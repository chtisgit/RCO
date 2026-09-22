#!/usr/bin/env python3
"""Build and load a tiny packed Qwen3.5-MoE checkpoint.

This is a bounded integration audit for the output schema. It creates a
one-layer, two-expert text model, maps synthetic RCO qparams into all logical
expert projections, writes indexed safetensors shards, then verifies both a
clean meta load and an exact materialized expert round trip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from quant.compressed_layout import compressed_state_from_bundle


BITS = 2
GROUP_SIZE = 8
NUM_EXPERTS = 2
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _quantization_config() -> dict:
    return {
        "config_groups": {
            "group_0": {
                "input_activations": None,
                "output_activations": None,
                "targets": ["re:.*experts.*"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": GROUP_SIZE,
                    "num_bits": BITS,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "group",
                    "symmetric": True,
                    "type": "int",
                },
            },
        },
        "format": "pack-quantized",
        "ignore": [],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }


def _bundle(shape: tuple[int, int], salt: int) -> dict:
    rows, columns = shape
    codes = (
        (torch.arange(rows * columns).reshape(rows, columns) + salt) % 4
    ).to(torch.uint8)
    groups = (columns + GROUP_SIZE - 1) // GROUP_SIZE
    row = torch.arange(rows, dtype=torch.float32).reshape(-1, 1)
    group = torch.arange(groups, dtype=torch.float32).reshape(1, -1)
    scales = (0.03125 + row * 0.0005 + group * 0.002).contiguous()
    return {
        "bits": BITS,
        "shape": shape,
        "sym": True,
        "act_order": False,
        "perm": None,
        "group_size": GROUP_SIZE,
        "scales": scales,
        "zeros": torch.full_like(scales, 2.0),
        "qweight": codes,
    }


def _decode(bundle: dict) -> torch.Tensor:
    codes = bundle["qweight"].float()
    columns = codes.shape[1]
    group_index = torch.arange(columns) // int(bundle["group_size"])
    return (
        bundle["scales"][:, group_index]
        * (codes - bundle["zeros"][:, group_index])
    )


def _loading_info(info: dict) -> dict:
    fields = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    return {field: [str(item) for item in info.get(field, [])] for field in fields}


def _clean(info: dict) -> bool:
    return all(not values for values in info.values())


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def audit(checkpoint_dir: Path) -> dict:
    import compressed_tensors
    import transformers
    from huggingface_hub import save_torch_state_dict
    from transformers import (
        AutoModelForCausalLM,
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeTextConfig,
    )

    started = time.perf_counter()
    config = Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        layer_types=["full_attention"],
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts_per_tok=1,
        num_experts=NUM_EXPERTS,
        tie_word_embeddings=False,
        dtype="float32",
    )
    config.architectures = ["Qwen3_5MoeForCausalLM"]
    torch.manual_seed(1234)
    source_model = Qwen3_5MoeForCausalLM(config)
    state = source_model.state_dict()
    gate_up = state.pop("model.layers.0.mlp.experts.gate_up_proj")
    down = state.pop("model.layers.0.mlp.experts.down_proj")
    intermediate = config.moe_intermediate_size
    sources = {
        "gate_proj": gate_up[:, :intermediate],
        "up_proj": gate_up[:, intermediate:],
        "down_proj": down,
    }
    expected: dict[str, list[torch.Tensor]] = {
        projection: [] for projection in PROJECTIONS
    }
    hashes = {}
    for expert in range(NUM_EXPERTS):
        for projection_index, projection in enumerate(PROJECTIONS):
            prefix = f"model.layers.0.mlp.experts.{expert}.{projection}"
            bundle = _bundle(
                tuple(sources[projection][expert].shape),
                salt=expert * len(PROJECTIONS) + projection_index,
            )
            mapped = compressed_state_from_bundle(bundle)
            state.update({f"{prefix}.{name}": value for name, value in mapped.items()})
            expected[projection].append(_decode(bundle))
            hashes[prefix] = hashlib.sha256(
                mapped["weight_packed"].view(torch.uint8).numpy().tobytes()
            ).hexdigest()
    del source_model, sources, gate_up, down

    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    config_data = config.to_dict()
    config_data["quantization_config"] = _quantization_config()
    with (checkpoint_dir / "config.json").open("w") as handle:
        json.dump(config_data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    save_torch_state_dict(
        state,
        checkpoint_dir,
        max_shard_size="20KB",
        safe_serialization=True,
        metadata={"format": "pt"},
    )
    del state

    meta_model, raw_meta_info = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        device_map="meta",
        dtype=torch.float32,
        local_files_only=True,
        output_loading_info=True,
    )
    meta_info = _loading_info(raw_meta_info)
    meta_devices = sorted({str(value.device) for value in meta_model.parameters()})
    meta_class = type(meta_model).__name__
    del meta_model

    cpu_model, raw_cpu_info = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        dtype=torch.float32,
        local_files_only=True,
        output_loading_info=True,
    )
    cpu_info = _loading_info(raw_cpu_info)
    actual_gate_up = cpu_model.model.layers[0].mlp.experts.gate_up_proj.detach()
    actual_down = cpu_model.model.layers[0].mlp.experts.down_proj.detach()
    expected_gate_up = torch.cat(
        (torch.stack(expected["gate_proj"]), torch.stack(expected["up_proj"])),
        dim=1,
    )
    expected_down = torch.stack(expected["down_proj"])
    errors = {
        "gate_up_max_abs": float((actual_gate_up - expected_gate_up).abs().max()),
        "down_max_abs": float((actual_down - expected_down).abs().max()),
    }
    runtime_shapes = {
        "gate_up_proj": list(actual_gate_up.shape),
        "down_proj": list(actual_down.shape),
    }
    del cpu_model

    index_path = checkpoint_dir / "model.safetensors.index.json"
    with index_path.open() as handle:
        index = json.load(handle)
    shards = sorted(set(index["weight_map"].values()))
    files = {
        path.name: path.stat().st_size for path in sorted(checkpoint_dir.iterdir())
    }
    exact = all(value == 0.0 for value in errors.values())
    passed = (
        _clean(meta_info)
        and _clean(cpu_info)
        and meta_devices == ["meta"]
        and len(shards) >= 2
        and sum(key.endswith("weight_packed") for key in index["weight_map"]) == 6
        and exact
    )
    return {
        "schema": 1,
        "result": "passed" if passed else "failed",
        "transformers": transformers.__version__,
        "compressed_tensors": compressed_tensors.__version__,
        "torch": torch.__version__,
        "model_class": meta_class,
        "architecture": {
            "layers": 1,
            "experts": NUM_EXPERTS,
            "hidden_size": config.hidden_size,
            "moe_intermediate_size": config.moe_intermediate_size,
        },
        "quantization": {
            "bits": BITS,
            "group_size": GROUP_SIZE,
            "symmetric": True,
            "act_order": False,
            "mapped_logical_projection_count": len(hashes),
        },
        "mapped_packed_sha256": hashes,
        "checkpoint": {
            "files": files,
            "total_bytes": sum(files.values()),
            "shard_count": len(shards),
            "shards": shards,
            "indexed_tensor_count": len(index["weight_map"]),
            "packed_tensor_count": sum(
                key.endswith("weight_packed") for key in index["weight_map"]
            ),
        },
        "meta_load": {
            "parameter_devices": meta_devices,
            "loading_info": meta_info,
            "clean": _clean(meta_info),
        },
        "materialized_load": {
            "loading_info": cpu_info,
            "clean": _clean(cpu_info),
            "runtime_expert_shapes": runtime_shapes,
            "max_abs_errors": errors,
            "exact": exact,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a tiny indexed Qwen3.5-MoE checkpoint from mapped RCO "
            "qparams and verify clean meta and exact materialized loads."
        )
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--temporary-parent",
        type=Path,
        default=None,
        help="Parent for the automatically deleted tiny checkpoint",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    parent = None if args.temporary_parent is None else str(args.temporary_parent)
    with tempfile.TemporaryDirectory(prefix="rco-tiny-packed-", dir=parent) as temporary:
        report = audit(Path(temporary) / "checkpoint")
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _write_json(args.output, report)
        print(f"Wrote {args.output}")
    return 0 if report["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
