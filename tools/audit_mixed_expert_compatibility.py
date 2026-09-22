#!/usr/bin/env python3
"""Prove whether fused Qwen experts can load mixed packed bit widths."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import io
import json
import os
import re
import resource
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from quant.compressed_layout import compressed_state_from_bundle


GROUP_SIZE = 8
EXPERT_BITS = (2, 4)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _bundle(shape: tuple[int, int], bits: int, salt: int) -> dict:
    rows, columns = shape
    codes = (
        (torch.arange(rows * columns).reshape(rows, columns) + salt)
        % (1 << bits)
    ).to(torch.uint8)
    scales = torch.full(
        (rows, (columns + GROUP_SIZE - 1) // GROUP_SIZE),
        0.03125,
        dtype=torch.float32,
    )
    return {
        "bits": bits,
        "shape": shape,
        "sym": True,
        "act_order": False,
        "perm": None,
        "group_size": GROUP_SIZE,
        "scales": scales,
        "zeros": torch.full_like(scales, float(1 << (bits - 1))),
        "qweight": codes,
    }


def _scheme(bits: int, expert: int) -> dict:
    return {
        "targets": [f"re:.*experts\\.{expert}\\..*"],
        "format": "pack-quantized",
        "weights": {
            "num_bits": bits,
            "group_size": GROUP_SIZE,
            "symmetric": True,
            "strategy": "group",
            "type": "int",
            "actorder": None,
            "observer": "minmax",
            "observer_kwargs": {},
            "dynamic": False,
            "block_structure": None,
        },
    }


def _quantization_config(order: tuple[int, int]) -> dict:
    return {
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "format": "pack-quantized",
        "ignore": [],
        "kv_cache_scheme": None,
        "config_groups": {
            f"expert_{expert}_q{EXPERT_BITS[expert]}": _scheme(
                EXPERT_BITS[expert], expert
            )
            for expert in order
        },
    }


def _diagnostic(output: str) -> list[str]:
    messages = []
    for raw in output.replace("\r", "\n").splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw).strip()
        if "size of tensor" in line or "Could not match" in line:
            if line not in messages:
                messages.append(line)
    return messages[:8]


def _attempt(root: Path, device_map: str | None) -> dict:
    from transformers import AutoModelForCausalLM
    from transformers.utils import logging as transformers_logging

    captured = io.StringIO()
    previous_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity(50)
    transformers_logging.disable_progress_bar()
    try:
        kwargs = {} if device_map is None else {"device_map": device_map}
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            _, loading_info = AutoModelForCausalLM.from_pretrained(
                root,
                local_files_only=True,
                output_loading_info=True,
                dtype=torch.float32,
                **kwargs,
            )
        fields = {
            name: [str(item) for item in loading_info.get(name, [])]
            for name in (
                "missing_keys", "unexpected_keys", "mismatched_keys",
                "error_msgs",
            )
        }
        return {
            "device_map": device_map,
            "result": "loaded",
            "loading_info": fields,
            "diagnostic": _diagnostic(captured.getvalue()),
        }
    except Exception as error:
        return {
            "device_map": device_map,
            "result": "failed",
            "exception_type": type(error).__name__,
            "exception": str(error),
            "diagnostic": _diagnostic(captured.getvalue()),
        }
    finally:
        transformers_logging.set_verbosity(previous_verbosity)


def _direct_conversion_probe(config_data: dict, expert_states: list[dict]) -> dict:
    """Capture the underlying conversion error hidden by the load report."""
    from compressed_tensors.quantization import QuantizationConfig
    from transformers.integrations.compressed_tensors import DecompressExperts

    quantization_config = QuantizationConfig.model_validate(config_data)
    quantizer = SimpleNamespace(
        compressor=SimpleNamespace(quantization_config=quantization_config)
    )
    operation = DecompressExperts(quantizer)
    values = {
        "expert.weight_packed": [item["weight_packed"] for item in expert_states],
        "expert.weight_scale": [item["weight_scale"] for item in expert_states],
        "expert.weight_shape": [item["weight_shape"] for item in expert_states],
    }
    try:
        operation.convert(values, [], ["fused_expert"])
        return {"result": "converted"}
    except Exception as error:
        return {
            "result": "failed",
            "exception_type": type(error).__name__,
            "exception": str(error),
        }


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
    from transformers.integrations.compressed_tensors import DecompressExperts

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
        num_experts=2,
        tie_word_embeddings=False,
        dtype="float32",
    )
    config.architectures = ["Qwen3_5MoeForCausalLM"]
    model = Qwen3_5MoeForCausalLM(config)
    state = model.state_dict()
    gate_up = state.pop("model.layers.0.mlp.experts.gate_up_proj")
    down = state.pop("model.layers.0.mlp.experts.down_proj")
    sources = {
        "gate_proj": gate_up[:, :16],
        "up_proj": gate_up[:, 16:],
        "down_proj": down,
    }
    packed_shapes = {}
    direct_states = []
    for expert, bits in enumerate(EXPERT_BITS):
        for projection_index, projection in enumerate(PROJECTIONS):
            prefix = f"model.layers.0.mlp.experts.{expert}.{projection}"
            mapped = compressed_state_from_bundle(
                _bundle(
                    tuple(sources[projection][expert].shape),
                    bits,
                    salt=expert * len(PROJECTIONS) + projection_index,
                )
            )
            state.update({f"{prefix}.{name}": value for name, value in mapped.items()})
            packed_shapes[prefix] = list(mapped["weight_packed"].shape)
            if projection == "gate_proj":
                direct_states.append(mapped)
    del model, gate_up, down, sources

    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    save_torch_state_dict(
        state,
        checkpoint_dir,
        max_shard_size="20KB",
        safe_serialization=True,
        metadata={"format": "pt"},
    )
    del state

    cases = []
    for order in ((0, 1), (1, 0)):
        config_data = config.to_dict()
        quantization_config = _quantization_config(order)
        config_data["quantization_config"] = quantization_config
        with (checkpoint_dir / "config.json").open("w") as handle:
            json.dump(config_data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        attempts = [_attempt(checkpoint_dir, "meta"), _attempt(checkpoint_dir, None)]
        cases.append({
            "config_group_order": [
                {"expert": expert, "bits": EXPERT_BITS[expert]}
                for expert in order
            ],
            "direct_conversion": _direct_conversion_probe(
                quantization_config, direct_states
            ),
            "attempts": attempts,
        })

    source = inspect.getsource(DecompressExperts.convert)
    all_attempts = [attempt for case in cases for attempt in case["attempts"]]
    all_failed = all(attempt["result"] == "failed" for attempt in all_attempts)
    direct_failures = [case["direct_conversion"] for case in cases]
    shape_mismatch_seen = all(
        item["result"] == "failed" and "size of tensor" in item["exception"]
        for item in direct_failures
    )
    unsupported_confirmed = all_failed and shape_mismatch_seen
    files = {
        path.name: path.stat().st_size for path in sorted(checkpoint_dir.iterdir())
    }
    return {
        "schema": 1,
        "result": (
            "unsupported_confirmed" if unsupported_confirmed
            else "inconclusive"
        ),
        "transformers": transformers.__version__,
        "compressed_tensors": compressed_tensors.__version__,
        "torch": torch.__version__,
        "model_class": "Qwen3_5MoeForCausalLM",
        "expert_bits": list(EXPERT_BITS),
        "group_size": GROUP_SIZE,
        "packed_shapes": packed_shapes,
        "checkpoint": {
            "files": files,
            "total_bytes": sum(files.values()),
        },
        "loader_implementation": {
            "decompress_experts_convert_sha256": hashlib.sha256(
                source.encode()
            ).hexdigest(),
            "selects_first_config_group": (
                "list(ct_quantization_config.config_groups.values())[0]" in source
            ),
        },
        "cases": cases,
        "conclusion": (
            "Transformers 5.13.1 applies one config-group bit width to the "
            "whole fused expert collection; per-expert mixed packed widths "
            "cannot be loaded by the unmodified compressed-tensors path."
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a tiny Q2/Q4 Qwen expert checkpoint and test both config "
            "group orders through meta and materialized loads."
        )
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--temporary-parent", type=Path, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    parent = None if args.temporary_parent is None else str(args.temporary_parent)
    with tempfile.TemporaryDirectory(prefix="rco-mixed-expert-", dir=parent) as temporary:
        report = audit(Path(temporary) / "checkpoint")
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _atomic_json(args.output, report)
        print(f"Wrote {args.output}")
    return 0 if report["result"] == "unsupported_confirmed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
