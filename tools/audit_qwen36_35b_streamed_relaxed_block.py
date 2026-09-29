#!/usr/bin/env python3
"""Run reproducible streamed relaxed backward through a genuine 35B block."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForImageTextToText

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoint_stream import SafeTensorPrefixLoader
from dense_oracle import tensor_sha256
from native_runtime import NativeManifestRelaxedLinearSource
from native_store import NativeCandidateStore
from quant.ggml_native import GGMLNativeCodec, GGMLType
from qwen35_native import Qwen35LinearAttentionGeometry
from search.relaxed import (
    CheckpointedRelaxedBlockStats,
    CheckpointedStreamingRelaxedBlock,
    RelaxedExpertsBinding,
    RelaxedLinearBinding,
    RelaxedRouterBinding,
    StreamingRelaxedLinearStats,
)


EXPERT_DESTINATIONS = {
    "gate": "blk.0.ffn_gate_exps.weight",
    "up": "blk.0.ffn_up_exps.weight",
    "down": "blk.0.ffn_down_exps.weight",
}
ROUTER_DESTINATION = "blk.0.ffn_gate_inp.weight"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_oracle(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        values = {
            name: handle.get_tensor(name)
            for name in handle.keys() if name != "__metadata_json__"
        }
        metadata = json.loads(bytes(
            handle.get_tensor("__metadata_json__").tolist()).decode("utf-8"))
    if set(values) != {"input_ids", "block_input", "block_output"}:
        raise RuntimeError("retained genuine-block oracle inventory differs")
    return values, metadata


def _instantiate_model(model_dir: Path):
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(
            config, attn_implementation="eager")
    model.requires_grad_(False)
    return model


def _initial_logits(group_count: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(
        -0.8, 0.8, steps=group_count * 2,
        dtype=torch.float32, device=device,
    ).reshape(group_count, 2).requires_grad_(True)


def _forward_block(
    block: torch.nn.Module,
    hidden: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(hidden.shape[1], device=device).view(1, -1)
    mask = torch.ones(hidden.shape[:2], dtype=torch.long, device=device)
    output = block(
        hidden,
        position_embeddings=(None, None),
        attention_mask=mask,
        position_ids=positions,
        past_key_values=None,
        use_cache=False,
    )
    return output[0] if isinstance(output, tuple) else output


def _run_once(
    *,
    model_dir: Path,
    loader: SafeTensorPrefixLoader,
    store: NativeCandidateStore,
    entries: list[dict[str, Any]],
    block_input: torch.Tensor,
    block_target: torch.Tensor,
    device: torch.device,
    rows_per_chunk: int,
) -> dict[str, Any]:
    model = _instantiate_model(model_dir)
    block = model.model.language_model.layers[0]
    block.eval()
    logits = _initial_logits(len(entries), device)
    geometry = Qwen35LinearAttentionGeometry.from_model_dir(model_dir)
    entry_by_destination = {
        entry["destination_name"]: entry for entry in entries
    }
    group_by_destination = {
        entry["destination_name"]: index
        for index, entry in enumerate(entries)
    }

    linear_bindings = []
    persistent_sources = []
    for group_index, entry in enumerate(entries):
        destination = entry["destination_name"]
        if destination in set(EXPERT_DESTINATIONS.values()) | {ROUTER_DESTINATION}:
            continue
        source = NativeManifestRelaxedLinearSource(
            store,
            entry,
            reference_type=GGMLType.Q4_0,
            alternative_types=(GGMLType.Q2_0,),
            geometry=geometry,
            rows_per_chunk=rows_per_chunk,
        )
        linear_stats = StreamingRelaxedLinearStats()
        linear_bindings.append(RelaxedLinearBinding(
            module_path=entry["source_name"].removesuffix(".weight"),
            group_index=group_index,
            source=source,
            stats=linear_stats,
        ))
        persistent_sources.append((destination, source, linear_stats))

    router_entry = entry_by_destination[ROUTER_DESTINATION]
    router_source = NativeManifestRelaxedLinearSource(
        store,
        router_entry,
        reference_type=GGMLType.Q4_0,
        alternative_types=(GGMLType.Q2_0,),
        geometry=geometry,
        rows_per_chunk=rows_per_chunk,
    )
    router_stats = StreamingRelaxedLinearStats()
    router_binding = RelaxedRouterBinding(
        module_path=router_entry["source_name"].removesuffix(".weight"),
        group_index=group_by_destination[ROUTER_DESTINATION],
        source=router_source,
        stats=router_stats,
    )
    persistent_sources.append((ROUTER_DESTINATION, router_source, router_stats))

    expert_entries = {
        projection: entry_by_destination[destination]
        for projection, destination in EXPERT_DESTINATIONS.items()
    }
    expert_source_records: list[
        tuple[str, int, NativeManifestRelaxedLinearSource]
    ] = []

    def expert_source_factory(projection: str, expert_index: int):
        source = NativeManifestRelaxedLinearSource(
            store,
            expert_entries[projection],
            reference_type=GGMLType.Q4_0,
            alternative_types=(GGMLType.Q2_0,),
            geometry=geometry,
            rows_per_chunk=rows_per_chunk,
            expert_index=expert_index,
        )
        expert_source_records.append((projection, expert_index, source))
        return source

    expert_operation_stats = {
        projection: StreamingRelaxedLinearStats()
        for projection in EXPERT_DESTINATIONS
    }
    experts_module_path = expert_entries["gate"]["source_name"].rpartition(".")[0]
    expert_binding = RelaxedExpertsBinding(
        module_path=experts_module_path,
        gate_group_index=group_by_destination[EXPERT_DESTINATIONS["gate"]],
        up_group_index=group_by_destination[EXPERT_DESTINATIONS["up"]],
        down_group_index=group_by_destination[EXPERT_DESTINATIONS["down"]],
        source_factory=expert_source_factory,
        gate_stats=expert_operation_stats["gate"],
        up_stats=expert_operation_stats["up"],
        down_stats=expert_operation_stats["down"],
    )

    block_stats = CheckpointedRelaxedBlockStats()
    prefix = "model.language_model.layers.0"
    wrapper = CheckpointedStreamingRelaxedBlock(
        block,
        model=model,
        path=prefix,
        checkpoint_loader=loader,
        logits=logits,
        bindings=linear_bindings,
        expert_bindings=[expert_binding],
        router_bindings=[router_binding],
        device=device,
        stats=block_stats,
    )
    model.model.language_model.layers[0] = wrapper
    hidden = block_input.to(device=device).detach().requires_grad_(True)
    target = block_target.to(device=device)
    started = time.perf_counter()
    output = _forward_block(wrapper, hidden, device)
    if wrapper.module.input_layernorm.weight.device.type != "meta":
        raise RuntimeError("genuine block remained materialized after forward")
    loss = F.mse_loss(output.float(), target.float())
    loss.backward()
    elapsed = time.perf_counter() - started
    if wrapper.module.input_layernorm.weight.device.type != "meta":
        raise RuntimeError("genuine block remained materialized after backward")
    if block_stats.load_passes != 2 or block_stats.release_passes != 2:
        raise RuntimeError("genuine block did not reload exactly once in backward")
    if logits.grad is None or not torch.isfinite(logits.grad).all():
        raise RuntimeError("genuine-block assignment gradient is invalid")

    group_gradient_norms = logits.grad.float().norm(dim=1)
    if torch.count_nonzero(group_gradient_norms).item() != len(entries):
        raise RuntimeError("one or more genuine-block groups has zero gradient")
    expert_indices = sorted({
        expert_index for _, expert_index, _ in expert_source_records
    })
    if not expert_indices:
        raise RuntimeError("genuine-block routing selected no experts")

    persistent_records = [{
        "tensor": destination,
        "linear_stats": asdict(linear_stats),
        "native_source_stats": asdict(source.stats),
    } for destination, source, linear_stats in persistent_sources]
    expert_native_stats = {
        "source_instance_count": len(expert_source_records),
        "unique_expert_indices": expert_indices,
        "unique_expert_count": len(expert_indices),
        "reference_passes": sum(
            source.stats.reference_passes
            for _, _, source in expert_source_records),
        "alternative_passes": sum(
            source.stats.alternative_passes
            for _, _, source in expert_source_records),
        "reference_payload_bytes_read": sum(
            source.stats.reference_payload_bytes_read
            for _, _, source in expert_source_records),
        "alternative_payload_bytes_read": sum(
            source.stats.alternative_payload_bytes_read
            for _, _, source in expert_source_records),
        "max_resident_decoded_bytes": max(
            source.stats.max_resident_decoded_bytes
            for _, _, source in expert_source_records),
    }
    packed_bytes_read = (
        sum(
            source.stats.reference_payload_bytes_read
            + source.stats.alternative_payload_bytes_read
            for _, source, _ in persistent_sources
        )
        + expert_native_stats["reference_payload_bytes_read"]
        + expert_native_stats["alternative_payload_bytes_read"]
    )
    max_native_working_set = max(
        max(source.stats.max_resident_decoded_bytes
            for _, source, _ in persistent_sources),
        expert_native_stats["max_resident_decoded_bytes"],
    )
    max_linear_chunk = max(
        max(record["linear_stats"]["max_materialized_chunk_bytes"]
            for record in persistent_records),
        *(stats.max_materialized_chunk_bytes
          for stats in expert_operation_stats.values()),
    )
    result = {
        "loss": float(loss.item()),
        "output_sha256": tensor_sha256(output),
        "input_gradient_sha256": tensor_sha256(hidden.grad),
        "assignment_gradient_sha256": tensor_sha256(logits.grad),
        "assignment_gradient": logits.grad.detach().cpu().tolist(),
        "group_gradient_norms": group_gradient_norms.detach().cpu().tolist(),
        "nonzero_group_gradient_count": int(torch.count_nonzero(
            group_gradient_norms).item()),
        "output_finite": bool(torch.isfinite(output).all()),
        "input_gradient_finite": bool(torch.isfinite(hidden.grad).all()),
        "elapsed_seconds": elapsed,
        "block_stats": asdict(block_stats),
        "persistent_candidates": persistent_records,
        "expert_operation_stats": {
            projection: asdict(stats)
            for projection, stats in expert_operation_stats.items()
        },
        "expert_native_source_stats": expert_native_stats,
        "packed_payload_bytes_read": packed_bytes_read,
        "max_native_source_working_set_bytes": max_native_working_set,
        "max_materialized_linear_chunk_bytes": max_linear_chunk,
    }
    del output, loss, hidden, logits, wrapper, model
    gc.collect()
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model_dir = args.model_dir.resolve(strict=True)
    oracle_path = args.oracle.resolve(strict=True)
    identity_path = args.identity.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    store_path = args.store.resolve(strict=True)
    identity = _load_json(identity_path)
    manifest = _load_json(manifest_path)
    oracle, oracle_metadata = _load_oracle(oracle_path)
    if oracle_metadata["revision"] != identity["revision"]:
        raise RuntimeError("genuine oracle and identity revisions differ")
    entries = [
        entry for entry in manifest["entries"]
        if entry["rco_search"]
        and entry["destination_name"].startswith("blk.0.")
    ]
    if len(entries) != 13:
        raise RuntimeError(f"expected 13 block decisions, found {len(entries)}")
    codec = GGMLNativeCodec(args.ggml_library)
    store = NativeCandidateStore(store_path, codec)
    store_revision = store.index["source"].get(
        "dense_revision", store.index["source"].get("revision"))
    if store_revision != identity["revision"]:
        raise RuntimeError("genuine candidate store and identity revisions differ")
    if set(store.index["tensors"]) != {
        entry["destination_name"] for entry in entries
    }:
        raise RuntimeError("genuine store does not exactly cover block decisions")
    loader = SafeTensorPrefixLoader(model_dir)
    schema_model = _instantiate_model(model_dir)
    schema = loader.assert_prefix_schema(
        schema_model, "model.language_model.layers.0")
    del schema_model

    first = _run_once(
        model_dir=model_dir,
        loader=loader,
        store=store,
        entries=entries,
        block_input=oracle["block_input"],
        block_target=oracle["block_output"],
        device=device,
        rows_per_chunk=args.rows_per_chunk,
    )
    second = _run_once(
        model_dir=model_dir,
        loader=loader,
        store=store,
        entries=entries,
        block_input=oracle["block_input"],
        block_target=oracle["block_output"],
        device=device,
        rows_per_chunk=args.rows_per_chunk,
    )
    reproducible = all(first[key] == second[key] for key in (
        "loss",
        "output_sha256",
        "input_gradient_sha256",
        "assignment_gradient_sha256",
        "assignment_gradient",
    ))
    if not reproducible:
        raise RuntimeError("genuine streamed relaxed backward is not reproducible")

    cuda = {
        "available": torch.cuda.is_available(),
        "built_version": torch.version.cuda,
        "allocated_peak_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else 0),
        "reserved_peak_bytes": (
            torch.cuda.max_memory_reserved(device)
            if device.type == "cuda" else 0),
    }
    if device.type == "cuda":
        cuda["device_name"] = torch.cuda.get_device_name(device)
    return {
        "schema": 1,
        "status": "pass",
        "scope": (
            "reproducible relaxed forward/backward through the complete genuine "
            "Qwen3.6-35B-A3B layer 0 with 13 native decisions, routed expert "
            "slice streaming, and checkpoint reload; this is not yet a complete "
            "40-layer search or CUDA gate"
        ),
        "source": {
            "repo_id": identity["repo_id"],
            "revision": identity["revision"],
            "model_dir": str(model_dir),
            "identity_path": str(identity_path),
            "identity_sha256": _sha256_file(identity_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
        },
        "oracle": {
            "path": str(oracle_path),
            "sha256": _sha256_file(oracle_path),
            "metadata": oracle_metadata,
            "objective": "mean squared error against retained dense block output",
        },
        "candidate_store": {
            "path": str(store_path),
            "index_sha256": _sha256_file(
                store_path / "native-candidate-index.json"),
            "decision_count": len(entries),
            "reference_type": GGMLType.Q4_0.name,
            "alternative_type": GGMLType.Q2_0.name,
            "rows_per_chunk": args.rows_per_chunk,
            "expert_policy": (
                "decode only routed expert slices from aggregated payloads"),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "cuda": cuda,
        },
        "block_schema": schema,
        "initial_logits": _initial_logits(
            len(entries), torch.device("cpu")).detach().tolist(),
        "runs": [first, second],
        "same_seed_reproducible": reproducible,
        "memory": {
            "peak_process_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "max_materialized_block_bytes": max(
                run["block_stats"]["max_materialized_block_bytes"]
                for run in (first, second)),
            "max_native_source_working_set_bytes": max(
                run["max_native_source_working_set_bytes"]
                for run in (first, second)),
            "max_materialized_linear_chunk_bytes": max(
                run["max_materialized_linear_chunk_bytes"]
                for run in (first, second)),
            "max_cuda_allocated_bytes": cuda["allocated_peak_bytes"],
            "max_cuda_reserved_bytes": cuda["reserved_peak_bytes"],
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit(args)
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "loss": report["runs"][0]["loss"],
        "nonzero_group_gradient_count": report["runs"][0][
            "nonzero_group_gradient_count"],
        "unique_routed_expert_count": report["runs"][0][
            "expert_native_source_stats"]["unique_expert_count"],
        "same_seed_reproducible": report["same_seed_reproducible"],
        "memory": report["memory"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
