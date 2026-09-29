#!/usr/bin/env python3
"""Validate streamed relaxed backward on the complete Qwen3.5-2B block 0."""

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

import numpy as np
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
from search.quant import WeightInterpolation
from search.relaxed import (
    CheckpointedRelaxedBlockStats,
    CheckpointedStreamingRelaxedBlock,
    RelaxedLinearBinding,
    StreamingRelaxedLinearStats,
)


MAX_OUTPUT_RELATIVE_ERROR = 0.02
MAX_INPUT_GRADIENT_RELATIVE_ERROR = 0.05
MAX_ASSIGNMENT_GRADIENT_RELATIVE_ERROR = 0.05
MAX_LOSS_ABSOLUTE_ERROR = 0.001
MIN_GRADIENT_COSINE_SIMILARITY = 0.999


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
        raise RuntimeError("retained block oracle inventory differs")
    return values, metadata


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
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            f"Qwen development block returned {type(output).__name__}")
    return output


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.detach().float().cpu() - expected.detach().float().cpu()
    denominator = torch.linalg.vector_norm(expected.detach().float().cpu())
    numerator = torch.linalg.vector_norm(difference)
    return float(numerator / denominator) if denominator else float(numerator)


def _comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual.detach().float().cpu() - expected.detach().float().cpu()
    return {
        "max_absolute_error": float(difference.abs().max()),
        "rmse": float(torch.sqrt(difference.square().mean())),
        "relative_frobenius_error": _relative_error(actual, expected),
        "cosine_similarity": float(F.cosine_similarity(
            actual.detach().float().cpu().reshape(1, -1),
            expected.detach().float().cpu().reshape(1, -1),
        )),
    }


def _instantiate_model(model_dir: Path):
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(
            config, attn_implementation="eager")
    model.requires_grad_(False)
    return model


def _initial_logits(group_count: int, device: torch.device) -> torch.Tensor:
    values = torch.linspace(
        -0.6, 0.6, steps=group_count * 2,
        dtype=torch.float32, device=device,
    ).reshape(group_count, 2)
    return values.requires_grad_(True)


def _run_streamed(
    *, model_dir: Path, loader: SafeTensorPrefixLoader,
    store: NativeCandidateStore, entries: list[dict[str, Any]],
    hidden_values: torch.Tensor, target: torch.Tensor,
    device: torch.device, rows_per_chunk: int,
) -> dict[str, Any]:
    model = _instantiate_model(model_dir)
    block = model.model.language_model.layers[0]
    block.eval()
    logits = _initial_logits(len(entries), device)
    geometry = Qwen35LinearAttentionGeometry.from_model_dir(model_dir)
    block_stats = CheckpointedRelaxedBlockStats()
    bindings = []
    source_records = []
    for group_index, entry in enumerate(entries):
        source = NativeManifestRelaxedLinearSource(
            store,
            entry,
            reference_type=GGMLType.Q4_0,
            alternative_types=(GGMLType.Q2_0,),
            geometry=geometry,
            rows_per_chunk=rows_per_chunk,
        )
        linear_stats = StreamingRelaxedLinearStats()
        bindings.append(RelaxedLinearBinding(
            module_path=entry["source_name"].removesuffix(".weight"),
            group_index=group_index,
            source=source,
            stats=linear_stats,
        ))
        source_records.append((entry, source, linear_stats))

    prefix = "model.language_model.layers.0"
    wrapper = CheckpointedStreamingRelaxedBlock(
        block,
        model=model,
        path=prefix,
        checkpoint_loader=loader,
        logits=logits,
        bindings=bindings,
        device=device,
        stats=block_stats,
    )
    model.model.language_model.layers[0] = wrapper
    hidden = hidden_values.to(device=device).detach().requires_grad_(True)
    expected = target.to(device=device)
    started = time.perf_counter()
    output = _forward_block(wrapper, hidden, device)
    if wrapper.module.input_layernorm.weight.device.type != "meta":
        raise RuntimeError("streamed block remained materialized after forward")
    loss = F.mse_loss(output.float(), expected.float())
    loss.backward()
    if wrapper.module.input_layernorm.weight.device.type != "meta":
        raise RuntimeError("streamed block remained materialized after backward")
    elapsed = time.perf_counter() - started
    if block_stats.load_passes != 2 or block_stats.release_passes != 2:
        raise RuntimeError(
            "streamed block was not loaded once for forward and recomputation")
    if logits.grad is None or not torch.isfinite(logits.grad).all():
        raise RuntimeError("streamed assignment gradient is missing or non-finite")

    candidate_records = []
    for entry, source, linear_stats in source_records:
        if (
            linear_stats.forward_reference_passes != 2
            or linear_stats.backward_reference_passes != 1
            or linear_stats.forward_alternative_passes != 2
            or linear_stats.backward_alternative_passes != 1
        ):
            raise RuntimeError(
                f"unexpected candidate reread counts for "
                f"{entry['destination_name']}")
        candidate_records.append({
            "tensor": entry["destination_name"],
            "source_tensor": entry["source_name"],
            "linear_stats": asdict(linear_stats),
            "native_source_stats": asdict(source.stats),
        })

    result = {
        "loss": float(loss.item()),
        "output": output.detach().cpu(),
        "input_gradient": hidden.grad.detach().cpu(),
        "assignment_gradient": logits.grad.detach().cpu(),
        "output_sha256": tensor_sha256(output),
        "input_gradient_sha256": tensor_sha256(hidden.grad),
        "assignment_gradient_sha256": tensor_sha256(logits.grad),
        "elapsed_seconds": elapsed,
        "block_stats": asdict(block_stats),
        "candidates": candidate_records,
    }
    del output, loss, hidden, logits, wrapper, model
    gc.collect()
    return result


def _run_dense(
    *, model_dir: Path, loader: SafeTensorPrefixLoader,
    store: NativeCandidateStore, entries: list[dict[str, Any]],
    hidden_values: torch.Tensor, target: torch.Tensor,
    device: torch.device, rows_per_chunk: int,
) -> dict[str, Any]:
    model = _instantiate_model(model_dir)
    prefix = "model.language_model.layers.0"
    block_bytes = loader.load_prefix(model, prefix, device=device)
    block = model.model.language_model.layers[0]
    block.eval()
    logits = _initial_logits(len(entries), device)
    geometry = Qwen35LinearAttentionGeometry.from_model_dir(model_dir)
    persistent_delta_bytes = 0
    decoded_records = []
    for group_index, entry in enumerate(entries):
        source = NativeManifestRelaxedLinearSource(
            store,
            entry,
            reference_type=GGMLType.Q4_0,
            alternative_types=(GGMLType.Q2_0,),
            geometry=geometry,
            rows_per_chunk=rows_per_chunk,
        )
        reference = np.concatenate([
            rows for _, rows in source.iter_reference_rows()
        ])
        delta = np.concatenate([
            rows for _, rows in source.iter_delta_rows(0)
        ])
        module = model.get_submodule(
            entry["source_name"].removesuffix(".weight"))
        reference_tensor = torch.from_numpy(reference).to(
            device=device, dtype=module.weight.dtype)
        delta_tensor = torch.from_numpy(delta).to(
            device=device, dtype=module.weight.dtype)
        with torch.no_grad():
            module.weight.copy_(reference_tensor)
        torch.nn.utils.parametrize.register_parametrization(
            module,
            "weight",
            WeightInterpolation([delta_tensor], logits, group_index),
        )
        persistent_delta_bytes += delta_tensor.numel() * delta_tensor.element_size()
        decoded_records.append({
            "tensor": entry["destination_name"],
            "reference_bytes": reference_tensor.numel()
            * reference_tensor.element_size(),
            "delta_bytes": delta_tensor.numel() * delta_tensor.element_size(),
        })

    hidden = hidden_values.to(device=device).detach().requires_grad_(True)
    expected = target.to(device=device)
    started = time.perf_counter()
    output = _forward_block(block, hidden, device)
    loss = F.mse_loss(output.float(), expected.float())
    loss.backward()
    elapsed = time.perf_counter() - started
    if logits.grad is None or not torch.isfinite(logits.grad).all():
        raise RuntimeError("dense assignment gradient is missing or non-finite")
    result = {
        "loss": float(loss.item()),
        "output": output.detach().cpu(),
        "input_gradient": hidden.grad.detach().cpu(),
        "assignment_gradient": logits.grad.detach().cpu(),
        "output_sha256": tensor_sha256(output),
        "input_gradient_sha256": tensor_sha256(hidden.grad),
        "assignment_gradient_sha256": tensor_sha256(logits.grad),
        "elapsed_seconds": elapsed,
        "resident_block_bytes": block_bytes,
        "persistent_delta_bytes": persistent_delta_bytes,
        "candidates": decoded_records,
    }
    del output, loss, hidden, logits, block, model
    gc.collect()
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model_dir = args.model_dir.resolve(strict=True)
    oracle_path = args.oracle.resolve(strict=True)
    identity_path = args.identity.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    store_path = args.store.resolve(strict=True)
    identity = _load_json(identity_path)
    manifest = _load_json(manifest_path)
    oracle, oracle_metadata = _load_oracle(oracle_path)
    if oracle_metadata["revision"] != identity["revision"]:
        raise RuntimeError("oracle and identity revisions differ")
    entries = [
        entry for entry in manifest["entries"]
        if entry["rco_search"]
        and entry["destination_name"].startswith("blk.0.")
    ]
    if len(entries) != 8 or any(len(entry["source_shape"]) != 2 for entry in entries):
        raise RuntimeError("expected eight complete 2-D block-0 decisions")
    codec = GGMLNativeCodec(args.ggml_library)
    store = NativeCandidateStore(store_path, codec)
    if store.index["source"]["revision"] != identity["revision"]:
        raise RuntimeError("candidate store and identity revisions differ")
    loader = SafeTensorPrefixLoader(model_dir)
    schema_model = _instantiate_model(model_dir)
    schema = loader.assert_prefix_schema(
        schema_model, "model.language_model.layers.0")
    del schema_model

    streamed = _run_streamed(
        model_dir=model_dir,
        loader=loader,
        store=store,
        entries=entries,
        hidden_values=oracle["block_input"],
        target=oracle["block_output"],
        device=device,
        rows_per_chunk=args.rows_per_chunk,
    )
    streamed_repeat = _run_streamed(
        model_dir=model_dir,
        loader=loader,
        store=store,
        entries=entries,
        hidden_values=oracle["block_input"],
        target=oracle["block_output"],
        device=device,
        rows_per_chunk=args.rows_per_chunk,
    )
    streamed_peak_rss = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    reproducible = all(
        streamed[key] == streamed_repeat[key]
        for key in (
            "loss", "output_sha256", "input_gradient_sha256",
            "assignment_gradient_sha256",
        )
    )
    if not reproducible:
        raise RuntimeError("streamed relaxed block is not reproducible")

    dense = _run_dense(
        model_dir=model_dir,
        loader=loader,
        store=store,
        entries=entries,
        hidden_values=oracle["block_input"],
        target=oracle["block_output"],
        device=device,
        rows_per_chunk=args.rows_per_chunk,
    )
    total_peak_rss = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    comparisons = {
        "output": _comparison(streamed["output"], dense["output"]),
        "input_gradient": _comparison(
            streamed["input_gradient"], dense["input_gradient"]),
        "assignment_gradient": _comparison(
            streamed["assignment_gradient"], dense["assignment_gradient"]),
        "loss_absolute_error": abs(streamed["loss"] - dense["loss"]),
    }
    tolerances = {
        "output_relative_frobenius_error": MAX_OUTPUT_RELATIVE_ERROR,
        "input_gradient_relative_frobenius_error": (
            MAX_INPUT_GRADIENT_RELATIVE_ERROR),
        "assignment_gradient_relative_frobenius_error": (
            MAX_ASSIGNMENT_GRADIENT_RELATIVE_ERROR),
        "minimum_gradient_cosine_similarity": (
            MIN_GRADIENT_COSINE_SIMILARITY),
        "loss_absolute_error": MAX_LOSS_ABSOLUTE_ERROR,
    }
    tolerance_pass = (
        comparisons["output"]["relative_frobenius_error"]
        <= MAX_OUTPUT_RELATIVE_ERROR
        and comparisons["input_gradient"]["relative_frobenius_error"]
        <= MAX_INPUT_GRADIENT_RELATIVE_ERROR
        and comparisons["assignment_gradient"]["relative_frobenius_error"]
        <= MAX_ASSIGNMENT_GRADIENT_RELATIVE_ERROR
        and comparisons["input_gradient"]["cosine_similarity"]
        >= MIN_GRADIENT_COSINE_SIMILARITY
        and comparisons["assignment_gradient"]["cosine_similarity"]
        >= MIN_GRADIENT_COSINE_SIMILARITY
        and comparisons["loss_absolute_error"] <= MAX_LOSS_ABSOLUTE_ERROR
    )
    if not tolerance_pass:
        raise RuntimeError(
            f"streamed block differs from dense relaxed oracle: {comparisons}")

    max_linear_chunk = max(
        record["linear_stats"]["max_materialized_chunk_bytes"]
        for record in streamed["candidates"])
    max_native_chunk = max(
        record["native_source_stats"]["max_resident_decoded_bytes"]
        for record in streamed["candidates"])
    total_packed_reads = sum(
        record["native_source_stats"]["reference_payload_bytes_read"]
        + record["native_source_stats"]["alternative_payload_bytes_read"]
        for record in streamed["candidates"])

    def serializable_run(value: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item for key, item in value.items()
            if not isinstance(item, torch.Tensor)
        }

    return {
        "schema": 1,
        "status": "pass",
        "scope": (
            "complete Qwen3.5-2B layer-0 streamed relaxed forward/backward "
            "against released dense interpolation on eight native Q2_0/Q4_0 "
            "decisions; this is not yet the fused-expert 35B or CUDA gate"
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
        },
        "candidate_store": {
            "path": str(store_path),
            "index_sha256": _sha256_file(
                store_path / "native-candidate-index.json"),
            "decision_count": len(entries),
            "reference_type": GGMLType.Q4_0.name,
            "alternative_type": GGMLType.Q2_0.name,
            "rows_per_chunk": args.rows_per_chunk,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
        "block_schema": schema,
        "initial_logits": _initial_logits(
            len(entries), torch.device("cpu")).detach().tolist(),
        "streamed": serializable_run(streamed),
        "streamed_repeat": serializable_run(streamed_repeat),
        "dense_reference": serializable_run(dense),
        "comparison": comparisons,
        "tolerances": tolerances,
        "tolerance_pass": tolerance_pass,
        "same_seed_reproducible": reproducible,
        "memory": {
            "streamed_peak_process_rss_bytes": streamed_peak_rss,
            "total_peak_after_dense_reference_bytes": total_peak_rss,
            "streamed_max_materialized_weight_chunk_bytes": max_linear_chunk,
            "streamed_max_native_source_working_set_bytes": max_native_chunk,
            "streamed_packed_payload_bytes_read": total_packed_reads,
            "dense_persistent_delta_bytes": dense["persistent_delta_bytes"],
            "dense_resident_block_bytes": dense["resident_block_bytes"],
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
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
        "comparison": report["comparison"],
        "same_seed_reproducible": report["same_seed_reproducible"],
        "memory": report["memory"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
