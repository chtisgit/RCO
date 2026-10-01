#!/usr/bin/env python3
"""Compare decoded native block-0 assignments with the retained dense oracle."""

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
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForImageTextToText

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoint_stream import SafeTensorPrefixLoader
from dense_oracle import tensor_sha256
from native_store import NativeCandidateStore
from quant.ggml_native import GGMLNativeCodec, GGMLType
from qwen35_native import (
    Qwen35LinearAttentionGeometry,
    restore_source_matrix,
)
from validation import cross_device_tensor_match


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


def _errors(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference64 = reference.detach().to(device="cpu", dtype=torch.float64)
    actual64 = actual.detach().to(device="cpu", dtype=torch.float64)
    difference = actual64 - reference64
    denominator = float(torch.linalg.vector_norm(reference64))
    return {
        "max_absolute_error": float(difference.abs().max()),
        "mean_absolute_error": float(difference.abs().mean()),
        "mean_signed_error": float(difference.mean()),
        "rmse": float(torch.sqrt(torch.mean(difference.square()))),
        "relative_frobenius_error": (
            float(torch.linalg.vector_norm(difference)) / denominator
            if denominator else 0.0
        ),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(
            reference64.reshape(1, -1), actual64.reshape(1, -1))),
    }


def _load_oracle(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        values = {
            name: handle.get_tensor(name)
            for name in handle.keys() if name != "__metadata_json__"
        }
        metadata = json.loads(bytes(
            handle.get_tensor("__metadata_json__").tolist()).decode("utf-8"))
    required = {"input_ids", "block_input", "block_output"}
    if set(values) != required:
        raise RuntimeError(f"oracle tensor inventory differs: {sorted(values)}")
    return values, metadata


def _forward_block(
    block: torch.nn.Module,
    hidden: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(hidden.shape[1], device=device).view(1, -1)
    mask = torch.ones(
        hidden.shape[:2], dtype=torch.long, device=device)
    with torch.inference_mode():
        return block(
            hidden,
            position_embeddings=(None, None),
            attention_mask=mask,
            position_ids=positions,
            past_key_values=None,
            use_cache=False,
        )


def _assignment_records(
    entries: list[dict[str, Any]],
) -> dict[str, dict[str, GGMLType]]:
    names = [entry["destination_name"] for entry in entries]
    return {
        "uniform_q2_0": {name: GGMLType.Q2_0 for name in names},
        "uniform_q4_0": {name: GGMLType.Q4_0 for name in names},
        "alternating_q2_0_q4_0": {
            name: (GGMLType.Q2_0 if index % 2 == 0 else GGMLType.Q4_0)
            for index, name in enumerate(names)
        },
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model_dir = args.model_dir.resolve(strict=True)
    oracle_path = args.oracle.resolve(strict=True)
    manifest = _load_json(args.manifest.resolve(strict=True))
    identity = _load_json(args.identity.resolve(strict=True))
    entries = [
        entry for entry in manifest["entries"]
        if entry["destination_name"].startswith("blk.0.")
        and entry["rco_search"]
    ]
    if len(entries) != 8:
        raise RuntimeError(f"expected eight block-0 decision groups, found {len(entries)}")
    oracle, oracle_metadata = _load_oracle(oracle_path)
    if oracle_metadata["revision"] != identity["revision"]:
        raise RuntimeError("oracle and checkpoint identity revisions differ")
    if oracle_metadata["layer"] != "0":
        raise RuntimeError("oracle does not describe layer 0")

    codec = GGMLNativeCodec(args.ggml_library)
    store_path = args.store.resolve(strict=True)
    store = NativeCandidateStore(store_path, codec)
    if store.index["source"]["revision"] != identity["revision"]:
        raise RuntimeError("candidate store and checkpoint identity revisions differ")
    if store.index["tensor_count"] != len(entries):
        raise RuntimeError("candidate store does not cover every block decision group")

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(
            config, attn_implementation="eager")
    loader = SafeTensorPrefixLoader(model_dir)
    prefix = "model.language_model.layers.0"
    schema = loader.assert_prefix_schema(model, prefix)
    block_bytes = loader.load_prefix(
        model, prefix, device=device, dtype=torch.bfloat16)
    block = model.model.language_model.layers[0]
    block.eval()
    hidden = oracle["block_input"].to(device=device)
    expected = oracle["block_output"].to(device=device)
    dense_outputs = [_forward_block(block, hidden, device) for _ in range(2)]
    if not torch.equal(dense_outputs[0], dense_outputs[1]):
        raise RuntimeError("direct dense block output is not exactly reproducible")
    dense_exact = torch.equal(dense_outputs[0].cpu(), oracle["block_output"])
    dense_cross_device = cross_device_tensor_match(
        oracle["block_output"],
        dense_outputs[0],
        absolute_tolerance=args.dense_oracle_atol,
        relative_tolerance=args.dense_oracle_rtol,
    )
    if device.type == "cpu" and not dense_exact:
        raise RuntimeError("direct dense block does not reproduce retained oracle")
    if device.type == "cuda" and not dense_cross_device["within_tolerance"]:
        raise RuntimeError(
            "CUDA dense block differs from the retained CPU oracle beyond "
            f"the cross-device tolerance: {dense_cross_device}")

    geometry = Qwen35LinearAttentionGeometry.from_model_dir(model_dir)
    entry_by_name = {entry["destination_name"]: entry for entry in entries}
    results = []
    max_decoded_bytes = 0
    max_restored_bytes = 0
    max_install_bytes = 0
    for assignment_name, assignment in _assignment_records(entries).items():
        # Restore exact dense copy-only tensors and source parameters before
        # each assignment, then overwrite all eight decision groups.
        loader.load_prefix(model, prefix, device=device, dtype=torch.bfloat16)
        realized_payload = 0
        realized_aligned = 0
        installed = []
        for tensor_name, candidate_type in assignment.items():
            entry = entry_by_name[tensor_name]
            metadata = store.metadata(tensor_name, candidate_type)
            decoded = np.empty(
                tuple(reversed(metadata["gguf_shape"])), dtype=np.float32)
            store.decode_into(
                tensor_name, candidate_type, decoded,
                rows_per_chunk=args.rows_per_chunk)
            restored = np.empty_like(decoded)
            restore_source_matrix(
                decoded,
                entry["normalized_source_name"],
                geometry,
                out=restored,
            )
            module_path = entry["source_name"].rpartition(".")[0]
            target = model.get_submodule(module_path).weight
            if tuple(target.shape) != tuple(restored.shape):
                raise RuntimeError(
                    f"restored candidate shape differs for {tensor_name}")
            candidate = torch.from_numpy(restored).to(
                device=device, dtype=target.dtype)
            with torch.no_grad():
                target.copy_(candidate)
            max_decoded_bytes = max(max_decoded_bytes, decoded.nbytes)
            max_restored_bytes = max(max_restored_bytes, restored.nbytes)
            max_install_bytes = max(
                max_install_bytes, candidate.numel() * candidate.element_size())
            realized_payload += int(metadata["payload_bytes"])
            realized_aligned += int(metadata["aligned_gguf_bytes"])
            installed.append({
                "tensor": tensor_name,
                "source_tensor": entry["source_name"],
                "ggml_type": candidate_type.name,
                "payload_bytes": metadata["payload_bytes"],
                "aligned_gguf_bytes": metadata["aligned_gguf_bytes"],
                "sha256": metadata["sha256"],
            })
            del candidate, restored, decoded

        outputs = [_forward_block(block, hidden, device) for _ in range(2)]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if not torch.equal(outputs[0], outputs[1]):
            raise RuntimeError(f"assignment is not reproducible: {assignment_name}")
        if not torch.isfinite(outputs[0]).all():
            raise RuntimeError(f"assignment produced non-finite output: {assignment_name}")
        results.append({
            "name": assignment_name,
            "assignment": {
                name: candidate_type.name
                for name, candidate_type in assignment.items()
            },
            "realized_payload_bytes": realized_payload,
            "realized_aligned_gguf_bytes": realized_aligned,
            "installed": installed,
            "output_shape": list(outputs[0].shape),
            "output_dtype": str(outputs[0].dtype).replace("torch.", ""),
            "output_sha256": tensor_sha256(outputs[0]),
            "exactly_reproducible": True,
            "finite": True,
            "errors": _errors(expected, outputs[0]),
        })
        del outputs

    released_bytes = loader.release_prefix(model, prefix)
    gc.collect()
    cuda = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "allocated_peak_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0),
        "reserved_peak_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0),
    }
    if device.type == "cuda":
        cuda["device_name"] = torch.cuda.get_device_name(device)

    return {
        "schema": 1,
        "status": "pass",
        "scope": (
            "Qwen3.5-2B layer-0 numerical and memory comparison for complete "
            "native assignments"
            + (" on CUDA" if device.type == "cuda" else " on CPU")
            + "; this is not an end-to-end loss or 35B gate"
        ),
        "source": {
            "repo_id": identity["repo_id"],
            "revision": identity["revision"],
            "model_dir": str(model_dir),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device": str(device),
            "cuda": cuda,
        },
        "oracle": {
            "path": str(oracle_path),
            "bytes": oracle_path.stat().st_size,
            "sha256": _sha256_file(oracle_path),
            "metadata": oracle_metadata,
            "dense_direct_matches_retained_oracle_exactly": dense_exact,
            "dense_direct_cross_device_comparison": dense_cross_device,
            "dense_output_sha256": tensor_sha256(dense_outputs[0]),
        },
        "block": {
            "prefix": prefix,
            "schema": schema,
            "resident_bf16_bytes": block_bytes,
            "released_bytes": released_bytes,
            "released_to_meta": all(
                parameter.device.type == "meta"
                for parameter in block.parameters()),
        },
        "candidate_store": {
            "path": str(store_path),
            "index_sha256": _sha256_file(
                store_path / "native-candidate-index.json"),
            "persistent_decoded_cache": False,
            "max_decoded_fp32_bytes": max_decoded_bytes,
            "max_inverse_transform_fp32_bytes": max_restored_bytes,
            "max_install_bf16_bytes": max_install_bytes,
        },
        "assignments": results,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        ),
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
    parser.add_argument("--dense-oracle-atol", type=float, default=0.02)
    parser.add_argument("--dense-oracle-rtol", type=float, default=0.02)
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/qwen35_2b_block0_native_output.json"))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(args)
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "device": report["environment"]["device"],
        "assignments": {
            value["name"]: value["errors"]["relative_frobenius_error"]
            for value in report["assignments"]
        },
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
