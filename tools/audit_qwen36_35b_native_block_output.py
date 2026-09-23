#!/usr/bin/env python3
"""Evaluate complete native assignments against the genuine 35B block oracle."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
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
from model_adapter import get_model_adapter
from native_store import NativeCandidateStore
from quant.ggml_native import GGMLNativeCodec, GGMLType
from qwen35_native import Qwen35LinearAttentionGeometry, matrix_permutations


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
    mask = torch.ones(hidden.shape[:2], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = block(
            hidden,
            position_embeddings=(None, None),
            attention_mask=mask,
            position_ids=positions,
            past_key_values=None,
            use_cache=False,
        )
    return output[0] if isinstance(output, tuple) else output


def _errors(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference64 = reference.detach().to(device="cpu", dtype=torch.float64)
    actual64 = actual.detach().to(device="cpu", dtype=torch.float64)
    difference = actual64 - reference64
    reference_norm = float(torch.linalg.vector_norm(reference64))
    actual_norm = float(torch.linalg.vector_norm(actual64))
    dot = float(torch.sum(reference64 * actual64))
    return {
        "max_absolute_error": float(difference.abs().max()),
        "mean_absolute_error": float(difference.abs().mean()),
        "mean_signed_error": float(difference.mean()),
        "rmse": float(torch.sqrt(torch.mean(difference.square()))),
        "relative_frobenius_error": (
            float(torch.linalg.vector_norm(difference)) / reference_norm
            if reference_norm else 0.0),
        "cosine_similarity": (
            dot / (reference_norm * actual_norm)
            if reference_norm and actual_norm else 0.0),
    }


def _assignments(entries: list[dict[str, Any]]) -> dict[str, dict[str, GGMLType]]:
    names = [entry["destination_name"] for entry in entries]
    return {
        "uniform_q2_0": {name: GGMLType.Q2_0 for name in names},
        "uniform_q4_0": {name: GGMLType.Q4_0 for name in names},
        "alternating_q2_0_q4_0": {
            name: GGMLType.Q2_0 if index % 2 == 0 else GGMLType.Q4_0
            for index, name in enumerate(names)
        },
    }


def _expected_hashes(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        name, separator, digest = value.partition("=")
        if not separator or not name or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(
                "expected output hashes must use assignment=64-lowercase-hex")
        if name in result:
            raise ValueError(f"duplicate expected output hash for {name}")
        result[name] = digest
    return result


def _install_candidate(
    *,
    model: torch.nn.Module,
    store: NativeCandidateStore,
    entry: dict[str, Any],
    candidate_type: GGMLType,
    geometry: Qwen35LinearAttentionGeometry,
    rows_per_chunk: int,
) -> dict[str, int]:
    target = model.get_parameter(entry["source_name"])
    source_shape = tuple(int(value) for value in entry["source_shape"])
    candidate_shape = tuple(int(value) for value in entry.get(
        "candidate_source_shape", source_shape))
    if tuple(target.shape) != source_shape:
        raise RuntimeError(
            f"target shape changed for {entry['source_name']}: {tuple(target.shape)}")
    max_decoded = 0
    max_install = 0
    installed_rows = 0

    if len(candidate_shape) == 2:
        row_order, column_order = matrix_permutations(
            entry["normalized_source_name"], candidate_shape, geometry)
        if row_order is not None and column_order is not None:
            raise RuntimeError("simultaneous row/column permutations are unsupported")
        for start, decoded in store.iter_decoded_rows(
            entry["destination_name"], candidate_type,
            rows_per_chunk=rows_per_chunk,
        ):
            stop = start + decoded.shape[0]
            values = torch.from_numpy(decoded).to(
                device=target.device, dtype=target.dtype)
            with torch.no_grad():
                if row_order is not None:
                    indices = torch.from_numpy(row_order[start:stop]).to(
                        device=target.device)
                    target.index_copy_(0, indices, values)
                elif column_order is not None:
                    indices = torch.from_numpy(column_order).to(
                        device=target.device)
                    target[start:stop, indices] = values
                else:
                    target[start:stop].copy_(values)
            installed_rows += decoded.shape[0]
            max_decoded = max(max_decoded, decoded.nbytes)
            max_install = max(max_install, values.numel() * values.element_size())
    elif len(candidate_shape) == 3:
        source_view = entry.get("source_view")
        if source_view is None:
            view_start, view_stop = 0, source_shape[1]
        else:
            if source_view.get("axis") != 1:
                raise RuntimeError(f"unsupported source view: {source_view}")
            view_start = int(source_view["start"])
            view_stop = int(source_view["stop"])
        rows_per_expert = candidate_shape[1]
        for flat_start, decoded in store.iter_decoded_rows(
            entry["destination_name"], candidate_type,
            rows_per_chunk=rows_per_chunk,
        ):
            chunk_offset = 0
            while chunk_offset < decoded.shape[0]:
                flat_row = flat_start + chunk_offset
                expert, row = divmod(flat_row, rows_per_expert)
                run = min(decoded.shape[0] - chunk_offset, rows_per_expert - row)
                values = torch.from_numpy(
                    decoded[chunk_offset:chunk_offset + run]).to(
                        device=target.device, dtype=target.dtype)
                with torch.no_grad():
                    target[
                        expert,
                        view_start + row:view_start + row + run,
                        :,
                    ].copy_(values)
                installed_rows += run
                chunk_offset += run
                max_install = max(
                    max_install, values.numel() * values.element_size())
            max_decoded = max(max_decoded, decoded.nbytes)
        if view_stop - view_start != rows_per_expert:
            raise RuntimeError("candidate/source-view row counts differ")
    else:
        raise RuntimeError(f"unsupported candidate rank: {candidate_shape}")

    expected_rows = math.prod(candidate_shape[:-1])
    if installed_rows != expected_rows:
        raise RuntimeError(
            f"installed {installed_rows} rows for {entry['destination_name']}; "
            f"expected {expected_rows}")
    return {
        "installed_rows": installed_rows,
        "max_decoded_fp32_bytes": max_decoded,
        "max_install_bf16_bytes": max_install,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    expected_hashes = _expected_hashes(args.expected_output_sha256)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model_dir = args.model_dir.resolve(strict=True)
    identity = _load_json(args.identity.resolve(strict=True))
    manifest = _load_json(args.manifest.resolve(strict=True))
    oracle_path = args.oracle.resolve(strict=True)
    oracle, oracle_metadata = _load_oracle(oracle_path)
    if oracle_metadata["revision"] != identity["revision"]:
        raise RuntimeError("oracle and checkpoint identity revisions differ")
    if oracle_metadata["layer"] != "0":
        raise RuntimeError("oracle does not describe block 0")
    entries = [
        entry for entry in manifest["entries"]
        if entry["destination_name"].startswith("blk.0.")
        and entry["rco_search"]
    ]
    if len(entries) != 13:
        raise RuntimeError(f"expected 13 block decision groups, found {len(entries)}")

    codec = GGMLNativeCodec(args.ggml_library)
    store_path = args.store.resolve(strict=True)
    store = NativeCandidateStore(store_path, codec)
    if store.index["source"]["revision"] != identity["revision"]:
        raise RuntimeError("candidate store and checkpoint revisions differ")
    if store.index["tensor_count"] != len(entries):
        raise RuntimeError("candidate store does not cover every block group")

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(
            config, attn_implementation="eager")
    model.eval()
    adapter = get_model_adapter(model)
    loader = SafeTensorPrefixLoader(model_dir)
    loader.move_runtime_buffers(model, device)
    prefix = f"{adapter.layers_path}.0"
    schema = loader.assert_prefix_schema(model, prefix)
    block_bytes = loader.load_prefix(
        model, prefix, device=device, dtype=torch.bfloat16)
    block = adapter.layers[0]
    block.eval()
    hidden = oracle["block_input"].to(device=device)
    expected = oracle["block_output"]
    dense_outputs = [
        _forward_block(block, hidden, device).detach().cpu().contiguous()
        for _ in range(2)
    ]
    if not torch.equal(dense_outputs[0], dense_outputs[1]):
        raise RuntimeError("direct dense block output is not exactly reproducible")
    if not torch.equal(dense_outputs[0], expected):
        raise RuntimeError("direct dense block does not reproduce retained oracle")

    geometry = Qwen35LinearAttentionGeometry.from_model_dir(model_dir)
    entry_by_name = {entry["destination_name"]: entry for entry in entries}
    results = []
    max_decoded = 0
    max_install = 0
    for assignment_name, assignment in _assignments(entries).items():
        loader.load_prefix(model, prefix, device=device, dtype=torch.bfloat16)
        installed = []
        realized_payload = 0
        realized_aligned = 0
        for tensor_name, candidate_type in assignment.items():
            entry = entry_by_name[tensor_name]
            install_stats = _install_candidate(
                model=model,
                store=store,
                entry=entry,
                candidate_type=candidate_type,
                geometry=geometry,
                rows_per_chunk=args.rows_per_chunk,
            )
            metadata = store.metadata(tensor_name, candidate_type)
            max_decoded = max(
                max_decoded, install_stats["max_decoded_fp32_bytes"])
            max_install = max(
                max_install, install_stats["max_install_bf16_bytes"])
            realized_payload += int(metadata["payload_bytes"])
            realized_aligned += int(metadata["aligned_gguf_bytes"])
            installed.append({
                "tensor": tensor_name,
                "source_tensor": entry["source_name"],
                "source_view": entry.get("source_view"),
                "ggml_type": candidate_type.name,
                "payload_bytes": metadata["payload_bytes"],
                "aligned_gguf_bytes": metadata["aligned_gguf_bytes"],
                "sha256": metadata["sha256"],
                **install_stats,
            })

        outputs = [
            _forward_block(block, hidden, device).detach().cpu().contiguous()
            for _ in range(2)
        ]
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

    actual_hashes = {item["name"]: item["output_sha256"] for item in results}
    if expected_hashes and actual_hashes != expected_hashes:
        raise RuntimeError(
            f"assignment output hashes differ from independent process: "
            f"expected {expected_hashes}, actual {actual_hashes}")

    released_bytes = loader.release_prefix(model, prefix)
    gc.collect()
    cuda: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "allocated_peak_bytes": 0,
        "reserved_peak_bytes": 0,
    }
    if device.type == "cuda":
        cuda.update({
            "device_name": torch.cuda.get_device_name(device),
            "allocated_peak_bytes": torch.cuda.max_memory_allocated(device),
            "reserved_peak_bytes": torch.cuda.max_memory_reserved(device),
        })
    return {
        "schema": 1,
        "status": "pass",
        "scope": (
            "genuine Qwen3.6-35B-A3B block-0 output comparison for complete "
            "BF16-derived native assignments; this is not an end-to-end, "
            "authentic-GSQ, llama.cpp-matmul, or CUDA gate"
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
            "dense_direct_matches_retained_oracle_exactly": True,
            "dense_output_sha256": tensor_sha256(dense_outputs[0]),
        },
        "block": {
            "prefix": prefix,
            "schema": schema,
            "resident_bf16_bytes": block_bytes,
            "released_bytes": released_bytes,
            "released_to_meta": all(
                parameter.device.type == "meta"
                for parameter in adapter.layers[0].parameters()),
        },
        "candidate_store": {
            "path": str(store_path),
            "index_sha256": _sha256_file(
                store_path / "native-candidate-index.json"),
            "persistent_decoded_cache": False,
            "rows_per_chunk": args.rows_per_chunk,
            "max_decoded_fp32_bytes": max_decoded,
            "max_install_bf16_bytes": max_install,
        },
        "assignments": results,
        "cross_process_reproducibility": {
            "expected_output_sha256": expected_hashes or None,
            "all_assignment_output_hashes_match": bool(expected_hashes),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument(
        "--expected-output-sha256", action="append", default=[],
        metavar="ASSIGNMENT=SHA256")
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/qwen36_35b_block0_native_output.json"))
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
            item["name"]: item["errors"]["relative_frobenius_error"]
            for item in report["assignments"]
        },
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
