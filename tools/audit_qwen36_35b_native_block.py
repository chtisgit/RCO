#!/usr/bin/env python3
"""Generate and validate a bounded native candidate store for Qwen3.6 block 0."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import resource
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from native_store import NativeCandidateStore, NativeCandidateStoreWriter
from quant.ggml_native import GGMLNativeCodec, GGMLType
from qwen35_native import SafetensorGGUFRowSource, generate_native_block_candidates


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


def _shape_coverage(
    entries: list[dict[str, Any]], codec: GGMLNativeCodec,
) -> dict[str, Any]:
    counts = Counter(
        tuple(int(value) for value in entry["destination_gguf_shape"])
        for entry in entries if entry["rco_search"]
    )
    rows = []
    tested_widths: dict[int, dict[str, Any]] = {}
    for shape, count in sorted(counts.items()):
        row_count = math.prod(shape[1:])
        types = {}
        for candidate_type in (GGMLType.Q2_0, GGMLType.Q4_0):
            geometry = codec.geometry(candidate_type, shape[0])
            types[candidate_type.name] = {
                **geometry,
                "payload_bytes": row_count * int(geometry["row_size"]),
            }
        rows.append({
            "gguf_shape": list(shape),
            "tensor_count": count,
            "row_width": shape[0],
            "row_count": row_count,
            "candidate_types": types,
        })
        if shape[0] not in tested_widths:
            source = np.linspace(-1.0, 1.0, 2 * shape[0], dtype=np.float32).reshape(
                2, shape[0])
            type_tests = {}
            for candidate_type in (GGMLType.Q2_0, GGMLType.Q4_0):
                packed = codec.quantize_rows(source, candidate_type)
                decoded = np.empty_like(source)
                codec.dequantize_rows_into(packed, candidate_type, decoded)
                type_tests[candidate_type.name] = {
                    "payload_bytes": len(packed),
                    "finite": bool(np.isfinite(decoded).all()),
                    "max_absolute_error": float(np.max(np.abs(decoded - source))),
                }
            tested_widths[shape[0]] = {
                "row_width": shape[0],
                "rows_tested": 2,
                "candidate_types": type_tests,
            }
    return {
        "unique_gguf_shape_count": len(rows),
        "unique_row_width_count": len(tested_widths),
        "shapes": rows,
        "cpu_native_round_trip_by_row_width": [
            tested_widths[key] for key in sorted(tested_widths)
        ],
    }


def _validate_candidate(
    source: SafetensorGGUFRowSource,
    store: NativeCandidateStore,
    codec: GGMLNativeCodec,
    entry: dict[str, Any],
    candidate_type: GGMLType,
    *,
    rows_per_chunk: int,
    stats: dict[str, int],
) -> dict[str, Any]:
    metadata = store.metadata(entry["destination_name"], candidate_type)
    packed_chunks = store.iter_payload(
        entry["destination_name"], candidate_type,
        chunk_bytes=rows_per_chunk * int(metadata["row_size"]),
    )
    count = 0
    max_abs = 0.0
    sum_abs = 0.0
    sum_signed = 0.0
    sum_squared = 0.0
    reference_sum_squared = 0.0
    compared_bytes = 0
    independent_digest = hashlib.sha256()
    sentinel = object()
    source_rows = source.iter_rows(
        entry, rows_per_chunk=rows_per_chunk, stats=stats)
    for rows, packed in itertools.zip_longest(
        source_rows, packed_chunks, fillvalue=sentinel,
    ):
        if rows is sentinel or packed is sentinel:
            raise RuntimeError(
                f"source/payload chunk count differs for "
                f"{entry['destination_name']}/{candidate_type.name}")
        expected = codec.quantize_rows(rows, candidate_type)
        if packed != expected:
            raise RuntimeError(
                f"stored native bytes differ from independent quantization for "
                f"{entry['destination_name']}/{candidate_type.name}")
        independent_digest.update(expected)
        compared_bytes += len(expected)
        decoded = np.empty_like(rows)
        codec.dequantize_rows_into(packed, candidate_type, decoded)
        reference64 = rows.astype(np.float64)
        difference = decoded.astype(np.float64) - reference64
        absolute = np.abs(difference)
        max_abs = max(max_abs, float(np.max(absolute)))
        sum_abs += float(np.sum(absolute, dtype=np.float64))
        sum_signed += float(np.sum(difference, dtype=np.float64))
        sum_squared += float(np.sum(np.square(difference), dtype=np.float64))
        reference_sum_squared += float(np.sum(
            np.square(reference64), dtype=np.float64))
        count += rows.size
        stats["max_validation_scratch_bytes"] = max(
            stats.get("max_validation_scratch_bytes", 0),
            rows.nbytes + decoded.nbytes + reference64.nbytes
            + difference.nbytes + absolute.nbytes,
        )
    if compared_bytes != metadata["payload_bytes"]:
        raise RuntimeError("validated payload byte count differs from store metadata")
    if independent_digest.hexdigest() != metadata["sha256"]:
        raise RuntimeError("independent candidate digest differs from store metadata")
    return {
        "tensor": entry["destination_name"],
        "source_tensor": entry["source_name"],
        "source_view": entry.get("source_view"),
        "source_shape": entry["source_shape"],
        "gguf_shape": metadata["gguf_shape"],
        "ggml_type": candidate_type.name,
        "ggml_type_id": int(candidate_type),
        "payload_bytes": metadata["payload_bytes"],
        "aligned_gguf_bytes": metadata["aligned_gguf_bytes"],
        "sha256": metadata["sha256"],
        "stored_bytes_equal_independent_chunking": True,
        "elements": count,
        "errors": {
            "max_absolute_error": max_abs,
            "mean_absolute_error": sum_abs / count,
            "mean_signed_error": sum_signed / count,
            "rmse": math.sqrt(sum_squared / count),
            "relative_frobenius_error": (
                math.sqrt(sum_squared / reference_sum_squared)
                if reference_sum_squared else 0.0),
        },
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    model_dir = args.model_dir.resolve(strict=True)
    identity = _load_json(args.identity.resolve(strict=True))
    manifest = _load_json(args.manifest.resolve(strict=True))
    if identity.get("status") != "pass" or manifest.get("status") != "pass":
        raise RuntimeError("identity and canonical manifest must both pass")
    if manifest["source"]["revision"] != identity["revision"]:
        raise RuntimeError("identity and manifest revisions differ")
    llama_cpp = args.llama_cpp.resolve(strict=True)
    import subprocess
    llama_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=llama_cpp, check=True,
        text=True, capture_output=True).stdout.strip()
    if llama_revision != manifest["llama_cpp_revision"]:
        raise RuntimeError("llama.cpp checkout differs from canonical manifest")

    all_searched = [entry for entry in manifest["entries"] if entry["rco_search"]]
    entries = [
        entry for entry in manifest["entries"]
        if entry["destination_name"].startswith(f"blk.{args.block}.")
    ]
    searched = [entry for entry in entries if entry["rco_search"]]
    copied = [entry for entry in entries if not entry["rco_search"]]
    if (len(entries), len(searched), len(copied)) != (19, 13, 6):
        raise RuntimeError(
            f"unexpected block inventory: {len(entries)}/"
            f"{len(searched)}/{len(copied)}")
    codec = GGMLNativeCodec(args.ggml_library)
    full_shape_coverage = _shape_coverage(all_searched, codec)
    block_shape_coverage = _shape_coverage(searched, codec)
    store_output = args.store_output.resolve()
    if store_output.exists():
        raise FileExistsError(
            f"candidate store already exists; refusing to replace it: {store_output}")
    store_output.parent.mkdir(parents=True, exist_ok=True)

    file_records = {record["path"]: record for record in identity["files"]}
    relevant_shards = sorted({entry["source_shard"] for entry in entries})
    source = SafetensorGGUFRowSource(model_dir)
    validation_stats: dict[str, int] = {
        "chunk_count": 0,
        "max_dense_chunk_bytes": 0,
        "max_source_rows_per_chunk": 0,
        "max_validation_scratch_bytes": 0,
    }
    temporary_parent = (
        None if args.temporary_parent is None
        else str(args.temporary_parent.resolve(strict=True))
    )
    with tempfile.TemporaryDirectory(
        prefix="rco-qwen36-35b-block-", dir=temporary_parent,
    ) as directory_name:
        store_path = Path(directory_name) / "native-store"
        writer = NativeCandidateStoreWriter(
            store_path,
            codec,
            source={
                "repo_id": identity["repo_id"],
                "revision": identity["revision"],
                "block": args.block,
                "llama_cpp_revision": llama_revision,
                "source_shards": {
                    name: file_records[name]["sha256"] for name in relevant_shards
                },
            },
        )
        generation = generate_native_block_candidates(
            model_dir,
            entries,
            writer,
            codec,
            rows_per_chunk=args.rows_per_chunk,
        )
        store = NativeCandidateStore(store_path, codec)
        validations = []
        for entry in searched:
            for candidate_type in (GGMLType.Q2_0, GGMLType.Q4_0):
                validations.append(_validate_candidate(
                    source, store, codec, entry, candidate_type,
                    rows_per_chunk=args.validation_rows_per_chunk,
                    stats=validation_stats,
                ))
        if store.index["tensor_count"] != len(searched):
            raise RuntimeError("native store tensor count is incomplete")
        if store.index["candidate_count"] != len(validations):
            raise RuntimeError("native store candidate count is incomplete")
        index_sha256 = _sha256_file(
            store_path / "native-candidate-index.json")
        os.replace(store_path, store_output)

    block_source_bytes = sum(
        item["logical_bytes"] for item in identity["text_inventory"]
        if item["name"].startswith(
            f"model.language_model.layers.{args.block}.")
    )
    total_payload = sum(item["payload_bytes"] for item in validations)
    return {
        "schema": 1,
        "status": "pass",
        "scope": (
            "complete BF16-derived native Q2_0/Q4_0 candidate store for "
            "genuine Qwen3.6-35B-A3B block 0; CUDA and authentic GSQ import "
            "remain separate gates"
        ),
        "source": {
            "repo_id": identity["repo_id"],
            "revision": identity["revision"],
            "model_dir": str(model_dir),
            "block_source_logical_bytes": block_source_bytes,
            "source_shards": {
                name: file_records[name]["sha256"] for name in relevant_shards
            },
        },
        "llama_cpp": {"path": str(llama_cpp), "revision": llama_revision},
        "ggml_library": {
            "path": str(codec.library_path),
            "sha256": codec.library_sha256,
        },
        "shape_coverage": {
            "full_model": full_shape_coverage,
            "block": block_shape_coverage,
            "cpu_quantize_dequantize": "pass",
            "cuda": "not_run_cuda_initialization_failed",
        },
        "block": {
            "index": args.block,
            "canonical_tensor_count": len(entries),
            "searched_tensor_count": len(searched),
            "copied_tensor_count": len(copied),
            "complete_manifest_coverage": True,
            "copied": [
                {
                    "tensor": entry["destination_name"],
                    "source_tensor": entry["source_name"],
                    "copy_reason": entry["copy_reason"],
                    "gguf_shape": entry["destination_gguf_shape"],
                    "ggml_type": entry["destination_ggml_type"],
                }
                for entry in copied
            ],
        },
        "candidate_store": {
            "path": str(store_output),
            "index": str(store_output / "native-candidate-index.json"),
            "index_sha256": index_sha256,
            **generation,
            "total_payload_bytes": total_payload,
            "total_aligned_gguf_bytes": sum(
                item["aligned_gguf_bytes"] for item in validations),
        },
        "validation": {
            "candidate_count": len(validations),
            "validation_rows_per_chunk": args.validation_rows_per_chunk,
            "all_stored_bytes_equal_independent_chunking": True,
            **validation_stats,
            "candidates": validations,
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
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--store-output", type=Path, required=True)
    parser.add_argument("--temporary-parent", type=Path)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument("--validation-rows-per-chunk", type=int, default=16)
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/qwen36_35b_block0_native_candidates.json"))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(args)
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "store": report["candidate_store"]["path"],
        "candidate_count": report["validation"]["candidate_count"],
        "payload_bytes": report["candidate_store"]["total_payload_bytes"],
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
