#!/usr/bin/env python3
"""Build the missing BF16-derived routed-expert Q2_0 override store."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_qwen36_35b_native_block import _validate_candidate  # noqa: E402
from audit_qwen36_35b_native_database import _bf16_chunks  # noqa: E402
from native_store import (  # noqa: E402
    NATIVE_CANDIDATE_INDEX,
    NativeCandidateStore,
    NativeCandidateStoreWriter,
)
from quant.ggml_native import GGMLNativeCodec, GGMLType  # noqa: E402
from qwen35_native import SafetensorGGUFRowSource  # noqa: E402


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


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    model_dir = args.model_dir.resolve(strict=True)
    identity_path = args.identity.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    output_store = args.output_store.resolve()
    report_path = args.report.resolve()
    identity = _load_json(identity_path)
    manifest = _load_json(manifest_path)
    entries = sorted(
        (
            entry for entry in manifest["entries"]
            if entry.get("rco_search")
            and entry["source_category"] == "routed_expert"
        ),
        key=lambda entry: entry["destination_name"],
    )
    if len(entries) != 120:
        raise RuntimeError(f"expected 120 routed-expert groups, found {len(entries)}")
    problem = {
        "dense_revision": identity["revision"],
        "identity_sha256": _sha256_file(identity_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "candidate_type": "Q2_0",
        "candidate_count": len(entries),
        "rows_per_chunk": args.rows_per_chunk,
    }
    problem_sha256 = hashlib.sha256(json.dumps(
        problem, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source_descriptor = {
        "kind": "bf16_native_q2_routed_overlay",
        **problem,
        "problem_sha256": problem_sha256,
    }
    codec = GGMLNativeCodec(args.ggml_library.resolve(strict=True))
    index_path = output_store / NATIVE_CANDIDATE_INDEX
    writer = NativeCandidateStoreWriter(
        output_store,
        codec,
        source=source_descriptor,
        resume=index_path.exists(),
    )
    dense_source = SafetensorGGUFRowSource(model_dir)
    generation_stats: dict[str, int] = {}
    for index, entry in enumerate(entries, start=1):
        provenance = {
            "kind": "bf16_native_quantization",
            "dense_revision": identity["revision"],
            "source_tensor": entry["source_name"],
            "source_shard": entry["source_shard"],
            "source_view": entry.get("source_view"),
            "converter_transforms": entry["converter_transforms"],
            "rows_per_chunk": args.rows_per_chunk,
            "overlay_purpose": "uniform_bf16_derived_q2_control",
        }
        writer.write_packed_chunks(
            entry["destination_name"],
            GGMLType.Q2_0,
            entry["destination_gguf_shape"],
            _bf16_chunks(
                dense_source, codec, entry, GGMLType.Q2_0,
                rows_per_chunk=args.rows_per_chunk, stats=generation_stats),
            provenance=provenance,
        )
        writer.finalize()
        if shutil.disk_usage(output_store.parent).free < args.min_free_bytes:
            raise RuntimeError("free-space reserve crossed while building overlay")
        print(json.dumps({
            "phase": "generation", "candidate": index,
            "candidate_count": len(entries), "tensor": entry["destination_name"],
        }, sort_keys=True), flush=True)

    store = NativeCandidateStore(output_store, codec)
    if int(store.index["tensor_count"]) != 120:
        raise RuntimeError("overlay tensor count differs")
    validation_stats: dict[str, int] = {}
    validation = []
    for index, entry in enumerate(entries, start=1):
        record = _validate_candidate(
            dense_source, store, codec, entry, GGMLType.Q2_0,
            rows_per_chunk=args.validation_rows_per_chunk,
            stats=validation_stats,
        )
        validation.append(record)
        print(json.dumps({
            "phase": "validation", "candidate": index,
            "candidate_count": len(entries), "tensor": entry["destination_name"],
        }, sort_keys=True), flush=True)

    report = {
        "schema": "rco.qwen36.bf16_q2_routed_overlay.v1",
        "status": "complete",
        "scope": (
            "BF16-derived Q2_0 replacements for the 120 routed-expert groups "
            "whose production-store Q2_0 candidates are authentic-GSQ-derived"
        ),
        "problem": problem,
        "problem_sha256": problem_sha256,
        "store": str(output_store),
        "store_index_sha256": _sha256_file(index_path),
        "candidate_count": len(validation),
        "payload_bytes": sum(item["payload_bytes"] for item in validation),
        "all_payloads_equal_independent_bf16_quantization": all(
            item["stored_bytes_equal_independent_chunking"] for item in validation),
        "writer": {
            "resumed_candidate_count": writer.resumed_candidate_count,
            "reused_candidate_count": writer.reused_candidate_count,
            "written_candidate_count": writer.written_candidate_count,
        },
        "generation_stats": generation_stats,
        "validation_stats": validation_stats,
        "candidates": validation,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(report_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-store", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument("--validation-rows-per-chunk", type=int, default=16)
    parser.add_argument("--min-free-bytes", type=int, default=32 << 30)
    args = parser.parse_args()
    if min(
        args.rows_per_chunk,
        args.validation_rows_per_chunk,
        args.min_free_bytes,
    ) < 1:
        parser.error("numeric arguments must be positive")
    return args


if __name__ == "__main__":
    result = build(_parse_args())
    print(json.dumps({
        "status": result["status"],
        "candidate_count": result["candidate_count"],
        "payload_bytes": result["payload_bytes"],
    }, indent=2, sort_keys=True))
