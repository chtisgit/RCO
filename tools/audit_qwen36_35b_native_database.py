#!/usr/bin/env python3
"""Build and validate the complete mixed-provenance 35B native database."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

import torch
import transformers

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_qwen36_35b_gsq_block_import import _gsq_chunks, _projection
from audit_qwen36_35b_native_block import _shape_coverage, _validate_candidate
from gsq_q2 import GSQCheckpoint
from native_store import NativeCandidateStore, NativeCandidateStoreWriter
from quant.ggml_native import GGMLNativeCodec, GGMLType
from qwen35_native import SafetensorGGUFRowSource


_BLOCK = re.compile(r"^blk\.(\d+)\.")
_CANDIDATE_TYPES = (GGMLType.Q2_0, GGMLType.Q4_0)


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


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _projected_bytes(
    entries: list[dict[str, Any]], codec: GGMLNativeCodec,
) -> dict[str, Any]:
    by_type = {candidate_type.name: 0 for candidate_type in _CANDIDATE_TYPES}
    by_unit: dict[str, dict[str, int]] = {}
    for entry in entries:
        shape = tuple(int(value) for value in entry["destination_gguf_shape"])
        row_count = math.prod(shape[1:])
        match = _BLOCK.match(entry["destination_name"])
        unit = f"block_{int(match.group(1))}" if match else entry["destination_name"]
        unit_costs = by_unit.setdefault(
            unit, {candidate_type.name: 0 for candidate_type in _CANDIDATE_TYPES})
        for candidate_type in _CANDIDATE_TYPES:
            value = row_count * int(codec.geometry(
                candidate_type, shape[0])["row_size"])
            by_type[candidate_type.name] += value
            unit_costs[candidate_type.name] += value
    return {
        "by_type": by_type,
        "combined_payload_bytes": sum(by_type.values()),
        "by_unit": by_unit,
    }


def _units(entries: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    blocks: dict[int, list[dict[str, Any]]] = {}
    globals_: list[tuple[str, list[dict[str, Any]]]] = []
    for entry in entries:
        match = _BLOCK.match(entry["destination_name"])
        if match:
            blocks.setdefault(int(match.group(1)), []).append(entry)
        elif entry.get("rco_search"):
            globals_.append((entry["destination_name"], [entry]))
    if sorted(blocks) != list(range(40)):
        raise RuntimeError(f"manifest block inventory differs: {sorted(blocks)}")
    return [
        (f"block_{index}", blocks[index]) for index in range(40)
    ] + sorted(globals_)


def _is_authentic_gsq(entry: dict[str, Any], candidate_type: GGMLType) -> bool:
    return (
        candidate_type == GGMLType.Q2_0
        and entry["source_category"] == "routed_expert"
    )


def _bf16_chunks(
    source: SafetensorGGUFRowSource,
    codec: GGMLNativeCodec,
    entry: dict[str, Any],
    candidate_type: GGMLType,
    *,
    rows_per_chunk: int,
    stats: dict[str, int],
) -> Iterator[bytes]:
    for rows in source.iter_rows(
        entry, rows_per_chunk=rows_per_chunk, stats=stats,
    ):
        yield codec.quantize_rows(rows, candidate_type)


def _source_descriptor(
    identity: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
    gsq_revision: str,
    llama_revision: str,
) -> dict[str, Any]:
    return {
        "dense_repo_id": identity["repo_id"],
        "dense_revision": identity["revision"],
        "gsq_repo_id": "ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ",
        "gsq_revision": gsq_revision,
        "llama_cpp_revision": llama_revision,
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest_entry_count": len(manifest["entries"]),
        "candidate_policy": (
            "authentic GSQ-derived Q2_0 for every routed-expert aggregate; "
            "BF16-derived Q2_0 elsewhere; BF16-derived Q4_0 everywhere"
        ),
    }


def _validate_gsq_candidate(
    checkpoint: GSQCheckpoint,
    codec: GGMLNativeCodec,
    store: NativeCandidateStore,
    entry: dict[str, Any],
    *,
    accepted_bound: float,
) -> dict[str, Any]:
    match = _BLOCK.match(entry["destination_name"])
    if match is None:
        raise ValueError("GSQ candidate is outside a decoder block")
    projection = _projection(entry)
    expert_count = int(entry["candidate_source_shape"][0])
    records: list[dict[str, Any]] = []
    expected_digest = hashlib.sha256()
    expected_bytes = 0
    for payload in _gsq_chunks(
        checkpoint,
        codec,
        layer=int(match.group(1)),
        projection=projection,
        expert_count=expert_count,
        accepted_max_weight_error=accepted_bound,
        records=records,
    ):
        expected_digest.update(payload)
        expected_bytes += len(payload)
    metadata = store.metadata(entry["destination_name"], GGMLType.Q2_0)
    actual_digest = hashlib.sha256()
    actual_bytes = 0
    for payload in store.iter_payload(entry["destination_name"], GGMLType.Q2_0):
        actual_digest.update(payload)
        actual_bytes += len(payload)
    if expected_bytes != int(metadata["payload_bytes"]) or actual_bytes != expected_bytes:
        raise RuntimeError(
            f"GSQ payload byte count differs for {entry['destination_name']}")
    if not (
        expected_digest.hexdigest()
        == actual_digest.hexdigest()
        == metadata["sha256"]
    ):
        raise RuntimeError(
            f"GSQ payload hash differs for {entry['destination_name']}")
    max_error_record = max(
        records, key=lambda record: record["max_weight_absolute_error"])
    max_error = float(max_error_record["max_weight_absolute_error"])
    if max_error > accepted_bound:
        raise RuntimeError(
            f"GSQ candidate exceeds accepted error: {entry['destination_name']}")
    return {
        "tensor": entry["destination_name"],
        "source_tensor": entry["source_name"],
        "ggml_type": GGMLType.Q2_0.name,
        "ggml_type_id": int(GGMLType.Q2_0),
        "origin": "direct_authentic_gsq2_to_q2_0",
        "projection": projection,
        "expert_count": expert_count,
        "payload_bytes": actual_bytes,
        "aligned_gguf_bytes": metadata["aligned_gguf_bytes"],
        "sha256": metadata["sha256"],
        "source_group_count": sum(record["group_count"] for record in records),
        "scale_rounding_count": sum(
            record["scale_rounding_count"] for record in records),
        "scale_underflow_count": sum(
            record["scale_underflow_count"] for record in records),
        "max_scale_absolute_error": max(
            record["max_scale_absolute_error"] for record in records),
        "max_weight_absolute_error": max_error,
        "max_error_expert": int(max_error_record["expert"]),
        "stock_decodes_match_mapped_gsq_exactly": True,
        "stored_payload_matches_direct_mapping_exactly": True,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    model_dir = args.model_dir.resolve(strict=True)
    identity_path = args.identity.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    relationship_path = args.relationship.resolve(strict=True)
    gsq_audit_path = args.gsq_audit.resolve(strict=True)
    acceptance_path = args.acceptance.resolve(strict=True)
    identity = _load_json(identity_path)
    manifest = _load_json(manifest_path)
    relationship = _load_json(relationship_path)
    gsq_audit = _load_json(gsq_audit_path)
    acceptance = _load_json(acceptance_path)
    if identity.get("status") != "pass" or manifest.get("status") != "pass":
        raise RuntimeError("identity and canonical manifest must both pass")
    if manifest["source"]["revision"] != identity["revision"]:
        raise RuntimeError("identity and manifest revisions differ")
    revisions = gsq_audit["source_revision_evidence"]
    if len(revisions) != 1:
        raise RuntimeError(f"GSQ audit has ambiguous revisions: {revisions}")
    gsq_revision = next(iter(revisions))
    if gsq_revision != relationship["gsq_release"]["revision"]:
        raise RuntimeError("GSQ audit and relationship revisions differ")
    if (
        relationship["dense_base"]["repo_id"] != identity["repo_id"]
        or relationship["dense_base"]["revision"] != identity["revision"]
    ):
        raise RuntimeError("dense identity and GSQ relationship differ")
    if acceptance.get("stage0_gate") != "PASSED_BY_EXPLICIT_USER_ACCEPTANCE":
        raise RuntimeError("accepted GSQ-to-Q2_0 deviation gate is absent")
    accepted_bound = float(acceptance["accepted_max_actual_weight_error"])

    llama_cpp = args.llama_cpp.resolve(strict=True)
    llama_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=llama_cpp, check=True,
        text=True, capture_output=True).stdout.strip()
    if llama_revision != manifest["llama_cpp_revision"]:
        raise RuntimeError("llama.cpp checkout differs from canonical manifest")

    searched = [entry for entry in manifest["entries"] if entry.get("rco_search")]
    if len(manifest["entries"]) != 733 or len(searched) != 512:
        raise RuntimeError(
            f"unexpected manifest inventory: {len(manifest['entries'])}/"
            f"{len(searched)}")
    routed = [
        entry for entry in searched
        if entry["source_category"] == "routed_expert"
    ]
    if len(routed) != 120:
        raise RuntimeError(f"expected 120 routed candidates, found {len(routed)}")

    codec = GGMLNativeCodec(args.ggml_library.resolve(strict=True))
    projection = _projected_bytes(searched, codec)
    if projection["combined_payload_bytes"] != 29_243_911_680:
        raise RuntimeError(
            f"candidate byte projection changed: "
            f"{projection['combined_payload_bytes']}")

    output = args.store_output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace complete store: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = (
        args.stage.resolve()
        if args.stage is not None
        else output.parent / f".{output.name}.stage"
    )
    if stage == output:
        raise ValueError("stage and output paths must differ")
    stage.parent.mkdir(parents=True, exist_ok=True)
    if os.stat(stage.parent).st_dev != os.stat(output.parent).st_dev:
        raise ValueError("stage and output must be on the same filesystem")
    stage_bytes_before = _tree_bytes(stage)
    free_before = shutil.disk_usage(output.parent).free
    remaining_projection = max(
        0, projection["combined_payload_bytes"] - stage_bytes_before)
    if free_before - remaining_projection < args.min_free_bytes:
        raise RuntimeError(
            "projected candidate database would cross the free-space floor")

    source_descriptor = _source_descriptor(
        identity, manifest_path, manifest, gsq_revision, llama_revision)
    writer = NativeCandidateStoreWriter(
        stage,
        codec,
        source=source_descriptor,
        resume=True,
    )
    resumed_candidates = writer.resumed_candidate_count
    dense_source = SafetensorGGUFRowSource(model_dir)
    gsq_checkpoint = GSQCheckpoint(args.gsq_dir.resolve(strict=True))
    generation_stats: dict[str, int] = {
        "chunk_count": 0,
        "max_dense_chunk_bytes": 0,
        "max_source_rows_per_chunk": 0,
    }
    unit_reports = []
    generated_groups = 0
    generation_started = time.perf_counter()
    units = _units(manifest["entries"])
    for unit_index, (unit_name, unit_entries) in enumerate(units):
        unit_started = time.perf_counter()
        unit_searched = [entry for entry in unit_entries if entry.get("rco_search")]
        for entry in unit_searched:
            for candidate_type in _CANDIDATE_TYPES:
                if _is_authentic_gsq(entry, candidate_type):
                    match = _BLOCK.match(entry["destination_name"])
                    if match is None:
                        raise RuntimeError("routed expert is outside a block")
                    records: list[dict[str, Any]] = []
                    chunks = _gsq_chunks(
                        gsq_checkpoint,
                        codec,
                        layer=int(match.group(1)),
                        projection=_projection(entry),
                        expert_count=int(entry["candidate_source_shape"][0]),
                        accepted_max_weight_error=accepted_bound,
                        records=records,
                    )
                    provenance = {
                        "kind": "direct_authentic_gsq2_to_q2_0",
                        "gsq_revision": gsq_revision,
                        "projection": _projection(entry),
                        "expert_order": list(range(
                            int(entry["candidate_source_shape"][0]))),
                        "mapping": acceptance["mapping_unchanged"],
                        "accepted_max_actual_weight_error": accepted_bound,
                    }
                else:
                    chunks = _bf16_chunks(
                        dense_source,
                        codec,
                        entry,
                        candidate_type,
                        rows_per_chunk=args.rows_per_chunk,
                        stats=generation_stats,
                    )
                    provenance = {
                        "kind": "bf16_native_quantization",
                        "dense_revision": identity["revision"],
                        "source_tensor": entry["source_name"],
                        "source_shard": entry["source_shard"],
                        "source_view": entry.get("source_view"),
                        "converter_transforms": entry["converter_transforms"],
                        "rows_per_chunk": args.rows_per_chunk,
                    }
                writer.write_packed_chunks(
                    entry["destination_name"],
                    candidate_type,
                    entry["destination_gguf_shape"],
                    chunks,
                    provenance=provenance,
                )
        writer.finalize()
        generated_groups += len(unit_searched)
        free_now = shutil.disk_usage(output.parent).free
        if free_now < args.min_free_bytes:
            raise RuntimeError("candidate generation crossed the free-space floor")
        unit_report = {
            "index": unit_index,
            "name": unit_name,
            "searched_tensor_count": len(unit_searched),
            "candidate_count": 2 * len(unit_searched),
            "cumulative_searched_tensor_count": generated_groups,
            "elapsed_seconds": time.perf_counter() - unit_started,
            "free_bytes_after": free_now,
        }
        unit_reports.append(unit_report)
        print(json.dumps({"phase": "generation", **unit_report}, sort_keys=True),
              flush=True)
    generation_seconds = time.perf_counter() - generation_started

    store = NativeCandidateStore(stage, codec)
    if store.index["tensor_count"] != 512 or store.index["candidate_count"] != 1024:
        raise RuntimeError("complete candidate store inventory differs")
    actual_payload = sum(
        int(metadata["payload_bytes"])
        for candidates in store.index["tensors"].values()
        for metadata in candidates.values()
    )
    if actual_payload != projection["combined_payload_bytes"]:
        raise RuntimeError("complete store payload differs from exact projection")

    validation_stats: dict[str, int] = {
        "chunk_count": 0,
        "max_dense_chunk_bytes": 0,
        "max_source_rows_per_chunk": 0,
        "max_validation_scratch_bytes": 0,
    }
    validation_records = []
    validation_units = []
    validation_started = time.perf_counter()
    for unit_index, (unit_name, unit_entries) in enumerate(units):
        unit_started = time.perf_counter()
        unit_records = []
        for entry in unit_entries:
            if not entry.get("rco_search"):
                continue
            for candidate_type in _CANDIDATE_TYPES:
                if _is_authentic_gsq(entry, candidate_type):
                    record = _validate_gsq_candidate(
                        gsq_checkpoint,
                        codec,
                        store,
                        entry,
                        accepted_bound=accepted_bound,
                    )
                else:
                    record = {
                        **_validate_candidate(
                            dense_source,
                            store,
                            codec,
                            entry,
                            candidate_type,
                            rows_per_chunk=args.validation_rows_per_chunk,
                            stats=validation_stats,
                        ),
                        "origin": "bf16_native_quantization",
                    }
                unit_records.append(record)
        validation_records.extend(unit_records)
        unit_report = {
            "index": unit_index,
            "name": unit_name,
            "candidate_count": len(unit_records),
            "elapsed_seconds": time.perf_counter() - unit_started,
        }
        validation_units.append(unit_report)
        print(json.dumps({"phase": "validation", **unit_report}, sort_keys=True),
              flush=True)
    validation_seconds = time.perf_counter() - validation_started

    if len(validation_records) != 1024:
        raise RuntimeError("validated candidate count differs")
    gsq_records = [
        record for record in validation_records
        if record["origin"] == "direct_authentic_gsq2_to_q2_0"
    ]
    if len(gsq_records) != 120:
        raise RuntimeError("validated GSQ candidate count differs")
    observed_gsq_error = max(
        record["max_weight_absolute_error"] for record in gsq_records)
    if observed_gsq_error > accepted_bound:
        raise RuntimeError("full GSQ import exceeded accepted error")

    block0_store_path = args.block0_production_store.resolve(strict=True)
    block0_store = NativeCandidateStore(block0_store_path, codec)
    block0_regression = []
    for entry in searched:
        if not entry["destination_name"].startswith("blk.0."):
            continue
        for candidate_type in _CANDIDATE_TYPES:
            expected = block0_store.metadata(entry["destination_name"], candidate_type)
            actual = store.metadata(entry["destination_name"], candidate_type)
            if expected["sha256"] != actual["sha256"]:
                raise RuntimeError("full database changed proven block-0 candidate")
            block0_regression.append({
                "tensor": entry["destination_name"],
                "ggml_type": candidate_type.name,
                "sha256": actual["sha256"],
                "exact_match": True,
            })

    index_sha256 = _sha256_file(stage / "native-candidate-index.json")
    stage_bytes_complete = _tree_bytes(stage)
    os.replace(stage, output)
    directory_fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    free_after = shutil.disk_usage(output.parent).free

    return {
        "schema": 1,
        "status": "pass",
        "scope": (
            "complete 512-group production native Q2_0/Q4_0 candidate "
            "database for the text-only Qwen3.6-35B-A3B model; routed-expert "
            "Q2_0 candidates come directly from the authentic GSQ checkpoint, "
            "all other Q2_0 and every Q4_0 candidate come from matching BF16"
        ),
        "source": {
            "dense_repo_id": identity["repo_id"],
            "dense_revision": identity["revision"],
            "dense_model_dir": str(model_dir),
            "identity_path": str(identity_path),
            "identity_sha256": _sha256_file(identity_path),
            "gsq_repo_id": "ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ",
            "gsq_revision": gsq_revision,
            "gsq_directory": str(args.gsq_dir.resolve(strict=True)),
            "gsq_audit_path": str(gsq_audit_path),
            "gsq_audit_sha256": _sha256_file(gsq_audit_path),
            "relationship_path": str(relationship_path),
            "relationship_sha256": _sha256_file(relationship_path),
            "acceptance_path": str(acceptance_path),
            "acceptance_sha256": _sha256_file(acceptance_path),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "llama_cpp_path": str(llama_cpp),
            "llama_cpp_revision": llama_revision,
            "ggml_library": str(codec.library_path),
            "ggml_library_sha256": codec.library_sha256,
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
            "entry_count": len(manifest["entries"]),
            "searched_tensor_count": len(searched),
            "routed_expert_tensor_count": len(routed),
            "shape_coverage": _shape_coverage(searched, codec),
        },
        "capacity": {
            "projection": projection,
            "minimum_free_bytes": args.min_free_bytes,
            "free_bytes_before": free_before,
            "stage_bytes_before": stage_bytes_before,
            "stage_bytes_complete": stage_bytes_complete,
            "free_bytes_after": free_after,
        },
        "generation": {
            "stage_path": str(stage),
            "atomic_output_path": str(output),
            "resumed_candidate_count": resumed_candidates,
            "reused_candidate_count": writer.reused_candidate_count,
            "written_candidate_count": writer.written_candidate_count,
            "rows_per_chunk": args.rows_per_chunk,
            "seconds": generation_seconds,
            **generation_stats,
            "units": unit_reports,
        },
        "candidate_store": {
            "path": str(output),
            "index": str(output / "native-candidate-index.json"),
            "index_sha256": index_sha256,
            "tensor_count": store.index["tensor_count"],
            "candidate_count": store.index["candidate_count"],
            "total_payload_bytes": actual_payload,
            "q2_0_payload_bytes": projection["by_type"][GGMLType.Q2_0.name],
            "q4_0_payload_bytes": projection["by_type"][GGMLType.Q4_0.name],
            "routed_q2_0_origin": "direct_authentic_gsq2_to_q2_0",
            "other_candidate_origin": "matching_bf16_native_quantization",
        },
        "validation": {
            "candidate_count": len(validation_records),
            "bf16_candidate_count": len(validation_records) - len(gsq_records),
            "authentic_gsq_candidate_count": len(gsq_records),
            "validation_rows_per_chunk": args.validation_rows_per_chunk,
            "all_bf16_payloads_equal_independent_quantization": True,
            "all_gsq_payloads_equal_direct_mapping": True,
            "all_gsq_stock_decodes_match_mapped_values": True,
            "observed_gsq_max_weight_absolute_error": observed_gsq_error,
            "accepted_gsq_max_weight_absolute_error": accepted_bound,
            "gsq_within_accepted_bound": True,
            "seconds": validation_seconds,
            **validation_stats,
            "units": validation_units,
            "candidates": validation_records,
        },
        "block0_production_regression": {
            "source_store": str(block0_store_path),
            "candidate_count": len(block0_regression),
            "all_payload_hashes_match_exactly": True,
            "candidates": block0_regression,
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
    parser.add_argument("--gsq-dir", type=Path, required=True)
    parser.add_argument("--gsq-audit", type=Path, required=True)
    parser.add_argument("--relationship", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--block0-production-store", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--store-output", type=Path, required=True)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument("--validation-rows-per-chunk", type=int, default=16)
    parser.add_argument("--min-free-bytes", type=int, default=100 << 30)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rows_per_chunk <= 0 or args.validation_rows_per_chunk <= 0:
        raise ValueError("row chunk sizes must be positive")
    if args.min_free_bytes < 0:
        raise ValueError("free-space floor must be non-negative")
    report = audit(args)
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "store": report["candidate_store"]["path"],
        "tensor_count": report["candidate_store"]["tensor_count"],
        "candidate_count": report["candidate_store"]["candidate_count"],
        "payload_bytes": report["candidate_store"]["total_payload_bytes"],
        "resumed_candidate_count": report["generation"][
            "resumed_candidate_count"],
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
