#!/usr/bin/env python3
"""Build a block store whose routed Q2_0 candidates come directly from GSQ."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gsq_q2 import GSQCheckpoint, repack_gsq2_to_q2_0
from native_store import NativeCandidateStore, NativeCandidateStoreWriter
from quant.ggml_native import GGMLNativeCodec, GGMLType


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


def _projection(entry: dict[str, Any]) -> str:
    name = entry["normalized_source_name"]
    for projection in ("gate_proj", "up_proj", "down_proj"):
        if name.endswith(f".experts.{projection}.weight"):
            return projection
    raise ValueError(f"entry is not a routed projection: {name}")


def _gsq_chunks(
    checkpoint: GSQCheckpoint,
    codec: GGMLNativeCodec,
    *,
    layer: int,
    projection: str,
    expert_count: int,
    accepted_max_weight_error: float,
    records: list[dict[str, Any]],
) -> Iterator[bytes]:
    for expert in range(expert_count):
        prefix = (
            f"model.language_model.layers.{layer}.mlp.experts."
            f"{expert}.{projection}")
        codes, scales = checkpoint.expert(prefix)
        payload = repack_gsq2_to_q2_0(codes, scales)
        mapped_scales = scales.astype(np.float16).astype(np.float32)
        scale_error = np.abs(
            mapped_scales.astype(np.float64) - scales.astype(np.float64))
        magnitudes = np.abs(codes.astype(np.int16) - 2).reshape(
            codes.shape[0], codes.shape[1] // 128, 128)
        max_weight_error = float(
            (magnitudes * scale_error[..., None]).max(initial=0))
        if max_weight_error > accepted_max_weight_error:
            raise RuntimeError(
                f"GSQ scale mapping exceeds accepted error at {prefix}: "
                f"{max_weight_error}")

        decoded = np.empty(codes.shape, dtype=np.float32)
        codec.dequantize_rows_into(payload, GGMLType.Q2_0, decoded)
        expected = (
            (codes.astype(np.int16) - 2).astype(np.float32)
            * np.repeat(mapped_scales, 128, axis=1)
        )
        if not np.array_equal(decoded, expected):
            raise RuntimeError(f"stock Q2_0 decode differs from mapped GSQ: {prefix}")
        records.append({
            "expert": expert,
            "source_prefix": prefix,
            "logical_shape": list(codes.shape),
            "payload_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "group_count": int(scales.size),
            "scale_rounding_count": int((mapped_scales != scales).sum()),
            "scale_underflow_count": int(
                ((mapped_scales == 0) & (scales != 0)).sum()),
            "max_scale_absolute_error": float(scale_error.max(initial=0)),
            "max_weight_absolute_error": max_weight_error,
            "stock_decode_matches_mapped_gsq_exactly": True,
        })
        yield payload


def _compare_store_with_gguf(
    store: NativeCandidateStore,
    tensor_name: str,
    gguf_tensor,
) -> dict[str, Any]:
    if int(gguf_tensor.tensor_type) != int(GGMLType.Q2_0):
        raise RuntimeError(f"proven GGUF tensor is not Q2_0: {tensor_name}")
    raw = np.asarray(gguf_tensor.data).view(np.uint8).reshape(-1)
    offset = 0
    for chunk in store.iter_payload(tensor_name, GGMLType.Q2_0):
        stop = offset + len(chunk)
        if not np.array_equal(np.frombuffer(chunk, dtype=np.uint8), raw[offset:stop]):
            raise RuntimeError(f"imported GSQ bytes differ from proven GGUF: {tensor_name}")
        offset = stop
    if offset != raw.size:
        raise RuntimeError(f"imported/proven GGUF sizes differ: {tensor_name}")
    return {
        "tensor": tensor_name,
        "payload_bytes": offset,
        "proven_gguf_payload_sha256": hashlib.sha256(raw).hexdigest(),
        "exact_payload_match": True,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    manifest = _load_json(args.manifest.resolve(strict=True))
    relationship = _load_json(args.relationship.resolve(strict=True))
    gsq_audit = _load_json(args.gsq_audit.resolve(strict=True))
    acceptance = _load_json(args.acceptance.resolve(strict=True))
    if manifest.get("status") != "pass" or relationship.get("status") != "pass":
        raise RuntimeError("canonical manifest and relationship audit must pass")
    revisions = gsq_audit["source_revision_evidence"]
    if len(revisions) != 1:
        raise RuntimeError(f"GSQ audit has ambiguous revisions: {revisions}")
    gsq_revision = next(iter(revisions))
    if gsq_revision != relationship["gsq_release"]["revision"]:
        raise RuntimeError("GSQ audit and relationship revisions differ")
    if acceptance.get("stage0_gate") != "PASSED_BY_EXPLICIT_USER_ACCEPTANCE":
        raise RuntimeError("accepted GSQ-to-Q2_0 deviation gate is absent")
    accepted_bound = float(acceptance["accepted_max_actual_weight_error"])

    entries = [
        entry for entry in manifest["entries"]
        if entry["destination_name"].startswith(f"blk.{args.block}.")
        and entry["rco_search"]
    ]
    if len(entries) != 13:
        raise RuntimeError(f"expected 13 block decision groups, found {len(entries)}")
    routed = [entry for entry in entries if entry["source_category"] == "routed_expert"]
    if len(routed) != 3:
        raise RuntimeError(f"expected three routed projections, found {len(routed)}")

    codec = GGMLNativeCodec(args.ggml_library)
    base_store_path = args.base_store.resolve(strict=True)
    base_store = NativeCandidateStore(base_store_path, codec)
    if base_store.index["source"]["revision"] != manifest["source"]["revision"]:
        raise RuntimeError("BF16 candidate store and dense manifest revisions differ")
    gsq_dir = args.gsq_dir.resolve(strict=True)
    checkpoint = GSQCheckpoint(gsq_dir)
    output = args.store_output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing candidate store: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.llama_cpp.resolve(strict=True) / "gguf-py"))
    from gguf import GGUFReader
    gguf_path = args.gsq_gguf.resolve(strict=True)
    gguf_reader = GGUFReader(gguf_path, "r")
    gguf_tensors = {tensor.name: tensor for tensor in gguf_reader.tensors}

    expert_records: dict[str, list[dict[str, Any]]] = {}
    temporary_parent = (
        None if args.temporary_parent is None
        else str(args.temporary_parent.resolve(strict=True))
    )
    with tempfile.TemporaryDirectory(
        prefix="rco-qwen36-gsq-import-", dir=temporary_parent,
    ) as directory_name:
        store_path = Path(directory_name) / "native-store"
        writer = NativeCandidateStoreWriter(
            store_path,
            codec,
            source={
                "dense_repo_id": manifest["source"]["repo_id"],
                "dense_revision": manifest["source"]["revision"],
                "gsq_repo_id": "ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ",
                "gsq_revision": gsq_revision,
                "block": args.block,
                "candidate_policy": (
                    "authentic GSQ-derived Q2_0 for routed experts; "
                    "BF16-derived Q2_0 elsewhere; BF16-derived Q4_0 everywhere"),
            },
        )
        for entry in entries:
            tensor_name = entry["destination_name"]
            for candidate_type in (GGMLType.Q2_0, GGMLType.Q4_0):
                if candidate_type == GGMLType.Q2_0 and entry in routed:
                    projection = _projection(entry)
                    records: list[dict[str, Any]] = []
                    expert_records[projection] = records
                    chunks = _gsq_chunks(
                        checkpoint,
                        codec,
                        layer=args.block,
                        projection=projection,
                        expert_count=256,
                        accepted_max_weight_error=accepted_bound,
                        records=records,
                    )
                    provenance = {
                        "kind": "direct_gsq2_to_q2_0",
                        "gsq_revision": gsq_revision,
                        "projection": projection,
                        "expert_order": list(range(256)),
                        "mapping": acceptance["mapping_unchanged"],
                        "accepted_max_actual_weight_error": accepted_bound,
                    }
                else:
                    base_metadata = base_store.metadata(tensor_name, candidate_type)
                    chunks = base_store.iter_payload(tensor_name, candidate_type)
                    provenance = {
                        "kind": "copy_bf16_derived_native_candidate",
                        "source_store": str(base_store_path),
                        "source_sha256": base_metadata["sha256"],
                    }
                writer.write_packed_chunks(
                    tensor_name,
                    candidate_type,
                    entry["destination_gguf_shape"],
                    chunks,
                    provenance=provenance,
                )
        writer.finalize()
        store = NativeCandidateStore(store_path, codec)
        gguf_comparisons = [
            _compare_store_with_gguf(
                store, entry["destination_name"],
                gguf_tensors[entry["destination_name"]])
            for entry in routed
        ]
        copied_candidates = []
        for entry in entries:
            for candidate_type in (GGMLType.Q2_0, GGMLType.Q4_0):
                if candidate_type == GGMLType.Q2_0 and entry in routed:
                    continue
                source_meta = base_store.metadata(
                    entry["destination_name"], candidate_type)
                target_meta = store.metadata(
                    entry["destination_name"], candidate_type)
                if source_meta["sha256"] != target_meta["sha256"]:
                    raise RuntimeError("copied BF16-derived candidate hash changed")
                copied_candidates.append({
                    "tensor": entry["destination_name"],
                    "ggml_type": candidate_type.name,
                    "sha256": target_meta["sha256"],
                    "exact_copy": True,
                })
        index_sha256 = _sha256_file(
            store_path / "native-candidate-index.json")
        total_payload = sum(
            int(metadata["payload_bytes"])
            for candidates in store.index["tensors"].values()
            for metadata in candidates.values()
        )
        os.replace(store_path, output)

    flat_records = [
        record for records in expert_records.values() for record in records
    ]
    max_error = max(
        record["max_weight_absolute_error"] for record in flat_records)
    if max_error > accepted_bound:
        raise RuntimeError("import exceeded accepted maximum weight error")
    return {
        "schema": 1,
        "status": "pass",
        "scope": (
            "genuine block-0 production candidate store with direct authentic "
            "GSQ-to-Q2_0 routed experts; CUDA and full-model search remain open"),
        "dense_source": manifest["source"],
        "gsq_source": {
            "repo_id": "ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ",
            "revision": gsq_revision,
            "directory": str(gsq_dir),
            "prior_audit": str(args.gsq_audit),
        },
        "accepted_deviations": {
            **acceptance,
            "observed_block0_max_weight_absolute_error": max_error,
            "within_accepted_bound": True,
        },
        "ggml_library": {
            "path": str(codec.library_path),
            "sha256": codec.library_sha256,
        },
        "proven_gsq_gguf": {
            "path": str(gguf_path),
            "sha256": _sha256_file(gguf_path),
            "routed_q2_0_comparisons": gguf_comparisons,
            "all_imported_routed_payloads_match_exactly": True,
        },
        "candidate_store": {
            "path": str(output),
            "index": str(output / "native-candidate-index.json"),
            "index_sha256": index_sha256,
            "tensor_count": len(entries),
            "candidate_count": len(entries) * 2,
            "total_payload_bytes": total_payload,
            "routed_q2_0_origin": "direct_gsq2_to_q2_0",
            "other_candidate_origin": "exact_copy_from_bf16_derived_store",
        },
        "routed_experts": {
            "projection_count": len(routed),
            "expert_count_per_projection": 256,
            "expert_payload_count": len(flat_records),
            "source_group_count": sum(
                record["group_count"] for record in flat_records),
            "scale_rounding_count": sum(
                record["scale_rounding_count"] for record in flat_records),
            "scale_underflow_count": sum(
                record["scale_underflow_count"] for record in flat_records),
            "max_weight_absolute_error": max_error,
            "all_stock_decodes_match_mapped_gsq_exactly": True,
            "projections": expert_records,
        },
        "copied_candidates": copied_candidates,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsq-dir", type=Path, required=True)
    parser.add_argument("--gsq-audit", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--relationship", type=Path, required=True)
    parser.add_argument("--base-store", type=Path, required=True)
    parser.add_argument("--store-output", type=Path, required=True)
    parser.add_argument("--gsq-gguf", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--temporary-parent", type=Path)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/qwen36_35b_block0_gsq_import.json"))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(args)
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "store": report["candidate_store"]["path"],
        "payload_bytes": report["candidate_store"]["total_payload_bytes"],
        "max_weight_absolute_error": report[
            "routed_experts"]["max_weight_absolute_error"],
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
