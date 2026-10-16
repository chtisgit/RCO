#!/usr/bin/env python3
"""Phase 6 of RCO_PLAN_NEW.md: the Q3_K candidate store for the pruned model.

For each of the 120 routed-expert tensors it writes a BF16-derived Q3_K
payload, quantized per expert with that expert's row of the Phase 5 fit
imatrix (``reports/qwen36_prune24_imatrix.json``), into a new native store.

The payloads keep all 256 experts, so the streaming evaluator can install
them in place of the loaded GSQ tensor (as in Phase 0) while the frequency
mask restricts routing.  The 24 pruned experts per layer never receive a
token; their imatrix rows are the layer-mean fallback of Phase 0a, and their
bytes are never used.  What the Phase 8 builder writes is the kept experts'
slices in ascending order: 232 x 450,560 = 104,529,920 bytes per tensor.
Their sha256 is recorded as ``kept_payload``.

Per tensor the report records, over the kept experts only, the relative RMSE
and imatrix-weighted relative error against BF16 for authentic GSQ Q2_0 and
for Q3_K, plus the absolute token-weighted squared error (sum over kept
experts and columns of squared weight error times the fit sum of squared
inputs).  The Phase 7 imatrix control is ranked by the reduction in
imatrix-weighted relative error (Q2_0 minus Q3_K).  It is scale-free; the
absolute token-weighted error lives in each tensor's own output space (gate
and up pre-activations, the residual for down), so it is recorded but does
not compare across families.

Before a payload is accepted, the BF16 rows must reproduce the checksum of
the stored Q4_0 candidate, as in Phase 0.  Resumable per tensor.  Nothing
here reads the validation part or the held-out half.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_q3k_viability import (  # noqa: E402
    GSQ_Q2_0_TYPE,
    _atomic_json,
    _load_json,
    _parse_name,
    _sha256_file,
    expert_importance,
    expert_names,
)
from native_store import NativeCandidateStore, NativeCandidateStoreWriter  # noqa: E402
from quant.ggml_importance import quantize_rows_with_importance  # noqa: E402
from quant.ggml_native import GGMLNativeCodec, GGMLType  # noqa: E402

SCHEMA = "rco.qwen36.prune24_q3k_store.v1"
LAYERS = 40
EXPERTS = 256
KEPT = 232
KEPT_PAYLOAD_BYTES = 104_529_920
GSQ_KEPT_BYTES = 68_419_584


class _Errors:
    def __init__(self) -> None:
        self.squared = 0.0
        self.weighted = 0.0
        self.token_weighted = 0.0

    def add(self, decoded, rows, importance, sums) -> None:
        difference = np.square(decoded - rows, dtype=np.float64)
        self.squared += float(difference.sum())
        self.weighted += float((difference * importance).sum())
        self.token_weighted += float((difference * sums).sum())


_WORKER: dict[str, Any] = {}


def _worker_init(settings: dict[str, Any]) -> None:
    from qwen35_native import SafetensorGGUFRowSource

    _WORKER.update(settings)
    _WORKER["codec"] = GGMLNativeCodec(settings["ggml_library"])
    _WORKER["source"] = SafetensorGGUFRowSource(settings["model_dir"])
    with np.load(settings["imatrix_path"]) as matrix:
        _WORKER["imatrix"] = {key: matrix[key] for key in matrix.files}
    _WORKER["mask"] = np.load(settings["mask_path"])


def _quantize_tensor(name: str) -> dict[str, Any]:
    codec: GGMLNativeCodec = _WORKER["codec"]
    store_root = Path(_WORKER["store_root"])
    sidecar = store_root / "tensors" / quote(name, safe="") / "Q3_K.metrics.json"
    if sidecar.exists():
        return _load_json(sidecar)
    started = time.perf_counter()
    entry = _WORKER["manifest_entries"][name]
    layer, family = _parse_name(name)
    experts, rows_per_expert, width = (
        int(value) for value in entry["candidate_source_shape"])
    if experts != EXPERTS:
        raise RuntimeError(f"expert count differs for {name}")
    imatrix = _WORKER["imatrix"]
    sums = imatrix[f"blk{layer}_down_sum" if family == "ffn_down_exps"
                   else f"blk{layer}_gate_up_sum"]
    counts = imatrix[f"blk{layer}_count"]
    pruned = _WORKER["mask"][layer]
    if counts[pruned].any() or sums[pruned].any():
        raise RuntimeError(f"pruned experts have imatrix data in {name}")
    if not (counts[~pruned] > 0).all():
        raise RuntimeError(f"a kept expert has no fit tokens in {name}")
    importance, fallback_experts = expert_importance(sums, counts)
    if fallback_experts != EXPERTS - KEPT or importance.shape != (experts, width):
        raise RuntimeError(f"unexpected imatrix fallback for {name}")
    offset, gsq_bytes = _WORKER["gsq_offsets"][name]
    gsq_row_size = codec.geometry(GGMLType.Q2_0, width)["row_size"]
    if gsq_bytes != experts * rows_per_expert * gsq_row_size:
        raise RuntimeError(f"GSQ payload size differs for {name}")
    gsq = np.memmap(_WORKER["gguf_path"], dtype=np.uint8, mode="r",
                    offset=offset, shape=(gsq_bytes,))
    expert_bytes = rows_per_expert * gsq_row_size
    errors = {"gsq_q2_0": _Errors(), "q3_k_imatrix": _Errors()}
    energy = 0.0
    weighted_energy = 0.0
    q4_0_digest = hashlib.sha256()
    kept_digest = hashlib.sha256()
    kept_bytes = 0

    def chunks():
        nonlocal energy, weighted_energy, kept_bytes
        expert = -1
        for expert, rows in enumerate(_WORKER["source"].iter_rows(
            entry, rows_per_chunk=rows_per_expert,
        )):
            rows = np.ascontiguousarray(rows, dtype=np.float32)
            if rows.shape != (rows_per_expert, width):
                raise RuntimeError(f"source chunk is not one expert: {name}")
            q4_0_digest.update(codec.quantize_rows(rows, GGMLType.Q4_0))
            payload = quantize_rows_with_importance(
                codec, rows, GGMLType.Q3_K, importance[expert])
            if not pruned[expert]:
                kept_digest.update(payload)
                kept_bytes += len(payload)
                weights = importance[expert]
                token_sums = sums[expert]
                squared = np.square(rows, dtype=np.float64)
                energy += float(squared.sum())
                weighted_energy += float((squared * weights).sum())
                decoded = np.empty_like(rows)
                codec.dequantize_rows_into(
                    gsq[expert * expert_bytes:(expert + 1) * expert_bytes],
                    GGMLType.Q2_0, decoded)
                errors["gsq_q2_0"].add(decoded, rows, weights, token_sums)
                codec.dequantize_rows_into(payload, GGMLType.Q3_K, decoded)
                errors["q3_k_imatrix"].add(decoded, rows, weights, token_sums)
            yield payload
        if expert != experts - 1:
            raise RuntimeError(f"source expert count differs for {name}")

    writer = NativeCandidateStoreWriter(
        store_root, codec, source=_WORKER["store_source"])
    store_entry = writer.write_packed_chunks(
        name, GGMLType.Q3_K, entry["destination_gguf_shape"], chunks(),
        provenance={
            "kind": "bf16_derived_imatrix_native_candidate",
            "source_tensor": entry["source_name"],
            "source_shard": entry["source_shard"],
            "source_view": entry.get("source_view"),
            "imatrix_sha256": _WORKER["imatrix_sha256"],
            "mask_sha256": _WORKER["mask_sha256"],
        })
    stored_q4_0_sha256 = _WORKER["stored_q4_0"][name]
    if q4_0_digest.hexdigest() != stored_q4_0_sha256:
        raise RuntimeError(f"BF16 rows do not reproduce the stored Q4_0: {name}")
    if kept_bytes != KEPT_PAYLOAD_BYTES:
        raise RuntimeError(f"kept Q3_K bytes differ for {name}: {kept_bytes}")
    metrics = {
        variant: {
            "relative_rmse": math.sqrt(accumulator.squared / energy),
            "weighted_relative_error": math.sqrt(accumulator.weighted / weighted_energy),
            "token_weighted_squared_error": accumulator.token_weighted,
        }
        for variant, accumulator in errors.items()
    }
    result = {
        "name": name,
        "layer": layer,
        "family": family,
        "store_entry": store_entry,
        "kept_payload": {"sha256": kept_digest.hexdigest(), "bytes": kept_bytes},
        "stored_q4_0_sha256_reproduced": q4_0_digest.hexdigest(),
        "gsq_q2_0_kept_bytes": KEPT * expert_bytes,
        "q3_k_incremental_bytes": kept_bytes - KEPT * expert_bytes,
        "metrics": metrics,
        "weighted_relative_error_reduction": (
            metrics["gsq_q2_0"]["weighted_relative_error"]
            - metrics["q3_k_imatrix"]["weighted_relative_error"]),
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(sidecar, result)
    return result


def run(args: argparse.Namespace) -> None:
    from native_gguf import import_pinned_gguf

    output = args.reports / "qwen36_prune24_q3k_store.json"
    if output.exists() and _load_json(output).get("status") == "complete":
        print("store: already complete", flush=True)
        return
    started = time.perf_counter()
    imatrix_report = _load_json(args.imatrix_report)
    if imatrix_report.get("status") != "complete" or imatrix_report["corpus"].get("limit"):
        raise RuntimeError("the fit imatrix is not complete")
    imatrix_path = Path(imatrix_report["imatrix"]["path"])
    imatrix_sha256 = _sha256_file(imatrix_path)
    if imatrix_sha256 != imatrix_report["imatrix"]["sha256"]:
        raise RuntimeError("imatrix file differs from its report")
    mask_path = Path(imatrix_report["mask"]["path"])
    mask_sha256 = _sha256_file(mask_path)
    if mask_sha256 != imatrix_report["mask"]["sha256"]:
        raise RuntimeError("mask file differs from the imatrix report")
    mask = np.load(mask_path)
    if mask.shape != (LAYERS, EXPERTS) or mask.dtype != bool or (
        mask.sum(axis=1) != EXPERTS - KEPT
    ).any():
        raise RuntimeError("mask does not prune 24 experts per layer")
    manifest = _load_json(args.manifest)
    names = expert_names()
    manifest_entries = {
        entry["destination_name"]: entry for entry in manifest["entries"]
        if entry["destination_name"] in names}
    if len(manifest_entries) != len(names):
        raise RuntimeError("manifest does not map every routed-expert tensor")
    codec = GGMLNativeCodec(args.ggml_library)
    native = NativeCandidateStore(args.native_store, codec)
    stored_q4_0 = {name: native.metadata(name, GGMLType.Q4_0)["sha256"]
                   for name in names}
    gguf = import_pinned_gguf(args.gguf_python)
    reader = gguf.GGUFReader(args.gguf)
    offsets = {}
    for tensor in reader.tensors:
        if tensor.name in manifest_entries:
            if int(tensor.tensor_type) != GSQ_Q2_0_TYPE:
                raise RuntimeError(f"GSQ tensor is not Q2_0: {tensor.name}")
            offsets[tensor.name] = (int(tensor.data_offset), int(tensor.n_bytes))
    del reader
    if len(offsets) != len(names):
        raise RuntimeError("GSQ GGUF does not contain every routed-expert tensor")
    gguf_sha256 = _sha256_file(args.gguf)
    store_root = args.work / "q3k_store"
    store_source = {
        "kind": "bf16_derived_imatrix_q3_k_routed_experts_prune24",
        "model_dir": str(args.model_dir),
        "manifest_sha256": _sha256_file(args.manifest),
        "imatrix_sha256": imatrix_sha256,
        "mask_sha256": mask_sha256,
    }
    settings = {
        "ggml_library": str(args.ggml_library),
        "model_dir": str(args.model_dir),
        "imatrix_path": str(imatrix_path),
        "imatrix_sha256": imatrix_sha256,
        "mask_path": str(mask_path),
        "mask_sha256": mask_sha256,
        "manifest_entries": manifest_entries,
        "gsq_offsets": offsets,
        "gguf_path": str(args.gguf),
        "store_root": str(store_root),
        "store_source": store_source,
        "stored_q4_0": stored_q4_0,
    }
    if args.only:
        _worker_init(settings)
        for name in args.only:
            print(json.dumps(_quantize_tensor(name)["metrics"]), flush=True)
        return
    results = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init, initargs=(settings,),
    ) as pool:
        for result in pool.map(_quantize_tensor, names):
            results[result["name"]] = result
            print(json.dumps({
                "tensor": result["name"],
                "gsq_weighted": round(
                    result["metrics"]["gsq_q2_0"]["weighted_relative_error"], 4),
                "q3k_weighted": round(
                    result["metrics"]["q3_k_imatrix"]["weighted_relative_error"], 4),
                "seconds": round(result["wall_seconds"], 1),
            }), flush=True)
    for name, result in results.items():
        if result["gsq_q2_0_kept_bytes"] != GSQ_KEPT_BYTES:
            raise RuntimeError(f"GSQ kept bytes differ for {name}")
    writer = NativeCandidateStoreWriter(store_root, codec, source=store_source)
    writer._entries = {name: {"Q3_K": results[name]["store_entry"]} for name in names}
    index_path = writer.finalize()
    reduction = {name: results[name]["weighted_relative_error_reduction"] for name in names}
    ranked = sorted(names, key=lambda name: (-reduction[name], name))
    down = [name for name in ranked if results[name]["family"] == "ffn_down_exps"]
    weighted = {
        variant: [results[name]["metrics"][variant]["weighted_relative_error"]
                  for name in names]
        for variant in ("gsq_q2_0", "q3_k_imatrix")}
    _atomic_json(output, {
        "schema": SCHEMA,
        "status": "complete",
        "gsq_gguf_sha256": gguf_sha256,
        "imatrix_report": str(args.imatrix_report),
        "imatrix_sha256": imatrix_sha256,
        "mask": {"path": str(mask_path), "sha256": mask_sha256},
        "store": {"root": str(store_root), "index_sha256": _sha256_file(index_path)},
        "summary": {
            "tensor_count": len(names),
            "q3_k_kept_bytes_per_tensor": KEPT_PAYLOAD_BYTES,
            "q3_k_incremental_bytes_per_tensor": KEPT_PAYLOAD_BYTES - GSQ_KEPT_BYTES,
            "median_weighted_relative_error": {
                variant: float(np.median(values)) for variant, values in weighted.items()},
            "q3_k_better_on_every_tensor": all(
                q3k < gsq for gsq, q3k in zip(weighted["gsq_q2_0"], weighted["q3_k_imatrix"])),
            "ranking_by_weighted_relative_error_reduction": ranked,
            "top23_by_weighted_relative_error_reduction": ranked[:23],
            "top23_down_by_weighted_relative_error_reduction": down[:23],
        },
        "tensors": results,
        "wall_seconds": time.perf_counter() - started,
    })
    print("store: complete", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/qwen36_35b_base")
    parser.add_argument("--manifest", type=Path,
                        default=RCO / "reports/qwen36_35b_base_gguf_manifest.json")
    parser.add_argument("--imatrix-report", type=Path,
                        default=RCO / "reports/qwen36_prune24_imatrix.json")
    parser.add_argument("--gguf", type=Path,
                        default=ROOT / "results/Qwen3.6-35B-A3B-GSQ-hybrid.gguf")
    parser.add_argument("--gguf-python", type=Path,
                        default=ROOT / "repos/llama.cpp/gguf-py")
    parser.add_argument("--ggml-library", type=Path,
                        default=ROOT / "experiment/build-cpu/bin/libggml-base.so.0.24.0")
    parser.add_argument("--native-store", type=Path,
                        default=ROOT / "data/qwen36_35b_native")
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_prune24_q3k")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", nargs="+", metavar="TENSOR",
                        help="smoke test: quantize these tensors in-process, no report")
    args = parser.parse_args()
    for key in ("model_dir", "manifest", "imatrix_report", "gguf", "gguf_python",
                "ggml_library", "native_store", "work", "reports"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main() -> int:
    run(_parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
