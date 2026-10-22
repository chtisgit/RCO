#!/usr/bin/env python3
"""Test A of PLAN_POST_RCO_RELEASE.md: an I-quant store for the pruned model.

The Phase 6 store builder (``build_qwen36_prune24_q3k_store.py``) for a
llama.cpp I-quant type (``--type IQ2_XS`` or ``IQ2_XXS``) instead of Q3_K:
for each of the 120 routed-expert tensors, the BF16 rows are quantized per
expert with that expert's row of the Phase 5 fit imatrix.  Payloads keep all
256 experts (the 24 pruned ones per layer with the layer-mean fallback; they
are never routed), so the streaming evaluator installs them in place of the
loaded GSQ tensor.

Per tensor, over the kept experts only: relative RMSE, imatrix-weighted
relative error and token-weighted squared error against BF16, for authentic
GSQ Q2_0 and the new type.  Before a payload is accepted, the BF16 rows must
reproduce the checksum of the stored Q4_0 candidate, as in Phase 6.
Resumable per tensor.  Nothing here reads the validation part or the
held-out half.
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
from build_qwen36_prune24_q3k_store import _Errors  # noqa: E402
from native_store import NativeCandidateStore, NativeCandidateStoreWriter  # noqa: E402
from quant.ggml_importance import quantize_rows_with_importance  # noqa: E402
from quant.ggml_native import GGMLType  # noqa: E402
from quant.ggml_native_iq import IQ_TYPE_IDS, IQGGMLNativeCodec  # noqa: E402

SCHEMA = "rco.qwen36.post_release_iq_store.v1"
LAYERS = 40
EXPERTS = 256
KEPT = 232

_WORKER: dict[str, Any] = {}


def _worker_init(settings: dict[str, Any]) -> None:
    from qwen35_native import SafetensorGGUFRowSource

    _WORKER.update(settings)
    _WORKER["codec"] = IQGGMLNativeCodec(settings["ggml_library"])
    _WORKER["source"] = SafetensorGGUFRowSource(settings["model_dir"])
    with np.load(settings["imatrix_path"]) as matrix:
        _WORKER["imatrix"] = {key: matrix[key] for key in matrix.files}
    _WORKER["mask"] = np.load(settings["mask_path"])


def _quantize_tensor(name: str) -> dict[str, Any]:
    codec: IQGGMLNativeCodec = _WORKER["codec"]
    type_name = _WORKER["type_name"]
    type_id = IQ_TYPE_IDS[type_name]
    store_root = Path(_WORKER["store_root"])
    sidecar = store_root / "tensors" / quote(name, safe="") / f"{type_name}.metrics.json"
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
    variant = type_name.lower() + "_imatrix"
    errors = {"gsq_q2_0": _Errors(), variant: _Errors()}
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
            payload = quantize_rows_with_importance(codec, rows, type_id, importance[expert])
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
                codec.dequantize_rows_into(payload, type_id, decoded)
                errors[variant].add(decoded, rows, weights, token_sums)
            yield payload
        if expert != experts - 1:
            raise RuntimeError(f"source expert count differs for {name}")

    writer = NativeCandidateStoreWriter(store_root, codec, source=_WORKER["store_source"])
    store_entry = writer.write_packed_chunks(
        name, type_id, entry["destination_gguf_shape"], chunks(),
        provenance={
            "kind": "bf16_derived_imatrix_native_candidate",
            "source_tensor": entry["source_name"],
            "source_shard": entry["source_shard"],
            "source_view": entry.get("source_view"),
            "imatrix_sha256": _WORKER["imatrix_sha256"],
            "mask_sha256": _WORKER["mask_sha256"],
        })
    if q4_0_digest.hexdigest() != _WORKER["stored_q4_0"][name]:
        raise RuntimeError(f"BF16 rows do not reproduce the stored Q4_0: {name}")
    expected_kept = KEPT * rows_per_expert * int(codec.geometry(type_id, width)["row_size"])
    if kept_bytes != expected_kept:
        raise RuntimeError(f"kept {type_name} bytes differ for {name}: {kept_bytes}")
    metrics = {
        key: {
            "relative_rmse": math.sqrt(accumulator.squared / energy),
            "weighted_relative_error": math.sqrt(accumulator.weighted / weighted_energy),
            "token_weighted_squared_error": accumulator.token_weighted,
        }
        for key, accumulator in errors.items()
    }
    result = {
        "name": name,
        "layer": layer,
        "family": family,
        "type": type_name,
        "store_entry": store_entry,
        "kept_payload": {"sha256": kept_digest.hexdigest(), "bytes": kept_bytes},
        "stored_q4_0_sha256_reproduced": q4_0_digest.hexdigest(),
        "gsq_q2_0_kept_bytes": KEPT * expert_bytes,
        "incremental_bytes": kept_bytes - KEPT * expert_bytes,
        "metrics": metrics,
        "weighted_relative_error_reduction": (
            metrics["gsq_q2_0"]["weighted_relative_error"]
            - metrics[variant]["weighted_relative_error"]),
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(sidecar, result)
    return result


def run(args: argparse.Namespace) -> None:
    from native_gguf import import_pinned_gguf

    type_name = args.type
    output = args.reports / f"qwen36_post_release_iq_store_{type_name}.json"
    if output.exists() and _load_json(output).get("status") == "complete":
        print(f"store {type_name}: already complete", flush=True)
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
    names = args.only or expert_names()
    manifest_entries = {
        entry["destination_name"]: entry for entry in manifest["entries"]
        if entry["destination_name"] in names}
    if len(manifest_entries) != len(names):
        raise RuntimeError("manifest does not map every routed-expert tensor")
    codec = IQGGMLNativeCodec(args.ggml_library)
    native = NativeCandidateStore(args.native_store, codec)
    stored_q4_0 = {name: native.metadata(name, GGMLType.Q4_0)["sha256"] for name in names}
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
    store_root = args.work / f"store_{type_name}"
    store_source = {
        "kind": f"bf16_derived_imatrix_{type_name.lower()}_routed_experts_prune24",
        "model_dir": str(args.model_dir),
        "manifest_sha256": _sha256_file(args.manifest),
        "imatrix_sha256": imatrix_sha256,
        "mask_sha256": mask_sha256,
    }
    settings = {
        "type_name": type_name,
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
            result = _quantize_tensor(name)
            print(json.dumps({"tensor": name, "metrics": result["metrics"],
                              "seconds": round(result["wall_seconds"], 1)}), flush=True)
        return
    gguf_sha256 = _sha256_file(args.gguf)
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
                "gsq_weighted": round(result["metrics"]["gsq_q2_0"]["weighted_relative_error"], 4),
                "new_weighted": round(result["metrics"][f"{type_name.lower()}_imatrix"]
                                      ["weighted_relative_error"], 4),
                "seconds": round(result["wall_seconds"], 1),
            }), flush=True)
    writer = NativeCandidateStoreWriter(store_root, codec, source=store_source)
    writer._entries = {name: {type_name: results[name]["store_entry"]} for name in names}
    index_path = writer.finalize()
    variant = f"{type_name.lower()}_imatrix"
    summary = {}
    for family in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "all"):
        members = [name for name in names if family == "all" or results[name]["family"] == family]
        for key in ("gsq_q2_0", variant):
            for metric in ("relative_rmse", "weighted_relative_error"):
                summary.setdefault(family, {}).setdefault(key, {})[f"median_{metric}"] = float(
                    np.median([results[name]["metrics"][key][metric] for name in members]))
        summary[family]["new_better_weighted_on"] = sum(
            results[name]["weighted_relative_error_reduction"] > 0 for name in members)
        summary[family]["tensors"] = len(members)
    _atomic_json(output, {
        "schema": SCHEMA,
        "status": "complete",
        "type": type_name,
        "gsq_gguf_sha256": gguf_sha256,
        "imatrix_report": str(args.imatrix_report),
        "imatrix_sha256": imatrix_sha256,
        "mask": {"path": str(mask_path), "sha256": mask_sha256},
        "store": {"root": str(store_root), "index_sha256": _sha256_file(index_path)},
        "kept_bytes_per_tensor": {
            "gsq_q2_0": results[names[0]]["gsq_q2_0_kept_bytes"],
            type_name: results[names[0]]["kept_payload"]["bytes"]},
        "summary": summary,
        "tensors": results,
        "wall_seconds": time.perf_counter() - started,
    })
    print(f"store {type_name}: complete; " + json.dumps(summary["all"]), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", required=True, choices=("IQ2_XS", "IQ2_XXS"))
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/qwen36_35b_base")
    parser.add_argument("--manifest", type=Path,
                        default=RCO / "reports/qwen36_35b_base_gguf_manifest.json")
    parser.add_argument("--imatrix-report", type=Path,
                        default=RCO / "reports/qwen36_prune24_imatrix.json")
    parser.add_argument("--gguf", type=Path,
                        default=ROOT / "results/Qwen3.6-35B-A3B-GSQ-hybrid.gguf")
    parser.add_argument("--gguf-python", type=Path, default=ROOT / "repos/llama.cpp/gguf-py")
    parser.add_argument("--ggml-library", type=Path,
                        default=ROOT / "experiment/build-cpu/bin/libggml-base.so.0.24.0")
    parser.add_argument("--native-store", type=Path, default=ROOT / "data/qwen36_35b_native")
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_post_release_q2_types")
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
