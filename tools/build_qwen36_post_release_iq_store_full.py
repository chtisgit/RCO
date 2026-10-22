#!/usr/bin/env python3
"""I-quant routed-expert store for the unpruned model (PLAN_POST_RCO_RELEASE.md).

``build_qwen36_post_release_iq_store.py`` builds for the pruned (P24) model:
24 experts per layer get a fallback importance and are left out of the
metrics.  Here all 256 experts are quantized from BF16 with their own row of
Unsloth's published imatrix (``in_sum2 / counts`` in float32, as
``llama-quantize`` uses it; every expert has data), to ``--type`` (IQ2_XS or
IQ3_XXS), for the families in ``--families``.

Per tensor, over all experts: relative RMSE and imatrix-weighted relative
error against BF16 for authentic GSQ Q2_0 and the new type, and the bytes of
both.  Before a payload is accepted, the BF16 rows must reproduce the
checksum of the stored Q4_0 candidate, as in Phase 6.  Resumable per tensor.
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
    FAMILIES,
    GSQ_Q2_0_TYPE,
    _atomic_json,
    _load_json,
    _parse_name,
    _sha256_file,
)
from audit_qwen36_unsloth_imatrix import expert_arrays, read_imatrix  # noqa: E402
from native_store import NativeCandidateStore, NativeCandidateStoreWriter  # noqa: E402
from quant.ggml_importance import quantize_rows_with_importance  # noqa: E402
from quant.ggml_native import GGMLType  # noqa: E402
from quant.ggml_native_iq import IQ_TYPE_IDS, IQGGMLNativeCodec  # noqa: E402

SCHEMA = "rco.qwen36.post_release_iq_store_full.v1"
LAYERS = 40
EXPERTS = 256

_WORKER: dict[str, Any] = {}


def _worker_init(settings: dict[str, Any]) -> None:
    from qwen35_native import SafetensorGGUFRowSource

    _WORKER.update(settings)
    _WORKER["codec"] = IQGGMLNativeCodec(settings["ggml_library"])
    _WORKER["source"] = SafetensorGGUFRowSource(settings["model_dir"])
    _, _WORKER["imatrix"] = read_imatrix(Path(settings["imatrix_path"]),
                                         Path(settings["gguf_python"]))


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
    experts, rows_per_expert, width = (int(v) for v in entry["candidate_source_shape"])
    if experts != EXPERTS:
        raise RuntimeError(f"expert count differs for {name}")
    sums, counts = expert_arrays(_WORKER["imatrix"], layer, family.split("_")[1])
    if sums.shape != (EXPERTS, width) or not (counts > 0).all():
        raise RuntimeError(f"imatrix shape or counts unexpected for {name}")
    importance = (sums / counts.astype(np.float32)[:, None]).astype(np.float32)
    offset, gsq_bytes = _WORKER["gsq_offsets"][name]
    gsq_row_size = codec.geometry(GGMLType.Q2_0, width)["row_size"]
    if gsq_bytes != experts * rows_per_expert * gsq_row_size:
        raise RuntimeError(f"GSQ payload size differs for {name}")
    gsq = np.memmap(_WORKER["gguf_path"], dtype=np.uint8, mode="r",
                    offset=offset, shape=(gsq_bytes,))
    expert_bytes = rows_per_expert * gsq_row_size
    keys = ("gsq_q2_0", type_name.lower())
    squared = dict.fromkeys(keys, 0.0)
    weighted = dict.fromkeys(keys, 0.0)
    energy = weighted_energy = 0.0
    q4_0_digest = hashlib.sha256()
    new_bytes = 0

    def chunks():
        nonlocal energy, weighted_energy, new_bytes
        expert = -1
        for expert, rows in enumerate(_WORKER["source"].iter_rows(
            entry, rows_per_chunk=rows_per_expert,
        )):
            rows = np.ascontiguousarray(rows, dtype=np.float32)
            if rows.shape != (rows_per_expert, width):
                raise RuntimeError(f"source chunk is not one expert: {name}")
            q4_0_digest.update(codec.quantize_rows(rows, GGMLType.Q4_0))
            payload = quantize_rows_with_importance(codec, rows, type_id, importance[expert])
            new_bytes += len(payload)
            weights = importance[expert]
            square = np.square(rows, dtype=np.float64)
            energy += float(square.sum())
            weighted_energy += float((square * weights).sum())
            decoded = np.empty_like(rows)
            for key, data, ggml_type in (
                ("gsq_q2_0", gsq[expert * expert_bytes:(expert + 1) * expert_bytes],
                 GGMLType.Q2_0),
                (type_name.lower(), payload, type_id),
            ):
                codec.dequantize_rows_into(data, ggml_type, decoded)
                difference = np.square(decoded - rows, dtype=np.float64)
                squared[key] += float(difference.sum())
                weighted[key] += float((difference * weights).sum())
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
        })
    if q4_0_digest.hexdigest() != _WORKER["stored_q4_0"][name]:
        raise RuntimeError(f"BF16 rows do not reproduce the stored Q4_0: {name}")
    metrics = {key: {"relative_rmse": math.sqrt(squared[key] / energy),
                     "weighted_relative_error": math.sqrt(weighted[key] / weighted_energy)}
               for key in keys}
    result = {
        "name": name,
        "layer": layer,
        "family": family,
        "type": type_name,
        "store_entry": store_entry,
        "bytes": {"gsq_q2_0": gsq_bytes, type_name: new_bytes},
        "metrics": metrics,
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(sidecar, result)
    return result


def run(args: argparse.Namespace) -> None:
    from native_gguf import import_pinned_gguf

    type_name = args.type
    output = args.reports / f"qwen36_post_release_iq_store_full_{type_name}.json"
    if output.exists() and _load_json(output).get("status") == "complete":
        print(f"store {type_name} (all experts): already complete", flush=True)
        return
    started = time.perf_counter()
    imatrix_sha256 = _sha256_file(args.imatrix)
    names = args.only or [f"blk.{layer}.{family}.weight"
                          for layer in range(LAYERS) for family in args.families]
    manifest = _load_json(args.manifest)
    manifest_entries = {e["destination_name"]: e for e in manifest["entries"]
                        if e["destination_name"] in set(names)}
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
        "kind": f"bf16_derived_imatrix_{type_name.lower()}_routed_experts_all",
        "model_dir": str(args.model_dir),
        "manifest_sha256": _sha256_file(args.manifest),
        "imatrix_sha256": imatrix_sha256,
    }
    settings = {
        "type_name": type_name,
        "ggml_library": str(args.ggml_library),
        "gguf_python": str(args.gguf_python),
        "model_dir": str(args.model_dir),
        "imatrix_path": str(args.imatrix),
        "imatrix_sha256": imatrix_sha256,
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
                              "bytes": result["bytes"],
                              "seconds": round(result["wall_seconds"], 1)}), flush=True)
        return
    results = {}
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init, initargs=(settings,),
    ) as pool:
        for result in pool.map(_quantize_tensor, names):
            results[result["name"]] = result
            print(json.dumps({
                "tensor": result["name"],
                "gsq_weighted": round(result["metrics"]["gsq_q2_0"]["weighted_relative_error"], 4),
                "new_weighted": round(result["metrics"][type_name.lower()]
                                      ["weighted_relative_error"], 4),
                "seconds": round(result["wall_seconds"], 1),
            }), flush=True)
    writer = NativeCandidateStoreWriter(store_root, codec, source=store_source)
    writer._entries = {name: {type_name: results[name]["store_entry"]} for name in names}
    index_path = writer.finalize()
    summary = {}
    for family in (*args.families, "all"):
        members = [n for n in names if family == "all" or results[n]["family"] == family]
        summary[family] = {
            "tensors": len(members),
            "bytes": {key: sum(results[n]["bytes"][key] for n in members)
                      for key in ("gsq_q2_0", type_name)},
            **{f"median_{metric}": {key: float(np.median(
                [results[n]["metrics"][key][metric] for n in members]))
                for key in ("gsq_q2_0", type_name.lower())}
               for metric in ("relative_rmse", "weighted_relative_error")},
        }
    _atomic_json(output, {
        "schema": SCHEMA,
        "status": "complete",
        "type": type_name,
        "families": list(args.families),
        "gsq_gguf_sha256": _sha256_file(args.gguf),
        "imatrix_path": str(args.imatrix),
        "imatrix_sha256": imatrix_sha256,
        "store": {"root": str(store_root), "index_sha256": _sha256_file(index_path)},
        "summary": summary,
        "tensors": results,
        "wall_seconds": time.perf_counter() - started,
    })
    print(f"store {type_name} (all experts): complete; " + json.dumps(summary["all"]), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--type", required=True, choices=("IQ2_XS", "IQ3_XXS"))
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--imatrix", type=Path,
                        default=ROOT / "data/unsloth_imatrix/imatrix_unsloth.gguf")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/qwen36_35b_base")
    parser.add_argument("--manifest", type=Path,
                        default=RCO / "reports/qwen36_35b_base_gguf_manifest.json")
    parser.add_argument("--gguf", type=Path,
                        default=ROOT / "results/Qwen3.6-35B-A3B-GSQ-hybrid.gguf")
    parser.add_argument("--gguf-python", type=Path, default=ROOT / "repos/llama.cpp/gguf-py")
    parser.add_argument("--ggml-library", type=Path,
                        default=ROOT / "experiment/build-cpu/bin/libggml-base.so.0.24.0")
    parser.add_argument("--native-store", type=Path, default=ROOT / "data/qwen36_35b_native")
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_post_release_full_iq")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", nargs="+", metavar="TENSOR",
                        help="smoke test: quantize these tensors in-process, no report")
    args = parser.parse_args()
    for key in ("imatrix", "model_dir", "manifest", "gguf", "gguf_python", "ggml_library",
                "native_store", "work", "reports"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main() -> int:
    run(_parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
