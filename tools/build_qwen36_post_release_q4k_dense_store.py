#!/usr/bin/env python3
"""Q4_K store for the parts other than the routed experts (PLAN_POST_RCO_RELEASE.md).

Three groups, all Q8_0 in the GSQ GGUF and in the release:

- ``attention``: ``attn_qkv``, ``attn_gate``, ``ssm_out`` (30 linear-attention
  layers) and ``attn_q``, ``attn_k``, ``attn_v``, ``attn_output`` (10 full-
  attention layers), 130 tensors.  ``ssm_alpha``/``ssm_beta`` stay as they
  are (BF16, 8 MB in all);
- ``shared_expert``: ``ffn_{gate,up,down}_shexp``, 120 tensors;
- ``output``: ``output.weight``.

Each tensor is quantized from BF16 to Q4_K with Unsloth's published imatrix
(``in_sum2 / counts`` in float32, as ``llama-quantize`` uses it), except
``output.weight``: Unsloth's imatrix has no entry for it (llama.cpp's
imatrix tool skips the output by default), so it is quantized without one,
as ``llama-quantize`` does then.  Before a payload is accepted, the BF16 rows
must reproduce the checksum of the stored Q4_0 candidate, as in Phase 6.
Per tensor: relative RMSE and imatrix-weighted relative error against BF16
for the GGUF's Q8_0 and for Q4_K, and the bytes of both.  Resumable per
tensor.
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

from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file  # noqa: E402
from audit_qwen36_unsloth_imatrix import read_imatrix  # noqa: E402
from native_store import NativeCandidateStore, NativeCandidateStoreWriter  # noqa: E402
from quant.ggml_importance import quantize_rows_with_importance  # noqa: E402
from quant.ggml_native import GGMLNativeCodec, GGMLType  # noqa: E402

SCHEMA = "rco.qwen36.post_release_q4k_dense_store.v1"
OUTPUT = "qwen36_post_release_q4k_dense_store.json"
Q8_0_TYPE = 8
ROWS_PER_CHUNK = 4096
LINEAR_LAYERS = [layer for layer in range(40) if layer % 4 != 3]
FULL_LAYERS = [layer for layer in range(40) if layer % 4 == 3]


def groups() -> dict[str, list[str]]:
    attention = sorted(
        [f"blk.{l}.{f}.weight" for l in LINEAR_LAYERS for f in ("attn_qkv", "attn_gate", "ssm_out")]
        + [f"blk.{l}.{f}.weight" for l in FULL_LAYERS
           for f in ("attn_q", "attn_k", "attn_v", "attn_output")],
        key=lambda n: (int(n.split(".")[1]), n))
    shared = [f"blk.{l}.ffn_{f}_shexp.weight" for l in range(40) for f in ("gate", "up", "down")]
    return {"attention": attention, "shared_expert": shared, "output": ["output.weight"]}


_WORKER: dict[str, Any] = {}


def _worker_init(settings: dict[str, Any]) -> None:
    from native_gguf import import_pinned_gguf
    from qwen35_native import SafetensorGGUFRowSource

    _WORKER.update(settings)
    _WORKER["codec"] = GGMLNativeCodec(settings["ggml_library"])
    _WORKER["source"] = SafetensorGGUFRowSource(settings["model_dir"])
    _WORKER["gguf"] = import_pinned_gguf(settings["gguf_python"])


def _quantize_tensor(name: str) -> dict[str, Any]:
    codec: GGMLNativeCodec = _WORKER["codec"]
    gguf = _WORKER["gguf"]
    sidecar = Path(_WORKER["store_root"]) / "tensors" / quote(name, safe="") / "Q4_K.metrics.json"
    if sidecar.exists():
        return _load_json(sidecar)
    started = time.perf_counter()
    entry = _WORKER["manifest_entries"][name]
    rows_total, width = (int(v) for v in entry["candidate_source_shape"])
    importance = _WORKER["importance"].get(name)
    weights = np.ones(width, dtype=np.float32) if importance is None else importance
    offset, q8_bytes = _WORKER["gguf_offsets"][name]
    q8_row = width // 32 * 34
    if q8_bytes != rows_total * q8_row:
        raise RuntimeError(f"Q8_0 payload size differs for {name}")
    q8 = np.memmap(_WORKER["gguf_path"], dtype=np.uint8, mode="r",
                   offset=offset, shape=(q8_bytes,))
    squared = {"q8_0": 0.0, "q4_k": 0.0}
    weighted = {"q8_0": 0.0, "q4_k": 0.0}
    energy = weighted_energy = 0.0
    q4_0_digest = hashlib.sha256()
    q4k_bytes = 0
    row = 0

    def chunks():
        nonlocal energy, weighted_energy, q4k_bytes, row
        for rows in _WORKER["source"].iter_rows(entry, rows_per_chunk=ROWS_PER_CHUNK):
            rows = np.ascontiguousarray(rows, dtype=np.float32)
            count = rows.shape[0]
            q4_0_digest.update(codec.quantize_rows(rows, GGMLType.Q4_0))
            payload = (codec.quantize_rows(rows, GGMLType.Q4_K) if importance is None
                       else quantize_rows_with_importance(codec, rows, GGMLType.Q4_K, importance))
            q4k_bytes += len(payload)
            square = np.square(rows, dtype=np.float64)
            energy += float(square.sum())
            weighted_energy += float((square * weights).sum())
            decoded = {
                "q8_0": gguf.quants.dequantize(
                    np.asarray(q8[row * q8_row:(row + count) * q8_row]),
                    gguf.GGMLQuantizationType.Q8_0).reshape(count, width),
                "q4_k": np.empty_like(rows),
            }
            codec.dequantize_rows_into(payload, GGMLType.Q4_K, decoded["q4_k"])
            for key, values in decoded.items():
                difference = np.square(values.astype(np.float32) - rows, dtype=np.float64)
                squared[key] += float(difference.sum())
                weighted[key] += float((difference * weights).sum())
            row += count
            yield payload

    writer = NativeCandidateStoreWriter(Path(_WORKER["store_root"]), codec,
                                        source=_WORKER["store_source"])
    store_entry = writer.write_packed_chunks(
        name, GGMLType.Q4_K, entry["destination_gguf_shape"], chunks(),
        provenance={
            "kind": "bf16_derived_q4_k" + ("" if importance is None else "_imatrix"),
            "source_tensor": entry["source_name"],
            "source_shard": entry["source_shard"],
            "imatrix_sha256": None if importance is None else _WORKER["imatrix_sha256"],
        })
    if row != rows_total:
        raise RuntimeError(f"source row count differs for {name}: {row}")
    if q4_0_digest.hexdigest() != _WORKER["stored_q4_0"][name]:
        raise RuntimeError(f"BF16 rows do not reproduce the stored Q4_0: {name}")
    result = {
        "name": name,
        "group": _WORKER["group_of"][name],
        "imatrix": importance is not None,
        "store_entry": store_entry,
        "bytes": {"q8_0": q8_bytes, "q4_k": q4k_bytes},
        "metrics": {key: {"relative_rmse": math.sqrt(squared[key] / energy),
                          "weighted_relative_error": math.sqrt(weighted[key] / weighted_energy)}
                    for key in squared},
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(sidecar, result)
    return result


def run(args: argparse.Namespace) -> None:
    from native_gguf import import_pinned_gguf

    output = args.reports / OUTPUT
    if output.exists() and _load_json(output).get("status") == "complete":
        print("store Q4_K dense: already complete", flush=True)
        return
    started = time.perf_counter()
    named = groups()
    names = args.only or [n for members in named.values() for n in members]
    group_of = {n: g for g, members in named.items() for n in members}
    header, tensors = read_imatrix(args.imatrix, args.gguf_python)
    imatrix_sha256 = _sha256_file(args.imatrix)
    importance = {}
    for name in names:
        if name == "output.weight":
            if f"{name}.in_sum2" in tensors:
                raise RuntimeError("Unsloth's imatrix unexpectedly has output.weight")
            continue
        sums = tensors[f"{name}.in_sum2"].reshape(-1)
        counts = tensors[f"{name}.counts"].reshape(-1)
        if counts.shape != (1,) or counts[0] <= 0:
            raise RuntimeError(f"unexpected imatrix counts for {name}: {counts}")
        importance[name] = (sums / np.float32(counts[0])).astype(np.float32)
    manifest = _load_json(args.manifest)
    manifest_entries = {e["destination_name"]: e for e in manifest["entries"]
                        if e["destination_name"] in set(names)}
    if len(manifest_entries) != len(names):
        raise RuntimeError("manifest does not map every tensor")
    for name in names:
        if importance.get(name) is not None and importance[name].shape != (
                int(manifest_entries[name]["candidate_source_shape"][1]),):
            raise RuntimeError(f"imatrix width differs for {name}")
    codec = GGMLNativeCodec(args.ggml_library)
    native = NativeCandidateStore(args.native_store, codec)
    stored_q4_0 = {name: native.metadata(name, GGMLType.Q4_0)["sha256"] for name in names}
    gguf = import_pinned_gguf(args.gguf_python)
    reader = gguf.GGUFReader(args.gguf)
    offsets = {}
    for tensor in reader.tensors:
        if tensor.name in manifest_entries:
            if int(tensor.tensor_type) != Q8_0_TYPE:
                raise RuntimeError(f"GSQ GGUF tensor is not Q8_0: {tensor.name}")
            offsets[tensor.name] = (int(tensor.data_offset), int(tensor.n_bytes))
    del reader
    if len(offsets) != len(names):
        raise RuntimeError("GSQ GGUF does not contain every tensor")
    store_root = args.work / "store_Q4_K"
    store_source = {
        "kind": "bf16_derived_q4_k_dense",
        "model_dir": str(args.model_dir),
        "manifest_sha256": _sha256_file(args.manifest),
        "imatrix_sha256": imatrix_sha256,
    }
    settings = {
        "ggml_library": str(args.ggml_library),
        "gguf_python": str(args.gguf_python),
        "model_dir": str(args.model_dir),
        "importance": importance,
        "imatrix_sha256": imatrix_sha256,
        "manifest_entries": manifest_entries,
        "gguf_offsets": offsets,
        "gguf_path": str(args.gguf),
        "store_root": str(store_root),
        "store_source": store_source,
        "stored_q4_0": stored_q4_0,
        "group_of": group_of,
    }
    if args.only:
        _worker_init(settings)
        for name in args.only:
            result = _quantize_tensor(name)
            print(json.dumps({"tensor": name, "metrics": result["metrics"],
                              "bytes": result["bytes"],
                              "seconds": round(result["wall_seconds"], 1)}), flush=True)
        return
    # largest first, so output.weight does not finish last on one worker
    order = sorted(names, key=lambda n: -int(np.prod(manifest_entries[n]["candidate_source_shape"])))
    results = {}
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init, initargs=(settings,),
    ) as pool:
        for result in pool.map(_quantize_tensor, order):
            results[result["name"]] = result
            print(json.dumps({
                "tensor": result["name"],
                "q8_0_weighted": round(result["metrics"]["q8_0"]["weighted_relative_error"], 5),
                "q4_k_weighted": round(result["metrics"]["q4_k"]["weighted_relative_error"], 5),
                "seconds": round(result["wall_seconds"], 1),
            }), flush=True)
    writer = NativeCandidateStoreWriter(store_root, codec, source=store_source)
    writer._entries = {name: {"Q4_K": results[name]["store_entry"]} for name in names}
    index_path = writer.finalize()
    summary = {}
    for group, members in named.items():
        summary[group] = {
            "tensors": len(members),
            "bytes": {key: sum(results[n]["bytes"][key] for n in members)
                      for key in ("q8_0", "q4_k")},
            **{f"median_{metric}": {key: float(np.median(
                [results[n]["metrics"][key][metric] for n in members]))
                for key in ("q8_0", "q4_k")}
               for metric in ("relative_rmse", "weighted_relative_error")},
        }
    _atomic_json(output, {
        "schema": SCHEMA,
        "status": "complete",
        "type": "Q4_K",
        "gsq_gguf_sha256": _sha256_file(args.gguf),
        "imatrix": {"path": str(args.imatrix), "sha256": imatrix_sha256,
                    "header": {k: (v if isinstance(v, (int, float, str)) else list(v))
                               for k, v in header.items()},
                    "without": ["output.weight"]},
        "groups": named,
        "store": {"root": str(store_root), "index_sha256": _sha256_file(index_path)},
        "summary": summary,
        "tensors": results,
        "wall_seconds": time.perf_counter() - started,
    })
    print("store Q4_K dense: complete; " + json.dumps(
        {g: {"bytes": s["bytes"], "weighted": s["median_weighted_relative_error"]}
         for g, s in summary.items()}), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
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
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_post_release_q4k_dense")
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
