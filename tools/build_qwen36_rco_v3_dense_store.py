#!/usr/bin/env python3
"""Q6_K or Q8_0 store for attention/SSM and shared expert (RCO_PLAN_V3.md).

The 250 dense search tensors of ``build_qwen36_post_release_q4k_dense_store``
(``attention`` and ``shared_expert`` groups; ``output.weight`` is fixed at
Q4_K in v3 and not built here):

- ``--type Q6_K``: quantized from BF16 with Unsloth's imatrix (``in_sum2 /
  counts`` in float32, as ``llama-quantize`` uses it);
- ``--type Q8_0``: the GSQ GGUF's own Q8_0 bytes, copied, so that the
  evaluator can install them over a center GGUF that holds another type.

Before a payload is accepted, the BF16 rows must reproduce the checksum of
the stored Q4_0 candidate, as in Phase 6.  Per tensor: relative RMSE and
imatrix-weighted relative error against BF16, and the bytes.  Resumable per
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
from build_qwen36_post_release_q4k_dense_store import Q8_0_TYPE, ROWS_PER_CHUNK, groups  # noqa: E402
from native_store import NativeCandidateStore, NativeCandidateStoreWriter  # noqa: E402
from quant.ggml_importance import quantize_rows_with_importance  # noqa: E402
from quant.ggml_native import GGMLType  # noqa: E402
from quant.ggml_native_mixed import MIXED_TYPE_IDS, MixedGGMLNativeCodec  # noqa: E402

SCHEMA = "rco.qwen36.rco_v3_dense_store.v1"

_WORKER: dict[str, Any] = {}


def _worker_init(settings: dict[str, Any]) -> None:
    from qwen35_native import SafetensorGGUFRowSource

    _WORKER.update(settings)
    _WORKER["codec"] = MixedGGMLNativeCodec(settings["ggml_library"])
    _WORKER["source"] = SafetensorGGUFRowSource(settings["model_dir"])


def _quantize_tensor(name: str) -> dict[str, Any]:
    codec: MixedGGMLNativeCodec = _WORKER["codec"]
    type_name = _WORKER["type_name"]
    type_id = MIXED_TYPE_IDS[type_name]
    sidecar = (Path(_WORKER["store_root"]) / "tensors" / quote(name, safe="")
               / f"{type_name}.metrics.json")
    if sidecar.exists():
        return _load_json(sidecar)
    started = time.perf_counter()
    entry = _WORKER["manifest_entries"][name]
    rows_total, width = (int(v) for v in entry["candidate_source_shape"])
    importance = _WORKER["importance"][name]
    offset, q8_bytes = _WORKER["gguf_offsets"][name]
    q8_row = width // 32 * 34
    if q8_bytes != rows_total * q8_row:
        raise RuntimeError(f"Q8_0 payload size differs for {name}")
    q8 = np.memmap(_WORKER["gguf_path"], dtype=np.uint8, mode="r",
                   offset=offset, shape=(q8_bytes,))
    squared = weighted = energy = weighted_energy = 0.0
    q4_0_digest = hashlib.sha256()
    new_bytes = 0
    row = 0

    def chunks():
        nonlocal squared, weighted, energy, weighted_energy, new_bytes, row
        for rows in _WORKER["source"].iter_rows(entry, rows_per_chunk=ROWS_PER_CHUNK):
            rows = np.ascontiguousarray(rows, dtype=np.float32)
            count = rows.shape[0]
            q4_0_digest.update(codec.quantize_rows(rows, GGMLType.Q4_0))
            if type_name == "Q8_0":
                payload = bytes(q8[row * q8_row:(row + count) * q8_row])
            else:
                payload = quantize_rows_with_importance(codec, rows, type_id, importance)
            new_bytes += len(payload)
            decoded = np.empty_like(rows)
            codec.dequantize_rows_into(payload, type_id, decoded)
            square = np.square(rows, dtype=np.float64)
            difference = np.square(decoded - rows, dtype=np.float64)
            energy += float(square.sum())
            weighted_energy += float((square * importance).sum())
            squared += float(difference.sum())
            weighted += float((difference * importance).sum())
            row += count
            yield payload

    writer = NativeCandidateStoreWriter(Path(_WORKER["store_root"]), codec,
                                        source=_WORKER["store_source"])
    store_entry = writer.write_packed_chunks(
        name, type_id, entry["destination_gguf_shape"], chunks(),
        provenance={
            "kind": ("gsq_gguf_q8_0_copy" if type_name == "Q8_0"
                     else f"bf16_derived_{type_name.lower()}_imatrix"),
            "source_tensor": entry["source_name"],
            "source_shard": entry["source_shard"],
            "imatrix_sha256": None if type_name == "Q8_0" else _WORKER["imatrix_sha256"],
        })
    if row != rows_total:
        raise RuntimeError(f"source row count differs for {name}: {row}")
    if q4_0_digest.hexdigest() != _WORKER["stored_q4_0"][name]:
        raise RuntimeError(f"BF16 rows do not reproduce the stored Q4_0: {name}")
    result = {
        "name": name,
        "group": _WORKER["group_of"][name],
        "type": type_name,
        "store_entry": store_entry,
        "bytes": {"q8_0": q8_bytes, type_name.lower(): new_bytes},
        "metrics": {"relative_rmse": math.sqrt(squared / energy),
                    "weighted_relative_error": math.sqrt(weighted / weighted_energy)},
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(sidecar, result)
    return result


def run(args: argparse.Namespace) -> None:
    from native_gguf import import_pinned_gguf

    type_name = args.type
    output = args.reports / f"qwen36_rco_v3_dense_store_{type_name}.json"
    if output.exists() and _load_json(output).get("status") == "complete":
        print(f"dense store {type_name}: already complete", flush=True)
        return
    started = time.perf_counter()
    named = {g: m for g, m in groups().items() if g != "output"}
    names = args.only or [n for members in named.values() for n in members]
    group_of = {n: g for g, members in named.items() for n in members}
    header, tensors = read_imatrix(args.imatrix, args.gguf_python)
    imatrix_sha256 = _sha256_file(args.imatrix)
    importance = {}
    for name in names:
        counts = tensors[f"{name}.counts"].reshape(-1)
        if counts.shape != (1,) or counts[0] <= 0:
            raise RuntimeError(f"unexpected imatrix counts for {name}: {counts}")
        importance[name] = (tensors[f"{name}.in_sum2"].reshape(-1)
                            / np.float32(counts[0])).astype(np.float32)
    manifest = _load_json(args.manifest)
    manifest_entries = {e["destination_name"]: e for e in manifest["entries"]
                        if e["destination_name"] in set(names)}
    if len(manifest_entries) != len(names):
        raise RuntimeError("manifest does not map every tensor")
    for name in names:
        if importance[name].shape != (int(manifest_entries[name]["candidate_source_shape"][1]),):
            raise RuntimeError(f"imatrix width differs for {name}")
    codec = MixedGGMLNativeCodec(args.ggml_library)
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
    store_root = args.work / f"store_{type_name}"
    store_source = {
        "kind": f"rco_v3_dense_{type_name.lower()}",
        "model_dir": str(args.model_dir),
        "manifest_sha256": _sha256_file(args.manifest),
        "imatrix_sha256": imatrix_sha256,
        "gsq_gguf": str(args.gguf),
    }
    settings = {
        "type_name": type_name,
        "ggml_library": str(args.ggml_library),
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
    order = sorted(names, key=lambda n: -int(np.prod(manifest_entries[n]["candidate_source_shape"])))
    results = {}
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init, initargs=(settings,),
    ) as pool:
        for result in pool.map(_quantize_tensor, order):
            results[result["name"]] = result
            print(json.dumps({"tensor": result["name"],
                              "weighted": round(result["metrics"]["weighted_relative_error"], 5),
                              "seconds": round(result["wall_seconds"], 1)}), flush=True)
    writer = NativeCandidateStoreWriter(store_root, codec, source=store_source)
    writer._entries = {name: {type_name: results[name]["store_entry"]} for name in names}
    index_path = writer.finalize()
    summary = {}
    for group, members in named.items():
        summary[group] = {
            "tensors": len(members),
            "bytes": {key: sum(results[n]["bytes"][key] for n in members)
                      for key in ("q8_0", type_name.lower())},
            **{f"median_{metric}": float(np.median(
                [results[n]["metrics"][metric] for n in members]))
               for metric in ("relative_rmse", "weighted_relative_error")},
        }
    _atomic_json(output, {
        "schema": SCHEMA,
        "status": "complete",
        "type": type_name,
        "gsq_gguf_sha256": _sha256_file(args.gguf),
        "imatrix": {"path": str(args.imatrix), "sha256": imatrix_sha256,
                    "header": {k: (v if isinstance(v, (int, float, str)) else list(v))
                               for k, v in header.items()},
                    "used": type_name != "Q8_0"},
        "groups": named,
        "store": {"root": str(store_root), "index_sha256": _sha256_file(index_path)},
        "summary": summary,
        "tensors": results,
        "wall_seconds": time.perf_counter() - started,
    })
    print(f"dense store {type_name}: complete; " + json.dumps(
        {g: {"bytes": s["bytes"], "weighted": s["median_weighted_relative_error"]}
         for g, s in summary.items()}), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--type", required=True, choices=("Q6_K", "Q8_0"))
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
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_rco_v3")
    parser.add_argument("--reports", type=Path, default=RCO / "reports/rco_v3")
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
