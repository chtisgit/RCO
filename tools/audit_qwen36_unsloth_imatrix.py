#!/usr/bin/env python3
"""Unsloth's published imatrix for Qwen3.6-35B-A3B (PLAN_POST_RCO_RELEASE.md).

``imatrix_unsloth.gguf_file`` from ``unsloth/Qwen3.6-35B-A3B-GGUF`` is
llama.cpp's GGUF imatrix: per tensor ``in_sum2`` (sum of squared inputs per
column; per expert for the routed experts) and ``counts`` (tokens; per
expert).  Subcommands:

``reproduce --tensor T [--experts N]``
    Quantize the first N experts of tensor T from the BF16 checkpoint with
    Unsloth's imatrix (``in_sum2 / counts``, as ``llama-quantize`` uses it) to
    the type T has in Unsloth's UD-Q2_K_XL GGUF, and compare the bytes with
    that GGUF.  Identical bytes mean we read the imatrix as llama.cpp does.
``convert``
    Write it in the layout of our Phase 5 imatrix (``blk{L}_gate_up_sum``,
    ``blk{L}_down_sum``, ``blk{L}_count``) with the release's 24 pruned experts
    per layer zeroed, so that ``build_qwen36_post_release_iq_store.py`` can
    use it unchanged for the pruned (P24) comparison with test A.  Report:
    ``reports/qwen36_unsloth_imatrix_p24.json`` (the fields the store builder
    reads, plus provenance).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file  # noqa: E402

LAYERS = 40
EXPERTS = 256
SOURCE_URL = ("https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/"
              "imatrix_unsloth.gguf_file")
SOURCE_REVISION = "a483e9e6cbd595906af30beda3187c2663a1118c"


def read_imatrix(path: Path, gguf_python: Path) -> tuple[dict, dict[str, np.ndarray]]:
    from native_gguf import import_pinned_gguf

    gguf = import_pinned_gguf(gguf_python)
    reader = gguf.GGUFReader(path)
    header = {}
    for key in ("general.type", "imatrix.datasets", "imatrix.chunk_count", "imatrix.chunk_size"):
        header[key] = reader.fields[key].contents()
    if header["general.type"] != "imatrix":
        raise RuntimeError("not a GGUF imatrix")
    tensors = {t.name: np.array(t.data, dtype=np.float32) for t in reader.tensors}
    return header, tensors


def expert_arrays(tensors: dict[str, np.ndarray], layer: int, family: str):
    """in_sum2 [experts, columns] and counts [experts] for a routed-expert tensor."""
    sums = tensors[f"blk.{layer}.ffn_{family}_exps.weight.in_sum2"]
    counts = tensors[f"blk.{layer}.ffn_{family}_exps.weight.counts"].reshape(-1)
    if sums.shape[0] != EXPERTS or counts.shape != (EXPERTS,):
        raise RuntimeError(f"unexpected imatrix shape for layer {layer} {family}: {sums.shape}")
    return sums, counts


def run_reproduce(args) -> int:
    from native_gguf import import_pinned_gguf
    from qwen35_native import SafetensorGGUFRowSource
    from quant.ggml_importance import quantize_rows_with_importance
    from quant.ggml_native_iq import IQGGMLNativeCodec

    _, tensors = read_imatrix(args.imatrix, args.gguf_python)
    layer, family = int(args.tensor.split(".")[1]), args.tensor.split(".")[2].split("_")[1]
    sums, counts = expert_arrays(tensors, layer, family)
    gguf = import_pinned_gguf(args.gguf_python)
    reader = gguf.GGUFReader(args.unsloth_gguf)
    target = next(t for t in reader.tensors if t.name == args.tensor)
    type_id = int(target.tensor_type)
    codec = IQGGMLNativeCodec(args.ggml_library)
    manifest = _load_json(args.manifest)
    entry = next(e for e in manifest["entries"] if e["destination_name"] == args.tensor)
    experts, rows_per_expert, width = (int(v) for v in entry["candidate_source_shape"])
    row_size = int(codec.geometry(type_id, width)["row_size"])
    expert_bytes = rows_per_expert * row_size
    unsloth = np.memmap(args.unsloth_gguf, dtype=np.uint8, mode="r",
                        offset=int(target.data_offset), shape=(int(target.n_bytes),))
    if int(target.n_bytes) != experts * expert_bytes:
        raise RuntimeError("Unsloth tensor size differs from the expected geometry")
    source = SafetensorGGUFRowSource(args.model_dir)
    results = []
    for expert, rows in enumerate(source.iter_rows(entry, rows_per_chunk=rows_per_expert)):
        if expert >= args.experts:
            break
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        # float32 division, as llama-quantize does it
        importance = (sums[expert] / np.float32(counts[expert])).astype(np.float32)
        ours = quantize_rows_with_importance(codec, rows, type_id, importance)
        theirs = bytes(unsloth[expert * expert_bytes:(expert + 1) * expert_bytes])
        equal_rows = sum(ours[i * row_size:(i + 1) * row_size] == theirs[i * row_size:(i + 1) * row_size]
                         for i in range(rows_per_expert))
        results.append({"expert": expert, "identical": ours == theirs,
                        "identical_rows": int(equal_rows), "rows": rows_per_expert})
        print(f"{args.tensor} expert {expert} ({codec.type_name(type_id)}): identical "
              f"{ours == theirs}, rows {equal_rows}/{rows_per_expert}", flush=True)
    return 0 if all(r["identical"] for r in results) else 1


def run_convert(args) -> int:
    header, tensors = read_imatrix(args.imatrix, args.gguf_python)
    mask_report = _load_json(args.reports / "qwen36_gsq_e6_prune24_chat_frequency_mask.json")
    mask_path = Path(mask_report["mask"]["path"])
    if _sha256_file(mask_path) != mask_report["mask"]["sha256"]:
        raise RuntimeError("frequency mask differs from its report")
    mask = np.load(mask_path)
    arrays, gate_up_equal = {}, []
    for layer in range(LAYERS):
        gate, gate_counts = expert_arrays(tensors, layer, "gate")
        up, up_counts = expert_arrays(tensors, layer, "up")
        down, down_counts = expert_arrays(tensors, layer, "down")
        if not (np.array_equal(gate_counts, up_counts) and np.array_equal(gate_counts, down_counts)):
            raise RuntimeError(f"expert counts differ between families in layer {layer}")
        gate_up_equal.append(bool(np.array_equal(gate, up)))
        pruned = mask[layer]
        counts = np.where(pruned, 0, np.rint(gate_counts)).astype(np.int64)
        arrays[f"blk{layer}_gate_up_sum"] = np.where(pruned[:, None], 0.0, gate)
        arrays[f"blk{layer}_down_sum"] = np.where(pruned[:, None], 0.0, down)
        arrays[f"blk{layer}_count"] = counts
    if not all(gate_up_equal):
        raise RuntimeError("gate and up in_sum2 differ; they share one input")
    args.work.mkdir(parents=True, exist_ok=True)
    path = args.work / "imatrix_unsloth_p24.npz"
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)
    counts_all = np.stack([arrays[f"blk{l}_count"] for l in range(LAYERS)])
    _atomic_json(args.reports / "qwen36_unsloth_imatrix_p24.json", {
        "schema": "rco.qwen36.unsloth_imatrix_p24.v1",
        "status": "complete",
        "source": {"url": SOURCE_URL, "revision": SOURCE_REVISION,
                   "path": str(args.imatrix), "sha256": _sha256_file(args.imatrix),
                   "header": {k: (v if isinstance(v, (int, float, str)) else list(v))
                              for k, v in header.items()}},
        "corpus": {"limit": None, "description": "Unsloth's calibration text (not published)"},
        "imatrix": {"path": str(path), "sha256": _sha256_file(path),
                    "layout": "our Phase 5 layout; pruned experts zeroed"},
        "mask": {"path": str(mask_path), "sha256": mask_report["mask"]["sha256"]},
        "kept_expert_counts": {"min": int(counts_all[~mask].min()),
                               "median": float(np.median(counts_all[~mask]))},
    })
    print(f"convert: {path}; kept-expert tokens min {counts_all[~mask].min()}, "
          f"median {np.median(counts_all[~mask])}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("reproduce", "convert"))
    parser.add_argument("--imatrix", type=Path,
                        default=ROOT / "data/unsloth_imatrix/imatrix_unsloth.gguf")
    parser.add_argument("--unsloth-gguf", type=Path,
                        default=Path("/extraspace/IronWolf/christian/LLMs/"
                                     "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"))
    parser.add_argument("--tensor", default="blk.0.ffn_gate_exps.weight")
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/qwen36_35b_base")
    parser.add_argument("--manifest", type=Path,
                        default=RCO / "reports/qwen36_35b_base_gguf_manifest.json")
    parser.add_argument("--gguf-python", type=Path, default=ROOT / "repos/llama.cpp/gguf-py")
    parser.add_argument("--ggml-library", type=Path,
                        default=ROOT / "experiment/build-cpu/bin/libggml-base.so.0.24.0")
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_post_release_unsloth_imatrix")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    args = parser.parse_args()
    return {"reproduce": run_reproduce, "convert": run_convert}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
