#!/usr/bin/env python3
"""Phase 4 of RCO_PLAN_NEW.md: llama.cpp document NLL against the evaluator.

Compares, per v2 calibration document, the llama.cpp NLL of the P24 GGUF and
of the unpruned GSQ-E6 control (``audit_qwen36_gsq_gguf_nll.py``) with the
streaming evaluator's NLL of the same models: the consensus-mask score report
for P24 and the router-statistics report for the unpruned model.

It reports how closely each runtime agrees per document, and whether
llama.cpp reproduces the evaluator's pruning gain (P24 minus unpruned, paired
over documents).  No pass criterion was fixed in advance; the report is
descriptive.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

RCO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RCO / "src"))

from release_quality import paired_bootstrap_mean_ci  # noqa: E402

SCHEMA = "rco.qwen36.p24_v2_llama_nll_comparison.v1"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ci(deltas: np.ndarray) -> dict:
    lower, upper = paired_bootstrap_mean_ci([float(value) for value in deltas])
    return {"mean": float(deltas.mean()), "ci95": [lower, upper]}


def _agreement(llama: np.ndarray, native: np.ndarray, strata: list[str]) -> dict:
    difference = llama - native
    per_stratum = {}
    for stratum in dict.fromkeys(strata):
        rows = np.array([item == stratum for item in strata])
        per_stratum[stratum] = {
            "llama_mean_nll": float(llama[rows].mean()),
            "evaluator_mean_nll": float(native[rows].mean()),
            "mean_difference": float(difference[rows].mean()),
        }
    return {
        "llama_mean_nll": float(llama.mean()),
        "evaluator_mean_nll": float(native.mean()),
        "difference_llama_minus_evaluator": _ci(difference),
        "mean_absolute_difference": float(np.abs(difference).mean()),
        "max_absolute_difference": float(np.abs(difference).max()),
        "pearson_r": float(np.corrcoef(llama, native)[0, 1]),
        "per_stratum": per_stratum,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    reports = RCO / "reports"
    parser.add_argument("--p24-llama", type=Path,
                        default=reports / "qwen36_gsq_e6_p24_v2_llama_nll.json")
    parser.add_argument("--control-llama", type=Path,
                        default=reports / "qwen36_gsq_e6_v2_llama_nll.json")
    parser.add_argument("--p24-evaluator", type=Path,
                        default=reports / "qwen36_gsq_e6_prune24_score_consensus.json")
    parser.add_argument("--control-evaluator", type=Path,
                        default=reports / "qwen36_gsq_e6_prune24_router_stats.json")
    parser.add_argument("--p24-build", type=Path,
                        default=reports / "qwen36_gsq_e6_p24_gguf.json")
    parser.add_argument("--control-build", type=Path,
                        default=reports / "qwen36_gsq_e6_gguf.json")
    parser.add_argument("--output", type=Path,
                        default=reports / "qwen36_gsq_e6_p24_v2_llama_nll_comparison.json")
    args = parser.parse_args()

    llama = {"p24": _load(args.p24_llama), "control": _load(args.control_llama)}
    native = {"p24": _load(args.p24_evaluator), "control": _load(args.control_evaluator)}
    build = {"p24": _load(args.p24_build), "control": _load(args.control_build)}

    # The two llama.cpp runs differ only in the model.
    identities = [dict(llama[v]["identity"], model=None) for v in llama]
    if identities[0] != identities[1]:
        raise ValueError("llama.cpp reports differ in more than the model")
    for variant in llama:
        if llama[variant]["status"] != "complete" or native[variant]["status"] != "complete":
            raise ValueError(f"{variant}: an input report is incomplete")
        if llama[variant]["identity"]["model"]["sha256"] != build[variant]["output"]["sha256"]:
            raise ValueError(f"{variant}: llama.cpp scored a GGUF other than the build output")
        if build[variant]["inputs"]["gsq_gguf"]["sha256"] != native[variant]["identity"]["gsq_gguf_sha256"]:
            raise ValueError(f"{variant}: GGUF and evaluator start from different GSQ files")
    if build["p24"]["inputs"]["mask"]["sha256"] != native["p24"]["mask"]["sha256"]:
        raise ValueError("P24 GGUF and evaluator score use different masks")
    if native["p24"]["identity"]["calibration"] != native["control"]["identity"]["calibration"]:
        raise ValueError("evaluator reports use different calibration sets")

    strata = native["p24"]["identity"]["calibration"]["documents"]
    llama_nll, native_nll = {}, {}
    for variant in llama:
        documents = llama[variant]["documents"]
        if [item["index"] for item in documents] != list(range(len(strata))):
            raise ValueError(f"{variant}: llama.cpp documents out of order")
        llama_nll[variant] = np.array([item["mean_nll"] for item in documents])
        native_nll[variant] = np.array(native[variant]["document_mean_nll"], dtype=np.float64)
        if len(native_nll[variant]) != len(strata) or not np.isfinite(llama_nll[variant]).all():
            raise ValueError(f"{variant}: per-document scores do not line up")

    gain_llama = llama_nll["p24"] - llama_nll["control"]
    gain_native = native_nll["p24"] - native_nll["control"]
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "question": ("Does llama.cpp, running the physically pruned P24 GGUF, reproduce the "
                     "streaming evaluator's per-document NLL and its pruning gain over the "
                     "unpruned GSQ-E6 model?"),
        "criteria": "none fixed in advance; descriptive",
        "documents": len(strata),
        "inputs": {name: str(path) for name, path in vars(args).items() if name != "output"},
        "agreement": {variant: _agreement(llama_nll[variant], native_nll[variant], strata)
                      for variant in llama},
        "pruning_gain_p24_minus_unpruned": {
            "llama_cpp": _ci(gain_llama),
            "evaluator": _ci(gain_native),
            "difference_llama_minus_evaluator": _ci(gain_llama - gain_native),
            "pearson_r_per_document": float(np.corrcoef(gain_llama, gain_native)[0, 1]),
            "documents_improved": {"llama_cpp": int((gain_llama < 0).sum()),
                                   "evaluator": int((gain_native < 0).sum())},
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("agreement", "pruning_gain_p24_minus_unpruned")},
                     indent=1, default=lambda value: round(value, 5) if math.isfinite(value) else value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
