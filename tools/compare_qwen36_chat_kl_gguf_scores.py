#!/usr/bin/env python3
"""Pairwise comparison of llama.cpp held-out chat KL scores.

Reads the score reports of ``audit_qwen36_chat_kl_gguf.py score`` for the
given labels (for example the release candidate, authentic GSQ and an
outside GGUF of the same base model) and, for every ordered pair A, B with A
listed before B, reports A minus B: per-conversation mean top-20 KL to BF16
and NLL, template-token KL, and per stratum, each with the paired bootstrap
95% CI used by the Phase 8 gate.  The scores must share helper, llama.cpp
build, parameters, corpus, reference and conversation order.

    compare_qwen36_chat_kl_gguf_scores.py --labels rco_v2 gsq unsloth_ud_q2_k_xl \
        --output reports/qwen36_chat_kl_gguf_comparison_v2_heldout.json
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

RCO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file  # noqa: E402
from search_qwen36_prune24_q3k import _paired  # noqa: E402

SCHEMA = "rco.qwen36.chat_kl_gguf_comparison.v1"
SHARED = ("helper", "llama_cpp", "parameters", "corpus", "reference_sha256")


def compare(a: dict, b: dict) -> dict:
    """A minus B on per-conversation values; negative means A is closer to BF16."""
    template = [i for i in range(len(a["strata"]))
                if np.isfinite(a["conversation_template_kl"][i])
                and np.isfinite(b["conversation_template_kl"][i])]
    strata = np.asarray(a["strata"])
    return {
        "kl": _paired(a["conversation_mean_top20_kl"], b["conversation_mean_top20_kl"]),
        "nll": _paired(a["conversation_mean_nll"], b["conversation_mean_nll"]),
        "template_kl": _paired([a["conversation_template_kl"][i] for i in template],
                               [b["conversation_template_kl"][i] for i in template]),
        "per_stratum_kl": {
            stratum: _paired([a["conversation_mean_top20_kl"][i] for i in members],
                             [b["conversation_mean_top20_kl"][i] for i in members])
            for stratum in sorted(set(strata))
            for members in [np.flatnonzero(strata == stratum)]},
    }


def check_compatible(scores: dict[str, dict]) -> None:
    first_label, first = next(iter(scores.items()))
    for label, score in scores.items():
        if score.get("status") != "complete":
            raise RuntimeError(f"score {label} is not complete")
        for key in SHARED:
            if score["identity"][key] != first["identity"][key]:
                raise RuntimeError(f"scores {first_label} and {label} differ in {key}")
        if score["conversation_ids"] != first["conversation_ids"]:
            raise RuntimeError(f"scores {first_label} and {label} cover other conversations")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--split", default="heldout", choices=("calibration", "heldout"))
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {label: args.reports / f"qwen36_chat_kl_gguf_{label}_v2_{args.split}.json"
             for label in args.labels}
    scores = {label: _load_json(path) for label, path in paths.items()}
    check_compatible(scores)
    report = {
        "schema": SCHEMA, "status": "complete", "split": args.split,
        "metric": "per-conversation mean top-20 KL(BF16 || model), not renormalized; "
                  "A minus B, paired bootstrap 95% CI",
        "means": {label: {"kl": s["mean_top20_kl_to_bf16"], "nll": s["mean_nll"]}
                  for label, s in scores.items()},
        "pairs": {f"{a}_minus_{b}": compare(scores[a], scores[b])
                  for a, b in combinations(args.labels, 2)},
        "inputs": {label: {"report": str(path), "report_sha256": _sha256_file(path),
                           "model": scores[label]["identity"]["model"]}
                   for label, path in paths.items()},
        "shared_identity": {key: next(iter(scores.values()))["identity"][key] for key in SHARED},
    }
    _atomic_json(args.output, report)
    for label, means in report["means"].items():
        print(f"{label}: KL {means['kl']:.5f} NLL {means['nll']:.5f}")
    for name, pair in report["pairs"].items():
        kl = pair["kl"]
        print(f"{name}: KL {kl['mean_delta']:+.5f} CI [{kl['ci95'][0]:+.5f}, "
              f"{kl['ci95'][1]:+.5f}], {kl['improved_conversations']}/{kl['conversations']} lower")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
