#!/usr/bin/env python3
"""Step 1 crosscheck of RCO_PLAN_V3.md: the center GGUF in the exact evaluator.

The evaluator loads the center GGUF (``gate``/``up`` IQ2_XS, ``down``
IQ3_XXS, dense Q6_K, output Q4_K) through ``MixedGGUFManifestPrefixLoader``
and installs from the v3 stores.  One pass over the validation part (the
upgrade-rate run's setting), three variants:

- ``center``;
- ``center_dense_q8_0``: the 250 dense tensors installed from the Q8_0 store.
  Must reproduce the upgrade-rate run's ``iq2_xs_down_iq3_xxs+output_q4k``
  exactly (within 1e-9);
- ``center_dense_q4_k``: the 250 from the Q4_K store.  Must reproduce
  ``iq2_xs_down_iq3_xxs+dense_q4k+output_q4k``.

``center`` minus ``center_dense_q8_0`` is the first measurement of dense Q6_K.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

RCO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file  # noqa: E402
from rco_v3_common import (  # noqa: E402
    EVALUATOR_PATHS,
    add_evaluator_arguments,
    center_assignment,
    evaluator,
    gpu_bytes,
    load_store_reports,
    searched,
    upgrade_keys,
)
from search_qwen36_prune24_q3k import _paired, _parts  # noqa: E402

SCHEMA = "rco.qwen36.rco_v3_crosscheck.v1"
OUTPUT = "rco_v3/qwen36_rco_v3_crosscheck.json"
TOLERANCE = 1e-9
TARGETS = {"center_dense_q8_0": "iq2_xs_down_iq3_xxs+output_q4k",
           "center_dense_q4_k": "iq2_xs_down_iq3_xxs+dense_q4k+output_q4k"}


def run(args) -> int:
    from native_gguf import import_pinned_gguf
    from search.streamed_chat_exact import ExactVariant

    output = args.reports / OUTPUT
    if output.exists() and _load_json(output).get("status") in ("pass", "fail"):
        print(f"crosscheck: already done ({_load_json(output)['status']})", flush=True)
        return 0
    started = time.perf_counter()
    gguf_report = _load_json(args.gguf_report)
    if gguf_report.get("status") != "built" or gguf_report["assignment_source"] != "center":
        raise RuntimeError("the center GGUF is not built")
    gguf_path = Path(gguf_report["output"]["path"])
    if _sha256_file(gguf_path) != gguf_report["output"]["sha256"]:
        raise RuntimeError("center GGUF differs from its report")
    upgrade_rate = _load_json(args.reports / "qwen36_post_release_upgrade_rate.json")
    reports = load_store_reports()
    center = center_assignment()
    dense = searched()["attention"] + searched()["shared_expert"]
    assignments = {"center": center,
                   "center_dense_q8_0": {**center, **{n: "Q8_0" for n in dense}},
                   "center_dense_q4_k": {**center, **{n: "Q4_K" for n in dense}}}
    gguf = import_pinned_gguf(args.gguf_python)
    gsq_bytes = {t.name: int(t.n_bytes) for t in gguf.GGUFReader(args.gsq_gguf).tensors}
    if gpu_bytes(center, reports, gsq_bytes) != gguf_report["budget"]["gpu_bytes"]:
        raise RuntimeError("center GPU bytes differ between the registry and the GGUF")
    parts, data_identity = _parts(args)
    exact, identity = evaluator(args, gguf_path, reports)
    variants = [ExactVariant(label, upgrades=upgrade_keys(a, center))
                for label, a in assignments.items()]
    rows = parts["validation"]["rows"]
    result = exact.evaluate(rows, variants)
    arms = {}
    for number, variant in enumerate(variants):
        kl = result.row_mean_kl(number).tolist()
        nll = result.row_mean_nll(number).tolist()
        arms[variant.label] = {
            "conversation_mean_top20_kl": kl, "conversation_mean_nll": nll,
            "mean_top20_kl_to_bf16": float(np.mean(kl)), "mean_nll": float(np.mean(nll)),
            "gpu_bytes": gpu_bytes(assignments[variant.label], reports, gsq_bytes),
            "installs": len(variant.upgrades)}
    reproduced = {
        label: float(np.max(np.abs(np.asarray(arms[label]["conversation_mean_top20_kl"])
                                   - upgrade_rate["arms"][target]["conversation_mean_top20_kl"])))
        for label, target in TARGETS.items()}
    ok = all(v <= TOLERANCE for v in reproduced.values())
    strata = np.asarray([c["stratum"] for c in parts["validation"]["conversations"]])

    def compare(a, b):
        ka, kb = arms[a]["conversation_mean_top20_kl"], arms[b]["conversation_mean_top20_kl"]
        gb = (arms[a]["gpu_bytes"] - arms[b]["gpu_bytes"]) / 1e9
        kl = _paired(ka, kb)
        return {"kl": kl, "gigabytes": gb, "kl_per_gigabyte": kl["mean_delta"] / gb,
                "nll": _paired(arms[a]["conversation_mean_nll"], arms[b]["conversation_mean_nll"]),
                "per_stratum_kl": {
                    s: _paired([ka[i] for i in m], [kb[i] for i in m])
                    for s in sorted(set(strata)) for m in [np.flatnonzero(strata == s)]}}

    pairs = {"dense_q6_k_minus_q8_0": compare("center", "center_dense_q8_0"),
             "dense_q4_k_minus_q6_k": compare("center_dense_q4_k", "center")}
    _atomic_json(output, {
        "schema": SCHEMA, "status": "pass" if ok else "fail",
        "metric": "per-conversation mean top-20 KL(BF16 || model), not renormalized; "
                  "A minus B, paired bootstrap 95% CI; bytes on the GPU (without token_embd)",
        "identity": {**data_identity, **identity, "part": "validation",
                     "conversation_ids": [c["id"] for c in parts["validation"]["conversations"]]},
        "reproduces_upgrade_rate": {"targets": TARGETS, "max_abs_difference": reproduced,
                                    "tolerance": TOLERANCE, "ok": ok},
        "means": {k: a["mean_top20_kl_to_bf16"] for k, a in arms.items()},
        "pairs": pairs, "arms": arms, "stats": asdict(result.stats),
        "wall_seconds": time.perf_counter() - started})
    for label, arm in arms.items():
        print(f"{label}: KL {arm['mean_top20_kl_to_bf16']:.5f}, GPU {arm['gpu_bytes'] / 1e9:.3f} GB",
              flush=True)
    for name, pair in pairs.items():
        kl = pair["kl"]
        print(f"{name}: {kl['mean_delta']:+.5f} CI [{kl['ci95'][0]:+.5f}, {kl['ci95'][1]:+.5f}] "
              f"{kl['improved_conversations']}/{kl['conversations']}, {pair['gigabytes']:+.3f} GB, "
              f"{pair['kl_per_gigabyte']:+.4f}/GB", flush=True)
    print(f"reproduces the upgrade-rate run: {reproduced}", flush=True)
    return 0 if ok else 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_evaluator_arguments(parser)
    parser.add_argument("--gguf-report", type=Path,
                        default=RCO / "reports/rco_v3/qwen36_rco_v3_center_gguf.json")
    parser.add_argument("--gsq-gguf", type=Path,
                        default=RCO.parents[1] / "results/Qwen3.6-35B-A3B-GSQ-hybrid.gguf")
    args = parser.parse_args()
    for key in (*EVALUATOR_PATHS, "gguf_report", "gsq_gguf"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
