#!/usr/bin/env python3
"""Test A of PLAN_POST_RCO_RELEASE.md: GSQ Q2_0 against imatrix I-quants.

One pass of the Phase 7 exact evaluator (``search.streamed_chat_exact``)
over the validation part of the chat v2 calibration half, all arms on
identical batches, every arm on GSQ-E6 with the frequency mask (P24):

- ``base``: routed experts GSQ Q2_0;
- ``iq2_xs_gate_up``, ``iq2_xs_down``, ``iq2_xs_all``: those tensors from the
  IQ2_XS store (``build_qwen36_post_release_iq_store.py``);
- ``iq2_xxs_all``: all 120 from the IQ2_XXS store;
- ``release``: the release's 23 Q3_K upgrades (Phase 7 promoted set, the
  Phase 6 store), the other 97 GSQ;
- ``release_iq2_xs_rest``: the 23 Q3_K, the other 97 IQ2_XS.

Reports every arm minus ``base`` and ``release_iq2_xs_rest`` minus
``release`` (per-conversation mean top-20 KL, paired bootstrap 95% CI), per
stratum, and kept-expert bytes per arm.  ``base`` and ``release`` must
reproduce the Phase 7 extended promotion's arms exactly (the replay did).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file, expert_names  # noqa: E402
from search_qwen36_prune24_q3k import _evaluator, _paired, _parts, _run_variants  # noqa: E402

SCHEMA = "rco.qwen36.post_release_q2_types.v1"
OUTPUT = "qwen36_post_release_q2_types.json"
TOLERANCE = 1e-9
IQ_STORES = ("IQ2_XS", "IQ2_XXS")


def arm_assignments(release: list[str]) -> dict[str, dict[str, str]]:
    """Arm label -> {tensor: type}; tensors not listed keep GSQ Q2_0."""
    names = expert_names()
    gate_up = [n for n in names if "ffn_down_exps" not in n]
    down = [n for n in names if "ffn_down_exps" in n]
    return {
        "base": {},
        "iq2_xs_gate_up": {n: "IQ2_XS" for n in gate_up},
        "iq2_xs_down": {n: "IQ2_XS" for n in down},
        "iq2_xs_all": {n: "IQ2_XS" for n in names},
        "iq2_xxs_all": {n: "IQ2_XXS" for n in names},
        "release": {n: "Q3_K" for n in release},
        "release_iq2_xs_rest": {n: ("Q3_K" if n in release else "IQ2_XS") for n in names},
    }


def kept_bytes(assignment: dict[str, str], per_type: dict[str, int]) -> int:
    return sum(per_type[assignment.get(name, "Q2_0")] for name in expert_names())


def run(args) -> int:
    from native_store import NativeCandidateStore
    from quant.ggml_native_iq import IQ_TYPE_IDS, IQGGMLNativeCodec
    from search.streamed_chat_exact import ExactVariant
    from search.typed_upgrades import (
        SingleTypeExpertStore, TypedUpgradeStore, check_variant_keys, typed_key)

    output = args.reports / OUTPUT
    if output.exists() and _load_json(output).get("status") == "complete":
        print("test A: already complete", flush=True)
        return 0
    started = time.perf_counter()
    promotion = _load_json(args.reports / "qwen36_prune24_q3k_promotion_extended.json")
    candidate = _load_json(args.reports / "qwen36_gsq_rco_v2_candidate_gguf.json")
    release = sorted(promotion["promoted_tensors"])
    if sorted(candidate["upgraded_tensors"]) != release:
        raise RuntimeError("the release's upgrades differ from the promoted set")
    stores = {}
    for type_name in IQ_STORES:
        report = _load_json(args.reports / f"qwen36_post_release_iq_store_{type_name}.json")
        if report.get("status") != "complete":
            raise RuntimeError(f"the {type_name} store is not complete")
        root = Path(report["store"]["root"])
        if _sha256_file(root / "native-candidate-index.json") != report["store"]["index_sha256"]:
            raise RuntimeError(f"the {type_name} store index differs from its report")
        stores[type_name] = report

    parts, data_identity = _parts(args)
    evaluator, mask, q3k_report, identity = _evaluator(args)
    codec = IQGGMLNativeCodec(args.ggml_library)
    manifest = _load_json(args.manifest)
    typed = {"Q3_K": evaluator.upgrade_store}
    for type_name, report in stores.items():
        typed[type_name] = SingleTypeExpertStore(
            NativeCandidateStore(Path(report["store"]["root"]), codec), manifest,
            args.model_dir, IQ_TYPE_IDS[type_name], rows_per_chunk=args.install_rows)
    evaluator.upgrade_store = TypedUpgradeStore(typed)

    assignments = arm_assignments(release)
    variants = []
    for label, assignment in assignments.items():
        keys = frozenset(typed_key(t, n) for n, t in assignment.items())
        check_variant_keys(keys)
        variants.append(ExactVariant(label, prune_mask=mask, upgrades=keys))
    arms, stats = _run_variants(evaluator, parts["validation"]["rows"], variants)

    reproduced = {
        label: float(np.max(np.abs(np.asarray(arms[label]["conversation_mean_top20_kl"])
                                   - promotion["arms"][source]["conversation_mean_top20_kl"])))
        for label, source in (("base", "base"), ("release", promotion["promoted"]))}
    reproduced_ok = all(value <= TOLERANCE for value in reproduced.values())
    per_type = {"Q2_0": q3k_report["tensors"][release[0]]["gsq_q2_0_kept_bytes"],
                "Q3_K": q3k_report["summary"]["q3_k_kept_bytes_per_tensor"],
                **{t: r["kept_bytes_per_tensor"][t] for t, r in stores.items()}}
    strata = np.asarray([c["stratum"] for c in parts["validation"]["conversations"]])

    def compare(a: str, b: str) -> dict:
        ka, kb = arms[a]["conversation_mean_top20_kl"], arms[b]["conversation_mean_top20_kl"]
        return {
            "kl": _paired(ka, kb),
            "nll": _paired(arms[a]["conversation_mean_nll"], arms[b]["conversation_mean_nll"]),
            "per_stratum_kl": {
                stratum: _paired([ka[i] for i in members], [kb[i] for i in members])
                for stratum in sorted(set(strata))
                for members in [np.flatnonzero(strata == stratum)]},
        }

    pairs = {f"{label}_minus_base": compare(label, "base") for label in arms if label != "base"}
    pairs["release_iq2_xs_rest_minus_release"] = compare("release_iq2_xs_rest", "release")
    for label in arms:
        arms[label]["kept_expert_bytes"] = kept_bytes(assignments[label], per_type)
        arms[label]["types"] = {t: sum(v == t for v in assignments[label].values())
                                for t in sorted(set(assignments[label].values()))}
    _atomic_json(output, {
        "schema": SCHEMA, "status": "complete",
        "question": "GSQ Q2_0 against imatrix IQ2_XS / IQ2_XXS on the routed experts",
        "metric": "per-conversation mean top-20 KL(BF16 || model), not renormalized; "
                  "A minus B, paired bootstrap 95% CI",
        "identity": {**data_identity, **identity, "part": "validation",
                     "conversation_ids": [c["id"] for c in parts["validation"]["conversations"]],
                     "iq_store_index_sha256": {t: r["store"]["index_sha256"]
                                               for t, r in stores.items()}},
        "release_tensors": release,
        "kept_bytes_per_tensor": per_type,
        "reproduces_phase7_promotion": {"max_abs_difference": reproduced,
                                        "tolerance": TOLERANCE, "ok": reproduced_ok},
        "means": {label: arm["mean_top20_kl_to_bf16"] for label, arm in arms.items()},
        "pairs": pairs,
        "arms": arms, "stats": stats,
        "wall_seconds": time.perf_counter() - started})
    for label, arm in arms.items():
        print(f"{label}: KL {arm['mean_top20_kl_to_bf16']:.5f} "
              f"bytes {arm['kept_expert_bytes'] / 1e9:.3f} GB", flush=True)
    for name, pair in pairs.items():
        kl = pair["kl"]
        print(f"{name}: {kl['mean_delta']:+.5f} CI [{kl['ci95'][0]:+.5f}, "
              f"{kl['ci95'][1]:+.5f}] {kl['improved_conversations']}/{kl['conversations']}",
              flush=True)
    print(f"reproduces Phase 7: {reproduced}", flush=True)
    return 0 if reproduced_ok else 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path,
                        default=RCO / "reports/qwen36_chat_corpus_v2_manifest.json")
    parser.add_argument("--split-report", type=Path,
                        default=RCO / "reports/qwen36_chat_corpus_v2_calibration_split.json")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/qwen36_35b_base")
    parser.add_argument("--manifest", type=Path,
                        default=RCO / "reports/qwen36_35b_base_gguf_manifest.json")
    parser.add_argument("--gguf", type=Path,
                        default=ROOT / "results/Qwen3.6-35B-A3B-GSQ-hybrid.gguf")
    parser.add_argument("--gguf-python", type=Path, default=ROOT / "repos/llama.cpp/gguf-py")
    parser.add_argument("--ggml-library", type=Path,
                        default=ROOT / "experiment/build-cpu/bin/libggml-base.so.0.24.0")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows-per-chunk", type=int, default=1024)
    parser.add_argument("--install-rows", type=int, default=4096)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--position-chunk", type=int, default=256)
    parser.add_argument("--microbatch-tokens", type=int, default=8448)
    parser.add_argument("--device-hidden-bytes", type=int, default=3 << 30)
    args = parser.parse_args()
    for key in ("corpus", "split_report", "model_dir", "manifest", "gguf", "gguf_python",
                "ggml_library", "reports"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
