#!/usr/bin/env python3
"""What expert upgrades buy on an unpruned IQ2_XS base (PLAN_POST_RCO_RELEASE.md).

The Q4_K result (``audit_qwen36_post_release_q4k_dense.py``) prices Q4_K for
attention/SSM, shared expert and output at about 0.0098 KL per GB saved.
Whether that trade pays depends on what the freed bytes buy in the routed
experts.  Same setting (Phase 7 exact evaluator, validation part, GSQ-E6,
unpruned); routed experts from ``build_qwen36_post_release_iq_store_full.py``
(Unsloth's imatrix, all 256 experts).  Forward variants:

- ``base``: GSQ Q2_0, everything else as loaded; must reproduce the Q4_K
  run's ``base`` exactly;
- ``iq2_xs``: all 120 routed-expert tensors IQ2_XS;
- ``iq2_xs_down_iq3_xxs``: gate/up IQ2_XS, down IQ3_XXS (Unsloth's layout
  for most of its down tensors);
- ``iq2_xs+dense_q4k`` and ``iq2_xs_down_iq3_xxs+dense_q4k``: plus
  attention/SSM and shared expert from the Q4_K store.

Each is scored with the loaded Q8_0 output head and with the Q4_K one
(``+output_q4k``), as in the Q4_K run.  Reported: arm bytes against
``base``, every arm minus ``base`` and minus ``iq2_xs``, and KL per GB for
the IQ3_XXS ``down`` upgrade and for the Q4_K trade on the IQ2_XS base.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_post_release_q4k_dense import HEAD_SUFFIX, _head_weight, _SecondHead  # noqa: E402
from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file  # noqa: E402
from search_qwen36_prune24_q3k import _evaluator, _paired, _parts  # noqa: E402

SCHEMA = "rco.qwen36.post_release_upgrade_rate.v1"
OUTPUT = "qwen36_post_release_upgrade_rate.json"
TOLERANCE = 1e-9


def _checked(report_path: Path, kind: str) -> dict:
    report = _load_json(report_path)
    if report.get("status") != "complete":
        raise RuntimeError(f"{report_path} is not complete")
    root = Path(report["store"]["root"])
    if _sha256_file(root / "native-candidate-index.json") != report["store"]["index_sha256"]:
        raise RuntimeError(f"{report_path}: store index differs from its report ({kind})")
    return report


def run(args) -> int:
    from native_store import NativeCandidateStore
    from quant.ggml_native import GGMLType
    from quant.ggml_native_iq import IQ_TYPE_IDS, IQGGMLNativeCodec
    from search.streamed_chat_exact import ExactVariant
    from search.typed_upgrades import SingleTypeExpertStore, TypedUpgradeStore, typed_key

    output = args.reports / OUTPUT
    if output.exists() and _load_json(output).get("status") == "complete":
        print("upgrade rate: already complete", flush=True)
        return 0
    started = time.perf_counter()
    q4k_run = _load_json(args.reports / "qwen36_post_release_q4k_dense.json")
    reports = {
        "IQ2_XS": _checked(args.reports / "qwen36_post_release_iq_store_full_IQ2_XS.json", "IQ2_XS"),
        "IQ3_XXS": _checked(args.reports / "qwen36_post_release_iq_store_full_IQ3_XXS.json",
                            "IQ3_XXS"),
        "Q4_K": _checked(args.reports / "qwen36_post_release_q4k_dense_store.json", "Q4_K"),
    }
    if reports["IQ3_XXS"]["families"] != ["ffn_down_exps"]:
        raise RuntimeError("the IQ3_XXS store must hold exactly the down tensors")
    parts, data_identity = _parts(args)
    evaluator, _, _, identity = _evaluator(args)
    codec = IQGGMLNativeCodec(args.ggml_library)
    manifest = _load_json(args.manifest)
    stores = {label: NativeCandidateStore(Path(r["store"]["root"]), codec)
              for label, r in reports.items()}
    type_ids = {"IQ2_XS": IQ_TYPE_IDS["IQ2_XS"], "IQ3_XXS": IQ_TYPE_IDS["IQ3_XXS"],
                "Q4_K": int(GGMLType.Q4_K)}
    evaluator.upgrade_store = TypedUpgradeStore({
        label: SingleTypeExpertStore(store, manifest, args.model_dir, type_ids[label],
                                     rows_per_chunk=args.install_rows)
        for label, store in stores.items()})
    second = _SecondHead(evaluator, _head_weight(stores["Q4_K"], "output.weight",
                                                 evaluator.device))
    evaluator._row_losses = second

    experts = list(reports["IQ2_XS"]["tensors"])
    down = [n for n in experts if "ffn_down_exps" in n]
    gate_up = [n for n in experts if "ffn_down_exps" not in n]
    dense = (reports["Q4_K"]["groups"]["attention"]
             + reports["Q4_K"]["groups"]["shared_expert"])
    if sorted(down) != sorted(reports["IQ3_XXS"]["tensors"]):
        raise RuntimeError("IQ3_XXS store and IQ2_XS down tensors differ")
    iq2 = frozenset(typed_key("IQ2_XS", n) for n in experts)
    iq3_down = (frozenset(typed_key("IQ2_XS", n) for n in gate_up)
                | frozenset(typed_key("IQ3_XXS", n) for n in down))
    dense_q4k = frozenset(typed_key("Q4_K", n) for n in dense)
    variants = [ExactVariant("base"),
                ExactVariant("iq2_xs", upgrades=iq2),
                ExactVariant("iq2_xs_down_iq3_xxs", upgrades=iq3_down),
                ExactVariant("iq2_xs+dense_q4k", upgrades=iq2 | dense_q4k),
                ExactVariant("iq2_xs_down_iq3_xxs+dense_q4k", upgrades=iq3_down | dense_q4k)]

    def tensor_bytes(label, name):
        return reports[label]["tensors"][name]["bytes"][label]

    gsq_bytes = {n: reports["IQ2_XS"]["tensors"][n]["bytes"]["gsq_q2_0"] for n in experts}
    q8_bytes = {n: reports["Q4_K"]["tensors"][n]["bytes"]["q8_0"]
                for n in dense + ["output.weight"]}

    def delta_bytes(variant, head: bool) -> int:
        total = 0
        for key in variant.upgrades:
            label, name = key.split(":", 1)
            if label == "Q4_K":
                total += reports["Q4_K"]["tensors"][name]["bytes"]["q4_k"] - q8_bytes[name]
            else:
                total += tensor_bytes(label, name) - gsq_bytes[name]
        if head:
            total += (reports["Q4_K"]["tensors"]["output.weight"]["bytes"]["q4_k"]
                      - q8_bytes["output.weight"])
        return total

    rows = parts["validation"]["rows"]
    result = evaluator.evaluate(rows, variants)
    if any(len(v) != len(variants) for v in second.results.values()) or \
            len(second.results) != len(rows):
        raise RuntimeError("second-head results do not cover every row and variant")
    arms = {}
    for number, variant in enumerate(variants):
        for head in (False, True):
            if head:
                kl = [second.results[id(row)][number][0] for row in rows]
                nll = [second.results[id(row)][number][1] for row in rows]
            else:
                kl, nll = result.kl[number], result.nll[number]
            per_kl = [float(values.mean()) for values in kl]
            per_nll = [float(values.mean()) for values in nll]
            arms[variant.label + (HEAD_SUFFIX if head else "")] = {
                "conversation_mean_top20_kl": per_kl,
                "conversation_mean_nll": per_nll,
                "mean_top20_kl_to_bf16": float(np.mean(per_kl)),
                "mean_nll": float(np.mean(per_nll)),
                "bytes_minus_base": delta_bytes(variant, head),
            }

    reproduced = {
        label: float(np.max(np.abs(np.asarray(arms[label]["conversation_mean_top20_kl"])
                                   - q4k_run["arms"][label]["conversation_mean_top20_kl"])))
        for label in ("base", "base" + HEAD_SUFFIX)}
    reproduced_ok = all(value <= TOLERANCE for value in reproduced.values())
    strata = np.asarray([c["stratum"] for c in parts["validation"]["conversations"]])

    def compare(a: str, b: str) -> dict:
        ka, kb = arms[a]["conversation_mean_top20_kl"], arms[b]["conversation_mean_top20_kl"]
        gb = (arms[a]["bytes_minus_base"] - arms[b]["bytes_minus_base"]) / 1e9
        kl = _paired(ka, kb)
        return {
            "kl": kl,
            "gigabytes": gb,
            "kl_per_gigabyte": kl["mean_delta"] / gb if gb else None,
            "nll": _paired(arms[a]["conversation_mean_nll"], arms[b]["conversation_mean_nll"]),
            "per_stratum_kl": {
                stratum: _paired([ka[i] for i in members], [kb[i] for i in members])
                for stratum in sorted(set(strata))
                for members in [np.flatnonzero(strata == stratum)]},
        }

    pairs = {f"{label}_minus_base": compare(label, "base")
             for label in arms if label != "base"}
    pairs.update({f"{label}_minus_iq2_xs": compare(label, "iq2_xs")
                  for label in arms if label not in ("base", "base" + HEAD_SUFFIX, "iq2_xs")})
    _atomic_json(output, {
        "schema": SCHEMA, "status": "complete",
        "question": "KL per GB of an IQ3_XXS down upgrade and of Q4_K for the other parts, "
                    "on an unpruned IQ2_XS base",
        "metric": "per-conversation mean top-20 KL(BF16 || model), not renormalized; "
                  "A minus B, paired bootstrap 95% CI; bytes from the stores",
        "identity": {**data_identity, **identity, "part": "validation",
                     "conversation_ids": [c["id"] for c in parts["validation"]["conversations"]],
                     "store_index_sha256": {k: r["store"]["index_sha256"]
                                            for k, r in reports.items()}},
        "reproduces_q4k_run_base": {"max_abs_difference": reproduced, "tolerance": TOLERANCE,
                                    "ok": reproduced_ok},
        "means": {label: arm["mean_top20_kl_to_bf16"] for label, arm in arms.items()},
        "bytes_minus_base": {label: arm["bytes_minus_base"] for label, arm in arms.items()},
        "pairs": pairs, "arms": arms, "stats": _stats(result.stats),
        "wall_seconds": time.perf_counter() - started})
    for label, arm in arms.items():
        print(f"{label}: KL {arm['mean_top20_kl_to_bf16']:.5f}, "
              f"bytes {arm['bytes_minus_base'] / 1e9:+.3f} GB", flush=True)
    for name, pair in pairs.items():
        kl = pair["kl"]
        rate = "" if pair["kl_per_gigabyte"] is None else f", {pair['kl_per_gigabyte']:+.4f}/GB"
        print(f"{name}: {kl['mean_delta']:+.5f} CI [{kl['ci95'][0]:+.5f}, "
              f"{kl['ci95'][1]:+.5f}] {kl['improved_conversations']}/{kl['conversations']}, "
              f"{pair['gigabytes']:+.3f} GB{rate}", flush=True)
    print(f"reproduces Q4_K run base: {reproduced}", flush=True)
    return 0 if reproduced_ok else 4


def _stats(stats) -> dict:
    from dataclasses import asdict

    return asdict(stats)


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
