#!/usr/bin/env python3
"""Our chat imatrix against Unsloth's for IQ2_XS routed experts (PLAN_POST_RCO_RELEASE.md).

Test A's setting, unchanged: the Phase 7 exact evaluator over the validation
part of the chat v2 calibration half, GSQ-E6 with the frequency mask (P24),
all arms in one pass over identical batches:

- ``base``: routed experts GSQ Q2_0;
- ``iq2_xs_ours``: all 120 from test A's IQ2_XS store (Phase 5 fit imatrix,
  from the fit part of the same chat corpus);
- ``iq2_xs_unsloth``: all 120 from a store built the same way with Unsloth's
  published imatrix (``audit_qwen36_unsloth_imatrix.py convert``), which never
  saw our corpus.

``iq2_xs_ours`` must reproduce test A's ``iq2_xs_all`` and ``base`` test A's
``base`` exactly.  Reported: each arm minus ``base`` and ``iq2_xs_unsloth``
minus ``iq2_xs_ours``, per stratum.
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

from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file, expert_names  # noqa: E402
from search_qwen36_prune24_q3k import _evaluator, _paired, _parts, _run_variants  # noqa: E402

SCHEMA = "rco.qwen36.post_release_imatrix_source.v1"
OUTPUT = "qwen36_post_release_imatrix_source.json"
TOLERANCE = 1e-9


def _store(report_path: Path) -> dict:
    report = _load_json(report_path)
    if report.get("status") != "complete" or report.get("type") != "IQ2_XS":
        raise RuntimeError(f"{report_path} is not a complete IQ2_XS store")
    root = Path(report["store"]["root"])
    if _sha256_file(root / "native-candidate-index.json") != report["store"]["index_sha256"]:
        raise RuntimeError(f"{report_path}: store index differs from its report")
    return report


def run(args) -> int:
    from native_store import NativeCandidateStore
    from quant.ggml_native_iq import IQ_TYPE_IDS, IQGGMLNativeCodec
    from search.streamed_chat_exact import ExactVariant
    from search.typed_upgrades import SingleTypeExpertStore, TypedUpgradeStore, typed_key

    output = args.reports / OUTPUT
    if output.exists() and _load_json(output).get("status") == "complete":
        print("imatrix source: already complete", flush=True)
        return 0
    started = time.perf_counter()
    test_a = _load_json(args.reports / "qwen36_post_release_q2_types.json")
    stores = {"OURS": _store(args.reports / "qwen36_post_release_iq_store_IQ2_XS.json"),
              "UNSLOTH": _store(args.unsloth_store_report)}
    parts, data_identity = _parts(args)
    evaluator, mask, _, identity = _evaluator(args)
    codec = IQGGMLNativeCodec(args.ggml_library)
    manifest = _load_json(args.manifest)
    evaluator.upgrade_store = TypedUpgradeStore({
        label: SingleTypeExpertStore(
            NativeCandidateStore(Path(report["store"]["root"]), codec), manifest,
            args.model_dir, IQ_TYPE_IDS["IQ2_XS"], rows_per_chunk=args.install_rows)
        for label, report in stores.items()})
    names = expert_names()
    variants = [ExactVariant("base", prune_mask=mask, upgrades=frozenset()),
                ExactVariant("iq2_xs_ours", prune_mask=mask,
                             upgrades=frozenset(typed_key("OURS", n) for n in names)),
                ExactVariant("iq2_xs_unsloth", prune_mask=mask,
                             upgrades=frozenset(typed_key("UNSLOTH", n) for n in names))]
    arms, stats = _run_variants(evaluator, parts["validation"]["rows"], variants)

    reproduced = {
        label: float(np.max(np.abs(np.asarray(arms[label]["conversation_mean_top20_kl"])
                                   - test_a["arms"][source]["conversation_mean_top20_kl"])))
        for label, source in (("base", "base"), ("iq2_xs_ours", "iq2_xs_all"))}
    reproduced_ok = all(value <= TOLERANCE for value in reproduced.values())
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

    pairs = {"iq2_xs_ours_minus_base": compare("iq2_xs_ours", "base"),
             "iq2_xs_unsloth_minus_base": compare("iq2_xs_unsloth", "base"),
             "iq2_xs_unsloth_minus_iq2_xs_ours": compare("iq2_xs_unsloth", "iq2_xs_ours")}
    _atomic_json(output, {
        "schema": SCHEMA, "status": "complete",
        "question": "IQ2_XS routed experts: our chat imatrix against Unsloth's published imatrix",
        "metric": "per-conversation mean top-20 KL(BF16 || model), not renormalized; "
                  "A minus B, paired bootstrap 95% CI",
        "identity": {**data_identity, **identity, "part": "validation",
                     "conversation_ids": [c["id"] for c in parts["validation"]["conversations"]],
                     "store_index_sha256": {k: r["store"]["index_sha256"]
                                            for k, r in stores.items()},
                     "imatrix_sha256": {k: r["imatrix_sha256"] for k, r in stores.items()}},
        "reproduces_test_a": {"max_abs_difference": reproduced, "tolerance": TOLERANCE,
                              "ok": reproduced_ok},
        "store_summaries": {k: r["summary"]["all"] for k, r in stores.items()},
        "means": {label: arm["mean_top20_kl_to_bf16"] for label, arm in arms.items()},
        "pairs": pairs, "arms": arms, "stats": stats,
        "wall_seconds": time.perf_counter() - started})
    for label, arm in arms.items():
        print(f"{label}: KL {arm['mean_top20_kl_to_bf16']:.5f}", flush=True)
    for name, pair in pairs.items():
        kl = pair["kl"]
        print(f"{name}: {kl['mean_delta']:+.5f} CI [{kl['ci95'][0]:+.5f}, "
              f"{kl['ci95'][1]:+.5f}] {kl['improved_conversations']}/{kl['conversations']}",
              flush=True)
    print(f"reproduces test A: {reproduced}", flush=True)
    return 0 if reproduced_ok else 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unsloth-store-report", type=Path,
                        default=RCO / "reports/post_release_unsloth_imatrix/"
                                      "qwen36_post_release_iq_store_IQ2_XS.json")
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
    for key in ("unsloth_store_report", "corpus", "split_report", "model_dir", "manifest",
                "gguf", "gguf_python", "ggml_library", "reports"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
