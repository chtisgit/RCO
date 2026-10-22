#!/usr/bin/env python3
"""Q4_K for the parts other than the routed experts (PLAN_POST_RCO_RELEASE.md).

Test A's setting (Phase 7 exact evaluator, validation part of the chat v2
calibration half, GSQ-E6), but without the prune mask: routed experts GSQ
Q2_0, all 256.  Forward variants, one pass over identical batches:

- ``p24_base``: test A's ``base`` (P24 mask); must reproduce it exactly;
- ``base``: unpruned, everything as loaded (attention, shared expert, output
  Q8_0 from the GSQ GGUF);
- ``attention_q4k``, ``shared_expert_q4k``, ``attention_shared_expert_q4k``:
  those groups installed from the Q4_K store
  (``build_qwen36_post_release_q4k_dense_store.py``).

``output.weight`` is outside the decoder blocks, where the evaluator installs
nothing.  It only enters the loss, so each forward variant is scored twice:
with the loaded Q8_0 head and with the Q4_K head (dequantized once to BF16 on
the GPU).  Arms with the Q4_K head carry the suffix ``+output_q4k``; ``all``
is ``attention_shared_expert_q4k+output_q4k``.  Before the pass, the second
scoring is checked on one row with the loaded head: it must equal the
evaluator's own loss exactly.

Reported: every arm minus ``base``, per stratum; bytes per group.
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

from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file  # noqa: E402
from search_qwen36_prune24_q3k import _evaluator, _paired, _parts  # noqa: E402

SCHEMA = "rco.qwen36.post_release_q4k_dense.v1"
OUTPUT = "qwen36_post_release_q4k_dense.json"
TOLERANCE = 1e-9
HEAD_SUFFIX = "+output_q4k"


def _row_losses_with(evaluator, hidden, row, weight):
    """``StreamedChatExactEvaluator._row_losses`` with a given head weight."""
    import torch
    import torch.nn.functional as F

    from search.streamed_chat_exact import top_k_kl

    positions = torch.nonzero(row.scored[1:].bool()).flatten().to(hidden.device)
    targets = row.tokens[1:].to(hidden.device)[positions]
    normed = evaluator.adapter.text_model.norm(hidden[positions])
    kl, nll = [], []
    for start in range(0, positions.numel(), evaluator.position_chunk):
        stop = min(start + evaluator.position_chunk, positions.numel())
        log_probs = torch.log_softmax(F.linear(normed[start:stop], weight).float(), dim=-1)
        nll.append((-log_probs.gather(-1, targets[start:stop, None])).squeeze(-1)
                   .double().cpu().numpy())
        kl.append(top_k_kl(row.reference_values[start:stop].to(hidden.device),
                           row.reference_indices[start:stop].to(hidden.device),
                           log_probs).double().cpu().numpy())
        del log_probs
    return np.concatenate(kl), np.concatenate(nll)


class _SecondHead:
    """Wraps the evaluator's ``_row_losses``: also scores with a second head.

    The evaluator scores variant by variant, each row once per variant, so the
    k-th call for a row belongs to variant k.
    """

    def __init__(self, evaluator, weight):
        self.evaluator, self.weight = evaluator, weight
        self.original = evaluator._row_losses
        self.results: dict[int, list] = {}
        self.checked = False

    def __call__(self, hidden, row):
        kl, nll = self.original(hidden, row)
        if not self.checked:
            own = _row_losses_with(self.evaluator, hidden, row,
                                   self.evaluator.adapter.lm_head().weight)
            if not (np.array_equal(own[0], kl) and np.array_equal(own[1], nll)):
                raise RuntimeError("second-head scoring differs from the evaluator's own loss")
            self.checked = True
        self.results.setdefault(id(row), []).append(
            _row_losses_with(self.evaluator, hidden, row, self.weight))
        return kl, nll


def _head_weight(store, name, device):
    import torch

    from quant.ggml_native import GGMLType

    entry = store.index["tensors"][name]["Q4_K"]
    width, rows = (int(v) for v in entry["gguf_shape"])
    weight = torch.empty((rows, width), dtype=torch.bfloat16, device=device)
    for start, decoded in store.iter_decoded_rows(name, GGMLType.Q4_K, rows_per_chunk=8192):
        weight[start:start + decoded.shape[0]].copy_(torch.from_numpy(decoded))
    return weight


def run(args) -> int:
    from native_store import NativeCandidateStore
    from quant.ggml_native import GGMLNativeCodec, GGMLType
    from search.streamed_chat_exact import ExactVariant
    from search.typed_upgrades import SingleTypeExpertStore

    output = args.reports / OUTPUT
    if output.exists() and _load_json(output).get("status") == "complete":
        print("Q4_K dense: already complete", flush=True)
        return 0
    started = time.perf_counter()
    test_a = _load_json(args.reports / "qwen36_post_release_q2_types.json")
    store_report = _load_json(args.reports / "qwen36_post_release_q4k_dense_store.json")
    if store_report.get("status") != "complete":
        raise RuntimeError("the Q4_K dense store is not complete")
    root = Path(store_report["store"]["root"])
    if _sha256_file(root / "native-candidate-index.json") != store_report["store"]["index_sha256"]:
        raise RuntimeError("Q4_K dense store index differs from its report")
    groups = store_report["groups"]
    parts, data_identity = _parts(args)
    evaluator, mask, _, identity = _evaluator(args)
    codec = GGMLNativeCodec(args.ggml_library)
    native = NativeCandidateStore(root, codec)
    evaluator.upgrade_store = SingleTypeExpertStore(
        native, _load_json(args.manifest), args.model_dir, GGMLType.Q4_K,
        rows_per_chunk=args.install_rows)
    second = _SecondHead(evaluator, _head_weight(native, "output.weight", evaluator.device))
    evaluator._row_losses = second

    attention, shared = frozenset(groups["attention"]), frozenset(groups["shared_expert"])
    variants = [ExactVariant("p24_base", prune_mask=mask),
                ExactVariant("base"),
                ExactVariant("attention_q4k", upgrades=attention),
                ExactVariant("shared_expert_q4k", upgrades=shared),
                ExactVariant("attention_shared_expert_q4k", upgrades=attention | shared)]
    rows = parts["validation"]["rows"]
    result = evaluator.evaluate(rows, variants)
    arms = {}
    for number, variant in enumerate(variants):
        for head in ("", HEAD_SUFFIX):
            if head:
                kl = [second.results[id(row)][number][0] for row in rows]
                nll = [second.results[id(row)][number][1] for row in rows]
            else:
                kl, nll = result.kl[number], result.nll[number]
            per_kl = [float(values.mean()) for values in kl]
            per_nll = [float(values.mean()) for values in nll]
            arms[variant.label + head] = {
                "conversation_mean_top20_kl": per_kl,
                "conversation_mean_nll": per_nll,
                "mean_top20_kl_to_bf16": float(np.mean(per_kl)),
                "mean_nll": float(np.mean(per_nll)),
                "pruned": variant.prune_mask is not None,
                "q4k_groups": [g for g, members in (("attention", attention),
                                                    ("shared_expert", shared))
                               if members <= variant.upgrades] + (["output"] if head else []),
            }
    if any(len(v) != len(variants) for v in second.results.values()) or \
            len(second.results) != len(rows):
        raise RuntimeError("second-head results do not cover every row and variant")

    reproduced = float(np.max(np.abs(
        np.asarray(arms["p24_base"]["conversation_mean_top20_kl"])
        - test_a["arms"]["base"]["conversation_mean_top20_kl"])))
    reproduced_ok = reproduced <= TOLERANCE
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

    pairs = {f"{label}_minus_base": compare(label, "base")
             for label in arms if label not in ("base", "p24_base", "p24_base" + HEAD_SUFFIX)}
    bytes_saved = {g: s["bytes"]["q8_0"] - s["bytes"]["q4_k"]
                   for g, s in store_report["summary"].items()}
    _atomic_json(output, {
        "schema": SCHEMA, "status": "complete",
        "question": "Q4_K instead of Q8_0 for attention/SSM, shared expert and output",
        "metric": "per-conversation mean top-20 KL(BF16 || model), not renormalized; "
                  "A minus B, paired bootstrap 95% CI",
        "identity": {**data_identity, **identity, "part": "validation",
                     "conversation_ids": [c["id"] for c in parts["validation"]["conversations"]],
                     "store_index_sha256": store_report["store"]["index_sha256"],
                     "imatrix_sha256": store_report["imatrix"]["sha256"]},
        "reproduces_test_a_base": {"max_abs_difference": reproduced, "tolerance": TOLERANCE,
                                   "ok": reproduced_ok},
        "bytes": {g: s["bytes"] for g, s in store_report["summary"].items()},
        "bytes_saved": {**bytes_saved, "all": sum(bytes_saved.values())},
        "means": {label: arm["mean_top20_kl_to_bf16"] for label, arm in arms.items()},
        "pairs": pairs, "arms": arms, "stats": _stats(result.stats),
        "wall_seconds": time.perf_counter() - started})
    for label, arm in arms.items():
        print(f"{label}: KL {arm['mean_top20_kl_to_bf16']:.5f}", flush=True)
    for name, pair in pairs.items():
        kl = pair["kl"]
        print(f"{name}: {kl['mean_delta']:+.5f} CI [{kl['ci95'][0]:+.5f}, "
              f"{kl['ci95'][1]:+.5f}] {kl['improved_conversations']}/{kl['conversations']}",
              flush=True)
    print(f"bytes saved: {bytes_saved}", flush=True)
    print(f"reproduces test A base: {reproduced}", flush=True)
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
