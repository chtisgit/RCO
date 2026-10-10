#!/usr/bin/env python3
"""Phase 4 chat check 2 of RCO_PLAN_NEW.md: chat KL against BF16.

Scores a chat corpus with the streaming evaluator and the Phase 3
machinery, over the scored (assistant) tokens only.  The corpus is chat
corpus v1 (``build_qwen36_chat_corpus.py``) by default, or chat corpus v2
(``build_qwen36_chat_corpus_v2.py``) via ``--corpus``.  For v2 this tool
also gives Phase 4b its BF16 reference, router statistics and exact mask
scores on the calibration half, and Phase 8 its held-out scores.  Use
``--attn-implementation sdpa`` for v2: eager attention runs out of memory on
its 8k-token conversations.

Subcommands, run in this order and each resumable per batch:

``reference``
    Streamed BF16 forward over the chosen split; stores the top-20
    log-probabilities and token ids at every scored position.
``score --label L [--mask M] [--router-stats]``
    GSQ-E6 forward, unpruned or with exact pruned routing for mask ``M``
    (pruned router logits -inf, softmax, top-8, renormalize).  Per
    conversation: mean top-20 KL to BF16 and mean NLL over the scored
    tokens, and the same restricted to template tokens (added tokens such as
    ``<|im_end|>``, ``</think>``, ``<tool_call>``).  ``--router-stats``
    (unpruned only) also accumulates router statistics over every real
    token, padding excluded.
``compare [--base-label B --candidate-label C]``
    Candidate minus base per conversation, paired bootstrap 95% CI; by
    default P24 minus unpruned.  Pass (the Phase 4 chat check 2 rule): the
    CI upper bound of the mean KL difference is below +0.005.  Exit status 3
    on failure.  Other label pairs get their own report name.

Conversations are sorted by length and batched up to a padded-token budget.
Each batch is right-padded to its longest conversation.  The model is
causal, so padding cannot change any scored position; padded positions are
never scored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_prune24_prelim import (  # noqa: E402
    EXPERTS,
    LAYERS,
    POSITION_CHUNK,
    PRUNE_PER_LAYER,
    TOP_K_REFERENCE,
    RouterHooks,
    _gsq_e6_evaluator,
    _save_npz,
    top_k_kl,
)
from audit_qwen36_q3k_viability import (  # noqa: E402
    _atomic_json,
    _load_json,
    _sha256_file,
)
from release_corpus import canonical_json_bytes, sha256_bytes  # noqa: E402

SCHEMA = "rco.qwen36.chat_kl.v1"
PAD_TOKEN = 248044  # <|endoftext|>, never scored
KL_MARGIN = 0.005


class MaskedTap:
    """Wrap the evaluator's loss; read log-probs only at scored positions."""

    def __init__(self) -> None:
        import search.streaming as streaming

        self.streaming = streaming
        self.original = streaming._chunked_causal_cross_entropy_details
        self.mode: str | None = None
        self.reference: list[tuple[np.ndarray, np.ndarray]] | None = None
        self.result: list[dict[str, np.ndarray]] = []
        streaming._chunked_causal_cross_entropy_details = self._wrapped

    def close(self) -> None:
        self.streaming._chunked_causal_cross_entropy_details = self.original

    def _wrapped(self, hidden_states, labels, lm_head, **kwargs):
        import torch

        loss_mask = kwargs.get("loss_mask")
        if loss_mask is None or self.mode not in ("reference", "compare"):
            raise RuntimeError("masked tap needs a loss mask and a mode")
        self.result = []
        for row in range(hidden_states.shape[0]):
            # Hidden state at p predicts token p + 1.
            positions = torch.nonzero(loss_mask[row, 1:].bool()).flatten().to(
                hidden_states.device)
            targets = labels[row, 1:].to(hidden_states.device)[positions]
            out = {"values": [], "indices": [], "kl": [], "nll": []}
            for start in range(0, positions.numel(), POSITION_CHUNK):
                chunk = positions[start:start + POSITION_CHUNK]
                log_probs = torch.log_softmax(
                    lm_head(hidden_states[row, chunk]).float(), dim=-1)
                nll = -log_probs.gather(-1, targets[start:start + POSITION_CHUNK, None])
                out["nll"].append(nll.squeeze(-1).double().cpu().numpy())
                if self.mode == "reference":
                    values, indices = log_probs.topk(TOP_K_REFERENCE, dim=-1)
                    out["values"].append(values.cpu().numpy())
                    out["indices"].append(indices.int().cpu().numpy())
                else:
                    ref_values, ref_indices = self.reference[row]
                    stop = start + chunk.numel()
                    out["kl"].append(top_k_kl(
                        torch.from_numpy(ref_values[start:stop]).to(log_probs.device),
                        torch.from_numpy(ref_indices[start:stop]).to(log_probs.device),
                        log_probs).double().cpu().numpy())
                del log_probs
            self.result.append({key: np.concatenate(value) for key, value in out.items()
                                if value})
        return self.original(hidden_states, labels, lm_head, **kwargs)


def _model(args):
    """Empty model skeleton.  Chat corpus v2 needs SDPA: eager attention
    materializes the full attention matrix and runs out of memory on
    8k-token conversations."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(
            config, attn_implementation=args.attn_implementation)
    model.eval()
    return model


def _valid_tokens(conversations, members):
    """Flattened bool mask of the real (unpadded) tokens of a batch."""
    import torch

    width = max(conversations[i]["token_count"] for i in members)
    valid = torch.zeros((len(members), width), dtype=torch.bool)
    for row, index in enumerate(members):
        valid[row, :conversations[index]["token_count"]] = True
    return valid.reshape(-1)


def load_corpus(args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = _load_json(args.corpus)
    canonical = report["canonical_manifest"]
    if sha256_bytes(canonical_json_bytes(canonical)) != report["canonical_manifest_sha256"]:
        raise RuntimeError("chat corpus canonical manifest hash differs")
    tokens_path = Path(canonical["tokens"]["path"])
    if _sha256_file(tokens_path) != canonical["tokens"]["sha256"]:
        raise RuntimeError("chat corpus token file hash differs")
    rows = {row["id"]: row for row in _load_json(tokens_path)}
    conversations = []
    for meta in canonical["conversations"]:
        if meta["split"] != args.split:
            continue
        row = rows[meta["id"]]
        if (len(row["tokens"]) != meta["token_count"]
                or sum(row["scored"][1:]) != meta["scored_target_count"]):
            raise RuntimeError(f"{meta['id']}: tokens differ from the manifest")
        conversations.append({**meta, "tokens": row["tokens"], "scored": row["scored"]})
    if args.limit:
        conversations = conversations[:args.limit]
    identity = {"corpus_manifest_sha256": report["canonical_manifest_sha256"],
                "split": args.split, "conversation_count": len(conversations),
                "conversation_ids": [c["id"] for c in conversations]}
    return conversations, identity


def batches(conversations: list[dict], budget: int) -> list[list[int]]:
    """Longest first; each batch's padded size stays within the token budget."""
    order = sorted(range(len(conversations)),
                   key=lambda i: (-conversations[i]["token_count"], conversations[i]["id"]))
    result, current = [], []
    for index in order:
        width = conversations[current[0]]["token_count"] if current else \
            conversations[index]["token_count"]
        if current and (len(current) + 1) * width > budget:
            result.append(current)
            current = []
        current.append(index)
    if current:
        result.append(current)
    return result


def _batch_tensors(conversations, members):
    import torch

    width = max(conversations[i]["token_count"] for i in members)
    input_ids = torch.full((len(members), width), PAD_TOKEN, dtype=torch.long)
    loss_mask = torch.zeros((len(members), width), dtype=torch.long)
    for row, index in enumerate(members):
        tokens = conversations[index]["tokens"]
        input_ids[row, :len(tokens)] = torch.tensor(tokens)
        loss_mask[row, :len(tokens)] = torch.tensor(conversations[index]["scored"])
    loss_mask[:, 0] = 0
    return input_ids, loss_mask


def _run(args, evaluator, assignment, conversations, tap, label, before_batch):
    """Run every batch once; return per-conversation arrays in corpus order."""
    results: list[dict[str, np.ndarray] | None] = [None] * len(conversations)
    for number, members in enumerate(batches(conversations, args.batch_tokens)):
        path = args.work / label / f"batch{number:03d}.npz"
        if path.exists():
            with np.load(path) as data:
                if list(data["members"]) != members:
                    raise RuntimeError(f"{path}: batch membership changed")
                for row, index in enumerate(members):
                    results[index] = {key[len(f"{row}_"):]: data[key] for key in data.files
                                      if key.startswith(f"{row}_")}
            continue
        before_batch(members)
        input_ids, loss_mask = _batch_tensors(conversations, members)
        started = time.perf_counter()
        evaluation = evaluator.evaluate(input_ids, assignment, loss_mask=loss_mask)
        arrays, differences = {"members": np.asarray(members)}, []
        for row, index in enumerate(members):
            item = dict(tap.result[row])
            if item["nll"].shape[0] != conversations[index]["scored_target_count"]:
                raise RuntimeError(f"{conversations[index]['id']}: scored count differs")
            # The reported NLL is the evaluator's own masked NLL, as in Phase 3.
            # The tap's NLL, from BF16 lm_head logits like the KL, only guards
            # against misaligned positions, which would be off by nats.
            difference = abs(float(item["nll"].mean()) - evaluation.document_mean_nll[row])
            differences.append(difference)
            if difference > 0.05:
                raise RuntimeError(f"{conversations[index]['id']}: NLL cross-check failed")
            item["eval_nll"] = np.asarray([evaluation.document_mean_nll[row]])
            results[index] = item
            arrays.update({f"{row}_{key}": value for key, value in item.items()})
        _save_npz(path, **arrays)
        print(f"{label}: batch {number} ({len(members)} x {input_ids.shape[1]}) NLL "
              f"{float(np.mean(evaluation.document_mean_nll)):.5f} (tap NLL within "
              f"{max(differences):.4f}) in "
              f"{time.perf_counter() - started:.0f}s", flush=True)
    return results


def _reference_path(args) -> Path:
    return args.reports / f"qwen36_chat_kl_reference{args.suffix}.json"


def run_reference(args) -> None:
    import torch

    from checkpoint_stream import SafeTensorPrefixLoader
    from search.streaming import StreamingHardCausalEvaluator

    output = _reference_path(args)
    if output.exists() and _load_json(output).get("status") == "complete":
        print("reference: already complete", flush=True)
        return
    started = time.perf_counter()
    conversations, identity = load_corpus(args)
    model = _model(args)
    evaluator = StreamingHardCausalEvaluator(
        model, SafeTensorPrefixLoader(args.model_dir), SimpleNamespace(), [], [0, 1],
        device=torch.device(args.device), vocab_chunk_size=args.vocab_chunk_size)
    tap = MaskedTap()
    tap.mode = "reference"
    try:
        results = _run(args, evaluator, torch.zeros(0, dtype=torch.long), conversations,
                       tap, "reference", lambda members: None)
    finally:
        tap.close()
    path = args.work / "reference_bf16_top20.npz"
    _save_npz(path, values=np.concatenate([r["values"] for r in results]),
              indices=np.concatenate([r["indices"] for r in results]),
              counts=np.asarray([r["values"].shape[0] for r in results]))
    nll = [float(r["eval_nll"][0]) for r in results]
    _atomic_json(output, {
        "schema": SCHEMA + ".reference",
        "status": "complete",
        "teacher": "pinned BF16 safetensors, streamed",
        "attn_implementation": args.attn_implementation,
        "corpus": identity,
        "reference": {"path": str(path), "sha256": _sha256_file(path),
                      "top_k": TOP_K_REFERENCE},
        "bf16_mean_nll": float(np.mean(nll)),
        "conversation_mean_nll": nll,
        "batch_tokens": args.batch_tokens,
        "wall_seconds": time.perf_counter() - started,
    })
    print(f"reference: complete, BF16 NLL {np.mean(nll):.5f}", flush=True)


def _load_reference(args, identity):
    report = _load_json(_reference_path(args))
    if report.get("status") != "complete" or report["corpus"] != identity:
        raise RuntimeError("chat BF16 reference is incomplete or for another corpus")
    path = Path(report["reference"]["path"])
    if _sha256_file(path) != report["reference"]["sha256"]:
        raise RuntimeError("chat BF16 reference file differs from its report")
    with np.load(path) as data:
        offsets = np.concatenate([[0], np.cumsum(data["counts"])])
        values, indices = data["values"], data["indices"]
    return [(values[a:b], indices[a:b]) for a, b in zip(offsets[:-1], offsets[1:])], report


def run_score(args) -> None:
    from transformers import AutoTokenizer

    output = args.reports / f"qwen36_chat_kl_score_{args.label}{args.suffix}.json"
    if output.exists() and _load_json(output).get("status") == "complete":
        print(f"score {args.label}: already complete", flush=True)
        return
    started = time.perf_counter()
    conversations, identity = load_corpus(args)
    reference, reference_report = _load_reference(args, identity)
    if reference_report.get("attn_implementation", "eager") != args.attn_implementation:
        raise RuntimeError("score and reference must use the same attention implementation")
    mask, mask_identity = None, None
    if args.router_stats:
        if args.mask is not None:
            raise ValueError("router statistics are collected on the unpruned model only")
        if any((args.work / f"score_{args.label}").glob("batch*.npz")):
            raise RuntimeError(
                f"router statistics need one uninterrupted pass; remove "
                f"{args.work / f'score_{args.label}'} and rerun")
    if args.mask is not None:
        mask = np.load(args.mask)
        if mask.shape != (LAYERS, EXPERTS) or mask.dtype != bool or not bool(
                (mask.sum(axis=1) == PRUNE_PER_LAYER).all()):
            raise ValueError(f"mask must be 40 x 256 bool with {PRUNE_PER_LAYER} per layer")
        mask_identity = {"path": str(args.mask), "sha256": _sha256_file(args.mask)}
    model = _model(args)
    evaluator, assignment, model_identity = _gsq_e6_evaluator(args, model)
    hooks = RouterHooks(model, prune_mask=mask, collect=args.router_stats)
    tap = MaskedTap()
    tap.mode = "compare"

    def before_batch(members):
        tap.reference = [reference[index] for index in members]
        hooks.valid = _valid_tokens(conversations, members)

    try:
        results = _run(args, evaluator, assignment, conversations, tap,
                       f"score_{args.label}", before_batch)
    finally:
        tap.close()
        hooks.close()
    router_stats = None
    if args.router_stats:
        path = args.work / f"router_stats_{args.label}.npz"
        _save_npz(path, counts=hooks.counts, weight_sum=hooks.weight_sum,
                  probability_sum=hooks.probability_sum)
        router_stats = {
            "path": str(path), "sha256": _sha256_file(path),
            "tokens": "every real token of every conversation (context and replies), "
                      "padding excluded",
            "routed_slots_per_layer": int(hooks.counts[0].sum()),
            "experts_never_selected": int((hooks.counts == 0).sum()),
            "min_count": int(hooks.counts.min()),
            "median_count": float(np.median(hooks.counts)),
            "max_count": int(hooks.counts.max()),
        }
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    template_ids = np.asarray(sorted(tokenizer.added_tokens_decoder))
    template_kl, template_counts = [], []
    for conversation, result in zip(conversations, results):
        tokens = np.asarray(conversation["tokens"])
        targets = tokens[1:][np.asarray(conversation["scored"][1:], dtype=bool)]
        is_template = np.isin(targets, template_ids)
        template_counts.append(int(is_template.sum()))
        template_kl.append(float(result["kl"][is_template].mean()))
    _atomic_json(output, {
        "schema": SCHEMA + ".score",
        "status": "complete",
        "label": args.label,
        "semantics": ("unpruned" if mask is None else
                      "exact: pruned router logits -inf, softmax, top-8, renormalize"),
        "mask": mask_identity,
        "identity": {**model_identity, "corpus": identity,
                     "reference_sha256": reference_report["reference"]["sha256"],
                     "attn_implementation": args.attn_implementation},
        "router_stats": router_stats,
        "mean_top20_kl_to_bf16": float(np.mean([r["kl"].mean() for r in results])),
        "mean_nll": float(np.mean([r["eval_nll"][0] for r in results])),
        "conversation_mean_top20_kl": [float(r["kl"].mean()) for r in results],
        "conversation_mean_nll": [float(r["eval_nll"][0]) for r in results],
        "max_tap_nll_difference": float(max(abs(r["nll"].mean() - r["eval_nll"][0])
                                            for r in results)),
        "conversation_template_kl": template_kl,
        "conversation_template_tokens": template_counts,
        "strata": [c["stratum"] for c in conversations],
        "batch_tokens": args.batch_tokens,
        "wall_seconds": time.perf_counter() - started,
    })
    print(f"score {args.label}: KL {np.mean([r['kl'].mean() for r in results]):.5f} "
          f"NLL {np.mean([r['eval_nll'][0] for r in results]):.5f}", flush=True)


def run_compare(args) -> int:
    from release_quality import paired_bootstrap_mean_ci

    reports = {label: _load_json(args.reports / f"qwen36_chat_kl_score_{label}{args.suffix}.json")
               for label in (args.base_label, args.candidate_label)}
    base, candidate = reports[args.base_label], reports[args.candidate_label]
    if any(r.get("status") != "complete" for r in reports.values()):
        raise RuntimeError("a score report is incomplete")
    if (base["identity"]["corpus"] != candidate["identity"]["corpus"]
            or base["identity"]["reference_sha256"] != candidate["identity"]["reference_sha256"]):
        raise RuntimeError("score reports use different corpora or references")

    def delta(key, rows=None):
        values = (np.asarray(candidate[key]) - np.asarray(base[key]))
        values = values if rows is None else values[rows]
        lower, upper = paired_bootstrap_mean_ci(values.tolist())
        return {"mean": float(values.mean()), "ci95": [lower, upper]}

    strata = np.asarray(base["strata"])
    kl = delta("conversation_mean_top20_kl")
    status = "pass" if kl["ci95"][1] < KL_MARGIN else "fail"
    report = {
        "schema": SCHEMA + ".comparison",
        "status": status,
        "rule": (f"pass if the 95% CI upper bound of mean per-conversation top-20 KL "
                 f"({args.candidate_label} minus {args.base_label}) is below +{KL_MARGIN}"),
        "conversations": len(strata),
        "mean_top20_kl": {label: r["mean_top20_kl_to_bf16"] for label, r in reports.items()},
        "mean_nll": {label: r["mean_nll"] for label, r in reports.items()},
        "delta_top20_kl": kl,
        "for_information": {
            "delta_nll": delta("conversation_mean_nll"),
            "delta_template_token_kl": delta("conversation_template_kl"),
            "per_stratum_delta_top20_kl": {
                stratum: delta("conversation_mean_top20_kl", strata == stratum)
                for stratum in dict.fromkeys(base["strata"])},
        },
    }
    pair = ("" if (args.base_label, args.candidate_label) == ("unpruned", "p24")
            else f"_{args.candidate_label}_vs_{args.base_label}")
    _atomic_json(args.reports / f"qwen36_chat_kl_comparison{pair}{args.suffix}.json", report)
    print(f"chat KL {args.candidate_label} - {args.base_label}: {kl['mean']:+.5f} "
          f"[{kl['ci95'][0]:+.5f}, {kl['ci95'][1]:+.5f}] -> {status}", flush=True)
    return 0 if status == "pass" else 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("reference", "score", "compare"))
    parser.add_argument("--label")
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--base-label", default="unpruned")
    parser.add_argument("--candidate-label", default="p24")
    parser.add_argument("--split", default="heldout", choices=("calibration", "heldout"))
    parser.add_argument("--limit", type=int, help="first N conversations only (smoke tests)")
    parser.add_argument("--suffix", default="", help="report name suffix (smoke tests)")
    parser.add_argument("--corpus", type=Path,
                        default=RCO / "reports/qwen36_chat_corpus_v1_manifest.json")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/qwen36_35b_base")
    parser.add_argument("--manifest", type=Path,
                        default=RCO / "reports/qwen36_35b_base_gguf_manifest.json")
    parser.add_argument("--gguf", type=Path,
                        default=ROOT / "results/Qwen3.6-35B-A3B-GSQ-hybrid.gguf")
    parser.add_argument("--gguf-python", type=Path,
                        default=ROOT / "repos/llama.cpp/gguf-py")
    parser.add_argument("--ggml-library", type=Path,
                        default=ROOT / "experiment/build-cpu/bin/libggml-base.so.0.24.0")
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_chat_kl")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows-per-chunk", type=int, default=1024)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--batch-tokens", type=int, default=12_800)
    parser.add_argument("--attn-implementation", default="eager", choices=("eager", "sdpa"))
    parser.add_argument("--router-stats", action="store_true",
                        help="score: also collect router statistics (unpruned only)")
    args = parser.parse_args()
    for key in ("corpus", "model_dir", "manifest", "gguf", "gguf_python",
                "ggml_library", "work", "reports"):
        setattr(args, key, getattr(args, key).resolve())
    if args.mask is not None:
        args.mask = args.mask.resolve(strict=True)
    if args.command == "score" and not args.label:
        parser.error("score needs --label")
    return args


def main() -> int:
    args = _parse_args()
    if args.command == "compare":
        return run_compare(args)
    {"reference": run_reference, "score": run_score}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
