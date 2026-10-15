#!/usr/bin/env python3
"""Phase 5 of RCO_PLAN_NEW.md: fit/validation split and the per-expert imatrix.

Subcommands, run in this order:

``split``
    Splits the chat corpus v2 calibration half into a fit part and a
    validation part (safeguard 1 of Phase 4b, "Outcome and decision"):
    validation quotas per stratum, thinking on and off as evenly as each
    stratum allows (largest remainder), random within each group with seed
    20261016.  Writes ``reports/qwen36_chat_corpus_v2_calibration_split.json``.
``imatrix``
    Streamed BF16 forward over the fit conversations with the frequency
    mask's exact pruned routing (pruned router logits -inf, softmax, top-8,
    renormalize).  Per layer and expert: the per-column sum of squared
    expert inputs over the real tokens routed to that expert (gate/up: the
    routed hidden state; down: act(gate) * up; routing weights not applied,
    as in Phase 0 and llama.cpp's ``imatrix`` for ``MUL_MAT_ID``), plus token
    counts.  Padding is excluded.  The running sums are saved after every
    batch, so an interrupted run resumes at the next batch.
"""

from __future__ import annotations

import argparse
import json
import re
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

from audit_qwen36_chat_kl import (  # noqa: E402
    _batch_tensors,
    _model,
    _valid_tokens,
    batches,
    load_corpus,
)
from audit_qwen36_prune24_prelim import (  # noqa: E402
    EXPERTS,
    LAYERS,
    PRUNE_PER_LAYER,
    RouterHooks,
    _save_npz,
)
from audit_qwen36_q3k_viability import (  # noqa: E402
    _atomic_json,
    _load_json,
    _sha256_file,
    expert_importance,
)

SCHEMA = "rco.qwen36.chat_imatrix.v1"
SEED = 20261016
VALIDATION_QUOTAS = {"chat": 18, "tool-call": 7, "tool-answer": 7, "reasoning": 13,
                     "document": 18}
TOP_K = 8


# -- split -------------------------------------------------------------------


def _largest_remainder(total: int, sizes: list[int]) -> list[int]:
    """Split ``total`` over groups in proportion to ``sizes``; ties go to the
    earlier group."""
    shares = [total * size / sum(sizes) for size in sizes]
    counts = [int(share) for share in shares]
    order = sorted(range(len(sizes)), key=lambda i: (-(shares[i] - counts[i]), i))
    for i in order[:total - sum(counts)]:
        counts[i] += 1
    return counts


def choose_validation(conversations: list[dict[str, Any]], quotas: dict[str, int],
                      seed: int) -> list[str]:
    """Validation ids: per stratum, thinking on before off, sorted ids, one
    generator drawn in that fixed order."""
    rng = np.random.default_rng(seed)
    if set(quotas) != {c["stratum"] for c in conversations}:
        raise ValueError("quotas must name every stratum exactly")
    chosen = []
    for stratum in sorted(quotas):
        groups = [sorted(c["id"] for c in conversations
                         if c["stratum"] == stratum and c["thinking"] == thinking)
                  for thinking in (True, False)]
        groups = [group for group in groups if group]
        for group, count in zip(groups, _largest_remainder(
                quotas[stratum], [len(group) for group in groups])):
            chosen += [str(i) for i in rng.choice(group, size=count, replace=False)]
    return sorted(chosen)


def _part_summary(conversations: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, int]] = {}
    for c in conversations:
        key = f"{c['stratum']}/{'thinking' if c['thinking'] else 'no-thinking'}"
        group = groups.setdefault(key, {"conversations": 0, "scored_targets": 0})
        group["conversations"] += 1
        group["scored_targets"] += c["scored_target_count"]
    return {"conversations": len(conversations),
            "tokens": sum(c["token_count"] for c in conversations),
            "scored_targets": sum(c["scored_target_count"] for c in conversations),
            "groups": dict(sorted(groups.items()))}


def run_split(args) -> None:
    output = args.reports / "qwen36_chat_corpus_v2_calibration_split.json"
    conversations, identity = load_corpus(
        SimpleNamespace(corpus=args.corpus, split="calibration", limit=None))
    validation = choose_validation(conversations, VALIDATION_QUOTAS, SEED)
    chosen = set(validation)
    fit = sorted(c["id"] for c in conversations if c["id"] not in chosen)
    result = {
        "schema": SCHEMA + ".split",
        "status": "complete",
        "corpus_manifest_sha256": identity["corpus_manifest_sha256"],
        "rule": ("validation quotas per stratum; within a stratum thinking on and off "
                 "by largest remainder (ties to thinking on); random within each group "
                 "(numpy default_rng(seed), strata sorted, thinking on before off, ids "
                 "sorted)"),
        "seed": SEED,
        "validation_quotas": VALIDATION_QUOTAS,
        "fit": fit,
        "validation": validation,
        "summary": {
            "fit": _part_summary([c for c in conversations if c["id"] not in chosen]),
            "validation": _part_summary([c for c in conversations if c["id"] in chosen]),
        },
    }
    if args.check_only:
        print(json.dumps(result["summary"], indent=1))
        return
    if output.exists():
        previous = _load_json(output)
        if (previous["fit"], previous["validation"]) != (fit, validation):
            raise RuntimeError(f"{output} exists with a different split")
        print(f"{output.name}: unchanged", flush=True)
        return
    _atomic_json(output, result)
    print(f"{output.name}: fit {len(fit)}, validation {len(validation)}", flush=True)


def load_part(args, part: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The calibration conversations of one part of the split, in corpus order."""
    split = _load_json(args.split_report)
    conversations, identity = load_corpus(
        SimpleNamespace(corpus=args.corpus, split="calibration", limit=None))
    if split["corpus_manifest_sha256"] != identity["corpus_manifest_sha256"]:
        raise RuntimeError("the split was made for a different corpus")
    wanted = set(split[part])
    selected = [c for c in conversations if c["id"] in wanted]
    if len(selected) != len(wanted):
        raise RuntimeError(f"{part}: conversations missing from the corpus")
    return selected, {**identity, "part": part, "split_sha256": _sha256_file(args.split_report),
                      "conversation_count": len(selected),
                      "conversation_ids": [c["id"] for c in selected]}


# -- imatrix -----------------------------------------------------------------


def accumulate_expert_inputs(hidden, top_k_index, gate_up_proj, act_fn,
                             gate_up_sum, down_sum, counts) -> None:
    """Add one layer's routed expert inputs (real tokens only) to the sums.

    ``hidden`` is [tokens, hidden_size], ``top_k_index`` [tokens, top_k],
    ``gate_up_proj`` [experts, 2 * intermediate, hidden_size].
    """
    import torch
    import torch.nn.functional as F

    squared = hidden.double().square()
    for slot in range(top_k_index.shape[1]):
        gate_up_sum.index_add_(0, top_k_index[:, slot], squared)
    counts += torch.bincount(top_k_index.reshape(-1), minlength=counts.shape[0])
    for expert in torch.unique(top_k_index).tolist():
        tokens = (top_k_index == expert).any(dim=-1).nonzero().squeeze(-1)
        gate, up = F.linear(hidden[tokens], gate_up_proj[expert]).chunk(2, dim=-1)
        down_sum[expert] += (act_fn(gate) * up).double().square().sum(dim=0)


class ExpertInputHooks:
    """Pre-hooks on every routed-expert module; counts real tokens only."""

    def __init__(self, model, device) -> None:
        import torch

        self.valid = None
        self.gate_up, self.down, self.counts = {}, {}, {}
        self.handles = []
        for name, module in model.named_modules():
            if type(module).__name__ != "Qwen3_5MoeExperts":
                continue
            layer = int(re.search(r"layers\.(\d+)\.", name).group(1))
            self.handles.append(module.register_forward_pre_hook(
                self._hook(layer), with_kwargs=True))
        if len(self.handles) != LAYERS:
            raise RuntimeError(f"found {len(self.handles)} expert modules, expected {LAYERS}")
        config = model.config.get_text_config()
        for layer in range(LAYERS):
            self.gate_up[layer] = torch.zeros(EXPERTS, config.hidden_size,
                                              dtype=torch.float64, device=device)
            self.down[layer] = torch.zeros(EXPERTS, config.moe_intermediate_size,
                                           dtype=torch.float64, device=device)
            self.counts[layer] = torch.zeros(EXPERTS, dtype=torch.int64, device=device)

    def _hook(self, layer: int):
        def hook(module, hook_args, hook_kwargs):
            values = list(hook_args) + [hook_kwargs[key] for key in
                                        ("hidden_states", "top_k_index", "top_k_weights")
                                        if key in hook_kwargs]
            hidden, top_k_index = values[0], values[1]
            if self.valid is None or self.valid.shape[0] != hidden.shape[0]:
                raise RuntimeError("valid-token mask does not match the routed tokens")
            valid = self.valid.to(hidden.device)
            accumulate_expert_inputs(
                hidden[valid], top_k_index[valid], module.gate_up_proj, module.act_fn,
                self.gate_up[layer], self.down[layer], self.counts[layer])
        return hook

    def arrays(self) -> dict[str, np.ndarray]:
        result = {}
        for layer in range(LAYERS):
            result[f"blk{layer}_gate_up_sum"] = self.gate_up[layer].cpu().numpy()
            result[f"blk{layer}_down_sum"] = self.down[layer].cpu().numpy()
            result[f"blk{layer}_count"] = self.counts[layer].cpu().numpy()
        return result

    def load(self, arrays) -> None:
        import torch

        for layer in range(LAYERS):
            for store, key in ((self.gate_up, "gate_up_sum"), (self.down, "down_sum"),
                               (self.counts, "count")):
                store[layer].copy_(torch.from_numpy(arrays[f"blk{layer}_{key}"]))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _load_mask(path: Path) -> np.ndarray:
    mask = np.load(path)
    if mask.shape != (LAYERS, EXPERTS) or mask.dtype != bool or not bool(
            (mask.sum(axis=1) == PRUNE_PER_LAYER).all()):
        raise ValueError(f"mask must be 40 x 256 bool with {PRUNE_PER_LAYER} per layer")
    return mask


def run_imatrix(args) -> None:
    import torch

    from checkpoint_stream import SafeTensorPrefixLoader
    from search.streaming import StreamingHardCausalEvaluator

    output = args.reports / "qwen36_prune24_imatrix.json"
    matrix_path = args.work / "imatrix.npz"
    partial_path = args.work / "imatrix_partial.npz"
    if output.exists() and _load_json(output).get("status") == "complete":
        if _sha256_file(matrix_path) != _load_json(output)["imatrix"]["sha256"]:
            raise RuntimeError("existing imatrix file differs from its report")
        print("imatrix: already complete", flush=True)
        return
    started = time.perf_counter()
    conversations, identity = load_part(args, "fit")
    if args.limit:
        conversations = conversations[:args.limit]
        identity = {**identity, "limit": args.limit,
                    "conversation_ids": [c["id"] for c in conversations]}
    mask = _load_mask(args.mask)
    run_identity = {"corpus": identity,
                    "mask": {"path": str(args.mask), "sha256": _sha256_file(args.mask)},
                    "model_dir": str(args.model_dir), "batch_tokens": args.batch_tokens,
                    "attn_implementation": args.attn_implementation}
    device = torch.device(args.device)
    model = _model(args)
    evaluator = StreamingHardCausalEvaluator(
        model, SafeTensorPrefixLoader(args.model_dir), SimpleNamespace(), [], [0, 1],
        device=device, vocab_chunk_size=args.vocab_chunk_size)
    routing = RouterHooks(model, prune_mask=mask, collect=False)
    inputs = ExpertInputHooks(model, device)
    plan = batches(conversations, args.batch_tokens)
    done, nll = 0, {}
    if partial_path.exists():
        with np.load(partial_path, allow_pickle=False) as data:
            if json.loads(str(data["identity"])) != run_identity:
                raise RuntimeError(f"{partial_path} was made with a different identity")
            done = int(data["batches_done"])
            nll = json.loads(str(data["nll"]))
            inputs.load(data)
        print(f"imatrix: resuming after batch {done - 1}", flush=True)
    try:
        for number, members in enumerate(plan):
            if number < done:
                continue
            input_ids, loss_mask = _batch_tensors(conversations, members)
            inputs.valid = _valid_tokens(conversations, members)
            batch_started = time.perf_counter()
            evaluation = evaluator.evaluate(input_ids, torch.zeros(0, dtype=torch.long),
                                            loss_mask=loss_mask)
            for row, index in enumerate(members):
                nll[conversations[index]["id"]] = evaluation.document_mean_nll[row]
            _save_npz(partial_path, **inputs.arrays(), batches_done=np.asarray(number + 1),
                      identity=np.asarray(json.dumps(run_identity, sort_keys=True)),
                      nll=np.asarray(json.dumps(nll)))
            print(f"imatrix: batch {number + 1}/{len(plan)} ({len(members)} x "
                  f"{input_ids.shape[1]}) in {time.perf_counter() - batch_started:.0f}s",
                  flush=True)
    finally:
        inputs.close()
        routing.close()
    arrays = inputs.arrays()
    kept = ~mask
    real_tokens = sum(c["token_count"] for c in conversations)
    fallback, kept_counts = {}, []
    for layer in range(LAYERS):
        layer_counts = arrays[f"blk{layer}_count"]
        if int(layer_counts.sum()) != real_tokens * TOP_K:
            raise RuntimeError(f"layer {layer}: routed-token count differs")
        if int(layer_counts[mask[layer]].sum()) != 0:
            raise RuntimeError(f"layer {layer}: a pruned expert received tokens")
        # The fallback is computed over the kept experts only.
        _, fallback[str(layer)] = expert_importance(
            arrays[f"blk{layer}_gate_up_sum"][kept[layer]], layer_counts[kept[layer]])
        kept_counts.append(layer_counts[kept[layer]])
    _save_npz(matrix_path, **arrays)
    kept_counts = np.stack(kept_counts)
    _atomic_json(output, {
        "schema": SCHEMA + ".imatrix",
        "status": "complete",
        "source": ("streamed BF16 forward over the fit part of the chat v2 calibration half, "
                   "routing restricted to the frequency mask's kept experts"),
        **run_identity,
        "imatrix": {
            "path": str(matrix_path),
            "sha256": _sha256_file(matrix_path),
            "convention": (
                "per expert and input column: sum of squared matmul inputs over the real "
                "(unpadded) tokens routed to that expert; gate/up input is the routed "
                "hidden state, down input is act(gate)*up; routing weights are not "
                "applied; pruned experts are all zero"),
            "real_tokens": real_tokens,
            "routed_tokens_per_layer": real_tokens * TOP_K,
            "kept_experts_without_tokens_per_layer": fallback,
            "min_tokens_per_kept_expert": int(kept_counts.min()),
            "median_tokens_per_kept_expert": float(np.median(kept_counts)),
        },
        "pruned_bf16_fit": {
            "mean_nll": float(np.mean(list(nll.values()))),
            "conversation_mean_nll": [nll[c["id"]] for c in conversations],
        },
        "wall_seconds_this_run": time.perf_counter() - started,
    })
    partial_path.unlink()
    print(f"imatrix: complete, {sum(fallback.values())} kept experts without tokens",
          flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("split", "imatrix"))
    parser.add_argument("--corpus", type=Path,
                        default=RCO / "reports/qwen36_chat_corpus_v2_manifest.json")
    parser.add_argument("--split-report", type=Path,
                        default=RCO / "reports/qwen36_chat_corpus_v2_calibration_split.json")
    parser.add_argument("--check-only", action="store_true",
                        help="split: print the summary without writing the report")
    parser.add_argument("--mask", type=Path,
                        default=ROOT / "data/qwen36_prune24_chat/frequency_mask.npy")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/qwen36_35b_base")
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_prune24_imatrix")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--limit", type=int, help="imatrix: first N fit conversations (smoke)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--batch-tokens", type=int, default=12_800)
    parser.add_argument("--attn-implementation", default="sdpa", choices=("eager", "sdpa"))
    args = parser.parse_args()
    args.mask = args.mask.resolve()
    return args


def main() -> int:
    args = _parse_args()
    {"split": run_split, "imatrix": run_imatrix}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
