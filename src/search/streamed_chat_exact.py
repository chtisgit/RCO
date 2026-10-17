"""Exact, block-major chat evaluation of several model variants in one pass.

``audit_qwen36_chat_kl.py score`` runs ``StreamingHardCausalEvaluator``,
which streams the whole model once per batch: about 470 s per 12,800
padded tokens, so one exact score of the 160 fit conversations takes about
3 h.  This evaluator reuses the block streaming of
``StreamedChatPruneObjective`` (each block loaded once, every micro-batch
run through it) without gradients and with exact routing, and evaluates
several variants in the same pass.

A variant is an optional prune mask (exact routing as in
``audit_qwen36_chat_kl.py``: pruned router logits -inf, softmax, top-8,
renormalize) plus a set of routed-expert tensors upgraded from the loaded
GSQ tensor to a candidate-store payload.  Per block, the variants are
grouped by their upgrades in that block.  The block is loaded from GSQ, and
the groups run in order of growing upgrade sets: a group whose set contains
the previous one only installs the extra tensors; otherwise the block is
released and loaded again.

A variant's embedding is either the configured one (``"store"``: the
GSQ-E6 Q6_K one, through ``embedding_store``) or the checkpoint's own
(``"checkpoint"``: authentic GSQ's BF16 ``token_embd``); each embedding kind
runs once per pass.  Hidden states stay on
the GPU while they fit ``device_hidden_bytes`` and otherwise wait in host
memory between blocks.

The loss is ``audit_qwen36_prune24_prelim.top_k_kl``: KL(BF16 || model)
over the reference's top-20 tokens, not renormalized, at the scored
positions only, which is the metric of the chat KL scores and the Phase 8
gate.  Per row the evaluator returns the per-position KL and NLL.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from search.streamed_prune import _sync
from search.streamed_prune_chat import (
    ChatRow,
    StreamedChatPruneObjective,
    _Pass,
    microbatches,
)


def exact_pruned_routing(router_logits, prune_mask, top_k: int):
    """llama.cpp-equivalent routing with pruned experts removed.

    Identical to ``audit_qwen36_prune24_prelim.exact_pruned_routing``
    (checked in ``tests/test_streamed_chat_exact.py``).
    """
    masked = router_logits.masked_fill(prune_mask, float("-inf"))
    probabilities = torch.softmax(masked, dim=-1, dtype=torch.float)
    weights, indices = torch.topk(probabilities, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return masked, weights.to(router_logits.dtype), indices


def top_k_kl(reference_values, reference_indices, model_log_probs):
    """KL(reference || model) restricted to the reference top-k tokens.

    Identical to ``audit_qwen36_prune24_prelim.top_k_kl``.
    """
    gathered = model_log_probs.gather(-1, reference_indices.long())
    return (reference_values.exp() * (reference_values - gathered)).sum(dim=-1)


@dataclass(frozen=True)
class ExactVariant:
    label: str
    prune_mask: Optional[np.ndarray] = None   # [layers, experts] bool
    upgrades: frozenset = field(default_factory=frozenset)
    embedding: str = "store"   # "store" (embedding_store) or "checkpoint"


@dataclass
class ExactStats:
    rows: int = 0
    tokens: int = 0
    padded_tokens: int = 0
    scored_targets: int = 0
    microbatch_count: int = 0
    variants: int = 0
    block_loads: int = 0
    installs: int = 0
    load_seconds: float = 0.0
    install_seconds: float = 0.0
    forward_seconds: float = 0.0
    loss_seconds: float = 0.0
    total_seconds: float = 0.0
    hidden_on_device: bool = True
    cuda_max_allocated: int = 0
    cuda_max_reserved: int = 0


@dataclass
class ExactResult:
    kl: list[list[np.ndarray]]    # [variant][row] per scored position
    nll: list[list[np.ndarray]]
    stats: ExactStats

    def row_mean_kl(self, variant: int) -> np.ndarray:
        return np.asarray([values.mean() for values in self.kl[variant]])

    def row_mean_nll(self, variant: int) -> np.ndarray:
        return np.asarray([values.mean() for values in self.nll[variant]])


class StreamedChatExactEvaluator(StreamedChatPruneObjective):
    """Forward-only exact evaluation of variants over chat rows."""

    def __init__(self, model, checkpoint_loader, *, upgrade_store: Any = None,
                 device_hidden_bytes: int = 3 << 30, **kwargs) -> None:
        super().__init__(model, checkpoint_loader, **kwargs)
        self.upgrade_store = upgrade_store
        self.device_hidden_bytes = int(device_hidden_bytes)

    def _exact_block(self, state: _Pass, layer_index: int, number: int,
                     hidden: torch.Tensor, prune_row: Optional[torch.Tensor]) -> torch.Tensor:
        path = f"{self.adapter.layers_path}.{layer_index}"

        def hook(module, inputs, output):
            router_logits, weights, indices = output
            masked, weights, indices = exact_pruned_routing(
                router_logits, prune_row, module.top_k)
            if bool(prune_row[indices].any()):
                raise RuntimeError(f"pruned expert selected in layer {layer_index}")
            return masked, weights, indices

        layer_type = self.adapter.text_model.config.layer_types[layer_index]
        text_position_ids, position_embeddings, mask = self._context(
            state, number, hidden, layer_type)
        handle = None
        if prune_row is not None:
            handle = self.model.get_submodule(
                f"{path}.{self.router_suffix}").register_forward_hook(hook)
        try:
            output = self.adapter.layers[layer_index](
                hidden, position_embeddings=position_embeddings, attention_mask=mask,
                position_ids=text_position_ids, past_key_values=None, use_cache=False)
        finally:
            if handle is not None:
                handle.remove()
        return output[0] if isinstance(output, tuple) else output

    def _upgrades_in(self, variant: ExactVariant, layer_index: int) -> frozenset:
        return frozenset(name for name in variant.upgrades
                         if self.upgrade_store.block_index(name) == layer_index)

    def _row_losses(self, hidden: torch.Tensor, row: ChatRow) -> tuple[np.ndarray, np.ndarray]:
        positions = torch.nonzero(row.scored[1:].bool()).flatten().to(hidden.device)
        targets = row.tokens[1:].to(hidden.device)[positions]
        normed = self.adapter.text_model.norm(hidden[positions])
        weight = self.adapter.lm_head().weight
        kl, nll = [], []
        for start in range(0, positions.numel(), self.position_chunk):
            stop = min(start + self.position_chunk, positions.numel())
            log_probs = torch.log_softmax(
                F.linear(normed[start:stop], weight).float(), dim=-1)
            nll.append((-log_probs.gather(-1, targets[start:stop, None])).squeeze(-1)
                       .double().cpu().numpy())
            kl.append(top_k_kl(row.reference_values[start:stop].to(hidden.device),
                               row.reference_indices[start:stop].to(hidden.device),
                               log_probs).double().cpu().numpy())
            del log_probs
        return np.concatenate(kl), np.concatenate(nll)

    @torch.no_grad()
    def evaluate(self, rows: Sequence[ChatRow], variants: Sequence[ExactVariant]) -> ExactResult:
        rows = list(rows)
        variants = list(variants)
        if not rows or not variants:
            raise ValueError("need at least one row and one variant")
        if any(v.upgrades for v in variants) and self.upgrade_store is None:
            raise ValueError("upgrades need an upgrade store")
        if any(v.embedding not in ("store", "checkpoint") for v in variants):
            raise ValueError("a variant's embedding is 'store' or 'checkpoint'")
        stats = ExactStats(variants=len(variants))
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        lengths = [int(row.tokens.numel()) for row in rows]
        groups = microbatches(lengths, self.microbatch_tokens)
        input_batches, shapes = [], []
        for members in groups:
            width = lengths[members[0]]
            ids = torch.full((len(members), width), self.pad_token, dtype=torch.long)
            for slot, index in enumerate(members):
                ids[slot, :lengths[index]] = rows[index].tokens
            input_batches.append(ids)
            shapes.append((len(members), width))
        stats.rows, stats.tokens = len(rows), sum(lengths)
        stats.padded_tokens = sum(r * w for r, w in shapes)
        stats.scored_targets = sum(int(row.scored.sum()) for row in rows)
        stats.microbatch_count = len(groups)
        hidden_size = self.adapter.text_model.config.hidden_size
        stats.hidden_on_device = (
            stats.padded_tokens * hidden_size * 2 * len(variants) <= self.device_hidden_bytes)
        home = self.device if stats.hidden_on_device else torch.device("cpu")
        prune_rows = [
            None if v.prune_mask is None
            else torch.from_numpy(np.asarray(v.prune_mask, dtype=bool)).to(self.device)
            for v in variants]

        state = _Pass(self, stats, shapes, None)
        loaded: list[str] = []
        try:
            self.loader.move_runtime_buffers(self.model, self.device)
            embedded = {}
            for kind in sorted({v.embedding for v in variants}):
                selected = self.embedding_selected
                if kind == "checkpoint":
                    self.embedding_selected = ()
                try:
                    embedded[kind] = [item.to(home) for item in self._embed(input_batches, stats)]
                finally:
                    self.embedding_selected = selected
            hidden = [list(embedded[v.embedding]) for v in variants]
            del embedded
            for path in self.adapter.final_module_paths:
                dtype = self.checkpoint_dtype if path == "lm_head" else None
                self._load(path, dtype, stats, loaded)

            for layer_index in range(self.layer_count):
                block = f"{self.adapter.layers_path}.{layer_index}"
                by_set: dict[frozenset, list[int]] = {}
                for number, variant in enumerate(variants):
                    key = (self._upgrades_in(variant, layer_index)
                           if variant.upgrades else frozenset())
                    by_set.setdefault(key, []).append(number)
                installed: Optional[frozenset] = None
                try:
                    for key in sorted(by_set, key=lambda s: (len(s), sorted(s))):
                        if installed is None or not installed <= key:
                            if installed is not None:
                                self.loader.release_prefix(self.model, block)
                                installed = None
                            _sync(self.device)
                            load_started = time.perf_counter()
                            self.loader.load_prefix(self.model, block, device=self.device,
                                                    dtype=self.checkpoint_dtype)
                            _sync(self.device)
                            stats.load_seconds += time.perf_counter() - load_started
                            stats.block_loads += 1
                            installed = frozenset()
                        install_started = time.perf_counter()
                        for name in sorted(key - installed):
                            self.upgrade_store.install_layer_weight(self.model, name, 1)
                            stats.installs += 1
                        _sync(self.device)
                        stats.install_seconds += time.perf_counter() - install_started
                        installed = key
                        forward_started = time.perf_counter()
                        for number in by_set[key]:
                            prune_row = (None if prune_rows[number] is None
                                         else prune_rows[number][layer_index])
                            hidden[number] = [
                                self._exact_block(state, layer_index, micro,
                                                  item.to(self.device), prune_row).to(home)
                                for micro, item in enumerate(hidden[number])]
                        _sync(self.device)
                        stats.forward_seconds += time.perf_counter() - forward_started
                finally:
                    if installed is not None:
                        self.loader.release_prefix(self.model, block)

            loss_started = time.perf_counter()
            kl: list[list[Any]] = [[None] * len(rows) for _ in variants]
            nll: list[list[Any]] = [[None] * len(rows) for _ in variants]
            for number in range(len(variants)):
                for members, batch in zip(groups, hidden[number]):
                    batch = batch.to(self.device)
                    for slot, index in enumerate(members):
                        kl[number][index], nll[number][index] = self._row_losses(
                            batch[slot, :lengths[index]], rows[index])
                    del batch
                hidden[number] = None
            _sync(self.device)
            stats.loss_seconds = time.perf_counter() - loss_started
            return ExactResult(kl, nll, stats)
        finally:
            state.masks.clear()
            state.rotary.clear()
            for path in reversed(loaded):
                self.loader.release_prefix(self.model, path)
            gc.collect()
            if self.device.type == "cuda":
                stats.cuda_max_allocated = torch.cuda.max_memory_allocated(self.device)
                stats.cuda_max_reserved = torch.cuda.max_memory_reserved(self.device)
                torch.cuda.empty_cache()
            stats.total_seconds = time.perf_counter() - started


__all__ = [
    "ExactResult",
    "ExactStats",
    "ExactVariant",
    "StreamedChatExactEvaluator",
    "exact_pruned_routing",
    "top_k_kl",
]
