"""Exact evaluation of variants that share the center below their first change.

``StreamedChatExactEvaluator`` runs every variant through every block.  In
the v3 screening (RCO_PLAN_V3.md step 2) each variant differs from the
center in one tensor, so below that tensor's block its hidden states equal
the center's.  This evaluator runs one center trunk through all blocks and
forks a variant from the trunk's input to the block of its first upgrade;
from there the variant runs as in the parent (same grouping by upgrade set,
same reloads and installs, same micro-batches), so its result equals the
parent's exactly (``tools/screen_qwen36_rco_v3.py check``).  A variant
without upgrades is the trunk.

Restrictions: no prune masks, one embedding kind.
"""

from __future__ import annotations

import gc
import time
from typing import Optional, Sequence

import torch

from search.streamed_chat_exact import (
    ExactResult,
    ExactStats,
    ExactVariant,
    StreamedChatExactEvaluator,
)
from search.streamed_prune import _sync
from search.streamed_prune_chat import ChatRow, _Pass, microbatches

TRUNK = -1


class SharedPrefixExactEvaluator(StreamedChatExactEvaluator):
    """Variants fork from one center trunk at their first upgraded block."""

    @torch.no_grad()
    def evaluate(self, rows: Sequence[ChatRow], variants: Sequence[ExactVariant]) -> ExactResult:
        rows = list(rows)
        variants = list(variants)
        if not rows or not variants:
            raise ValueError("need at least one row and one variant")
        if any(v.prune_mask is not None for v in variants):
            raise ValueError("the shared-prefix evaluator takes no prune masks")
        if len({v.embedding for v in variants}) != 1 or variants[0].embedding not in (
                "store", "checkpoint"):
            raise ValueError("all variants need the same embedding, 'store' or 'checkpoint'")
        if any(v.upgrades for v in variants) and self.upgrade_store is None:
            raise ValueError("upgrades need an upgrade store")
        first = [min((self.upgrade_store.block_index(k) for k in v.upgrades), default=None)
                 for v in variants]
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
        forked = sum(f is not None for f in first)
        stats.hidden_on_device = (
            stats.padded_tokens * hidden_size * 2 * (forked + 1) <= self.device_hidden_bytes)
        home = self.device if stats.hidden_on_device else torch.device("cpu")

        state = _Pass(self, stats, shapes, None)
        loaded: list[str] = []
        try:
            self.loader.move_runtime_buffers(self.model, self.device)
            kind = variants[0].embedding
            selected = self.embedding_selected
            if kind == "checkpoint":
                self.embedding_selected = ()
            try:
                trunk = [item.to(home) for item in self._embed(input_batches, stats)]
            finally:
                self.embedding_selected = selected
            hidden: dict[int, list] = {TRUNK: trunk}
            for path in self.adapter.final_module_paths:
                dtype = self.checkpoint_dtype if path == "lm_head" else None
                self._load(path, dtype, stats, loaded)

            for layer_index in range(self.layer_count):
                block = f"{self.adapter.layers_path}.{layer_index}"
                for number, layer in enumerate(first):
                    if layer == layer_index:
                        hidden[number] = list(hidden[TRUNK])
                by_set: dict[frozenset, list[int]] = {frozenset(): [TRUNK]}
                for number in hidden:
                    if number != TRUNK:
                        key = self._upgrades_in(variants[number], layer_index)
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
                            hidden[number] = [
                                self._exact_block(state, layer_index, micro,
                                                  item.to(self.device), None).to(home)
                                for micro, item in enumerate(hidden[number])]
                        _sync(self.device)
                        stats.forward_seconds += time.perf_counter() - forward_started
                finally:
                    if installed is not None:
                        self.loader.release_prefix(self.model, block)

            loss_started = time.perf_counter()
            kl: list[list] = [[None] * len(rows) for _ in variants]
            nll: list[list] = [[None] * len(rows) for _ in variants]
            sources = {number: TRUNK if layer is None else number
                       for number, layer in enumerate(first)}
            done: dict[int, tuple[list, list]] = {}
            for number in range(len(variants)):
                source = sources[number]
                if source not in done:
                    kl_rows, nll_rows = [None] * len(rows), [None] * len(rows)
                    for members, batch in zip(groups, hidden[source]):
                        batch = batch.to(self.device)
                        for slot, index in enumerate(members):
                            kl_rows[index], nll_rows[index] = self._row_losses(
                                batch[slot, :lengths[index]], rows[index])
                        del batch
                    if source != TRUNK:
                        hidden[source] = None
                    done[source] = (kl_rows, nll_rows)
                kl[number], nll[number] = list(done[source][0]), list(done[source][1])
            hidden.clear()
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


__all__ = ["SharedPrefixExactEvaluator"]
