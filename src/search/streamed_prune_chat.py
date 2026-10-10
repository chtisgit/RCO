"""Block-streamed expert-pruning objective for variable-length conversations.

``streamed_prune.StreamedPruneObjective`` takes fixed-length rows, scores
every position, and runs the whole batch through each block at once, so
GPU memory bounds the tokens per step.  Chat rows are 100 to 8,192 tokens
long, only the final assistant turn is scored, and every Gumbel sample
needs its own copy of each row.  This objective keeps the search
(``StreamedPruneSearch``), the surrogate routing and the loss, and changes
how a pass is executed:

* Rows are grouped into micro-batches of similar length, padded on the
  right up to a padded-token budget.  The model is causal, so padding never
  changes a real position, and padded positions are never scored.
* The decoder loop is driven here, micro-batch by micro-batch, with the
  rotary embeddings and masks that ``Qwen3_5MoeTextModel.forward`` builds
  for an unpadded batch of that shape.
* Each block is one autograd function.  Forward loads the block once, runs
  every micro-batch without gradients, keeps only the block inputs (in host
  memory), and releases the block.  Backward loads the block once more,
  then recomputes and backpropagates one micro-batch at a time.  Every block
  is therefore loaded exactly twice per pass, as before, however many
  tokens the step holds; GPU memory is bounded by the largest micro-batch.
* The loss per row is ``compact_top_k_kl`` (``compute_kl_loss`` with a
  compact top-k reference) at the scored positions only, averaged over the
  row's scored positions.  The objective is the mean over variants of each
  variant's mean row KL, as in ``StreamedPruneObjective``.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model_adapter import get_model_adapter
from search.streamed_prune import (
    StreamedPruneResult,
    StreamedPruneStats,
    _sync,
    compact_top_k_kl,
    surrogate_routing_weights,
)
from search.streaming import StreamingMemoryStats, _install_selected_candidates


@dataclass(frozen=True)
class ChatRow:
    """One conversation: its tokens, scored mask and compact reference.

    ``scored[p]`` marks token ``p`` as a scored target, so the hidden state
    at ``p - 1`` predicts it.  ``scored[0]`` must be 0.  The reference holds
    one row per scored target, in order.
    """

    tokens: torch.Tensor
    scored: torch.Tensor
    reference_values: torch.Tensor
    reference_indices: torch.Tensor

    def __post_init__(self) -> None:
        if self.tokens.ndim != 1 or self.scored.shape != self.tokens.shape:
            raise ValueError("tokens and scored must be 1-D and of equal length")
        if bool(self.scored[0]):
            raise ValueError("the first token has no prediction and cannot be scored")
        count = int(self.scored.sum())
        if count == 0:
            raise ValueError("a row needs at least one scored target")
        if (self.reference_values.shape[0] != count
                or self.reference_indices.shape != self.reference_values.shape):
            raise ValueError("reference rows must match the scored targets")


class ChatRowSet:
    """Indexable row collection that ``StreamedPruneSearch.take_step`` accepts.

    ``take_step`` reads ``shape[0]`` and indexes with a tensor of row
    numbers; the reference travels inside each row, so the same set is
    passed for the input ids and both reference arguments.
    """

    def __init__(self, rows: Sequence[ChatRow]) -> None:
        self.rows = list(rows)

    @property
    def shape(self) -> tuple[int]:
        return (len(self.rows),)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: Any) -> "ChatRowSet":
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        if isinstance(index, int):
            return ChatRowSet([self.rows[index]])
        return ChatRowSet([self.rows[i] for i in index])


def microbatches(lengths: Sequence[int], budget: int) -> list[list[int]]:
    """Longest first; a micro-batch's padded size stays within ``budget``.

    A row longer than the budget forms a micro-batch of its own.
    """
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    result: list[list[int]] = []
    current: list[int] = []
    for index in order:
        if current and (len(current) + 1) * lengths[current[0]] > budget:
            result.append(current)
            current = []
        current.append(index)
    if current:
        result.append(current)
    return result


@dataclass
class StreamedChatPruneStats(StreamedPruneStats):
    rows: int = 0
    tokens: int = 0
    padded_tokens: int = 0
    scored_targets: int = 0
    microbatch_count: int = 0
    host_activation_bytes: int = 0


class _Pass:
    """What every block of one pass needs."""

    def __init__(self, objective: "StreamedChatPruneObjective", stats, shapes,
                 token_variants) -> None:
        self.objective = objective
        self.stats = stats
        self.shapes = shapes                  # [(rows, width)] per micro-batch
        self.token_variants = token_variants  # flattened variant ids per micro-batch
        self.masks: dict[tuple[int, str], Any] = {}
        self.rotary: dict[int, Any] = {}


class _StreamedBlock(torch.autograd.Function):
    """One decoder block over all micro-batches, loaded once per direction."""

    @staticmethod
    def forward(ctx, state: _Pass, layer_index: int, survival: torch.Tensor,
                *hidden: torch.Tensor):
        objective = state.objective
        ctx.state, ctx.layer_index = state, layer_index
        ctx.save_for_backward(survival)
        stored = []
        outputs = []
        with objective._loaded_block(layer_index, state.stats):
            for number, item in enumerate(hidden):
                stored.append(item.detach().to("cpu"))
                with torch.no_grad():
                    outputs.append(objective._block(
                        state, layer_index, number, item, survival[:, layer_index]))
        state.stats.host_activation_bytes += sum(
            item.numel() * item.element_size() for item in stored)
        ctx.stored = stored
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        state, layer_index = ctx.state, ctx.layer_index
        objective = state.objective
        (survival,) = ctx.saved_tensors
        want_hidden = any(ctx.needs_input_grad[3:])
        grad_survival = torch.zeros_like(survival)
        grad_hidden = []
        with objective._loaded_block(layer_index, state.stats):
            for number, (stored, grad) in enumerate(zip(ctx.stored, grad_outputs)):
                hidden = stored.to(objective.device).requires_grad_(want_hidden)
                active = survival.detach().requires_grad_(True)
                with torch.enable_grad():
                    output = objective._block(
                        state, layer_index, number, hidden, active[:, layer_index])
                inputs = (hidden, active) if want_hidden else (active,)
                grads = torch.autograd.grad(output, inputs, grad, allow_unused=True)
                if grads[-1] is not None:
                    grad_survival += grads[-1]
                grad_hidden.append(grads[0] if want_hidden else None)
                del output, hidden, active, grads
        ctx.stored = None
        return (None, None, grad_survival, *grad_hidden)


class StreamedChatPruneObjective:
    """Streamed surrogate-pruned forward over conversations, masked top-k KL."""

    def __init__(
        self,
        model: nn.Module,
        checkpoint_loader: Any,
        *,
        device: torch.device | str,
        checkpoint_dtype: Optional[torch.dtype] = None,
        embedding_store: Any = None,
        embedding_selected: Sequence[tuple[str, int]] = (),
        position_chunk: int = 256,
        microbatch_tokens: int = 8192,
        pad_token: int = 0,
        router_suffix: str = "mlp.gate",
    ) -> None:
        self.model = model
        self.adapter = get_model_adapter(model)
        self.loader = checkpoint_loader
        self.device = torch.device(device)
        self.checkpoint_dtype = checkpoint_dtype
        self.embedding_store = embedding_store
        self.embedding_selected = tuple(embedding_selected)
        self.position_chunk = int(position_chunk)
        self.microbatch_tokens = int(microbatch_tokens)
        self.pad_token = int(pad_token)
        self.router_suffix = router_suffix
        self.model.requires_grad_(False)
        prefixes = [
            *self.adapter.embedding_paths,
            *(f"{self.adapter.layers_path}.{index}"
              for index in range(len(self.adapter.layers))),
            *self.adapter.final_module_paths,
        ]
        if hasattr(self.loader, "assert_prefix_schema"):
            for prefix in dict.fromkeys(prefixes):
                self.loader.assert_prefix_schema(self.model, prefix)

    @property
    def layer_count(self) -> int:
        return len(self.adapter.layers)

    # -- block execution ---------------------------------------------------

    class _Loaded:
        def __init__(self, objective, layer_index, stats):
            self.objective, self.layer_index, self.stats = objective, layer_index, stats
            self.path = f"{objective.adapter.layers_path}.{layer_index}"
            self.loaded = False

        def __enter__(self):
            objective = self.objective
            _sync(objective.device)
            started = time.perf_counter()
            self.loaded = True
            objective.loader.load_prefix(objective.model, self.path, device=objective.device,
                                         dtype=objective.checkpoint_dtype)
            _sync(objective.device)
            self.stats.load_seconds += time.perf_counter() - started
            self.stats.block_loads[self.layer_index] += 1

        def __exit__(self, *exc):
            if self.loaded:
                self.objective.loader.release_prefix(self.objective.model, self.path)
                self.stats.block_releases[self.layer_index] += 1
            return False

    def _loaded_block(self, layer_index: int, stats) -> "_Loaded":
        return self._Loaded(self, layer_index, stats)

    def _context(self, state: _Pass, number: int, hidden: torch.Tensor, layer_type: str):
        """Rotary embeddings and mask, as the text model builds them unpadded."""
        from transformers.masking_utils import (
            create_causal_mask,
            create_recurrent_attention_mask,
        )

        rows, width = state.shapes[number]
        if number not in state.rotary:
            position_ids = torch.arange(width, device=hidden.device).view(1, 1, -1).expand(
                4, rows, -1)
            text_position_ids = position_ids[0]
            state.rotary[number] = (
                text_position_ids,
                self.adapter.text_model.rotary_emb(hidden, position_ids[1:]))
        text_position_ids, position_embeddings = state.rotary[number]
        key = (number, layer_type)
        if key not in state.masks:
            builder = (create_causal_mask if layer_type == "full_attention"
                       else create_recurrent_attention_mask)
            state.masks[key] = builder(
                config=self.adapter.text_model.config, inputs_embeds=hidden,
                attention_mask=None, past_key_values=None, position_ids=text_position_ids)
        return text_position_ids, position_embeddings, state.masks[key]

    def _block(self, state: _Pass, layer_index: int, number: int,
               hidden: torch.Tensor, layer_survival: torch.Tensor) -> torch.Tensor:
        path = f"{self.adapter.layers_path}.{layer_index}"
        token_variant = state.token_variants[number]

        def hook(module, inputs, output):
            logits, weights, indices = output
            return logits, surrogate_routing_weights(
                weights, indices, layer_survival, token_variant), indices

        layer_type = self.adapter.text_model.config.layer_types[layer_index]
        text_position_ids, position_embeddings, mask = self._context(
            state, number, hidden, layer_type)
        handle = self.model.get_submodule(f"{path}.{self.router_suffix}").register_forward_hook(hook)
        try:
            output = self.adapter.layers[layer_index](
                hidden, position_embeddings=position_embeddings, attention_mask=mask,
                position_ids=text_position_ids, past_key_values=None, use_cache=False)
        finally:
            handle.remove()
        return output[0] if isinstance(output, tuple) else output

    # -- pass ----------------------------------------------------------------

    def _load(self, path: str, dtype, stats, loaded: list[str]) -> None:
        _sync(self.device)
        started = time.perf_counter()
        loaded.append(path)
        self.loader.load_prefix(self.model, path, device=self.device, dtype=dtype)
        _sync(self.device)
        stats.load_seconds += time.perf_counter() - started

    def _embed(self, batches: list[torch.Tensor], stats) -> list[torch.Tensor]:
        loaded: list[str] = []
        try:
            for path in self.adapter.embedding_paths:
                self._load(path, self.checkpoint_dtype, stats, loaded)
            if self.embedding_selected:
                _install_selected_candidates(
                    self.model, self.embedding_store, self.embedding_selected,
                    StreamingMemoryStats())
            with torch.no_grad():
                return [self.adapter.embeddings[0](ids.to(self.device)) for ids in batches]
        finally:
            for path in reversed(loaded):
                self.loader.release_prefix(self.model, path)

    def _row_kl(self, hidden: torch.Tensor, row: ChatRow) -> torch.Tensor:
        """Mean compact top-k KL over one row's scored targets."""
        positions = torch.nonzero(row.scored[1:].bool()).flatten().to(hidden.device)
        selected = self.adapter.text_model.norm(hidden[positions])
        values = row.reference_values.to(hidden.device)
        indices = row.reference_indices.to(hidden.device)
        weight = self.adapter.lm_head().weight

        def chunk_kl(chunk_hidden, chunk_values, chunk_indices):
            return compact_top_k_kl(F.linear(chunk_hidden, weight), chunk_values, chunk_indices)

        pieces = []
        for start in range(0, positions.numel(), self.position_chunk):
            stop = min(start + self.position_chunk, positions.numel())
            pieces.append(checkpoint(chunk_kl, selected[start:stop], values[start:stop],
                                     indices[start:stop], use_reentrant=False))
        return torch.cat(pieces).mean()

    def run(
        self,
        rows: ChatRowSet,
        survival: torch.Tensor,
        row_variant: torch.Tensor,
        reference_values: Any = None,
        reference_indices: Any = None,
        *,
        backward: bool = True,
    ) -> StreamedPruneResult:
        """Forward all rows; optionally backpropagate the objective.

        The reference arguments are accepted for ``take_step`` and ignored:
        each row carries its own reference.
        """
        rows = rows.rows if isinstance(rows, ChatRowSet) else list(rows)
        variants = survival.shape[0]
        if survival.shape[1] != self.layer_count:
            raise ValueError("survival layer count differs from the model")
        if row_variant.shape != (len(rows),):
            raise ValueError("row_variant must name one variant per row")
        rows_per_variant = torch.bincount(row_variant, minlength=variants)
        if bool((rows_per_variant == 0).any()):
            raise ValueError("every variant needs at least one row")
        stats = StreamedChatPruneStats(
            block_loads=[0] * self.layer_count, block_releases=[0] * self.layer_count)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()

        lengths = [int(row.tokens.numel()) for row in rows]
        groups = microbatches(lengths, self.microbatch_tokens)
        input_batches, shapes, token_variants = [], [], []
        for members in groups:
            width = lengths[members[0]]
            ids = torch.full((len(members), width), self.pad_token, dtype=torch.long)
            for slot, index in enumerate(members):
                ids[slot, :lengths[index]] = rows[index].tokens
            input_batches.append(ids)
            shapes.append((len(members), width))
            token_variants.append(row_variant[members].to(self.device).repeat_interleave(width))
        stats.rows = len(rows)
        stats.tokens = sum(lengths)
        stats.padded_tokens = sum(r * w for r, w in shapes)
        stats.scored_targets = sum(int(row.scored.sum()) for row in rows)
        stats.microbatch_count = len(groups)

        state = _Pass(self, stats, shapes, token_variants)
        loaded: list[str] = []
        try:
            self.loader.move_runtime_buffers(self.model, self.device)
            hidden = self._embed(input_batches, stats)
            for path in self.adapter.final_module_paths:
                dtype = self.checkpoint_dtype if path == "lm_head" else None
                self._load(path, dtype, stats, loaded)
            active_survival = survival.to(self.device)

            forward_started = time.perf_counter()
            with torch.set_grad_enabled(backward):
                for index in range(self.layer_count):
                    hidden = list(_StreamedBlock.apply(state, index, active_survival, *hidden))
                _sync(self.device)
                stats.forward_seconds = time.perf_counter() - forward_started

                loss_started = time.perf_counter()
                row_kl = [None] * len(rows)
                for members, batch in zip(groups, hidden):
                    for slot, index in enumerate(members):
                        row_kl[index] = self._row_kl(batch[slot, :lengths[index]], rows[index])
                row_kl = torch.stack(row_kl)
                row_weight = 1.0 / (
                    variants * rows_per_variant.to(self.device)[
                        row_variant.to(self.device)].to(row_kl.dtype))
                objective = (row_kl * row_weight).sum()
                _sync(self.device)
                stats.loss_seconds = time.perf_counter() - loss_started

            if backward:
                backward_started = time.perf_counter()
                objective.backward()
                _sync(self.device)
                stats.backward_seconds = time.perf_counter() - backward_started
            return StreamedPruneResult(
                row_kl.detach().cpu(), float(objective.detach().item()), stats)
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


def rows_from_corpus(conversations: Sequence[dict[str, Any]],
                     reference: Sequence[tuple[np.ndarray, np.ndarray]]) -> ChatRowSet:
    """Build rows from ``audit_qwen36_chat_kl`` conversations and reference."""
    rows = []
    for conversation, (values, indices) in zip(conversations, reference, strict=True):
        scored = torch.tensor(conversation["scored"], dtype=torch.bool)
        scored[0] = False
        rows.append(ChatRow(
            tokens=torch.tensor(conversation["tokens"], dtype=torch.long),
            scored=scored,
            reference_values=torch.from_numpy(np.asarray(values)),
            reference_indices=torch.from_numpy(np.asarray(indices))))
    return ChatRowSet(rows)


__all__ = [
    "ChatRow",
    "ChatRowSet",
    "StreamedChatPruneObjective",
    "StreamedChatPruneStats",
    "microbatches",
    "rows_from_corpus",
]
