"""Bounded-memory relaxed interpolation for one linear weight.

The released relaxed RCO path retains every dense candidate delta for the
duration of the search.  This module keeps the same interpolation and softmax
gradient, but delegates reference and alternative-delta reads to a row source.
Only one decoded row chunk is materialized at a time.  Backward saves the
linear input and probabilities, then rereads the reference and alternatives;
it never saves a candidate delta in the autograd context.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from manifold import project_gradient, retraction, vector_transport


class RelaxedLinearRowSource(Protocol):
    """External row source used by :func:`streaming_relaxed_linear`.

    Alternatives exclude the reference candidate.  Logits contain one entry
    for every alternative followed by the reference entry, matching the
    released ``WeightInterpolation`` convention.
    """

    in_features: int
    out_features: int
    alternative_count: int

    def iter_reference_rows(self) -> Iterable[tuple[int, Any]]:
        """Yield ``(start_row, values)`` chunks for the reference weight."""

    def iter_delta_rows(
        self, alternative_index: int,
    ) -> Iterable[tuple[int, Any]]:
        """Yield alternative-minus-reference row chunks."""

    def read_reference_rows(self, row_indices: Any) -> Any:
        """Read selected reference rows in the caller's order."""

    def read_delta_rows(self, alternative_index: int, row_indices: Any) -> Any:
        """Read selected alternative-minus-reference rows."""


@dataclass
class StreamingRelaxedLinearStats:
    """Decoded-I/O and active-chunk telemetry for one or more calls."""

    forward_reference_passes: int = 0
    forward_alternative_passes: int = 0
    backward_reference_passes: int = 0
    backward_alternative_passes: int = 0
    forward_reference_bytes: int = 0
    forward_alternative_bytes: int = 0
    backward_reference_bytes: int = 0
    backward_alternative_bytes: int = 0
    max_materialized_chunk_bytes: int = 0

    def _record(self, phase: str, kind: str, tensor: torch.Tensor) -> None:
        field = f"{phase}_{kind}_bytes"
        size = tensor.numel() * tensor.element_size()
        setattr(self, field, getattr(self, field) + size)
        self.max_materialized_chunk_bytes = max(
            self.max_materialized_chunk_bytes, size)

    def _record_pass(self, phase: str, kind: str) -> None:
        field = f"{phase}_{kind}_passes"
        setattr(self, field, getattr(self, field) + 1)


def _serialized_cost_problem(
    logits: torch.Tensor,
    low_costs: Iterable[int],
    high_costs: Iterable[int],
    target_cost: int | None = None,
) -> tuple[int, torch.Tensor, float, int | None]:
    low = tuple(int(value) for value in low_costs)
    high = tuple(int(value) for value in high_costs)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("serialized-cost logits must have shape [groups, 2]")
    if not low or len(low) != len(high) or len(low) != logits.shape[0]:
        raise ValueError("serialized cost vectors must match the logit groups")
    if any(value < 0 for value in low):
        raise ValueError("low serialized costs must be non-negative")
    increments = tuple(
        high_value - low_value
        for low_value, high_value in zip(low, high)
    )
    if any(value <= 0 for value in increments):
        raise ValueError("every high serialized cost must exceed its low cost")
    base = sum(low)
    residual = None if target_cost is None else int(target_cost) - base
    if residual is not None and not 0 <= residual <= sum(increments):
        raise ValueError("target serialized cost is outside candidate bounds")
    scale = float(max(increments))
    normalized = torch.zeros(
        logits.shape, device=logits.device, dtype=logits.dtype)
    normalized[:, 1] = torch.as_tensor(
        increments, device=logits.device, dtype=logits.dtype) / scale
    return base, normalized, scale, residual


def expected_serialized_cost(
    logits: torch.Tensor,
    low_costs: Iterable[int],
    high_costs: Iterable[int],
) -> float:
    """Return the relaxed expected bytes for per-group low/high choices."""
    low = tuple(int(value) for value in low_costs)
    high = tuple(int(value) for value in high_costs)
    base, _, _, _ = _serialized_cost_problem(logits, low, high)
    probabilities = torch.softmax(logits.detach().double(), dim=-1)[:, 1]
    increments = torch.as_tensor(
        [high_value - low_value
         for low_value, high_value in zip(low, high)],
        device=probabilities.device,
        dtype=torch.float64,
    )
    return float(base + torch.dot(probabilities, increments).item())


def project_serialized_cost_gradient_(
    logits: torch.Tensor,
    low_costs: Iterable[int],
    high_costs: Iterable[int],
) -> tuple[float, float, float]:
    """Project ``logits.grad`` onto the exact-byte budget tangent plane."""
    if logits.grad is None:
        raise ValueError("serialized-cost logits have no gradient")
    _, normalized, _, _ = _serialized_cost_problem(
        logits, low_costs, high_costs)
    return project_gradient(logits, normalized)


def retract_serialized_cost_(
    logits: torch.Tensor,
    low_costs: Iterable[int],
    high_costs: Iterable[int],
    target_cost: int,
    *,
    tolerance_bytes: float = 4096.0,
) -> float:
    """Retract relaxed logits to a nonuniform serialized-byte constraint."""
    if tolerance_bytes <= 0:
        raise ValueError("serialized-cost tolerance must be positive")
    _, normalized, scale, residual = _serialized_cost_problem(
        logits, low_costs, high_costs, target_cost)
    assert residual is not None
    retraction(
        logits,
        normalized,
        residual / scale,
        tol=tolerance_bytes / scale,
    )
    return expected_serialized_cost(logits, low_costs, high_costs)


def transport_serialized_cost_momentum_(
    optimizer: torch.optim.Optimizer,
    logits: torch.Tensor,
    low_costs: Iterable[int],
    high_costs: Iterable[int],
) -> None:
    """Project Adam's first moment onto the updated exact-byte tangent."""
    _, normalized, _, _ = _serialized_cost_problem(
        logits, low_costs, high_costs)
    vector_transport(optimizer, logits, normalized)


def _validated_rows(
    rows: Iterable[tuple[int, Any]],
    *,
    source: RelaxedLinearRowSource,
    device: torch.device,
    dtype: torch.dtype,
    phase: str,
    kind: str,
    stats: StreamingRelaxedLinearStats,
):
    """Move one row chunk at a time and reject gaps, overlap, or bad shapes."""
    expected_start = 0
    for raw_start, raw_values in rows:
        start = int(raw_start)
        if start != expected_start:
            raise ValueError(
                f"{kind} rows start at {start}, expected {expected_start}")
        values = torch.as_tensor(raw_values).to(device=device, dtype=dtype)
        if values.ndim != 2 or values.shape[1] != source.in_features:
            raise ValueError(
                f"{kind} chunk has shape {tuple(values.shape)}, expected "
                f"[rows, {source.in_features}]")
        if values.shape[0] == 0:
            raise ValueError(f"{kind} source yielded an empty row chunk")
        stop = start + values.shape[0]
        if stop > source.out_features:
            raise ValueError(
                f"{kind} rows end at {stop}, beyond {source.out_features}")
        stats._record(phase, kind, values)
        yield start, stop, values
        expected_start = stop
    if expected_start != source.out_features:
        raise ValueError(
            f"{kind} source ended at row {expected_start}, expected "
            f"{source.out_features}")


class _StreamingRelaxedLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        logits: torch.Tensor,
        bias: torch.Tensor | None,
        source: RelaxedLinearRowSource,
        stats: StreamingRelaxedLinearStats,
    ) -> torch.Tensor:
        if input_tensor.ndim < 1:
            raise ValueError("linear input must have at least one dimension")
        if input_tensor.shape[-1] != source.in_features:
            raise ValueError(
                f"input has {input_tensor.shape[-1]} features, expected "
                f"{source.in_features}")
        if logits.ndim != 1 or logits.numel() != source.alternative_count + 1:
            raise ValueError(
                f"logits must have shape ({source.alternative_count + 1},)")
        if bias is not None and bias.shape != (source.out_features,):
            raise ValueError(
                f"bias has shape {tuple(bias.shape)}, expected "
                f"({source.out_features},)")

        probabilities = torch.softmax(
            logits.to(device=input_tensor.device, dtype=input_tensor.dtype),
            dim=0,
        )
        output = input_tensor.new_empty(
            (*input_tensor.shape[:-1], source.out_features))

        stats._record_pass("forward", "reference")
        for start, stop, reference in _validated_rows(
            source.iter_reference_rows(),
            source=source,
            device=input_tensor.device,
            dtype=input_tensor.dtype,
            phase="forward",
            kind="reference",
            stats=stats,
        ):
            bias_chunk = None
            if bias is not None:
                bias_chunk = bias[start:stop].to(
                    device=input_tensor.device, dtype=input_tensor.dtype)
            output[..., start:stop] = F.linear(
                input_tensor, reference, bias_chunk)

        for alternative_index in range(source.alternative_count):
            stats._record_pass("forward", "alternative")
            for start, stop, delta in _validated_rows(
                source.iter_delta_rows(alternative_index),
                source=source,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
                phase="forward",
                kind="alternative",
                stats=stats,
            ):
                output[..., start:stop].add_(
                    F.linear(input_tensor, delta)
                    * probabilities[alternative_index])

        ctx.source = source
        ctx.stats = stats
        ctx.logits_device = logits.device
        ctx.logits_dtype = logits.dtype
        ctx.bias_device = None if bias is None else bias.device
        ctx.bias_dtype = None if bias is None else bias.dtype
        ctx.save_for_backward(input_tensor, probabilities)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_tensor, probabilities = ctx.saved_tensors
        source = ctx.source
        stats = ctx.stats
        grad_output = grad_output.to(
            device=input_tensor.device, dtype=input_tensor.dtype)
        grad_input = torch.zeros_like(input_tensor)

        stats._record_pass("backward", "reference")
        for start, stop, reference in _validated_rows(
            source.iter_reference_rows(),
            source=source,
            device=input_tensor.device,
            dtype=input_tensor.dtype,
            phase="backward",
            kind="reference",
            stats=stats,
        ):
            grad_input.add_(F.linear(
                grad_output[..., start:stop], reference.transpose(0, 1)))

        probability_gradients = []
        for alternative_index in range(source.alternative_count):
            stats._record_pass("backward", "alternative")
            probability_gradient = input_tensor.new_zeros(())
            for start, stop, delta in _validated_rows(
                source.iter_delta_rows(alternative_index),
                source=source,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
                phase="backward",
                kind="alternative",
                stats=stats,
            ):
                grad_slice = grad_output[..., start:stop]
                delta_output = F.linear(input_tensor, delta)
                probability_gradient.add_((grad_slice * delta_output).sum())
                grad_input.add_(
                    F.linear(grad_slice, delta.transpose(0, 1))
                    * probabilities[alternative_index])
            probability_gradients.append(probability_gradient)

        # The reference has zero derivative in
        # W_ref + sum_i p_i (W_i - W_ref).  Softmax coupling still gives its
        # logit a generally nonzero gradient.
        probability_gradients.append(input_tensor.new_zeros(()))
        d_probability = torch.stack(probability_gradients)
        centered = d_probability - (probabilities * d_probability).sum()
        grad_logits = (probabilities * centered).to(
            device=ctx.logits_device, dtype=ctx.logits_dtype)

        grad_bias = None
        if ctx.bias_device is not None:
            reduce_dims = tuple(range(grad_output.ndim - 1))
            grad_bias = grad_output.sum(dim=reduce_dims).to(
                device=ctx.bias_device, dtype=ctx.bias_dtype)
        return grad_input, grad_logits, grad_bias, None, None


def streaming_relaxed_linear(
    input_tensor: torch.Tensor,
    logits: torch.Tensor,
    source: RelaxedLinearRowSource,
    *,
    bias: torch.Tensor | None = None,
    stats: StreamingRelaxedLinearStats | None = None,
) -> torch.Tensor:
    """Apply released-RCO interpolation while streaming all weight rows.

    The source is read once in forward and again in backward.  Only
    ``input_tensor`` and the softmax probabilities are retained by autograd.
    """
    if stats is None:
        stats = StreamingRelaxedLinearStats()
    return _StreamingRelaxedLinear.apply(
        input_tensor, logits, bias, source, stats)


class _StreamingRelaxedEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        source: RelaxedLinearRowSource,
        stats: StreamingRelaxedLinearStats,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if input_ids.dtype not in {
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
        }:
            raise TypeError("embedding indices must have an integer dtype")
        if logits.ndim != 1 or logits.numel() != source.alternative_count + 1:
            raise ValueError(
                f"logits must have shape ({source.alternative_count + 1},)")
        flat_ids = input_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        unique_ids, inverse = torch.unique(
            flat_ids, sorted=True, return_inverse=True)
        if len(unique_ids) == 0:
            raise ValueError("embedding input is empty")
        if int(unique_ids[0]) < 0 or int(unique_ids[-1]) >= source.out_features:
            raise IndexError("embedding index is out of range")
        device = input_ids.device
        dtype = output_dtype
        probabilities = torch.softmax(
            logits.to(device=device, dtype=dtype), dim=0)

        stats._record_pass("forward", "reference")
        reference = torch.as_tensor(
            source.read_reference_rows(unique_ids.numpy()),
        ).to(device=device, dtype=dtype)
        stats._record("forward", "reference", reference)
        mixed = reference.clone()
        for alternative_index in range(source.alternative_count):
            stats._record_pass("forward", "alternative")
            delta = torch.as_tensor(source.read_delta_rows(
                alternative_index, unique_ids.numpy()),
            ).to(device=device, dtype=dtype)
            stats._record("forward", "alternative", delta)
            mixed.add_(delta * probabilities[alternative_index])
        output = mixed[inverse.to(device=device)].reshape(
            *input_ids.shape, source.in_features)
        ctx.source = source
        ctx.stats = stats
        ctx.logits_device = logits.device
        ctx.logits_dtype = logits.dtype
        ctx.save_for_backward(
            unique_ids, inverse, probabilities)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        unique_ids, inverse, probabilities = ctx.saved_tensors
        source = ctx.source
        stats = ctx.stats
        flattened_gradient = grad_output.reshape(-1, source.in_features)
        accumulated = torch.zeros(
            len(unique_ids),
            source.in_features,
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        accumulated.index_add_(
            0, inverse.to(device=grad_output.device), flattened_gradient)
        probability_gradients = []
        for alternative_index in range(source.alternative_count):
            stats._record_pass("backward", "alternative")
            delta = torch.as_tensor(source.read_delta_rows(
                alternative_index, unique_ids.numpy()),
            ).to(device=grad_output.device, dtype=grad_output.dtype)
            stats._record("backward", "alternative", delta)
            probability_gradients.append((accumulated * delta).sum())
        probability_gradients.append(grad_output.new_zeros(()))
        d_probability = torch.stack(probability_gradients)
        probabilities = probabilities.to(
            device=grad_output.device, dtype=grad_output.dtype)
        grad_logits = probabilities * (
            d_probability - (probabilities * d_probability).sum())
        return (
            None,
            grad_logits.to(
                device=ctx.logits_device, dtype=ctx.logits_dtype),
            None,
            None,
            None,
        )


def streaming_relaxed_embedding(
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    source: RelaxedLinearRowSource,
    *,
    dtype: torch.dtype | None = None,
    stats: StreamingRelaxedLinearStats | None = None,
) -> torch.Tensor:
    """Embed token IDs while reading only their native candidate rows."""
    if stats is None:
        stats = StreamingRelaxedLinearStats()
    if dtype is None:
        dtype = logits.dtype
    return _StreamingRelaxedEmbedding.apply(
        input_ids, logits, source, stats, dtype)


def _mixed_row_chunks(
    source: RelaxedLinearRowSource,
    probabilities: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    phase: str,
    stats: StreamingRelaxedLinearStats,
):
    stats._record_pass(phase, "reference")
    reference_rows = iter(_validated_rows(
        source.iter_reference_rows(),
        source=source,
        device=device,
        dtype=dtype,
        phase=phase,
        kind="reference",
        stats=stats,
    ))
    delta_rows = []
    for alternative_index in range(source.alternative_count):
        stats._record_pass(phase, "alternative")
        delta_rows.append(iter(_validated_rows(
            source.iter_delta_rows(alternative_index),
            source=source,
            device=device,
            dtype=dtype,
            phase=phase,
            kind="alternative",
            stats=stats,
        )))
    for start, stop, reference in reference_rows:
        deltas = []
        mixed = reference.clone()
        for alternative_index, rows in enumerate(delta_rows):
            try:
                delta_start, delta_stop, delta = next(rows)
            except StopIteration as error:
                raise ValueError("alternative rows ended before reference") from error
            if delta_start != start or delta_stop != stop:
                raise ValueError("reference and alternative row chunks differ")
            mixed.add_(delta * probabilities[alternative_index])
            deltas.append(delta)
        yield start, stop, mixed, deltas
    for rows in delta_rows:
        try:
            next(rows)
        except StopIteration:
            continue
        raise ValueError("alternative rows extend beyond reference")


class _StreamingRelaxedCausalCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
        active_indices: torch.Tensor,
        source: RelaxedLinearRowSource,
        stats: StreamingRelaxedLinearStats,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3 or labels.ndim != 2:
            raise ValueError("expected hidden [B,S,H] and labels [B,S]")
        if hidden_states.shape[:2] != labels.shape:
            raise ValueError("hidden-state and label shapes differ")
        if hidden_states.shape[-1] != source.in_features:
            raise ValueError("output row source width differs from hidden size")
        if logits.ndim != 1 or logits.numel() != source.alternative_count + 1:
            raise ValueError(
                f"logits must have shape ({source.alternative_count + 1},)")
        shifted_hidden = hidden_states[:, :-1].reshape(
            -1, hidden_states.shape[-1])
        shifted_labels = labels[:, 1:].reshape(-1).to(
            device=hidden_states.device, dtype=torch.long)
        active = active_indices.to(device=hidden_states.device, dtype=torch.long)
        selected_hidden = shifted_hidden.index_select(0, active)
        selected_labels = shifted_labels.index_select(0, active)
        if torch.any(selected_labels < 0) or torch.any(
            selected_labels >= source.out_features
        ):
            raise IndexError("selected causal target is outside the vocabulary")
        accumulation_dtype = (
            torch.float64
            if hidden_states.dtype == torch.float64
            else torch.float32
        )
        probabilities = torch.softmax(
            logits.to(device=hidden_states.device, dtype=hidden_states.dtype),
            dim=0,
        )
        normalizer = None
        target_logits = torch.zeros(
            len(active), device=hidden_states.device,
            dtype=accumulation_dtype)
        for start, stop, mixed, _ in _mixed_row_chunks(
            source,
            probabilities,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            phase="forward",
            stats=stats,
        ):
            chunk_logits = F.linear(selected_hidden, mixed).to(
                dtype=accumulation_dtype)
            chunk_lse = torch.logsumexp(chunk_logits, dim=-1)
            normalizer = (
                chunk_lse if normalizer is None
                else torch.logaddexp(normalizer, chunk_lse))
            in_chunk = (selected_labels >= start) & (selected_labels < stop)
            if torch.any(in_chunk):
                target_logits[in_chunk] = chunk_logits[
                    in_chunk, selected_labels[in_chunk] - start]
        loss = (normalizer - target_logits).mean()
        ctx.source = source
        ctx.stats = stats
        ctx.hidden_shape = hidden_states.shape
        ctx.logits_device = logits.device
        ctx.logits_dtype = logits.dtype
        ctx.accumulation_dtype = accumulation_dtype
        ctx.save_for_backward(
            selected_hidden,
            selected_labels,
            active_indices.detach().to(device="cpu", dtype=torch.long),
            probabilities,
            normalizer,
        )
        return loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        (selected_hidden, selected_labels, active_indices,
         probabilities, normalizer) = ctx.saved_tensors
        source = ctx.source
        stats = ctx.stats
        selected_gradient = torch.zeros_like(selected_hidden)
        probability_gradients = [
            selected_hidden.new_zeros(())
            for _ in range(source.alternative_count)
        ]
        scale = grad_loss.to(
            device=selected_hidden.device, dtype=ctx.accumulation_dtype,
        ) / len(selected_labels)
        for start, stop, mixed, deltas in _mixed_row_chunks(
            source,
            probabilities,
            device=selected_hidden.device,
            dtype=selected_hidden.dtype,
            phase="backward",
            stats=stats,
        ):
            chunk_logits = F.linear(selected_hidden, mixed).to(
                dtype=ctx.accumulation_dtype)
            d_logits = torch.exp(chunk_logits - normalizer[:, None])
            in_chunk = (selected_labels >= start) & (selected_labels < stop)
            if torch.any(in_chunk):
                d_logits[
                    in_chunk, selected_labels[in_chunk] - start] -= 1.0
            d_logits.mul_(scale)
            d_values = d_logits.to(dtype=selected_hidden.dtype)
            selected_gradient.add_(F.linear(
                d_values, mixed.transpose(0, 1)))
            for alternative_index, delta in enumerate(deltas):
                delta_output = F.linear(selected_hidden, delta).to(
                    dtype=ctx.accumulation_dtype)
                probability_gradients[alternative_index].add_(
                    (d_logits * delta_output).sum().to(
                        dtype=selected_hidden.dtype))

        probability_gradients.append(selected_hidden.new_zeros(()))
        d_probability = torch.stack(probability_gradients)
        centered = d_probability - (probabilities * d_probability).sum()
        grad_logits = probabilities * centered
        shifted_gradient = torch.zeros(
            ctx.hidden_shape[0] * (ctx.hidden_shape[1] - 1),
            ctx.hidden_shape[2],
            device=selected_hidden.device,
            dtype=selected_hidden.dtype,
        )
        shifted_gradient.index_copy_(
            0,
            active_indices.to(device=selected_hidden.device),
            selected_gradient,
        )
        grad_hidden = torch.zeros(
            ctx.hidden_shape,
            device=selected_hidden.device,
            dtype=selected_hidden.dtype,
        )
        grad_hidden[:, :-1] = shifted_gradient.reshape(
            ctx.hidden_shape[0], ctx.hidden_shape[1] - 1, ctx.hidden_shape[2])
        return (
            grad_hidden,
            None,
            grad_logits.to(
                device=ctx.logits_device, dtype=ctx.logits_dtype),
            None,
            None,
            None,
        )


def streaming_relaxed_causal_cross_entropy(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    source: RelaxedLinearRowSource,
    *,
    loss_mask: torch.Tensor | None = None,
    stats: StreamingRelaxedLinearStats | None = None,
) -> tuple[torch.Tensor, int]:
    """Exact causal CE with native mixed output rows streamed in chunks."""
    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected hidden [B,S,H] and labels [B,S]")
    shifted_count = labels[:, 1:].numel()
    if shifted_count == 0:
        raise ValueError("causal cross-entropy requires at least two tokens")
    if loss_mask is None:
        active_indices = torch.arange(shifted_count, dtype=torch.long)
    else:
        if loss_mask.shape != labels.shape:
            raise ValueError("loss mask and labels must have the same shape")
        active_indices = torch.nonzero(
            loss_mask[:, 1:].reshape(-1).to(device="cpu", dtype=torch.bool),
            as_tuple=False,
        ).flatten()
    token_count = len(active_indices)
    if token_count == 0:
        raise ValueError("loss mask selects no next-token targets")
    if stats is None:
        stats = StreamingRelaxedLinearStats()
    loss = _StreamingRelaxedCausalCrossEntropy.apply(
        hidden_states, labels, logits, active_indices, source, stats)
    return loss, token_count


class DenseDeltaRowSource:
    """Dense oracle source for equivalence tests, not production storage."""

    def __init__(
        self,
        reference: torch.Tensor,
        alternatives: Iterable[torch.Tensor],
        *,
        rows_per_chunk: int = 16,
    ) -> None:
        if reference.ndim != 2:
            raise ValueError("reference must be a 2-D linear weight")
        if rows_per_chunk < 1:
            raise ValueError("rows_per_chunk must be positive")
        candidates = tuple(alternatives)
        if not candidates:
            raise ValueError("at least one alternative is required")
        if any(candidate.shape != reference.shape for candidate in candidates):
            raise ValueError("all alternatives must match the reference shape")
        self.reference = reference.detach()
        self.alternatives = tuple(candidate.detach() for candidate in candidates)
        self.rows_per_chunk = int(rows_per_chunk)
        self.out_features, self.in_features = reference.shape
        self.alternative_count = len(candidates)

    def iter_reference_rows(self):
        for start in range(0, self.out_features, self.rows_per_chunk):
            stop = min(start + self.rows_per_chunk, self.out_features)
            yield start, self.reference[start:stop]

    def iter_delta_rows(self, alternative_index: int):
        if not 0 <= alternative_index < self.alternative_count:
            raise IndexError("alternative index is out of range")
        candidate = self.alternatives[alternative_index]
        for start in range(0, self.out_features, self.rows_per_chunk):
            stop = min(start + self.rows_per_chunk, self.out_features)
            yield start, candidate[start:stop] - self.reference[start:stop]

    def read_reference_rows(self, row_indices: Any):
        return self.reference[torch.as_tensor(row_indices, dtype=torch.long)]

    def read_delta_rows(self, alternative_index: int, row_indices: Any):
        if not 0 <= alternative_index < self.alternative_count:
            raise IndexError("alternative index is out of range")
        indices = torch.as_tensor(row_indices, dtype=torch.long)
        return (
            self.alternatives[alternative_index][indices]
            - self.reference[indices])


@dataclass(frozen=True)
class RelaxedLinearBinding:
    """Bind one model linear to a group logit row and external row source."""

    module_path: str
    group_index: int
    source: RelaxedLinearRowSource
    stats: StreamingRelaxedLinearStats


@dataclass(frozen=True)
class RelaxedExpertsBinding:
    """Bind one fused Qwen expert collection to three candidate groups."""

    module_path: str
    gate_group_index: int
    up_group_index: int
    down_group_index: int
    source_factory: Callable[[str, int], RelaxedLinearRowSource]
    gate_stats: StreamingRelaxedLinearStats
    up_stats: StreamingRelaxedLinearStats
    down_stats: StreamingRelaxedLinearStats


@dataclass(frozen=True)
class RelaxedRouterBinding:
    """Bind a fused Qwen top-k router's direct weight parameter."""

    module_path: str
    group_index: int
    source: RelaxedLinearRowSource
    stats: StreamingRelaxedLinearStats


@dataclass
class CheckpointedRelaxedBlockStats:
    """Materialization telemetry for a checkpointed streamed block."""

    load_passes: int = 0
    release_passes: int = 0
    checkpoint_bytes_read: int = 0
    max_materialized_block_bytes: int = 0


@contextmanager
def _patch_relaxed_linears(
    model: nn.Module,
    logits: torch.Tensor,
    bindings: tuple[RelaxedLinearBinding, ...],
):
    restored = []
    try:
        for binding in bindings:
            if not 0 <= binding.group_index < logits.shape[0]:
                raise IndexError(
                    f"group index {binding.group_index} is outside logits")
            module = model.get_submodule(binding.module_path)
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"{binding.module_path!r} is not torch.nn.Linear")
            if (
                module.in_features != binding.source.in_features
                or module.out_features != binding.source.out_features
            ):
                raise ValueError(
                    f"row source shape differs for {binding.module_path!r}")
            had_instance_forward = "forward" in module.__dict__
            instance_forward = module.__dict__.get("forward")
            restored.append((module, had_instance_forward, instance_forward))

            def relaxed_forward(
                input_tensor: torch.Tensor,
                *,
                _binding: RelaxedLinearBinding = binding,
                _module: nn.Linear = module,
            ) -> torch.Tensor:
                return streaming_relaxed_linear(
                    input_tensor,
                    logits[_binding.group_index],
                    _binding.source,
                    bias=_module.bias,
                    stats=_binding.stats,
                )

            module.forward = relaxed_forward
        yield
    finally:
        for module, had_instance_forward, instance_forward in reversed(restored):
            if had_instance_forward:
                module.forward = instance_forward
            else:
                del module.forward


@contextmanager
def _patch_relaxed_experts(
    model: nn.Module,
    logits: torch.Tensor,
    bindings: tuple[RelaxedExpertsBinding, ...],
):
    restored = []
    try:
        for binding in bindings:
            group_indices = (
                binding.gate_group_index,
                binding.up_group_index,
                binding.down_group_index,
            )
            if any(not 0 <= index < logits.shape[0] for index in group_indices):
                raise IndexError("expert group index is outside logits")
            module = model.get_submodule(binding.module_path)
            required = (
                "num_experts", "hidden_dim", "intermediate_dim", "act_fn")
            if any(not hasattr(module, name) for name in required):
                raise TypeError(
                    f"{binding.module_path!r} is not a supported fused expert "
                    "collection")
            had_instance_forward = "forward" in module.__dict__
            instance_forward = module.__dict__.get("forward")
            restored.append((module, had_instance_forward, instance_forward))

            def relaxed_experts_forward(
                hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor,
                *,
                _binding: RelaxedExpertsBinding = binding,
                _module: nn.Module = module,
            ) -> torch.Tensor:
                final_hidden_states = torch.zeros_like(hidden_states)
                with torch.no_grad():
                    expert_mask = F.one_hot(
                        top_k_index, num_classes=_module.num_experts)
                    expert_mask = expert_mask.permute(2, 1, 0)
                    expert_hit = torch.greater(
                        expert_mask.sum(dim=(-1, -2)), 0).nonzero()

                for raw_expert_index in expert_hit:
                    expert_index = int(raw_expert_index[0].item())
                    if expert_index == _module.num_experts:
                        continue
                    top_k_pos, token_index = torch.where(
                        expert_mask[expert_index])
                    current_state = hidden_states[token_index]
                    gate = streaming_relaxed_linear(
                        current_state,
                        logits[_binding.gate_group_index],
                        _binding.source_factory("gate", expert_index),
                        stats=_binding.gate_stats,
                    )
                    up = streaming_relaxed_linear(
                        current_state,
                        logits[_binding.up_group_index],
                        _binding.source_factory("up", expert_index),
                        stats=_binding.up_stats,
                    )
                    current_hidden_states = _module.act_fn(gate) * up
                    current_hidden_states = streaming_relaxed_linear(
                        current_hidden_states,
                        logits[_binding.down_group_index],
                        _binding.source_factory("down", expert_index),
                        stats=_binding.down_stats,
                    )
                    current_hidden_states = (
                        current_hidden_states
                        * top_k_weights[token_index, top_k_pos, None])
                    final_hidden_states.index_add_(
                        0,
                        token_index,
                        current_hidden_states.to(final_hidden_states.dtype),
                    )
                return final_hidden_states

            module.forward = relaxed_experts_forward
        yield
    finally:
        for module, had_instance_forward, instance_forward in reversed(restored):
            if had_instance_forward:
                module.forward = instance_forward
            else:
                del module.forward


@contextmanager
def _patch_relaxed_routers(
    model: nn.Module,
    logits: torch.Tensor,
    bindings: tuple[RelaxedRouterBinding, ...],
):
    restored = []
    try:
        for binding in bindings:
            if not 0 <= binding.group_index < logits.shape[0]:
                raise IndexError("router group index is outside logits")
            module = model.get_submodule(binding.module_path)
            required = ("top_k", "num_experts", "hidden_dim")
            if any(not hasattr(module, name) for name in required):
                raise TypeError(
                    f"{binding.module_path!r} is not a supported top-k router")
            if (
                binding.source.in_features != module.hidden_dim
                or binding.source.out_features != module.num_experts
            ):
                raise ValueError("router row source shape differs")
            had_instance_forward = "forward" in module.__dict__
            instance_forward = module.__dict__.get("forward")
            restored.append((module, had_instance_forward, instance_forward))

            def relaxed_router_forward(
                hidden_states: torch.Tensor,
                *,
                _binding: RelaxedRouterBinding = binding,
                _module: nn.Module = module,
            ):
                hidden_states = hidden_states.reshape(-1, _module.hidden_dim)
                router_logits = streaming_relaxed_linear(
                    hidden_states,
                    logits[_binding.group_index],
                    _binding.source,
                    stats=_binding.stats,
                )
                router_probabilities = F.softmax(
                    router_logits, dtype=torch.float, dim=-1)
                router_top_value, router_indices = torch.topk(
                    router_probabilities, _module.top_k, dim=-1)
                router_top_value /= router_top_value.sum(
                    dim=-1, keepdim=True)
                router_scores = router_top_value.to(router_logits.dtype)
                return router_logits, router_scores, router_indices

            module.forward = relaxed_router_forward
        yield
    finally:
        for module, had_instance_forward, instance_forward in reversed(restored):
            if had_instance_forward:
                module.forward = instance_forward
            else:
                del module.forward


class CheckpointedStreamingRelaxedBlock(nn.Module):
    """Reload one block for forward and checkpoint recomputation/backward.

    The block's searched ``nn.Linear`` calls are temporarily redirected to
    external native row sources.  Nonsearched operations use the reloaded
    checkpoint tensors.  Non-reentrant checkpointing discards forward
    intermediates and invokes this same load/patch/run/release region during
    backward, keeping materialized model blocks bounded to the active block.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        model: nn.Module,
        path: str,
        checkpoint_loader: Any,
        logits: torch.Tensor,
        bindings: Iterable[RelaxedLinearBinding],
        expert_bindings: Iterable[RelaxedExpertsBinding] = (),
        router_bindings: Iterable[RelaxedRouterBinding] = (),
        device: torch.device | str,
        stats: CheckpointedRelaxedBlockStats | None = None,
    ) -> None:
        super().__init__()
        if logits.ndim != 2:
            raise ValueError("relaxed group logits must be two-dimensional")
        self.module = module
        object.__setattr__(self, "_root_model", model)
        object.__setattr__(self, "_checkpoint_loader", checkpoint_loader)
        object.__setattr__(self, "_logits", logits)
        object.__setattr__(self, "_bindings", tuple(bindings))
        object.__setattr__(self, "_expert_bindings", tuple(expert_bindings))
        object.__setattr__(self, "_router_bindings", tuple(router_bindings))
        object.__setattr__(
            self,
            "_streaming_stats",
            stats if stats is not None else CheckpointedRelaxedBlockStats(),
        )
        self.path = path
        self.device = torch.device(device)
        self.train(module.training)

    @property
    def streaming_stats(self) -> CheckpointedRelaxedBlockStats:
        return object.__getattribute__(self, "_streaming_stats")

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["module"], name)

    def _run(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        remaining_args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        model = object.__getattribute__(self, "_root_model")
        loader = object.__getattribute__(self, "_checkpoint_loader")
        bindings = object.__getattribute__(self, "_bindings")
        expert_bindings = object.__getattribute__(self, "_expert_bindings")
        router_bindings = object.__getattribute__(self, "_router_bindings")
        stats = object.__getattribute__(self, "_streaming_stats")
        loaded = False
        try:
            # Mark the prefix before I/O so partial materialization is also
            # released on failure.
            loaded = True
            block_bytes = loader.load_prefix(
                model, self.path, device=self.device)
            stats.load_passes += 1
            stats.checkpoint_bytes_read += block_bytes
            stats.max_materialized_block_bytes = max(
                stats.max_materialized_block_bytes, block_bytes)
            with (
                _patch_relaxed_linears(model, logits, bindings),
                _patch_relaxed_experts(model, logits, expert_bindings),
                _patch_relaxed_routers(model, logits, router_bindings),
            ):
                return self.module(hidden_states, *remaining_args, **kwargs)
        finally:
            if loaded:
                loader.release_prefix(model, self.path)
                stats.release_passes += 1

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        logits = object.__getattribute__(self, "_logits")

        def run(hidden: torch.Tensor, active_logits: torch.Tensor):
            return self._run(hidden, active_logits, args, kwargs)

        return checkpoint(
            run,
            hidden_states,
            logits,
            use_reentrant=False,
            preserve_rng_state=True,
        )


__all__ = [
    "CheckpointedRelaxedBlockStats",
    "CheckpointedStreamingRelaxedBlock",
    "DenseDeltaRowSource",
    "RelaxedExpertsBinding",
    "RelaxedLinearBinding",
    "RelaxedLinearRowSource",
    "RelaxedRouterBinding",
    "StreamingRelaxedLinearStats",
    "expected_serialized_cost",
    "project_serialized_cost_gradient_",
    "retract_serialized_cost_",
    "streaming_relaxed_causal_cross_entropy",
    "streaming_relaxed_embedding",
    "streaming_relaxed_linear",
    "transport_serialized_cost_momentum_",
]
