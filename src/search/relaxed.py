"""Bounded-memory relaxed interpolation for one linear weight.

The released relaxed RCO path retains every dense candidate delta for the
duration of the search.  This module keeps the same interpolation and softmax
gradient, but delegates reference and alternative-delta reads to a row source.
Only one decoded row chunk is materialized at a time.  Backward saves the
linear input and probabilities, then rereads the reference and alternatives;
it never saves a candidate delta in the autograd context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import torch
import torch.nn.functional as F


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


__all__ = [
    "DenseDeltaRowSource",
    "RelaxedLinearRowSource",
    "StreamingRelaxedLinearStats",
    "streaming_relaxed_linear",
]
