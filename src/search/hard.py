"""Low-memory hard-assignment search primitives.

This module intentionally does not know how a model is loaded.  Its evaluator
receives one exact-budget candidate assignment at a time, which lets a caller
stream only the selected packed weights through a model runner.  No dense
candidate mixtures or autograd graph through model weights are required.

The first implementation supports two bitwidths and equal-size groups.  That is
the routed-expert use case: each group contains the same gate/up/down shapes and
therefore has the same parameter count.  Supporting unequal groups requires a
fast constrained sampler rather than silently approximating the budget.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

from common import get_layer_weights, set_layer_weights

logger = logging.getLogger(__name__)


class HardCandidateModel:
    """Apply one packed candidate per group without retaining alternatives."""

    def __init__(self, model, weight_store, groups, bitwidths):
        if len(bitwidths) != 2:
            raise ValueError("HardCandidateModel currently requires two bitwidths")
        self.model = model
        self.weight_store = weight_store
        self.groups = groups
        self.bitwidths = tuple(sorted(int(bits) for bits in bitwidths))
        self.model.requires_grad_(False)

    @torch.no_grad()
    def apply(self, assignment: torch.Tensor) -> None:
        if assignment.shape != (len(self.groups),):
            raise ValueError(
                f"assignment has shape {assignment.shape}, expected "
                f"({len(self.groups)},)")
        for group_index, group in enumerate(self.groups):
            choice = int(assignment[group_index])
            if choice not in (0, 1):
                raise ValueError(f"assignment choice must be 0 or 1, got {choice}")
            bitwidth = self.bitwidths[choice]
            for name in group.layer_names:
                if bitwidth == 0:
                    current = get_layer_weights(self.model, name)
                    if current is None:
                        raise ValueError(f"Cannot resolve weight for {name!r}")
                    weight = torch.zeros_like(current, device="cpu")
                else:
                    weight = self.weight_store.get_layer_weight(name, bitwidth)
                set_layer_weights(self.model, name, weight)

    def layer_assignment(self, assignment: torch.Tensor) -> dict[str, int]:
        result = {}
        for group_index, group in enumerate(self.groups):
            bitwidth = self.bitwidths[int(assignment[group_index])]
            for name in group.layer_names:
                result[name] = bitwidth
        return result


def high_choice_count(n_groups: int, low_bits: float, high_bits: float,
                      target_bits: float, tolerance: float = 1e-6) -> int:
    """Return the exact number of high-bit groups required by the target."""
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")
    if not low_bits < high_bits:
        raise ValueError("low_bits must be smaller than high_bits")
    if not low_bits <= target_bits <= high_bits:
        raise ValueError("target_bits must lie between low_bits and high_bits")
    exact = n_groups * (target_bits - low_bits) / (high_bits - low_bits)
    rounded = round(exact)
    if abs(exact - rounded) > tolerance:
        raise ValueError(
            f"target {target_bits} cannot be realized exactly by {n_groups} "
            f"equal groups with choices {low_bits} and {high_bits}; "
            f"it requires {exact:.6f} high-bit groups"
        )
    return int(rounded)


def exact_budget_assignment(scores: torch.Tensor, n_high: int) -> torch.Tensor:
    """Select exactly ``n_high`` groups, returning 0=low and 1=high."""
    if scores.ndim != 1:
        raise ValueError(f"scores must be one-dimensional, got {scores.shape}")
    if not 0 <= n_high <= scores.numel():
        raise ValueError("n_high is outside the assignment size")
    result = torch.zeros_like(scores, dtype=torch.long)
    if n_high:
        indices = scores.topk(n_high, sorted=False).indices
        result[indices] = 1
    return result


def realized_average_bits(assignment: torch.Tensor, low_bits: float,
                          high_bits: float) -> float:
    choices = torch.tensor(
        [low_bits, high_bits], dtype=torch.float64,
        device=assignment.device)
    return choices[assignment.long()].mean().item()


def optimize_hard_spsa(
    evaluate: Callable[[torch.Tensor], float],
    *,
    n_groups: int,
    low_bits: float,
    high_bits: float,
    target_bits: float,
    n_steps: int = 100,
    lr: float = 0.05,
    perturbation: float = 0.1,
    seed: int = 42,
    initial_scores: Optional[torch.Tensor] = None,
    log_interval: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Optimize exact-budget hard choices with paired SPSA evaluations.

    ``evaluate`` may stream a complete model and returns a scalar loss.  It is
    called twice per step and never participates in autograd.  Only the
    ``n_groups`` score vector and Adam state remain resident.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if perturbation <= 0:
        raise ValueError("perturbation must be positive")
    n_high = high_choice_count(
        n_groups, low_bits, high_bits, target_bits)

    if initial_scores is None:
        scores = torch.zeros(n_groups, dtype=torch.float32)
    else:
        if initial_scores.shape != (n_groups,):
            raise ValueError(
                f"initial_scores has shape {initial_scores.shape}, expected "
                f"({n_groups},)")
        scores = initial_scores.detach().to(dtype=torch.float32).clone()
    scores.requires_grad_(True)
    optimizer = torch.optim.Adam([scores], lr=lr)
    generator = torch.Generator(device=scores.device)
    generator.manual_seed(seed)
    history = []

    for step in range(n_steps):
        delta = torch.empty_like(scores).bernoulli_(0.5, generator=generator)
        delta.mul_(2).sub_(1)
        plus = exact_budget_assignment(
            scores.detach() + perturbation * delta, n_high)
        minus = exact_budget_assignment(
            scores.detach() - perturbation * delta, n_high)

        loss_plus = float(evaluate(plus))
        loss_minus = float(evaluate(minus))
        if not torch.isfinite(torch.tensor([loss_plus, loss_minus])).all():
            raise FloatingPointError(
                f"non-finite evaluator loss at step {step}: "
                f"{loss_plus}, {loss_minus}")

        # SPSA estimates d(loss)/d(score) from a paired directional probe.
        gradient = ((loss_plus - loss_minus) / (2.0 * perturbation)) * delta
        optimizer.zero_grad(set_to_none=True)
        scores.grad = gradient
        optimizer.step()

        chosen = plus if loss_plus <= loss_minus else minus
        entry = {
            "step": step,
            "loss_plus": loss_plus,
            "loss_minus": loss_minus,
            "best_loss": min(loss_plus, loss_minus),
            "gradient_norm": gradient.norm().item(),
            "n_high": int(chosen.sum().item()),
            "realized_bits": realized_average_bits(
                chosen, low_bits, high_bits),
        }
        history.append(entry)
        if step % log_interval == 0 or step == n_steps - 1:
            logger.info(
                "[Hard SPSA %d] loss+=%.6f loss-=%.6f grad=%.5f bits=%.4f",
                step, loss_plus, loss_minus, entry["gradient_norm"],
                entry["realized_bits"],
            )

    final_assignment = exact_budget_assignment(scores.detach(), n_high)
    return scores.detach(), final_assignment, history


__all__ = [
    "HardCandidateModel",
    "exact_budget_assignment",
    "high_choice_count",
    "optimize_hard_spsa",
    "realized_average_bits",
]
