"""Low-memory hard-assignment search primitives.

This module intentionally does not know how a model is loaded.  Its evaluator
receives one exact-budget candidate assignment at a time, which lets a caller
stream only the selected packed weights through a model runner.  No dense
candidate mixtures or autograd graph through model weights are required.

Two-choice searches may use unequal serialized costs. Small problems use a
sparse per-item dynamic program; production-scale problems with repeated tensor
geometries collapse into bounded cost classes while retaining the exact global
budget distribution and its score gradient.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
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


def _cost_vectors(
    low_costs: Sequence[int] | torch.Tensor,
    high_costs: Sequence[int] | torch.Tensor,
) -> tuple[list[int], list[int]]:
    low = [int(value) for value in low_costs]
    high = [int(value) for value in high_costs]
    if not low or len(low) != len(high):
        raise ValueError("cost vectors must be non-empty and have equal length")
    if any(value < 0 for value in low):
        raise ValueError("low costs must be non-negative")
    if any(high_value <= low_value for low_value, high_value in zip(low, high)):
        raise ValueError("every high cost must exceed its low cost")
    return low, high


def _normalized_cost_problem(
    low_costs: Sequence[int] | torch.Tensor,
    high_costs: Sequence[int] | torch.Tensor,
    target_cost: int,
) -> tuple[list[int], int, int]:
    low, high = _cost_vectors(low_costs, high_costs)
    base = sum(low)
    residual = int(target_cost) - base
    increments = [high_value - low_value
                  for low_value, high_value in zip(low, high)]
    divisor = math.gcd(*increments)
    if residual < 0 or residual > sum(increments) or residual % divisor:
        raise ValueError(
            f"target cost {target_cost} is not reachable from candidate costs")
    return [value // divisor for value in increments], residual // divisor, base


def _cost_classes(increments: Sequence[int]) -> list[tuple[int, list[int]]]:
    grouped: dict[int, list[int]] = {}
    for index, increment in enumerate(increments):
        grouped.setdefault(increment, []).append(index)
    return sorted(grouped.items(), reverse=True)


def _use_class_knapsack(increments: Sequence[int]) -> bool:
    """Select the scalable path only when repeated costs make it worthwhile."""
    return len(increments) >= 64 and len(set(increments)) <= 16


def _extend_class_dp(
    previous: np.ndarray,
    previous_gcd: int,
    weight: int,
    values: np.ndarray,
    target: int,
    *,
    maximize: bool,
) -> tuple[np.ndarray, int, np.ndarray | None]:
    """Add one bounded cost class to a compressed dense dynamic program."""
    new_gcd = weight if previous_gcd == 0 else math.gcd(previous_gcd, weight)
    previous_stride = 0 if previous_gcd == 0 else previous_gcd // new_gcd
    previous_max = 0 if previous_gcd == 0 else (
        (len(previous) - 1) * previous_gcd)
    maximum = min(target, previous_max + (len(values) - 1) * weight)
    result = np.full(maximum // new_gcd + 1, -np.inf, dtype=np.float64)
    parent = (
        np.full(len(result), -1, dtype=np.int16) if maximize else None)
    old_indices = (
        np.zeros(1, dtype=np.int64) if previous_gcd == 0 else
        np.arange(len(previous), dtype=np.int64) * previous_stride)
    for count, value in enumerate(values):
        offset = count * weight // new_gcd
        stop = np.searchsorted(old_indices, len(result) - offset)
        if stop == 0:
            break
        positions = old_indices[:stop] + offset
        candidate = previous[:stop] + value
        if maximize:
            assert parent is not None
            better = candidate > result[positions]
            result[positions[better]] = candidate[better]
            parent[positions[better]] = count
        else:
            result[positions] = np.logaddexp(result[positions], candidate)
    return result, new_gcd, parent


def _class_exact_cost_assignment(
    scores: torch.Tensor,
    increments: Sequence[int],
    target: int,
) -> torch.Tensor:
    """Exact max-score projection using bounded repeated-cost classes."""
    classes = _cost_classes(increments)
    detached = scores.detach().to(device="cpu", dtype=torch.float64).tolist()
    dp = np.array([0.0], dtype=np.float64)
    divisor = 0
    stages = []
    ranked_classes: list[list[int]] = []
    for weight, indices in classes:
        ranked = sorted(indices, key=lambda index: (-detached[index], index))
        values = np.zeros(len(ranked) + 1, dtype=np.float64)
        if ranked:
            values[1:] = np.cumsum([detached[index] for index in ranked])
        old_divisor = divisor
        dp, divisor, parent = _extend_class_dp(
            dp, divisor, weight, values, target, maximize=True)
        stages.append((old_divisor, divisor, weight, parent))
        ranked_classes.append(ranked)
    if target % divisor or target // divisor >= len(dp):
        raise ValueError("target cost has no exact candidate assignment")
    state = target
    result = torch.zeros_like(scores, dtype=torch.long)
    for stage_index in range(len(classes) - 1, -1, -1):
        old_divisor, divisor, weight, parent = stages[stage_index]
        assert parent is not None
        count = int(parent[state // divisor])
        if count < 0:
            raise ValueError("target cost has no exact candidate assignment")
        for index in ranked_classes[stage_index][:count]:
            result[index] = 1
        state -= count * weight
        if old_divisor and state % old_divisor:
            raise RuntimeError("class knapsack backtracking lost alignment")
    if state != 0:
        raise RuntimeError("class knapsack backtracking failed")
    return result


def _log_elementary_symmetric(scores: torch.Tensor) -> torch.Tensor:
    """Return log elementary-symmetric sums for every subset cardinality."""
    zero = scores.sum() * 0.0
    values = torch.stack((zero,))
    for score in scores:
        excluded = torch.cat((values, values.new_full((1,), -torch.inf)))
        included = torch.cat((values.new_full((1,), -torch.inf), values + score))
        values = torch.logaddexp(excluded, included)
    return values


def _lookup_class_dp(values: np.ndarray, divisor: int, cost: int) -> float:
    if cost < 0:
        return -math.inf
    if divisor == 0:
        return float(values[0]) if cost == 0 else -math.inf
    if cost % divisor:
        return -math.inf
    index = cost // divisor
    return float(values[index]) if index < len(values) else -math.inf


def _logsumexp_numpy(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if not math.isfinite(maximum):
        return -math.inf
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def _class_count_posterior(
    prefix: tuple[np.ndarray, int],
    suffix: tuple[np.ndarray, int],
    weight: int,
    class_values: np.ndarray,
    target: int,
    log_partition: float,
) -> np.ndarray:
    """Return P(class cardinality=k | exact total cost)."""
    prefix_values, prefix_gcd = prefix
    suffix_values, suffix_gcd = suffix
    if prefix_gcd == 0:
        prefix_costs = np.zeros(1, dtype=np.int64)
    else:
        prefix_costs = (
            np.arange(len(prefix_values), dtype=np.int64) * prefix_gcd)
    log_numerators = np.full(len(class_values), -np.inf, dtype=np.float64)
    for count, class_value in enumerate(class_values):
        remaining = target - prefix_costs - count * weight
        if suffix_gcd == 0:
            valid = remaining == 0
            suffix_indices = np.zeros(len(remaining), dtype=np.int64)
        else:
            valid = (remaining >= 0) & (remaining % suffix_gcd == 0)
            suffix_indices = np.where(valid, remaining // suffix_gcd, 0)
            valid &= suffix_indices < len(suffix_values)
        if not np.any(valid):
            continue
        terms = (
            prefix_values[valid]
            + suffix_values[suffix_indices[valid]]
            + class_value
        )
        log_numerators[count] = _logsumexp_numpy(terms)
    posterior = np.exp(log_numerators - log_partition)
    posterior /= posterior.sum()
    return posterior


def _sample_class_subset(
    scores: np.ndarray,
    count: int,
    uniforms: np.ndarray,
) -> np.ndarray:
    """Sample a product-weighted fixed-cardinality subset exactly."""
    size = len(scores)
    result = np.zeros(size, dtype=np.int64)
    if count == 0:
        return result
    if count == size:
        result.fill(1)
        return result
    suffix = np.full((size + 1, count + 1), -np.inf, dtype=np.float64)
    suffix[size, 0] = 0.0
    for index in range(size - 1, -1, -1):
        suffix[index, 0] = 0.0
        maximum = min(count, size - index)
        suffix[index, 1:maximum + 1] = np.logaddexp(
            suffix[index + 1, 1:maximum + 1],
            scores[index] + suffix[index + 1, :maximum],
        )
    remaining = count
    for index in range(size - 1):
        if remaining == 0:
            break
        if remaining == size - index:
            result[index:] = 1
            remaining = 0
            break
        log_high = scores[index] + suffix[index + 1, remaining - 1]
        probability = math.exp(log_high - suffix[index, remaining])
        if uniforms[index] < probability:
            result[index] = 1
            remaining -= 1
    if remaining:
        result[-1] = 1
        remaining -= 1
    if remaining:
        raise RuntimeError("fixed-cardinality subset sampler failed")
    return result


def _sample_class_exact_cost_assignment(
    scores: torch.Tensor,
    increments: Sequence[int],
    target: int,
    uniforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale exact Gibbs sampling through repeated serialized-cost classes."""
    classes = _cost_classes(increments)
    class_log_values = [
        _log_elementary_symmetric(scores[indices])
        for _, indices in classes
    ]
    detached_values = [
        values.detach().to(device="cpu", dtype=torch.float64).numpy()
        for values in class_log_values
    ]

    suffix: list[tuple[np.ndarray, int]] = [
        (np.array([0.0]), 0) for _ in range(len(classes) + 1)]
    for class_index in range(len(classes) - 1, -1, -1):
        weight, _ = classes[class_index]
        next_values, next_gcd = suffix[class_index + 1]
        values, divisor, _ = _extend_class_dp(
            next_values,
            next_gcd,
            weight,
            detached_values[class_index],
            target,
            maximize=False,
        )
        suffix[class_index] = (values, divisor)
    log_partition = _lookup_class_dp(*suffix[0], target)
    if not math.isfinite(log_partition):
        raise ValueError("target cost has no exact candidate assignment")

    prefix: list[tuple[np.ndarray, int]] = [
        (np.array([0.0]), 0)]
    for class_index, (weight, _) in enumerate(classes):
        previous_values, previous_gcd = prefix[-1]
        values, divisor, _ = _extend_class_dp(
            previous_values,
            previous_gcd,
            weight,
            detached_values[class_index],
            target,
            maximize=False,
        )
        prefix.append((values, divisor))

    uniform_values = uniforms.detach().to(
        device="cpu", dtype=torch.float64).numpy()
    remaining = target
    counts: list[int] = []
    for class_index, (weight, _) in enumerate(classes):
        conditional = np.full(
            len(detached_values[class_index]), -np.inf, dtype=np.float64)
        for count, class_value in enumerate(detached_values[class_index]):
            rest = _lookup_class_dp(
                *suffix[class_index + 1], remaining - count * weight)
            conditional[count] = class_value + rest
        normalization = _logsumexp_numpy(conditional)
        probabilities = np.exp(conditional - normalization)
        cumulative = np.cumsum(probabilities)
        count = min(
            int(np.searchsorted(
                cumulative, uniform_values[class_index], side="right")),
            len(probabilities) - 1,
        )
        counts.append(count)
        remaining -= count * weight
    if remaining != 0:
        raise RuntimeError("class-count sampler did not meet its target")

    assignment = torch.zeros_like(scores, dtype=torch.long)
    cursor = len(classes)
    detached_scores = scores.detach().to(
        device="cpu", dtype=torch.float64).numpy()
    for (_, indices), count in zip(classes, counts):
        class_uniforms = uniform_values[cursor:cursor + len(indices) - 1]
        cursor += len(indices) - 1
        chosen = _sample_class_subset(
            detached_scores[indices], count, class_uniforms)
        for local_index, choice in enumerate(chosen):
            if choice:
                assignment[indices[local_index]] = 1
    if cursor != scores.numel():
        raise RuntimeError("class sampler did not consume its uniform draws")

    log_z = scores.sum() * 0.0 + log_partition
    for class_index, (weight, _) in enumerate(classes):
        posterior = _class_count_posterior(
            prefix[class_index],
            suffix[class_index + 1],
            weight,
            detached_values[class_index],
            target,
            log_partition,
        )
        probability = torch.from_numpy(posterior).to(
            device=scores.device, dtype=scores.dtype)
        values = class_log_values[class_index]
        log_z = log_z + torch.dot(
            probability, values - values.detach())
    log_probability = (
        scores[assignment.to(dtype=torch.bool)].sum() - log_z)
    return assignment, log_probability


def realized_cost(
    assignment: torch.Tensor,
    low_costs: Sequence[int] | torch.Tensor,
    high_costs: Sequence[int] | torch.Tensor,
) -> int:
    """Return the exact serialized cost of a binary assignment."""
    low, high = _cost_vectors(low_costs, high_costs)
    if assignment.shape != (len(low),):
        raise ValueError(
            f"assignment has shape {assignment.shape}, expected ({len(low)},)")
    choices = assignment.detach().to(device="cpu", dtype=torch.long).tolist()
    if any(choice not in (0, 1) for choice in choices):
        raise ValueError("assignment choices must be 0 or 1")
    return sum((high[index] if choice else low[index])
               for index, choice in enumerate(choices))


def exact_cost_assignment(
    scores: torch.Tensor,
    low_costs: Sequence[int] | torch.Tensor,
    high_costs: Sequence[int] | torch.Tensor,
    target_cost: int,
) -> torch.Tensor:
    """Maximize high-choice scores under an exact serialized-byte budget."""
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    increments, target, _ = _normalized_cost_problem(
        low_costs, high_costs, target_cost)
    if len(increments) != scores.numel():
        raise ValueError("cost vectors and scores must have equal length")
    if _use_class_knapsack(increments):
        return _class_exact_cost_assignment(scores, increments, target)

    # A sparse multiple-choice knapsack keeps only exactly reachable costs.
    # Parent maps are retained for deterministic backtracking. Equal-score
    # ties preserve the low choice, so results are stable across runs.
    states = {0: 0.0}
    parents: list[dict[int, int]] = []
    detached = scores.detach().to(device="cpu", dtype=torch.float64)
    for index, increment in enumerate(increments):
        next_states = dict(states)
        choices = {cost: 0 for cost in states}
        advantage = float(detached[index])
        for cost, value in states.items():
            new_cost = cost + increment
            if new_cost > target:
                continue
            candidate = value + advantage
            if new_cost not in next_states or candidate > next_states[new_cost]:
                next_states[new_cost] = candidate
                choices[new_cost] = 1
        states = next_states
        parents.append(choices)
    if target not in states:
        raise ValueError(
            f"target cost {target_cost} has no exact candidate assignment")

    result = torch.zeros_like(scores, dtype=torch.long)
    remaining = target
    for index in range(scores.numel() - 1, -1, -1):
        choice = parents[index][remaining]
        result[index] = choice
        if choice:
            remaining -= increments[index]
    if remaining != 0:
        raise RuntimeError("exact-cost assignment backtracking failed")
    return result


def sample_exact_cost_assignment(
    scores: torch.Tensor,
    low_costs: Sequence[int] | torch.Tensor,
    high_costs: Sequence[int] | torch.Tensor,
    target_cost: int,
    *,
    uniforms: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample an exact-cost assignment and its differentiable log probability.

    The distribution is proportional to ``exp(sum(scores[high]))`` over only
    the assignments that exactly meet ``target_cost``. A suffix log-partition
    dynamic program provides exact sequential conditionals without enumerating
    candidate combinations.
    """
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    increments, target, _ = _normalized_cost_problem(
        low_costs, high_costs, target_cost)
    if len(increments) != scores.numel():
        raise ValueError("cost vectors and scores must have equal length")
    if uniforms is None:
        uniforms = torch.rand_like(scores)
    elif uniforms.shape != scores.shape:
        raise ValueError(
            f"uniforms has shape {uniforms.shape}, expected {scores.shape}")
    uniforms = uniforms.detach().to(device=scores.device, dtype=scores.dtype)
    if _use_class_knapsack(increments):
        return _sample_class_exact_cost_assignment(
            scores, increments, target, uniforms)

    zero = scores.sum() * 0.0
    suffix: list[dict[int, torch.Tensor]] = [dict() for _ in range(
        scores.numel() + 1)]
    suffix[-1] = {0: zero}
    for index in range(scores.numel() - 1, -1, -1):
        current: dict[int, torch.Tensor] = {}
        for cost, log_weight in suffix[index + 1].items():
            if cost in current:
                current[cost] = torch.logaddexp(current[cost], log_weight)
            else:
                current[cost] = log_weight
            high_cost = cost + increments[index]
            if high_cost <= target:
                high_weight = log_weight + scores[index]
                if high_cost in current:
                    current[high_cost] = torch.logaddexp(
                        current[high_cost], high_weight)
                else:
                    current[high_cost] = high_weight
        suffix[index] = current
    if target not in suffix[0]:
        raise ValueError(
            f"target cost {target_cost} has no exact candidate assignment")

    assignment = torch.zeros_like(scores, dtype=torch.long)
    log_probability = zero
    remaining = target
    for index, increment in enumerate(increments):
        low_weight = suffix[index + 1].get(remaining)
        high_suffix = suffix[index + 1].get(remaining - increment)
        high_weight = (
            None if high_suffix is None else scores[index] + high_suffix)
        if low_weight is None:
            choose_high = True
            total = high_weight
        elif high_weight is None:
            choose_high = False
            total = low_weight
        else:
            total = torch.logaddexp(low_weight, high_weight)
            probability_high = torch.exp(high_weight - total).detach()
            choose_high = bool(uniforms[index] < probability_high)
        chosen_weight = high_weight if choose_high else low_weight
        if chosen_weight is None or total is None:
            raise RuntimeError("exact-cost sampler reached an infeasible state")
        log_probability = log_probability + chosen_weight - total
        if choose_high:
            assignment[index] = 1
            remaining -= increment
    if remaining != 0:
        raise RuntimeError("exact-cost sampler did not meet its target")
    return assignment, log_probability


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


def sample_plackett_luce_assignment(
    scores: torch.Tensor,
    n_high: int,
    *,
    uniforms: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample an exact-cardinality assignment and return its log probability.

    Gumbel top-k samples an ordered Plackett-Luce draw without replacement.
    The evaluator only consumes the resulting unordered assignment, while the
    ordered draw supplies an exact, differentiable log probability for the
    REINFORCE estimator. Computation and persistent state are O(n_groups).
    """
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not 0 <= n_high <= scores.numel():
        raise ValueError("n_high is outside the assignment size")
    if uniforms is None:
        uniforms = torch.rand_like(scores)
    elif uniforms.shape != scores.shape:
        raise ValueError(
            f"uniforms has shape {uniforms.shape}, expected {scores.shape}")
    tiny = torch.finfo(scores.dtype).eps
    uniforms = uniforms.to(device=scores.device, dtype=scores.dtype).clamp(
        tiny, 1.0 - tiny)
    gumbels = -torch.log(-torch.log(uniforms))
    assignment = torch.zeros_like(scores, dtype=torch.long)
    if n_high == 0:
        return assignment, scores.sum() * 0.0

    selected = torch.topk(
        scores.detach() + gumbels, n_high, sorted=True).indices
    assignment[selected] = 1

    # For the sampled order, P(i_t) = exp(score_i_t) / sum_remaining exp(score).
    # A detached common shift keeps exponentials finite without changing the
    # probability or its gradient. Cumulative subtraction avoids an O(N*K)
    # mask or dynamic candidate tensor.
    shift = scores.detach().max()
    weights = torch.exp(scores - shift)
    selected_weights = weights[selected]
    removed_before = torch.cat((
        torch.zeros(1, dtype=weights.dtype, device=weights.device),
        torch.cumsum(selected_weights[:-1], dim=0),
    ))
    denominators = (weights.sum() - removed_before).clamp_min(
        torch.finfo(weights.dtype).tiny)
    log_probability = (
        scores[selected] - shift - torch.log(denominators)).sum()
    return assignment, log_probability


def optimize_hard_reinforce(
    evaluate: Callable[[torch.Tensor], float],
    *,
    n_groups: int,
    low_bits: float,
    high_bits: float,
    target_bits: float,
    n_steps: int = 100,
    lr: float = 0.05,
    baseline_decay: float = 0.9,
    seed: int = 42,
    initial_scores: Optional[torch.Tensor] = None,
    log_interval: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Optimize exact-budget choices with paired antithetic REINFORCE draws."""
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if not 0.0 <= baseline_decay < 1.0:
        raise ValueError("baseline_decay must lie in [0, 1)")
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
    baseline = None
    history = []

    for step in range(n_steps):
        uniforms = torch.rand(
            scores.shape, generator=generator, device=scores.device,
            dtype=scores.dtype)
        plus, log_probability_plus = sample_plackett_luce_assignment(
            scores, n_high, uniforms=uniforms)
        minus, log_probability_minus = sample_plackett_luce_assignment(
            scores, n_high, uniforms=1.0 - uniforms)
        loss_plus = float(evaluate(plus))
        loss_minus = float(evaluate(minus))
        if not torch.isfinite(torch.tensor([loss_plus, loss_minus])).all():
            raise FloatingPointError(
                f"non-finite evaluator loss at step {step}: "
                f"{loss_plus}, {loss_minus}")

        pair_mean = 0.5 * (loss_plus + loss_minus)
        if baseline is None:
            baseline = pair_mean
        objective = 0.5 * (
            (loss_plus - baseline) * log_probability_plus
            + (loss_minus - baseline) * log_probability_minus
        )
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradient_norm = scores.grad.norm().item()
        optimizer.step()
        baseline = (
            baseline_decay * baseline + (1.0 - baseline_decay) * pair_mean)

        chosen = plus if loss_plus <= loss_minus else minus
        entry = {
            "step": step,
            "loss_plus": loss_plus,
            "loss_minus": loss_minus,
            "best_loss": min(loss_plus, loss_minus),
            "baseline": baseline,
            "gradient_norm": gradient_norm,
            "n_high": int(chosen.sum().item()),
            "realized_bits": realized_average_bits(
                chosen, low_bits, high_bits),
        }
        history.append(entry)
        if step % log_interval == 0 or step == n_steps - 1:
            logger.info(
                "[Hard REINFORCE %d] loss+=%.6f loss-=%.6f "
                "baseline=%.6f grad=%.5f bits=%.4f",
                step, loss_plus, loss_minus, baseline, gradient_norm,
                entry["realized_bits"],
            )

    final_assignment = exact_budget_assignment(scores.detach(), n_high)
    return scores.detach(), final_assignment, history


def optimize_cost_spsa(
    evaluate: Callable[[torch.Tensor], float],
    *,
    low_costs: Sequence[int] | torch.Tensor,
    high_costs: Sequence[int] | torch.Tensor,
    target_cost: int,
    n_steps: int = 100,
    lr: float = 0.05,
    perturbation: float = 0.1,
    seed: int = 42,
    initial_scores: Optional[torch.Tensor] = None,
    log_interval: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Optimize hard choices at an exact, nonuniform serialized-byte cost."""
    low, high = _cost_vectors(low_costs, high_costs)
    n_groups = len(low)
    # Fail before invoking an expensive evaluator if the target is impossible.
    exact_cost_assignment(torch.zeros(n_groups), low, high, target_cost)
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if perturbation <= 0:
        raise ValueError("perturbation must be positive")
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
    incumbent_loss = float("inf")
    incumbent_assignment = None

    for step in range(n_steps):
        delta = torch.empty_like(scores).bernoulli_(0.5, generator=generator)
        delta.mul_(2).sub_(1)
        plus = exact_cost_assignment(
            scores.detach() + perturbation * delta,
            low, high, target_cost)
        minus = exact_cost_assignment(
            scores.detach() - perturbation * delta,
            low, high, target_cost)
        loss_plus = float(evaluate(plus))
        loss_minus = float(evaluate(minus))
        if not torch.isfinite(torch.tensor([loss_plus, loss_minus])).all():
            raise FloatingPointError(
                f"non-finite evaluator loss at step {step}: "
                f"{loss_plus}, {loss_minus}")
        gradient = ((loss_plus - loss_minus) / (2.0 * perturbation)) * delta
        optimizer.zero_grad(set_to_none=True)
        scores.grad = gradient
        optimizer.step()
        chosen = plus if loss_plus <= loss_minus else minus
        chosen_loss = min(loss_plus, loss_minus)
        if chosen_loss < incumbent_loss:
            incumbent_loss = chosen_loss
            incumbent_assignment = chosen.clone()
        entry = {
            "step": step,
            "loss_plus": loss_plus,
            "loss_minus": loss_minus,
            "best_loss": min(loss_plus, loss_minus),
            "incumbent_loss": incumbent_loss,
            "gradient_norm": gradient.norm().item(),
            "n_high": int(chosen.sum().item()),
            "realized_cost": realized_cost(chosen, low, high),
        }
        history.append(entry)
        if step % log_interval == 0 or step == n_steps - 1:
            logger.info(
                "[Cost SPSA %d] loss+=%.6f loss-=%.6f grad=%.5f cost=%d",
                step, loss_plus, loss_minus, entry["gradient_norm"],
                entry["realized_cost"],
            )
    if incumbent_assignment is None:
        raise RuntimeError("cost SPSA completed without an evaluated assignment")
    return scores.detach(), incumbent_assignment, history


def optimize_cost_reinforce(
    evaluate: Callable[[torch.Tensor], float],
    *,
    low_costs: Sequence[int] | torch.Tensor,
    high_costs: Sequence[int] | torch.Tensor,
    target_cost: int,
    n_steps: int = 100,
    lr: float = 0.05,
    baseline_decay: float = 0.9,
    seed: int = 42,
    initial_scores: Optional[torch.Tensor] = None,
    initial_incumbent_assignment: Optional[torch.Tensor] = None,
    initial_incumbent_loss: Optional[float] = None,
    initial_step: int = 0,
    initial_baseline: Optional[float] = None,
    initial_optimizer_state: Optional[Mapping[str, Any]] = None,
    step_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    log_interval: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Optimize exact-cost choices with paired antithetic REINFORCE draws.

    ``initial_*`` state and ``step_callback`` form a fail-closed restart
    boundary.  The callback runs after each completed optimizer step and
    receives detached tensors suitable for durable serialization.  Restoring
    all of that state reproduces the uninterrupted trajectory exactly.
    """
    low, high = _cost_vectors(low_costs, high_costs)
    n_groups = len(low)
    exact_cost_assignment(torch.zeros(n_groups), low, high, target_cost)
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if not 0.0 <= baseline_decay < 1.0:
        raise ValueError("baseline_decay must lie in [0, 1)")
    if initial_step < 0:
        raise ValueError("initial_step must be nonnegative")
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
    for _ in range(initial_step):
        # Sampling consumes exactly one tensor per completed step. Advancing
        # here avoids storing backend-specific generator bytes in JSON.
        torch.rand(
            scores.shape, generator=generator, device=scores.device,
            dtype=scores.dtype)
    baseline = initial_baseline
    history = []
    if ((initial_incumbent_assignment is None)
            != (initial_incumbent_loss is None)):
        raise ValueError(
            "initial incumbent assignment and loss must be provided together")
    if initial_incumbent_assignment is None:
        incumbent_loss = float("inf")
        incumbent_assignment = None
    else:
        if initial_incumbent_assignment.shape != (n_groups,):
            raise ValueError(
                "initial incumbent assignment has the wrong shape")
        if realized_cost(
            initial_incumbent_assignment, low, high,
        ) != int(target_cost):
            raise ValueError("initial incumbent does not meet the exact budget")
        incumbent_loss = float(initial_incumbent_loss)
        if not math.isfinite(incumbent_loss):
            raise ValueError("initial incumbent loss must be finite")
        incumbent_assignment = initial_incumbent_assignment.detach().clone()

    if initial_step:
        if initial_scores is None or initial_baseline is None:
            raise ValueError(
                "resuming requires initial scores and moving baseline")
        if incumbent_assignment is None or initial_optimizer_state is None:
            raise ValueError(
                "resuming requires incumbent and Adam optimizer state")
        exp_avg = torch.as_tensor(
            initial_optimizer_state.get("exp_avg"), dtype=scores.dtype,
            device=scores.device)
        exp_avg_sq = torch.as_tensor(
            initial_optimizer_state.get("exp_avg_sq"), dtype=scores.dtype,
            device=scores.device)
        if exp_avg.shape != scores.shape or exp_avg_sq.shape != scores.shape:
            raise ValueError("initial Adam moment shape differs from scores")
        optimizer_step = int(initial_optimizer_state.get("step", -1))
        if optimizer_step != initial_step:
            raise ValueError("initial Adam step differs from initial_step")
        optimizer.state[scores] = {
            "step": torch.tensor(float(optimizer_step), device=scores.device),
            "exp_avg": exp_avg.clone(),
            "exp_avg_sq": exp_avg_sq.clone(),
        }
    elif initial_optimizer_state is not None:
        raise ValueError("initial optimizer state requires a positive step")

    for local_step in range(n_steps):
        step = initial_step + local_step
        uniforms = torch.rand(
            scores.shape, generator=generator, device=scores.device,
            dtype=scores.dtype)
        plus, log_probability_plus = sample_exact_cost_assignment(
            scores, low, high, target_cost, uniforms=uniforms)
        minus, log_probability_minus = sample_exact_cost_assignment(
            scores, low, high, target_cost, uniforms=1.0 - uniforms)
        loss_plus = float(evaluate(plus))
        loss_minus = float(evaluate(minus))
        if not torch.isfinite(torch.tensor([loss_plus, loss_minus])).all():
            raise FloatingPointError(
                f"non-finite evaluator loss at step {step}: "
                f"{loss_plus}, {loss_minus}")
        pair_mean = 0.5 * (loss_plus + loss_minus)
        if baseline is None:
            baseline = pair_mean
        objective = 0.5 * (
            (loss_plus - baseline) * log_probability_plus
            + (loss_minus - baseline) * log_probability_minus)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradient_norm = scores.grad.norm().item()
        optimizer.step()
        baseline = (
            baseline_decay * baseline + (1.0 - baseline_decay) * pair_mean)
        chosen = plus if loss_plus <= loss_minus else minus
        chosen_loss = min(loss_plus, loss_minus)
        if chosen_loss < incumbent_loss:
            incumbent_loss = chosen_loss
            incumbent_assignment = chosen.clone()
        entry = {
            "step": step,
            "loss_plus": loss_plus,
            "loss_minus": loss_minus,
            "best_loss": min(loss_plus, loss_minus),
            "incumbent_loss": incumbent_loss,
            "baseline": baseline,
            "gradient_norm": gradient_norm,
            "n_high": int(chosen.sum().item()),
            "realized_cost": realized_cost(chosen, low, high),
        }
        history.append(entry)
        if step_callback is not None:
            adam_state = optimizer.state[scores]
            step_callback({
                "completed_steps": step + 1,
                "scores": scores.detach().clone(),
                "baseline": float(baseline),
                "incumbent_assignment": incumbent_assignment.detach().clone(),
                "incumbent_loss": float(incumbent_loss),
                "optimizer_state": {
                    "step": int(adam_state["step"].item()),
                    "exp_avg": adam_state["exp_avg"].detach().clone(),
                    "exp_avg_sq": adam_state["exp_avg_sq"].detach().clone(),
                },
                "history_entry": dict(entry),
            })
        if step % log_interval == 0 or local_step == n_steps - 1:
            logger.info(
                "[Cost REINFORCE %d] loss+=%.6f loss-=%.6f "
                "baseline=%.6f grad=%.5f cost=%d",
                step, loss_plus, loss_minus, baseline, gradient_norm,
                entry["realized_cost"],
            )
    if incumbent_assignment is None:
        raise RuntimeError(
            "cost REINFORCE completed without an evaluated assignment")
    return scores.detach(), incumbent_assignment, history


__all__ = [
    "HardCandidateModel",
    "exact_budget_assignment",
    "exact_cost_assignment",
    "high_choice_count",
    "optimize_cost_reinforce",
    "optimize_cost_spsa",
    "optimize_hard_spsa",
    "optimize_hard_reinforce",
    "realized_average_bits",
    "realized_cost",
    "sample_exact_cost_assignment",
    "sample_plackett_luce_assignment",
]
