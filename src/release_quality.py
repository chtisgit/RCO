"""Statistics for document-paired release-quality evaluation."""

from __future__ import annotations

import math
import random
from typing import Sequence


def partial_perplexity_ratio_certificate(
    *,
    baseline_mean_nll: float,
    observed_mean_nll: Sequence[float],
    observed_token_counts: Sequence[int],
    total_token_count: int,
    maximum_ratio: float,
) -> dict[str, float | int | bool | str]:
    """Certify failure when even zero loss on all unseen tokens cannot pass."""
    if not math.isfinite(baseline_mean_nll) or baseline_mean_nll < 0:
        raise ValueError("baseline mean NLL must be finite and nonnegative")
    if len(observed_mean_nll) != len(observed_token_counts):
        raise ValueError("observed NLL and token-count vectors must align")
    if not observed_mean_nll:
        raise ValueError("at least one observed result is required")
    if any(not math.isfinite(value) or value < 0 for value in observed_mean_nll):
        raise ValueError("observed NLLs must be finite and nonnegative")
    if any(count <= 0 for count in observed_token_counts):
        raise ValueError("observed token counts must be positive")
    observed_token_count = sum(observed_token_counts)
    if total_token_count < observed_token_count or total_token_count <= 0:
        raise ValueError("total token count must cover all observed tokens")
    if not math.isfinite(maximum_ratio) or maximum_ratio <= 0:
        raise ValueError("maximum perplexity ratio must be positive and finite")

    observed_nll_sum = math.fsum(
        value * count
        for value, count in zip(
            observed_mean_nll, observed_token_counts, strict=True)
    )
    candidate_mean_nll_lower_bound = observed_nll_sum / total_token_count
    ratio_lower_bound = math.exp(
        candidate_mean_nll_lower_bound - baseline_mean_nll)
    maximum_candidate_mean_nll = baseline_mean_nll + math.log(maximum_ratio)
    return {
        "assumption_for_unseen_tokens": "zero_nll_best_case",
        "observed_token_count": observed_token_count,
        "total_token_count": total_token_count,
        "observed_nll_sum": observed_nll_sum,
        "baseline_mean_nll": baseline_mean_nll,
        "maximum_perplexity_ratio": maximum_ratio,
        "maximum_candidate_mean_nll": maximum_candidate_mean_nll,
        "candidate_mean_nll_lower_bound": candidate_mean_nll_lower_bound,
        "candidate_perplexity_ratio_lower_bound": ratio_lower_bound,
        "failure_proven": ratio_lower_bound > maximum_ratio,
    }


def weighted_mean_nll(
    document_mean_nll: Sequence[float],
    document_token_counts: Sequence[int],
) -> float:
    if len(document_mean_nll) != len(document_token_counts) or not document_mean_nll:
        raise ValueError("document NLL and token-count vectors must align")
    if any(count <= 0 for count in document_token_counts):
        raise ValueError("every document must contain predicted tokens")
    if any(not math.isfinite(value) for value in document_mean_nll):
        raise ValueError("document NLLs must be finite")
    total = sum(document_token_counts)
    return math.fsum(
        value * count
        for value, count in zip(
            document_mean_nll, document_token_counts, strict=True)
    ) / total


def paired_bootstrap_mean_ci(
    deltas: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 20261001,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if not deltas or any(not math.isfinite(value) for value in deltas):
        raise ValueError("paired deltas must be nonempty and finite")
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    generator = random.Random(seed)
    count = len(deltas)
    means = []
    for _ in range(samples):
        means.append(math.fsum(
            deltas[generator.randrange(count)] for _ in range(count)
        ) / count)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    lower_index = max(0, math.floor(tail * samples))
    upper_index = min(samples - 1, math.ceil((1.0 - tail) * samples) - 1)
    return means[lower_index], means[upper_index]
