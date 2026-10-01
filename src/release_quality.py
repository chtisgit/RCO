"""Statistics for document-paired release-quality evaluation."""

from __future__ import annotations

import math
import random
from typing import Sequence


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
