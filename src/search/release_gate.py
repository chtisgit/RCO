"""Release-quality policy for a production Qwen native assignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class ReleasePolicy:
    """Versioned minimum evidence required before final GGUF construction."""

    minimum_documents: int = 100
    minimum_predicted_tokens: int = 100_000
    minimum_corpus_strata: int = 4
    maximum_perplexity_ratio_to_bf16: float = 1.15
    maximum_paired_nll_delta_ci95_upper: float = 0.0
    maximum_repeat_nll_delta: float = 1e-6
    minimum_generation_prompts: int = 32
    maximum_cuda_reserved_bytes: int = 10 << 30


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _integer(value: Any, default: int = -1) -> int:
    return default if value is None else int(value)


def _digest(value: Any, length: int = 64) -> bool:
    return isinstance(value, str) and len(value) == length


def evaluate_release_gate(
    evidence: Mapping[str, Any],
    policy: ReleasePolicy = ReleasePolicy(),
) -> dict[str, Any]:
    """Evaluate complete measured evidence without permitting silent waivers."""
    candidate = evidence["candidate"]
    corpus = evidence["heldout_corpus"]
    perplexity = evidence["perplexity"]
    generation = evidence["generation"]
    runtime = evidence["runtime"]

    candidate_nll = _finite_float(perplexity.get("candidate_mean_nll"))
    bf16_nll = _finite_float(perplexity.get("bf16_mean_nll"))
    repeat_nll = _finite_float(perplexity.get("repeat_candidate_mean_nll"))
    paired_ci_upper = _finite_float(
        perplexity.get("paired_candidate_minus_incumbent_ci95_upper"))
    perplexity_ratio_lower_bound = _finite_float(
        perplexity.get("candidate_perplexity_ratio_lower_bound"))
    finite_losses = all(value is not None for value in (
        candidate_nll, bf16_nll, repeat_nll, paired_ci_upper))
    perplexity_ratio = (
        math.exp(candidate_nll - bf16_nll)
        if finite_losses and candidate_nll - bf16_nll < 700  # type: ignore[operator]
        else math.inf
    )
    calibration_hashes = set(candidate.get("search_calibration_sha256", []))
    heldout_hash = corpus.get("sha256")
    perplexity_failure_proven = (
        perplexity.get("failure_proven_from_partial_candidate_evaluation") is True
        and perplexity_ratio_lower_bound is not None
        and perplexity_ratio_lower_bound
        > policy.maximum_perplexity_ratio_to_bf16
    )

    checks = {
        "exact_byte_target": (
            _integer(candidate.get("assignment_cost_bytes")) >= 0
            and _integer(candidate.get("assignment_cost_bytes"))
            == _integer(candidate.get("target_cost_bytes"))
        ),
        "candidate_identity_pinned": (
            _digest(candidate.get("model_revision"), length=40)
            and _digest(candidate.get("candidate_store_index_sha256"))
            and _digest(candidate.get("assignment_sha256"))
        ),
        "hard_assignment_reproduced": (
            candidate.get("independent_hard_replay") is True
        ),
        "heldout_is_disjoint": (
            _digest(heldout_hash) and heldout_hash not in calibration_hashes
        ),
        "heldout_manifest_pinned": _digest(heldout_hash),
        "minimum_documents": (
            _integer(corpus.get("document_count")) >= policy.minimum_documents
        ),
        "minimum_predicted_tokens": (
            _integer(corpus.get("predicted_token_count"))
            >= policy.minimum_predicted_tokens
        ),
        "minimum_corpus_strata": (
            len(set(corpus.get("strata", []))) >= policy.minimum_corpus_strata
        ),
        "finite_exact_full_vocabulary_nll": (
            finite_losses
            and perplexity.get("objective")
            == "exact full-vocabulary causal cross-entropy"
            and _integer(perplexity.get("nonfinite_token_count")) == 0
        ),
        "perplexity_ratio_to_bf16": (
            perplexity_ratio <= policy.maximum_perplexity_ratio_to_bf16
        ),
        "paired_nll_improves_incumbent": (
            paired_ci_upper is not None
            and paired_ci_upper
            <= policy.maximum_paired_nll_delta_ci95_upper
        ),
        "perplexity_reproduced": (
            candidate_nll is not None
            and repeat_nll is not None
            and abs(candidate_nll - repeat_nll)
            <= policy.maximum_repeat_nll_delta
            and _digest(perplexity.get("token_sequence_sha256"))
            and perplexity.get("token_sequence_sha256")
            == perplexity.get("repeat_token_sequence_sha256")
        ),
        "minimum_generation_prompts": (
            _integer(generation.get("prompt_count"))
            >= policy.minimum_generation_prompts
        ),
        "generation_manifest_pinned": _digest(
            generation.get("prompt_manifest_sha256")),
        "generation_assertions_pass": (
            _integer(generation.get("assertions_passed"))
            == _integer(generation.get("assertions_total"))
            and _integer(generation.get("assertions_total")) > 0
        ),
        "generation_reproduced": generation.get("deterministic_repeat") is True,
        "generation_is_well_formed": (
            _integer(generation.get("nonfinite_run_count")) == 0
            and _integer(generation.get("invalid_utf8_count")) == 0
            and _integer(generation.get("replacement_character_count")) == 0
            and _integer(generation.get("empty_output_count")) == 0
        ),
        "cuda_kernel_matrix": runtime.get("cuda_kernel_matrix_pass") is True,
        "cuda_one_step_memory": runtime.get("cuda_one_step_pass") is True,
        "cuda_multi_step_memory": runtime.get("cuda_multi_step_pass") is True,
        "cuda_reserved_below_limit": (
            _integer(runtime.get("max_cuda_reserved_bytes")) >= 0
            and _integer(runtime.get("max_cuda_reserved_bytes"))
            < policy.maximum_cuda_reserved_bytes
        ),
        "partial_cuda_offload": runtime.get("partial_cuda_offload_pass") is True,
        "unmodified_llama_cpp": runtime.get("unmodified_llama_cpp") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "policy": asdict(policy),
        "measurements": {
            "candidate_perplexity": (
                math.exp(candidate_nll)
                if candidate_nll is not None and candidate_nll < 700
                else None
            ),
            "bf16_perplexity": (
                math.exp(bf16_nll)
                if bf16_nll is not None and bf16_nll < 700
                else None
            ),
            "perplexity_ratio_to_bf16": (
                perplexity_ratio if math.isfinite(perplexity_ratio) else None
            ),
            "candidate_perplexity_ratio_lower_bound": (
                perplexity_ratio_lower_bound
            ),
            "perplexity_failure_proven": perplexity_failure_proven,
            "repeat_nll_delta": (
                abs(candidate_nll - repeat_nll)
                if candidate_nll is not None and repeat_nll is not None
                else None
            ),
        },
        "checks": checks,
        "failed_checks": failed,
        "authorized_for_final_gguf_construction": not failed,
    }
