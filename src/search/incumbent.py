"""Fail-closed selection of reproducible hard-assignment incumbents."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class HardCandidateEvidence:
    """A hard projection and the independent report that reproduces it."""

    label: str
    primary: Mapping[str, Any]
    repeat: Mapping[str, Any]
    primary_sha256: str


def _problem_identity(report: Mapping[str, Any]) -> tuple[Any, ...]:
    source = report["source"]
    store = report["candidate_store"]
    calibration = report["calibration"]
    return (
        source["repo_id"],
        source["revision"],
        store["index_sha256"],
        store["decision_count"],
        store["low_type"],
        store["high_type"],
        store["target_cost"],
        tuple(calibration["input_ids"]),
        calibration["objective"],
        calibration["vocab_chunk_size"],
    )


def _validate_report(report: Mapping[str, Any], label: str) -> None:
    if report.get("status") != "pass":
        raise ValueError(f"{label} did not pass")
    hard_loss = float(report["hard_evaluation"]["loss"])
    if not math.isfinite(hard_loss):
        raise ValueError(f"{label} has a non-finite hard loss")
    assignment = report["relaxed_step"]["assignment"]
    decision_count = int(report["candidate_store"]["decision_count"])
    if len(assignment) != decision_count:
        raise ValueError(f"{label} has the wrong assignment length")
    if any(choice not in (0, 1) for choice in assignment):
        raise ValueError(f"{label} has a non-binary assignment")
    assignment_cost = int(report["relaxed_step"]["assignment_cost"])
    target_cost = int(report["candidate_store"]["target_cost"])
    if assignment_cost != target_cost:
        raise ValueError(f"{label} misses the exact byte target")


def select_reproducible_hard_incumbent(
    evidence: Sequence[HardCandidateEvidence],
) -> dict[str, Any]:
    """Validate hard/repeat pairs and return the minimum-loss incumbent.

    All candidates must describe exactly the same model revision, candidate
    store, calibration tokens, objective, and serialized-byte target. A repeat
    must reference the supplied primary report digest and reproduce its
    assignment and exact loss.
    """
    if not evidence:
        raise ValueError("at least one hard candidate is required")
    labels = [item.label for item in evidence]
    if len(set(labels)) != len(labels):
        raise ValueError("candidate labels must be unique")

    expected_identity = None
    candidates: list[dict[str, Any]] = []
    for item in evidence:
        _validate_report(item.primary, f"{item.label} primary")
        _validate_report(item.repeat, f"{item.label} repeat")
        identity = _problem_identity(item.primary)
        if expected_identity is None:
            expected_identity = identity
        elif identity != expected_identity:
            raise ValueError(f"{item.label} describes a different hard problem")
        if _problem_identity(item.repeat) != identity:
            raise ValueError(f"{item.label} repeat describes a different problem")

        reference = item.repeat.get("reproducibility_reference")
        if not isinstance(reference, Mapping):
            raise ValueError(f"{item.label} repeat has no primary reference")
        if reference.get("sha256") != item.primary_sha256:
            raise ValueError(f"{item.label} repeat references the wrong primary")
        if item.repeat.get("same_assignment_reproducible") is not True:
            raise ValueError(f"{item.label} assignment was not reproduced")

        primary_step = item.primary["relaxed_step"]
        repeat_step = item.repeat["relaxed_step"]
        if (
            repeat_step["assignment"] != primary_step["assignment"]
            or repeat_step["assignment_bits"] != primary_step["assignment_bits"]
            or repeat_step["assignment_cost"] != primary_step["assignment_cost"]
            or item.repeat["hard_evaluation"]["loss"]
            != item.primary["hard_evaluation"]["loss"]
            or item.repeat["hard_evaluation"]["token_count"]
            != item.primary["hard_evaluation"]["token_count"]
        ):
            raise ValueError(f"{item.label} hard evidence did not reproduce exactly")

        candidates.append({
            "label": item.label,
            "hard_loss": float(item.primary["hard_evaluation"]["loss"]),
            "relaxed_loss": float(primary_step["relaxed_loss"]),
            "assignment_cost": int(primary_step["assignment_cost"]),
            "assignment_bits": primary_step["assignment_bits"],
            "step_report_sha256": primary_step["sha256"],
            "primary_report_sha256": item.primary_sha256,
            "reproduced": True,
        })

    candidates.sort(key=lambda candidate: (candidate["hard_loss"], candidate["label"]))
    incumbent = candidates[0]
    for candidate in candidates:
        delta = candidate["hard_loss"] - incumbent["hard_loss"]
        candidate["hard_loss_delta_from_incumbent"] = delta
        candidate["hard_loss_regression_percent"] = (
            100.0 * delta / incumbent["hard_loss"])
        candidate["selected"] = candidate is incumbent

    return {
        "problem": {
            "repo_id": expected_identity[0],
            "revision": expected_identity[1],
            "candidate_store_index_sha256": expected_identity[2],
            "decision_count": expected_identity[3],
            "low_type": expected_identity[4],
            "high_type": expected_identity[5],
            "target_cost": expected_identity[6],
            "calibration_input_ids": list(expected_identity[7]),
            "objective": expected_identity[8],
            "vocab_chunk_size": expected_identity[9],
        },
        "candidates": candidates,
        "incumbent": dict(incumbent),
    }
