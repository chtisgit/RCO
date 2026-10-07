#!/usr/bin/env python3
"""Paired held-out comparison of the GSQ-RCO candidate against authentic GSQ.

Both inputs are llama.cpp ``rco.gguf_document_nll.v1`` reports scored with the
same helper, llama.cpp revision, corpus and runtime parameters; only the model
may differ.  A repeat candidate report must be bit-identical per document.
The BF16 reference comes from the native held-out report, whose token sequence
must match.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from release_quality import paired_bootstrap_mean_ci  # noqa: E402


SCHEMA = "rco.qwen36.gsq_rco_candidate_heldout_comparison.v1"
PERPLEXITY_RATIO_GATE = 1.15


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _documents(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if report.get("status") != "complete":
        raise RuntimeError("held-out report is not complete")
    return {document["id"]: document for document in report["documents"]}


def _require_same_scoring(left: dict[str, Any], right: dict[str, Any]) -> None:
    for key in ("corpus_manifest", "helper", "llama_cpp", "parameters",
                "token_sequence_sha256", "schema"):
        if left["identity"][key] != right["identity"][key]:
            raise RuntimeError(f"held-out scoring identity differs: {key}")
    if left["environment"] != right["environment"]:
        raise RuntimeError("held-out scoring environment differs")


def compare(
    candidate: dict[str, Any],
    repeat: dict[str, Any],
    incumbent: dict[str, Any],
    bf16_reference: dict[str, Any],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    _require_same_scoring(candidate, incumbent)
    _require_same_scoring(candidate, repeat)
    if candidate["identity"]["model"] != repeat["identity"]["model"]:
        raise RuntimeError("repeat report scored a different model")
    if candidate["identity"]["model"]["sha256"] == incumbent["identity"]["model"]["sha256"]:
        raise RuntimeError("candidate and incumbent are the same model")
    if (bf16_reference["problem"]["token_sequence_sha256"]
            != candidate["identity"]["token_sequence_sha256"]):
        raise RuntimeError("BF16 reference token sequence differs")

    candidate_documents = _documents(candidate)
    repeat_documents = _documents(repeat)
    incumbent_documents = _documents(incumbent)
    order = candidate["aggregate"]["document_ids"]
    if (order != incumbent["aggregate"]["document_ids"]
            or order != repeat["aggregate"]["document_ids"]
            or order != bf16_reference["runs"]["bf16"]["aggregate"]["document_ids"]):
        raise RuntimeError("held-out document order differs")

    repeat_max_abs = 0.0
    deltas = []
    weighted_delta_sum = []
    token_count = 0
    improved = 0
    domains: dict[str, dict[str, float]] = {}
    for document_id in order:
        current = candidate_documents[document_id]
        again = repeat_documents[document_id]
        baseline = incumbent_documents[document_id]
        if current["predicted_token_count"] != baseline["predicted_token_count"]:
            raise RuntimeError(f"predicted token count differs: {document_id}")
        repeat_max_abs = max(repeat_max_abs, abs(current["nll_sum"] - again["nll_sum"]))
        delta = current["mean_nll"] - baseline["mean_nll"]
        deltas.append(delta)
        weighted_delta_sum.append(current["nll_sum"] - baseline["nll_sum"])
        token_count += current["predicted_token_count"]
        improved += delta < 0
        domain = domains.setdefault(document_id.split("-", 1)[0], {
            "documents": 0, "predicted_tokens": 0, "nll_sum_delta": 0.0})
        domain["documents"] += 1
        domain["predicted_tokens"] += current["predicted_token_count"]
        domain["nll_sum_delta"] += current["nll_sum"] - baseline["nll_sum"]
    if repeat_max_abs > 1e-6:
        raise RuntimeError(f"candidate repeat is not reproducible: {repeat_max_abs}")

    ci_lower, ci_upper = paired_bootstrap_mean_ci(deltas, samples=samples, seed=seed)
    bf16_nll = bf16_reference["runs"]["bf16"]["aggregate"]["mean_nll"]
    candidate_nll = candidate["aggregate"]["mean_nll"]
    incumbent_nll = incumbent["aggregate"]["mean_nll"]
    candidate_ratio = math.exp(candidate_nll - bf16_nll)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "scope": (
            "llama.cpp held-out quality of the ungated evaluation-only GSQ-RCO "
            "candidate; not a release certification"
        ),
        "models": {
            "candidate": candidate["identity"]["model"],
            "incumbent": incumbent["identity"]["model"],
        },
        "scoring": {
            key: candidate["identity"][key]
            for key in ("corpus_manifest", "helper", "llama_cpp", "parameters",
                        "token_sequence_sha256")
        },
        "reproducibility": {
            "candidate_repeat_max_abs_document_nll_sum_delta": repeat_max_abs,
        },
        "quality": {
            "document_count": len(order),
            "predicted_token_count": token_count,
            "bf16_mean_nll": bf16_nll,
            "incumbent_mean_nll": incumbent_nll,
            "candidate_mean_nll": candidate_nll,
            "incumbent_perplexity": math.exp(incumbent_nll),
            "candidate_perplexity": math.exp(candidate_nll),
            "bf16_perplexity": math.exp(bf16_nll),
            "token_weighted_candidate_minus_incumbent_mean_nll": (
                math.fsum(weighted_delta_sum) / token_count),
            "paired_candidate_minus_incumbent_mean_nll": math.fsum(deltas) / len(deltas),
            "paired_candidate_minus_incumbent_ci95_lower": ci_lower,
            "paired_candidate_minus_incumbent_ci95_upper": ci_upper,
            "documents_improved": improved,
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
            "bf16_gap_closed_fraction": (
                (incumbent_nll - candidate_nll) / (incumbent_nll - bf16_nll)),
            "incumbent_perplexity_ratio_to_bf16": math.exp(incumbent_nll - bf16_nll),
            "candidate_perplexity_ratio_to_bf16": candidate_ratio,
            "perplexity_ratio_gate": PERPLEXITY_RATIO_GATE,
            "perplexity_ratio_gate_passed": candidate_ratio <= PERPLEXITY_RATIO_GATE,
            "domains": {
                name: {
                    "documents": int(values["documents"]),
                    "predicted_tokens": int(values["predicted_tokens"]),
                    "token_weighted_candidate_minus_incumbent_mean_nll": (
                        values["nll_sum_delta"] / values["predicted_tokens"]),
                }
                for name, values in sorted(domains.items())
            },
        },
        "improvement_over_incumbent_passed": ci_upper < 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-repeat", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--bf16-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261001)
    args = parser.parse_args()
    report = compare(
        _load_json(args.candidate),
        _load_json(args.candidate_repeat),
        _load_json(args.incumbent),
        _load_json(args.bf16_reference),
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(report["quality"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
