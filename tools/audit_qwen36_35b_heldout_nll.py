#!/usr/bin/env python3
"""Run resumable exact held-out NLL evaluation through streamed Qwen weights."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
import platform
import resource
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import transformers
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoint_stream import SafeTensorPrefixLoader  # noqa: E402
from native_runtime import NativeManifestWeightStore  # noqa: E402
from native_store import NativeCandidateStore  # noqa: E402
from quant.ggml_native import GGMLNativeCodec, GGMLType  # noqa: E402
from release_corpus import canonical_json_bytes, sha256_bytes  # noqa: E402
from release_quality import (  # noqa: E402
    paired_bootstrap_mean_ci,
    partial_perplexity_ratio_certificate,
    weighted_mean_nll,
)
from search.hard import realized_cost  # noqa: E402
from search.streaming import StreamingHardCausalEvaluator  # noqa: E402


VARIANTS = ("bf16", "incumbent", "candidate")
MAXIMUM_CANDIDATE_PERPLEXITY_RATIO = 1.15


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _assignment_bits(values: torch.Tensor) -> str:
    return "".join(str(int(value)) for value in values.tolist())


def _load_corpus(
    report_path: Path,
    tokenizer: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], torch.Tensor]:
    report = _load_json(report_path)
    canonical = report["canonical_manifest"]
    actual_manifest_sha = sha256_bytes(canonical_json_bytes(canonical))
    if actual_manifest_sha != report["canonical_manifest_sha256"]:
        raise RuntimeError("held-out canonical manifest digest mismatch")
    corpus_path = Path(canonical["corpus"]["path"]).resolve(strict=True)
    if _sha256_file(corpus_path) != canonical["corpus"]["sha256"]:
        raise RuntimeError("held-out corpus payload digest mismatch")
    documents = [
        json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    metadata = canonical["documents"]
    if len(documents) != len(metadata):
        raise RuntimeError("held-out corpus document count mismatch")
    token_rows = []
    sequence_digest = hashlib.sha256()
    for document, expected in zip(documents, metadata, strict=True):
        if document["id"] != expected["id"]:
            raise RuntimeError("held-out corpus document order mismatch")
        content = document["text"].encode("utf-8")
        if sha256_bytes(content) != expected["content_sha256"]:
            raise RuntimeError(f"held-out content mismatch: {document['id']}")
        tokens = [int(value) for value in tokenizer.encode(
            document["text"], add_special_tokens=False)]
        token_bytes = canonical_json_bytes(tokens)
        if (
            len(tokens) != expected["token_count"]
            or sha256_bytes(token_bytes) != expected["token_ids_sha256"]
        ):
            raise RuntimeError(f"held-out token mismatch: {document['id']}")
        token_rows.append(tokens)
        sequence_digest.update(len(tokens).to_bytes(8, "little"))
        sequence_digest.update(token_bytes)
    if sequence_digest.hexdigest() != canonical["token_sequence_sha256"]:
        raise RuntimeError("held-out combined token-sequence digest mismatch")
    widths = {len(row) for row in token_rows}
    if len(widths) != 1:
        raise RuntimeError("held-out documents do not share one sequence length")
    return report, metadata, torch.tensor(token_rows, dtype=torch.long)


def _aggregate(run: dict[str, Any]) -> dict[str, Any]:
    documents = [
        item for batch in sorted(run["batches"], key=lambda item: item["index"])
        for item in batch["documents"]
    ]
    means = [float(item["mean_nll"]) for item in documents]
    counts = [int(item["predicted_token_count"]) for item in documents]
    mean_nll = weighted_mean_nll(means, counts)
    return {
        "document_count": len(documents),
        "predicted_token_count": sum(counts),
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll) if mean_nll < 700 else None,
        "nonfinite_token_count": 0,
        "document_mean_nll": means,
        "document_ids": [item["id"] for item in documents],
        "max_cuda_allocated_bytes": max(
            (item["memory"]["cuda_max_allocated"] for item in run["batches"]),
            default=0,
        ),
        "max_cuda_reserved_bytes": max(
            (item["memory"]["cuda_max_reserved"] for item in run["batches"]),
            default=0,
        ),
        "max_process_peak_rss_bytes": max(
            (item["memory"]["process_peak_rss"] for item in run["batches"]),
            default=0,
        ),
        "elapsed_seconds": sum(
            item["memory"]["total_seconds"] for item in run["batches"]),
    }


def _early_failure_certificate(
    report: dict[str, Any], *, total_token_count: int,
) -> dict[str, Any] | None:
    runs = report["runs"]
    bf16 = runs.get("bf16", {}).get("aggregate")
    candidate_batches = runs.get("candidate", {}).get("batches", [])
    if bf16 is None or not candidate_batches:
        return None
    documents = [
        document
        for batch in sorted(candidate_batches, key=lambda item: item["index"])
        for document in batch["documents"]
    ]
    certificate = partial_perplexity_ratio_certificate(
        baseline_mean_nll=float(bf16["mean_nll"]),
        observed_mean_nll=[float(item["mean_nll"]) for item in documents],
        observed_token_counts=[
            int(item["predicted_token_count"]) for item in documents
        ],
        total_token_count=total_token_count,
        maximum_ratio=MAXIMUM_CANDIDATE_PERPLEXITY_RATIO,
    )
    certificate.update({
        "criterion": "candidate_perplexity_ratio_to_bf16_lte_1.15",
        "proof": (
            "causal cross-entropy is nonnegative, so assigning zero NLL to "
            "every unseen token is a valid lower bound on the final mean NLL"
        ),
        "candidate_completed_batch_count": len(candidate_batches),
        "candidate_completed_document_count": len(documents),
    })
    return certificate


def _finalize_early_failure(
    report: dict[str, Any], certificate: dict[str, Any], *, started: float,
) -> dict[str, Any]:
    report["status"] = "fail"
    report["quality"] = {
        "objective": "exact full-vocabulary causal cross-entropy",
        "gate": "candidate_perplexity_ratio_to_bf16_lte_1.15",
        "result": "fail",
        "failure_proven_from_partial_candidate_evaluation": True,
        "early_failure_certificate": certificate,
        "nonfinite_token_count": 0,
    }
    report["elapsed_seconds"] = sum(
        float(batch["memory"]["total_seconds"])
        for run in report["runs"].values()
        for batch in run["batches"]
    )
    report["peak_process_rss_bytes"] = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    report["wall_seconds_this_invocation"] = time.perf_counter() - started
    return report


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda is unavailable")
    variants = tuple(dict.fromkeys(args.variant))
    if not variants or any(item not in VARIANTS for item in variants):
        raise ValueError(f"variants must come from {VARIANTS}")

    model_dir = args.model_dir.resolve(strict=True)
    identity_path = args.identity.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    store_path = args.store.resolve(strict=True)
    corpus_report_path = args.corpus_manifest.resolve(strict=True)
    incumbent_report_path = args.incumbent_gguf_report.resolve(strict=True)
    candidate_report_path = args.candidate_report.resolve(strict=True)
    identity = _load_json(identity_path)
    manifest = _load_json(manifest_path)
    entries = sorted((
        entry for entry in manifest["entries"] if entry.get("rco_search")
    ), key=lambda entry: entry["destination_name"])
    if len(entries) != 512:
        raise RuntimeError(f"expected 512 decision groups, found {len(entries)}")
    names = [entry["destination_name"] for entry in entries]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    corpus_report, document_metadata, input_ids = _load_corpus(
        corpus_report_path, tokenizer)
    if input_ids.shape[0] < 100:
        raise RuntimeError("held-out corpus contains fewer than 100 documents")

    codec = GGMLNativeCodec(args.ggml_library.resolve(strict=True))
    packed_store = NativeCandidateStore(store_path, codec)
    weight_store = NativeManifestWeightStore(
        packed_store, {"entries": entries}, model_dir,
        rows_per_chunk=args.rows_per_chunk,
    )
    low_costs = [
        int(packed_store.metadata(name, GGMLType.Q2_0)["aligned_gguf_bytes"])
        for name in names
    ]
    high_costs = [
        int(packed_store.metadata(name, GGMLType.Q4_0)["aligned_gguf_bytes"])
        for name in names
    ]

    candidate_report = _load_json(candidate_report_path)
    candidate_bits = candidate_report["incumbent"]["assignment_bits"]
    candidate = torch.tensor([int(value) for value in candidate_bits], dtype=torch.long)
    if len(candidate) != len(names):
        raise RuntimeError("candidate assignment length mismatch")
    incumbent_report = _load_json(incumbent_report_path)
    choices = incumbent_report["assignment"]["choices"]
    if set(choices) != set(names):
        raise RuntimeError("incumbent GGUF choices do not match decision groups")
    incumbent = torch.tensor([
        0 if choices[name] == GGMLType.Q2_0.name
        else 1 if choices[name] == GGMLType.Q4_0.name
        else -1
        for name in names
    ], dtype=torch.long)
    if bool((incumbent < 0).any().item()):
        raise RuntimeError("incumbent GGUF contains an unsupported choice")
    if torch.equal(candidate, incumbent):
        raise RuntimeError("candidate and incumbent assignments are identical")
    assignments = {
        "bf16": torch.empty(0, dtype=torch.long),
        "incumbent": incumbent,
        "candidate": candidate,
    }
    assignment_evidence = {}
    for label in ("incumbent", "candidate"):
        value = assignments[label]
        cost = realized_cost(value, low_costs, high_costs)
        if cost != args.target_cost:
            raise RuntimeError(f"{label} cost {cost} != {args.target_cost}")
        bits = _assignment_bits(value)
        assignment_evidence[label] = {
            "assignment_bits": bits,
            "assignment_sha256": hashlib.sha256(bits.encode()).hexdigest(),
            "realized_aligned_gguf_bytes": cost,
        }

    problem = {
        "model_revision": identity["revision"],
        "identity_sha256": _sha256_file(identity_path),
        "gguf_manifest_sha256": _sha256_file(manifest_path),
        "candidate_store_index_sha256": _sha256_file(
            store_path / "native-candidate-index.json"),
        "corpus_manifest_sha256": corpus_report["canonical_manifest_sha256"],
        "token_sequence_sha256": corpus_report[
            "canonical_manifest"]["token_sequence_sha256"],
        "incumbent_gguf_report_sha256": _sha256_file(incumbent_report_path),
        "candidate_report_sha256": _sha256_file(candidate_report_path),
        "assignments": assignment_evidence,
        "variants": list(variants),
        "batch_size": args.batch_size,
        "vocab_chunk_size": args.vocab_chunk_size,
        "rows_per_chunk": args.rows_per_chunk,
        "device": str(device),
    }
    problem_sha256 = sha256_bytes(canonical_json_bytes(problem))
    if args.output.exists():
        report = _load_json(args.output)
        if report.get("problem_sha256") != problem_sha256:
            raise RuntimeError("existing held-out report describes another problem")
    else:
        report = {
            "schema": 1,
            "status": "in_progress",
            "scope": (
                "exact full-vocabulary causal cross-entropy on the pinned "
                "100-document Qwen3.6 release corpus with one decoder block "
                "resident at a time"
            ),
            "problem": problem,
            "problem_sha256": problem_sha256,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "cuda_built_version": torch.version.cuda,
                "cuda_device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda" else None
                ),
            },
            "runs": {label: {"batches": []} for label in variants},
        }
        _atomic_json(args.output, report)

    total_token_count = input_ids.shape[0] * (input_ids.shape[1] - 1)
    certificate = _early_failure_certificate(
        report, total_token_count=total_token_count)
    if certificate is not None and certificate["failure_proven"]:
        report = _finalize_early_failure(
            report, certificate, started=started)
        _atomic_json(args.output, report)
        return report

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(
            config, attn_implementation="eager")
    model.eval()
    loader = SafeTensorPrefixLoader(model_dir)
    groups = [SimpleNamespace(layer_names=(name,)) for name in names]
    candidate_evaluator = StreamingHardCausalEvaluator(
        model, loader, weight_store, groups, [2, 4], device=device,
        vocab_chunk_size=args.vocab_chunk_size,
    )
    bf16_evaluator = StreamingHardCausalEvaluator(
        model, loader, weight_store, [], [2, 4], device=device,
        vocab_chunk_size=args.vocab_chunk_size,
    )

    new_batch_count = 0
    for label in variants:
        run = report["runs"][label]
        evaluator = bf16_evaluator if label == "bf16" else candidate_evaluator
        assignment = assignments[label]
        for start in range(0, input_ids.shape[0], args.batch_size):
            batch_index = start // args.batch_size
            stop = min(start + args.batch_size, input_ids.shape[0])
            expected_ids = [
                item["id"] for item in document_metadata[start:stop]
            ]
            if batch_index < len(run["batches"]):
                if (
                    run["batches"][batch_index]["index"] != batch_index
                    or [item["id"] for item in run["batches"][batch_index]["documents"]]
                    != expected_ids
                ):
                    raise RuntimeError("existing held-out batch order mismatch")
                continue
            evaluation = evaluator.evaluate(input_ids[start:stop], assignment)
            expected_count = (stop - start) * (input_ids.shape[1] - 1)
            if evaluation.token_count != expected_count:
                raise RuntimeError("held-out evaluation returned wrong token count")
            if len(evaluation.document_mean_nll) != stop - start:
                raise RuntimeError("held-out evaluation returned wrong document count")
            if any(not math.isfinite(value)
                   for value in evaluation.document_mean_nll):
                raise RuntimeError("held-out evaluation returned non-finite NLL")
            batch = {
                "index": batch_index,
                "start_document": start,
                "stop_document": stop,
                "mean_nll": evaluation.loss,
                "predicted_token_count": evaluation.token_count,
                "documents": [
                    {
                        "id": identifier,
                        "mean_nll": mean_nll,
                        "predicted_token_count": token_count,
                    }
                    for identifier, mean_nll, token_count in zip(
                        expected_ids,
                        evaluation.document_mean_nll,
                        evaluation.document_token_counts,
                        strict=True,
                    )
                ],
                "memory": asdict(evaluation.memory),
            }
            run["batches"].append(batch)
            _atomic_json(args.output, report)
            print(json.dumps({
                "variant": label, "batch": batch_index,
                "documents": [start, stop], "mean_nll": evaluation.loss,
                "seconds": evaluation.memory.total_seconds,
                "cuda_reserved_bytes": evaluation.memory.cuda_max_reserved,
            }, sort_keys=True), flush=True)
            new_batch_count += 1
            certificate = _early_failure_certificate(
                report, total_token_count=total_token_count)
            if certificate is not None and certificate["failure_proven"]:
                report = _finalize_early_failure(
                    report, certificate, started=started)
                _atomic_json(args.output, report)
                return report
            if (
                args.stop_after_new_batches is not None
                and new_batch_count >= args.stop_after_new_batches
            ):
                report["wall_seconds_this_invocation"] = (
                    time.perf_counter() - started)
                return report
        run["aggregate"] = _aggregate(run)
        _atomic_json(args.output, report)

    if all(label in report["runs"] and report["runs"][label].get("aggregate")
           for label in variants):
        report["status"] = "pass"
        report["elapsed_seconds"] = sum(
            run["aggregate"]["elapsed_seconds"]
            for run in report["runs"].values()
        )
        report["peak_process_rss_bytes"] = int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        if all(label in report["runs"] for label in VARIANTS):
            bf16 = report["runs"]["bf16"]["aggregate"]
            incumbent_result = report["runs"]["incumbent"]["aggregate"]
            candidate_result = report["runs"]["candidate"]["aggregate"]
            if not (
                bf16["document_ids"] == incumbent_result["document_ids"]
                == candidate_result["document_ids"]
            ):
                raise RuntimeError("held-out variant document order differs")
            deltas = [
                candidate_value - incumbent_value
                for candidate_value, incumbent_value in zip(
                    candidate_result["document_mean_nll"],
                    incumbent_result["document_mean_nll"], strict=True,
                )
            ]
            ci_lower, ci_upper = paired_bootstrap_mean_ci(
                deltas, samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            )
            report["quality"] = {
                "objective": "exact full-vocabulary causal cross-entropy",
                "bf16_mean_nll": bf16["mean_nll"],
                "incumbent_mean_nll": incumbent_result["mean_nll"],
                "candidate_mean_nll": candidate_result["mean_nll"],
                "candidate_perplexity_ratio_to_bf16": math.exp(
                    candidate_result["mean_nll"] - bf16["mean_nll"]),
                "paired_candidate_minus_incumbent_mean_nll": math.fsum(
                    deltas) / len(deltas),
                "paired_candidate_minus_incumbent_ci95_lower": ci_lower,
                "paired_candidate_minus_incumbent_ci95_upper": ci_upper,
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_seed": args.bootstrap_seed,
                "token_sequence_sha256": problem["token_sequence_sha256"],
                "nonfinite_token_count": 0,
            }
        if args.reference_report is not None:
            reference_path = args.reference_report.resolve(strict=True)
            reference = _load_json(reference_path)
            if (
                reference["problem"]["token_sequence_sha256"]
                != problem["token_sequence_sha256"]
                or "candidate" not in reference["runs"]
                or "candidate" not in report["runs"]
            ):
                raise RuntimeError("repeat reference describes another corpus")
            primary = reference["runs"]["candidate"]["aggregate"]
            repeat = report["runs"]["candidate"]["aggregate"]
            delta = abs(primary["mean_nll"] - repeat["mean_nll"])
            report["reproducibility"] = {
                "reference_path": str(reference_path),
                "reference_sha256": _sha256_file(reference_path),
                "token_sequence_sha256_matches": True,
                "document_ids_match": (
                    primary["document_ids"] == repeat["document_ids"]),
                "candidate_mean_nll_absolute_delta": delta,
                "candidate_mean_nll_within_1e_6": delta <= 1e-6,
            }
            if not (
                report["reproducibility"]["document_ids_match"]
                and report["reproducibility"]["candidate_mean_nll_within_1e_6"]
            ):
                raise RuntimeError("candidate held-out repeat did not reproduce")
    _atomic_json(args.output, report)
    report["wall_seconds_this_invocation"] = time.perf_counter() - started
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--incumbent-gguf-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--target-cost", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261001)
    parser.add_argument("--stop-after-new-batches", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if (args.stop_after_new_batches is not None
            and args.stop_after_new_batches < 1):
        raise ValueError("--stop-after-new-batches must be positive")
    report = audit(args)
    print(json.dumps({
        "status": report["status"], "output": str(args.output),
        "completed_variants": sorted(
            label for label, run in report["runs"].items()
            if run.get("aggregate") is not None
        ),
        "quality": report.get("quality"),
        "reproducibility": report.get("reproducibility"),
        "wall_seconds_this_invocation": report["wall_seconds_this_invocation"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
