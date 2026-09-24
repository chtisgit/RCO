#!/usr/bin/env python3
"""Run a reproducible exact-byte search step through all 35B text blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import transformers
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from native_runtime import NativeManifestWeightStore
from native_store import NativeCandidateStore
from quant.ggml_native import GGMLNativeCodec, GGMLType
from search.hard import optimize_cost_reinforce, realized_cost
from search.streaming import StreamingHardCausalEvaluator
from checkpoint_stream import SafeTensorPrefixLoader


CALIBRATION_TEXT = (
    "A bounded native GGML search streams every Qwen decoder block while "
    "evaluating an exact serialized byte budget."
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
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


def _assignment_key(assignment: torch.Tensor) -> str:
    return "".join(str(int(value)) for value in assignment.tolist())


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda is unavailable")

    model_dir = args.model_dir.resolve(strict=True)
    identity_path = args.identity.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    store_path = args.store.resolve(strict=True)
    identity = _load_json(identity_path)
    manifest = _load_json(manifest_path)
    entries = sorted((
        entry for entry in manifest["entries"]
        if entry.get("rco_search")
        and entry["destination_name"].startswith("blk.0.")
    ), key=lambda entry: entry["destination_name"])
    if len(entries) != 13:
        raise RuntimeError(
            f"expected 13 block-0 decision groups, found {len(entries)}")

    codec = GGMLNativeCodec(args.ggml_library.resolve(strict=True))
    packed_store = NativeCandidateStore(store_path, codec)
    store_revision = packed_store.index["source"].get(
        "revision", packed_store.index["source"].get("dense_revision"))
    if store_revision != identity["revision"]:
        raise RuntimeError("candidate store and checkpoint revisions differ")
    weight_store = NativeManifestWeightStore(
        packed_store,
        {"entries": entries},
        model_dir,
        rows_per_chunk=args.rows_per_chunk,
    )

    names = [entry["destination_name"] for entry in entries]
    low_costs = [
        int(packed_store.metadata(name, GGMLType.Q2_0)[
            "aligned_gguf_bytes"])
        for name in names
    ]
    high_costs = [
        int(packed_store.metadata(name, GGMLType.Q4_0)[
            "aligned_gguf_bytes"])
        for name in names
    ]

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(
            config, attn_implementation="eager")
    model.eval()
    loader = SafeTensorPrefixLoader(model_dir)
    groups = [SimpleNamespace(layer_names=(name,)) for name in names]
    evaluator = StreamingHardCausalEvaluator(
        model,
        loader,
        weight_store,
        groups,
        [2, 4],
        device=device,
        vocab_chunk_size=args.vocab_chunk_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True)
    input_ids = tokenizer(
        CALIBRATION_TEXT,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=args.sequence_length,
    )["input_ids"]
    if input_ids.shape[1] < args.sequence_length:
        raise RuntimeError(
            f"calibration text produced only {input_ids.shape[1]} tokens")
    input_ids = input_ids[:, :args.sequence_length].contiguous()

    runs = []
    for run_index in range(args.repeat_runs):
        evaluations = []

        def evaluate(assignment: torch.Tensor) -> float:
            exact_cost = realized_cost(
                assignment, low_costs, high_costs)
            if exact_cost != args.target_cost:
                raise RuntimeError(
                    f"optimizer emitted cost {exact_cost}, expected "
                    f"{args.target_cost}")
            evaluation = evaluator.evaluate(input_ids, assignment)
            record = {
                "index": len(evaluations),
                "assignment_bits": _assignment_key(assignment),
                "assignment": assignment.tolist(),
                "realized_aligned_gguf_bytes": exact_cost,
                "loss": evaluation.loss,
                "token_count": evaluation.token_count,
                "memory": asdict(evaluation.memory),
            }
            evaluations.append(record)
            print(json.dumps({
                "run": run_index,
                "evaluation": record["index"],
                "assignment_bits": record["assignment_bits"],
                "loss": record["loss"],
                "seconds": record["memory"]["total_seconds"],
                "peak_rss_bytes": record["memory"]["process_peak_rss"],
            }, sort_keys=True), flush=True)
            return evaluation.loss

        scores, assignment, history = optimize_cost_reinforce(
            evaluate,
            low_costs=low_costs,
            high_costs=high_costs,
            target_cost=args.target_cost,
            n_steps=args.steps,
            lr=args.learning_rate,
            baseline_decay=args.baseline_decay,
            seed=args.seed,
            log_interval=args.steps + 1,
        )
        run = {
            "index": run_index,
            "seed": args.seed,
            "steps": args.steps,
            "scores": [float(value) for value in scores],
            "selected_assignment_bits": _assignment_key(assignment),
            "selected_assignment": assignment.tolist(),
            "selected_assignment_cost": realized_cost(
                assignment, low_costs, high_costs),
            "history": history,
            "evaluations": evaluations,
        }
        runs.append(run)

    reference_report = None
    reference_path = None
    if args.reference_report is not None:
        reference_path = args.reference_report.resolve(strict=True)
        reference_report = _load_json(reference_path)
        if (
            reference_report["source"]["revision"] != identity["revision"]
            or reference_report["calibration"]["input_ids"]
            != input_ids.tolist()[0]
            or reference_report["candidate_store"]["target_cost"]
            != args.target_cost
            or reference_report["search"]["seed"] != args.seed
            or reference_report["search"]["steps_per_run"] != args.steps
        ):
            raise RuntimeError(
                "reference report does not describe the same search problem")
        reference = reference_report["search"]["runs"][0]
        runs_to_compare = runs
    else:
        reference = runs[0]
        runs_to_compare = runs[1:]

    def _matches_reference(run: dict[str, Any]) -> bool:
        return (
            run["scores"] == reference["scores"]
            and run["selected_assignment"] == reference["selected_assignment"]
            and [item["assignment"] for item in run["evaluations"]]
            == [item["assignment"] for item in reference["evaluations"]]
            and [item["loss"] for item in run["evaluations"]]
            == [item["loss"] for item in reference["evaluations"]]
        )

    reproducible = (
        None if not runs_to_compare
        else all(_matches_reference(run) for run in runs_to_compare)
    )
    if reproducible is False:
        raise RuntimeError("same-seed full-model runs were not reproducible")
    all_evaluations = [
        evaluation for run in runs for evaluation in run["evaluations"]
    ]
    if any(item["token_count"] != args.sequence_length - 1
           for item in all_evaluations):
        raise RuntimeError("full-model evaluation returned a wrong token count")
    if any(item["memory"]["loaded_blocks"] != 40
           for item in all_evaluations):
        raise RuntimeError("full-model evaluation did not stream all 40 blocks")
    losses = [item["loss"] for item in all_evaluations]
    distinct_assignments = sorted({
        item["assignment_bits"] for item in all_evaluations
    })
    gradient_norms = [
        float(step["gradient_norm"])
        for run in runs for step in run["history"]
    ]

    cuda_available = torch.cuda.is_available()
    cuda = {
        "available": cuda_available,
        "built_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }
    if cuda_available:
        cuda["device_name"] = torch.cuda.get_device_name(0)

    return {
        "schema": 1,
        "status": "pass",
        "scope": (
            "exact-byte antithetic REINFORCE through the complete 40-layer "
            "Qwen3.6-35B-A3B text stack, with native Q2_0/Q4_0 choices for "
            "the 13 block-0 tensors and dense BF16 weights for all remaining "
            "tensors; this does not represent the not-yet-generated full "
            "512-group candidate database"
        ),
        "source": {
            "repo_id": identity["repo_id"],
            "revision": identity["revision"],
            "model_dir": str(model_dir),
            "identity_path": str(identity_path),
            "identity_sha256": _sha256_file(identity_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "cuda": cuda,
        },
        "calibration": {
            "text": CALIBRATION_TEXT,
            "sequence_length": args.sequence_length,
            "input_ids": input_ids.tolist()[0],
            "next_token_count": args.sequence_length - 1,
            "objective": "exact full-vocabulary causal cross-entropy",
            "vocab_chunk_size": args.vocab_chunk_size,
        },
        "candidate_store": {
            "path": str(store_path),
            "index_sha256": _sha256_file(
                store_path / "native-candidate-index.json"),
            "tensor_names": names,
            "low_type": GGMLType.Q2_0.name,
            "high_type": GGMLType.Q4_0.name,
            "low_aligned_costs": low_costs,
            "high_aligned_costs": high_costs,
            "uniform_low_cost": sum(low_costs),
            "uniform_high_cost": sum(high_costs),
            "target_cost": args.target_cost,
            "rows_per_chunk": args.rows_per_chunk,
        },
        "search": {
            "estimator": "antithetic_reinforce",
            "seed": args.seed,
            "steps_per_run": args.steps,
            "repeat_runs": args.repeat_runs,
            "learning_rate": args.learning_rate,
            "baseline_decay": args.baseline_decay,
            "evaluation_count": len(all_evaluations),
            "all_assignments_exact_cost": all(
                item["realized_aligned_gguf_bytes"] == args.target_cost
                for item in all_evaluations),
            "distinct_evaluated_assignments": distinct_assignments,
            "distinct_evaluated_assignment_count": len(distinct_assignments),
            "minimum_evaluated_loss": min(losses),
            "maximum_evaluated_loss": max(losses),
            "mean_evaluated_loss": sum(losses) / len(losses),
            "nonzero_gradient_step_count": sum(
                value > 0.0 for value in gradient_norms),
            "same_seed_reproducible": reproducible,
            "reproducibility_reference": (
                None if reference_path is None else {
                    "path": str(reference_path),
                    "sha256": _sha256_file(reference_path),
                }
            ),
            "runs": runs,
        },
        "memory": {
            "max_process_peak_rss_bytes": max(
                item["memory"]["process_peak_rss"]
                for item in all_evaluations),
            "max_cuda_allocated_bytes": max(
                item["memory"]["cuda_max_allocated"]
                for item in all_evaluations),
            "max_cuda_reserved_bytes": max(
                item["memory"]["cuda_max_reserved"]
                for item in all_evaluations),
            "max_resident_block_bytes": max(
                item["memory"]["max_block_bytes"]
                for item in all_evaluations),
            "max_candidate_chunk_bytes": max(
                item["memory"]["max_candidate_bytes"]
                for item in all_evaluations),
        },
        "peak_process_rss_bytes": (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024),
        "elapsed_seconds": time.perf_counter() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--target-cost", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--repeat-runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--baseline-decay", type=float, default=0.9)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sequence_length < 2:
        raise ValueError("sequence length must be at least two")
    if args.steps < 1 or args.repeat_runs < 1:
        raise ValueError("steps and repeat runs must be positive")
    report = audit(args)
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "evaluation_count": report["search"]["evaluation_count"],
        "same_seed_reproducible": report["search"][
            "same_seed_reproducible"],
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
