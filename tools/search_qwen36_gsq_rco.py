#!/usr/bin/env python3
"""Run exact-byte hard search over authentic-GSQ retain/upgrade choices."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
from typing import Any

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForImageTextToText

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gguf_checkpoint_stream import GGUFManifestPrefixLoader  # noqa: E402
from gsq_upgrade_runtime import GSQUpgradeWeightStore  # noqa: E402
from native_runtime import NativeManifestWeightStore  # noqa: E402
from native_store import NativeCandidateStore  # noqa: E402
from quant.ggml_native import GGMLNativeCodec  # noqa: E402
from release_corpus import canonical_json_bytes, sha256_bytes  # noqa: E402
from qwen36_runtime_provenance import (  # noqa: E402
    SEARCH_RUNTIME_IMPLEMENTATION_FILES,
    runtime_provenance,
)
from search.hard import (  # noqa: E402
    exact_cost_assignment,
    optimize_cost_reinforce,
    realized_cost,
)
from search.streaming import StreamingHardCausalEvaluator  # noqa: E402


def _load_json(path: Path) -> Any:
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


def _assignment_bits(assignment: torch.Tensor) -> str:
    return "".join(str(int(value)) for value in assignment.tolist())


def _fold_for_step(run_index: int, step: int, fold_count: int) -> int:
    if min(run_index, step) < 0 or fold_count < 1:
        raise ValueError("run, step, and fold count are outside their domains")
    return (run_index + step) % fold_count


def _json_optimizer_checkpoint(state: dict[str, Any]) -> dict[str, Any]:
    optimizer = state["optimizer_state"]
    return {
        "completed_steps": int(state["completed_steps"]),
        "scores": state["scores"].tolist(),
        "baseline": float(state["baseline"]),
        "incumbent_assignment": state["incumbent_assignment"].tolist(),
        "incumbent_loss": float(state["incumbent_loss"]),
        "optimizer_state": {
            "step": int(optimizer["step"]),
            "exp_avg": optimizer["exp_avg"].tolist(),
            "exp_avg_sq": optimizer["exp_avg_sq"].tolist(),
        },
    }


def _drop_memory_measurements(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_memory_measurements(item)
            for key, item in value.items()
            if key != "memory"
        }
    if isinstance(value, list):
        return [_drop_memory_measurements(item) for item in value]
    return value


def _replay_projection(report: dict[str, Any]) -> dict[str, Any]:
    """Select deterministic search evidence for independent replay."""
    ignored = {
        "problem", "problem_sha256", "replay", "status", "warning",
        "wall_seconds_this_invocation",
    }
    return _drop_memory_measurements({
        key: value for key, value in report.items() if key not in ignored
    })


def _replay_problem(problem: dict[str, Any]) -> dict[str, Any]:
    """Return the problem identity shared by primary and replay runs."""
    return {
        key: value for key, value in problem.items()
        if key != "replay_reference_sha256"
    }


def search(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda is unavailable")
    model_dir = args.model_dir.resolve(strict=True)
    identity_path = args.identity.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    policy_path = args.policy.resolve(strict=True)
    calibration_path = args.calibration.resolve(strict=True)
    gguf_path = args.gguf.resolve(strict=True)
    gguf_python = args.gguf_python.resolve(strict=True)
    store_path = args.store.resolve(strict=True)
    ggml_library = args.ggml_library.resolve(strict=True)
    output_path = args.output.resolve()
    reference_path = (
        None if args.reference_report is None
        else args.reference_report.resolve(strict=True))
    reference_report = (
        None if reference_path is None else _load_json(reference_path))
    reference_sha256 = (
        None if reference_path is None else _sha256_file(reference_path))
    if reference_path == output_path:
        raise RuntimeError("replay reference and output paths must differ")
    if reference_report is not None and reference_report.get("status") != (
        "promotion_complete_pending_independent_replay"
    ):
        raise RuntimeError(
            "replay reference has not completed full-corpus promotion")
    identity = _load_json(identity_path)
    manifest = _load_json(manifest_path)
    policy = _load_json(policy_path)
    calibration = _load_json(calibration_path)
    calibration_manifest = calibration["canonical_manifest"]
    calibration_manifest_sha256 = sha256_bytes(
        canonical_json_bytes(calibration_manifest))
    if calibration.get("canonical_manifest_sha256") != (
        calibration_manifest_sha256
    ):
        raise RuntimeError("calibration canonical manifest hash differs")
    tokens_path = Path(calibration_manifest["tokens"]["path"]).resolve(strict=True)
    if _sha256_file(tokens_path) != calibration_manifest["tokens"]["sha256"]:
        raise RuntimeError("calibration token file hash differs")
    token_rows = _load_json(tokens_path)
    input_ids = torch.tensor(token_rows, dtype=torch.long)
    if input_ids.shape != (
        calibration_manifest["document_count"],
        calibration_manifest["token_count_per_document"],
    ):
        raise RuntimeError("calibration token matrix shape differs")
    strata = sorted(calibration_manifest["stratum_document_counts"])
    by_stratum = {
        stratum: [
            index for index, document in enumerate(
                calibration_manifest["documents"])
            if document["stratum"] == stratum
        ]
        for stratum in strata
    }
    if any(len(indices) != 10 for indices in by_stratum.values()):
        raise RuntimeError("calibration strata do not each contain ten documents")
    if args.documents_per_evaluation != len(strata):
        raise RuntimeError(
            "documents-per-evaluation must equal the number of strata so every "
            "optimization fold contains exactly one document per stratum")
    folds = [
        [by_stratum[stratum][fold] for stratum in strata]
        for fold in range(10)
    ]

    problem = {
        "dense_revision": identity["revision"],
        "identity_sha256": _sha256_file(identity_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "policy_sha256": _sha256_file(policy_path),
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "calibration_token_sha256": _sha256_file(tokens_path),
        "gguf_sha256": _sha256_file(gguf_path),
        "candidate_store_index_sha256": _sha256_file(
            store_path / "native-candidate-index.json"),
        "runtime_provenance": runtime_provenance(
            repository=Path(__file__).resolve().parents[1],
            gguf_python=gguf_python, ggml_library=ggml_library,
            implementation_files=(
                *SEARCH_RUNTIME_IMPLEMENTATION_FILES,
                "src/release_corpus.py",
            )),
        "device": str(device),
        "rows_per_chunk": args.rows_per_chunk,
        "vocab_chunk_size": args.vocab_chunk_size,
        "documents_per_evaluation": args.documents_per_evaluation,
        "stratified_folds": folds,
    }
    if reference_sha256 is not None:
        problem["replay_reference_sha256"] = reference_sha256
        reference_problem = reference_report.get("problem")
        if not isinstance(reference_problem, dict) or _replay_problem(
            reference_problem
        ) != _replay_problem(problem):
            raise RuntimeError(
                "replay reference describes a different search problem")
    problem_sha256 = hashlib.sha256(json.dumps(
        problem, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if output_path.exists():
        report = _load_json(output_path)
        if report.get("problem_sha256") != problem_sha256:
            raise RuntimeError("existing search report describes another problem")
    else:
        report = {
            "schema": "rco.qwen36.gsq_hard_search.v2",
            "status": "in_progress",
            "scope": (
                "exact-hard search over stratified folds of the complete "
                "calibration corpus using policy-admitted precision upgrades "
                "of the authentic GSQ incumbent"),
            "problem": problem,
            "problem_sha256": problem_sha256,
            "calibration": {
                "document_count": input_ids.shape[0],
                "sequence_length": input_ids.shape[1],
                "predicted_token_count": calibration_manifest[
                    "predicted_token_count"],
                "stratum_document_counts": calibration_manifest[
                    "stratum_document_counts"],
                "token_sequence_sha256": calibration_manifest[
                    "token_sequence_sha256"],
            },
            "baselines": {},
            "runs": [],
        }
        _atomic_json(output_path, report)

    codec = GGMLNativeCodec(ggml_library)
    native_store = NativeCandidateStore(store_path, codec)
    native_weight_store = NativeManifestWeightStore(
        native_store, manifest, model_dir, rows_per_chunk=args.rows_per_chunk)
    upgrade_store = GSQUpgradeWeightStore(
        policy, manifest, model_dir, native_weight_store,
        rows_per_chunk=args.rows_per_chunk)
    names = list(upgrade_store.names)
    if len(names) != policy["inventory"]["admitted_upgrade_count"]:
        raise RuntimeError("runtime and policy upgrade inventories differ")
    low_costs = [0] * len(names)
    high_costs = [upgrade_store.incremental_gguf_bytes(name) for name in names]
    target_cost = policy["budget"][
        "maximum_reachable_allowance_under_comparison_cap_bytes"]
    # Fail before loading the model if the policy target is not actually exact.
    projected = exact_cost_assignment(
        torch.zeros(len(names)), low_costs, high_costs, target_cost)
    if realized_cost(projected, low_costs, high_costs) != target_cost:
        raise RuntimeError("policy target projection is not exact")
    budget = {
        "mandatory_floor_file_bytes": policy["budget"][
            "mandatory_floor_file_bytes"],
        "target_incremental_gguf_bytes": target_cost,
        "target_complete_file_bytes": (
            policy["budget"]["mandatory_floor_file_bytes"] + target_cost),
        "comparison_cap_bytes": policy["budget"][
            "comparison_prototype_file_cap_bytes"],
        "unreachable_slack_bytes": policy["budget"][
            "unreachable_slack_below_comparison_cap_bytes"],
        "tensor_names": names,
        "upgrade_incremental_gguf_bytes": high_costs,
    }
    if report.get("budget") not in (None, budget):
        raise RuntimeError("existing report budget differs")
    report["budget"] = budget

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(
            config, attn_implementation="eager")
    model.eval()
    loader = GGUFManifestPrefixLoader(
        gguf_path, manifest, model_dir,
        gguf_python=gguf_python, ggml_library=ggml_library,
        rows_per_chunk=args.rows_per_chunk)
    groups = [SimpleNamespace(layer_names=(name,)) for name in names]
    evaluator = StreamingHardCausalEvaluator(
        model, loader, upgrade_store, groups, [0, 1], device=device,
        vocab_chunk_size=args.vocab_chunk_size,
        checkpoint_dtype=torch.bfloat16)

    def ensure_baseline(fold_index: int) -> dict[str, Any]:
        key = str(fold_index)
        if key in report["baselines"]:
            return report["baselines"][key]
        fold_input = input_ids[folds[fold_index]]
        baseline_evaluation = evaluator.evaluate(
            fold_input, torch.zeros(len(names), dtype=torch.long))
        expected_tokens = args.documents_per_evaluation * (
            input_ids.shape[1] - 1)
        if baseline_evaluation.token_count != expected_tokens:
            raise RuntimeError("calibration fold baseline token count differs")
        baseline = {
            "assignment": "authentic_retain",
            "fold_index": fold_index,
            "document_indices": folds[fold_index],
            "mean_nll": baseline_evaluation.loss,
            "predicted_token_count": baseline_evaluation.token_count,
            "document_mean_nll": list(
                baseline_evaluation.document_mean_nll),
            "document_token_counts": list(
                baseline_evaluation.document_token_counts),
            "memory": asdict(baseline_evaluation.memory),
        }
        report["baselines"][key] = baseline
        _atomic_json(output_path, report)
        print(json.dumps({
            "control": "authentic_retain",
            "mean_nll": baseline_evaluation.loss,
            "predicted_token_count": baseline_evaluation.token_count,
            "seconds": baseline_evaluation.memory.total_seconds,
            "peak_rss_bytes": baseline_evaluation.memory.process_peak_rss,
        }, sort_keys=True), flush=True)
        return baseline

    if args.baseline_only:
        ensure_baseline(args.baseline_fold)
        report["status"] = "baseline_complete"
        report["wall_seconds_this_invocation"] = time.perf_counter() - started
        _atomic_json(output_path, report)
        return report

    search_config = {
        "steps_per_run": args.steps,
        "repeat_runs": args.repeat_runs,
        "seed_start": args.seed_start,
        "learning_rate": args.learning_rate,
        "baseline_decay": args.baseline_decay,
        "fold_schedule": "(run_index + optimizer_step) modulo 10",
        "objective": "delta_mean_nll_from_authentic_fold_baseline",
        "promotion": "exact hard loss on all 50 calibration documents",
    }
    if report.get("search_config") not in (None, search_config):
        raise RuntimeError("existing report search configuration differs")
    report["search_config"] = search_config
    if len(report["runs"]) > args.repeat_runs:
        raise RuntimeError("report contains more runs than requested")
    if any(run["index"] != index for index, run in enumerate(report["runs"])):
        raise RuntimeError("report run indices are not contiguous")
    if any(run.get("status") != "complete" for run in report["runs"][:-1]):
        raise RuntimeError("only the final recorded run may be incomplete")

    for run_index in range(args.repeat_runs):
        already_complete = False
        if run_index < len(report["runs"]):
            run = report["runs"][run_index]
            already_complete = run.get("status") == "complete"
        else:
            run = {
                "index": run_index,
                "seed": args.seed_start + run_index,
                "status": "in_progress",
                "evaluations": [],
                "history": [],
            }
            report["runs"].append(run)
            _atomic_json(output_path, report)
        if run.get("seed") != args.seed_start + run_index:
            raise RuntimeError("recorded run seed differs")
        evaluations = run["evaluations"]
        checkpoint = run.get("optimizer_checkpoint")
        completed_steps = (
            0 if checkpoint is None else int(checkpoint["completed_steps"]))
        if not 0 <= completed_steps <= args.steps:
            raise RuntimeError("optimizer checkpoint step is outside run")
        if len(run.get("history", [])) != completed_steps:
            raise RuntimeError("run history and optimizer checkpoint differ")
        for record_index, record in enumerate(evaluations):
            recorded_assignment = torch.tensor(
                record["assignment"], dtype=torch.long)
            if recorded_assignment.shape != (len(names),):
                raise RuntimeError("recorded assignment shape differs")
            if not torch.all(
                (recorded_assignment == 0) | (recorded_assignment == 1)
            ):
                raise RuntimeError("recorded assignment is not binary")
            if record["index"] != record_index:
                raise RuntimeError("evaluation indices are not contiguous")
            if record["assignment_bits"] != _assignment_bits(
                recorded_assignment
            ):
                raise RuntimeError("recorded assignment bits differ")
            if realized_cost(
                recorded_assignment, low_costs, high_costs,
            ) != target_cost:
                raise RuntimeError("recorded assignment violates exact budget")
            recorded_fold = int(record["fold_index"])
            if not 0 <= recorded_fold < len(folds):
                raise RuntimeError("recorded fold is outside calibration set")
            expected_fold = _fold_for_step(
                run_index, int(record["optimizer_step"]), len(folds))
            if recorded_fold != expected_fold:
                raise RuntimeError("recorded fold violates the step schedule")
            if record["document_indices"] != folds[recorded_fold]:
                raise RuntimeError("recorded fold documents differ")
            try:
                baseline_loss = report["baselines"][str(recorded_fold)][
                    "mean_nll"]
            except KeyError as error:
                raise RuntimeError("recorded evaluation lacks its baseline") from error
            if record["objective_loss"] != record["mean_nll"] - baseline_loss:
                raise RuntimeError("recorded normalized objective differs")
        if checkpoint is not None:
            if len(checkpoint["scores"]) != len(names):
                raise RuntimeError("checkpoint score count differs")
            incumbent = torch.tensor(
                checkpoint["incumbent_assignment"], dtype=torch.long)
            if incumbent.shape != (len(names),) or realized_cost(
                incumbent, low_costs, high_costs,
            ) != target_cost:
                raise RuntimeError("checkpoint incumbent violates exact budget")
            if not run["history"] or run["history"][-1][
                "incumbent_loss"
            ] != checkpoint["incumbent_loss"]:
                raise RuntimeError("checkpoint incumbent and history differ")
        if already_complete:
            if checkpoint is None or completed_steps != args.steps:
                raise RuntimeError("complete run lacks a final checkpoint")
            if run.get("selected_assignment") != checkpoint[
                "incumbent_assignment"
            ]:
                raise RuntimeError("complete run selection differs from checkpoint")
            continue
        cached = {
            (int(record["fold_index"]), record["assignment_bits"]): record
            for record in evaluations
        }
        evaluation_call = completed_steps * 2

        def evaluate(assignment: torch.Tensor) -> float:
            nonlocal evaluation_call
            step = evaluation_call // 2
            fold_index = _fold_for_step(run_index, step, len(folds))
            baseline = ensure_baseline(fold_index)
            bits = _assignment_bits(assignment)
            cache_key = (fold_index, bits)
            evaluation_call += 1
            if cache_key in cached:
                return float(cached[cache_key]["objective_loss"])
            fold_input_ids = input_ids[folds[fold_index]]
            cost = realized_cost(assignment, low_costs, high_costs)
            if cost != target_cost:
                raise RuntimeError(
                    f"optimizer emitted cost {cost}, expected {target_cost}")
            evaluation = evaluator.evaluate(fold_input_ids, assignment)
            record = {
                "index": len(evaluations),
                "optimizer_step": step,
                "fold_index": fold_index,
                "document_indices": folds[fold_index],
                "assignment_bits": bits,
                "assignment": assignment.tolist(),
                "upgrade_count": int(assignment.sum()),
                "incremental_gguf_bytes": cost,
                "complete_file_bytes": (
                    budget["mandatory_floor_file_bytes"] + cost),
                "mean_nll": evaluation.loss,
                "delta_mean_nll_from_authentic": (
                    evaluation.loss - baseline["mean_nll"]),
                "objective_loss": evaluation.loss - baseline["mean_nll"],
                "predicted_token_count": evaluation.token_count,
                "document_mean_nll": list(evaluation.document_mean_nll),
                "document_token_counts": list(evaluation.document_token_counts),
                "memory": asdict(evaluation.memory),
            }
            evaluations.append(record)
            cached[cache_key] = record
            _atomic_json(output_path, report)
            print(json.dumps({
                "run": run_index,
                "evaluation": record["index"],
                "upgrade_count": record["upgrade_count"],
                "mean_nll": record["mean_nll"],
                "delta_mean_nll_from_authentic": record[
                    "delta_mean_nll_from_authentic"],
                "seconds": record["memory"]["total_seconds"],
            }, sort_keys=True), flush=True)
            return record["objective_loss"]

        def save_step(state: dict[str, Any]) -> None:
            run["optimizer_checkpoint"] = _json_optimizer_checkpoint(state)
            run["history"].append(state["history_entry"])
            _atomic_json(output_path, report)

        continuation: dict[str, Any] = {}
        if checkpoint is not None:
            continuation = {
                "initial_scores": torch.tensor(
                    checkpoint["scores"], dtype=torch.float32),
                "initial_incumbent_assignment": torch.tensor(
                    checkpoint["incumbent_assignment"], dtype=torch.long),
                "initial_incumbent_loss": checkpoint["incumbent_loss"],
                "initial_step": completed_steps,
                "initial_baseline": checkpoint["baseline"],
                "initial_optimizer_state": checkpoint["optimizer_state"],
            }
        if completed_steps == args.steps:
            if checkpoint is None:
                raise RuntimeError("completed run lacks optimizer checkpoint")
            scores = torch.tensor(checkpoint["scores"], dtype=torch.float32)
            assignment = torch.tensor(
                checkpoint["incumbent_assignment"], dtype=torch.long)
        else:
            scores, assignment, _ = optimize_cost_reinforce(
                evaluate,
                low_costs=low_costs,
                high_costs=high_costs,
                target_cost=target_cost,
                n_steps=args.steps - completed_steps,
                lr=args.learning_rate,
                baseline_decay=args.baseline_decay,
                seed=args.seed_start + run_index,
                step_callback=save_step,
                log_interval=args.steps + 1,
                **continuation,
            )
        selected_bits = _assignment_bits(assignment)
        selected_records = [
            record for record in evaluations
            if record["assignment_bits"] == selected_bits
            and record["objective_loss"]
            == run["optimizer_checkpoint"]["incumbent_loss"]
        ]
        if not selected_records:
            raise RuntimeError("incumbent evaluation record is missing")
        selected_record = selected_records[0]
        run.update({
            "status": "complete",
            "scores": [float(value) for value in scores],
            "selected_assignment_bits": selected_bits,
            "selected_assignment": assignment.tolist(),
            "selected_upgrade_count": int(assignment.sum()),
            "selected_incremental_gguf_bytes": realized_cost(
                assignment, low_costs, high_costs),
            "selected_screen_mean_nll": selected_record["mean_nll"],
            "selected_screen_fold_index": selected_record["fold_index"],
            "selected_screen_delta_mean_nll_from_authentic": (
                run["optimizer_checkpoint"]["incumbent_loss"]),
        })
        _atomic_json(output_path, report)

    report["provisional_candidates"] = [
        {
            "run_index": run["index"],
            "assignment_bits": run["selected_assignment_bits"],
            "assignment": run["selected_assignment"],
            "upgrade_count": run["selected_upgrade_count"],
            "incremental_gguf_bytes": run["selected_incremental_gguf_bytes"],
            "screen_fold_index": run["selected_screen_fold_index"],
            "screen_delta_mean_nll_from_authentic": run[
                "selected_screen_delta_mean_nll_from_authentic"],
        }
        for run in report["runs"]
    ]
    report["status"] = "search_complete_pending_full_corpus_promotion"
    if args.promote_full_corpus:
        if "full_corpus_baseline" not in report:
            baseline_evaluation = evaluator.evaluate(
                input_ids, torch.zeros(len(names), dtype=torch.long))
            if baseline_evaluation.token_count != calibration_manifest[
                "predicted_token_count"
            ]:
                raise RuntimeError("full-corpus baseline token count differs")
            report["full_corpus_baseline"] = {
                "assignment": "authentic_retain",
                "mean_nll": baseline_evaluation.loss,
                "predicted_token_count": baseline_evaluation.token_count,
                "document_mean_nll": list(
                    baseline_evaluation.document_mean_nll),
                "document_token_counts": list(
                    baseline_evaluation.document_token_counts),
                "memory": asdict(baseline_evaluation.memory),
            }
            _atomic_json(output_path, report)
        promotions = report.setdefault("full_corpus_promotions", [])
        promoted_bits = {item["assignment_bits"] for item in promotions}
        for candidate in report["provisional_candidates"]:
            if candidate["assignment_bits"] in promoted_bits:
                continue
            assignment = torch.tensor(candidate["assignment"], dtype=torch.long)
            evaluation = evaluator.evaluate(input_ids, assignment)
            if evaluation.token_count != calibration_manifest[
                "predicted_token_count"
            ]:
                raise RuntimeError("full-corpus promotion token count differs")
            document_deltas = [
                candidate_loss - baseline_loss
                for candidate_loss, baseline_loss in zip(
                    evaluation.document_mean_nll,
                    report["full_corpus_baseline"]["document_mean_nll"],
                    strict=True)
            ]
            promotion = {
                **candidate,
                "mean_nll": evaluation.loss,
                "delta_mean_nll_from_authentic": (
                    evaluation.loss
                    - report["full_corpus_baseline"]["mean_nll"]),
                "predicted_token_count": evaluation.token_count,
                "document_mean_nll": list(evaluation.document_mean_nll),
                "document_token_counts": list(evaluation.document_token_counts),
                "document_delta_mean_nll_from_authentic": document_deltas,
                "document_improvement_count": sum(
                    delta < 0.0 for delta in document_deltas),
                "memory": asdict(evaluation.memory),
            }
            promotions.append(promotion)
            promoted_bits.add(candidate["assignment_bits"])
            _atomic_json(output_path, report)
        report["selected"] = min(
            promotions, key=lambda item: item["mean_nll"])
        report["status"] = "promotion_complete_pending_independent_replay"
    if reference_report is not None:
        if _sha256_file(reference_path) != reference_sha256:
            raise RuntimeError("replay reference changed during execution")
        expected = _replay_projection(reference_report)
        actual = _replay_projection(report)
        if actual != expected:
            raise RuntimeError(
                "independent replay differs from the reference trajectory")
        projection_sha256 = hashlib.sha256(json.dumps(
            actual, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        report["replay"] = {
            "passed": True,
            "reference_path": str(reference_path),
            "reference_sha256": reference_sha256,
            "deterministic_projection_sha256": projection_sha256,
        }
        report["status"] = (
            "independent_replay_complete_pending_release_gates")
    report["wall_seconds_this_invocation"] = time.perf_counter() - started
    pending_promotion = (
        "full-corpus promotion, "
        if not args.promote_full_corpus else ""
    )
    report["warning"] = (
        "Search completion is not release authorization. Exact "
        + pending_promotion
        + "independent hard replay, pinned-llama.cpp parity, held-out quality, "
        "and deterministic generation gates remain required."
    )
    _atomic_json(output_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--gguf-python", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeat-runs", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--baseline-decay", type=float, default=0.9)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--promote-full-corpus", action="store_true")
    parser.add_argument("--documents-per-evaluation", type=int, default=5)
    parser.add_argument("--baseline-fold", type=int, default=0)
    args = parser.parse_args()
    if min(
        args.rows_per_chunk, args.vocab_chunk_size, args.steps,
        args.repeat_runs, args.documents_per_evaluation,
    ) < 1:
        parser.error("chunk sizes, steps, and repeats must be positive")
    if not 0 <= args.baseline_fold < 10:
        parser.error("--baseline-fold must be between 0 and 9")
    if not args.baseline_only and args.steps < 10:
        parser.error("--steps must be at least 10 so every run covers all folds")
    if not args.baseline_only and args.repeat_runs < 2:
        parser.error("--repeat-runs must be at least 2 for independent searches")
    if args.reference_report is not None and not args.promote_full_corpus:
        parser.error("--reference-report requires --promote-full-corpus")
    if args.reference_report is not None and args.baseline_only:
        parser.error("--reference-report cannot be used with --baseline-only")
    return args


if __name__ == "__main__":
    result = search(_parse_args())
    print(json.dumps({
        "status": result["status"],
        "baseline_mean_nll_by_fold": {
            key: value["mean_nll"]
            for key, value in result["baselines"].items()
        },
        "completed_runs": sum(
            run.get("status") == "complete" for run in result["runs"]),
        "selected": result.get("selected"),
        "provisional_candidates": result.get("provisional_candidates"),
    }, indent=2, sort_keys=True))
