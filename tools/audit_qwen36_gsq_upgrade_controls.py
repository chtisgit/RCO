#!/usr/bin/env python3
"""Test genuine precision upgrades over the untouched GSQ-hybrid baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import transformers
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gguf_checkpoint_stream import GGUFManifestPrefixLoader  # noqa: E402
from native_gguf import import_pinned_gguf  # noqa: E402
from native_runtime import NativeManifestWeightStore  # noqa: E402
from native_store import NATIVE_CANDIDATE_INDEX, NativeCandidateStore  # noqa: E402
from quant.ggml_native import GGMLNativeCodec, GGMLType  # noqa: E402
from search.streaming import StreamingHardCausalEvaluator  # noqa: E402


CALIBRATION_TEXT = (
    "A bounded native GGML search streams every Qwen decoder block while "
    "evaluating an exact serialized byte budget."
)
CONTROL_ORDER = (
    "routed_expert_q4",
    "output_head_bf16",
    "linear_attention_bf16",
    "self_attention_bf16",
    "shared_expert_bf16",
    "all_q8_bf16",
    "all_eligible_upgrades",
)
ROUTED_SUBFAMILY_CONTROLS = (
    "routed_gate_q4",
    "routed_up_q4",
    "routed_down_q4",
)
ROUTED_COMBINATION_CONTROLS = ("routed_gate_up_q4",)
ALL_CONTROLS = (
    CONTROL_ORDER + ROUTED_SUBFAMILY_CONTROLS + ROUTED_COMBINATION_CONTROLS)
LOGIT_THRESHOLDS = {
    "maximum_position_relative_rmse": 0.20,
    "minimum_position_cosine_similarity": 0.98,
    "minimum_top1_agreement_fraction": 2 / 3,
}


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


def _logit_comparison(
    candidate: np.ndarray, reference: np.ndarray,
) -> dict[str, Any]:
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise ValueError("candidate and reference logits must have equal 2-D shape")
    positions = []
    top1_matches = 0
    for position, (actual, expected) in enumerate(zip(candidate, reference)):
        delta = actual.astype(np.float64) - expected.astype(np.float64)
        rmse = float(np.sqrt(np.mean(np.square(delta))))
        mae = float(np.mean(np.abs(delta)))
        reference_rms = float(np.sqrt(np.mean(np.square(
            expected.astype(np.float64)))))
        relative_rmse = rmse / reference_rms if reference_rms else float("inf")
        actual64 = actual.astype(np.float64)
        expected64 = expected.astype(np.float64)
        denominator = float(np.linalg.norm(actual64) * np.linalg.norm(expected64))
        cosine = float(np.dot(actual64, expected64) / denominator) if denominator else 0.0
        actual_top1 = int(np.argmax(actual))
        expected_top1 = int(np.argmax(expected))
        top1_matches += actual_top1 == expected_top1
        positions.append({
            "position": position,
            "mean_absolute_error": mae,
            "rmse": rmse,
            "reference_rms": reference_rms,
            "relative_rmse": relative_rmse,
            "cosine_similarity": cosine,
            "candidate_top1": actual_top1,
            "reference_top1": expected_top1,
            "top1_agrees": actual_top1 == expected_top1,
        })
    top1_fraction = top1_matches / len(positions)
    maximum_relative_rmse = max(item["relative_rmse"] for item in positions)
    minimum_cosine = min(item["cosine_similarity"] for item in positions)
    passed = (
        maximum_relative_rmse
        <= LOGIT_THRESHOLDS["maximum_position_relative_rmse"]
        and minimum_cosine
        >= LOGIT_THRESHOLDS["minimum_position_cosine_similarity"]
        and top1_fraction
        >= LOGIT_THRESHOLDS["minimum_top1_agreement_fraction"]
    )
    return {
        "positions": positions,
        "maximum_position_relative_rmse": maximum_relative_rmse,
        "minimum_position_cosine_similarity": minimum_cosine,
        "top1_agreement_fraction": top1_fraction,
        "passed": passed,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda is unavailable")
    model_dir = args.model_dir.resolve(strict=True)
    identity_path = args.identity.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    gguf_path = args.gguf.resolve(strict=True)
    gguf_python = args.gguf_python.resolve(strict=True)
    ggml_library = args.ggml_library.resolve(strict=True)
    store_path = args.store.resolve(strict=True)
    output_path = args.output.resolve()
    identity = _load_json(identity_path)
    manifest = _load_json(manifest_path)
    entries = sorted(
        (entry for entry in manifest["entries"] if entry.get("rco_search")),
        key=lambda entry: entry["destination_name"],
    )
    if len(entries) != 512:
        raise RuntimeError("searchable manifest inventory differs")

    gguf = import_pinned_gguf(gguf_python)
    reader = gguf.GGUFReader(gguf_path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    original_types = {
        entry["destination_name"]: tensors[
            entry["destination_name"]].tensor_type.name
        for entry in entries
    }
    type_counts = Counter(original_types.values())
    if type_counts != Counter({
        "Q2_0": 120, "Q8_0": 251, "F32": 80, "BF16": 61,
    }):
        raise RuntimeError(f"unexpected original searchable types: {type_counts}")
    routed_entries = [
        entry for entry in entries
        if original_types[entry["destination_name"]] == "Q2_0"
    ]
    q8_entries = [
        entry for entry in entries
        if original_types[entry["destination_name"]] == "Q8_0"
    ]
    if any(entry["source_category"] != "routed_expert"
           for entry in routed_entries):
        raise RuntimeError("original Q2 inventory is not exactly routed experts")
    routed_by_control = {
        "routed_expert_q4": routed_entries,
        "all_eligible_upgrades": routed_entries,
        "routed_gate_q4": [entry for entry in routed_entries
                            if ".ffn_gate_exps." in entry["destination_name"]],
        "routed_up_q4": [entry for entry in routed_entries
                          if ".ffn_up_exps." in entry["destination_name"]],
        "routed_down_q4": [entry for entry in routed_entries
                            if ".ffn_down_exps." in entry["destination_name"]],
    }
    if {label: len(routed_by_control[label])
            for label in ROUTED_SUBFAMILY_CONTROLS} != {
                label: 40 for label in ROUTED_SUBFAMILY_CONTROLS}:
        raise RuntimeError("routed Q4 subfamily inventory differs")
    routed_by_control["routed_gate_up_q4"] = [
        *routed_by_control["routed_gate_q4"],
        *routed_by_control["routed_up_q4"],
    ]
    if len(routed_by_control["routed_gate_up_q4"]) != 80:
        raise RuntimeError("routed gate+up Q4 inventory differs")
    q8_by_control = {
        "output_head_bf16": [
            entry for entry in q8_entries if entry["source_category"] == "lm_head"],
        "linear_attention_bf16": [
            entry for entry in q8_entries
            if entry["source_category"] == "linear_attention"],
        "self_attention_bf16": [
            entry for entry in q8_entries
            if entry["source_category"] == "self_attention"],
        "shared_expert_bf16": [
            entry for entry in q8_entries
            if entry["source_category"] == "shared_expert"],
        "all_q8_bf16": q8_entries,
        "all_eligible_upgrades": q8_entries,
    }
    expected_counts = {
        "output_head_bf16": 1,
        "linear_attention_bf16": 90,
        "self_attention_bf16": 40,
        "shared_expert_bf16": 120,
        "all_q8_bf16": 251,
        "all_eligible_upgrades": 251,
    }
    if {name: len(values) for name, values in q8_by_control.items()} != expected_counts:
        raise RuntimeError("Q8 upgrade-family inventory differs")
    override_sources = {
        label: sorted({entry["source_name"] for entry in values})
        for label, values in q8_by_control.items()
    }
    # A fused dense source must never be only partially selected.
    entries_by_source: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        entries_by_source.setdefault(entry["source_name"], []).append(entry)
    for label, sources in override_sources.items():
        selected_names = {
            entry["destination_name"] for entry in q8_by_control[label]}
        for source in sources:
            source_names = {
                entry["destination_name"] for entry in entries_by_source[source]}
            if not source_names <= selected_names:
                raise RuntimeError(
                    f"{label} partially selects fused source {source}")

    codec = GGMLNativeCodec(ggml_library)
    store = NativeCandidateStore(store_path, codec)
    for entry in routed_entries:
        metadata = store.metadata(entry["destination_name"], GGMLType.Q4_0)
        if metadata["provenance"]["kind"] != "bf16_native_quantization":
            raise RuntimeError("routed Q4 upgrade is not BF16-derived")
    weight_store = NativeManifestWeightStore(
        store, {"entries": entries}, model_dir,
        rows_per_chunk=args.rows_per_chunk)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    input_ids = tokenizer(
        CALIBRATION_TEXT, return_tensors="pt", add_special_tokens=False,
        truncation=True, max_length=args.sequence_length,
    )["input_ids"]
    if input_ids.shape[1] != args.sequence_length:
        raise RuntimeError("calibration text is too short")
    controls = tuple(args.controls)
    problem = {
        "dense_revision": identity["revision"],
        "identity_sha256": _sha256_file(identity_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "gguf_sha256": _sha256_file(gguf_path),
        "candidate_store_index_sha256": _sha256_file(
            store_path / NATIVE_CANDIDATE_INDEX),
        "original_searchable_type_counts": dict(sorted(type_counts.items())),
        "controls": list(controls),
        "input_ids": input_ids.tolist()[0],
        "device": str(device),
        "rows_per_chunk": args.rows_per_chunk,
        "vocab_chunk_size": args.vocab_chunk_size,
    }
    problem_sha256 = hashlib.sha256(json.dumps(
        problem, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if output_path.exists():
        report = _load_json(output_path)
        if report.get("problem_sha256") != problem_sha256:
            raise RuntimeError("existing upgrade-control report describes another problem")
    else:
        report = {
            "schema": "rco.qwen36.gsq_upgrade_controls.v1",
            "status": "in_progress",
            "scope": (
                "complete-model native screens that preserve the untouched GSQ "
                "hybrid and apply only genuine BF16-derived precision upgrades"
            ),
            "problem": problem,
            "problem_sha256": problem_sha256,
            "policy_inventory": {
                "retain_payload_count": 512,
                "q2_to_q4_upgrade_count": len(routed_entries),
                "q8_to_bf16_upgrade_count": len(q8_entries),
                "mandatory_floor_count": type_counts["F32"] + type_counts["BF16"],
                "mandatory_floor_type_counts": {
                    "BF16": type_counts["BF16"], "F32": type_counts["F32"]},
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "results": {},
        }
        _atomic_json(output_path, report)

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)

    def evaluate(
        *, dense_sources: list[str], routed_q4_entries: list[dict[str, Any]],
        capture_logits: bool = False,
    ):
        with init_empty_weights(include_buffers=False):
            model = AutoModelForImageTextToText.from_config(
                config, attn_implementation="eager")
        model.eval()
        loader = GGUFManifestPrefixLoader(
            gguf_path,
            manifest,
            model_dir,
            gguf_python=gguf_python,
            ggml_library=ggml_library,
            dense_override_sources=dense_sources,
            rows_per_chunk=args.rows_per_chunk,
        )
        groups = (
            [SimpleNamespace(layer_names=(entry["destination_name"],))
             for entry in routed_q4_entries]
        )
        evaluator = StreamingHardCausalEvaluator(
            model, loader, weight_store, groups, [2, 4], device=device,
            vocab_chunk_size=args.vocab_chunk_size,
            checkpoint_dtype=torch.bfloat16)
        assignment = torch.ones(len(groups), dtype=torch.long)
        result = evaluator.evaluate(
            input_ids,
            assignment,
            capture_logit_positions=(
                tuple(range(input_ids.shape[1] - 1)) if capture_logits else ()),
        )
        max_chunk = loader.max_decoded_chunk_bytes
        del evaluator, loader, model
        gc.collect()
        return result, max_chunk

    if "authentic_gsq" not in report["results"]:
        if args.baseline_nll_report is not None:
            baseline_report = _load_json(args.baseline_nll_report.resolve(strict=True))
            baseline_problem = baseline_report.get("problem", {})
            for key in (
                "dense_revision", "identity_sha256", "manifest_sha256",
                "gguf_sha256", "candidate_store_index_sha256", "input_ids",
                "device", "rows_per_chunk", "vocab_chunk_size",
            ):
                if baseline_problem.get(key) != problem.get(key):
                    raise RuntimeError(f"baseline NLL report differs on {key}")
            baseline_result = baseline_report.get("results", {}).get(
                "authentic_gsq")
            if baseline_result is None:
                raise RuntimeError("baseline NLL report lacks authentic_gsq")
            report["results"]["authentic_gsq"] = baseline_result
            report["authentic_gsq_reused_from"] = {
                "path": str(args.baseline_nll_report.resolve()),
                "problem_sha256": baseline_report["problem_sha256"],
            }
        else:
            evaluation, max_chunk = evaluate(
                dense_sources=[], routed_q4_entries=[])
            report["results"]["authentic_gsq"] = {
                "mean_nll": evaluation.loss,
                "predicted_token_count": evaluation.token_count,
                "max_decoded_chunk_bytes": max_chunk,
                "memory": asdict(evaluation.memory),
            }
        _atomic_json(output_path, report)
        print(json.dumps({
            "control": "authentic_gsq",
            "mean_nll": report["results"]["authentic_gsq"]["mean_nll"],
            "reused": args.baseline_nll_report is not None,
        }, sort_keys=True), flush=True)
    baseline = float(report["results"]["authentic_gsq"]["mean_nll"])

    new_control_count = 0
    fresh_logits: dict[str, np.ndarray] = {}
    for label in controls:
        if label in report["results"]:
            continue
        dense_sources = override_sources.get(label, [])
        evaluation, max_chunk = evaluate(
            dense_sources=dense_sources,
            routed_q4_entries=routed_by_control.get(label, []),
            capture_logits=args.logit_output is not None,
        )
        if evaluation.captured_logits is not None:
            fresh_logits[label] = evaluation.captured_logits.numpy()
        delta = evaluation.loss - baseline
        result = {
            "mean_nll": evaluation.loss,
            "delta_mean_nll_from_authentic_gsq": delta,
            "bounded_by_plus_0_5_nll": delta <= 0.5,
            "improves_fixed_screen": delta < 0,
            "predicted_token_count": evaluation.token_count,
            "dense_override_source_count": len(dense_sources),
            "dense_override_tensor_count": len(q8_by_control.get(label, [])),
            "routed_q4_tensor_count": len(routed_by_control.get(label, [])),
            "max_decoded_chunk_bytes": max_chunk,
            "memory": asdict(evaluation.memory),
        }
        report["results"][label] = result
        _atomic_json(output_path, report)
        print(json.dumps({
            "control": label,
            "mean_nll": evaluation.loss,
            "delta_mean_nll_from_authentic_gsq": delta,
            "improves": delta < 0,
        }, sort_keys=True), flush=True)
        new_control_count += 1
        if (args.stop_after_new_controls is not None
                and new_control_count >= args.stop_after_new_controls):
            return report
    report["status"] = "complete"
    report["ranking_by_delta_mean_nll"] = sorted(
        controls,
        key=lambda label: report["results"][label][
            "delta_mean_nll_from_authentic_gsq"],
    )
    report["admitted_on_fixed_screen"] = [
        label for label in controls
        if report["results"][label]["bounded_by_plus_0_5_nll"]
    ]
    report["warning"] = (
        "Admission here is provisional native screening only. Search and final "
        "release still require pinned-llama.cpp replay and held-out gates."
    )
    report["wall_seconds_this_invocation"] = time.perf_counter() - started
    _atomic_json(output_path, report)

    if args.logit_output is not None:
        baseline_path = args.baseline_logits.resolve(strict=True)
        with np.load(baseline_path) as baseline_archive:
            baseline_tokens = np.asarray(baseline_archive["tokens"], dtype=np.int64)
            baseline_logits = np.asarray(
                baseline_archive["native"], dtype=np.float32)
        if baseline_tokens.tolist() != input_ids.tolist()[0]:
            raise RuntimeError("baseline-logit tokens differ from upgrade screen")
        expected_shape = (
            input_ids.shape[1] - 1, config.get_text_config().vocab_size)
        if baseline_logits.shape != expected_shape:
            raise RuntimeError(
                f"baseline logits have shape {baseline_logits.shape}, "
                f"expected {expected_shape}")
        logit_path = args.logit_output.resolve()
        logit_problem = {
            "nll_problem_sha256": problem_sha256,
            "baseline_logits_sha256": _sha256_file(baseline_path),
            "baseline_logits_key": "native",
            "input_ids": input_ids.tolist()[0],
            "controls": list(controls),
            "thresholds": LOGIT_THRESHOLDS,
        }
        logit_problem_sha256 = hashlib.sha256(json.dumps(
            logit_problem, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if logit_path.exists():
            logit_report = _load_json(logit_path)
            if logit_report.get("problem_sha256") != logit_problem_sha256:
                raise RuntimeError("existing logit report describes another problem")
        else:
            logit_report = {
                "schema": "rco.qwen36.gsq_upgrade_logits.v1",
                "status": "in_progress",
                "scope": (
                    "full-vocabulary native logit drift for genuine upgrades "
                    "relative to the authentic GSQ native replay"),
                "problem": logit_problem,
                "problem_sha256": logit_problem_sha256,
                "results": {},
            }
            _atomic_json(logit_path, logit_report)
        new_logit_count = 0
        for label in controls:
            if label in logit_report["results"]:
                continue
            if label in fresh_logits:
                candidate_logits = fresh_logits[label]
            else:
                evaluation, _ = evaluate(
                    dense_sources=override_sources.get(label, []),
                    routed_q4_entries=routed_by_control.get(label, []),
                    capture_logits=True,
                )
                candidate_logits = evaluation.captured_logits.numpy()
            comparison = _logit_comparison(
                candidate_logits, baseline_logits)
            comparison["nll_bounded_by_plus_0_5"] = report["results"][label][
                "bounded_by_plus_0_5_nll"]
            comparison["admitted_on_fixed_screen"] = (
                comparison["passed"]
                and comparison["nll_bounded_by_plus_0_5"])
            logit_report["results"][label] = comparison
            _atomic_json(logit_path, logit_report)
            print(json.dumps({
                "logit_control": label,
                "passed": comparison["passed"],
                "maximum_position_relative_rmse": comparison[
                    "maximum_position_relative_rmse"],
                "minimum_position_cosine_similarity": comparison[
                    "minimum_position_cosine_similarity"],
                "top1_agreement_fraction": comparison[
                    "top1_agreement_fraction"],
            }, sort_keys=True), flush=True)
            new_logit_count += 1
            if (args.stop_after_new_logit_controls is not None
                    and new_logit_count >= args.stop_after_new_logit_controls):
                return report
        logit_report["status"] = "complete"
        logit_report["admitted_on_fixed_screen"] = [
            label for label in controls
            if logit_report["results"][label]["admitted_on_fixed_screen"]
        ]
        logit_report["warning"] = (
            "These are native fixed-sequence drift bounds, not release proof. "
            "Pinned-llama.cpp replay and held-out gates remain mandatory."
        )
        _atomic_json(logit_path, logit_report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--gguf-python", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument("--controls", nargs="+", choices=ALL_CONTROLS,
                        default=list(CONTROL_ORDER))
    parser.add_argument("--stop-after-new-controls", type=int)
    parser.add_argument("--baseline-nll-report", type=Path)
    parser.add_argument("--baseline-logits", type=Path)
    parser.add_argument("--logit-output", type=Path)
    parser.add_argument("--stop-after-new-logit-controls", type=int)
    args = parser.parse_args()
    if min(args.sequence_length, args.vocab_chunk_size, args.rows_per_chunk) < 1:
        parser.error("invalid numeric argument")
    if len(set(args.controls)) != len(args.controls):
        parser.error("--controls must not contain duplicates")
    if (args.stop_after_new_controls is not None
            and args.stop_after_new_controls < 1):
        parser.error("--stop-after-new-controls must be positive")
    if (args.baseline_logits is None) != (args.logit_output is None):
        parser.error("--baseline-logits and --logit-output must be used together")
    if (args.stop_after_new_logit_controls is not None
            and args.stop_after_new_logit_controls < 1):
        parser.error("--stop-after-new-logit-controls must be positive")
    if (args.stop_after_new_logit_controls is not None
            and args.logit_output is None):
        parser.error("--stop-after-new-logit-controls requires --logit-output")
    return args


if __name__ == "__main__":
    parsed_args = _parse_args()
    result = audit(parsed_args)
    summary = {
        "status": result["status"],
        "completed": sorted(result["results"]),
        "nll_admitted_on_fixed_screen": result.get("admitted_on_fixed_screen"),
    }
    if parsed_args.logit_output is not None:
        logit_result = _load_json(parsed_args.logit_output.resolve())
        summary["logit_and_nll_admitted_on_fixed_screen"] = (
            logit_result.get("admitted_on_fixed_screen"))
    print(json.dumps(summary, indent=2, sort_keys=True))
