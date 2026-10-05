#!/usr/bin/env python3
"""Replay authentic retain and admitted-upgrade choices through the new runtime."""

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

import numpy as np
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_qwen36_gsq_upgrade_controls import (  # noqa: E402
    CALIBRATION_TEXT,
    LOGIT_THRESHOLDS,
    _logit_comparison,
)
from gguf_checkpoint_stream import GGUFManifestPrefixLoader  # noqa: E402
from gsq_upgrade_runtime import GSQUpgradeWeightStore  # noqa: E402
from native_runtime import NativeManifestWeightStore  # noqa: E402
from native_store import NativeCandidateStore  # noqa: E402
from quant.ggml_native import GGMLNativeCodec  # noqa: E402
from search.streaming import StreamingHardCausalEvaluator  # noqa: E402
from qwen36_runtime_provenance import (  # noqa: E402
    AUDIT_RUNTIME_IMPLEMENTATION_FILES,
    runtime_provenance,
    runtime_report_status,
)


CONTROL_ORDER = ("authentic_retain", "all_admitted_upgrades")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_authentic_loss(report: dict[str, Any]) -> float:
    try:
        return float(report["results"]["authentic_gsq"]["mean_nll"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "baseline controls lack authentic_gsq mean NLL") from error


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


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda is unavailable")
    model_dir = args.model_dir.resolve(strict=True)
    identity_path = args.identity.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    policy_path = args.policy.resolve(strict=True)
    gguf_path = args.gguf.resolve(strict=True)
    gguf_python = args.gguf_python.resolve(strict=True)
    store_path = args.store.resolve(strict=True)
    ggml_library = args.ggml_library.resolve(strict=True)
    baseline_logits_path = args.baseline_logits.resolve(strict=True)
    baseline_controls_path = args.baseline_controls.resolve(strict=True)
    output_path = args.output.resolve()
    identity = _load_json(identity_path)
    manifest = _load_json(manifest_path)
    policy = _load_json(policy_path)
    baseline_controls = _load_json(baseline_controls_path)
    expected_authentic_loss = _expected_authentic_loss(baseline_controls)
    with np.load(baseline_logits_path) as archive:
        baseline_tokens = np.asarray(archive["tokens"], dtype=np.int64)
        baseline_logits = np.asarray(archive["native"], dtype=np.float32)

    problem = {
        "identity_sha256": _sha256_file(identity_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "policy_sha256": _sha256_file(policy_path),
        "gguf_sha256": _sha256_file(gguf_path),
        "candidate_store_index_sha256": _sha256_file(
            store_path / "native-candidate-index.json"),
        "baseline_logits_sha256": _sha256_file(baseline_logits_path),
        "baseline_controls_sha256": _sha256_file(baseline_controls_path),
        "runtime_provenance": runtime_provenance(
            repository=Path(__file__).resolve().parents[1],
            gguf_python=gguf_python, ggml_library=ggml_library,
            implementation_files=AUDIT_RUNTIME_IMPLEMENTATION_FILES),
        "controls": list(args.controls),
        "device": str(device),
        "rows_per_chunk": args.rows_per_chunk,
        "vocab_chunk_size": args.vocab_chunk_size,
    }
    problem_sha256 = hashlib.sha256(json.dumps(
        problem, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if output_path.exists():
        report = _load_json(output_path)
        if report.get("problem_sha256") != problem_sha256:
            raise RuntimeError("existing runtime report describes another problem")
    else:
        report = {
            "schema": "rco.qwen36.gsq_upgrade_runtime.v1",
            "status": "partial",
            "scope": (
                "real-model integration replay of authentic retain no-ops and "
                "policy-admitted precision upgrades"),
            "problem": problem,
            "problem_sha256": problem_sha256,
            "results": {},
        }
        _atomic_json(output_path, report)
    report["status"] = runtime_report_status(report["results"])
    if report["status"] == "failed":
        raise RuntimeError(
            "existing runtime report contains a failed authentic-retain gate")
    _atomic_json(output_path, report)

    codec = GGMLNativeCodec(ggml_library)
    native_store = NativeCandidateStore(store_path, codec)
    native_weight_store = NativeManifestWeightStore(
        native_store, manifest, model_dir, rows_per_chunk=args.rows_per_chunk)
    upgrade_store = GSQUpgradeWeightStore(
        policy, manifest, model_dir, native_weight_store,
        rows_per_chunk=args.rows_per_chunk)
    names = list(upgrade_store.names)
    if len(names) != 331:
        raise RuntimeError(f"expected 331 admitted upgrades, found {len(names)}")
    groups = [SimpleNamespace(layer_names=(name,)) for name in names]
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    input_ids = tokenizer(
        CALIBRATION_TEXT, return_tensors="pt", add_special_tokens=False,
        truncation=True, max_length=len(baseline_tokens),
    )["input_ids"]
    if input_ids.tolist()[0] != baseline_tokens.tolist():
        raise RuntimeError("runtime control tokens differ from baseline logits")

    def evaluate(assignment: torch.Tensor):
        with init_empty_weights(include_buffers=False):
            model = AutoModelForImageTextToText.from_config(
                config, attn_implementation="eager")
        model.eval()
        loader = GGUFManifestPrefixLoader(
            gguf_path, manifest, model_dir,
            gguf_python=gguf_python, ggml_library=ggml_library,
            rows_per_chunk=args.rows_per_chunk)
        evaluator = StreamingHardCausalEvaluator(
            model, loader, upgrade_store, groups, [0, 1], device=device,
            vocab_chunk_size=args.vocab_chunk_size,
            checkpoint_dtype=torch.bfloat16)
        evaluation = evaluator.evaluate(
            input_ids, assignment,
            capture_logit_positions=tuple(range(input_ids.shape[1] - 1)))
        return evaluation

    for label in args.controls:
        if label in report["results"]:
            continue
        assignment = torch.full(
            (len(names),),
            0 if label == "authentic_retain" else 1,
            dtype=torch.long)
        evaluation = evaluate(assignment)
        candidate_logits = evaluation.captured_logits.numpy()
        comparison = _logit_comparison(candidate_logits, baseline_logits)
        upgrade_bytes = sum(
            upgrade_store.incremental_gguf_bytes(name)
            for name, choice in zip(names, assignment.tolist()) if choice)
        result = {
            "mean_nll": evaluation.loss,
            "predicted_token_count": evaluation.token_count,
            "upgrade_count": int(assignment.sum()),
            "incremental_gguf_bytes": upgrade_bytes,
            "complete_file_bytes": (
                policy["budget"]["mandatory_floor_file_bytes"] + upgrade_bytes),
            "assignment_sha256": hashlib.sha256(
                bytes(assignment.tolist())).hexdigest(),
            "logits_sha256": hashlib.sha256(
                candidate_logits.astype("<f4", copy=False).tobytes()).hexdigest(),
            "logit_comparison_to_authentic": comparison,
            "memory": asdict(evaluation.memory),
        }
        if label == "authentic_retain":
            result["expected_mean_nll"] = expected_authentic_loss
            result["exact_mean_nll_match"] = (
                evaluation.loss == expected_authentic_loss)
            result["exact_logit_match"] = np.array_equal(
                candidate_logits, baseline_logits)
            if not (result["exact_mean_nll_match"]
                    and result["exact_logit_match"]):
                report["results"][label] = result
                report["status"] = runtime_report_status(report["results"])
                _atomic_json(output_path, report)
                raise RuntimeError("all-retain runtime does not reproduce authentic GSQ")
        report["results"][label] = result
        report["status"] = runtime_report_status(report["results"])
        _atomic_json(output_path, report)
        print(json.dumps({
            "control": label,
            "mean_nll": evaluation.loss,
            "upgrade_count": result["upgrade_count"],
            "incremental_gguf_bytes": upgrade_bytes,
            "logit_passed": comparison["passed"],
        }, sort_keys=True), flush=True)

    report["status"] = runtime_report_status(report["results"])
    report["logit_thresholds"] = LOGIT_THRESHOLDS
    report["wall_seconds_this_invocation"] = time.perf_counter() - started
    report["warning"] = (
        "This is a fixed-sequence native integration gate, not a search or "
        "release-quality result."
    )
    _atomic_json(output_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--gguf-python", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--baseline-logits", type=Path, required=True)
    parser.add_argument("--baseline-controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controls", nargs="+", choices=CONTROL_ORDER,
                        default=list(CONTROL_ORDER))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    args = parser.parse_args()
    if min(args.rows_per_chunk, args.vocab_chunk_size) < 1:
        parser.error("chunk sizes must be positive")
    if len(set(args.controls)) != len(args.controls):
        parser.error("--controls must not contain duplicates")
    return args


if __name__ == "__main__":
    result = audit(_parse_args())
    print(json.dumps({
        "status": result["status"],
        "completed": sorted(result["results"]),
    }, indent=2, sort_keys=True))
