#!/usr/bin/env python3
"""Localize destructive BF16-to-Q2 substitutions by broad tensor family."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import transformers
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoint_stream import SafeTensorPrefixLoader  # noqa: E402
from native_runtime import NativeManifestWeightStore  # noqa: E402
from native_store import (  # noqa: E402
    NATIVE_CANDIDATE_INDEX,
    NativeCandidateOverlayStore,
    NativeCandidateStore,
)
from quant.ggml_native import GGMLNativeCodec, GGMLType  # noqa: E402
from search.streaming import StreamingHardCausalEvaluator  # noqa: E402


CALIBRATION_TEXT = (
    "A bounded native GGML search streams every Qwen decoder block while "
    "evaluating an exact serialized byte budget."
)
FAMILY_ORDER = (
    "embedding_and_output",
    "linear_attention_projections",
    "self_attention_projections",
    "routed_expert_matrices",
    "shared_expert_matrices",
    "router_and_shared_gate",
    "ssm_alpha_beta",
)
SUBFAMILY_ORDER = (
    "token_embedding",
    "output_head",
    "linear_attention_qkv",
    "linear_attention_gate",
    "linear_attention_output",
    "routed_expert_down",
    "routed_expert_gate",
    "routed_expert_up",
    "shared_expert_down",
    "shared_expert_gate",
    "shared_expert_up",
)
INTERACTION_ORDER = (
    "linear_qkv_plus_output",
    "linear_gate_plus_output",
    "routed_down_plus_gate",
)


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


def family_for_entry(entry: dict[str, Any]) -> str:
    name = entry["destination_name"]
    category = entry["source_category"]
    if category in {"embedding", "lm_head"}:
        return "embedding_and_output"
    if category == "self_attention":
        return "self_attention_projections"
    if category == "routed_expert":
        return "routed_expert_matrices"
    if category == "shared_expert":
        return (
            "router_and_shared_gate"
            if "gate_inp_shexp" in name else "shared_expert_matrices"
        )
    if category == "router":
        return "router_and_shared_gate"
    if category == "linear_attention":
        if ".ssm_alpha." in name or ".ssm_beta." in name:
            return "ssm_alpha_beta"
        if any(token in name for token in (
            ".attn_qkv.", ".attn_gate.", ".ssm_out.",
        )):
            return "linear_attention_projections"
    raise ValueError(f"unclassified searchable tensor: {category}/{name}")


def subfamily_for_entry(entry: dict[str, Any]) -> str | None:
    name = entry["destination_name"]
    category = entry["source_category"]
    if category == "embedding":
        return "token_embedding"
    if category == "lm_head":
        return "output_head"
    if category == "linear_attention":
        if ".attn_qkv." in name:
            return "linear_attention_qkv"
        if ".attn_gate." in name:
            return "linear_attention_gate"
        if ".ssm_out." in name:
            return "linear_attention_output"
    if category in {"routed_expert", "shared_expert"}:
        if "gate_inp_shexp" in name:
            return None
        prefix = "routed_expert" if category == "routed_expert" else "shared_expert"
        for projection in ("down", "gate", "up"):
            if f"ffn_{projection}_" in name:
                return f"{prefix}_{projection}"
    return None


def interaction_families_for_entry(entry: dict[str, Any]) -> tuple[str, ...]:
    subfamily = subfamily_for_entry(entry)
    selected = []
    if subfamily in {"linear_attention_qkv", "linear_attention_output"}:
        selected.append("linear_qkv_plus_output")
    if subfamily in {"linear_attention_gate", "linear_attention_output"}:
        selected.append("linear_gate_plus_output")
    if subfamily in {"routed_expert_down", "routed_expert_gate"}:
        selected.append("routed_down_plus_gate")
    return tuple(selected)


def _error_records(
    database_report: dict[str, Any], overlay_report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    records = {
        record["tensor"]: record
        for record in database_report["validation"]["candidates"]
        if record.get("origin") == "bf16_native_quantization"
        and record["ggml_type"] == "Q2_0"
    }
    records.update({
        record["tensor"]: record for record in overlay_report["candidates"]
        if record["ggml_type"] == "Q2_0"
    })
    if len(records) != 512:
        raise RuntimeError(f"expected 512 BF16 Q2 error records, found {len(records)}")
    return records


def _aggregate_errors(
    names: list[str], records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    elements = 0
    sum_squared = 0.0
    sum_absolute = 0.0
    source_sum_squared = 0.0
    maximum_absolute = 0.0
    for name in names:
        record = records[name]
        count = int(record["elements"])
        errors = record["errors"]
        squared = float(errors["rmse"]) ** 2 * count
        relative = float(errors["relative_frobenius_error"])
        elements += count
        sum_squared += squared
        sum_absolute += float(errors["mean_absolute_error"]) * count
        maximum_absolute = max(
            maximum_absolute, float(errors["max_absolute_error"]))
        if relative:
            source_sum_squared += squared / (relative ** 2)
    return {
        "elements": elements,
        "maximum_absolute_error": maximum_absolute,
        "mean_absolute_error": sum_absolute / elements,
        "rmse": math.sqrt(sum_squared / elements),
        "relative_frobenius_error": math.sqrt(
            sum_squared / source_sum_squared) if source_sum_squared else 0.0,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda is unavailable")
    model_dir = args.model_dir.resolve(strict=True)
    identity_path = args.identity.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    store_path = args.store.resolve(strict=True)
    overlay_path = args.overlay_store.resolve(strict=True)
    database_report_path = args.database_report.resolve(strict=True)
    overlay_report_path = args.overlay_report.resolve(strict=True)
    output_path = args.output.resolve()
    identity = _load_json(identity_path)
    manifest = _load_json(manifest_path)
    entries = sorted(
        (entry for entry in manifest["entries"] if entry.get("rco_search")),
        key=lambda entry: entry["destination_name"],
    )
    if len(entries) != 512:
        raise RuntimeError(f"expected 512 search groups, found {len(entries)}")
    if args.granularity == "broad":
        family_order = FAMILY_ORDER
        selector = family_for_entry
        expected_selected = 512
    elif args.granularity == "subfamily":
        family_order = SUBFAMILY_ORDER
        selector = subfamily_for_entry
        expected_selected = 332
    else:
        family_order = INTERACTION_ORDER
        selector = None
        expected_counts = {
            "linear_qkv_plus_output": 60,
            "linear_gate_plus_output": 60,
            "routed_down_plus_gate": 80,
        }
    families: dict[str, list[dict[str, Any]]] = {
        name: [] for name in family_order}
    for entry in entries:
        selected_families = (
            interaction_families_for_entry(entry)
            if args.granularity == "interaction"
            else (selector(entry),)
        )
        for selected_family in selected_families:
            if selected_family is not None:
                families[selected_family].append(entry)
    if args.granularity == "interaction":
        if {name: len(values) for name, values in families.items()} != expected_counts:
            raise RuntimeError("interaction family coverage differs")
    elif sum(map(len, families.values())) != expected_selected or any(
        not values for values in families.values()
    ):
        raise RuntimeError("family partition is incomplete")
    requested_families = tuple(args.families or family_order)
    unknown_requested = sorted(set(requested_families) - set(family_order))
    if unknown_requested:
        raise RuntimeError(
            f"families do not belong to {args.granularity}: {unknown_requested}")

    codec = GGMLNativeCodec(args.ggml_library.resolve(strict=True))
    base_store = NativeCandidateStore(store_path, codec)
    overlay_store = NativeCandidateStore(overlay_path, codec)
    store = NativeCandidateOverlayStore(base_store, overlay_store)
    for entry in entries:
        provenance = store.metadata(
            entry["destination_name"], GGMLType.Q2_0)["provenance"]
        if provenance["kind"] != "bf16_native_quantization":
            raise RuntimeError("family control includes non-BF16 Q2 provenance")
    weight_store = NativeManifestWeightStore(
        store, {"entries": entries}, model_dir,
        rows_per_chunk=args.rows_per_chunk)
    records = _error_records(
        _load_json(database_report_path), _load_json(overlay_report_path))
    for entry in entries:
        name = entry["destination_name"]
        if records[name]["sha256"] != store.metadata(
                name, GGMLType.Q2_0)["sha256"]:
            raise RuntimeError(f"error record and candidate payload differ: {name}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    input_ids = tokenizer(
        CALIBRATION_TEXT, return_tensors="pt", add_special_tokens=False,
        truncation=True, max_length=args.sequence_length,
    )["input_ids"]
    if input_ids.shape[1] != args.sequence_length:
        raise RuntimeError("calibration text is too short")
    problem = {
        "dense_revision": identity["revision"],
        "identity_sha256": _sha256_file(identity_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "base_store_index_sha256": _sha256_file(
            store_path / NATIVE_CANDIDATE_INDEX),
        "overlay_store_index_sha256": _sha256_file(
            overlay_path / NATIVE_CANDIDATE_INDEX),
        "database_report_sha256": _sha256_file(database_report_path),
        "overlay_report_sha256": _sha256_file(overlay_report_path),
        "input_ids": input_ids.tolist()[0],
        "families": list(requested_families),
        "granularity": args.granularity,
        "family_tensor_counts": {
            name: len(families[name]) for name in requested_families},
        "device": str(device),
        "rows_per_chunk": args.rows_per_chunk,
        "vocab_chunk_size": args.vocab_chunk_size,
    }
    problem_sha256 = hashlib.sha256(json.dumps(
        problem, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if output_path.exists():
        report = _load_json(output_path)
        if report.get("problem_sha256") != problem_sha256:
            raise RuntimeError("existing family-control report describes another problem")
    else:
        report = {
            "schema": "rco.qwen36.q2_family_controls.v1",
            "status": "in_progress",
            "scope": (
                "complete-model native screens starting from BF16 and replacing "
                "one broad tensor family at a time with BF16-derived Q2_0"
            ),
            "problem": problem,
            "problem_sha256": problem_sha256,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "excluded_fixed_families": {
                "ssm_convolution_and_state_scalars": (
                    "not searchable and never replaced in uniform Q2: 30 conv1d "
                    "F32 tensors plus 90 F32 state/norm vectors"
                )
            },
            "results": {},
        }
        _atomic_json(output_path, report)

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(
            config, attn_implementation="eager")
    model.eval()
    loader = SafeTensorPrefixLoader(model_dir)
    baseline_key = "bf16"
    if baseline_key not in report["results"]:
        evaluator = StreamingHardCausalEvaluator(
            model, loader, weight_store, [], [2, 4], device=device,
            vocab_chunk_size=args.vocab_chunk_size)
        evaluation = evaluator.evaluate(
            input_ids, torch.empty(0, dtype=torch.long))
        report["results"][baseline_key] = {
            "mean_nll": evaluation.loss,
            "predicted_token_count": evaluation.token_count,
            "memory": asdict(evaluation.memory),
        }
        _atomic_json(output_path, report)
        print(json.dumps({
            "control": baseline_key, "mean_nll": evaluation.loss,
        }, sort_keys=True), flush=True)
    baseline = float(report["results"][baseline_key]["mean_nll"])

    new_family_count = 0
    for family in requested_families:
        if family in report["results"]:
            continue
        family_entries = families[family]
        names = [entry["destination_name"] for entry in family_entries]
        evaluator = StreamingHardCausalEvaluator(
            model,
            loader,
            weight_store,
            [SimpleNamespace(layer_names=(name,)) for name in names],
            [2, 4],
            device=device,
            vocab_chunk_size=args.vocab_chunk_size,
        )
        evaluation = evaluator.evaluate(
            input_ids, torch.zeros(len(names), dtype=torch.long))
        delta = evaluation.loss - baseline
        result = {
            "tensor_count": len(names),
            "tensors": names,
            "mean_nll": evaluation.loss,
            "delta_mean_nll_from_bf16": delta,
            "catastrophic_by_plus_2_nll": delta > 2.0,
            "predicted_token_count": evaluation.token_count,
            "q2_aligned_gguf_bytes": sum(int(store.metadata(
                name, GGMLType.Q2_0)["aligned_gguf_bytes"]) for name in names),
            "reconstruction_error": _aggregate_errors(names, records),
            "memory": asdict(evaluation.memory),
        }
        report["results"][family] = result
        _atomic_json(output_path, report)
        print(json.dumps({
            "control": family,
            "tensor_count": len(names),
            "mean_nll": evaluation.loss,
            "delta_mean_nll_from_bf16": delta,
            "catastrophic": delta > 2.0,
        }, sort_keys=True), flush=True)
        new_family_count += 1
        if (args.stop_after_new_families is not None
                and new_family_count >= args.stop_after_new_families):
            return report
    report["status"] = "complete"
    report["ranking_by_delta_mean_nll"] = sorted(
        requested_families,
        key=lambda name: report["results"][name]["delta_mean_nll_from_bf16"],
        reverse=True,
    )
    report["catastrophic_families"] = [
        name for name in requested_families
        if report["results"][name]["catastrophic_by_plus_2_nll"]
    ]
    report["wall_seconds_this_invocation"] = time.perf_counter() - started
    report["warning"] = (
        "These three-target native screens localize candidate damage but are "
        "not release-quality evaluations; step-2 runtime parity remains failed."
    )
    _atomic_json(output_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--overlay-store", type=Path, required=True)
    parser.add_argument("--database-report", type=Path, required=True)
    parser.add_argument("--overlay-report", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument("--granularity", choices=("broad", "subfamily", "interaction"),
                        default="broad")
    parser.add_argument("--families", nargs="+",
                        choices=FAMILY_ORDER + SUBFAMILY_ORDER + INTERACTION_ORDER)
    parser.add_argument("--stop-after-new-families", type=int)
    args = parser.parse_args()
    if min(args.sequence_length, args.vocab_chunk_size, args.rows_per_chunk) < 1:
        parser.error("invalid numeric argument")
    if args.families is not None and len(set(args.families)) != len(args.families):
        parser.error("--families must not contain duplicates")
    if (args.stop_after_new_families is not None
            and args.stop_after_new_families < 1):
        parser.error("--stop-after-new-families must be positive")
    return args


if __name__ == "__main__":
    result = audit(_parse_args())
    print(json.dumps({
        "status": result["status"],
        "completed": sorted(result["results"]),
        "catastrophic_families": result.get("catastrophic_families"),
    }, indent=2, sort_keys=True))
