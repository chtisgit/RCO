#!/usr/bin/env python3
"""Evaluate a relaxed step's projected assignment through the hard runtime."""

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

from checkpoint_stream import SafeTensorPrefixLoader
from native_runtime import NativeManifestWeightStore
from native_store import NativeCandidateStore
from quant.ggml_native import GGMLNativeCodec, GGMLType
from search.hard import realized_cost
from search.streaming import StreamingHardCausalEvaluator


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


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model_dir = args.model_dir.resolve(strict=True)
    identity_path = args.identity.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    store_path = args.store.resolve(strict=True)
    step_path = args.step_report.resolve(strict=True)
    identity = _load_json(identity_path)
    manifest = _load_json(manifest_path)
    step = _load_json(step_path)
    entries = sorted(
        (entry for entry in manifest["entries"] if entry.get("rco_search")),
        key=lambda entry: entry["destination_name"],
    )
    if len(entries) != 512:
        raise RuntimeError(f"expected 512 decisions, found {len(entries)}")
    names = [entry["destination_name"] for entry in entries]

    codec = GGMLNativeCodec(args.ggml_library.resolve(strict=True))
    packed_store = NativeCandidateStore(store_path, codec)
    store_index_path = store_path / "native-candidate-index.json"
    store_index_sha256 = _sha256_file(store_index_path)
    revision = packed_store.index["source"].get(
        "revision", packed_store.index["source"].get("dense_revision"))
    if revision != identity["revision"]:
        raise RuntimeError("candidate store and checkpoint revisions differ")
    if (
        step["source"]["revision"] != identity["revision"]
        or step["candidate_store"]["index_sha256"] != store_index_sha256
        or step["candidate_store"]["decision_count"] != len(entries)
    ):
        raise RuntimeError("relaxed step describes a different search problem")

    optimization = step["run"].get("optimization")
    if not isinstance(optimization, dict):
        raise RuntimeError("step report has no projected relaxed assignment")
    assignment = torch.tensor(
        optimization["projected_assignment"], dtype=torch.long)
    if assignment.shape != (len(entries),):
        raise RuntimeError("projected assignment has the wrong group count")
    if torch.any((assignment != 0) & (assignment != 1)):
        raise RuntimeError("projected assignment is not binary")

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
    target_cost = int(optimization["target_aligned_gguf_bytes"])
    exact_cost = realized_cost(assignment, low_costs, high_costs)
    if (
        exact_cost != target_cost
        or exact_cost != optimization["projected_assignment_cost"]
    ):
        raise RuntimeError("projected assignment does not meet its byte target")

    weight_store = NativeManifestWeightStore(
        packed_store,
        {"entries": entries},
        model_dir,
        rows_per_chunk=args.rows_per_chunk,
    )
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(
            config, attn_implementation="eager")
    model.eval()
    groups = [SimpleNamespace(layer_names=(name,)) for name in names]
    evaluator = StreamingHardCausalEvaluator(
        model,
        SafeTensorPrefixLoader(model_dir),
        weight_store,
        groups,
        [2, 4],
        device=device,
        vocab_chunk_size=args.vocab_chunk_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    input_ids = tokenizer(
        CALIBRATION_TEXT,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=args.sequence_length,
    )["input_ids"]
    if input_ids.shape[1] < args.sequence_length:
        raise RuntimeError("calibration text is shorter than requested sequence")
    input_ids = input_ids[:, :args.sequence_length].contiguous()
    if input_ids.tolist()[0] != step["calibration"]["input_ids"]:
        raise RuntimeError("hard evaluation tokens differ from relaxed step")

    evaluation = evaluator.evaluate(input_ids, assignment)
    if evaluation.token_count != args.sequence_length - 1:
        raise RuntimeError("hard evaluation returned the wrong token count")
    if evaluation.memory.loaded_blocks != 40:
        raise RuntimeError("hard evaluation did not stream all decoder blocks")
    cuda = {
        "available": torch.cuda.is_available(),
        "built_version": torch.version.cuda,
        "allocated_peak_bytes": evaluation.memory.cuda_max_allocated,
        "reserved_peak_bytes": evaluation.memory.cuda_max_reserved,
    }
    if device.type == "cuda":
        cuda["device_name"] = torch.cuda.get_device_name(device)
    return {
        "schema": 1,
        "status": "pass",
        "scope": (
            "authoritative hard native evaluation of the exact-cost assignment "
            "projected by one complete streamed relaxed production step"
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
        "candidate_store": {
            "path": str(store_path),
            "index_sha256": store_index_sha256,
            "decision_count": len(entries),
            "low_type": GGMLType.Q2_0.name,
            "high_type": GGMLType.Q4_0.name,
            "rows_per_chunk": args.rows_per_chunk,
            "target_cost": target_cost,
        },
        "relaxed_step": {
            "path": str(step_path),
            "sha256": _sha256_file(step_path),
            "relaxed_loss": step["run"]["loss"],
            "updated_logits_sha256": optimization["updated_logits_sha256"],
            "assignment_bits": optimization["projected_assignment_bits"],
            "assignment": assignment.tolist(),
            "assignment_cost": exact_cost,
        },
        "calibration": {
            "text": CALIBRATION_TEXT,
            "sequence_length": args.sequence_length,
            "input_ids": input_ids.tolist()[0],
            "objective": "exact full-vocabulary causal cross-entropy",
            "vocab_chunk_size": args.vocab_chunk_size,
        },
        "hard_evaluation": {
            "loss": evaluation.loss,
            "token_count": evaluation.token_count,
            "memory": asdict(evaluation.memory),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "cuda": cuda,
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
    parser.add_argument("--step-report", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sequence_length < 2:
        raise ValueError("sequence length must be at least two")
    report = audit(args)
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "hard_loss": report["hard_evaluation"]["loss"],
        "assignment_cost": report["relaxed_step"]["assignment_cost"],
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
