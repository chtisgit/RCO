#!/usr/bin/env python3
"""Run resumable BF16 and uniform native Q2_0/Q4_0 full-model controls."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
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
from search.hard import realized_cost  # noqa: E402
from search.streaming import StreamingHardCausalEvaluator  # noqa: E402


CALIBRATION_TEXT = (
    "A bounded native GGML search streams every Qwen decoder block while "
    "evaluating an exact serialized byte budget."
)
CONTROL_ORDER = ("bf16", "uniform_q4_0", "uniform_q2_0")
CONTROL_CANDIDATE_TYPES = {
    "uniform_q4_0": GGMLType.Q4_0,
    "uniform_q2_0": GGMLType.Q2_0,
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


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda is unavailable")
    model_dir = args.model_dir.resolve(strict=True)
    identity_path = args.identity.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    store_path = args.store.resolve(strict=True)
    output_path = args.output.resolve()
    identity = _load_json(identity_path)
    manifest = _load_json(manifest_path)
    entries = sorted(
        (entry for entry in manifest["entries"] if entry.get("rco_search")),
        key=lambda entry: entry["destination_name"],
    )
    if len(entries) != 512:
        raise RuntimeError(f"expected 512 search groups, found {len(entries)}")
    names = [entry["destination_name"] for entry in entries]
    codec = GGMLNativeCodec(args.ggml_library.resolve(strict=True))
    base_store = NativeCandidateStore(store_path, codec)
    if base_store.index["source"].get(
        "revision", base_store.index["source"].get("dense_revision")
    ) != identity["revision"]:
        raise RuntimeError("candidate store and BF16 identity revisions differ")
    overlay_paths = [path.resolve(strict=True) for path in args.overlay_store]
    overlay_stores = [NativeCandidateStore(path, codec) for path in overlay_paths]
    packed_store = (
        NativeCandidateOverlayStore(base_store, *overlay_stores)
        if overlay_stores else base_store
    )
    controls = tuple(args.controls)
    candidate_provenance_counts: dict[str, dict[str, int]] = {}
    for label in controls:
        candidate_type = CONTROL_CANDIDATE_TYPES.get(label)
        if candidate_type is None:
            continue
        counts: dict[str, int] = {}
        for name in names:
            kind = packed_store.metadata(
                name, candidate_type)["provenance"]["kind"]
            counts[kind] = counts.get(kind, 0) + 1
        candidate_provenance_counts[label] = counts
        if counts != {"bf16_native_quantization": len(names)}:
            raise RuntimeError(
                f"{label} is not uniformly BF16-derived: {counts}")
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
    assignments = {
        "bf16": torch.empty(0, dtype=torch.long),
        "uniform_q4_0": torch.ones(len(names), dtype=torch.long),
        "uniform_q2_0": torch.zeros(len(names), dtype=torch.long),
    }
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    input_ids = tokenizer(
        CALIBRATION_TEXT, return_tensors="pt", add_special_tokens=False,
        truncation=True, max_length=args.sequence_length,
    )["input_ids"]
    if input_ids.shape[1] != args.sequence_length:
        raise RuntimeError("calibration text is too short")
    problem = {
        "model_revision": identity["revision"],
        "identity_sha256": _sha256_file(identity_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "candidate_store_index_sha256": _sha256_file(
            store_path / NATIVE_CANDIDATE_INDEX),
        "overlay_store_index_sha256s": [
            _sha256_file(path / NATIVE_CANDIDATE_INDEX)
            for path in overlay_paths
        ],
        "input_ids": input_ids.tolist()[0],
        "controls": list(controls),
        "candidate_provenance_counts": candidate_provenance_counts,
        "device": str(device),
        "rows_per_chunk": args.rows_per_chunk,
        "vocab_chunk_size": args.vocab_chunk_size,
    }
    problem_sha256 = hashlib.sha256(
        json.dumps(problem, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if output_path.exists():
        report = _load_json(output_path)
        if report.get("problem_sha256") != problem_sha256:
            raise RuntimeError("existing uniform-control report describes another problem")
    else:
        report = {
            "schema": "rco.qwen36.uniform_native_controls.v1",
            "status": "in_progress",
            "scope": (
                "selected complete-model BF16 and uniform BF16-derived native "
                "Q2_0/Q4_0 controls on one fixed causal sequence"
            ),
            "problem": problem,
            "problem_sha256": problem_sha256,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
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
    candidate_evaluator = StreamingHardCausalEvaluator(
        model, loader, weight_store,
        [SimpleNamespace(layer_names=(name,)) for name in names],
        [2, 4], device=device, vocab_chunk_size=args.vocab_chunk_size,
    )
    bf16_evaluator = StreamingHardCausalEvaluator(
        model, loader, weight_store, [], [2, 4], device=device,
        vocab_chunk_size=args.vocab_chunk_size,
    )
    new_control_count = 0
    for label in controls:
        if label in report["results"]:
            continue
        assignment = assignments[label]
        evaluator = bf16_evaluator if label == "bf16" else candidate_evaluator
        evaluation = evaluator.evaluate(input_ids, assignment)
        result = {
            "mean_nll": evaluation.loss,
            "perplexity": (
                float(torch.exp(torch.tensor(evaluation.loss)).item())
                if evaluation.loss < 80 else None
            ),
            "predicted_token_count": evaluation.token_count,
            "assignment_bits": (
                None if label == "bf16"
                else "".join(str(int(value)) for value in assignment.tolist())
            ),
            "realized_aligned_gguf_bytes": (
                None if label == "bf16"
                else realized_cost(assignment, low_costs, high_costs)
            ),
            "memory": asdict(evaluation.memory),
        }
        report["results"][label] = result
        _atomic_json(output_path, report)
        print(json.dumps({"control": label, **result}, sort_keys=True), flush=True)
        new_control_count += 1
        if (args.stop_after_new_controls is not None
                and new_control_count >= args.stop_after_new_controls):
            return report
    report["status"] = "complete"
    report["elapsed_seconds"] = sum(
        value["memory"]["total_seconds"] for value in report["results"].values())
    report["ordering"] = sorted(
        report["results"], key=lambda label: report["results"][label]["mean_nll"])
    interpretation: dict[str, Any] = {
        "warning": (
            "Native results are screening evidence only because milestone "
            "5e8d1ec failed strict native-versus-llama.cpp parity."
        ),
    }
    if "bf16" in report["results"]:
        for label in ("uniform_q4_0", "uniform_q2_0"):
            if label in report["results"]:
                interpretation[f"{label}_catastrophic_relative_to_bf16"] = (
                    report["results"][label]["mean_nll"]
                    - report["results"]["bf16"]["mean_nll"] > 2.0
                )
    report["interpretation"] = interpretation
    report["wall_seconds_this_invocation"] = time.perf_counter() - started
    _atomic_json(output_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument(
        "--overlay-store", type=Path, action="append", default=[],
        help="candidate store whose per-type payloads override the base store",
    )
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument(
        "--controls", nargs="+", choices=CONTROL_ORDER,
        default=list(CONTROL_ORDER),
        help="controls to run, in the requested order",
    )
    parser.add_argument("--stop-after-new-controls", type=int)
    args = parser.parse_args()
    if min(args.sequence_length, args.vocab_chunk_size, args.rows_per_chunk) < 1:
        parser.error("invalid numeric argument")
    if args.stop_after_new_controls is not None and args.stop_after_new_controls < 1:
        parser.error("--stop-after-new-controls must be positive")
    if len(set(args.controls)) != len(args.controls):
        parser.error("--controls must not contain duplicates")
    return args


if __name__ == "__main__":
    result = audit(_parse_args())
    print(json.dumps({
        "status": result["status"], "results": result["results"],
    }, indent=2, sort_keys=True))
