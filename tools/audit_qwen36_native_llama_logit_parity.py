#!/usr/bin/env python3
"""Compare full-vocabulary native-runtime and pinned-llama.cpp logits."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
import platform
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import transformers
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoint_stream import SafeTensorPrefixLoader  # noqa: E402
from native_runtime import NativeManifestWeightStore  # noqa: E402
from native_store import NATIVE_CANDIDATE_INDEX, NativeCandidateStore  # noqa: E402
from quant.ggml_native import GGMLNativeCodec, GGMLType  # noqa: E402
from search.streaming import StreamingHardCausalEvaluator  # noqa: E402


CALIBRATION_TEXT = (
    "A bounded native GGML search streams every Qwen decoder block while "
    "evaluating an exact serialized byte budget."
)

# Declared before measuring the production artifacts. These tolerances permit
# ordinary BF16/GGML kernel differences but reject materially different logits.
MAXIMUM_POSITION_RMSE = 0.25
MAXIMUM_POSITION_MEAN_ABSOLUTE_ERROR = 0.15
MINIMUM_POSITION_COSINE_SIMILARITY = 0.999
REQUIRE_TOP1_AGREEMENT = True
MAXIMUM_LAYER_RELATIVE_RMSE = 0.05
MINIMUM_LAYER_COSINE_SIMILARITY = 0.999


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


def _atomic_npz(
    path: Path, *, tokens: np.ndarray, native: np.ndarray, llama: np.ndarray,
    native_layers: np.ndarray, llama_layers: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle, tokens=tokens, native=native, llama=llama,
                native_layers=native_layers, llama_layers=llama_layers)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_bundle(path: Path, identifier: str, text: str, tokens: list[int]) -> None:
    encoded_id = identifier.encode("utf-8")
    encoded_text = text.encode("utf-8")
    with path.open("wb") as handle:
        handle.write(b"RCONLL1\0")
        handle.write(struct.pack("<II", 1, 1))
        for payload in (encoded_id, encoded_text):
            handle.write(struct.pack("<I", len(payload)))
            handle.write(payload)
        handle.write(struct.pack("<I", len(tokens)))
        handle.write(struct.pack(f"<{len(tokens)}i", *tokens))


def _read_llama_logits(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        if handle.read(8) != b"RCOLOG1\0":
            raise RuntimeError("invalid llama.cpp logit-dump magic")
        version, positions, vocabulary = struct.unpack("<III", handle.read(12))
        if version != 1 or positions < 1 or vocabulary < 1:
            raise RuntimeError("invalid llama.cpp logit-dump header")
        tokens = np.fromfile(handle, dtype="<i4", count=positions + 1)
        logits = np.fromfile(
            handle, dtype="<f4", count=positions * vocabulary,
        ).reshape(positions, vocabulary)
        if handle.read(1):
            raise RuntimeError("unexpected trailing logit-dump bytes")
    return tokens, logits


def _read_llama_layers(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        if handle.read(8) != b"RCOLAY1\0":
            raise RuntimeError("invalid llama.cpp layer-dump magic")
        version, count, tokens, hidden = struct.unpack("<IIII", handle.read(16))
        if version != 1 or min(count, tokens, hidden) < 1:
            raise RuntimeError("invalid llama.cpp layer-dump header")
        result = np.empty((count, tokens, hidden), dtype=np.float32)
        for expected in range(count):
            index, = struct.unpack("<I", handle.read(4))
            if index != expected:
                raise RuntimeError("llama.cpp layer-dump order mismatch")
            values = np.fromfile(handle, dtype="<f4", count=tokens * hidden)
            if values.size != tokens * hidden:
                raise RuntimeError("truncated llama.cpp layer dump")
            result[index] = values.reshape(tokens, hidden)
        if handle.read(1):
            raise RuntimeError("unexpected trailing layer-dump bytes")
    return result


def _comparison(
    native: np.ndarray, llama: np.ndarray, tokens: list[int],
) -> tuple[list[dict[str, Any]], bool]:
    if native.shape != llama.shape:
        raise RuntimeError(
            f"native/llama logit shapes differ: {native.shape} and {llama.shape}")
    positions = []
    overall_pass = True
    for index, (native_row, llama_row) in enumerate(
        zip(native.astype(np.float64), llama.astype(np.float64), strict=True)
    ):
        difference = native_row - llama_row
        rmse = float(np.sqrt(np.mean(np.square(difference))))
        mean_absolute = float(np.mean(np.abs(difference)))
        maximum_absolute = float(np.max(np.abs(difference)))
        denominator = float(np.linalg.norm(native_row) * np.linalg.norm(llama_row))
        cosine = float(np.dot(native_row, llama_row) / denominator)
        native_top = np.argsort(native_row)[-10:][::-1]
        llama_top = np.argsort(llama_row)[-10:][::-1]
        top1_agreement = bool(native_top[0] == llama_top[0])
        passed = (
            rmse <= MAXIMUM_POSITION_RMSE
            and mean_absolute <= MAXIMUM_POSITION_MEAN_ABSOLUTE_ERROR
            and cosine >= MINIMUM_POSITION_COSINE_SIMILARITY
            and (top1_agreement or not REQUIRE_TOP1_AGREEMENT)
        )
        overall_pass = overall_pass and passed
        target = tokens[index + 1]
        native_lse = float(np.logaddexp.reduce(native_row))
        llama_lse = float(np.logaddexp.reduce(llama_row))
        positions.append({
            "position": index,
            "target_token": target,
            "rmse": rmse,
            "mean_absolute_error": mean_absolute,
            "maximum_absolute_error": maximum_absolute,
            "cosine_similarity": cosine,
            "top1_agreement": top1_agreement,
            "native_top1_token": int(native_top[0]),
            "llama_top1_token": int(llama_top[0]),
            "top10_overlap_count": len(set(native_top.tolist()) & set(llama_top.tolist())),
            "native_target_nll": native_lse - float(native_row[target]),
            "llama_target_nll": llama_lse - float(llama_row[target]),
            "passed": passed,
        })
    return positions, overall_pass


def _layer_comparison(
    native: np.ndarray, llama: np.ndarray,
) -> tuple[list[dict[str, Any]], int | None]:
    if native.shape != llama.shape:
        raise RuntimeError(
            f"native/llama layer shapes differ: {native.shape} and {llama.shape}")
    results = []
    first_failed_layer = None
    for state_index, (native_state, llama_state) in enumerate(
        zip(native.astype(np.float64), llama.astype(np.float64), strict=True)
    ):
        position_results = []
        state_passed = True
        for position, (native_row, llama_row) in enumerate(
            zip(native_state, llama_state, strict=True)
        ):
            difference = native_row - llama_row
            rmse = float(np.sqrt(np.mean(np.square(difference))))
            reference_rms = float(np.sqrt(np.mean(np.square(llama_row))))
            relative_rmse = rmse / reference_rms if reference_rms else math.inf
            denominator = float(np.linalg.norm(native_row) * np.linalg.norm(llama_row))
            cosine = float(np.dot(native_row, llama_row) / denominator)
            passed = (
                relative_rmse <= MAXIMUM_LAYER_RELATIVE_RMSE
                and cosine >= MINIMUM_LAYER_COSINE_SIMILARITY
            )
            state_passed = state_passed and passed
            position_results.append({
                "position": position,
                "rmse": rmse,
                "reference_rms": reference_rms,
                "relative_rmse": relative_rmse,
                "maximum_absolute_error": float(np.max(np.abs(difference))),
                "cosine_similarity": cosine,
                "passed": passed,
            })
        layer = state_index - 1
        if not state_passed and first_failed_layer is None:
            first_failed_layer = layer
        results.append({
            "state": "model_input" if layer == -1 else "layer_output",
            "layer": layer,
            "passed": state_passed,
            "positions": position_results,
        })
    return results, first_failed_layer


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda is unavailable")
    model_dir = args.model_dir.resolve(strict=True)
    identity_path = args.identity.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    store_path = args.store.resolve(strict=True)
    gguf_report_path = args.gguf_report.resolve(strict=True)
    gguf_path = args.gguf.resolve(strict=True)
    helper_path = args.helper.resolve(strict=True)
    layer_helper_path = args.layer_helper.resolve(strict=True)
    output_path = args.output.resolve()
    logits_path = args.logits.resolve()

    identity = _load_json(identity_path)
    manifest = _load_json(manifest_path)
    gguf_report = _load_json(gguf_report_path)
    entries = sorted(
        (entry for entry in manifest["entries"] if entry.get("rco_search")),
        key=lambda entry: entry["destination_name"],
    )
    if len(entries) != 512:
        raise RuntimeError(f"expected 512 search groups, found {len(entries)}")
    names = [entry["destination_name"] for entry in entries]
    choices = gguf_report["assignment"]["choices"]
    if set(choices) != set(names):
        raise RuntimeError("GGUF assignment and manifest groups differ")
    assignment = torch.tensor([
        0 if choices[name] == "Q2_0" else 1 if choices[name] == "Q4_0" else -1
        for name in names
    ], dtype=torch.long)
    if bool((assignment < 0).any().item()):
        raise RuntimeError("GGUF assignment contains a non-Q2_0/Q4_0 choice")
    if gguf_path.stat().st_size != gguf_report["output"]["bytes"]:
        raise RuntimeError("GGUF byte size differs from construction report")
    gguf_sha256 = _sha256_file(gguf_path)
    if gguf_sha256 != gguf_report["output"]["sha256"]:
        raise RuntimeError("GGUF digest differs from construction report")

    codec = GGMLNativeCodec(args.ggml_library.resolve(strict=True))
    packed_store = NativeCandidateStore(store_path, codec)
    if packed_store.index["source"].get(
        "revision", packed_store.index["source"].get("dense_revision")
    ) != identity["revision"]:
        raise RuntimeError("candidate store and BF16 identity revisions differ")
    weight_store = NativeManifestWeightStore(
        packed_store, {"entries": entries}, model_dir,
        rows_per_chunk=args.rows_per_chunk,
    )
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(
            config, attn_implementation="eager")
    model.eval()
    evaluator = StreamingHardCausalEvaluator(
        model,
        SafeTensorPrefixLoader(model_dir),
        weight_store,
        [SimpleNamespace(layer_names=(name,)) for name in names],
        [2, 4],
        device=device,
        vocab_chunk_size=args.vocab_chunk_size,
        checkpoint_dtype={
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[args.native_compute_dtype],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokens = [int(value) for value in tokenizer.encode(
        CALIBRATION_TEXT, add_special_tokens=False)[:args.sequence_length]]
    if len(tokens) != args.sequence_length:
        raise RuntimeError("calibration text is too short")
    text = tokenizer.decode(
        tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if [int(value) for value in tokenizer.encode(
            text, add_special_tokens=False)] != tokens:
        raise RuntimeError("calibration prefix does not round-trip through tokenizer")
    input_ids = torch.tensor([tokens], dtype=torch.long)
    native_result = evaluator.evaluate(
        input_ids, assignment,
        capture_logit_positions=tuple(range(len(tokens) - 1)),
        capture_layer_outputs=True,
    )
    if native_result.captured_logits is None:
        raise RuntimeError("native evaluator did not return captured logits")
    native_logits = native_result.captured_logits.numpy()
    if native_result.captured_model_input is None:
        raise RuntimeError("native evaluator did not capture model input")
    native_layers = np.stack([
        native_result.captured_model_input.numpy(),
        *(value.numpy() for value in native_result.captured_layer_outputs),
    ])

    with tempfile.TemporaryDirectory(prefix="rco-logit-parity-") as temporary:
        bundle = Path(temporary) / "input.bundle"
        raw_logits = Path(temporary) / "llama-logits.bin"
        raw_layers = Path(temporary) / "llama-layers.bin"
        _write_bundle(bundle, "calibration-prefix", text, tokens)
        completed = subprocess.run([
            str(helper_path), "--model", str(gguf_path),
            "--input", str(bundle), "--output", str(raw_logits),
            "--gpu-layers", str(args.gpu_layers),
            "--threads", str(args.threads), "--ubatch", str(args.ubatch),
        ], check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"llama.cpp logit helper failed ({completed.returncode}): "
                f"{completed.stderr[-4000:]}")
        llama_tokens, llama_logits = _read_llama_logits(raw_logits)
        completed = subprocess.run([
            str(layer_helper_path), "--model", str(gguf_path),
            "--input", str(bundle), "--output", str(raw_layers),
            "--expected-layers", str(len(native_result.captured_layer_outputs)),
            "--gpu-layers", str(args.gpu_layers),
            "--threads", str(args.threads), "--ubatch", str(args.ubatch),
        ], check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"llama.cpp layer helper failed ({completed.returncode}): "
                f"{completed.stderr[-4000:]}")
        llama_layers = _read_llama_layers(raw_layers)[:, :len(tokens) - 1]
    if llama_tokens.tolist() != tokens:
        raise RuntimeError("llama.cpp helper returned different token IDs")

    position_results, passed = _comparison(native_logits, llama_logits, tokens)
    layer_results, first_failed_layer = _layer_comparison(
        native_layers, llama_layers)
    passed = passed and first_failed_layer is None
    _atomic_npz(
        logits_path,
        tokens=np.asarray(tokens, dtype=np.int32),
        native=native_logits,
        llama=llama_logits,
        native_layers=native_layers,
        llama_layers=llama_layers,
    )
    report = {
        "schema": "rco.qwen36.native_llama_logit_parity.v1",
        "status": "pass" if passed else "fail",
        "scope": "existing BF16-derived Q2_0/Q4_0 prototype assignment",
        "thresholds_declared_before_production_measurement": {
            "maximum_position_rmse": MAXIMUM_POSITION_RMSE,
            "maximum_position_mean_absolute_error": MAXIMUM_POSITION_MEAN_ABSOLUTE_ERROR,
            "minimum_position_cosine_similarity": MINIMUM_POSITION_COSINE_SIMILARITY,
            "require_top1_agreement": REQUIRE_TOP1_AGREEMENT,
            "maximum_layer_relative_rmse": MAXIMUM_LAYER_RELATIVE_RMSE,
            "minimum_layer_cosine_similarity": MINIMUM_LAYER_COSINE_SIMILARITY,
        },
        "tokens": {"text": text, "input_ids": tokens},
        "positions": position_results,
        "layer_states": layer_results,
        "first_failed_position": next(
            (item["position"] for item in position_results if not item["passed"]), None),
        "first_failed_layer": first_failed_layer,
        "native": {
            "loss": native_result.loss,
            "token_count": native_result.token_count,
            "memory": asdict(native_result.memory),
        },
        "artifacts": {
            "bf16_identity": {"path": str(identity_path), "sha256": _sha256_file(identity_path)},
            "manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)},
            "candidate_store": {
                "path": str(store_path),
                "index_sha256": _sha256_file(store_path / NATIVE_CANDIDATE_INDEX),
            },
            "gguf_report": {"path": str(gguf_report_path), "sha256": _sha256_file(gguf_report_path)},
            "gguf": {"path": str(gguf_path), "bytes": gguf_path.stat().st_size, "sha256": gguf_sha256},
            "helper": {"path": str(helper_path), "sha256": _sha256_file(helper_path)},
            "layer_helper": {
                "path": str(layer_helper_path),
                "sha256": _sha256_file(layer_helper_path),
            },
            "logits": {"path": str(logits_path), "sha256": _sha256_file(logits_path)},
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "gpu_layers": args.gpu_layers,
            "threads": args.threads,
            "ubatch": args.ubatch,
            "rows_per_chunk": args.rows_per_chunk,
            "native_compute_dtype": args.native_compute_dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(output_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--gguf-report", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--layer-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument(
        "--native-compute-dtype", choices=("bf16", "fp32"), default="fp32")
    parser.add_argument("--gpu-layers", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ubatch", type=int, default=64)
    args = parser.parse_args()
    if args.sequence_length < 2 or min(
        args.vocab_chunk_size, args.rows_per_chunk, args.threads, args.ubatch,
    ) < 1 or args.gpu_layers < 0:
        parser.error("invalid numeric argument")
    return args


if __name__ == "__main__":
    result = audit(_parse_args())
    print(json.dumps({
        "status": result["status"],
        "first_failed_position": result["first_failed_position"],
        "first_failed_layer": result["first_failed_layer"],
        "positions": result["positions"],
    }, indent=2, sort_keys=True))
