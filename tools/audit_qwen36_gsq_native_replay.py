#!/usr/bin/env python3
"""Replay the untouched GSQ-hybrid payloads through the native streamer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
import platform
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_qwen36_native_llama_logit_parity import (  # noqa: E402
    CALIBRATION_TEXT,
    MAXIMUM_LAYER_RELATIVE_RMSE,
    MAXIMUM_POSITION_MEAN_ABSOLUTE_ERROR,
    MAXIMUM_POSITION_RMSE,
    MINIMUM_LAYER_COSINE_SIMILARITY,
    MINIMUM_POSITION_COSINE_SIMILARITY,
    REQUIRE_TOP1_AGREEMENT,
    _atomic_json,
    _atomic_npz,
    _comparison,
    _layer_comparison,
    _read_llama_layers,
    _read_llama_logits,
    _write_bundle,
)
from gguf_checkpoint_stream import GGUFManifestPrefixLoader  # noqa: E402
from native_gguf import import_pinned_gguf  # noqa: E402
from search.streaming import StreamingHardCausalEvaluator  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


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
    helper = args.helper.resolve(strict=True)
    layer_helper = args.layer_helper.resolve(strict=True)
    output_path = args.output.resolve()
    logits_path = args.logits.resolve()
    identity = _load_json(identity_path)
    manifest = _load_json(manifest_path)
    if len(manifest["entries"]) != 733:
        raise RuntimeError("manifest does not contain 733 text tensors")
    gguf_sha256 = _sha256_file(gguf_path)
    if args.gguf_sha256 and gguf_sha256 != args.gguf_sha256:
        raise RuntimeError("untouched GSQ GGUF digest differs")

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(
            config, attn_implementation="eager")
    model.eval()
    loader = GGUFManifestPrefixLoader(
        gguf_path, manifest, model_dir,
        gguf_python=gguf_python,
        ggml_library=args.ggml_library.resolve(strict=True),
        rows_per_chunk=args.rows_per_chunk)
    evaluator = StreamingHardCausalEvaluator(
        model,
        loader,
        SimpleNamespace(),
        [],
        [2, 4],
        device=device,
        vocab_chunk_size=args.vocab_chunk_size,
        checkpoint_dtype=torch.bfloat16,
    )
    prefixes = [
        *evaluator.adapter.embedding_paths,
        *(f"{evaluator.adapter.layers_path}.{index}"
          for index in range(len(evaluator.adapter.layers))),
        *(path for path in evaluator.adapter.final_module_paths if path != "lm_head"),
        "lm_head",
    ]
    schema = [loader.assert_prefix_schema(model, prefix) for prefix in prefixes]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokens = [int(value) for value in tokenizer.encode(
        CALIBRATION_TEXT, add_special_tokens=False)[:args.sequence_length]]
    if len(tokens) != args.sequence_length:
        raise RuntimeError("calibration text is too short")
    text = tokenizer.decode(
        tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if [int(value) for value in tokenizer.encode(
            text, add_special_tokens=False)] != tokens:
        raise RuntimeError("calibration prefix does not round-trip")
    input_ids = torch.tensor([tokens], dtype=torch.long)
    native_result = evaluator.evaluate(
        input_ids,
        torch.empty(0, dtype=torch.long),
        capture_logit_positions=tuple(range(len(tokens) - 1)),
        capture_layer_outputs=True,
    )
    if native_result.captured_logits is None:
        raise RuntimeError("native replay returned no logits")
    if native_result.captured_model_input is None:
        raise RuntimeError("native replay returned no model input")
    native_logits = native_result.captured_logits.numpy()
    native_layers = np.stack([
        native_result.captured_model_input.numpy(),
        *(value.numpy() for value in native_result.captured_layer_outputs),
    ])

    with tempfile.TemporaryDirectory(prefix="rco-gsq-replay-") as temporary:
        temporary_path = Path(temporary)
        bundle = temporary_path / "input.bundle"
        raw_logits = temporary_path / "llama-logits.bin"
        raw_layers = temporary_path / "llama-layers.bin"
        _write_bundle(bundle, "calibration-prefix", text, tokens)
        logit_command = [
            str(helper), "--model", str(gguf_path),
            "--input", str(bundle), "--output", str(raw_logits),
            "--gpu-layers", str(args.gpu_layers),
            "--threads", str(args.threads), "--ubatch", str(args.ubatch),
        ]
        completed = subprocess.run(
            logit_command, check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"llama logit helper failed: {completed.stderr[-4000:]}")
        llama_tokens, llama_logits = _read_llama_logits(raw_logits)
        layer_command = [
            str(layer_helper), "--model", str(gguf_path),
            "--input", str(bundle), "--output", str(raw_layers),
            "--gpu-layers", str(args.gpu_layers),
            "--threads", str(args.threads), "--ubatch", str(args.ubatch),
        ]
        completed = subprocess.run(
            layer_command, check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"llama layer helper failed: {completed.stderr[-4000:]}")
        llama_layers = _read_llama_layers(raw_layers)[:, :len(tokens) - 1]
    if llama_tokens.tolist() != tokens:
        raise RuntimeError("llama helper token sequence differs")

    positions, output_passed = _comparison(native_logits, llama_logits, tokens)
    layers, first_failed_layer = _layer_comparison(native_layers, llama_layers)
    gguf = import_pinned_gguf(gguf_python)
    reader = gguf.GGUFReader(gguf_path)
    type_counts: dict[str, int] = {}
    for tensor in reader.tensors:
        type_counts[tensor.tensor_type.name] = type_counts.get(
            tensor.tensor_type.name, 0) + 1
    report = {
        "schema": "rco.qwen36.gsq_native_replay.v1",
        "status": "pass" if output_passed and first_failed_layer is None else "mismatch",
        "scope": (
            "untouched GSQ-hybrid payloads inverse-mapped and streamed into the "
            "native HF evaluator, compared with the same GGUF in pinned llama.cpp"
        ),
        "source": {
            "model": str(gguf_path),
            "sha256": gguf_sha256,
            "bytes": gguf_path.stat().st_size,
            "tensor_count": len(reader.tensors),
            "tensor_type_counts": dict(sorted(type_counts.items())),
            "manifest_sha256": _sha256_file(manifest_path),
            "identity_sha256": _sha256_file(identity_path),
            "dense_revision": identity["revision"],
        },
        "input": {"text": text, "tokens": tokens},
        "native": {
            "device": str(device),
            "compute_dtype": "bfloat16",
            "mean_nll": native_result.loss,
            "predicted_token_count": native_result.token_count,
            "memory": asdict(native_result.memory),
            "max_decoded_chunk_bytes": loader.max_decoded_chunk_bytes,
            "prefix_schema": schema,
        },
        "llama_cpp": {
            "logit_helper": str(helper),
            "logit_helper_sha256": _sha256_file(helper),
            "layer_helper": str(layer_helper),
            "layer_helper_sha256": _sha256_file(layer_helper),
            "gpu_layers": args.gpu_layers,
            "threads": args.threads,
            "ubatch": args.ubatch,
            "logit_command": logit_command,
            "layer_command": layer_command,
        },
        "thresholds": {
            "maximum_position_rmse": MAXIMUM_POSITION_RMSE,
            "maximum_position_mean_absolute_error": (
                MAXIMUM_POSITION_MEAN_ABSOLUTE_ERROR),
            "minimum_position_cosine_similarity": (
                MINIMUM_POSITION_COSINE_SIMILARITY),
            "require_top1_agreement": REQUIRE_TOP1_AGREEMENT,
            "maximum_layer_relative_rmse": MAXIMUM_LAYER_RELATIVE_RMSE,
            "minimum_layer_cosine_similarity": MINIMUM_LAYER_COSINE_SIMILARITY,
        },
        "comparison": {
            "output_passed": output_passed,
            "first_failed_layer": first_failed_layer,
            "positions": positions,
            "layers": layers,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "warning": (
            "A mismatch is expected to be interpreted with the already proven "
            "native-versus-llama recurrent execution drift; this replay tests "
            "untouched GSQ provenance and inverse mapping, not release quality."
        ),
    }
    _atomic_npz(
        logits_path,
        tokens=np.asarray(tokens, dtype=np.int32),
        native=native_logits,
        llama=llama_logits,
        native_layers=native_layers,
        llama_layers=llama_layers,
    )
    report["arrays"] = {
        "path": str(logits_path),
        "sha256": _sha256_file(logits_path),
    }
    _atomic_json(output_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--gguf-sha256")
    parser.add_argument("--gguf-python", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--layer-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ubatch", type=int, default=64)
    args = parser.parse_args()
    if min(
        args.sequence_length, args.vocab_chunk_size, args.rows_per_chunk,
        args.threads, args.ubatch,
    ) < 1 or args.gpu_layers < 0:
        parser.error("invalid numeric argument")
    return args


if __name__ == "__main__":
    result = audit(_parse_args())
    print(json.dumps({
        "status": result["status"],
        "native_mean_nll": result["native"]["mean_nll"],
        "output_passed": result["comparison"]["output_passed"],
        "first_failed_layer": result["comparison"]["first_failed_layer"],
    }, indent=2, sort_keys=True))
