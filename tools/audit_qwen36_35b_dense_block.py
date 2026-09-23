#!/usr/bin/env python3
"""Retain a bounded, exactly reproducible dense Qwen3.6-35B block oracle."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import resource
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil
import torch
import transformers
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoint_stream import SafeTensorPrefixLoader
from dense_oracle import write_block_oracle
from model_adapter import get_model_adapter


CALIBRATION_TEXT = (
    "A bounded native GGML search keeps one genuine routed expert block "
    "resident while all other Qwen weights remain on disk."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--expected-oracle-sha256")
    parser.add_argument("--oracle-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


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


def _forward_block(
    block: torch.nn.Module,
    hidden: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(hidden.shape[1], device=device).view(1, -1)
    mask = torch.ones(hidden.shape[:2], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = block(
            hidden,
            position_embeddings=(None, None),
            attention_mask=mask,
            position_ids=positions,
            past_key_values=None,
            use_cache=False,
        )
    if isinstance(output, tuple):
        output = output[0]
    return output


def main() -> None:
    args = parse_args()
    if args.sequence_length < 2:
        raise ValueError("sequence length must be at least two")
    if (
        args.expected_oracle_sha256 is not None
        and re.fullmatch(r"[0-9a-f]{64}", args.expected_oracle_sha256) is None
    ):
        raise ValueError("expected oracle SHA-256 must be 64 lowercase hex digits")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    process = psutil.Process()
    initial_rss = process.memory_info().rss
    model_dir = args.model_dir.resolve(strict=True)
    identity = json.loads(args.identity.resolve(strict=True).read_text())
    if identity.get("status") != "pass":
        raise RuntimeError("checkpoint identity audit did not pass")
    if identity["repo_id"] != "Qwen/Qwen3.6-35B-A3B":
        raise RuntimeError(f"unexpected dense source: {identity['repo_id']}")

    torch.manual_seed(20260923)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokens = tokenizer(
        CALIBRATION_TEXT,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=args.sequence_length,
    )["input_ids"]
    if tokens.shape[1] < args.sequence_length:
        raise RuntimeError(f"calibration text produced only {tokens.shape[1]} tokens")
    tokens = tokens[:, :args.sequence_length].to(device)

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model_started = time.perf_counter()
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(
            config, attn_implementation="eager")
    model.eval()
    adapter = get_model_adapter(model)
    if adapter.num_hidden_layers != 40 or adapter.num_experts != 256:
        raise RuntimeError(
            f"unexpected model geometry: {adapter.num_hidden_layers} layers, "
            f"{adapter.num_experts} experts")
    loader = SafeTensorPrefixLoader(model_dir)
    loader.move_runtime_buffers(model, device)
    model_init_seconds = time.perf_counter() - model_started

    embedding_prefix = adapter.embedding_paths[0]
    embedding_schema = loader.assert_prefix_schema(model, embedding_prefix)
    embedding_bytes = loader.load_prefix(
        model, embedding_prefix, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        hidden = adapter.embeddings[0](tokens).contiguous()
    embedding_released = loader.release_prefix(model, embedding_prefix)

    block_prefix = f"{adapter.layers_path}.0"
    block_schema = loader.assert_prefix_schema(model, block_prefix)
    load_started = time.perf_counter()
    block_bytes = loader.load_prefix(
        model, block_prefix, device=device, dtype=torch.bfloat16)
    block_load_seconds = time.perf_counter() - load_started
    block = adapter.layers[0]
    block.eval()

    forward_seconds = []
    outputs = []
    for _ in range(2):
        pass_started = time.perf_counter()
        output = _forward_block(block, hidden, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds.append(time.perf_counter() - pass_started)
        outputs.append(output.detach().to(device="cpu").contiguous())
    if not torch.equal(outputs[0], outputs[1]):
        raise RuntimeError("dense block output is not exactly reproducible")
    if not torch.isfinite(outputs[0]).all():
        raise RuntimeError("dense block output contains non-finite values")

    oracle_destination = args.oracle_output
    guarded_temporary: Path | None = None
    if args.expected_oracle_sha256 is not None:
        oracle_destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{oracle_destination.name}.", suffix=".guarded",
            dir=oracle_destination.parent)
        os.close(descriptor)
        guarded_temporary = Path(temporary_name)
        guarded_temporary.unlink()
    try:
        oracle = write_block_oracle(
            guarded_temporary or oracle_destination,
            {
                "input_ids": tokens.detach().cpu(),
                "block_input": hidden.detach().cpu(),
                "block_output": outputs[0],
            },
            metadata={
                "repo_id": identity["repo_id"],
                "revision": identity["revision"],
                "layer": 0,
                "calibration_text": CALIBRATION_TEXT,
                "attention_implementation": "eager",
            },
        )
        if (
            args.expected_oracle_sha256 is not None
            and oracle["sha256"] != args.expected_oracle_sha256
        ):
            raise RuntimeError(
                f"oracle SHA-256 {oracle['sha256']} differs from independent "
                f"process result {args.expected_oracle_sha256}")
        if guarded_temporary is not None:
            os.replace(guarded_temporary, oracle_destination)
            directory_fd = os.open(oracle_destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            oracle["path"] = str(oracle_destination)
    finally:
        if guarded_temporary is not None:
            guarded_temporary.unlink(missing_ok=True)
    released_bytes = loader.release_prefix(model, block_prefix)
    del outputs, output, block, hidden
    gc.collect()

    expected_block_bytes = sum(
        item["logical_bytes"] for item in identity["text_inventory"]
        if item["name"].startswith(block_prefix + ".")
    )
    expected_embedding_bytes = sum(
        item["logical_bytes"] for item in identity["text_inventory"]
        if item["name"].startswith(embedding_prefix + ".")
    )
    if block_bytes != expected_block_bytes or embedding_bytes != expected_embedding_bytes:
        raise RuntimeError("streamed resident bytes differ from identity inventory")
    if released_bytes != block_bytes or embedding_released != embedding_bytes:
        raise RuntimeError("streamed prefixes did not release their complete payload")

    cuda: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "allocated_peak_bytes": 0,
        "reserved_peak_bytes": 0,
    }
    if device.type == "cuda":
        cuda.update({
            "device_name": torch.cuda.get_device_name(device),
            "allocated_peak_bytes": torch.cuda.max_memory_allocated(device),
            "reserved_peak_bytes": torch.cuda.max_memory_reserved(device),
        })
    report = {
        "schema": 1,
        "status": "pass",
        "scope": (
            "genuine Qwen3.6-35B-A3B layer-0 BF16 numerical oracle with "
            "only the embedding or one transformer block resident at a time"
        ),
        "source": {
            "repo_id": identity["repo_id"],
            "revision": identity["revision"],
            "model_dir": str(model_dir),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device": str(device),
            "cuda": cuda,
        },
        "model": {
            "class": type(model).__name__,
            "text_layer_count": adapter.num_hidden_layers,
            "hidden_size": adapter.hidden_size,
            "expert_count": adapter.num_experts,
            "active_experts": adapter.num_experts_per_tok,
        },
        "calibration": {
            "text": CALIBRATION_TEXT,
            "sequence_length": int(tokens.shape[1]),
            "input_ids": tokens.detach().cpu().tolist()[0],
            "input_sha256": oracle["tensors"]["block_input"]["sha256"],
            "output_sha256": oracle["tensors"]["block_output"]["sha256"],
        },
        "oracle": oracle,
        "streaming": {
            "embedding": {
                "prefix": embedding_prefix,
                "schema": embedding_schema,
                "resident_bf16_bytes": embedding_bytes,
                "released_bytes": embedding_released,
            },
            "block": {
                "index": 0,
                "prefix": block_prefix,
                "schema": block_schema,
                "source_tensor_count": block_schema["checkpoint_count"],
                "resident_bf16_bytes": block_bytes,
                "released_bytes": released_bytes,
                "released_to_meta": all(
                    parameter.device.type == "meta"
                    for parameter in adapter.layers[0].parameters()),
            },
            "simultaneous_embedding_and_block_residency": False,
        },
        "reproducibility": {
            "passes": 2,
            "exact_output_equality": True,
            "finite": True,
            "output_sha256": oracle["tensors"]["block_output"]["sha256"],
            "independent_process_oracle_sha256": args.expected_oracle_sha256,
            "independent_process_byte_match": (
                args.expected_oracle_sha256 is not None),
        },
        "memory": {
            "initial_rss_bytes": initial_rss,
            "peak_process_rss_bytes": (
                int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024),
        },
        "timing": {
            "model_init_seconds": model_init_seconds,
            "block_load_seconds": block_load_seconds,
            "forward_seconds": forward_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }
    _atomic_json(args.report_output, report)
    print(json.dumps({
        "status": report["status"],
        "oracle": str(args.oracle_output),
        "report": str(args.report_output),
        "block_bytes": block_bytes,
        "output_sha256": report["reproducibility"]["output_sha256"],
        "peak_process_rss_bytes": report["memory"]["peak_process_rss_bytes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
