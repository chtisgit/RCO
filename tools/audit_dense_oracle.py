#!/usr/bin/env python3
"""Reproduce dense 2B calibration loss and retain a block-0 oracle."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense_oracle import write_block_oracle
from model_adapter import get_model_adapter


CALIBRATION_TEXT = (
    "Native GGML candidates must preserve the model objective while keeping "
    "only one decoder block resident at a time. This fixed sentence is the "
    "Qwen3.5 dense calibration oracle."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--oracle-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def _cuda_metrics() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "available": False,
            "device_count": torch.cuda.device_count(),
            "allocated_peak_bytes": None,
            "reserved_peak_bytes": None,
        }
    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
        "reserved_peak_bytes": torch.cuda.max_memory_reserved(),
    }


def main() -> None:
    args = parse_args()
    if args.sequence_length < 2:
        raise ValueError("sequence length must be at least two")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    torch.manual_seed(20260923)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260923)
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    started = time.perf_counter()
    initial_rss = process.memory_info().rss

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True)
    encoded = tokenizer(
        CALIBRATION_TEXT,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=args.sequence_length,
    )
    input_ids = encoded["input_ids"]
    if input_ids.shape[1] < args.sequence_length:
        raise ValueError(
            f"calibration text produced only {input_ids.shape[1]} tokens")
    input_ids = input_ids[:, :args.sequence_length].to(args.device)
    attention_mask = torch.ones_like(input_ids)

    load_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.to(args.device)
    model.eval()
    load_seconds = time.perf_counter() - load_started
    loaded_rss = process.memory_info().rss
    adapter = get_model_adapter(model)
    if adapter.num_hidden_layers != 24:
        raise ValueError(f"expected 24 layers, found {adapter.num_hidden_layers}")

    captured: dict[str, torch.Tensor] = {}

    def capture_input(_module, values):
        captured["block_input"] = values[0].detach().to(device="cpu").contiguous()

    def capture_output(_module, _values, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["block_output"] = hidden.detach().to(device="cpu").contiguous()

    pre_handle = adapter.layers[0].register_forward_pre_hook(capture_input)
    post_handle = adapter.layers[0].register_forward_hook(capture_output)
    losses = []
    pass_seconds = []
    with torch.inference_mode():
        for pass_index in range(2):
            pass_started = time.perf_counter()
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                use_cache=False,
            )
            if args.device == "cuda":
                torch.cuda.synchronize()
            pass_seconds.append(time.perf_counter() - pass_started)
            losses.append(float(result.loss.detach().cpu()))
            if pass_index == 0:
                pre_handle.remove()
                post_handle.remove()

    if losses[0] != losses[1]:
        raise RuntimeError(f"dense losses are not exactly reproducible: {losses}")
    if set(captured) != {"block_input", "block_output"}:
        raise RuntimeError(f"failed to capture complete block oracle: {captured.keys()}")
    if captured["block_input"].shape != captured["block_output"].shape:
        raise RuntimeError("block input/output shapes differ")

    oracle = write_block_oracle(
        args.oracle_output,
        {
            "input_ids": input_ids.detach().cpu(),
            "block_input": captured["block_input"],
            "block_output": captured["block_output"],
        },
        metadata={
            "repo_id": "Qwen/Qwen3.5-2B-Base",
            "revision": args.revision,
            "layer": 0,
            "calibration_text": CALIBRATION_TEXT,
            "loss": repr(losses[0]),
        },
    )
    report = {
        "schema": 1,
        "status": "pass",
        "scope": "dense full-model causal loss and layer-0 numerical oracle",
        "source": {
            "repo_id": "Qwen/Qwen3.5-2B-Base",
            "revision": args.revision,
            "model_dir": str(args.model_dir.resolve()),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device": args.device,
            "cuda": _cuda_metrics(),
        },
        "model": {
            "class": type(model).__name__,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "text_layer_count": adapter.num_hidden_layers,
            "hidden_size": adapter.hidden_size,
        },
        "calibration": {
            "text": CALIBRATION_TEXT,
            "sequence_length": input_ids.shape[1],
            "input_ids": input_ids.detach().cpu().tolist()[0],
            "losses": losses,
            "exactly_reproducible": True,
        },
        "oracle": oracle,
        "memory": {
            "initial_rss_bytes": initial_rss,
            "rss_after_load_bytes": loaded_rss,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
        "timing": {
            "load_seconds": load_seconds,
            "forward_seconds": pass_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "loss": losses[0],
        "peak_process_rss_bytes": report["memory"]["peak_process_rss_bytes"],
        "oracle": oracle["path"],
        "report": str(args.report_output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

