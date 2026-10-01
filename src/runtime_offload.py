"""Validation helpers for llama.cpp partial CUDA-offload evidence."""

from __future__ import annotations

import re
from typing import Any


_FAILURE_MARKERS = (
    "failed to initialize cuda",
    "no cuda-capable device",
    "no usable gpu",
    "ignoring --gpu-layers",
    "ignored --gpu-layers",
)


def _required_match(pattern: str, text: str, label: str) -> re.Match[str]:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"missing llama.cpp {label} evidence")
    return match


def parse_partial_cuda_offload(
    stderr: str,
    *,
    requested_layers: int,
    expected_total_layers: int,
    device: str = "CUDA0",
) -> dict[str, Any]:
    """Parse and strictly validate one llama.cpp partial-offload run."""

    lowered = stderr.lower()
    failures = [marker for marker in _FAILURE_MARKERS if marker in lowered]
    if failures:
        raise ValueError(f"llama.cpp reported CUDA failure: {failures}")

    escaped_device = re.escape(device)
    _required_match(
        rf"using device\s+{escaped_device}\s+\([^\n]+\)",
        stderr,
        f"{device} selection",
    )
    offload = _required_match(
        r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU",
        stderr,
        "layer offload",
    )
    offloaded_layers = int(offload.group(1))
    total_layers = int(offload.group(2))
    if total_layers != expected_total_layers:
        raise ValueError(
            f"llama.cpp reported {total_layers} offloadable layers; "
            f"expected {expected_total_layers}")
    if offloaded_layers != requested_layers:
        raise ValueError(
            f"llama.cpp offloaded {offloaded_layers} layers; "
            f"requested {requested_layers}")
    if not 0 < offloaded_layers < total_layers:
        raise ValueError(
            f"offload must be partial, got {offloaded_layers}/{total_layers}")

    model_buffer = _required_match(
        rf"{escaped_device}\s+model buffer size\s*=\s*([0-9.]+)\s+MiB",
        stderr,
        f"{device} model buffer",
    )
    compute_buffer = _required_match(
        rf"{escaped_device}\s+compute buffer size\s*=\s*([0-9.]+)\s+MiB",
        stderr,
        f"{device} compute buffer",
    )
    cpu_buffer = _required_match(
        r"CPU_Mapped\s+model buffer size\s*=\s*([0-9.]+)\s+MiB",
        stderr,
        "CPU mapped model buffer",
    )
    prompt_timing = _required_match(
        r"prompt eval time\s*=\s*([0-9.]+)\s+ms\s*/\s*(\d+)\s+tokens",
        stderr,
        "prompt evaluation",
    )
    generation_timing = _required_match(
        r"(?<!prompt )eval time\s*=\s*([0-9.]+)\s+ms\s*/\s*(\d+)\s+tokens",
        stderr,
        "generation evaluation",
    )

    cuda_model_mib = float(model_buffer.group(1))
    cuda_compute_mib = float(compute_buffer.group(1))
    cpu_model_mib = float(cpu_buffer.group(1))
    if cuda_model_mib <= 0 or cuda_compute_mib <= 0 or cpu_model_mib <= 0:
        raise ValueError("llama.cpp reported a non-positive runtime buffer")

    evidence_patterns = (
        "using device", "offloading output layer", "offloading ", "offloaded ",
        "model buffer size", "kv buffer size", "rs buffer size",
        "compute buffer size", "prompt eval time", "eval time",
        "memory breakdown", f"- {device}",
    )
    evidence = [
        line for line in stderr.splitlines()
        if any(pattern.lower() in line.lower() for pattern in evidence_patterns)
    ]
    return {
        "device": device,
        "requested_layers": requested_layers,
        "offloaded_layers": offloaded_layers,
        "total_offloadable_layers": total_layers,
        "partial_offload": True,
        "cuda_model_buffer_mib": cuda_model_mib,
        "cuda_compute_buffer_mib": cuda_compute_mib,
        "cpu_mapped_model_buffer_mib": cpu_model_mib,
        "prompt_eval_milliseconds": float(prompt_timing.group(1)),
        "prompt_tokens": int(prompt_timing.group(2)),
        "generation_eval_milliseconds": float(generation_timing.group(1)),
        "generated_tokens": int(generation_timing.group(2)),
        "cuda_failure_markers": [],
        "loader_and_execution_evidence": evidence,
    }
