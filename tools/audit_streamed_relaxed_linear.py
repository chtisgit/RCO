#!/usr/bin/env python3
"""Retain a deterministic dense-equivalence oracle for streamed backward."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.quant import WeightInterpolation
from search.relaxed import (
    DenseDeltaRowSource,
    StreamingRelaxedLinearStats,
    streaming_relaxed_linear,
)


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


def _sha256_tensors(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        values = tensor.detach().cpu().contiguous()
        digest.update(str(values.dtype).encode())
        digest.update(str(tuple(values.shape)).encode())
        digest.update(values.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def _run_case(
    *, seed: int, dtype: torch.dtype, rows_per_chunk: int,
    batch: int, sequence: int, in_features: int, out_features: int,
    alternatives_count: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32).to(dtype)

    reference = randn(out_features, in_features)
    alternatives = [
        reference + (0.05 * (index + 1)) * randn(out_features, in_features)
        for index in range(alternatives_count)
    ]
    bias_values = randn(out_features)
    input_values = randn(batch, sequence, in_features)
    target = randn(batch, sequence, out_features)
    logits_values = randn(alternatives_count + 1)

    dense_input = input_values.clone().requires_grad_(True)
    dense_logits = logits_values.clone().requires_grad_(True)
    dense_bias = bias_values.clone().requires_grad_(True)
    interpolation = WeightInterpolation(
        [candidate - reference for candidate in alternatives],
        dense_logits.unsqueeze(0),
        0,
    )
    dense_output = F.linear(
        dense_input, interpolation(reference), dense_bias)
    dense_loss = F.mse_loss(dense_output.float(), target.float())
    dense_loss.backward()

    streamed_input = input_values.clone().requires_grad_(True)
    streamed_logits = logits_values.clone().requires_grad_(True)
    streamed_bias = bias_values.clone().requires_grad_(True)
    stats = StreamingRelaxedLinearStats()
    streamed_output = streaming_relaxed_linear(
        streamed_input,
        streamed_logits,
        DenseDeltaRowSource(
            reference, alternatives, rows_per_chunk=rows_per_chunk),
        bias=streamed_bias,
        stats=stats,
    )
    saved_tensors = tuple(streamed_output.grad_fn.saved_tensors)
    saved_tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in saved_tensors)
    saved_shapes = [list(tensor.shape) for tensor in saved_tensors]
    expected_saved_shapes = [
        list(streamed_input.shape),
        [alternatives_count + 1],
    ]
    if len(saved_tensors) != 2 or saved_shapes != expected_saved_shapes:
        raise RuntimeError(
            "streamed autograd retained values beyond input/probabilities: "
            f"{saved_shapes}")
    streamed_loss = F.mse_loss(streamed_output.float(), target.float())
    streamed_loss.backward()

    errors = {
        "output_max_abs": _max_error(streamed_output, dense_output),
        "loss_abs": abs(float(streamed_loss.item()) - float(dense_loss.item())),
        "input_gradient_max_abs": _max_error(
            streamed_input.grad, dense_input.grad),
        "assignment_gradient_max_abs": _max_error(
            streamed_logits.grad, dense_logits.grad),
        "bias_gradient_max_abs": _max_error(
            streamed_bias.grad, dense_bias.grad),
    }
    tolerance = {
        torch.float64: 1e-10,
        torch.float32: 2e-6,
        # Streaming distributes the matmul over reference and delta chunks,
        # while the released path first rounds the mixed BF16 weight.  A
        # 0.04 bound covers that expected accumulation-order difference; the
        # report retains each substantially smaller gradient error separately.
        torch.bfloat16: 4e-2,
    }[dtype]
    passed = all(value <= tolerance for value in errors.values())
    if not passed:
        raise RuntimeError(
            f"streamed relaxed case exceeds {tolerance}: {errors}")

    result_hash = _sha256_tensors(
        streamed_output,
        streamed_input.grad,
        streamed_logits.grad,
        streamed_bias.grad,
    )
    return {
        "seed": seed,
        "dtype": str(dtype).removeprefix("torch."),
        "shape": {
            "batch": batch,
            "sequence": sequence,
            "in_features": in_features,
            "out_features": out_features,
            "alternatives": alternatives_count,
        },
        "rows_per_chunk": rows_per_chunk,
        "tolerance": tolerance,
        "errors": errors,
        "dense_loss": float(dense_loss.item()),
        "streamed_loss": float(streamed_loss.item()),
        "saved_tensor_count": len(saved_tensors),
        "saved_tensor_shapes": saved_shapes,
        "saved_tensor_bytes": saved_tensor_bytes,
        "dense_delta_bytes_avoided": (
            alternatives_count * reference.numel() * reference.element_size()),
        "stats": asdict(stats),
        "result_sha256": result_hash,
        "status": "pass",
    }


def audit() -> dict[str, Any]:
    specifications = [
        (101, torch.float64, 1),
        (103, torch.float64, 4),
        (107, torch.float32, 3),
        (109, torch.bfloat16, 2),
    ]
    cases = [
        _run_case(
            seed=seed,
            dtype=dtype,
            rows_per_chunk=rows,
            batch=2,
            sequence=5,
            in_features=11,
            out_features=13,
            alternatives_count=2,
        )
        for seed, dtype, rows in specifications
    ]
    repeats = [
        _run_case(
            seed=seed,
            dtype=dtype,
            rows_per_chunk=rows,
            batch=2,
            sequence=5,
            in_features=11,
            out_features=13,
            alternatives_count=2,
        )
        for seed, dtype, rows in specifications
    ]
    reproducible = [case["result_sha256"] for case in cases] == [
        case["result_sha256"] for case in repeats]
    if not reproducible:
        raise RuntimeError("streamed relaxed oracle is not reproducible")

    error_names = tuple(cases[0]["errors"])
    worst_errors = {
        name: max(case["errors"][name] for case in cases)
        for name in error_names
    }
    return {
        "schema": 1,
        "status": "pass",
        "scope": (
            "tiny linear dense-equivalence gate for the Phase 5 streamed "
            "relaxed-backward fallback"
        ),
        "reference": "released search.quant.WeightInterpolation",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "cases": cases,
        "worst_errors": worst_errors,
        "all_cases_pass": all(case["status"] == "pass" for case in cases),
        "same_seed_reproducible": reproducible,
        "memory_invariant": {
            "autograd_saved_values": "linear input and softmax probabilities",
            "candidate_or_delta_saved_by_autograd": False,
            "source_reread_in_backward": True,
            "maximum_autograd_saved_tensor_bytes": max(
                case["saved_tensor_bytes"] for case in cases),
            "largest_materialized_weight_chunk_bytes": max(
                case["stats"]["max_materialized_chunk_bytes"]
                for case in cases),
            "maximum_dense_delta_bytes_avoided": max(
                case["dense_delta_bytes_avoided"] for case in cases),
        },
        "limitations": [
            "This gate uses a dense oracle row source, not native GGML files.",
            "It validates one parametrized linear operation, not block reload "
            "or a complete model backward pass.",
            "CUDA memory and kernel behavior remain untested while CUDA "
            "initialization is unavailable.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    _atomic_json(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "worst_errors": report["worst_errors"],
        "same_seed_reproducible": report["same_seed_reproducible"],
        "memory_invariant": report["memory_invariant"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
