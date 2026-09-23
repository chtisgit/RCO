#!/usr/bin/env python3
"""Compare exact-byte hard estimators on the genuine 35B block oracle."""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
import platform
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_qwen36_35b_native_block_output import (
    _forward_block,
    _install_candidate,
    _load_json,
    _load_oracle,
)
from checkpoint_stream import SafeTensorPrefixLoader
from dense_oracle import tensor_sha256
from model_adapter import get_model_adapter
from native_store import NativeCandidateStore
from quant.ggml_native import GGMLNativeCodec, GGMLType
from qwen35_native import Qwen35LinearAttentionGeometry
from search.hard import (
    exact_cost_assignment,
    optimize_cost_reinforce,
    optimize_cost_spsa,
    realized_cost,
)


CALIBRATION_TEXTS = (
    "A bounded native GGML search keeps one genuine routed expert block "
    "resident while all other Qwen weights remain on disk.",
    "Exact serialized byte budgets prevent a mixed precision optimizer from "
    "quietly publishing a model larger than the requested deployment target.",
    "Antithetic samples compare two feasible assignments on the same hidden "
    "states so calibration noise cannot dominate the estimator direction.",
    "The final GGUF must preserve every selected native payload and execute in "
    "an unmodified runtime on both ordinary and routed expert tensor shapes.",
)


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


def _key(bits: tuple[int, ...] | torch.Tensor) -> str:
    values = bits.tolist() if isinstance(bits, torch.Tensor) else bits
    return "".join(str(int(value)) for value in values)


def _losses(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    reference64 = reference.to(device="cpu", dtype=torch.float64)
    actual64 = actual.to(device="cpu", dtype=torch.float64)
    squared = (actual64 - reference64).square()
    reference_squared = reference64.square()
    token_loss = (
        squared.sum(dim=-1) / reference_squared.sum(dim=-1).clamp_min(1e-30)
    ).reshape(-1)
    normalized_mse = float(squared.sum() / reference_squared.sum())
    return {
        "normalized_mse": normalized_mse,
        "relative_frobenius_error": normalized_mse ** 0.5,
        "token_normalized_mse_mean": float(token_loss.mean()),
        "token_normalized_mse_variance": float(token_loss.var(unbiased=False)),
        "token_normalized_mse": [float(value) for value in token_loss],
    }


def _feasible_assignments(
    low_costs: list[int], high_costs: list[int], target_cost: int,
) -> list[tuple[int, ...]]:
    # Gray-code ordering reduces candidate rewrites between neighboring model
    # evaluations while leaving the exhaustively evaluated set unchanged.
    result = []
    for value in range(1 << len(low_costs)):
        gray = value ^ (value >> 1)
        bits = tuple((gray >> index) & 1 for index in range(len(low_costs)))
        cost = sum(high_costs[index] if bit else low_costs[index]
                   for index, bit in enumerate(bits))
        if cost == target_cost:
            result.append(bits)
    if not result:
        raise ValueError(f"no assignment realizes target cost {target_cost}")
    return result


def _curve_summary(curves: list[list[float]]) -> list[dict[str, float]]:
    return [
        {
            "step": step,
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "population_variance": statistics.pvariance(values),
            "minimum": min(values),
            "maximum": max(values),
        }
        for step, values in enumerate(zip(*curves))
    ]


def _compare_estimator(
    name: str,
    landscape: dict[str, float],
    low_costs: list[int],
    high_costs: list[int],
    target_cost: int,
    *,
    seeds: list[int],
    steps: int,
    learning_rate: float,
    perturbation: float,
    baseline_decay: float,
    optimum: float,
) -> dict[str, Any]:
    runs = []
    for seed in seeds:
        evaluated: list[float] = []

        def evaluate(assignment: torch.Tensor) -> float:
            if realized_cost(assignment, low_costs, high_costs) != target_cost:
                raise RuntimeError("optimizer emitted an infeasible assignment")
            key = _key(assignment)
            if key not in landscape:
                raise RuntimeError(f"optimizer emitted unknown assignment {key}")
            value = landscape[key]
            evaluated.append(value)
            return value

        common = {
            "low_costs": low_costs,
            "high_costs": high_costs,
            "target_cost": target_cost,
            "n_steps": steps,
            "lr": learning_rate,
            "seed": seed,
            "log_interval": steps + 1,
        }
        if name == "spsa":
            scores, assignment, history = optimize_cost_spsa(
                evaluate, perturbation=perturbation, **common)
        elif name == "reinforce":
            scores, assignment, history = optimize_cost_reinforce(
                evaluate, baseline_decay=baseline_decay, **common)
        else:
            raise ValueError(name)
        final_key = _key(assignment)
        final_loss = landscape[final_key]
        pair_best = [float(item["best_loss"]) for item in history]
        best_curve = []
        best = float("inf")
        for value in pair_best:
            best = min(best, value)
            best_curve.append(best)
        best = min(best, final_loss)
        runs.append({
            "seed": seed,
            "evaluation_count": len(evaluated),
            "all_evaluations_exact_cost": all(
                int(item["realized_cost"]) == target_cost for item in history),
            "evaluated_loss_mean": statistics.fmean(evaluated),
            "evaluated_loss_population_variance": statistics.pvariance(evaluated),
            "best_seen_loss": best,
            "reached_global_optimum": abs(best - optimum) <= 1e-15,
            "final_assignment_bits": final_key,
            "final_assignment": assignment.tolist(),
            "final_cost": realized_cost(assignment, low_costs, high_costs),
            "final_loss": final_loss,
            "score_values": [float(value) for value in scores],
            "best_seen_curve": best_curve,
            "history": history,
        })
    best_losses = [run["best_seen_loss"] for run in runs]
    final_losses = [run["final_loss"] for run in runs]
    evaluated_variances = [
        run["evaluated_loss_population_variance"] for run in runs]
    return {
        "name": name,
        "seed_count": len(seeds),
        "steps": steps,
        "evaluations_per_seed": 2 * steps,
        "learning_rate": learning_rate,
        "perturbation": perturbation if name == "spsa" else None,
        "baseline_decay": baseline_decay if name == "reinforce" else None,
        "all_evaluations_exact_cost": all(
            run["all_evaluations_exact_cost"] for run in runs),
        "global_optimum_success_count": sum(
            run["reached_global_optimum"] for run in runs),
        "best_seen_loss_mean": statistics.fmean(best_losses),
        "best_seen_loss_median": statistics.median(best_losses),
        "best_seen_loss_population_variance": statistics.pvariance(best_losses),
        "final_loss_mean": statistics.fmean(final_losses),
        "final_loss_median": statistics.median(final_losses),
        "final_loss_population_variance": statistics.pvariance(final_losses),
        "evaluated_loss_variance_mean": statistics.fmean(evaluated_variances),
        "best_seen_convergence": _curve_summary(
            [run["best_seen_curve"] for run in runs]),
        "runs": runs,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type != "cpu":
        raise ValueError("this controlled comparison currently requires CPU")
    model_dir = args.model_dir.resolve(strict=True)
    identity = _load_json(args.identity.resolve(strict=True))
    manifest = _load_json(args.manifest.resolve(strict=True))
    oracle_path = args.oracle.resolve(strict=True)
    oracle, oracle_metadata = _load_oracle(oracle_path)
    if oracle_metadata["revision"] != identity["revision"]:
        raise RuntimeError("oracle and identity revisions differ")
    entries = sorted((
        entry for entry in manifest["entries"]
        if entry["destination_name"].startswith("blk.0.")
        and entry["rco_search"]
    ), key=lambda entry: entry["destination_name"])
    if len(entries) != 13:
        raise RuntimeError(f"expected 13 block groups, found {len(entries)}")

    codec = GGMLNativeCodec(args.ggml_library.resolve(strict=True))
    store_path = args.store.resolve(strict=True)
    store = NativeCandidateStore(store_path, codec)
    names = [entry["destination_name"] for entry in entries]
    low_costs = [
        int(store.metadata(name, GGMLType.Q2_0)["aligned_gguf_bytes"])
        for name in names
    ]
    high_costs = [
        int(store.metadata(name, GGMLType.Q4_0)["aligned_gguf_bytes"])
        for name in names
    ]
    feasible = _feasible_assignments(low_costs, high_costs, args.target_cost)
    # Exercise the production projection against the selected target before
    # paying the model-evaluation cost.
    exact_cost_assignment(
        torch.zeros(len(names)), low_costs, high_costs, args.target_cost)

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(
            config, attn_implementation="eager")
    model.eval()
    adapter = get_model_adapter(model)
    loader = SafeTensorPrefixLoader(model_dir)
    loader.move_runtime_buffers(model, device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    token_batches = []
    for text in CALIBRATION_TEXTS:
        tokens = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=16,
        )["input_ids"]
        if tokens.shape[1] < 16:
            raise RuntimeError(
                f"calibration text produced only {tokens.shape[1]} tokens")
        token_batches.append(tokens[:, :16].to(device))

    embedding_prefix = adapter.embedding_paths[0]
    embedding_schema = loader.assert_prefix_schema(model, embedding_prefix)
    embedding_bytes = loader.load_prefix(
        model, embedding_prefix, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        hidden_batches = [
            adapter.embeddings[0](tokens).contiguous()
            for tokens in token_batches
        ]
    embedding_released = loader.release_prefix(model, embedding_prefix)
    if not torch.equal(hidden_batches[0].cpu(), oracle["block_input"]):
        raise RuntimeError("first calibration embedding differs from retained oracle")

    prefix = f"{adapter.layers_path}.0"
    schema = loader.assert_prefix_schema(model, prefix)
    block_bytes = loader.load_prefix(
        model, prefix, device=device, dtype=torch.bfloat16)
    block = adapter.layers[0]
    block.eval()
    dense_batches = [
        _forward_block(block, hidden, device).detach().cpu().contiguous()
        for hidden in hidden_batches
    ]
    if not torch.equal(dense_batches[0], oracle["block_output"]):
        raise RuntimeError("dense block does not reproduce retained oracle")
    expected = torch.cat(dense_batches, dim=0)

    geometry = Qwen35LinearAttentionGeometry.from_model_dir(model_dir)
    current: tuple[int, ...] | None = None
    landscape_records = []
    max_decoded = 0
    max_install = 0
    assignment_started = time.perf_counter()
    for bits in feasible:
        changed = []
        for index, bit in enumerate(bits):
            if current is not None and current[index] == bit:
                continue
            candidate_type = GGMLType.Q4_0 if bit else GGMLType.Q2_0
            install = _install_candidate(
                model=model,
                store=store,
                entry=entries[index],
                candidate_type=candidate_type,
                geometry=geometry,
                rows_per_chunk=args.rows_per_chunk,
            )
            max_decoded = max(max_decoded, install["max_decoded_fp32_bytes"])
            max_install = max(max_install, install["max_install_bf16_bytes"])
            changed.append(names[index])
        current = bits
        output = torch.cat([
            _forward_block(block, hidden, device).detach().cpu().contiguous()
            for hidden in hidden_batches
        ], dim=0)
        if not torch.isfinite(output).all():
            raise RuntimeError(f"assignment {_key(bits)} produced non-finite output")
        metrics = _losses(expected, output)
        landscape_records.append({
            "assignment_bits": _key(bits),
            "assignment": list(bits),
            "high_tensor_count": sum(bits),
            "realized_aligned_gguf_bytes": sum(
                high_costs[index] if bit else low_costs[index]
                for index, bit in enumerate(bits)),
            "changed_tensors_from_previous": changed,
            "output_sha256": tensor_sha256(output),
            **metrics,
        })
    landscape_seconds = time.perf_counter() - assignment_started
    landscape = {
        record["assignment_bits"]: record["normalized_mse"]
        for record in landscape_records
    }
    optimum_record = min(
        landscape_records, key=lambda record: record["normalized_mse"])

    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    comparisons = [
        _compare_estimator(
            name, landscape, low_costs, high_costs, args.target_cost,
            seeds=seeds,
            steps=args.steps,
            learning_rate=args.learning_rate,
            perturbation=args.perturbation,
            baseline_decay=args.baseline_decay,
            optimum=optimum_record["normalized_mse"],
        )
        for name in ("spsa", "reinforce")
    ]
    default = min(
        comparisons,
        key=lambda item: (
            item["best_seen_loss_median"],
            item["best_seen_loss_mean"],
            item["final_loss_population_variance"],
        ),
    )["name"]

    # Reinstall and reproduce the optimum after every other landscape point.
    optimum_bits = tuple(optimum_record["assignment"])
    for index, bit in enumerate(optimum_bits):
        if current[index] == bit:
            continue
        _install_candidate(
            model=model,
            store=store,
            entry=entries[index],
            candidate_type=(GGMLType.Q4_0 if bit else GGMLType.Q2_0),
            geometry=geometry,
            rows_per_chunk=args.rows_per_chunk,
        )
    reproduced = torch.cat([
        _forward_block(block, hidden, device).detach().cpu().contiguous()
        for hidden in hidden_batches
    ], dim=0)
    if tensor_sha256(reproduced) != optimum_record["output_sha256"]:
        raise RuntimeError("optimum output did not reproduce after traversal")

    released_bytes = loader.release_prefix(model, prefix)
    gc.collect()
    return {
        "schema": 1,
        "status": "pass",
        "scope": (
            "exhaustive exact-byte loss landscape and matched-seed hard SPSA "
            "versus antithetic REINFORCE comparison on the retained genuine "
            "Qwen3.6-35B-A3B block-0 calibration batches; this is a block "
            "quality estimator comparison, not an end-to-end or CUDA gate"
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
        },
        "oracle": {
            "path": str(oracle_path),
            "sha256": _sha256_file(oracle_path),
            "metadata": oracle_metadata,
            "input_ids": oracle["input_ids"].tolist(),
            "block_input_sha256": tensor_sha256(oracle["block_input"]),
            "dense_output_sha256": tensor_sha256(expected),
            "dense_direct_matches_exactly": True,
        },
        "calibration": {
            "batch_count": len(CALIBRATION_TEXTS),
            "sequence_length": 16,
            "texts": list(CALIBRATION_TEXTS),
            "input_ids": [tokens.cpu().tolist()[0] for tokens in token_batches],
            "block_input_sha256": [
                tensor_sha256(hidden.cpu()) for hidden in hidden_batches
            ],
            "dense_output_sha256": [
                tensor_sha256(output) for output in dense_batches
            ],
            "first_batch_matches_retained_oracle_exactly": True,
        },
        "block": {
            "embedding_prefix": embedding_prefix,
            "embedding_schema": embedding_schema,
            "embedding_resident_bf16_bytes": embedding_bytes,
            "embedding_released_bytes": embedding_released,
            "simultaneous_embedding_and_block_residency": False,
            "prefix": prefix,
            "schema": schema,
            "resident_bf16_bytes": block_bytes,
            "released_bytes": released_bytes,
            "released_to_meta": all(
                parameter.device.type == "meta"
                for parameter in adapter.layers[0].parameters()),
        },
        "candidate_store": {
            "path": str(store_path),
            "index_sha256": _sha256_file(
                store_path / "native-candidate-index.json"),
            "tensor_names": names,
            "low_type": GGMLType.Q2_0.name,
            "high_type": GGMLType.Q4_0.name,
            "low_aligned_costs": low_costs,
            "high_aligned_costs": high_costs,
            "uniform_low_cost": sum(low_costs),
            "uniform_high_cost": sum(high_costs),
            "target_cost": args.target_cost,
            "max_decoded_fp32_bytes": max_decoded,
            "max_install_bf16_bytes": max_install,
        },
        "loss_landscape": {
            "metric": "sum_squared_error / sum_squared_dense_output",
            "feasible_assignment_count": len(landscape_records),
            "all_assignments_exact_cost": all(
                record["realized_aligned_gguf_bytes"] == args.target_cost
                for record in landscape_records),
            "evaluation_seconds": landscape_seconds,
            "global_optimum": optimum_record,
            "records": landscape_records,
            "optimum_reproduced_after_traversal": True,
        },
        "comparison": {
            "matched_seeds": seeds,
            "steps_per_seed": args.steps,
            "identical_cached_real_model_landscape": True,
            "estimators": comparisons,
            "selection_rule": (
                "lowest median best-seen loss, then mean best-seen loss, then "
                "final-loss variance"
            ),
            "recommended_default": default,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--target-cost", type=int, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--perturbation", type=float, default=0.25)
    parser.add_argument("--baseline-decay", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(args)
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "feasible_assignments": report["loss_landscape"][
            "feasible_assignment_count"],
        "global_optimum": report["loss_landscape"]["global_optimum"][
            "normalized_mse"],
        "recommended_default": report["comparison"]["recommended_default"],
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
