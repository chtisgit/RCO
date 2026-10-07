#!/usr/bin/env python3
"""Phase 3 steps d-e of RCO_PLAN_NEW.md: streamed RCO expert-pruning search.

The model is GSQ-E6, which is authentic GSQ with the Phase 1 Q6_K
token_embd.  Exactly 24 of 256 experts are pruned in every layer.  The
objective is the top-20 KL to the cached BF16 reference from step a.
``alpha`` is initialised from step b's router probability sums, using
``init_alpha_from_router_scores``.

Subcommands:

``pilot``
    One optimizer step in a scratch directory, to measure timing (step d).
    It writes ``reports/qwen36_gsq_e6_prune24_pilot.json`` and applies the
    speed gate of at most 30 minutes per step.
``search --seed S``
    A full run (step e).  It checkpoints atomically after every step, and a
    restart resumes the trajectory.  When finished, it writes the
    deterministic final mask and
    ``reports/qwen36_gsq_e6_prune24_search_seed{S}.json``.
``score --mask M --label L``
    Exact scoring of a 40 x 256 mask on all of v2 (steps f and g).  It
    uses the step a-c scorer: pruned router logits are set to -inf, then
    softmax, top-8, and renormalization; per-document top-20 KL to BF16 and
    NLL.  The result is compared with the frequency baseline by paired
    bootstrap.  All mask comparisons use this one scorer.  The search itself
    optimizes RCO's surrogate objective, where the reference top-20 is
    renormalized.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_prune24_prelim import (  # noqa: E402
    EXPERTS,
    LAYERS,
    PRUNE_PER_LAYER,
    _load_reference,
)
from audit_qwen36_q3k_viability import (  # noqa: E402
    _atomic_json,
    _calibration,
    _empty_model,
    _load_json,
    _sha256_file,
)


SCHEMA = "rco.qwen36.prune24_search.v1"
STEP_GATE_SECONDS = 30 * 60


def _router_scores(args) -> tuple[np.ndarray, dict[str, Any]]:
    report = _load_json(args.reports / "qwen36_gsq_e6_prune24_router_stats.json")
    if report.get("status") != "complete":
        raise RuntimeError("router statistics are not complete")
    path = Path(report["router_stats"]["path"])
    sha256 = _sha256_file(path)
    if sha256 != report["router_stats"]["sha256"]:
        raise RuntimeError("router statistics differ from their report")
    with np.load(path) as data:
        return data["probability_sum"], {"router_stats_sha256": sha256}


def initial_alpha(probability_sum: np.ndarray, spread: float):
    """``init_alpha_from_router_scores`` on the GSQ-E6 router probability sums."""
    import torch

    from search.prune import init_alpha_from_router_scores

    alpha = torch.zeros(LAYERS * EXPERTS, 2, dtype=torch.float32)
    init_alpha_from_router_scores(
        alpha, torch.from_numpy(np.asarray(probability_sum, dtype=np.float32)),
        PRUNE_PER_LAYER * LAYERS, LAYERS, EXPERTS, spread=spread)
    return alpha


def _setup(args):
    import torch

    from audit_qwen36_gsq_e6 import NAME, EmbeddingQ6KWeightStore, _entry
    from gguf_parallel_stream import ParallelGGUFManifestPrefixLoader
    from quant.ggml_native import GGMLNativeCodec
    from search.streamed_prune import StreamedPruneObjective

    values, indices, reference = _load_reference(args)
    input_ids, calibration = _calibration(args)
    if calibration != reference["calibration"]:
        raise RuntimeError("reference was built on a different calibration corpus")
    scores, router_identity = _router_scores(args)
    embedding = _load_json(args.reports / "qwen36_gsq_e6_embedding.json")
    if _load_json(args.reports / "qwen36_gsq_e6_calibration.json").get("status") != "pass":
        raise RuntimeError("Phase 1 GSQ-E6 has not passed")
    store = EmbeddingQ6KWeightStore(
        GGMLNativeCodec(args.ggml_library), Path(embedding["payload"]["path"]),
        embedding["payload"]["sha256"], _entry(args.manifest), rows_per_chunk=4096)
    loader = ParallelGGUFManifestPrefixLoader(
        args.gguf, _load_json(args.manifest), args.model_dir,
        gguf_python=args.gguf_python, ggml_library=args.ggml_library,
        rows_per_chunk=args.rows_per_chunk, workers=args.decode_workers)
    model = _empty_model(args.model_dir)
    objective = StreamedPruneObjective(
        model, loader, device=torch.device(args.device),
        checkpoint_dtype=torch.bfloat16, embedding_store=store,
        embedding_selected=[(NAME, 1)], position_chunk=args.position_chunk)
    identity = {
        "gsq_gguf_sha256": _sha256_file(args.gguf),
        "embedding_sha256": embedding["payload"]["sha256"],
        "reference_sha256": reference["reference"]["sha256"],
        **router_identity,
        "calibration_manifest_sha256": calibration["calibration_manifest_sha256"],
        "calibration_token_sha256": calibration["calibration_token_sha256"],
    }
    return (objective, input_ids, torch.from_numpy(values),
            torch.from_numpy(indices), scores, identity)


def _config(args):
    from search.streamed_prune import StreamedPruneConfig

    return StreamedPruneConfig(
        layers=LAYERS, experts=EXPERTS, prune_per_layer=PRUNE_PER_LAYER,
        steps=args.steps, lr=args.lr, tau_init=args.tau_init, tau_min=args.tau_min,
        gumbel_samples=args.gumbel_samples, antithetic=True,
        documents_per_sample=args.documents_per_sample, seed=args.seed)


def _summary_line(record: dict[str, Any], steps: int) -> str:
    stats = record["stats"]
    return (
        f"step {record['step'] + 1}/{steps} tau {record['tau']:.3f} "
        f"KL {record['objective']:.5f} "
        f"[{' '.join(f'{value:.4f}' for value in record['variant_kl'])}] "
        f"grad {record['projected_grad_norm']:.3g} "
        f"E[prune] {record['expected_prune_per_layer']:.2f} "
        f"decided>0.99 {record['decided_0_99']} "
        f"swaps {record['sample_masks_differ_from_deterministic']} | "
        f"load {stats['load_seconds']:.0f}s fwd {stats['forward_seconds']:.0f}s "
        f"loss {stats['loss_seconds']:.0f}s bwd {stats['backward_seconds']:.0f}s "
        f"step {record['step_seconds']:.0f}s "
        f"peak {stats['cuda_max_allocated'] / 2**30:.1f} GiB")


def _hyperparameters(args) -> dict[str, Any]:
    return {
        "router_spread": args.router_spread,
        "rows_per_chunk": args.rows_per_chunk,
        "decode_workers": args.decode_workers,
        "position_chunk": args.position_chunk,
        "device": args.device,
    }


def run_pilot(args: argparse.Namespace) -> None:
    from search.streamed_prune import StreamedPruneSearch

    started = time.perf_counter()
    objective, input_ids, values, indices, scores, identity = _setup(args)
    config = _config(args)
    search = StreamedPruneSearch(config, initial_alpha(scores, args.router_spread))
    record = search.take_step(objective, input_ids, values, indices)
    print(_summary_line(record, config.steps), flush=True)
    step_seconds = record["step_seconds"]
    report = {
        "schema": SCHEMA + ".pilot",
        "status": "pass" if step_seconds <= STEP_GATE_SECONDS else "fail",
        "gate": f"one optimizer step <= {STEP_GATE_SECONDS} s",
        "identity": identity,
        "config": asdict(config),
        "hyperparameters": _hyperparameters(args),
        "rows_per_step": len(record["documents"]),
        "tokens_per_step": len(record["documents"]) * int(input_ids.shape[1]),
        "step": {key: value for key, value in record.items()
                 if key != "deterministic_mask_packed"},
        "projected_hours_per_run": step_seconds * config.steps / 3600,
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.reports / "qwen36_gsq_e6_prune24_pilot.json", report)
    print(f"pilot: {report['status']}, {step_seconds:.0f} s per step, "
          f"{report['projected_hours_per_run']:.1f} h per {config.steps}-step run",
          flush=True)


def run_search(args: argparse.Namespace) -> None:
    import torch

    from search.streamed_prune import StreamedPruneSearch, deterministic_mask

    run_dir = args.work / f"search_seed{args.seed}"
    output = args.reports / f"qwen36_gsq_e6_prune24_search_seed{args.seed}.json"
    if output.exists() and _load_json(output).get("status") == "complete":
        print(f"search seed {args.seed}: already complete", flush=True)
        return
    objective, input_ids, values, indices, scores, identity = _setup(args)
    config = _config(args)
    run_manifest = {
        "schema": SCHEMA + ".run",
        "identity": identity,
        "config": asdict(config),
        "hyperparameters": _hyperparameters(args),
    }
    manifest_path = run_dir / "run.json"
    state_path = run_dir / "state.pt"
    search = StreamedPruneSearch(config, initial_alpha(scores, args.router_spread))
    if manifest_path.exists():
        previous = _load_json(manifest_path)
        if previous != json.loads(json.dumps(run_manifest)):
            raise RuntimeError(f"{manifest_path} differs from this run; refusing to resume")
        if state_path.exists():
            search.load(state_path)
            print(f"search seed {args.seed}: resuming at step {search.step}", flush=True)
    else:
        _atomic_json(manifest_path, run_manifest)
    initial_mask = deterministic_mask(search.alpha, PRUNE_PER_LAYER, LAYERS, EXPERTS)
    if search.step == 0:
        np.save(run_dir / "initial_mask.npy", initial_mask.numpy())

    while search.step < config.steps:
        record = search.take_step(objective, input_ids, values, indices)
        search.save(state_path)
        remaining = (config.steps - search.step) * record["step_seconds"] / 3600
        print(_summary_line(record, config.steps) + f" | ETA {remaining:.1f} h",
              flush=True)

    mask = deterministic_mask(search.alpha, PRUNE_PER_LAYER, LAYERS, EXPERTS).numpy()
    mask_path = run_dir / "final_mask.npy"
    np.save(mask_path, mask)
    initial = np.load(run_dir / "initial_mask.npy")
    frequency_path = args.work / "frequency_mask.npy"
    frequency = np.load(frequency_path) if frequency_path.exists() else None
    probabilities = torch.softmax(search.alpha.detach(), dim=1)[:, 1]
    history = [{key: value for key, value in item.items()
                if key != "deterministic_mask_packed"} for item in search.history]
    _atomic_json(output, {
        "schema": SCHEMA + ".search",
        "status": "complete",
        **run_manifest,
        "final_mask": {
            "path": str(mask_path),
            "sha256": _sha256_file(mask_path),
            "pruned_per_layer": mask.sum(axis=1).tolist(),
            "experts_changed_from_router_init": int((mask != initial).sum()) // 2,
            "experts_changed_from_frequency_baseline": (
                int((mask != frequency).sum()) // 2 if frequency is not None else None),
            "pruned_probability_min": float(probabilities.view(LAYERS, EXPERTS)[
                torch.from_numpy(mask)].min()),
            "kept_probability_max": float(probabilities.view(LAYERS, EXPERTS)[
                ~torch.from_numpy(mask)].max()),
        },
        "state": {"path": str(state_path), "sha256": _sha256_file(state_path)},
        "history": history,
        "search_hours": sum(item["step_seconds"] for item in history) / 3600,
        "note": (
            "surrogate search only; exact scoring and promotion are step f"),
    })
    print(f"search seed {args.seed}: complete, final mask {mask_path}", flush=True)


def run_score(args: argparse.Namespace) -> None:
    from audit_qwen36_prune24_prelim import _score_gsq_e6
    from release_quality import paired_bootstrap_mean_ci

    if not args.label or args.mask is None:
        raise SystemExit("score needs --mask and --label")
    output = args.reports / f"qwen36_gsq_e6_prune24_score_{args.label}.json"
    if output.exists() and _load_json(output).get("status") == "complete":
        print(f"score {args.label}: already complete", flush=True)
        return
    started = time.perf_counter()
    mask = np.load(args.mask)
    if mask.shape != (LAYERS, EXPERTS) or mask.dtype != bool:
        raise ValueError("mask must be a 40 x 256 boolean array")
    if not bool((mask.sum(axis=1) == PRUNE_PER_LAYER).all()):
        raise ValueError(f"mask must prune {PRUNE_PER_LAYER} experts per layer")
    mask_sha256 = _sha256_file(args.mask)
    summary = _score_gsq_e6(args, f"score_{args.label}", mask, False)
    comparison = None
    baseline_path = args.reports / "qwen36_gsq_e6_prune24_frequency_baseline.json"
    if baseline_path.exists():
        baseline = _load_json(baseline_path)
        comparison = {"baseline_mask_sha256": baseline["mask"]["sha256"],
                      "baseline_mean_top20_kl_to_bf16": baseline["mean_top20_kl_to_bf16"],
                      "baseline_mean_nll": baseline["mean_nll"]}
        frequency = np.load(baseline["mask"]["path"])
        comparison["experts_changed"] = int((mask != frequency).sum()) // 2
        for key in ("top20_kl", "nll"):
            delta = (np.asarray(summary[f"document_mean_{key}"])
                     - np.asarray(baseline[f"document_mean_{key}"]))
            lower, upper = paired_bootstrap_mean_ci(
                delta.tolist(), samples=10_000, seed=20261001)
            comparison[f"delta_{key}"] = {
                "mean": float(delta.mean()), "ci95_lower": lower, "ci95_upper": upper}
    _atomic_json(output, {
        "schema": SCHEMA + ".score",
        "status": "complete",
        "semantics": (
            "exact: pruned router logits -inf, softmax, top-8, renormalize "
            "(equivalent to llama.cpp qwen35moe with experts removed)"),
        "mask": {"path": str(args.mask), "sha256": mask_sha256,
                 "pruned_per_layer": PRUNE_PER_LAYER},
        **summary,
        "versus_frequency_baseline": comparison,
        "wall_seconds": time.perf_counter() - started,
    })
    line = (f"score {args.label}: KL {summary['mean_top20_kl_to_bf16']:.5f} "
            f"NLL {summary['mean_nll']:.5f}")
    if comparison is not None:
        kl = comparison["delta_top20_kl"]
        line += (f"; vs frequency baseline KL {kl['mean']:+.5f} "
                 f"[{kl['ci95_lower']:+.5f}, {kl['ci95_upper']:+.5f}]")
    print(line, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("pilot", "search", "score"))
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--tau-init", type=float, default=1.0)
    parser.add_argument("--tau-min", type=float, default=0.05)
    parser.add_argument("--gumbel-samples", type=int, default=4)
    parser.add_argument("--documents-per-sample", type=int, default=4)
    parser.add_argument("--router-spread", type=float, default=5.0)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/qwen36_35b_base")
    parser.add_argument("--manifest", type=Path,
                        default=RCO / "reports/qwen36_35b_base_gguf_manifest.json")
    parser.add_argument("--calibration", type=Path,
                        default=RCO / "reports/qwen36_35b_calibration_corpus_v2_manifest.json")
    parser.add_argument("--gguf", type=Path,
                        default=ROOT / "results/Qwen3.6-35B-A3B-GSQ-hybrid.gguf")
    parser.add_argument("--gguf-python", type=Path,
                        default=ROOT / "repos/llama.cpp/gguf-py")
    parser.add_argument("--ggml-library", type=Path,
                        default=ROOT / "experiment/build-cpu/bin/libggml-base.so.0.24.0")
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_prune24")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows-per-chunk", type=int, default=1024)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--position-chunk", type=int, default=256)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--batch-documents", type=int, default=25)
    args = parser.parse_args()
    for key in ("model_dir", "manifest", "calibration", "gguf", "gguf_python",
                "ggml_library", "work", "reports"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main() -> int:
    args = _parse_args()
    if args.mask is not None:
        args.mask = args.mask.resolve()
    {"pilot": run_pilot, "search": run_search, "score": run_score}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
