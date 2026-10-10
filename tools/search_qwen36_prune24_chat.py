#!/usr/bin/env python3
"""Phase 4b of RCO_PLAN_NEW.md: the Phase 3 pruning search on chat corpus v2.

The search is Phase 3's (``StreamedPruneSearch``, same defaults); the data
and the objective change.  The rows are the conversations of the chat v2
calibration half, of 100 to 8,192 tokens, and the objective is the top-20
KL to BF16 over each conversation's scored tokens (the final assistant
turn), averaged per conversation.  ``StreamedChatPruneObjective`` runs
them in length-sorted micro-batches with one block load per direction.

Inputs, all from ``audit_qwen36_chat_kl.py`` with ``--attn-implementation
sdpa`` on the calibration split:

* the BF16 reference (``reference``);
* the unpruned GSQ-E6 score with ``--router-stats``: router probability
  sums for the initial ``alpha`` and selection counts for the frequency
  baseline.

Subcommands:

``baseline-mask``
    The frequency baseline on chat data: each layer's 24 least-selected
    experts.  Score it with ``audit_qwen36_chat_kl.py score --mask``.
``pilot``
    One optimizer step in a scratch directory, for timing; the speed gate
    is at most 30 minutes per step.
``search --seed S``
    A full run, checkpointed after every step and resumable.  Writes the
    final deterministic mask and the run report.

Masks are compared with exact routing by ``audit_qwen36_chat_kl.py score``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_chat_kl import PAD_TOKEN, _load_reference, _model, load_corpus  # noqa: E402
from audit_qwen36_prune24_prelim import (  # noqa: E402
    EXPERTS,
    LAYERS,
    PRUNE_PER_LAYER,
    frequency_prune_mask,
)
from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file  # noqa: E402
from search_qwen36_prune24 import (  # noqa: E402
    STEP_GATE_SECONDS,
    _config,
    _summary_line,
    initial_alpha,
)

SCHEMA = "rco.qwen36.prune24_chat_search.v1"
PREFIX = "qwen36_gsq_e6_prune24_chat"


def _corpus_args(args) -> SimpleNamespace:
    return SimpleNamespace(corpus=args.corpus, split="calibration", limit=None,
                           reports=args.reports, suffix=args.suffix)


def _router_stats(args) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    report = _load_json(args.reports / f"qwen36_chat_kl_score_unpruned{args.suffix}.json")
    stats = report.get("router_stats") if report.get("status") == "complete" else None
    if stats is None:
        raise RuntimeError("the unpruned calibration score has no router statistics")
    path = Path(stats["path"])
    if _sha256_file(path) != stats["sha256"]:
        raise RuntimeError("router statistics differ from their report")
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    return arrays, {"router_stats_sha256": stats["sha256"],
                    "unpruned_score_corpus": report["identity"]["corpus"]}


def run_baseline_mask(args) -> None:
    arrays, identity = _router_stats(args)
    mask = frequency_prune_mask(arrays["counts"], arrays["probability_sum"], PRUNE_PER_LAYER)
    path = args.work / "frequency_mask.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mask)
    counts = arrays["counts"]
    _atomic_json(args.reports / f"{PREFIX}_frequency_mask.json", {
        "schema": SCHEMA + ".frequency_mask",
        "status": "complete",
        **identity,
        "mask": {"path": str(path), "sha256": _sha256_file(path),
                 "pruned_per_layer": PRUNE_PER_LAYER,
                 "rule": ("fewest GSQ-E6 selections on the chat v2 calibration half "
                          "(all real tokens); ties by router probability mass, then index"),
                 "unpruned_routed_slot_share_of_pruned_experts":
                     float(counts[mask].sum() / counts.sum())},
    })
    print(f"baseline-mask: {path}", flush=True)


def _setup(args):
    import torch

    from audit_qwen36_gsq_e6 import NAME, EmbeddingQ6KWeightStore, _entry
    from gguf_parallel_stream import ParallelGGUFManifestPrefixLoader
    from quant.ggml_native import GGMLNativeCodec
    from search.streamed_prune_chat import StreamedChatPruneObjective, rows_from_corpus

    corpus_args = _corpus_args(args)
    conversations, corpus_identity = load_corpus(corpus_args)
    reference, reference_report = _load_reference(corpus_args, corpus_identity)
    if reference_report.get("attn_implementation") != args.attn_implementation:
        raise RuntimeError("the reference used another attention implementation")
    rows = rows_from_corpus(conversations, reference)
    arrays, router_identity = _router_stats(args)
    if router_identity.pop("unpruned_score_corpus") != corpus_identity:
        raise RuntimeError("router statistics come from another corpus split")
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
    objective = StreamedChatPruneObjective(
        _model(args), loader, device=torch.device(args.device),
        checkpoint_dtype=torch.bfloat16, embedding_store=store,
        embedding_selected=[(NAME, 1)], position_chunk=args.position_chunk,
        microbatch_tokens=args.microbatch_tokens, pad_token=PAD_TOKEN)
    identity = {
        "gsq_gguf_sha256": _sha256_file(args.gguf),
        "embedding_sha256": embedding["payload"]["sha256"],
        "corpus": corpus_identity,
        "reference_sha256": reference_report["reference"]["sha256"],
        **router_identity,
        "attn_implementation": args.attn_implementation,
    }
    return objective, rows, arrays["probability_sum"], identity


def _hyperparameters(args) -> dict[str, Any]:
    return {"router_spread": args.router_spread, "rows_per_chunk": args.rows_per_chunk,
            "decode_workers": args.decode_workers, "position_chunk": args.position_chunk,
            "microbatch_tokens": args.microbatch_tokens, "device": args.device}


def _step_line(record: dict[str, Any], steps: int) -> str:
    stats = record["stats"]
    return (_summary_line(record, steps)
            + f" | {stats['rows']} rows {stats['tokens']} tok "
              f"({stats['padded_tokens']} padded, {stats['microbatch_count']} mb)")


def run_pilot(args) -> None:
    from search.streamed_prune import StreamedPruneSearch

    started = time.perf_counter()
    objective, rows, scores, identity = _setup(args)
    config = _config(args)
    search = StreamedPruneSearch(config, initial_alpha(scores, args.router_spread))
    record = search.take_step(objective, rows, rows, rows)
    print(_step_line(record, config.steps), flush=True)
    lengths = np.asarray([row.tokens.numel() for row in rows.rows])
    step_seconds = record["step_seconds"]
    tokens = record["stats"]["tokens"]
    # Compute grows with tokens, loading does not; project with the mean.
    per_token = (step_seconds - record["stats"]["load_seconds"]) / tokens
    expected_tokens = float(lengths.mean()) * len(record["documents"])
    projected = record["stats"]["load_seconds"] + per_token * expected_tokens
    report = {
        "schema": SCHEMA + ".pilot",
        "status": "pass" if projected <= STEP_GATE_SECONDS else "fail",
        "gate": f"projected mean step <= {STEP_GATE_SECONDS} s",
        "identity": identity,
        "config": asdict(config),
        "hyperparameters": _hyperparameters(args),
        "step": {key: value for key, value in record.items()
                 if key != "deterministic_mask_packed"},
        "conversation_tokens": {"mean": float(lengths.mean()), "max": int(lengths.max()),
                                "count": int(lengths.size)},
        "expected_tokens_per_step": expected_tokens,
        "projected_mean_step_seconds": projected,
        "projected_hours_per_run": projected * config.steps / 3600,
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.reports / f"{PREFIX}_pilot.json", report)
    print(f"pilot: {report['status']}, this step {step_seconds:.0f} s for {tokens} tokens; "
          f"projected {projected:.0f} s per mean step, "
          f"{report['projected_hours_per_run']:.1f} h per {config.steps}-step run", flush=True)


def run_search(args) -> None:
    import torch

    from search.streamed_prune import StreamedPruneSearch, deterministic_mask

    run_dir = args.work / f"search_seed{args.seed}"
    output = args.reports / f"{PREFIX}_search_seed{args.seed}.json"
    if output.exists() and _load_json(output).get("status") == "complete":
        print(f"search seed {args.seed}: already complete", flush=True)
        return
    objective, rows, scores, identity = _setup(args)
    config = _config(args)
    run_manifest = {"schema": SCHEMA + ".run", "identity": identity,
                    "config": asdict(config), "hyperparameters": _hyperparameters(args)}
    manifest_path, state_path = run_dir / "run.json", run_dir / "state.pt"
    search = StreamedPruneSearch(config, initial_alpha(scores, args.router_spread))
    if manifest_path.exists():
        if _load_json(manifest_path) != json.loads(json.dumps(run_manifest)):
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
        record = search.take_step(objective, rows, rows, rows)
        search.save(state_path)
        recent = search.history[-20:]
        remaining = (config.steps - search.step) * float(
            np.mean([item["step_seconds"] for item in recent])) / 3600
        print(_step_line(record, config.steps) + f" | ETA {remaining:.1f} h", flush=True)

    mask = deterministic_mask(search.alpha, PRUNE_PER_LAYER, LAYERS, EXPERTS).numpy()
    mask_path = run_dir / "final_mask.npy"
    np.save(mask_path, mask)
    initial = np.load(run_dir / "initial_mask.npy")
    frequency_path = args.work / "frequency_mask.npy"
    frequency = np.load(frequency_path) if frequency_path.exists() else None
    probabilities = torch.softmax(search.alpha.detach(), dim=1)[:, 1].view(LAYERS, EXPERTS)
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
            "pruned_probability_min": float(probabilities[torch.from_numpy(mask)].min()),
            "kept_probability_max": float(probabilities[~torch.from_numpy(mask)].max()),
        },
        "state": {"path": str(state_path), "sha256": _sha256_file(state_path)},
        "history": history,
        "search_hours": sum(item["step_seconds"] for item in history) / 3600,
        "note": "surrogate search only; masks are compared by exact scoring",
    })
    print(f"search seed {args.seed}: complete, final mask {mask_path}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("baseline-mask", "pilot", "search"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--tau-init", type=float, default=1.0)
    parser.add_argument("--tau-min", type=float, default=0.05)
    parser.add_argument("--gumbel-samples", type=int, default=4)
    parser.add_argument("--documents-per-sample", type=int, default=4,
                        help="conversations per antithetic pair")
    parser.add_argument("--router-spread", type=float, default=5.0)
    parser.add_argument("--corpus", type=Path,
                        default=RCO / "reports/qwen36_chat_corpus_v2_manifest.json")
    parser.add_argument("--suffix", default="_v2_calibration",
                        help="report suffix of the chat KL reference and unpruned score")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/qwen36_35b_base")
    parser.add_argument("--manifest", type=Path,
                        default=RCO / "reports/qwen36_35b_base_gguf_manifest.json")
    parser.add_argument("--gguf", type=Path,
                        default=ROOT / "results/Qwen3.6-35B-A3B-GSQ-hybrid.gguf")
    parser.add_argument("--gguf-python", type=Path, default=ROOT / "repos/llama.cpp/gguf-py")
    parser.add_argument("--ggml-library", type=Path,
                        default=ROOT / "experiment/build-cpu/bin/libggml-base.so.0.24.0")
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_prune24_chat")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa", choices=("sdpa",))
    parser.add_argument("--rows-per-chunk", type=int, default=1024)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--position-chunk", type=int, default=256)
    parser.add_argument("--microbatch-tokens", type=int, default=8448)
    args = parser.parse_args()
    for key in ("corpus", "model_dir", "manifest", "gguf", "gguf_python",
                "ggml_library", "work", "reports"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main() -> int:
    args = _parse_args()
    {"baseline-mask": run_baseline_mask, "pilot": run_pilot,
     "search": run_search}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
