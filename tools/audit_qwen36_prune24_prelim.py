#!/usr/bin/env python3
"""Phase 3 steps a-c of RCO_PLAN_NEW.md: forward-only pruning groundwork.

Subcommands, each resumable per document batch and run in this order:

``reference``
    Streamed BF16 forward over calibration v2.  For every predicted position
    it stores the top-20 next-token log-probabilities and their token ids.
    This is the teacher for the pruning objective (top-20 KL).
``router-stats``
    Streamed GSQ-E6 forward (authentic GSQ with the Phase 1 Q6_K token_embd)
    over v2.  Per expert: selection count, routed-weight sum, and full router
    probability sum.  Also the unpruned GSQ-E6 top-20 KL to BF16 and NLL per
    document.
``baseline``
    Frequency baseline: prune each layer's 24 least-selected experts
    (ties: lower router-probability mass, then lower index).  Exact GSQ-E6
    scoring under llama.cpp pruning semantics: pruned router logits are set
    to -inf, then softmax, top-8, and renormalization.  With experts
    physically removed, llama.cpp ``qwen35moe`` does softmax over the kept
    experts, top-k, and normalizes the weights, which gives the same result.
"""

from __future__ import annotations

import argparse
import math
import os
import re
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

from audit_qwen36_q3k_viability import (  # noqa: E402
    _atomic_json,
    _calibration,
    _empty_model,
    _load_json,
    _sha256_file,
)


SCHEMA = "rco.qwen36.prune24_prelim.v1"
LAYERS = 40
EXPERTS = 256
PRUNE_PER_LAYER = 24
TOP_K_REFERENCE = 20
POSITION_CHUNK = 256


# --------------------------------------------------------------------------
# pure helpers (unit tested)
# --------------------------------------------------------------------------


def exact_pruned_routing(router_logits, prune_mask, top_k: int):
    """Return llama.cpp-equivalent routing with pruned experts removed."""
    import torch

    masked = router_logits.masked_fill(prune_mask, float("-inf"))
    probabilities = torch.softmax(masked, dim=-1, dtype=torch.float)
    weights, indices = torch.topk(probabilities, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return masked, weights.to(router_logits.dtype), indices


def frequency_prune_mask(counts: np.ndarray, probability_mass: np.ndarray,
                         per_layer: int) -> np.ndarray:
    """Prune the ``per_layer`` least-selected experts in every layer."""
    counts = np.asarray(counts)
    mass = np.asarray(probability_mass)
    if counts.shape != mass.shape or counts.ndim != 2:
        raise ValueError("counts and probability mass must be [layers, experts]")
    if not 0 < per_layer < counts.shape[1]:
        raise ValueError("per-layer prune count is out of range")
    mask = np.zeros(counts.shape, dtype=bool)
    for layer in range(counts.shape[0]):
        experts = np.arange(counts.shape[1])
        order = np.lexsort((experts, mass[layer], counts[layer]))
        mask[layer, order[:per_layer]] = True
    return mask


def top_k_kl(reference_values, reference_indices, model_log_probs):
    """KL(reference || model) restricted to the reference top-k tokens."""
    import torch

    gathered = model_log_probs.gather(-1, reference_indices.long())
    return (reference_values.exp() * (reference_values - gathered)).sum(dim=-1)


# --------------------------------------------------------------------------
# streamed evaluation plumbing
# --------------------------------------------------------------------------


class LossTap:
    """Wrap the evaluator's loss to read top-k log-probs from the final hidden states."""

    def __init__(self) -> None:
        import search.streaming as streaming

        self.streaming = streaming
        self.original = streaming._chunked_causal_cross_entropy_details
        self.mode: str | None = None
        self.reference: tuple[Any, Any] | None = None
        self.result: Any = None
        streaming._chunked_causal_cross_entropy_details = self._wrapped

    def close(self) -> None:
        self.streaming._chunked_causal_cross_entropy_details = self.original

    def _wrapped(self, hidden_states, labels, lm_head, **kwargs):
        import torch

        batch, sequence, _ = hidden_states.shape
        predicted = sequence - 1
        if self.mode == "reference":
            values = np.empty((batch, predicted, TOP_K_REFERENCE), dtype=np.float32)
            indices = np.empty((batch, predicted, TOP_K_REFERENCE), dtype=np.int32)
        elif self.mode == "compare":
            reference_values, reference_indices = self.reference
            kl = np.empty((batch, predicted), dtype=np.float64)
        else:
            raise RuntimeError("loss tap mode is not set")
        for row in range(batch):
            for start in range(0, predicted, POSITION_CHUNK):
                stop = min(start + POSITION_CHUNK, predicted)
                logits = lm_head(hidden_states[row, start:stop]).float()
                log_probs = torch.log_softmax(logits, dim=-1)
                if self.mode == "reference":
                    top_values, top_indices = log_probs.topk(TOP_K_REFERENCE, dim=-1)
                    values[row, start:stop] = top_values.cpu().numpy()
                    indices[row, start:stop] = top_indices.int().cpu().numpy()
                else:
                    device = log_probs.device
                    kl[row, start:stop] = top_k_kl(
                        torch.from_numpy(reference_values[row, start:stop]).to(device),
                        torch.from_numpy(reference_indices[row, start:stop]).to(device),
                        log_probs).double().cpu().numpy()
                del logits, log_probs
        self.result = (values, indices) if self.mode == "reference" else kl
        return self.original(hidden_states, labels, lm_head, **kwargs)


def _router_modules(model) -> dict[int, Any]:
    modules = {}
    for name, module in model.named_modules():
        if type(module).__name__ == "Qwen3_5MoeTopKRouter":
            modules[int(re.search(r"layers\.(\d+)\.", name).group(1))] = module
    if sorted(modules) != list(range(LAYERS)):
        raise RuntimeError(f"found router layers {sorted(modules)}")
    return modules


class RouterHooks:
    """Optionally apply exact pruning, and optionally accumulate router statistics."""

    def __init__(self, model, *, prune_mask: np.ndarray | None, collect: bool) -> None:
        import torch

        self.torch = torch
        self.prune_mask = prune_mask
        self.collect = collect
        # Optional flattened bool mask of the tokens to count (excludes padding).
        self.valid = None
        self.counts = np.zeros((LAYERS, EXPERTS), dtype=np.int64)
        self.weight_sum = np.zeros((LAYERS, EXPERTS), dtype=np.float64)
        self.probability_sum = np.zeros((LAYERS, EXPERTS), dtype=np.float64)
        self.handles = [
            module.register_forward_hook(self._hook(layer))
            for layer, module in _router_modules(model).items()]

    def _hook(self, layer: int):
        torch = self.torch

        def hook(module, inputs, output):
            router_logits, weights, indices = output
            if self.prune_mask is not None:
                mask = torch.from_numpy(self.prune_mask[layer]).to(router_logits.device)
                router_logits, weights, indices = exact_pruned_routing(
                    router_logits, mask, module.top_k)
                if bool(mask[indices].any()):
                    raise RuntimeError(f"pruned expert selected in layer {layer}")
            if self.collect:
                counted_logits, counted_weights, counted = router_logits, weights, indices
                if self.valid is not None:
                    if self.valid.shape[0] != indices.shape[0]:
                        raise RuntimeError("valid-token mask does not match the routed tokens")
                    valid = self.valid.to(indices.device)
                    counted_logits, counted_weights, counted = (
                        router_logits[valid], weights[valid], indices[valid])
                flat = counted.reshape(-1)
                self.counts[layer] += torch.bincount(
                    flat, minlength=EXPERTS).cpu().numpy()
                weight_sum = torch.zeros(EXPERTS, dtype=torch.float64,
                                         device=flat.device)
                weight_sum.index_add_(0, flat, counted_weights.reshape(-1).double())
                self.weight_sum[layer] += weight_sum.cpu().numpy()
                self.probability_sum[layer] += torch.softmax(
                    counted_logits, dim=-1, dtype=torch.float).double().sum(
                        dim=0).cpu().numpy()
            return router_logits, weights, indices
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _batches(document_count: int, size: int) -> list[list[int]]:
    return [list(range(start, min(start + size, document_count)))
            for start in range(0, document_count, size)]


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _gsq_e6_evaluator(args, model):
    import torch

    from audit_qwen36_gsq_e6 import NAME, EmbeddingQ6KWeightStore, _entry
    from gguf_parallel_stream import ParallelGGUFManifestPrefixLoader
    from quant.ggml_native import GGMLNativeCodec
    from search.streaming import StreamingHardCausalEvaluator

    embedding = _load_json(args.reports / "qwen36_gsq_e6_embedding.json")
    calibration = _load_json(args.reports / "qwen36_gsq_e6_calibration.json")
    if calibration.get("status") != "pass":
        raise RuntimeError("Phase 1 GSQ-E6 has not passed")
    store = EmbeddingQ6KWeightStore(
        GGMLNativeCodec(args.ggml_library), Path(embedding["payload"]["path"]),
        embedding["payload"]["sha256"], _entry(args.manifest),
        rows_per_chunk=4096)
    # Bit-identical to GGUFManifestPrefixLoader (tests/test_gguf_parallel_stream.py).
    loader = ParallelGGUFManifestPrefixLoader(
        args.gguf, _load_json(args.manifest), args.model_dir,
        gguf_python=args.gguf_python, ggml_library=args.ggml_library,
        rows_per_chunk=args.rows_per_chunk, workers=args.decode_workers)
    evaluator = StreamingHardCausalEvaluator(
        model, loader, store, [SimpleNamespace(layer_names=(NAME,))], [0, 1],
        device=torch.device(args.device), vocab_chunk_size=args.vocab_chunk_size,
        checkpoint_dtype=torch.bfloat16)
    return evaluator, torch.ones(1, dtype=torch.long), {
        "gsq_gguf_sha256": _sha256_file(args.gguf),
        "embedding_sha256": embedding["payload"]["sha256"],
    }


def _load_reference(args) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    report = _load_json(args.reports / "qwen36_gsq_e6_prune24_reference.json")
    if report.get("status") != "complete":
        raise RuntimeError("BF16 reference is not complete")
    path = Path(report["reference"]["path"])
    if _sha256_file(path) != report["reference"]["sha256"]:
        raise RuntimeError("BF16 reference file differs from its report")
    with np.load(path) as data:
        return data["values"], data["indices"], report


def _run_batches(args, evaluator, assignment, input_ids, tap, label, before_batch):
    """Run every document batch once, saving per-batch results for resumption."""
    batch_dir = args.work / label
    results = []
    for index, documents in enumerate(_batches(input_ids.shape[0], args.batch_documents)):
        path = batch_dir / f"batch{index:03d}.npz"
        if path.exists():
            with np.load(path) as data:
                results.append({key: data[key] for key in data.files})
            continue
        before_batch(documents)
        started = time.perf_counter()
        evaluation = evaluator.evaluate(input_ids[documents], assignment)
        arrays = {
            "documents": np.asarray(documents),
            "document_mean_nll": np.asarray(evaluation.document_mean_nll),
        }
        if tap.mode == "reference":
            arrays["values"], arrays["indices"] = tap.result
        else:
            arrays["kl"] = tap.result
        extra = getattr(before_batch, "collect", None)
        if extra is not None:
            arrays.update(extra())
        _save_npz(path, **arrays)
        results.append(arrays)
        print(f"{label}: batch {index} ({len(documents)} documents) NLL "
              f"{float(np.mean(arrays['document_mean_nll'])):.5f} in "
              f"{time.perf_counter() - started:.0f}s "
              f"(peak RSS {evaluation.memory.process_peak_rss / 2**30:.1f} GiB)",
              flush=True)
    return results


def run_reference(args: argparse.Namespace) -> None:
    import torch

    from checkpoint_stream import SafeTensorPrefixLoader
    from search.streaming import StreamingHardCausalEvaluator

    output = args.reports / "qwen36_gsq_e6_prune24_reference.json"
    if output.exists() and _load_json(output).get("status") == "complete":
        print("reference: already complete", flush=True)
        return
    started = time.perf_counter()
    input_ids, calibration = _calibration(args)
    model = _empty_model(args.model_dir)
    evaluator = StreamingHardCausalEvaluator(
        model, SafeTensorPrefixLoader(args.model_dir), SimpleNamespace(), [], [0, 1],
        device=torch.device(args.device), vocab_chunk_size=args.vocab_chunk_size)
    tap = LossTap()
    tap.mode = "reference"
    try:
        results = _run_batches(
            args, evaluator, torch.zeros(0, dtype=torch.long), input_ids, tap,
            "reference", lambda documents: None)
    finally:
        tap.close()
    values = np.concatenate([item["values"] for item in results])
    indices = np.concatenate([item["indices"] for item in results])
    nll = np.concatenate([item["document_mean_nll"] for item in results])
    path = args.work / "reference_bf16_top20.npz"
    _save_npz(path, values=values, indices=indices)
    _atomic_json(output, {
        "schema": SCHEMA + ".reference",
        "status": "complete",
        "teacher": "pinned BF16 safetensors, streamed",
        "model_dir": str(args.model_dir),
        "calibration": calibration,
        "reference": {
            "path": str(path),
            "sha256": _sha256_file(path),
            "top_k": TOP_K_REFERENCE,
            "shape": list(values.shape),
            "values": "float32 log-probabilities, descending",
            "indices": "int32 token ids",
            "mean_top_k_probability_mass": float(np.exp(values).sum(-1).mean()),
        },
        "bf16_mean_nll": float(nll.mean()),
        "document_mean_nll": nll.tolist(),
        "batch_documents": args.batch_documents,
        "wall_seconds": time.perf_counter() - started,
    })
    print(f"reference: complete, BF16 NLL {nll.mean():.5f}", flush=True)


def _score_gsq_e6(args, label, prune_mask, collect):
    import torch  # noqa: F401

    values, indices, reference = _load_reference(args)
    input_ids, calibration = _calibration(args)
    if calibration != reference["calibration"]:
        raise RuntimeError("reference was built on a different calibration corpus")
    model = _empty_model(args.model_dir)
    evaluator, assignment, identity = _gsq_e6_evaluator(args, model)
    hooks = RouterHooks(model, prune_mask=prune_mask, collect=collect)
    tap = LossTap()
    tap.mode = "compare"

    def before_batch(documents):
        tap.reference = (values[documents], indices[documents])
        hooks.counts[:] = 0
        hooks.weight_sum[:] = 0
        hooks.probability_sum[:] = 0

    if collect:
        before_batch.collect = lambda: {
            "counts": hooks.counts.copy(),
            "weight_sum": hooks.weight_sum.copy(),
            "probability_sum": hooks.probability_sum.copy(),
        }
    try:
        results = _run_batches(args, evaluator, assignment, input_ids, tap,
                               label, before_batch)
    finally:
        tap.close()
        hooks.close()
    kl = np.concatenate([item["kl"] for item in results])
    nll = np.concatenate([item["document_mean_nll"] for item in results])
    summary = {
        "identity": {**identity, "calibration": calibration,
                     "reference_sha256": reference["reference"]["sha256"]},
        "mean_top20_kl_to_bf16": float(kl.mean()),
        "mean_nll": float(nll.mean()),
        "document_mean_top20_kl": kl.mean(axis=1).tolist(),
        "document_mean_nll": nll.tolist(),
        "batch_documents": args.batch_documents,
    }
    if collect:
        summary["_router"] = {
            key: sum(item[key] for item in results)
            for key in ("counts", "weight_sum", "probability_sum")}
    return summary


def run_router_stats(args: argparse.Namespace) -> None:
    output = args.reports / "qwen36_gsq_e6_prune24_router_stats.json"
    if output.exists() and _load_json(output).get("status") == "complete":
        print("router-stats: already complete", flush=True)
        return
    started = time.perf_counter()
    summary = _score_gsq_e6(args, "router_stats", None, True)
    router = summary.pop("_router")
    path = args.work / "router_stats_gsq_e6.npz"
    _save_npz(path, **router)
    counts = router["counts"]
    _atomic_json(output, {
        "schema": SCHEMA + ".router_stats",
        "status": "complete",
        "model": "GSQ-E6 (authentic GSQ + Phase 1 Q6_K token_embd), unpruned",
        **summary,
        "router_stats": {
            "path": str(path),
            "sha256": _sha256_file(path),
            "routed_slots_per_layer": int(counts[0].sum()),
            "experts_never_selected": int((counts == 0).sum()),
            "min_count": int(counts.min()),
            "median_count": float(np.median(counts)),
            "max_count": int(counts.max()),
        },
        "wall_seconds": time.perf_counter() - started,
    })
    print(f"router-stats: complete, GSQ-E6 KL {summary['mean_top20_kl_to_bf16']:.5f} "
          f"NLL {summary['mean_nll']:.5f}", flush=True)


def run_baseline(args: argparse.Namespace) -> None:
    from release_quality import paired_bootstrap_mean_ci

    output = args.reports / "qwen36_gsq_e6_prune24_frequency_baseline.json"
    if output.exists() and _load_json(output).get("status") == "complete":
        print("baseline: already complete", flush=True)
        return
    started = time.perf_counter()
    stats_report = _load_json(args.reports / "qwen36_gsq_e6_prune24_router_stats.json")
    stats_path = Path(stats_report["router_stats"]["path"])
    if _sha256_file(stats_path) != stats_report["router_stats"]["sha256"]:
        raise RuntimeError("router statistics differ from their report")
    with np.load(stats_path) as data:
        mask = frequency_prune_mask(
            data["counts"], data["probability_sum"], PRUNE_PER_LAYER)
        counts = data["counts"]
    mask_path = args.work / "frequency_mask.npy"
    np.save(mask_path, mask)
    summary = _score_gsq_e6(args, "frequency_baseline", mask, False)
    unpruned_kl = np.asarray(stats_report["document_mean_top20_kl"])
    unpruned_nll = np.asarray(stats_report["document_mean_nll"])
    deltas = {}
    for key, base in (("top20_kl", unpruned_kl), ("nll", unpruned_nll)):
        delta = np.asarray(summary[f"document_mean_{key}"]) - base
        lower, upper = paired_bootstrap_mean_ci(delta.tolist(), samples=10_000,
                                                seed=20261001)
        deltas[key] = {"mean": float(delta.mean()), "ci95_lower": lower,
                       "ci95_upper": upper}
    pruned_share = float(counts[mask].sum() / counts.sum())
    _atomic_json(output, {
        "schema": SCHEMA + ".frequency_baseline",
        "status": "complete",
        "semantics": (
            "exact: pruned router logits -inf, softmax, top-8, renormalize "
            "(equivalent to llama.cpp qwen35moe with experts removed)"),
        "mask": {
            "path": str(mask_path), "sha256": _sha256_file(mask_path),
            "pruned_per_layer": PRUNE_PER_LAYER,
            "rule": "fewest GSQ-E6 selections on v2; ties by router probability mass, then index",
            "unpruned_routed_slot_share_of_pruned_experts": pruned_share,
        },
        **summary,
        "unpruned_mean_top20_kl_to_bf16": float(unpruned_kl.mean()),
        "unpruned_mean_nll": float(unpruned_nll.mean()),
        "delta_vs_unpruned": deltas,
        "wall_seconds": time.perf_counter() - started,
    })
    print(f"baseline: complete, KL {summary['mean_top20_kl_to_bf16']:.5f} "
          f"(+{deltas['top20_kl']['mean']:.5f}), NLL {summary['mean_nll']:.5f} "
          f"(+{deltas['nll']['mean']:.5f}); pruned experts carried "
          f"{100 * pruned_share:.2f}% of routed slots", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("reference", "router-stats", "baseline"))
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
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--batch-documents", type=int, default=25)
    args = parser.parse_args()
    for key in ("model_dir", "manifest", "calibration", "gguf", "gguf_python",
                "ggml_library", "work", "reports"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main() -> int:
    args = _parse_args()
    {"reference": run_reference, "router-stats": run_router_stats,
     "baseline": run_baseline}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
