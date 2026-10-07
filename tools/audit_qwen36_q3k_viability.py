#!/usr/bin/env python3
"""Phase 0 of RCO_PLAN_NEW.md: is Q3_K a sensible upgrade for GSQ Q2_0 experts?

Subcommands, each resumable and run in this order:

``imatrix``
    One streamed BF16 forward over the pinned calibration corpus.  For every
    routed-expert tensor it accumulates the per-expert, per-column sum of the
    squared matmul inputs and the routed token counts (the convention of
    llama.cpp ``imatrix`` for ``MUL_MAT_ID``).  Gate and up share the routed
    hidden state as input; down sees the expert's intermediate activation.
``quantize``
    BF16-derived, imatrix-weighted Q3_K payloads for all 120 routed-expert
    tensors in a new native candidate store.  Each tensor's BF16 rows must
    reproduce the stored Q4_0 candidate checksum.  The relative RMSE and
    imatrix-weighted relative error are recorded against BF16 for authentic
    GSQ Q2_0 and imatrix-Q3_K; sampled tensors also get plain Q3_K and Q4_K.
``evaluate``
    Exact calibration NLL in the streaming evaluator for authentic GSQ,
    GSQ + Q3_K on all 40 ``ffn_down_exps``, GSQ + stored Q4_0 on the same 40,
    and GSQ + Q3_K on all 120 routed-expert tensors (the Q3_K ceiling).
``report``
    Gate evaluation and go/no-go status.

The held-out corpus is not touched.  No GGUF is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

import numpy as np

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from native_store import NativeCandidateStore, NativeCandidateStoreWriter  # noqa: E402
from quant.ggml_importance import quantize_rows_with_importance  # noqa: E402
from quant.ggml_native import GGMLNativeCodec, GGMLType  # noqa: E402
from release_corpus import canonical_json_bytes, sha256_bytes  # noqa: E402
from release_quality import paired_bootstrap_mean_ci  # noqa: E402


SCHEMA = "rco.qwen36.q3k_viability.v1"
LAYERS = 40
EXPERTS = 256
FAMILIES = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
SAMPLED_LAYERS = (0, 10, 20, 30, 39)
GSQ_Q2_0_TYPE = 42
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20261001
MEDIAN_REDUCTION_GATE = 0.30
SEARCH_BASELINE_TOLERANCE = 1e-6


def expert_names() -> list[str]:
    return [f"blk.{layer}.{family}.weight"
            for layer in range(LAYERS) for family in FAMILIES]


def sampled_names() -> list[str]:
    return [f"blk.{layer}.{family}.weight"
            for layer in SAMPLED_LAYERS for family in FAMILIES]


def _parse_name(name: str) -> tuple[int, str]:
    match = re.fullmatch(r"blk\.(\d+)\.(ffn_(?:gate|up|down)_exps)\.weight", name)
    if match is None:
        raise ValueError(f"not a routed-expert tensor: {name}")
    return int(match.group(1)), match.group(2)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
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


def expert_importance(
    sums: np.ndarray, counts: np.ndarray, *, floor_fraction: float = 1e-8,
) -> tuple[np.ndarray, int]:
    """Return per-expert mean squared inputs and the number of fallback experts.

    Experts that received no calibration tokens use the mean of the other
    experts' importances.  A tiny floor keeps every column strictly positive.
    """
    sums = np.asarray(sums, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    if sums.ndim != 2 or counts.shape != (sums.shape[0],):
        raise ValueError("sums must be [experts, columns] with one count per expert")
    seen = counts > 0
    if not seen.any():
        raise ValueError("no expert received a calibration token")
    means = np.zeros_like(sums)
    means[seen] = sums[seen] / counts[seen, None]
    means[~seen] = means[seen].mean(axis=0)
    floor = floor_fraction * float(means.max())
    return np.maximum(means, floor).astype(np.float32), int((~seen).sum())


def _calibration(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    import torch

    calibration = _load_json(args.calibration)
    manifest = calibration["canonical_manifest"]
    manifest_sha256 = sha256_bytes(canonical_json_bytes(manifest))
    if calibration.get("canonical_manifest_sha256") != manifest_sha256:
        raise RuntimeError("calibration canonical manifest hash differs")
    tokens_path = Path(manifest["tokens"]["path"]).resolve(strict=True)
    if _sha256_file(tokens_path) != manifest["tokens"]["sha256"]:
        raise RuntimeError("calibration token file hash differs")
    input_ids = torch.tensor(_load_json(tokens_path), dtype=torch.long)
    if input_ids.shape != (
        manifest["document_count"], manifest["token_count_per_document"],
    ):
        raise RuntimeError("calibration token matrix shape differs")
    identity = {
        "calibration_manifest_sha256": manifest_sha256,
        "calibration_token_sha256": manifest["tokens"]["sha256"],
        "document_count": int(input_ids.shape[0]),
        "sequence_length": int(input_ids.shape[1]),
        "documents": [document["stratum"] for document in manifest["documents"]],
    }
    return input_ids, identity


def _empty_model(model_dir: Path) -> Any:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(
            config, attn_implementation="eager")
    model.eval()
    return model


# --------------------------------------------------------------------------
# imatrix
# --------------------------------------------------------------------------


def run_imatrix(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F

    from checkpoint_stream import SafeTensorPrefixLoader
    from search.streaming import StreamingHardCausalEvaluator

    output = args.reports / "qwen36_q3k_viability_imatrix.json"
    matrix_path = args.work / "imatrix.npz"
    if output.exists() and _load_json(output).get("status") == "complete":
        if _sha256_file(matrix_path) != _load_json(output)["imatrix"]["sha256"]:
            raise RuntimeError("existing imatrix file differs from its report")
        print("imatrix: already complete", flush=True)
        return
    started = time.perf_counter()
    device = torch.device(args.device)
    input_ids, calibration = _calibration(args)
    model = _empty_model(args.model_dir)
    loader = SafeTensorPrefixLoader(args.model_dir)
    evaluator = StreamingHardCausalEvaluator(
        model, loader, SimpleNamespace(), [], [0, 1], device=device,
        vocab_chunk_size=args.vocab_chunk_size)

    gate_up = {}
    down = {}
    counts = {}
    handles = []

    def make_hook(layer: int):
        def hook(module, hook_args, hook_kwargs):
            values = list(hook_args) + [
                hook_kwargs[key] for key in
                ("hidden_states", "top_k_index", "top_k_weights")
                if key in hook_kwargs]
            hidden, top_k_index = values[0], values[1]
            if layer not in gate_up:
                gate_up[layer] = torch.zeros(
                    EXPERTS, hidden.shape[-1], dtype=torch.float64,
                    device=hidden.device)
                down[layer] = torch.zeros(
                    EXPERTS, module.down_proj.shape[-1], dtype=torch.float64,
                    device=hidden.device)
                counts[layer] = torch.zeros(
                    EXPERTS, dtype=torch.int64, device=hidden.device)
            squared = hidden.double().square()
            for slot in range(top_k_index.shape[1]):
                gate_up[layer].index_add_(0, top_k_index[:, slot], squared)
            counts[layer] += torch.bincount(
                top_k_index.reshape(-1), minlength=EXPERTS)
            for expert in torch.unique(top_k_index).tolist():
                tokens = (top_k_index == expert).any(dim=-1).nonzero().squeeze(-1)
                gate, up = F.linear(
                    hidden[tokens], module.gate_up_proj[expert]).chunk(2, dim=-1)
                intermediate = module.act_fn(gate) * up
                down[layer][expert] += intermediate.double().square().sum(dim=0)
        return hook

    for module_name, module in model.named_modules():
        if type(module).__name__ != "Qwen3_5MoeExperts":
            continue
        layer = int(re.search(r"layers\.(\d+)\.", module_name).group(1))
        handles.append(module.register_forward_pre_hook(
            make_hook(layer), with_kwargs=True))
    if len(handles) != LAYERS:
        raise RuntimeError(f"found {len(handles)} expert modules, expected {LAYERS}")
    try:
        evaluation = evaluator.evaluate(
            input_ids, torch.zeros(0, dtype=torch.long))
    finally:
        for handle in handles:
            handle.remove()
    if sorted(gate_up) != list(range(LAYERS)):
        raise RuntimeError("not every expert layer was observed")
    tokens = input_ids.numel()
    arrays = {}
    fallback = {}
    for layer in range(LAYERS):
        layer_counts = counts[layer].cpu().numpy()
        if int(layer_counts.sum()) != tokens * 8:
            raise RuntimeError(f"layer {layer} routed-token count differs")
        arrays[f"blk{layer}_gate_up_sum"] = gate_up[layer].cpu().numpy()
        arrays[f"blk{layer}_down_sum"] = down[layer].cpu().numpy()
        arrays[f"blk{layer}_count"] = layer_counts
        fallback[str(layer)] = int((layer_counts == 0).sum())
    args.work.mkdir(parents=True, exist_ok=True)
    temporary = matrix_path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, matrix_path)
    all_counts = np.stack([arrays[f"blk{layer}_count"] for layer in range(LAYERS)])
    _atomic_json(output, {
        "schema": SCHEMA + ".imatrix",
        "status": "complete",
        "source": "streamed BF16 forward over the pinned calibration corpus",
        "model_dir": str(args.model_dir),
        "calibration": calibration,
        "imatrix": {
            "path": str(matrix_path),
            "sha256": _sha256_file(matrix_path),
            "convention": (
                "per expert and input column: sum of squared matmul inputs over "
                "routed tokens; gate/up input is the routed hidden state, down "
                "input is act(gate)*up; routing weights are not applied"),
            "routed_tokens_per_layer": tokens * 8,
            "experts_without_tokens_per_layer": fallback,
            "min_tokens_per_expert": int(all_counts.min()),
            "median_tokens_per_expert": float(np.median(all_counts)),
        },
        "bf16_calibration": {
            "mean_nll": evaluation.loss,
            "predicted_token_count": evaluation.token_count,
            "document_mean_nll": list(evaluation.document_mean_nll),
        },
        "memory": asdict(evaluation.memory),
        "wall_seconds": time.perf_counter() - started,
    })
    print(json.dumps({"imatrix": "complete", "bf16_mean_nll": evaluation.loss,
                      "experts_without_tokens": sum(fallback.values())}), flush=True)


# --------------------------------------------------------------------------
# quantize
# --------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _worker_init(settings: dict[str, Any]) -> None:
    from qwen35_native import SafetensorGGUFRowSource

    _WORKER.update(settings)
    _WORKER["codec"] = GGMLNativeCodec(settings["ggml_library"])
    _WORKER["source"] = SafetensorGGUFRowSource(settings["model_dir"])
    with np.load(settings["imatrix_path"]) as matrix:
        _WORKER["imatrix"] = {key: matrix[key] for key in matrix.files}


class _ErrorAccumulator:
    def __init__(self) -> None:
        self.squared = 0.0
        self.weighted = 0.0

    def add(self, decoded: np.ndarray, rows: np.ndarray, importance: np.ndarray) -> None:
        difference = np.square(decoded - rows, dtype=np.float64)
        self.squared += float(difference.sum())
        self.weighted += float((difference * importance).sum())


def _quantize_tensor(name: str) -> dict[str, Any]:
    codec: GGMLNativeCodec = _WORKER["codec"]
    store_root = Path(_WORKER["store_root"])
    sidecar = store_root / "tensors" / quote(name, safe="") / "Q3_K.metrics.json"
    if sidecar.exists():
        return _load_json(sidecar)
    started = time.perf_counter()
    entry = _WORKER["manifest_entries"][name]
    layer, family = _parse_name(name)
    experts, rows_per_expert, width = (
        int(value) for value in entry["candidate_source_shape"])
    imatrix = _WORKER["imatrix"]
    sums = imatrix[f"blk{layer}_down_sum" if family == "ffn_down_exps"
                   else f"blk{layer}_gate_up_sum"]
    importance, fallback_experts = expert_importance(
        sums, imatrix[f"blk{layer}_count"])
    if importance.shape != (experts, width):
        raise RuntimeError(f"importance shape differs for {name}")
    sampled = name in _WORKER["sampled"]
    offset, gsq_bytes = _WORKER["gsq_offsets"][name]
    gsq_row_size = codec.geometry(GGMLType.Q2_0, width)["row_size"]
    if gsq_bytes != experts * rows_per_expert * gsq_row_size:
        raise RuntimeError(f"GSQ payload size differs for {name}")
    gsq = np.memmap(_WORKER["gguf_path"], dtype=np.uint8, mode="r",
                    offset=offset, shape=(gsq_bytes,))
    expert_bytes = rows_per_expert * gsq_row_size
    variants = ["gsq_q2_0", "q3_k_imatrix"] + (
        ["q3_k_plain", "q4_k_plain"] if sampled else [])
    errors = {variant: _ErrorAccumulator() for variant in variants}
    energy = 0.0
    weighted_energy = 0.0
    q4_0_digest = hashlib.sha256()

    def chunks():
        nonlocal energy, weighted_energy
        for expert, rows in enumerate(_WORKER["source"].iter_rows(
            entry, rows_per_chunk=rows_per_expert,
        )):
            rows = np.ascontiguousarray(rows, dtype=np.float32)
            if rows.shape != (rows_per_expert, width):
                raise RuntimeError(f"source chunk is not one expert: {name}")
            weights = importance[expert]
            squared = np.square(rows, dtype=np.float64)
            energy += float(squared.sum())
            weighted_energy += float((squared * weights).sum())
            q4_0_digest.update(codec.quantize_rows(rows, GGMLType.Q4_0))
            decoded = np.empty_like(rows)
            codec.dequantize_rows_into(
                gsq[expert * expert_bytes:(expert + 1) * expert_bytes],
                GGMLType.Q2_0, decoded)
            errors["gsq_q2_0"].add(decoded, rows, weights)
            payload = quantize_rows_with_importance(
                codec, rows, GGMLType.Q3_K, weights)
            codec.dequantize_rows_into(payload, GGMLType.Q3_K, decoded)
            errors["q3_k_imatrix"].add(decoded, rows, weights)
            if sampled:
                for variant, ggml_type in (
                    ("q3_k_plain", GGMLType.Q3_K), ("q4_k_plain", GGMLType.Q4_K),
                ):
                    codec.dequantize_rows_into(
                        codec.quantize_rows(rows, ggml_type), ggml_type, decoded)
                    errors[variant].add(decoded, rows, weights)
            yield payload
        if expert != experts - 1:
            raise RuntimeError(f"source expert count differs for {name}")

    writer = NativeCandidateStoreWriter(
        store_root, codec, source=_WORKER["store_source"])
    store_entry = writer.write_packed_chunks(
        name, GGMLType.Q3_K, entry["destination_gguf_shape"], chunks(),
        provenance={
            "kind": "bf16_derived_imatrix_native_candidate",
            "source_tensor": entry["source_name"],
            "source_shard": entry["source_shard"],
            "source_view": entry.get("source_view"),
            "imatrix_sha256": _WORKER["imatrix_sha256"],
        })
    stored_q4_0_sha256, stored_q4_0_bytes = _WORKER["stored_q4_0"][name]
    if q4_0_digest.hexdigest() != stored_q4_0_sha256:
        raise RuntimeError(f"BF16 rows do not reproduce the stored Q4_0: {name}")
    metrics = {
        variant: {
            "relative_rmse": math.sqrt(accumulator.squared / energy),
            "weighted_relative_error": math.sqrt(
                accumulator.weighted / weighted_energy),
        }
        for variant, accumulator in errors.items()
    }
    result = {
        "name": name,
        "layer": layer,
        "family": family,
        "sampled": sampled,
        "store_entry": store_entry,
        "stored_q4_0_sha256_reproduced": q4_0_digest.hexdigest(),
        "gsq_q2_0_payload_bytes": gsq_bytes,
        "q4_0_incremental_bytes": stored_q4_0_bytes - gsq_bytes,
        "q3_k_incremental_bytes": int(store_entry["payload_bytes"]) - gsq_bytes,
        "imatrix_fallback_experts": fallback_experts,
        "metrics": metrics,
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(sidecar, result)
    return result


def run_quantize(args: argparse.Namespace) -> None:
    from native_gguf import import_pinned_gguf

    output = args.reports / "qwen36_q3k_viability_tensors.json"
    if output.exists() and _load_json(output).get("status") == "complete":
        print("quantize: already complete", flush=True)
        return
    started = time.perf_counter()
    imatrix_report = _load_json(args.reports / "qwen36_q3k_viability_imatrix.json")
    if imatrix_report.get("status") != "complete":
        raise RuntimeError("imatrix is not complete")
    imatrix_path = Path(imatrix_report["imatrix"]["path"])
    imatrix_sha256 = _sha256_file(imatrix_path)
    if imatrix_sha256 != imatrix_report["imatrix"]["sha256"]:
        raise RuntimeError("imatrix file differs from its report")
    manifest = _load_json(args.manifest)
    names = expert_names()
    manifest_entries = {
        entry["destination_name"]: entry for entry in manifest["entries"]
        if entry["destination_name"] in names}
    if len(manifest_entries) != len(names):
        raise RuntimeError("manifest does not map every routed-expert tensor")
    codec = GGMLNativeCodec(args.ggml_library)
    native = NativeCandidateStore(args.native_store, codec)
    stored_q4_0 = {}
    for name in names:
        metadata = native.metadata(name, GGMLType.Q4_0)
        stored_q4_0[name] = (metadata["sha256"], int(metadata["payload_bytes"]))
    gguf = import_pinned_gguf(args.gguf_python)
    reader = gguf.GGUFReader(args.gguf)
    offsets = {}
    for tensor in reader.tensors:
        if tensor.name in manifest_entries:
            if int(tensor.tensor_type) != GSQ_Q2_0_TYPE:
                raise RuntimeError(f"GSQ tensor is not Q2_0: {tensor.name}")
            offsets[tensor.name] = (int(tensor.data_offset), int(tensor.n_bytes))
    del reader
    if len(offsets) != len(names):
        raise RuntimeError("GSQ GGUF does not contain every routed-expert tensor")
    gguf_sha256 = _sha256_file(args.gguf)
    store_root = args.work / "q3k_store"
    store_source = {
        "kind": "bf16_derived_imatrix_q3_k_routed_experts",
        "model_dir": str(args.model_dir),
        "manifest_sha256": _sha256_file(args.manifest),
        "imatrix_sha256": imatrix_sha256,
    }
    settings = {
        "ggml_library": str(args.ggml_library),
        "model_dir": str(args.model_dir),
        "imatrix_path": str(imatrix_path),
        "imatrix_sha256": imatrix_sha256,
        "manifest_entries": manifest_entries,
        "gsq_offsets": offsets,
        "gguf_path": str(args.gguf),
        "store_root": str(store_root),
        "store_source": store_source,
        "stored_q4_0": stored_q4_0,
        "sampled": set(sampled_names()),
    }
    import multiprocessing

    results = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init, initargs=(settings,),
    ) as pool:
        for result in pool.map(_quantize_tensor, names):
            results[result["name"]] = result
            print(json.dumps({
                "tensor": result["name"],
                "gsq_weighted": round(
                    result["metrics"]["gsq_q2_0"]["weighted_relative_error"], 4),
                "q3k_weighted": round(
                    result["metrics"]["q3_k_imatrix"]["weighted_relative_error"], 4),
                "seconds": round(result["wall_seconds"], 1),
            }), flush=True)
    writer = NativeCandidateStoreWriter(store_root, codec, source=store_source)
    writer._entries = {
        name: {"Q3_K": results[name]["store_entry"]} for name in names}
    index_path = writer.finalize()
    _atomic_json(output, {
        "schema": SCHEMA + ".tensors",
        "status": "complete",
        "gsq_gguf_sha256": gguf_sha256,
        "imatrix_sha256": imatrix_sha256,
        "store": {
            "root": str(store_root),
            "index_sha256": _sha256_file(index_path),
        },
        "sampled_tensors": sampled_names(),
        "tensors": results,
        "wall_seconds": time.perf_counter() - started,
    })


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------


def _expert_weight_store_class():
    from native_runtime import NativeManifestWeightStore

    class ExpertUpgradeWeightStore(NativeManifestWeightStore):
        """Choice 0 keeps the loaded GSQ tensor; choice 1 installs one type."""

        def __init__(self, store, manifest, model_dir, upgrade_type, *,
                     rows_per_chunk):
            super().__init__(store, manifest, model_dir,
                             rows_per_chunk=rows_per_chunk)
            self.upgrade_type = GGMLType(upgrade_type)

        def _type(self, bitwidth: int) -> GGMLType:
            if int(bitwidth) != 1:
                raise ValueError(f"only choice 1 installs a payload: {bitwidth}")
            return self.upgrade_type

        @staticmethod
        def is_retain_choice(name: str, choice: int) -> bool:
            if int(choice) not in (0, 1):
                raise ValueError("expert upgrade choice must be zero or one")
            return int(choice) == 0

    return ExpertUpgradeWeightStore


def evaluation_arms() -> dict[str, tuple[str | None, list[str]]]:
    down = [name for name in expert_names() if "ffn_down_exps" in name]
    return {
        "gsq": (None, []),
        "q3k_down40": ("Q3_K", down),
        "q4_0_down40": ("Q4_0", down),
        "q3k_all120": ("Q3_K", expert_names()),
    }


def run_evaluate(args: argparse.Namespace) -> None:
    import torch

    from gguf_checkpoint_stream import GGUFManifestPrefixLoader
    from search.streaming import StreamingHardCausalEvaluator

    output = args.reports / "qwen36_q3k_viability_evaluation.json"
    tensors_report = _load_json(args.reports / "qwen36_q3k_viability_tensors.json")
    if tensors_report.get("status") != "complete":
        raise RuntimeError("Q3_K store is not complete")
    q3k_root = Path(tensors_report["store"]["root"])
    if _sha256_file(q3k_root / "native-candidate-index.json") != (
        tensors_report["store"]["index_sha256"]
    ):
        raise RuntimeError("Q3_K store index differs from its report")
    device = torch.device(args.device)
    input_ids, calibration = _calibration(args)
    identity = {
        "gsq_gguf_sha256": tensors_report["gsq_gguf_sha256"],
        "q3k_store_index_sha256": tensors_report["store"]["index_sha256"],
        "native_store_index_sha256": _sha256_file(
            args.native_store / "native-candidate-index.json"),
        "calibration": calibration,
        "device": str(device),
        "rows_per_chunk": args.rows_per_chunk,
        "vocab_chunk_size": args.vocab_chunk_size,
    }
    report = _load_json(output) if output.exists() else {
        "schema": SCHEMA + ".evaluation", "status": "in_progress", "arms": {}}
    if report.get("identity") not in (None, identity):
        raise RuntimeError("existing evaluation report describes another problem")
    report["identity"] = identity
    manifest = _load_json(args.manifest)
    codec = GGMLNativeCodec(args.ggml_library)
    store_class = _expert_weight_store_class()
    stores = {
        "Q3_K": store_class(
            NativeCandidateStore(q3k_root, codec), manifest, args.model_dir,
            GGMLType.Q3_K, rows_per_chunk=args.rows_per_chunk),
        "Q4_0": store_class(
            NativeCandidateStore(args.native_store, codec), manifest,
            args.model_dir, GGMLType.Q4_0, rows_per_chunk=args.rows_per_chunk),
    }
    model = _empty_model(args.model_dir)
    loader = GGUFManifestPrefixLoader(
        args.gguf, manifest, args.model_dir, gguf_python=args.gguf_python,
        ggml_library=args.ggml_library, rows_per_chunk=args.rows_per_chunk)
    for label, (upgrade_type, names) in evaluation_arms().items():
        if label in report["arms"]:
            continue
        store = stores[upgrade_type or "Q3_K"]
        groups = [SimpleNamespace(layer_names=(name,)) for name in names]
        evaluator = StreamingHardCausalEvaluator(
            model, loader, store, groups, [0, 1], device=device,
            vocab_chunk_size=args.vocab_chunk_size,
            checkpoint_dtype=torch.bfloat16)
        started = time.perf_counter()
        evaluation = evaluator.evaluate(
            input_ids, torch.ones(len(names), dtype=torch.long))
        report["arms"][label] = {
            "upgrade_type": upgrade_type,
            "upgraded_tensors": names,
            "mean_nll": evaluation.loss,
            "predicted_token_count": evaluation.token_count,
            "document_mean_nll": list(evaluation.document_mean_nll),
            "document_token_counts": list(evaluation.document_token_counts),
            "memory": asdict(evaluation.memory),
            "wall_seconds": time.perf_counter() - started,
        }
        _atomic_json(output, report)
        print(json.dumps({"arm": label, "mean_nll": evaluation.loss}), flush=True)
    report["status"] = "complete"
    _atomic_json(output, report)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def build_report(
    imatrix_report: dict[str, Any],
    tensors_report: dict[str, Any],
    evaluation_report: dict[str, Any],
    search_baseline_mean_nll: float,
) -> dict[str, Any]:
    tensors = tensors_report["tensors"]
    sampled = tensors_report["sampled_tensors"]

    def weighted(name: str, variant: str) -> float:
        return tensors[name]["metrics"][variant]["weighted_relative_error"]

    reductions = {
        name: 1.0 - weighted(name, "q3_k_imatrix") / weighted(name, "gsq_q2_0")
        for name in tensors}
    sampled_reductions = [reductions[name] for name in sampled]
    imatrix_gain = [
        1.0 - weighted(name, "q3_k_imatrix") / weighted(name, "q3_k_plain")
        for name in sampled]
    tensor_level = {
        "sampled": {
            name: {
                variant: tensors[name]["metrics"][variant]
                for variant in ("gsq_q2_0", "q3_k_plain", "q3_k_imatrix",
                                "q4_k_plain")}
            for name in sampled},
        "sampled_weighted_error_reduction_vs_gsq": dict(
            zip(sampled, sampled_reductions)),
        "sampled_median_weighted_error_reduction_vs_gsq": float(
            np.median(sampled_reductions)),
        "sampled_all_q3k_imatrix_better_than_gsq": all(
            value > 0 for value in sampled_reductions),
        "sampled_median_imatrix_weighted_error_reduction_vs_plain_q3k": float(
            np.median(imatrix_gain)),
        "sampled_imatrix_better_than_plain_count": sum(
            value > 0 for value in imatrix_gain),
        "all_tensors_median_weighted_error_reduction_vs_gsq": float(
            np.median(list(reductions.values()))),
        "all_tensors_min_weighted_error_reduction_vs_gsq": float(
            min(reductions.values())),
        "family_median_weighted_error_reduction_vs_gsq": {
            family: float(np.median([
                value for name, value in reductions.items()
                if tensors[name]["family"] == family]))
            for family in FAMILIES},
    }
    arms = evaluation_report["arms"]
    baseline = arms["gsq"]["document_mean_nll"]
    costs = {
        "Q3_K": tensors[expert_names()[0]]["q3_k_incremental_bytes"],
        "Q4_0": tensors[expert_names()[0]]["q4_0_incremental_bytes"],
    }
    functional = {}
    for label, arm in arms.items():
        if label == "gsq":
            continue
        deltas = [
            value - base
            for value, base in zip(arm["document_mean_nll"], baseline)]
        lower, upper = paired_bootstrap_mean_ci(
            deltas, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)
        added = costs[arm["upgrade_type"]] * len(arm["upgraded_tensors"])
        mean_delta = math.fsum(deltas) / len(deltas)
        functional[label] = {
            "mean_nll": arm["mean_nll"],
            "token_weighted_delta_vs_gsq": arm["mean_nll"] - arms["gsq"]["mean_nll"],
            "paired_document_mean_delta_vs_gsq": mean_delta,
            "ci95_lower": lower,
            "ci95_upper": upper,
            "documents_improved": sum(value < 0 for value in deltas),
            "added_bytes": added,
            "nll_reduction_per_gb": -mean_delta / (added / 1e9),
        }
    ceiling = functional["q3k_all120"]["paired_document_mean_delta_vs_gsq"]
    functional["q3k_down40"]["fraction_of_q3k_ceiling"] = (
        functional["q3k_down40"]["paired_document_mean_delta_vs_gsq"] / ceiling
        if ceiling else None)
    gates = {
        "tensor_level_all_sampled_better_than_gsq": (
            tensor_level["sampled_all_q3k_imatrix_better_than_gsq"]),
        "tensor_level_median_reduction_at_least_30_percent": (
            tensor_level["sampled_median_weighted_error_reduction_vs_gsq"]
            >= MEDIAN_REDUCTION_GATE),
        "imatrix_better_than_plain_q3k": (
            tensor_level[
                "sampled_median_imatrix_weighted_error_reduction_vs_plain_q3k"]
            > 0),
        "q3k_down40_ci_upper_below_zero": functional["q3k_down40"]["ci95_upper"] < 0,
    }
    q3k_per_gb_not_worse = (
        functional["q3k_down40"]["nll_reduction_per_gb"]
        >= functional["q4_0_down40"]["nll_reduction_per_gb"])
    required = (
        gates["tensor_level_all_sampled_better_than_gsq"]
        and gates["tensor_level_median_reduction_at_least_30_percent"]
        and gates["q3k_down40_ci_upper_below_zero"])
    if not required:
        status = "no_go"
    elif not q3k_per_gb_not_worse:
        status = "stop_for_decision_q3k_per_gb_below_q4_0"
    else:
        status = "go"
    return {
        "schema": SCHEMA,
        "status": status,
        "scope": (
            "Phase 0 of RCO_PLAN_NEW.md on the unpruned GSQ model and the "
            "pinned 50-document calibration corpus; no held-out data, no GGUF"),
        "inputs": {
            "imatrix_sha256": imatrix_report["imatrix"]["sha256"],
            "q3k_store_index_sha256": tensors_report["store"]["index_sha256"],
            "gsq_gguf_sha256": tensors_report["gsq_gguf_sha256"],
            "evaluation_identity": evaluation_report["identity"],
        },
        "imatrix": {
            key: imatrix_report["imatrix"][key]
            for key in ("min_tokens_per_expert", "median_tokens_per_expert",
                        "experts_without_tokens_per_layer")},
        "bf16_calibration_mean_nll": imatrix_report["bf16_calibration"]["mean_nll"],
        "gsq_calibration_mean_nll": arms["gsq"]["mean_nll"],
        "gsq_reproduces_search_baseline": abs(
            arms["gsq"]["mean_nll"] - search_baseline_mean_nll
        ) <= SEARCH_BASELINE_TOLERANCE,
        "per_tensor_upgrade_bytes": costs,
        "tensor_level": tensor_level,
        "functional": functional,
        "gates": gates,
        "q3k_nll_reduction_per_gb_not_worse_than_q4_0": q3k_per_gb_not_worse,
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED},
    }


def run_report(args: argparse.Namespace) -> None:
    search = _load_json(args.reports / "qwen36_gsq_rco_search.json")
    report = build_report(
        _load_json(args.reports / "qwen36_q3k_viability_imatrix.json"),
        _load_json(args.reports / "qwen36_q3k_viability_tensors.json"),
        _load_json(args.reports / "qwen36_q3k_viability_evaluation.json"),
        float(search["full_corpus_baseline"]["mean_nll"]),
    )
    _atomic_json(args.reports / "qwen36_q3k_viability.json", report)
    print(json.dumps({
        "status": report["status"],
        "gates": report["gates"],
        "functional": {
            label: {key: value[key] for key in (
                "paired_document_mean_delta_vs_gsq", "ci95_lower", "ci95_upper",
                "nll_reduction_per_gb")}
            for label, value in report["functional"].items()},
        "median_weighted_error_reduction_vs_gsq": report["tensor_level"][
            "sampled_median_weighted_error_reduction_vs_gsq"],
    }, indent=2), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("imatrix", "quantize", "evaluate", "report"))
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/qwen36_35b_base")
    parser.add_argument("--manifest", type=Path,
                        default=RCO / "reports/qwen36_35b_base_gguf_manifest.json")
    parser.add_argument("--calibration", type=Path,
                        default=RCO / "reports/qwen36_35b_calibration_corpus_manifest.json")
    parser.add_argument("--gguf", type=Path,
                        default=ROOT / "results/Qwen3.6-35B-A3B-GSQ-hybrid.gguf")
    parser.add_argument("--gguf-python", type=Path,
                        default=ROOT / "repos/llama.cpp/gguf-py")
    parser.add_argument("--ggml-library", type=Path,
                        default=ROOT / "experiment/build-cpu/bin/libggml-base.so.0.24.0")
    parser.add_argument("--native-store", type=Path,
                        default=ROOT / "data/qwen36_35b_native")
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_q3k_phase0")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for key in ("model_dir", "manifest", "calibration", "gguf", "gguf_python",
                "ggml_library", "native_store", "work", "reports"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main() -> int:
    args = _parse_args()
    {"imatrix": run_imatrix, "quantize": run_quantize,
     "evaluate": run_evaluate, "report": run_report}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
