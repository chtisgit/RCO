#!/usr/bin/env python3
"""Run one streamed relaxed backward through all 512 production decisions."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import transformers
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoint_stream import SafeTensorPrefixLoader
from dense_oracle import tensor_sha256
from model_adapter import get_model_adapter
from native_runtime import NativeManifestRelaxedLinearSource
from native_store import NativeCandidateStore
from quant.ggml_native import GGMLNativeCodec, GGMLType
from qwen35_native import Qwen35LinearAttentionGeometry
from search.relaxed import (
    CheckpointedRelaxedBlockStats,
    CheckpointedStreamingRelaxedBlock,
    RelaxedExpertsBinding,
    RelaxedLinearBinding,
    RelaxedRouterBinding,
    StreamingRelaxedLinearStats,
    streaming_relaxed_causal_cross_entropy,
    streaming_relaxed_embedding,
)


CALIBRATION_TEXT = (
    "A bounded native GGML search streams every Qwen decoder block while "
    "evaluating an exact serialized byte budget."
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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


def _instantiate_model(model_dir: Path):
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(
            config, attn_implementation="eager")
    model.requires_grad_(False)
    model.eval()
    return model


def _initial_logits(group_count: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(
        -0.8, 0.8, steps=group_count * 2,
        dtype=torch.float32, device=device,
    ).reshape(group_count, 2).requires_grad_(True)


def _block_index(destination: str) -> int | None:
    if not destination.startswith("blk."):
        return None
    fields = destination.split(".", 2)
    if len(fields) != 3:
        raise ValueError(f"malformed block destination {destination!r}")
    return int(fields[1])


def _expert_projection(destination: str) -> str | None:
    for projection in ("gate", "up", "down"):
        if destination.endswith(f".ffn_{projection}_exps.weight"):
            return projection
    return None


def _is_router(destination: str) -> bool:
    return destination.endswith(".ffn_gate_inp.weight")


def _source_dtype(entry: dict[str, Any]) -> torch.dtype:
    dtypes = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
    }
    try:
        return dtypes[entry["source_dtype"]]
    except KeyError as error:
        raise ValueError(
            f"unsupported source dtype {entry.get('source_dtype')!r}") from error


@contextmanager
def _patch_embedding(
    module: torch.nn.Module,
    input_logits: torch.Tensor,
    source: NativeManifestRelaxedLinearSource,
    stats: StreamingRelaxedLinearStats,
    output_dtype: torch.dtype,
):
    had_instance_forward = "forward" in module.__dict__
    instance_forward = module.__dict__.get("forward")
    def relaxed_forward(input_ids: torch.Tensor) -> torch.Tensor:
        return streaming_relaxed_embedding(
            input_ids,
            input_logits,
            source,
            dtype=output_dtype,
            stats=stats,
        )

    module.forward = relaxed_forward
    try:
        yield
    finally:
        if had_instance_forward:
            module.forward = instance_forward
        else:
            del module.forward


def _aggregate_native_stats(sources: list[NativeManifestRelaxedLinearSource]):
    if not sources:
        raise RuntimeError("no native sources were exercised")
    return {
        "source_instance_count": len(sources),
        "reference_passes": sum(
            source.stats.reference_passes for source in sources),
        "alternative_passes": sum(
            source.stats.alternative_passes for source in sources),
        "reference_payload_bytes_read": sum(
            source.stats.reference_payload_bytes_read for source in sources),
        "alternative_payload_bytes_read": sum(
            source.stats.alternative_payload_bytes_read for source in sources),
        "max_resident_decoded_bytes": max(
            source.stats.max_resident_decoded_bytes for source in sources),
    }


def _run_once(
    *,
    model_dir: Path,
    loader: SafeTensorPrefixLoader,
    store: NativeCandidateStore,
    entries: list[dict[str, Any]],
    input_ids: torch.Tensor,
    device: torch.device,
    rows_per_chunk: int,
) -> dict[str, Any]:
    model = _instantiate_model(model_dir)
    adapter = get_model_adapter(model)
    geometry = Qwen35LinearAttentionGeometry.from_model_dir(model_dir)
    logits = _initial_logits(len(entries), device)
    group_by_destination = {
        entry["destination_name"]: index
        for index, entry in enumerate(entries)
    }
    entry_by_destination = {
        entry["destination_name"]: entry for entry in entries
    }
    native_sources: list[NativeManifestRelaxedLinearSource] = []
    operation_stats: list[StreamingRelaxedLinearStats] = []

    def make_source(
        entry: dict[str, Any], *, expert_index: int | None = None,
    ) -> NativeManifestRelaxedLinearSource:
        source = NativeManifestRelaxedLinearSource(
            store,
            entry,
            reference_type=GGMLType.Q4_0,
            alternative_types=(GGMLType.Q2_0,),
            geometry=geometry,
            rows_per_chunk=rows_per_chunk,
            expert_index=expert_index,
        )
        native_sources.append(source)
        return source

    embedding_entry = entry_by_destination["token_embd.weight"]
    output_entry = entry_by_destination["output.weight"]
    embedding_source = make_source(embedding_entry)
    output_source = make_source(output_entry)
    embedding_stats = StreamingRelaxedLinearStats()
    output_stats = StreamingRelaxedLinearStats()
    operation_stats.extend((embedding_stats, output_stats))

    originals = list(adapter.layers)
    block_stats: list[CheckpointedRelaxedBlockStats] = []
    expert_source_records: list[tuple[int, str, int]] = []
    for block_index, block in enumerate(originals):
        block_entries = [
            entry for entry in entries
            if _block_index(entry["destination_name"]) == block_index
        ]
        if not block_entries:
            raise RuntimeError(f"block {block_index} has no relaxed decisions")
        linear_bindings = []
        router_bindings = []
        expert_entries: dict[str, dict[str, Any]] = {}
        for entry in block_entries:
            destination = entry["destination_name"]
            projection = _expert_projection(destination)
            if projection is not None:
                expert_entries[projection] = entry
                continue
            stats = StreamingRelaxedLinearStats()
            operation_stats.append(stats)
            source = make_source(entry)
            if _is_router(destination):
                router_bindings.append(RelaxedRouterBinding(
                    module_path=entry["source_name"].removesuffix(".weight"),
                    group_index=group_by_destination[destination],
                    source=source,
                    stats=stats,
                ))
            else:
                linear_bindings.append(RelaxedLinearBinding(
                    module_path=entry["source_name"].removesuffix(".weight"),
                    group_index=group_by_destination[destination],
                    source=source,
                    stats=stats,
                ))
        if set(expert_entries) != {"gate", "up", "down"}:
            raise RuntimeError(
                f"block {block_index} lacks a complete routed-expert triplet")
        expert_stats = {
            projection: StreamingRelaxedLinearStats()
            for projection in expert_entries
        }
        operation_stats.extend(expert_stats.values())

        def expert_source_factory(
            projection: str,
            expert_index: int,
            *,
            _block_index: int = block_index,
            _entries: dict[str, dict[str, Any]] = expert_entries,
        ):
            expert_source_records.append(
                (_block_index, projection, expert_index))
            return make_source(
                _entries[projection], expert_index=expert_index)

        expert_path = expert_entries["gate"]["source_name"].rpartition(".")[0]
        expert_binding = RelaxedExpertsBinding(
            module_path=expert_path,
            gate_group_index=group_by_destination[
                expert_entries["gate"]["destination_name"]],
            up_group_index=group_by_destination[
                expert_entries["up"]["destination_name"]],
            down_group_index=group_by_destination[
                expert_entries["down"]["destination_name"]],
            source_factory=expert_source_factory,
            gate_stats=expert_stats["gate"],
            up_stats=expert_stats["up"],
            down_stats=expert_stats["down"],
        )
        stats = CheckpointedRelaxedBlockStats()
        block_stats.append(stats)
        prefix = f"{adapter.layers_path}.{block_index}"
        adapter.layers[block_index] = CheckpointedStreamingRelaxedBlock(
            block,
            model=model,
            path=prefix,
            checkpoint_loader=loader,
            logits=logits,
            bindings=linear_bindings,
            expert_bindings=[expert_binding],
            router_bindings=router_bindings,
            device=device,
            stats=stats,
        )

    embedding_module = adapter.embeddings[0]
    if embedding_module.weight.device.type != "meta":
        raise RuntimeError("embedding unexpectedly materialized before forward")
    loader.move_runtime_buffers(model, device)
    norm_path = next(
        path for path in adapter.final_module_paths if path != "lm_head")
    final_norm_bytes = loader.load_prefix(model, norm_path, device=device)
    started = time.perf_counter()
    try:
        with _patch_embedding(
            embedding_module,
            logits[group_by_destination["token_embd.weight"]],
            embedding_source,
            embedding_stats,
            _source_dtype(embedding_entry),
        ):
            device_ids = input_ids.to(device=device)
            attention_mask = torch.ones_like(device_ids)
            output = adapter.text_model(
                input_ids=device_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = output.last_hidden_state
            loss, token_count = streaming_relaxed_causal_cross_entropy(
                hidden_states,
                device_ids,
                logits[group_by_destination["output.weight"]],
                output_source,
                stats=output_stats,
            )
            loss.backward()
    finally:
        loader.release_prefix(model, norm_path)
    elapsed = time.perf_counter() - started

    if any(stats.load_passes != 2 or stats.release_passes != 2
           for stats in block_stats):
        raise RuntimeError("one or more blocks did not reload exactly in backward")
    if any(block.module.input_layernorm.weight.device.type != "meta"
           for block in adapter.layers):
        raise RuntimeError("one or more decoder blocks remained materialized")
    if logits.grad is None or not torch.isfinite(logits.grad).all():
        raise RuntimeError("full-model assignment gradient is invalid")
    gradient_norms = logits.grad.float().norm(dim=1)
    nonzero_count = int(torch.count_nonzero(gradient_norms).item())
    if nonzero_count != len(entries):
        zero_destinations = [
            entries[index]["destination_name"]
            for index, value in enumerate(gradient_norms)
            if value == 0
        ]
        raise RuntimeError(
            f"{len(entries) - nonzero_count} decisions have zero gradient: "
            f"{zero_destinations[:8]}")

    native_stats = _aggregate_native_stats(native_sources)
    run = {
        "loss": float(loss.item()),
        "token_count": token_count,
        "hidden_state_sha256": tensor_sha256(hidden_states),
        "assignment_gradient_sha256": tensor_sha256(logits.grad),
        "assignment_gradient": logits.grad.detach().cpu().tolist(),
        "group_gradient_norms": gradient_norms.detach().cpu().tolist(),
        "nonzero_group_gradient_count": nonzero_count,
        "all_gradients_finite": bool(torch.isfinite(logits.grad).all()),
        "elapsed_seconds": elapsed,
        "final_norm_checkpoint_bytes": final_norm_bytes,
        "block_stats": [asdict(stats) for stats in block_stats],
        "embedding_stats": asdict(embedding_stats),
        "output_stats": asdict(output_stats),
        "native_source_stats": native_stats,
        "packed_payload_bytes_read": (
            native_stats["reference_payload_bytes_read"]
            + native_stats["alternative_payload_bytes_read"]),
        "max_materialized_linear_chunk_bytes": max(
            stats.max_materialized_chunk_bytes for stats in operation_stats),
        "routed_source_call_count": len(expert_source_records),
        "unique_routed_experts_by_block": {
            str(block_index): len({
                expert_index
                for record_block, _, expert_index in expert_source_records
                if record_block == block_index
            })
            for block_index in range(len(originals))
        },
    }
    for index, module in enumerate(originals):
        adapter.layers[index] = module
    del output, hidden_states, loss, logits, model
    gc.collect()
    return run


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model_dir = args.model_dir.resolve(strict=True)
    identity_path = args.identity.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    store_path = args.store.resolve(strict=True)
    identity = _load_json(identity_path)
    manifest = _load_json(manifest_path)
    entries = sorted(
        (entry for entry in manifest["entries"] if entry.get("rco_search")),
        key=lambda entry: entry["destination_name"],
    )
    if len(entries) != 512:
        raise RuntimeError(f"expected 512 decisions, found {len(entries)}")

    codec = GGMLNativeCodec(args.ggml_library.resolve(strict=True))
    store = NativeCandidateStore(store_path, codec)
    revision = store.index["source"].get(
        "revision", store.index["source"].get("dense_revision"))
    if revision != identity["revision"]:
        raise RuntimeError("candidate store and checkpoint revisions differ")
    if set(store.index["tensors"]) != {
        entry["destination_name"] for entry in entries
    }:
        raise RuntimeError("candidate store does not exactly cover decisions")

    loader = SafeTensorPrefixLoader(model_dir)
    schema_model = _instantiate_model(model_dir)
    schema_adapter = get_model_adapter(schema_model)
    block_schemas = [
        loader.assert_prefix_schema(
            schema_model, f"{schema_adapter.layers_path}.{index}")
        for index in range(len(schema_adapter.layers))
    ]
    del schema_model

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    input_ids = tokenizer(
        CALIBRATION_TEXT,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=args.sequence_length,
    )["input_ids"]
    if input_ids.shape[1] < args.sequence_length:
        raise RuntimeError("calibration text is shorter than requested sequence")
    input_ids = input_ids[:, :args.sequence_length].contiguous()

    run = _run_once(
        model_dir=model_dir,
        loader=loader,
        store=store,
        entries=entries,
        input_ids=input_ids,
        device=device,
        rows_per_chunk=args.rows_per_chunk,
    )
    reference = None
    reproducible = None
    if args.reference_report is not None:
        reference_path = args.reference_report.resolve(strict=True)
        reference_report = _load_json(reference_path)
        if (
            reference_report["source"]["revision"] != identity["revision"]
            or reference_report["calibration"]["input_ids"]
            != input_ids.tolist()[0]
            or reference_report["candidate_store"]["index_sha256"]
            != _sha256_file(store_path / "native-candidate-index.json")
            or reference_report["candidate_store"]["rows_per_chunk"]
            != args.rows_per_chunk
        ):
            raise RuntimeError("reference report describes a different run")
        reference = {
            "path": str(reference_path),
            "sha256": _sha256_file(reference_path),
        }
        reference_run = reference_report["run"]
        reproducible = all(run[key] == reference_run[key] for key in (
            "loss",
            "token_count",
            "hidden_state_sha256",
            "assignment_gradient_sha256",
            "assignment_gradient",
            "group_gradient_norms",
        ))
        if not reproducible:
            raise RuntimeError("full-model relaxed backward did not reproduce")

    cuda = {
        "available": torch.cuda.is_available(),
        "built_version": torch.version.cuda,
        "allocated_peak_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else 0),
        "reserved_peak_bytes": (
            torch.cuda.max_memory_reserved(device)
            if device.type == "cuda" else 0),
    }
    if device.type == "cuda":
        cuda["device_name"] = torch.cuda.get_device_name(device)
    return {
        "schema": 1,
        "status": "pass",
        "scope": (
            "one complete relaxed forward/backward through all 40 "
            "Qwen3.6-35B-A3B text blocks and all 512 native Q2_0/Q4_0 "
            "decisions, including sparse embedding and streamed exact output "
            "loss; this proves the gradient path, not optimizer quality or CUDA"
        ),
        "source": {
            "repo_id": identity["repo_id"],
            "revision": identity["revision"],
            "model_dir": str(model_dir),
            "identity_path": str(identity_path),
            "identity_sha256": _sha256_file(identity_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
        },
        "candidate_store": {
            "path": str(store_path),
            "index_sha256": _sha256_file(
                store_path / "native-candidate-index.json"),
            "decision_count": len(entries),
            "reference_type": GGMLType.Q4_0.name,
            "alternative_type": GGMLType.Q2_0.name,
            "rows_per_chunk": args.rows_per_chunk,
        },
        "calibration": {
            "text": CALIBRATION_TEXT,
            "sequence_length": args.sequence_length,
            "input_ids": input_ids.tolist()[0],
            "objective": "exact full-vocabulary causal cross-entropy",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "cuda": cuda,
        },
        "block_schema": block_schemas,
        "initial_logits": _initial_logits(
            len(entries), torch.device("cpu")).detach().tolist(),
        "run": run,
        "reproducibility_reference": reference,
        "same_seed_reproducible": reproducible,
        "memory": {
            "peak_process_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "max_materialized_block_bytes": max(
                stats["max_materialized_block_bytes"]
                for stats in run["block_stats"]),
            "max_native_source_working_set_bytes": run[
                "native_source_stats"]["max_resident_decoded_bytes"],
            "max_materialized_linear_chunk_bytes": run[
                "max_materialized_linear_chunk_bytes"],
            "max_cuda_allocated_bytes": cuda["allocated_peak_bytes"],
            "max_cuda_reserved_bytes": cuda["reserved_peak_bytes"],
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sequence_length < 2:
        raise ValueError("sequence length must be at least two")
    report = audit(args)
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "loss": report["run"]["loss"],
        "nonzero_group_gradient_count": report["run"][
            "nonzero_group_gradient_count"],
        "same_seed_reproducible": report["same_seed_reproducible"],
        "memory": report["memory"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
