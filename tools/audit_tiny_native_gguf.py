#!/usr/bin/env python3
"""Build and load a complete tiny mixed-type Qwen3.5-MoE GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from native_gguf import import_pinned_gguf, write_selected_native_gguf
from native_store import NativeCandidateStore, NativeCandidateStoreWriter
from quant.ggml_native import GGMLNativeCodec, GGMLType


HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 64
NUM_EXPERTS = 3
VOCAB_SIZE = 248_320
SELECTED_TYPES = {
    "blk.0.attn_q.weight": GGMLType.Q2_0,
    "blk.0.ffn_gate_exps.weight": GGMLType.Q4_0,
    "blk.0.ffn_up_exps.weight": GGMLType.Q2_0,
    "blk.0.ffn_down_exps.weight": GGMLType.Q4_0,
}
EXPERT_TENSORS = {
    "gate": "blk.0.ffn_gate_exps.weight",
    "up": "blk.0.ffn_up_exps.weight",
    "down": "blk.0.ffn_down_exps.weight",
}
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def _sha256_bytes(value: bytes | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _deterministic_values(shape: tuple[int, ...], salt: int):
    import torch

    count = int(np.prod(shape))
    values = torch.arange(count, dtype=torch.float32).reshape(shape)
    return (
        torch.sin((values + salt * 17.0) / 53.0) * 0.75
        + torch.cos((values + salt * 7.0) / 29.0) * 0.25
        + salt * 0.03125
    )


def _make_checkpoint(checkpoint: Path, tokenizer_source: Path) -> dict[str, np.ndarray]:
    import torch
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    config = Qwen3_5MoeTextConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        layer_types=["full_attention"],
        moe_intermediate_size=INTERMEDIATE_SIZE,
        shared_expert_intermediate_size=INTERMEDIATE_SIZE,
        num_experts_per_tok=1,
        num_experts=NUM_EXPERTS,
        tie_word_embeddings=True,
        dtype="float32",
    )
    config.architectures = ["Qwen3_5MoeForCausalLM"]
    torch.manual_seed(1234)
    model = Qwen3_5MoeForCausalLM(config)
    experts = model.model.layers[0].mlp.experts
    expected: dict[str, np.ndarray] = {}
    with torch.no_grad():
        gate = _deterministic_values(
            (NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE), 1)
        up = _deterministic_values(
            (NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE), 2)
        down = _deterministic_values(
            (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE), 3)
        experts.gate_up_proj.copy_(torch.cat((gate, up), dim=1))
        experts.down_proj.copy_(down)
        expected = {
            "gate": gate.numpy().copy(),
            "up": up.numpy().copy(),
            "down": down.numpy().copy(),
        }
    checkpoint.mkdir(parents=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    del model
    for name in TOKENIZER_FILES:
        source = tokenizer_source / name
        if not source.is_file():
            raise FileNotFoundError(f"required tokenizer file is missing: {source}")
        shutil.copy2(source, checkpoint / name)
    return expected


def _convert_checkpoint(checkpoint: Path, reference: Path, llama_cpp: Path) -> None:
    command = [
        sys.executable,
        str(llama_cpp / "convert_hf_to_gguf.py"),
        str(checkpoint),
        "--outfile",
        str(reference),
        "--outtype",
        "f32",
        "--no-mtp",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    if "Model successfully exported" not in completed.stderr:
        raise RuntimeError("pinned converter did not report a successful export")


def _errors(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    denominator = float(np.linalg.norm(reference.astype(np.float64)))
    return {
        "max_absolute_error": float(np.max(np.abs(difference))),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "relative_frobenius_error": (
            float(np.linalg.norm(difference)) / denominator if denominator else 0.0
        ),
    }


def _probe_model(probe: Path, model: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(probe), str(model)], check=True, text=True, capture_output=True)
    result = json.loads(completed.stdout.strip())
    if result.get("status") != "pass":
        raise RuntimeError(f"unexpected llama model probe result: {result}")
    tensor_types = {
        name: int(count)
        for name, count in re.findall(
            r"- type\s+(\S+):\s+(\d+) tensors", completed.stderr)
    }
    return {
        **result,
        "probe_sha256": _sha256_file(probe),
        "loaded_tensor_types": tensor_types,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    llama_cpp = args.llama_cpp.resolve(strict=True)
    actual_revision = _git_revision(llama_cpp)
    if actual_revision != args.llama_revision:
        raise RuntimeError(
            f"llama.cpp revision mismatch: {actual_revision} != {args.llama_revision}")
    gguf_python = llama_cpp / "gguf-py"
    gguf = import_pinned_gguf(gguf_python)
    codec = GGMLNativeCodec(args.ggml_library)

    parent = None if args.temporary_parent is None else str(args.temporary_parent)
    with tempfile.TemporaryDirectory(
        prefix="rco-tiny-native-gguf-", dir=parent) as directory_name:
        directory = Path(directory_name)
        checkpoint = directory / "checkpoint"
        reference_path = directory / "reference-f32.gguf"
        output_path = directory / "mixed-native.gguf"
        store_path = directory / "native-store"
        expected_experts = _make_checkpoint(
            checkpoint, args.tokenizer_source.resolve(strict=True))
        _convert_checkpoint(checkpoint, reference_path, llama_cpp)

        reference = gguf.GGUFReader(reference_path)
        reference_tensors = {tensor.name: tensor for tensor in reference.tensors}
        reference_sha256 = _sha256_file(reference_path)
        missing = sorted(set(SELECTED_TYPES) - set(reference_tensors))
        if missing:
            raise RuntimeError(f"converted fixture is missing tensors: {missing}")

        expert_order_records = []
        for projection, tensor_name in EXPERT_TENSORS.items():
            actual = np.asarray(reference_tensors[tensor_name].data)
            expected = expected_experts[projection]
            if not np.array_equal(actual, expected):
                raise RuntimeError(
                    f"converter changed {projection} expert values or ordering")
            expert_order_records.append({
                "projection": projection,
                "tensor": tensor_name,
                "expert_order": list(range(NUM_EXPERTS)),
                "shape": list(actual.shape),
                "per_expert_sha256": [
                    _sha256_bytes(np.ascontiguousarray(actual[index]).tobytes())
                    for index in range(NUM_EXPERTS)
                ],
                "exactly_matches_source_fused_slice_order": True,
            })

        store_writer = NativeCandidateStoreWriter(
            store_path,
            codec,
            source={
                "kind": "deterministic-tiny-qwen35moe",
                "llama_cpp_revision": actual_revision,
                "expert_order": list(range(NUM_EXPERTS)),
            },
        )
        candidates: list[dict[str, Any]] = []
        for tensor_name in SELECTED_TYPES:
            values = np.asarray(reference_tensors[tensor_name].data)
            for candidate_type in (GGMLType.Q2_0, GGMLType.Q4_0):
                entry = store_writer.quantize_array(
                    tensor_name,
                    candidate_type,
                    values,
                    rows_per_chunk=2,
                    provenance={
                        "reference_tensor": tensor_name,
                        "reference_gguf_sha256": reference_sha256,
                        "expert_order": (
                            list(range(NUM_EXPERTS))
                            if tensor_name in EXPERT_TENSORS.values()
                            else None
                        ),
                    },
                )
                candidates.append({
                    "tensor": tensor_name,
                    "ggml_type": candidate_type.name,
                    "ggml_type_id": int(candidate_type),
                    "gguf_shape": entry["gguf_shape"],
                    "payload_bytes": entry["payload_bytes"],
                    "aligned_gguf_bytes": entry["aligned_gguf_bytes"],
                    "sha256": entry["sha256"],
                })
        store_writer.finalize()
        store = NativeCandidateStore(store_path, codec)
        selected = write_selected_native_gguf(
            reference_path,
            output_path,
            store,
            SELECTED_TYPES,
            gguf_python=gguf_python,
        )

        output = gguf.GGUFReader(output_path)
        output_tensors = {tensor.name: tensor for tensor in output.tensors}
        if list(reference_tensors) != list(output_tensors):
            raise RuntimeError("output tensor inventory or order changed")
        selected_validation = []
        for tensor_name, selected_type in SELECTED_TYPES.items():
            tensor = output_tensors[tensor_name]
            metadata = store.metadata(tensor_name, selected_type)
            payload = tensor.data.reshape(-1).tobytes()
            if int(tensor.tensor_type) != int(selected_type):
                raise RuntimeError(f"type mismatch after GGUF load: {tensor_name}")
            if _sha256_bytes(payload) != metadata["sha256"]:
                raise RuntimeError(f"payload mismatch after GGUF load: {tensor_name}")
            store_decoded = np.empty(tuple(reversed(metadata["gguf_shape"])), np.float32)
            store.decode_into(tensor_name, selected_type, store_decoded, rows_per_chunk=2)
            gguf_decoded = np.empty_like(store_decoded)
            codec.dequantize_rows_into(
                payload, selected_type, gguf_decoded.reshape(-1, gguf_decoded.shape[-1]))
            if not np.array_equal(store_decoded, gguf_decoded):
                raise RuntimeError(f"search/runtime decode mismatch: {tensor_name}")
            selected_validation.append({
                "tensor": tensor_name,
                "ggml_type": selected_type.name,
                "ggml_type_id": int(selected_type),
                "gguf_shape": [int(value) for value in tensor.shape],
                "payload_bytes": tensor.n_bytes,
                "data_offset": tensor.data_offset,
                "data_offset_aligned": tensor.data_offset % output.alignment == 0,
                "sha256": metadata["sha256"],
                "candidate_payload_exactly_preserved": True,
                "store_decode_equals_gguf_payload_decode": True,
                "errors": _errors(
                    np.asarray(reference_tensors[tensor_name].data), gguf_decoded),
            })

        unchanged = []
        for tensor_name, reference_tensor in reference_tensors.items():
            if tensor_name in SELECTED_TYPES:
                continue
            actual = output_tensors[tensor_name]
            reference_bytes = reference_tensor.data.reshape(-1).tobytes()
            actual_bytes = actual.data.reshape(-1).tobytes()
            identical = (
                int(reference_tensor.tensor_type) == int(actual.tensor_type)
                and tuple(reference_tensor.shape) == tuple(actual.shape)
                and reference_bytes == actual_bytes
            )
            if not identical:
                raise RuntimeError(f"unselected tensor changed: {tensor_name}")
            unchanged.append(tensor_name)

        runtime_probe = _probe_model(args.llama_probe.resolve(strict=True), output_path)
        if runtime_probe["embedding_length"] != HIDDEN_SIZE:
            raise RuntimeError("runtime loaded an unexpected embedding length")
        if runtime_probe["layer_count"] != 1:
            raise RuntimeError("runtime loaded an unexpected layer count")
        expected_runtime_types = {"f32": 14, "q4_0": 2, "q2_0": 2}
        if runtime_probe["loaded_tensor_types"] != expected_runtime_types:
            raise RuntimeError(
                "runtime tensor type inventory differs: "
                f"{runtime_probe['loaded_tensor_types']} != {expected_runtime_types}")
        output_size = output_path.stat().st_size
        output_sha256 = _sha256_file(output_path)
        reference_size = reference_path.stat().st_size

    return {
        "schema": 1,
        "status": "pass",
        "scope": (
            "complete deterministic one-layer Qwen3.5-MoE fixture; this proves "
            "expert aggregation, mixed native tensor types, byte preservation, "
            "and CPU model loading, not real-model accuracy or a CUDA memory gate"
        ),
        "llama_cpp": {
            "path": str(llama_cpp),
            "revision": actual_revision,
            "runtime_probe": str(args.llama_probe.resolve()),
        },
        "ggml_library": {
            "path": str(codec.library_path),
            "sha256": codec.library_sha256,
        },
        "fixture": {
            "architecture": "qwen35moe",
            "layer_count": 1,
            "tensor_count": len(reference_tensors),
            "hidden_size": HIDDEN_SIZE,
            "expert_intermediate_size": INTERMEDIATE_SIZE,
            "expert_count": NUM_EXPERTS,
            "experts_used": 1,
            "vocabulary_size": VOCAB_SIZE,
            "complete_model_inventory_preserved": True,
            "expert_aggregation": expert_order_records,
        },
        "candidate_store": {
            "decision_group_count": len(SELECTED_TYPES),
            "candidate_count": len(candidates),
            "candidate_types": [GGMLType.Q2_0.name, GGMLType.Q4_0.name],
            "candidates": candidates,
        },
        "assignment": {
            key: value.name for key, value in SELECTED_TYPES.items()
        },
        "mixed_type_count": len(set(SELECTED_TYPES.values())),
        "gguf_validation": {
            "reference_bytes": reference_size,
            "reference_sha256": reference_sha256,
            "output_bytes": output_size,
            "output_sha256": output_sha256,
            "alignment": output.alignment,
            "selected": selected_validation,
            "selected_writer_records": selected,
            "unchanged_tensor_count": len(unchanged),
            "unchanged_tensors_exact": True,
            "pinned_gguf_reader_load": True,
        },
        "unmodified_llama_cpp_model_load": runtime_probe,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-source", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--llama-revision", required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--llama-probe", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/qwen35_tiny_native_gguf.json"))
    parser.add_argument("--temporary-parent", type=Path, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(args)
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "tensor_count": report["fixture"]["tensor_count"],
        "candidate_count": report["candidate_store"]["candidate_count"],
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
