#!/usr/bin/env python3
"""Reproduce the bounded native-GGML codec and candidate-store proof."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from native_store import NativeCandidateStore, NativeCandidateStoreWriter
from quant.ggml_native import GGMLNativeCodec, GGMLType


def _errors(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    reference64 = reference.astype(np.float64)
    denominator = float(np.linalg.norm(reference64))
    return {
        "max_absolute_error": float(np.max(np.abs(difference))),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "mean_signed_error": float(np.mean(difference)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "relative_frobenius_error": (
            float(np.linalg.norm(difference)) / denominator
            if denominator else 0.0
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ggml-library", required=True)
    parser.add_argument(
        "--output",
        default="reports/native_ggml_store_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    codec = GGMLNativeCodec(args.ggml_library)
    ordinary = (
        np.sin(np.arange(9 * 64, dtype=np.float32) / 17.0)
        * np.linspace(0.25, 3.0, 9 * 64, dtype=np.float32)
    ).reshape(9, 64)
    experts = (
        np.cos(np.arange(3 * 7 * 64, dtype=np.float32) / 23.0)
        * np.linspace(2.0, 0.125, 3 * 7 * 64, dtype=np.float32)
    ).reshape(3, 7, 64)
    fixtures = {
        "blk.0.attn_q.weight": {
            "values": ordinary,
            "provenance": {
                "fixture": "ordinary-matrix",
                "source_tensor": "model.layers.0.self_attn.q_proj.weight",
            },
        },
        "blk.0.ffn_gate_exps.weight": {
            "values": experts,
            "provenance": {
                "fixture": "aggregated-routed-experts",
                "source_family": "model.layers.0.mlp.experts.*.gate_proj.weight",
                "expert_order": [0, 1, 2],
            },
        },
    }
    candidate_types = (GGMLType.Q2_0, GGMLType.Q4_0)
    records: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="rco-native-ggml-audit-") as directory:
        writer = NativeCandidateStoreWriter(
            directory,
            codec,
            source={
                "kind": "deterministic-synthetic-fixture",
                "purpose": "native-codec-and-store-proof",
            },
        )
        written: dict[tuple[str, GGMLType], dict[str, Any]] = {}
        for tensor_name, fixture in fixtures.items():
            values = fixture["values"]
            for ggml_type in candidate_types:
                entry = writer.quantize_array(
                    tensor_name,
                    ggml_type,
                    values,
                    provenance=fixture["provenance"],
                    rows_per_chunk=4,
                )
                written[(tensor_name, ggml_type)] = entry
        index_path = writer.finalize()
        store = NativeCandidateStore(directory, codec)

        for tensor_name, fixture in fixtures.items():
            values = fixture["values"]
            flattened = values.reshape(-1, values.shape[-1])
            for ggml_type in candidate_types:
                entry = written[(tensor_name, ggml_type)]
                decoded = np.empty_like(values)
                store.decode_into(
                    tensor_name, ggml_type, decoded, rows_per_chunk=3)
                stored = b"".join(store.iter_payload(
                    tensor_name, ggml_type, chunk_bytes=13))
                direct = codec.quantize_rows(flattened, ggml_type)
                if stored != direct:
                    raise RuntimeError(
                        f"chunked bytes differ from direct GGML output for "
                        f"{tensor_name}/{ggml_type.name}"
                    )
                records.append({
                    "tensor": tensor_name,
                    "ggml_type": ggml_type.name,
                    "gguf_shape": entry["gguf_shape"],
                    "block_size": entry["block_size"],
                    "type_size": entry["type_size"],
                    "row_size": entry["row_size"],
                    "row_count": entry["row_count"],
                    "payload_bytes": entry["payload_bytes"],
                    "aligned_gguf_bytes": entry["aligned_gguf_bytes"],
                    "sha256": entry["sha256"],
                    "chunked_bytes_equal_direct_native_call": True,
                    "errors": _errors(values, decoded),
                })

        index = json.loads(index_path.read_text())
        temporary_files = [
            str(path.relative_to(directory))
            for path in Path(directory).rglob("*.tmp")
        ]
        if temporary_files:
            raise RuntimeError(f"unpublished temporary files remain: {temporary_files}")
        if len(store._verified) != len(records):
            raise RuntimeError("not every candidate was integrity-verified")

    report = {
        "schema": 1,
        "status": "pass",
        "scope": (
            "deterministic ordinary and aggregated-expert tensors; this is "
            "not a real-model or CUDA memory gate"
        ),
        "ggml_library": {
            "path": str(codec.library_path),
            "sha256": codec.library_sha256,
        },
        "candidate_types": [value.name for value in candidate_types],
        "tensor_count": index["tensor_count"],
        "candidate_count": index["candidate_count"],
        "alignment": index["alignment"],
        "caller_owned_decode_buffer": True,
        "persistent_decoded_cache": False,
        "atomic_payload_publication": True,
        "atomic_index_publication": True,
        "temporary_files_after_finalize": temporary_files,
        "candidates": records,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "candidate_count": report["candidate_count"],
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

