#!/usr/bin/env python3
"""Phase 1 of RCO_PLAN_NEW.md: GSQ-E6, authentic GSQ with a Q6_K token_embd.

Subcommands, each resumable:

``quantize``
    Quantize the pinned BF16 ``embed_tokens`` (bit-identical to GSQ's BF16
    ``token_embd``, which is checked) to Q6_K and record its checksum and
    relative RMSE.  ``token_embd`` is a lookup table, not a matmul input, so
    no importance matrix applies.
``evaluate``
    Exact calibration NLL of GSQ-E6 in the streaming evaluator, paired by
    document against the Phase 0 GSQ arm scored with the identical setup.
    Gate: |token-weighted delta NLL| <= 0.002.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import mmap
import os
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
from quant.ggml_native import GGMLNativeCodec  # noqa: E402
from quant.ggml_raw import (  # noqa: E402
    GGML_TYPE_Q6_K,
    dequantize_rows_raw_into,
    quantize_rows_raw,
    raw_row_size,
)
from release_quality import paired_bootstrap_mean_ci  # noqa: E402


SCHEMA = "rco.qwen36.gsq_e6.v1"
NAME = "token_embd.weight"
GGML_TYPE_BF16 = 30
DELTA_GATE = 0.002
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20261001


class EmbeddingQ6KWeightStore:
    """Choice 0 keeps GSQ's BF16 embedding; choice 1 installs the Q6_K one."""

    cache = False

    def __init__(self, codec: GGMLNativeCodec, payload: Path, sha256: str,
                 entry: dict[str, Any], *, rows_per_chunk: int) -> None:
        self.codec = codec
        self.payload = payload
        self.entry = entry
        self.rows_per_chunk = rows_per_chunk
        self.rows, self.width = (int(value) for value in entry["source_shape"])
        self.row_size = raw_row_size(codec, GGML_TYPE_Q6_K, self.width)
        if payload.stat().st_size != self.rows * self.row_size:
            raise ValueError("Q6_K embedding payload size differs")
        if _sha256_file(payload) != sha256:
            raise ValueError("Q6_K embedding payload checksum differs")

    @staticmethod
    def candidate_location(name: str) -> str:
        if name != NAME:
            raise KeyError(name)
        return "embedding"

    @staticmethod
    def is_retain_choice(name: str, choice: int) -> bool:
        if name != NAME or int(choice) not in (0, 1):
            raise ValueError(f"invalid embedding choice {name}/{choice}")
        return int(choice) == 0

    def get_layer_storage_bytes(self, name: str, choice: int) -> int:
        return 0 if self.is_retain_choice(name, choice) else self.rows * self.row_size

    def install_layer_weight(self, model: Any, name: str, choice: int) -> dict[str, int]:
        import torch

        if self.is_retain_choice(name, choice):
            raise ValueError("retain needs no installation")
        target = model.get_parameter(self.entry["source_name"])
        if tuple(target.shape) != (self.rows, self.width):
            raise RuntimeError(f"embedding target shape changed: {tuple(target.shape)}")
        max_decoded = max_install = installed = 0
        with self.payload.open("rb") as handle, mmap.mmap(
            handle.fileno(), 0, access=mmap.ACCESS_READ,
        ) as payload:
            for start in range(0, self.rows, self.rows_per_chunk):
                stop = min(start + self.rows_per_chunk, self.rows)
                decoded = np.empty((stop - start, self.width), dtype=np.float32)
                packed = memoryview(payload)[start * self.row_size:stop * self.row_size]
                try:
                    dequantize_rows_raw_into(self.codec, packed, GGML_TYPE_Q6_K, decoded)
                finally:
                    packed.release()
                values = torch.from_numpy(decoded).to(
                    device=target.device, dtype=target.dtype)
                with torch.no_grad():
                    target[start:stop].copy_(values)
                installed += stop - start
                max_decoded = max(max_decoded, decoded.nbytes)
                max_install = max(max_install, values.numel() * values.element_size())
        if installed != self.rows:
            raise RuntimeError(f"installed {installed} embedding rows; expected {self.rows}")
        return {"installed_rows": installed, "max_decoded_fp32_bytes": max_decoded,
                "max_install_bf16_bytes": max_install}


def _entry(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    entries = [entry for entry in manifest["entries"]
               if entry["destination_name"] == NAME]
    if len(entries) != 1:
        raise RuntimeError("manifest does not map token_embd exactly once")
    return entries[0]


def run_quantize(args: argparse.Namespace) -> None:
    from native_gguf import import_pinned_gguf
    from qwen35_native import SafetensorGGUFRowSource

    output = args.reports / "qwen36_gsq_e6_embedding.json"
    if output.exists() and _load_json(output).get("status") == "complete":
        print("quantize: already complete", flush=True)
        return
    started = time.perf_counter()
    entry = _entry(args.manifest)
    rows, width = (int(value) for value in entry["source_shape"])
    codec = GGMLNativeCodec(args.ggml_library)
    gguf = import_pinned_gguf(args.gguf_python)
    reader = gguf.GGUFReader(args.gguf)
    tensor = next(item for item in reader.tensors if item.name == NAME)
    if int(tensor.tensor_type) != GGML_TYPE_BF16:
        raise RuntimeError("GSQ token_embd is not BF16")
    gsq = np.memmap(args.gguf, dtype=np.uint16, mode="r",
                    offset=int(tensor.data_offset), shape=(rows, width))
    del reader
    source = SafetensorGGUFRowSource(args.model_dir)
    args.work.mkdir(parents=True, exist_ok=True)
    payload_path = args.work / "token_embd.Q6_K.bin"
    temporary = payload_path.with_suffix(".tmp")
    digest = hashlib.sha256()
    written = 0
    squared_error = energy = 0.0
    start = 0
    with temporary.open("wb") as handle:
        for chunk in source.iter_rows(entry, rows_per_chunk=args.quantize_rows):
            chunk = np.ascontiguousarray(chunk, dtype=np.float32)
            stop = start + chunk.shape[0]
            gsq_rows = (gsq[start:stop].astype(np.uint32) << 16).view(np.float32)
            if not np.array_equal(gsq_rows, chunk):
                raise RuntimeError(f"GSQ token_embd differs from BF16 source at {start}")
            payload = quantize_rows_raw(codec, chunk, GGML_TYPE_Q6_K)
            decoded = dequantize_rows_raw_into(
                codec, payload, GGML_TYPE_Q6_K, np.empty_like(chunk))
            squared_error += float(np.square(decoded - chunk, dtype=np.float64).sum())
            energy += float(np.square(chunk, dtype=np.float64).sum())
            handle.write(payload)
            digest.update(payload)
            written += len(payload)
            start = stop
        handle.flush()
        os.fsync(handle.fileno())
    expected = rows * raw_row_size(codec, GGML_TYPE_Q6_K, width)
    if start != rows or written != expected:
        raise RuntimeError(f"Q6_K embedding has {written} bytes; expected {expected}")
    os.replace(temporary, payload_path)
    _atomic_json(output, {
        "schema": SCHEMA + ".embedding",
        "status": "complete",
        "tensor": NAME,
        "source": {
            "model_dir": str(args.model_dir),
            "source_tensor": entry["source_name"],
            "source_shard": entry["source_shard"],
            "gsq_gguf_sha256": _sha256_file(args.gguf),
            "gsq_bf16_token_embd_bit_identical_to_source": True,
        },
        "ggml_library_sha256": codec.library_sha256,
        "payload": {
            "path": str(payload_path),
            "ggml_type": "Q6_K",
            "ggml_type_id": GGML_TYPE_Q6_K,
            "gguf_shape": entry["destination_gguf_shape"],
            "payload_bytes": written,
            "bf16_payload_bytes": rows * width * 2,
            "saved_bytes": rows * width * 2 - written,
            "sha256": digest.hexdigest(),
        },
        "relative_rmse_vs_bf16": math.sqrt(squared_error / energy),
        "wall_seconds": time.perf_counter() - started,
    })
    print(f"quantize: Q6_K token_embd {written} bytes, relative RMSE "
          f"{math.sqrt(squared_error / energy):.5f}", flush=True)


def run_evaluate(args: argparse.Namespace) -> None:
    import torch

    from gguf_checkpoint_stream import GGUFManifestPrefixLoader
    from search.streaming import StreamingHardCausalEvaluator

    output = args.reports / "qwen36_gsq_e6_calibration.json"
    if output.exists() and _load_json(output).get("status") in ("pass", "stop"):
        print("evaluate: already complete", flush=True)
        return
    embedding = _load_json(args.reports / "qwen36_gsq_e6_embedding.json")
    phase0 = _load_json(args.reports / "qwen36_q3k_viability_evaluation.json")
    if embedding.get("status") != "complete" or phase0.get("status") != "complete":
        raise RuntimeError("Q6_K embedding or Phase 0 baseline is not complete")
    device = torch.device(args.device)
    input_ids, calibration = _calibration(args)
    baseline_identity = phase0["identity"]
    identity = {
        "gsq_gguf_sha256": _sha256_file(args.gguf),
        "calibration": calibration,
        "device": str(device),
        "rows_per_chunk": args.rows_per_chunk,
        "vocab_chunk_size": args.vocab_chunk_size,
    }
    for key, value in identity.items():
        if baseline_identity[key] != value:
            raise RuntimeError(f"Phase 0 GSQ baseline was scored differently: {key}")
    entry = _entry(args.manifest)
    codec = GGMLNativeCodec(args.ggml_library)
    store = EmbeddingQ6KWeightStore(
        codec, Path(embedding["payload"]["path"]), embedding["payload"]["sha256"],
        entry, rows_per_chunk=args.install_rows)
    manifest = _load_json(args.manifest)
    model = _empty_model(args.model_dir)
    loader = GGUFManifestPrefixLoader(
        args.gguf, manifest, args.model_dir, gguf_python=args.gguf_python,
        ggml_library=args.ggml_library, rows_per_chunk=args.rows_per_chunk)
    evaluator = StreamingHardCausalEvaluator(
        model, loader, store, [SimpleNamespace(layer_names=(NAME,))], [0, 1],
        device=device, vocab_chunk_size=args.vocab_chunk_size,
        checkpoint_dtype=torch.bfloat16)
    started = time.perf_counter()
    evaluation = evaluator.evaluate(input_ids, torch.ones(1, dtype=torch.long))
    gsq = phase0["arms"]["gsq"]
    deltas = [value - base for value, base in zip(
        evaluation.document_mean_nll, gsq["document_mean_nll"], strict=True)]
    lower, upper = paired_bootstrap_mean_ci(
        deltas, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)
    token_weighted = evaluation.loss - gsq["mean_nll"]
    status = "pass" if abs(token_weighted) <= DELTA_GATE else "stop"
    _atomic_json(output, {
        "schema": SCHEMA + ".calibration",
        "status": status,
        "scope": "GSQ with a Q6_K token_embd on the pinned calibration corpus",
        "identity": identity,
        "embedding_sha256": embedding["payload"]["sha256"],
        "gsq_mean_nll": gsq["mean_nll"],
        "gsq_e6_mean_nll": evaluation.loss,
        "predicted_token_count": evaluation.token_count,
        "token_weighted_delta_vs_gsq": token_weighted,
        "paired_document_mean_delta_vs_gsq": math.fsum(deltas) / len(deltas),
        "ci95_lower": lower,
        "ci95_upper": upper,
        "documents_worse": sum(value > 0 for value in deltas),
        "max_abs_document_delta": max(abs(value) for value in deltas),
        "gate": {"max_abs_token_weighted_delta": DELTA_GATE,
                 "passed": status == "pass"},
        "document_mean_nll": list(evaluation.document_mean_nll),
        "memory": asdict(evaluation.memory),
        "wall_seconds": time.perf_counter() - started,
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED},
    })
    print(f"evaluate: GSQ-E6 NLL {evaluation.loss:.7f} (delta "
          f"{token_weighted:+.7f}, CI [{lower:+.5f}, {upper:+.5f}]) -> {status}",
          flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("quantize", "evaluate"))
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
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_gsq_e6")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--quantize-rows", type=int, default=4096)
    parser.add_argument("--install-rows", type=int, default=4096)
    args = parser.parse_args()
    for key in ("model_dir", "manifest", "calibration", "gguf", "gguf_python",
                "ggml_library", "work", "reports"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main() -> int:
    args = _parse_args()
    {"quantize": run_quantize, "evaluate": run_evaluate}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
