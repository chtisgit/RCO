#!/usr/bin/env python3
"""Generate and validate every native candidate in Qwen3.5-2B block 0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from native_gguf import import_pinned_gguf
from native_store import NativeCandidateStore, NativeCandidateStoreWriter
from quant.ggml_native import GGMLNativeCodec, GGMLType
from qwen35_native import SafetensorGGUFRowSource, generate_native_block_candidates


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _make_block_checkpoint(
    model_dir: Path,
    checkpoint: Path,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    checkpoint.mkdir(parents=True)
    names_by_shard: dict[str, list[str]] = {}
    for entry in entries:
        names_by_shard.setdefault(entry["source_shard"], []).append(
            entry["source_name"])
    tensors = {}
    logical_bytes = 0
    for shard_name, names in names_by_shard.items():
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as handle:
            for name in names:
                tensor = handle.get_tensor(name)
                tensors[name] = tensor
                logical_bytes += tensor.numel() * tensor.element_size()
    save_file(tensors, checkpoint / "model.safetensors", metadata={"format": "pt"})
    del tensors
    shutil.copy2(model_dir / "config.json", checkpoint / "config.json")
    for name in TOKENIZER_FILES:
        shutil.copy2(model_dir / name, checkpoint / name)
    return {
        "source_tensor_count": len(entries),
        "source_logical_bytes": logical_bytes,
        "checkpoint_bytes": sum(
            path.stat().st_size for path in checkpoint.iterdir() if path.is_file()),
    }


def _convert_reference(checkpoint: Path, output: Path, llama_cpp: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(llama_cpp / "convert_hf_to_gguf.py"),
            str(checkpoint),
            "--outfile",
            str(output),
            "--outtype",
            "f32",
            "--no-mtp",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    if "Model successfully exported" not in completed.stderr:
        raise RuntimeError("pinned converter did not report successful export")


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


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    model_dir = args.model_dir.resolve(strict=True)
    manifest = _load_json(args.manifest.resolve(strict=True))
    identity = _load_json(args.identity.resolve(strict=True))
    if manifest.get("status") != "pass" or identity.get("status") != "pass":
        raise RuntimeError("identity and canonical manifest must both pass")
    if manifest["source"]["revision"] != identity["revision"]:
        raise RuntimeError("identity and canonical manifest revisions differ")
    weight_files = [
        record for record in identity["files"]
        if record["path"].endswith(".safetensors")
    ]
    if len(weight_files) != 1:
        raise RuntimeError(f"expected one 2B weight shard, found {len(weight_files)}")
    weight_sha256 = weight_files[0]["sha256"]
    entries = [
        entry for entry in manifest["entries"]
        if entry["destination_name"].startswith(f"blk.{args.block}.")
    ]
    searched = [entry for entry in entries if entry["rco_search"]]
    copied = [entry for entry in entries if not entry["rco_search"]]
    if len(entries) != 14 or len(searched) != 8 or len(copied) != 6:
        raise RuntimeError(
            f"unexpected block inventory: {len(entries)}/"
            f"{len(searched)}/{len(copied)}")
    llama_cpp = args.llama_cpp.resolve(strict=True)
    actual_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=llama_cpp, check=True,
        text=True, capture_output=True).stdout.strip()
    if actual_revision != manifest["llama_cpp_revision"]:
        raise RuntimeError("llama.cpp checkout differs from canonical manifest")
    gguf = import_pinned_gguf(llama_cpp / "gguf-py")
    codec = GGMLNativeCodec(args.ggml_library)
    store_output = args.store_output.resolve()
    if store_output.exists():
        raise FileExistsError(
            f"candidate store already exists; refusing to replace it: {store_output}")
    store_output.parent.mkdir(parents=True, exist_ok=True)

    temporary_parent = (
        None if args.temporary_parent is None
        else str(args.temporary_parent.resolve(strict=True))
    )
    with tempfile.TemporaryDirectory(
        prefix="rco-qwen35-2b-block-", dir=temporary_parent) as directory_name:
        directory = Path(directory_name)
        checkpoint = directory / "checkpoint"
        reference_path = directory / "block-reference-f32.gguf"
        store_path = directory / "native-store"
        checkpoint_record = _make_block_checkpoint(model_dir, checkpoint, entries)
        _convert_reference(checkpoint, reference_path, llama_cpp)
        reference = gguf.GGUFReader(reference_path)
        reference_tensors = {tensor.name: tensor for tensor in reference.tensors}
        expected_names = [entry["destination_name"] for entry in entries]
        if list(reference_tensors) != expected_names:
            raise RuntimeError(
                "pinned converter block tensor inventory/order differs from manifest")
        for entry in entries:
            tensor = reference_tensors[entry["destination_name"]]
            if [int(value) for value in tensor.shape] != entry["destination_gguf_shape"]:
                raise RuntimeError(
                    f"pinned converter shape differs for {entry['destination_name']}")

        writer = NativeCandidateStoreWriter(
            store_path,
            codec,
            source={
                "repo_id": identity["repo_id"],
                "revision": identity["revision"],
                "weight_sha256": weight_sha256,
                "block": args.block,
                "llama_cpp_revision": actual_revision,
            },
        )
        generation = generate_native_block_candidates(
            model_dir,
            entries,
            writer,
            codec,
            rows_per_chunk=args.rows_per_chunk,
        )
        store = NativeCandidateStore(store_path, codec)
        row_source = SafetensorGGUFRowSource(model_dir)
        validations = []
        for entry in searched:
            tensor_name = entry["destination_name"]
            reference_values = np.asarray(reference_tensors[tensor_name].data)
            streamed_values = np.concatenate(list(row_source.iter_rows(
                entry, rows_per_chunk=args.rows_per_chunk)))
            if not np.array_equal(streamed_values, reference_values):
                raise RuntimeError(
                    f"streamed transform differs from pinned converter: {tensor_name}")
            for candidate_type in (GGMLType.Q2_0, GGMLType.Q4_0):
                metadata = store.metadata(tensor_name, candidate_type)
                stored = b"".join(store.iter_payload(tensor_name, candidate_type))
                reference_bytes = codec.quantize_rows(reference_values, candidate_type)
                if stored != reference_bytes:
                    raise RuntimeError(
                        f"candidate differs from quantized converter output: "
                        f"{tensor_name}/{candidate_type.name}")
                decoded = np.empty_like(reference_values)
                store.decode_into(
                    tensor_name, candidate_type, decoded,
                    rows_per_chunk=args.rows_per_chunk)
                validations.append({
                    "tensor": tensor_name,
                    "source_tensor": entry["source_name"],
                    "ggml_type": candidate_type.name,
                    "ggml_type_id": int(candidate_type),
                    "gguf_shape": metadata["gguf_shape"],
                    "payload_bytes": metadata["payload_bytes"],
                    "aligned_gguf_bytes": metadata["aligned_gguf_bytes"],
                    "sha256": metadata["sha256"],
                    "streamed_transform_equals_pinned_converter": True,
                    "chunked_candidate_equals_quantized_converter_tensor": True,
                    "errors": _errors(reference_values, decoded),
                })

        store_index = _load_json(store_path / "native-candidate-index.json")
        if store_index["tensor_count"] != len(searched):
            raise RuntimeError("native store tensor count is incomplete")
        if store_index["candidate_count"] != len(validations):
            raise RuntimeError("native store candidate count is incomplete")
        reference_record = {
            "tensor_count": len(reference_tensors),
            "bytes": reference_path.stat().st_size,
            "sha256": _sha256_file(reference_path),
            "all_names_and_shapes_match_manifest": True,
        }
        os.replace(store_path, store_output)

    return {
        "schema": 1,
        "status": "pass",
        "scope": (
            "complete canonical Qwen3.5-2B transformer block 0 candidate "
            "database; this is not a full-model, 35B, CUDA, or kernel gate"
        ),
        "source": {
            "repo_id": identity["repo_id"],
            "revision": identity["revision"],
            "weight_sha256": weight_sha256,
            "model_dir": str(model_dir),
        },
        "llama_cpp": {
            "path": str(llama_cpp),
            "revision": actual_revision,
        },
        "ggml_library": {
            "path": str(codec.library_path),
            "sha256": codec.library_sha256,
        },
        "block": {
            "index": args.block,
            "canonical_tensor_count": len(entries),
            "searched_tensor_count": len(searched),
            "copied_tensor_count": len(copied),
            "complete_manifest_coverage": True,
            "copied": [
                {
                    "tensor": entry["destination_name"],
                    "source_tensor": entry["source_name"],
                    "copy_reason": entry["copy_reason"],
                    "converter_transforms": entry["converter_transforms"],
                    "gguf_shape": entry["destination_gguf_shape"],
                    "ggml_type": entry["destination_ggml_type"],
                }
                for entry in copied
            ],
        },
        "reference_conversion": {
            **checkpoint_record,
            **reference_record,
        },
        "candidate_store": {
            "path": str(store_output),
            "index": str(store_output / "native-candidate-index.json"),
            **generation,
            "total_payload_bytes": sum(
                record["payload_bytes"] for record in validations),
            "total_aligned_gguf_bytes": sum(
                record["aligned_gguf_bytes"] for record in validations),
        },
        "validation": {
            "candidate_count": len(validations),
            "all_streamed_transforms_equal_pinned_converter": True,
            "all_chunked_candidates_equal_quantized_converter_tensors": True,
            "candidates": validations,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--store-output", type=Path, required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument("--temporary-parent", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/qwen35_2b_block0_native_candidates.json"))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(args)
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "store": report["candidate_store"]["path"],
        "candidate_count": report["validation"]["candidate_count"],
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
