"""Immutable identity and tensor-inventory audits for local Qwen checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from safetensors import safe_open

from model_adapter import classify_qwen35_tensor


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def sha256_file(path: str | os.PathLike, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _numel(shape: list[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _download_revision(model_dir: Path, filenames: list[str]) -> tuple[str, dict[str, str]]:
    metadata_root = model_dir / ".cache" / "huggingface" / "download"
    revisions: dict[str, str] = {}
    etags: dict[str, str] = {}
    for filename in filenames:
        metadata_path = metadata_root / f"{filename}.metadata"
        if not metadata_path.is_file():
            raise ValueError(f"download metadata is missing for {filename}")
        lines = metadata_path.read_text().splitlines()
        if len(lines) < 2:
            raise ValueError(f"invalid download metadata at {metadata_path}")
        revisions[filename] = lines[0]
        etags[filename] = lines[1]
    unique = set(revisions.values())
    if len(unique) != 1:
        raise ValueError(f"checkpoint files came from mixed revisions: {revisions}")
    return unique.pop(), etags


def audit_qwen_checkpoint_identity(
    model_dir: str | os.PathLike,
    *,
    repo_id: str,
    expected_revision: str,
    expected_weight_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit one downloaded checkpoint without materializing tensor payloads."""
    root = Path(model_dir).resolve(strict=True)
    index_path = root / "model.safetensors.index.json"
    config_path = root / "config.json"
    if not index_path.is_file() or not config_path.is_file():
        raise ValueError(f"{root} is missing config.json or the safetensors index")

    config = json.loads(config_path.read_text())
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path} has no non-empty weight_map")
    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        if Path(shard_name).name != shard_name:
            raise ValueError(f"unsafe shard path in index: {shard_name}")
        if not (root / shard_name).is_file():
            raise ValueError(f"indexed shard is missing: {shard_name}")

    top_level_files = sorted(
        path.name for path in root.iterdir()
        if path.is_file()
    )
    downloaded_revision, etags = _download_revision(root, top_level_files)
    if downloaded_revision != expected_revision:
        raise ValueError(
            f"download revision {downloaded_revision} does not match expected "
            f"{expected_revision}"
        )

    file_records = []
    for filename in top_level_files:
        path = root / filename
        file_records.append({
            "path": filename,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "hub_etag": etags[filename],
        })
    if expected_weight_sha256 is not None:
        if len(shard_names) != 1:
            raise ValueError("one expected weight hash was supplied for multiple shards")
        actual = next(
            record["sha256"] for record in file_records
            if record["path"] == shard_names[0]
        )
        if actual != expected_weight_sha256:
            raise ValueError(
                f"weight shard SHA-256 {actual} does not match expected "
                f"{expected_weight_sha256}"
            )

    shard_keys: set[str] = set()
    inventory: list[dict[str, Any]] = []
    for shard_name in shard_names:
        with safe_open(root / shard_name, framework="pt", device="cpu") as shard:
            keys = set(shard.keys())
            duplicate = shard_keys.intersection(keys)
            if duplicate:
                raise ValueError(f"tensor keys occur in multiple shards: {sorted(duplicate)[:8]}")
            shard_keys.update(keys)
            for name in sorted(keys):
                tensor_slice = shard.get_slice(name)
                shape = [int(value) for value in tensor_slice.get_shape()]
                dtype = str(tensor_slice.get_dtype())
                if dtype not in _DTYPE_BYTES:
                    raise ValueError(f"unsupported safetensors dtype {dtype} for {name}")
                numel = _numel(shape)
                inventory.append({
                    "name": name,
                    "category": classify_qwen35_tensor(name),
                    "dtype": dtype,
                    "shape": shape,
                    "numel": numel,
                    "logical_bytes": numel * _DTYPE_BYTES[dtype],
                    "shard": shard_name,
                })

    indexed_keys = set(weight_map)
    if indexed_keys != shard_keys:
        missing = sorted(indexed_keys - shard_keys)
        unexpected = sorted(shard_keys - indexed_keys)
        raise ValueError(
            f"index/shard key mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected"
        )
    wrong_shards = [
        item["name"] for item in inventory
        if weight_map[item["name"]] != item["shard"]
    ]
    if wrong_shards:
        raise ValueError(f"index assigns tensors to wrong shards: {wrong_shards[:8]}")

    categories = Counter(item["category"] for item in inventory)
    unknown = [item["name"] for item in inventory if item["category"] == "unknown"]
    if unknown:
        raise ValueError(f"unclassified checkpoint tensors: {unknown[:8]}")
    text_inventory = [
        item for item in inventory if item["category"] not in {"vision", "mtp"}
    ]
    logical_bytes = sum(item["logical_bytes"] for item in inventory)
    declared_bytes = int(index.get("metadata", {}).get("total_size", -1))
    if declared_bytes != logical_bytes:
        raise ValueError(
            f"index declares {declared_bytes} bytes but tensor inventory has "
            f"{logical_bytes}"
        )

    text_config = config.get("text_config", config)
    layer_types = list(text_config.get("layer_types", []))
    return {
        "schema": 1,
        "status": "pass",
        "repo_id": repo_id,
        "revision": downloaded_revision,
        "model_dir": str(root),
        "config": {
            "architecture": config.get("architectures", [None])[0],
            "model_type": config.get("model_type"),
            "text_model_type": text_config.get("model_type"),
            "dtype": text_config.get("dtype"),
            "tie_word_embeddings": config.get("tie_word_embeddings"),
            "hidden_size": text_config.get("hidden_size"),
            "intermediate_size": text_config.get("intermediate_size"),
            "num_hidden_layers": text_config.get("num_hidden_layers"),
            "vocab_size": text_config.get("vocab_size"),
            "layer_type_counts": dict(sorted(Counter(layer_types).items())),
        },
        "files": file_records,
        "shard_count": len(shard_names),
        "tensor_count": len(inventory),
        "text_tensor_count": len(text_inventory),
        "omitted_tensor_count": categories["vision"] + categories["mtp"],
        "logical_weight_bytes": logical_bytes,
        "text_logical_weight_bytes": sum(
            item["logical_bytes"] for item in text_inventory),
        "category_counts": dict(sorted(categories.items())),
        "unknown_tensors": unknown,
        "text_inventory": text_inventory,
    }

