"""Persistence helpers for small dense numerical-oracle tensors."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def tensor_sha256(tensor: torch.Tensor, chunk_bytes: int = 8 << 20) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    raw = value.view(torch.uint8).reshape(-1).numpy()
    view = memoryview(raw)
    for start in range(0, len(view), chunk_bytes):
        digest.update(view[start:start + chunk_bytes])
    return digest.hexdigest()


def file_sha256(path: str | os.PathLike, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def write_block_oracle(
    path: str | os.PathLike,
    tensors: dict[str, torch.Tensor],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Atomically persist caller-owned CPU tensors in safetensors format."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = {
        name: value.detach().to(device="cpu").contiguous()
        for name, value in tensors.items()
    }
    encoded_metadata = {key: str(value) for key, value in metadata.items()}
    metadata_bytes = json.dumps(
        encoded_metadata, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    stored_values = {
        "__metadata_json__": torch.tensor(list(metadata_bytes), dtype=torch.uint8),
        **values,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        # safetensors metadata is a Rust hash map whose serialized key order is
        # not stable across processes. Store canonical JSON as a tensor so the
        # complete oracle artifact, not only its numerical tensors, is byte
        # reproducible.
        save_file(stored_values, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
        "tensors": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype).replace("torch.", ""),
                "sha256": tensor_sha256(value),
            }
            for name, value in values.items()
        },
        "metadata": encoded_metadata,
    }
