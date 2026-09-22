"""Atomic store for exact native GGML candidate payloads."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import quote

import numpy as np

from quant.ggml_native import GGMLNativeCodec, GGMLType


NATIVE_CANDIDATE_INDEX = "native-candidate-index.json"
NATIVE_CANDIDATE_SCHEMA = 1


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


def _atomic_json(value: dict[str, Any], path: Path) -> None:
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
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


class NativeCandidateStoreWriter:
    """Write one immutable payload at a time and atomically publish an index."""

    def __init__(
        self,
        root: str | os.PathLike,
        codec: GGMLNativeCodec,
        *,
        source: dict[str, Any],
        alignment: int = 32,
    ):
        self.root = Path(root)
        self.codec = codec
        self.source = dict(source)
        self.alignment = alignment
        _align_up(0, alignment)
        self.root.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, dict[str, Any]] = {}

    def _payload_path(self, tensor_name: str, type_name: str) -> Path:
        if not tensor_name or tensor_name in {".", ".."}:
            raise ValueError("tensor_name must be non-empty and relative")
        encoded = quote(tensor_name, safe="")
        return self.root / "tensors" / encoded / f"{type_name}.bin"

    def write_packed_chunks(
        self,
        tensor_name: str,
        ggml_type: GGMLType | int,
        gguf_shape: Iterable[int],
        chunks: Iterable[bytes | bytearray | memoryview],
        *,
        provenance: dict[str, Any],
    ) -> dict[str, Any]:
        """Write already-native chunks, validating exact expected byte count."""
        shape = tuple(int(value) for value in gguf_shape)
        if len(shape) < 2 or any(value <= 0 for value in shape):
            raise ValueError(f"invalid GGUF shape: {shape}")
        row_width = shape[0]
        row_count = int(np.prod(shape[1:], dtype=np.int64))
        geometry = self.codec.geometry(ggml_type, row_width)
        payload_bytes = row_count * int(geometry["row_size"])
        type_name = str(geometry["ggml_type_name"])
        destination = self._payload_path(tensor_name, type_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        written = 0
        try:
            with os.fdopen(descriptor, "wb") as handle:
                for chunk in chunks:
                    view = memoryview(chunk).cast("B")
                    handle.write(view)
                    digest.update(view)
                    written += len(view)
                    if written > payload_bytes:
                        raise ValueError(
                            f"candidate {tensor_name}/{type_name} exceeds expected "
                            f"size {payload_bytes}"
                        )
                if written != payload_bytes:
                    raise ValueError(
                        f"candidate {tensor_name}/{type_name} has {written} bytes; "
                        f"expected {payload_bytes}"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

        entry = {
            **geometry,
            "path": destination.relative_to(self.root).as_posix(),
            "offset": 0,
            "gguf_shape": list(shape),
            "row_width": row_width,
            "row_count": row_count,
            "payload_bytes": payload_bytes,
            "aligned_gguf_bytes": _align_up(payload_bytes, self.alignment),
            "alignment": self.alignment,
            "byte_order": "little-endian-native-ggml",
            "sha256": digest.hexdigest(),
            "provenance": dict(provenance),
        }
        self._entries.setdefault(tensor_name, {})[type_name] = entry
        return entry

    def quantize_array(
        self,
        tensor_name: str,
        ggml_type: GGMLType | int,
        values: np.ndarray,
        *,
        provenance: dict[str, Any],
        rows_per_chunk: int = 16,
    ) -> dict[str, Any]:
        """Quantize conventional row-major values and publish exact bytes."""
        source = np.asarray(values)
        if source.ndim < 2:
            raise ValueError("candidate tensors must have at least two dimensions")
        gguf_shape = tuple(reversed(source.shape))
        return self.write_packed_chunks(
            tensor_name,
            ggml_type,
            gguf_shape,
            self.codec.iter_quantized_rows(
                source, ggml_type, rows_per_chunk=rows_per_chunk),
            provenance=provenance,
        )

    def finalize(self) -> Path:
        candidates = sum(len(values) for values in self._entries.values())
        index = {
            "schema": NATIVE_CANDIDATE_SCHEMA,
            "format": "native-ggml-candidates",
            "alignment": self.alignment,
            "source": self.source,
            "generator": {
                "library_path": str(self.codec.library_path),
                "library_sha256": self.codec.library_sha256,
            },
            "tensor_count": len(self._entries),
            "candidate_count": candidates,
            "tensors": self._entries,
        }
        path = self.root / NATIVE_CANDIDATE_INDEX
        _atomic_json(index, path)
        return path


class NativeCandidateStore:
    """Indexed native candidates with one-time integrity verification."""

    def __init__(self, root: str | os.PathLike, codec: GGMLNativeCodec):
        self.root = Path(root)
        self.codec = codec
        with (self.root / NATIVE_CANDIDATE_INDEX).open(encoding="utf-8") as handle:
            self.index = json.load(handle)
        if self.index.get("schema") != NATIVE_CANDIDATE_SCHEMA:
            raise ValueError(
                f"unsupported native candidate schema: {self.index.get('schema')}")
        self._verified: set[tuple[str, str]] = set()

    def metadata(
        self, tensor_name: str, ggml_type: GGMLType | int,
    ) -> dict[str, Any]:
        type_name = self.codec.type_name(ggml_type)
        try:
            return self.index["tensors"][tensor_name][type_name]
        except KeyError as error:
            raise KeyError(f"candidate not found: {tensor_name}/{type_name}") from error

    def _path(self, metadata: dict[str, Any]) -> Path:
        path = (self.root / metadata["path"]).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as error:
            raise ValueError(f"candidate path escapes store root: {path}") from error
        return path

    def _verify(self, tensor_name: str, type_name: str, metadata: dict[str, Any]) -> Path:
        path = self._path(metadata)
        key = (tensor_name, type_name)
        if key in self._verified:
            return path
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(8 << 20):
                digest.update(chunk)
                size += len(chunk)
        if size != metadata["payload_bytes"]:
            raise ValueError(f"candidate size mismatch at {path}: {size}")
        if digest.hexdigest() != metadata["sha256"]:
            raise ValueError(f"candidate checksum mismatch at {path}")
        self._verified.add(key)
        return path

    def decode_into(
        self,
        tensor_name: str,
        ggml_type: GGMLType | int,
        out: np.ndarray,
        *,
        rows_per_chunk: int = 16,
    ) -> np.ndarray:
        """Decode through a read-only mmap into a caller-provided array."""
        metadata = self.metadata(tensor_name, ggml_type)
        type_name = str(metadata["ggml_type_name"])
        path = self._verify(tensor_name, type_name, metadata)
        expected_shape = tuple(reversed(metadata["gguf_shape"]))
        if out.dtype != np.float32 or out.shape != expected_shape or not out.flags.c_contiguous:
            raise ValueError(
                f"out must be C-contiguous float32 with shape {expected_shape}")
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")
        rows = out.reshape(-1, metadata["row_width"])
        row_size = int(metadata["row_size"])
        with path.open("rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as payload:
                view = memoryview(payload)
                try:
                    for start in range(0, rows.shape[0], rows_per_chunk):
                        stop = min(start + rows_per_chunk, rows.shape[0])
                        packed = view[start * row_size:stop * row_size]
                        try:
                            self.codec.dequantize_rows_into(
                                packed, ggml_type, rows[start:stop])
                        finally:
                            packed.release()
                finally:
                    view.release()
        return out

    def iter_payload(
        self,
        tensor_name: str,
        ggml_type: GGMLType | int,
        *,
        chunk_bytes: int = 8 << 20,
    ) -> Iterator[bytes]:
        """Yield verified native bytes for a future streaming GGUF writer."""
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        metadata = self.metadata(tensor_name, ggml_type)
        type_name = str(metadata["ggml_type_name"])
        path = self._verify(tensor_name, type_name, metadata)
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_bytes):
                yield chunk
