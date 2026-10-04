"""Atomic store for exact native GGML candidate payloads."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import tempfile
from copy import deepcopy
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
        resume: bool = False,
    ):
        self.root = Path(root)
        self.codec = codec
        self.source = dict(source)
        self.alignment = alignment
        _align_up(0, alignment)
        self.root.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, dict[str, Any]] = {}
        self.resumed_candidate_count = 0
        self.reused_candidate_count = 0
        self.written_candidate_count = 0
        index_path = self.root / NATIVE_CANDIDATE_INDEX
        if index_path.exists():
            if not resume:
                raise FileExistsError(
                    f"candidate store index already exists: {index_path}")
            with index_path.open(encoding="utf-8") as handle:
                index = json.load(handle)
            expected_generator = {
                "library_path": str(self.codec.library_path),
                "library_sha256": self.codec.library_sha256,
            }
            if index.get("schema") != NATIVE_CANDIDATE_SCHEMA:
                raise ValueError("resumed candidate store schema differs")
            if index.get("format") != "native-ggml-candidates":
                raise ValueError("resumed candidate store format differs")
            if int(index.get("alignment", -1)) != self.alignment:
                raise ValueError("resumed candidate store alignment differs")
            if index.get("source") != self.source:
                raise ValueError("resumed candidate store source differs")
            if index.get("generator") != expected_generator:
                raise ValueError("resumed candidate store generator differs")
            entries = index.get("tensors")
            if not isinstance(entries, dict):
                raise ValueError("resumed candidate store has no tensor index")
            self._entries = entries
            self.resumed_candidate_count = sum(
                len(candidates) for candidates in entries.values())

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
        existing = self._entries.get(tensor_name, {}).get(type_name)
        if existing is not None:
            expected = {
                "gguf_shape": list(shape),
                "row_width": row_width,
                "row_count": row_count,
                "payload_bytes": payload_bytes,
                "aligned_gguf_bytes": _align_up(payload_bytes, self.alignment),
                "alignment": self.alignment,
                "byte_order": "little-endian-native-ggml",
                "provenance": dict(provenance),
                "path": destination.relative_to(self.root).as_posix(),
            }
            for field, value in expected.items():
                if existing.get(field) != value:
                    raise ValueError(
                        f"resumed candidate {tensor_name}/{type_name} has "
                        f"different {field}")
            digest = hashlib.sha256()
            size = 0
            with destination.open("rb") as handle:
                while chunk := handle.read(8 << 20):
                    digest.update(chunk)
                    size += len(chunk)
            if size != payload_bytes:
                raise ValueError(
                    f"resumed candidate size mismatch at {destination}")
            if digest.hexdigest() != existing.get("sha256"):
                raise ValueError(
                    f"resumed candidate checksum mismatch at {destination}")
            self.reused_candidate_count += 1
            return dict(existing)

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
        self.written_candidate_count += 1
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
        expected_shape = tuple(reversed(metadata["gguf_shape"]))
        if out.dtype != np.float32 or out.shape != expected_shape or not out.flags.c_contiguous:
            raise ValueError(
                f"out must be C-contiguous float32 with shape {expected_shape}")
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")
        rows = out.reshape(-1, metadata["row_width"])
        for start, decoded in self.iter_decoded_rows(
            tensor_name, ggml_type, rows_per_chunk=rows_per_chunk,
        ):
            np.copyto(rows[start:start + decoded.shape[0]], decoded)
        return out

    def iter_decoded_rows(
        self,
        tensor_name: str,
        ggml_type: GGMLType | int,
        *,
        rows_per_chunk: int = 16,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield bounded decoded FP32 row chunks and their flat row offsets."""
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")
        metadata = self.metadata(tensor_name, ggml_type)
        type_name = str(metadata["ggml_type_name"])
        path = self._verify(tensor_name, type_name, metadata)
        row_width = int(metadata["row_width"])
        row_count = int(metadata["row_count"])
        row_size = int(metadata["row_size"])
        with path.open("rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as payload:
                view = memoryview(payload)
                try:
                    for start in range(0, row_count, rows_per_chunk):
                        stop = min(start + rows_per_chunk, row_count)
                        packed = view[start * row_size:stop * row_size]
                        try:
                            decoded = np.empty(
                                (stop - start, row_width), dtype=np.float32)
                            self.codec.dequantize_rows_into(
                                packed, ggml_type, decoded)
                        finally:
                            packed.release()
                        yield start, decoded
                finally:
                    view.release()

    def iter_decoded_row_indices(
        self,
        tensor_name: str,
        ggml_type: GGMLType | int,
        row_indices: Iterable[int],
        *,
        rows_per_chunk: int = 16,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield decoded rows in a caller-selected order with bounded scratch.

        Native GGML rows have fixed byte offsets.  This permits inverse
        converter permutations to be applied without decoding a complete
        matrix or retaining a full reordered candidate.
        """
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")
        metadata = self.metadata(tensor_name, ggml_type)
        type_name = str(metadata["ggml_type_name"])
        path = self._verify(tensor_name, type_name, metadata)
        row_width = int(metadata["row_width"])
        row_count = int(metadata["row_count"])
        row_size = int(metadata["row_size"])
        indices = np.asarray(tuple(int(value) for value in row_indices), dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("row_indices must be a non-empty vector")
        if np.any(indices < 0) or np.any(indices >= row_count):
            raise IndexError("row index is outside the candidate tensor")

        with path.open("rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as payload:
                view = memoryview(payload)
                try:
                    for output_start in range(0, len(indices), rows_per_chunk):
                        selected = indices[
                            output_start:output_start + rows_per_chunk]
                        decoded = np.empty(
                            (len(selected), row_width), dtype=np.float32)
                        run_output_start = 0
                        while run_output_start < len(selected):
                            run_output_stop = run_output_start + 1
                            while (
                                run_output_stop < len(selected)
                                and selected[run_output_stop]
                                == selected[run_output_stop - 1] + 1
                            ):
                                run_output_stop += 1
                            first_row = int(selected[run_output_start])
                            last_row = int(selected[run_output_stop - 1]) + 1
                            packed = view[
                                first_row * row_size:last_row * row_size]
                            try:
                                self.codec.dequantize_rows_into(
                                    packed,
                                    ggml_type,
                                    decoded[run_output_start:run_output_stop],
                                )
                            finally:
                                packed.release()
                            run_output_start = run_output_stop
                        yield output_start, decoded
                finally:
                    view.release()

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


class NativeCandidateOverlayStore:
    """Read candidates from a base store with exact per-type overrides.

    The merged in-memory index lets existing manifest adapters see the complete
    tensor inventory, while payload access remains delegated to the immutable
    source store that owns each candidate.
    """

    def __init__(
        self,
        base: NativeCandidateStore,
        *overlays: NativeCandidateStore,
    ) -> None:
        self.base = base
        self.overlays = tuple(overlays)
        self.codec = base.codec
        stores = (base, *self.overlays)
        for store in stores:
            if store.codec.library_sha256 != self.codec.library_sha256:
                raise ValueError("overlay stores use different GGML libraries")
        tensors = deepcopy(base.index["tensors"])
        self._owners: dict[tuple[str, str], NativeCandidateStore] = {
            (name, type_name): base
            for name, candidates in base.index["tensors"].items()
            for type_name in candidates
        }
        for overlay in self.overlays:
            for name, candidates in overlay.index["tensors"].items():
                for type_name, metadata in candidates.items():
                    tensors.setdefault(name, {})[type_name] = deepcopy(metadata)
                    self._owners[(name, type_name)] = overlay
        self.index = {
            **deepcopy(base.index),
            "source": {
                "kind": "native_candidate_overlay",
                "base": deepcopy(base.index.get("source")),
                "overlays": [
                    deepcopy(store.index.get("source")) for store in self.overlays
                ],
            },
            "tensor_count": len(tensors),
            "candidate_count": sum(len(values) for values in tensors.values()),
            "tensors": tensors,
        }

    def _owner(
        self, tensor_name: str, ggml_type: GGMLType | int,
    ) -> NativeCandidateStore:
        type_name = self.codec.type_name(ggml_type)
        try:
            return self._owners[(tensor_name, type_name)]
        except KeyError as error:
            raise KeyError(
                f"candidate not found: {tensor_name}/{type_name}") from error

    def metadata(
        self, tensor_name: str, ggml_type: GGMLType | int,
    ) -> dict[str, Any]:
        return self._owner(tensor_name, ggml_type).metadata(tensor_name, ggml_type)

    def iter_decoded_rows(self, tensor_name, ggml_type, **kwargs):
        return self._owner(tensor_name, ggml_type).iter_decoded_rows(
            tensor_name, ggml_type, **kwargs)

    def iter_decoded_row_indices(self, tensor_name, ggml_type, row_indices, **kwargs):
        return self._owner(tensor_name, ggml_type).iter_decoded_row_indices(
            tensor_name, ggml_type, row_indices, **kwargs)

    def iter_payload(self, tensor_name, ggml_type, **kwargs):
        return self._owner(tensor_name, ggml_type).iter_payload(
            tensor_name, ggml_type, **kwargs)

    def decode_into(self, tensor_name, ggml_type, out, **kwargs):
        return self._owner(tensor_name, ggml_type).decode_into(
            tensor_name, ggml_type, out, **kwargs)
