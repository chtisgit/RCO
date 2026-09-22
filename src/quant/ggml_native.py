"""Thin, bounded bindings for native GGML row quantization.

The published candidate bytes must be the bytes llama.cpp executes.  This
module therefore calls the pinned GGML shared library directly instead of
reimplementing a quantizer in Python.  It intentionally exposes only the
row-oriented operations needed by the candidate store.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from enum import IntEnum
from pathlib import Path
from typing import Iterator

import numpy as np


class GGMLType(IntEnum):
    """Stable GGML type IDs from the pinned llama.cpp ``ggml.h``."""

    Q4_0 = 2
    Q3_K = 11
    Q4_K = 12
    Q2_0 = 42


_TYPE_NAMES = {
    GGMLType.Q4_0: "Q4_0",
    GGMLType.Q3_K: "Q3_K",
    GGMLType.Q4_K: "Q4_K",
    GGMLType.Q2_0: "Q2_0",
}


class _GGMLTypeTraits(ctypes.Structure):
    _fields_ = [
        ("type_name", ctypes.c_char_p),
        ("blck_size", ctypes.c_int64),
        ("blck_size_interleave", ctypes.c_int64),
        ("type_size", ctypes.c_size_t),
        ("is_quantized", ctypes.c_bool),
        ("to_float", ctypes.c_void_p),
        ("from_float_ref", ctypes.c_void_p),
    ]


_TO_FLOAT = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


class GGMLNativeCodec:
    """Quantize and dequantize row chunks through a specific GGML library."""

    def __init__(self, library_path: str | os.PathLike | None = None):
        configured = library_path or os.environ.get("RCO_GGML_LIBRARY")
        if not configured:
            raise ValueError(
                "GGML shared library is required; pass library_path or set "
                "RCO_GGML_LIBRARY"
            )
        self.library_path = Path(configured).expanduser().resolve(strict=True)
        self.library_sha256 = _sha256_file(self.library_path)
        self._library = ctypes.CDLL(str(self.library_path))
        self._configure_functions()

    def _configure_functions(self) -> None:
        self._library.ggml_blck_size.argtypes = [ctypes.c_int]
        self._library.ggml_blck_size.restype = ctypes.c_int64
        self._library.ggml_type_size.argtypes = [ctypes.c_int]
        self._library.ggml_type_size.restype = ctypes.c_size_t
        self._library.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        self._library.ggml_row_size.restype = ctypes.c_size_t
        self._library.ggml_type_name.argtypes = [ctypes.c_int]
        self._library.ggml_type_name.restype = ctypes.c_char_p
        self._library.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
        self._library.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        self._library.ggml_quantize_chunk.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        self._library.ggml_quantize_chunk.restype = ctypes.c_size_t
        self._library.ggml_get_type_traits.argtypes = [ctypes.c_int]
        self._library.ggml_get_type_traits.restype = ctypes.POINTER(
            _GGMLTypeTraits)

    @staticmethod
    def type_name(ggml_type: GGMLType | int) -> str:
        try:
            return _TYPE_NAMES[GGMLType(ggml_type)]
        except (KeyError, ValueError) as error:
            raise ValueError(f"unsupported native candidate type: {ggml_type}") from error

    def _type_id(self, ggml_type: GGMLType | int) -> int:
        value = GGMLType(ggml_type)
        expected = self.type_name(value).lower()
        actual_raw = self._library.ggml_type_name(int(value))
        if actual_raw is None:
            raise RuntimeError(f"GGML library has no type name for ID {int(value)}")
        actual = actual_raw.decode("ascii")
        if actual.lower() != expected.lower():
            raise RuntimeError(
                f"GGML type ID mismatch for {value.name}: library reports {actual}")
        return int(value)

    def geometry(self, ggml_type: GGMLType | int, row_width: int) -> dict[str, int | str]:
        type_id = self._type_id(ggml_type)
        block_size = int(self._library.ggml_blck_size(type_id))
        type_size = int(self._library.ggml_type_size(type_id))
        if block_size <= 0 or type_size <= 0:
            raise ValueError(f"GGML type {self.type_name(ggml_type)} has invalid geometry")
        if row_width <= 0 or row_width % block_size:
            raise ValueError(
                f"row width {row_width} is not divisible by "
                f"{self.type_name(ggml_type)} block size {block_size}"
            )
        row_size = int(self._library.ggml_row_size(type_id, row_width))
        return {
            "ggml_type": type_id,
            "ggml_type_name": self.type_name(ggml_type),
            "block_size": block_size,
            "type_size": type_size,
            "row_size": row_size,
        }

    def quantize_rows(
        self,
        rows: np.ndarray,
        ggml_type: GGMLType | int,
    ) -> bytes:
        """Return exact native bytes for a C-contiguous FP32 row chunk."""
        source = np.asarray(rows, dtype=np.float32, order="C")
        if source.ndim != 2:
            raise ValueError(f"expected a 2D row chunk, got shape {source.shape}")
        nrows, row_width = source.shape
        if nrows <= 0:
            raise ValueError("cannot quantize an empty row chunk")
        geometry = self.geometry(ggml_type, row_width)
        type_id = int(geometry["ggml_type"])
        if self._library.ggml_quantize_requires_imatrix(type_id):
            raise ValueError(
                f"{geometry['ggml_type_name']} requires an importance matrix"
            )
        expected = nrows * int(geometry["row_size"])
        destination = np.empty(expected, dtype=np.uint8)
        written = int(self._library.ggml_quantize_chunk(
            type_id,
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_void_p(destination.ctypes.data),
            0,
            nrows,
            row_width,
            None,
        ))
        if written != expected:
            raise RuntimeError(
                f"GGML wrote {written} bytes for {nrows} rows; expected {expected}"
            )
        return destination.tobytes()

    def dequantize_rows_into(
        self,
        payload: bytes | bytearray | memoryview | np.ndarray,
        ggml_type: GGMLType | int,
        out: np.ndarray,
    ) -> np.ndarray:
        """Decode packed rows directly into a caller-owned FP32 array."""
        if not isinstance(out, np.ndarray):
            raise TypeError("out must be a NumPy array")
        if out.dtype != np.float32 or out.ndim != 2 or not out.flags.c_contiguous:
            raise ValueError("out must be a C-contiguous 2D float32 array")
        nrows, row_width = out.shape
        geometry = self.geometry(ggml_type, row_width)
        row_size = int(geometry["row_size"])
        packed = np.frombuffer(payload, dtype=np.uint8)
        expected = nrows * row_size
        if packed.nbytes != expected:
            raise ValueError(
                f"packed payload has {packed.nbytes} bytes; expected {expected}"
            )
        traits = self._library.ggml_get_type_traits(
            int(geometry["ggml_type"])).contents
        if not traits.to_float:
            raise ValueError(f"{geometry['ggml_type_name']} has no dequantizer")
        to_float = _TO_FLOAT(traits.to_float)
        for row in range(nrows):
            to_float(
                ctypes.c_void_p(packed.ctypes.data + row * row_size),
                out[row].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                row_width,
            )
        return out

    def iter_quantized_rows(
        self,
        rows: np.ndarray,
        ggml_type: GGMLType | int,
        *,
        rows_per_chunk: int,
    ) -> Iterator[bytes]:
        """Quantize an array in independently bounded row chunks."""
        source = np.asarray(rows)
        if source.ndim < 2:
            raise ValueError(f"expected at least two dimensions, got {source.shape}")
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")
        flattened = source.reshape(-1, source.shape[-1])
        for start in range(0, flattened.shape[0], rows_per_chunk):
            yield self.quantize_rows(
                flattened[start:start + rows_per_chunk], ggml_type)

