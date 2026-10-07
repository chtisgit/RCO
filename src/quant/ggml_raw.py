"""Native GGML rows for types outside the provenance-hashed codec enum.

``GGMLNativeCodec`` only admits the candidate types of the original search.
The v2 plan also writes a Q6_K ``token_embd``.  These helpers call the same
pinned library by raw type ID and check the library's own type name, so the
hashed codec file stays byte-identical.
"""

from __future__ import annotations

import ctypes

import numpy as np

from quant.ggml_native import _TO_FLOAT, GGMLNativeCodec


GGML_TYPE_Q6_K = 14
_EXPECTED_NAMES = {GGML_TYPE_Q6_K: "q6_K"}


def raw_row_size(codec: GGMLNativeCodec, type_id: int, row_width: int) -> int:
    """Return the packed row size after checking the library's type name."""
    expected = _EXPECTED_NAMES.get(int(type_id))
    if expected is None:
        raise ValueError(f"unsupported raw GGML type: {type_id}")
    actual = codec._library.ggml_type_name(int(type_id))
    if actual is None or actual.decode("ascii") != expected:
        raise RuntimeError(f"GGML type ID {type_id} is not {expected}")
    block = int(codec._library.ggml_blck_size(int(type_id)))
    if row_width <= 0 or row_width % block:
        raise ValueError(f"row width {row_width} is not divisible by {block}")
    return int(codec._library.ggml_row_size(int(type_id), row_width))


def quantize_rows_raw(codec: GGMLNativeCodec, rows: np.ndarray, type_id: int) -> bytes:
    """Quantize a C-contiguous FP32 row chunk without an importance matrix."""
    source = np.asarray(rows, dtype=np.float32, order="C")
    if source.ndim != 2 or source.shape[0] <= 0:
        raise ValueError(f"expected a non-empty 2D row chunk, got {source.shape}")
    nrows, row_width = source.shape
    if codec._library.ggml_quantize_requires_imatrix(int(type_id)):
        raise ValueError(f"GGML type {type_id} requires an importance matrix")
    expected = nrows * raw_row_size(codec, type_id, row_width)
    destination = np.empty(expected, dtype=np.uint8)
    written = int(codec._library.ggml_quantize_chunk(
        int(type_id),
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_void_p(destination.ctypes.data),
        0,
        nrows,
        row_width,
        None,
    ))
    if written != expected:
        raise RuntimeError(
            f"GGML wrote {written} bytes for {nrows} rows; expected {expected}")
    return destination.tobytes()


def dequantize_rows_raw_into(
    codec: GGMLNativeCodec,
    payload: bytes | bytearray | memoryview | np.ndarray,
    type_id: int,
    out: np.ndarray,
) -> np.ndarray:
    """Decode packed rows into a caller-owned C-contiguous FP32 array."""
    if out.dtype != np.float32 or out.ndim != 2 or not out.flags.c_contiguous:
        raise ValueError("out must be a C-contiguous 2D float32 array")
    nrows, row_width = out.shape
    row_size = raw_row_size(codec, type_id, row_width)
    packed = np.frombuffer(payload, dtype=np.uint8)
    if packed.nbytes != nrows * row_size:
        raise ValueError(
            f"packed payload has {packed.nbytes} bytes; expected {nrows * row_size}")
    traits = codec._library.ggml_get_type_traits(int(type_id)).contents
    if not traits.to_float:
        raise ValueError(f"GGML type {type_id} has no dequantizer")
    to_float = _TO_FLOAT(traits.to_float)
    for row in range(nrows):
        to_float(
            ctypes.c_void_p(packed.ctypes.data + row * row_size),
            out[row].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            row_width,
        )
    return out


__all__ = [
    "GGML_TYPE_Q6_K",
    "dequantize_rows_raw_into",
    "quantize_rows_raw",
    "raw_row_size",
]
