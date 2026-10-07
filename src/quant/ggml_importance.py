"""Importance-weighted (imatrix) native GGML row quantization.

``GGMLNativeCodec.quantize_rows`` always passes a null importance matrix.
This module calls the same pinned ``ggml_quantize_chunk`` with a per-column
importance vector, which is how llama.cpp applies an imatrix: every row in
the chunk shares one weight per input column.  It lives in a separate file so
the provenance-hashed codec stays byte-identical.
"""

from __future__ import annotations

import ctypes

import numpy as np

from quant.ggml_native import GGMLNativeCodec, GGMLType


def quantize_rows_with_importance(
    codec: GGMLNativeCodec,
    rows: np.ndarray,
    ggml_type: GGMLType | int,
    importance: np.ndarray,
) -> bytes:
    """Return native bytes for ``rows`` quantized with column importances."""
    source = np.asarray(rows, dtype=np.float32, order="C")
    if source.ndim != 2:
        raise ValueError(f"expected a 2D row chunk, got shape {source.shape}")
    nrows, row_width = source.shape
    if nrows <= 0:
        raise ValueError("cannot quantize an empty row chunk")
    weights = np.ascontiguousarray(importance, dtype=np.float32)
    if weights.shape != (row_width,):
        raise ValueError(
            f"importance has shape {weights.shape}; expected ({row_width},)")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("importance must be finite and nonnegative")
    if not np.any(weights > 0):
        raise ValueError("importance must have a positive entry")
    geometry = codec.geometry(ggml_type, row_width)
    expected = nrows * int(geometry["row_size"])
    destination = np.empty(expected, dtype=np.uint8)
    written = int(codec._library.ggml_quantize_chunk(
        int(geometry["ggml_type"]),
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_void_p(destination.ctypes.data),
        0,
        nrows,
        row_width,
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    ))
    if written != expected:
        raise RuntimeError(
            f"GGML wrote {written} bytes for {nrows} rows; expected {expected}")
    return destination.tobytes()


__all__ = ["quantize_rows_with_importance"]
