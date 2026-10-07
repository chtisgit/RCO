"""Faster decoding of quantized tensors for streamed GGUF prefixes.

``GGUFManifestPrefixLoader`` decodes Q2_0 with one ctypes call per row,
sequentially, and decodes Q8_0 with gguf-py's NumPy code.  The streamed
pruning search loads every block twice per step, so decoding dominates its
run time.  This subclass changes two things for Q2_0 and Q8_0, and the
decoded values stay bit-identical:

* The pinned ggml library decodes a whole chunk in one call.  Packed rows
  are contiguous, the decoded rows are contiguous, and the row width is a
  multiple of the block size.  So one call over ``rows * width`` values
  processes exactly the same blocks as per-row calls.  Q8_0 decoding is
  ``fp16 scale * int8`` in both implementations.
* Chunks are decoded on a thread pool, because ctypes releases the GIL during
  foreign calls.  Chunks are still yielded in order.

Other types (F32, BF16) keep the base path.  The base loader lives in a
provenance-hashed file, which is why this is a subclass.
"""

from __future__ import annotations

import ctypes
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from gguf_checkpoint_stream import GGUFManifestPrefixLoader
from quant.ggml_native import _TO_FLOAT, GGMLNativeCodec


NATIVE_TYPES = {"Q2_0": 42, "Q8_0": 8}


def decode_native_chunk(
    codec: GGMLNativeCodec, type_name: str, packed: np.ndarray, out: np.ndarray,
) -> np.ndarray:
    """Decode contiguous packed rows of one native type with one library call."""
    if out.dtype != np.float32 or out.ndim != 2 or not out.flags.c_contiguous:
        raise ValueError("out must be a C-contiguous 2D float32 array")
    type_id = NATIVE_TYPES[type_name]
    library = codec._library
    actual = library.ggml_type_name(type_id)
    if actual is None or actual.decode("ascii") != type_name.lower():
        raise RuntimeError(f"GGML type ID {type_id} is not {type_name}")
    rows, width = out.shape
    if width % int(library.ggml_blck_size(type_id)):
        raise ValueError(f"row width {width} is not a whole number of blocks")
    packed = np.ascontiguousarray(packed).view(np.uint8).reshape(-1)
    if packed.nbytes != rows * int(library.ggml_row_size(type_id, width)):
        raise ValueError(f"packed {type_name} chunk has the wrong size")
    traits = library.ggml_get_type_traits(type_id).contents
    _TO_FLOAT(traits.to_float)(
        ctypes.c_void_p(packed.ctypes.data),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rows * width,
    )
    return out


class ParallelGGUFManifestPrefixLoader(GGUFManifestPrefixLoader):
    """``GGUFManifestPrefixLoader`` with chunked, threaded native decoding."""

    def __init__(self, *args: Any, workers: int = 8, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers = int(workers)
        self._executor = ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix="gguf-decode")

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def _decoded_rows(self, entry: dict[str, Any]):
        tensor = self.tensors[entry["destination_name"]]
        type_name = tensor.tensor_type.name
        if type_name not in NATIVE_TYPES:
            yield from super()._decoded_rows(entry)
            return
        data = np.asarray(tensor.data)
        packed_rows = data.reshape(-1, data.shape[-1])
        expected_rows = int(np.prod(
            entry["candidate_source_shape"][:-1], dtype=np.int64))
        if len(packed_rows) != expected_rows:
            raise ValueError(
                f"GGUF row count differs for {tensor.name}: "
                f"{len(packed_rows)} != {expected_rows}")
        width = int(entry["candidate_source_shape"][-1])

        def decode(start: int) -> tuple[int, np.ndarray]:
            stop = min(start + self.rows_per_chunk, expected_rows)
            decoded = np.empty((stop - start, width), dtype=np.float32)
            decode_native_chunk(
                self.native_codec, type_name, packed_rows[start:stop], decoded)
            return start, decoded

        starts = iter(range(0, expected_rows, self.rows_per_chunk))
        pending: deque = deque()
        for start in starts:
            pending.append(self._executor.submit(decode, start))
            if len(pending) >= 2 * self.workers:
                break
        while pending:
            start, decoded = pending.popleft().result()
            following = next(starts, None)
            if following is not None:
                pending.append(self._executor.submit(decode, following))
            self.max_decoded_chunk_bytes = max(
                self.max_decoded_chunk_bytes, decoded.nbytes)
            yield start, decoded


__all__ = ["NATIVE_TYPES", "ParallelGGUFManifestPrefixLoader", "decode_native_chunk"]
