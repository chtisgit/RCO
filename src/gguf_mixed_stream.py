"""Streamed GGUF prefixes with the v3 search's types (RCO_PLAN_V3.md).

``ParallelGGUFManifestPrefixLoader`` decodes Q2_0 and Q8_0 with the pinned
ggml library and leaves every other type to gguf-py's NumPy dequantizers.
The v3 center GGUF also holds Q4_K, Q6_K, IQ2_XS and IQ3_XXS tensors, and its
loaded weights must equal what the candidate stores install, which decode
through ggml.  So this subclass decodes those types with ggml's ``to_float``
too, in whole chunks on the thread pool, exactly as the parent does Q2_0 and
Q8_0 (blocks are independent, so a chunk call equals per-row calls).  F32 and
BF16 keep the base path.
"""

from __future__ import annotations

import ctypes
from collections import deque
from typing import Any

import numpy as np

from gguf_parallel_stream import ParallelGGUFManifestPrefixLoader
from quant.ggml_native import _TO_FLOAT

MIXED_STREAM_TYPES = {"Q2_0": 42, "Q8_0": 8, "Q4_K": 12, "Q6_K": 14,
                      "IQ2_XS": 17, "IQ3_XXS": 18}


def decode_chunk(library, type_name: str, packed: np.ndarray, out: np.ndarray) -> np.ndarray:
    if out.dtype != np.float32 or out.ndim != 2 or not out.flags.c_contiguous:
        raise ValueError("out must be a C-contiguous 2D float32 array")
    type_id = MIXED_STREAM_TYPES[type_name]
    actual = library.ggml_type_name(type_id)
    if actual is None or actual.decode("ascii").lower() != type_name.lower():
        raise RuntimeError(f"GGML type ID {type_id} is not {type_name}")
    rows, width = out.shape
    if width % int(library.ggml_blck_size(type_id)):
        raise ValueError(f"row width {width} is not a whole number of blocks")
    packed = np.ascontiguousarray(packed).view(np.uint8).reshape(-1)
    if packed.nbytes != rows * int(library.ggml_row_size(type_id, width)):
        raise ValueError(f"packed {type_name} chunk has the wrong size")
    traits = library.ggml_get_type_traits(type_id).contents
    _TO_FLOAT(traits.to_float)(ctypes.c_void_p(packed.ctypes.data),
                               out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                               rows * width)
    return out


class MixedGGUFManifestPrefixLoader(ParallelGGUFManifestPrefixLoader):
    def _decoded_rows(self, entry: dict[str, Any]):
        tensor = self.tensors[entry["destination_name"]]
        type_name = tensor.tensor_type.name
        if type_name not in MIXED_STREAM_TYPES:
            yield from super()._decoded_rows(entry)
            return
        data = np.asarray(tensor.data)
        packed_rows = data.reshape(-1, data.shape[-1])
        expected_rows = int(np.prod(entry["candidate_source_shape"][:-1], dtype=np.int64))
        if len(packed_rows) != expected_rows:
            raise ValueError(f"GGUF row count differs for {tensor.name}: "
                             f"{len(packed_rows)} != {expected_rows}")
        width = int(entry["candidate_source_shape"][-1])
        library = self.native_codec._library

        def decode(start: int) -> tuple[int, np.ndarray]:
            stop = min(start + self.rows_per_chunk, expected_rows)
            decoded = np.empty((stop - start, width), dtype=np.float32)
            decode_chunk(library, type_name, packed_rows[start:stop], decoded)
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
            self.max_decoded_chunk_bytes = max(self.max_decoded_chunk_bytes, decoded.nbytes)
            yield start, decoded


__all__ = ["MIXED_STREAM_TYPES", "MixedGGUFManifestPrefixLoader", "decode_chunk"]
