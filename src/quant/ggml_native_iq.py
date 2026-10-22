"""``GGMLNativeCodec`` with llama.cpp's I-quant types.

The provenance-hashed codec in ``ggml_native.py`` accepts only Q4_0, Q3_K,
Q4_K and Q2_0.  This subclass also accepts IQ2_XXS, IQ2_XS, IQ3_XXS and
IQ2_S, checks each type id against the library's own type name, and calls
``ggml_quantize_init`` for them once: the I-quant quantizers need their grid
tables, and without the call the pinned library aborts inside
``ggml_quantize_chunk``.  These types require an importance matrix, so they
are quantized through ``quant.ggml_importance.quantize_rows_with_importance``.
"""

from __future__ import annotations

import ctypes

from quant.ggml_native import GGMLNativeCodec

IQ_TYPES = {16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 22: "IQ2_S"}
IQ_TYPE_IDS = {name: type_id for type_id, name in IQ_TYPES.items()}


class IQGGMLNativeCodec(GGMLNativeCodec):
    def __init__(self, library_path=None):
        super().__init__(library_path)
        self._library.ggml_quantize_init.argtypes = [ctypes.c_int]
        self._library.ggml_quantize_init.restype = None
        for type_id in IQ_TYPES:
            self._type_id(type_id)
            self._library.ggml_quantize_init(type_id)

    @staticmethod
    def type_name(ggml_type) -> str:
        if int(ggml_type) in IQ_TYPES:
            return IQ_TYPES[int(ggml_type)]
        return GGMLNativeCodec.type_name(ggml_type)

    def _type_id(self, ggml_type) -> int:
        if int(ggml_type) not in IQ_TYPES:
            return super()._type_id(ggml_type)
        actual_raw = self._library.ggml_type_name(int(ggml_type))
        if actual_raw is None or actual_raw.decode("ascii").lower() != IQ_TYPES[
                int(ggml_type)].lower():
            raise RuntimeError(f"GGML type ID mismatch for {IQ_TYPES[int(ggml_type)]}: "
                               f"library reports {actual_raw!r}")
        return int(ggml_type)


__all__ = ["IQGGMLNativeCodec", "IQ_TYPES", "IQ_TYPE_IDS"]
