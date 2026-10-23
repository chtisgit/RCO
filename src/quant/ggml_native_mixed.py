"""``IQGGMLNativeCodec`` with Q8_0 and Q6_K as well (RCO_PLAN_V3.md).

The v3 search stores Q8_0 (copied from the GSQ GGUF) and Q6_K candidates
next to Q4_K and the I-quants.  The provenance-hashed codec accepts only
Q4_0, Q3_K, Q4_K and Q2_0, and ``IQGGMLNativeCodec`` adds the I-quants; this
subclass adds Q8_0 and Q6_K, checking each type id against the library's own
type name.  Quantizing, decoding and the geometry are the base class's.
"""

from __future__ import annotations

from quant.ggml_native_iq import IQ_TYPE_IDS, IQGGMLNativeCodec

EXTRA_TYPES = {8: "Q8_0", 14: "Q6_K"}
MIXED_TYPE_IDS = {**IQ_TYPE_IDS, "Q8_0": 8, "Q6_K": 14, "Q4_K": 12, "Q2_0": 42}


class MixedGGMLNativeCodec(IQGGMLNativeCodec):
    def __init__(self, library_path=None):
        super().__init__(library_path)
        for type_id in EXTRA_TYPES:
            self._type_id(type_id)

    @staticmethod
    def type_name(ggml_type) -> str:
        if int(ggml_type) in EXTRA_TYPES:
            return EXTRA_TYPES[int(ggml_type)]
        return IQGGMLNativeCodec.type_name(ggml_type)

    def _type_id(self, ggml_type) -> int:
        if int(ggml_type) not in EXTRA_TYPES:
            return super()._type_id(ggml_type)
        actual_raw = self._library.ggml_type_name(int(ggml_type))
        expected = EXTRA_TYPES[int(ggml_type)]
        if actual_raw is None or actual_raw.decode("ascii").lower() != expected.lower():
            raise RuntimeError(f"GGML type ID mismatch for {expected}: "
                               f"library reports {actual_raw!r}")
        return int(ggml_type)


__all__ = ["EXTRA_TYPES", "MIXED_TYPE_IDS", "MixedGGMLNativeCodec"]
