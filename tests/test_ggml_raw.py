import os
import sys
from pathlib import Path
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GGML_LIBRARY = os.environ.get("RCO_GGML_LIBRARY")


@unittest.skipUnless(GGML_LIBRARY, "RCO_GGML_LIBRARY is not configured")
class RawQ6KTest(unittest.TestCase):
    def setUp(self):
        from quant.ggml_native import GGMLNativeCodec

        self.codec = GGMLNativeCodec(GGML_LIBRARY)

    def test_round_trip_geometry_and_error(self):
        from quant.ggml_raw import (
            GGML_TYPE_Q6_K, dequantize_rows_raw_into, quantize_rows_raw,
            raw_row_size)

        rows = np.random.default_rng(5).standard_normal((3, 512)).astype(np.float32)
        self.assertEqual(raw_row_size(self.codec, GGML_TYPE_Q6_K, 512), 420)
        payload = quantize_rows_raw(self.codec, rows, GGML_TYPE_Q6_K)
        self.assertEqual(len(payload), 3 * 420)
        decoded = dequantize_rows_raw_into(
            self.codec, payload, GGML_TYPE_Q6_K, np.empty_like(rows))
        relative = np.sqrt(np.square(decoded - rows).sum() / np.square(rows).sum())
        self.assertLess(relative, 0.03)

    def test_rejects_unknown_type_and_bad_width(self):
        from quant.ggml_raw import GGML_TYPE_Q6_K, raw_row_size

        with self.assertRaises(ValueError):
            raw_row_size(self.codec, 2, 512)
        with self.assertRaises(ValueError):
            raw_row_size(self.codec, GGML_TYPE_Q6_K, 300)


if __name__ == "__main__":
    unittest.main()
