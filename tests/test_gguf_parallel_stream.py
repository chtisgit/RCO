import os
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GGML_LIBRARY = os.environ.get("RCO_GGML_LIBRARY")


@unittest.skipUnless(GGML_LIBRARY, "RCO_GGML_LIBRARY is not configured")
class ParallelQ2Test(unittest.TestCase):
    def setUp(self):
        from quant.ggml_native import GGMLNativeCodec, GGMLType

        self.codec = GGMLNativeCodec(GGML_LIBRARY)
        rows = np.random.default_rng(3).standard_normal((37, 256)).astype(np.float32)
        payload = self.codec.quantize_rows(rows, GGMLType.Q2_0)
        self.packed = np.frombuffer(payload, dtype=np.uint8).reshape(37, -1)

    def test_one_call_chunk_matches_per_row_decoder(self):
        from gguf_parallel_stream import decode_native_chunk
        from quant.ggml_native import GGMLType

        expected = self.codec.dequantize_rows_into(
            self.packed, GGMLType.Q2_0, np.empty((37, 256), dtype=np.float32))
        got = decode_native_chunk(self.codec, "Q2_0", self.packed, np.empty_like(expected))
        np.testing.assert_array_equal(got, expected)

    def test_q8_0_is_scale_times_int8(self):
        from gguf_parallel_stream import decode_native_chunk

        rng = np.random.default_rng(4)
        scales = rng.uniform(0.01, 0.1, size=(3, 2)).astype(np.float16)
        quants = rng.integers(-128, 128, size=(3, 2, 32), dtype=np.int8)
        packed = np.concatenate(
            [scales.view(np.uint8).reshape(3, 2, 2), quants.view(np.uint8)],
            axis=2).reshape(3, 68)
        got = decode_native_chunk(
            self.codec, "Q8_0", packed, np.empty((3, 64), dtype=np.float32))
        expected = (scales.astype(np.float32)[..., None]
                    * quants.astype(np.float32)).reshape(3, 64)
        np.testing.assert_array_equal(got, expected)

    def test_threaded_rows_match_base_loader_in_order(self):
        from gguf_checkpoint_stream import GGUFManifestPrefixLoader
        from gguf_parallel_stream import ParallelGGUFManifestPrefixLoader

        entry = {"destination_name": "t", "candidate_source_shape": [37, 256]}
        tensor = SimpleNamespace(
            name="t", tensor_type=SimpleNamespace(name="Q2_0"), data=self.packed)
        results = []
        for cls, extra in ((GGUFManifestPrefixLoader, {}),
                           (ParallelGGUFManifestPrefixLoader, {"workers": 3})):
            loader = object.__new__(cls)
            loader.tensors = {"t": tensor}
            loader.native_codec = self.codec
            loader.rows_per_chunk = 5
            loader.max_decoded_chunk_bytes = 0
            if extra:
                from concurrent.futures import ThreadPoolExecutor

                loader.workers = extra["workers"]
                loader._executor = ThreadPoolExecutor(extra["workers"])
            results.append(list(loader._decoded_rows(entry)))
            if extra:
                loader.close()
        base, parallel = results
        self.assertEqual([start for start, _ in parallel], list(range(0, 37, 5)))
        self.assertEqual([start for start, _ in base], [start for start, _ in parallel])
        for (_, want), (_, got) in zip(base, parallel):
            np.testing.assert_array_equal(got, want)


if __name__ == "__main__":
    unittest.main()
