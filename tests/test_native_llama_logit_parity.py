import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools" / "audit_qwen36_native_llama_logit_parity.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_qwen36_native_llama_logit_parity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class NativeLlamaLogitParityTest(unittest.TestCase):
    def test_reads_versioned_logit_dump(self):
        tokens = np.asarray([3, 4, 5], dtype="<i4")
        logits = np.arange(14, dtype="<f4").reshape(2, 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logits.bin"
            with path.open("wb") as handle:
                handle.write(b"RCOLOG1\0")
                handle.write(struct.pack("<III", 1, 2, 7))
                handle.write(tokens.tobytes())
                handle.write(logits.tobytes())
            actual_tokens, actual_logits = MODULE._read_llama_logits(path)
        np.testing.assert_array_equal(actual_tokens, tokens)
        np.testing.assert_array_equal(actual_logits, logits)

    def test_comparison_passes_identity_and_rejects_large_shift(self):
        logits = np.asarray([
            [0.0, 0.5, 2.0, -1.0],
            [1.0, -0.5, 0.25, 3.0],
        ], dtype=np.float32)
        positions, passed = MODULE._comparison(logits, logits.copy(), [0, 2, 3])
        self.assertTrue(passed)
        self.assertTrue(all(item["passed"] for item in positions))

        shifted = logits.copy()
        shifted[0, 0] += 2.0
        positions, passed = MODULE._comparison(logits, shifted, [0, 2, 3])
        self.assertFalse(passed)
        self.assertFalse(positions[0]["passed"])

    def test_reads_and_compares_layer_dump(self):
        layers = np.arange(24, dtype="<f4").reshape(2, 3, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layers.bin"
            with path.open("wb") as handle:
                handle.write(b"RCOLAY1\0")
                handle.write(struct.pack("<IIII", 1, 2, 3, 4))
                for index, values in enumerate(layers):
                    handle.write(struct.pack("<I", index))
                    handle.write(values.tobytes())
            actual = MODULE._read_llama_layers(path)
        np.testing.assert_array_equal(actual, layers)
        results, first_failed = MODULE._layer_comparison(layers, layers.copy())
        self.assertIsNone(first_failed)
        self.assertTrue(all(item["passed"] for item in results))

        shifted = layers.copy()
        shifted[1] += 10.0
        results, first_failed = MODULE._layer_comparison(layers, shifted)
        self.assertEqual(first_failed, 0)
        self.assertFalse(results[1]["passed"])


if __name__ == "__main__":
    unittest.main()
