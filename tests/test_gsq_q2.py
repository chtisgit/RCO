import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gsq_q2 import repack_gsq2_to_q2_0


class GSQQ2Test(unittest.TestCase):
    def test_repack_reverses_integer_lanes_and_duplicates_negative_scale(self):
        codes = np.tile(
            np.array([0, 1, 2, 3], dtype=np.uint8), 32).reshape(1, 128)
        scales = np.array([[0.5]], dtype=np.float32)
        payload = np.frombuffer(
            repack_gsq2_to_q2_0(codes, scales), dtype=np.uint8).reshape(2, 18)
        np.testing.assert_array_equal(
            payload[:, :2].reshape(-1).view("<f2"),
            np.array([-0.5, -0.5], dtype=np.float16),
        )
        self.assertTrue(np.all(payload[:, 2:] == 0x1B))

    def test_repack_rejects_misaligned_or_out_of_range_input(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            repack_gsq2_to_q2_0(
                np.zeros((1, 64), dtype=np.uint8),
                np.ones((1, 1), dtype=np.float32),
            )
        codes = np.zeros((1, 128), dtype=np.uint8)
        codes[0, 0] = 4
        with self.assertRaisesRegex(ValueError, "invalid"):
            repack_gsq2_to_q2_0(codes, np.ones((1, 1), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
