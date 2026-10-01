import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from release_quality import paired_bootstrap_mean_ci, weighted_mean_nll


class ReleaseQualityTest(unittest.TestCase):
    def test_weighted_mean_uses_predicted_token_counts(self):
        self.assertAlmostEqual(weighted_mean_nll([1.0, 3.0], [1, 3]), 2.5)

    def test_weighted_mean_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            weighted_mean_nll([], [])
        with self.assertRaises(ValueError):
            weighted_mean_nll([1.0], [0])
        with self.assertRaises(ValueError):
            weighted_mean_nll([math.inf], [1])

    def test_paired_bootstrap_is_deterministic_and_order_sensitive(self):
        first = paired_bootstrap_mean_ci(
            [-1.0, 0.0, 1.0], samples=1000, seed=7)
        second = paired_bootstrap_mean_ci(
            [-1.0, 0.0, 1.0], samples=1000, seed=7)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], 0.0)
        self.assertGreaterEqual(first[1], 0.0)

    def test_constant_delta_has_degenerate_interval(self):
        self.assertEqual(
            paired_bootstrap_mean_ci([0.25] * 10, samples=100, seed=3),
            (0.25, 0.25),
        )


if __name__ == "__main__":
    unittest.main()
