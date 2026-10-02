import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from release_quality import (
    paired_bootstrap_mean_ci,
    partial_perplexity_ratio_certificate,
    weighted_mean_nll,
)


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

    def test_partial_ratio_certificate_proves_unrecoverable_failure(self):
        result = partial_perplexity_ratio_certificate(
            baseline_mean_nll=1.0,
            observed_mean_nll=[8.0],
            observed_token_counts=[20],
            total_token_count=100,
            maximum_ratio=1.15,
        )
        self.assertTrue(result["failure_proven"])
        self.assertAlmostEqual(result["candidate_mean_nll_lower_bound"], 1.6)
        self.assertGreater(
            result["candidate_perplexity_ratio_lower_bound"], 1.15)

    def test_partial_ratio_certificate_does_not_overclaim(self):
        result = partial_perplexity_ratio_certificate(
            baseline_mean_nll=1.0,
            observed_mean_nll=[2.0],
            observed_token_counts=[10],
            total_token_count=100,
            maximum_ratio=1.15,
        )
        self.assertFalse(result["failure_proven"])

    def test_partial_ratio_certificate_rejects_negative_nll(self):
        with self.assertRaises(ValueError):
            partial_perplexity_ratio_certificate(
                baseline_mean_nll=1.0,
                observed_mean_nll=[-0.1],
                observed_token_counts=[10],
                total_token_count=100,
                maximum_ratio=1.15,
            )


if __name__ == "__main__":
    unittest.main()
