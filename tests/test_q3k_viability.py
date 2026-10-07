import os
import sys
from pathlib import Path
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_qwen36_q3k_viability import (  # noqa: E402
    build_report,
    expert_importance,
    expert_names,
    sampled_names,
)

GGML_LIBRARY = os.environ.get("RCO_GGML_LIBRARY")


class ExpertImportanceTest(unittest.TestCase):
    def test_means_fallback_and_floor(self):
        sums = np.array([[4.0, 0.0], [0.0, 0.0], [2.0, 6.0]])
        counts = np.array([2, 0, 2])
        importance, fallback = expert_importance(sums, counts)
        self.assertEqual(fallback, 1)
        np.testing.assert_allclose(importance[0], [2.0, 3e-8], rtol=1e-6)
        np.testing.assert_allclose(importance[2], [1.0, 3.0])
        np.testing.assert_allclose(importance[1], [1.5, 1.5])
        self.assertTrue(np.all(importance > 0))

    def test_rejects_unobserved_layer(self):
        with self.assertRaises(ValueError):
            expert_importance(np.ones((2, 3)), np.zeros(2, dtype=np.int64))


@unittest.skipUnless(GGML_LIBRARY, "RCO_GGML_LIBRARY is not configured")
class ImportanceQuantizationTest(unittest.TestCase):
    def setUp(self):
        from quant.ggml_native import GGMLNativeCodec, GGMLType

        self.codec = GGMLNativeCodec(GGML_LIBRARY)
        self.type = GGMLType.Q3_K
        rng = np.random.default_rng(3)
        self.rows = rng.standard_normal((8, 512)).astype(np.float32)
        self.importance = rng.gamma(0.3, 1.0, 512).astype(np.float32)

    def test_lowers_weighted_error_at_equal_size(self):
        from quant.ggml_importance import quantize_rows_with_importance

        plain = self.codec.quantize_rows(self.rows, self.type)
        weighted = quantize_rows_with_importance(
            self.codec, self.rows, self.type, self.importance)
        self.assertEqual(len(plain), len(weighted))

        def error(payload):
            decoded = self.codec.dequantize_rows_into(
                payload, self.type, np.empty_like(self.rows))
            return float((self.importance * (decoded - self.rows) ** 2).sum())

        self.assertLess(error(weighted), error(plain))

    def test_rejects_invalid_importance(self):
        from quant.ggml_importance import quantize_rows_with_importance

        for importance in (np.ones(256), -np.ones(512), np.zeros(512)):
            with self.assertRaises(ValueError):
                quantize_rows_with_importance(
                    self.codec, self.rows, self.type, importance)


def _fake_reports(down_delta, ceiling_delta, q4_delta):
    metrics = {
        "gsq_q2_0": {"relative_rmse": 0.4, "weighted_relative_error": 0.4},
        "q3_k_imatrix": {"relative_rmse": 0.15, "weighted_relative_error": 0.12},
        "q3_k_plain": {"relative_rmse": 0.14, "weighted_relative_error": 0.14},
        "q4_k_plain": {"relative_rmse": 0.07, "weighted_relative_error": 0.07},
    }
    tensors = {
        name: {
            "family": name.split(".")[2],
            "metrics": metrics,
            "q3_k_incremental_bytes": 39_845_888,
            "q4_0_incremental_bytes": 75_497_472,
        }
        for name in expert_names()
    }
    baseline = [2.0, 1.5, 1.8, 2.2]
    down = [name for name in expert_names() if "down" in name]

    def arm(delta, upgrade_type, names):
        return {
            "upgrade_type": upgrade_type,
            "upgraded_tensors": names,
            "mean_nll": 1.9 + delta,
            "document_mean_nll": [value + delta for value in baseline],
        }

    imatrix = {
        "imatrix": {
            "sha256": "i", "min_tokens_per_expert": 1,
            "median_tokens_per_expert": 300.0,
            "experts_without_tokens_per_layer": {},
        },
        "bf16_calibration": {"mean_nll": 1.4},
    }
    return (
        imatrix,
        {"tensors": tensors, "sampled_tensors": sampled_names(),
         "store": {"index_sha256": "s"}, "gsq_gguf_sha256": "g"},
        {"identity": {}, "arms": {
            "gsq": arm(0.0, None, []),
            "q3k_down40": arm(down_delta, "Q3_K", down),
            "q4_0_down40": arm(q4_delta, "Q4_0", down),
            "q3k_all120": arm(ceiling_delta, "Q3_K", expert_names()),
        }},
    )


class BuildReportTest(unittest.TestCase):
    def test_go_when_q3k_gains_more_per_gb(self):
        report = build_report(*_fake_reports(-0.02, -0.05, -0.03), 1.9)
        self.assertEqual(report["status"], "go")
        self.assertTrue(report["gsq_reproduces_search_baseline"])
        self.assertAlmostEqual(
            report["functional"]["q3k_down40"]["fraction_of_q3k_ceiling"], 0.4)

    def test_stop_for_decision_when_q4_0_is_better_per_gb(self):
        report = build_report(*_fake_reports(-0.01, -0.05, -0.04), 1.9)
        self.assertEqual(
            report["status"], "stop_for_decision_q3k_per_gb_below_q4_0")

    def test_no_go_without_functional_gain(self):
        report = build_report(*_fake_reports(0.01, -0.05, -0.04), 1.9)
        self.assertEqual(report["status"], "no_go")


if __name__ == "__main__":
    unittest.main()
