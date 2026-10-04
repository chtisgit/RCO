import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from audit_qwen36_gsq_upgrade_controls import _logit_comparison


class GSQUpgradeControlsTest(unittest.TestCase):
    def test_identical_logits_pass_all_bounds(self):
        logits = np.array([[1.0, 2.0, -1.0], [4.0, 0.0, 3.0]], np.float32)
        result = _logit_comparison(logits, logits.copy())
        self.assertTrue(result["passed"])
        self.assertEqual(result["top1_agreement_fraction"], 1.0)
        self.assertEqual(result["maximum_position_relative_rmse"], 0.0)

    def test_large_orthogonal_drift_fails(self):
        reference = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
        candidate = np.array([[0.0, 1.0], [1.0, 0.0]], np.float32)
        result = _logit_comparison(candidate, reference)
        self.assertFalse(result["passed"])
        self.assertEqual(result["top1_agreement_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
