import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from build_qwen36_gsq_upgrade_policy import _maximum_reachable


class GSQUpgradePolicyTest(unittest.TestCase):
    def test_maximum_reachable_uses_bounded_multiplicities(self):
        self.assertEqual(_maximum_reachable([32, 32, 96], 128, 32), 128)
        self.assertEqual(_maximum_reachable([64, 96], 128, 32), 96)

    def test_maximum_reachable_rejects_misaligned_cost(self):
        with self.assertRaises(ValueError):
            _maximum_reachable([33], 128, 32)


if __name__ == "__main__":
    unittest.main()
