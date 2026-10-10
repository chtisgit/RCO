import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent / "src"))

import search_qwen36_prune24_chat as chat_search  # noqa: E402
from audit_qwen36_prune24_prelim import frequency_prune_mask  # noqa: E402


class BaselineMaskTest(unittest.TestCase):
    def _setup(self, directory, *, tamper=False):
        root = Path(directory)
        rng = np.random.default_rng(0)
        counts = rng.integers(0, 1000, size=(40, 256))
        probability = rng.random((40, 256))
        stats = root / "router_stats_unpruned.npz"
        np.savez(stats, counts=counts, weight_sum=probability, probability_sum=probability)
        sha256 = hashlib.sha256(stats.read_bytes()).hexdigest()
        report = {"status": "complete", "identity": {"corpus": {"split": "calibration"}},
                  "router_stats": {"path": str(stats),
                                   "sha256": "0" * 64 if tamper else sha256}}
        (root / "qwen36_chat_kl_score_unpruned_v2_calibration.json").write_text(
            json.dumps(report))
        args = SimpleNamespace(reports=root, work=root / "work", suffix="_v2_calibration")
        return args, counts, probability

    def test_prunes_the_least_selected_experts_of_the_chat_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            args, counts, probability = self._setup(directory)
            with mock.patch("builtins.print"):
                chat_search.run_baseline_mask(args)
            mask = np.load(args.work / "frequency_mask.npy")
            report = json.loads(
                (args.reports / "qwen36_gsq_e6_prune24_chat_frequency_mask.json").read_text())
        np.testing.assert_array_equal(mask, frequency_prune_mask(counts, probability, 24))
        self.assertEqual(report["mask"]["pruned_per_layer"], 24)
        self.assertAlmostEqual(
            report["mask"]["unpruned_routed_slot_share_of_pruned_experts"],
            counts[mask].sum() / counts.sum())

    def test_refuses_statistics_that_differ_from_their_report(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = self._setup(directory, tamper=True)
            with self.assertRaisesRegex(RuntimeError, "differ from their report"):
                chat_search.run_baseline_mask(args)


if __name__ == "__main__":
    unittest.main()
