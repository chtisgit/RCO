import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch


TOOL = Path(__file__).resolve().parents[1] / "tools" / "build_qwen36_prune24_consensus.py"
LAYERS, EXPERTS, PRUNE = 40, 256, 24


def _top(score):
    mask = np.zeros(score.shape, dtype=bool)
    np.put_along_axis(mask, np.argsort(-score, axis=1, kind="stable")[:, :PRUNE], True, axis=1)
    return mask


class ConsensusTest(unittest.TestCase):
    def _write_run(self, work, seed, advantage, final=None, step=300):
        run = work / f"search_seed{seed}"
        run.mkdir(parents=True)
        alpha = torch.zeros(LAYERS * EXPERTS, 2)
        alpha[:, 1] = torch.from_numpy(advantage.reshape(-1))
        torch.save({"step": step, "alpha": alpha}, run / "state.pt")
        np.save(run / "final_mask.npy", _top(advantage) if final is None else final)

    def _build(self, work):
        return subprocess.run(
            [sys.executable, str(TOOL), "--work", str(work), "--reports", str(work)],
            capture_output=True, text=True)

    def test_agreed_first_then_mean_advantage_never_unpruned(self):
        rng = np.random.default_rng(3)
        # Correlated seeds, so that they agree on some experts but not all.
        shared = rng.normal(size=(LAYERS, EXPERTS))
        advantages = [(shared + rng.normal(size=shared.shape)).astype(np.float32)
                      for _ in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            for seed, advantage in enumerate(advantages):
                self._write_run(work, seed, advantage)
            result = self._build(work)
            self.assertEqual(result.returncode, 0, result.stderr)
            mask = np.load(work / "consensus_mask.npy")
            report = json.loads((work / "qwen36_gsq_e6_prune24_consensus.json").read_text())

        finals = [_top(a) for a in advantages]
        both, one = finals[0] & finals[1], finals[0] ^ finals[1]
        mean = (advantages[0] + advantages[1]) / 2
        expected = np.zeros_like(mask)
        for layer in range(LAYERS):
            chosen = list(np.flatnonzero(both[layer]))
            ranked = sorted(np.flatnonzero(one[layer]), key=lambda e: -mean[layer, e])
            expected[layer, chosen + ranked[:PRUNE - len(chosen)]] = True
        np.testing.assert_array_equal(mask, expected)
        self.assertTrue(0 < both.sum() < LAYERS * PRUNE)
        self.assertFalse((mask & ~(finals[0] | finals[1])).any())
        self.assertEqual(report["agreed_by_both_seeds"], int(both.sum()))
        self.assertEqual(report["filled_from_one_seed"], LAYERS * PRUNE - int(both.sum()))

    def test_refuses_a_final_mask_that_is_not_its_alpha_top_24(self):
        rng = np.random.default_rng(4)
        advantage = rng.normal(size=(LAYERS, EXPERTS)).astype(np.float32)
        wrong = _top(-advantage)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            self._write_run(work, 0, advantage, final=wrong)
            self._write_run(work, 1, advantage)
            result = self._build(work)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not the top-24", result.stderr)

    def test_refuses_an_unfinished_run(self):
        advantage = np.random.default_rng(5).normal(size=(LAYERS, EXPERTS)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            self._write_run(work, 0, advantage, step=150)
            self._write_run(work, 1, advantage)
            result = self._build(work)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected 300", result.stderr)


if __name__ == "__main__":
    unittest.main()
