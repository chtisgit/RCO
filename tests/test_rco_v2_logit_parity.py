import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_qwen36_rco_v2_logit_parity as P  # noqa: E402


class _Store:
    def __init__(self, label):
        self.label = label

    def candidate_location(self, name):
        return self.label

    def is_retain_choice(self, name, choice):
        return choice == 0

    def get_layer_storage_bytes(self, name, choice):
        return len(self.label)

    def install_layer_weight(self, model, name, choice):
        return {"store": self.label}


class StoreTest(unittest.TestCase):
    def test_dispatch(self):
        store = P.CombinedWeightStore("token_embd.weight", _Store("embedding"), _Store("experts"))
        self.assertEqual(store.candidate_location("token_embd.weight"), "embedding")
        self.assertEqual(store.candidate_location("blk.10.ffn_down_exps.weight"), "experts")
        self.assertEqual(store.install_layer_weight(None, "blk.1.ffn_up_exps.weight", 1),
                         {"store": "experts"})
        self.assertFalse(store.cache)

    def test_assignments(self):
        plan = P.assignments(["a", "b"])
        self.assertEqual(plan["control"], [1, 0, 0])
        self.assertEqual(plan["base"], [1, 0, 0])
        self.assertEqual(plan["candidate"], [1, 1, 1])


class ReproductionTest(unittest.TestCase):
    def test_identical_and_different(self):
        with tempfile.TemporaryDirectory() as directory:
            arrays = {"tokens": np.arange(3), "native": np.ones((2, 4)), "llama": np.ones((2, 4)),
                      "native_layers": np.zeros((2, 2, 4)), "llama_layers": np.zeros((2, 2, 4))}
            a, b, c = (Path(directory) / name for name in ("a.npz", "b.npz", "c.npz"))
            np.savez(a, **arrays)
            np.savez(b, **arrays)
            np.savez(c, **{**arrays, "llama": np.ones((2, 4)) * 1.5})
            self.assertTrue(P.reproduction(a, b)["identical"])
            result = P.reproduction(a, c)
            self.assertFalse(result["identical"])
            self.assertEqual(result["arrays"]["llama"]["max_abs_difference"], 0.5)


class GateTest(unittest.TestCase):
    def test_gates(self):
        per = {"control": {"mean_logit_rmse": 0.1, "top1_agreement_rate": 0.96,
                           "layer_mean_relative_rmse": [0.0, 0.04, 0.05]},
               "candidate": {"mean_logit_rmse": 0.11, "top1_agreement_rate": 0.95,
                             "layer_mean_relative_rmse": [0.0, 0.05, 0.08]}}
        result = P.gates(per, "candidate")
        self.assertTrue(result["logit_rmse_ratio"]["passed"])
        self.assertTrue(result["top1_agreement_difference"]["passed"])
        self.assertEqual(result["layer_relative_rmse"]["failing_layers"], [1])
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
