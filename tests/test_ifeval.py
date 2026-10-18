import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_qwen36_ifeval as I  # noqa: E402


def _r(key, thinking, strict_all, each=(True,)):
    return {"key": key, "thinking": thinking, "instruction_ids": ["x"] * len(each),
            "strict": {"all": strict_all, "each": list(each)},
            "loose": {"all": strict_all, "each": list(each)}}


def _report(results):
    return {"identity": {"input_data": 1, "seed": 2, "decoding": 3, "parameters": 4},
            "per_generation": results, "accuracy": I.accuracies(results),
            "accuracy_by_mode": {}}


class IFEvalTest(unittest.TestCase):
    def test_projection(self):
        self.assertAlmostEqual(I.projected_hours(3600, 30, 541), 541 / 30)

    def test_accuracies(self):
        acc = I.accuracies([_r(1, False, True, (True, True)), _r(2, False, False, (True, False))])
        self.assertEqual(acc["prompt_level_strict"], 0.5)
        self.assertEqual(acc["instruction_level_strict"], 0.75)

    def test_compare_equal_passes(self):
        results = [_r(k, t, k % 3 != 0) for k in range(200) for t in (False, True)]
        out = I.compare_reports(_report(results), _report(results))
        self.assertEqual(out["status"], "pass")
        self.assertEqual(out["ci95"], [0.0, 0.0])

    def test_compare_much_worse_fails(self):
        base = [_r(k, t, True) for k in range(200) for t in (False, True)]
        cand = [_r(k, t, k % 5 != 0) for k in range(200) for t in (False, True)]
        out = I.compare_reports(_report(cand), _report(base))
        self.assertAlmostEqual(out["mean_delta"], -0.2)
        self.assertEqual(out["status"], "fail")

    def test_units_must_match(self):
        with self.assertRaises(RuntimeError):
            I.compare_reports(_report([_r(1, False, True)]), _report([_r(2, False, True)]))


if __name__ == "__main__":
    unittest.main()
