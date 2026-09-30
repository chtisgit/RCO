import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.incumbent import (
    HardCandidateEvidence,
    select_reproducible_hard_incumbent,
)


def _report(loss=2.0, relaxed_loss=3.0, bits="01", reference=None):
    report = {
        "status": "pass",
        "source": {"repo_id": "model", "revision": "revision"},
        "candidate_store": {
            "index_sha256": "store",
            "decision_count": 2,
            "low_type": "Q2_0",
            "high_type": "Q4_0",
            "target_cost": 30,
        },
        "relaxed_step": {
            "assignment": [0, 1],
            "assignment_bits": bits,
            "assignment_cost": 30,
            "relaxed_loss": relaxed_loss,
            "sha256": "step",
        },
        "calibration": {
            "input_ids": [1, 2, 3],
            "objective": "cross entropy",
            "vocab_chunk_size": 16,
        },
        "hard_evaluation": {"loss": loss, "token_count": 2},
        "reproducibility_reference": reference,
        "same_assignment_reproducible": reference is not None,
    }
    return report


def _evidence(label, loss, relaxed_loss, digest):
    primary = _report(loss, relaxed_loss)
    repeat = copy.deepcopy(primary)
    repeat["reproducibility_reference"] = {"sha256": digest}
    repeat["same_assignment_reproducible"] = True
    return HardCandidateEvidence(label, primary, repeat, digest)


class HardIncumbentTest(unittest.TestCase):
    def test_selects_reproduced_hard_loss_not_relaxed_loss(self):
        result = select_reproducible_hard_incumbent([
            _evidence("step-1", loss=1.0, relaxed_loss=5.0, digest="one"),
            _evidence("step-2", loss=2.0, relaxed_loss=1.0, digest="two"),
        ])
        self.assertEqual(result["incumbent"]["label"], "step-1")
        self.assertEqual(result["candidates"][1]["hard_loss_regression_percent"], 100.0)

    def test_rejects_repeat_with_wrong_primary_digest(self):
        evidence = _evidence("step", 1.0, 1.0, "correct")
        evidence.repeat["reproducibility_reference"]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "wrong primary"):
            select_reproducible_hard_incumbent([evidence])

    def test_rejects_different_problem_identity(self):
        first = _evidence("first", 1.0, 1.0, "first")
        second = _evidence("second", 2.0, 2.0, "second")
        second.primary["source"]["revision"] = "other"
        second.repeat["source"]["revision"] = "other"
        with self.assertRaisesRegex(ValueError, "different hard problem"):
            select_reproducible_hard_incumbent([first, second])

    def test_rejects_non_exact_assignment_cost(self):
        evidence = _evidence("step", 1.0, 1.0, "digest")
        evidence.primary["relaxed_step"]["assignment_cost"] = 29
        with self.assertRaisesRegex(ValueError, "exact byte target"):
            select_reproducible_hard_incumbent([evidence])


if __name__ == "__main__":
    unittest.main()
