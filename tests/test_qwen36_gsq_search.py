import json
import sys
from pathlib import Path
import unittest

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from search_qwen36_gsq_rco import (  # noqa: E402
    _fold_for_step,
    _json_optimizer_checkpoint,
    _replay_problem,
    _replay_projection,
)


class Qwen36GSQSearchTest(unittest.TestCase):
    def test_each_ten_step_run_covers_every_fold(self):
        for run_index in range(4):
            self.assertEqual(
                sorted(_fold_for_step(run_index, step, 10)
                       for step in range(10)),
                list(range(10)),
            )

    def test_optimizer_checkpoint_is_json_serializable(self):
        state = {
            "completed_steps": 3,
            "scores": torch.tensor([0.25, -0.5]),
            "baseline": 0.125,
            "incumbent_assignment": torch.tensor([1, 0]),
            "incumbent_loss": -0.25,
            "optimizer_state": {
                "step": 3,
                "exp_avg": torch.tensor([0.1, 0.2]),
                "exp_avg_sq": torch.tensor([0.01, 0.02]),
            },
        }
        checkpoint = _json_optimizer_checkpoint(state)
        self.assertEqual(json.loads(json.dumps(checkpoint)), checkpoint)
        self.assertEqual(checkpoint["completed_steps"], 3)
        self.assertEqual(checkpoint["incumbent_assignment"], [1, 0])

    def test_replay_projection_ignores_only_provenance_and_measurements(self):
        report = {
            "schema": "search",
            "status": "complete",
            "problem": {"device": "cuda"},
            "problem_sha256": "a",
            "wall_seconds_this_invocation": 10.0,
            "warning": "pending",
            "runs": [{
                "scores": [0.25],
                "memory": {"total_seconds": 5.0},
                "evaluations": [{"mean_nll": 1.0, "memory": {"rss": 2}}],
            }],
        }
        projected = _replay_projection(report)
        self.assertEqual(projected, {
            "schema": "search",
            "runs": [{"scores": [0.25], "evaluations": [{"mean_nll": 1.0}]}],
        })
        changed = json.loads(json.dumps(report))
        changed["runs"][0]["scores"] = [0.5]
        self.assertNotEqual(_replay_projection(changed), projected)

    def test_replay_problem_allows_only_the_reference_digest_to_differ(self):
        primary = {"model": "a", "runtime": {"code": "b"}}
        replay = {**primary, "replay_reference_sha256": "c"}
        self.assertEqual(_replay_problem(primary), _replay_problem(replay))
        changed = {**replay, "model": "different"}
        self.assertNotEqual(_replay_problem(primary), _replay_problem(changed))


if __name__ == "__main__":
    unittest.main()
