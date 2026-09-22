import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None

if torch is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from search.hard import (
        HardCandidateModel,
        exact_budget_assignment,
        high_choice_count,
        optimize_hard_reinforce,
        optimize_hard_spsa,
        realized_average_bits,
        sample_plackett_luce_assignment,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class HardSearchTest(unittest.TestCase):
    def test_candidate_runtime_keeps_only_selected_weight(self):
        class Store:
            def __init__(self):
                self.calls = []

            def get_layer_weight(self, name, bits):
                self.calls.append((name, bits))
                return torch.full((2, 2), float(bits))

        model = nn.Module()
        model.layers = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        groups = [SimpleNamespace(layer_names=["layers.0"])]
        store = Store()
        runtime = HardCandidateModel(model, store, groups, [2, 4])
        runtime.apply(torch.tensor([1]))
        self.assertEqual(store.calls, [("layers.0", 4)])
        self.assertTrue(torch.equal(
            model.layers[0].weight, torch.full((2, 2), 4.0)))
        self.assertEqual(runtime.layer_assignment(torch.tensor([0])),
                         {"layers.0": 2})

    def test_exact_budget_assignment(self):
        scores = torch.tensor([0.2, 1.0, -1.0, 0.5])
        assignment = exact_budget_assignment(scores, n_high=2)
        self.assertEqual(assignment.tolist(), [0, 1, 0, 1])
        self.assertEqual(high_choice_count(4, 2.0, 4.0, 3.0), 2)
        self.assertEqual(realized_average_bits(assignment, 2.0, 4.0), 3.0)

    def test_unrealizable_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be realized exactly"):
            high_choice_count(3, 2.0, 4.0, 3.0)

    def test_spsa_preserves_budget_and_improves_synthetic_choice(self):
        # The optimum selects high precision for groups 0 and 1. The objective
        # is deliberately discrete, like a streamed model evaluation.
        importance = torch.tensor([8.0, 4.0, 2.0, 1.0])

        def evaluate(assignment):
            return float((importance * (1 - assignment.float())).sum())

        _, assignment, history = optimize_hard_spsa(
            evaluate,
            n_groups=4,
            low_bits=2.0,
            high_bits=4.0,
            target_bits=3.0,
            n_steps=80,
            lr=0.15,
            perturbation=0.25,
            seed=4,
            log_interval=100,
        )
        self.assertEqual(int(assignment.sum()), 2)
        self.assertEqual(realized_average_bits(assignment, 2.0, 4.0), 3.0)
        self.assertEqual(assignment.tolist(), [1, 1, 0, 0])
        self.assertTrue(all(item["n_high"] == 2 for item in history))

    def test_plackett_luce_sample_has_exact_budget_and_score_gradient(self):
        scores = torch.tensor(
            [0.2, -0.4, 1.0, 0.5], requires_grad=True)
        assignment, log_probability = sample_plackett_luce_assignment(
            scores, 2, uniforms=torch.tensor([0.1, 0.8, 0.4, 0.6]))
        self.assertEqual(int(assignment.sum()), 2)
        log_probability.backward()
        self.assertIsNotNone(scores.grad)
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertGreater(scores.grad.norm().item(), 0.0)

    def test_reinforce_preserves_budget_and_learns_synthetic_choice(self):
        importance = torch.tensor([8.0, 4.0, 2.0, 1.0])

        def evaluate(assignment):
            return float((importance * (1 - assignment.float())).sum())

        _, assignment, history = optimize_hard_reinforce(
            evaluate,
            n_groups=4,
            low_bits=2.0,
            high_bits=4.0,
            target_bits=3.0,
            n_steps=160,
            lr=0.08,
            baseline_decay=0.9,
            seed=7,
            log_interval=200,
        )
        self.assertEqual(assignment.tolist(), [1, 1, 0, 0])
        self.assertTrue(all(item["n_high"] == 2 for item in history))
        self.assertTrue(all(item["realized_bits"] == 3.0 for item in history))


if __name__ == "__main__":
    unittest.main()
