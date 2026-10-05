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
        exact_cost_assignment,
        high_choice_count,
        optimize_cost_reinforce,
        optimize_cost_spsa,
        optimize_hard_reinforce,
        optimize_hard_spsa,
        realized_average_bits,
        realized_cost,
        sample_exact_cost_assignment,
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

    def test_exact_nonuniform_cost_assignment(self):
        low = [10, 10, 10, 10]
        high = [11, 12, 13, 14]
        scores = torch.tensor([5.0, 1.0, 4.0, 2.0])
        assignment = exact_cost_assignment(scores, low, high, target_cost=45)
        self.assertEqual(assignment.tolist(), [1, 0, 0, 1])
        self.assertEqual(realized_cost(assignment, low, high), 45)
        with self.assertRaisesRegex(ValueError, "not reachable|no exact"):
            exact_cost_assignment(scores, low, high, target_cost=40 + 11)

    def test_exact_cost_sample_has_budget_and_score_gradient(self):
        low = [10, 10, 10, 10]
        high = [11, 12, 13, 14]
        scores = torch.tensor(
            [0.2, -0.4, 1.0, 0.5], requires_grad=True)
        assignment, log_probability = sample_exact_cost_assignment(
            scores, low, high, target_cost=45,
            uniforms=torch.tensor([0.1, 0.8, 0.4, 0.6]))
        self.assertEqual(realized_cost(assignment, low, high), 45)
        log_probability.backward()
        self.assertIsNotNone(scores.grad)
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertGreater(scores.grad.norm().item(), 0.0)

    def test_cost_class_sampler_preserves_exact_distribution_gradient(self):
        size = 64
        selected_count = 20
        low = [10] * size
        high = [11] * size
        scores = torch.zeros(size, requires_grad=True)
        uniforms = torch.linspace(0.01, 0.99, size)
        assignment, log_probability = sample_exact_cost_assignment(
            scores,
            low,
            high,
            target_cost=sum(low) + selected_count,
            uniforms=uniforms,
        )
        self.assertEqual(int(assignment.sum()), selected_count)
        log_probability.backward()
        expected = assignment.float() - selected_count / size
        self.assertTrue(torch.allclose(scores.grad, expected, atol=1e-6))

    def test_cost_class_projection_matches_countwise_global_optimum(self):
        torch.manual_seed(9)
        scores = torch.randn(64)
        low = [100] * 64
        increments = [1] * 32 + [3] * 32
        high = [base + increment
                for base, increment in zip(low, increments)]
        target_increment = 40
        assignment = exact_cost_assignment(
            scores, low, high, sum(low) + target_increment)

        best_value = -float("inf")
        best_counts = None
        for first_count in range(33):
            remainder = target_increment - first_count
            if remainder < 0 or remainder % 3:
                continue
            second_count = remainder // 3
            if second_count > 32:
                continue
            value = (
                scores[:32].topk(first_count).values.sum()
                + scores[32:].topk(second_count).values.sum()
            ).item()
            if value > best_value:
                best_value = value
                best_counts = (first_count, second_count)

        self.assertEqual(realized_cost(assignment, low, high),
                         sum(low) + target_increment)
        self.assertEqual(
            (int(assignment[:32].sum()), int(assignment[32:].sum())),
            best_counts,
        )
        self.assertAlmostEqual(
            float(scores[assignment.bool()].sum()), best_value, places=5)

    def test_cost_optimizers_preserve_exact_nonuniform_budget(self):
        low = [10, 10, 10, 10]
        high = [11, 12, 13, 14]
        importance = torch.tensor([8.0, 1.0, 2.0, 7.0])

        def evaluate(assignment):
            return float((importance * (1 - assignment.float())).sum())

        for optimizer, steps in (
            (optimize_cost_spsa, 5),
            (optimize_cost_reinforce, 5),
        ):
            _, assignment, history = optimizer(
                evaluate,
                low_costs=low,
                high_costs=high,
                target_cost=45,
                n_steps=steps,
                lr=0.1,
                seed=7,
                log_interval=steps + 1,
            )
            self.assertEqual(realized_cost(assignment, low, high), 45)
            self.assertTrue(all(item["realized_cost"] == 45
                                for item in history))

    def test_cost_optimizers_return_best_evaluated_incumbent(self):
        low = [10, 10, 10, 10]
        high = [11, 12, 13, 14]
        importance = torch.tensor([8.0, 1.0, 2.0, 7.0])
        for optimizer in (optimize_cost_spsa, optimize_cost_reinforce):
            observed = {}

            def evaluate(assignment):
                loss = float((importance * (1 - assignment.float())).sum())
                observed[tuple(assignment.tolist())] = loss
                return loss

            _, assignment, history = optimizer(
                evaluate,
                low_costs=low,
                high_costs=high,
                target_cost=45,
                n_steps=8,
                lr=0.1,
                seed=3,
                log_interval=20,
            )
            self.assertEqual(observed[tuple(assignment.tolist())],
                             min(observed.values()))
            self.assertEqual(history[-1]["incumbent_loss"],
                             min(observed.values()))

    def test_cost_reinforce_continuation_preserves_better_incumbent(self):
        low = [10, 10, 10, 10]
        high = [11, 12, 13, 14]
        incumbent = torch.tensor([1, 0, 0, 1])

        def evaluate(assignment):
            return 2.0 + float(assignment.sum())

        _, selected, history = optimize_cost_reinforce(
            evaluate,
            low_costs=low,
            high_costs=high,
            target_cost=45,
            n_steps=3,
            seed=11,
            initial_incumbent_assignment=incumbent,
            initial_incumbent_loss=1.0,
            log_interval=10,
        )
        self.assertTrue(torch.equal(selected, incumbent))
        self.assertTrue(all(item["incumbent_loss"] == 1.0 for item in history))

    def test_cost_reinforce_resume_matches_uninterrupted_state(self):
        low = [10, 10, 10, 10]
        high = [11, 12, 13, 14]
        importance = torch.tensor([8.0, 1.0, 2.0, 7.0])

        def evaluate(assignment):
            return float((importance * (1 - assignment.float())).sum())

        full_scores, full_selected, full_history = optimize_cost_reinforce(
            evaluate,
            low_costs=low,
            high_costs=high,
            target_cost=45,
            n_steps=5,
            lr=0.1,
            seed=17,
            log_interval=20,
        )
        checkpoints = []
        _, _, first_history = optimize_cost_reinforce(
            evaluate,
            low_costs=low,
            high_costs=high,
            target_cost=45,
            n_steps=2,
            lr=0.1,
            seed=17,
            step_callback=checkpoints.append,
            log_interval=20,
        )
        checkpoint = checkpoints[-1]
        resumed_scores, resumed_selected, resumed_history = (
            optimize_cost_reinforce(
                evaluate,
                low_costs=low,
                high_costs=high,
                target_cost=45,
                n_steps=3,
                lr=0.1,
                seed=17,
                initial_scores=checkpoint["scores"],
                initial_incumbent_assignment=checkpoint[
                    "incumbent_assignment"],
                initial_incumbent_loss=checkpoint["incumbent_loss"],
                initial_step=checkpoint["completed_steps"],
                initial_baseline=checkpoint["baseline"],
                initial_optimizer_state=checkpoint["optimizer_state"],
                log_interval=20,
            )
        )
        self.assertTrue(torch.equal(resumed_scores, full_scores))
        self.assertTrue(torch.equal(resumed_selected, full_selected))
        self.assertEqual(first_history + resumed_history, full_history)

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
