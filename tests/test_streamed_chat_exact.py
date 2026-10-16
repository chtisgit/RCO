import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import audit_qwen36_prune24_prelim as prelim  # noqa: E402
import search_qwen36_prune24_q3k as q3k_search  # noqa: E402
from search import streamed_chat_exact as exact  # noqa: E402


class ExactHelpersTest(unittest.TestCase):
    def test_routing_and_kl_match_the_scoring_tool(self):
        import torch

        generator = torch.Generator().manual_seed(0)
        logits = torch.randn(32, 256, generator=generator).bfloat16()
        mask = torch.zeros(256, dtype=torch.bool)
        mask[torch.randperm(256, generator=generator)[:24]] = True
        for got, want in zip(exact.exact_pruned_routing(logits, mask, 8),
                             prelim.exact_pruned_routing(logits, mask, 8)):
            self.assertTrue(torch.equal(got, want))
        log_probs = torch.log_softmax(torch.randn(5, 100, generator=generator), dim=-1)
        values, indices = torch.log_softmax(
            torch.randn(5, 100, generator=generator), dim=-1).topk(20, dim=-1)
        self.assertTrue(torch.equal(exact.top_k_kl(values, indices, log_probs),
                                    prelim.top_k_kl(values, indices, log_probs)))


class FoldTest(unittest.TestCase):
    def test_folds_partition_and_balance_strata(self):
        conversations = [{"id": f"c{i:03d}", "stratum": ("a", "b", "c")[i % 3],
                          "token_count": 100 + 7 * i} for i in range(160)]
        folds = q3k_search.stratified_folds(conversations, 10)
        self.assertEqual(sorted(i for fold in folds for i in fold), list(range(160)))
        self.assertEqual({len(fold) for fold in folds}, {16})
        for stratum in "abc":
            counts = [sum(conversations[i]["stratum"] == stratum for i in fold)
                      for fold in folds]
            self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(folds, q3k_search.stratified_folds(conversations, 10))


class SearchStateTest(unittest.TestCase):
    def test_update_matches_optimize_hard_reinforce(self):
        import tempfile

        import torch

        from search.hard import optimize_hard_reinforce

        weights = torch.linspace(-1, 1, 120)

        def evaluate(assignment):
            return float((assignment.float() * weights).sum())

        # optimize_hard_reinforce draws from a generator on the scores' device
        # seeded with ``seed``; Search does the same on the CPU.
        _, final, history = optimize_hard_reinforce(
            evaluate, n_groups=120, low_bits=2.0, high_bits=3.0,
            target_bits=2.0 + 23 / 120, n_steps=6, lr=0.1, seed=11)
        search = q3k_search.Search(120, lr=0.1, baseline_decay=0.9, seed=11)
        with tempfile.TemporaryDirectory() as directory:
            for step in range(6):
                draws = search.draw()
                record = search.update(draws, (evaluate(draws[0][0]), evaluate(draws[1][0])))
                search.history.append(record)
                if step == 2:
                    path = Path(directory) / "state.pt"
                    search.save(path)
                    search = q3k_search.Search(120, lr=0.1, baseline_decay=0.9, seed=0)
                    search.load(path)
                self.assertAlmostEqual(record["loss_plus"], history[step]["loss_plus"], places=5)
                self.assertAlmostEqual(record["loss_minus"], history[step]["loss_minus"], places=5)
        self.assertEqual(search.final(), final.tolist())


if __name__ == "__main__":
    unittest.main()
