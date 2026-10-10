import sys
import tempfile
from pathlib import Path
import unittest

import torch
from safetensors.torch import save_file


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint_stream import SafeTensorPrefixLoader  # noqa: E402
from search.prune import install_wrappers, remove_wrappers  # noqa: E402
from search.streamed_prune import (  # noqa: E402
    StreamedPruneConfig,
    StreamedPruneObjective,
    StreamedPruneSearch,
    compact_top_k_kl,
    ste_survival,
)
from search.streamed_prune_chat import (  # noqa: E402
    ChatRow,
    ChatRowSet,
    StreamedChatPruneObjective,
    microbatches,
)
from test_streamed_prune import EXPERTS, LAYERS, VOCAB, _tiny_model  # noqa: E402


def _streamed(directory, budget=None, *, fixed=False, **kwargs):
    model = _tiny_model(meta=True)
    loader = SafeTensorPrefixLoader(directory)
    if fixed:
        return model, StreamedPruneObjective(model, loader, device="cpu", position_chunk=3)
    return model, StreamedChatPruneObjective(
        model, loader, device="cpu", position_chunk=3, microbatch_tokens=budget, **kwargs)


class MicrobatchTest(unittest.TestCase):
    def test_longest_first_within_budget(self):
        self.assertEqual(microbatches([3, 9, 5, 5, 2], 10), [[1], [2, 3], [0, 4]])
        self.assertEqual(microbatches([12], 10), [[0]])


class ChatObjectiveTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.resident = _tiny_model()
        self.state = {name: value.detach().clone()
                      for name, value in self.resident.state_dict().items()}
        self.alpha = torch.randn(LAYERS * EXPERTS, 2, dtype=torch.float64)
        self.noises = [torch.randn(LAYERS * EXPERTS, 2, dtype=torch.float64)
                       for _ in range(2)]

    def _survival(self, alpha):
        return torch.stack([ste_survival(alpha, noise, 0.7, 3, LAYERS, EXPERTS)[0]
                            for noise in self.noises])

    def test_full_rows_match_fixed_length_objective(self):
        input_ids = torch.randint(0, VOCAB, (4, 9))
        teacher = torch.log_softmax(torch.randn(4, 8, VOCAB), dim=-1)
        values, indices = teacher.topk(5, dim=-1)
        values, indices = values.float(), indices.int()
        row_variant = torch.tensor([0, 1, 0, 1])
        rows = ChatRowSet([
            ChatRow(input_ids[i], torch.tensor([False] + [True] * 8),
                    values[i], indices[i]) for i in range(4)])
        results, grads = [], []
        with tempfile.TemporaryDirectory() as directory:
            save_file(self.state, Path(directory) / "model.safetensors")
            for fixed in (True, False):
                alpha = self.alpha.clone().requires_grad_(True)
                model, objective = _streamed(directory, 18, fixed=fixed)
                if fixed:
                    result = objective.run(input_ids, self._survival(alpha), row_variant,
                                           values, indices)
                else:
                    result = objective.run(rows, self._survival(alpha), row_variant)
                    self.assertEqual(result.stats.microbatch_count, 2)
                    self.assertTrue(all(p.device.type == "meta" for p in model.parameters()))
                self.assertEqual(result.stats.block_loads, [2] * LAYERS)
                self.assertEqual(result.stats.block_releases, [2] * LAYERS)
                results.append(result)
                grads.append(alpha.grad)
        torch.testing.assert_close(results[1].row_kl, results[0].row_kl, rtol=1e-6, atol=1e-9)
        self.assertAlmostEqual(results[1].objective, results[0].objective, places=9)
        self.assertGreater(float(grads[0].abs().sum()), 0.0)
        torch.testing.assert_close(grads[1], grads[0], rtol=1e-5, atol=1e-9)

    def test_variable_rows_match_resident_per_row(self):
        lengths = [11, 4, 7, 9, 6]
        row_variant = torch.tensor([0, 1, 1, 0, 1])
        rows = []
        for length in lengths:
            tokens = torch.randint(1, VOCAB, (length,))
            scored = torch.zeros(length, dtype=torch.bool)
            scored[length // 2:] = True
            count = int(scored.sum())
            teacher = torch.log_softmax(torch.randn(count, VOCAB), dim=-1)
            values, indices = teacher.topk(5, dim=-1)
            rows.append(ChatRow(tokens, scored, values.float(), indices.int()))

        # Resident: RCO's wrappers, one unpadded row at a time.
        resident_alpha = self.alpha.clone().requires_grad_(True)
        self.resident.requires_grad_(False)
        shared = {"ste_masks": None}
        wrappers = install_wrappers(self.resident, LAYERS, EXPERTS, shared)
        survival = self._survival(resident_alpha)
        weights = 1.0 / (2 * torch.bincount(row_variant)[row_variant].double())
        expected, total = [], 0.0
        try:
            for row, variant, weight in zip(rows, row_variant.tolist(), weights):
                shared["ste_masks"] = survival[variant]
                logits = self.resident(input_ids=row.tokens[None]).logits[0, :-1]
                positions = torch.nonzero(row.scored[1:]).flatten()
                kl = compact_top_k_kl(logits[positions], row.reference_values,
                                      row.reference_indices).mean()
                expected.append(float(kl))
                total = total + weight * kl
        finally:
            remove_wrappers(wrappers)
        total.backward()

        with tempfile.TemporaryDirectory() as directory:
            save_file(self.state, Path(directory) / "model.safetensors")
            streamed_alpha = self.alpha.clone().requires_grad_(True)
            _, objective = _streamed(directory, 16, pad_token=0)
            result = objective.run(ChatRowSet(rows), self._survival(streamed_alpha),
                                   row_variant)
        self.assertGreater(result.stats.microbatch_count, 1)
        self.assertEqual(result.stats.tokens, sum(lengths))
        self.assertEqual(result.stats.block_loads, [2] * LAYERS)
        # The KL is computed in float32, as in compute_kl_loss.
        torch.testing.assert_close(result.row_kl, torch.tensor(expected),
                                   rtol=1e-5, atol=1e-7)
        self.assertAlmostEqual(result.objective, float(total), places=6)
        self.assertGreater(float(resident_alpha.grad.abs().sum()), 0.0)
        torch.testing.assert_close(streamed_alpha.grad, resident_alpha.grad,
                                   rtol=1e-5, atol=1e-9)

    def test_search_step_accepts_row_sets(self):
        rows = ChatRowSet([
            ChatRow(torch.randint(1, VOCAB, (n,)), torch.tensor([False] + [True] * (n - 1)),
                    torch.randn(n - 1, 5).sort(-1, descending=True).values,
                    torch.randint(0, VOCAB, (n - 1, 5)).int()) for n in (5, 8, 6, 7)])
        config = StreamedPruneConfig(layers=LAYERS, experts=EXPERTS, prune_per_layer=3,
                                     steps=2, gumbel_samples=4, documents_per_sample=2, seed=3)
        search = StreamedPruneSearch(config, torch.zeros(LAYERS * EXPERTS, 2))
        with tempfile.TemporaryDirectory() as directory:
            save_file(self.state, Path(directory) / "model.safetensors")
            _, objective = _streamed(directory, 16)
            record = search.take_step(objective, rows, rows, rows)
        self.assertEqual(len(record["documents"]), 8)
        self.assertAlmostEqual(record["expected_prune_per_layer"], 3.0, places=2)
        self.assertEqual(record["stats"]["rows"], 8)


if __name__ == "__main__":
    unittest.main()
