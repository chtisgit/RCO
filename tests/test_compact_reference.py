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
    from metrics import (
        COMPACT_REFERENCE_SCHEMA,
        compute_kl_loss,
        compute_reference_log_probs,
        summarize_compact_reference_mass,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class CompactReferenceTest(unittest.TestCase):
    class TinyLm(nn.Module if nn is not None else object):
        def __init__(self, vocab=11, hidden=7):
            super().__init__()
            self.embedding = nn.Embedding(vocab, hidden)
            self.head = nn.Linear(hidden, vocab, bias=False)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, input_ids):
            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    def test_compact_topk_matches_existing_topk_loss_and_has_gradient(self):
        torch.manual_seed(7)
        teacher = self.TinyLm()
        student = self.TinyLm()
        student.load_state_dict(teacher.state_dict())
        with torch.no_grad():
            student.head.weight[0, 0] += 0.25

        tokens = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
        full = compute_reference_log_probs(teacher, tokens, batch_size=2)
        compact = compute_reference_log_probs(
            teacher, tokens, batch_size=2, topk=3)

        full_loss = compute_kl_loss(student, tokens, full[0], topk=3)
        compact_loss = compute_kl_loss(student, tokens, compact[0], topk=3)
        self.assertTrue(torch.allclose(full_loss, compact_loss, atol=2e-4))

        compact_loss.backward()
        self.assertIsNotNone(student.head.weight.grad)
        self.assertGreater(student.head.weight.grad.abs().sum().item(), 0.0)

        self.assertEqual(
            compact[0]["schema_version"], COMPACT_REFERENCE_SCHEMA)
        expected_mass = full[0].float().exp().topk(
            3, dim=-1).values.sum(dim=-1)
        self.assertTrue(torch.allclose(
            compact[0]["retained_mass"].float(), expected_mass,
            atol=5e-4, rtol=0.0))

        mask = torch.tensor([
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
        ])
        stats = summarize_compact_reference_mass(compact, mask)
        selected_mass = compact[0]["retained_mass"].float()[
            mask[:, 1:].bool()]
        self.assertEqual(stats["token_count"], 4)
        self.assertAlmostEqual(
            stats["retained_mass_mean"], selected_mass.mean().item(), places=7)
        self.assertAlmostEqual(
            stats["omitted_mass_max"],
            1.0 - selected_mass.min().item(), places=7)

        compact_bytes = sum(
            compact[0][key].nbytes
            for key in ("values", "indices", "retained_mass"))
        self.assertLess(compact_bytes, full[0].nbytes)

    def test_mass_summary_rejects_legacy_compact_cache(self):
        legacy = [{
            "values": torch.zeros(1, 2, 1),
            "indices": torch.zeros(1, 2, 1, dtype=torch.int32),
        }]
        self.assertIsNone(summarize_compact_reference_mass(legacy))

    def test_compact_objective_matches_full_kl_when_topk_covers_vocabulary(self):
        torch.manual_seed(11)
        teacher = self.TinyLm(vocab=9)
        student = self.TinyLm(vocab=9)
        tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        full = compute_reference_log_probs(teacher, tokens, batch_size=2)
        compact = compute_reference_log_probs(
            teacher, tokens, batch_size=2, topk=9)

        full_loss = compute_kl_loss(student, tokens, full[0])
        compact_loss = compute_kl_loss(student, tokens, compact[0], topk=9)
        self.assertTrue(torch.allclose(
            full_loss, compact_loss, atol=3e-4, rtol=0.0))
        self.assertTrue(torch.allclose(
            compact[0]["retained_mass"].float(),
            torch.ones_like(compact[0]["retained_mass"].float()),
            atol=5e-4, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
