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
    from metrics import compute_kl_loss, compute_reference_log_probs


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

        compact_bytes = (
            compact[0]["values"].nbytes + compact[0]["indices"].nbytes)
        self.assertLess(compact_bytes, full[0].nbytes)


if __name__ == "__main__":
    unittest.main()
