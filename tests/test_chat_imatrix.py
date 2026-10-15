import sys
from pathlib import Path
import unittest

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent / "src"))

import build_qwen36_chat_imatrix as imatrix  # noqa: E402


def _conversations():
    rows = []
    for stratum, on, off in (("chat", 10, 10), ("tool-call", 6, 6), ("reasoning", 8, 0)):
        for thinking, count in ((True, on), (False, off)):
            rows += [{"id": f"{stratum}-{thinking}-{i:02d}", "stratum": stratum,
                      "thinking": thinking} for i in range(count)]
    return rows


class SplitTest(unittest.TestCase):
    def test_largest_remainder_ties_go_to_the_earlier_group(self):
        self.assertEqual(imatrix._largest_remainder(7, [12, 12]), [4, 3])
        self.assertEqual(imatrix._largest_remainder(18, [34, 29]), [10, 8])
        self.assertEqual(imatrix._largest_remainder(13, [48]), [13])

    def test_quotas_balance_and_determinism(self):
        conversations = _conversations()
        quotas = {"chat": 5, "tool-call": 3, "reasoning": 2}
        chosen = imatrix.choose_validation(conversations, quotas, seed=7)
        self.assertEqual(chosen, imatrix.choose_validation(conversations, quotas, seed=7))
        self.assertNotEqual(chosen, imatrix.choose_validation(conversations, quotas, seed=8))
        self.assertEqual(chosen, sorted(chosen))
        by_id = {c["id"]: c for c in conversations}
        picked = [by_id[i] for i in chosen]
        count = lambda s, t: sum(c["stratum"] == s and c["thinking"] == t for c in picked)
        self.assertEqual((count("chat", True), count("chat", False)), (3, 2))
        self.assertEqual((count("tool-call", True), count("tool-call", False)), (2, 1))
        self.assertEqual(count("reasoning", True), 2)

    def test_quotas_must_name_every_stratum(self):
        with self.assertRaisesRegex(ValueError, "every stratum"):
            imatrix.choose_validation(_conversations(), {"chat": 1}, seed=7)


class AccumulateTest(unittest.TestCase):
    def test_sums_match_a_per_token_loop(self):
        import torch
        import torch.nn.functional as F

        generator = torch.Generator().manual_seed(0)
        experts, hidden_size, intermediate, tokens, top_k = 6, 8, 4, 10, 2
        hidden = torch.randn(tokens, hidden_size, generator=generator)
        top_k_index = torch.stack([torch.randperm(experts, generator=generator)[:top_k]
                                   for _ in range(tokens)])
        gate_up_proj = torch.randn(experts, 2 * intermediate, hidden_size, generator=generator)
        act_fn = F.silu
        gate_up = torch.zeros(experts, hidden_size, dtype=torch.float64)
        down = torch.zeros(experts, intermediate, dtype=torch.float64)
        counts = torch.zeros(experts, dtype=torch.int64)
        imatrix.accumulate_expert_inputs(hidden, top_k_index, gate_up_proj, act_fn,
                                         gate_up, down, counts)

        want_gate_up = torch.zeros_like(gate_up)
        want_down = torch.zeros_like(down)
        want_counts = torch.zeros_like(counts)
        for token in range(tokens):
            for expert in top_k_index[token].tolist():
                x = hidden[token]
                gate, up = (gate_up_proj[expert] @ x).chunk(2)
                want_gate_up[expert] += x.double().square()
                want_down[expert] += (act_fn(gate) * up).double().square()
                want_counts[expert] += 1
        self.assertTrue(torch.equal(counts, want_counts))
        self.assertTrue(torch.allclose(gate_up, want_gate_up))
        self.assertTrue(torch.allclose(down, want_down, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
