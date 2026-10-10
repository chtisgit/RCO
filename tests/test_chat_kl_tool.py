import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent / "src"))

import audit_qwen36_chat_kl as chat_kl  # noqa: E402


def _conversations(lengths):
    return [{"id": f"c{i}", "token_count": n, "tokens": list(range(1, n + 1)),
             "scored": [0] * (n // 2) + [1] * (n - n // 2)}
            for i, n in enumerate(lengths)]


class BatchingTest(unittest.TestCase):
    def test_longest_first_within_the_padded_budget(self):
        conversations = _conversations([5, 9, 3, 9, 4])
        # Ties by id: c1 before c3.
        self.assertEqual(chat_kl.batches(conversations, 18), [[1, 3], [0, 4, 2]])
        self.assertEqual(chat_kl.batches(conversations, 8), [[1], [3], [0], [4, 2]])

    def test_padding_masks_and_valid_tokens(self):
        conversations = _conversations([5, 3])
        input_ids, loss_mask = chat_kl._batch_tensors(conversations, [0, 1])
        self.assertEqual(input_ids.tolist(), [
            [1, 2, 3, 4, 5], [1, 2, 3, chat_kl.PAD_TOKEN, chat_kl.PAD_TOKEN]])
        # Padded positions and the first token are never scored.
        self.assertEqual(loss_mask.tolist(), [[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
        valid = chat_kl._valid_tokens(conversations, [0, 1])
        self.assertEqual(valid.tolist(), [True] * 5 + [True] * 3 + [False] * 2)


class CompareTest(unittest.TestCase):
    def _score(self, label, kl, base_kl=None):
        return {"status": "complete", "label": label,
                "identity": {"corpus": {"split": "heldout"}, "reference_sha256": "r"},
                "mean_top20_kl_to_bf16": float(np.mean(kl)), "mean_nll": 1.0,
                "conversation_mean_top20_kl": kl,
                "conversation_mean_nll": [1.0] * len(kl),
                "conversation_template_kl": kl,
                "strata": ["chat", "tool-call"] * (len(kl) // 2)}

    def _run(self, base_label, candidate_label, delta):
        base = [0.5, 0.4, 0.6, 0.3] * 5
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            for label, values in ((base_label, base),
                                  (candidate_label, [v + delta for v in base])):
                (reports / f"qwen36_chat_kl_score_{label}_v2.json").write_text(
                    json.dumps(self._score(label, values)))
            args = SimpleNamespace(reports=reports, suffix="_v2", base_label=base_label,
                                   candidate_label=candidate_label)
            with mock.patch("builtins.print"):
                status = chat_kl.run_compare(args)
            names = sorted(path.name for path in reports.glob("qwen36_chat_kl_comparison*"))
            report = json.loads((reports / names[0]).read_text())
        return status, names, report

    def test_default_pair_keeps_its_name_and_applies_the_margin(self):
        status, names, report = self._run("unpruned", "p24", 0.004)
        self.assertEqual(names, ["qwen36_chat_kl_comparison_v2.json"])
        self.assertEqual((status, report["status"]), (0, "pass"))
        status, _, report = self._run("unpruned", "p24", 0.006)
        self.assertEqual((status, report["status"]), (3, "fail"))

    def test_other_pairs_get_their_own_report(self):
        _, names, report = self._run("frequency", "seed0_final", -0.01)
        self.assertEqual(names, ["qwen36_chat_kl_comparison_seed0_final_vs_frequency_v2.json"])
        self.assertAlmostEqual(report["delta_top20_kl"]["mean"], -0.01)
        self.assertEqual(set(report["for_information"]["per_stratum_delta_top20_kl"]),
                         {"chat", "tool-call"})


class ModelTest(unittest.TestCase):
    def test_skeleton_uses_the_requested_attention(self):
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )

        # The tools load the multimodal skeleton, as the real checkpoint does.
        config = Qwen3_5MoeConfig(text_config=dict(
            vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8, moe_intermediate_size=8,
            shared_expert_intermediate_size=8, num_experts=4, num_experts_per_tok=2,
            linear_num_value_heads=2, linear_num_key_heads=2, linear_key_head_dim=8,
            linear_value_head_dim=8, layer_types=["full_attention"]))
        for implementation in ("eager", "sdpa"):
            with tempfile.TemporaryDirectory() as directory:
                config.save_pretrained(directory)
                args = SimpleNamespace(model_dir=Path(directory),
                                       attn_implementation=implementation)
                model = chat_kl._model(args)
            self.assertEqual(model.config._attn_implementation, implementation)
            self.assertTrue(all(p.device.type == "meta" for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
