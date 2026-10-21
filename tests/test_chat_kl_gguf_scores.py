import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import compare_qwen36_chat_kl_gguf_scores as C  # noqa: E402


def score(kl, identity="x"):
    return {"status": "complete", "conversation_ids": ["a", "b", "c"],
            "strata": ["chat", "chat", "tool"],
            "conversation_mean_top20_kl": kl, "conversation_mean_nll": [1.0, 1.0, 1.0],
            "conversation_template_kl": [0.1, float("nan"), 0.1],
            "identity": {key: identity for key in C.SHARED}}


class CompareScoresTest(unittest.TestCase):
    def test_a_minus_b(self):
        pair = C.compare(score([0.01, 0.02, 0.03]), score([0.02, 0.03, 0.03]))
        self.assertAlmostEqual(pair["kl"]["mean_delta"], -0.02 / 3)
        self.assertEqual(pair["kl"]["improved_conversations"], 2)
        self.assertEqual(pair["template_kl"]["conversations"], 2)
        self.assertEqual(sorted(pair["per_stratum_kl"]), ["chat", "tool"])
        self.assertEqual(pair["per_stratum_kl"]["tool"]["conversations"], 1)

    def test_incompatible_scores_are_refused(self):
        with self.assertRaises(RuntimeError):
            C.check_compatible({"a": score([0, 0, 0]), "b": score([0, 0, 0], identity="y")})


if __name__ == "__main__":
    unittest.main()
