import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_qwen36_generation_gate as G  # noqa: E402

PROMPT = {"id": "knowledge-france", "category": "knowledge", "prompt": "Capital?",
          "assertion": {"type": "exact", "expected": "Paris"}}


def _gen(prompt_id, thinking, seed, answer, raw=None, stop="eos", reasoning=""):
    raw = raw if raw is not None else ("x\n</think>\n\n" + answer if thinking else answer)
    return {"prompt_id": prompt_id, "thinking": thinking, "seed": seed, "raw_content": raw,
            "content": answer, "reasoning_content": reasoning, "stop_type": stop,
            "score": G.score(PROMPT, thinking, raw, answer, stop)}


def _report(label, generations, seeds=(1, 2, 3)):
    return {"status": "complete", "label": label, "generations": generations,
            "identity": {"manifest_sha256": "m", "parameters": {"p": 1}, "seeds": list(seeds)}}


class ScoreTest(unittest.TestCase):
    def test_well_formed(self):
        self.assertTrue(G.score(PROMPT, True, "a</think>\n\nParis", "Paris", "eos")["well_formed"])
        s = G.score(PROMPT, True, "no close Paris", "Paris", "eos")
        self.assertFalse(s["form_checks"]["think_closed"])
        self.assertFalse(G.score(PROMPT, False, "Paris", "Paris", "limit")["well_formed"])
        self.assertFalse(G.score(PROMPT, False, "Pa�ris", "Paris", "eos")["well_formed"])
        self.assertFalse(G.score(PROMPT, False, " ", " ", "eos")["well_formed"])

    def test_correct(self):
        self.assertTrue(G.score(PROMPT, False, "Paris", " Paris\n", "eos")["correct"])
        self.assertFalse(G.score(PROMPT, False, "Paris.", "Paris.", "eos")["correct"])


class CompareTest(unittest.TestCase):
    def reports(self, candidate_answers, base_answers, repeat_answer="Paris"):
        cand = _report("rco_v2", [_gen("k", False, s, a) for s, a in zip((1, 2, 3), candidate_answers)])
        base = _report("gsq", [_gen("k", False, s, a) for s, a in zip((1, 2, 3), base_answers)])
        rep = _report("rco_v2_repeat", [_gen("k", False, 1, repeat_answer)], seeds=(1,))
        return cand, base, rep

    def test_pass_when_not_worse(self):
        result = G.compare_reports(*self.reports(["Paris", "Paris.", "Paris"],
                                                 ["Paris.", "Paris", "Paris"]))
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["absolute_rule_all_correct"]["rco_v2"])

    def test_fail_when_worse(self):
        result = G.compare_reports(*self.reports(["Paris.", "Paris.", "Paris"],
                                                 ["Paris.", "Paris", "Paris"]))
        self.assertEqual(result["cells_worse_than_gsq"], ["k thinking off"])
        self.assertEqual(result["status"], "fail")

    def test_repeat_must_match(self):
        result = G.compare_reports(*self.reports(["Paris"] * 3, ["Paris"] * 3, repeat_answer="Lyon"))
        self.assertEqual(result["repeat_differing"], ["k thinking off"])
        self.assertEqual(result["status"], "fail")


if __name__ == "__main__":
    unittest.main()
