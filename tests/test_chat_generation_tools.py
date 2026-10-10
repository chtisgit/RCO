import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent / "src"))

import audit_qwen36_chat_generation as generation  # noqa: E402
import build_qwen36_chat_corpus_v2 as corpus_v2  # noqa: E402
import build_qwen36_generation_manifest as manifest  # noqa: E402
import compare_qwen36_chat_generation as gate  # noqa: E402


# Qwen/Qwen3.6-35B-A3B model card, best practices.  llama.cpp's own defaults
# differ (min_p 0.05, no presence penalty), so every value is pinned.
CARD = {
    True: {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
           "presence_penalty": 1.5, "repeat_penalty": 1.0, "repeat_last_n": 64},
    False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 1.5, "repeat_penalty": 1.0, "repeat_last_n": 64},
}


def _prompts(name):
    return json.loads((TOOLS / name).read_text(encoding="utf-8"))


class DecodingTest(unittest.TestCase):
    def test_v1_is_greedy_once_with_per_mode_limits(self):
        prompts = _prompts("qwen36_chat_generation_prompts.json")
        seeds, settings = generation.decoding_plan(prompts)
        self.assertEqual(seeds, [None])
        for thinking, mode in ((True, "thinking_on"), (False, "thinking_off")):
            self.assertEqual(settings({"thinking": thinking}), {
                "temperature": 0, "seed": prompts["decoding"]["seed"],
                "max_tokens": prompts["max_tokens"][mode]})

    def test_v2_samples_with_the_model_card_settings(self):
        prompts = _prompts("qwen36_chat_generation_prompts_v2.json")
        seeds, settings = generation.decoding_plan(prompts)
        self.assertEqual(seeds, [20261010, 20261011, 20261012])
        for thinking in (True, False):
            self.assertEqual(settings({"thinking": thinking}),
                             {**CARD[thinking], "max_tokens": 32768})

    def test_v2_keeps_the_v1_prompts(self):
        self.assertEqual(_prompts("qwen36_chat_generation_prompts_v2.json")["prompts"],
                         _prompts("qwen36_chat_generation_prompts.json")["prompts"])

    def test_every_tool_uses_the_same_card_settings(self):
        self.assertEqual(corpus_v2.SAMPLING, CARD)
        v2 = manifest.DECODING["model-card-v2"]
        self.assertEqual(v2["thinking_on"], CARD[True])
        self.assertEqual(v2["thinking_off"], CARD[False])


class ToolCallCheckTest(unittest.TestCase):
    TOOLS = [{"type": "function", "function": {
        "name": "get_weather",
        "parameters": {"type": "object",
                       "properties": {"city": {}, "unit": {}}, "required": ["city"]}}}]

    def _call(self, name="get_weather", arguments='{"city": "Athens"}'):
        return [{"function": {"name": name, "arguments": arguments}}]

    def test_accepts_a_valid_call(self):
        self.assertEqual(generation.check_tool_calls(self._call(), self.TOOLS), (True, "ok"))

    def test_rejects_each_kind_of_invalid_call(self):
        cases = {
            "no tool call": None,
            "unknown tool": self._call(name="get_time"),
            "not JSON": self._call(arguments="{city: Athens}"),
            "not an object": self._call(arguments='["Athens"]'),
            "unknown arguments": self._call(arguments='{"city": "Athens", "day": 1}'),
            "missing required": self._call(arguments='{"unit": "C"}'),
        }
        for reason, calls in cases.items():
            ok, message = generation.check_tool_calls(calls, self.TOOLS)
            self.assertFalse(ok, reason)
            self.assertIn(reason, message)


class GateTest(unittest.TestCase):
    """compare_qwen36_chat_generation on synthetic reports."""

    def _report(self, failures, prompts_sha256="p"):
        """``failures`` maps (prompt, turn, check) to failed seeds out of 3."""
        responses = []
        for (prompt, turn, check), failed in failures.items():
            for seed in range(3):
                responses.append({"prompt_id": prompt, "turn": turn, "seed": seed,
                                  "checks": {check: seed >= failed},
                                  "tokens_per_second": 10.0})
        return {"status": "complete", "identity": {"prompts_sha256": prompts_sha256},
                "summary": {"failed_checks": sum(failures.values()),
                            "checks": len(responses)},
                "responses": responses}

    def _run(self, candidate, control, **control_kwargs):
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            for device in ("cpu", "gpu"):
                for label, failures in (("p24", candidate), ("unpruned", control)):
                    report = self._report(failures, **(control_kwargs if label == "unpruned"
                                                       else {}))
                    (reports / f"qwen36_chat_generation_{label}_{device}_v2.json").write_text(
                        json.dumps(report))
            argv = ["compare", "--reports", str(reports), "--suffix", "_v2"]
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                status = gate.main()
            result = json.loads(
                (reports / "qwen36_chat_generation_comparison_v2.json").read_text())
        return status, result

    def test_counts_only_failures_beyond_the_control(self):
        status, result = self._run(
            {("a", 0, "stopped"): 1, ("b", 0, "tool_call"): 2, ("c", 0, "stopped"): 0},
            {("a", 0, "stopped"): 1, ("b", 0, "tool_call"): 1, ("c", 0, "stopped"): 2})
        self.assertEqual(status, 3)
        self.assertEqual(result["status"], "fail")
        counted = {(e["device"], e["prompt_id"]) for e in result["counted_failures"]}
        self.assertEqual(counted, {("cpu", "b"), ("gpu", "b")})
        self.assertEqual({e["prompt_id"] for e in result["shared_failures"]}, {"a"})
        self.assertEqual({e["prompt_id"] for e in result["control_only_failures"]}, {"c"})

    def test_equal_failures_pass(self):
        status, result = self._run({("a", 0, "stopped"): 2}, {("a", 0, "stopped"): 2})
        self.assertEqual((status, result["status"]), (0, "pass"))

    def test_turn_the_control_never_reached_counts_as_passed(self):
        status, result = self._run({("m", 2, "stopped"): 1}, {("m", 0, "stopped"): 0})
        self.assertEqual(status, 3)
        entry = [e for e in result["counted_failures"] if e["device"] == "cpu"][0]
        self.assertEqual((entry["turn"], entry["control_of"]), (3, 0))

    def test_refuses_different_prompt_sets(self):
        with self.assertRaises(RuntimeError):
            self._run({("a", 0, "stopped"): 0}, {("a", 0, "stopped"): 0},
                      prompts_sha256="other")


if __name__ == "__main__":
    unittest.main()
