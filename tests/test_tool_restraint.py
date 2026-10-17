import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import audit_qwen36_tool_restraint as T  # noqa: E402

MANIFEST = json.loads((Path(T.__file__).parent / "qwen36_tool_restraint_prompts.json").read_text())
PROMPTS = {p["id"]: p for p in MANIFEST["prompts"]}


def _reply(content="", calls=(), results=(), stop="eos"):
    return {"content": content, "tool_calls": list(calls), "results": list(results),
            "stop_type": stop}


def _call(manifest_prompt, name, arguments):
    result, distractors = T.tool_result(MANIFEST, manifest_prompt, name, arguments)
    return {"name": name, "arguments": arguments, "result": result, "distractors": distractors}


class ToolResultTest(unittest.TestCase):
    def test_arguments_and_product(self):
        result, distractors = T.tool_result(MANIFEST, PROMPTS["related-yen"], "convert_currency",
                                            {"amount": 200, "from_currency": "EUR", "to_currency": "USD"})
        self.assertEqual(result["amount"], 200)
        self.assertEqual(result["converted_amount"], 216.84)
        self.assertIn("216.84", distractors)
        self.assertIn("1.0842", distractors)

    def test_override_has_no_distractors(self):
        result, distractors = T.tool_result(MANIFEST, PROMPTS["given-rate"], "convert_currency",
                                            {"amount": 200, "from_currency": "EUR", "to_currency": "USD"})
        self.assertEqual(result["converted_amount"], 220.0)
        self.assertEqual(distractors, [])
        result, _ = T.tool_result(MANIFEST, PROMPTS["given-jacket"], "get_current_weather",
                                  {"location": "Madrid"})
        self.assertEqual((result["location"], result["temperature"]), ("Madrid", 31))

    def test_missing_argument(self):
        result, _ = T.tool_result(MANIFEST, PROMPTS["chat-greeting"], "get_current_weather", {})
        self.assertEqual(result["location"], "unknown")


class DistractorTest(unittest.TestCase):
    def test_numbers_and_phrases(self):
        found = T.find_distractors("It is 13.7°C with light rain, wind -23.4 km/h.",
                                   ["13.7", "light rain", "81", "23.4"])
        self.assertEqual(found, ["13.7", "light rain", "23.4"])
        self.assertEqual(T.find_distractors("1,084.20 dollars", ["1084.2"]), ["1084.2"])
        self.assertEqual(T.find_distractors("Flight LH 4417 at 07:35", ["LH 4417", "07:35"]),
                         ["LH 4417", "07:35"])
        self.assertEqual(T.find_distractors("In 1981 it was 813.7", ["81", "13.7"]), [])


class CheckTest(unittest.TestCase):
    def test_checks(self):
        self.assertTrue(T.check_answer({"type": "contains_any", "accept": ["Au"]}, "The symbol is **Au**."))
        self.assertFalse(T.check_answer({"type": "contains_any", "accept": ["Au"]}, "Australia"))
        self.assertTrue(T.check_answer({"type": "contains_any", "accept": ["300"]}, "Profit: $300.00"))
        self.assertTrue(T.check_answer({"type": "starts_with_any", "accept": ["no"]}, "**No**, it's hot."))
        self.assertFalse(T.check_answer({"type": "starts_with_any", "accept": ["no"]}, "Not really"))
        self.assertFalse(T.check_answer({"type": "nonempty"}, "  "))
        self.assertFalse(T.check_answer(PROMPTS["given-birthday"]["check"],
                                        "Born 2015-03-04, today 2025-08-21: she is nine."))
        self.assertTrue(T.check_answer(PROMPTS["given-birthday"]["check"], "She is 10 years old."))


class ScoreTest(unittest.TestCase):
    prompt = PROMPTS["unrelated-multiply"]

    def score(self, replies):
        return T.score_conversation(self.prompt, replies, MANIFEST["round_limit"])

    def weather(self):
        return _call(self.prompt, "get_current_weather", {"location": "Athens"})

    def test_clean_answer(self):
        s = self.score([_reply("17 × 23 = 391.")])
        self.assertFalse(s["failed"])
        self.assertFalse(s["unnecessary_call"])

    def test_single_call_is_not_a_failure(self):
        s = self.score([_reply(calls=[{}], results=[self.weather()]), _reply("It is 391.")])
        self.assertTrue(s["unnecessary_call"])
        self.assertFalse(s["failed"])

    def test_repeated_call(self):
        s = self.score([_reply(calls=[{}], results=[self.weather()]),
                        _reply(calls=[{}], results=[self.weather()]), _reply("391")])
        self.assertTrue(s["r1_repeated"] and s["failed"])

    def test_two_calls_in_one_reply(self):
        s = self.score([_reply(calls=[{}, {}], results=[self.weather(), self.weather()]),
                        _reply("391")])
        self.assertTrue(s["r1_repeated"])

    def test_influence(self):
        s = self.score([_reply(calls=[{}], results=[self.weather()]),
                        _reply("391, and it is 13.7 degrees in Athens.")])
        self.assertTrue(s["r2_influence"])
        self.assertEqual(s["distractors_found"], ["13.7"])
        s = self.score([_reply(calls=[{}], results=[self.weather()]), _reply("It is 381.")])
        self.assertTrue(s["r2_influence"])

    def test_wrong_without_tool_is_recorded_not_failed(self):
        s = self.score([_reply("It is 381.")])
        self.assertTrue(s["wrong_without_tool"])
        self.assertFalse(s["failed"])

    def test_no_answer(self):
        self.assertTrue(self.score([_reply("39", stop="limit")])["no_answer"])
        s = self.score([_reply(calls=[{}], results=[self.weather()])] * MANIFEST["round_limit"])
        self.assertTrue(s["no_answer"] and s["round_limit_reached"] and s["r1_repeated"])


if __name__ == "__main__":
    unittest.main()
