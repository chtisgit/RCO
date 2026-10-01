import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generation_assertions import (
    evaluate_generation_assertion,
    validate_prompt_manifest,
)


def _manifest():
    categories = (
        "instruction", "extraction", "arithmetic_reasoning",
        "code", "knowledge", "multilingual",
    )
    prompts = []
    for index in range(32):
        prompts.append({
            "id": f"prompt-{index}",
            "category": categories[index % len(categories)],
            "prompt": f"Prompt {index}",
            "assertion": {"type": "exact", "expected": "ok"},
        })
    return {"schema": 1, "prompts": prompts}


class GenerationAssertionsTest(unittest.TestCase):
    def test_validates_complete_manifest(self):
        summary = validate_prompt_manifest(_manifest())
        self.assertEqual(summary["prompt_count"], 32)
        self.assertEqual(len(summary["categories"]), 6)

    def test_rejects_duplicate_ids_and_missing_categories(self):
        manifest = _manifest()
        manifest["prompts"][1]["id"] = manifest["prompts"][0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_prompt_manifest(manifest)
        manifest = _manifest()
        for item in manifest["prompts"]:
            if item["category"] == "multilingual":
                item["category"] = "instruction"
        with self.assertRaisesRegex(ValueError, "missing categories"):
            validate_prompt_manifest(manifest)

    def test_rejects_invalid_regex_and_negative_tolerance(self):
        manifest = _manifest()
        manifest["prompts"][0]["assertion"] = {
            "type": "regex", "pattern": "[",
        }
        with self.assertRaises(Exception):
            validate_prompt_manifest(manifest)
        manifest = copy.deepcopy(_manifest())
        manifest["prompts"][0]["assertion"] = {
            "type": "numeric", "expected": 1, "absolute_tolerance": -1,
        }
        with self.assertRaisesRegex(ValueError, "negative"):
            validate_prompt_manifest(manifest)

    def test_exact_and_regex_require_the_complete_stripped_output(self):
        self.assertTrue(evaluate_generation_assertion(
            {"type": "exact", "expected": "BLUE"}, "\nBLUE \n")["passed"])
        self.assertFalse(evaluate_generation_assertion(
            {"type": "exact", "expected": "BLUE"}, "BLUE.")["passed"])
        self.assertTrue(evaluate_generation_assertion(
            {"type": "regex", "pattern": r"A\d{2}"}, " A17 ")["passed"])
        self.assertFalse(evaluate_generation_assertion(
            {"type": "regex", "pattern": r"A\d{2}"}, "x A17")["passed"])

    def test_numeric_requires_one_finite_number_within_tolerance(self):
        assertion = {
            "type": "numeric", "expected": 0.6, "absolute_tolerance": 0.001,
        }
        self.assertTrue(evaluate_generation_assertion(
            assertion, "0.6005")["passed"])
        self.assertFalse(evaluate_generation_assertion(
            assertion, "0.6 or 60% ")["passed"])
        self.assertFalse(evaluate_generation_assertion(
            assertion, "0.61")["passed"])


if __name__ == "__main__":
    unittest.main()
