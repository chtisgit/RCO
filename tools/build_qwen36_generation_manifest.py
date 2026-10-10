#!/usr/bin/env python3
"""Build the pinned Qwen3.6 generation-prompt manifest.

``--decoding model-card-v2`` (the default since 2026-10-10) uses the
Qwen3.6-35B-A3B model card's sampling for each mode, three seeds, thinking
off and on, and the card's 32,768-token output limit; the assertions apply
to the final answer.  ``--decoding greedy-v1`` rebuilds the superseded
greedy manifest byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generation_assertions import validate_prompt_manifest  # noqa: E402
from release_corpus import canonical_json_bytes  # noqa: E402


def _exact(identifier: str, category: str, prompt: str, expected: str) -> dict:
    return {
        "id": identifier, "category": category, "prompt": prompt,
        "assertion": {"type": "exact", "expected": expected},
    }


def _numeric(
    identifier: str, prompt: str, expected: float, tolerance: float = 0.0,
) -> dict:
    return {
        "id": identifier, "category": "arithmetic_reasoning", "prompt": prompt,
        "assertion": {
            "type": "numeric", "expected": expected,
            "absolute_tolerance": tolerance,
        },
    }


PROMPTS = [
    _exact("instruction-blue", "instruction",
           "Reply with exactly the uppercase word BLUE and nothing else.", "BLUE"),
    _exact("instruction-rco", "instruction",
           "Output exactly these three uppercase letters: RCO", "RCO"),
    _exact("instruction-cedar", "instruction",
           "Repeat the lowercase word cedar and output nothing else.", "cedar"),
    _exact("instruction-ok", "instruction",
           "Respond with exactly OK, with no punctuation.", "OK"),
    _exact("instruction-vienna", "instruction",
           "Write the single word Vienna and nothing else.", "Vienna"),
    _exact("instruction-symbol", "instruction",
           "Output exactly the symbol # and no other characters.", "#"),

    _exact("extract-color", "extraction",
           "Aster=red; Birch=blue; Cedar=green. What is Birch's value? "
           "Answer with the value only.", "blue"),
    _exact("extract-order", "extraction",
           "Record: customer Mira, order ZX-417, total 28 euros. Return only "
           "the order code.", "ZX-417"),
    _exact("extract-city", "extraction",
           "Sentence: 'The conference moved from Linz to Graz on Tuesday.' "
           "Return only the destination city.", "Graz"),
    _exact("extract-email", "extraction",
           "Contact line: Name: Ada; Email: ada@example.org; Team: Core. "
           "Return only the email address.", "ada@example.org"),
    _exact("extract-third", "extraction",
           "Values in order are quartz, amber, cobalt, silver. Return only "
           "the third value.", "cobalt"),

    _numeric("arithmetic-sum",
             "Compute 37 + 58. Return only the number.", 95),
    _numeric("arithmetic-division",
             "Compute 144 divided by 12. Return only the number.", 12),
    _numeric("arithmetic-product",
             "Compute 17 times 19. Return only the number.", 323),
    _numeric("reasoning-distance",
             "A train travels at 60 km/h for 2.5 hours. How many kilometres "
             "does it travel? Return only the number.", 150),
    _numeric("reasoning-sequence",
             "The sequence is 2, 6, 12, 20. Return only the next number.", 30),
    _numeric("reasoning-probability",
             "A bag has 3 red and 2 blue balls. Give the probability of "
             "drawing red as a decimal, and output only that number.", 0.6, 0.001),

    _exact("code-python-def", "code",
           "Which Python keyword begins a function definition? Return only "
           "the keyword.", "def"),
    _exact("code-sql-select", "code",
           "Which SQL keyword retrieves rows from a table? Return only the "
           "uppercase keyword.", "SELECT"),
    _exact("code-cpp-null", "code",
           "What is the C++11 null pointer literal? Return only the literal.",
           "nullptr"),
    _exact("code-python-length", "code",
           "Give the Python expression that returns the length of variable "
           "items. Return only the expression.", "len(items)"),
    {
        "id": "code-json-object", "category": "code",
        "prompt": (
            "Return exactly this compact JSON object with keys in the shown "
            "order and no code fence: {\"ready\":true}"
        ),
        "assertion": {"type": "regex", "pattern": r'\{"ready":true\}'},
    },

    _exact("knowledge-france", "knowledge",
           "What is the capital of France? Return only the city name.", "Paris"),
    _exact("knowledge-water", "knowledge",
           "What is the chemical formula for water? Return only the formula.",
           "H2O"),
    _exact("knowledge-red-planet", "knowledge",
           "Which planet is known as the Red Planet? Return only its name.",
           "Mars"),
    _exact("knowledge-austen", "knowledge",
           "Who wrote Pride and Prejudice? Return only the author's full name.",
           "Jane Austen"),
    _exact("knowledge-photosynthesis", "knowledge",
           "Which gas do plants release during photosynthesis? Return only "
           "the gas name.", "oxygen"),

    _exact("multilingual-de", "multilingual",
           "Translate the English word 'house' into German. Return only the "
           "German word.", "Haus"),
    _exact("multilingual-es", "multilingual",
           "Translate the English word 'book' into Spanish. Return only the "
           "Spanish word.", "libro"),
    _exact("multilingual-fr", "multilingual",
           "Translate the English word 'water' into French. Return only the "
           "French word.", "eau"),
    _exact("multilingual-it", "multilingual",
           "Translate the English word 'sun' into Italian. Return only the "
           "Italian word.", "sole"),
    _exact("multilingual-pt", "multilingual",
           "A male speaker says 'thank you' in Portuguese. Return only the "
           "Portuguese word.", "obrigado"),
]


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


DECODING = {
    "greedy-v1": {
        "temperature": 0,
        "seed": 20261001,
        "reasoning": "off",
        "maximum_generated_tokens": 64,
    },
    "model-card-v2": {
        "source": "Qwen/Qwen3.6-35B-A3B model card, best practices",
        "seeds": [20261010, 20261011, 20261012],
        "reasoning": ["off", "on"],
        "maximum_generated_tokens": 32768,
        "thinking_on": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                        "presence_penalty": 1.5, "repeat_penalty": 1.0,
                        "repeat_last_n": 64},
        "thinking_off": {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.0,
                         "presence_penalty": 1.5, "repeat_penalty": 1.0,
                         "repeat_last_n": 64},
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decoding", choices=sorted(DECODING), default="model-card-v2")
    args = parser.parse_args()
    canonical_manifest = {
        "schema": 1,
        "decoding": DECODING[args.decoding],
        "prompts": PROMPTS,
    }
    summary = validate_prompt_manifest(canonical_manifest)
    digest = hashlib.sha256(canonical_json_bytes(canonical_manifest)).hexdigest()
    report = {
        "schema": 1,
        "status": "pass",
        "scope": (
            "pinned deterministic generation prompts and machine-checkable "
            "assertions for Qwen3.6 release-quality evaluation"
            if args.decoding == "greedy-v1" else
            "pinned generation prompts, sampled with the model card's settings, "
            "and machine-checkable assertions for Qwen3.6 release-quality evaluation"
        ),
        "canonical_manifest_sha256": digest,
        "summary": summary,
        "canonical_manifest": canonical_manifest,
    }
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"], "output": str(args.output),
        "canonical_manifest_sha256": digest, **summary,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
