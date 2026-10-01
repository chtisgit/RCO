"""Validation and evaluation for deterministic generation assertions."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


REQUIRED_CATEGORIES = frozenset({
    "instruction", "extraction", "arithmetic_reasoning",
    "code", "knowledge", "multilingual",
})
ASSERTION_TYPES = frozenset({"exact", "regex", "numeric"})
_NUMBER = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def validate_prompt_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("schema") != 1:
        raise ValueError("prompt manifest schema must be 1")
    prompts = manifest.get("prompts")
    if not isinstance(prompts, list) or len(prompts) < 32:
        raise ValueError("prompt manifest must contain at least 32 prompts")
    identifiers: set[str] = set()
    categories: set[str] = set()
    for index, item in enumerate(prompts):
        if not isinstance(item, Mapping):
            raise ValueError(f"prompt {index} is not an object")
        identifier = item.get("id")
        category = item.get("category")
        prompt = item.get("prompt")
        assertion = item.get("assertion")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"prompt {index} has no id")
        if identifier in identifiers:
            raise ValueError(f"duplicate prompt id: {identifier}")
        identifiers.add(identifier)
        if category not in REQUIRED_CATEGORIES:
            raise ValueError(f"prompt {identifier} has invalid category: {category}")
        categories.add(str(category))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompt {identifier} is empty")
        if not isinstance(assertion, Mapping):
            raise ValueError(f"prompt {identifier} has no assertion")
        assertion_type = assertion.get("type")
        if assertion_type not in ASSERTION_TYPES:
            raise ValueError(
                f"prompt {identifier} has invalid assertion type: {assertion_type}")
        if assertion_type == "exact":
            if not isinstance(assertion.get("expected"), str):
                raise ValueError(f"prompt {identifier} exact value is invalid")
        elif assertion_type == "regex":
            pattern = assertion.get("pattern")
            if not isinstance(pattern, str):
                raise ValueError(f"prompt {identifier} regex is invalid")
            re.compile(pattern)
        else:
            expected = float(assertion.get("expected"))
            tolerance = float(assertion.get("absolute_tolerance"))
            if not math.isfinite(expected) or not math.isfinite(tolerance):
                raise ValueError(f"prompt {identifier} numeric bound is non-finite")
            if tolerance < 0:
                raise ValueError(f"prompt {identifier} tolerance is negative")
    missing = sorted(REQUIRED_CATEGORIES - categories)
    if missing:
        raise ValueError(f"prompt manifest is missing categories: {missing}")
    return {
        "prompt_count": len(prompts),
        "categories": sorted(categories),
        "assertion_types": sorted({
            str(item["assertion"]["type"]) for item in prompts
        }),
    }


def evaluate_generation_assertion(
    assertion: Mapping[str, Any], output: str,
) -> dict[str, Any]:
    if not isinstance(output, str):
        raise TypeError("generation output must be text")
    normalized = output.strip()
    assertion_type = assertion.get("type")
    if assertion_type == "exact":
        expected = str(assertion["expected"])
        return {
            "passed": normalized == expected,
            "type": "exact",
            "normalized_output": normalized,
            "expected": expected,
        }
    if assertion_type == "regex":
        pattern = str(assertion["pattern"])
        return {
            "passed": re.fullmatch(pattern, normalized) is not None,
            "type": "regex",
            "normalized_output": normalized,
            "pattern": pattern,
        }
    if assertion_type == "numeric":
        matches = _NUMBER.findall(normalized)
        expected = float(assertion["expected"])
        tolerance = float(assertion["absolute_tolerance"])
        actual = float(matches[0]) if len(matches) == 1 else None
        return {
            "passed": (
                actual is not None and math.isfinite(actual)
                and abs(actual - expected) <= tolerance
            ),
            "type": "numeric",
            "normalized_output": normalized,
            "parsed_value": actual,
            "expected": expected,
            "absolute_tolerance": tolerance,
            "numeric_match_count": len(matches),
        }
    raise ValueError(f"unsupported assertion type: {assertion_type}")
