#!/usr/bin/env python3
"""Score IFEval responses with Google's reference scorer (no judge model).

Runs in the IFEval venv (``experiment/ifeval-venv``) with
``experiment/ifeval-src`` on ``sys.path``:

    ifeval_score.py --source DIR --input-data JSONL --responses JSONL --output JSON

``--responses`` holds one ``{"key", "prompt", "response"}`` object per line.
The output gives, per key, strict and loose: whether every instruction is
followed, and the per-instruction list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input-data", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    from instruction_following_eval import evaluation_lib

    inputs = {item.key: item for item in evaluation_lib.read_prompt_list(str(args.input_data))}
    responses = [json.loads(line) for line in args.responses.read_text().splitlines() if line]
    result = {}
    for item in responses:
        example = inputs[item["key"]]
        if example.prompt != item["prompt"]:
            raise RuntimeError(f"prompt {item['key']} differs from the input data")
        mapping = {example.prompt: item["response"]}
        strict = evaluation_lib.test_instruction_following_strict(example, mapping)
        loose = evaluation_lib.test_instruction_following_loose(example, mapping)
        result[str(item["key"])] = {
            "instruction_ids": list(example.instruction_id_list),
            "strict": {"all": bool(strict.follow_all_instructions),
                       "each": [bool(v) for v in strict.follow_instruction_list]},
            "loose": {"all": bool(loose.follow_all_instructions),
                      "each": [bool(v) for v in loose.follow_instruction_list]},
        }
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
