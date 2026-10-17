#!/usr/bin/env python3
"""Phase 8 of RCO_PLAN_NEW.md: tool-call guardrails T1 and T2 (safeguard 4).

``T1, first token``
    On every thinking-off tool prompt where the reference calls the tool
    with P(``<tool_call>``) >= 0.9 at the first reply position, the
    candidate's greedy first token (``probe_qwen36_template_tokens.py
    first-token``) must be ``<tool_call>``.  The reference is BF16 from the
    saved top 20 for the chat corpus v2 prompts and the Q6_K GGUF's probe for
    the gate prompts (no BF16 reply exists there).  The prompt set is
    recomputed and must equal the 17 fixed in the plan.
``T2, sampled``
    The three gate tool prompts with 20 seeds each
    (``qwen36_chat_generation_prompts_v2_tools20.json``, GPU): each must call
    the tool, by the generation audit's ``tool_call`` check, in at least 18
    of 20.

``--reference-label`` adds another model (authentic GSQ) as a reference arm:
evaluated the same way, reported next to the candidate, not gated.  Exit
status 3 when the candidate fails.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_chat_kl import _load_reference, load_corpus  # noqa: E402
from audit_qwen36_q3k_viability import _atomic_json, _load_json  # noqa: E402

SCHEMA = "rco.qwen36.tool_call_guardrails.v1"
TOOL_CALL = "<tool_call>"
TOOL_CALL_ID = 248058
T1_THRESHOLD = 0.9
T2_MINIMUM = 18
T2_SEEDS = 20
T1_PROMPTS = sorted(
    [f"chat-v2-calibration-tool-call-{n:02d}" for n in (5, 11, 13, 15, 17, 19, 21, 23)]
    + [f"chat-v2-heldout-tool-call-{n:02d}" for n in (3, 5, 9, 13, 19, 21, 23)]
    + ["tool-weather", "tool-flights"])


def _first_token_report(args, label: str) -> dict[str, Any]:
    report = _load_json(args.reports / f"qwen36_template_token_probe_first_token_{label}.json")
    if report.get("status") != "complete":
        raise RuntimeError(f"first-token probe {label} is not complete")
    return report


def bf16_first_reply_probabilities(args) -> dict[str, float]:
    """BF16 P(<tool_call>) at the first scored position of each corpus
    conversation, from the saved top 20 (0 if outside the top 20)."""
    result = {}
    for split in ("calibration", "heldout"):
        corpus_args = SimpleNamespace(corpus=args.corpus, split=split, limit=None,
                                      reports=args.reports, suffix=f"_v2_{split}")
        conversations, identity = load_corpus(corpus_args)
        reference, _ = _load_reference(corpus_args, identity)
        for conversation, (values, indices) in zip(conversations, reference):
            hit = np.flatnonzero(indices[0] == TOOL_CALL_ID)
            result[conversation["id"]] = float(np.exp(values[0][hit[0]])) if hit.size else 0.0
    return result


def reference_probabilities(args, prompt_ids: list[str]) -> dict[str, float]:
    bf16 = bf16_first_reply_probabilities(args)
    q6k = {item["id"]: item for item in _first_token_report(args, "q6k_reference")["prompts"]}
    result = {}
    for prompt_id in prompt_ids:
        if q6k[prompt_id]["source"] == "gate":
            log_prob = q6k[prompt_id]["tool_call_log_prob"]
            result[prompt_id] = 0.0 if log_prob is None else math.exp(log_prob)
        else:
            result[prompt_id] = bf16[prompt_id]
    return result


def t1(args, label: str, reference: dict[str, float]) -> dict[str, Any]:
    prompts = {item["id"]: item for item in _first_token_report(args, label)["prompts"]}
    if set(prompts) != set(reference):
        raise RuntimeError(f"{label}: probe prompts differ from the reference's")
    rows = []
    for prompt_id in T1_PROMPTS:
        item = prompts[prompt_id]
        log_prob = item["tool_call_log_prob"]
        rows.append({"id": prompt_id, "reference_p_tool_call": reference[prompt_id],
                     "greedy_token": item["greedy_token"],
                     "p_tool_call": None if log_prob is None else math.exp(log_prob),
                     "pass": item["greedy_token"] == TOOL_CALL})
    others = sorted(set(prompts) - set(T1_PROMPTS))
    return {"pass": all(row["pass"] for row in rows),
            "greedy_tool_call": sum(row["pass"] for row in rows), "of": len(rows),
            "prompts": rows,
            "other_prompts_greedy_tool_call": sum(prompts[p]["greedy_token"] == TOOL_CALL
                                                  for p in others),
            "other_prompts": len(others)}


def t2(args, label: str) -> dict[str, Any]:
    report = _load_json(args.reports / f"qwen36_chat_generation_{label}_gpu_v2_tools20.json")
    if report.get("status") != "complete":
        raise RuntimeError(f"tools20 generation {label} is not complete")
    counts: dict[str, list[int]] = {}
    for item in report["responses"]:
        entry = counts.setdefault(item["prompt_id"], [0, 0])
        entry[0] += bool(item["checks"]["tool_call"])
        entry[1] += 1
    if sorted(counts) != ["tool-currency", "tool-flights", "tool-weather"] or any(
            of != T2_SEEDS for _, of in counts.values()):
        raise RuntimeError(f"{label}: expected 3 tool prompts x {T2_SEEDS} seeds")
    rows = {prompt: {"tool_calls": calls, "of": of, "pass": calls >= T2_MINIMUM}
            for prompt, (calls, of) in sorted(counts.items())}
    return {"pass": all(row["pass"] for row in rows.values()), "prompts": rows,
            "model": report["identity"]["model"], "server": report["identity"]["server"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True)
    parser.add_argument("--reference-label")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--corpus", type=Path,
                        default=RCO / "reports/qwen36_chat_corpus_v2_manifest.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.reports / f"qwen36_tool_call_guardrails_{args.label}.json"

    probe_ids = [item["id"] for item in _first_token_report(args, args.label)["prompts"]]
    reference = reference_probabilities(args, probe_ids)
    selected = sorted(p for p, value in reference.items() if value >= T1_THRESHOLD)
    if selected != T1_PROMPTS:
        raise RuntimeError(f"reference selects {selected}, not the plan's {T1_PROMPTS}")
    arms = {label: {"t1": t1(args, label, reference), "t2": t2(args, label)}
            for label in filter(None, (args.label, args.reference_label))}
    candidate = arms[args.label]
    status = "pass" if candidate["t1"]["pass"] and candidate["t2"]["pass"] else "fail"
    report = {
        "schema": SCHEMA, "status": status, "candidate": args.label,
        "reference_arm": args.reference_label,
        "rules": {
            "t1": (f"greedy first reply token is {TOOL_CALL} on every thinking-off tool "
                   f"prompt with reference P({TOOL_CALL}) >= {T1_THRESHOLD}"),
            "t2": f"each gate tool prompt calls the tool in >= {T2_MINIMUM} of {T2_SEEDS} seeds"},
        "t1_reference_probabilities": reference,
        "arms": arms,
    }
    _atomic_json(output, report)
    for label, arm in arms.items():
        print(f"{label}: T1 {arm['t1']['greedy_tool_call']}/{arm['t1']['of']} "
              f"({'pass' if arm['t1']['pass'] else 'FAIL'}), T2 "
              + ", ".join(f"{p} {r['tool_calls']}/{r['of']}" for p, r in arm["t2"]["prompts"].items())
              + f" ({'pass' if arm['t2']['pass'] else 'FAIL'})", flush=True)
    print(f"guardrails: {status}", flush=True)
    return 0 if status == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
