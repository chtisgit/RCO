#!/usr/bin/env python3
"""Phase 4 chat check 1 of RCO_PLAN_NEW.md: gate over the generation runs.

Reads the four ``audit_qwen36_chat_generation.py`` reports (P24 and the
unpruned control, each on CPU and GPU).  Failures are counted per check,
prompt, turn and device over the seeds (one seed for greedy v1).  A key
counts against P24 when P24 fails it on more seeds than the control does;
other failures are reported.  Pass: no counted key.  Exit status 3 on
failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RCO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RCO / "tools"))

from audit_qwen36_q3k_viability import _atomic_json, _load_json  # noqa: E402

SCHEMA = "rco.qwen36.chat_generation_comparison.v1"


def _failures(report: dict) -> dict[tuple[str, int, str], list[int]]:
    """Map (prompt, turn, check) to [failed seeds, seeds that reached it]."""
    counts: dict[tuple[str, int, str], list[int]] = {}
    for item in report["responses"]:
        for name, ok in item["checks"].items():
            entry = counts.setdefault((item["prompt_id"], item["turn"], name), [0, 0])
            entry[0] += not ok
            entry[1] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--candidate", default="p24")
    parser.add_argument("--control", default="unpruned")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    result = {"schema": SCHEMA, "devices": {}, "counted_failures": [],
              "shared_failures": [], "control_only_failures": []}
    for device in ("cpu", "gpu"):
        reports = {label: _load_json(
            args.reports / f"qwen36_chat_generation_{label}_{device}{args.suffix}.json")
            for label in (args.candidate, args.control)}
        if any(r.get("status") != "complete" for r in reports.values()):
            raise RuntimeError(f"{device}: a generation report is incomplete")
        if (reports[args.candidate]["identity"]["prompts_sha256"]
                != reports[args.control]["identity"]["prompts_sha256"]):
            raise RuntimeError(f"{device}: runs used different prompt sets")
        candidate = _failures(reports[args.candidate])
        control = _failures(reports[args.control])
        for key in sorted(set(candidate) | set(control)):
            # A turn the control never reached (it ended a conversation with a
            # tool call) counts as passed for the control.
            (failed, reached), (control_failed, control_reached) = (
                candidate.get(key, [0, 0]), control.get(key, [0, 0]))
            entry = {"device": device, "prompt_id": key[0], "turn": key[1] + 1,
                     "check": key[2], "failed": failed, "of": reached,
                     "control_failed": control_failed, "control_of": control_reached}
            if failed > control_failed:
                result["counted_failures"].append(entry)
            elif failed:
                result["shared_failures"].append(entry)
            elif control_failed:
                result["control_only_failures"].append(entry)
        result["devices"][device] = {
            label: {"responses": len(r["responses"]),
                    "failed_checks": r["summary"]["failed_checks"],
                    "checks": r["summary"]["checks"],
                    "mean_tokens_per_second": sum(
                        item["tokens_per_second"] for item in r["responses"])
                    / len(r["responses"])}
            for label, r in reports.items()}
    result["status"] = "pass" if not result["counted_failures"] else "fail"
    result["rule"] = (f"on CPU and GPU, {args.candidate} fails no check on a prompt and turn "
                      f"on more seeds than {args.control} does")
    _atomic_json(args.reports / f"qwen36_chat_generation_comparison{args.suffix}.json", result)
    print(json.dumps({key: result[key] for key in
                      ("status", "devices", "counted_failures", "shared_failures",
                       "control_only_failures")}, indent=1), flush=True)
    return 0 if result["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
