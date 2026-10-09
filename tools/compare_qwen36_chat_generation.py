#!/usr/bin/env python3
"""Phase 4 chat check 1 of RCO_PLAN_NEW.md: gate over the generation runs.

Reads the four ``audit_qwen36_chat_generation.py`` reports (P24 and the
unpruned control, each on CPU and GPU).  A P24 check failure counts against
P24 only when the control passes the same check on the same prompt, turn
and device; failures the control shares are reported.  Pass: no counted
failure.  Exit status 3 on failure.
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


def _checks(report: dict) -> dict[tuple[str, int, str], bool]:
    return {(item["prompt_id"], item["turn"], name): ok
            for item in report["responses"] for name, ok in item["checks"].items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--candidate", default="p24")
    parser.add_argument("--control", default="unpruned")
    args = parser.parse_args()

    result = {"schema": SCHEMA, "devices": {}, "counted_failures": [],
              "shared_failures": [], "control_only_failures": []}
    for device in ("cpu", "gpu"):
        reports = {label: _load_json(
            args.reports / f"qwen36_chat_generation_{label}_{device}.json")
            for label in (args.candidate, args.control)}
        if any(r.get("status") != "complete" for r in reports.values()):
            raise RuntimeError(f"{device}: a generation report is incomplete")
        if (reports[args.candidate]["identity"]["prompts_sha256"]
                != reports[args.control]["identity"]["prompts_sha256"]):
            raise RuntimeError(f"{device}: runs used different prompt sets")
        candidate, control = (_checks(reports[args.candidate]), _checks(reports[args.control]))
        for key, ok in sorted(candidate.items()):
            entry = {"device": device, "prompt_id": key[0], "turn": key[1] + 1, "check": key[2]}
            # A turn the control never reached (it ended a conversation with a
            # tool call) gives no evidence either way; it counts against P24.
            control_ok = control.get(key, True)
            if not ok and control_ok:
                result["counted_failures"].append(entry)
            elif not ok:
                result["shared_failures"].append(entry)
        result["control_only_failures"] += [
            {"device": device, "prompt_id": k[0], "turn": k[1] + 1, "check": k[2]}
            for k, ok in sorted(control.items()) if not ok and candidate.get(k, False)]
        result["devices"][device] = {
            label: {"responses": len(r["responses"]),
                    "failed_checks": r["summary"]["failed_checks"],
                    "checks": r["summary"]["checks"],
                    "mean_tokens_per_second": sum(
                        item["tokens_per_second"] for item in r["responses"])
                    / len(r["responses"])}
            for label, r in reports.items()}
    result["status"] = "pass" if not result["counted_failures"] else "fail"
    result["rule"] = (f"every {args.candidate} check passes on CPU and GPU, unless "
                      f"{args.control} fails the same check on the same prompt, turn and device")
    _atomic_json(args.reports / "qwen36_chat_generation_comparison.json", result)
    print(json.dumps({key: result[key] for key in
                      ("status", "devices", "counted_failures", "shared_failures",
                       "control_only_failures")}, indent=1), flush=True)
    return 0 if result["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
