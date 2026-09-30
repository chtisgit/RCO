#!/usr/bin/env python3
"""Evaluate measured Qwen3.6 release evidence using the pinned RCO policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.release_gate import evaluate_release_gate  # noqa: E402


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.evidence.open(encoding="utf-8") as handle:
        evidence = json.load(handle)
    decision = evaluate_release_gate(evidence)
    report = {
        "schema": 1,
        "status": (
            "pass" if decision["authorized_for_final_gguf_construction"]
            else "fail"
        ),
        "scope": (
            "preconstruction release-quality authorization for the selected "
            "Qwen3.6-35B-A3B native assignment"
        ),
        "evidence_path": str(args.evidence.resolve()),
        **decision,
    }
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "authorized_for_final_gguf_construction": report[
            "authorized_for_final_gguf_construction"],
        "failed_checks": report["failed_checks"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0 if report["authorized_for_final_gguf_construction"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
