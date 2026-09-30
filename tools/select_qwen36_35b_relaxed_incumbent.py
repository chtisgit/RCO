#!/usr/bin/env python3
"""Select a Qwen3.6 relaxed projection using reproducible hard loss only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.incumbent import (  # noqa: E402
    HardCandidateEvidence,
    select_reproducible_hard_incumbent,
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", action="append", nargs=3, required=True,
        metavar=("LABEL", "PRIMARY", "REPEAT"),
        help="candidate label, primary hard report, and independent repeat",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evidence = []
    report_paths = []
    for label, primary_name, repeat_name in args.candidate:
        primary_path = Path(primary_name).resolve(strict=True)
        repeat_path = Path(repeat_name).resolve(strict=True)
        primary_sha256 = _sha256(primary_path)
        repeat_sha256 = _sha256(repeat_path)
        evidence.append(HardCandidateEvidence(
            label=label,
            primary=_load_json(primary_path),
            repeat=_load_json(repeat_path),
            primary_sha256=primary_sha256,
        ))
        report_paths.append({
            "label": label,
            "primary": {"path": str(primary_path), "sha256": primary_sha256},
            "repeat": {"path": str(repeat_path), "sha256": repeat_sha256},
        })

    selection = select_reproducible_hard_incumbent(evidence)
    report = {
        "schema": 1,
        "status": "pass",
        "scope": (
            "incumbent-preserving selection among exact-byte relaxed "
            "projections using independently reproduced hard native loss"
        ),
        "policy": {
            "metric": "exact full-vocabulary causal cross-entropy",
            "direction": "minimize",
            "requires_independent_exact_replay": True,
            "relaxed_loss_is_not_a_promotion_metric": True,
        },
        "reports": report_paths,
        **selection,
    }
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "incumbent": report["incumbent"]["label"],
        "hard_loss": report["incumbent"]["hard_loss"],
        "candidate_count": len(report["candidates"]),
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
