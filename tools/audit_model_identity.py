#!/usr/bin/env python3
"""Record immutable identity and text tensor inventory for a Qwen checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_identity import audit_qwen_checkpoint_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--weight-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_qwen_checkpoint_identity(
        args.model_dir,
        repo_id=args.repo_id,
        expected_revision=args.revision,
        expected_weight_sha256=args.weight_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "revision": report["revision"],
        "tensor_count": report["tensor_count"],
        "text_tensor_count": report["text_tensor_count"],
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

