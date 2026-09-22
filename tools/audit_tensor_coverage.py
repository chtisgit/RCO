#!/usr/bin/env python3
"""Audit Qwen3.5/Qwen3.6 source tensor-name coverage without loading weights."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_adapter import classify_qwen35_tensor


def names_from_safetensors_index(path: Path) -> list[str]:
    with path.open() as handle:
        data = json.load(handle)
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{path} does not contain a weight_map object")
    return list(weight_map)


def names_from_conversion_manifest(path: Path) -> list[str]:
    """Read the source tensor map emitted by this workspace's GGUF tooling."""
    with path.open() as handle:
        data = json.load(handle)
    families = data.get("families")
    if not isinstance(families, list):
        raise ValueError(f"{path} does not contain a families array")

    names = []
    seen = set()
    for family in families:
        for name in family.get("source_tensors", {}):
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names


def build_report(names: list[str], source: Path) -> dict:
    categories = Counter(classify_qwen35_tensor(name) for name in names)
    unknown = [name for name in names if classify_qwen35_tensor(name) == "unknown"]
    text_count = sum(
        count for category, count in categories.items()
        if category not in {"vision", "mtp", "unknown"}
    )
    return {
        "status": "PASS" if not unknown else "FAIL_UNKNOWN_TENSORS",
        "source": str(source),
        "tensor_count": len(names),
        "text_tensor_count": text_count,
        "omitted_tensor_count": categories["vision"] + categories["mtp"],
        "category_counts": dict(sorted(categories.items())),
        "unknown_tensors": unknown,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--index", type=Path,
        help="Hugging Face model.safetensors.index.json",
    )
    source.add_argument(
        "--conversion-manifest", type=Path,
        help="Existing conversion manifest containing source_tensors maps",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.index:
        names = names_from_safetensors_index(args.index)
        source = args.index
    else:
        names = names_from_conversion_manifest(args.conversion_manifest)
        source = args.conversion_manifest

    report = build_report(names, source)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")

    if report["unknown_tensors"]:
        print(
            f"error: {len(report['unknown_tensors'])} unclassified tensors",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
