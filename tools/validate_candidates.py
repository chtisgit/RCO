#!/usr/bin/env python3
"""Validate selected RCO candidates against a dense safetensors source."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.quant import parse_bitwidth_map
from store import LoadMode, WeightStore
from validation import load_assignment, validate_candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream one source tensor/slice and one decoded candidate at a "
            "time, reporting numerical error and storage statistics."))
    parser.add_argument("--source-model", required=True,
                        help="Dense Hugging Face safetensors directory.")
    parser.add_argument("--layer-dir", required=True,
                        help="RCO candidate database directory.")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--assignment",
                        help="RCO text assignment or JSON result.")
    choice.add_argument("--bitwidth", type=int,
                        help="Validate this candidate width for every layer.")
    parser.add_argument("--tensor", action="append", default=[],
                        help="Restrict validation to a layer name; repeatable.")
    parser.add_argument("--max-tensors", type=int, default=None,
                        help="Validate only the first N sorted tensors.")
    parser.add_argument("--bitwidth-map", default="",
                        help="Logical-to-actual bit mapping used by the search.")
    parser.add_argument("--chunk-elements", type=int, default=1 << 20,
                        help="Maximum FP32/FP64 error scratch elements.")
    parser.add_argument("--worst-count", type=int, default=20,
                        help="Keep this many worst tensors per error ranking.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write the bounded aggregate JSON report here.")
    parser.add_argument("--details-jsonl", type=Path, default=None,
                        help="Opt in to one JSON record per validated tensor.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.chunk_elements < 1:
        raise SystemExit("--chunk-elements must be positive")
    if args.worst_count < 0:
        raise SystemExit("--worst-count must be nonnegative")
    if args.max_tensors is not None and args.max_tensors < 1:
        raise SystemExit("--max-tensors must be positive")

    if args.assignment:
        assignment = load_assignment(args.assignment)
    else:
        store = WeightStore(
            args.layer_dir, mode=LoadMode.LAZY, cache=False).load()
        assignment = {
            name: args.bitwidth for name in store.get_layer_names()
        }
    if args.tensor:
        requested = set(args.tensor)
        missing = sorted(requested - set(assignment))
        if missing:
            raise SystemExit(
                f"requested tensors absent from assignment: {missing}")
        assignment = {
            name: bits for name, bits in assignment.items()
            if name in requested
        }
    if args.max_tensors is not None:
        assignment = dict(sorted(assignment.items())[:args.max_tensors])

    details = None
    try:
        if args.details_jsonl is not None:
            args.details_jsonl.parent.mkdir(parents=True, exist_ok=True)
            details = args.details_jsonl.open("w")
        report = validate_candidates(
            args.source_model,
            args.layer_dir,
            assignment,
            bitwidth_map=parse_bitwidth_map(args.bitwidth_map),
            chunk_elements=args.chunk_elements,
            worst_count=args.worst_count,
            details_handle=details,
        )
    finally:
        if details is not None:
            details.close()

    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
