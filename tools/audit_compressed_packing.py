#!/usr/bin/env python3
"""Compare RCO's packed-state mapper with compressed-tensors exactly."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from quant.compressed_layout import pack_codes_to_int32


def audit(seed: int = 23) -> dict:
    import compressed_tensors
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

    generator = torch.Generator().manual_seed(seed)
    cases = []
    digest = hashlib.sha256()
    for bits in range(1, 9):
        for shape in ((1, 1), (3, 17), (2, 31), (4, 32), (3, 33), (2, 67)):
            codes = torch.randint(
                0, 1 << bits, shape, dtype=torch.uint8,
                generator=generator)
            actual = pack_codes_to_int32(codes, bits)
            signed = (
                codes.to(torch.int16) - (1 << (bits - 1))).to(torch.int8)
            expected = pack_to_int32(signed, bits)
            equal = torch.equal(actual, expected)
            digest.update(actual.contiguous().view(torch.uint8).numpy().tobytes())
            cases.append({
                "bits": bits,
                "shape": list(shape),
                "packed_shape": list(actual.shape),
                "equal": equal,
            })
    return {
        "schema": 1,
        "compressed_tensors": compressed_tensors.__version__,
        "seed": seed,
        "case_count": len(cases),
        "all_equal": all(item["equal"] for item in cases),
        "packed_sha256": digest.hexdigest(),
        "cases": cases,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare 1-8 bit row-aligned packing with compressed-tensors.")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = audit(args.seed)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(encoded)
    else:
        args.output.write_text(encoded)
        print(f"Wrote {args.output}")
    return 0 if report["all_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
