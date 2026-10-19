#!/usr/bin/env python3
"""Build the GSQ-RCO release GGUF from the gated Phase 8/9 candidate.

The release changes nothing but ``general.name`` (the candidate inherited
authentic GSQ's).  The copy is written by llama.cpp's own
``gguf_new_metadata.py`` and then checked against the candidate:

- the candidate's SHA-256 is the one every Phase 8 and 9 report names;
- every metadata key except ``general.name`` has the same type and value;
- every tensor has the same name, order, type, shape and data bytes.

Runs with an interpreter that has gguf-py's dependencies (numpy, PyYAML):

    build_qwen36_release_gguf.py --candidate GGUF --output GGUF --name NAME \\
        --expected-sha256 HEX --gguf-py DIR --report JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "rco.qwen36.gsq_rco_v2_release_gguf.v1"
NAME_KEY = "general.name"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fields(reader) -> dict:
    result = {}
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        result[key] = ([int(t) for t in field.types], field.contents())
    return result


def _tensors(reader) -> list[tuple[str, str, list[int], str]]:
    return [(t.name, t.tensor_type.name, [int(n) for n in t.shape],
             hashlib.sha256(memoryview(t.data).cast("B")).hexdigest())
            for t in reader.tensors]


def compare(candidate, release, name: str) -> dict:
    """Compare two opened GGUFReaders; return the validation record."""
    before, after = _fields(candidate), _fields(release)
    differing = sorted(k for k in set(before) | set(after)
                       if k != NAME_KEY and before.get(k) != after.get(k))
    old_tensors, new_tensors = _tensors(candidate), _tensors(release)
    return {
        "metadata_keys": len(after),
        "metadata_identical_except_name": not differing,
        "metadata_differing": differing,
        "name_before": before.get(NAME_KEY, (None, None))[1],
        "name_after": after.get(NAME_KEY, (None, None))[1],
        "name_set": after.get(NAME_KEY, (None, None))[1] == name,
        "tensor_count": len(new_tensors),
        "tensor_order_identical": [t[0] for t in old_tensors] == [t[0] for t in new_tensors],
        "tensors_identical": old_tensors == new_tensors,
        "tensor_types": dict(sorted(_count(t[1] for t in new_tensors).items())),
    }


def _count(items) -> dict:
    counts: dict = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--gguf-py", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    candidate_sha = _sha256_file(args.candidate)
    if candidate_sha != args.expected_sha256:
        print(f"candidate sha256 {candidate_sha} is not the gated {args.expected_sha256}",
              file=sys.stderr)
        return 3
    if args.output.exists():
        print(f"{args.output} exists; refusing to overwrite", file=sys.stderr)
        return 3
    script = args.gguf_py / "gguf/scripts/gguf_new_metadata.py"
    partial = args.output.with_name(args.output.name + ".partial")
    subprocess.run([sys.executable, str(script), "--general-name", args.name,
                    str(args.candidate), str(partial)], check=True,
                   env={**os.environ, "PYTHONPATH": str(args.gguf_py)})
    sys.path.insert(0, str(args.gguf_py))
    from gguf import GGUFReader

    validation = compare(GGUFReader(args.candidate), GGUFReader(partial), args.name)
    ok = (validation["metadata_identical_except_name"] and validation["name_set"]
          and validation["tensor_order_identical"] and validation["tensors_identical"])
    if ok:
        partial.rename(args.output)
    final = args.output if ok else partial
    revision = subprocess.run(["git", "-C", str(args.gguf_py.parent), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    report = {
        "schema": SCHEMA,
        "status": "built" if ok else "failed",
        "candidate": {"path": str(args.candidate), "sha256": candidate_sha,
                      "bytes": args.candidate.stat().st_size},
        "tool": {"script": str(script), "sha256": _sha256_file(script),
                 "llama_cpp_revision": revision},
        "output": {"path": str(final), "sha256": _sha256_file(final),
                   "bytes": final.stat().st_size},
        "validation": validation,
        "wall_seconds": time.perf_counter() - started,
    }
    args.report.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"release gguf: {report['status']}; {final.name} {report['output']['bytes']} bytes, "
          f"sha256 {report['output']['sha256']}; {validation['tensor_count']} tensors identical "
          f"{validation['tensors_identical']}; metadata differing {validation['metadata_differing']}",
          flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
