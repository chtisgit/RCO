#!/usr/bin/env python3
"""llama-bench of the release GGUF and GSQ at realistic context depths.

For the release model card's Speed section: prompt processing (pp512,
pp4096) and decode (tg128) at depths 0, 8,192 and 32,768, at the Usage
placement (``-ngl 99 --n-cpu-moe 4``), on one llama.cpp build.  Each model
is benchmarked in its own llama-bench process; the raw rows are kept.

    bench_qwen36_release_speed.py --bench BIN --model LABEL=GGUF ... --report JSON
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_qwen36_q3k_viability import _atomic_json, _sha256_file  # noqa: E402

SCHEMA = "rco.qwen36.release_speed.v1"
PLACEMENT = ["-ngl", "99", "-ncmoe", "4"]


def test_name(row: dict) -> str:
    """``pp4096@d8192`` / ``tg128@d0`` for one llama-bench JSON row."""
    kind = f"pp{row['n_prompt']}" if row["n_prompt"] else f"tg{row['n_gen']}"
    return f"{kind}@d{row['n_depth']}"


def summarize(rows: list[dict]) -> dict:
    return {test_name(r): {"avg_ts": round(r["avg_ts"], 2), "stddev_ts": round(r.get("stddev_ts", 0.0), 2)}
            for r in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="LABEL=GGUF")
    parser.add_argument("--prompt", default="512,4096")
    parser.add_argument("--gen", default="128")
    parser.add_argument("--depth", default="0,8192,32768")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema": SCHEMA, "bench": {"path": str(args.bench), "sha256": _sha256_file(args.bench)},
              "placement": PLACEMENT, "models": {}}
    for item in args.model:
        label, path = item.split("=", 1)
        command = [str(args.bench), "-m", path, *PLACEMENT, "-t", str(args.threads),
                   "-p", args.prompt, "-n", args.gen, "-d", args.depth,
                   "-r", str(args.repetitions), "-o", "json"]
        started = time.perf_counter()
        done = subprocess.run(command, capture_output=True, text=True)
        entry = {"path": path, "sha256": _sha256_file(Path(path)), "command": command,
                 "wall_seconds": round(time.perf_counter() - started, 1)}
        if done.returncode != 0:
            entry.update(runs=False, stderr_tail=done.stderr[-3000:])
        else:
            rows = json.loads(done.stdout)
            entry.update(runs=True, tokens_per_second=summarize(rows), rows=rows)
        report["models"][label] = entry
        print(f"bench {label}: {entry.get('tokens_per_second', 'FAILED')}", flush=True)
        _atomic_json(args.report, report)
    report["status"] = "done" if all(m["runs"] for m in report["models"].values()) else "failed"
    _atomic_json(args.report, report)
    return 0 if report["status"] == "done" else 3


if __name__ == "__main__":
    raise SystemExit(main())
