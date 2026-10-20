#!/usr/bin/env python3
"""Where a llama-server setup loses decode speed against the card's numbers.

The user's server (pinned build 911f6cdc, ``-t 16``, q8_0 KV cache, 128k
context) decoded at about 24 tok/s, the card says 70.  Two stages:

``bench``: llama-bench at the Usage placement (``-ngl 99 -ncmoe 4``,
``-fa 1``), pp512 and tg128 at depths 0, 8,192 and 32,768, for every
combination of build, threads (8, 16) and KV type (f16, q8_0).

``server``: llama-server end to end with named argument sets (the user's
exact command and variants).  Each set gets prompts of about 16, 8,192 and
32,768 tokens and generates 256 tokens with ``ignore_eos``; the server's own
timings are recorded.

    bench_qwen36_speed_factors.py bench --build LABEL=BINDIR ... --model GGUF --report JSON
    bench_qwen36_speed_factors.py server --set LABEL=BINDIR::ARGS ... --model GGUF --report JSON
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_qwen36_chat_generation import _post, _wait_ready  # noqa: E402
from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file  # noqa: E402

SCHEMA = "rco.qwen36.speed_factors.v1"
ROOT = Path(__file__).resolve().parents[3]
DEPTHS = [0, 8192, 32768]


def bench_binary(bindir: Path) -> list[str]:
    exe = bindir / "llama-bench"
    return [str(exe)] if exe.exists() else [str(bindir / "llama"), "bench"]


def row_key(row: dict) -> str:
    kind = f"pp{row['n_prompt']}" if row["n_prompt"] else f"tg{row['n_gen']}"
    return f"t{row['n_threads']}/{row['type_k']}/{kind}@d{row['n_depth']}"


def stage_bench(args, report: dict) -> None:
    for item in args.build:
        label, bindir = item.split("=", 1)
        for kv in ("f16", "q8_0"):
            name = f"{label}/{kv}"
            if report["bench"].get(name, {}).get("runs"):
                continue
            command = [*bench_binary(Path(bindir)), "-m", str(args.model), "-ngl", "99",
                       "-ncmoe", "4", "-fa", "1", "-t", "8,16", "-ctk", kv, "-ctv", kv,
                       "-p", "512", "-n", "128", "-d", ",".join(map(str, DEPTHS)), "-r", "3",
                       "-o", "json"]
            started = time.perf_counter()
            done = subprocess.run(command, capture_output=True, text=True)
            entry = {"command": command, "wall_seconds": round(time.perf_counter() - started, 1)}
            if done.returncode != 0:
                entry.update(runs=False, stderr_tail=done.stderr[-3000:])
            else:
                rows = json.loads(done.stdout)
                entry.update(runs=True, rows=rows, build=rows[0].get("build_commit"),
                             tokens_per_second={row_key(r): round(r["avg_ts"], 2) for r in rows})
            report["bench"][name] = entry
            print(f"bench {name}: {entry.get('tokens_per_second', 'FAILED')}", flush=True)
            _atomic_json(args.report, report)


def _prompt_tokens(base: str, text: str, length: int) -> list[int]:
    tokens = _post(base + "/tokenize", {"content": text})["tokens"]
    while len(tokens) < length:
        tokens = tokens + tokens
    return tokens[:length]


def stage_server(args, report: dict) -> None:
    text = (ROOT / "RCO_PLAN_NEW.md").read_text()
    for item in args.set:
        label, rest = item.split("=", 1)
        bindir, extra = rest.split("::", 1)
        if report["server"].get(label, {}).get("runs"):
            continue
        command = [str(Path(bindir) / "llama-server"), "--model", str(args.model),
                   *shlex.split(extra), "--host", "127.0.0.1", "--port", str(args.port), "--no-webui"]
        log_path = ROOT / f"speed_factors_{label}.server.log"
        base = f"http://127.0.0.1:{args.port}"
        entry = {"command": command, "server_log": str(log_path), "requests": []}
        with log_path.open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                _wait_ready(base, process, timeout=1800)
                with urllib_open(base + "/props") as props:
                    entry["build_info"] = props.get("build_info")
                for length in (16, *DEPTHS[1:]):
                    tokens = _prompt_tokens(base, text, length)
                    reply = _post(base + "/completion", {
                        "prompt": tokens, "n_predict": 256, "ignore_eos": True,
                        "cache_prompt": False, "temperature": 0.6, "top_p": 0.95,
                        "top_k": 20, "min_p": 0.0, "seed": 20261020})
                    timings = reply.get("timings", {})
                    entry["requests"].append({"prompt_tokens": length, **timings})
                    print(f"server {label}: prompt {length}: pp "
                          f"{timings.get('prompt_per_second', 0):.1f}, decode "
                          f"{timings.get('predicted_per_second', 0):.2f} tok/s", flush=True)
                entry["runs"] = True
            except Exception as error:  # noqa: BLE001 - recorded, next set continues
                entry.update(runs=False, error=repr(error))
                print(f"server {label}: FAILED {error!r}", flush=True)
            finally:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
        report["server"][label] = entry
        _atomic_json(args.report, report)


class urllib_open:
    """``with urllib_open(url) as data`` for a GET returning JSON."""

    def __init__(self, url: str):
        self.url = url

    def __enter__(self) -> dict:
        import urllib.request
        with urllib.request.urlopen(self.url, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def __exit__(self, *exc) -> bool:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("bench", "server"))
    parser.add_argument("--build", action="append", default=[], help="LABEL=BINDIR")
    parser.add_argument("--set", action="append", default=[], help="LABEL=BINDIR::ARGS")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = _load_json(args.report) if args.report.exists() else {
        "schema": SCHEMA, "model": {"path": str(args.model), "sha256": _sha256_file(args.model)},
        "bench": {}, "server": {}}
    (stage_bench if args.stage == "bench" else stage_server)(args, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
