#!/usr/bin/env python3
"""Coding reproduction of PLAN_POST_RCO_RELEASE.md (user report, 2026-10-22).

The user saw the release loop on a coding task ("I made a mistake, let me
rewrite this", the same wrong character on every rewrite) where the GSQ GGUF
did not.  This runs the user's prompt as a single user turn in
``llama-server --jinja`` (same build and offload as the Phase 9 generation
gate), thinking off and on, the generation manifest's 3 seeds and the model
card's sampling per mode, and records per generation:

- the stop type (``eos`` or ``limit``: a generation that hits the token limit
  did not finish);
- self-correction phrases and the most repeated non-trivial line;
- the code blocks of the final answer by language, CSS brace balance, and
  ``node --check`` on each JavaScript block.

Subcommands: ``run --model GGUF --label L`` (resumable per generation) and
``summary`` (one table over all labels present).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_chat_generation import _post, _wait_ready  # noqa: E402
from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file  # noqa: E402

SCHEMA = "rco.qwen36.coding_repro.v1"
PROMPT = "Write a web-based Mario Kart game. Use HTML/CSS/JS. Make one file per type."
MODES = (True, False)
CORRECTION = re.compile(r"(made a mistake|let me rewrite|let me fix|wait,|I made an error|"
                        r"typo|that's wrong|let me correct)", re.IGNORECASE)
FENCE = re.compile(r"```([A-Za-z0-9_+-]*)[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)


def code_blocks(answer: str) -> list[dict]:
    blocks = []
    for language, body in FENCE.findall(answer):
        language = language.lower()
        language = {"javascript": "js", "htm": "html"}.get(language, language)
        blocks.append({"language": language, "chars": len(body), "body": body})
    return blocks


def node_check(source: str) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(source)
        path = handle.name
    try:
        result = subprocess.run(["node", "--check", path], capture_output=True, text=True,
                                timeout=60)
        message = (result.stderr.strip().splitlines() or [""])
        return {"ok": result.returncode == 0,
                "error": "" if result.returncode == 0 else "\n".join(message[:6])}
    finally:
        os.unlink(path)


def analyse(raw: str, answer: str) -> dict:
    lines = [line.strip() for line in raw.splitlines() if len(line.strip()) >= 20]
    counts = collections.Counter(lines)
    top_line, top_count = counts.most_common(1)[0] if counts else ("", 0)
    blocks = code_blocks(answer)
    checks = []
    for block in blocks:
        item = {"language": block["language"], "chars": block["chars"]}
        if block["language"] == "css":
            item["brace_balance"] = block["body"].count("{") - block["body"].count("}")
        if block["language"] == "js":
            item["node_check"] = node_check(block["body"])
        checks.append(item)
    return {"correction_phrases": len(CORRECTION.findall(raw)),
            "correction_phrases_in_answer": len(CORRECTION.findall(answer)),
            "most_repeated_line": top_line[:200], "most_repeated_line_count": top_count,
            "distinct_line_fraction": (len(counts) / len(lines)) if lines else 1.0,
            "code_blocks": checks,
            "languages": sorted({c["language"] for c in checks}),
            "js_syntax_errors": sum(not c["node_check"]["ok"] for c in checks if "node_check" in c),
            "css_unbalanced": sum(c.get("brace_balance", 0) != 0 for c in checks)}


def run(args) -> int:
    manifest = _load_json(args.manifest)["canonical_manifest"]
    decoding = manifest["decoding"]
    seeds = decoding["seeds"]
    output = args.reports / f"qwen36_coding_repro_{args.label}.json"
    revision = subprocess.run(["git", "-C", str(args.llama_dir), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    identity = {
        "model": {"path": str(args.model), "sha256": _sha256_file(args.model)},
        "server": {"path": str(args.server), "sha256": _sha256_file(args.server)},
        "llama_cpp_revision": revision, "prompt": PROMPT, "seeds": seeds,
        "decoding": decoding,
        "parameters": {"gpu_layers": args.gpu_layers, "n_cpu_moe": args.n_cpu_moe,
                       "threads": args.threads, "context": args.context, "fit": "off"},
    }
    report = {"schema": SCHEMA, "status": "incomplete", "label": args.label,
              "identity": identity, "generations": []}
    if output.exists():
        previous = _load_json(output)
        if previous.get("identity") != identity:
            raise RuntimeError(f"{output} describes another run; refusing to resume")
        if previous.get("status") == "complete":
            print(f"{args.label}: already complete", flush=True)
            return 0
        report = previous
    complete = {(g["thinking"], g["seed"]) for g in report["generations"]}
    base = f"http://127.0.0.1:{args.port}"
    log = (ROOT / f"coding_repro_{args.label}.server.log").open("w")
    command = [str(args.server), "-m", str(args.model), "--jinja", "-ngl", str(args.gpu_layers),
               "-t", str(args.threads), "-c", str(args.context), "-np", "1", "--fit", "off",
               "--host", "127.0.0.1", "--port", str(args.port), "--no-webui",
               "--n-cpu-moe", str(args.n_cpu_moe)]
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        _wait_ready(base, process, timeout=1800)
        for thinking in MODES:
            for seed in seeds:
                if (thinking, seed) in complete:
                    continue
                settings = {**decoding["thinking_on" if thinking else "thinking_off"],
                            "max_tokens": decoding["maximum_generated_tokens"]}
                body = {"messages": [{"role": "user", "content": PROMPT}],
                        "chat_template_kwargs": {"enable_thinking": thinking},
                        **settings, "seed": seed, "verbose": True}
                started = time.perf_counter()
                response = _post(base + "/v1/chat/completions", body)
                choice = response["choices"][0]
                verbose = response.get("__verbose", {})
                raw = verbose.get("content", "")
                answer = choice["message"].get("content") or ""
                item = {"thinking": thinking, "seed": seed, "raw_content": raw,
                        "content": answer, "stop_type": verbose.get("stop_type"),
                        "tokens_predicted": verbose.get("tokens_predicted"),
                        "tokens_per_second": response.get("timings", {}).get(
                            "predicted_per_second", 0.0),
                        "wall_seconds": time.perf_counter() - started,
                        "analysis": analyse(raw, answer)}
                report["generations"].append(item)
                _atomic_json(output, report)
                a = item["analysis"]
                print(f"{args.label}: thinking {'on' if thinking else 'off'} seed {seed}: "
                      f"stop {item['stop_type']}, {item['tokens_predicted']} tokens, "
                      f"corrections {a['correction_phrases']}, top line x"
                      f"{a['most_repeated_line_count']}, languages {a['languages']}, "
                      f"js errors {a['js_syntax_errors']}, css unbalanced {a['css_unbalanced']}",
                      flush=True)
        report["status"] = "complete"
        _atomic_json(output, report)
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
        log.close()
    return 0


def summary(args) -> int:
    rows = []
    for path in sorted(args.reports.glob("qwen36_coding_repro_*.json")):
        report = _load_json(path)
        for g in report["generations"]:
            a = g["analysis"]
            rows.append((report["label"], "on" if g["thinking"] else "off", g["seed"],
                         g["stop_type"], g["tokens_predicted"], a["correction_phrases"],
                         a["most_repeated_line_count"], ",".join(a["languages"]),
                         a["js_syntax_errors"], a["css_unbalanced"]))
    print("label thinking seed stop tokens corrections top_line_count languages "
          "js_errors css_unbalanced")
    for row in rows:
        print(*row)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "summary"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--n-cpu-moe", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--server", type=Path,
                        default=ROOT / "experiment/build-cuda-mmqfix/bin/llama-server")
    parser.add_argument("--llama-dir", type=Path, default=ROOT / "experiment/llama.cpp-mmqfix")
    parser.add_argument("--manifest", type=Path,
                        default=RCO / "reports/qwen36_35b_generation_prompt_manifest_v2.json")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    args = parser.parse_args()
    for key in ("server", "llama_dir", "manifest", "reports"):
        setattr(args, key, getattr(args, key).resolve())
    if args.command == "summary":
        return summary(args)
    if args.model is None or args.label is None:
        parser.error("run needs --model and --label")
    args.model = args.model.resolve()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
