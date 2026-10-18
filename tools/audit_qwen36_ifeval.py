#!/usr/bin/env python3
"""Phase 9 IFEval of RCO_PLAN_NEW.md (decision D, 2026-10-18).

The 541 IFEval prompts (Google's ``instruction_following_eval``, data and
scorer pinned by commit and sha256) as single user turns in
``llama-server --jinja`` at P3 (``-ngl 99 --n-cpu-moe 4``), thinking off and
on, one seed (20261023), the model card's sampling: 1,082 generations per
model.  The scorer sees the final answer only (the content after
``</think>``) and runs in its own venv (``tools/ifeval_score.py``).

Subcommands:

``run --model GGUF --label L``
    Resumable per generation; each prompt runs thinking off, then on.  After
    30 prompts, a projected total above twice the 9-hour estimate stops the
    run at a decision point (exit 4).
``score --label L``
    The four standard accuracies, and per generation the strict and loose
    results.
``compare``
    Non-inferiority against GSQ: the per-unit difference in prompt-level
    strict pass (candidate - GSQ), paired over prompt x mode, paired
    bootstrap 95% CI (10,000 samples, seed 20261001); pass if its lower
    bound is at least -0.04 (exit 3 if not).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_chat_generation import _post, _wait_ready  # noqa: E402
from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file  # noqa: E402

SCHEMA = "rco.qwen36.ifeval.v1"
SEED = 20261023
MODES = (False, True)
MARGIN = 0.04
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20261001
ESTIMATE_HOURS = 9.0
GUARD_PROMPTS = 30
DECODING_SOURCE = RCO / "tools/qwen36_chat_generation_prompts_v2.json"


def load_prompts(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _settings(thinking: bool) -> dict:
    decoding = _load_json(DECODING_SOURCE)["decoding"]
    return {**decoding["thinking_on" if thinking else "thinking_off"],
            "max_tokens": decoding["max_tokens"]}


def projected_hours(elapsed_seconds: float, prompts_done: int, prompts_total: int) -> float:
    return elapsed_seconds / prompts_done * prompts_total / 3600


def run(args) -> int:
    prompts = load_prompts(args.input_data)
    stem = f"qwen36_ifeval_{args.label}"
    output = args.work / f"{stem}.generations.json"
    revision = subprocess.run(["git", "-C", str(args.llama_dir), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    identity = {
        "model": {"path": str(args.model), "sha256": _sha256_file(args.model)},
        "server": {"path": str(args.server), "sha256": _sha256_file(args.server)},
        "llama_cpp_revision": revision,
        "input_data": {"path": str(args.input_data), "sha256": _sha256_file(args.input_data)},
        "seed": SEED, "decoding": {"thinking_off": _settings(False), "thinking_on": _settings(True)},
        "parameters": {"gpu_layers": args.gpu_layers, "n_cpu_moe": args.n_cpu_moe,
                       "threads": args.threads, "context": args.context, "fit": "off"},
    }
    report = {"schema": SCHEMA + ".generations", "status": "incomplete", "label": args.label,
              "identity": identity, "generations": []}
    if output.exists():
        previous = _load_json(output)
        if previous.get("identity") != identity:
            raise RuntimeError(f"{output} describes another run; refusing to resume")
        if previous.get("status") == "complete":
            print(f"{stem}: already complete", flush=True)
            return 0
        report = previous
    complete = {(g["key"], g["thinking"]) for g in report["generations"]}
    base = f"http://127.0.0.1:{args.port}"
    log = (ROOT / f"{stem}.server.log").open("w")
    command = [str(args.server), "-m", str(args.model), "--jinja", "-ngl", str(args.gpu_layers),
               "-t", str(args.threads), "-c", str(args.context), "-np", "1", "--fit", "off",
               "--host", "127.0.0.1", "--port", str(args.port), "--no-webui"]
    if args.n_cpu_moe:
        command += ["--n-cpu-moe", str(args.n_cpu_moe)]
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        _wait_ready(base, process, timeout=1800)
        started = time.perf_counter()
        new_prompts = 0
        guard_checked = len({key for key, _ in complete}) >= GUARD_PROMPTS
        for prompt in prompts:
            ran = False
            for thinking in MODES:
                if (prompt["key"], thinking) in complete:
                    continue
                body = {"messages": [{"role": "user", "content": prompt["prompt"]}],
                        "chat_template_kwargs": {"enable_thinking": thinking},
                        **_settings(thinking), "seed": SEED, "verbose": True}
                response = _post(base + "/v1/chat/completions", body)
                choice = response["choices"][0]
                message = choice["message"]
                verbose = response.get("__verbose", {})
                item = {"key": prompt["key"], "thinking": thinking,
                        "raw_content": verbose.get("content", ""),
                        "content": message.get("content") or "",
                        "reasoning_content": message.get("reasoning_content") or "",
                        "stop_type": verbose.get("stop_type"),
                        "finish_reason": choice.get("finish_reason"),
                        "tokens_predicted": verbose.get("tokens_predicted"),
                        "tokens_per_second": response.get("timings", {}).get(
                            "predicted_per_second", 0.0)}
                report["generations"].append(item)
                _atomic_json(output, report)
                ran = True
                print(f"{stem}: {prompt['key']} thinking {'on' if thinking else 'off'}: "
                      f"stop {item['stop_type']}, {item['tokens_predicted']} tokens", flush=True)
            new_prompts += ran
            if not guard_checked and new_prompts >= GUARD_PROMPTS:
                guard_checked = True
                hours = projected_hours(time.perf_counter() - started, new_prompts, len(prompts))
                report["budget_guard"] = {"prompts": new_prompts, "projected_hours": hours,
                                          "limit_hours": 2 * ESTIMATE_HOURS}
                _atomic_json(output, report)
                print(f"{stem}: projected {hours:.1f} h for all prompts", flush=True)
                if hours > 2 * ESTIMATE_HOURS:
                    print(f"{stem}: DECISION, projection above {2 * ESTIMATE_HOURS} h", flush=True)
                    return 4
        report["status"] = "complete"
        report["summary"] = {"generations": len(report["generations"]),
                             "stopped_on_eos": sum(g["stop_type"] == "eos"
                                                   for g in report["generations"]),
                             "wall_seconds_this_invocation": time.perf_counter() - started}
        _atomic_json(output, report)
        print(f"{stem}: complete, {report['summary']}", flush=True)
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
        log.close()
    return 0


def accuracies(results: list[dict]) -> dict[str, float]:
    out = {}
    for mode in ("strict", "loose"):
        out[f"prompt_level_{mode}"] = float(np.mean([r[mode]["all"] for r in results]))
        each = [v for r in results for v in r[mode]["each"]]
        out[f"instruction_level_{mode}"] = float(np.mean(each))
    return out


def score(args) -> int:
    stem = f"qwen36_ifeval_{args.label}"
    generations = _load_json(args.work / f"{stem}.generations.json")
    if generations.get("status") != "complete":
        raise RuntimeError(f"{stem} generations are not complete")
    prompts = {p["key"]: p for p in load_prompts(args.input_data)}
    results: dict[bool, dict] = {}
    for thinking in MODES:
        with tempfile.TemporaryDirectory(prefix="rco-ifeval-") as temporary:
            responses = Path(temporary) / "responses.jsonl"
            scored = Path(temporary) / "scored.json"
            responses.write_text("".join(json.dumps(
                {"key": g["key"], "prompt": prompts[g["key"]]["prompt"], "response": g["content"]})
                + "\n" for g in generations["generations"] if g["thinking"] == thinking))
            done = subprocess.run(
                [str(args.ifeval_python), str(RCO / "tools/ifeval_score.py"),
                 "--source", str(args.ifeval_source), "--input-data", str(args.input_data),
                 "--responses", str(responses), "--output", str(scored)],
                env={**os.environ, "NLTK_DATA": str(args.nltk_data)},
                capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"IFEval scorer failed: {done.stderr[-3000:]}")
            results[thinking] = json.loads(scored.read_text())
    per_generation = []
    for g in generations["generations"]:
        r = results[g["thinking"]][str(g["key"])]
        per_generation.append({"key": g["key"], "thinking": g["thinking"],
                               "stop_type": g["stop_type"], "tokens_predicted": g["tokens_predicted"],
                               **r})
    by_mode = {("on" if t else "off"): accuracies([r for r in per_generation if r["thinking"] == t])
               for t in MODES}
    by_type: dict[str, list[bool]] = defaultdict(list)
    for r in per_generation:
        for name, ok in zip(r["instruction_ids"], r["strict"]["each"]):
            by_type[name].append(ok)
    report = {
        "schema": SCHEMA, "status": "scored", "label": args.label,
        "identity": {**generations["identity"],
                     "scorer": {"source": str(args.ifeval_source),
                                "commit": subprocess.run(
                                    ["git", "-C", str(args.ifeval_source), "rev-parse", "HEAD"],
                                    capture_output=True, text=True, check=True).stdout.strip(),
                                "wrapper_sha256": _sha256_file(RCO / "tools/ifeval_score.py")},
                     "generations_sha256": _sha256_file(args.work / f"{stem}.generations.json")},
        "accuracy": accuracies(per_generation), "accuracy_by_mode": by_mode,
        "instruction_type_strict": {k: float(np.mean(v)) for k, v in sorted(by_type.items())},
        "stopped_on_eos": sum(r["stop_type"] == "eos" for r in per_generation),
        "generations": len(per_generation),
        "per_generation": per_generation,
    }
    _atomic_json(args.reports / f"{stem}.json", report)
    print(f"{stem}: {report['accuracy']}; by mode {by_mode}", flush=True)
    return 0


def compare_reports(candidate: dict, base: dict) -> dict[str, Any]:
    from release_quality import paired_bootstrap_mean_ci

    for key in ("input_data", "seed", "decoding", "parameters"):
        if candidate["identity"][key] != base["identity"][key]:
            raise RuntimeError(f"the candidate and GSQ differ in {key}")
    index = {(r["key"], r["thinking"]): r for r in base["per_generation"]}
    units = sorted((r["key"], r["thinking"]) for r in candidate["per_generation"])
    if sorted(index) != units:
        raise RuntimeError("the candidate and GSQ scored different prompt x mode units")
    cand = {(r["key"], r["thinking"]): r for r in candidate["per_generation"]}
    deltas = [float(cand[u]["strict"]["all"]) - float(index[u]["strict"]["all"]) for u in units]
    low, high = paired_bootstrap_mean_ci(deltas, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)
    by_mode = {}
    for thinking in MODES:
        d = [x for x, u in zip(deltas, units) if u[1] == thinking]
        by_mode["on" if thinking else "off"] = {"mean_delta": float(np.mean(d)),
                                                "candidate_better": int(sum(x > 0 for x in d)),
                                                "gsq_better": int(sum(x < 0 for x in d))}
    passed = low >= -MARGIN
    return {"schema": SCHEMA + ".comparison", "status": "pass" if passed else "fail",
            "rule": f"paired bootstrap 95% CI lower bound of prompt-level strict pass "
                    f"(rco_v2 - gsq) >= -{MARGIN}",
            "units": len(units), "mean_delta": float(np.mean(deltas)), "ci95": [low, high],
            "margin": MARGIN, "candidate_better": int(sum(x > 0 for x in deltas)),
            "gsq_better": int(sum(x < 0 for x in deltas)), "by_mode": by_mode,
            "accuracy": {"rco_v2": candidate["accuracy"], "gsq": base["accuracy"]},
            "accuracy_by_mode": {"rco_v2": candidate["accuracy_by_mode"],
                                 "gsq": base["accuracy_by_mode"]},
            "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED}}


def compare(args) -> int:
    load = lambda label: _load_json(args.reports / f"qwen36_ifeval_{label}.json")  # noqa: E731
    result = compare_reports(load("rco_v2"), load("gsq"))
    _atomic_json(args.reports / "qwen36_ifeval_comparison_rco_v2_vs_gsq.json", result)
    print(f"ifeval: {result['status']}; delta {result['mean_delta']:+.4f} CI "
          f"[{result['ci95'][0]:+.4f}, {result['ci95'][1]:+.4f}]; accuracy {result['accuracy']}",
          flush=True)
    return 0 if result["status"] == "pass" else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "score", "compare"))
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
    parser.add_argument("--ifeval-source", type=Path, default=ROOT / "experiment/ifeval-src")
    parser.add_argument("--ifeval-python", type=Path,
                        default=ROOT / "experiment/ifeval-venv/bin/python")
    parser.add_argument("--nltk-data", type=Path, default=ROOT / "experiment/ifeval-venv/nltk_data")
    parser.add_argument("--input-data", type=Path, default=ROOT / (
        "experiment/ifeval-src/instruction_following_eval/data/input_data.jsonl"))
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_ifeval")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    args = parser.parse_args()
    for key in ("server", "llama_dir", "ifeval_source", "nltk_data", "input_data", "work",
                "reports"):
        setattr(args, key, getattr(args, key).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    if args.command in ("run", "score") and args.label is None:
        parser.error(f"{args.command} needs --label")
    if args.command == "run":
        if args.model is None:
            parser.error("run needs --model")
        args.model = args.model.resolve()
        return run(args)
    if args.command == "score":
        return score(args)
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
