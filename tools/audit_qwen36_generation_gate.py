#!/usr/bin/env python3
"""Phase 9 generation gate of RCO_PLAN_NEW.md.

The 32 prompts of manifest v2 (``reports/qwen36_35b_generation_prompt_manifest_v2.json``)
as single user turns in ``llama-server --jinja``, thinking off and on, the
manifest's 3 seeds and the model card's sampling per mode: 192 generations
per model.  The assertion (``src/generation_assertions.py``) applies to the
final answer, the content after ``</think>``.

Subcommands:

``run --model GGUF --label L [--seeds S,...] [--suffix X]``
    Resumable per generation.  ``--seeds 20261010 --suffix _repeat`` is the
    candidate's repeat run.
``compare``
    Pass if every candidate generation is well-formed, the candidate fails
    no prompt x mode on more seeds than GSQ does, and the repeat reproduces
    the candidate's seed-20261010 outputs exactly (exit 3 if not).

Well-formed: stopped on the end-of-turn token, closes ``<think>`` with
thinking on, no U+FFFD replacement character, and a non-empty answer.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_chat_generation import _post, _wait_ready  # noqa: E402
from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file  # noqa: E402
from generation_assertions import evaluate_generation_assertion  # noqa: E402

SCHEMA = "rco.qwen36.generation_gate.v1"
MODES = (False, True)


def well_formed(thinking: bool, raw: str, answer: str, stop_type: str | None) -> dict[str, bool]:
    checks = {"stopped": stop_type == "eos",
              "no_replacement_character": "�" not in raw,
              "nonempty_answer": bool(answer.strip())}
    if thinking:
        checks["think_closed"] = "</think>" in raw
    return checks


def score(prompt: dict, thinking: bool, raw: str, answer: str, stop_type: str | None) -> dict:
    form = well_formed(thinking, raw, answer, stop_type)
    assertion = evaluate_generation_assertion(prompt["assertion"], answer)
    return {"well_formed": all(form.values()), "form_checks": form,
            "correct": bool(assertion["passed"]), "assertion": assertion}


def _manifest(args) -> tuple[dict, str]:
    report = _load_json(args.manifest)
    return report["canonical_manifest"], report["canonical_manifest_sha256"]


def _settings(decoding: dict, thinking: bool) -> dict:
    return {**decoding["thinking_on" if thinking else "thinking_off"],
            "max_tokens": decoding["maximum_generated_tokens"]}


def _transcript(report: dict) -> str:
    lines = [f"# Generation gate: {report['label']}", "",
             f"Model `{Path(report['identity']['model']['path']).name}`, "
             f"{report['identity']['parameters']}.", ""]
    for item in report["generations"]:
        s = item["score"]
        flags = [k for k, v in s["form_checks"].items() if not v]
        lines += [f"## {item['prompt_id']}, thinking {'on' if item['thinking'] else 'off'}, "
                  f"seed {item['seed']}", "",
                  f"*{'correct' if s['correct'] else '**WRONG**'}; stop `{item['stop_type']}`, "
                  f"{item['tokens_predicted']} tokens"
                  + (f"; **malformed: {', '.join(flags)}**" if flags else "") + "*", "",
                  "**User:** " + item["user"], "", "````", item["raw_content"], "````", ""]
    return "\n".join(lines)


def run(args) -> int:
    manifest, manifest_sha = _manifest(args)
    decoding = manifest["decoding"]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else decoding["seeds"]
    stem = f"qwen36_generation_gate_{args.label}{args.suffix}"
    output = args.reports / f"{stem}.json"
    revision = subprocess.run(["git", "-C", str(args.llama_dir), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    identity = {
        "model": {"path": str(args.model), "sha256": _sha256_file(args.model)},
        "server": {"path": str(args.server), "sha256": _sha256_file(args.server)},
        "llama_cpp_revision": revision, "manifest_sha256": manifest_sha, "seeds": seeds,
        "only": args.only,
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
            print(f"{stem}: already complete", flush=True)
            return 0
        report = previous
    complete = {(g["prompt_id"], g["thinking"], g["seed"]) for g in report["generations"]}
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
        for seed in seeds:
            for thinking in MODES:
                for prompt in manifest["prompts"]:
                    if (prompt["id"], thinking, seed) in complete \
                            or (args.only and prompt["id"] not in args.only.split(",")):
                        continue
                    body = {"messages": [{"role": "user", "content": prompt["prompt"]}],
                            "chat_template_kwargs": {"enable_thinking": thinking},
                            **_settings(decoding, thinking), "seed": seed, "verbose": True}
                    response = _post(base + "/v1/chat/completions", body)
                    choice = response["choices"][0]
                    message = choice["message"]
                    verbose = response.get("__verbose", {})
                    raw = verbose.get("content", "")
                    answer = message.get("content") or ""
                    item = {"prompt_id": prompt["id"], "category": prompt["category"],
                            "thinking": thinking, "seed": seed, "user": prompt["prompt"],
                            "raw_content": raw, "content": answer,
                            "reasoning_content": message.get("reasoning_content") or "",
                            "stop_type": verbose.get("stop_type"),
                            "finish_reason": choice.get("finish_reason"),
                            "tokens_predicted": verbose.get("tokens_predicted"),
                            "tokens_per_second": response.get("timings", {}).get(
                                "predicted_per_second", 0.0),
                            "score": score(prompt, thinking, raw, answer, verbose.get("stop_type"))}
                    report["generations"].append(item)
                    _atomic_json(output, report)
                    s = item["score"]
                    print(f"{stem}: {prompt['id']} thinking {'on' if thinking else 'off'} "
                          f"seed {seed}: correct {s['correct']}, well-formed {s['well_formed']}, "
                          f"{item['tokens_predicted']} tokens", flush=True)
        scores = [g["score"] for g in report["generations"]]
        report["status"] = "complete"
        report["summary"] = {"generations": len(scores),
                             "correct": sum(s["correct"] for s in scores),
                             "well_formed": sum(s["well_formed"] for s in scores),
                             "wall_seconds_this_invocation": time.perf_counter() - started}
        _atomic_json(output, report)
        (args.reports / f"{stem}.md").write_text(_transcript(report), encoding="utf-8")
        print(f"{stem}: complete, {report['summary']}", flush=True)
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
        log.close()
    return 0


def _cells(report: dict) -> dict[tuple[str, bool], list[dict]]:
    cells: dict[tuple[str, bool], list[dict]] = {}
    for item in report["generations"]:
        cells.setdefault((item["prompt_id"], item["thinking"]), []).append(item)
    return cells


def compare_reports(candidate: dict, base: dict, repeat: dict) -> dict[str, Any]:
    for report in (candidate, base, repeat):
        if report.get("status") != "complete":
            raise RuntimeError(f"{report.get('label')} is not complete")
    if candidate["identity"]["manifest_sha256"] != base["identity"]["manifest_sha256"] \
            or candidate["identity"]["parameters"] != base["identity"]["parameters"]:
        raise RuntimeError("the candidate and GSQ ran different manifests or settings")
    malformed = [{"prompt_id": g["prompt_id"], "thinking": g["thinking"], "seed": g["seed"],
                  "failed": [k for k, v in g["score"]["form_checks"].items() if not v]}
                 for g in candidate["generations"] if not g["score"]["well_formed"]]
    per_cell: dict[str, dict[str, int]] = {}
    worse: list[str] = []
    candidate_cells, base_cells = _cells(candidate), _cells(base)
    for key in sorted(candidate_cells):
        name = f"{key[0]} thinking {'on' if key[1] else 'off'}"
        c = sum(not g["score"]["correct"] for g in candidate_cells[key])
        b = sum(not g["score"]["correct"] for g in base_cells[key])
        per_cell[name] = {"rco_v2": c, "gsq": b}
        if c > b:
            worse.append(name)
    first = {(g["prompt_id"], g["thinking"]): g for g in candidate["generations"]
             if g["seed"] in repeat["identity"]["seeds"]}
    differing = [f"{g['prompt_id']} thinking {'on' if g['thinking'] else 'off'}"
                 for g in repeat["generations"]
                 if (g["raw_content"], g["content"], g["reasoning_content"]) != (
                     first[(g["prompt_id"], g["thinking"])]["raw_content"],
                     first[(g["prompt_id"], g["thinking"])]["content"],
                     first[(g["prompt_id"], g["thinking"])]["reasoning_content"])]
    checks = {"candidate_well_formed": not malformed, "correct_relative_to_gsq": not worse,
              "repeat_identical": not differing}

    def totals(report):
        scores = [g["score"] for g in report["generations"]]
        return {"generations": len(scores), "wrong": sum(not s["correct"] for s in scores),
                "malformed": sum(not s["well_formed"] for s in scores)}

    return {"schema": SCHEMA + ".comparison", "status": "pass" if all(checks.values()) else "fail",
            "rule": "every rco_v2 generation well-formed; rco_v2 fails no prompt x mode on more "
                    "seeds than gsq; the repeat reproduces seed 20261010 exactly",
            "checks": checks, "totals": {"rco_v2": totals(candidate), "gsq": totals(base)},
            "absolute_rule_all_correct": {"rco_v2": totals(candidate)["wrong"] == 0,
                                          "gsq": totals(base)["wrong"] == 0},
            "malformed_candidate": malformed, "cells_worse_than_gsq": worse,
            "repeat_differing": differing, "per_cell_wrong_seeds": per_cell,
            "manifest_sha256": candidate["identity"]["manifest_sha256"]}


def compare(args) -> int:
    load = lambda label: _load_json(args.reports / f"qwen36_generation_gate_{label}.json")  # noqa: E731
    result = compare_reports(load("rco_v2"), load("gsq"), load("rco_v2_repeat"))
    _atomic_json(args.reports / "qwen36_generation_gate_comparison_rco_v2_vs_gsq.json", result)
    print(f"generation gate: {result['status']}; checks {result['checks']}; totals "
          f"{result['totals']}; worse cells {result['cells_worse_than_gsq']}; repeat differing "
          f"{result['repeat_differing']}", flush=True)
    return 0 if result["status"] == "pass" else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "compare"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--seeds", help="comma-separated subset of the manifest's seeds")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--only", help="comma-separated prompt ids (smoke tests)")
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
    if args.command == "run":
        if args.model is None or args.label is None:
            parser.error("run needs --model and --label")
        args.model = args.model.resolve()
        return run(args)
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
