#!/usr/bin/env python3
"""Phase 9 of RCO_PLAN_NEW.md: the tool restraint check.

Every prompt (``qwen36_tool_restraint_prompts.json``) offers tools that do
not fit the request.  The model may call one anyway; the harness answers
each call with the tool's fixed result and lets the model continue, until
it replies without a tool call or the round limit is reached.

Subcommands:

``run --model GGUF --label L``
    The prompts x seeds in ``llama-server --jinja`` with the model card's
    sampling per mode (the setup of ``audit_qwen36_chat_generation.py``).
    Resumable per conversation.
``validate``
    On the Q6_K reference's run: drop a prompt where the reference calls a
    tool in 3 or more of the 5 seeds, or fails the answer check in any seed.
    Fewer than 12 kept prompts is a decision point (exit 4).
``compare``
    The candidate against authentic GSQ on the kept prompts: pass if the
    candidate fails no prompt on more seeds than GSQ does (exit 3 if not).

A conversation fails on
``R1``  a second tool round, or two or more calls in one reply;
``R2``  after any tool call, a final answer that fails its check or holds a
        distractor value from a tool result;
``no answer``  no plain-text reply stopped on the end-of-turn token.
A single unnecessary call followed by a correct, clean answer is recorded,
not a failure; so is a wrong answer without any tool call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))

from audit_qwen36_chat_generation import _post, _wait_ready, check_tool_calls  # noqa: E402
from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file  # noqa: E402

SCHEMA = "rco.qwen36.tool_restraint.v1"
DROP_CALLING_SEEDS = 3
MINIMUM_KEPT = 12
_NUMBER = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")


# -- tool results and distractors ---------------------------------------------------


def _render(template: Any, arguments: dict[str, Any]) -> Any:
    if isinstance(template, dict):
        if "$multiply" in template:
            a, b = (_render(item, arguments) for item in template["$multiply"])
            try:
                return round(float(a) * float(b), template.get("$round", 2))
            except (TypeError, ValueError):
                return "unknown"
        return {key: _render(value, arguments) for key, value in template.items()}
    if isinstance(template, list):
        return [_render(item, arguments) for item in template]
    if isinstance(template, str):
        whole = re.fullmatch(r"\{(\w+)\}", template)
        if whole:
            return arguments.get(whole.group(1), "unknown")
        return re.sub(r"\{(\w+)\}", lambda m: str(arguments.get(m.group(1), "unknown")), template)
    return template


def tool_result(manifest: dict, prompt: dict, name: str, arguments: dict[str, Any]) -> tuple[dict, list[str]]:
    """The result a call gets, and its distractor values (none if the
    prompt overrides the tool with the user's own data)."""
    overrides = prompt.get("tool_result_overrides", {})
    if name not in manifest["tool_results"]:
        return {"error": f"unknown tool {name}"}, []
    result = _render(overrides.get(name, manifest["tool_results"][name]), arguments)
    if name in overrides:
        return result, []
    distractors = []
    for item in manifest["distractors"].get(name, []):
        if item.startswith("$"):
            value = result.get(item[1:]) if isinstance(result, dict) else None
            if isinstance(value, (int, float)):
                distractors.append(repr(value))
        else:
            distractors.append(item)
    return result, distractors


def _numbers(text: str) -> list[float]:
    return [float(match.replace(",", "")) for match in _NUMBER.findall(text)]


def find_distractors(answer: str, distractors: list[str]) -> list[str]:
    """Distractors in the answer: numbers numerically, text as a phrase."""
    found = []
    numbers = _numbers(answer)
    for item in distractors:
        try:
            value = float(item)
        except ValueError:
            if re.search(r"(?<!\w)" + re.escape(item) + r"(?!\w)", answer, re.IGNORECASE):
                found.append(item)
            continue
        if any(abs(abs(number) - abs(value)) < 1e-9 for number in numbers):
            found.append(item)
    return found


# -- answer checks ------------------------------------------------------------------


def check_answer(check: dict, answer: str) -> bool:
    text = answer.strip()
    kind = check["type"]
    if kind == "nonempty":
        return bool(text)
    if kind == "contains_any":
        return any(re.search(r"(?<!\w)" + re.escape(item) + r"(?!\w)", text, re.IGNORECASE)
                   for item in check["accept"])
    if kind == "starts_with_any":
        head = re.sub(r"^[\s*_#>\"'`\-]+", "", text).lower()
        return any(re.match(re.escape(item.lower()) + r"(?!\w)", head) for item in check["accept"])
    raise ValueError(f"unknown check type {kind}")


def score_conversation(prompt: dict, replies: list[dict], round_limit: int) -> dict[str, Any]:
    """R1, R2 and no-answer for one conversation from its recorded replies."""
    rounds = [reply for reply in replies if reply["tool_calls"]]
    final = replies[-1] if replies and not replies[-1]["tool_calls"] else None
    answer = (final or {}).get("content") or ""
    distractors = sorted({d for reply in rounds for call in reply["results"]
                          for d in call["distractors"]})
    found = find_distractors(answer, distractors) if final else []
    check_passed = bool(final) and check_answer(prompt["check"], answer)
    no_answer = final is None or final["stop_type"] != "eos" or not answer.strip()
    r1 = len(rounds) >= 2 or any(len(reply["tool_calls"]) >= 2 for reply in rounds)
    r2 = bool(rounds) and not no_answer and (not check_passed or bool(found))
    return {
        "tool_rounds": len(rounds),
        "tool_calls": sum(len(reply["tool_calls"]) for reply in rounds),
        "tools_called": [call["name"] for reply in rounds for call in reply["results"]],
        "round_limit_reached": len(rounds) >= round_limit and final is None,
        "final_answer": answer,
        "check_passed": check_passed,
        "distractors_found": found,
        "r1_repeated": r1, "r2_influence": r2, "no_answer": no_answer,
        "failed": r1 or r2 or no_answer,
        "unnecessary_call": bool(rounds),
        "wrong_without_tool": not rounds and not no_answer and not check_passed,
    }


# -- run ------------------------------------------------------------------------------


def _settings(manifest: dict, prompt: dict) -> dict:
    decoding = _load_json(RCO / manifest["decoding_source"])["decoding"]
    mode = "thinking_on" if prompt["thinking"] else "thinking_off"
    return {**decoding[mode], "max_tokens": decoding["max_tokens"]}


def run_conversation(base: str, manifest: dict, prompt: dict, seed: int) -> list[dict]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt["turns"][0]}]
    replies: list[dict] = []
    for _ in range(manifest["round_limit"] + 1):
        body = {"messages": messages, "tools": prompt["tools"],
                "chat_template_kwargs": {"enable_thinking": prompt["thinking"]},
                **_settings(manifest, prompt), "seed": seed, "verbose": True}
        response = _post(base + "/v1/chat/completions", body)
        choice = response["choices"][0]
        message = choice["message"]
        verbose = response.get("__verbose", {})
        calls = message.get("tool_calls") or []
        reply = {"raw_content": verbose.get("content", ""), "content": message.get("content"),
                 "reasoning_content": message.get("reasoning_content"), "tool_calls": calls,
                 "stop_type": verbose.get("stop_type"), "finish_reason": choice.get("finish_reason"),
                 "tokens_predicted": verbose.get("tokens_predicted"),
                 "tokens_per_second": response.get("timings", {}).get("predicted_per_second", 0.0),
                 "results": []}
        if calls:
            reply["calls_valid"], reply["calls_note"] = check_tool_calls(calls, prompt["tools"])
        replies.append(reply)
        if not calls:
            break
        if sum(1 for r in replies if r["tool_calls"]) >= manifest["round_limit"]:
            break
        assistant = {"role": "assistant", "content": message.get("content") or "",
                     "tool_calls": calls}
        if message.get("reasoning_content"):
            assistant["reasoning_content"] = message["reasoning_content"]
        messages.append(assistant)
        for number, call in enumerate(calls):
            function = call.get("function", {})
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            result, distractors = tool_result(manifest, prompt, function.get("name"), arguments)
            reply["results"].append({"name": function.get("name"), "arguments": arguments,
                                     "result": result, "distractors": distractors})
            messages.append({"role": "tool", "tool_call_id": call.get("id") or f"call_{number}",
                             "content": json.dumps(result, ensure_ascii=False)})
    return replies


def _transcript(report: dict) -> str:
    lines = [f"# Tool restraint: {report['label']}", "",
             f"Model `{Path(report['identity']['model']['path']).name}`, "
             f"{report['identity']['parameters']}.", ""]
    for item in report["conversations"]:
        s = item["score"]
        flags = [name for name in ("r1_repeated", "r2_influence", "no_answer") if s[name]]
        lines += [f"## {item['prompt_id']}, seed {item['seed']}", "",
                  f"*{item['kind']}, {item['offer']}, thinking "
                  f"{'on' if item['thinking'] else 'off'}; {s['tool_rounds']} tool rounds, "
                  f"check {'pass' if s['check_passed'] else 'FAIL'}"
                  + (f", distractors {s['distractors_found']}" if s["distractors_found"] else "")
                  + (f"; **FAILED: {', '.join(flags)}**" if flags else "") + "*", "",
                  "**User:** " + item["user"], ""]
        for number, reply in enumerate(item["replies"]):
            lines += [f"**Reply {number + 1}** (stop `{reply['stop_type']}`, "
                      f"{reply['tokens_predicted']} tokens):", "", "````", reply["raw_content"], "````", ""]
            for call in reply["results"]:
                lines += [f"Tool `{call['name']}` {json.dumps(call['arguments'], ensure_ascii=False)} "
                          f"→ `{json.dumps(call['result'], ensure_ascii=False)}`", ""]
    return "\n".join(lines)


def run(args) -> int:
    manifest = _load_json(args.prompts)
    selected = manifest["prompts"]
    if args.kept_from:
        kept = _load_json(args.kept_from)
        if kept.get("status") != "pass" or kept["prompts_sha256"] != _sha256_file(args.prompts):
            raise RuntimeError("the validation report has not passed for this prompt file")
        selected = [p for p in selected if p["id"] in kept["kept"]]
    if args.only:
        selected = [p for p in selected if p["id"] in args.only.split(",")]
    stem = f"qwen36_tool_restraint_{args.label}{args.suffix}"
    output = args.reports / f"{stem}.json"
    revision = subprocess.run(["git", "-C", str(args.llama_dir), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    identity = {
        "model": {"path": str(args.model.resolve()), "sha256": _sha256_file(args.model)},
        "server": {"path": str(args.server), "sha256": _sha256_file(args.server)},
        "llama_cpp_revision": revision, "prompts_sha256": _sha256_file(args.prompts),
        "prompt_ids": [p["id"] for p in selected], "seeds": manifest["seeds"],
        "parameters": {"gpu_layers": args.gpu_layers, "n_cpu_moe": args.n_cpu_moe,
                       "threads": args.threads, "context": args.context},
    }
    report = {"schema": SCHEMA, "status": "incomplete", "label": args.label,
              "identity": identity, "conversations": []}
    if output.exists():
        previous = _load_json(output)
        if previous.get("identity") != identity:
            raise RuntimeError(f"{output} describes another run; refusing to resume")
        if previous.get("status") == "complete":
            print(f"{stem}: already complete", flush=True)
            return 0
        report = previous
    complete = {(c["prompt_id"], c["seed"]) for c in report["conversations"]}
    base = f"http://127.0.0.1:{args.port}"
    log = (ROOT / f"{stem}.server.log").open("w")
    command = [str(args.server), "-m", str(args.model), "--jinja", "-ngl", str(args.gpu_layers),
               "-t", str(args.threads), "-c", str(args.context), "-np", "1",
               "--host", "127.0.0.1", "--port", str(args.port), "--no-webui"]
    if args.n_cpu_moe:
        command[6:6] = ["--n-cpu-moe", str(args.n_cpu_moe)]
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        _wait_ready(base, process, timeout=1800)
        started = time.perf_counter()
        for seed in manifest["seeds"]:
            for prompt in selected:
                if (prompt["id"], seed) in complete:
                    continue
                replies = run_conversation(base, manifest, prompt, seed)
                score = score_conversation(prompt, replies, manifest["round_limit"])
                report["conversations"].append({
                    "prompt_id": prompt["id"], "seed": seed, "kind": prompt["kind"],
                    "offer": prompt["offer"], "thinking": prompt["thinking"],
                    "user": prompt["turns"][0], "replies": replies, "score": score})
                _atomic_json(output, report)
                print(f"{stem}: {prompt['id']} seed {seed}: {score['tool_rounds']} rounds, "
                      f"check {score['check_passed']}, failed {score['failed']}, "
                      f"{sum(r['tokens_predicted'] or 0 for r in replies)} tokens", flush=True)
        scores = [c["score"] for c in report["conversations"]]
        report["status"] = "complete"
        report["summary"] = {
            "conversations": len(scores),
            "failed": sum(s["failed"] for s in scores),
            "unnecessary_call": sum(s["unnecessary_call"] for s in scores),
            "wrong_without_tool": sum(s["wrong_without_tool"] for s in scores),
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


# -- validate and compare --------------------------------------------------------------


def _per_prompt(report: dict) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for item in report["conversations"]:
        result.setdefault(item["prompt_id"], []).append(item["score"])
    return result


def validate(args) -> int:
    reference = _load_json(args.reports / f"qwen36_tool_restraint_{args.reference_label}.json")
    if reference.get("status") != "complete":
        raise RuntimeError("the reference run is not complete")
    if reference["identity"]["prompts_sha256"] != _sha256_file(args.prompts):
        raise RuntimeError("the reference ran another prompt file")
    rows, kept = {}, []
    for prompt_id, scores in _per_prompt(reference).items():
        calling = sum(s["unnecessary_call"] for s in scores)
        check_failures = sum(not s["check_passed"] for s in scores)
        reasons = []
        if calling >= DROP_CALLING_SEEDS:
            reasons.append(f"reference calls a tool in {calling} of {len(scores)} seeds")
        if check_failures:
            reasons.append(f"reference fails the answer check in {check_failures} seeds")
        rows[prompt_id] = {"calling_seeds": calling, "check_failures": check_failures,
                           "seeds": len(scores), "dropped": bool(reasons), "reasons": reasons}
        if not reasons:
            kept.append(prompt_id)
    status = "pass" if len(kept) >= MINIMUM_KEPT else "decision"
    report = {"schema": SCHEMA + ".validation", "status": status,
              "rule": (f"drop a prompt where the reference calls a tool in >= {DROP_CALLING_SEEDS} "
                       f"seeds or fails the answer check in any seed; >= {MINIMUM_KEPT} must remain"),
              "reference": reference["identity"]["model"],
              "prompts_sha256": _sha256_file(args.prompts),
              "kept": kept, "kept_count": len(kept), "prompts": rows,
              "reference_summary": reference["summary"]}
    _atomic_json(args.reports / f"qwen36_tool_restraint_validation_{args.reference_label}.json", report)
    print(f"validation: {status}, kept {len(kept)} of {len(rows)}: "
          + "; ".join(f"{p} dropped ({', '.join(r['reasons'])})" for p, r in rows.items() if r["dropped"]),
          flush=True)
    return 0 if status == "pass" else 4


def compare(args) -> int:
    validation = _load_json(args.reports / f"qwen36_tool_restraint_validation_{args.reference_label}.json")
    if validation.get("status") != "pass":
        raise RuntimeError("the prompt validation has not passed")
    arms = {}
    for label in (args.candidate_label, args.control_label):
        report = _load_json(args.reports / f"qwen36_tool_restraint_{label}.json")
        if report.get("status") != "complete" or report["identity"]["prompts_sha256"] != validation["prompts_sha256"]:
            raise RuntimeError(f"{label}: run not complete or of another prompt file")
        if sorted(report["identity"]["prompt_ids"]) != sorted(validation["kept"]):
            raise RuntimeError(f"{label}: ran other prompts than the kept set")
        arms[label] = _per_prompt(report)
    rows, failing = {}, []
    for prompt_id in validation["kept"]:
        counts = {label: {"failed": sum(s["failed"] for s in arms[label][prompt_id]),
                          "r1": sum(s["r1_repeated"] for s in arms[label][prompt_id]),
                          "r2": sum(s["r2_influence"] for s in arms[label][prompt_id]),
                          "no_answer": sum(s["no_answer"] for s in arms[label][prompt_id]),
                          "unnecessary_call": sum(s["unnecessary_call"] for s in arms[label][prompt_id]),
                          "wrong_without_tool": sum(s["wrong_without_tool"] for s in arms[label][prompt_id])}
                  for label in arms}
        rows[prompt_id] = counts
        if counts[args.candidate_label]["failed"] > counts[args.control_label]["failed"]:
            failing.append(prompt_id)
    totals = {label: {key: sum(rows[p][label][key] for p in rows) for key in rows[validation["kept"][0]][label]}
              for label in arms}
    status = "fail" if failing else "pass"
    _atomic_json(args.reports / f"qwen36_tool_restraint_comparison_{args.candidate_label}_vs_{args.control_label}.json", {
        "schema": SCHEMA + ".comparison", "status": status,
        "rule": (f"{args.candidate_label} fails no kept prompt on more seeds than "
                 f"{args.control_label} does"),
        "prompts_failing_rule": failing, "totals": totals, "per_prompt": rows,
        "validation_sha256": _sha256_file(
            args.reports / f"qwen36_tool_restraint_validation_{args.reference_label}.json")})
    print(f"tool restraint: {status}; totals {totals}; failing prompts {failing}", flush=True)
    return 0 if status == "pass" else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "validate", "compare"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--gpu-layers", type=int, default=20)
    parser.add_argument("--n-cpu-moe", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--server", type=Path,
                        default=ROOT / "experiment/build-cuda-mmqfix/bin/llama-server")
    parser.add_argument("--llama-dir", type=Path, default=ROOT / "experiment/llama.cpp-mmqfix")
    parser.add_argument("--prompts", type=Path,
                        default=RCO / "tools/qwen36_tool_restraint_prompts.json")
    parser.add_argument("--kept-from", type=Path, help="validation report; run its kept prompts")
    parser.add_argument("--reference-label", default="q6k_reference")
    parser.add_argument("--candidate-label", default="rco_v2")
    parser.add_argument("--control-label", default="gsq")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--only", help="comma-separated prompt ids (smoke tests)")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    if args.command == "run":
        if args.model is None or args.label is None:
            parser.error("run needs --model and --label")
        return run(args)
    return validate(args) if args.command == "validate" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
