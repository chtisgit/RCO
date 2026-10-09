#!/usr/bin/env python3
"""Phase 4 chat check 1 of RCO_PLAN_NEW.md: chat generation in llama-server.

Starts the pinned ``llama-server`` with ``--jinja``, so the GGUF's own chat
template is applied, and sends the fixed prompts
(``qwen36_chat_generation_prompts.json``) through ``/v1/chat/completions``
with greedy decoding.  Multi-turn prompts feed the model's own replies back
as history.  ``verbose`` returns the raw generated text and the stop type.

Checks per response:

``stopped``
    Generation ended on an end-of-generation token (``<|im_end|>``), not on
    the token limit.
``think_closed`` (thinking on)
    The raw text closes the ``<think>`` block that the generation prompt
    opens.
``tool_call`` (tool prompts)
    llama-server parsed at least one tool call from the template's
    ``<tool_call>`` format.  Each names a provided tool, its arguments are a
    JSON object, use only that tool's parameter names, and include every
    required one.

The report is rewritten after every prompt, and a restart skips completed
prompts.  A readable Markdown transcript is written next to it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))

from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file  # noqa: E402

SCHEMA = "rco.qwen36.chat_generation.v1"


def _post(url: str, body: dict, timeout: float = 3600) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_ready(base: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as response:
                if json.loads(response.read()).get("status") == "ok":
                    return
        except (urllib.error.URLError, ConnectionError, json.JSONDecodeError):
            pass
        time.sleep(2)
    raise TimeoutError("llama-server did not become ready")


def check_tool_calls(tool_calls: list[dict] | None, tools: list[dict]) -> tuple[bool, str]:
    if not tool_calls:
        return False, "no tool call parsed"
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    for call in tool_calls:
        function = call.get("function", {})
        name = function.get("name")
        if name not in schemas:
            return False, f"unknown tool {name!r}"
        try:
            arguments = json.loads(function.get("arguments", ""))
        except (TypeError, json.JSONDecodeError):
            return False, f"{name}: arguments are not JSON"
        if not isinstance(arguments, dict):
            return False, f"{name}: arguments are not an object"
        allowed = set(schemas[name].get("properties", {}))
        if set(arguments) - allowed:
            return False, f"{name}: unknown arguments {sorted(set(arguments) - allowed)}"
        missing = set(schemas[name].get("required", [])) - set(arguments)
        if missing:
            return False, f"{name}: missing required {sorted(missing)}"
    return True, "ok"


def _transcript(report: dict) -> str:
    lines = [f"# Chat generation: {report['label']} on {report['device']}", "",
             f"Model `{Path(report['identity']['model']['path']).name}`, "
             f"llama.cpp `{report['identity']['llama_cpp_revision']}`, "
             f"{report['identity']['parameters']}.", ""]
    for item in report["responses"]:
        checks = ", ".join(f"{name}: {'pass' if ok else 'FAIL'}"
                           for name, ok in item["checks"].items())
        lines += [f"## {item['prompt_id']}, turn {item['turn'] + 1}", "",
                  f"*{item['kind']}, thinking {'on' if item['thinking'] else 'off'}; "
                  f"stop `{item['stop_type']}`, {item['tokens_predicted']} tokens, "
                  f"{item['tokens_per_second']:.1f} tok/s; {checks}*", "",
                  "**User:** " + item["user"], "", "**Raw output:**", "", "````",
                  item["raw_content"], "````", ""]
        if item["tool_calls"]:
            lines += ["**Parsed tool calls:**", "", "```json",
                      json.dumps(item["tool_calls"], indent=2, ensure_ascii=False), "```", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", required=True, choices=("cpu", "gpu"))
    parser.add_argument("--gpu-layers", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--server", type=Path,
                        default=ROOT / "experiment/build-cuda/bin/llama-server")
    parser.add_argument("--llama-dir", type=Path, default=ROOT / "repos/llama.cpp")
    parser.add_argument("--prompts", type=Path,
                        default=RCO / "tools/qwen36_chat_generation_prompts.json")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--only", help="comma-separated prompt ids (smoke tests)")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    stem = f"qwen36_chat_generation_{args.label}_{args.device}{args.suffix}"
    output = args.reports / f"{stem}.json"
    prompts = _load_json(args.prompts)
    selected = [p for p in prompts["prompts"]
                if not args.only or p["id"] in args.only.split(",")]
    revision = subprocess.run(["git", "-C", str(args.llama_dir), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    identity = {
        "model": {"path": str(args.model.resolve()), "sha256": _sha256_file(args.model)},
        "server": {"path": str(args.server), "sha256": _sha256_file(args.server)},
        "llama_cpp_revision": revision,
        "prompts_sha256": _sha256_file(args.prompts),
        "parameters": {"gpu_layers": args.gpu_layers, "threads": args.threads,
                       "context": args.context,
                       "devices": "none" if args.device == "cpu" else "default"},
    }
    report = {"schema": SCHEMA, "status": "incomplete", "label": args.label,
              "device": args.device, "identity": identity, "responses": []}
    if args.device == "cpu" and args.gpu_layers != 0:
        parser.error("a CPU run needs --gpu-layers 0")
    if output.exists():
        previous = _load_json(output)
        if previous.get("identity") != identity:
            raise RuntimeError("existing report identity does not match this run")
        if previous.get("status") == "complete":
            print(f"{stem}: already complete", flush=True)
            return 0
        report = previous
    done = {item["prompt_id"] for item in report["responses"]}
    # Drop a partly finished multi-turn prompt; it is rerun from turn 1.
    complete = {p["id"] for p in selected if p["id"] in done and
                sum(r["prompt_id"] == p["id"] for r in report["responses"]) == len(p["turns"])}
    report["responses"] = [r for r in report["responses"] if r["prompt_id"] in complete]

    base = f"http://127.0.0.1:{args.port}"
    log = (ROOT / f"{stem}.server.log").open("w")
    # A CPU run uses no device at all: no layer offload and no op offload.
    devices = ["--device", "none"] if args.device == "cpu" else []
    process = subprocess.Popen(
        [str(args.server), "-m", str(args.model), "--jinja", "-ngl", str(args.gpu_layers),
         *devices, "-t", str(args.threads), "-c", str(args.context), "-np", "1",
         "--host", "127.0.0.1", "--port", str(args.port), "--no-webui"],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        _wait_ready(base, process, timeout=900)
        started = time.perf_counter()
        for prompt in selected:
            if prompt["id"] in complete:
                continue
            tools = prompt.get("tools")
            max_tokens = prompts["max_tokens"]["thinking_on" if prompt["thinking"]
                                               else "thinking_off"]
            messages: list[dict[str, Any]] = []
            for turn, user in enumerate(prompt["turns"]):
                messages.append({"role": "user", "content": user})
                body = {"messages": messages,
                        "chat_template_kwargs": {"enable_thinking": prompt["thinking"]},
                        "temperature": 0, "seed": prompts["decoding"]["seed"],
                        "max_tokens": max_tokens, "verbose": True, "return_tokens": True}
                if tools:
                    body["tools"] = tools
                rendered = _post(base + "/apply-template", body).get("prompt")
                response = _post(base + "/v1/chat/completions", body)
                choice = response["choices"][0]
                message = choice["message"]
                verbose = response.get("__verbose", {})
                raw = verbose.get("content", "")
                checks = {"stopped": verbose.get("stop_type") == "eos"}
                notes = {}
                if prompt["thinking"]:
                    checks["think_closed"] = "</think>" in raw
                if prompt["kind"] == "tool":
                    checks["tool_call"], notes["tool_call"] = check_tool_calls(
                        message.get("tool_calls"), tools)
                timings = response.get("timings", {})
                report["responses"].append({
                    "prompt_id": prompt["id"], "kind": prompt["kind"],
                    "thinking": prompt["thinking"], "turn": turn, "user": user,
                    "rendered_prompt_sha256": hashlib.sha256(
                        (rendered or "").encode("utf-8")).hexdigest(),
                    "finish_reason": choice.get("finish_reason"),
                    "stop_type": verbose.get("stop_type"),
                    "tokens_predicted": verbose.get("tokens_predicted"),
                    "tokens_per_second": timings.get("predicted_per_second", 0.0),
                    "raw_content": raw,
                    "content": message.get("content"),
                    "reasoning_content": message.get("reasoning_content"),
                    "tool_calls": message.get("tool_calls"),
                    "checks": checks, "notes": notes,
                })
                print(f"{stem}: {prompt['id']} turn {turn + 1}: "
                      f"{verbose.get('tokens_predicted')} tokens, stop "
                      f"{verbose.get('stop_type')}, {checks}", flush=True)
                if message.get("tool_calls"):
                    break  # no tool results are supplied, so the turn ends here
                messages.append({"role": "assistant", "content": message.get("content") or ""})
            _atomic_json(output, report)
        checks = [ok for item in report["responses"] for ok in item["checks"].values()]
        report["status"] = "complete"
        report["summary"] = {"responses": len(report["responses"]),
                             "checks": len(checks), "failed_checks": checks.count(False),
                             "wall_seconds": time.perf_counter() - started}
        _atomic_json(output, report)
        (args.reports / f"{stem}.md").write_text(_transcript(report), encoding="utf-8")
        print(f"{stem}: complete, {checks.count(False)} of {len(checks)} checks failed",
              flush=True)
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
