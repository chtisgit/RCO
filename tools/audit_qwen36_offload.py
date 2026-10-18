#!/usr/bin/env python3
"""Phase 9 CUDA/offload checks of RCO_PLAN_NEW.md.

Placements (each for the candidate and for authentic GSQ):

  P0  CPU only (``-ngl 0 --device none``), the agreement reference
  P1  ``-ngl 20``, the placement of every earlier gate
  P2  ``-ngl 99 --n-cpu-moe 40``, all experts on the CPU
  P3  ``-ngl 99 --n-cpu-moe N``, the smallest N where both models load with
      context 65,536 (N = 40, 36, 32, ... downward)
  P4  ``-ngl 99``, context 8,192: full offload, recorded whether it fits

Stages, each stored in the report as it finishes (the run resumes):

  p3           find N for P3, and whether P4 fits
  agreement    KL(P0 || Pk) over the BF16-scored chat v2 calibration
               conversations (first 3 by id per stratum), with the
               placement helper ``tools/gguf_chat_topk.cpp``
  lengths      single decodes of 2..512 prompt tokens: no crash
  determinism  5 manifest v2 prompts generated twice with one seed
  bench        ``llama bench`` pp512 / tg128 (information)
  defaults     llama-server with no placement arguments (information)

``report`` applies the pass rules and writes the final summary.
Every server runs with ``--fit off`` so that the placement is exactly the
one recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_chat_generation import _post, _wait_ready  # noqa: E402
from audit_qwen36_chat_kl_gguf import _corpus, write_bundle  # noqa: E402
from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file  # noqa: E402

SCHEMA = "rco.qwen36.cuda_offload.v1"
MODELS = {
    "rco_v2": ROOT / "results/Qwen3.6-35B-A3B-GSQ-RCO-v2-UNGATED.gguf",
    "gsq": ROOT / "results/Qwen3.6-35B-A3B-GSQ-hybrid.gguf",
}
BUILDS = {
    "fixed": {"source": ROOT / "experiment/llama.cpp-mmqfix",
              "bin": ROOT / "experiment/build-cuda-mmqfix/bin",
              "helper": ROOT / "experiment/helpers-mmqfix/rco-gguf-chat-topk"},
    "upstream": {"source": ROOT / "experiment/llama.cpp-upstream",
                 "bin": ROOT / "experiment/build-cuda-upstream/bin",
                 "helper": ROOT / "experiment/helpers-upstream/rco-gguf-chat-topk"},
    "pinned": {"source": ROOT / "repos/llama.cpp",
               "bin": ROOT / "experiment/build-cuda/bin", "helper": None},
}
LAYERS = 40
P3_CONTEXT = 65536
P4_CONTEXT = 8192
P3_STEPS = list(range(40, -1, -4))
PER_STRATUM = 3
KL_FACTOR, KL_SLACK = 2.0, 0.001
LENGTHS = range(2, 513)
DETERMINISM_SEED = 20261010
GENERATION_MANIFEST = RCO / "reports/qwen36_35b_generation_prompt_manifest_v2.json"


def placement_args(placement: dict, style: str) -> list[str]:
    """Command-line flags for a placement: ``server`` / ``bench`` / ``helper``."""
    if placement["device_none"]:
        return {"server": ["-ngl", "0", "--device", "none"],
                "bench": ["-ngl", "0", "-dev", "none"],
                "helper": ["--device-none"]}[style]
    layers = ["--gpu-layers" if style == "helper" else "-ngl", str(placement["gpu_layers"])]
    moe = placement.get("n_cpu_moe", 0)
    if moe:
        layers += ["--n-cpu-moe" if style != "bench" else "-ncmoe", str(moe)]
    return layers


def placements(report: dict) -> dict[str, dict]:
    found = {"P0": {"gpu_layers": 0, "device_none": True},
             "P1": {"gpu_layers": 20, "device_none": False},
             "P2": {"gpu_layers": 99, "n_cpu_moe": 40, "device_none": False}}
    p3 = report["stages"].get("p3")
    if p3:
        if p3["n"] is not None:
            found["P3"] = {"gpu_layers": 99, "n_cpu_moe": p3["n"], "device_none": False}
        if p3["p4_fits"]:
            found["P4"] = {"gpu_layers": 99, "device_none": False}
    return found


def _revision(source: Path) -> str:
    return subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def _buffers(text: str) -> dict[str, float]:
    """Model, KV, recurrent-state and compute buffer sizes per device, MiB."""
    sizes: dict[str, float] = {}
    for kind, device, mib in re.findall(
            r"(load_tensors|llama_kv_cache|llama_memory_recurrent|sched_reserve|llama_context):"
            r"\s+(\S+)\s+\w+ buffer size =\s+([\d.]+) MiB", text):
        key = f"{device}:{kind}"
        sizes[key] = sizes.get(key, 0.0) + float(mib)
    return sizes


def _cuda_mib(sizes: dict[str, float]) -> float:
    return round(sum(v for k, v in sizes.items() if k.startswith("CUDA0:")), 2)


def _gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=30).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


class Server:
    """llama-server at one placement; the log is kept for the buffer sizes."""

    def __init__(self, args, build: str, model: str, extra: list[str], context: int,
                 name: str, fit_off: bool = True):
        self.base = f"http://127.0.0.1:{args.port}"
        self.log_path = args.work / f"{name}.server.log"
        command = [str(BUILDS[build]["bin"] / "llama-server"), "-m", str(MODELS[model]),
                   "--jinja", "-t", str(args.threads), "-np", "1", "--host", "127.0.0.1",
                   "--port", str(args.port), "--no-webui", *extra]
        if context:
            command += ["-c", str(context)]
        if fit_off:
            command += ["--fit", "off"]
        self.command = command
        self.log = self.log_path.open("w")
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT,
                                        start_new_session=True)

    def ready(self, timeout: float = 1800) -> bool:
        try:
            _wait_ready(self.base, self.process, timeout=timeout)
            return True
        except (RuntimeError, TimeoutError):
            return False

    def alive(self) -> bool:
        return self.process.poll() is None

    def text(self) -> str:
        self.log.flush()
        return self.log_path.read_text(errors="replace")

    def stop(self) -> None:
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self.log.close()


def _short_completion(server: Server) -> bool:
    try:
        reply = _post(server.base + "/completion",
                      {"prompt": "The capital of France is", "n_predict": 8,
                       "temperature": 0, "cache_prompt": False}, timeout=600)
        return bool(reply.get("content")) and server.alive()
    except Exception:  # noqa: BLE001 - any failure means the placement does not run
        return False


# -- stage p3 -----------------------------------------------------------------------


def stage_p3(args, report: dict) -> dict:
    tries: dict[str, list[dict]] = {model: [] for model in MODELS}
    smallest: dict[str, int | None] = {}
    for model in MODELS:
        smallest[model] = None
        for n in P3_STEPS:
            server = Server(args, "fixed", model, ["-ngl", "99", "--n-cpu-moe", str(n)],
                            P3_CONTEXT, f"p3_{model}_n{n}")
            loaded = server.ready() and _short_completion(server)
            used = _gpu_used_mib() if loaded else None
            sizes = _buffers(server.text())
            server.stop()
            tries[model].append({"n_cpu_moe": n, "runs": loaded, "cuda_buffers_mib": _cuda_mib(sizes),
                                 "nvidia_smi_used_mib": used, "buffers": sizes})
            print(f"p3 {model} n_cpu_moe {n}: {'runs' if loaded else 'fails'}, "
                  f"CUDA {_cuda_mib(sizes)} MiB", flush=True)
            if not loaded:
                break
            smallest[model] = n
    values = [v for v in smallest.values()]
    n = None if any(v is None for v in values) else max(values)
    p4: dict[str, Any] = {}
    for model in MODELS:
        server = Server(args, "fixed", model, ["-ngl", "99"], P4_CONTEXT, f"p4_{model}")
        loaded = server.ready() and _short_completion(server)
        sizes = _buffers(server.text())
        server.stop()
        p4[model] = {"runs": loaded, "cuda_buffers_mib": _cuda_mib(sizes), "buffers": sizes}
        print(f"p4 {model}: {'runs' if loaded else 'does not fit'}", flush=True)
    return {"tries": tries, "smallest_per_model": smallest, "n": n,
            "p4": p4, "p4_fits": all(item["runs"] for item in p4.values())}


# -- stage agreement ------------------------------------------------------------------


def _agreement_conversations(args):
    corpus_args = SimpleNamespace(corpus=args.corpus, split="calibration", limit=None,
                                  reports=args.reports)
    conversations, identity, reference, _ = _corpus(corpus_args)
    chosen: list[int] = []
    for stratum in sorted({c["stratum"] for c in conversations}):
        ids = sorted((c["id"], i) for i, c in enumerate(conversations) if c["stratum"] == stratum)
        chosen += [i for _, i in ids[:PER_STRATUM]]
    return [conversations[i] for i in chosen], [reference[i] for i in chosen], identity


def _helper(args, build: str, model: str, bundle: Path, placement: dict, name: str,
            topk_out: Path | None = None) -> dict:
    lines = args.work / f"{name}.jsonl"
    command = [str(BUILDS[build]["helper"]), "--model", str(MODELS[model]), "--input", str(bundle),
               *placement_args(placement, "helper"), "--threads", str(args.threads),
               "--ubatch", str(args.ubatch)]
    if topk_out is not None:
        command += ["--topk-out", str(topk_out)]
    started = time.perf_counter()
    with lines.open("w") as out, (args.work / f"{name}.err").open("w") as err:
        status = subprocess.run(command, stdout=out, stderr=err).returncode
    stderr = (args.work / f"{name}.err").read_text(errors="replace")
    if status != 0:
        return {"runs": False, "exit_status": status, "stderr_tail": stderr[-2000:]}
    rows = [json.loads(line) for line in lines.read_text().splitlines()
            if '"type":"conversation"' in line]
    return {"runs": True, "seconds": round(time.perf_counter() - started, 1),
            "buffers": _buffers(stderr),
            "per_conversation": {row["id"]: float(np.mean(row["kl"])) for row in rows},
            "mean": float(np.mean([np.mean(row["kl"]) for row in rows]))}


def stage_agreement(args, report: dict, build: str, model: str) -> dict:
    conversations, reference, identity = _agreement_conversations(args)
    bf16_bundle = args.work / "agreement_bf16.bundle"
    write_bundle(bf16_bundle, conversations, reference)
    reference_bundle = args.work / f"agreement_{model}_P0.bundle"
    result: dict[str, Any] = {"conversations": [c["id"] for c in conversations],
                              "corpus": identity, "reference": "fixed build P0"}
    found = placements(report)
    if build == "fixed":
        result["P0_vs_bf16"] = _helper(args, "fixed", model, bf16_bundle, found["P0"],
                                       f"agreement_fixed_{model}_P0", topk_out=reference_bundle)
        names = [p for p in ("P1", "P2", "P3", "P4") if p in found]
    else:
        names = ["P1"]
    if not reference_bundle.exists():
        raise RuntimeError(f"{reference_bundle} is missing: run the fixed build first")
    result["reference_bundle_sha256"] = _sha256_file(reference_bundle)
    for name in names:
        result[name] = {"placement": found[name],
                        "vs_P0": _helper(args, build, model, reference_bundle, found[name],
                                         f"agreement_{build}_{model}_{name}")}
        print(f"agreement {build} {model} {name}: {result[name]['vs_P0'].get('mean')}", flush=True)
    return result


# -- stage lengths --------------------------------------------------------------------


def _length_tokens(args) -> list[int]:
    corpus_args = SimpleNamespace(corpus=args.corpus, split="calibration", limit=None,
                                  reports=args.reports)
    conversations, _, _, _ = _corpus(corpus_args)
    longest = max(conversations, key=lambda c: (len(c["tokens"]), c["id"]))
    return [int(t) for t in longest["tokens"][:max(LENGTHS)]]


def stage_lengths(args, report: dict, build: str, model: str, placement_name: str) -> dict:
    placement = placements(report)[placement_name]
    tokens = _length_tokens(args)
    crashed: list[int] = []
    errors: list[str] = []
    name = f"lengths_{build}_{model}_{placement_name}"
    server = None
    restarts = 0
    for n in LENGTHS:
        if server is None:
            server = Server(args, build, model, placement_args(placement, "server"), 8192,
                            f"{name}_{restarts}")
            if not server.ready():
                server.stop()
                return {"runs": False, "placement": placement, "first_length": n}
        try:
            _post(server.base + "/completion", {"prompt": tokens[:n], "n_predict": 1,
                                                "temperature": 0, "cache_prompt": False},
                  timeout=600)
            if not server.alive():
                raise RuntimeError("server exited")
        except Exception:  # noqa: BLE001 - a crash shows as a failed request
            crashed.append(n)
            tail = server.text()[-3000:]
            errors.append(next((line for line in tail.splitlines()
                                if "error" in line.lower() or "fault" in line.lower()), tail[-200:]))
            server.stop()
            server = None
            restarts += 1
            print(f"{name}: length {n} failed", flush=True)
    if server is not None:
        server.stop()
    print(f"{name}: {len(crashed)} failing lengths {crashed}", flush=True)
    return {"runs": True, "placement": placement, "lengths": [min(LENGTHS), max(LENGTHS)],
            "crashed": crashed, "first_errors": errors[:5],
            "llama_cpp_revision": _revision(BUILDS[build]["source"])}


# -- stage determinism ----------------------------------------------------------------


def _determinism_prompts() -> tuple[list[dict], dict]:
    manifest = _load_json(GENERATION_MANIFEST)["canonical_manifest"]
    seen, chosen = set(), []
    for prompt in manifest["prompts"]:
        if prompt["category"] not in seen:
            seen.add(prompt["category"])
            chosen.append(prompt)
    return chosen[:5], manifest["decoding"]


def stage_determinism(args, report: dict, placement_name: str) -> dict:
    placement = placements(report)[placement_name]
    prompts, decoding = _determinism_prompts()
    sampling = {k: v for k, v in decoding["thinking_on"].items()}
    server = Server(args, "fixed", "rco_v2", placement_args(placement, "server"), 65536,
                    f"determinism_{placement_name}")
    if not server.ready():
        server.stop()
        return {"runs": False, "placement": placement}
    results = []
    try:
        for prompt in prompts:
            outputs = []
            for _ in range(2):
                reply = _post(server.base + "/v1/chat/completions", {
                    "messages": [{"role": "user", "content": prompt["prompt"]}],
                    "chat_template_kwargs": {"enable_thinking": True},
                    "seed": DETERMINISM_SEED, "max_tokens": decoding["maximum_generated_tokens"],
                    "cache_prompt": False, **sampling})
                message = reply["choices"][0]["message"]
                outputs.append({"content": message.get("content") or "",
                                "reasoning_content": message.get("reasoning_content") or "",
                                "tokens": reply["usage"]["completion_tokens"]})
            identical = outputs[0] == outputs[1]
            results.append({"id": prompt["id"], "identical": identical, "outputs": outputs})
            print(f"determinism {placement_name} {prompt['id']}: "
                  f"{'identical' if identical else 'DIFFERENT'}", flush=True)
    finally:
        server.stop()
    return {"runs": True, "placement": placement, "seed": DETERMINISM_SEED,
            "thinking": True, "prompts": results,
            "identical": all(item["identical"] for item in results)}


# -- stage bench / defaults (information) ---------------------------------------------


def stage_bench(args, report: dict, build: str, model: str, placement_name: str) -> dict:
    placement = placements(report)[placement_name]
    binary = BUILDS[build]["bin"]
    base = [str(binary / "llama-bench")] if (binary / "llama-bench").exists() \
        else [str(binary / "llama"), "bench"]
    command = [*base, "-m", str(MODELS[model]), *placement_args(placement, "bench"),
               "-t", str(args.threads), "-p", "512", "-n", "128", "-r", "3", "-o", "json"]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        return {"runs": False, "placement": placement, "stderr_tail": done.stderr[-2000:]}
    rows = json.loads(done.stdout)
    summary = {("pp512" if row["n_prompt"] else "tg128"): round(row["avg_ts"], 2) for row in rows}
    print(f"bench {build} {model} {placement_name}: {summary}", flush=True)
    return {"runs": True, "placement": placement, "tokens_per_second": summary, "rows": rows}


def stage_defaults(args, report: dict, build: str, model: str) -> dict:
    server = Server(args, build, model, [], 0, f"defaults_{build}_{model}", fit_off=False)
    if not server.ready():
        text = server.text()
        server.stop()
        return {"runs": False, "log_tail": text[-3000:]}
    try:
        reply = _post(server.base + "/completion", {
            "prompt": "Write a short paragraph about rivers.", "n_predict": 128,
            "temperature": 0, "ignore_eos": True, "cache_prompt": False})
        speed = reply.get("timings", {}).get("predicted_per_second")
        used = _gpu_used_mib()
    finally:
        text = server.text()
        server.stop()
    fit_lines = [line for line in text.splitlines() if "fit" in line.lower()][:40]
    context = re.findall(r"n_ctx\s*=\s*(\d+)", text)
    print(f"defaults {build} {model}: {speed} tokens/s", flush=True)
    return {"runs": True, "generation_tokens_per_second": speed, "nvidia_smi_used_mib": used,
            "buffers": _buffers(text), "n_ctx": int(context[0]) if context else None,
            "fit_log": fit_lines}


# -- run / report ---------------------------------------------------------------------


def _plan(report: dict) -> list[tuple[str, tuple]]:
    steps: list[tuple[str, tuple]] = [("p3", ())]
    found = placements(report) if report["stages"].get("p3") else {}
    steps += [("agreement", ("fixed", model)) for model in MODELS]
    steps += [("agreement", ("upstream", model)) for model in MODELS]
    for build in ("fixed", "pinned", "upstream"):
        steps += [("lengths", (build, "rco_v2", p)) for p in ("P1", "P3") if p in found]
    steps += [("determinism", (p,)) for p in ("P1", "P3") if p in found]
    for build in ("fixed", "upstream"):
        names = [p for p in found if build == "fixed" or p in ("P1", "P3")]
        steps += [("bench", (build, model, p)) for model in MODELS for p in names]
        steps += [("defaults", (build, model)) for model in MODELS]
    return steps


STAGES = {"p3": stage_p3, "agreement": stage_agreement, "lengths": stage_lengths,
          "determinism": stage_determinism, "bench": stage_bench, "defaults": stage_defaults}


def run(args) -> int:
    args.work.mkdir(parents=True, exist_ok=True)
    identity = {
        "models": {k: {"path": str(v), "sha256": _sha256_file(v)} for k, v in MODELS.items()},
        "builds": {k: {"source": str(v["source"]), "revision": _revision(v["source"]),
                       "helper_sha256": _sha256_file(v["helper"]) if v["helper"] else None}
                   for k, v in BUILDS.items()},
        "helper_source_sha256": _sha256_file(RCO / "tools/gguf_chat_topk.cpp"),
        "threads": args.threads, "ubatch": args.ubatch,
    }
    if args.output.exists():
        report = _load_json(args.output)
        if report["identity"] != json.loads(json.dumps(identity)):
            raise RuntimeError(f"{args.output} describes another run; refusing to resume")
    else:
        report = {"schema": SCHEMA, "status": "running", "identity": identity, "stages": {}}
    while True:
        pending = [(stage, key) for stage, key in _plan(report)
                   if "/".join((stage, *key)) not in report["stages"]
                   and not (stage == "p3" and "p3" in report["stages"])]
        if not pending:
            break
        stage, key = pending[0]
        print(f"== {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {stage} {' '.join(key)}", flush=True)
        result = STAGES[stage](args, report, *key)
        report["stages"]["/".join((stage, *key)) if key else stage] = result
        _atomic_json(args.output, report)
    return evaluate(args, report)


def evaluate(args, report: dict) -> int:
    stages = report["stages"]
    checks: dict[str, Any] = {}
    p3 = stages["p3"]
    checks["p3_found"] = p3["n"] is not None
    loads = {}
    agreement = {}
    for build in ("fixed", "upstream"):
        for name in ("P1", "P2", "P3", "P4"):
            rows = {m: stages.get(f"agreement/{build}/{m}", {}).get(name) for m in MODELS}
            if any(row is None for row in rows.values()):
                continue
            ok_runs = all(row["vs_P0"]["runs"] for row in rows.values())
            loads[f"{build}/{name}"] = ok_runs
            if ok_runs:
                cand, base = rows["rco_v2"]["vs_P0"]["mean"], rows["gsq"]["vs_P0"]["mean"]
                limit = KL_FACTOR * base + KL_SLACK
                agreement[f"{build}/{name}"] = {"rco_v2": cand, "gsq": base, "limit": limit,
                                                "pass": cand <= limit}
    checks["loads_and_runs"] = all(loads.values()) and bool(loads)
    checks["agreement"] = all(item["pass"] for item in agreement.values()) and bool(agreement)
    lengths = {k: v for k, v in stages.items() if k.startswith("lengths/")}
    checks["no_length_crash_fixed"] = all(
        v["runs"] and not v["crashed"] for k, v in lengths.items() if "/fixed/" in k)
    determinism = {k: v for k, v in stages.items() if k.startswith("determinism/")}
    checks["determinism"] = bool(determinism) and all(
        v["runs"] and v["identical"] for v in determinism.values())
    status = "pass" if all(checks.values()) else "fail"
    report["evaluation"] = {
        "checks": checks, "status": status, "loads": loads, "agreement": agreement,
        "agreement_rule": f"rco_v2 mean KL(P0 || Pk) <= {KL_FACTOR} x gsq's + {KL_SLACK}",
        "length_crashes": {k: v.get("crashed") for k, v in lengths.items()},
        "p3_n_cpu_moe": p3["n"], "p4_fits": p3["p4_fits"],
        "information": {
            "upstream_length_crash_counted": False,
            "bench": {k: v.get("tokens_per_second") for k, v in stages.items()
                      if k.startswith("bench/")},
            "defaults": {k: {"tokens_per_second": v.get("generation_tokens_per_second"),
                             "n_ctx": v.get("n_ctx"), "nvidia_smi_used_mib": v.get("nvidia_smi_used_mib")}
                         for k, v in stages.items() if k.startswith("defaults/")},
        },
    }
    report["status"] = status
    _atomic_json(args.output, report)
    print(f"offload: {status}; checks {checks}; P3 n_cpu_moe {p3['n']}; agreement "
          f"{ {k: (round(v['rco_v2'], 5), round(v['gsq'], 5)) for k, v in agreement.items()} }; "
          f"length crashes { {k: v.get('crashed') for k, v in lengths.items()} }", flush=True)
    return 0 if status == "pass" else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--corpus", type=Path,
                        default=RCO / "reports/qwen36_chat_corpus_v2_manifest.json")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_offload")
    parser.add_argument("--output", type=Path,
                        default=RCO / "reports/qwen36_cuda_offload_rco_v2.json")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ubatch", type=int, default=256)
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    for key in ("corpus", "reports", "work", "output"):
        setattr(args, key, getattr(args, key).resolve())
    if args.command == "report":
        return evaluate(args, _load_json(args.output))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
