#!/usr/bin/env python3
"""Phase 4b diagnostic: do pruning masks change template-token decisions?

Gate 1 for the chat-searched P24 build failed once because the model answered
a tool prompt directly instead of calling the tool.  On the calibration half,
the template-token KL of that mask was +0.010 (CI including 0).  This tool
looks at the template tokens themselves, without sampling.

``corpus``
    Reads the saved exact scores of ``audit_qwen36_chat_kl.py`` (per-target
    NLL, PyTorch, exact routing) and the BF16 top-20 reference for one split.
    For each template token that the reference model wrote (``<tool_call>``,
    ``</tool_call>``, ``<think>``, ``</think>``, ``<|im_end|>``, ...), it
    reports every label's mean log-probability of that token, overall and
    per stratum, and the paired difference to the base label.

``first-token``
    Starts ``llama-server --jinja`` on one GGUF and, for every tool prompt
    with thinking off (the gate's prompts and the tool-call prompts of chat
    corpus v2), renders the prompt with its tools, takes the next-token
    distribution at the first reply position (``/completion``, one token,
    raw probabilities) and records the log-probability of ``<tool_call>``.
    With thinking off the reply starts right after the empty think block, so
    this position is the call-or-answer decision.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_chat_generation import _post, _wait_ready  # noqa: E402
from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file  # noqa: E402

SCHEMA = "rco.qwen36.template_token_probe.v1"
TOOL_CALL = "<tool_call>"
TOP_N = 100


def _template_ids(model_dir: Path) -> dict[int, str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    return {int(i): token.content for i, token in tokenizer.added_tokens_decoder.items()}


# -- corpus ------------------------------------------------------------------


def _label_log_probs(work: Path, label: str, count: int) -> list[np.ndarray]:
    """Per-conversation log-probability of each scored target, in corpus order."""
    rows: list[np.ndarray | None] = [None] * count
    for path in sorted((work / f"score_{label}").glob("batch*.npz")):
        with np.load(path) as data:
            for row, index in enumerate(data["members"]):
                rows[int(index)] = -data[f"{row}_nll"]
    if any(row is None for row in rows):
        raise RuntimeError(f"{label}: scores are incomplete in {work}")
    return rows


def _reference_log_probs(work: Path, targets: list[np.ndarray]) -> list[np.ndarray]:
    """BF16 log-probability of each target; NaN where it is outside the top 20."""
    with np.load(work / "reference_bf16_top20.npz") as data:
        values, indices, counts = data["values"], data["indices"], data["counts"]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    result = []
    for number, target in enumerate(targets):
        v = values[offsets[number]:offsets[number + 1]]
        i = indices[offsets[number]:offsets[number + 1]]
        hit = i == target[:, None]
        result.append(np.where(hit.any(axis=1), (v * hit).sum(axis=1), np.nan))
    return result


def run_corpus(args) -> int:
    from audit_qwen36_chat_kl import load_corpus
    from release_quality import paired_bootstrap_mean_ci

    conversations, identity = load_corpus(SimpleArgs(corpus=args.corpus, split=args.split))
    names = _template_ids(args.model_dir)
    targets = []
    for conversation in conversations:
        tokens = np.asarray(conversation["tokens"])
        targets.append(tokens[1:][np.asarray(conversation["scored"][1:], dtype=bool)])
    labels = [args.base_label, *args.labels]
    log_probs = {label: _label_log_probs(args.work, label, len(conversations))
                 for label in labels}
    for label, rows in log_probs.items():
        for conversation, row, target in zip(conversations, rows, targets):
            if row.shape != target.shape:
                raise RuntimeError(f"{label} {conversation['id']}: scored count differs")
    reference = _reference_log_probs(args.work, targets)

    def summarize(selected: list[tuple[int, np.ndarray]]) -> dict[str, Any]:
        """``selected``: (conversation index, positions) pairs."""
        values = {label: np.concatenate([log_probs[label][c][p] for c, p in selected])
                  for label in labels}
        bf16 = np.concatenate([reference[c][p] for c, p in selected])
        out: dict[str, Any] = {
            "positions": int(bf16.size),
            "conversations": len(selected),
            "bf16_in_top20": int(np.isfinite(bf16).sum()),
            "mean_log_prob": {"bf16_where_in_top20": float(np.nanmean(bf16))
                              if np.isfinite(bf16).any() else None,
                              **{label: float(values[label].mean()) for label in labels}},
            "mean_prob": {label: float(np.exp(values[label]).mean()) for label in labels},
        }
        # Paired over conversations: each conversation's mean over its positions.
        per_conversation = {label: np.asarray([log_probs[label][c][p].mean()
                                               for c, p in selected]) for label in labels}
        out["delta_log_prob_vs_" + args.base_label] = {}
        for label in args.labels:
            delta = per_conversation[label] - per_conversation[args.base_label]
            lower, upper = (paired_bootstrap_mean_ci(delta.tolist()) if delta.size > 1
                            else (float("nan"), float("nan")))
            out["delta_log_prob_vs_" + args.base_label][label] = {
                "mean": float(delta.mean()), "ci95": [lower, upper]}
        return out

    strata = sorted({c["stratum"] for c in conversations})
    tokens_report = {}
    for token_id, name in sorted(names.items()):
        hits = [(c, np.flatnonzero(t == token_id)) for c, t in enumerate(targets)]
        hits = [(c, p) for c, p in hits if p.size]
        if not hits:
            continue
        entry = {"token_id": token_id, "all": summarize(hits), "per_stratum": {}}
        for stratum in strata:
            subset = [(c, p) for c, p in hits if conversations[c]["stratum"] == stratum]
            if subset:
                entry["per_stratum"][stratum] = summarize(subset)
        tokens_report[name] = entry
        delta = entry["all"]["delta_log_prob_vs_" + args.base_label]
        print(f"{name!r}: {entry['all']['positions']} positions; " + ", ".join(
            f"{label} {d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
            for label, d in delta.items()), flush=True)

    _atomic_json(args.output or args.reports
                 / f"qwen36_template_token_probe_corpus{args.suffix}.json", {
        "schema": SCHEMA + ".corpus",
        "status": "complete",
        "what": ("log-probability, under each label's exact PyTorch score, of every "
                 "template token the reference model wrote in a scored reply; deltas "
                 "are paired per conversation (mean over its positions), 95% bootstrap"),
        "identity": {"corpus": identity,
                     "score_reports": {label: _artifact(args.reports / (
                         f"qwen36_chat_kl_score_{label}{args.suffix}.json")) for label in labels},
                     "reference": _artifact(args.work / "reference_bf16_top20.npz")},
        "base_label": args.base_label,
        "tokens": tokens_report,
    })
    return 0


class SimpleArgs:
    def __init__(self, **kwargs):
        self.limit = None
        self.__dict__.update(kwargs)


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256_file(path)}


# -- first-token ---------------------------------------------------------------


def _tool_prompts(args) -> list[dict[str, Any]]:
    prompts = []
    for prompt in _load_json(args.prompts)["prompts"]:
        if prompt["kind"] == "tool" and not prompt["thinking"]:
            prompts.append({"id": prompt["id"], "source": "gate",
                            "messages": [{"role": "user", "content": prompt["turns"][0]}],
                            "tools": prompt["tools"]})
    with args.corpus_prompts.open(encoding="utf-8") as handle:
        for line in handle:
            prompt = json.loads(line)
            if prompt["stratum"] == "tool-call" and not prompt["thinking"]:
                prompts.append({"id": prompt["id"], "source": prompt["split"],
                                "messages": prompt["messages"], "tools": prompt["tools"]})
    return prompts


def _server_command(args) -> list[str]:
    command = [str(args.server), "-m", str(args.model), "--jinja", "-ngl", str(args.gpu_layers),
               "-t", str(args.threads), "-c", str(args.context), "-np", "1",
               "--host", "127.0.0.1", "--port", str(args.port), "--no-webui"]
    if args.n_cpu_moe:
        command += ["--n-cpu-moe", str(args.n_cpu_moe)]
    return command


def run_first_token(args) -> int:
    output = args.reports / f"qwen36_template_token_probe_first_token_{args.label}.json"
    prompts = _tool_prompts(args)
    identity = {"model": _artifact(args.model), "server": _artifact(args.server),
                "prompts": _artifact(args.prompts),
                "corpus_prompts": _artifact(args.corpus_prompts),
                "parameters": {"gpu_layers": args.gpu_layers, "n_cpu_moe": args.n_cpu_moe,
                               "threads": args.threads, "context": args.context}}
    if output.exists() and _load_json(output).get("status") == "complete":
        if _load_json(output)["identity"] != identity:
            raise RuntimeError(f"{output} exists with a different identity")
        print(f"{output.name}: already complete", flush=True)
        return 0
    base = f"http://127.0.0.1:{args.port}"
    log = (ROOT / f"{output.stem}.server.log").open("w")
    process = subprocess.Popen(_server_command(args), stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    results = []
    try:
        _wait_ready(base, process, timeout=900)
        tool_call_id = _post(base + "/tokenize", {"content": TOOL_CALL, "add_special": False,
                                                 "parse_special": True})["tokens"]
        if len(tool_call_id) != 1:
            raise RuntimeError(f"{TOOL_CALL} is not one token: {tool_call_id}")
        tool_call_id = tool_call_id[0]
        for prompt in prompts:
            request = {"messages": prompt["messages"], "tools": prompt["tools"],
                       "chat_template_kwargs": {"enable_thinking": False}}
            text = _post(base + "/apply-template", request)["prompt"]
            tokens = _post(base + "/tokenize", {"content": text, "add_special": False,
                                                "parse_special": True})["tokens"]
            response = _post(base + "/completion", {
                "prompt": tokens, "n_predict": 1, "n_probs": TOP_N, "temperature": 0,
                "post_sampling_probs": False, "cache_prompt": False})
            first = response["completion_probabilities"][0]
            top = first["top_logprobs"]
            hit = [entry["logprob"] for entry in top if entry["id"] == tool_call_id]
            results.append({
                "id": prompt["id"], "source": prompt["source"],
                "prompt_tokens": len(tokens),
                "tool_call_log_prob": hit[0] if hit else None,
                "tool_call_log_prob_upper_bound": None if hit else top[-1]["logprob"],
                "greedy_token": first.get("token"),
                "top5": [{"token": e["token"], "logprob": e["logprob"]} for e in top[:5]],
            })
            r = results[-1]
            print(f"{args.label} {prompt['id']}: P({TOOL_CALL}) "
                  + (f"{np.exp(r['tool_call_log_prob']):.4f}" if hit else
                     f"< {np.exp(r['tool_call_log_prob_upper_bound']):.2e}")
                  + f", greedy {r['greedy_token']!r}", flush=True)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
    probs = {source: [np.exp(r["tool_call_log_prob"]) if r["tool_call_log_prob"] is not None
                      else 0.0 for r in results if r["source"] == source]
             for source in dict.fromkeys(r["source"] for r in results)}
    _atomic_json(output, {
        "schema": SCHEMA + ".first_token",
        "status": "complete",
        "label": args.label,
        "what": (f"raw next-token log-probability of {TOOL_CALL} at the first reply "
                 "position of each tool prompt with thinking off (llama-server, "
                 "n_probs, before sampling); absent from the top "
                 f"{TOP_N} means below the upper bound given"),
        "identity": identity,
        "tool_call_token_id": tool_call_id,
        "summary": {source: {"prompts": len(p), "mean_prob": float(np.mean(p)),
                             "min_prob": float(np.min(p)),
                             "greedy_tool_call": sum(r["greedy_token"] == TOOL_CALL
                                                     for r in results
                                                     if r["source"] == source)}
                    for source, p in probs.items()},
        "prompts": results,
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("corpus", "first-token"))
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/qwen36_35b_base")
    # corpus
    parser.add_argument("--corpus", type=Path,
                        default=RCO / "reports/qwen36_chat_corpus_v2_manifest.json")
    parser.add_argument("--split", default="heldout", choices=("calibration", "heldout"))
    parser.add_argument("--suffix", default="_v2_heldout")
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_chat_kl_v2_heldout")
    parser.add_argument("--base-label", default="unpruned")
    parser.add_argument("--labels", nargs="+",
                        default=["chat_consensus", "p24_raw_text", "frequency"])
    parser.add_argument("--output", type=Path)
    # first-token
    parser.add_argument("--model", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--prompts", type=Path,
                        default=RCO / "tools/qwen36_chat_generation_prompts_v2.json")
    parser.add_argument("--corpus-prompts", type=Path,
                        default=ROOT / "data/qwen36_chat_corpus_v2/prompts.jsonl")
    parser.add_argument("--server", type=Path,
                        default=ROOT / "experiment/build-cuda/bin/llama-server")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--n-cpu-moe", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()
    if args.command == "corpus":
        return run_corpus(args)
    if args.model is None or args.label is None:
        parser.error("first-token needs --model and --label")
    return run_first_token(args)


if __name__ == "__main__":
    raise SystemExit(main())
