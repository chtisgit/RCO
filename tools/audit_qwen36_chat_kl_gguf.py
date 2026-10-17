#!/usr/bin/env python3
"""Phase 8 of RCO_PLAN_NEW.md: the held-out chat KL gate.

Subcommands:

``score --label L --model GGUF``
    Scores one GGUF in llama.cpp with ``rco-gguf-chat-kl``: every
    conversation of the split decoded alone from the corpus token ids, and
    at each scored position the NLL of the next token and KL(BF16 || model)
    over the saved BF16 top 20 (not renormalized), the metric of
    ``audit_qwen36_chat_kl.py``.  Resumable per conversation.
``crosscheck``
    One pass of the PyTorch evaluator (``search.streamed_chat_exact``) over
    the split, identical batches for all arms: the candidate (frequency mask
    plus its Q3_K upgrades), authentic GSQ (unpruned, its own BF16
    embedding), unpruned GSQ-E6, and the frequency base.
``gate``
    The candidate minus authentic GSQ in llama.cpp, per-conversation mean
    KL, paired bootstrap 95% CI: pass if the upper bound is below 0.  Checks
    against the crosscheck: per-conversation NLL of each model within 0.05
    across the two runtimes (a misaligned position would be off by nats),
    and the same pass or fail on the evaluator's own CI.  Exit status 3 on
    a failed gate, 4 when a check fails (a decision point).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
import os
import struct
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

from audit_qwen36_chat_kl import _load_reference, load_corpus  # noqa: E402
from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file  # noqa: E402

SCHEMA = "rco.qwen36.chat_kl_gguf.v1"
BUNDLE_MAGIC = b"RCOCKL1\0"
TOP_K = 20
NLL_TOLERANCE = 0.05
PREFIX = "qwen36_gsq_rco_v2"


# -- corpus -----------------------------------------------------------------------


def _corpus(args):
    corpus_args = SimpleNamespace(corpus=args.corpus, split=args.split, limit=args.limit,
                                  reports=args.reports, suffix=f"_v2_{args.split}")
    conversations, identity = load_corpus(corpus_args)
    reference, reference_report = _load_reference(
        SimpleNamespace(**{**vars(corpus_args), "limit": None}), {
            **identity, **_full_identity(args)})
    reference = reference[:len(conversations)]
    return conversations, identity, reference, reference_report


def _full_identity(args) -> dict[str, Any]:
    """The reference report covers the whole split; ``--limit`` smoke tests
    still check it against the full corpus identity."""
    if not args.limit:
        return {}
    _, identity = load_corpus(SimpleNamespace(corpus=args.corpus, split=args.split, limit=None,
                                              reports=args.reports,
                                              suffix=f"_v2_{args.split}"))
    return identity


def scored_positions(conversation: dict[str, Any]) -> np.ndarray:
    """Logit positions t whose next token t + 1 is scored (position 0 never)."""
    scored = np.asarray(conversation["scored"], dtype=bool)
    return np.flatnonzero(scored[1:]).astype(np.int32)


def write_bundle(path: Path, conversations, reference) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        handle.write(BUNDLE_MAGIC)
        handle.write(struct.pack("<III", 1, TOP_K, len(conversations)))
        for conversation, (values, indices) in zip(conversations, reference, strict=True):
            positions = scored_positions(conversation)
            if values.shape != (positions.size, TOP_K) or indices.shape != values.shape:
                raise RuntimeError(f"{conversation['id']}: reference rows differ from the "
                                   f"scored positions")
            identifier = conversation["id"].encode()
            tokens = np.asarray(conversation["tokens"], dtype=np.int32)
            handle.write(struct.pack("<I", len(identifier)) + identifier)
            handle.write(struct.pack("<I", tokens.size) + tokens.tobytes())
            handle.write(struct.pack("<I", positions.size) + positions.tobytes())
            handle.write(np.ascontiguousarray(indices, dtype=np.int32).tobytes())
            handle.write(np.ascontiguousarray(values, dtype=np.float32).tobytes())
    temporary.replace(path)


def _template_ids(args) -> np.ndarray:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    return np.asarray(sorted(tokenizer.added_tokens_decoder))


def _template_kl(conversations, kl_rows, template_ids):
    template_kl, template_counts = [], []
    for conversation, kl in zip(conversations, kl_rows, strict=True):
        tokens = np.asarray(conversation["tokens"])
        targets = tokens[1:][np.asarray(conversation["scored"][1:], dtype=bool)]
        is_template = np.isin(targets, template_ids)
        template_counts.append(int(is_template.sum()))
        template_kl.append(float(np.asarray(kl)[is_template].mean())
                           if is_template.any() else float("nan"))
    return template_kl, template_counts


def _git_revision(directory: Path) -> str:
    return subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


# -- score ------------------------------------------------------------------------


def _score_paths(args) -> tuple[Path, Path]:
    tag = f"{args.label}_v2_{args.split}{args.suffix}"
    return (args.reports / f"qwen36_chat_kl_gguf_{tag}.json",
            args.work / f"{tag}.jsonl")


def run_score(args) -> None:
    output, lines_path = _score_paths(args)
    if output.exists() and _load_json(output).get("status") == "complete":
        print(f"score {args.label}: already complete", flush=True)
        return
    started = time.perf_counter()
    conversations, corpus_identity, reference, reference_report = _corpus(args)
    args.work.mkdir(parents=True, exist_ok=True)
    identity = {
        "model": {"path": str(args.model), "sha256": _sha256_file(args.model)},
        "helper": {"path": str(args.helper), "sha256": _sha256_file(args.helper),
                   "source_sha256": _sha256_file(RCO / "tools/gguf_chat_kl.cpp")},
        "llama_cpp": {"path": str(args.llama_dir), "revision": _git_revision(args.llama_dir)},
        "parameters": {"gpu_layers": args.gpu_layers, "threads": args.threads,
                       "ubatch": args.ubatch},
        "corpus": corpus_identity,
        "reference_sha256": reference_report["reference"]["sha256"],
    }
    manifest_path = lines_path.with_suffix(".identity.json")
    if manifest_path.exists() and _load_json(manifest_path) != json.loads(json.dumps(identity)):
        raise RuntimeError(f"{manifest_path} describes another run; refusing to resume")
    _atomic_json(manifest_path, identity)
    bundle = lines_path.with_suffix(".bundle")
    write_bundle(bundle, conversations, reference)

    done: list[dict[str, Any]] = []
    if lines_path.exists():
        with lines_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.endswith("\n"):
                    done.append(json.loads(line))
        with lines_path.open("w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(item) + "\n" for item in done)
    for number, item in enumerate(done):
        if item["index"] != number or item["id"] != conversations[number]["id"]:
            raise RuntimeError(f"{lines_path}: line {number} is out of order")
    environment = None
    if len(done) < len(conversations):
        command = [str(args.helper), "--model", str(args.model), "--input", str(bundle),
                   "--gpu-layers", str(args.gpu_layers), "--threads", str(args.threads),
                   "--ubatch", str(args.ubatch), "--start", str(len(done))]
        print(f"score {args.label}: {len(conversations) - len(done)} conversations from "
              f"{len(done)}", flush=True)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        assert process.stdout is not None
        with lines_path.open("a", encoding="utf-8") as handle:
            for line in process.stdout:
                item = json.loads(line)
                if item["type"] == "environment":
                    environment = item
                    continue
                number = len(done)
                if item["index"] != number or item["id"] != conversations[number]["id"]:
                    raise RuntimeError(f"helper returned {item['id']} at {number}")
                if len(item["kl"]) != scored_positions(conversations[number]).size:
                    raise RuntimeError(f"{item['id']}: scored count differs")
                handle.write(json.dumps(item) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                done.append(item)
                if number % 20 == 0 or number + 1 == len(conversations):
                    print(f"score {args.label}: {number + 1}/{len(conversations)} "
                          f"{item['id']} ({item['token_count']} tokens) KL "
                          f"{np.mean(item['kl']):.5f} in {item['seconds']:.1f} s", flush=True)
        if process.wait() != 0:
            raise RuntimeError(f"helper exited with status {process.returncode}")
    if len(done) != len(conversations):
        raise RuntimeError("helper stopped early")

    template_kl, template_counts = _template_kl(conversations, [d["kl"] for d in done],
                                                _template_ids(args))
    kl = [float(np.mean(d["kl"])) for d in done]
    nll = [float(np.mean(d["nll"])) for d in done]
    _atomic_json(output, {
        "schema": SCHEMA + ".score", "status": "complete", "label": args.label,
        "runtime": "llama.cpp", "identity": identity, "environment": environment,
        "metric": "per conversation: mean over scored targets of KL(BF16 || model) over the "
                  "BF16 top 20, not renormalized; and mean NLL",
        "mean_top20_kl_to_bf16": float(np.mean(kl)), "mean_nll": float(np.mean(nll)),
        "conversation_ids": [c["id"] for c in conversations],
        "conversation_mean_top20_kl": kl, "conversation_mean_nll": nll,
        "conversation_template_kl": template_kl,
        "conversation_template_tokens": template_counts,
        "strata": [c["stratum"] for c in conversations],
        "decode_seconds": float(sum(d["seconds"] for d in done)),
        "wall_seconds_this_invocation": time.perf_counter() - started,
    })
    print(f"score {args.label}: KL {np.mean(kl):.5f} NLL {np.mean(nll):.5f}", flush=True)


# -- PyTorch crosscheck -------------------------------------------------------------


def _crosscheck_path(args) -> Path:
    return args.reports / f"{PREFIX}_chat_kl_crosscheck_v2_{args.split}{args.suffix}.json"


def run_crosscheck(args) -> None:
    from search.streamed_chat_exact import ExactVariant
    from search.streamed_prune_chat import rows_from_corpus
    from search_qwen36_prune24_q3k import _evaluator, _paired

    output = _crosscheck_path(args)
    if output.exists() and _load_json(output).get("status") == "complete":
        print("crosscheck: already complete", flush=True)
        return
    started = time.perf_counter()
    conversations, corpus_identity, reference, reference_report = _corpus(args)
    build = _load_json(args.candidate_report)
    if build.get("status") != "built":
        raise RuntimeError("the candidate GGUF report is not complete")
    upgrades = frozenset(build["upgraded_tensors"])
    evaluator, mask, _, identity = _evaluator(args)
    if identity["mask_sha256"] != build["inputs"]["mask"]["sha256"]:
        raise RuntimeError("the candidate GGUF was built with another mask")
    rows = rows_from_corpus(conversations, reference).rows
    variants = [
        ExactVariant("candidate", prune_mask=mask, upgrades=upgrades),
        ExactVariant("gsq", embedding="checkpoint"),
        ExactVariant("gsq_e6"),
        ExactVariant("base", prune_mask=mask),
    ]
    result = evaluator.evaluate(rows, variants)
    template_ids = _template_ids(args)
    arms = {}
    for number, variant in enumerate(variants):
        kl = result.row_mean_kl(number)
        template_kl, template_counts = _template_kl(conversations, result.kl[number],
                                                    template_ids)
        arms[variant.label] = {
            "conversation_mean_top20_kl": kl.tolist(),
            "conversation_mean_nll": result.row_mean_nll(number).tolist(),
            "conversation_template_kl": template_kl,
            "conversation_template_tokens": template_counts,
            "mean_top20_kl_to_bf16": float(kl.mean()),
            "mean_nll": float(result.row_mean_nll(number).mean()),
            "pruned": variant.prune_mask is not None, "embedding": variant.embedding,
            "upgrades": sorted(variant.upgrades)}

    def delta(a, b):
        return _paired(arms[a]["conversation_mean_top20_kl"], arms[b]["conversation_mean_top20_kl"])

    _atomic_json(output, {
        "schema": SCHEMA + ".crosscheck", "status": "complete", "runtime": "pytorch",
        "identity": {**identity, "corpus": corpus_identity,
                     "reference_sha256": reference_report["reference"]["sha256"],
                     "candidate_gguf_report_sha256": _sha256_file(args.candidate_report),
                     "attn_implementation": "sdpa"},
        "arms": arms,
        "deltas": {"candidate_minus_gsq": delta("candidate", "gsq"),
                   "embedding_swap_gsq_e6_minus_gsq": delta("gsq_e6", "gsq"),
                   "pruning_base_minus_gsq_e6": delta("base", "gsq_e6"),
                   "upgrades_candidate_minus_base": delta("candidate", "base"),
                   "candidate_minus_gsq_e6": delta("candidate", "gsq_e6")},
        "strata": [c["stratum"] for c in conversations],
        "conversation_ids": [c["id"] for c in conversations],
        "stats": asdict(result.stats),
        "wall_seconds": time.perf_counter() - started})
    print("crosscheck: " + ", ".join(f"{label} {arm['mean_top20_kl_to_bf16']:.5f}"
                                     for label, arm in arms.items()), flush=True)


# -- gate -------------------------------------------------------------------------


def run_gate(args) -> int:
    from search_qwen36_prune24_q3k import _paired

    def score(label):
        report = _load_json(args.reports / f"qwen36_chat_kl_gguf_{label}_v2_{args.split}"
                                           f"{args.suffix}.json")
        if report.get("status") != "complete":
            raise RuntimeError(f"llama.cpp score {label} is not complete")
        return report

    candidate, gsq = score(args.candidate_label), score(args.base_label)
    for key in ("helper", "llama_cpp", "parameters", "corpus", "reference_sha256"):
        if candidate["identity"][key] != gsq["identity"][key]:
            raise RuntimeError(f"the two llama.cpp scores differ in {key}")
    build = _load_json(args.candidate_report)
    if candidate["identity"]["model"]["sha256"] != build["output"]["sha256"]:
        raise RuntimeError("the candidate score is not of the built candidate GGUF")
    crosscheck = _load_json(_crosscheck_path(args))
    if crosscheck.get("status") != "complete":
        raise RuntimeError("the PyTorch crosscheck is not complete")
    if crosscheck["conversation_ids"] != candidate["conversation_ids"]:
        raise RuntimeError("the crosscheck covers other conversations")

    gate = _paired(candidate["conversation_mean_top20_kl"], gsq["conversation_mean_top20_kl"])
    passed = gate["ci95"][1] < 0
    strata = np.asarray(candidate["strata"])
    per_stratum = {}
    for stratum in sorted(set(strata)):
        members = np.flatnonzero(strata == stratum)
        per_stratum[stratum] = _paired(
            [candidate["conversation_mean_top20_kl"][i] for i in members],
            [gsq["conversation_mean_top20_kl"][i] for i in members])
    template = [i for i in range(len(strata))
                if np.isfinite(candidate["conversation_template_kl"][i])
                and np.isfinite(gsq["conversation_template_kl"][i])]

    runtimes = {}
    for label, llama, arm in (("candidate", candidate, "candidate"), ("gsq", gsq, "gsq")):
        torch_arm = crosscheck["arms"][arm]
        nll_gap = np.abs(np.asarray(llama["conversation_mean_nll"])
                         - torch_arm["conversation_mean_nll"])
        kl_llama = np.asarray(llama["conversation_mean_top20_kl"])
        kl_torch = np.asarray(torch_arm["conversation_mean_top20_kl"])
        runtimes[label] = {
            "max_abs_nll_difference": float(nll_gap.max()),
            "mean_nll_llama_minus_pytorch": float(np.mean(
                np.asarray(llama["conversation_mean_nll"]) - torch_arm["conversation_mean_nll"])),
            "mean_kl_llama": float(kl_llama.mean()), "mean_kl_pytorch": float(kl_torch.mean()),
            "kl_correlation": float(np.corrcoef(kl_llama, kl_torch)[0, 1]),
            "position_check": bool(nll_gap.max() <= NLL_TOLERANCE)}
    torch_gate = crosscheck["deltas"]["candidate_minus_gsq"]
    agree = (torch_gate["ci95"][1] < 0) == passed
    checks_ok = agree and all(item["position_check"] for item in runtimes.values())
    status = ("pass" if passed else "fail") if checks_ok else "decision"
    report = {
        "schema": SCHEMA + ".gate", "status": status,
        "gate": "candidate minus authentic GSQ (llama.cpp), per-conversation mean top-20 KL "
                "to BF16, paired bootstrap 95% CI upper bound below 0",
        "llama_cpp_gate_pass": passed,
        "candidate_minus_gsq": gate,
        "candidate_minus_gsq_nll": _paired(candidate["conversation_mean_nll"],
                                           gsq["conversation_mean_nll"]),
        "candidate_minus_gsq_template_kl": {**_paired(
            [candidate["conversation_template_kl"][i] for i in template],
            [gsq["conversation_template_kl"][i] for i in template])},
        "per_stratum": per_stratum,
        "means": {"candidate": {"kl": candidate["mean_top20_kl_to_bf16"],
                                "nll": candidate["mean_nll"]},
                  "gsq": {"kl": gsq["mean_top20_kl_to_bf16"], "nll": gsq["mean_nll"]}},
        "crosscheck": {"pytorch_candidate_minus_gsq": torch_gate,
                       "pytorch_gate_pass": torch_gate["ci95"][1] < 0,
                       "agree": agree, "runtimes": runtimes,
                       "nll_tolerance": NLL_TOLERANCE,
                       "steps_pytorch": {k: v for k, v in crosscheck["deltas"].items()
                                         if k != "candidate_minus_gsq"},
                       "means_pytorch": {k: v["mean_top20_kl_to_bf16"]
                                         for k, v in crosscheck["arms"].items()}},
        "inputs": {"candidate": candidate["identity"], "gsq": gsq["identity"]["model"],
                   "crosscheck_sha256": _sha256_file(_crosscheck_path(args))},
    }
    _atomic_json(args.reports / f"{PREFIX}_chat_kl_gate_v2_{args.split}{args.suffix}.json",
                 report)
    print(f"gate: {status}; candidate - GSQ {gate['mean_delta']:+.5f} CI "
          f"[{gate['ci95'][0]:+.5f}, {gate['ci95'][1]:+.5f}], {gate['improved_conversations']}"
          f"/{gate['conversations']} better; PyTorch {torch_gate['mean_delta']:+.5f} CI "
          f"[{torch_gate['ci95'][0]:+.5f}, {torch_gate['ci95'][1]:+.5f}]; NLL gaps "
          + ", ".join(f"{k} {v['max_abs_nll_difference']:.4f}" for k, v in runtimes.items()),
          flush=True)
    return {"pass": 0, "fail": 3, "decision": 4}[status]


# -- main -------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("score", "crosscheck", "gate"))
    parser.add_argument("--label")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--candidate-label", default="rco_v2")
    parser.add_argument("--base-label", default="gsq")
    parser.add_argument("--split", default="heldout", choices=("calibration", "heldout"))
    parser.add_argument("--limit", type=int, help="first N conversations only (smoke tests)")
    parser.add_argument("--suffix", default="", help="report name suffix (smoke tests)")
    parser.add_argument("--corpus", type=Path,
                        default=RCO / "reports/qwen36_chat_corpus_v2_manifest.json")
    parser.add_argument("--candidate-report", type=Path,
                        default=RCO / "reports/qwen36_gsq_rco_v2_candidate_gguf.json")
    parser.add_argument("--helper", type=Path,
                        default=ROOT / "experiment/helpers-mmqfix/rco-gguf-chat-kl")
    parser.add_argument("--llama-dir", type=Path, default=ROOT / "experiment/llama.cpp-mmqfix")
    parser.add_argument("--gpu-layers", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ubatch", type=int, default=256)
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_chat_kl_gguf")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    # PyTorch crosscheck (search_qwen36_prune24_q3k._evaluator)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/qwen36_35b_base")
    parser.add_argument("--manifest", type=Path,
                        default=RCO / "reports/qwen36_35b_base_gguf_manifest.json")
    parser.add_argument("--gguf", type=Path,
                        default=ROOT / "results/Qwen3.6-35B-A3B-GSQ-hybrid.gguf")
    parser.add_argument("--gguf-python", type=Path, default=ROOT / "repos/llama.cpp/gguf-py")
    parser.add_argument("--ggml-library", type=Path,
                        default=ROOT / "experiment/build-cpu/bin/libggml-base.so.0.24.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows-per-chunk", type=int, default=1024)
    parser.add_argument("--install-rows", type=int, default=4096)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--position-chunk", type=int, default=256)
    parser.add_argument("--microbatch-tokens", type=int, default=8448)
    parser.add_argument("--device-hidden-bytes", type=int, default=3 << 30)
    args = parser.parse_args()
    for key in ("corpus", "candidate_report", "helper", "llama_dir", "work", "reports",
                "model_dir", "manifest", "gguf", "gguf_python", "ggml_library"):
        setattr(args, key, getattr(args, key).resolve())
    if args.model is not None:
        args.model = args.model.resolve()
    if args.command == "score" and (args.label is None or args.model is None):
        parser.error("score needs --label and --model")
    return args


def main() -> int:
    args = _parse_args()
    return {"score": run_score, "crosscheck": run_crosscheck,
            "gate": run_gate}[args.command](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
