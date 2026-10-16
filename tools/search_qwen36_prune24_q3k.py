#!/usr/bin/env python3
"""Phase 7 of RCO_PLAN_NEW.md: RCO quant search, 23 of 120 tensors to Q3_K.

Base: GSQ-E6 (Q6_K embedding) with the frequency mask's exact pruned
routing.  Choices: for each of the 120 routed-expert tensors, keep GSQ Q2_0
or install the Phase 6 Q3_K payload; exactly 23 are upgraded.  The loss is
the mean per-conversation top-20 KL to BF16 over the scored tokens, the
metric of the Phase 8 gate.  Every evaluation is exact
(``search.streamed_chat_exact``); search data is the fit part of the chat
v2 calibration half, selection and the gate use the validation part, and
the held-out half is never read.

Subcommands, in order; each writes one report and is resumable:

``crosscheck``
    Unpruned GSQ-E6 and the frequency base, on two of the existing
    ``audit_qwen36_chat_kl.py`` calibration batches (the longest conversation
    and the batch with the most rows) with the same composition; the
    per-conversation KL must equal the existing scores (within 1e-9).
``baseline``
    The frequency base on each of the 10 fit folds, with the micro-batching
    every search step on that fold uses.
``pilot``
    One search step on fold 0, for timing.
``search --run R``
    Paired antithetic REINFORCE over exact-cardinality Plackett-Luce draws,
    as ``search.hard.optimize_hard_reinforce``, checkpointed every step.
    Step s of run R uses fold (R + s) mod 10.  The loss of a draw is its
    mean conversation KL minus the fold's base.
``promote``
    One validation pass over: the base, unpruned GSQ-E6, both controls
    (Phase 6 top 23 by weighted error reduction, and the top 23
    ``ffn_down_exps``), the ceiling (all 120), and each run's final and
    best-screened assignment.  The best search candidate by validation KL
    is promoted; it must beat both controls, otherwise the better control
    is named instead and the report says so.
``replay``
    An independent validation pass over the promoted assignment, the base
    and unpruned GSQ-E6, and the validation gate (safeguard 2): promoted
    minus unpruned GSQ-E6, paired bootstrap 95% CI upper bound below 0.
    Also reported: base minus unpruned (the pruning cost) and promoted
    minus base (the upgrades' gain), safeguard 3.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_chat_kl import PAD_TOKEN, _load_reference, _model, load_corpus  # noqa: E402
from audit_qwen36_q3k_viability import (  # noqa: E402
    _atomic_json,
    _load_json,
    _sha256_file,
    expert_names,
)

SCHEMA = "rco.qwen36.prune24_q3k_search.v1"
PREFIX = "qwen36_prune24_q3k"
UPGRADES = 23
FOLDS = 10
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20261001
CROSSCHECK_BATCH_TOKENS = 12_800
CROSSCHECK_TOLERANCE = 1e-9


# -- data ----------------------------------------------------------------------


def stratified_folds(conversations: list[dict[str, Any]], count: int) -> list[list[int]]:
    """Deal conversations round-robin into folds: stratum by stratum, longest
    first, each stratum continuing where the previous one stopped, so every
    stratum and every fold size differ by at most one across folds."""
    order = sorted(range(len(conversations)), key=lambda i: (
        conversations[i]["stratum"], -conversations[i]["token_count"], conversations[i]["id"]))
    folds: list[list[int]] = [[] for _ in range(count)]
    for position, index in enumerate(order):
        folds[position % count].append(index)
    return [sorted(fold) for fold in folds]


def _parts(args):
    from search.streamed_prune_chat import rows_from_corpus

    split = _load_json(args.split_report)
    corpus_args = SimpleNamespace(corpus=args.corpus, split="calibration", limit=None,
                                  reports=args.reports, suffix="_v2_calibration")
    conversations, identity = load_corpus(corpus_args)
    if split["corpus_manifest_sha256"] != identity["corpus_manifest_sha256"]:
        raise RuntimeError("the split was made for a different corpus")
    reference, reference_report = _load_reference(corpus_args, identity)
    if reference_report.get("attn_implementation") != "sdpa":
        raise RuntimeError("the reference must use SDPA")
    rows = rows_from_corpus(conversations, reference).rows
    by_id = {c["id"]: number for number, c in enumerate(conversations)}
    parts = {}
    for part in ("fit", "validation"):
        numbers = sorted(by_id[i] for i in split[part])
        parts[part] = {
            "conversations": [conversations[n] for n in numbers],
            "rows": [rows[n] for n in numbers],
            "calibration_index": numbers,
        }
    identity = {
        "corpus_manifest_sha256": identity["corpus_manifest_sha256"],
        "split_sha256": _sha256_file(args.split_report),
        "reference_sha256": reference_report["reference"]["sha256"],
        "attn_implementation": "sdpa",
    }
    return parts, identity


def _evaluator(args):
    import torch

    from audit_qwen36_gsq_e6 import NAME, EmbeddingQ6KWeightStore, _entry
    from audit_qwen36_q3k_viability import _expert_weight_store_class
    from gguf_parallel_stream import ParallelGGUFManifestPrefixLoader
    from native_store import NativeCandidateStore
    from quant.ggml_native import GGMLNativeCodec, GGMLType
    from search.streamed_chat_exact import StreamedChatExactEvaluator

    store_report = _load_json(args.reports / "qwen36_prune24_q3k_store.json")
    if store_report.get("status") != "complete":
        raise RuntimeError("the Phase 6 Q3_K store is not complete")
    q3k_root = Path(store_report["store"]["root"])
    if _sha256_file(q3k_root / "native-candidate-index.json") != store_report["store"]["index_sha256"]:
        raise RuntimeError("Q3_K store index differs from its report")
    mask_path = Path(store_report["mask"]["path"])
    if _sha256_file(mask_path) != store_report["mask"]["sha256"]:
        raise RuntimeError("frequency mask differs from the store report")
    embedding = _load_json(args.reports / "qwen36_gsq_e6_embedding.json")
    if _load_json(args.reports / "qwen36_gsq_e6_calibration.json").get("status") != "pass":
        raise RuntimeError("Phase 1 GSQ-E6 has not passed")
    codec = GGMLNativeCodec(args.ggml_library)
    manifest = _load_json(args.manifest)
    embedding_store = EmbeddingQ6KWeightStore(
        codec, Path(embedding["payload"]["path"]), embedding["payload"]["sha256"],
        _entry(args.manifest), rows_per_chunk=4096)
    upgrade_store = _expert_weight_store_class()(
        NativeCandidateStore(q3k_root, codec), manifest, args.model_dir, GGMLType.Q3_K,
        rows_per_chunk=args.install_rows)
    loader = ParallelGGUFManifestPrefixLoader(
        args.gguf, manifest, args.model_dir, gguf_python=args.gguf_python,
        ggml_library=args.ggml_library, rows_per_chunk=args.rows_per_chunk,
        workers=args.decode_workers)
    evaluator = StreamedChatExactEvaluator(
        _model(SimpleNamespace(model_dir=args.model_dir, attn_implementation="sdpa")),
        loader, upgrade_store=upgrade_store, device=torch.device(args.device),
        checkpoint_dtype=torch.bfloat16, embedding_store=embedding_store,
        embedding_selected=[(NAME, 1)], position_chunk=args.position_chunk,
        microbatch_tokens=args.microbatch_tokens, pad_token=PAD_TOKEN,
        device_hidden_bytes=args.device_hidden_bytes)
    identity = {
        "gsq_gguf_sha256": _sha256_file(args.gguf),
        "embedding_sha256": embedding["payload"]["sha256"],
        "q3k_store_index_sha256": store_report["store"]["index_sha256"],
        "mask_sha256": store_report["mask"]["sha256"],
        "microbatch_tokens": args.microbatch_tokens,
    }
    return evaluator, np.load(mask_path), store_report, identity


def _variants(mask, assignments: dict[str, list[str] | None]):
    from search.streamed_chat_exact import ExactVariant

    return [ExactVariant(label, prune_mask=None if names is None else mask,
                         upgrades=frozenset(names or ()))
            for label, names in assignments.items()]


def _names(assignment) -> list[str]:
    names = expert_names()
    return [names[i] for i in np.flatnonzero(np.asarray(assignment))]


def _stats(stats) -> dict[str, Any]:
    return asdict(stats)


def _run_variants(evaluator, rows, variants):
    result = evaluator.evaluate(rows, variants)
    return {variant.label: {
        "conversation_mean_top20_kl": result.row_mean_kl(n).tolist(),
        "conversation_mean_nll": result.row_mean_nll(n).tolist(),
        "mean_top20_kl_to_bf16": float(result.row_mean_kl(n).mean()),
        "mean_nll": float(result.row_mean_nll(n).mean()),
        "upgrades": sorted(variant.upgrades),
        "pruned": variant.prune_mask is not None,
    } for n, variant in enumerate(variants)}, _stats(result.stats)


def _paired(candidate: list[float], base: list[float]) -> dict[str, Any]:
    from release_quality import paired_bootstrap_mean_ci

    deltas = [c - b for c, b in zip(candidate, base, strict=True)]
    low, high = paired_bootstrap_mean_ci(deltas, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)
    return {"mean_delta": float(np.mean(deltas)), "ci95": [low, high],
            "improved_conversations": int(sum(d < 0 for d in deltas)),
            "conversations": len(deltas)}


# -- crosscheck ------------------------------------------------------------------


def run_crosscheck(args) -> int:
    """Rerun two of the existing scorer's batches with the same composition.

    Different batch shapes change bf16 results (and near-tie routing), by up
    to about 0.005 KL per conversation; with the same composition the new
    evaluator must reproduce the existing scores exactly.
    """
    from audit_qwen36_chat_kl import batches
    from search.streamed_prune_chat import rows_from_corpus

    output = args.reports / f"{PREFIX}_crosscheck.json"
    if output.exists() and _load_json(output).get("status") in ("pass", "fail"):
        print(f"crosscheck: already {_load_json(output)['status']}", flush=True)
        return 0
    started = time.perf_counter()
    corpus_args = SimpleNamespace(corpus=args.corpus, split="calibration", limit=None,
                                  reports=args.reports, suffix="_v2_calibration")
    conversations, corpus_identity = load_corpus(corpus_args)
    reference, reference_report = _load_reference(corpus_args, corpus_identity)
    rows = rows_from_corpus(conversations, reference).rows
    evaluator, mask, _, identity = _evaluator(args)
    existing = {}
    for label in ("unpruned", "frequency"):
        old = _load_json(args.reports / f"qwen36_chat_kl_score_{label}_v2_calibration.json")
        if (old["identity"]["reference_sha256"] != reference_report["reference"]["sha256"]
                or old["batch_tokens"] != CROSSCHECK_BATCH_TOKENS):
            raise RuntimeError(f"{label}: existing score differs in reference or batching")
        if label == "frequency" and old["mask"]["sha256"] != identity["mask_sha256"]:
            raise RuntimeError("existing frequency score used another mask")
        existing[label] = old["conversation_mean_top20_kl"]
    old_batches = batches(conversations, CROSSCHECK_BATCH_TOKENS)
    chosen = [0, max(range(len(old_batches)), key=lambda n: (len(old_batches[n]), -n))]
    evaluator.microbatch_tokens = CROSSCHECK_BATCH_TOKENS
    checked = []
    for number in chosen:
        members = old_batches[number]
        arms, stats = _run_variants(evaluator, [rows[i] for i in members],
                                    _variants(mask, {"unpruned": None, "frequency": []}))
        if stats["microbatch_count"] != 1:
            raise RuntimeError("the batch did not stay one micro-batch")
        differences = {label: float(np.max(np.abs(
            np.asarray(arms[label]["conversation_mean_top20_kl"])
            - [existing[label][i] for i in members]))) for label in existing}
        checked.append({"batch": number, "conversation_ids": [conversations[i]["id"]
                                                               for i in members],
                        "max_abs_difference": differences, "stats": stats})
        print(f"crosscheck: batch {number} ({len(members)} rows) {differences}", flush=True)
    passed = all(value <= CROSSCHECK_TOLERANCE
                 for item in checked for value in item["max_abs_difference"].values())
    _atomic_json(output, {
        "schema": SCHEMA + ".crosscheck", "status": "pass" if passed else "fail",
        "gate": (f"same batch composition as audit_qwen36_chat_kl.py score "
                 f"(--batch-tokens {CROSSCHECK_BATCH_TOKENS}): per-conversation KL equal "
                 f"within {CROSSCHECK_TOLERANCE}"),
        "identity": {**identity, "corpus": corpus_identity,
                     "reference_sha256": reference_report["reference"]["sha256"]},
        "batches": checked, "wall_seconds": time.perf_counter() - started})
    print(f"crosscheck: {'pass' if passed else 'fail'}", flush=True)
    return 0 if passed else 3


# -- baseline and search -----------------------------------------------------------


def _fit_folds(parts):
    fit = parts["fit"]
    folds = stratified_folds(fit["conversations"], FOLDS)
    return fit, folds


def run_baseline(args) -> None:
    output = args.reports / f"{PREFIX}_fold_baseline.json"
    if output.exists() and _load_json(output).get("status") == "complete":
        print("baseline: already complete", flush=True)
        return
    started = time.perf_counter()
    parts, data_identity = _parts(args)
    fit, folds = _fit_folds(parts)
    evaluator, mask, _, identity = _evaluator(args)
    report = _load_json(output) if output.exists() else {
        "schema": SCHEMA + ".fold_baseline", "status": "in_progress", "folds": {}}
    full_identity = {**data_identity, **identity, "part": "fit",
                     "folds": [[fit["conversations"][i]["id"] for i in fold] for fold in folds]}
    if report.get("identity") not in (None, full_identity):
        raise RuntimeError("existing fold baseline describes another problem")
    report["identity"] = full_identity
    for number, fold in enumerate(folds):
        if str(number) in report["folds"]:
            continue
        arms, stats = _run_variants(evaluator, [fit["rows"][i] for i in fold],
                                    _variants(mask, {"base": []}))
        report["folds"][str(number)] = {**arms["base"], "stats": stats}
        _atomic_json(output, report)
        print(f"baseline: fold {number} KL {arms['base']['mean_top20_kl_to_bf16']:.5f} "
              f"in {stats['total_seconds']:.0f} s", flush=True)
    report["status"] = "complete"
    report["wall_seconds"] = report.get("wall_seconds", 0.0) + time.perf_counter() - started
    _atomic_json(output, report)


class Search:
    """Paired antithetic REINFORCE with exact-cardinality Plackett-Luce draws.

    The update is ``search.hard.optimize_hard_reinforce``'s, step for step,
    with state that survives a restart.
    """

    def __init__(self, groups: int, *, lr: float, baseline_decay: float, seed: int) -> None:
        import torch

        self.scores = torch.zeros(groups, dtype=torch.float32, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.scores], lr=lr)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.baseline_decay = baseline_decay
        self.baseline: float | None = None
        self.history: list[dict[str, Any]] = []

    def draw(self):
        import torch

        from search.hard import sample_plackett_luce_assignment

        uniforms = torch.rand(self.scores.shape, generator=self.generator,
                              dtype=self.scores.dtype)
        plus, log_plus = sample_plackett_luce_assignment(self.scores, UPGRADES, uniforms=uniforms)
        minus, log_minus = sample_plackett_luce_assignment(
            self.scores, UPGRADES, uniforms=1.0 - uniforms)
        return (plus, log_plus), (minus, log_minus)

    def update(self, draws, losses) -> dict[str, Any]:
        (plus, log_plus), (minus, log_minus) = draws
        loss_plus, loss_minus = losses
        pair_mean = 0.5 * (loss_plus + loss_minus)
        if self.baseline is None:
            self.baseline = pair_mean
        objective = 0.5 * ((loss_plus - self.baseline) * log_plus
                           + (loss_minus - self.baseline) * log_minus)
        self.optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradient_norm = float(self.scores.grad.norm())
        self.optimizer.step()
        self.baseline = self.baseline_decay * self.baseline + (
            1.0 - self.baseline_decay) * pair_mean
        return {"loss_plus": loss_plus, "loss_minus": loss_minus,
                "baseline": self.baseline, "gradient_norm": gradient_norm,
                "plus": plus.tolist(), "minus": minus.tolist()}

    def final(self) -> list[int]:
        from search.hard import exact_budget_assignment

        return exact_budget_assignment(self.scores.detach(), UPGRADES).tolist()

    def save(self, path: Path) -> None:
        import torch

        temporary = path.with_suffix(".tmp")
        torch.save({"scores": self.scores.detach(), "optimizer": self.optimizer.state_dict(),
                    "generator": self.generator.get_state(), "baseline": self.baseline,
                    "history": self.history}, temporary)
        temporary.replace(path)

    def load(self, path: Path) -> None:
        import torch

        state = torch.load(path, weights_only=False)
        with torch.no_grad():
            self.scores.copy_(state["scores"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"])
        self.baseline = state["baseline"]
        self.history = state["history"]


def _search_config(args) -> dict[str, Any]:
    return {"upgrades": UPGRADES, "groups": len(expert_names()), "folds": FOLDS,
            "steps": args.steps, "lr": args.lr, "baseline_decay": args.baseline_decay,
            "seed": args.seed_base + args.run,
            "fold_schedule": "(run + step) mod 10",
            "objective": "fold mean conversation top-20 KL minus the fold's base"}


def _take_step(search, evaluator, mask, fit, folds, baseline, fold_number):
    draws = search.draw()
    rows = [fit["rows"][i] for i in folds[fold_number]]
    arms, stats = _run_variants(evaluator, rows, _variants(mask, {
        "plus": _names(draws[0][0]), "minus": _names(draws[1][0])}))
    base = baseline["folds"][str(fold_number)]["mean_top20_kl_to_bf16"]
    losses = (arms["plus"]["mean_top20_kl_to_bf16"] - base,
              arms["minus"]["mean_top20_kl_to_bf16"] - base)
    record = search.update(draws, losses)
    record.update({"fold": fold_number, "stats": stats})
    return record


def _baseline(args, data_identity, identity, fit, folds):
    baseline = _load_json(args.reports / f"{PREFIX}_fold_baseline.json")
    expected = {**data_identity, **identity, "part": "fit",
                "folds": [[fit["conversations"][i]["id"] for i in fold] for fold in folds]}
    if baseline.get("status") != "complete" or baseline["identity"] != expected:
        raise RuntimeError("the fold baseline is incomplete or for another problem")
    return baseline


def run_pilot(args) -> None:
    started = time.perf_counter()
    parts, data_identity = _parts(args)
    fit, folds = _fit_folds(parts)
    evaluator, mask, _, identity = _evaluator(args)
    baseline = _baseline(args, data_identity, identity, fit, folds)
    config = _search_config(args)
    search = Search(config["groups"], lr=args.lr, baseline_decay=args.baseline_decay,
                    seed=config["seed"])
    record = _take_step(search, evaluator, mask, fit, folds, baseline, 0)
    seconds = record["stats"]["total_seconds"]
    fold_tokens = [sum(fit["conversations"][i]["token_count"] for i in fold) for fold in folds]
    report = {
        "schema": SCHEMA + ".pilot", "status": "complete",
        "identity": {**data_identity, **identity}, "config": config, "step": record,
        "fold_tokens": fold_tokens,
        "projected_hours_per_run": seconds * args.steps / 3600,
        "wall_seconds": time.perf_counter() - started}
    _atomic_json(args.reports / f"{PREFIX}_pilot.json", report)
    print(f"pilot: step {seconds:.0f} s on fold 0 ({fold_tokens[0]} tokens; folds "
          f"{min(fold_tokens)}-{max(fold_tokens)}); load {record['stats']['load_seconds']:.0f} s, "
          f"install {record['stats']['install_seconds']:.0f} s, forward "
          f"{record['stats']['forward_seconds']:.0f} s; losses {record['loss_plus']:+.5f} "
          f"{record['loss_minus']:+.5f}", flush=True)


def run_search(args) -> None:
    output = args.reports / f"{PREFIX}_search_run{args.run}.json"
    if output.exists() and _load_json(output).get("status") == "complete":
        print(f"search run {args.run}: already complete", flush=True)
        return
    parts, data_identity = _parts(args)
    fit, folds = _fit_folds(parts)
    evaluator, mask, _, identity = _evaluator(args)
    baseline = _baseline(args, data_identity, identity, fit, folds)
    config = _search_config(args)
    run_dir = args.work / f"search_run{args.run}"
    manifest = {"identity": {**data_identity, **identity}, "config": config}
    manifest_path, state_path = run_dir / "run.json", run_dir / "state.pt"
    search = Search(config["groups"], lr=args.lr, baseline_decay=args.baseline_decay,
                    seed=config["seed"])
    if manifest_path.exists():
        if _load_json(manifest_path) != json.loads(json.dumps(manifest)):
            raise RuntimeError(f"{manifest_path} differs from this run; refusing to resume")
        if state_path.exists():
            search.load(state_path)
            print(f"search run {args.run}: resuming at step {len(search.history)}", flush=True)
    else:
        _atomic_json(manifest_path, manifest)
    while len(search.history) < args.steps:
        step = len(search.history)
        record = _take_step(search, evaluator, mask, fit, folds, baseline,
                            (args.run + step) % FOLDS)
        record["step"] = step
        search.history.append(record)
        search.save(state_path)
        recent = [item["stats"]["total_seconds"] for item in search.history[-10:]]
        print(f"search run {args.run}: step {step + 1}/{args.steps} fold {record['fold']} "
              f"loss {record['loss_plus']:+.5f} {record['loss_minus']:+.5f} baseline "
              f"{record['baseline']:+.5f} grad {record['gradient_norm']:.4f} in "
              f"{record['stats']['total_seconds']:.0f} s | ETA "
              f"{(args.steps - step - 1) * np.mean(recent) / 3600:.1f} h", flush=True)
    evaluated = [(item["loss_plus"], item["plus"], item["step"]) for item in search.history] + \
                [(item["loss_minus"], item["minus"], item["step"]) for item in search.history]
    best_loss, best_assignment, best_step = min(evaluated, key=lambda item: item[0])
    names = expert_names()
    final = search.final()
    _atomic_json(output, {
        "schema": SCHEMA + ".search", "status": "complete", **manifest,
        "final": {"assignment": final, "tensors": _names(final)},
        "best_screened": {"assignment": best_assignment, "tensors": _names(best_assignment),
                          "screen_delta_kl": best_loss, "step": best_step,
                          "fold": search.history[best_step]["fold"]},
        "scores": dict(zip(names, search.scores.detach().tolist())),
        "state": {"path": str(state_path), "sha256": _sha256_file(state_path)},
        "history": search.history,
        "search_hours": sum(item["stats"]["total_seconds"] for item in search.history) / 3600,
        "note": "screening only; candidates are compared on the validation part"})
    print(f"search run {args.run}: complete", flush=True)


# -- promotion, replay and the validation gate ---------------------------------------


def _candidates(args, store_report) -> dict[str, list[str]]:
    summary = store_report["summary"]
    candidates: dict[str, list[str]] = {}
    seen: dict[tuple[str, ...], str] = {}
    for run in range(args.runs):
        report = _load_json(args.reports / f"{PREFIX}_search_run{run}.json")
        if report.get("status") != "complete":
            raise RuntimeError(f"search run {run} is not complete")
        for kind in ("final", "best_screened"):
            tensors = tuple(sorted(report[kind]["tensors"]))
            if len(tensors) != UPGRADES:
                raise RuntimeError(f"run {run} {kind} upgrades {len(tensors)} tensors")
            label = f"run{run}_{kind}"
            if tensors in seen:
                print(f"promote: {label} equals {seen[tensors]}", flush=True)
                continue
            seen[tensors] = label
            candidates[label] = list(tensors)
    return {
        "base": [],
        "unpruned": None,
        "control_imatrix": summary["top23_by_weighted_relative_error_reduction"],
        "control_down": summary["top23_down_by_weighted_relative_error_reduction"],
        "ceiling": expert_names(),
        **candidates,
    }


def run_promote(args) -> None:
    output = args.reports / f"{PREFIX}_promotion.json"
    if output.exists() and _load_json(output).get("status") == "complete":
        print("promote: already complete", flush=True)
        return
    started = time.perf_counter()
    parts, data_identity = _parts(args)
    evaluator, mask, store_report, identity = _evaluator(args)
    assignments = _candidates(args, store_report)
    arms, stats = _run_variants(evaluator, parts["validation"]["rows"],
                                _variants(mask, assignments))
    search_labels = [label for label in arms if label.startswith("run")]
    best = min(search_labels, key=lambda label: (arms[label]["mean_top20_kl_to_bf16"], label))
    controls = ("control_imatrix", "control_down")
    beats = {control: arms[best]["mean_top20_kl_to_bf16"] < arms[control]["mean_top20_kl_to_bf16"]
             for control in controls}
    promoted = best if all(beats.values()) else min(
        controls, key=lambda label: arms[label]["mean_top20_kl_to_bf16"])
    base = arms["base"]["conversation_mean_top20_kl"]
    _atomic_json(output, {
        "schema": SCHEMA + ".promotion", "status": "complete",
        "identity": {**data_identity, **identity, "part": "validation",
                     "conversation_ids": [c["id"] for c in parts["validation"]["conversations"]]},
        "rule": ("best search candidate by validation mean KL; it must beat both controls, "
                 "otherwise the better control is promoted and reported as such"),
        "best_search_candidate": best,
        "best_search_candidate_beats_controls": beats,
        "promoted": promoted,
        "promoted_tensors": arms[promoted]["upgrades"],
        "deltas_vs_base": {label: _paired(arm["conversation_mean_top20_kl"], base)
                           for label, arm in arms.items() if label != "base"},
        "ceiling_gain_fraction": {
            label: ((arms[label]["mean_top20_kl_to_bf16"] - arms["base"]["mean_top20_kl_to_bf16"])
                    / (arms["ceiling"]["mean_top20_kl_to_bf16"]
                       - arms["base"]["mean_top20_kl_to_bf16"]))
            for label in arms if label not in ("base", "unpruned", "ceiling")},
        "arms": arms, "stats": stats,
        "wall_seconds": time.perf_counter() - started})
    print(f"promote: {promoted} (best search candidate {best}, beats controls {beats})",
          flush=True)


def run_replay(args) -> int:
    output = args.reports / f"{PREFIX}_replay.json"
    if output.exists() and _load_json(output).get("status") in ("pass", "fail"):
        print(f"replay: already {_load_json(output)['status']}", flush=True)
        return 0
    started = time.perf_counter()
    promotion = _load_json(args.reports / f"{PREFIX}_promotion.json")
    if promotion.get("status") != "complete":
        raise RuntimeError("promotion is not complete")
    parts, data_identity = _parts(args)
    evaluator, mask, _, identity = _evaluator(args)
    arms, stats = _run_variants(evaluator, parts["validation"]["rows"], _variants(mask, {
        "promoted": promotion["promoted_tensors"], "base": [], "unpruned": None}))
    reproduced = {
        label: float(np.max(np.abs(np.asarray(arms[label]["conversation_mean_top20_kl"])
                                   - promotion["arms"][source]["conversation_mean_top20_kl"])))
        for label, source in (("promoted", promotion["promoted"]), ("base", "base"),
                              ("unpruned", "unpruned"))}
    unpruned = arms["unpruned"]["conversation_mean_top20_kl"]
    gate = _paired(arms["promoted"]["conversation_mean_top20_kl"], unpruned)
    passed = gate["ci95"][1] < 0
    _atomic_json(output, {
        "schema": SCHEMA + ".replay", "status": "pass" if passed else "fail",
        "gate": "promoted minus unpruned GSQ-E6 on validation: bootstrap 95% CI upper bound < 0",
        "identity": {**data_identity, **identity, "part": "validation"},
        "promoted": promotion["promoted"], "promoted_tensors": promotion["promoted_tensors"],
        "max_abs_difference_from_promotion_pass": reproduced,
        "promoted_minus_unpruned": gate,
        "base_minus_unpruned": _paired(arms["base"]["conversation_mean_top20_kl"], unpruned),
        "promoted_minus_base": _paired(arms["promoted"]["conversation_mean_top20_kl"],
                                       arms["base"]["conversation_mean_top20_kl"]),
        "arms": arms, "stats": stats,
        "wall_seconds": time.perf_counter() - started})
    print(f"replay: {'pass' if passed else 'fail'}; promoted - unpruned "
          f"{gate['mean_delta']:+.5f} CI {gate['ci95']}; reproduced within {reproduced}",
          flush=True)
    return 0 if passed else 3


# -- main ----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "crosscheck", "baseline", "pilot", "search", "promote", "replay"))
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--baseline-decay", type=float, default=0.9)
    parser.add_argument("--seed-base", type=int, default=20261016)
    parser.add_argument("--corpus", type=Path,
                        default=RCO / "reports/qwen36_chat_corpus_v2_manifest.json")
    parser.add_argument("--split-report", type=Path,
                        default=RCO / "reports/qwen36_chat_corpus_v2_calibration_split.json")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/qwen36_35b_base")
    parser.add_argument("--manifest", type=Path,
                        default=RCO / "reports/qwen36_35b_base_gguf_manifest.json")
    parser.add_argument("--gguf", type=Path,
                        default=ROOT / "results/Qwen3.6-35B-A3B-GSQ-hybrid.gguf")
    parser.add_argument("--gguf-python", type=Path, default=ROOT / "repos/llama.cpp/gguf-py")
    parser.add_argument("--ggml-library", type=Path,
                        default=ROOT / "experiment/build-cpu/bin/libggml-base.so.0.24.0")
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_prune24_q3k")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows-per-chunk", type=int, default=1024)
    parser.add_argument("--install-rows", type=int, default=4096)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--position-chunk", type=int, default=256)
    parser.add_argument("--microbatch-tokens", type=int, default=8448)
    parser.add_argument("--device-hidden-bytes", type=int, default=3 << 30)
    args = parser.parse_args()
    for key in ("corpus", "split_report", "model_dir", "manifest", "gguf", "gguf_python",
                "ggml_library", "work", "reports"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main() -> int:
    args = _parse_args()
    return {"crosscheck": run_crosscheck, "baseline": run_baseline, "pilot": run_pilot,
            "search": run_search, "promote": run_promote,
            "replay": run_replay}[args.command](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
