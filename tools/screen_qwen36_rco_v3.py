#!/usr/bin/env python3
"""Step 2 of RCO_PLAN_V3.md: every single change from the center, alone.

620 changes, each one tensor away from the center GGUF:

- 40 ``down`` back to IQ2_XS;
- 80 ``gate``/``up`` up to IQ3_XXS;
- 250 dense (attention/SSM, shared expert) up to Q8_0;
- 250 dense down to Q4_K.

Data: fit folds 0 and 1 (``search_qwen36_prune24_q3k.stratified_folds``,
10 folds of the 160 fit conversations; the search uses the same folds).
Each change's effect is its per-conversation mean top-20 KL minus the
center's, paired over the 32 conversations, with a paired bootstrap 95% CI,
and its rate ``kl_per_gigabyte_added``: that KL change divided by the bytes
it adds on the GPU (without ``token_embd``), in GB.  Usually negative in both
directions: an upgrade lowers KL by that much per GB bought, a downgrade
raises it by that much per GB saved.

Subcommands:

``check``
    ``SharedPrefixExactEvaluator`` (hidden states in host memory) against
    the plain evaluator (hidden states on the GPU) on fold 0: the center and
    one change of each kind, at different blocks.  Every per-conversation
    KL and NLL must agree within 1e-9.  Exit 4 otherwise.
``screen``
    The 620 changes on each fold, in chunks that fit ``--host-hidden-bytes``
    of hidden states; each chunk also evaluates the center, which must
    equal the fold's first center exactly.  Resumable per chunk.  When both
    folds are done, the per-change summary is written.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

RCO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file  # noqa: E402
from rco_v3_common import (  # noqa: E402
    EVALUATOR_PATHS,
    add_evaluator_arguments,
    center_assignment,
    evaluator,
    load_store_reports,
    searched,
    store_entry,
    upgrade_keys,
)
from search_qwen36_prune24_q3k import FOLDS, _paired, _parts, stratified_folds  # noqa: E402

SCHEMA = "rco.qwen36.rco_v3_screening.v1"
CHECK_OUTPUT = "rco_v3/qwen36_rco_v3_screening_check.json"
OUTPUT = "rco_v3/qwen36_rco_v3_screening.json"
SCREEN_FOLDS = (0, 1)
TOLERANCE = 1e-9
CHECK_CHANGES = (("blk.3.ffn_down_exps.weight", "IQ2_XS"),
                 ("blk.17.ffn_gate_exps.weight", "IQ3_XXS"),
                 ("blk.25.attn_qkv.weight", "Q8_0"),
                 ("blk.39.ffn_down_shexp.weight", "Q4_K"))


def layer(name: str) -> int:
    return int(name.split(".")[1])


def changes() -> list[tuple[str, str, str]]:
    """(kind, tensor, type) for the 620 single changes, in a fixed order."""
    groups = searched()
    dense = groups["attention"] + groups["shared_expert"]
    return ([("down_iq2_xs", n, "IQ2_XS") for n in groups["down"]]
            + [("gate_up_iq3_xxs", n, "IQ3_XXS") for n in groups["gate_up"]]
            + [("dense_q8_0", n, "Q8_0") for n in dense]
            + [("dense_q4_k", n, "Q4_K") for n in dense])


def label(tensor: str, type_name: str) -> str:
    return f"{tensor}:{type_name}"


def _variants(items, center):
    from search.streamed_chat_exact import ExactVariant

    out = [ExactVariant("center")]
    for tensor, type_name in items:
        out.append(ExactVariant(label(tensor, type_name),
                                upgrades=upgrade_keys({**center, tensor: type_name}, center)))
    return out


def _arms(result, variants):
    return {v.label: {"conversation_mean_top20_kl": result.row_mean_kl(n).tolist(),
                      "conversation_mean_nll": result.row_mean_nll(n).tolist()}
            for n, v in enumerate(variants)}


def _setup(args):
    from search.streamed_chat_exact_shared import SharedPrefixExactEvaluator

    gguf_report = _load_json(args.gguf_report)
    if gguf_report.get("status") != "built" or gguf_report["assignment_source"] != "center":
        raise RuntimeError("the center GGUF is not built")
    gguf_path = Path(gguf_report["output"]["path"])
    reports = load_store_reports()
    parts, data_identity = _parts(args)
    fit = parts["fit"]
    folds = stratified_folds(fit["conversations"], FOLDS)
    exact, identity = evaluator(args, gguf_path, reports, SharedPrefixExactEvaluator)
    if identity["gguf_sha256"] != gguf_report["output"]["sha256"]:
        raise RuntimeError("center GGUF differs from its report")
    identity = {**data_identity, **identity, "part": "fit",
                "folds": {str(f): [fit["conversations"][i]["id"] for i in folds[f]]
                          for f in SCREEN_FOLDS}}
    return exact, identity, reports, fit, folds


def run_check(args) -> int:
    from search.streamed_chat_exact import StreamedChatExactEvaluator

    output = args.reports / CHECK_OUTPUT
    if output.exists() and _load_json(output).get("status") in ("pass", "fail"):
        print(f"check: already done ({_load_json(output)['status']})", flush=True)
        return 0 if _load_json(output)["status"] == "pass" else 4
    started = time.perf_counter()
    exact, identity, _, fit, folds = _setup(args)
    center = center_assignment()
    variants = _variants([(t, k) for t, k in CHECK_CHANGES], center)
    rows = [fit["rows"][i] for i in folds[0]]
    plain = StreamedChatExactEvaluator.evaluate(exact, rows, variants)
    device_hidden = exact.device_hidden_bytes
    exact.device_hidden_bytes = 0
    try:
        shared = exact.evaluate(rows, variants)
    finally:
        exact.device_hidden_bytes = device_hidden
    a, b = _arms(plain, variants), _arms(shared, variants)
    differences = {v.label: max(
        float(np.max(np.abs(np.asarray(a[v.label][k]) - b[v.label][k])))
        for k in ("conversation_mean_top20_kl", "conversation_mean_nll")) for v in variants}
    ok = all(d <= TOLERANCE for d in differences.values()) and not shared.stats.hidden_on_device
    _atomic_json(output, {
        "schema": SCHEMA + ".check", "status": "pass" if ok else "fail",
        "gate": f"shared-prefix evaluator (host hidden) equals the plain one (GPU hidden) "
                f"within {TOLERANCE}, per conversation, KL and NLL",
        "identity": identity, "changes": [list(c) for c in CHECK_CHANGES],
        "max_abs_difference": differences,
        "means": {k: float(np.mean(v["conversation_mean_top20_kl"])) for k, v in b.items()},
        "stats": {"plain": asdict(plain.stats), "shared": asdict(shared.stats)},
        "wall_seconds": time.perf_counter() - started})
    print(f"check: {'pass' if ok else 'fail'} {differences}; plain "
          f"{plain.stats.total_seconds:.0f} s, shared {shared.stats.total_seconds:.0f} s", flush=True)
    return 0 if ok else 4


def _chunk_size(args, exact, rows) -> int:
    from search.streamed_prune_chat import microbatches

    lengths = [int(r.tokens.numel()) for r in rows]
    padded = sum(len(m) * lengths[m[0]] for m in microbatches(lengths, exact.microbatch_tokens))
    per_variant = padded * exact.adapter.text_model.config.hidden_size * 2
    return max(1, args.host_hidden_bytes // per_variant - 1)


def run_screen(args) -> int:
    output = args.reports / OUTPUT
    if output.exists() and _load_json(output).get("status") == "complete":
        print("screen: already complete", flush=True)
        return 0
    started = time.perf_counter()
    exact, identity, reports, fit, folds = _setup(args)
    center = center_assignment()
    items = changes()
    report = _load_json(output) if output.exists() else {
        "schema": SCHEMA, "status": "in_progress", "folds": {}}
    if report.get("identity") not in (None, identity):
        raise RuntimeError("existing screening report describes another problem")
    report["identity"] = identity
    order = sorted(items, key=lambda c: (layer(c[1]), items.index(c)))
    for fold in SCREEN_FOLDS:
        rows = [fit["rows"][i] for i in folds[fold]]
        size = _chunk_size(args, exact, rows)
        done = report["folds"].setdefault(str(fold), {"center": None, "arms": {}, "chunks": []})
        while True:
            left = [(t, k) for _, t, k in order if label(t, k) not in done["arms"]]
            if not left:
                break
            chunk = left[:size]
            variants = _variants(chunk, center)
            result = exact.evaluate(rows, variants)
            arms = _arms(result, variants)
            if done["center"] is None:
                done["center"] = arms["center"]
            elif arms["center"] != done["center"]:
                raise RuntimeError(f"fold {fold}: the center differs between chunks")
            for v in variants[1:]:
                done["arms"][v.label] = arms[v.label]
            done["chunks"].append({"variants": len(chunk), "first": variants[1].label,
                                   "last": variants[-1].label, "stats": asdict(result.stats)})
            _atomic_json(output, report)
            print(f"screen: fold {fold} {len(done['arms'])}/{len(items)} "
                  f"({len(chunk)} in {result.stats.total_seconds:.0f} s; load "
                  f"{result.stats.load_seconds:.0f}, forward {result.stats.forward_seconds:.0f}, "
                  f"install {result.stats.install_seconds:.0f}, {result.stats.block_loads} loads)",
                  flush=True)
    report["summary"] = _summary(report, items, center, reports, fit, folds)
    report["status"] = "complete"
    report["metric"] = ("per-conversation mean top-20 KL(BF16 || model), not renormalized; change "
                        "minus center over folds 0 and 1, paired bootstrap 95% CI; "
                        "kl_per_gigabyte_added = KL change / bytes added on the GPU "
                        "(without token_embd), in GB")
    report["wall_seconds"] = report.get("wall_seconds", 0.0) + time.perf_counter() - started
    _atomic_json(output, report)
    for kind, group in report["summary"]["by_kind"].items():
        print(f"screen: {kind}: {group['changes']} changes, median rate "
              f"{group['median_kl_per_gigabyte_added']:+.4f}/GB, "
              f"{group['significant']} significant", flush=True)
    return 0


def _summary(report, items, center, reports, fit, folds):
    strata = [fit["conversations"][i]["stratum"] for f in SCREEN_FOLDS for i in folds[f]]
    base_kl = [x for f in SCREEN_FOLDS
               for x in report["folds"][str(f)]["center"]["conversation_mean_top20_kl"]]
    base_nll = [x for f in SCREEN_FOLDS
                for x in report["folds"][str(f)]["center"]["conversation_mean_nll"]]
    per_change = {}
    for kind, tensor, type_name in items:
        name = label(tensor, type_name)
        kl = [x for f in SCREEN_FOLDS
              for x in report["folds"][str(f)]["arms"][name]["conversation_mean_top20_kl"]]
        nll = [x for f in SCREEN_FOLDS
               for x in report["folds"][str(f)]["arms"][name]["conversation_mean_nll"]]
        added = (int(store_entry(reports, tensor, type_name)["payload_bytes"])
                 - int(store_entry(reports, tensor, center[tensor])["payload_bytes"]))
        pair = _paired(kl, base_kl)
        per_change[name] = {
            "kind": kind, "tensor": tensor, "from": center[tensor], "to": type_name,
            "layer": layer(tensor), "gpu_bytes_added": added,
            "kl": pair, "nll": _paired(nll, base_nll),
            "kl_per_gigabyte_added": pair["mean_delta"] / (added / 1e9),
            "per_fold_kl_delta": {str(f): float(np.mean(
                np.asarray(report["folds"][str(f)]["arms"][name]["conversation_mean_top20_kl"])
                - report["folds"][str(f)]["center"]["conversation_mean_top20_kl"]))
                for f in SCREEN_FOLDS}}
    by_kind = {}
    for kind in dict.fromkeys(k for k, _, _ in items):
        group = [c for c in per_change.values() if c["kind"] == kind]
        rates = [c["kl_per_gigabyte_added"] for c in group]
        by_kind[kind] = {
            "changes": len(group),
            "sum_kl_delta": float(sum(c["kl"]["mean_delta"] for c in group)),
            "sum_gpu_bytes_added": int(sum(c["gpu_bytes_added"] for c in group)),
            "median_kl_per_gigabyte_added": float(np.median(rates)),
            "significant": sum(c["kl"]["ci95"][1] < 0 or c["kl"]["ci95"][0] > 0 for c in group)}
    return {"center_mean_top20_kl": float(np.mean(base_kl)), "conversations": len(base_kl),
            "strata": strata, "by_kind": by_kind, "changes": per_change}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("check", "screen"))
    add_evaluator_arguments(parser)
    parser.add_argument("--gguf-report", type=Path,
                        default=RCO / "reports/rco_v3/qwen36_rco_v3_center_gguf.json")
    parser.add_argument("--host-hidden-bytes", type=int, default=8_000_000_000)
    args = parser.parse_args()
    for key in (*EVALUATOR_PATHS, "gguf_report"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main() -> int:
    args = _parse_args()
    return run_check(args) if args.command == "check" else run_screen(args)


if __name__ == "__main__":
    raise SystemExit(main())
