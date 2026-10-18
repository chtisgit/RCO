#!/usr/bin/env python3
"""Phase 9 cross-runtime logit parity of RCO_PLAN_NEW.md.

The Phase 4 differential audit (``audit_qwen36_p24_logit_parity.py``) with
three variants and the same tokens, helpers and settings:

``control``    native unpruned GSQ-E6 / ``results/Qwen3.6-35B-A3B-GSQ-E6-UNGATED.gguf``
``base``       native GSQ-E6 with the candidate's frequency mask (exact routing) /
               the diagnostic ``results/Qwen3.6-35B-A3B-GSQ-E6-P24-FREQ-UNGATED.gguf``
``candidate``  base plus the 23 Q3_K upgrades from the Phase 6 store /
               the candidate ``results/Qwen3.6-35B-A3B-GSQ-RCO-v2-UNGATED.gguf``

The question: does llama.cpp compute the upgrades as the native evaluator
does?  The native side always streams the 256-expert authentic GSQ GGUF.

Setup check: the control is measured again and must reproduce Phase 4's
control measurements exactly (``data/qwen36_p24_parity/control_*.npz``).
If it does not, the run stops before any other variant is read.

Pass criteria (Phase 4's), for base and for candidate against control:
mean logit RMSE <= 1.2 x control; top-1 agreement >= control - 0.03; every
layer's mean relative RMSE <= 1.2 x control + 0.01.  For information: the
upgrade effect (candidate - base) and the pruning effect (base - control),
compared between the runtimes, and the strict thresholds.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_native_llama_logit_parity import (  # noqa: E402
    _comparison,
    _layer_comparison,
    _read_llama_layers,
    _read_llama_logits,
    _write_bundle,
)
from audit_qwen36_p24_logit_parity import (  # noqa: E402
    LAYER_RATIO,
    LAYER_SLACK,
    LOGIT_RMSE_RATIO,
    TOP1_AGREEMENT_SLACK,
    _documents,
    _effect,
    _run_helper,
)
from audit_qwen36_prune24_prelim import RouterHooks  # noqa: E402
from audit_qwen36_q3k_viability import (  # noqa: E402
    _atomic_json,
    _empty_model,
    _expert_weight_store_class,
    _load_json,
    _sha256_file,
)

SCHEMA = "rco.qwen36.rco_v2_logit_parity.v1"
VARIANTS = ("control", "base", "candidate")
COMPARED = ("base", "candidate")


class CombinedWeightStore:
    """The Q6_K embedding store and the Q3_K expert upgrade store as one,
    dispatching by candidate name (choice 0 retains, choice 1 installs)."""

    cache = False

    def __init__(self, embedding_name: str, embedding_store: Any, expert_store: Any) -> None:
        self.embedding_name = embedding_name
        self.embedding_store = embedding_store
        self.expert_store = expert_store

    def _store(self, name: str) -> Any:
        return self.embedding_store if name == self.embedding_name else self.expert_store

    def candidate_location(self, name: str) -> int | str:
        return self._store(name).candidate_location(name)

    def is_retain_choice(self, name: str, choice: int) -> bool:
        return self._store(name).is_retain_choice(name, choice)

    def get_layer_storage_bytes(self, name: str, choice: int) -> int:
        return self._store(name).get_layer_storage_bytes(name, choice)

    def install_layer_weight(self, model: Any, name: str, choice: int) -> dict[str, int]:
        return self._store(name).install_layer_weight(model, name, choice)


def assignments(upgrades: list[str]) -> dict[str, list[int]]:
    """Group order: the embedding, then the upgrades.  Every variant uses the
    Q6_K embedding; only the candidate installs the upgrades."""
    retain = [1] + [0] * len(upgrades)
    return {"control": retain, "base": retain, "candidate": [1] * (len(upgrades) + 1)}


def _evaluator(args, model, upgrades: list[str], store_report: dict) -> Any:
    import torch

    from audit_qwen36_gsq_e6 import NAME, EmbeddingQ6KWeightStore, _entry
    from gguf_parallel_stream import ParallelGGUFManifestPrefixLoader
    from native_store import NativeCandidateStore
    from quant.ggml_native import GGMLNativeCodec, GGMLType
    from search.streaming import StreamingHardCausalEvaluator

    codec = GGMLNativeCodec(args.ggml_library)
    manifest = _load_json(args.manifest)
    embedding = _load_json(args.reports / "qwen36_gsq_e6_embedding.json")
    embedding_store = EmbeddingQ6KWeightStore(
        codec, Path(embedding["payload"]["path"]), embedding["payload"]["sha256"],
        _entry(args.manifest), rows_per_chunk=4096)
    expert_store = _expert_weight_store_class()(
        NativeCandidateStore(Path(store_report["store"]["root"]), codec), manifest,
        args.model_dir, GGMLType.Q3_K, rows_per_chunk=args.install_rows)
    loader = ParallelGGUFManifestPrefixLoader(
        args.gguf, manifest, args.model_dir,
        gguf_python=args.gguf_python, ggml_library=args.ggml_library,
        rows_per_chunk=args.rows_per_chunk, workers=args.decode_workers)
    groups = [SimpleNamespace(layer_names=(name,)) for name in (NAME, *upgrades)]
    return StreamingHardCausalEvaluator(
        model, loader, CombinedWeightStore(NAME, embedding_store, expert_store), groups, [0, 1],
        device=torch.device(args.device), vocab_chunk_size=args.vocab_chunk_size,
        checkpoint_dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[args.native_compute_dtype])


def _measure(args, evaluator, assignment, model, mask, variant, gguf, document) -> Path:
    """As Phase 4's ``_measure``; the mask applies to every pruned variant."""
    path = args.work / f"{variant}_{document['id']}.npz"
    if path.exists():
        return path
    import torch

    tokens = document["tokens"]
    hooks = RouterHooks(model, prune_mask=None if variant == "control" else mask, collect=False)
    try:
        native = evaluator.evaluate(
            torch.tensor([tokens], dtype=torch.long), torch.tensor(assignment, dtype=torch.long),
            capture_logit_positions=tuple(range(len(tokens) - 1)),
            capture_layer_outputs=True)
    finally:
        hooks.close()
    native_layers = np.stack([native.captured_model_input.numpy(),
                              *(value.numpy() for value in native.captured_layer_outputs)])
    with tempfile.TemporaryDirectory(prefix="rco-v2-parity-") as temporary:
        bundle = Path(temporary) / "input.bundle"
        raw_logits = Path(temporary) / "logits.bin"
        raw_layers = Path(temporary) / "layers.bin"
        _write_bundle(bundle, document["id"], document["text"], tokens)
        _run_helper(args.helper, gguf, bundle, raw_logits, args)
        llama_tokens, llama_logits = _read_llama_logits(raw_logits)
        _run_helper(args.layer_helper, gguf, bundle, raw_layers, args,
                    "--expected-layers", str(len(native.captured_layer_outputs)))
        llama_layers = _read_llama_layers(raw_layers)[:, :len(tokens) - 1]
    if llama_tokens.tolist() != tokens:
        raise RuntimeError("llama.cpp helper returned different token IDs")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, tokens=np.asarray(tokens, dtype=np.int32),
             native=native.captured_logits.numpy(), llama=llama_logits,
             native_layers=native_layers, llama_layers=llama_layers)
    temporary.replace(path)
    print(json.dumps({"measured": variant, "document": document["id"],
                      "tokens": len(tokens), "native_loss": native.loss}), flush=True)
    return path


def reproduction(new: Path, old: Path) -> dict[str, Any]:
    """Whether a remeasured control file equals Phase 4's, array by array."""
    a, b = np.load(new), np.load(old)
    arrays = {}
    for key in ("tokens", "native", "llama", "native_layers", "llama_layers"):
        same = a[key].shape == b[key].shape and np.array_equal(a[key], b[key])
        arrays[key] = {"identical": bool(same),
                       "max_abs_difference": None if a[key].shape != b[key].shape else
                       float(np.max(np.abs(a[key].astype(np.float64) - b[key].astype(np.float64))))}
    return {"identical": all(item["identical"] for item in arrays.values()), "arrays": arrays}


def summarize(paths, documents) -> dict[str, dict[str, Any]]:
    per_variant = {}
    for variant in VARIANTS:
        rmse, top1, strict_pass, layer_sums = [], [], True, None
        for document in documents:
            data = np.load(paths[(variant, document["id"])])
            positions, passed = _comparison(data["native"], data["llama"], document["tokens"])
            layers, first_failed = _layer_comparison(data["native_layers"], data["llama_layers"])
            strict_pass = strict_pass and passed and first_failed is None
            rmse += [item["rmse"] for item in positions]
            top1 += [item["top1_agreement"] for item in positions]
            relative = np.array([[item["relative_rmse"] for item in state["positions"]]
                                 for state in layers])
            layer_sums = relative if layer_sums is None else np.concatenate(
                [layer_sums, relative], axis=1)
        per_variant[variant] = {
            "positions": len(rmse), "mean_logit_rmse": float(np.mean(rmse)),
            "top1_agreement_rate": float(np.mean(top1)),
            "layer_mean_relative_rmse": layer_sums.mean(axis=1).tolist(),
            "strict_original_thresholds_pass": strict_pass}
    return per_variant


def gates(per_variant: dict[str, dict[str, Any]], variant: str) -> dict[str, Any]:
    control, other = per_variant["control"], per_variant[variant]
    limits = [LAYER_RATIO * value + LAYER_SLACK for value in control["layer_mean_relative_rmse"]]
    failing = [index - 1 for index, (value, limit) in enumerate(
        zip(other["layer_mean_relative_rmse"], limits)) if value > limit]
    ratio = other["mean_logit_rmse"] / control["mean_logit_rmse"]
    top1 = other["top1_agreement_rate"] - control["top1_agreement_rate"]
    result = {
        "logit_rmse_ratio": {"value": ratio, "limit": LOGIT_RMSE_RATIO,
                             "passed": ratio <= LOGIT_RMSE_RATIO},
        "top1_agreement_difference": {"value": top1, "limit": -TOP1_AGREEMENT_SLACK,
                                      "passed": top1 >= -TOP1_AGREEMENT_SLACK},
        "layer_relative_rmse": {"failing_layers": failing, "ratio": LAYER_RATIO,
                                "slack": LAYER_SLACK, "passed": not failing},
    }
    result["passed"] = all(gate["passed"] for gate in result.values())
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    reports = {"control": _load_json(args.control_report), "base": _load_json(args.base_report),
               "candidate": _load_json(args.candidate_report)}
    ggufs = {}
    for variant, report in reports.items():
        path = Path(report["output"]["path"])
        if path.stat().st_size != report["output"]["bytes"]:
            raise RuntimeError(f"{variant} GGUF size differs from its build report")
        ggufs[variant] = path
    if reports["control"].get("variant") != "unpruned_parity_control":
        raise RuntimeError("control build report is not the unpruned parity control")
    store_report = _load_json(args.reports / "qwen36_prune24_q3k_store.json")
    mask_sha = store_report["mask"]["sha256"]
    for variant in COMPARED:
        if reports[variant]["inputs"]["mask"]["sha256"] != mask_sha:
            raise RuntimeError(f"{variant} GGUF was built with another mask than the Q3_K store")
    mask_path = Path(store_report["mask"]["path"])
    if _sha256_file(mask_path) != mask_sha:
        raise RuntimeError("frequency mask differs from its reports")
    if _sha256_file(Path(store_report["store"]["root"]) / "native-candidate-index.json") \
            != store_report["store"]["index_sha256"]:
        raise RuntimeError("Q3_K store index differs from its report")
    mask = np.load(mask_path)
    upgrades = list(reports["candidate"]["upgraded_tensors"])
    plan = assignments(upgrades)

    documents, calibration = _documents(args)
    model = _empty_model(args.model_dir)
    evaluator = _evaluator(args, model, upgrades, store_report)
    paths: dict[tuple[str, str], Path] = {}
    for document in documents:
        paths[("control", document["id"])] = _measure(
            args, evaluator, plan["control"], model, mask, "control", ggufs["control"], document)
    reproduced = {document["id"]: reproduction(
        paths[("control", document["id"])],
        args.phase4_work / f"control_{document['id']}.npz") for document in documents}
    if not all(item["identical"] for item in reproduced.values()):
        return {"schema": SCHEMA, "status": "decision",
                "reason": "the control does not reproduce Phase 4's measurements exactly",
                "control_reproduction": reproduced}
    for variant in COMPARED:
        for document in documents:
            paths[(variant, document["id"])] = _measure(
                args, evaluator, plan[variant], model, mask, variant, ggufs[variant], document)

    per_variant = summarize(paths, documents)
    variant_gates = {variant: gates(per_variant, variant) for variant in COMPARED}
    effects = []
    for document in documents:
        data = {variant: np.load(paths[(variant, document["id"])]) for variant in VARIANTS}

        def effect(a: str, b: str) -> dict[str, Any]:
            return {"logits": _effect(data[a]["native"] - data[b]["native"],
                                      data[a]["llama"] - data[b]["llama"]),
                    "layers": [_effect(data[a]["native_layers"][s] - data[b]["native_layers"][s],
                                       data[a]["llama_layers"][s] - data[b]["llama_layers"][s])
                               for s in range(1, data[a]["native_layers"].shape[0])]}

        effects.append({"document": document["id"],
                        "upgrades_candidate_minus_base": effect("candidate", "base"),
                        "pruning_base_minus_control": effect("base", "control")})

    def _artifact(path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": _sha256_file(path)}

    import torch
    import transformers
    return {
        "schema": SCHEMA,
        "status": "pass" if all(g["passed"] for g in variant_gates.values()) else "fail",
        "question": "does llama.cpp compute the candidate's Q3_K upgrades (and pruning) as the "
                    "native evaluator does, beyond the unpruned control's divergence?",
        "control_reproduction": reproduced,
        "gates": variant_gates,
        "variants": per_variant,
        "documents": [{key: document[key] for key in ("id", "index", "stratum")}
                      | {"tokens": len(document["tokens"])} for document in documents],
        "effects": effects,
        "upgraded_tensors": upgrades,
        "artifacts": {
            **{f"{variant}_gguf_report": _artifact(path) for variant, path in (
                ("control", args.control_report), ("base", args.base_report),
                ("candidate", args.candidate_report))},
            **{f"{variant}_gguf_sha256": reports[variant]["output"]["sha256"]
               for variant in VARIANTS},
            "native_gsq_gguf": {"path": str(args.gguf)},
            "mask": {"path": str(mask_path), "sha256": mask_sha},
            "q3k_store_index_sha256": store_report["store"]["index_sha256"],
            "helper": _artifact(args.helper), "layer_helper": _artifact(args.layer_helper),
            "calibration": calibration | {"documents": None},
            "measurements": {f"{variant}_{name}": _artifact(path)
                             for (variant, name), path in paths.items()},
        },
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "device": args.device,
            "native_compute_dtype": args.native_compute_dtype,
            "gpu_layers": args.gpu_layers, "threads": args.threads, "ubatch": args.ubatch,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--control-report", type=Path,
                        default=RCO / "reports/qwen36_gsq_e6_gguf.json")
    parser.add_argument("--base-report", type=Path,
                        default=RCO / "reports/qwen36_gsq_e6_p24_frequency_gguf.json")
    parser.add_argument("--candidate-report", type=Path,
                        default=RCO / "reports/qwen36_gsq_rco_v2_candidate_gguf.json")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/qwen36_35b_base")
    parser.add_argument("--manifest", type=Path,
                        default=RCO / "reports/qwen36_35b_base_gguf_manifest.json")
    parser.add_argument("--calibration", type=Path,
                        default=RCO / "reports/qwen36_35b_calibration_corpus_v2_manifest.json")
    parser.add_argument("--gguf", type=Path,
                        default=ROOT / "results/Qwen3.6-35B-A3B-GSQ-hybrid.gguf")
    parser.add_argument("--gguf-python", type=Path, default=ROOT / "repos/llama.cpp/gguf-py")
    parser.add_argument("--ggml-library", type=Path,
                        default=ROOT / "experiment/build-cpu/bin/libggml-base.so.0.24.0")
    parser.add_argument("--helper", type=Path,
                        default=ROOT / "experiment/helpers/rco-gguf-logit-dump")
    parser.add_argument("--layer-helper", type=Path,
                        default=ROOT / "experiment/helpers/rco-gguf-layer-dump")
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_rco_v2_parity")
    parser.add_argument("--phase4-work", type=Path, default=ROOT / "data/qwen36_p24_parity")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--output", type=Path,
                        default=RCO / "reports/qwen36_gsq_rco_v2_logit_parity.json")
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--native-compute-dtype", choices=("bf16", "fp32"), default="fp32")
    parser.add_argument("--rows-per-chunk", type=int, default=1024)
    parser.add_argument("--install-rows", type=int, default=4096)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ubatch", type=int, default=64)
    args = parser.parse_args()
    for key in ("control_report", "base_report", "candidate_report", "model_dir", "manifest",
                "calibration", "gguf", "gguf_python", "ggml_library", "helper", "layer_helper",
                "work", "phase4_work", "reports", "output"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main() -> int:
    args = _parse_args()
    report = audit(args)
    _atomic_json(args.output, report)
    print(json.dumps({"status": report["status"], "gates": report.get("gates"),
                      "control_reproduced": all(item["identical"] for item in
                                                report["control_reproduction"].values())},
                     indent=2), flush=True)
    return {"pass": 0, "fail": 3, "decision": 4}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
