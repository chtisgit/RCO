#!/usr/bin/env python3
"""Phase 4 of RCO_PLAN_NEW.md: differential logit parity for the pruned model.

Strict native-vs-llama.cpp logit parity has never passed for this
architecture, not even unpruned.  The gap starts at layer 1 and grows through
the recurrent layers (``reports/qwen36_native_llama_logit_parity.json``,
``old/RCO_DIAGNOSTIC.md``).  This audit therefore asks the narrower Phase 4
question: does pruning add divergence?  It runs the same comparison on two
variants with the same tokens and settings:

``control``
    Native unpruned GSQ-E6 against llama.cpp on the unpruned GSQ-E6 GGUF.
``p24``
    Native GSQ-E6 with the promoted mask applied by exact routing (pruned
    router logits -inf, softmax, top-8, renormalize), against llama.cpp on
    the P24 GGUF, where the pruned experts are physically removed.

The native side always streams the 256-expert authentic GSQ GGUF, so the two
pruning implementations are independent.

Inputs are prefixes of the first calibration v2 document in each stratum.
Each prefix is the longest one of at most ``--prefix-tokens`` tokens that
round-trips through the tokenizer, which the llama.cpp helpers require.

Pass criteria (declared before the first measurement; see RCO_PLAN_NEW.md):

* mean per-position logit RMSE: p24 <= 1.2 x control;
* top-1 agreement rate: p24 >= control - 0.03;
* every layer state: mean relative RMSE p24 <= 1.2 x control + 0.01;
* the native run never selects a pruned expert, and llama.cpp runs the P24
  GGUF.

The strict absolute thresholds of the original audit are also reported for
both variants, for information only.  So is the pruning effect itself:
how well ``p24 - control`` agrees between the two runtimes.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
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
    MAXIMUM_LAYER_RELATIVE_RMSE,
    MAXIMUM_POSITION_MEAN_ABSOLUTE_ERROR,
    MAXIMUM_POSITION_RMSE,
    MINIMUM_LAYER_COSINE_SIMILARITY,
    MINIMUM_POSITION_COSINE_SIMILARITY,
    _comparison,
    _layer_comparison,
    _read_llama_layers,
    _read_llama_logits,
    _write_bundle,
)
from audit_qwen36_prune24_prelim import RouterHooks  # noqa: E402
from audit_qwen36_q3k_viability import (  # noqa: E402
    _atomic_json,
    _calibration,
    _empty_model,
    _load_json,
    _sha256_file,
)

SCHEMA = "rco.qwen36.p24_logit_parity.v1"
VARIANTS = ("control", "p24")
STRATA = ("language", "knowledge", "reasoning", "code", "multilingual")
LOGIT_RMSE_RATIO = 1.2
TOP1_AGREEMENT_SLACK = 0.03
LAYER_RATIO = 1.2
LAYER_SLACK = 0.01


def _evaluator(args: argparse.Namespace, model: Any) -> tuple[Any, Any]:
    """The Phase 3 GSQ-E6 evaluator, with a selectable compute dtype."""
    import torch

    from audit_qwen36_gsq_e6 import NAME, EmbeddingQ6KWeightStore, _entry
    from gguf_parallel_stream import ParallelGGUFManifestPrefixLoader
    from quant.ggml_native import GGMLNativeCodec
    from search.streaming import StreamingHardCausalEvaluator

    embedding = _load_json(args.reports / "qwen36_gsq_e6_embedding.json")
    store = EmbeddingQ6KWeightStore(
        GGMLNativeCodec(args.ggml_library), Path(embedding["payload"]["path"]),
        embedding["payload"]["sha256"], _entry(args.manifest), rows_per_chunk=4096)
    loader = ParallelGGUFManifestPrefixLoader(
        args.gguf, _load_json(args.manifest), args.model_dir,
        gguf_python=args.gguf_python, ggml_library=args.ggml_library,
        rows_per_chunk=args.rows_per_chunk, workers=args.decode_workers)
    evaluator = StreamingHardCausalEvaluator(
        model, loader, store, [SimpleNamespace(layer_names=(NAME,))], [0, 1],
        device=torch.device(args.device), vocab_chunk_size=args.vocab_chunk_size,
        checkpoint_dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[args.native_compute_dtype])
    return evaluator, torch.ones(1, dtype=torch.long)


def _documents(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from transformers import AutoTokenizer

    input_ids, calibration = _calibration(args)
    manifest = _load_json(args.calibration)["canonical_manifest"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    documents = []
    for stratum in STRATA:
        index = next(i for i, document in enumerate(manifest["documents"])
                     if document["stratum"] == stratum)
        row = input_ids[index].tolist()
        for length in range(min(args.prefix_tokens, len(row)), 1, -1):
            tokens = row[:length]
            text = tokenizer.decode(tokens, skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
            if tokenizer.encode(text, add_special_tokens=False) == tokens:
                break
        else:
            raise RuntimeError(f"no round-tripping prefix for document {index}")
        documents.append({"id": manifest["documents"][index]["id"], "index": index,
                          "stratum": stratum, "tokens": tokens, "text": text})
    return documents, calibration


def _run_helper(helper: Path, gguf: Path, bundle: Path, output: Path,
                args: argparse.Namespace, *extra: str) -> None:
    completed = subprocess.run([
        str(helper), "--model", str(gguf), "--input", str(bundle), "--output", str(output),
        *extra, "--gpu-layers", str(args.gpu_layers), "--threads", str(args.threads),
        "--ubatch", str(args.ubatch)], check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{helper.name} failed ({completed.returncode}): "
                           f"{completed.stderr[-4000:]}")


def _measure(args, evaluator, assignment, model, mask, variant, gguf, document) -> Path:
    """Native and llama.cpp logits and layer states for one variant and document."""
    path = args.work / f"{variant}_{document['id']}.npz"
    if path.exists():
        return path
    import torch

    tokens = document["tokens"]
    hooks = RouterHooks(model, prune_mask=mask if variant == "p24" else None, collect=False)
    try:
        native = evaluator.evaluate(
            torch.tensor([tokens], dtype=torch.long), assignment,
            capture_logit_positions=tuple(range(len(tokens) - 1)),
            capture_layer_outputs=True)
    finally:
        hooks.close()
    native_layers = np.stack([native.captured_model_input.numpy(),
                              *(value.numpy() for value in native.captured_layer_outputs)])
    with tempfile.TemporaryDirectory(prefix="rco-p24-parity-") as temporary:
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


def _effect(native_delta: np.ndarray, llama_delta: np.ndarray) -> dict[str, float]:
    native_delta = native_delta.astype(np.float64).ravel()
    llama_delta = llama_delta.astype(np.float64).ravel()
    norm = float(np.linalg.norm(native_delta))
    return {
        "native_effect_rms": float(np.sqrt(np.mean(np.square(native_delta)))),
        "llama_effect_rms": float(np.sqrt(np.mean(np.square(llama_delta)))),
        "cosine_similarity": float(np.dot(native_delta, llama_delta)
                                   / (norm * float(np.linalg.norm(llama_delta)))),
        "relative_difference": float(np.linalg.norm(native_delta - llama_delta) / norm),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    reports = {variant: _load_json(path) for variant, path in
               (("control", args.control_report), ("p24", args.p24_report))}
    ggufs = {}
    for variant, report in reports.items():
        path = Path(report["output"]["path"])
        if path.stat().st_size != report["output"]["bytes"]:
            raise RuntimeError(f"{variant} GGUF size differs from its build report")
        ggufs[variant] = path
    # The P24 report predates the builder's "variant" field.
    if reports["control"].get("variant") != "unpruned_parity_control":
        raise RuntimeError("control build report is not the unpruned parity control")
    if reports["p24"]["metadata_overrides"].get("qwen35moe.expert_count") != 232:
        raise RuntimeError("P24 build report is not a 232-expert build")
    mask_sha256 = reports["p24"]["inputs"]["mask"]["sha256"]
    mask_path = Path(reports["p24"]["inputs"]["mask"]["path"])
    if _sha256_file(mask_path) != mask_sha256:
        raise RuntimeError("mask differs from the one the P24 GGUF was built with")
    mask = np.load(mask_path)

    documents, calibration = _documents(args)
    model = _empty_model(args.model_dir)
    evaluator, assignment = _evaluator(args, model)
    paths = {(variant, document["id"]): _measure(
                 args, evaluator, assignment, model, mask, variant, ggufs[variant], document)
             for document in documents for variant in VARIANTS}

    per_variant: dict[str, dict[str, Any]] = {}
    per_document = []
    for variant in VARIANTS:
        rmse, top1, strict_pass, layer_sums = [], [], True, None
        for document in documents:
            data = np.load(paths[(variant, document["id"])])
            positions, passed = _comparison(data["native"], data["llama"], document["tokens"])
            layers, first_failed = _layer_comparison(data["native_layers"], data["llama_layers"])
            strict_pass = strict_pass and passed and first_failed is None
            rmse += [item["rmse"] for item in positions]
            top1 += [item["top1_agreement"] for item in positions]
            layer_relative = np.array([[item["relative_rmse"] for item in state["positions"]]
                                       for state in layers])
            layer_sums = layer_relative if layer_sums is None else np.concatenate(
                [layer_sums, layer_relative], axis=1)
            per_document.append({
                "variant": variant, "document": document["id"],
                "mean_logit_rmse": float(np.mean([item["rmse"] for item in positions])),
                "top1_agreement_rate": float(np.mean([item["top1_agreement"]
                                                      for item in positions])),
                "mean_abs_target_nll_difference": float(np.mean([
                    abs(item["native_target_nll"] - item["llama_target_nll"])
                    for item in positions])),
                "first_failed_strict_layer": first_failed,
            })
        per_variant[variant] = {
            "positions": len(rmse),
            "mean_logit_rmse": float(np.mean(rmse)),
            "top1_agreement_rate": float(np.mean(top1)),
            "layer_mean_relative_rmse": layer_sums.mean(axis=1).tolist(),
            "strict_original_thresholds_pass": strict_pass,
        }

    control, p24 = per_variant["control"], per_variant["p24"]
    layer_limits = [LAYER_RATIO * value + LAYER_SLACK
                    for value in control["layer_mean_relative_rmse"]]
    failing_layers = [index - 1 for index, (value, limit) in enumerate(
        zip(p24["layer_mean_relative_rmse"], layer_limits)) if value > limit]
    gates = {
        "logit_rmse_ratio": {
            "value": p24["mean_logit_rmse"] / control["mean_logit_rmse"],
            "limit": LOGIT_RMSE_RATIO},
        "top1_agreement_difference": {
            "value": p24["top1_agreement_rate"] - control["top1_agreement_rate"],
            "limit": -TOP1_AGREEMENT_SLACK},
        "layer_relative_rmse": {"failing_layers": failing_layers,
                                "ratio": LAYER_RATIO, "slack": LAYER_SLACK},
    }
    gates["logit_rmse_ratio"]["passed"] = gates["logit_rmse_ratio"]["value"] <= LOGIT_RMSE_RATIO
    gates["top1_agreement_difference"]["passed"] = (
        gates["top1_agreement_difference"]["value"] >= -TOP1_AGREEMENT_SLACK)
    gates["layer_relative_rmse"]["passed"] = not failing_layers
    passed = all(gate["passed"] for gate in gates.values())

    effects = []
    for document in documents:
        control_data = np.load(paths[("control", document["id"])])
        p24_data = np.load(paths[("p24", document["id"])])
        effects.append({
            "document": document["id"],
            "logits": _effect(p24_data["native"] - control_data["native"],
                              p24_data["llama"] - control_data["llama"]),
            "layers": [_effect(p24_data["native_layers"][state] - control_data["native_layers"][state],
                               p24_data["llama_layers"][state] - control_data["llama_layers"][state])
                       for state in range(1, p24_data["native_layers"].shape[0])],
        })

    def _artifact(path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": _sha256_file(path)}

    import torch
    import transformers
    return {
        "schema": SCHEMA,
        "status": "pass" if passed else "fail",
        "question": "does pruning add native-vs-llama.cpp divergence beyond the unpruned control?",
        "gates": gates,
        "variants": per_variant,
        "documents": [{key: document[key] for key in ("id", "index", "stratum")}
                      | {"tokens": len(document["tokens"])} for document in documents],
        "per_document": per_document,
        "pruning_effect_agreement": effects,
        "strict_thresholds_for_information": {
            "maximum_position_rmse": MAXIMUM_POSITION_RMSE,
            "maximum_position_mean_absolute_error": MAXIMUM_POSITION_MEAN_ABSOLUTE_ERROR,
            "minimum_position_cosine_similarity": MINIMUM_POSITION_COSINE_SIMILARITY,
            "maximum_layer_relative_rmse": MAXIMUM_LAYER_RELATIVE_RMSE,
            "minimum_layer_cosine_similarity": MINIMUM_LAYER_COSINE_SIMILARITY,
        },
        "artifacts": {
            "control_gguf_report": _artifact(args.control_report),
            "p24_gguf_report": _artifact(args.p24_report),
            "control_gguf_sha256": reports["control"]["output"]["sha256"],
            "p24_gguf_sha256": reports["p24"]["output"]["sha256"],
            "native_gsq_gguf": {"path": str(args.gguf),
                                "sha256": reports["p24"]["inputs"]["gsq_gguf"]["sha256"]},
            "mask": {"path": str(mask_path), "sha256": mask_sha256},
            "helper": _artifact(args.helper),
            "layer_helper": _artifact(args.layer_helper),
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
    parser.add_argument("--p24-report", type=Path,
                        default=RCO / "reports/qwen36_gsq_e6_p24_gguf.json")
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
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_p24_parity")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--output", type=Path,
                        default=RCO / "reports/qwen36_gsq_e6_p24_logit_parity.json")
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--native-compute-dtype", choices=("bf16", "fp32"), default="fp32")
    parser.add_argument("--rows-per-chunk", type=int, default=1024)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ubatch", type=int, default=64)
    args = parser.parse_args()
    for key in ("control_report", "p24_report", "model_dir", "manifest", "calibration",
                "gguf", "gguf_python", "ggml_library", "helper", "layer_helper", "work",
                "reports", "output"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main() -> int:
    args = _parse_args()
    report = audit(args)
    _atomic_json(args.output, report)
    print(json.dumps({"status": report["status"], "gates": report["gates"]}, indent=2),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
