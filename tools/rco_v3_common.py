"""Shared definitions of the v3 search (RCO_PLAN_V3.md).

- the 370 searched tensors and their choices;
- the candidate stores, one label per store, and which store holds a
  (tensor, type) choice;
- the center assignment (``gate``/``up`` IQ2_XS, ``down`` IQ3_XXS, dense
  Q6_K) that the center GGUF holds;
- bytes on the GPU (every tensor except ``token_embd``; metadata not counted)
  against the budget of 12,000,000,000 bytes;
- the exact evaluator over a v3 GGUF, with every store installable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_q3k_viability import _load_json, _sha256_file  # noqa: E402
from build_qwen36_post_release_q4k_dense_store import groups as dense_groups  # noqa: E402

BUDGET_GPU_BYTES = 12_000_000_000
LAYERS = 40
EXPERT_CHOICES = ("IQ2_XS", "IQ3_XXS")
DENSE_CHOICES = ("Q8_0", "Q6_K", "Q4_K")

STORE_REPORTS = {
    "IQ2_XS": RCO / "reports/qwen36_post_release_iq_store_full_IQ2_XS.json",
    "IQ3_XXS_DOWN": RCO / "reports/qwen36_post_release_iq_store_full_IQ3_XXS.json",
    "IQ3_XXS_GATE_UP": RCO / "reports/rco_v3/qwen36_post_release_iq_store_full_IQ3_XXS.json",
    "Q8_0": RCO / "reports/rco_v3/qwen36_rco_v3_dense_store_Q8_0.json",
    "Q6_K": RCO / "reports/rco_v3/qwen36_rco_v3_dense_store_Q6_K.json",
    "Q4_K": RCO / "reports/qwen36_post_release_q4k_dense_store.json",
}
STORE_TYPES = {"IQ2_XS": "IQ2_XS", "IQ3_XXS_DOWN": "IQ3_XXS", "IQ3_XXS_GATE_UP": "IQ3_XXS",
               "Q8_0": "Q8_0", "Q6_K": "Q6_K", "Q4_K": "Q4_K"}


def searched() -> dict[str, list[str]]:
    dense = dense_groups()
    return {
        "down": [f"blk.{l}.ffn_down_exps.weight" for l in range(LAYERS)],
        "gate_up": [f"blk.{l}.ffn_{f}_exps.weight" for l in range(LAYERS) for f in ("gate", "up")],
        "attention": dense["attention"],
        "shared_expert": dense["shared_expert"],
    }


def choices(name: str) -> tuple[str, ...]:
    return EXPERT_CHOICES if "_exps." in name else DENSE_CHOICES


def store_label(name: str, type_name: str) -> str:
    if type_name == "IQ3_XXS":
        return "IQ3_XXS_DOWN" if "ffn_down_exps" in name else "IQ3_XXS_GATE_UP"
    return type_name


def center_assignment() -> dict[str, str]:
    groups = searched()
    center = {n: "IQ3_XXS" for n in groups["down"]}
    center.update({n: "IQ2_XS" for n in groups["gate_up"]})
    center.update({n: "Q6_K" for n in groups["attention"] + groups["shared_expert"]})
    return center


def load_store_reports(labels=None) -> dict[str, dict[str, Any]]:
    """Each store report, its index checked against the report."""
    reports = {}
    for label in labels or STORE_REPORTS:
        report = _load_json(STORE_REPORTS[label])
        if report.get("status") != "complete":
            raise RuntimeError(f"store {label} is not complete")
        root = Path(report["store"]["root"])
        if _sha256_file(root / "native-candidate-index.json") != report["store"]["index_sha256"]:
            raise RuntimeError(f"store {label}: index differs from its report")
        reports[label] = report
    return reports


def store_entry(reports, name: str, type_name: str) -> dict[str, Any]:
    label = store_label(name, type_name)
    entry = reports[label]["tensors"][name]["store_entry"]
    if entry["ggml_type_name"] != type_name:
        raise RuntimeError(f"{label} holds {entry['ggml_type_name']} for {name}, not {type_name}")
    return entry


def gpu_bytes(assignment: dict[str, str], reports, gsq_bytes: dict[str, int]) -> int:
    """Bytes of every tensor except ``token_embd`` for an assignment.

    ``gsq_bytes``: the GSQ GGUF's tensor bytes by name; unsearched tensors
    keep them, except ``output.weight`` (Q4_K store)."""
    total = 0
    for name, size in gsq_bytes.items():
        if name == "token_embd.weight":
            continue
        if name == "output.weight":
            total += int(store_entry(reports, name, "Q4_K")["payload_bytes"])
        elif name in assignment:
            total += int(store_entry(reports, name, assignment[name])["payload_bytes"])
        else:
            total += size
    return total


def upgrade_keys(assignment: dict[str, str], center: dict[str, str]) -> frozenset:
    """Typed install keys for the tensors where an assignment leaves the center."""
    from search.typed_upgrades import typed_key

    return frozenset(typed_key(store_label(n, t), n)
                     for n, t in assignment.items() if t != center[n])


def evaluator(args, gguf_path: Path, reports, evaluator_class=None):
    """The exact evaluator over a v3 GGUF; every store installable by label.

    ``evaluator_class``: a ``StreamedChatExactEvaluator`` subclass to build
    instead (the screening's ``SharedPrefixExactEvaluator``)."""
    import torch

    from audit_qwen36_chat_kl import PAD_TOKEN, _model
    from audit_qwen36_gsq_e6 import NAME, EmbeddingQ6KWeightStore, _entry
    from gguf_mixed_stream import MixedGGUFManifestPrefixLoader
    from native_store import NativeCandidateStore
    from quant.ggml_native import GGMLNativeCodec
    from quant.ggml_native_mixed import MIXED_TYPE_IDS, MixedGGMLNativeCodec
    from search.streamed_chat_exact import StreamedChatExactEvaluator
    from search.typed_upgrades import SingleTypeExpertStore, TypedUpgradeStore

    embedding = _load_json(args.reports / "qwen36_gsq_e6_embedding.json")
    codec = MixedGGMLNativeCodec(args.ggml_library)
    manifest = _load_json(args.manifest)
    embedding_store = EmbeddingQ6KWeightStore(
        GGMLNativeCodec(args.ggml_library), Path(embedding["payload"]["path"]),
        embedding["payload"]["sha256"], _entry(args.manifest), rows_per_chunk=4096)
    upgrade_store = TypedUpgradeStore({
        label: SingleTypeExpertStore(
            NativeCandidateStore(Path(report["store"]["root"]), codec), manifest,
            args.model_dir, MIXED_TYPE_IDS[STORE_TYPES[label]], rows_per_chunk=args.install_rows)
        for label, report in reports.items()})
    loader = MixedGGUFManifestPrefixLoader(
        gguf_path, manifest, args.model_dir, gguf_python=args.gguf_python,
        ggml_library=args.ggml_library, rows_per_chunk=args.rows_per_chunk,
        workers=args.decode_workers)
    model = _model(SimpleNamespace(model_dir=args.model_dir, attn_implementation="sdpa"))
    result = (evaluator_class or StreamedChatExactEvaluator)(
        model, loader, upgrade_store=upgrade_store, device=torch.device(args.device),
        checkpoint_dtype=torch.bfloat16, embedding_store=embedding_store,
        embedding_selected=[(NAME, 1)], position_chunk=args.position_chunk,
        microbatch_tokens=args.microbatch_tokens, pad_token=PAD_TOKEN,
        device_hidden_bytes=args.device_hidden_bytes)
    identity = {
        "gguf_sha256": _sha256_file(gguf_path),
        "embedding_sha256": embedding["payload"]["sha256"],
        "store_index_sha256": {k: r["store"]["index_sha256"] for k, r in reports.items()},
        "microbatch_tokens": args.microbatch_tokens,
    }
    return result, identity


def add_evaluator_arguments(parser) -> None:
    parser.add_argument("--corpus", type=Path,
                        default=RCO / "reports/qwen36_chat_corpus_v2_manifest.json")
    parser.add_argument("--split-report", type=Path,
                        default=RCO / "reports/qwen36_chat_corpus_v2_calibration_split.json")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/qwen36_35b_base")
    parser.add_argument("--manifest", type=Path,
                        default=RCO / "reports/qwen36_35b_base_gguf_manifest.json")
    parser.add_argument("--gguf-python", type=Path, default=ROOT / "repos/llama.cpp/gguf-py")
    parser.add_argument("--ggml-library", type=Path,
                        default=ROOT / "experiment/build-cpu/bin/libggml-base.so.0.24.0")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows-per-chunk", type=int, default=1024)
    parser.add_argument("--install-rows", type=int, default=4096)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--position-chunk", type=int, default=256)
    parser.add_argument("--microbatch-tokens", type=int, default=8448)
    parser.add_argument("--device-hidden-bytes", type=int, default=3 << 30)


EVALUATOR_PATHS = ("corpus", "split_report", "model_dir", "manifest", "gguf_python",
                   "ggml_library", "reports")
