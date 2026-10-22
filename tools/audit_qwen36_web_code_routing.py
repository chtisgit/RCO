#!/usr/bin/env python3
"""Coding regression, part 1 of PLAN_POST_RCO_RELEASE.md: pruned experts on web code.

The release prunes each layer's 24 experts least selected by GSQ-E6 on the
chat v2 calibration half (``qwen36_gsq_e6_prune24_chat_frequency_mask.json``);
there those experts carried 2.14% of the routed slots (uniform routing:
24/256 = 9.4%).  This counts, with the same model (GSQ-E6, unpruned) and the
same router hook as the Phase 3 router statistics, how often they are
selected on web code and on contrast text, in 512-token windows:

- ``gen_css``, ``gen_html``, ``gen_js``: the code blocks GSQ wrote for the
  user's prompt in the coding reproduction (``qwen36_coding_repro_gsq.json``),
  concatenated per language;
- ``css``, ``svelte``, ``js``, ``ts``: files of the pinned llama.cpp checkout
  (web UI and scripts; nothing downloaded);
- ``cpp``: the held-out raw-text corpus's 25 C++ documents (where the release
  is not worse than GSQ);
- ``prose``: its 25 knowledge (Wikipedia) documents.

Per stratum: the share of routed slots that go to pruned experts, overall,
per layer and per window (bootstrap CI over windows), and the ratio to the
chat share.  Resumable per stratum.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_q3k_viability import _atomic_json, _empty_model, _load_json, _sha256_file  # noqa: E402
from audit_qwen36_prune24_prelim import EXPERTS, LAYERS, _router_modules, _save_npz  # noqa: E402

SCHEMA = "rco.qwen36.web_code_routing.v1"
OUTPUT = "qwen36_post_release_web_code_routing.json"
WINDOW = 512
CAP = 25
SEED = 20261001
UNIFORM_SHARE = 24 / 256


def _windows(tokens: list[int], limit: int) -> list[list[int]]:
    return [tokens[i:i + WINDOW] for i in range(0, len(tokens) - WINDOW + 1, WINDOW)][:limit]


def _files(root: Path, pattern: str, exclude: tuple[str, ...] = ()) -> list[Path]:
    return sorted(p for p in root.glob(pattern)
                  if p.is_file() and not any(part in p.parts for part in exclude))


def build_corpus(args, tokenizer) -> dict[str, list[dict]]:
    """Stratum -> windows, each {id, source, sha256, tokens}."""
    def encode(text: str) -> list[int]:
        return [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]

    strata: dict[str, list[dict]] = {}
    repro = _load_json(args.reports / "qwen36_coding_repro_gsq.json")
    from audit_qwen36_coding_repro import code_blocks
    for language in ("css", "html", "js"):
        text = "\n\n".join(block["body"] for g in repro["generations"]
                           for block in code_blocks(g["content"])
                           if block["language"] == language)
        strata[f"gen_{language}"] = [
            {"id": f"gen_{language}-{i:02d}", "source": "qwen36_coding_repro_gsq.json",
             "tokens": window}
            for i, window in enumerate(_windows(encode(text), CAP))]

    ui = args.llama_dir / "tools/ui"
    exclude = ("node_modules", "tests", "dist", "build")
    file_sets = {
        "css": (_files(ui / "src", "**/*.css", exclude), CAP),
        "svelte": (_files(ui / "src", "**/*.svelte", exclude), 1),
        "ts": (_files(ui / "src", "**/*.ts", exclude), 1),
        "js": (_files(args.llama_dir, "**/*.js", exclude + ("experiment",)), 4),
    }
    for stratum, (paths, per_file) in file_sets.items():
        windows = []
        for path in paths:
            data = path.read_bytes()
            for i, window in enumerate(_windows(encode(data.decode("utf-8")), per_file)):
                windows.append({"id": f"{stratum}-{path.relative_to(args.llama_dir)}#{i}",
                                "source": str(path.relative_to(args.llama_dir)),
                                "sha256": _sha256_file(path), "tokens": window})
        strata[stratum] = windows[:CAP]

    heldout = _load_json(args.heldout)["canonical_manifest"]
    corpus = Path(heldout["corpus"]["path"])
    if _sha256_file(corpus) != heldout["corpus"]["sha256"]:
        raise RuntimeError("held-out corpus differs from its manifest")
    documents = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line]
    stratum_of = {d["id"]: d["stratum"] for d in heldout["documents"]}
    for stratum, source in (("cpp", "code"), ("prose", "knowledge")):
        strata[stratum] = [
            {"id": f"{stratum}-{d['id']}", "source": d["id"], "tokens": encode(d["text"])[:WINDOW]}
            for d in documents if stratum_of[d["id"]] == source][:CAP]
        if any(len(w["tokens"]) != WINDOW for w in strata[stratum]):
            raise RuntimeError(f"a {stratum} document is shorter than {WINDOW} tokens")
    return strata


class PrunedHits:
    """Per window and layer: routed slots on pruned experts; per layer and expert: counts."""

    def __init__(self, model, mask: np.ndarray) -> None:
        import torch

        self.torch = torch
        self.mask = mask
        self.batch = 0
        self.hits = None
        self.counts = None
        self.handles = [module.register_forward_hook(self._hook(layer))
                        for layer, module in _router_modules(model).items()]

    def reset(self, batch: int) -> None:
        self.batch = batch
        self.hits = np.zeros((batch, LAYERS), dtype=np.int64)
        self.counts = np.zeros((LAYERS, EXPERTS), dtype=np.int64)

    def _hook(self, layer: int):
        torch = self.torch

        def hook(module, inputs, output):
            _, _, indices = output
            if indices.shape[0] != self.batch * WINDOW:
                raise RuntimeError(f"routed {indices.shape[0]} tokens, expected "
                                   f"{self.batch} x {WINDOW}")
            mask = torch.from_numpy(self.mask[layer]).to(indices.device)
            pruned = mask[indices].reshape(self.batch, WINDOW, -1)
            self.hits[:, layer] += pruned.sum(dim=(1, 2)).cpu().numpy()
            self.counts[layer] += torch.bincount(indices.reshape(-1),
                                                 minlength=EXPERTS).cpu().numpy()
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _bootstrap(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(values), size=(10_000, len(values)))
    means = values[draws].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def run(args) -> int:
    import torch
    from transformers import AutoTokenizer

    from audit_qwen36_prune24_prelim import _gsq_e6_evaluator

    output = args.reports / OUTPUT
    if output.exists() and _load_json(output).get("status") == "complete":
        print("web-code routing: already complete", flush=True)
        return 0
    started = time.perf_counter()
    mask_report = _load_json(args.reports / "qwen36_gsq_e6_prune24_chat_frequency_mask.json")
    mask_path = Path(mask_report["mask"]["path"])
    if _sha256_file(mask_path) != mask_report["mask"]["sha256"]:
        raise RuntimeError("frequency mask differs from its report")
    mask = np.load(mask_path)
    if mask.shape != (LAYERS, EXPERTS) or set(mask.sum(axis=1)) != {24}:
        raise RuntimeError("unexpected frequency mask shape")
    chat_share = float(mask_report["mask"]["unpruned_routed_slot_share_of_pruned_experts"])

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    strata = build_corpus(args, tokenizer)
    for stratum, windows in strata.items():
        print(f"{stratum}: {len(windows)} windows", flush=True)

    model = _empty_model(args.model_dir)
    evaluator, assignment, identity = _gsq_e6_evaluator(args, model)
    hits = PrunedHits(model, mask)
    results = {}
    try:
        for stratum, windows in strata.items():
            if not windows:
                continue
            path = args.work / f"{stratum}.npz"
            if path.exists():
                with np.load(path) as data:
                    results[stratum] = {key: data[key] for key in data.files}
                continue
            input_ids = torch.tensor([w["tokens"] for w in windows], dtype=torch.long)
            hits.reset(len(windows))
            batch_started = time.perf_counter()
            evaluation = evaluator.evaluate(input_ids, assignment)
            arrays = {"hits": hits.hits.copy(), "counts": hits.counts.copy(),
                      "document_mean_nll": np.asarray(evaluation.document_mean_nll)}
            _save_npz(path, **arrays)
            results[stratum] = arrays
            share = arrays["hits"].sum() / (len(windows) * WINDOW * 8 * LAYERS)
            print(f"{stratum}: pruned share {share:.4f} ({share / chat_share:.1f}x chat), "
                  f"GSQ-E6 NLL {arrays['document_mean_nll'].mean():.4f}, "
                  f"{time.perf_counter() - batch_started:.0f}s", flush=True)
    finally:
        hits.close()

    slots = WINDOW * 8
    summary = {}
    for stratum, arrays in results.items():
        window_share = arrays["hits"].sum(axis=1) / (slots * LAYERS)
        layer_share = arrays["hits"].sum(axis=0) / (slots * len(window_share))
        counts = arrays["counts"]
        order = np.argsort(layer_share)[::-1]
        summary[stratum] = {
            "windows": int(len(window_share)), "tokens": int(len(window_share) * WINDOW),
            "pruned_share": float(window_share.mean()),
            "pruned_share_ci95": _bootstrap(window_share),
            "ratio_to_chat": float(window_share.mean() / chat_share),
            "window_share_min_max": [float(window_share.min()), float(window_share.max())],
            "layer_share": layer_share.tolist(),
            "top_layers": [{"layer": int(i), "share": float(layer_share[i])} for i in order[:5]],
            "layers_above_uniform": int((layer_share > UNIFORM_SHARE).sum()),
            "pruned_experts_never_selected": int((counts[mask] == 0).sum()),
            "gsq_e6_mean_nll": float(arrays["document_mean_nll"].mean()),
        }
    _atomic_json(output, {
        "schema": SCHEMA, "status": "complete",
        "question": "how often web code selects the experts the release prunes",
        "model": "GSQ-E6 (authentic GSQ + Q6_K token_embd), unpruned, as for the mask's counts",
        "metric": "share of routed slots (8 per token per layer) on pruned experts",
        "mask": {"sha256": mask_report["mask"]["sha256"], "chat_share": chat_share,
                 "uniform_share": UNIFORM_SHARE},
        "identity": {**identity, "llama_dir": str(args.llama_dir),
                     "heldout_manifest_sha256": _sha256_file(args.heldout)},
        "corpus": {stratum: [{k: v for k, v in w.items() if k != "tokens"} for w in windows]
                   for stratum, windows in strata.items()},
        "window_tokens": WINDOW,
        "strata": summary,
        "wall_seconds": time.perf_counter() - started})
    print("stratum windows share CI95 ratio_to_chat top_layer", flush=True)
    for stratum, s in summary.items():
        top = s["top_layers"][0]
        print(f"{stratum} {s['windows']} {s['pruned_share']:.4f} "
              f"[{s['pruned_share_ci95'][0]:.4f}, {s['pruned_share_ci95'][1]:.4f}] "
              f"{s['ratio_to_chat']:.2f}x layer {top['layer']} {top['share']:.3f}", flush=True)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/qwen36_35b_base")
    parser.add_argument("--manifest", type=Path,
                        default=RCO / "reports/qwen36_35b_base_gguf_manifest.json")
    parser.add_argument("--gguf", type=Path,
                        default=ROOT / "results/Qwen3.6-35B-A3B-GSQ-hybrid.gguf")
    parser.add_argument("--gguf-python", type=Path, default=ROOT / "repos/llama.cpp/gguf-py")
    parser.add_argument("--ggml-library", type=Path,
                        default=ROOT / "experiment/build-cpu/bin/libggml-base.so.0.24.0")
    parser.add_argument("--llama-dir", type=Path, default=ROOT / "repos/llama.cpp")
    parser.add_argument("--heldout", type=Path,
                        default=RCO / "reports/qwen36_35b_heldout_corpus_manifest.json")
    parser.add_argument("--work", type=Path,
                        default=ROOT / "data/qwen36_post_release_web_code_routing")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows-per-chunk", type=int, default=1024)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--corpus-only", action="store_true",
                        help="build and print the corpus, no forward")
    args = parser.parse_args()
    for key in ("model_dir", "manifest", "gguf", "gguf_python", "ggml_library", "llama_dir",
                "heldout", "work", "reports"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main() -> int:
    args = _parse_args()
    if args.corpus_only:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        for stratum, windows in build_corpus(args, tokenizer).items():
            print(stratum, len(windows), sorted({w["source"] for w in windows})[:6])
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
