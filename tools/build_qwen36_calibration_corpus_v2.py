#!/usr/bin/env python3
"""Build calibration corpus v2 for Phases 2-7 of RCO_PLAN_NEW.md.

Same licensed local sources and strata as v1
(``build_qwen36_calibration_corpus.py``), but larger: 50 documents per
stratum of 512 tokens each by default.  Segments are spread across each
source by a van der Corput order over aligned start positions, and every
document shares no ``--disjoint-ngram-width``-token n-gram with the held-out
release corpus, with calibration v1, or with any other v2 document.
The output manifest keeps the v1 schema so existing loaders accept it.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterator, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_qwen36_calibration_corpus import (  # noqa: E402
    REASONING_SOURCE_IDS,
    _atomic_bytes,
    _git_revision,
    _load_raw_source,
    _ngram_hashes,
    _sha256_file,
    _validate_canonical_manifest_sha256,
    _wikipedia_sources,
)
from release_corpus import (  # noqa: E402
    canonical_json_bytes,
    normalize_text,
    sha256_bytes,
)


STRATA = ("language", "knowledge", "reasoning", "code", "multilingual")


def van_der_corput_order(count: int) -> list[int]:
    """Return 0..count-1 ordered so every prefix is spread across the range."""
    if count < 0:
        raise ValueError("count must be nonnegative")
    keys = []
    for index in range(count):
        value, denominator, remaining = 0.0, 1.0, index
        while remaining:
            denominator *= 2.0
            remaining, bit = divmod(remaining, 2)
            value += bit / denominator
        keys.append((value, index))
    return [index for _, index in sorted(keys)]


def spread_starts(token_count: int, *, initial: int, length: int, stride: int) -> list[int]:
    if min(initial, length, stride) < 1:
        raise ValueError("segment parameters must be positive")
    starts = list(range(initial, token_count - length + 1, stride))
    return [starts[index] for index in van_der_corput_order(len(starts))]


class SourceCursor:
    """Yield non-overlapping disjoint segments of one source in spread order."""

    def __init__(self, tokens: Sequence[int], *, initial: int, length: int,
                 stride: int, width: int,
                 accept: Callable[[list[int]], bool] = lambda segment: True) -> None:
        self.tokens = tokens
        self.accept = accept
        self.rejected = 0
        self.length = length
        self.width = width
        self.starts = iter(spread_starts(
            len(tokens), initial=initial, length=length, stride=stride))
        self.taken: list[tuple[int, int]] = []

    def next_segment(self, forbidden: set[bytes]) -> tuple[int, list[int], set[bytes]] | None:
        for start in self.starts:
            stop = start + self.length
            if any(start < other_stop and other_start < stop
                   for other_start, other_stop in self.taken):
                continue
            segment = [int(value) for value in self.tokens[start:stop]]
            ngrams = _ngram_hashes(segment, self.width)
            if ngrams & forbidden:
                continue
            if not self.accept(segment):
                self.rejected += 1
                continue
            self.taken.append((start, stop))
            return start, segment, ngrams
        return None


def round_robin(cursors: Sequence[Any], quota: int, forbidden: set[bytes]) -> Iterator[tuple[int, int, list[int]]]:
    """Take one segment per source in turn until ``quota`` or exhaustion."""
    active = list(range(len(cursors)))
    produced = 0
    while produced < quota and active:
        for source_index in list(active):
            if produced == quota:
                break
            found = cursors[source_index].next_segment(forbidden)
            if found is None:
                active.remove(source_index)
                continue
            start, segment, ngrams = found
            forbidden |= ngrams
            produced += 1
            yield source_index, start, segment
    if produced < quota:
        raise RuntimeError(f"sources provide only {produced} of {quota} segments")


def build(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    started = time.perf_counter()
    tokenizer_dir = args.tokenizer_dir.resolve(strict=True)
    heldout_data_dir = args.heldout_data_dir.resolve(strict=True)
    heldout_report_path = args.heldout_report.resolve(strict=True)
    v1_report_path = args.v1_report.resolve(strict=True)
    llama_cpp = args.llama_cpp.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    revision = _git_revision(llama_cpp)
    if revision != args.llama_cpp_revision:
        raise RuntimeError(f"llama.cpp revision differs: {revision}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True, trust_remote_code=False)

    heldout_report = json.loads(heldout_report_path.read_text(encoding="utf-8"))
    _validate_canonical_manifest_sha256(heldout_report)
    heldout_manifest = heldout_report["canonical_manifest"]
    heldout_corpus_path = heldout_data_dir / "corpus.jsonl"
    if _sha256_file(heldout_corpus_path) != heldout_manifest["corpus"]["sha256"]:
        raise RuntimeError("held-out corpus hash differs from its manifest")
    heldout_rows = [json.loads(line) for line in heldout_corpus_path.read_text(
        encoding="utf-8").splitlines() if line]
    forbidden: set[bytes] = set()
    for row in heldout_rows:
        forbidden |= _ngram_hashes(
            tokenizer.encode(row["text"], add_special_tokens=False), args.disjoint_ngram_width)
    heldout_ngram_count = len(forbidden)

    v1_report = json.loads(v1_report_path.read_text(encoding="utf-8"))
    _validate_canonical_manifest_sha256(v1_report)
    v1_tokens_meta = v1_report["canonical_manifest"]["tokens"]
    v1_tokens_path = Path(v1_tokens_meta["path"]).resolve(strict=True)
    if _sha256_file(v1_tokens_path) != v1_tokens_meta["sha256"]:
        raise RuntimeError("calibration v1 token file hash differs")
    v1_ngrams: set[bytes] = set()
    for tokens in json.loads(v1_tokens_path.read_text(encoding="utf-8")):
        v1_ngrams |= _ngram_hashes(tokens, args.disjoint_ngram_width)
    forbidden |= v1_ngrams

    pools = {
        "language": [row for row in heldout_rows if row["stratum"] == "general"
                     and row["id"] not in REASONING_SOURCE_IDS],
        "knowledge": [row for row in heldout_rows if row["stratum"] == "knowledge"],
        "reasoning": [row for row in heldout_rows if row["id"] in REASONING_SOURCE_IDS],
        "code": [row for row in heldout_rows if row["stratum"] == "code"],
        "multilingual": sorted(
            (row for row in heldout_rows if row["stratum"] == "multilingual"),
            key=lambda row: (row["language"], row["id"])),
    }
    if len(pools["reasoning"]) != len(REASONING_SOURCE_IDS):
        raise RuntimeError("reasoning sources are missing from the held-out corpus")
    wikipedia = _wikipedia_sources(heldout_data_dir / "sources")

    def round_trips(segment: list[int]) -> bool:
        text = tokenizer.decode(
            segment, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return tokenizer.encode(text, add_special_tokens=False) == segment

    rejected_round_trip = Counter()
    v2_only: set[bytes] = set()
    calibration_rows: list[dict[str, Any]] = []
    token_rows: list[list[int]] = []
    manifest_documents: list[dict[str, Any]] = []
    for stratum in STRATA:
        pool = pools[stratum]
        sources = []
        cursors = []
        for row in pool:
            raw_text, local_path = _load_raw_source(
                row, heldout_data_dir=heldout_data_dir, llama_cpp=llama_cpp,
                wikipedia=wikipedia)
            tokens = tokenizer.encode(normalize_text(raw_text), add_special_tokens=False)
            sources.append((row, local_path))
            cursors.append(SourceCursor(
                tokens, initial=args.initial_offset, length=args.tokens_per_document,
                stride=args.offset_stride, width=args.disjoint_ngram_width,
                accept=round_trips))
        before = set(forbidden)
        for index, (source_index, start, tokens) in enumerate(round_robin(
            cursors, args.documents_per_stratum, forbidden,
        )):
            row, local_path = sources[source_index]
            text = tokenizer.decode(
                tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if tokenizer.encode(text, add_special_tokens=False) != tokens:
                raise RuntimeError(f"segment does not round-trip: {row['id']}@{start}")
            document_id = f"calibration-v2-{stratum}-{index:02d}"
            source = dict(row["source"])
            source.update({
                "heldout_source_id": row["id"],
                "source_token_offset": start,
                "local_source": str(local_path),
                "local_source_sha256": (
                    _sha256_file(local_path) if local_path.is_file() else None),
            })
            calibration_rows.append({
                "id": document_id, "stratum": stratum, "language": row["language"],
                "text": text, "source": source})
            token_rows.append(tokens)
            content = text.encode("utf-8")
            manifest_documents.append({
                "id": document_id,
                "stratum": stratum,
                "language": row["language"],
                "source": source,
                "content_bytes": len(content),
                "content_sha256": sha256_bytes(content),
                "token_count": len(tokens),
                "predicted_token_count": len(tokens) - 1,
                "token_ids_sha256": sha256_bytes(canonical_json_bytes(tokens)),
            })
        v2_only |= forbidden - before
        rejected_round_trip[stratum] = sum(cursor.rejected for cursor in cursors)

    # Independent re-check of every disjointness claim on the final rows.
    heldout_check: set[bytes] = set()
    for row in heldout_rows:
        heldout_check |= _ngram_hashes(
            tokenizer.encode(row["text"], add_special_tokens=False), args.disjoint_ngram_width)
    seen: set[bytes] = set()
    for tokens in token_rows:
        ngrams = _ngram_hashes(tokens, args.disjoint_ngram_width)
        if ngrams & (heldout_check | v1_ngrams | seen):
            raise RuntimeError("calibration v2 disjointness invariant failed")
        seen |= ngrams

    counts = dict(sorted(Counter(row["stratum"] for row in calibration_rows).items()))
    corpus_bytes = b"".join(canonical_json_bytes(row) for row in calibration_rows)
    tokens_bytes = canonical_json_bytes(token_rows)
    corpus_path = output_dir / "corpus.jsonl"
    tokens_path = output_dir / "tokens.json"
    _atomic_bytes(corpus_path, corpus_bytes)
    _atomic_bytes(tokens_path, tokens_bytes)
    licenses = []
    for name in ("project-gutenberg-permission.html", "CC-BY-SA-4.0.txt",
                 "llama.cpp-MIT.txt"):
        destination = output_dir / "licenses" / name
        _atomic_bytes(destination, (heldout_data_dir / "licenses" / name).read_bytes())
        licenses.append({"path": str(destination), "sha256": _sha256_file(destination)})
    sequence_digest = hashlib.sha256()
    for tokens in token_rows:
        sequence_digest.update(len(tokens).to_bytes(8, "little"))
        sequence_digest.update(canonical_json_bytes(tokens))
    manifest = {
        "schema": 1,
        "version": 2,
        "corpus": {"path": str(corpus_path), "bytes": len(corpus_bytes),
                   "sha256": sha256_bytes(corpus_bytes)},
        "tokens": {"path": str(tokens_path), "bytes": len(tokens_bytes),
                   "sha256": sha256_bytes(tokens_bytes)},
        "document_count": len(calibration_rows),
        "documents": manifest_documents,
        "predicted_token_count": sum(len(tokens) - 1 for tokens in token_rows),
        "token_count_per_document": args.tokens_per_document,
        "token_sequence_sha256": sequence_digest.hexdigest(),
        "stratum_document_counts": counts,
        "source_counts_per_stratum": {
            stratum: len({row["source"]["heldout_source_id"]
                          for row in calibration_rows if row["stratum"] == stratum})
            for stratum in STRATA},
        "tokenizer": {"path": str(tokenizer_dir),
                      "tokenizer_json_sha256": _sha256_file(tokenizer_dir / "tokenizer.json")},
        "heldout_disjointness": {
            "heldout_report": str(heldout_report_path),
            "heldout_report_sha256": _sha256_file(heldout_report_path),
            "heldout_corpus_sha256": heldout_manifest["corpus"]["sha256"],
            "ngram_width": args.disjoint_ngram_width,
            "heldout_unique_ngram_count": heldout_ngram_count,
            "calibration_unique_ngram_count": len(seen),
            "shared_ngram_count": 0,
        },
        "v1_disjointness": {
            "v1_report": str(v1_report_path),
            "v1_report_sha256": _sha256_file(v1_report_path),
            "v1_token_sha256": v1_tokens_meta["sha256"],
            "v1_unique_ngram_count": len(v1_ngrams),
            "shared_ngram_count": 0,
        },
        "within_v2_shared_ngram_count": 0,
        "round_trip_rejected_candidates": dict(rejected_round_trip),
        "selection": {
            "initial_offset": args.initial_offset,
            "offset_stride": args.offset_stride,
            "order": "round robin over sources; van der Corput order over starts",
        },
        "llama_cpp": {"path": str(llama_cpp), "revision": revision},
        "licenses": licenses,
    }
    report = {
        "schema": "rco.qwen36.calibration_corpus.v2",
        "status": "pass",
        "scope": (
            "licensed stratified calibration corpus v2 for pruning, imatrix, "
            "and quant search; token-level disjoint from the release corpus, "
            "from calibration v1, and internally"),
        "canonical_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "canonical_manifest": manifest,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_bytes(output_path, json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--heldout-data-dir", type=Path, required=True)
    parser.add_argument("--heldout-report", type=Path, required=True)
    parser.add_argument("--v1-report", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--llama-cpp-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents-per-stratum", type=int, default=50)
    parser.add_argument("--tokens-per-document", type=int, default=512)
    parser.add_argument("--disjoint-ngram-width", type=int, default=32)
    parser.add_argument("--initial-offset", type=int, default=1536)
    parser.add_argument("--offset-stride", type=int, default=128)
    args = parser.parse_args()
    if min(args.documents_per_stratum, args.tokens_per_document,
           args.disjoint_ngram_width, args.initial_offset, args.offset_stride) < 1:
        parser.error("numeric corpus parameters must be positive")
    if args.disjoint_ngram_width > args.tokens_per_document:
        parser.error("disjoint n-gram width exceeds document length")
    return args


if __name__ == "__main__":
    result = build(_parse_args())
    manifest = result["canonical_manifest"]
    print(json.dumps({
        "status": result["status"],
        "document_count": manifest["document_count"],
        "predicted_token_count": manifest["predicted_token_count"],
        "stratum_document_counts": manifest["stratum_document_counts"],
        "source_counts_per_stratum": manifest["source_counts_per_stratum"],
        "token_sequence_sha256": manifest["token_sequence_sha256"],
    }, indent=2, sort_keys=True))
