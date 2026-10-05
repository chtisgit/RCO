#!/usr/bin/env python3
"""Build a licensed, stratified corpus disjoint from Qwen3.6 held-out data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from acquire_qwen36_release_corpus import _strip_gutenberg  # noqa: E402
from release_corpus import (  # noqa: E402
    canonical_json_bytes,
    normalize_text,
    sha256_bytes,
)


REASONING_SOURCE_IDS = (
    "general-gutenberg-1497",  # The Republic
    "general-gutenberg-1232",  # The Prince
    "general-gutenberg-1080",  # A Modest Proposal
    "general-gutenberg-6130",  # The Iliad
    "general-gutenberg-2600",  # War and Peace
)
GENERAL_SOURCE_IDS = (
    "general-gutenberg-1342", "general-gutenberg-84",
    "general-gutenberg-11", "general-gutenberg-1661",
    "general-gutenberg-98", "general-gutenberg-2701",
    "general-gutenberg-74", "general-gutenberg-76",
    "general-gutenberg-345", "general-gutenberg-5200",
)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_canonical_manifest_sha256(report: dict[str, Any]) -> str:
    manifest = report["canonical_manifest"]
    actual = sha256_bytes(canonical_json_bytes(manifest))
    recorded = report.get("canonical_manifest_sha256")
    if recorded != actual:
        raise RuntimeError(
            "held-out canonical manifest hash differs: "
            f"{recorded!r} != {actual}")
    return actual


def _ngram_hashes(tokens: Sequence[int], width: int) -> set[bytes]:
    if width < 1:
        raise ValueError("n-gram width must be positive")
    if len(tokens) < width:
        return set()
    return {
        hashlib.blake2b(
            b"".join(int(token).to_bytes(4, "little")
                     for token in tokens[start:start + width]),
            digest_size=16,
        ).digest()
        for start in range(len(tokens) - width + 1)
    }


def _choose_disjoint_segment(
    tokens: Sequence[int],
    *,
    requested_start: int,
    token_count: int,
    forbidden_ngrams: set[bytes],
    ngram_width: int,
    stride: int,
) -> tuple[int, list[int]]:
    if min(requested_start, token_count, stride) < 1:
        raise ValueError("segment parameters must be positive")
    for start in range(requested_start, len(tokens) - token_count + 1, stride):
        segment = [int(value) for value in tokens[start:start + token_count]]
        if not (_ngram_hashes(segment, ngram_width) & forbidden_ngrams):
            return start, segment
    raise ValueError("source has no sufficiently long disjoint segment")


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        text=True, capture_output=True,
    ).stdout.strip()


def _wikipedia_sources(source_dir: Path) -> dict[int, tuple[str, Path]]:
    pages: dict[int, tuple[str, Path]] = {}
    for path in sorted(source_dir.glob("wikipedia-*-wikitext.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for page in data["query"]["pages"]:
            revision = page.get("revisions", [{}])[0]
            content = revision.get("slots", {}).get("main", {}).get("content")
            if page.get("missing") or not content:
                raise RuntimeError(f"cached Wikipedia page is missing: {path}")
            page_id = int(page["pageid"])
            if page_id in pages:
                raise RuntimeError(f"duplicate cached Wikipedia page: {page_id}")
            pages[page_id] = (content, path)
    return pages


def _load_raw_source(
    heldout: dict[str, Any],
    *,
    heldout_data_dir: Path,
    llama_cpp: Path,
    wikipedia: dict[int, tuple[str, Path]],
) -> tuple[str, Path]:
    source = heldout["source"]
    if source["kind"] == "project_gutenberg":
        identifier = heldout["id"].removeprefix("general-gutenberg-")
        path = heldout_data_dir / "sources" / f"gutenberg-{identifier}.txt"
        return _strip_gutenberg(path.read_text(encoding="utf-8")), path
    if source["kind"] == "wikipedia":
        page_id = int(source["page_id"])
        try:
            return wikipedia[page_id]
        except KeyError as error:
            raise RuntimeError(f"cached Wikipedia page is absent: {page_id}") from error
    if source["kind"] == "git":
        path = llama_cpp / source["path"]
        return path.read_text(encoding="utf-8"), path
    raise RuntimeError(f"unsupported held-out source kind: {source['kind']}")


def build(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    started = time.perf_counter()
    tokenizer_dir = args.tokenizer_dir.resolve(strict=True)
    heldout_data_dir = args.heldout_data_dir.resolve(strict=True)
    heldout_corpus_path = heldout_data_dir / "corpus.jsonl"
    heldout_report_path = args.heldout_report.resolve(strict=True)
    llama_cpp = args.llama_cpp.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_path = args.output.resolve()
    revision = _git_revision(llama_cpp)
    if revision != args.llama_cpp_revision:
        raise RuntimeError(
            f"llama.cpp revision differs: {revision} != {args.llama_cpp_revision}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True, trust_remote_code=False)
    heldout_report = json.loads(heldout_report_path.read_text(encoding="utf-8"))
    _validate_canonical_manifest_sha256(heldout_report)
    heldout_manifest = heldout_report["canonical_manifest"]
    if _sha256_file(heldout_corpus_path) != heldout_manifest["corpus"]["sha256"]:
        raise RuntimeError("held-out corpus hash differs from its manifest")
    heldout_rows = [
        json.loads(line) for line in heldout_corpus_path.read_text(
            encoding="utf-8").splitlines() if line]
    if len(heldout_rows) != heldout_manifest["document_count"]:
        raise RuntimeError("held-out document count differs")
    heldout_by_id = {row["id"]: row for row in heldout_rows}
    if len(heldout_by_id) != len(heldout_rows):
        raise RuntimeError("held-out IDs are not unique")

    heldout_tokens = [
        [int(value) for value in tokenizer.encode(
            row["text"], add_special_tokens=False)]
        for row in heldout_rows
    ]
    forbidden_ngrams: set[bytes] = set()
    for tokens in heldout_tokens:
        forbidden_ngrams.update(_ngram_hashes(tokens, args.disjoint_ngram_width))
    heldout_ngram_count = len(forbidden_ngrams)
    wikipedia = _wikipedia_sources(heldout_data_dir / "sources")

    selected: list[tuple[dict[str, Any], int, str]] = []
    for source_id in GENERAL_SOURCE_IDS:
        selected.append((heldout_by_id[source_id], args.initial_offset, "language"))
    knowledge = [row for row in heldout_rows if row["stratum"] == "knowledge"]
    for row in knowledge[:10]:
        selected.append((row, args.initial_offset, "knowledge"))
    for source_id in REASONING_SOURCE_IDS:
        selected.append((heldout_by_id[source_id], args.initial_offset, "reasoning"))
        selected.append((heldout_by_id[source_id], args.second_reasoning_offset,
                         "reasoning"))
    code = [row for row in heldout_rows if row["stratum"] == "code"]
    for row in code[:10]:
        selected.append((row, args.initial_offset, "code"))
    multilingual = [
        row for row in heldout_rows if row["stratum"] == "multilingual"]
    by_language: dict[str, list[dict[str, Any]]] = {}
    for row in multilingual:
        by_language.setdefault(row["language"], []).append(row)
    for language in ("de", "es", "fr", "it", "pt"):
        for row in by_language[language][:2]:
            selected.append((row, args.initial_offset, "multilingual"))
    if len(selected) != 50:
        raise RuntimeError(f"expected 50 calibration selections, got {len(selected)}")

    calibration_rows = []
    token_rows = []
    manifest_documents = []
    calibration_ngrams: set[bytes] = set()
    source_cache: dict[str, tuple[list[int], Path]] = {}
    stratum_indices: Counter[str] = Counter()
    for heldout, requested_start, stratum in selected:
        source_id = heldout["id"]
        if source_id not in source_cache:
            raw_text, local_path = _load_raw_source(
                heldout, heldout_data_dir=heldout_data_dir,
                llama_cpp=llama_cpp, wikipedia=wikipedia)
            raw_tokens = [int(value) for value in tokenizer.encode(
                normalize_text(raw_text), add_special_tokens=False)]
            source_cache[source_id] = (raw_tokens, local_path)
        raw_tokens, local_path = source_cache[source_id]
        all_forbidden = forbidden_ngrams | calibration_ngrams
        start, tokens = _choose_disjoint_segment(
            raw_tokens,
            requested_start=requested_start,
            token_count=args.tokens_per_document,
            forbidden_ngrams=all_forbidden,
            ngram_width=args.disjoint_ngram_width,
            stride=args.offset_stride,
        )
        text = tokenizer.decode(
            tokens, skip_special_tokens=False,
            clean_up_tokenization_spaces=False)
        round_trip = [int(value) for value in tokenizer.encode(
            text, add_special_tokens=False)]
        if round_trip != tokens:
            raise RuntimeError(f"calibration segment does not round-trip: {source_id}")
        document_ngrams = _ngram_hashes(tokens, args.disjoint_ngram_width)
        if document_ngrams & all_forbidden:
            raise RuntimeError("calibration n-gram disjointness invariant failed")
        calibration_ngrams.update(document_ngrams)
        stratum_index = stratum_indices[stratum]
        stratum_indices[stratum] += 1
        document_id = f"calibration-{stratum}-{stratum_index:02d}"
        source = dict(heldout["source"])
        source.update({
            "heldout_source_id": source_id,
            "source_token_offset": start,
            "local_source": str(local_path),
            "local_source_sha256": (
                _sha256_file(local_path) if local_path.is_file() else None),
        })
        row = {
            "id": document_id,
            "stratum": stratum,
            "language": heldout["language"],
            "text": text,
            "source": source,
        }
        calibration_rows.append(row)
        token_rows.append(tokens)
        content = text.encode("utf-8")
        token_bytes = canonical_json_bytes(tokens)
        manifest_documents.append({
            "id": document_id,
            "stratum": stratum,
            "language": heldout["language"],
            "source": source,
            "content_bytes": len(content),
            "content_sha256": sha256_bytes(content),
            "token_count": len(tokens),
            "predicted_token_count": len(tokens) - 1,
            "token_ids_sha256": sha256_bytes(token_bytes),
        })

    expected_counts = {
        "code": 10, "knowledge": 10, "language": 10,
        "multilingual": 10, "reasoning": 10,
    }
    if dict(sorted(stratum_indices.items())) != expected_counts:
        raise RuntimeError(f"calibration strata differ: {stratum_indices}")

    corpus_bytes = b"".join(canonical_json_bytes(row) for row in calibration_rows)
    tokens_bytes = canonical_json_bytes(token_rows)
    corpus_path = output_dir / "corpus.jsonl"
    tokens_path = output_dir / "tokens.json"
    _atomic_bytes(corpus_path, corpus_bytes)
    _atomic_bytes(tokens_path, tokens_bytes)
    license_dir = output_dir / "licenses"
    license_dir.mkdir(parents=True, exist_ok=True)
    licenses = []
    for name in (
        "project-gutenberg-permission.html", "CC-BY-SA-4.0.txt",
        "llama.cpp-MIT.txt",
    ):
        source_path = heldout_data_dir / "licenses" / name
        destination = license_dir / name
        _atomic_bytes(destination, source_path.read_bytes())
        licenses.append({
            "path": str(destination), "sha256": _sha256_file(destination)})

    sequence_digest = hashlib.sha256()
    for tokens in token_rows:
        encoded = canonical_json_bytes(tokens)
        sequence_digest.update(len(tokens).to_bytes(8, "little"))
        sequence_digest.update(encoded)
    manifest = {
        "schema": 1,
        "corpus": {
            "path": str(corpus_path), "bytes": len(corpus_bytes),
            "sha256": sha256_bytes(corpus_bytes),
        },
        "tokens": {
            "path": str(tokens_path), "bytes": len(tokens_bytes),
            "sha256": sha256_bytes(tokens_bytes),
        },
        "document_count": len(calibration_rows),
        "documents": manifest_documents,
        "predicted_token_count": sum(len(row) - 1 for row in token_rows),
        "token_count_per_document": args.tokens_per_document,
        "token_sequence_sha256": sequence_digest.hexdigest(),
        "stratum_document_counts": expected_counts,
        "tokenizer": {
            "path": str(tokenizer_dir),
            "tokenizer_json_sha256": _sha256_file(tokenizer_dir / "tokenizer.json"),
        },
        "heldout_disjointness": {
            "heldout_report": str(heldout_report_path),
            "heldout_report_sha256": _sha256_file(heldout_report_path),
            "heldout_corpus_sha256": heldout_manifest["corpus"]["sha256"],
            "ngram_width": args.disjoint_ngram_width,
            "heldout_unique_ngram_count": heldout_ngram_count,
            "calibration_unique_ngram_count": len(calibration_ngrams),
            "shared_ngram_count": 0,
        },
        "llama_cpp": {"path": str(llama_cpp), "revision": revision},
        "licenses": licenses,
    }
    report = {
        "schema": "rco.qwen36.calibration_corpus.v1",
        "status": "pass",
        "scope": (
            "licensed stratified search calibration corpus with explicit "
            "token-level separation from the immutable release corpus"),
        "canonical_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "canonical_manifest": manifest,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_bytes(output_path, json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True,
    ).encode("utf-8") + b"\n")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--heldout-data-dir", type=Path, required=True)
    parser.add_argument("--heldout-report", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--llama-cpp-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens-per-document", type=int, default=256)
    parser.add_argument("--disjoint-ngram-width", type=int, default=32)
    parser.add_argument("--initial-offset", type=int, default=1536)
    parser.add_argument("--second-reasoning-offset", type=int, default=3072)
    parser.add_argument("--offset-stride", type=int, default=256)
    args = parser.parse_args()
    if min(
        args.tokens_per_document, args.disjoint_ngram_width,
        args.initial_offset, args.second_reasoning_offset,
        args.offset_stride,
    ) < 1:
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
        "token_sequence_sha256": manifest["token_sequence_sha256"],
        "shared_heldout_ngram_count": manifest["heldout_disjointness"][
            "shared_ngram_count"],
    }, indent=2, sort_keys=True))
