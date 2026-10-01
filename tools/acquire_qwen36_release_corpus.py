#!/usr/bin/env python3
"""Acquire and pin the licensed Qwen3.6 release-evaluation corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from release_corpus import (  # noqa: E402
    canonical_json_bytes,
    prepare_document,
    sha256_bytes,
)


LLAMA_CPP_REVISION = "911f6cdc8ab8a530b2bee09ee61471a6f3178eeb"
USER_AGENT = "RCO-release-quality-audit/1.0 (noncommercial local evaluation)"
GUTENBERG_WORKS = (
    (1342, "Pride and Prejudice"), (84, "Frankenstein"),
    (11, "Alice's Adventures in Wonderland"),
    (1661, "The Adventures of Sherlock Holmes"),
    (98, "A Tale of Two Cities"), (2701, "Moby Dick"),
    (74, "The Adventures of Tom Sawyer"),
    (76, "Adventures of Huckleberry Finn"), (345, "Dracula"),
    (5200, "Metamorphosis"), (1952, "The Yellow Wallpaper"),
    (46, "A Christmas Carol"), (174, "The Picture of Dorian Gray"),
    (1400, "Great Expectations"), (1260, "Jane Eyre"),
    (768, "Wuthering Heights"), (4300, "Ulysses"),
    (1232, "The Prince"), (1080, "A Modest Proposal"),
    (2591, "Grimms' Fairy Tales"), (2600, "War and Peace"),
    (6130, "The Iliad"), (1497, "The Republic"),
    (160, "The Awakening"), (514, "Little Women"),
)
ENGLISH_WIKIPEDIA_TITLES = (
    "Quantum mechanics", "Photosynthesis", "French Revolution", "DNA",
    "Plate tectonics", "Roman Empire", "Compiler", "Black hole",
    "Periodic table", "Supply and demand", "Renaissance", "Immune system",
    "Algorithm", "Solar System", "World War I", "Evolution",
    "General relativity", "Climate change", "Ancient Egypt",
    "Printing press", "Human brain", "Internet", "Mathematics",
    "Electricity", "Philosophy",
)
MULTILINGUAL_WIKIPEDIA_TITLES = (
    ("de", "Quantenmechanik"), ("de", "Photosynthese"),
    ("de", "Französische Revolution"), ("de", "Desoxyribonukleinsäure"),
    ("de", "Schwarzes Loch"), ("es", "Mecánica cuántica"),
    ("es", "Fotosíntesis"), ("es", "Revolución francesa"),
    ("es", "Ácido desoxirribonucleico"), ("es", "Agujero negro"),
    ("fr", "Mécanique quantique"), ("fr", "Photosynthèse"),
    ("fr", "Révolution française"), ("fr", "Acide désoxyribonucléique"),
    ("fr", "Trou noir"), ("it", "Meccanica quantistica"),
    ("it", "Fotosintesi"), ("it", "Rivoluzione francese"),
    ("it", "DNA"), ("it", "Buco nero"),
    ("pt", "Mecânica quântica"), ("pt", "Fotossíntese"),
    ("pt", "Revolução Francesa"), ("pt", "Ácido desoxirribonucleico"),
    ("pt", "Buraco negro"),
)
CODE_FILES = (
    "src/llama-vocab.cpp", "src/llama-model.cpp", "src/llama-context.cpp",
    "src/llama-sampler.cpp", "src/llama-graph.cpp", "src/llama-kv-cache.cpp",
    "src/llama-arch.cpp", "src/llama-model-loader.cpp",
    "src/llama-quant.cpp", "src/llama-grammar.cpp", "common/arg.cpp",
    "common/common.cpp", "common/chat.cpp", "common/sampling.cpp",
    "common/json-schema-to-grammar.cpp", "common/console.cpp",
    "common/speculative.cpp", "common/peg-parser.cpp",
    "common/chat-diff-analyzer.cpp", "tools/perplexity/perplexity.cpp",
    "tools/llama-bench/llama-bench.cpp", "tools/imatrix/imatrix.cpp",
    "tools/server/server-context.cpp", "tools/server/server-models.cpp",
    "tools/server/server-tools.cpp",
)


def _request(url: str) -> tuple[bytes, dict[str, str]]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(6):
        try:
            with urlopen(request, timeout=60) as response:
                return response.read(), {
                    key.lower(): value for key, value in response.headers.items()
                }
        except HTTPError as error:
            if error.code not in (429, 503) or attempt == 5:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else float(2 ** attempt)
            time.sleep(min(60.0, max(1.0, delay)))
    raise AssertionError("unreachable retry loop")


def _cached_request(url: str, path: Path) -> bytes:
    if path.is_file():
        return path.read_bytes()
    body, _ = _request(url)
    _atomic_bytes(path, body)
    return body


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


def _strip_gutenberg(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start = next((index + 1 for index, line in enumerate(lines)
                  if "*** START OF THE PROJECT GUTENBERG EBOOK" in line), None)
    end = next((index for index, line in enumerate(lines)
                if "*** END OF THE PROJECT GUTENBERG EBOOK" in line), None)
    if start is None or end is None or end <= start:
        raise RuntimeError("Project Gutenberg markers are missing")
    return "\n".join(lines[start:end])


def _wikipedia_documents(
    language: str,
    titles: tuple[str, ...],
    cache_path: Path,
) -> list[tuple[str, dict[str, Any]]]:
    query = urlencode({
        "action": "query", "prop": "revisions", "redirects": "1",
        "rvprop": "ids|timestamp|content", "rvslots": "main",
        "titles": "|".join(titles),
        "format": "json", "formatversion": "2",
    })
    url = f"https://{language}.wikipedia.org/w/api.php?{query}"
    body = _cached_request(url, cache_path)
    results = []
    for page in json.loads(body)["query"]["pages"]:
        revision = page.get("revisions", [{}])[0]
        content = revision.get("slots", {}).get("main", {}).get("content")
        if page.get("missing") or not content:
            raise RuntimeError(
                f"Wikipedia page is missing or empty: {language}:{page.get('title')}")
        page_title = page["title"]
        results.append((content, {
            "api_url": url,
            "content_format": "mediawiki_wikitext",
            "language": language,
            "page_id": int(page["pageid"]),
            "revision_id": int(revision["revid"]),
            "revision_timestamp": revision["timestamp"],
            "title": page_title,
            "url": (
                f"https://{language}.wikipedia.org/wiki/"
                f"{page_title.replace(' ', '_')}?oldid={revision['revid']}"
            ),
        }))
    if len(results) != len(titles):
        raise RuntimeError(
            f"Wikipedia returned {len(results)} of {len(titles)} {language} pages")
    return results


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        text=True, capture_output=True,
    ).stdout.strip()


def acquire(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    started = time.perf_counter()
    tokenizer_dir = args.tokenizer_dir.resolve(strict=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True, trust_remote_code=False)
    calibration = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    forbidden = [calibration["calibration"]["input_ids"]]
    llama_cpp = args.llama_cpp.resolve(strict=True)
    revision = _git_revision(llama_cpp)
    if revision != args.llama_cpp_revision:
        raise RuntimeError(
            f"llama.cpp revision mismatch: {revision} != "
            f"{args.llama_cpp_revision}")

    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    sources = data_dir / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    licenses = data_dir / "licenses"
    licenses.mkdir(parents=True, exist_ok=True)
    pg_policy = _cached_request(
        "https://www.gutenberg.org/policy/permission.html",
        licenses / "project-gutenberg-permission.html",
    )
    cc_by_sa = _cached_request(
        "https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt",
        licenses / "CC-BY-SA-4.0.txt",
    )
    _atomic_bytes(licenses / "project-gutenberg-permission.html", pg_policy)
    _atomic_bytes(licenses / "CC-BY-SA-4.0.txt", cc_by_sa)
    _atomic_bytes(
        licenses / "llama.cpp-MIT.txt", (llama_cpp / "LICENSE").read_bytes())

    raw_documents: list[dict[str, Any]] = []
    for book_id, title in GUTENBERG_WORKS:
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        body = _cached_request(url, sources / f"gutenberg-{book_id}.txt")
        raw_documents.append({
            "id": f"general-gutenberg-{book_id}", "stratum": "general",
            "language": "en", "text": _strip_gutenberg(body.decode("utf-8")),
            "source": {
                "kind": "project_gutenberg", "title": title, "url": url,
                "license": "public domain work; Gutenberg wrapper removed",
                "license_file": "licenses/project-gutenberg-permission.html",
            },
        })
    english_batches = (
        ENGLISH_WIKIPEDIA_TITLES[:15], ENGLISH_WIKIPEDIA_TITLES[15:],
    )
    for batch_index, titles in enumerate(english_batches):
        for text, source in _wikipedia_documents(
            "en", titles,
            sources / f"wikipedia-en-{batch_index}-wikitext.json",
        ):
            source.update({
                "kind": "wikipedia", "license": "CC BY-SA 4.0",
                "license_file": "licenses/CC-BY-SA-4.0.txt",
            })
            raw_documents.append({
                "id": f"knowledge-wikipedia-{source['page_id']}",
                "stratum": "knowledge", "language": "en", "text": text,
                "source": source,
            })
    for index, relative_name in enumerate(CODE_FILES):
        path = llama_cpp / relative_name
        raw_documents.append({
            "id": f"code-llama-cpp-{index:02d}", "stratum": "code",
            "language": "cpp", "text": path.read_text(encoding="utf-8"),
            "source": {
                "kind": "git", "path": relative_name, "revision": revision,
                "url": (
                    "https://github.com/ggml-org/llama.cpp/blob/"
                    f"{revision}/{relative_name}"
                ),
                "source_sha256": _sha256_file(path), "license": "MIT",
                "license_file": "licenses/llama.cpp-MIT.txt",
            },
        })
    multilingual_by_language: dict[str, list[str]] = {}
    for language, title in MULTILINGUAL_WIKIPEDIA_TITLES:
        multilingual_by_language.setdefault(language, []).append(title)
    for language, titles in multilingual_by_language.items():
        for text, source in _wikipedia_documents(
            language, tuple(titles),
            sources / f"wikipedia-{language}-wikitext.json",
        ):
            source.update({
                "kind": "wikipedia", "license": "CC BY-SA 4.0",
                "license_file": "licenses/CC-BY-SA-4.0.txt",
            })
            raw_documents.append({
                "id": f"multilingual-{language}-wikipedia-{source['page_id']}",
                "stratum": "multilingual", "language": language, "text": text,
                "source": source,
            })

    if len(raw_documents) != 100:
        raise RuntimeError(f"expected 100 source documents, got {len(raw_documents)}")
    documents = []
    manifest_documents = []
    sequence_digest = hashlib.sha256()
    for raw in raw_documents:
        text, tokens = prepare_document(
            tokenizer, raw.pop("text"), token_count=args.tokens_per_document,
            forbidden_sequences=forbidden,
        )
        document = {**raw, "text": text}
        content_bytes = text.encode("utf-8")
        token_bytes = canonical_json_bytes(tokens)
        documents.append(document)
        manifest_documents.append({
            key: value for key, value in {
                **raw,
                "content_bytes": len(content_bytes),
                "content_sha256": sha256_bytes(content_bytes),
                "token_count": len(tokens),
                "predicted_token_count": len(tokens) - 1,
                "token_ids_sha256": sha256_bytes(token_bytes),
            }.items()
        })
        sequence_digest.update(len(tokens).to_bytes(8, "little"))
        sequence_digest.update(token_bytes)

    corpus_bytes = b"".join(canonical_json_bytes(item) for item in documents)
    corpus_path = data_dir / "corpus.jsonl"
    _atomic_bytes(corpus_path, corpus_bytes)
    strata = sorted({item["stratum"] for item in manifest_documents})
    canonical_manifest = {
        "schema": 1,
        "corpus": {
            "path": str(corpus_path), "bytes": len(corpus_bytes),
            "sha256": sha256_bytes(corpus_bytes),
        },
        "document_count": len(manifest_documents),
        "documents": manifest_documents,
        "predicted_token_count": sum(
            item["predicted_token_count"] for item in manifest_documents),
        "strata": strata,
        "stratum_document_counts": {
            stratum: sum(item["stratum"] == stratum
                         for item in manifest_documents)
            for stratum in strata
        },
        "token_count_per_document": args.tokens_per_document,
        "token_sequence_sha256": sequence_digest.hexdigest(),
        "tokenizer": {
            "path": str(tokenizer_dir),
            "tokenizer_json_sha256": _sha256_file(tokenizer_dir / "tokenizer.json"),
        },
        "search_disjointness": {
            "calibration_report": str(args.calibration_report.resolve()),
            "calibration_input_ids": forbidden[0],
            "calibration_sequence_absent_from_every_document": True,
        },
        "licenses": [
            {
                "path": str(licenses / "project-gutenberg-permission.html"),
                "sha256": _sha256_file(
                    licenses / "project-gutenberg-permission.html"),
            },
            {
                "path": str(licenses / "CC-BY-SA-4.0.txt"),
                "sha256": _sha256_file(licenses / "CC-BY-SA-4.0.txt"),
            },
            {
                "path": str(licenses / "llama.cpp-MIT.txt"),
                "sha256": _sha256_file(licenses / "llama.cpp-MIT.txt"),
            },
        ],
    }
    manifest_bytes = canonical_json_bytes(canonical_manifest)
    report = {
        "schema": 1,
        "status": "pass",
        "scope": (
            "immutable, locally licensed, search-disjoint held-out corpus for "
            "Qwen3.6 release-quality evaluation"
        ),
        "canonical_manifest_sha256": sha256_bytes(manifest_bytes),
        "canonical_manifest": canonical_manifest,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_bytes(args.output, json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True,
    ).encode("utf-8") + b"\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens-per-document", type=int, default=1050)
    parser.add_argument("--llama-cpp-revision", default=LLAMA_CPP_REVISION)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.tokens_per_document < 2:
        raise ValueError("--tokens-per-document must be at least two")
    report = acquire(args)
    manifest = report["canonical_manifest"]
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "canonical_manifest_sha256": report["canonical_manifest_sha256"],
        "document_count": manifest["document_count"],
        "predicted_token_count": manifest["predicted_token_count"],
        "strata": manifest["strata"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
