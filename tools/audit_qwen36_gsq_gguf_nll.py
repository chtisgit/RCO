#!/usr/bin/env python3
"""Evaluate a GGUF on the pinned held-out corpus, one reset document at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from release_corpus import canonical_json_bytes, sha256_bytes  # noqa: E402


BUNDLE_MAGIC = b"RCONLL1\0"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_corpus(
    report_path: Path, tokenizer: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[int]]]:
    report = _load_json(report_path)
    canonical = report["canonical_manifest"]
    if sha256_bytes(canonical_json_bytes(canonical)) != report["canonical_manifest_sha256"]:
        raise RuntimeError("held-out canonical manifest digest mismatch")
    corpus_path = Path(canonical["corpus"]["path"]).resolve(strict=True)
    if _sha256_file(corpus_path) != canonical["corpus"]["sha256"]:
        raise RuntimeError("held-out corpus payload digest mismatch")
    documents = [
        json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    metadata = canonical["documents"]
    if len(documents) != len(metadata):
        raise RuntimeError("held-out corpus document count mismatch")
    token_rows: list[list[int]] = []
    sequence_digest = hashlib.sha256()
    for document, expected in zip(documents, metadata, strict=True):
        if document["id"] != expected["id"]:
            raise RuntimeError("held-out corpus document order mismatch")
        if sha256_bytes(document["text"].encode("utf-8")) != expected["content_sha256"]:
            raise RuntimeError(f"held-out content mismatch: {document['id']}")
        tokens = [int(value) for value in tokenizer.encode(
            document["text"], add_special_tokens=False)]
        token_bytes = canonical_json_bytes(tokens)
        if (len(tokens) != expected["token_count"]
                or sha256_bytes(token_bytes) != expected["token_ids_sha256"]):
            raise RuntimeError(f"held-out token mismatch: {document['id']}")
        token_rows.append(tokens)
        sequence_digest.update(len(tokens).to_bytes(8, "little"))
        sequence_digest.update(token_bytes)
    if sequence_digest.hexdigest() != canonical["token_sequence_sha256"]:
        raise RuntimeError("held-out combined token-sequence digest mismatch")
    return report, documents, token_rows


def _write_bundle(
    path: Path, documents: list[dict[str, Any]], token_rows: list[list[int]],
) -> None:
    with path.open("wb") as handle:
        handle.write(BUNDLE_MAGIC)
        handle.write(struct.pack("<II", 1, len(documents)))
        for document, tokens in zip(documents, token_rows, strict=True):
            identifier = document["id"].encode("utf-8")
            text = document["text"].encode("utf-8")
            if max(len(identifier), len(text), len(tokens)) >= 2**32:
                raise ValueError("input bundle field exceeds uint32")
            handle.write(struct.pack("<I", len(identifier)))
            handle.write(identifier)
            handle.write(struct.pack("<I", len(text)))
            handle.write(text)
            handle.write(struct.pack("<I", len(tokens)))
            handle.write(struct.pack(f"<{len(tokens)}i", *tokens))


def _aggregate(documents: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = sum(int(item["predicted_token_count"]) for item in documents)
    total_nll = math.fsum(float(item["nll_sum"]) for item in documents)
    mean_nll = total_nll / predicted if predicted else None
    return {
        "document_count": len(documents),
        "predicted_token_count": predicted,
        "nll_sum": total_nll,
        "mean_nll": mean_nll,
        "perplexity": (
            math.exp(mean_nll) if mean_nll is not None and mean_nll < 700 else None
        ),
        "elapsed_seconds": math.fsum(float(item["seconds"]) for item in documents),
        "document_ids": [item["id"] for item in documents],
        "document_mean_nll": [float(item["mean_nll"]) for item in documents],
    }


def _git_revision(directory: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True,
    ).strip()


def audit(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    started = time.perf_counter()
    model = args.model.resolve(strict=True)
    model_dir = args.model_dir.resolve(strict=True)
    corpus_manifest = args.corpus_manifest.resolve(strict=True)
    helper = args.helper.resolve(strict=True)
    llama_dir = args.llama_dir.resolve(strict=True)
    output = args.output.resolve()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    corpus_report, documents, token_rows = _load_corpus(corpus_manifest, tokenizer)
    source = Path(__file__).with_name("gguf_document_nll.cpp").resolve(strict=True)
    identity = {
        "schema": "rco.gguf_document_nll.v1",
        "model": {
            "path": str(model), "bytes": model.stat().st_size,
            "sha256": _sha256_file(model),
        },
        "model_dir": str(model_dir),
        "corpus_manifest": {
            "path": str(corpus_manifest),
            "canonical_manifest_sha256": corpus_report["canonical_manifest_sha256"],
        },
        "token_sequence_sha256": corpus_report["canonical_manifest"]["token_sequence_sha256"],
        "helper": {
            "path": str(helper), "sha256": _sha256_file(helper),
            "source_path": str(source), "source_sha256": _sha256_file(source),
        },
        "llama_cpp": {"path": str(llama_dir), "revision": _git_revision(llama_dir)},
        "parameters": {
            "gpu_layers": args.gpu_layers, "threads": args.threads,
            "ubatch": args.ubatch, "parse_special": args.parse_special,
        },
    }
    fingerprint = sha256_bytes(canonical_json_bytes(identity))
    if output.exists():
        report = _load_json(output)
        if report.get("fingerprint") != fingerprint or report.get("identity") != identity:
            raise RuntimeError("existing report identity does not match this run")
        if report.get("status") == "complete":
            return report
    else:
        report = {
            "status": "incomplete", "fingerprint": fingerprint,
            "identity": identity, "environment": None, "documents": [],
            "aggregate": _aggregate([]), "invocations": [],
        }
        _atomic_json(output, report)

    completed = report["documents"]
    for index, item in enumerate(completed):
        if item["index"] != index or item["id"] != documents[index]["id"]:
            raise RuntimeError("existing report document prefix is not contiguous")
    start = len(completed)
    if start >= len(documents):
        report["status"] = "complete"
        report["aggregate"] = _aggregate(completed)
        _atomic_json(output, report)
        return report
    count = len(documents) - start
    if args.stop_after_documents is not None:
        count = min(count, args.stop_after_documents)

    output.parent.mkdir(parents=True, exist_ok=True)
    bundle_handle = tempfile.NamedTemporaryFile(
        prefix="rco-nll-", suffix=".bundle", dir=output.parent, delete=False)
    bundle_path = Path(bundle_handle.name)
    bundle_handle.close()
    stderr_handle = tempfile.NamedTemporaryFile(
        prefix="rco-nll-", suffix=".stderr", dir=output.parent, delete=False)
    stderr_path = Path(stderr_handle.name)
    stderr_handle.close()
    invocation = {
        "start_document": start, "document_count": count,
        "started_unix_seconds": time.time(), "result": "running",
    }
    report["invocations"].append(invocation)
    _atomic_json(output, report)
    process: subprocess.Popen[str] | None = None
    try:
        _write_bundle(bundle_path, documents, token_rows)
        command = [
            str(helper), "--model", str(model), "--input", str(bundle_path),
            "--gpu-layers", str(args.gpu_layers), "--threads", str(args.threads),
            "--ubatch", str(args.ubatch), "--start-document", str(start),
            "--document-count", str(count),
        ] + (["--parse-special"] if args.parse_special else [])
        with stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=stderr, text=True,
                encoding="utf-8")
            assert process.stdout is not None
            for line in process.stdout:
                event = json.loads(line)
                if event.get("type") == "environment":
                    if report["environment"] not in (None, event):
                        raise RuntimeError("helper environment changed during resume")
                    report["environment"] = event
                    _atomic_json(output, report)
                    continue
                expected_index = len(report["documents"])
                if event.get("type") != "document" or event.get("index") != expected_index:
                    raise RuntimeError("unexpected helper output sequence")
                expected = documents[expected_index]
                expected_tokens = len(token_rows[expected_index])
                if (event.get("id") != expected["id"]
                        or event.get("token_count") != expected_tokens
                        or event.get("predicted_token_count") != expected_tokens - 1):
                    raise RuntimeError("helper document metadata mismatch")
                nll_sum = float(event["nll_sum"])
                mean_nll = float(event["mean_nll"])
                if (not math.isfinite(nll_sum) or not math.isfinite(mean_nll)
                        or abs(mean_nll - nll_sum / (expected_tokens - 1)) > 1e-10):
                    raise RuntimeError("helper returned invalid NLL values")
                report["documents"].append(event)
                report["aggregate"] = _aggregate(report["documents"])
                _atomic_json(output, report)
            return_code = process.wait()
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        invocation["return_code"] = return_code
        invocation["stderr_tail"] = stderr_text[-16000:]
        invocation["elapsed_seconds"] = time.perf_counter() - started
        if return_code != 0:
            invocation["result"] = "failed"
            _atomic_json(output, report)
            raise RuntimeError(f"GGUF NLL helper failed with status {return_code}")
        if len(report["documents"]) != start + count:
            invocation["result"] = "failed"
            _atomic_json(output, report)
            raise RuntimeError("GGUF NLL helper returned too few documents")
        invocation["result"] = "complete"
        report["status"] = (
            "complete" if len(report["documents"]) == len(documents) else "incomplete"
        )
        report["aggregate"] = _aggregate(report["documents"])
        _atomic_json(output, report)
        return report
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        invocation["result"] = "failed"
        invocation["return_code"] = None if process is None else process.returncode
        if stderr_path.exists():
            invocation["stderr_tail"] = stderr_path.read_text(
                encoding="utf-8", errors="replace")[-16000:]
        invocation["elapsed_seconds"] = time.perf_counter() - started
        _atomic_json(output, report)
        raise
    finally:
        bundle_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--llama-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ubatch", type=int, default=64)
    parser.add_argument("--stop-after-documents", type=int)
    parser.add_argument(
        "--parse-special", action="store_true",
        help="have llama.cpp match control-token strings in the text, as the HF "
             "tokenizer does (needed when a document contains e.g. <|im_end|>)")
    args = parser.parse_args()
    if args.gpu_layers < 0 or args.threads < 1 or args.ubatch < 1:
        parser.error("invalid execution parameter")
    if args.stop_after_documents is not None and args.stop_after_documents < 1:
        parser.error("--stop-after-documents must be positive")
    return args


if __name__ == "__main__":
    result = audit(_parse_args())
    print(json.dumps({
        "status": result["status"], "aggregate": result["aggregate"],
    }, indent=2, sort_keys=True))
