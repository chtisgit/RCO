#!/usr/bin/env python3
"""Build and validate the selected production Qwen3.6-35B native GGUF."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_qwen35_2b_resumable_gguf import (
    LLAMA_CPP_REVISION,
    _atomic_json,
    _git_revision,
    _load_json,
    _run_generation,
    _run_probe,
    _sha256_file,
    _validate_output,
)
from native_gguf import import_pinned_gguf, write_resumable_selected_native_gguf
from native_store import NativeCandidateStore
from quant.ggml_native import GGMLNativeCodec, GGMLType
from search.hard import realized_cost


def _production_assignment(
    report: dict[str, Any],
) -> tuple[dict[str, GGMLType], list[int]]:
    names = report["candidate_store"]["tensor_names"]
    choices = report["search"]["runs"][0]["selected_assignment"]
    if len(names) != 512 or len(choices) != 512:
        raise ValueError("production report does not contain 512 choices")
    if any(choice not in (0, 1) for choice in choices):
        raise ValueError("production assignment has a non-binary choice")
    return {
        name: (GGMLType.Q4_0 if choice else GGMLType.Q2_0)
        for name, choice in zip(names, choices)
    }, choices


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    reference = args.reference.resolve(strict=True)
    reference_hash = _sha256_file(reference)
    if reference_hash != args.reference_sha256:
        raise RuntimeError("reference GGUF checksum differs")
    llama_cpp = args.llama_cpp.resolve(strict=True)
    revision = _git_revision(llama_cpp)
    if revision != LLAMA_CPP_REVISION:
        raise RuntimeError("pinned llama.cpp revision differs")

    codec = GGMLNativeCodec(args.ggml_library.resolve(strict=True))
    store_path = args.store.resolve(strict=True)
    store = NativeCandidateStore(store_path, codec)
    search_path = args.search_report.resolve(strict=True)
    search = _load_json(search_path)
    assignment, choices = _production_assignment(search)
    if set(assignment) != set(store.index["tensors"]):
        raise RuntimeError("selected assignment does not cover the candidate store")
    names = search["candidate_store"]["tensor_names"]
    low_costs = [
        int(store.metadata(name, GGMLType.Q2_0)["aligned_gguf_bytes"])
        for name in names
    ]
    high_costs = [
        int(store.metadata(name, GGMLType.Q4_0)["aligned_gguf_bytes"])
        for name in names
    ]
    selected_cost = realized_cost(
        torch.tensor(choices), low_costs, high_costs)
    target_cost = int(search["candidate_store"]["target_cost"])
    if selected_cost != target_cost:
        raise RuntimeError("selected assignment does not meet its search budget")

    manifest_path = args.manifest.resolve(strict=True)
    manifest = _load_json(manifest_path)
    manifest_names = [entry["destination_name"] for entry in manifest["entries"]]
    if len(manifest_names) != 733 or len(set(manifest_names)) != 733:
        raise RuntimeError("production manifest inventory differs")
    searched_names = {
        entry["destination_name"] for entry in manifest["entries"]
        if entry.get("rco_search")
    }
    if searched_names != set(assignment):
        raise RuntimeError("manifest search inventory differs from assignment")
    copy_reasons: dict[str, int] = {}
    for entry in manifest["entries"]:
        if not entry.get("rco_search"):
            reason = entry["copy_reason"]
            copy_reasons[reason] = copy_reasons.get(reason, 0) + 1

    output = args.output_model.resolve()
    if output.exists():
        raise FileExistsError(f"output model already exists: {output}")
    interrupted_started = time.perf_counter()
    interrupted = write_resumable_selected_native_gguf(
        reference,
        output,
        store,
        assignment,
        gguf_python=llama_cpp / "gguf-py",
        chunk_bytes=args.chunk_bytes,
        stop_after_tensors=args.interrupt_after_tensors,
    )
    interrupted_seconds = time.perf_counter() - interrupted_started
    if interrupted["status"] != "incomplete" or output.exists():
        raise RuntimeError("interruption proof published a complete output")

    resume_started = time.perf_counter()
    construction = write_resumable_selected_native_gguf(
        reference,
        output,
        store,
        assignment,
        gguf_python=llama_cpp / "gguf-py",
        resume=True,
        chunk_bytes=args.chunk_bytes,
    )
    resume_seconds = time.perf_counter() - resume_started
    if construction["status"] != "complete":
        raise RuntimeError("resumed production GGUF did not complete")

    gguf = import_pinned_gguf(llama_cpp / "gguf-py")
    validation_started = time.perf_counter()
    validation = _validate_output(output, construction, gguf)
    validation_seconds = time.perf_counter() - validation_started
    if validation["tensor_count"] != 733:
        raise RuntimeError("production output tensor count differs")
    if validation["selected_tensor_count"] != 512:
        raise RuntimeError("production selected tensor count differs")
    if validation["copied_reference_tensor_count"] != 221:
        raise RuntimeError("production copied tensor count differs")

    probe = _run_probe(args.llama_probe.resolve(strict=True), output)
    if probe["layer_count"] != 40 or probe["embedding_length"] != 2048:
        raise RuntimeError(f"probe loaded unexpected geometry: {probe}")
    generation = _run_generation(
        args.llama_executable.resolve(strict=True), output, output.parent)
    selected = validation["selected"]
    selected_q2 = sum(item["ggml_type"] == "Q2_0" for item in selected)
    selected_q4 = sum(item["ggml_type"] == "Q4_0" for item in selected)
    return {
        "schema": 1,
        "status": "pass",
        "scope": (
            "restartable bounded-copy construction of the selected complete "
            "text-only Qwen3.6-35B-A3B production GGUF, exhaustive byte-level "
            "assignment/inventory validation, and unmodified pinned llama.cpp "
            "CPU load and generation"
        ),
        "environment": {
            "python": platform.python_version(),
            "llama_cpp_path": str(llama_cpp),
            "llama_cpp_revision": revision,
            "ggml_library": str(codec.library_path),
            "ggml_library_sha256": codec.library_sha256,
        },
        "reference": {
            "path": str(reference),
            "bytes": reference.stat().st_size,
            "sha256": reference_hash,
            "role": (
                "proven text-only metadata, canonical tensor order, and 221 "
                "explicit nonsearch tensor payloads only; all 512 searchable "
                "payloads come from the production native candidate store"
            ),
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
            "tensor_count": len(manifest_names),
            "searched_tensor_count": len(searched_names),
            "copied_tensor_count": len(manifest_names) - len(searched_names),
            "copy_reasons": copy_reasons,
        },
        "candidate_store": {
            "path": str(store_path),
            "index_sha256": _sha256_file(
                store_path / "native-candidate-index.json"),
            "tensor_count": int(store.index["tensor_count"]),
            "candidate_count": int(store.index["candidate_count"]),
        },
        "assignment": {
            "search_report_path": str(search_path),
            "search_report_sha256": _sha256_file(search_path),
            "target_aligned_gguf_bytes": target_cost,
            "realized_aligned_gguf_bytes": selected_cost,
            "q2_0_tensor_count": selected_q2,
            "q4_0_tensor_count": selected_q4,
            "choices": {name: value.name for name, value in assignment.items()},
        },
        "interruption_and_resume": {
            "interrupted_after_tensor_count": args.interrupt_after_tensors,
            "interrupted_completed_tensor_count": interrupted[
                "completed_tensor_count"],
            "incomplete_output_not_published": True,
            "interrupted_seconds": interrupted_seconds,
            "resume_status": construction["status"],
            "resume_seconds": resume_seconds,
            "plan_sha256": construction["plan_sha256"],
        },
        "output": {
            "path": str(output),
            "bytes": construction["output_bytes"],
            "sha256": construction["output_sha256"],
            "max_copy_chunk_bytes": max(
                interrupted["max_copy_chunk_bytes"],
                construction["max_copy_chunk_bytes"],
            ),
            "validation_seconds": validation_seconds,
            "validation": validation,
        },
        "unmodified_llama_cpp_model_load": probe,
        "unmodified_llama_cpp_cpu_generation": generation,
        "peak_process_rss_bytes": (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024),
        "peak_child_rss_bytes": (
            int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * 1024),
        "elapsed_seconds": time.perf_counter() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--search-report", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--llama-probe", type=Path, required=True)
    parser.add_argument("--llama-executable", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=8 << 20)
    parser.add_argument("--interrupt-after-tensors", type=int, default=366)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.chunk_bytes <= 0:
        raise ValueError("chunk bytes must be positive")
    if not 0 <= args.interrupt_after_tensors < 733:
        raise ValueError("interrupt count must lie in [0, 733)")
    report = audit(args)
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "model": report["output"]["path"],
        "model_sha256": report["output"]["sha256"],
        "selected_cost": report["assignment"]["realized_aligned_gguf_bytes"],
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
