#!/usr/bin/env python3
"""Build and validate a restartable mixed-native Qwen3.5-2B GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from native_gguf import (
    import_pinned_gguf,
    write_resumable_selected_native_gguf,
)
from native_store import NativeCandidateStore
from quant.ggml_native import GGMLNativeCodec, GGMLType


LLAMA_CPP_REVISION = "911f6cdc8ab8a530b2bee09ee61471a6f3178eeb"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: Any) -> str:
    view = memoryview(value).cast("B")
    digest = hashlib.sha256()
    try:
        for start in range(0, len(view), 16 << 20):
            digest.update(view[start:start + (16 << 20)])
    finally:
        view.release()
    return digest.hexdigest()


def _sha256_range(path: Path, offset: int, size: int) -> str:
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as handle:
        handle.seek(offset)
        while remaining:
            chunk = handle.read(min(16 << 20, remaining))
            if not chunk:
                raise RuntimeError(f"output is truncated at byte {handle.tell()}")
            digest.update(chunk)
            remaining -= len(chunk)
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


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        text=True, capture_output=True,
    ).stdout.strip()


def _mixed_assignment(report: dict[str, Any]) -> dict[str, GGMLType]:
    matches = [
        item for item in report["assignments"]
        if item["name"] == "alternating_q2_0_q4_0"]
    if len(matches) != 1:
        raise ValueError("assignment report has no unique mixed assignment")
    return {
        name: GGMLType[value]
        for name, value in matches[0]["assignment"].items()
    }


def _validate_output(
    output: Path,
    construction: dict[str, Any],
    gguf: Any,
) -> dict[str, Any]:
    reader = gguf.GGUFReader(output)
    tensors = list(reader.tensors)
    records = construction["tensors"]
    if len(tensors) != len(records):
        raise RuntimeError("output tensor inventory length differs")
    selected = []
    copied = 0
    payload_bytes = 0
    for tensor, record in zip(tensors, records):
        if tensor.name != record["tensor"]:
            raise RuntimeError("output tensor order differs from construction")
        if int(tensor.tensor_type) != int(record["ggml_type_id"]):
            raise RuntimeError(f"output type differs for {tensor.name}")
        if [int(value) for value in tensor.shape] != record["gguf_shape"]:
            raise RuntimeError(f"output shape differs for {tensor.name}")
        if int(tensor.n_bytes) != int(record["payload_bytes"]):
            raise RuntimeError(f"output size differs for {tensor.name}")
        # GGUFReader exposes Tensor.data_offset as an absolute file offset.
        absolute_offset = int(tensor.data_offset)
        if absolute_offset != int(record["data_offset"]):
            raise RuntimeError(f"output offset differs for {tensor.name}")
        actual_hash = _sha256_range(
            output, absolute_offset, int(tensor.n_bytes))
        if actual_hash != record["sha256"]:
            raise RuntimeError(f"output payload differs for {tensor.name}")
        if absolute_offset % int(reader.alignment):
            raise RuntimeError(f"output offset is unaligned for {tensor.name}")
        payload_bytes += int(tensor.n_bytes)
        if record["source"] == "candidate":
            selected.append({
                "tensor": tensor.name,
                "ggml_type": record["ggml_type"],
                "ggml_type_id": record["ggml_type_id"],
                "gguf_shape": record["gguf_shape"],
                "payload_bytes": record["payload_bytes"],
                "data_offset": absolute_offset,
                "sha256": actual_hash,
            })
        else:
            copied += 1
    return {
        "tensor_count": len(tensors),
        "selected_tensor_count": len(selected),
        "copied_reference_tensor_count": copied,
        "payload_bytes": payload_bytes,
        "alignment": int(reader.alignment),
        "all_offsets_aligned": True,
        "all_payloads_exact": True,
        "selected": selected,
    }


def _run_probe(probe: Path, model: Path) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        [str(probe), str(model)], check=True, text=True,
        capture_output=True, timeout=300)
    result = json.loads(completed.stdout.strip())
    if result.get("status") != "pass":
        raise RuntimeError(f"unexpected model probe result: {result}")
    result["loaded_tensor_types"] = {
        name: int(count)
        for name, count in re.findall(
            r"- type\s+(\S+):\s+(\d+) tensors", completed.stderr)
    }
    result["elapsed_seconds"] = time.perf_counter() - started
    result["stderr_sha256"] = hashlib.sha256(
        completed.stderr.encode()).hexdigest()
    return result


def _run_generation(executable: Path, model: Path, directory: Path) -> dict[str, Any]:
    generated_path = directory / "qwen35-2b-generated.txt"
    command = [
        str(executable), "cli", "--model", str(model),
        "--ctx-size", "512", "--predict", "4",
        "--batch-size", "32", "--ubatch-size", "32",
        "--threads", "4", "--threads-batch", "4",
        "--temp", "0", "--seed", "20260927",
        "--log-verbosity", "4", "--fit", "off", "--no-warmup",
        "--reasoning", "off", "--single-turn", "--simple-io",
        "--color", "off", "--device", "none", "--gpu-layers", "0",
        "--no-op-offload", "--prompt",
        "Reply with exactly one word: Hello.",
        "--output-file", str(generated_path),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command, check=False, text=True, capture_output=True, timeout=600)
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.splitlines()[-80:])
        raise RuntimeError(
            f"llama.cpp generation exited {completed.returncode}:\n{tail}")
    generated = generated_path.read_text(encoding="utf-8")
    if not generated.strip():
        raise RuntimeError("llama.cpp completed without generated text")
    evidence_patterns = (
        "print_info: arch", "print_info: n_layer", "load_tensors: offloaded",
        "model buffer size", "compute buffer size", "prompt eval time",
        "eval time",
    )
    return {
        "command": command,
        "exit_status": completed.returncode,
        "generated_text": generated,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "loader_and_execution_evidence": [
            line for line in completed.stderr.splitlines()
            if any(pattern in line for pattern in evidence_patterns)
        ],
        "cuda_initialization_failed_before_cpu_only_run": (
            "ggml_cuda_init: failed to initialize CUDA" in completed.stderr),
    }


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
    assignment_report_path = args.assignment_report.resolve(strict=True)
    assignment_report = _load_json(assignment_report_path)
    assignment = _mixed_assignment(assignment_report)
    if set(assignment) != set(store.index["tensors"]):
        raise RuntimeError("mixed assignment does not cover the candidate store")
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
    if interrupted["status"] != "incomplete":
        raise RuntimeError("interruption proof unexpectedly completed the output")
    if output.exists():
        raise RuntimeError("incomplete GGUF was published")

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
        raise RuntimeError("resumed GGUF construction did not complete")

    gguf = import_pinned_gguf(llama_cpp / "gguf-py")
    validation_started = time.perf_counter()
    validation = _validate_output(output, construction, gguf)
    validation_seconds = time.perf_counter() - validation_started
    if validation["tensor_count"] != 320:
        raise RuntimeError("Qwen3.5-2B output tensor count differs")
    if validation["selected_tensor_count"] != 8:
        raise RuntimeError("Qwen3.5-2B selected tensor count differs")

    probe = _run_probe(args.llama_probe.resolve(strict=True), output)
    if probe["layer_count"] != 24 or probe["embedding_length"] != 2048:
        raise RuntimeError(f"probe loaded unexpected geometry: {probe}")
    generation = _run_generation(
        args.llama_executable.resolve(strict=True), output, output.parent)
    return {
        "schema": 1,
        "status": "pass",
        "scope": (
            "restartable bounded-copy construction of a complete text-only "
            "Qwen3.5-2B GGUF with the proven mixed block-0 native assignment, "
            "complete byte-level inventory validation, and unmodified pinned "
            "llama.cpp CPU load and generation"
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
        },
        "candidate_store": {
            "path": str(store_path),
            "index_sha256": _sha256_file(
                store_path / "native-candidate-index.json"),
            "tensor_count": int(store.index["tensor_count"]),
            "candidate_count": int(store.index["candidate_count"]),
        },
        "assignment": {
            "source_path": str(assignment_report_path),
            "source_sha256": _sha256_file(assignment_report_path),
            "choices": {name: value.name for name, value in assignment.items()},
        },
        "interruption_and_resume": {
            "interrupted_after_tensor_count": args.interrupt_after_tensors,
            "interrupted_status": interrupted["status"],
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
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--assignment-report", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--llama-probe", type=Path, required=True)
    parser.add_argument("--llama-executable", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=8 << 20)
    parser.add_argument("--interrupt-after-tensors", type=int, default=160)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.chunk_bytes <= 0:
        raise ValueError("chunk bytes must be positive")
    if not 0 <= args.interrupt_after_tensors < 320:
        raise ValueError("interrupt count must lie in [0, 320)")
    report = audit(args)
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "model": report["output"]["path"],
        "model_sha256": report["output"]["sha256"],
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
