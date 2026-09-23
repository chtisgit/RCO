#!/usr/bin/env python3
"""Exercise genuine block candidates through an unmodified llama.cpp model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from native_gguf import import_pinned_gguf, write_selected_native_gguf
from native_store import NativeCandidateStore
from quant.ggml_native import GGMLNativeCodec, GGMLType


LLAMA_CPP_REVISION = "911f6cdc8ab8a530b2bee09ee61471a6f3178eeb"
ASSIGNMENTS = {
    "uniform_q2_0": GGMLType.Q2_0,
    "uniform_q4_0": GGMLType.Q4_0,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: Any) -> str:
    view = memoryview(value).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(view), 16 << 20):
        digest.update(view[start:start + (16 << 20)])
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


def _run_probe(probe: Path, model: Path) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        [str(probe), str(model)], check=True, text=True, capture_output=True,
        timeout=300,
    )
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
    output = directory / "generated.txt"
    command = [
        str(executable), "cli", "--model", str(model),
        "--ctx-size", "512", "--predict", "4",
        "--batch-size", "32", "--ubatch-size", "32",
        "--threads", "4", "--threads-batch", "4",
        "--temp", "0", "--seed", "20260923",
        "--log-verbosity", "4", "--fit", "off", "--no-warmup",
        "--reasoning", "off", "--single-turn", "--simple-io",
        "--color", "off", "--device", "none", "--gpu-layers", "0",
        "--no-op-offload", "--prompt",
        "Reply with exactly one word: Hello.", "--output-file", str(output),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command, check=False, text=True, capture_output=True, timeout=600)
    if completed.returncode != 0:
        stderr_tail = "\n".join(completed.stderr.splitlines()[-80:])
        raise RuntimeError(
            f"llama.cpp generation exited {completed.returncode}:\n{stderr_tail}")
    generated = output.read_text(encoding="utf-8")
    if not generated.strip():
        raise RuntimeError("llama.cpp completed without generated text")
    evidence_patterns = (
        "print_info: arch", "print_info: n_layer", "print_info: n_expert",
        "load_tensors: offloaded", "model buffer size", "compute buffer size",
        "prompt eval time", "eval time",
    )
    evidence = [
        line for line in completed.stderr.splitlines()
        if any(pattern in line for pattern in evidence_patterns)
    ]
    return {
        "command": command,
        "exit_status": completed.returncode,
        "generated_text": generated,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "loader_and_execution_evidence": evidence,
        "cuda_initialization_failed_before_cpu_only_run": (
            "ggml_cuda_init: failed to initialize CUDA" in completed.stderr
        ),
    }


def _validate_output(
    output_path: Path,
    store: NativeCandidateStore,
    selected_type: GGMLType,
    tensor_names: list[str],
    gguf: Any,
) -> dict[str, Any]:
    reader = gguf.GGUFReader(output_path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    missing = sorted(set(tensor_names) - set(tensors))
    if missing:
        raise RuntimeError(f"written GGUF is missing selected tensors: {missing}")
    records = []
    for name in tensor_names:
        tensor = tensors[name]
        metadata = store.metadata(name, selected_type)
        actual_hash = _sha256_array(tensor.data)
        if int(tensor.tensor_type) != int(selected_type):
            raise RuntimeError(f"written type mismatch for {name}")
        if tuple(int(value) for value in tensor.shape) != tuple(metadata["gguf_shape"]):
            raise RuntimeError(f"written shape mismatch for {name}")
        if int(tensor.n_bytes) != int(metadata["payload_bytes"]):
            raise RuntimeError(f"written byte count mismatch for {name}")
        if actual_hash != metadata["sha256"]:
            raise RuntimeError(f"written payload mismatch for {name}")
        records.append({
            "tensor": name,
            "ggml_type": selected_type.name,
            "ggml_type_id": int(selected_type),
            "gguf_shape": list(metadata["gguf_shape"]),
            "payload_bytes": int(tensor.n_bytes),
            "sha256": actual_hash,
            "data_offset": int(tensor.data_offset),
            "data_offset_aligned": int(tensor.data_offset) % int(reader.alignment) == 0,
        })
    return {
        "tensor_count": len(reader.tensors),
        "selected_tensor_count": len(records),
        "selected_payload_bytes": sum(item["payload_bytes"] for item in records),
        "selected_payloads_exact": True,
        "selected": records,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    llama_cpp = args.llama_cpp.resolve(strict=True)
    revision = _git_revision(llama_cpp)
    if revision != LLAMA_CPP_REVISION:
        raise RuntimeError(
            f"llama.cpp revision mismatch: {revision} != {LLAMA_CPP_REVISION}")
    reference = args.reference.resolve(strict=True)
    reference_sha256 = _sha256_file(reference)
    if reference_sha256 != args.reference_sha256:
        raise RuntimeError(
            f"reference GGUF mismatch: {reference_sha256} != "
            f"{args.reference_sha256}")
    codec = GGMLNativeCodec(args.ggml_library.resolve(strict=True))
    store = NativeCandidateStore(args.store.resolve(strict=True), codec)
    tensor_names = sorted(store.index["tensors"])
    if len(tensor_names) != 13:
        raise RuntimeError(f"expected 13 genuine block candidates, got {len(tensor_names)}")
    gguf = import_pinned_gguf(llama_cpp / "gguf-py")
    parent = args.temporary_parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for label, selected_type in ASSIGNMENTS.items():
        with tempfile.TemporaryDirectory(
            prefix=f"rco-qwen36-{label}-", dir=parent,
        ) as directory_name:
            directory = Path(directory_name)
            output = directory / f"qwen36-block0-{label}.gguf"
            assignment = {name: selected_type for name in tensor_names}
            write_started = time.perf_counter()
            writer_records = write_selected_native_gguf(
                reference, output, store, assignment,
                gguf_python=llama_cpp / "gguf-py",
            )
            write_seconds = time.perf_counter() - write_started
            validation = _validate_output(
                output, store, selected_type, tensor_names, gguf)
            if len(writer_records) != len(tensor_names):
                raise RuntimeError("writer did not report every selected tensor")
            model_sha256 = _sha256_file(output)
            model_bytes = output.stat().st_size
            probe = _run_probe(args.llama_probe.resolve(strict=True), output)
            if probe["layer_count"] != 40 or probe["embedding_length"] != 2048:
                raise RuntimeError(f"probe loaded unexpected model geometry: {probe}")
            generation = _run_generation(
                args.llama_executable.resolve(strict=True), output, directory)
            runs.append({
                "assignment": label,
                "ggml_type": selected_type.name,
                "ggml_type_id": int(selected_type),
                "write_seconds": write_seconds,
                "model_bytes": model_bytes,
                "model_sha256": model_sha256,
                "gguf_validation": validation,
                "unmodified_llama_cpp_model_load": probe,
                "unmodified_llama_cpp_cpu_generation": generation,
            })
    return {
        "schema": 1,
        "status": "pass",
        "scope": (
            "uniform Q2_0 and uniform Q4_0 assignments for all 13 eligible "
            "tensors in genuine Qwen3.6-35B-A3B block 0, copied unchanged into "
            "complete temporary GGUFs and exercised by unmodified llama.cpp on "
            "CPU; CUDA execution and end-to-end quality are not claimed"
        ),
        "llama_cpp": {
            "path": str(llama_cpp),
            "revision": revision,
            "executable": str(args.llama_executable.resolve()),
            "model_probe": str(args.llama_probe.resolve()),
        },
        "reference": {
            "path": str(reference),
            "bytes": reference.stat().st_size,
            "sha256": reference_sha256,
        },
        "candidate_store": {
            "path": str(args.store.resolve()),
            "index_sha256": _sha256_file(
                args.store.resolve() / "native-candidate-index.json"),
            "tensor_count": len(tensor_names),
            "candidate_count": int(store.index["candidate_count"]),
        },
        "runs": runs,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024),
        "peak_child_rss_bytes": (
            int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * 1024),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--llama-probe", type=Path, required=True)
    parser.add_argument("--llama-executable", type=Path, required=True)
    parser.add_argument("--temporary-parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(args)
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "runs": [item["assignment"] for item in report["runs"]],
        "elapsed_seconds": report["elapsed_seconds"],
        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
        "peak_child_rss_bytes": report["peak_child_rss_bytes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
