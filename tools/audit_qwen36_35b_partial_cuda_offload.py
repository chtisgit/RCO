#!/usr/bin/env python3
"""Prove partial CUDA offload of an existing Qwen3.6-35B GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runtime_offload import parse_partial_cuda_offload


LLAMA_CPP_REVISION = "911f6cdc8ab8a530b2bee09ee61471a6f3178eeb"


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


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        text=True, capture_output=True,
    ).stdout.strip()


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    model = args.model.resolve(strict=True)
    llama_cpp = args.llama_cpp.resolve(strict=True)
    executable = args.llama_executable.resolve(strict=True)
    revision = _git_revision(llama_cpp)
    if revision != args.llama_cpp_revision:
        raise RuntimeError(
            f"llama.cpp revision mismatch: {revision} != "
            f"{args.llama_cpp_revision}")

    model_sha256 = _sha256_file(model)
    if model_sha256 != args.model_sha256:
        raise RuntimeError(
            f"model GGUF mismatch: {model_sha256} != {args.model_sha256}")

    device_command = [str(executable), "cli", "--list-devices"]
    device_result = subprocess.run(
        device_command, check=False, text=True, capture_output=True,
        timeout=args.timeout_seconds,
    )
    if device_result.returncode != 0:
        raise RuntimeError(
            "llama.cpp device enumeration failed: "
            f"{device_result.stderr.strip()}")
    if f"{args.device}:" not in device_result.stdout:
        raise RuntimeError(
            f"llama.cpp did not enumerate {args.device}: "
            f"{device_result.stdout.strip()}")

    temporary_parent = args.temporary_parent.resolve()
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="rco-qwen36-partial-offload-", dir=temporary_parent,
    ) as directory_name:
        generated_path = Path(directory_name) / "generated.txt"
        command = [
            str(executable), "cli", "--model", str(model),
            "--ctx-size", str(args.context_size),
            "--predict", str(args.predict_tokens),
            "--batch-size", "32", "--ubatch-size", "32",
            "--threads", "4", "--threads-batch", "4",
            "--temp", "0", "--seed", str(args.seed),
            "--log-verbosity", "4", "--fit", "off", "--no-warmup",
            "--reasoning", "off", "--single-turn", "--simple-io",
            "--color", "off", "--device", args.device,
            "--gpu-layers", str(args.gpu_layers),
            "--prompt", args.prompt, "--output-file", str(generated_path),
        ]
        run_started = time.perf_counter()
        completed = subprocess.run(
            command, check=False, text=True, capture_output=True,
            timeout=args.timeout_seconds,
        )
        run_seconds = time.perf_counter() - run_started
        if completed.returncode != 0:
            stderr_tail = "\n".join(completed.stderr.splitlines()[-80:])
            raise RuntimeError(
                f"llama.cpp generation exited {completed.returncode}:\n"
                f"{stderr_tail}")
        if not generated_path.is_file():
            raise RuntimeError("llama.cpp did not create the generation output")
        generated = generated_path.read_text(encoding="utf-8")
        if not generated.strip():
            raise RuntimeError("llama.cpp completed without generated text")
        if "\ufffd" in generated:
            raise RuntimeError("llama.cpp generation contains replacement characters")

    offload = parse_partial_cuda_offload(
        completed.stderr,
        requested_layers=args.gpu_layers,
        expected_total_layers=args.expected_total_layers,
        device=args.device,
    )
    return {
        "schema": 1,
        "status": "pass",
        "scope": (
            "partial CUDA offload and short deterministic generation of the "
            "already-existing validated Qwen3.6-35B-A3B production GGUF; no "
            "GGUF construction or mutation is performed and no release-quality "
            "generation claim is made"
        ),
        "model": {
            "path": str(model),
            "bytes": model.stat().st_size,
            "sha256": model_sha256,
        },
        "llama_cpp": {
            "path": str(llama_cpp),
            "revision": revision,
            "unmodified_pinned_revision": True,
            "executable": str(executable),
            "executable_sha256": _sha256_file(executable),
        },
        "device_enumeration": {
            "command": device_command,
            "exit_status": device_result.returncode,
            "stdout": device_result.stdout,
            "stderr": device_result.stderr,
        },
        "generation": {
            "command": command,
            "exit_status": completed.returncode,
            "elapsed_seconds": run_seconds,
            "generated_text": generated,
            "generated_text_sha256": hashlib.sha256(
                generated.encode("utf-8")).hexdigest(),
            "stdout_sha256": hashlib.sha256(
                completed.stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(
                completed.stderr.encode("utf-8")).hexdigest(),
        },
        "partial_cuda_offload": offload,
        "partial_cuda_offload_pass": True,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024),
        "peak_child_rss_bytes": (
            int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * 1024),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--llama-executable", type=Path, required=True)
    parser.add_argument("--temporary-parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama-cpp-revision", default=LLAMA_CPP_REVISION)
    parser.add_argument("--device", default="CUDA0")
    parser.add_argument("--gpu-layers", type=int, default=4)
    parser.add_argument("--expected-total-layers", type=int, default=41)
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--predict-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--prompt", default="Reply with exactly one word: Hello.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gpu_layers <= 0:
        raise ValueError("--gpu-layers must be positive")
    report = audit(args)
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "offloaded_layers": report["partial_cuda_offload"]["offloaded_layers"],
        "cuda_model_buffer_mib": report[
            "partial_cuda_offload"]["cuda_model_buffer_mib"],
        "cuda_compute_buffer_mib": report[
            "partial_cuda_offload"]["cuda_compute_buffer_mib"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
