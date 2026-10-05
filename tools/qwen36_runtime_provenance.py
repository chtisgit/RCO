"""Stable implementation provenance for the Qwen 3.6 runtime gates."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence


COMMON_RUNTIME_IMPLEMENTATION_FILES = (
    "tools/qwen36_runtime_provenance.py",
    "src/checkpoint_stream.py",
    "src/gguf_checkpoint_stream.py",
    "src/gsq_upgrade_runtime.py",
    "src/native_gguf.py",
    "src/native_runtime.py",
    "src/native_store.py",
    "src/quant/ggml_native.py",
    "src/search/streaming.py",
)
AUDIT_RUNTIME_IMPLEMENTATION_FILES = (
    "tools/audit_qwen36_gsq_upgrade_runtime.py",
    *COMMON_RUNTIME_IMPLEMENTATION_FILES,
)
SEARCH_RUNTIME_IMPLEMENTATION_FILES = (
    "tools/search_qwen36_gsq_rco.py",
    "src/search/hard.py",
    *COMMON_RUNTIME_IMPLEMENTATION_FILES,
)
REQUIRED_RUNTIME_CONTROLS = ("authentic_retain", "all_admitted_upgrades")


def runtime_report_status(results: dict[str, Any]) -> str:
    if set(results) - set(REQUIRED_RUNTIME_CONTROLS):
        return "failed"
    retain = results.get("authentic_retain")
    if retain is not None:
        if not isinstance(retain, dict):
            return "failed"
        if retain.get("exact_mean_nll_match") is not True:
            return "failed"
        if retain.get("exact_logit_match") is not True:
            return "failed"
        comparison = retain.get("logit_comparison_to_authentic")
        if not isinstance(comparison, dict) or comparison.get("passed") is not True:
            return "failed"
    upgrades = results.get("all_admitted_upgrades")
    if upgrades is not None:
        if not isinstance(upgrades, dict):
            return "failed"
        comparison = upgrades.get("logit_comparison_to_authentic")
        if not isinstance(comparison, dict) or comparison.get("passed") is not True:
            return "failed"
    if all(control in results for control in REQUIRED_RUNTIME_CONTROLS):
        return "complete"
    return "partial"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _gguf_python_hashes(gguf_python: Path) -> dict[str, str]:
    package = gguf_python / "gguf"
    if not package.is_dir():
        raise ValueError(f"not a gguf-py directory: {gguf_python}")
    files = sorted(path for path in package.rglob("*.py") if path.is_file())
    if not files:
        raise ValueError(f"gguf-py package has no implementation files: {package}")
    return {
        path.relative_to(gguf_python).as_posix(): sha256_file(path)
        for path in files
    }


def _runtime_environment() -> dict[str, Any]:
    packages = {}
    for name in ("accelerate", "numpy", "safetensors", "torch", "transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "packages": packages,
    }


def runtime_provenance(
    *, repository: Path, gguf_python: Path, ggml_library: Path,
    implementation_files: Sequence[str],
) -> dict[str, Any]:
    """Return deterministic hashes for code and native/parser dependencies."""
    repository = repository.resolve(strict=True)
    gguf_python = gguf_python.resolve(strict=True)
    ggml_library = ggml_library.resolve(strict=True)
    implementation_hashes = {
        relative: sha256_file(repository / relative)
        for relative in implementation_files
    }
    implementation_revision = subprocess.run(
        [
            "git", "log", "-1", "--format=%H", "--",
            *implementation_files,
        ], cwd=repository, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    if not implementation_revision:
        raise RuntimeError("implementation files have no committed revision")
    implementation_sha256 = _json_sha256(implementation_hashes)
    gguf_files = _gguf_python_hashes(gguf_python)
    return {
        "environment": _runtime_environment(),
        "ggml_shared_library_sha256": sha256_file(ggml_library),
        "gguf_python_files": gguf_files,
        "gguf_python_sha256": _json_sha256(gguf_files),
        "rco_implementation_revision": implementation_revision,
        "rco_implementation_files": implementation_hashes,
        "rco_implementation_sha256": implementation_sha256,
    }
