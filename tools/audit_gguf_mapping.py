#!/usr/bin/env python3
"""Build a source-to-canonical-GGUF manifest using pinned llama.cpp."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gguf_manifest import build_gguf_manifest, parse_converter_dry_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--llama-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llama_cpp = args.llama_cpp.resolve(strict=True)
    actual_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=llama_cpp,
        text=True, check=True, capture_output=True,
    ).stdout.strip()
    if actual_revision != args.llama_revision:
        raise ValueError(
            f"llama.cpp is {actual_revision}, expected {args.llama_revision}")

    with tempfile.TemporaryDirectory(prefix="rco-gguf-map-") as directory:
        output_path = Path(directory) / "dry-run.gguf"
        command = [
            sys.executable,
            str(llama_cpp / "convert_hf_to_gguf.py"),
            str(args.model_dir.resolve(strict=True)),
            "--outfile", str(output_path),
            "--outtype", "bf16",
            "--no-mtp",
            "--dry-run",
        ]
        completed = subprocess.run(
            command, cwd=llama_cpp, text=True, capture_output=True, check=True,
        )
        dry_run_output = completed.stdout + "\n" + completed.stderr
        converter_records = parse_converter_dry_run(dry_run_output)
        if output_path.exists():
            raise RuntimeError("llama.cpp dry run unexpectedly wrote a GGUF")

    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    import gguf

    identity = json.loads(args.identity_report.read_text())
    block_count = int(identity["config"]["num_hidden_layers"])
    tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35, block_count)
    manifest = build_gguf_manifest(
        identity,
        converter_records,
        lambda name: tensor_map.get_name(
            key=name, try_suffixes=(".weight", ".bias")),
        llama_cpp_revision=actual_revision,
    )
    reported_command = list(command)
    reported_command[reported_command.index(str(output_path))] = "<temporary>/dry-run.gguf"
    manifest["converter_dry_run"] = {
        "command": reported_command,
        "return_code": completed.returncode,
        "wrote_output_file": False,
        "reported_tensor_count": len(converter_records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "canonical_tensor_count": manifest["canonical_tensor_count"],
        "decision_group_count": manifest["decision_group_count"],
        "copied_tensor_count": manifest["copied_tensor_count"],
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
