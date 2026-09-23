#!/usr/bin/env python3
"""Audit the pinned 35B BF16 base against the GSQ release and proven GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoint_relationship import audit_base_gsq_relationship


_TEXT_ASSETS = (
    "chat_template.jinja",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-identity", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--gsq-dir", type=Path, required=True)
    parser.add_argument("--gsq-audit", type=Path, required=True)
    parser.add_argument("--gsq-gguf", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_field(reader, key: str):
    field = reader.fields.get(key)
    if field is None:
        return None
    value = field.contents()
    return value.item() if hasattr(value, "item") else value


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve(strict=True)
    gsq_dir = args.gsq_dir.resolve(strict=True)
    gguf_path = args.gsq_gguf.resolve(strict=True)
    identity = json.loads(args.base_identity.read_text())
    manifest = json.loads(args.base_manifest.read_text())
    gsq_audit = json.loads(args.gsq_audit.read_text())
    revisions = gsq_audit["source_revision_evidence"]
    if len(revisions) != 1:
        raise ValueError(f"GSQ source audit has ambiguous revisions: {revisions}")
    gsq_revision = next(iter(revisions))

    base_record_hashes = {item["path"]: item["sha256"] for item in identity["files"]}
    shared_assets: dict[str, tuple[str, str]] = {}
    for name in _TEXT_ASSETS:
        if name not in base_record_hashes:
            raise ValueError(f"dense identity did not record required text asset {name}")
        gsq_path = gsq_dir / name
        if not gsq_path.is_file():
            raise ValueError(f"GSQ release is missing required text asset {name}")
        gsq_hash = sha256(gsq_path)
        audited_record = gsq_audit["source_hashes"].get(name)
        if not isinstance(audited_record, dict):
            raise ValueError(f"GSQ source audit has no structured record for {name}")
        if gsq_path.stat().st_size != audited_record.get("bytes"):
            raise ValueError(f"GSQ {name} size differs from its prior source audit")
        audited_hash = audited_record.get("sha256")
        if gsq_hash != audited_hash:
            raise ValueError(f"GSQ {name} differs from its prior source audit")
        shared_assets[name] = (base_record_hashes[name], gsq_hash)

    sys.path.insert(0, str(args.llama_cpp.resolve(strict=True) / "gguf-py"))
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path, "r")
    tensors = {
        tensor.name: [int(value) for value in tensor.shape]
        for tensor in reader.tensors
    }
    metadata_keys = (
        "general.architecture",
        "qwen35moe.block_count",
        "qwen35moe.embedding_length",
        "qwen35moe.expert_count",
        "qwen35moe.expert_used_count",
        "qwen35moe.expert_feed_forward_length",
    )
    report = audit_base_gsq_relationship(
        base_identity=identity,
        base_manifest=manifest,
        base_config=json.loads((base_dir / "config.json").read_text()),
        gsq_config=json.loads((gsq_dir / "config.json").read_text()),
        gsq_readme=(gsq_dir / "README.md").read_text(),
        shared_asset_hashes=shared_assets,
        gguf_tensors=tensors,
        gguf_metadata={key: scalar_field(reader, key) for key in metadata_keys},
        gsq_revision=gsq_revision,
    )
    report["gsq_release"]["audit_path"] = str(args.gsq_audit)
    report["canonical_inventory"]["gguf_path"] = str(gguf_path)
    report["canonical_inventory"]["gguf_sha256"] = sha256(gguf_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "tensor_count": report["canonical_inventory"]["tensor_count"],
        "dense_revision": report["dense_base"]["revision"],
        "gsq_revision": report["gsq_release"]["revision"],
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
