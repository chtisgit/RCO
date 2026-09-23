"""Fail-closed checks linking a dense Qwen base, GSQ release, and GGUF."""

from __future__ import annotations

import copy
import re
from typing import Any


_NON_SEMANTIC_CONFIG_KEYS = {"quantization_config", "transformers_version"}


def normalized_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove only release/quantizer metadata from an otherwise exact config."""
    result = copy.deepcopy(config)
    for key in _NON_SEMANTIC_CONFIG_KEYS:
        result.pop(key, None)
    return result


def declared_base_model(readme: str) -> tuple[str, str]:
    """Read the single base-model declaration from model-card front matter."""
    if not readme.startswith("---\n"):
        raise ValueError("GSQ README has no YAML front matter")
    try:
        front_matter = readme.split("---\n", 2)[1]
    except IndexError as error:
        raise ValueError("GSQ README has unterminated YAML front matter") from error
    values: dict[str, str] = {}
    for key in ("base_model", "base_model_relation"):
        matches = re.findall(rf"(?m)^{key}:\s*([^#\n]+?)\s*$", front_matter)
        if len(matches) != 1:
            raise ValueError(f"GSQ README must declare exactly one {key}")
        values[key] = matches[0].strip(" '\"")
    return values["base_model"], values["base_model_relation"]


def audit_base_gsq_relationship(
    *,
    base_identity: dict[str, Any],
    base_manifest: dict[str, Any],
    base_config: dict[str, Any],
    gsq_config: dict[str, Any],
    gsq_readme: str,
    shared_asset_hashes: dict[str, tuple[str, str]],
    gguf_tensors: dict[str, list[int]],
    gguf_metadata: dict[str, Any],
    gsq_revision: str,
) -> dict[str, Any]:
    """Prove lineage, structural identity, and exact canonical inventory."""
    expected_repo = base_identity["repo_id"]
    declared_repo, relation = declared_base_model(gsq_readme)
    if declared_repo != expected_repo:
        raise ValueError(
            f"GSQ declares base {declared_repo}, dense identity is {expected_repo}")
    if relation != "quantized":
        raise ValueError(f"GSQ base_model_relation is {relation}, expected quantized")

    if normalized_model_config(base_config) != normalized_model_config(gsq_config):
        raise ValueError("dense and GSQ model configs differ beyond quantization metadata")

    mismatched_assets = sorted(
        name for name, (base_hash, gsq_hash) in shared_asset_hashes.items()
        if base_hash != gsq_hash
    )
    if mismatched_assets:
        raise ValueError(f"dense and GSQ text assets differ: {mismatched_assets}")

    if base_identity.get("status") != "pass":
        raise ValueError("dense checkpoint identity audit did not pass")
    if base_manifest.get("status") != "pass":
        raise ValueError("dense canonical GGUF manifest did not pass")
    if base_manifest["source"] != {
        "repo_id": expected_repo,
        "revision": base_identity["revision"],
    }:
        raise ValueError("dense manifest source does not match dense identity")

    manifest_tensors = {
        entry["destination_name"]: [int(value) for value in entry["destination_gguf_shape"]]
        for entry in base_manifest["entries"]
    }
    if len(manifest_tensors) != len(base_manifest["entries"]):
        raise ValueError("dense manifest contains duplicate canonical destinations")
    missing = sorted(set(manifest_tensors) - set(gguf_tensors))
    extra = sorted(set(gguf_tensors) - set(manifest_tensors))
    shape_mismatches = sorted(
        name for name in set(manifest_tensors).intersection(gguf_tensors)
        if manifest_tensors[name] != [int(value) for value in gguf_tensors[name]]
    )
    if missing or extra or shape_mismatches:
        raise ValueError(
            "dense/GSQ GGUF inventory differs: "
            f"{len(missing)} missing, {len(extra)} extra, "
            f"{len(shape_mismatches)} shape mismatches")

    text_config = base_config["text_config"]
    expected_metadata = {
        "general.architecture": "qwen35moe",
        "qwen35moe.block_count": text_config["num_hidden_layers"],
        "qwen35moe.embedding_length": text_config["hidden_size"],
        "qwen35moe.expert_count": text_config["num_experts"],
        "qwen35moe.expert_used_count": text_config["num_experts_per_tok"],
        "qwen35moe.expert_feed_forward_length": text_config["moe_intermediate_size"],
    }
    wrong_metadata = {
        key: {"expected": value, "actual": gguf_metadata.get(key)}
        for key, value in expected_metadata.items()
        if gguf_metadata.get(key) != value
    }
    if wrong_metadata:
        raise ValueError(f"GSQ GGUF metadata differs from dense config: {wrong_metadata}")

    source_views = sum(
        entry.get("source_view") is not None for entry in base_manifest["entries"])
    return {
        "schema": 1,
        "status": "pass",
        "conclusion": (
            "The pinned BF16 checkpoint is the base model declared by the GSQ "
            "release and has the same text architecture and canonical GGUF "
            "inventory. This establishes source compatibility and lineage, not "
            "numerical equality between dense and quantized weights."
        ),
        "dense_base": {
            "repo_id": expected_repo,
            "revision": base_identity["revision"],
            "text_source_tensors": base_manifest["source_text_tensor_count"],
            "canonical_tensors": len(manifest_tensors),
        },
        "gsq_release": {
            "revision": gsq_revision,
            "declared_base_model": declared_repo,
            "declared_relation": relation,
        },
        "config": {
            "normalized_exact_match": True,
            "ignored_top_level_keys": sorted(_NON_SEMANTIC_CONFIG_KEYS),
        },
        "text_assets": {
            "exact_hash_match": True,
            "files": {
                name: hashes[0] for name, hashes in sorted(shared_asset_hashes.items())
            },
        },
        "canonical_inventory": {
            "exact_name_and_shape_match": True,
            "tensor_count": len(manifest_tensors),
            "fused_gate_up_source_views": source_views,
            "missing": [],
            "extra": [],
            "shape_mismatches": [],
        },
        "gguf_metadata": expected_metadata,
        "claim_limits": {
            "dense_and_quantized_values_equal": False,
            "gsq_is_dense_candidate_source": False,
            "bf16_is_higher_precision_candidate_source": True,
        },
    }
