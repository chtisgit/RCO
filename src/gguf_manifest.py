"""Canonical GGUF mapping and RCO decision-group manifest helpers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable, Iterable


_CONVERTER_TENSOR = re.compile(
    r"^INFO:hf-to-gguf:(?P<name>[^,]+),\s+"
    r"torch\.(?P<source_dtype>\S+)\s+-->\s+"
    r"(?P<ggml_type>\S+),\s+shape\s+=\s+\{(?P<shape>[^}]*)\}$"
)


def canonical_source_name(source_name: str) -> str:
    """Apply the name-only normalization used by the pinned converter."""
    name = source_name.replace("language_model.", "")
    if name.endswith(".mlp.experts.down_proj"):
        name += ".weight"
    if name.endswith(".linear_attn.dt_bias"):
        name = name.removesuffix(".dt_bias") + ".dt_proj.bias"
    return name


def converter_transforms(source_name: str) -> list[str]:
    """Describe semantic converter transforms relevant to candidate bytes."""
    transforms = ["strip_language_model_namespace"]
    if ".linear_attn." in source_name:
        if source_name.endswith(".A_log"):
            transforms.extend(["reorder_value_heads", "negative_exponential"])
        elif source_name.endswith(".dt_bias"):
            transforms.extend(["reorder_value_heads", "rename_dt_bias"])
        elif source_name.endswith(".conv1d.weight"):
            transforms.extend(["reorder_value_heads", "squeeze_conv1d"])
        elif source_name.endswith((
            ".in_proj_qkv.weight",
            ".in_proj_z.weight",
            ".in_proj_a.weight",
            ".in_proj_b.weight",
            ".out_proj.weight",
        )):
            transforms.append("reorder_value_heads")
    if source_name.endswith("norm.weight") and not source_name.endswith(
        "linear_attn.norm.weight"
    ):
        transforms.append("add_one_to_norm")
    return transforms


def parse_converter_dry_run(output: str) -> dict[str, dict[str, Any]]:
    """Parse final canonical tensor records from llama.cpp dry-run logs."""
    records: dict[str, dict[str, Any]] = {}
    for line in output.splitlines():
        match = _CONVERTER_TENSOR.match(line.strip())
        if match is None:
            continue
        name = match.group("name").strip()
        shape = [
            int(value.strip())
            for value in match.group("shape").split(",")
            if value.strip()
        ]
        if name in records:
            raise ValueError(f"converter emitted duplicate GGUF tensor {name}")
        records[name] = {
            "source_dtype": match.group("source_dtype"),
            "ggml_type": match.group("ggml_type"),
            "gguf_shape": shape,
        }
    if not records:
        raise ValueError("converter dry run contained no canonical tensors")
    return records


def candidate_eligibility(record: dict[str, Any]) -> tuple[bool, str | None]:
    """Select initial Q2_0/Q4_0 matrix groups from final GGUF geometry."""
    shape = record["destination_gguf_shape"]
    name = record["destination_name"]
    if not name.endswith(".weight"):
        return False, "not_a_weight_tensor"
    if len(shape) < 2:
        return False, "not_a_matrix"
    if shape[0] % 64:
        return False, "row_width_not_q2_0_aligned"
    return True, None


def build_gguf_manifest(
    identity_report: dict[str, Any],
    converter_records: dict[str, dict[str, Any]],
    map_name: Callable[[str], str | None],
    *,
    llama_cpp_revision: str,
) -> dict[str, Any]:
    """Map every text tensor and require exact agreement with dry-run output."""
    entries = []
    destinations = set()
    mapped_sources = set()
    for source in identity_report["text_inventory"]:
        normalized = canonical_source_name(source["name"])
        expansions: list[tuple[str, dict[str, int] | None, list[int], list[str]]]
        if normalized.endswith(".mlp.experts.gate_up_proj"):
            shape = [int(value) for value in source["shape"]]
            if len(shape) != 3 or shape[1] % 2:
                raise ValueError(
                    f"invalid fused gate/up shape for {source['name']}: {shape}")
            intermediate = shape[1] // 2
            expansions = []
            for projection, start, stop in (
                ("gate_proj", 0, intermediate),
                ("up_proj", intermediate, shape[1]),
            ):
                virtual_name = normalized.removesuffix(
                    "gate_up_proj") + projection + ".weight"
                expansions.append((
                    virtual_name,
                    {"axis": 1, "start": start, "stop": stop},
                    [shape[0], intermediate, shape[2]],
                    ["split_fused_gate_up"],
                ))
        else:
            expansions = [(normalized, None, list(source["shape"]), [])]

        for mapped_name, source_view, candidate_shape, extra_transforms in expansions:
            destination = map_name(mapped_name)
            if destination is None:
                raise ValueError(
                    f"cannot map source tensor {source['name']} as {mapped_name}")
            if destination in destinations:
                raise ValueError(f"multiple sources map to GGUF tensor {destination}")
            destinations.add(destination)
            mapped_sources.add(source["name"])
            try:
                converter = converter_records[destination]
            except KeyError as error:
                raise ValueError(
                    f"mapped GGUF tensor {destination} was not emitted by converter"
                ) from error
            entry = {
                "source_name": source["name"],
                "source_shard": source["shard"],
                "source_category": source["category"],
                "source_dtype": source["dtype"],
                "source_shape": source["shape"],
                "source_view": source_view,
                "candidate_source_shape": candidate_shape,
                "normalized_source_name": mapped_name,
                "destination_name": destination,
                "destination_ggml_type": converter["ggml_type"],
                "destination_gguf_shape": converter["gguf_shape"],
                "converter_transforms": (
                    converter_transforms(source["name"]) + extra_transforms),
            }
            eligible, reason = candidate_eligibility(entry)
            entry["rco_search"] = eligible
            entry["decision_group"] = destination if eligible else None
            entry["candidate_types"] = ["Q2_0", "Q4_0"] if eligible else []
            entry["copy_reason"] = reason
            entries.append(entry)

    converter_destinations = set(converter_records)
    missing_sources = sorted(converter_destinations - destinations)
    missing_outputs = sorted(destinations - converter_destinations)
    if missing_sources or missing_outputs:
        raise ValueError(
            f"source/converter coverage mismatch: {len(missing_sources)} outputs "
            f"without sources, {len(missing_outputs)} sources without outputs"
        )
    expected_sources = {item["name"] for item in identity_report["text_inventory"]}
    missing_source_tensors = sorted(expected_sources - mapped_sources)
    if missing_source_tensors:
        raise ValueError(
            f"text sources were not mapped: {missing_source_tensors[:8]}")
    searched = [entry for entry in entries if entry["rco_search"]]
    copied = [entry for entry in entries if not entry["rco_search"]]
    omitted = (
        identity_report["category_counts"].get("vision", 0)
        + identity_report["category_counts"].get("mtp", 0)
    )
    return {
        "schema": 1,
        "status": "pass",
        "source": {
            "repo_id": identity_report["repo_id"],
            "revision": identity_report["revision"],
        },
        "llama_cpp_revision": llama_cpp_revision,
        "architecture": "qwen35",
        "candidate_types": ["Q2_0", "Q4_0"],
        "source_text_tensor_count": len(expected_sources),
        "canonical_tensor_count": len(converter_records),
        "searched_tensor_count": len(searched),
        "copied_tensor_count": len(copied),
        "intentionally_omitted_tensor_count": omitted,
        "decision_group_count": len(searched),
        "unique_destination_count": len(destinations),
        "coverage": {
            "mapped": len(entries),
            "mapped_source_tensors": len(mapped_sources),
            "mapped_canonical_tensors": len(entries),
            "unmapped": 0,
            "duplicate_destinations": 0,
            "converter_outputs_without_sources": 0,
            "sources_without_converter_outputs": 0,
        },
        "source_category_counts": dict(sorted(Counter(
            source["category"] for source in identity_report["text_inventory"]
        ).items())),
        "copy_reason_counts": dict(sorted(Counter(
            entry["copy_reason"] for entry in copied).items())),
        "entries": entries,
    }
