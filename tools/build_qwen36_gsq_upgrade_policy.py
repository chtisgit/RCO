#!/usr/bin/env python3
"""Build the exact retain/upgrade policy and byte budgets for GSQ-RCO."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROUTED_POLICY_CONTROLS = (
    "routed_gate_q4",
    "routed_up_q4",
    "routed_down_q4",
)


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


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _maximum_reachable(costs: list[int], cap: int, unit: int) -> int:
    """Return the largest bounded-subset sum no greater than cap."""
    if cap < 0 or unit < 1 or cap % unit:
        raise ValueError("cap and costs must use the declared positive unit")
    counts = Counter(costs)
    if any(cost <= 0 or cost % unit for cost in counts):
        raise ValueError("upgrade costs must be positive unit multiples")
    cap_units = cap // unit
    reachable = 1
    mask = (1 << (cap_units + 1)) - 1
    for cost, count in sorted(counts.items()):
        cost_units = cost // unit
        batch = 1
        remaining = count
        while remaining:
            take = min(batch, remaining)
            reachable |= reachable << (cost_units * take)
            reachable &= mask
            remaining -= take
            batch <<= 1
    return (reachable.bit_length() - 1) * unit


def build(args: argparse.Namespace) -> dict[str, Any]:
    gguf_python = args.gguf_python.resolve(strict=True)
    import sys

    sys.path.insert(0, str(gguf_python))
    from gguf import GGUFReader  # noqa: PLC0415

    model_dir = args.model_dir.resolve(strict=True)
    identity_path = args.identity.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    control_path = args.controls.resolve(strict=True)
    logit_path = args.logits.resolve(strict=True)
    routed_control_path = args.routed_controls.resolve(strict=True)
    routed_logit_path = args.routed_logits.resolve(strict=True)
    routed_interaction_control_path = (
        args.routed_interaction_controls.resolve(strict=True))
    routed_interaction_logit_path = (
        args.routed_interaction_logits.resolve(strict=True))
    gguf_path = args.gguf.resolve(strict=True)
    store_path = args.store.resolve(strict=True)
    index_path = store_path / "native-candidate-index.json"
    output_path = args.output.resolve()

    identity = _load_json(identity_path)
    manifest = _load_json(manifest_path)
    controls = _load_json(control_path)
    logits = _load_json(logit_path)
    routed_controls = _load_json(routed_control_path)
    routed_logits = _load_json(routed_logit_path)
    routed_interaction_controls = _load_json(routed_interaction_control_path)
    routed_interaction_logits = _load_json(routed_interaction_logit_path)
    index = _load_json(index_path)
    if identity.get("status") != "pass":
        raise RuntimeError("dense identity is not passing")
    if controls.get("status") != "complete":
        raise RuntimeError("upgrade controls are incomplete")
    if logits.get("status") != "complete":
        raise RuntimeError("upgrade logit controls are incomplete")
    if routed_controls.get("status") != "complete":
        raise RuntimeError("routed upgrade controls are incomplete")
    if routed_logits.get("status") != "complete":
        raise RuntimeError("routed upgrade logit controls are incomplete")
    if routed_interaction_controls.get("status") != "complete":
        raise RuntimeError("routed interaction controls are incomplete")
    if routed_interaction_logits.get("status") != "complete":
        raise RuntimeError("routed interaction logit controls are incomplete")
    interaction_label = "routed_gate_up_q4"
    if not (
        routed_interaction_controls["results"][interaction_label][
            "bounded_by_plus_0_5_nll"]
        and routed_interaction_logits["results"][interaction_label]["passed"]
    ):
        raise RuntimeError("routed gate+up interaction did not pass")

    pinned = controls["problem"]
    actual_hashes = {
        "identity_sha256": _sha256_file(identity_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "gguf_sha256": _sha256_file(gguf_path),
        "candidate_store_index_sha256": _sha256_file(index_path),
    }
    for key, value in actual_hashes.items():
        if pinned.get(key) != value:
            raise RuntimeError(f"control report {key} differs from current input")

    reader = GGUFReader(gguf_path)
    alignment = int(reader.alignment)
    if alignment != int(index["alignment"]):
        raise RuntimeError("GGUF and candidate-store alignments differ")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    entries = {entry["destination_name"]: entry for entry in manifest["entries"]}
    if set(tensors) != set(entries):
        raise RuntimeError("manifest and GGUF tensor inventories differ")
    if len(tensors) != 733:
        raise RuntimeError("unexpected authentic GGUF tensor count")

    shard_hashes = {
        record["path"]: record["sha256"] for record in identity["files"]
    }
    family_control = {
        "lm_head": "output_head_bf16",
        "linear_attention": "linear_attention_bf16",
        "self_attention": "self_attention_bf16",
        "shared_expert": "shared_expert_bf16",
    }
    policies: list[dict[str, Any]] = []
    candidate_upgrade_costs: list[int] = []
    admitted_upgrade_costs: list[int] = []
    family_budget: dict[str, dict[str, int]] = {}
    searchable_type_counts: Counter[str] = Counter()

    for tensor in reader.tensors:
        entry = entries[tensor.name]
        original_payload = int(tensor.n_bytes)
        original_aligned = _align(original_payload, alignment)
        policy: dict[str, Any] = {
            "destination_name": tensor.name,
            "source_category": entry["source_category"],
            "searchable": bool(entry.get("rco_search")),
            "retain": {
                "kind": "authentic_gsq_gguf_slice",
                "ggml_type": tensor.tensor_type.name,
                "payload_offset": int(tensor.data_offset),
                "payload_bytes": original_payload,
                "aligned_gguf_bytes": original_aligned,
                "container_sha256": actual_hashes["gguf_sha256"],
            },
            "upgrade": None,
        }
        if entry.get("rco_search"):
            searchable_type_counts[tensor.tensor_type.name] += 1
            if tensor.tensor_type.name == "Q2_0":
                if entry["source_category"] != "routed_expert":
                    raise RuntimeError("non-routed Q2 tensor entered upgrade policy")
                if ".ffn_gate_exps." in tensor.name:
                    control = "routed_gate_q4"
                elif ".ffn_up_exps." in tensor.name:
                    control = "routed_up_q4"
                elif ".ffn_down_exps." in tensor.name:
                    control = "routed_down_q4"
                else:
                    raise RuntimeError(
                        f"unknown routed projection in {tensor.name}")
                control_results = routed_controls
                logit_results = routed_logits
                candidate = index["tensors"][tensor.name]["Q4_0"]
                upgrade = {
                    "kind": "bf16_derived_native_candidate",
                    "ggml_type": "Q4_0",
                    "control": control,
                    "path": candidate["path"],
                    "payload_bytes": int(candidate["payload_bytes"]),
                    "aligned_gguf_bytes": int(candidate["aligned_gguf_bytes"]),
                    "sha256": candidate["sha256"],
                    "provenance": candidate["provenance"],
                }
            elif tensor.tensor_type.name == "Q8_0":
                control = family_control.get(entry["source_category"])
                if control is None:
                    raise RuntimeError(
                        f"Q8 family has no control: {entry['source_category']}")
                control_results = controls
                logit_results = logits
                source_shard = entry["source_shard"]
                if source_shard not in shard_hashes:
                    raise RuntimeError(f"source shard is not identity-pinned: {source_shard}")
                payload_bytes = int(tensor.n_elements) * 2
                upgrade = {
                    "kind": "pinned_bf16_source",
                    "ggml_type": "BF16",
                    "control": control,
                    "payload_bytes": payload_bytes,
                    "aligned_gguf_bytes": _align(payload_bytes, alignment),
                    "dense_revision": identity["revision"],
                    "source_tensor": entry["source_name"],
                    "source_shard": source_shard,
                    "source_shard_sha256": shard_hashes[source_shard],
                    "converter_transforms": entry["converter_transforms"],
                    "materialization": (
                        "stream the pinned BF16 source through the canonical "
                        "GGUF transform; no decoded-GSQ requantization"),
                }
            elif tensor.tensor_type.name in {"BF16", "F32"}:
                upgrade = None
            else:
                raise RuntimeError(
                    f"unsupported searchable incumbent type: {tensor.tensor_type.name}")
            if upgrade is not None:
                nll_pass = control_results["results"][control][
                    "bounded_by_plus_0_5_nll"]
                logit_pass = logit_results["results"][control]["passed"]
                admitted = nll_pass and logit_pass
                upgrade["nll_screen_passed"] = nll_pass
                upgrade["logit_screen_passed"] = logit_pass
                upgrade["admitted_on_fixed_screen"] = admitted
                upgrade["policy_status"] = (
                    "admitted" if admitted
                    else "quarantined_pending_finer_family_controls")
                delta = upgrade["aligned_gguf_bytes"] - original_aligned
                if delta <= 0:
                    raise RuntimeError(f"upgrade does not increase precision bytes: {tensor.name}")
                upgrade["incremental_gguf_bytes"] = delta
                policy["upgrade"] = upgrade
                candidate_upgrade_costs.append(delta)
                if admitted:
                    admitted_upgrade_costs.append(delta)
                budget = family_budget.setdefault(control, {
                    "tensor_count": 0,
                    "original_aligned_gguf_bytes": 0,
                    "upgrade_aligned_gguf_bytes": 0,
                    "incremental_gguf_bytes": 0,
                })
                budget["tensor_count"] += 1
                budget["original_aligned_gguf_bytes"] += original_aligned
                budget["upgrade_aligned_gguf_bytes"] += upgrade[
                    "aligned_gguf_bytes"]
                budget["incremental_gguf_bytes"] += delta
        policies.append(policy)

    expected_types = Counter({"Q2_0": 120, "Q8_0": 251, "F32": 80, "BF16": 61})
    if searchable_type_counts != expected_types:
        raise RuntimeError(f"searchable type inventory differs: {searchable_type_counts}")
    if len(candidate_upgrade_costs) != 371:
        raise RuntimeError("candidate upgrade inventory differs")
    if len(admitted_upgrade_costs) != 331:
        raise RuntimeError("fixed-screen admitted upgrade inventory differs")
    for label, budget in family_budget.items():
        control_results = (
            routed_controls if label in ROUTED_POLICY_CONTROLS else controls)
        logit_results = (
            routed_logits if label in ROUTED_POLICY_CONTROLS else logits)
        budget["nll_screen_passed"] = control_results["results"][label][
            "bounded_by_plus_0_5_nll"]
        budget["logit_screen_passed"] = logit_results["results"][label]["passed"]
        budget["admitted_on_fixed_screen"] = (
            budget["nll_screen_passed"] and budget["logit_screen_passed"])

    tensor_region_bytes = sum(
        item["retain"]["aligned_gguf_bytes"] for item in policies)
    authentic_file_bytes = gguf_path.stat().st_size
    if int(reader.data_offset) + tensor_region_bytes != authentic_file_bytes:
        raise RuntimeError("authentic file size does not match aligned tensor accounting")
    legacy_cap = int(args.comparison_file_cap_bytes)
    allowance = legacy_cap - authentic_file_bytes
    if allowance < 0:
        raise RuntimeError("comparison file cap is below the authentic GSQ floor")
    reachable_allowance = _maximum_reachable(
        admitted_upgrade_costs, allowance, alignment)
    maximum_candidate_upgrade_bytes = sum(candidate_upgrade_costs)
    maximum_admitted_upgrade_bytes = sum(admitted_upgrade_costs)
    report = {
        "schema": "rco.qwen36.gsq_upgrade_policy.v1",
        "status": "pass",
        "scope": (
            "exact per-GGUF-tensor retain-or-upgrade policy; uniform Q2 is "
            "excluded and this artifact is not a final assignment"),
        "inputs": {
            "authentic_gsq_gguf": str(gguf_path),
            "authentic_gsq_gguf_sha256": actual_hashes["gguf_sha256"],
            "dense_revision": identity["revision"],
            "identity_sha256": actual_hashes["identity_sha256"],
            "manifest_sha256": actual_hashes["manifest_sha256"],
            "upgrade_controls_sha256": _sha256_file(control_path),
            "upgrade_logits_sha256": _sha256_file(logit_path),
            "routed_upgrade_controls_sha256": _sha256_file(routed_control_path),
            "routed_upgrade_logits_sha256": _sha256_file(routed_logit_path),
            "routed_interaction_controls_sha256": _sha256_file(
                routed_interaction_control_path),
            "routed_interaction_logits_sha256": _sha256_file(
                routed_interaction_logit_path),
            "candidate_store_index_sha256": actual_hashes[
                "candidate_store_index_sha256"],
        },
        "inventory": {
            "tensor_count": len(policies),
            "always_retain_only_count": len(policies) - len(candidate_upgrade_costs),
            "candidate_upgrade_count": len(candidate_upgrade_costs),
            "admitted_upgrade_count": len(admitted_upgrade_costs),
            "quarantined_upgrade_count": (
                len(candidate_upgrade_costs) - len(admitted_upgrade_costs)),
            "searchable_type_counts": dict(sorted(searchable_type_counts.items())),
            "candidate_upgrade_type_counts": {
                "Q2_0_to_Q4_0": 120, "Q8_0_to_BF16": 251},
            "admitted_upgrade_type_counts": {
                "Q2_0_to_Q4_0": 80, "Q8_0_to_BF16": 251},
            "quarantined_upgrade_type_counts": {"Q2_0_to_Q4_0": 40},
        },
        "budget": {
            "alignment": alignment,
            "metadata_and_padding_prefix_bytes": int(reader.data_offset),
            "authentic_tensor_region_bytes": tensor_region_bytes,
            "mandatory_floor_file_bytes": authentic_file_bytes,
            "all_candidate_upgrades_incremental_bytes": maximum_candidate_upgrade_bytes,
            "all_candidate_upgrades_file_bytes": (
                authentic_file_bytes + maximum_candidate_upgrade_bytes),
            "all_admitted_upgrades_incremental_bytes": maximum_admitted_upgrade_bytes,
            "all_admitted_upgrades_file_bytes": (
                authentic_file_bytes + maximum_admitted_upgrade_bytes),
            "comparison_prototype_file_cap_bytes": legacy_cap,
            "comparison_cap_upgrade_allowance_bytes": allowance,
            "maximum_reachable_allowance_under_comparison_cap_bytes": reachable_allowance,
            "maximum_reachable_file_bytes_under_comparison_cap": (
                authentic_file_bytes + reachable_allowance),
            "unreachable_slack_below_comparison_cap_bytes": allowance - reachable_allowance,
            "semantics": (
                "Start at the untouched authentic GSQ file-size floor and "
                "charge only positive aligned-byte deltas from currently "
                "admitted upgrades. The old prototype size is a comparison "
                "cap, never a reason to downgrade an incumbent tensor."),
        },
        "family_budget": dict(sorted(family_budget.items())),
        "controls": {
            label: {
                "nll": controls["results"][label],
                "logits": logits["results"][label],
            }
            for label in (
                "routed_expert_q4", "output_head_bf16",
                "linear_attention_bf16", "self_attention_bf16",
                "shared_expert_bf16", "all_q8_bf16",
                "all_eligible_upgrades")
        },
        "routed_subfamily_controls": {
            label: {
                "nll": routed_controls["results"][label],
                "logits": routed_logits["results"][label],
            }
            for label in ROUTED_POLICY_CONTROLS
        },
        "routed_interaction_control": {
            "label": interaction_label,
            "nll": routed_interaction_controls["results"][interaction_label],
            "logits": routed_interaction_logits["results"][interaction_label],
        },
        "warning": (
            "Fixed-screen admission is provisional. Routed gate and up Q4 "
            "alternatives passed their subfamily screens; routed down remains "
            "quarantined because its logit drift failed. Pinned-llama.cpp replay, "
            "search-disjoint calibration, and held-out release gates remain "
            "required before materializing a final GGUF."),
        "tensors": policies,
    }
    _atomic_json(output_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--routed-controls", type=Path, required=True)
    parser.add_argument("--routed-logits", type=Path, required=True)
    parser.add_argument("--routed-interaction-controls", type=Path, required=True)
    parser.add_argument("--routed-interaction-logits", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--gguf-python", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--comparison-file-cap-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.comparison_file_cap_bytes < 1:
        parser.error("--comparison-file-cap-bytes must be positive")
    return args


if __name__ == "__main__":
    result = build(_parse_args())
    print(json.dumps({
        "status": result["status"],
        "inventory": result["inventory"],
        "budget": result["budget"],
    }, indent=2, sort_keys=True))
