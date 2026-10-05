"""Bounded retain-or-upgrade installation for the authentic GSQ policy."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open

from checkpoint_stream import SafeTensorPrefixLoader


_BLOCK_NAME = re.compile(r"^blk\.(\d+)\.")
_GLOBAL_LOCATIONS = {
    "token_embd.weight": "embedding",
    "output.weight": "lm_head",
}


class GSQUpgradeWeightStore:
    """Install only admitted upgrades over an already loaded GSQ incumbent.

    Choice ``0`` is a true no-op: the exact tensor decoded by the authentic
    GGUF checkpoint loader remains installed. Choice ``1`` installs the
    policy-pinned higher-precision alternative.
    """

    cache = False

    def __init__(
        self,
        policy: Mapping[str, Any],
        manifest: Mapping[str, Any],
        model_dir: str | Path,
        native_weight_store: Any,
        *,
        rows_per_chunk: int = 16,
    ) -> None:
        if rows_per_chunk < 1:
            raise ValueError("rows_per_chunk must be positive")
        if policy.get("status") != "pass":
            raise ValueError("GSQ upgrade policy is not passing")
        self.rows_per_chunk = int(rows_per_chunk)
        self.native_weight_store = native_weight_store
        self.dense_loader = SafeTensorPrefixLoader(model_dir)
        manifest_entries = {
            entry["destination_name"]: dict(entry)
            for entry in manifest["entries"]
        }
        admitted = [
            record for record in policy["tensors"]
            if record.get("upgrade") is not None
            and record["upgrade"].get("policy_status") == "admitted"
        ]
        self.policy = {record["destination_name"]: dict(record) for record in admitted}
        if len(self.policy) != policy["inventory"]["admitted_upgrade_count"]:
            raise ValueError("admitted policy inventory differs")
        if set(self.policy) - set(manifest_entries):
            raise ValueError("policy contains tensors absent from manifest")
        self.entries = {name: manifest_entries[name] for name in self.policy}
        for name, record in self.policy.items():
            upgrade = record["upgrade"]
            entry = self.entries[name]
            if upgrade["kind"] == "bf16_derived_native_candidate":
                if upgrade["ggml_type"] != "Q4_0":
                    raise ValueError(f"unexpected native upgrade type for {name}")
                if name not in native_weight_store.entries:
                    raise ValueError(f"native store does not map {name}")
            elif upgrade["kind"] == "pinned_bf16_source":
                if upgrade["ggml_type"] != "BF16":
                    raise ValueError(f"unexpected dense upgrade type for {name}")
                if upgrade["source_tensor"] != entry["source_name"]:
                    raise ValueError(f"BF16 source name differs for {name}")
                if upgrade["source_shard"] != self.dense_loader.weight_map[
                    entry["source_name"]
                ]:
                    raise ValueError(f"BF16 source shard differs for {name}")
            else:
                raise ValueError(f"unsupported upgrade kind for {name}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.policy))

    def candidate_location(self, name: str) -> int | str:
        if name not in self.policy:
            raise KeyError(f"tensor is not admitted by the upgrade policy: {name}")
        match = _BLOCK_NAME.match(name)
        if match is not None:
            return int(match.group(1))
        try:
            return _GLOBAL_LOCATIONS[name]
        except KeyError as error:
            raise ValueError(f"unsupported streamed tensor location: {name}") from error

    def is_retain_choice(self, name: str, choice: int) -> bool:
        if name not in self.policy:
            raise KeyError(name)
        if int(choice) not in (0, 1):
            raise ValueError("GSQ upgrade choice must be zero or one")
        return int(choice) == 0

    def incremental_gguf_bytes(self, name: str) -> int:
        try:
            return int(self.policy[name]["upgrade"]["incremental_gguf_bytes"])
        except KeyError as error:
            raise KeyError(f"unknown admitted upgrade: {name}") from error

    def get_layer_storage_bytes(self, name: str, choice: int) -> int:
        if self.is_retain_choice(name, choice):
            return 0
        return int(self.policy[name]["upgrade"]["payload_bytes"])

    @staticmethod
    def _target(model: torch.nn.Module, source_name: str) -> torch.Tensor:
        try:
            return model.get_parameter(source_name)
        except AttributeError:
            return model.get_buffer(source_name)

    @torch.no_grad()
    def _install_bf16(self, model: torch.nn.Module, name: str) -> dict[str, int]:
        entry = self.entries[name]
        upgrade = self.policy[name]["upgrade"]
        source_name = entry["source_name"]
        target = self._target(model, source_name)
        source_shape = tuple(int(value) for value in entry["source_shape"])
        if tuple(target.shape) != source_shape:
            raise RuntimeError(
                f"BF16 target shape changed for {source_name}: {tuple(target.shape)}")
        shard_path = self.dense_loader.root / upgrade["source_shard"]
        installed_rows = 0
        source_bytes_read = 0
        max_install = 0
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            source = handle.get_slice(source_name)
            if tuple(source.get_shape()) != source_shape:
                raise RuntimeError(f"BF16 source shape differs for {source_name}")
            if len(source_shape) == 2:
                for start in range(0, source_shape[0], self.rows_per_chunk):
                    stop = min(start + self.rows_per_chunk, source_shape[0])
                    values = source[start:stop].to(
                        device=target.device, dtype=target.dtype)
                    target[start:stop].copy_(values)
                    installed_rows += stop - start
                    chunk_bytes = values.numel() * values.element_size()
                    source_bytes_read += chunk_bytes
                    max_install = max(max_install, chunk_bytes)
            elif len(source_shape) == 3:
                source_view = entry.get("source_view")
                if source_view is None:
                    view_start, view_stop = 0, source_shape[1]
                else:
                    if source_view.get("axis") != 1:
                        raise RuntimeError(f"unsupported BF16 source view: {source_view}")
                    view_start = int(source_view["start"])
                    view_stop = int(source_view["stop"])
                for expert in range(source_shape[0]):
                    for row in range(view_start, view_stop, self.rows_per_chunk):
                        stop = min(row + self.rows_per_chunk, view_stop)
                        values = source[expert, row:stop, :].to(
                            device=target.device, dtype=target.dtype)
                        target[expert, row:stop, :].copy_(values)
                        installed_rows += stop - row
                        chunk_bytes = values.numel() * values.element_size()
                        source_bytes_read += chunk_bytes
                        max_install = max(max_install, chunk_bytes)
            else:
                raise RuntimeError(
                    f"BF16 upgrade requires a matrix or expert stack: {source_shape}")
        if len(source_shape) == 2:
            expected_rows = source_shape[0]
        else:
            source_view = entry.get("source_view") or {
                "start": 0, "stop": source_shape[1]}
            expected_rows = source_shape[0] * (
                int(source_view["stop"]) - int(source_view["start"]))
        if installed_rows != expected_rows:
            raise RuntimeError(
                f"installed {installed_rows} BF16 rows for {name}; "
                f"expected {expected_rows}")
        return {
            "installed_rows": installed_rows,
            "max_decoded_fp32_bytes": 0,
            "max_install_bf16_bytes": max_install,
            "source_payload_bytes_read": source_bytes_read,
        }

    def install_layer_weight(
        self, model: torch.nn.Module, name: str, choice: int,
    ) -> dict[str, int]:
        if self.is_retain_choice(name, choice):
            return {
                "installed_rows": 0,
                "max_decoded_fp32_bytes": 0,
                "max_install_bf16_bytes": 0,
            }
        upgrade = self.policy[name]["upgrade"]
        if upgrade["kind"] == "bf16_derived_native_candidate":
            return self.native_weight_store.install_layer_weight(model, name, 4)
        return self._install_bf16(model, name)


__all__ = ["GSQUpgradeWeightStore"]
