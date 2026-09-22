"""Tensor-prefix streaming for sharded safetensors checkpoints."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from safetensors import safe_open


class SafeTensorPrefixLoader:
    """Materialize and release checkpoint-backed module prefixes on demand."""

    def __init__(self, checkpoint: str | Path):
        path = Path(checkpoint)
        if path.is_file() and path.name.endswith(".index.json"):
            self.root = path.parent
            self.index_path = path
        elif path.is_dir():
            self.root = path
            self.index_path = path / "model.safetensors.index.json"
        else:
            raise FileNotFoundError(path)

        if self.index_path.exists():
            with self.index_path.open() as handle:
                index = json.load(handle)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict):
                raise ValueError(
                    f"{self.index_path} has no weight_map object")
            self.weight_map = dict(weight_map)
        else:
            shards = sorted(self.root.glob("*.safetensors"))
            if len(shards) != 1:
                raise FileNotFoundError(
                    f"No model.safetensors.index.json and expected exactly "
                    f"one .safetensors file in {self.root}, found {len(shards)}")
            with safe_open(shards[0], framework="pt", device="cpu") as handle:
                self.weight_map = {name: shards[0].name for name in handle.keys()}

    def names_for_prefix(self, prefix: str) -> list[str]:
        marker = prefix + "."
        return sorted(
            name for name in self.weight_map
            if name == prefix or name.startswith(marker)
        )

    @staticmethod
    def _set_tensor(model: nn.Module, name: str, value: torch.Tensor) -> None:
        parent_path, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        if attribute in parent._parameters:
            old = parent._parameters[attribute]
            requires_grad = bool(old.requires_grad) if old is not None else False
            parent._parameters[attribute] = nn.Parameter(
                value, requires_grad=requires_grad)
        elif attribute in parent._buffers:
            parent._buffers[attribute] = value
        else:
            raise KeyError(
                f"Checkpoint tensor {name!r} does not map to a parameter or buffer")

    def load_prefix(
        self,
        model: nn.Module,
        prefix: str,
        device: torch.device | str,
        dtype: Optional[torch.dtype] = None,
    ) -> int:
        """Load all tensors below *prefix*, returning their resident bytes."""
        names = self.names_for_prefix(prefix)
        if not names:
            raise KeyError(f"Checkpoint contains no tensors below {prefix!r}")
        by_shard = defaultdict(list)
        for name in names:
            by_shard[self.weight_map[name]].append(name)

        resident_bytes = 0
        for shard_name, shard_names in by_shard.items():
            shard_path = self.root / shard_name
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for name in shard_names:
                    value = handle.get_tensor(name)
                    if dtype is not None and value.is_floating_point():
                        value = value.to(dtype=dtype)
                    value = value.to(device=device)
                    self._set_tensor(model, name, value)
                    resident_bytes += value.numel() * value.element_size()
        return resident_bytes

    def move_runtime_buffers(
        self, model: nn.Module, device: torch.device | str,
    ) -> int:
        """Move small config-created buffers that are absent from checkpoint.

        ``init_empty_weights(include_buffers=False)`` preserves values such as
        rotary frequencies on CPU. They must follow activations to the compute
        device, while checkpoint-backed buffers remain prefix-streamed.
        """
        moved_bytes = 0
        checkpoint_names = set(self.weight_map)
        for name, buffer in list(model.named_buffers(recurse=True)):
            if name in checkpoint_names or buffer.device.type == "meta":
                continue
            moved = buffer.to(device=device)
            self._set_tensor(model, name, moved)
            moved_bytes += moved.numel() * moved.element_size()
        return moved_bytes

    @classmethod
    def release_prefix(cls, model: nn.Module, prefix: str) -> int:
        """Replace a materialized module subtree with shape-only meta tensors."""
        module = model.get_submodule(prefix)
        released_bytes = 0
        for relative_name, parameter in list(module.named_parameters(recurse=True)):
            if parameter.device.type != "meta":
                released_bytes += parameter.numel() * parameter.element_size()
            cls._set_tensor(
                module,
                relative_name,
                torch.empty_like(parameter, device="meta"),
            )
        for relative_name, buffer in list(module.named_buffers(recurse=True)):
            if buffer.device.type != "meta":
                released_bytes += buffer.numel() * buffer.element_size()
            cls._set_tensor(
                module,
                relative_name,
                torch.empty_like(buffer, device="meta"),
            )
        return released_bytes


__all__ = ["SafeTensorPrefixLoader"]
