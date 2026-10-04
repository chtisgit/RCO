"""Bounded inverse-conversion loader for an untouched Qwen GGUF checkpoint."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from checkpoint_stream import SafeTensorPrefixLoader
from native_gguf import import_pinned_gguf
from quant.ggml_native import GGMLNativeCodec, GGMLType
from qwen35_native import (
    Qwen35LinearAttentionGeometry,
    _reordered_head_indices,
    matrix_permutations,
)


_NON_NUMERIC_TRANSFORMS = {
    "strip_language_model_namespace",
    "rename_dt_bias",
    "split_fused_gate_up",
}


def _restore_vector(
    canonical: np.ndarray,
    entry: dict[str, Any],
    geometry: Qwen35LinearAttentionGeometry,
) -> np.ndarray:
    value = np.asarray(canonical, dtype=np.float32).reshape(-1)
    transforms = set(entry["converter_transforms"])
    if "reorder_value_heads" in transforms:
        order = _reordered_head_indices(
            geometry.num_key_heads, geometry.num_value_heads, 1)
        if len(order) != len(value):
            raise ValueError("reordered vector length differs from V-head geometry")
        restored = np.empty_like(value)
        restored[order] = value
        value = restored
    if "negative_exponential" in transforms:
        if np.any(value >= 0):
            raise ValueError("negative-exponential GGUF values must be negative")
        value = np.log(-value)
    if "add_one_to_norm" in transforms:
        value = value - 1.0
    supported = _NON_NUMERIC_TRANSFORMS | {
        "reorder_value_heads", "negative_exponential", "add_one_to_norm",
    }
    unknown = transforms - supported
    if unknown:
        raise ValueError(f"unsupported GGUF vector transforms: {sorted(unknown)}")
    return np.ascontiguousarray(value)


def _restore_conv1d(
    canonical: np.ndarray,
    entry: dict[str, Any],
    geometry: Qwen35LinearAttentionGeometry,
) -> np.ndarray:
    value = np.asarray(canonical, dtype=np.float32)
    source_shape = tuple(int(item) for item in entry["source_shape"])
    if len(source_shape) != 3 or source_shape[1] != 1:
        raise ValueError(f"unexpected conv1d source shape: {source_shape}")
    squeezed_shape = (source_shape[0], source_shape[2])
    if value.shape != squeezed_shape:
        raise ValueError(
            f"conv1d canonical shape differs: {value.shape} != {squeezed_shape}")
    qk_channels = 2 * geometry.num_key_heads * geometry.key_head_dim
    value_channels = geometry.num_value_heads * geometry.value_head_dim
    if qk_channels + value_channels != source_shape[0]:
        raise ValueError("conv1d channel count differs from model geometry")
    value_order = _reordered_head_indices(
        geometry.num_key_heads,
        geometry.num_value_heads,
        geometry.value_head_dim,
    )
    order = np.concatenate((
        np.arange(qk_channels, dtype=np.int64),
        qk_channels + value_order,
    ))
    restored = np.empty_like(value)
    restored[order] = value
    return np.ascontiguousarray(restored[:, None, :])


class GGUFManifestPrefixLoader:
    """Stream an original GGUF into its inverse-mapped HF parameter prefixes."""

    def __init__(
        self,
        gguf_path: str | Path,
        manifest: dict[str, Any],
        model_dir: str | Path,
        *,
        gguf_python: str | Path,
        ggml_library: str | Path,
        rows_per_chunk: int = 16,
    ) -> None:
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")
        self.gguf = import_pinned_gguf(gguf_python)
        self.reader = self.gguf.GGUFReader(Path(gguf_path).resolve(strict=True))
        self.tensors = {tensor.name: tensor for tensor in self.reader.tensors}
        self.geometry = Qwen35LinearAttentionGeometry.from_model_dir(model_dir)
        self.native_codec = GGMLNativeCodec(ggml_library)
        self.rows_per_chunk = rows_per_chunk
        self.by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in manifest["entries"]:
            destination = entry["destination_name"]
            if destination not in self.tensors:
                raise KeyError(f"GGUF tensor is absent: {destination}")
            self.by_source[entry["source_name"]].append(entry)
        self.weight_map = {name: "gguf" for name in self.by_source}
        self.max_decoded_chunk_bytes = 0

    def names_for_prefix(self, prefix: str) -> list[str]:
        marker = prefix + "."
        return sorted(
            name for name in self.weight_map
            if name == prefix or name.startswith(marker)
        )

    def _decoded_rows(self, entry: dict[str, Any]):
        tensor = self.tensors[entry["destination_name"]]
        data = np.asarray(tensor.data)
        if data.ndim < 1:
            raise ValueError(f"GGUF tensor has no dimensions: {tensor.name}")
        packed_rows = data.reshape(-1, data.shape[-1])
        expected_rows = int(np.prod(
            entry["candidate_source_shape"][:-1], dtype=np.int64))
        if len(packed_rows) != expected_rows:
            raise ValueError(
                f"GGUF row count differs for {tensor.name}: "
                f"{len(packed_rows)} != {expected_rows}")
        for start in range(0, expected_rows, self.rows_per_chunk):
            stop = min(start + self.rows_per_chunk, expected_rows)
            packed = np.ascontiguousarray(packed_rows[start:stop])
            if tensor.tensor_type.name == "Q2_0":
                row_width = int(entry["candidate_source_shape"][-1])
                decoded = np.empty((stop - start, row_width), dtype=np.float32)
                self.native_codec.dequantize_rows_into(
                    packed, GGMLType.Q2_0, decoded)
            else:
                decoded = self.gguf.quants.dequantize(
                    packed, tensor.tensor_type)
                decoded = np.array(decoded, dtype=np.float32, order="C", copy=True)
            self.max_decoded_chunk_bytes = max(
                self.max_decoded_chunk_bytes, decoded.nbytes)
            yield start, decoded

    @staticmethod
    def _target_dtype(
        model: nn.Module, name: str, requested: Optional[torch.dtype],
    ) -> torch.dtype:
        if requested is not None:
            return requested
        parent_path, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        if attribute in parent._parameters:
            return parent._parameters[attribute].dtype
        if attribute in parent._buffers:
            return parent._buffers[attribute].dtype
        raise KeyError(f"manifest source does not map to model state: {name}")

    def _load_matrix_entry(
        self,
        target: torch.Tensor,
        entry: dict[str, Any],
    ) -> None:
        source_shape = tuple(int(value) for value in entry["source_shape"])
        candidate_shape = tuple(int(value) for value in entry[
            "candidate_source_shape"])
        if len(source_shape) == 2:
            row_order, column_order = matrix_permutations(
                entry["normalized_source_name"], source_shape, self.geometry)
            for start, decoded in self._decoded_rows(entry):
                stop = start + len(decoded)
                values = torch.from_numpy(decoded).to(
                    device=target.device, dtype=target.dtype)
                if row_order is not None:
                    indices = torch.from_numpy(row_order[start:stop]).to(
                        device=target.device)
                    target.index_copy_(0, indices, values)
                elif column_order is not None:
                    indices = torch.from_numpy(column_order).to(
                        device=target.device)
                    target[start:stop, indices] = values
                else:
                    target[start:stop].copy_(values)
            return
        if len(source_shape) == 3 and entry.get("rco_search"):
            source_view = entry.get("source_view")
            if source_view is None:
                view_start, view_stop = 0, source_shape[1]
            else:
                if source_view.get("axis") != 1:
                    raise ValueError("unsupported expert source view")
                view_start = int(source_view["start"])
                view_stop = int(source_view["stop"])
            rows_per_expert = candidate_shape[1]
            for flat_start, decoded in self._decoded_rows(entry):
                offset = 0
                while offset < len(decoded):
                    flat_row = flat_start + offset
                    expert, row = divmod(flat_row, rows_per_expert)
                    run = min(len(decoded) - offset, rows_per_expert - row)
                    values = torch.from_numpy(decoded[offset:offset + run]).to(
                        device=target.device, dtype=target.dtype)
                    target[
                        expert, view_start + row:view_start + row + run, :,
                    ].copy_(values)
                    offset += run
            if view_stop - view_start != rows_per_expert:
                raise ValueError("expert view row count differs")
            return
        raise ValueError(f"unsupported streamed matrix source shape: {source_shape}")

    @torch.no_grad()
    def load_prefix(
        self,
        model: nn.Module,
        prefix: str,
        device: torch.device | str,
        dtype: Optional[torch.dtype] = None,
    ) -> int:
        names = self.names_for_prefix(prefix)
        if not names:
            raise KeyError(f"GGUF manifest has no tensors below {prefix!r}")
        resident_bytes = 0
        for name in names:
            entries = self.by_source[name]
            source_shape = tuple(int(value) for value in entries[0]["source_shape"])
            if any(tuple(entry["source_shape"]) != source_shape for entry in entries):
                raise ValueError(f"manifest source shape differs for {name}")
            target_dtype = self._target_dtype(model, name, dtype)
            target = torch.empty(source_shape, dtype=target_dtype, device=device)
            SafeTensorPrefixLoader._set_tensor(model, name, target)
            resident_bytes += target.numel() * target.element_size()
            if len(source_shape) == 1:
                if len(entries) != 1:
                    raise ValueError(f"vector source has multiple GGUF entries: {name}")
                canonical = np.concatenate([
                    chunk for _, chunk in self._decoded_rows(entries[0])
                ]).reshape(-1)
                restored = _restore_vector(canonical, entries[0], self.geometry)
                target.copy_(torch.from_numpy(restored).to(
                    device=target.device, dtype=target.dtype))
            elif len(source_shape) == 3 and not entries[0].get("rco_search"):
                if len(entries) != 1 or "squeeze_conv1d" not in entries[0][
                    "converter_transforms"]:
                    raise ValueError(f"unsupported fixed 3-D source: {name}")
                canonical = np.concatenate([
                    chunk for _, chunk in self._decoded_rows(entries[0])
                ]).reshape(entries[0]["candidate_source_shape"][0], -1)
                restored = _restore_conv1d(canonical, entries[0], self.geometry)
                target.copy_(torch.from_numpy(restored).to(
                    device=target.device, dtype=target.dtype))
            else:
                for entry in entries:
                    self._load_matrix_entry(target, entry)
        return resident_bytes

    def validate_prefix_schema(self, model: nn.Module, prefix: str) -> dict[str, Any]:
        module = model.get_submodule(prefix)
        expected = {
            f"{prefix}.{name}" if name else prefix
            for name in module.state_dict().keys()
        }
        checkpoint = set(self.names_for_prefix(prefix))
        return {
            "prefix": prefix,
            "expected_count": len(expected),
            "checkpoint_count": len(checkpoint),
            "missing": sorted(expected - checkpoint),
            "unexpected": sorted(checkpoint - expected),
        }

    def assert_prefix_schema(self, model: nn.Module, prefix: str) -> dict[str, Any]:
        report = self.validate_prefix_schema(model, prefix)
        if report["missing"] or report["unexpected"]:
            raise ValueError(f"GGUF manifest schema differs for {prefix}: {report}")
        return report

    def move_runtime_buffers(
        self, model: nn.Module, device: torch.device | str,
    ) -> int:
        moved_bytes = 0
        checkpoint_names = set(self.weight_map)
        for name, buffer in list(model.named_buffers(recurse=True)):
            if name in checkpoint_names or buffer.device.type == "meta":
                continue
            moved = buffer.to(device=device)
            SafeTensorPrefixLoader._set_tensor(model, name, moved)
            moved_bytes += moved.numel() * moved.element_size()
        return moved_bytes

    def release_prefix(self, model: nn.Module, prefix: str) -> int:
        released_bytes = 0
        for name in self.names_for_prefix(prefix):
            parent_path, _, attribute = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            tensor = (
                parent._parameters[attribute]
                if attribute in parent._parameters
                else parent._buffers[attribute]
            )
            if tensor.device.type != "meta":
                released_bytes += tensor.numel() * tensor.element_size()
            SafeTensorPrefixLoader._set_tensor(
                model, name, torch.empty_like(tensor, device="meta"))
        return released_bytes


__all__ = ["GGUFManifestPrefixLoader"]
