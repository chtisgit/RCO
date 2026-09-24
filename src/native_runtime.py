"""Install canonical native GGML candidates into streamed Qwen blocks."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping

import torch

from native_store import NativeCandidateStore
from quant.ggml_native import GGMLType
from qwen35_native import Qwen35LinearAttentionGeometry, matrix_permutations


_BLOCK_NAME = re.compile(r"^blk\.(\d+)\.")


class NativeManifestWeightStore:
    """Streaming adapter from canonical GGUF names to HF model parameters."""

    cache = False

    def __init__(
        self,
        store: NativeCandidateStore,
        manifest: Mapping[str, Any],
        model_dir: str | Path,
        *,
        rows_per_chunk: int = 16,
    ) -> None:
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")
        self.store = store
        self.rows_per_chunk = rows_per_chunk
        self.geometry = Qwen35LinearAttentionGeometry.from_model_dir(model_dir)
        entries = [
            entry for entry in manifest["entries"]
            if entry.get("rco_search")
            and entry["destination_name"] in store.index["tensors"]
        ]
        self.entries = {
            entry["destination_name"]: entry for entry in entries
        }
        if len(self.entries) != int(store.index["tensor_count"]):
            raise ValueError(
                "manifest does not map every native candidate-store tensor")

    def block_index(self, name: str) -> int:
        if name not in self.entries:
            raise KeyError(f"unknown native candidate {name!r}")
        match = _BLOCK_NAME.match(name)
        if match is None:
            raise ValueError(f"candidate is outside a canonical block: {name}")
        return int(match.group(1))

    @staticmethod
    def _type(bitwidth: int) -> GGMLType:
        try:
            return {2: GGMLType.Q2_0, 4: GGMLType.Q4_0}[int(bitwidth)]
        except KeyError as error:
            raise ValueError(f"unsupported native bitwidth {bitwidth}") from error

    def get_layer_storage_bytes(self, name: str, bitwidth: int) -> int:
        return int(self.store.metadata(
            name, self._type(bitwidth))["payload_bytes"])

    def install_layer_weight(
        self, model: torch.nn.Module, name: str, bitwidth: int,
    ) -> dict[str, int]:
        """Decode bounded rows directly into the materialized target weight."""
        entry = self.entries[name]
        candidate_type = self._type(bitwidth)
        target = model.get_parameter(entry["source_name"])
        source_shape = tuple(int(value) for value in entry["source_shape"])
        candidate_shape = tuple(int(value) for value in entry.get(
            "candidate_source_shape", source_shape))
        if tuple(target.shape) != source_shape:
            raise RuntimeError(
                f"target shape changed for {entry['source_name']}: "
                f"{tuple(target.shape)}")
        max_decoded = 0
        max_install = 0
        installed_rows = 0

        if len(candidate_shape) == 2:
            row_order, column_order = matrix_permutations(
                entry["normalized_source_name"], candidate_shape, self.geometry)
            if row_order is not None and column_order is not None:
                raise RuntimeError(
                    "simultaneous row/column permutations are unsupported")
            for start, decoded in self.store.iter_decoded_rows(
                name, candidate_type, rows_per_chunk=self.rows_per_chunk,
            ):
                stop = start + decoded.shape[0]
                values = torch.from_numpy(decoded).to(
                    device=target.device, dtype=target.dtype)
                with torch.no_grad():
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
                installed_rows += decoded.shape[0]
                max_decoded = max(max_decoded, decoded.nbytes)
                max_install = max(
                    max_install, values.numel() * values.element_size())
        elif len(candidate_shape) == 3:
            source_view = entry.get("source_view")
            if source_view is None:
                view_start, view_stop = 0, source_shape[1]
            else:
                if source_view.get("axis") != 1:
                    raise RuntimeError(f"unsupported source view: {source_view}")
                view_start = int(source_view["start"])
                view_stop = int(source_view["stop"])
            rows_per_expert = candidate_shape[1]
            for flat_start, decoded in self.store.iter_decoded_rows(
                name, candidate_type, rows_per_chunk=self.rows_per_chunk,
            ):
                chunk_offset = 0
                while chunk_offset < decoded.shape[0]:
                    flat_row = flat_start + chunk_offset
                    expert, row = divmod(flat_row, rows_per_expert)
                    run = min(
                        decoded.shape[0] - chunk_offset,
                        rows_per_expert - row)
                    values = torch.from_numpy(
                        decoded[chunk_offset:chunk_offset + run]).to(
                            device=target.device, dtype=target.dtype)
                    with torch.no_grad():
                        target[
                            expert,
                            view_start + row:view_start + row + run,
                            :,
                        ].copy_(values)
                    installed_rows += run
                    chunk_offset += run
                    max_install = max(
                        max_install, values.numel() * values.element_size())
                max_decoded = max(max_decoded, decoded.nbytes)
            if view_stop - view_start != rows_per_expert:
                raise RuntimeError("candidate/source-view row counts differ")
        else:
            raise RuntimeError(f"unsupported candidate rank: {candidate_shape}")

        expected_rows = math.prod(candidate_shape[:-1])
        if installed_rows != expected_rows:
            raise RuntimeError(
                f"installed {installed_rows} rows for {name}; "
                f"expected {expected_rows}")
        return {
            "installed_rows": installed_rows,
            "max_decoded_fp32_bytes": max_decoded,
            "max_install_bf16_bytes": max_install,
        }


__all__ = ["NativeManifestWeightStore"]
