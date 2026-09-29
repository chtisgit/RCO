"""Install canonical native GGML candidates into streamed Qwen blocks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from native_store import NativeCandidateStore
from quant.ggml_native import GGMLType
from qwen35_native import Qwen35LinearAttentionGeometry, matrix_permutations


_BLOCK_NAME = re.compile(r"^blk\.(\d+)\.")
_GLOBAL_LOCATIONS = {
    "token_embd.weight": "embedding",
    "output.weight": "lm_head",
}


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

    def candidate_location(self, name: str) -> int | str:
        """Return the streamed lifetime in which a candidate is installed."""
        if name not in self.entries:
            raise KeyError(f"unknown native candidate {name!r}")
        match = _BLOCK_NAME.match(name)
        if match is not None:
            return int(match.group(1))
        try:
            return _GLOBAL_LOCATIONS[name]
        except KeyError as error:
            raise ValueError(
                f"candidate has no supported streaming location: {name}"
            ) from error

    def block_index(self, name: str) -> int:
        """Return a decoder-block index for block-local candidates."""
        location = self.candidate_location(name)
        if not isinstance(location, int):
            raise ValueError(f"candidate is outside a canonical block: {name}")
        return location

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


@dataclass
class NativeRelaxedSourceStats:
    """Packed-I/O and decoded working-set accounting for one row source."""

    reference_passes: int = 0
    alternative_passes: int = 0
    reference_payload_bytes_read: int = 0
    alternative_payload_bytes_read: int = 0
    max_resident_decoded_bytes: int = 0


class NativeManifestRelaxedLinearSource:
    """Stream one manifest matrix's native candidates in HF linear order.

    The reference is normally the highest native type and alternatives are the
    lower native types.  Deltas are formed from aligned decoded row chunks and
    released by the consumer immediately.  Row/column converter permutations
    are inverted without materializing a complete candidate matrix.
    """

    def __init__(
        self,
        store: NativeCandidateStore,
        entry: Mapping[str, Any],
        *,
        reference_type: GGMLType | int,
        alternative_types: tuple[GGMLType | int, ...],
        geometry: Qwen35LinearAttentionGeometry,
        rows_per_chunk: int = 16,
        expert_index: int | None = None,
    ) -> None:
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")
        if not alternative_types:
            raise ValueError("at least one alternative type is required")
        if not entry.get("rco_search"):
            raise ValueError("manifest entry is not searchable")
        source_shape = tuple(int(value) for value in entry["source_shape"])
        candidate_shape = tuple(int(value) for value in entry.get(
            "candidate_source_shape", source_shape))
        if len(candidate_shape) == 2:
            if candidate_shape != source_shape or expert_index is not None:
                raise ValueError(
                    "2-D relaxed source shape or expert index differs")
            out_features, in_features = candidate_shape
            flat_row_indices = None
        elif len(candidate_shape) == 3:
            if expert_index is None:
                raise ValueError("3-D relaxed source requires an expert index")
            experts, out_features, in_features = candidate_shape
            source_view = entry.get("source_view")
            if source_view is None:
                expected_candidate_shape = source_shape
            else:
                if source_view.get("axis") != 1 or len(source_shape) != 3:
                    raise ValueError("unsupported relaxed expert source view")
                view_start = int(source_view["start"])
                view_stop = int(source_view["stop"])
                expected_candidate_shape = (
                    source_shape[0], view_stop - view_start, source_shape[2])
            if candidate_shape != expected_candidate_shape:
                raise ValueError("expert candidate/source-view shape differs")
            if not 0 <= expert_index < experts:
                raise IndexError("expert index is outside the candidate tensor")
            flat_row_indices = (
                expert_index * out_features
                + np.arange(out_features, dtype=np.int64))
        else:
            raise ValueError(
                "relaxed linear source requires a 2-D matrix or 3-D expert stack")
        self.store = store
        self.entry = dict(entry)
        self.tensor_name = str(entry["destination_name"])
        self.reference_type = GGMLType(reference_type)
        self.alternative_types = tuple(
            GGMLType(value) for value in alternative_types)
        if self.reference_type in self.alternative_types:
            raise ValueError("reference type also appears as an alternative")
        if len(set(self.alternative_types)) != len(self.alternative_types):
            raise ValueError("alternative types must be unique")
        self.rows_per_chunk = int(rows_per_chunk)
        self.out_features = out_features
        self.in_features = in_features
        self.alternative_count = len(self.alternative_types)
        self.stats = NativeRelaxedSourceStats()
        self.expert_index = expert_index

        if len(candidate_shape) == 2:
            row_order, self._column_order = matrix_permutations(
                str(entry["normalized_source_name"]), source_shape, geometry)
            self._column_transform_multiplier = (
                1 if self._column_order is None else 2)
            if row_order is None:
                self._source_to_canonical_rows = None
            else:
                inverse = np.empty_like(row_order)
                inverse[row_order] = np.arange(len(row_order), dtype=np.int64)
                self._source_to_canonical_rows = inverse
        else:
            self._column_order = None
            self._column_transform_multiplier = 1
            self._source_to_canonical_rows = flat_row_indices

        expected_gguf_shape = list(reversed(candidate_shape))
        for candidate_type in (
            self.reference_type, *self.alternative_types,
        ):
            metadata = store.metadata(self.tensor_name, candidate_type)
            if metadata["gguf_shape"] != expected_gguf_shape:
                raise ValueError(
                    f"candidate shape changed for {self.tensor_name}/"
                    f"{candidate_type.name}")

    def _payload_bytes(self, candidate_type: GGMLType) -> int:
        metadata = self.store.metadata(self.tensor_name, candidate_type)
        if self.expert_index is None:
            return int(metadata["payload_bytes"])
        return self.out_features * int(metadata["row_size"])

    def _iter_type(self, candidate_type: GGMLType):
        if self._source_to_canonical_rows is None:
            rows = self.store.iter_decoded_rows(
                self.tensor_name,
                candidate_type,
                rows_per_chunk=self.rows_per_chunk,
            )
        else:
            rows = self.store.iter_decoded_row_indices(
                self.tensor_name,
                candidate_type,
                self._source_to_canonical_rows,
                rows_per_chunk=self.rows_per_chunk,
            )
        for start, decoded in rows:
            if self._column_order is None:
                restored = decoded
            else:
                restored = np.empty_like(decoded)
                restored[:, self._column_order] = decoded
                del decoded
            yield start, restored

    def _read_type_rows(
        self, candidate_type: GGMLType, row_indices: np.ndarray,
    ) -> np.ndarray:
        indices = np.asarray(row_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("row indices must be a non-empty vector")
        if np.any(indices < 0) or np.any(indices >= self.out_features):
            raise IndexError("requested source row is out of range")
        if self.expert_index is not None:
            canonical_indices = (
                self.expert_index * self.out_features + indices)
        elif self._source_to_canonical_rows is not None:
            canonical_indices = self._source_to_canonical_rows[indices]
        else:
            canonical_indices = indices
        chunks = self.store.iter_decoded_row_indices(
            self.tensor_name,
            candidate_type,
            canonical_indices,
            rows_per_chunk=min(self.rows_per_chunk, len(indices)),
        )
        decoded = np.concatenate([rows for _, rows in chunks])
        if self._column_order is None:
            return decoded
        restored = np.empty_like(decoded)
        restored[:, self._column_order] = decoded
        return restored

    def read_reference_rows(self, row_indices: Any) -> np.ndarray:
        indices = np.asarray(row_indices, dtype=np.int64)
        self.stats.reference_passes += 1
        metadata = self.store.metadata(self.tensor_name, self.reference_type)
        self.stats.reference_payload_bytes_read += (
            len(indices) * int(metadata["row_size"]))
        reference = self._read_type_rows(self.reference_type, indices)
        self.stats.max_resident_decoded_bytes = max(
            self.stats.max_resident_decoded_bytes,
            self._column_transform_multiplier * reference.nbytes,
        )
        return reference

    def read_delta_rows(
        self, alternative_index: int, row_indices: Any,
    ) -> np.ndarray:
        if not 0 <= alternative_index < self.alternative_count:
            raise IndexError("alternative index is out of range")
        indices = np.asarray(row_indices, dtype=np.int64)
        alternative_type = self.alternative_types[alternative_index]
        reference_metadata = self.store.metadata(
            self.tensor_name, self.reference_type)
        alternative_metadata = self.store.metadata(
            self.tensor_name, alternative_type)
        self.stats.alternative_passes += 1
        self.stats.reference_payload_bytes_read += (
            len(indices) * int(reference_metadata["row_size"]))
        self.stats.alternative_payload_bytes_read += (
            len(indices) * int(alternative_metadata["row_size"]))
        alternative = self._read_type_rows(alternative_type, indices)
        reference = self._read_type_rows(self.reference_type, indices)
        self.stats.max_resident_decoded_bytes = max(
            self.stats.max_resident_decoded_bytes,
            (self._column_transform_multiplier + 1)
            * max(alternative.nbytes, reference.nbytes),
        )
        alternative -= reference
        return alternative

    def iter_reference_rows(self):
        self.stats.reference_passes += 1
        self.stats.reference_payload_bytes_read += self._payload_bytes(
            self.reference_type)
        for start, reference in self._iter_type(self.reference_type):
            self.stats.max_resident_decoded_bytes = max(
                self.stats.max_resident_decoded_bytes,
                self._column_transform_multiplier * reference.nbytes,
            )
            yield start, reference

    def iter_delta_rows(self, alternative_index: int):
        if not 0 <= alternative_index < self.alternative_count:
            raise IndexError("alternative index is out of range")
        alternative_type = self.alternative_types[alternative_index]
        self.stats.alternative_passes += 1
        self.stats.alternative_payload_bytes_read += self._payload_bytes(
            alternative_type)
        self.stats.reference_payload_bytes_read += self._payload_bytes(
            self.reference_type)
        alternatives = self._iter_type(alternative_type)
        references = self._iter_type(self.reference_type)
        for (start, alternative), (reference_start, reference) in zip(
            alternatives, references, strict=True,
        ):
            if start != reference_start or alternative.shape != reference.shape:
                raise RuntimeError("native reference/alternative chunks differ")
            self.stats.max_resident_decoded_bytes = max(
                self.stats.max_resident_decoded_bytes,
                (self._column_transform_multiplier + 1)
                * max(alternative.nbytes, reference.nbytes),
            )
            alternative -= reference
            yield start, alternative


__all__ = [
    "NativeManifestRelaxedLinearSource",
    "NativeManifestWeightStore",
    "NativeRelaxedSourceStats",
]
