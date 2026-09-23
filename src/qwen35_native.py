"""Transform-aware, bounded native candidate generation for Qwen3.5."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from native_store import NativeCandidateStoreWriter
from quant.ggml_native import GGMLNativeCodec, GGMLType


@dataclass(frozen=True)
class Qwen35LinearAttentionGeometry:
    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int

    @classmethod
    def from_model_dir(cls, model_dir: str | Path):
        with (Path(model_dir) / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
        text = config.get("text_config", config)
        return cls(
            num_key_heads=int(text["linear_num_key_heads"]),
            num_value_heads=int(text["linear_num_value_heads"]),
            key_head_dim=int(text["linear_key_head_dim"]),
            value_head_dim=int(text["linear_value_head_dim"]),
        )

    def __post_init__(self):
        if min(
            self.num_key_heads,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
        ) <= 0:
            raise ValueError("linear-attention geometry must be positive")
        if self.num_value_heads % self.num_key_heads:
            raise ValueError("value-head count must be divisible by key-head count")


def _reordered_head_indices(
    num_key_heads: int,
    num_value_heads: int,
    head_dim: int,
) -> np.ndarray:
    """Return output-to-source indices for llama.cpp's tiled V-head order."""
    values_per_key = num_value_heads // num_key_heads
    return (
        np.arange(num_value_heads * head_dim, dtype=np.int64)
        .reshape(num_key_heads, values_per_key, head_dim)
        .transpose(1, 0, 2)
        .reshape(-1)
    )


def matrix_permutations(
    source_name: str,
    shape: Sequence[int],
    geometry: Qwen35LinearAttentionGeometry,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return row and column output-to-source permutations for one matrix."""
    if len(shape) != 2:
        raise ValueError(f"matrix source must be 2D, got {tuple(shape)}")
    rows, columns = (int(value) for value in shape)
    row_order: np.ndarray | None = None
    column_order: np.ndarray | None = None
    nk = geometry.num_key_heads
    nv = geometry.num_value_heads
    kd = geometry.key_head_dim
    vd = geometry.value_head_dim

    if source_name.endswith(".linear_attn.in_proj_qkv.weight"):
        q_size = nk * kd
        k_size = nk * kd
        v_size = nv * vd
        if rows != q_size + k_size + v_size:
            raise ValueError(f"unexpected QKV row count for {source_name}: {rows}")
        value_order = _reordered_head_indices(nk, nv, vd)
        row_order = np.concatenate((
            np.arange(q_size + k_size, dtype=np.int64),
            q_size + k_size + value_order,
        ))
    elif source_name.endswith(".linear_attn.in_proj_z.weight"):
        if rows != nv * vd:
            raise ValueError(f"unexpected Z row count for {source_name}: {rows}")
        row_order = _reordered_head_indices(nk, nv, vd)
    elif source_name.endswith((
        ".linear_attn.in_proj_a.weight",
        ".linear_attn.in_proj_b.weight",
    )):
        if rows != nv:
            raise ValueError(f"unexpected alpha/beta row count for {source_name}: {rows}")
        row_order = _reordered_head_indices(nk, nv, 1)
    elif source_name.endswith(".linear_attn.out_proj.weight"):
        if columns != nv * vd:
            raise ValueError(
                f"unexpected output-projection width for {source_name}: {columns}")
        column_order = _reordered_head_indices(nk, nv, vd)
    return row_order, column_order


def restore_source_matrix(
    canonical: np.ndarray,
    source_name: str,
    geometry: Qwen35LinearAttentionGeometry,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Undo canonical GGUF head ordering into a caller-owned HF matrix."""
    value = np.asarray(canonical)
    if value.ndim != 2:
        raise ValueError(f"canonical matrix must be 2D, got {value.shape}")
    if out is None:
        out = np.empty_like(value)
    if out.shape != value.shape or out.dtype != value.dtype or not out.flags.c_contiguous:
        raise ValueError(
            f"out must be C-contiguous {value.dtype} with shape {value.shape}")
    row_order, column_order = matrix_permutations(
        source_name, value.shape, geometry)
    if row_order is None and column_order is None:
        np.copyto(out, value)
    elif row_order is not None and column_order is None:
        out[row_order] = value
    elif row_order is None and column_order is not None:
        out[:, column_order] = value
    else:
        out[np.ix_(row_order, column_order)] = value
    return out


def _contiguous_runs(indices: np.ndarray) -> Iterator[tuple[int, int]]:
    if indices.ndim != 1 or not len(indices):
        raise ValueError("row indices must be a non-empty vector")
    start = int(indices[0])
    previous = start
    for raw_value in indices[1:]:
        value = int(raw_value)
        if value != previous + 1:
            yield start, previous + 1
            start = value
        previous = value
    yield start, previous + 1


class SafetensorGGUFRowSource:
    """Read only the source row runs needed for canonical GGUF output rows."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir).resolve(strict=True)
        self.geometry = Qwen35LinearAttentionGeometry.from_model_dir(
            self.model_dir)

    def iter_rows(
        self,
        entry: dict[str, Any],
        *,
        rows_per_chunk: int,
        stats: dict[str, int] | None = None,
    ) -> Iterator[np.ndarray]:
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")
        if not entry.get("rco_search"):
            raise ValueError(f"entry is not a decision group: {entry['destination_name']}")
        source_shape = tuple(int(value) for value in entry["source_shape"])
        if len(source_shape) != 2:
            raise ValueError(f"searched source is not a matrix: {source_shape}")
        expected_gguf_shape = tuple(reversed(source_shape))
        if tuple(entry["destination_gguf_shape"]) != expected_gguf_shape:
            raise ValueError(
                f"source/destination shape mismatch for {entry['source_name']}")
        unknown_transforms = set(entry["converter_transforms"]) - {
            "strip_language_model_namespace",
            "reorder_value_heads",
        }
        if unknown_transforms:
            raise ValueError(
                f"unsupported searched transforms for {entry['source_name']}: "
                f"{sorted(unknown_transforms)}")

        source_name = entry["source_name"]
        row_order, column_order = matrix_permutations(
            entry["normalized_source_name"], source_shape, self.geometry)
        if row_order is None:
            row_order = np.arange(source_shape[0], dtype=np.int64)
        shard = self.model_dir / entry["source_shard"]
        if not shard.is_file():
            raise FileNotFoundError(shard)

        from safetensors import safe_open
        import torch

        with safe_open(shard, framework="pt", device="cpu") as handle:
            tensor_slice = handle.get_slice(source_name)
            if tuple(tensor_slice.get_shape()) != source_shape:
                raise ValueError(
                    f"safetensors shape changed for {source_name}: "
                    f"{tuple(tensor_slice.get_shape())} != {source_shape}")
            for output_start in range(0, source_shape[0], rows_per_chunk):
                output_stop = min(output_start + rows_per_chunk, source_shape[0])
                indices = row_order[output_start:output_stop]
                pieces = [
                    tensor_slice[start:stop]
                    for start, stop in _contiguous_runs(indices)
                ]
                rows = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
                if column_order is not None:
                    order = torch.from_numpy(column_order)
                    rows = rows.index_select(1, order)
                output = rows.to(dtype=torch.float32).contiguous().numpy()
                if stats is not None:
                    stats["chunk_count"] = stats.get("chunk_count", 0) + 1
                    stats["max_dense_chunk_bytes"] = max(
                        stats.get("max_dense_chunk_bytes", 0), output.nbytes)
                    stats["max_source_rows_per_chunk"] = max(
                        stats.get("max_source_rows_per_chunk", 0), len(indices))
                yield output


def generate_native_block_candidates(
    model_dir: str | Path,
    entries: Iterable[dict[str, Any]],
    writer: NativeCandidateStoreWriter,
    codec: GGMLNativeCodec,
    *,
    candidate_types: Sequence[GGMLType] = (GGMLType.Q2_0, GGMLType.Q4_0),
    rows_per_chunk: int = 16,
) -> dict[str, Any]:
    """Generate every searched candidate in a canonical block manifest."""
    source = SafetensorGGUFRowSource(model_dir)
    selected = [entry for entry in entries if entry.get("rco_search")]
    if not selected:
        raise ValueError("block contains no searched tensors")
    stats: dict[str, int] = {
        "chunk_count": 0,
        "max_dense_chunk_bytes": 0,
        "max_source_rows_per_chunk": 0,
    }
    records: list[dict[str, Any]] = []
    for entry in selected:
        for candidate_type in candidate_types:
            chunks = (
                codec.quantize_rows(rows, candidate_type)
                for rows in source.iter_rows(
                    entry, rows_per_chunk=rows_per_chunk, stats=stats)
            )
            metadata = writer.write_packed_chunks(
                entry["destination_name"],
                candidate_type,
                entry["destination_gguf_shape"],
                chunks,
                provenance={
                    "source_tensor": entry["source_name"],
                    "source_shard": entry["source_shard"],
                    "converter_transforms": entry["converter_transforms"],
                    "rows_per_chunk": rows_per_chunk,
                },
            )
            records.append({
                "tensor": entry["destination_name"],
                "source_tensor": entry["source_name"],
                "ggml_type": candidate_type.name,
                "ggml_type_id": int(candidate_type),
                "gguf_shape": metadata["gguf_shape"],
                "payload_bytes": metadata["payload_bytes"],
                "aligned_gguf_bytes": metadata["aligned_gguf_bytes"],
                "sha256": metadata["sha256"],
                "converter_transforms": entry["converter_transforms"],
            })
    writer.finalize()
    return {
        "searched_tensor_count": len(selected),
        "candidate_count": len(records),
        "candidate_types": [value.name for value in candidate_types],
        "rows_per_chunk": rows_per_chunk,
        **stats,
        "candidates": records,
    }
