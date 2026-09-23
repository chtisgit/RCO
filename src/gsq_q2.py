"""Bounded direct repacking of published GSQ2 experts into native Q2_0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
from safetensors import safe_open


class GSQCheckpoint:
    """Read one published logical expert triplet at a time."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve(strict=True)
        index_path = self.root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        self.weight_map = dict(index["weight_map"])

    def _tensor(self, name: str):
        try:
            shard = self.weight_map[name]
        except KeyError as error:
            raise KeyError(f"GSQ checkpoint has no tensor {name}") from error
        with safe_open(self.root / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def expert(self, prefix: str) -> tuple[np.ndarray, np.ndarray]:
        """Return unpacked uint8 GSQ codes and FP32 group scales."""
        shape = tuple(int(value) for value in self._tensor(
            prefix + ".weight_shape").tolist())
        if len(shape) != 2 or shape[1] % 128:
            raise ValueError(f"invalid GSQ logical shape for {prefix}: {shape}")
        rows, width = shape
        packed_tensor = self._tensor(prefix + ".weight_packed").contiguous()
        packed = packed_tensor.numpy().view(np.uint32)
        scales = self._tensor(prefix + ".weight_scale").float().numpy()
        if packed.shape != (rows, width // 16):
            raise ValueError(f"invalid GSQ packed shape for {prefix}: {packed.shape}")
        if scales.shape != (rows, width // 128):
            raise ValueError(f"invalid GSQ scale shape for {prefix}: {scales.shape}")
        shifts = np.arange(16, dtype=np.uint32) * 2
        codes = ((packed[..., None] >> shifts) & 3).astype(np.uint8).reshape(
            rows, width)
        return codes, scales

    def projection_payloads(
        self,
        *,
        layer: int,
        projection: str,
        expert_count: int,
    ) -> Iterator[tuple[int, bytes, dict[str, float | int | str]]]:
        """Yield one expert's direct native payload and numerical audit."""
        if projection not in {"gate_proj", "up_proj", "down_proj"}:
            raise ValueError(f"unsupported expert projection {projection}")
        for expert in range(expert_count):
            prefix = (
                f"model.language_model.layers.{layer}.mlp.experts."
                f"{expert}.{projection}")
            codes, scales = self.expert(prefix)
            payload = repack_gsq2_to_q2_0(codes, scales)
            mapped_scales = scales.astype(np.float16).astype(np.float32)
            scale_error = np.abs(
                mapped_scales.astype(np.float64) - scales.astype(np.float64))
            code_magnitude = np.abs(codes.astype(np.int16) - 2).reshape(
                codes.shape[0], codes.shape[1] // 128, 128)
            weight_error = code_magnitude * scale_error[..., None]
            yield expert, payload, {
                "source_prefix": prefix,
                "row_count": int(codes.shape[0]),
                "row_width": int(codes.shape[1]),
                "group_count": int(scales.size),
                "scale_rounding_count": int((mapped_scales != scales).sum()),
                "scale_underflow_count": int(
                    ((mapped_scales == 0) & (scales != 0)).sum()),
                "max_scale_absolute_error": float(scale_error.max(initial=0)),
                "max_weight_absolute_error": float(weight_error.max(initial=0)),
            }


def repack_gsq2_to_q2_0(codes: np.ndarray, scales: np.ndarray) -> bytes:
    """Map GSQ codes/scales to stock Q2_0 blocks without weight requantization."""
    codes = np.asarray(codes)
    scales = np.asarray(scales, dtype=np.float32)
    if codes.dtype != np.uint8 or codes.ndim != 2:
        raise ValueError("GSQ codes must be a 2D uint8 array")
    rows, width = codes.shape
    if width % 128 or scales.shape != (rows, width // 128):
        raise ValueError(
            f"incompatible GSQ code/scale shapes: {codes.shape}, {scales.shape}")
    if np.any(codes > 3) or not np.isfinite(scales).all():
        raise ValueError("GSQ codes/scales contain invalid values")
    q = (3 - codes).reshape(-1, 16, 4)
    packed_codes = (
        q[..., 0]
        | q[..., 1] << 2
        | q[..., 2] << 4
        | q[..., 3] << 6
    ).astype(np.uint8)
    native_scales = np.repeat((-scales).astype("<f2"), 2, axis=1).reshape(-1)
    blocks = np.empty((native_scales.size, 18), dtype=np.uint8)
    blocks[:, :2] = native_scales.view(np.uint8).reshape(-1, 2)
    blocks[:, 2:] = packed_codes
    return blocks.tobytes()


__all__ = ["GSQCheckpoint", "repack_gsq2_to_q2_0"]
