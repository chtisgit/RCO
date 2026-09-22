"""Lossless RCO qparams mapping to compressed-tensors packed state."""

from __future__ import annotations

import math
import re
from typing import Any, Dict

import torch

from quant.qparams import load_qparams, pack_qweight, unpack_qweight


_ROUTED_EXPERT = re.compile(
    r"(?:^|\.)mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$")


def require_uniform_routed_expert_bits(assignment: Dict[str, int]) -> int | None:
    """Reject packed assignments the stock fused-expert loader cannot read.

    Transformers 5.13.1 chooses one compressed-tensors quantization scheme for
    the whole fused Qwen expert collection. Different packed widths have
    different row shapes and fail during logical-expert fusion. Return the
    common width, or ``None`` when the assignment has no routed experts.
    """
    selected = {
        int(bits) for name, bits in assignment.items()
        if _ROUTED_EXPERT.search(name)
    }
    if len(selected) > 1:
        widths = ", ".join(str(bits) for bits in sorted(selected))
        raise ValueError(
            "unmodified Transformers 5.13.1 compressed-tensors loading "
            "requires one packed bit width across all fused Qwen routed "
            f"experts; assignment contains {{{widths}}}")
    return next(iter(selected), None)


def pack_codes_to_int32(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned offset-binary codes in compressed-tensors row layout.

    Compressed-tensors represents signed integer values internally, then adds
    ``2**(bits - 1)`` before packing. RCO's symmetric qparams already store
    those offset unsigned codes. Both formats use least-significant-bit-first
    dense packing; compressed-tensors additionally begins every row on an
    INT32 boundary.
    """
    if codes.ndim != 2:
        raise ValueError(f"codes must be two-dimensional, got {codes.shape}")
    if not 1 <= bits <= 8:
        raise ValueError("bits must lie in [1, 8]")
    codes = codes.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    if codes.numel() and int(codes.max()) >= 1 << bits:
        raise ValueError(f"code outside {bits}-bit range")
    rows, columns = codes.shape
    words_per_row = math.ceil(columns * bits / 32)
    bytes_per_row = words_per_row * 4
    packed_rows = torch.zeros((rows, bytes_per_row), dtype=torch.uint8)
    for row in range(rows):
        packed = pack_qweight(codes[row], bits)
        packed_rows[row, :packed.numel()] = packed
    octets = packed_rows.reshape(rows, words_per_row, 4).to(torch.int64)
    words = (
        octets[..., 0]
        | (octets[..., 1] << 8)
        | (octets[..., 2] << 16)
        | (octets[..., 3] << 24)
    )
    return words.to(torch.int32).contiguous()


def compressed_state_from_bundle(bundle: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Map one supported RCO bundle without changing codes or scales."""
    bits = int(bundle["bits"])
    shape = tuple(int(value) for value in bundle["shape"])
    if len(shape) != 2:
        raise ValueError(f"compressed writer requires a 2-D weight, got {shape}")
    if not bundle.get("sym", False):
        raise ValueError(
            "direct compressed-tensors mapping currently requires symmetric qparams")
    if bundle.get("act_order", False) or bundle.get("perm") is not None:
        raise ValueError(
            "direct compressed-tensors mapping currently requires act_order=False")
    group_size = int(bundle["group_size"])
    expected_scale_shape = (shape[0], math.ceil(shape[1] / group_size))
    scales = bundle["scales"].detach().cpu().contiguous()
    zeros = bundle["zeros"].detach().cpu()
    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError(
            f"scale shape {tuple(scales.shape)} does not match "
            f"{expected_scale_shape}")
    symmetric_zero = float(1 << (bits - 1))
    if zeros.shape != scales.shape or not torch.equal(
            zeros.float(), torch.full_like(zeros.float(), symmetric_zero)):
        raise ValueError(
            f"symmetric {bits}-bit qparams must use zero point "
            f"{symmetric_zero} for every group")

    codes = bundle.get("qweight")
    if codes is None:
        codes = unpack_qweight(
            bundle["qweight_packed"], bits,
            tuple(int(value) for value in bundle["qweight_shape"]))
    if tuple(codes.shape) != shape:
        raise ValueError(
            f"code shape {tuple(codes.shape)} does not match weight shape {shape}")
    return {
        "weight_packed": pack_codes_to_int32(codes, bits),
        "weight_scale": scales,
        "weight_shape": torch.tensor(shape, dtype=torch.int64),
    }


def compressed_state_from_qparams(layer_dir, bits: int) -> Dict[str, torch.Tensor]:
    """Load, verify, and map one persisted qparams candidate."""
    return compressed_state_from_bundle(
        load_qparams(layer_dir, bits, unpack=False, verify=True))


__all__ = [
    "compressed_state_from_bundle",
    "compressed_state_from_qparams",
    "pack_codes_to_int32",
    "require_uniform_routed_expert_bits",
]
