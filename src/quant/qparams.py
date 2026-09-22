"""
Persisted quantization parameters per (layer, bitwidth).

run_quantize.py writes one of these alongside the dequantized fake-quant tensor:

    layer_dir/
      model.layers.0.self_attn.q_proj/
        4.pth              # dequantized FP/BF tensor (used by search pipeline)
        4_qparams.pt       # this file: integer codes + scales + zeros + perm + meta
        5.pth
        5_qparams.pt
        ...

Downstream checkpoint packers (vLLM compressed-tensors, Humming, GPTQModel, ...)
load the qparams bundle directly and pack into their target on-disk format. No
re-quantization of the dequantized tensor is needed.

Layout of a saved bundle (a single torch-pickled dict):

    qweight_packed uint8 [ceil(numel*bits/8)]   LSB-first packed integer codes.
    qweight_shape tuple[int, int]               logical code matrix shape.
    qweight_numel int                           logical code count.
    scales      original-dtype [d_row, n_groups]   per-(row, group) scale.
    zeros       original-dtype [d_row, n_groups]   per-(row, group) zero-point.
                                                Asymmetric quant: real-valued.
                                                Symmetric quant: zero-tensor.
    perm        int64 [d_col] | None              act-order column permutation that
                                                was applied to the input weights
                                                before quantization.  None when
                                                act_order=False.
    bits        int                              4..8.
    group_size  int                              group size along input dim.
    sym         bool                             symmetric grid?
    perchannel  bool                             per-channel scales?
    act_order   bool                             was activation-order GPTQ used?
    shape       tuple[int, int]                  original (out_features, in_features).
    dtype       str                              original weight dtype, e.g. "bfloat16".
    schema      int                              format version (currently 2).

Reconstruction of the dequantized weight (column-permuted form):

    W_q = scales[:, group_idx] * (qweight - zeros[:, group_idx])

where group_idx[c] = c // group_size. To recover the *original* column order,
apply inverse_perm = torch.argsort(perm) to the columns of W_q. Most fast
inference kernels (e.g. Marlin) prefer to keep the permuted layout and permute
activations at runtime instead.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import torch

QPARAMS_SCHEMA_VERSION = 2
QPARAMS_SUFFIX = "_qparams.pt"


def qparams_path(layer_dir: str | os.PathLike, bits: int) -> str:
    """Canonical path of the qparams bundle for one (layer, bitwidth)."""
    return os.path.join(str(layer_dir), f"{int(bits)}{QPARAMS_SUFFIX}")


def save_qparams(
    layer_dir: str | os.PathLike,
    *,
    bits: int,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    perm: Optional[torch.Tensor],
    handle: Any,
) -> str:
    """
    Persist a single (layer, bitwidth) qparams bundle. handle is the FastOBQ
    instance; we read group_size / sym / perchannel / act_order / shape / dtype
    off it so the call site stays tight.
    """
    quantizer = handle.quantizer_dict[bits]
    packed = pack_qweight(qweight, bits)
    bundle: Dict[str, Any] = {
        "qweight_packed": packed,
        "qweight_numel": int(qweight.numel()),
        "qweight_shape": tuple(int(x) for x in qweight.shape),
        "packing": "lsb_bitstream",
        "scales":  scales.detach().cpu().contiguous(),
        "zeros":   zeros.detach().cpu().contiguous(),
        "perm":    None if perm is None else perm.detach().cpu().contiguous(),
        "bits":       int(bits),
        "group_size": int(handle.group_size if handle.group_size else handle.d_col),
        "sym":        bool(quantizer.sym),
        "perchannel": bool(quantizer.perchannel),
        "act_order":  bool(handle.act_order),
        "shape":      tuple(int(x) for x in handle.W_shape),
        "dtype":      str(handle.W_dtype).replace("torch.", ""),
        "schema":     QPARAMS_SCHEMA_VERSION,
    }
    path = qparams_path(layer_dir, bits)
    torch.save(bundle, path)
    return path


def load_qparams(
    layer_dir: str | os.PathLike,
    bits: int,
    *,
    unpack: bool = True,
) -> Dict[str, Any]:
    """Load a qparams bundle, decoding packed codes only when requested.

    Schema-1 byte-per-code files remain readable.  Schema-2 files stay packed
    when ``unpack=False`` so streaming callers can bound their working set.
    """
    path = qparams_path(layer_dir, bits)
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    schema = bundle.get("schema", 0)
    if schema not in (1, QPARAMS_SCHEMA_VERSION):
        raise ValueError(
            f"qparams schema mismatch at {path}: got {schema}, expected 1 or "
            f"{QPARAMS_SCHEMA_VERSION}"
        )
    if schema == QPARAMS_SCHEMA_VERSION and unpack:
        bundle["qweight"] = unpack_qweight(
            bundle["qweight_packed"],
            bits=int(bundle["bits"]),
            shape=tuple(bundle["qweight_shape"]),
        )
    return bundle


def dequantize_from_qparams(bundle: Dict[str, Any]) -> torch.Tensor:
    """
    Reconstruct the dequantized weight tensor from a qparams bundle, in the
    *column-permuted* order used during quantization. Apply
    tensor[:, torch.argsort(bundle['perm'])] afterwards if you want the
    original column order.
    """
    if "qweight" in bundle:
        qweight = bundle["qweight"].to(torch.float32)
    else:
        qweight = unpack_qweight(
            bundle["qweight_packed"],
            bits=int(bundle["bits"]),
            shape=tuple(bundle["qweight_shape"]),
        ).to(torch.float32)
    scales  = bundle["scales"].to(torch.float32)
    zeros   = bundle["zeros"].to(torch.float32)
    group_size = bundle["group_size"]
    d_row, d_col = qweight.shape

    # Map each column to its group index, then broadcast to (d_row, d_col).
    group_idx = torch.arange(d_col, device=qweight.device) // group_size
    s = scales[:, group_idx]
    z = zeros[:, group_idx]
    w = s * (qweight - z)

    target_dtype = getattr(torch, bundle["dtype"])
    return w.to(target_dtype)


def pack_qweight(qweight: torch.Tensor, bits: int,
                 chunk_values: int = 1 << 20) -> torch.Tensor:
    """Pack unsigned integer codes into an LSB-first byte stream.

    Processing is chunked so temporary bit matrices remain small even for
    expert tensors.  Chunks except the final one end on a byte boundary.
    """
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    values = qweight.detach().to(device="cpu", dtype=torch.uint8).reshape(-1)
    if values.numel() and int(values.max()) >= (1 << bits):
        raise ValueError(f"qweight contains a value outside {bits}-bit range")

    # A multiple of 8 values always ends at a byte boundary for integer bits.
    chunk_values = max(8, (int(chunk_values) // 8) * 8)
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Packing qparams requires NumPy") from exc

    packed_chunks = []
    bit_positions = np.arange(bits, dtype=np.uint8)
    values_np = values.numpy()
    for start in range(0, values.numel(), chunk_values):
        chunk = values_np[start:start + chunk_values]
        bit_matrix = ((chunk[:, None] >> bit_positions) & 1).reshape(-1)
        packed_np = np.packbits(bit_matrix, bitorder="little")
        packed_chunks.append(torch.from_numpy(packed_np.copy()))
    if not packed_chunks:
        return torch.empty(0, dtype=torch.uint8)
    return torch.cat(packed_chunks).contiguous()


def unpack_qweight(packed: torch.Tensor, bits: int,
                   shape: tuple[int, ...],
                   chunk_values: int = 1 << 20) -> torch.Tensor:
    """Decode an LSB-first byte stream produced by :func:`pack_qweight`."""
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    numel = math.prod(shape)
    required_bytes = (numel * bits + 7) // 8
    stream = packed.detach().to(device="cpu", dtype=torch.uint8).reshape(-1)
    if stream.numel() != required_bytes:
        raise ValueError(
            f"packed byte count {stream.numel()} does not match "
            f"{numel} values at {bits} bits ({required_bytes} bytes)"
        )
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Unpacking qparams requires NumPy") from exc
    powers = (1 << np.arange(bits, dtype=np.uint16))
    chunk_values = max(8, (int(chunk_values) // 8) * 8)
    values = torch.empty(numel, dtype=torch.uint8)
    stream_np = stream.numpy()
    for start in range(0, numel, chunk_values):
        count = min(chunk_values, numel - start)
        bit_start = start * bits
        byte_start = bit_start // 8
        byte_end = (bit_start + count * bits + 7) // 8
        unpacked_bits = np.unpackbits(
            stream_np[byte_start:byte_end], bitorder="little",
            count=count * bits)
        bit_matrix = unpacked_bits.reshape(count, bits).astype(np.uint16)
        chunk = (bit_matrix * powers).sum(axis=1).astype(np.uint8)
        values[start:start + count].copy_(torch.from_numpy(chunk))
    return values.reshape(shape)
