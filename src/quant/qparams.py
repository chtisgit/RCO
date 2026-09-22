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
    bits        int                              1..8.
    group_size  int                              group size along input dim.
    sym         bool                             symmetric grid?
    perchannel  bool                             per-channel scales?
    act_order   bool                             was activation-order GPTQ used?
    shape       tuple[int, int]                  original (out_features, in_features).
    dtype       str                              original weight dtype, e.g. "bfloat16".
    checksums   dict[str, str | None]            SHA-256 of tensor payloads.
    schema      int                              format version (currently 3).

Reconstruction of the dequantized weight (column-permuted form):

    W_q = scales[:, group_idx] * (qweight - zeros[:, group_idx])

where group_idx[c] = c // group_size. To recover the *original* column order,
place column ``j`` at output column ``perm[j]`` (equivalently index W_q with
``argsort(perm)``). Most fast
inference kernels (e.g. Marlin) prefer to keep the permuted layout and permute
activations at runtime instead.
"""

from __future__ import annotations

import math
import os
import hashlib
import json
import tempfile
from typing import Any, Dict, Optional

import torch

QPARAMS_SCHEMA_VERSION = 3
QPARAMS_SUFFIX = "_qparams.pt"
CANDIDATE_INDEX_NAME = "candidate-index.json"


def tensor_sha256(tensor: torch.Tensor, chunk_bytes: int = 8 << 20) -> str:
    """Hash tensor dtype/shape and raw CPU bytes without one large byte copy."""
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    raw = value.view(torch.uint8).reshape(-1).numpy()
    view = memoryview(raw)
    for start in range(0, len(view), chunk_bytes):
        digest.update(view[start:start + chunk_bytes])
    return digest.hexdigest()


def _bundle_checksums(bundle: Dict[str, Any]) -> Dict[str, Optional[str]]:
    result: Dict[str, Optional[str]] = {}
    for name in ("qweight_packed", "scales", "zeros", "perm"):
        value = bundle.get(name)
        result[name] = None if value is None else tensor_sha256(value)
    metadata_names = (
        "qweight_numel", "qweight_shape", "packing", "bits", "group_size",
        "sym", "perchannel", "act_order", "shape", "dtype", "schema",
    )
    metadata = {name: bundle.get(name) for name in metadata_names}
    encoded = json.dumps(
        metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result["metadata"] = hashlib.sha256(encoded).hexdigest()
    return result


def _atomic_torch_save(bundle: Dict[str, Any], path: str) -> None:
    """Write a bundle in the destination directory and atomically publish it."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    os.close(descriptor)
    try:
        torch.save(bundle, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json_save(value: Dict[str, Any], path: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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
    bundle["checksums"] = _bundle_checksums(bundle)
    path = qparams_path(layer_dir, bits)
    _atomic_torch_save(bundle, path)
    return path


def load_qparams(
    layer_dir: str | os.PathLike,
    bits: int,
    *,
    unpack: bool = True,
    verify: bool = True,
    mmap: bool = False,
) -> Dict[str, Any]:
    """Load a qparams bundle, decoding packed codes only when requested.

    Schema-1 byte-per-code and schema-2 packed files remain readable. Schema-2
    and schema-3 files stay packed when ``unpack=False``. Schema-3 tensor
    checksums are verified before optional code unpacking.
    """
    path = qparams_path(layer_dir, bits)
    bundle = torch.load(
        path, map_location="cpu", weights_only=False, mmap=mmap)
    schema = bundle.get("schema", 0)
    if schema not in (1, 2, QPARAMS_SCHEMA_VERSION):
        raise ValueError(
            f"qparams schema mismatch at {path}: got {schema}, expected one "
            f"of 1, 2, {QPARAMS_SCHEMA_VERSION}"
        )
    if schema == QPARAMS_SCHEMA_VERSION and verify:
        expected = bundle.get("checksums")
        if not isinstance(expected, dict):
            raise ValueError(f"schema-3 qparams at {path} have no checksums")
        actual = _bundle_checksums(bundle)
        for name, digest in actual.items():
            if expected.get(name) != digest:
                raise ValueError(
                    f"qparams checksum mismatch at {path} for {name}")
    if schema >= 2 and unpack:
        bundle["qweight"] = unpack_qweight(
            bundle["qweight_packed"],
            bits=int(bundle["bits"]),
            shape=tuple(bundle["qweight_shape"]),
        )
    return bundle


def build_qparams_index(
    layer_dir: str | os.PathLike,
    *,
    output_name: str = CANDIDATE_INDEX_NAME,
) -> str:
    """Build an atomic metadata index without decoding candidate codes.

    Candidate bundles are memory-mapped and their already-recorded component
    checksums are copied into the index. Each candidate is a separate file, so
    its logical container offset is zero and ``file_bytes`` bounds the read.
    """
    root = os.path.abspath(os.fspath(layer_dir))
    entries: Dict[str, Dict[str, Any]] = {}
    suffix_length = len(QPARAMS_SUFFIX)
    for directory, _, files in os.walk(root):
        for filename in sorted(files):
            if not filename.endswith(QPARAMS_SUFFIX):
                continue
            stem = filename[:-suffix_length]
            try:
                bits = int(stem)
            except ValueError:
                continue
            path = os.path.join(directory, filename)
            layer_name = os.path.relpath(directory, root)
            bundle = load_qparams(
                directory, bits, unpack=False, verify=False, mmap=True)
            entry = {
                "path": os.path.relpath(path, root),
                "offset": 0,
                "file_bytes": os.path.getsize(path),
                "schema": int(bundle.get("schema", 0)),
                "bits": int(bundle["bits"]),
                "shape": [int(value) for value in bundle["shape"]],
                "dtype": str(bundle["dtype"]),
                "group_size": int(bundle["group_size"]),
                "sym": bool(bundle["sym"]),
                "perchannel": bool(bundle["perchannel"]),
                "act_order": bool(bundle["act_order"]),
                "checksums": bundle.get("checksums"),
            }
            entries.setdefault(layer_name, {})[str(bits)] = entry
            del bundle
    index = {
        "schema": 1,
        "format": "rco-qparams-files",
        "candidate_count": sum(len(item) for item in entries.values()),
        "layer_count": len(entries),
        "layers": entries,
    }
    output = os.path.join(root, output_name)
    _atomic_json_save(index, output)
    return output


def dequantize_from_qparams(
    bundle: Dict[str, Any],
    *,
    out: Optional[torch.Tensor] = None,
    restore_order: bool = False,
    column_chunk_size: int = 1024,
) -> torch.Tensor:
    """
    Reconstruct the dequantized weight tensor from a qparams bundle, in the
    *column-permuted* order used during quantization by default. Set
    ``restore_order=True`` for an ordinary linear weight. Dequantization works
    in column chunks and can write into a caller-provided CPU tensor, avoiding
    full-size FP32 scale, zero-point, and result intermediates.
    """
    if column_chunk_size < 1:
        raise ValueError("column_chunk_size must be positive")
    if "qweight" in bundle:
        qweight = bundle["qweight"].to(device="cpu", dtype=torch.uint8)
    else:
        qweight = unpack_qweight(
            bundle["qweight_packed"],
            bits=int(bundle["bits"]),
            shape=tuple(bundle["qweight_shape"]),
        )
    scales = bundle["scales"].to(device="cpu", dtype=torch.float32)
    zeros = bundle["zeros"].to(device="cpu", dtype=torch.float32)
    group_size = int(bundle["group_size"])
    d_row, d_col = qweight.shape
    target_dtype = getattr(torch, bundle["dtype"])
    if out is None:
        out = torch.empty((d_row, d_col), dtype=target_dtype, device="cpu")
    elif (out.device.type != "cpu" or tuple(out.shape) != (d_row, d_col)
          or out.dtype != target_dtype):
        raise ValueError(
            f"out must be CPU {target_dtype} with shape {(d_row, d_col)}, "
            f"got {out.device} {out.dtype} {tuple(out.shape)}")

    perm = bundle.get("perm") if restore_order else None
    if perm is not None:
        perm = perm.to(device="cpu", dtype=torch.long)
        if tuple(perm.shape) != (d_col,):
            raise ValueError(
                f"perm has shape {tuple(perm.shape)}, expected {(d_col,)}")
    for start in range(0, d_col, column_chunk_size):
        stop = min(start + column_chunk_size, d_col)
        group_idx = torch.arange(start, stop, dtype=torch.long) // group_size
        chunk = scales[:, group_idx] * (
            qweight[:, start:stop].float() - zeros[:, group_idx])
        converted = chunk.to(target_dtype)
        if perm is None:
            out[:, start:stop].copy_(converted)
        else:
            out[:, perm[start:stop]] = converted
        del group_idx, chunk, converted
    return out


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
