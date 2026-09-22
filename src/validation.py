"""Bounded-memory numerical validation for packed quantization candidates."""

from __future__ import annotations

import heapq
import json
import math
import re
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open

from search.quant import get_actual_bitwidth
from store import LoadMode, WeightStore


_LOGICAL_EXPERT = re.compile(
    r"^(?P<experts>.+\.experts)\.(?P<index>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)$"
)


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


@dataclass(frozen=True)
class TensorError:
    layer: str
    bitwidth: int
    source_tensor: str
    shape: list[int]
    elements: int
    max_absolute_error: float
    mean_absolute_error: float
    mean_signed_error: float
    root_mean_square_error: float
    relative_frobenius_error: Optional[float]
    reference_norm_zero: bool
    source_tensor_bytes: int
    candidate_tensor_bytes: int
    candidate_storage_bytes: int


@dataclass(frozen=True)
class _ErrorSums:
    elements: int
    max_abs: float
    sum_abs: float
    sum_signed: float
    sum_squared: float
    reference_sum_squared: float


def tensor_error(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    chunk_elements: int = 1 << 20,
) -> tuple[dict, _ErrorSums]:
    """Compute error metrics with a bounded FP32/FP64 scratch chunk."""
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: reference {tuple(reference.shape)}, "
            f"candidate {tuple(candidate.shape)}")
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")
    ref_flat = reference.detach().cpu().reshape(-1)
    candidate_flat = candidate.detach().cpu().reshape(-1)
    count = ref_flat.numel()
    max_abs = 0.0
    sum_abs = 0.0
    sum_signed = 0.0
    sum_squared = 0.0
    reference_sum_squared = 0.0
    for start in range(0, count, chunk_elements):
        stop = min(start + chunk_elements, count)
        ref = ref_flat[start:stop].float()
        diff = candidate_flat[start:stop].float().sub(ref)
        absolute = diff.abs()
        if absolute.numel():
            max_abs = max(max_abs, absolute.max().item())
        sum_abs += absolute.sum(dtype=torch.float64).item()
        sum_signed += diff.sum(dtype=torch.float64).item()
        sum_squared += diff.square().sum(dtype=torch.float64).item()
        reference_sum_squared += ref.square().sum(dtype=torch.float64).item()
        del ref, diff, absolute
    denominator_zero = reference_sum_squared == 0.0
    relative = (
        None if denominator_zero
        else math.sqrt(sum_squared / reference_sum_squared)
    )
    divisor = max(count, 1)
    metrics = {
        "max_absolute_error": max_abs,
        "mean_absolute_error": sum_abs / divisor,
        "mean_signed_error": sum_signed / divisor,
        "root_mean_square_error": math.sqrt(sum_squared / divisor),
        "relative_frobenius_error": relative,
        "reference_norm_zero": denominator_zero,
    }
    return metrics, _ErrorSums(
        count, max_abs, sum_abs, sum_signed, sum_squared,
        reference_sum_squared)


class SafeTensorReference:
    """Read one ordinary weight or one logical fused-expert slice at a time."""

    def __init__(self, model_path: str | Path):
        self.root = Path(model_path)
        if not self.root.is_dir():
            raise ValueError(f"model path is not a directory: {self.root}")
        index_path = self.root / "model.safetensors.index.json"
        if index_path.exists():
            with index_path.open() as handle:
                self.weight_map = json.load(handle)["weight_map"]
        else:
            self.weight_map = {}
            for path in sorted(self.root.glob("*.safetensors")):
                with safe_open(path, framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        if key in self.weight_map:
                            raise ValueError(f"duplicate safetensors key {key!r}")
                        self.weight_map[key] = path.name
        if not self.weight_map:
            raise ValueError(f"no safetensors weights found under {self.root}")

    def _read(self, key: str, selection=None) -> torch.Tensor:
        path = self.root / self.weight_map[key]
        with safe_open(path, framework="pt", device="cpu") as handle:
            if selection is None:
                return handle.get_tensor(key)
            return handle.get_slice(key)[selection]

    def get(self, layer_name: str) -> tuple[torch.Tensor, str]:
        for direct in (f"{layer_name}.weight", layer_name):
            if direct in self.weight_map:
                return self._read(direct), direct

        match = _LOGICAL_EXPERT.fullmatch(layer_name)
        if match is None:
            raise KeyError(
                f"no dense source tensor resolves candidate {layer_name!r}")
        expert = int(match.group("index"))
        projection = match.group("projection")
        fused_name = (
            f"{match.group('experts')}.down_proj"
            if projection == "down_proj"
            else f"{match.group('experts')}.gate_up_proj"
        )
        key = next(
            (name for name in (fused_name, f"{fused_name}.weight")
             if name in self.weight_map),
            None,
        )
        if key is None:
            raise KeyError(
                f"candidate {layer_name!r} requires fused source "
                f"{fused_name!r}, "
                "which is absent")
        path = self.root / self.weight_map[key]
        with safe_open(path, framework="pt", device="cpu") as handle:
            view = handle.get_slice(key)
            shape = view.get_shape()
            if len(shape) != 3 or not 0 <= expert < shape[0]:
                raise ValueError(
                    f"fused source {key!r} has incompatible shape {shape}")
            if projection == "down_proj":
                value = view[expert, :, :]
            else:
                if shape[1] % 2:
                    raise ValueError(
                        f"fused gate/up dimension is odd for {key!r}: {shape}")
                split = shape[1] // 2
                rows = slice(0, split) if projection == "gate_proj" else slice(split, None)
                value = view[expert, rows, :]
        return value, f"{key}[{expert}:{projection}]"


def load_assignment(path: str | Path) -> dict[str, int]:
    """Read an RCO text assignment or JSON result."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open() as handle:
            payload = json.load(handle)
        payload = payload.get("assignment", payload)
        if not isinstance(payload, dict):
            raise ValueError("JSON assignment must be an object")
        return {str(name): int(bits) for name, bits in payload.items()}
    result = {}
    with path.open() as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(
                    f"invalid assignment line {line_number}: {raw.rstrip()!r}")
            name, bits = line.rsplit(":", 1)
            result[name.strip()] = int(bits.strip())
    return result


def validate_candidates(
    source_model: str | Path,
    layer_dir: str | Path,
    assignment: dict[str, int],
    *,
    bitwidth_map: Optional[dict[int, float]] = None,
    chunk_elements: int = 1 << 20,
    worst_count: int = 20,
    details_handle=None,
) -> dict:
    """Validate candidates sequentially and return a bounded aggregate report."""
    if not assignment:
        raise ValueError("assignment is empty")
    if worst_count < 0:
        raise ValueError("worst_count must be nonnegative")
    started = time.perf_counter()
    source = SafeTensorReference(source_model)
    store = WeightStore(str(layer_dir), mode=LoadMode.LAZY, cache=False).load()
    unknown = sorted(set(assignment) - set(store.get_layer_names()))
    if unknown:
        raise KeyError(
            f"assignment contains {len(unknown)} layers absent from candidate "
            f"store; first: {unknown[:5]}")

    totals = _ErrorSums(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    source_bytes = 0
    candidate_tensor_bytes = 0
    candidate_storage_bytes = 0
    weighted_bits = 0.0
    max_active_bytes = 0
    worst_max = []
    worst_relative = []

    for order, layer in enumerate(sorted(assignment)):
        bits = int(assignment[layer])
        reference, source_name = source.get(layer)
        if bits == 0:
            candidate = torch.zeros_like(reference)
            storage_bytes = 0
        else:
            candidate = store.get_layer_weight(layer, bits)
            storage_bytes = store.get_layer_storage_bytes(layer, bits)
        metrics, sums = tensor_error(
            reference, candidate, chunk_elements=chunk_elements)
        reference_bytes = reference.numel() * reference.element_size()
        decoded_bytes = candidate.numel() * candidate.element_size()
        item = TensorError(
            layer=layer,
            bitwidth=bits,
            source_tensor=source_name,
            shape=list(candidate.shape),
            elements=sums.elements,
            source_tensor_bytes=reference_bytes,
            candidate_tensor_bytes=decoded_bytes,
            candidate_storage_bytes=storage_bytes,
            **metrics,
        )
        item_dict = asdict(item)
        if details_handle is not None:
            details_handle.write(json.dumps(item_dict, allow_nan=False) + "\n")

        totals = _ErrorSums(
            totals.elements + sums.elements,
            max(totals.max_abs, sums.max_abs),
            totals.sum_abs + sums.sum_abs,
            totals.sum_signed + sums.sum_signed,
            totals.sum_squared + sums.sum_squared,
            totals.reference_sum_squared + sums.reference_sum_squared,
        )
        source_bytes += reference_bytes
        candidate_tensor_bytes += decoded_bytes
        candidate_storage_bytes += storage_bytes
        weighted_bits += (
            get_actual_bitwidth(bits, bitwidth_map or {}) * sums.elements)
        max_active_bytes = max(
            max_active_bytes, reference_bytes + decoded_bytes)
        if worst_count:
            max_entry = (item.max_absolute_error, order, item_dict)
            relative_key = (
                -1.0 if item.relative_frobenius_error is None
                else item.relative_frobenius_error)
            relative_entry = (relative_key, order, item_dict)
            for heap, entry in (
                (worst_max, max_entry), (worst_relative, relative_entry)):
                if len(heap) < worst_count:
                    heapq.heappush(heap, entry)
                elif entry[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, entry)
        del reference, candidate

    divisor = max(totals.elements, 1)
    reference_zero = totals.reference_sum_squared == 0.0
    report = {
        "schema": 1,
        "source_model": str(Path(source_model).resolve()),
        "candidate_directory": str(Path(layer_dir).resolve()),
        "tensor_count": len(assignment),
        "element_count": totals.elements,
        "weighted_average_selected_bits": weighted_bits / divisor,
        "effective_candidate_file_bits_per_weight": (
            candidate_storage_bytes * 8 / divisor),
        "errors": {
            "max_absolute_error": totals.max_abs,
            "mean_absolute_error": totals.sum_abs / divisor,
            "mean_signed_error": totals.sum_signed / divisor,
            "root_mean_square_error": math.sqrt(
                totals.sum_squared / divisor),
            "relative_frobenius_error": (
                None if reference_zero else math.sqrt(
                    totals.sum_squared / totals.reference_sum_squared)),
            "reference_norm_zero": reference_zero,
        },
        "io_and_memory": {
            "logical_source_tensor_bytes_read": source_bytes,
            "logical_candidate_tensor_bytes_decoded": candidate_tensor_bytes,
            "candidate_file_bytes_read": candidate_storage_bytes,
            "max_active_source_plus_candidate_bytes": max_active_bytes,
            "process_peak_rss_bytes": _peak_rss_bytes(),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "worst_by_max_absolute_error": [
            entry[2] for entry in sorted(worst_max, reverse=True)],
        "worst_by_relative_frobenius_error": [
            entry[2] for entry in sorted(worst_relative, reverse=True)],
    }
    return report


__all__ = [
    "SafeTensorReference",
    "TensorError",
    "load_assignment",
    "tensor_error",
    "validate_candidates",
]
