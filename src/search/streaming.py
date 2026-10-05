"""Inference-only, block-streamed evaluation for hard assignments.

The ordinary RCO search runtime keeps a complete Hugging Face model resident
while it swaps candidate weights.  This module instead leaves a model skeleton
on the meta device and materializes one decoder block at a time.  The original
text-model ``forward`` still owns mask, position, rotary, and hybrid-layer
argument preparation: lightweight wrappers load and release a block exactly
when the canonical forward loop calls it.

This path deliberately supports hard assignments and inference objectives
only.  It does not pretend that device-map hooks provide a streamed backward.
"""

from __future__ import annotations

import gc
import logging
import re
import resource
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_adapter import ModelAdapter, get_model_adapter


logger = logging.getLogger(__name__)


_LOGICAL_EXPERT = re.compile(
    r"^(?P<experts>(?:.+\.)?experts)\.(?P<index>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)$"
)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _process_peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.  RCO's supported execution environment
    # is Linux/CUDA, so retain the platform-specific conversion explicitly.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _extract_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    if isinstance(output, Sequence) and output:
        return output[0]
    raise TypeError(
        f"Cannot extract hidden states from {type(output).__name__}")


@torch.no_grad()
def _copy_candidate(model: nn.Module, name: str, weight: torch.Tensor) -> None:
    """Copy one logical candidate into a materialized model weight.

    Normal linear modules resolve directly.  Current Qwen MoE implementations
    keep experts fused as ``gate_up_proj`` and ``down_proj`` 3-D parameters,
    while RCO's candidate identifiers address logical expert projections.  The
    latter are copied into the corresponding fused slice without unfusing the
    block or allocating a second fused tensor.
    """
    try:
        module = model.get_submodule(name)
    except AttributeError:
        module = None
    if module is not None and isinstance(getattr(module, "weight", None), torch.Tensor):
        target = module.weight
        if tuple(target.shape) != tuple(weight.shape):
            raise ValueError(
                f"Candidate {name!r} has shape {tuple(weight.shape)}, expected "
                f"{tuple(target.shape)}")
        target.copy_(weight.to(device=target.device, dtype=target.dtype))
        return

    match = _LOGICAL_EXPERT.fullmatch(name)
    if match is None:
        raise KeyError(f"Candidate {name!r} does not resolve to a model weight")

    experts = model.get_submodule(match.group("experts"))
    expert_index = int(match.group("index"))
    projection = match.group("projection")
    gate_up = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    if not (isinstance(gate_up, torch.Tensor)
            and isinstance(down, torch.Tensor)
            and gate_up.ndim == 3 and down.ndim == 3):
        raise KeyError(
            f"Candidate {name!r} is logical expert storage, but "
            f"{match.group('experts')!r} is not a supported fused expert module")
    if not 0 <= expert_index < gate_up.shape[0]:
        raise IndexError(
            f"Expert {expert_index} is outside {match.group('experts')!r}")

    intermediate = gate_up.shape[1] // 2
    if projection == "gate_proj":
        target = gate_up[expert_index, :intermediate]
    elif projection == "up_proj":
        target = gate_up[expert_index, intermediate:]
    else:
        target = down[expert_index]
    if tuple(target.shape) != tuple(weight.shape):
        raise ValueError(
            f"Candidate {name!r} has shape {tuple(weight.shape)}, expected "
            f"{tuple(target.shape)}")
    target.copy_(weight.to(device=target.device, dtype=target.dtype))


@dataclass
class StreamingMemoryStats:
    """High-water measurements from one streamed evaluation."""

    loaded_blocks: int = 0
    max_block_bytes: int = 0
    max_candidate_bytes: int = 0
    cuda_max_allocated: int = 0
    cuda_max_reserved: int = 0
    process_peak_rss: int = 0
    checkpoint_tensor_bytes_read: int = 0
    candidate_storage_bytes_read: int = 0
    checkpoint_load_seconds: float = 0.0
    candidate_decode_seconds: float = 0.0
    block_forward_seconds: float = 0.0
    loss_seconds: float = 0.0
    checkpoint_release_seconds: float = 0.0
    total_seconds: float = 0.0
    cuda_load_max_allocated: int = 0
    cuda_load_max_reserved: int = 0
    cuda_decode_max_allocated: int = 0
    cuda_decode_max_reserved: int = 0
    cuda_forward_max_allocated: int = 0
    cuda_forward_max_reserved: int = 0
    cuda_loss_max_allocated: int = 0
    cuda_loss_max_reserved: int = 0
    cuda_release_max_allocated: int = 0
    cuda_release_max_reserved: int = 0


_PHASE_FIELDS = {
    "load": (
        "checkpoint_load_seconds",
        "cuda_load_max_allocated",
        "cuda_load_max_reserved",
    ),
    "decode": (
        "candidate_decode_seconds",
        "cuda_decode_max_allocated",
        "cuda_decode_max_reserved",
    ),
    "forward": (
        "block_forward_seconds",
        "cuda_forward_max_allocated",
        "cuda_forward_max_reserved",
    ),
    "loss": (
        "loss_seconds",
        "cuda_loss_max_allocated",
        "cuda_loss_max_reserved",
    ),
    "release": (
        "checkpoint_release_seconds",
        "cuda_release_max_allocated",
        "cuda_release_max_reserved",
    ),
}


def _start_phase(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def _finish_phase(
    stats: StreamingMemoryStats,
    phase: str,
    device: torch.device,
    started: float,
) -> None:
    duration_field, allocated_field, reserved_field = _PHASE_FIELDS[phase]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    setattr(stats, duration_field,
            getattr(stats, duration_field) + time.perf_counter() - started)
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
        setattr(stats, allocated_field,
                max(getattr(stats, allocated_field), allocated))
        setattr(stats, reserved_field,
                max(getattr(stats, reserved_field), reserved))
        stats.cuda_max_allocated = max(stats.cuda_max_allocated, allocated)
        stats.cuda_max_reserved = max(stats.cuda_max_reserved, reserved)


@dataclass(frozen=True)
class StreamingEvaluation:
    """Scalar causal-LM result and its measured working-set bounds."""

    loss: float
    token_count: int
    memory: StreamingMemoryStats
    document_mean_nll: tuple[float, ...] = ()
    document_token_counts: tuple[int, ...] = ()
    captured_logit_positions: tuple[int, ...] = ()
    captured_logits: Optional[torch.Tensor] = None
    captured_model_input: Optional[torch.Tensor] = None
    captured_layer_outputs: tuple[torch.Tensor, ...] = ()


class _StreamingBlock(nn.Module):
    """Load, patch, execute, and release one decoder block on demand."""

    def __init__(
        self,
        module: nn.Module,
        *,
        model: nn.Module,
        path: str,
        checkpoint_loader: Any,
        weight_store: Any,
        selected: Sequence[tuple[str, int]],
        device: torch.device,
        stats: StreamingMemoryStats,
        capture_positions: tuple[int, ...] = (),
        captured_layers: Optional[dict[int, torch.Tensor]] = None,
        layer_index: int = -1,
        checkpoint_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.module = module
        object.__setattr__(self, "_root_model", model)
        object.__setattr__(self, "_checkpoint_loader", checkpoint_loader)
        object.__setattr__(self, "_weight_store", weight_store)
        self.path = path
        self.selected = tuple(selected)
        self.device = device
        object.__setattr__(self, "_stats", stats)
        self.capture_positions = capture_positions
        object.__setattr__(self, "_captured_layers", captured_layers)
        self.layer_index = layer_index
        self.checkpoint_dtype = checkpoint_dtype
        self.train(module.training)

    def __getattr__(self, name: str) -> Any:
        # Canonical HF forward loops inspect attributes such as layer_type or
        # attention_type before invoking a decoder layer.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["module"], name)

    @torch.no_grad()
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        loader = object.__getattribute__(self, "_checkpoint_loader")
        model = object.__getattribute__(self, "_root_model")
        store = object.__getattribute__(self, "_weight_store")
        stats = object.__getattribute__(self, "_stats")
        loaded = False
        try:
            captured_layers = object.__getattribute__(self, "_captured_layers")
            if captured_layers is not None and self.layer_index == 0:
                captured_layers[-1] = _extract_hidden(args[0])[
                    0, list(self.capture_positions)
                ].float().cpu()
            phase_started = _start_phase(self.device)
            try:
                # ``release_prefix`` is safe for tensors that are still meta.
                # Mark the prefix first so a mid-shard I/O/device failure does
                # not strand the tensors already materialized by load_prefix.
                loaded = True
                block_bytes = loader.load_prefix(
                    model, self.path, device=self.device,
                    dtype=self.checkpoint_dtype)
            finally:
                _finish_phase(stats, "load", self.device, phase_started)
            stats.loaded_blocks += 1
            stats.max_block_bytes = max(stats.max_block_bytes, block_bytes)
            stats.checkpoint_tensor_bytes_read += block_bytes

            phase_started = _start_phase(self.device)
            try:
                _install_selected_candidates(
                    model, store, self.selected, stats)
            finally:
                _finish_phase(stats, "decode", self.device, phase_started)

            phase_started = _start_phase(self.device)
            try:
                output = self.module(*args, **kwargs)
            finally:
                _finish_phase(stats, "forward", self.device, phase_started)
            if captured_layers is not None:
                hidden = _extract_hidden(output)
                captured_layers[self.layer_index] = hidden[
                    0, list(self.capture_positions)
                ].float().cpu()
            stats.process_peak_rss = max(
                stats.process_peak_rss, _process_peak_rss_bytes())
            return output
        finally:
            if loaded:
                phase_started = _start_phase(self.device)
                try:
                    loader.release_prefix(model, self.path)
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                finally:
                    _finish_phase(
                        stats, "release", self.device, phase_started)


def _zero_candidate(model: nn.Module, name: str) -> torch.Tensor:
    """Create a zero candidate with the logical target's 2-D shape."""
    try:
        module = model.get_submodule(name)
    except AttributeError:
        module = None
    if module is not None and isinstance(getattr(module, "weight", None), torch.Tensor):
        return torch.zeros_like(module.weight, device="cpu")

    match = _LOGICAL_EXPERT.fullmatch(name)
    if match is None:
        raise KeyError(f"Candidate {name!r} does not resolve to a model weight")
    experts = model.get_submodule(match.group("experts"))
    index = int(match.group("index"))
    projection = match.group("projection")
    if projection == "down_proj":
        target = experts.down_proj[index]
    else:
        split = experts.gate_up_proj.shape[1] // 2
        start = 0 if projection == "gate_proj" else split
        target = experts.gate_up_proj[index, start:start + split]
    return torch.zeros(target.shape, dtype=target.dtype, device="cpu")


@torch.no_grad()
def _install_selected_candidates(
    model: nn.Module,
    store: Any,
    selected: Sequence[tuple[str, int]],
    stats: StreamingMemoryStats,
) -> None:
    """Install one location's choices while accounting bounded store I/O."""
    for name, bitwidth in selected:
        if (hasattr(store, "is_retain_choice")
                and store.is_retain_choice(name, bitwidth)):
            continue
        if bitwidth == 0:
            candidate = _zero_candidate(model, name)
            stats.max_candidate_bytes = max(
                stats.max_candidate_bytes, _tensor_bytes(candidate))
            _copy_candidate(model, name, candidate)
            del candidate
        elif hasattr(store, "install_layer_weight"):
            if hasattr(store, "get_layer_storage_bytes"):
                stats.candidate_storage_bytes_read += (
                    store.get_layer_storage_bytes(name, bitwidth))
            install = store.install_layer_weight(model, name, bitwidth)
            stats.max_candidate_bytes = max(
                stats.max_candidate_bytes,
                int(install["max_decoded_fp32_bytes"]),
                int(install["max_install_bf16_bytes"]),
            )
        else:
            if hasattr(store, "get_layer_storage_bytes"):
                stats.candidate_storage_bytes_read += (
                    store.get_layer_storage_bytes(name, bitwidth))
            candidate = store.get_layer_weight(name, bitwidth)
            stats.max_candidate_bytes = max(
                stats.max_candidate_bytes, _tensor_bytes(candidate))
            _copy_candidate(model, name, candidate)
            del candidate


def chunked_causal_cross_entropy(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head: nn.Module,
    *,
    loss_mask: Optional[torch.Tensor] = None,
    vocab_chunk_size: int = 8192,
) -> tuple[torch.Tensor, int]:
    """Compute exact next-token CE without materializing full-vocabulary logits."""
    loss, token_count, _, _ = _chunked_causal_cross_entropy_details(
        hidden_states,
        labels,
        lm_head,
        loss_mask=loss_mask,
        vocab_chunk_size=vocab_chunk_size,
    )
    return loss, token_count


def _chunked_causal_cross_entropy_details(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head: nn.Module,
    *,
    loss_mask: Optional[torch.Tensor] = None,
    vocab_chunk_size: int = 8192,
) -> tuple[torch.Tensor, int, tuple[float, ...], tuple[int, ...]]:
    """Compute aggregate and per-document exact next-token cross-entropy."""
    if vocab_chunk_size < 1:
        raise ValueError("vocab_chunk_size must be positive")
    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected hidden states [B,S,H] and labels [B,S]")
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("Hidden-state and label batch/sequence shapes differ")

    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError("LM head must expose a 2-D weight tensor")
    bias = getattr(lm_head, "bias", None)
    shifted_hidden = hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1])
    shifted_labels = labels[:, 1:].reshape(-1).to(hidden_states.device)
    if shifted_labels.numel() == 0:
        raise ValueError("Causal cross-entropy requires at least two tokens")

    target_weight = F.embedding(shifted_labels, weight)
    target_logits = (target_weight * shifted_hidden).sum(dim=-1).float()
    del target_weight
    if bias is not None:
        target_logits.add_(bias[shifted_labels].float())

    normalizer = None
    for start in range(0, weight.shape[0], vocab_chunk_size):
        stop = min(start + vocab_chunk_size, weight.shape[0])
        chunk_bias = bias[start:stop] if bias is not None else None
        logits = F.linear(shifted_hidden, weight[start:stop], chunk_bias)
        chunk_lse = torch.logsumexp(logits.float(), dim=-1)
        normalizer = (chunk_lse if normalizer is None
                      else torch.logaddexp(normalizer, chunk_lse))
        del logits, chunk_lse

    losses = normalizer - target_logits
    losses_by_document = losses.reshape(labels.shape[0], labels.shape[1] - 1)
    if loss_mask is not None:
        if loss_mask.shape != labels.shape:
            raise ValueError("Loss mask and labels must have the same shape")
        active = loss_mask[:, 1:].reshape(-1).to(
            device=losses.device, dtype=torch.bool)
        token_count = int(active.sum().item())
        if token_count == 0:
            raise ValueError("Loss mask selects no next-token targets")
        loss = losses[active].mean()
        active_by_document = active.reshape(
            labels.shape[0], labels.shape[1] - 1)
        document_token_counts_tensor = active_by_document.sum(dim=1)
        if bool((document_token_counts_tensor == 0).any().item()):
            raise ValueError("Loss mask leaves a document without targets")
        document_sums = (
            losses_by_document * active_by_document.to(losses.dtype)
        ).sum(dim=1)
        document_means = document_sums / document_token_counts_tensor
    else:
        token_count = losses.numel()
        loss = losses.mean()
        document_token_counts_tensor = torch.full(
            (labels.shape[0],), labels.shape[1] - 1,
            dtype=torch.long, device=losses.device,
        )
        document_means = losses_by_document.mean(dim=1)
    return (
        loss,
        token_count,
        tuple(float(value) for value in document_means.cpu().tolist()),
        tuple(int(value) for value in document_token_counts_tensor.cpu().tolist()),
    )


class StreamingHardCausalEvaluator:
    """Evaluate exact-budget hard assignments with one dense block resident."""

    def __init__(
        self,
        model: nn.Module,
        checkpoint_loader: Any,
        weight_store: Any,
        groups: Sequence[Any],
        bitwidths: Sequence[int],
        *,
        device: torch.device | str,
        vocab_chunk_size: int = 8192,
        checkpoint_dtype: Optional[torch.dtype] = None,
    ) -> None:
        if len(bitwidths) != 2:
            raise ValueError("Streaming hard evaluation requires two bitwidths")
        if getattr(weight_store, "cache", False):
            raise ValueError(
                "Streaming evaluation requires WeightStore(cache=False)")
        self.model = model
        self.adapter: ModelAdapter = get_model_adapter(model)
        self.checkpoint_loader = checkpoint_loader
        self.weight_store = weight_store
        self.groups = tuple(groups)
        self.bitwidths = tuple(sorted(int(bits) for bits in bitwidths))
        self.device = torch.device(device)
        self.vocab_chunk_size = int(vocab_chunk_size)
        self.checkpoint_dtype = checkpoint_dtype
        self._group_by_name = self._index_groups()
        self._validate_checkpoint_schema()
        self.model.requires_grad_(False)

    def _validate_checkpoint_schema(self) -> None:
        prefixes = [
            *self.adapter.embedding_paths,
            *(f"{self.adapter.layers_path}.{index}"
              for index in range(len(self.adapter.layers))),
            *self.adapter.final_module_paths,
        ]
        for prefix in dict.fromkeys(prefixes):
            self.checkpoint_loader.assert_prefix_schema(self.model, prefix)

    def _index_groups(self) -> dict[str, int]:
        result: dict[str, int] = {}
        block_prefixes = tuple(
            f"{self.adapter.layers_path}.{index}"
            for index in range(len(self.adapter.layers)))
        for group_index, group in enumerate(self.groups):
            for name in group.layer_names:
                if name in result:
                    raise ValueError(f"Candidate {name!r} appears in two groups")
                if hasattr(self.weight_store, "candidate_location"):
                    location = self.weight_store.candidate_location(name)
                    if isinstance(location, int) and not (
                        0 <= location < len(block_prefixes)
                    ):
                        raise ValueError(
                            f"Candidate {name!r} has invalid block index "
                            f"{location}")
                    if not isinstance(location, int) and location not in {
                        "embedding", "lm_head",
                    }:
                        raise ValueError(
                            f"Candidate {name!r} has invalid streaming "
                            f"location {location!r}")
                elif hasattr(self.weight_store, "block_index"):
                    block_index = self.weight_store.block_index(name)
                    if not 0 <= block_index < len(block_prefixes):
                        raise ValueError(
                            f"Candidate {name!r} has invalid block index "
                            f"{block_index}")
                elif not any(name.startswith(prefix + ".")
                             for prefix in block_prefixes):
                    raise ValueError(
                        f"Candidate {name!r} is outside the decoder blocks")
                result[name] = group_index
        return result

    def layer_assignment(self, assignment: torch.Tensor) -> dict[str, int]:
        if assignment.shape != (len(self.groups),):
            raise ValueError(
                f"assignment has shape {assignment.shape}, expected "
                f"({len(self.groups)},)")
        result = {}
        for name, group_index in self._group_by_name.items():
            choice = int(assignment[group_index])
            if choice not in (0, 1):
                raise ValueError(
                    f"assignment choice must be 0 or 1, got {choice}")
            result[name] = self.bitwidths[choice]
        return result

    def _selected_by_location(
        self, assignment: torch.Tensor,
    ) -> tuple[
        dict[int, list[tuple[str, int]]],
        list[tuple[str, int]],
        list[tuple[str, int]],
    ]:
        selected = self.layer_assignment(assignment)
        by_block = {index: [] for index in range(len(self.adapter.layers))}
        embedding: list[tuple[str, int]] = []
        lm_head: list[tuple[str, int]] = []
        for name, bitwidth in selected.items():
            if hasattr(self.weight_store, "candidate_location"):
                location = self.weight_store.candidate_location(name)
                if isinstance(location, int):
                    by_block[location].append((name, bitwidth))
                elif location == "embedding":
                    embedding.append((name, bitwidth))
                elif location == "lm_head":
                    lm_head.append((name, bitwidth))
                else:  # Guarded by _index_groups; retain a local invariant.
                    raise RuntimeError(
                        f"unsupported candidate location {location!r}")
                continue
            if hasattr(self.weight_store, "block_index"):
                by_block[self.weight_store.block_index(name)].append(
                    (name, bitwidth))
                continue
            for index in range(len(self.adapter.layers)):
                prefix = f"{self.adapter.layers_path}.{index}."
                if name.startswith(prefix):
                    by_block[index].append((name, bitwidth))
                    break
        return by_block, embedding, lm_head

    @torch.inference_mode()
    def evaluate(
        self,
        input_ids: torch.Tensor,
        assignment: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        capture_logit_positions: Sequence[int] = (),
        capture_layer_outputs: bool = False,
    ) -> StreamingEvaluation:
        """Run canonical text forward with block weights streamed on demand."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        positions = tuple(int(value) for value in capture_logit_positions)
        if positions and input_ids.shape[0] != 1:
            raise ValueError("logit capture currently requires batch size one")
        if capture_layer_outputs and not positions:
            raise ValueError("layer capture requires captured logit positions")
        if len(set(positions)) != len(positions) or any(
            value < 0 or value >= input_ids.shape[1] for value in positions
        ):
            raise ValueError("logit capture positions are invalid or repeated")
        stats = StreamingMemoryStats(process_peak_rss=_process_peak_rss_bytes())
        evaluation_started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        selected, embedding_selected, lm_head_selected = (
            self._selected_by_location(assignment))
        layers = self.adapter.layers
        originals = list(layers)
        loaded_prefixes: list[str] = []
        captured_layers: dict[int, torch.Tensor] | None = (
            {} if capture_layer_outputs else None)
        try:
            phase_started = _start_phase(self.device)
            try:
                self.checkpoint_loader.move_runtime_buffers(
                    self.model, self.device)
                for path in self.adapter.embedding_paths:
                    loaded_prefixes.append(path)
                    stats.checkpoint_tensor_bytes_read += (
                        self.checkpoint_loader.load_prefix(
                            self.model, path, device=self.device,
                            dtype=self.checkpoint_dtype)
                    )
                if embedding_selected:
                    decode_started = _start_phase(self.device)
                    try:
                        _install_selected_candidates(
                            self.model,
                            self.weight_store,
                            embedding_selected,
                            stats,
                        )
                    finally:
                        _finish_phase(
                            stats, "decode", self.device, decode_started)
                for path in self.adapter.final_module_paths:
                    if path == "lm_head":
                        continue
                    loaded_prefixes.append(path)
                    stats.checkpoint_tensor_bytes_read += (
                        self.checkpoint_loader.load_prefix(
                            self.model, path, device=self.device)
                    )
            finally:
                _finish_phase(stats, "load", self.device, phase_started)

            for index, module in enumerate(originals):
                path = f"{self.adapter.layers_path}.{index}"
                layers[index] = _StreamingBlock(
                    module,
                    model=self.model,
                    path=path,
                    checkpoint_loader=self.checkpoint_loader,
                    weight_store=self.weight_store,
                    selected=selected[index],
                    device=self.device,
                    stats=stats,
                    capture_positions=positions,
                    captured_layers=captured_layers,
                    layer_index=index,
                    checkpoint_dtype=self.checkpoint_dtype,
                )

            model_kwargs = {
                "input_ids": input_ids.to(self.device),
                "use_cache": False,
            }
            if attention_mask is not None:
                model_kwargs["attention_mask"] = attention_mask.to(self.device)
            output = self.adapter.text_model(**model_kwargs)
            hidden_states = _extract_hidden(output)

            phase_started = _start_phase(self.device)
            try:
                loaded_prefixes.append("lm_head")
                stats.checkpoint_tensor_bytes_read += (
                    self.checkpoint_loader.load_prefix(
                        self.model, "lm_head", device=self.device,
                        dtype=self.checkpoint_dtype)
                )
            finally:
                _finish_phase(stats, "load", self.device, phase_started)
            if lm_head_selected:
                phase_started = _start_phase(self.device)
                try:
                    _install_selected_candidates(
                        self.model,
                        self.weight_store,
                        lm_head_selected,
                        stats,
                    )
                finally:
                    _finish_phase(stats, "decode", self.device, phase_started)
            captured_logits = None
            if positions:
                phase_started = _start_phase(self.device)
                try:
                    selected_hidden = hidden_states[0, list(positions)]
                    captured_logits = F.linear(
                        selected_hidden,
                        self.adapter.lm_head().weight,
                        getattr(self.adapter.lm_head(), "bias", None),
                    ).float().cpu()
                finally:
                    _finish_phase(stats, "loss", self.device, phase_started)
            phase_started = _start_phase(self.device)
            try:
                (loss, token_count, document_mean_nll,
                 document_token_counts) = _chunked_causal_cross_entropy_details(
                    hidden_states,
                    input_ids.to(self.device),
                    self.adapter.lm_head(),
                    loss_mask=(loss_mask.to(self.device)
                               if loss_mask is not None else None),
                    vocab_chunk_size=self.vocab_chunk_size,
                )
            finally:
                _finish_phase(stats, "loss", self.device, phase_started)
            value = float(loss.item())
            stats.process_peak_rss = max(
                stats.process_peak_rss, _process_peak_rss_bytes())
            return StreamingEvaluation(
                value, token_count, stats,
                document_mean_nll=document_mean_nll,
                document_token_counts=document_token_counts,
                captured_logit_positions=positions,
                captured_logits=captured_logits,
                captured_model_input=(
                    captured_layers[-1] if captured_layers is not None else None
                ),
                captured_layer_outputs=(
                    tuple(captured_layers[index] for index in range(len(originals)))
                    if captured_layers is not None else ()
                ),
            )
        finally:
            for index, module in enumerate(originals):
                layers[index] = module
            phase_started = _start_phase(self.device)
            try:
                for path in reversed(loaded_prefixes):
                    self.checkpoint_loader.release_prefix(self.model, path)
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            finally:
                _finish_phase(stats, "release", self.device, phase_started)
                stats.total_seconds = time.perf_counter() - evaluation_started


__all__ = [
    "StreamingEvaluation",
    "StreamingHardCausalEvaluator",
    "StreamingMemoryStats",
    "chunked_causal_cross_entropy",
]
