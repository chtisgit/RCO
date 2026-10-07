"""Block-streamed projected Gumbel-STE expert pruning.

``search.prune.optimize`` needs a resident Hugging Face model.  This module
runs the same search with one decoder block materialized at a time:

* Forward streams blocks 0..L-1.  Each block is wrapped in non-reentrant
  activation checkpointing, so only the block inputs are kept.
* Backward reaches the blocks in reverse order.  Checkpoint recomputation
  reloads the block, runs it again with gradients enabled, backpropagates,
  and releases it.  Every pass therefore loads each block exactly twice.
* The model weights are frozen.  The STE survival masks are the only
  tensors that require gradients.

The routing surrogate is ``MoEPruneWrapper``'s.  Top-k selection runs over
all experts, and the routing weights of the selected experts are multiplied
by their survival value, without renormalization.  Several Gumbel samples
(for example an antithetic pair) are evaluated in one pass.  Every row of
the batch belongs to one sample ("variant"), and a token's survival values
come from its own variant's mask, so all samples share each block load.

The loss is ``metrics.compute_kl_loss`` with a compact top-k reference: the
reference log-probabilities are renormalized over their top k, and the model
log-probabilities use the full-vocabulary normalizer.  The LM-head and KL
are computed per position chunk under checkpointing, so the full logits are
never materialized.

The optimizer step is ``optimize``'s per-layer path: per-layer gradient
projection, Adam, per-layer retraction, and transport of the first moment.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from manifold import (
    project_gradient_per_layer,
    retraction_per_layer,
    vector_transport_per_layer,
)
from model_adapter import get_model_adapter
from search.prune import budget_assignment_per_layer
from search.streaming import (
    StreamingMemoryStats,
    _extract_hidden,
    _install_selected_candidates,
)


# ----------------------------------------------------------------------------
# Pure pieces
# ----------------------------------------------------------------------------


def surrogate_routing_weights(
    weights: torch.Tensor,
    indices: torch.Tensor,
    layer_survival: torch.Tensor,
    token_variant: torch.Tensor,
) -> torch.Tensor:
    """Scale each selected expert's routing weight by its survival value.

    ``layer_survival`` is [variants, experts] and ``token_variant`` holds the
    variant of every flattened token.  This matches ``MoEPruneWrapper``,
    which computes in float32 and casts back to the weights' dtype.
    """
    kept = layer_survival[token_variant.unsqueeze(-1), indices]
    return (weights.float() * kept.float()).to(weights.dtype)


def compact_top_k_kl(
    logits: torch.Tensor,
    reference_values: torch.Tensor,
    reference_indices: torch.Tensor,
) -> torch.Tensor:
    """Per-position KL of ``metrics.compute_kl_loss`` with a compact reference."""
    logits = logits.float()
    model_log_z = logits.logsumexp(dim=-1, keepdim=True)
    model_top = logits.gather(-1, reference_indices.long()) - model_log_z
    reference = reference_values.float()
    reference = reference - reference.logsumexp(dim=-1, keepdim=True)
    return (reference.exp() * (reference - model_top)).sum(dim=-1)


def streamed_row_kl(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    reference_values: torch.Tensor,
    reference_indices: torch.Tensor,
    *,
    position_chunk: int = 256,
) -> torch.Tensor:
    """Mean next-token top-k KL per row, without materializing all logits.

    ``hidden_states`` is [rows, sequence, hidden]; the reference tensors are
    [rows, sequence - 1, k].  Each chunk of positions is checkpointed, so
    the backward pass recomputes its logits instead of storing them.
    """
    rows, sequence, width = hidden_states.shape
    predicted = sequence - 1
    if reference_values.shape[:2] != (rows, predicted):
        raise ValueError("reference shape does not match the hidden states")
    flat_hidden = hidden_states[:, :-1].reshape(-1, width)
    flat_values = reference_values.reshape(rows * predicted, -1)
    flat_indices = reference_indices.reshape(rows * predicted, -1)

    def chunk_kl(hidden, values, indices):
        return compact_top_k_kl(F.linear(hidden, lm_head_weight), values, indices)

    pieces = []
    for start in range(0, flat_hidden.shape[0], position_chunk):
        stop = min(start + position_chunk, flat_hidden.shape[0])
        pieces.append(checkpoint(
            chunk_kl, flat_hidden[start:stop], flat_values[start:stop],
            flat_indices[start:stop], use_reentrant=False))
    return torch.cat(pieces).view(rows, predicted).mean(dim=1)


def tau_at(step: int, steps: int, tau_init: float, tau_min: float) -> float:
    """``optimize``'s geometric temperature schedule."""
    progress = step / max(steps - 1, 1)
    return max(tau_min, tau_init * (tau_min / tau_init) ** progress)


def gumbel_noise(shape: Sequence[int], generator: torch.Generator) -> torch.Tensor:
    uniform = torch.rand(tuple(shape), generator=generator).clamp(1e-20)
    return -torch.log(-torch.log(uniform) + 1e-20)


def ste_survival(
    alpha: torch.Tensor,
    noise: torch.Tensor,
    tau: float,
    prune_per_layer: int,
    layers: int,
    experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard per-layer survival forward, soft keep probability backward."""
    noisy = (alpha + noise) / tau
    soft_keep = F.softmax(noisy, dim=1)[:, 0]
    hard_prune = budget_assignment_per_layer(
        noisy.detach(), prune_per_layer, layers, experts)
    hard_keep = 1.0 - hard_prune.float()
    survival = hard_keep + (soft_keep - soft_keep.detach())
    return survival.view(layers, experts), hard_prune.view(layers, experts).bool()


def deterministic_mask(
    alpha: torch.Tensor, prune_per_layer: int, layers: int, experts: int,
) -> torch.Tensor:
    """Noise-free per-layer assignment, as ``optimize`` returns at the end."""
    return budget_assignment_per_layer(
        alpha.detach(), prune_per_layer, layers, experts,
    ).view(layers, experts).bool()


# ----------------------------------------------------------------------------
# Streamed model
# ----------------------------------------------------------------------------


@dataclass
class StreamedPruneStats:
    """Timing and memory of one streamed forward and backward pass."""

    block_loads: list[int] = field(default_factory=list)
    block_releases: list[int] = field(default_factory=list)
    load_seconds: float = 0.0
    forward_seconds: float = 0.0
    loss_seconds: float = 0.0
    backward_seconds: float = 0.0
    total_seconds: float = 0.0
    cuda_max_allocated: int = 0
    cuda_max_reserved: int = 0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _PassState:
    """Values that every streamed block of one pass reads."""

    def __init__(self) -> None:
        self.survival: Optional[torch.Tensor] = None
        self.token_variant: Optional[torch.Tensor] = None


class CheckpointedStreamingPruneBlock(nn.Module):
    """Load a block, run it with surrogate routing, and release it.

    The same region runs again during backward under non-reentrant
    checkpointing, so the block is reloaded instead of kept resident.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        model: nn.Module,
        path: str,
        router_path: str,
        layer_index: int,
        checkpoint_loader: Any,
        state: _PassState,
        stats: StreamedPruneStats,
        device: torch.device,
        checkpoint_dtype: Optional[torch.dtype],
    ) -> None:
        super().__init__()
        self.module = module
        object.__setattr__(self, "_root_model", model)
        object.__setattr__(self, "_checkpoint_loader", checkpoint_loader)
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_stats", stats)
        self.path = path
        self.router_path = router_path
        self.layer_index = layer_index
        self.device = device
        self.checkpoint_dtype = checkpoint_dtype
        self.train(module.training)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["module"], name)

    def _run(self, hidden_states, survival, args, kwargs):
        model = object.__getattribute__(self, "_root_model")
        loader = object.__getattribute__(self, "_checkpoint_loader")
        state = object.__getattribute__(self, "_state")
        stats = object.__getattribute__(self, "_stats")
        layer_survival = survival[:, self.layer_index]
        token_variant = state.token_variant

        def hook(module, inputs, output):
            logits, weights, indices = output
            return logits, surrogate_routing_weights(
                weights, indices, layer_survival, token_variant), indices

        loaded = False
        handle = None
        try:
            _sync(self.device)
            started = time.perf_counter()
            loaded = True
            loader.load_prefix(model, self.path, device=self.device,
                               dtype=self.checkpoint_dtype)
            _sync(self.device)
            stats.load_seconds += time.perf_counter() - started
            stats.block_loads[self.layer_index] += 1
            handle = model.get_submodule(self.router_path).register_forward_hook(hook)
            return self.module(hidden_states, *args, **kwargs)
        finally:
            if handle is not None:
                handle.remove()
            if loaded:
                loader.release_prefix(model, self.path)
                stats.block_releases[self.layer_index] += 1

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        survival = object.__getattribute__(self, "_state").survival

        def run(hidden, active_survival):
            return self._run(hidden, active_survival, args, kwargs)

        return checkpoint(run, hidden_states, survival,
                          use_reentrant=False, preserve_rng_state=False)


@dataclass(frozen=True)
class StreamedPruneResult:
    """Per-row losses of one pass and how the pass behaved."""

    row_kl: torch.Tensor
    objective: float
    stats: StreamedPruneStats


class StreamedPruneObjective:
    """Streamed surrogate-pruned forward, top-k KL, and backward to the masks."""

    def __init__(
        self,
        model: nn.Module,
        checkpoint_loader: Any,
        *,
        device: torch.device | str,
        checkpoint_dtype: Optional[torch.dtype] = None,
        embedding_store: Any = None,
        embedding_selected: Sequence[tuple[str, int]] = (),
        position_chunk: int = 256,
        router_suffix: str = "mlp.gate",
    ) -> None:
        self.model = model
        self.adapter = get_model_adapter(model)
        self.loader = checkpoint_loader
        self.device = torch.device(device)
        self.checkpoint_dtype = checkpoint_dtype
        self.embedding_store = embedding_store
        self.embedding_selected = tuple(embedding_selected)
        self.position_chunk = int(position_chunk)
        self.router_suffix = router_suffix
        # Loaders re-wrap tensors as parameters with the skeleton's flag.
        self.model.requires_grad_(False)
        prefixes = [
            *self.adapter.embedding_paths,
            *(f"{self.adapter.layers_path}.{index}"
              for index in range(len(self.adapter.layers))),
            *self.adapter.final_module_paths,
        ]
        if hasattr(self.loader, "assert_prefix_schema"):
            for prefix in dict.fromkeys(prefixes):
                self.loader.assert_prefix_schema(self.model, prefix)

    @property
    def layer_count(self) -> int:
        return len(self.adapter.layers)

    def _load(self, path: str, dtype: Optional[torch.dtype],
              stats: StreamedPruneStats, loaded: list[str]) -> None:
        _sync(self.device)
        started = time.perf_counter()
        loaded.append(path)
        self.loader.load_prefix(self.model, path, device=self.device, dtype=dtype)
        _sync(self.device)
        stats.load_seconds += time.perf_counter() - started

    def _embed(self, input_ids: torch.Tensor, stats: StreamedPruneStats) -> torch.Tensor:
        loaded: list[str] = []
        try:
            for path in self.adapter.embedding_paths:
                self._load(path, self.checkpoint_dtype, stats, loaded)
            if self.embedding_selected:
                _install_selected_candidates(
                    self.model, self.embedding_store, self.embedding_selected,
                    StreamingMemoryStats())
            with torch.no_grad():
                return self.adapter.embeddings[0](input_ids.to(self.device))
        finally:
            for path in reversed(loaded):
                self.loader.release_prefix(self.model, path)

    def run(
        self,
        input_ids: torch.Tensor,
        survival: torch.Tensor,
        row_variant: torch.Tensor,
        reference_values: torch.Tensor,
        reference_indices: torch.Tensor,
        *,
        backward: bool = True,
    ) -> StreamedPruneResult:
        """Forward all rows; optionally backpropagate the objective.

        ``survival`` is [variants, layers, experts].  The objective is the
        mean over variants of each variant's mean row KL, which is
        ``optimize``'s average of per-sample losses.
        """
        variants = survival.shape[0]
        if survival.shape[1] != self.layer_count:
            raise ValueError("survival layer count differs from the model")
        if row_variant.shape != (input_ids.shape[0],):
            raise ValueError("row_variant must name one variant per row")
        rows_per_variant = torch.bincount(row_variant, minlength=variants)
        if bool((rows_per_variant == 0).any()):
            raise ValueError("every variant needs at least one row")
        stats = StreamedPruneStats(
            block_loads=[0] * self.layer_count,
            block_releases=[0] * self.layer_count)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()

        state = _PassState()
        state.survival = survival.to(self.device)
        state.token_variant = row_variant.to(self.device).repeat_interleave(
            input_ids.shape[1])
        layers = self.adapter.layers
        originals = list(layers)
        loaded: list[str] = []
        try:
            self.loader.move_runtime_buffers(self.model, self.device)
            inputs_embeds = self._embed(input_ids, stats)
            for path in self.adapter.final_module_paths:
                dtype = self.checkpoint_dtype if path == "lm_head" else None
                self._load(path, dtype, stats, loaded)
            for index, module in enumerate(originals):
                path = f"{self.adapter.layers_path}.{index}"
                layers[index] = CheckpointedStreamingPruneBlock(
                    module, model=self.model, path=path,
                    router_path=f"{path}.{self.router_suffix}",
                    layer_index=index, checkpoint_loader=self.loader,
                    state=state, stats=stats, device=self.device,
                    checkpoint_dtype=self.checkpoint_dtype)

            forward_started = time.perf_counter()
            with torch.set_grad_enabled(backward):
                output = self.adapter.text_model(
                    inputs_embeds=inputs_embeds, use_cache=False)
                hidden_states = _extract_hidden(output)
                _sync(self.device)
                stats.forward_seconds = time.perf_counter() - forward_started

                loss_started = time.perf_counter()
                row_kl = streamed_row_kl(
                    hidden_states, self.adapter.lm_head().weight,
                    reference_values.to(self.device),
                    reference_indices.to(self.device),
                    position_chunk=self.position_chunk)
                row_weight = 1.0 / (
                    variants * rows_per_variant.to(self.device)[
                        row_variant.to(self.device)].to(row_kl.dtype))
                objective = (row_kl * row_weight).sum()
                _sync(self.device)
                stats.loss_seconds = time.perf_counter() - loss_started

            if backward:
                backward_started = time.perf_counter()
                objective.backward()
                _sync(self.device)
                stats.backward_seconds = time.perf_counter() - backward_started
            return StreamedPruneResult(
                row_kl.detach().cpu(), float(objective.detach().item()), stats)
        finally:
            for index, module in enumerate(originals):
                layers[index] = module
            for path in reversed(loaded):
                self.loader.release_prefix(self.model, path)
            gc.collect()
            if self.device.type == "cuda":
                stats.cuda_max_allocated = torch.cuda.max_memory_allocated(self.device)
                stats.cuda_max_reserved = torch.cuda.max_memory_reserved(self.device)
                torch.cuda.empty_cache()
            stats.total_seconds = time.perf_counter() - started


# ----------------------------------------------------------------------------
# Search loop with per-step checkpoints
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamedPruneConfig:
    layers: int
    experts: int
    prune_per_layer: int
    steps: int
    lr: float = 0.1
    tau_init: float = 1.0
    tau_min: float = 0.05
    gumbel_samples: int = 4
    antithetic: bool = True
    documents_per_sample: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.prune_per_layer < self.experts:
            raise ValueError("per-layer prune count is out of range")
        if self.antithetic and self.gumbel_samples % 2:
            raise ValueError("antithetic sampling needs an even sample count")
        if self.steps < 1 or self.documents_per_sample < 1:
            raise ValueError("steps and documents per sample must be positive")


class StreamedPruneSearch:
    """``optimize`` with ``per_layer_budget=True``, streamed and resumable.

    Each base noise draws its own document subset.  Its antithetic partner
    reuses those documents, and all samples of one step run in one pass.
    """

    def __init__(self, config: StreamedPruneConfig, initial_alpha: torch.Tensor) -> None:
        expected = (config.layers * config.experts, 2)
        if tuple(initial_alpha.shape) != expected:
            raise ValueError(f"alpha must have shape {expected}")
        self.config = config
        self.alpha = initial_alpha.detach().to(torch.float32).clone().requires_grad_(True)
        self.optimizer = torch.optim.Adam([self.alpha], lr=config.lr)
        self.costs = torch.tensor([0.0, 1.0])
        self.generator = torch.Generator().manual_seed(config.seed)
        self.step = 0
        self.history: list[dict[str, Any]] = []

    # -- checkpointing ---------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "alpha": self.alpha.detach().clone(),
            "optimizer": self.optimizer.state_dict(),
            "generator": self.generator.get_state(),
            "step": self.step,
            "history": self.history,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["config"] != asdict(self.config):
            raise ValueError("checkpoint was written with a different config")
        with torch.no_grad():
            self.alpha.copy_(state["alpha"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"])
        self.step = int(state["step"])
        self.history = list(state["history"])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(self.state_dict(), temporary)
        os.replace(temporary, path)

    def load(self, path: Path) -> None:
        self.load_state_dict(torch.load(path, weights_only=False))

    # -- one step ----------------------------------------------------------

    def sample(self, document_count: int):
        """Draw this step's survival masks, rows, and row variants."""
        config = self.config
        base = config.gumbel_samples // 2 if config.antithetic else config.gumbel_samples
        base = max(base, 1)
        tau = tau_at(self.step, config.steps, config.tau_init, config.tau_min)
        survivals, hard_masks, documents, row_variant = [], [], [], []
        for _ in range(base):
            noise = gumbel_noise(self.alpha.shape, self.generator)
            chosen = torch.randperm(document_count, generator=self.generator)[
                :config.documents_per_sample]
            for variant_noise in ((noise, -noise) if config.antithetic else (noise,)):
                survival, hard = ste_survival(
                    self.alpha, variant_noise, tau, config.prune_per_layer,
                    config.layers, config.experts)
                row_variant.extend([len(survivals)] * len(chosen))
                survivals.append(survival)
                hard_masks.append(hard)
                documents.append(chosen)
        return (tau, torch.stack(survivals), torch.stack(hard_masks),
                torch.cat(documents), torch.tensor(row_variant))

    def take_step(self, objective, input_ids, reference_values, reference_indices):
        """Run one optimizer step and append its history record."""
        config = self.config
        started = time.perf_counter()
        self.optimizer.zero_grad()
        tau, survival, hard_masks, documents, row_variant = self.sample(
            input_ids.shape[0])
        result = objective.run(
            input_ids[documents], survival, row_variant,
            reference_values[documents], reference_indices[documents])
        gradient = self.alpha.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("alpha gradient is missing or not finite")
        nonzero = int((gradient.abs() > 1e-12).any(dim=1).sum())
        if nonzero == 0:
            raise RuntimeError("alpha gradient is identically zero")
        raw_norm = float(gradient.norm())

        projection = project_gradient_per_layer(
            self.alpha, self.costs, config.layers, config.experts)
        projected_norm = float(self.alpha.grad.norm())
        self.optimizer.step()
        budget = retraction_per_layer(
            self.alpha, self.costs, config.prune_per_layer,
            config.layers, config.experts)
        vector_transport_per_layer(
            self.optimizer, self.alpha, self.costs, config.layers, config.experts)

        with torch.no_grad():
            probabilities = torch.softmax(self.alpha, dim=1)
            entropy = float(-(probabilities * (probabilities + 1e-10).log()).sum(1).mean())
            confidence = probabilities.max(dim=1).values
        variant_kl = [
            float(result.row_kl[row_variant == variant].mean())
            for variant in range(survival.shape[0])]
        mask = deterministic_mask(
            self.alpha, config.prune_per_layer, config.layers, config.experts)
        record = {
            "step": self.step,
            "tau": tau,
            "objective": result.objective,
            "variant_kl": variant_kl,
            "documents": documents.tolist(),
            "raw_grad_norm": raw_norm,
            "projected_grad_norm": projected_norm,
            "projection_coefficient": projection,
            "experts_with_gradient": nonzero,
            "expected_prune_per_layer": budget,
            "entropy": entropy,
            "decided_0_9": int((confidence > 0.9).sum()),
            "decided_0_99": int((confidence > 0.99).sum()),
            "sample_masks_differ_from_deterministic": [
                int((hard != mask).sum()) // 2 for hard in hard_masks],
            "deterministic_mask_packed": np.packbits(mask.numpy().reshape(-1)).tolist(),
            "stats": asdict(result.stats),
            "step_seconds": time.perf_counter() - started,
        }
        self.history.append(record)
        self.step += 1
        return record


__all__ = [
    "CheckpointedStreamingPruneBlock",
    "StreamedPruneConfig",
    "StreamedPruneObjective",
    "StreamedPruneResult",
    "StreamedPruneSearch",
    "StreamedPruneStats",
    "compact_top_k_kl",
    "deterministic_mask",
    "gumbel_noise",
    "ste_survival",
    "streamed_row_kl",
    "surrogate_routing_weights",
    "tau_at",
]
