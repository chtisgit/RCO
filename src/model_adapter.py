"""Model topology adapters used by quantization and search code.

RCO originally addressed decoder modules through model-specific attribute
chains such as ``model.model.layers``.  That is fragile for multimodal model
wrappers: Qwen3.5-MoE exposes its text decoder under
``model.language_model.layers`` while its text hyperparameters live in
``config.text_config``.  This module is the single authority for those paths.

The adapter deliberately describes topology only.  It does not load weights,
move modules, or instantiate a second model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence


QWEN35_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe"})
QWEN35_ARCHITECTURES = frozenset({
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
})


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _architectures(config: Any) -> Sequence[str]:
    value = _config_value(config, "architectures", ())
    return tuple(value or ())


def is_qwen35_config(config: Any) -> bool:
    """Return whether *config* describes a Qwen3.5/Qwen3.6 text or VLM model."""
    model_type = _config_value(config, "model_type")
    if model_type in QWEN35_MODEL_TYPES:
        return True
    return bool(QWEN35_ARCHITECTURES.intersection(_architectures(config)))


def get_text_config(config: Any) -> Any:
    """Return the decoder configuration, unwrapping multimodal configs."""
    text_config = _config_value(config, "text_config")
    if text_config is not None:
        return text_config
    return config


def _resolve(root: Any, path: str) -> Any:
    if not path:
        return root
    try:
        return root.get_submodule(path)
    except AttributeError as exc:
        raise ValueError(
            f"Model {type(root).__name__} does not expose required module "
            f"{path!r}."
        ) from exc


@dataclass(frozen=True)
class ModelAdapter:
    """Resolved text-decoder topology for an instantiated HF model."""

    model: Any
    family: str
    text_config: Any
    text_model_path: str
    layers_path: str
    embedding_paths: tuple[str, ...]
    final_module_paths: tuple[str, ...]

    @property
    def text_model(self) -> Any:
        return _resolve(self.model, self.text_model_path)

    @property
    def layers(self) -> Sequence[Any]:
        layers = _resolve(self.model, self.layers_path)
        if not hasattr(layers, "__len__") or not hasattr(layers, "__getitem__"):
            raise TypeError(
                f"{self.layers_path!r} resolved to {type(layers).__name__}, "
                "expected an ordered layer collection."
            )
        return layers

    @property
    def embeddings(self) -> tuple[Any, ...]:
        return tuple(_resolve(self.model, path) for path in self.embedding_paths)

    @property
    def final_modules(self) -> tuple[Any, ...]:
        return tuple(_resolve(self.model, path) for path in self.final_module_paths)

    @property
    def hidden_size(self) -> int:
        value = _config_value(self.text_config, "hidden_size")
        if value is None:
            raise ValueError("Text configuration has no hidden_size.")
        return int(value)

    @property
    def num_hidden_layers(self) -> int:
        value = _config_value(self.text_config, "num_hidden_layers")
        if value is None:
            return len(self.layers)
        return int(value)

    @property
    def num_experts(self) -> int:
        value = _config_value(self.text_config, "num_experts")
        if value is None:
            value = _config_value(self.text_config, "num_local_experts")
        if value is None:
            raise ValueError("Text configuration has no expert count.")
        return int(value)

    @property
    def num_experts_per_tok(self) -> int:
        value = _config_value(self.text_config, "num_experts_per_tok")
        if value is None:
            raise ValueError("Text configuration has no num_experts_per_tok.")
        return int(value)

    def mlp(self, layer_index: int) -> Any:
        layer = self.layers[layer_index]
        try:
            return layer.mlp
        except AttributeError as exc:
            raise ValueError(
                f"Layer {layer_index} ({type(layer).__name__}) has no mlp module."
            ) from exc

    def attention_name(self, layer_index: int) -> str:
        """Return the actual attention attribute for a hybrid decoder layer."""
        layer = self.layers[layer_index]
        for name in ("self_attn", "linear_attn"):
            if hasattr(layer, name):
                return name
        raise ValueError(
            f"Layer {layer_index} ({type(layer).__name__}) has no supported "
            "attention module."
        )

    def final_norm(self) -> Optional[Any]:
        norm = getattr(self.text_model, "norm", None)
        if norm is not None:
            return norm
        return getattr(self.text_model, "final_layer_norm", None)

    def lm_head(self) -> Any:
        return _resolve(self.model, "lm_head")


def get_model_adapter(model: Any) -> ModelAdapter:
    """Resolve the supported decoder topology for *model*.

    Qwen3.5/Qwen3.6 conditional-generation wrappers use a nested language
    model.  Existing decoder-only families retain their historical paths.
    """
    config = model.config
    model_type = _config_value(config, "model_type")

    if is_qwen35_config(config):
        try:
            _resolve(model, "model.language_model")
            text_model_path = "model.language_model"
        except ValueError:
            # Decoder-only Qwen3.5 classes expose model.layers directly;
            # conditional-generation wrappers add model.language_model.
            _resolve(model, "model.layers")
            text_model_path = "model"
        return ModelAdapter(
            model=model,
            family="qwen3_5",
            text_config=get_text_config(config),
            text_model_path=text_model_path,
            layers_path=f"{text_model_path}.layers",
            embedding_paths=(f"{text_model_path}.embed_tokens",),
            final_module_paths=(f"{text_model_path}.norm", "lm_head"),
        )

    if model_type == "opt":
        final = []
        decoder = _resolve(model, "model.decoder")
        if getattr(decoder, "final_layer_norm", None) is not None:
            final.append("model.decoder.final_layer_norm")
        if getattr(decoder, "project_out", None) is not None:
            final.append("model.decoder.project_out")
        final.append("lm_head")
        embedding_paths = ["model.decoder.embed_tokens"]
        if getattr(decoder, "embed_positions", None) is not None:
            embedding_paths.append("model.decoder.embed_positions")
        if getattr(decoder, "project_in", None) is not None:
            embedding_paths.append("model.decoder.project_in")
        return ModelAdapter(
            model=model,
            family="opt",
            text_config=config,
            text_model_path="model.decoder",
            layers_path="model.decoder.layers",
            embedding_paths=tuple(embedding_paths),
            final_module_paths=tuple(final),
        )

    if model_type in {"llama", "gemma", "gemma2", "phi3", "mistral"}:
        final = []
        base = _resolve(model, "model")
        if getattr(base, "norm", None) is not None:
            final.append("model.norm")
        final.append("lm_head")
        return ModelAdapter(
            model=model,
            family=str(model_type),
            text_config=config,
            text_model_path="model",
            layers_path="model.layers",
            embedding_paths=("model.embed_tokens",),
            final_module_paths=tuple(final),
        )

    raise ValueError(f"Model type {model_type!r} is not supported.")


TEXT_TENSOR_CATEGORIES = (
    "routed_expert",
    "shared_expert",
    "self_attention",
    "linear_attention",
    "router",
    "embedding",
    "normalization",
    "lm_head",
    "ordinary_text",
)


def classify_qwen35_tensor(name: str) -> str:
    """Classify one source-checkpoint tensor without loading its payload.

    The classifier is intentionally conservative.  Unknown tensors stay
    unknown so a coverage audit cannot silently omit a required text weight.
    """
    if name.startswith("model.visual.") or name.startswith("visual."):
        return "vision"
    if name.startswith("mtp.") or ".mtp." in name:
        return "mtp"
    if name.startswith("lm_head."):
        return "lm_head"
    if not name.startswith("model.language_model."):
        return "unknown"
    if ".mlp.experts." in name:
        return "routed_expert"
    if ".mlp.shared_expert." in name or ".mlp.shared_expert_gate." in name:
        return "shared_expert"
    if ".linear_attn." in name:
        return "linear_attention"
    if ".self_attn." in name:
        return "self_attention"
    if ".mlp.gate." in name or name.endswith(".mlp.gate.weight"):
        return "router"
    if ".embed_tokens." in name:
        return "embedding"
    if any("norm" in component for component in name.split(".")[:-1]):
        return "normalization"
    if ".layers." in name:
        return "ordinary_text"
    return "unknown"


def classify_qwen35_tensors(names: Iterable[str]) -> dict[str, list[str]]:
    """Group names by category, preserving input order within each group."""
    result: dict[str, list[str]] = {}
    for name in names:
        result.setdefault(classify_qwen35_tensor(name), []).append(name)
    return result


def assert_complete_qwen35_text_coverage(names: Iterable[str]) -> None:
    """Raise if a potentially required source tensor is unclassified."""
    unknown = classify_qwen35_tensors(names).get("unknown", [])
    if unknown:
        preview = ", ".join(repr(name) for name in unknown[:8])
        suffix = "" if len(unknown) <= 8 else f" (and {len(unknown) - 8} more)"
        raise ValueError(f"Unclassified Qwen3.5 tensors: {preview}{suffix}")
