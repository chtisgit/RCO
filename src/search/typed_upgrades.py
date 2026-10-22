"""Upgrade store for ``StreamedChatExactEvaluator`` with several payload types.

The evaluator's upgrade keys are tensor names, installed from one store of
one type.  Here a key is ``"TYPE:tensor"`` and is installed from the store
registered for that type, so one variant can mix, for example, Q3_K and
IQ2_XS payloads.  A variant must not hold two keys for the same tensor.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from native_runtime import NativeManifestWeightStore


def typed_key(type_name: str, tensor: str) -> str:
    return f"{type_name}:{tensor}"


def split_key(key: str) -> tuple[str, str]:
    type_name, separator, tensor = key.partition(":")
    if not separator or not type_name or not tensor:
        raise ValueError(f"upgrade key is not TYPE:tensor: {key!r}")
    return type_name, tensor


def check_variant_keys(keys: Iterable[str]) -> None:
    tensors = [split_key(key)[1] for key in keys]
    if len(tensors) != len(set(tensors)):
        raise ValueError("a variant installs two payloads for one tensor")


class SingleTypeExpertStore(NativeManifestWeightStore):
    """Choice 1 installs this store's type (any id the codec accepts)."""

    def __init__(self, store, manifest, model_dir, type_id: int, *, rows_per_chunk: int):
        super().__init__(store, manifest, model_dir, rows_per_chunk=rows_per_chunk)
        self.type_id = int(type_id)
        store.codec.type_name(self.type_id)

    def _type(self, bitwidth: int) -> int:
        if int(bitwidth) != 1:
            raise ValueError(f"only choice 1 installs a payload: {bitwidth}")
        return self.type_id


class TypedUpgradeStore:
    def __init__(self, stores: Mapping[str, Any]):
        if not stores:
            raise ValueError("no stores")
        self.stores = dict(stores)

    def _store(self, key: str) -> tuple[Any, str]:
        type_name, tensor = split_key(key)
        try:
            return self.stores[type_name], tensor
        except KeyError as error:
            raise KeyError(f"no store for type {type_name}") from error

    def block_index(self, key: str) -> int:
        store, tensor = self._store(key)
        return store.block_index(tensor)

    def install_layer_weight(self, model, key: str, choice: int):
        store, tensor = self._store(key)
        return store.install_layer_weight(model, tensor, choice)


__all__ = ["SingleTypeExpertStore", "TypedUpgradeStore", "check_variant_keys",
           "split_key", "typed_key"]
