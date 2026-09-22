"""Materialize exact native candidates into a GGUF container.

The final production writer will add resumability and model-scale streaming.
This module supplies the smaller invariant needed first: a selected candidate
payload is framed by pinned ``gguf-py`` without being decoded or requantized,
and every non-selected tensor is copied unchanged from a reference GGUF.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from native_store import NativeCandidateStore
from quant.ggml_native import GGMLType


def import_pinned_gguf(gguf_python: str | os.PathLike) -> Any:
    """Import ``gguf`` from an explicitly selected pinned llama.cpp tree."""
    path = Path(gguf_python).expanduser().resolve(strict=True)
    if not (path / "gguf" / "__init__.py").is_file():
        raise ValueError(f"not a gguf-py directory: {path}")
    path_string = str(path)
    loaded = sys.modules.get("gguf")
    if loaded is not None:
        loaded_path = Path(loaded.__file__).resolve()
        if path not in loaded_path.parents:
            raise RuntimeError(
                f"gguf is already imported from {loaded_path}, not {path}")
        return loaded
    sys.path.insert(0, path_string)
    try:
        return importlib.import_module("gguf")
    finally:
        sys.path.remove(path_string)


def _copy_metadata(reader: Any, writer: Any, gguf: Any) -> None:
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE:
            continue
        if field.name.startswith("GGUF."):
            continue
        value_type = field.types[0]
        sub_type = (
            field.types[-1]
            if value_type == gguf.GGUFValueType.ARRAY
            else None
        )
        value = field.contents()
        if value is not None:
            writer.add_key_value(
                field.name, value, value_type, sub_type=sub_type)


def write_selected_native_gguf(
    reference_path: str | os.PathLike,
    output_path: str | os.PathLike,
    store: NativeCandidateStore,
    assignment: Mapping[str, GGMLType | int],
    *,
    gguf_python: str | os.PathLike,
) -> list[dict[str, Any]]:
    """Copy a reference GGUF while substituting selected native payloads.

    This deliberately accepts only tensors already present in the reference.
    Their reference shapes are the schema oracle; a candidate with a different
    canonical shape is rejected before an output file is published.
    """
    if not assignment:
        raise ValueError("assignment must select at least one tensor")
    gguf = import_pinned_gguf(gguf_python)
    reference = Path(reference_path).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reader = gguf.GGUFReader(reference)
    names = {tensor.name for tensor in reader.tensors}
    unknown = sorted(set(assignment) - names)
    if unknown:
        raise ValueError(f"assignment names are absent from reference: {unknown}")

    architecture_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if architecture_field is None:
        raise ValueError("reference GGUF has no architecture metadata")
    architecture = architecture_field.contents()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    selected: list[dict[str, Any]] = []
    writer = gguf.GGUFWriter(
        temporary,
        arch=architecture,
        endianess=reader.endianess,
    )
    writer.data_alignment = int(reader.alignment)
    try:
        _copy_metadata(reader, writer, gguf)
        for tensor in reader.tensors:
            requested = assignment.get(tensor.name)
            if requested is None:
                writer.add_tensor(
                    tensor.name,
                    tensor.data,
                    raw_shape=tensor.data.shape,
                    raw_dtype=tensor.tensor_type,
                    tensor_endianess=reader.endianess,
                )
                continue

            candidate_type = GGMLType(requested)
            metadata = store.metadata(tensor.name, candidate_type)
            candidate_shape = tuple(int(value) for value in metadata["gguf_shape"])
            reference_shape = tuple(int(value) for value in tensor.shape)
            if candidate_shape != reference_shape:
                raise ValueError(
                    f"shape mismatch for {tensor.name}: candidate "
                    f"{candidate_shape}, reference {reference_shape}")
            payload = b"".join(store.iter_payload(tensor.name, candidate_type))
            if len(payload) != int(metadata["payload_bytes"]):
                raise ValueError(f"payload size mismatch for {tensor.name}")
            conventional_shape = tuple(reversed(candidate_shape))
            quant_type = gguf.GGMLQuantizationType(int(candidate_type))
            byte_shape = gguf.quant_shape_to_byte_shape(
                conventional_shape, quant_type)
            packed = np.frombuffer(payload, dtype=np.uint8).reshape(byte_shape)
            writer.add_tensor(
                tensor.name,
                packed,
                raw_dtype=quant_type,
                tensor_endianess=reader.endianess,
            )
            selected.append({
                "tensor": tensor.name,
                "ggml_type": candidate_type.name,
                "ggml_type_id": int(candidate_type),
                "gguf_shape": list(candidate_shape),
                "payload_bytes": len(payload),
                "sha256": metadata["sha256"],
            })

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        writer.close()
        temporary.unlink(missing_ok=True)
    return selected
