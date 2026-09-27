"""Materialize exact native candidates into a GGUF container.

The final production writer will add resumability and model-scale streaming.
This module supplies the smaller invariant needed first: a selected candidate
payload is framed by pinned ``gguf-py`` without being decoded or requantized,
and every non-selected tensor is copied unchanged from a reference GGUF.
"""

from __future__ import annotations

import importlib
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from native_store import NativeCandidateStore
from quant.ggml_native import GGMLType


RESUMABLE_GGUF_SCHEMA = 1


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


def _sha256_file(path: Path, *, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = limit
    with path.open("rb") as handle:
        while remaining is None or remaining > 0:
            chunk = handle.read(
                16 << 20 if remaining is None else min(16 << 20, remaining))
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    if remaining not in (None, 0):
        raise ValueError(f"file is shorter than expected: {path}")
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stream_array_bytes(
    array: np.ndarray[Any, Any], chunk_bytes: int,
):
    if not array.flags.c_contiguous:
        raise ValueError("reference GGUF tensor data is not C-contiguous")
    view = memoryview(array).cast("B")
    try:
        for start in range(0, len(view), chunk_bytes):
            yield view[start:start + chunk_bytes].tobytes()
    finally:
        view.release()


def _verify_completed_staging(
    staging: Path,
    state: Mapping[str, Any],
    alignment: int,
) -> None:
    committed = int(state["committed_bytes"])
    actual_size = staging.stat().st_size
    if actual_size < committed:
        raise ValueError("resumable GGUF staging file is shorter than committed state")
    if _sha256_file(staging, limit=int(state["header_bytes"])) != state[
        "header_sha256"
    ]:
        raise ValueError("resumable GGUF header checksum differs")
    with staging.open("rb") as handle:
        for record in state["completed_tensors"]:
            handle.seek(int(record["data_offset"]))
            remaining = int(record["payload_bytes"])
            digest = hashlib.sha256()
            while remaining:
                chunk = handle.read(min(16 << 20, remaining))
                if not chunk:
                    raise ValueError(
                        f"resumable tensor {record['tensor']!r} is truncated")
                digest.update(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError(
                    f"resumable tensor {record['tensor']!r} checksum differs")
            padded_stop = (
                (int(record["payload_bytes"]) + alignment - 1)
                // alignment * alignment)
            padding = padded_stop - int(record["payload_bytes"])
            if padding and handle.read(padding) != bytes(padding):
                raise ValueError(
                    f"resumable tensor {record['tensor']!r} padding differs")


def write_resumable_selected_native_gguf(
    reference_path: str | os.PathLike,
    output_path: str | os.PathLike,
    store: NativeCandidateStore,
    assignment: Mapping[str, GGMLType | int],
    *,
    gguf_python: str | os.PathLike,
    resume: bool = False,
    chunk_bytes: int = 8 << 20,
    stop_after_tensors: int | None = None,
) -> dict[str, Any]:
    """Stream, checkpoint, and atomically publish a selected native GGUF.

    The immutable header and tensor-info table are written first. Tensor bytes
    are then appended in canonical reference order and fsynced individually.
    A sidecar state file is published only after each tensor is durable, so a
    resume can verify every committed payload and discard only an uncommitted
    tail left by interruption.
    """
    if not assignment:
        raise ValueError("assignment must select at least one tensor")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if stop_after_tensors is not None and stop_after_tensors < 0:
        raise ValueError("stop_after_tensors must be non-negative")

    gguf = import_pinned_gguf(gguf_python)
    reference = Path(reference_path).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.partial")
    state_path = output.with_name(f".{output.name}.state.json")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed GGUF: {output}")
    if staging.exists() != state_path.exists():
        raise ValueError("resumable GGUF staging file and state are inconsistent")
    if staging.exists() and not resume:
        raise FileExistsError(
            f"resumable GGUF staging state already exists: {state_path}")
    if resume and not staging.exists():
        raise FileNotFoundError("no resumable GGUF staging state exists")

    reader = gguf.GGUFReader(reference)
    tensors = list(reader.tensors)
    tensor_by_name = {tensor.name: tensor for tensor in tensors}
    unknown = sorted(set(assignment) - set(tensor_by_name))
    if unknown:
        raise ValueError(f"assignment names are absent from reference: {unknown}")
    normalized_assignment = {
        name: GGMLType(value) for name, value in assignment.items()
    }

    tensor_plan = []
    for tensor in tensors:
        selected_type = normalized_assignment.get(tensor.name)
        if selected_type is None:
            tensor_plan.append({
                "tensor": tensor.name,
                "source": "reference",
                "ggml_type_id": int(tensor.tensor_type),
                "gguf_shape": [int(value) for value in tensor.shape],
                "payload_bytes": int(tensor.n_bytes),
            })
            continue
        metadata = store.metadata(tensor.name, selected_type)
        candidate_shape = tuple(int(value) for value in metadata["gguf_shape"])
        reference_shape = tuple(int(value) for value in tensor.shape)
        if candidate_shape != reference_shape:
            raise ValueError(
                f"shape mismatch for {tensor.name}: candidate "
                f"{candidate_shape}, reference {reference_shape}")
        tensor_plan.append({
            "tensor": tensor.name,
            "source": "candidate",
            "ggml_type": selected_type.name,
            "ggml_type_id": int(selected_type),
            "gguf_shape": list(candidate_shape),
            "payload_bytes": int(metadata["payload_bytes"]),
            "sha256": metadata["sha256"],
        })

    reference_identity = {
        "path": str(reference),
        "bytes": reference.stat().st_size,
        "sha256": _sha256_file(reference),
    }
    index_path = store.root / "native-candidate-index.json"
    plan = {
        "reference": reference_identity,
        "candidate_index_sha256": _sha256_file(index_path),
        "alignment": int(reader.alignment),
        "assignment": {
            name: selected_type.name
            for name, selected_type in sorted(normalized_assignment.items())
        },
        "tensors": tensor_plan,
    }
    plan_sha256 = _json_sha256(plan)

    if staging.exists():
        with state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("schema") != RESUMABLE_GGUF_SCHEMA:
            raise ValueError("resumable GGUF state schema differs")
        if state.get("format") != "resumable-native-gguf":
            raise ValueError("resumable GGUF state format differs")
        if state.get("plan_sha256") != plan_sha256:
            raise ValueError("resumable GGUF construction plan differs")
        completed = state.get("completed_tensors")
        if not isinstance(completed, list):
            raise ValueError("resumable GGUF state has no completed inventory")
        expected_prefix = [item["tensor"] for item in tensor_plan[:len(completed)]]
        if [item.get("tensor") for item in completed] != expected_prefix:
            raise ValueError("resumable GGUF completed inventory is not a prefix")
        _verify_completed_staging(staging, state, int(reader.alignment))
        committed = int(state["committed_bytes"])
        if staging.stat().st_size != committed:
            with staging.open("r+b") as handle:
                handle.truncate(committed)
                handle.flush()
                os.fsync(handle.fileno())
    else:
        architecture_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
        if architecture_field is None:
            raise ValueError("reference GGUF has no architecture metadata")
        writer = gguf.GGUFWriter(
            staging,
            arch=architecture_field.contents(),
            endianess=reader.endianess,
        )
        writer.data_alignment = int(reader.alignment)
        try:
            _copy_metadata(reader, writer, gguf)
            for tensor, item in zip(tensors, tensor_plan):
                if item["source"] == "reference":
                    writer.add_tensor_info(
                        tensor.name,
                        tensor.data.shape,
                        tensor.data.dtype,
                        int(tensor.n_bytes),
                        raw_dtype=tensor.tensor_type,
                    )
                else:
                    quant_type = gguf.GGMLQuantizationType(item["ggml_type_id"])
                    conventional_shape = tuple(reversed(item["gguf_shape"]))
                    byte_shape = gguf.quant_shape_to_byte_shape(
                        conventional_shape, quant_type)
                    writer.add_tensor_info(
                        tensor.name,
                        byte_shape,
                        np.dtype(np.uint8),
                        int(item["payload_bytes"]),
                        raw_dtype=quant_type,
                    )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()
            assert writer.fout is not None and len(writer.fout) == 1
            handle = writer.fout[0]
            writer.write_padding(handle, handle.tell())
            handle.flush()
            os.fsync(handle.fileno())
            header_bytes = handle.tell()
        finally:
            writer.close()
        state = {
            "schema": RESUMABLE_GGUF_SCHEMA,
            "format": "resumable-native-gguf",
            "plan_sha256": plan_sha256,
            "header_bytes": header_bytes,
            "header_sha256": _sha256_file(staging),
            "committed_bytes": header_bytes,
            "completed_tensors": [],
        }
        _atomic_json(state_path, state)

    completed_count = len(state["completed_tensors"])
    if stop_after_tensors is not None:
        write_limit = min(stop_after_tensors, len(tensors))
    else:
        write_limit = len(tensors)
    max_copy_chunk = 0
    with staging.open("ab") as handle:
        for index in range(completed_count, write_limit):
            tensor = tensors[index]
            item = tensor_plan[index]
            data_offset = handle.tell()
            digest = hashlib.sha256()
            written = 0
            if item["source"] == "candidate":
                chunks = store.iter_payload(
                    tensor.name,
                    normalized_assignment[tensor.name],
                    chunk_bytes=chunk_bytes,
                )
            else:
                chunks = _stream_array_bytes(tensor.data, chunk_bytes)
            for chunk in chunks:
                view = memoryview(chunk).cast("B")
                try:
                    handle.write(view)
                    digest.update(view)
                    written += len(view)
                    max_copy_chunk = max(max_copy_chunk, len(view))
                finally:
                    view.release()
            if written != int(item["payload_bytes"]):
                raise ValueError(
                    f"payload size mismatch for {tensor.name}: "
                    f"{written} != {item['payload_bytes']}")
            actual_sha256 = digest.hexdigest()
            if item["source"] == "candidate" and actual_sha256 != item["sha256"]:
                raise ValueError(f"candidate checksum mismatch for {tensor.name}")
            padding = (-written) % int(reader.alignment)
            if padding:
                handle.write(bytes(padding))
            handle.flush()
            os.fsync(handle.fileno())
            record = {
                **item,
                "data_offset": data_offset,
                "sha256": actual_sha256,
                "aligned_bytes": written + padding,
            }
            state["completed_tensors"].append(record)
            state["committed_bytes"] = handle.tell()
            _atomic_json(state_path, state)

    if len(state["completed_tensors"]) != len(tensors):
        return {
            "status": "incomplete",
            "output": str(output),
            "staging": str(staging),
            "state": str(state_path),
            "tensor_count": len(tensors),
            "completed_tensor_count": len(state["completed_tensors"]),
            "selected_tensor_count": len(normalized_assignment),
            "committed_bytes": int(state["committed_bytes"]),
            "max_copy_chunk_bytes": max_copy_chunk,
            "plan_sha256": plan_sha256,
        }

    with staging.open("rb") as handle:
        os.fsync(handle.fileno())
    output_sha256 = _sha256_file(staging)
    output_bytes = staging.stat().st_size
    os.replace(staging, output)
    state_path.unlink()
    directory_fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {
        "status": "complete",
        "output": str(output),
        "tensor_count": len(tensors),
        "selected_tensor_count": len(normalized_assignment),
        "output_bytes": output_bytes,
        "output_sha256": output_sha256,
        "max_copy_chunk_bytes": max_copy_chunk,
        "plan_sha256": plan_sha256,
        "tensors": state["completed_tensors"],
    }


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
