#!/usr/bin/env python3
"""Construct the ungated GSQ-RCO evaluation candidate from a replayed assignment.

The candidate exists only so the remaining release gates can execute it in
unmodified pinned llama.cpp. It is not a release artifact. Every retained
tensor is copied byte-for-byte from the authentic GSQ GGUF, every selected
``Q2_0 -> Q4_0`` upgrade is copied from the verified native candidate store,
and every selected ``Q8_0 -> BF16`` upgrade streams the pinned BF16 source
through the canonical converter row order without any arithmetic. Nothing is
decoded and requantized.

Construction is resumable: the header is written first and each tensor payload
is fsynced before a sidecar state records it. After publication the output is
re-read and validated independently, including an inverse-layout comparison of
every BF16 upgrade against its safetensors source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from native_gguf import (  # noqa: E402
    _atomic_json,
    _copy_metadata,
    _json_sha256,
    _sha256_file,
    _stream_array_bytes,
    _verify_completed_staging,
    import_pinned_gguf,
)
from native_store import NativeCandidateStore  # noqa: E402
from quant.ggml_native import GGMLNativeCodec, GGMLType  # noqa: E402
from qwen35_native import (  # noqa: E402
    Qwen35LinearAttentionGeometry,
    SafetensorGGUFRowSource,
    restore_source_matrix,
)


SCHEMA = "rco.qwen36.gsq_rco_candidate_gguf.v1"
STATE_SCHEMA = "rco.qwen36.gsq_rco_candidate_gguf_state.v1"
REPLAY_STATUS = "independent_replay_complete_pending_release_gates"
GGML_TYPE_BF16 = 30
WARNING = (
    "Ungated evaluation candidate. It must not be published or described as a "
    "release GSQ-RCO model until pinned-llama.cpp parity, held-out paired "
    "quality, repeat-NLL, deterministic generation, and CUDA/offload gates pass."
)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def bf16_bytes_from_exact_float32(values: np.ndarray) -> bytes:
    """Pack float32 values that are exactly representable in BF16.

    The pinned source is BF16, so the float32 rows produced by the canonical
    row source carry zero low mantissa bits. Any nonzero low bits mean the
    values were changed by arithmetic, which this candidate must never do.
    """
    array = np.ascontiguousarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    if np.any(bits & np.uint32(0xFFFF)):
        raise ValueError("float32 values are not exactly representable in BF16")
    return (bits >> np.uint32(16)).astype("<u2").tobytes()


def bf16_bytes_to_float32(payload: bytes | np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.frombuffer(memoryview(payload).cast("B"), dtype="<u2")
    if raw.size != int(np.prod(shape)):
        raise ValueError(f"BF16 payload has {raw.size} values, expected {shape}")
    return (raw.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(shape)


def _rechunk(chunks: Iterator[bytes], chunk_bytes: int) -> Iterator[bytes]:
    pending = bytearray()
    for chunk in chunks:
        pending += chunk
        while len(pending) >= chunk_bytes:
            yield bytes(pending[:chunk_bytes])
            del pending[:chunk_bytes]
    if pending:
        yield bytes(pending)


class CandidatePayloads:
    """Resolve each selected upgrade to exact payload bytes."""

    def __init__(
        self,
        policy_records: dict[str, dict[str, Any]],
        manifest_entries: dict[str, dict[str, Any]],
        store: NativeCandidateStore,
        model_dir: Path,
        *,
        rows_per_chunk: int,
    ) -> None:
        self.policy = policy_records
        self.entries = manifest_entries
        self.store = store
        self.source = SafetensorGGUFRowSource(model_dir)
        self.rows_per_chunk = rows_per_chunk

    def ggml_type_id(self, name: str) -> int:
        upgrade = self.policy[name]["upgrade"]
        if upgrade["kind"] == "bf16_derived_native_candidate":
            if upgrade["ggml_type"] != "Q4_0":
                raise ValueError(f"unexpected native upgrade type for {name}")
            return int(GGMLType.Q4_0)
        if upgrade["kind"] == "pinned_bf16_source":
            if upgrade["ggml_type"] != "BF16":
                raise ValueError(f"unexpected dense upgrade type for {name}")
            entry = self.entries[name]
            if entry.get("source_view") is not None:
                raise ValueError(f"BF16 upgrade has an unexpected view: {name}")
            if upgrade["source_tensor"] != entry["source_name"]:
                raise ValueError(f"BF16 source name differs for {name}")
            if upgrade["source_shard"] != entry["source_shard"]:
                raise ValueError(f"BF16 source shard differs for {name}")
            return GGML_TYPE_BF16
        raise ValueError(f"unsupported upgrade kind for {name}")

    def iter_payload(self, name: str, chunk_bytes: int) -> Iterator[bytes]:
        if self.ggml_type_id(name) == int(GGMLType.Q4_0):
            yield from self.store.iter_payload(
                name, GGMLType.Q4_0, chunk_bytes=chunk_bytes)
            return
        rows = self.source.iter_rows(
            self.entries[name], rows_per_chunk=self.rows_per_chunk)
        yield from _rechunk(
            (bf16_bytes_from_exact_float32(chunk) for chunk in rows), chunk_bytes)

    def digest(self, name: str, chunk_bytes: int) -> tuple[str, int]:
        if self.ggml_type_id(name) == int(GGMLType.Q4_0):
            metadata = self.store.metadata(name, GGMLType.Q4_0)
            return str(metadata["sha256"]), int(metadata["payload_bytes"])
        digest = hashlib.sha256()
        size = 0
        for chunk in self.iter_payload(name, chunk_bytes):
            digest.update(chunk)
            size += len(chunk)
        return digest.hexdigest(), size


def _selected_assignment(
    replay: dict[str, Any], reference: dict[str, Any],
) -> list[str]:
    if replay.get("status") != REPLAY_STATUS:
        raise ValueError(f"replay status is {replay.get('status')!r}")
    if replay.get("replay", {}).get("passed") is not True:
        raise ValueError("independent replay did not pass")
    bits = replay["selected"]["assignment_bits"]
    if bits != reference["selected"]["assignment_bits"]:
        raise ValueError("replay and reference selections differ")
    names = replay["budget"]["tensor_names"]
    if names != reference["budget"]["tensor_names"] or len(names) != len(bits):
        raise ValueError("replay and reference upgrade inventories differ")
    if set(bits) - {"0", "1"}:
        raise ValueError("assignment bits must be binary")
    return [name for name, bit in zip(names, bits) if bit == "1"]


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    gguf = import_pinned_gguf(args.gguf_python)
    replay_path = args.search_report.resolve(strict=True)
    reference_path = args.reference_report.resolve(strict=True)
    replay = _load_json(replay_path)
    reference_sha256 = _sha256_file(reference_path)
    if replay["replay"]["reference_sha256"] != reference_sha256:
        raise ValueError("replay was not made against this reference report")
    reference = _load_json(reference_path)
    selected = _selected_assignment(replay, reference)
    problem = replay["problem"]

    identities = {
        "policy": (args.policy, problem["policy_sha256"]),
        "manifest": (args.manifest, problem["manifest_sha256"]),
        "gsq_gguf": (args.gguf, problem["gguf_sha256"]),
        "candidate_store_index": (
            args.store / "native-candidate-index.json",
            problem["candidate_store_index_sha256"],
        ),
    }
    resolved = {}
    for label, (path, expected) in identities.items():
        path = path.resolve(strict=True)
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(f"{label} SHA-256 differs: {actual} != {expected}")
        resolved[label] = {"path": str(path), "sha256": actual}

    policy = _load_json(args.policy)
    manifest = _load_json(args.manifest)
    policy_records = {
        record["destination_name"]: record for record in policy["tensors"]}
    manifest_entries = {
        entry["destination_name"]: entry for entry in manifest["entries"]}
    for name in selected:
        upgrade = policy_records[name].get("upgrade") or {}
        if upgrade.get("policy_status") != "admitted":
            raise ValueError(f"selected tensor is not an admitted upgrade: {name}")

    codec = GGMLNativeCodec(args.ggml_library)
    store = NativeCandidateStore(args.store, codec)
    payloads = CandidatePayloads(
        policy_records, manifest_entries, store, args.model_dir,
        rows_per_chunk=args.rows_per_chunk)

    gsq_path = args.gguf.resolve(strict=True)
    reader = gguf.GGUFReader(gsq_path)
    tensors = list(reader.tensors)
    by_name = {tensor.name: tensor for tensor in tensors}
    missing = sorted(set(selected) - set(by_name))
    if missing:
        raise ValueError(f"selected tensors are absent from GSQ GGUF: {missing}")
    alignment = int(reader.alignment)

    selected_set = set(selected)
    tensor_plan = []
    for tensor in tensors:
        record = policy_records.get(tensor.name)
        retain = (record or {}).get("retain")
        if retain is not None:
            if int(retain["payload_bytes"]) != int(tensor.n_bytes):
                raise ValueError(f"policy retain size differs for {tensor.name}")
            if int(retain["payload_offset"]) != int(tensor.data_offset):
                raise ValueError(f"policy retain offset differs for {tensor.name}")
            if retain["ggml_type"] != tensor.tensor_type.name:
                raise ValueError(f"policy retain type differs for {tensor.name}")
        if tensor.name not in selected_set:
            tensor_plan.append({
                "tensor": tensor.name,
                "source": "authentic_gsq",
                "ggml_type": tensor.tensor_type.name,
                "ggml_type_id": int(tensor.tensor_type),
                "gguf_shape": [int(value) for value in tensor.shape],
                "payload_bytes": int(tensor.n_bytes),
            })
            continue
        upgrade = record["upgrade"]
        type_id = payloads.ggml_type_id(tensor.name)
        shape = [int(value) for value in manifest_entries[tensor.name][
            "destination_gguf_shape"]]
        if shape != [int(value) for value in tensor.shape]:
            raise ValueError(f"upgrade shape differs for {tensor.name}")
        sha256, size = payloads.digest(tensor.name, args.chunk_bytes)
        if size != int(upgrade["payload_bytes"]):
            raise ValueError(
                f"upgrade payload size differs for {tensor.name}: "
                f"{size} != {upgrade['payload_bytes']}")
        tensor_plan.append({
            "tensor": tensor.name,
            "source": upgrade["kind"],
            "ggml_type": gguf.GGMLQuantizationType(type_id).name,
            "ggml_type_id": type_id,
            "gguf_shape": shape,
            "payload_bytes": size,
            "sha256": sha256,
            "replaced_gsq_type": tensor.tensor_type.name,
        })

    plan = {
        "schema": SCHEMA,
        "inputs": resolved,
        "replay_report_sha256": _sha256_file(replay_path),
        "reference_report_sha256": reference_sha256,
        "alignment": alignment,
        "selected": sorted(selected),
        "tensors": tensor_plan,
    }
    plan_sha256 = _json_sha256(plan)

    output = args.output_model.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed GGUF: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.partial")
    state_path = output.with_name(f".{output.name}.state.json")
    if staging.exists() != state_path.exists():
        raise ValueError("staging file and state are inconsistent")

    if staging.exists():
        state = _load_json(state_path)
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("staging state schema differs")
        if state.get("plan_sha256") != plan_sha256:
            raise ValueError("staging construction plan differs")
        completed = [item["tensor"] for item in state["completed_tensors"]]
        if completed != [item["tensor"] for item in tensor_plan[:len(completed)]]:
            raise ValueError("staging inventory is not a plan prefix")
        _verify_completed_staging(staging, state, alignment)
        committed = int(state["committed_bytes"])
        if staging.stat().st_size != committed:
            with staging.open("r+b") as handle:
                handle.truncate(committed)
                handle.flush()
                os.fsync(handle.fileno())
        print(json.dumps({"resumed_after_tensors": len(completed)}), flush=True)
    else:
        writer = gguf.GGUFWriter(
            staging,
            arch=reader.get_field(gguf.Keys.General.ARCHITECTURE).contents(),
            endianess=reader.endianess,
        )
        writer.data_alignment = alignment
        try:
            _copy_metadata(reader, writer, gguf)
            for tensor, item in zip(tensors, tensor_plan):
                if item["source"] == "authentic_gsq":
                    writer.add_tensor_info(
                        tensor.name, tensor.data.shape, tensor.data.dtype,
                        int(tensor.n_bytes), raw_dtype=tensor.tensor_type)
                    continue
                quant_type = gguf.GGMLQuantizationType(item["ggml_type_id"])
                byte_shape = gguf.quant_shape_to_byte_shape(
                    tuple(reversed(item["gguf_shape"])), quant_type)
                writer.add_tensor_info(
                    tensor.name, byte_shape, np.dtype(np.uint8),
                    int(item["payload_bytes"]), raw_dtype=quant_type)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()
            handle = writer.fout[0]
            writer.write_padding(handle, handle.tell())
            handle.flush()
            os.fsync(handle.fileno())
            header_bytes = handle.tell()
        finally:
            writer.close()
        state = {
            "schema": STATE_SCHEMA,
            "plan_sha256": plan_sha256,
            "header_bytes": header_bytes,
            "header_sha256": _sha256_file(staging),
            "committed_bytes": header_bytes,
            "completed_tensors": [],
        }
        _atomic_json(state_path, state)

    with staging.open("ab") as handle:
        for index in range(len(state["completed_tensors"]), len(tensors)):
            tensor = tensors[index]
            item = tensor_plan[index]
            if item["source"] == "authentic_gsq":
                chunks = _stream_array_bytes(tensor.data, args.chunk_bytes)
            else:
                chunks = payloads.iter_payload(tensor.name, args.chunk_bytes)
            data_offset = handle.tell()
            digest = hashlib.sha256()
            written = 0
            for chunk in chunks:
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
            if written != item["payload_bytes"]:
                raise ValueError(f"payload size differs for {tensor.name}")
            actual = digest.hexdigest()
            if "sha256" in item and actual != item["sha256"]:
                raise ValueError(f"payload checksum differs for {tensor.name}")
            padding = (-written) % alignment
            if padding:
                handle.write(bytes(padding))
            handle.flush()
            os.fsync(handle.fileno())
            state["completed_tensors"].append({
                **item,
                "data_offset": data_offset,
                "sha256": actual,
                "aligned_bytes": written + padding,
            })
            state["committed_bytes"] = handle.tell()
            _atomic_json(state_path, state)
            if item["source"] != "authentic_gsq" or index % 100 == 0:
                print(json.dumps({
                    "tensor_index": index,
                    "tensor": tensor.name,
                    "source": item["source"],
                }), flush=True)

    output_sha256 = _sha256_file(staging)
    output_bytes = staging.stat().st_size
    os.replace(staging, output)
    state_path.unlink()
    directory_fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

    validation = validate(
        gguf, reader, output, tensor_plan, manifest_entries, args.model_dir)
    expected_bytes = int(replay["budget"]["target_complete_file_bytes"])
    if output_bytes != expected_bytes:
        raise ValueError(
            f"candidate size {output_bytes} differs from budget {expected_bytes}")
    counts: dict[str, int] = {}
    for item in tensor_plan:
        counts[item["source"]] = counts.get(item["source"], 0) + 1
    return {
        "schema": SCHEMA,
        "status": "candidate_constructed_pending_release_gates",
        "warning": WARNING,
        "plan_sha256": plan_sha256,
        "inputs": {
            **resolved,
            "replay_report": {
                "path": str(replay_path),
                "sha256": plan["replay_report_sha256"],
            },
            "reference_report": {
                "path": str(reference_path), "sha256": reference_sha256},
            "model_dir": str(args.model_dir.resolve(strict=True)),
            "dense_revision": problem["dense_revision"],
            "ggml_library_sha256": codec.library_sha256,
        },
        "output": {
            "path": str(output),
            "bytes": output_bytes,
            "sha256": output_sha256,
            "expected_bytes": expected_bytes,
        },
        "selection": {
            "upgrade_count": len(selected),
            "source_counts": counts,
            "incremental_gguf_bytes": output_bytes - int(
                replay["budget"]["mandatory_floor_file_bytes"]),
            "calibration_mean_nll": replay["selected"]["mean_nll"],
            "calibration_delta_mean_nll_from_authentic": replay["selected"][
                "delta_mean_nll_from_authentic"],
        },
        "validation": validation,
        "wall_seconds": time.monotonic() - started,
        "tensors": tensor_plan,
    }


def _tensor_sha256(tensor: Any, chunk_bytes: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    for chunk in _stream_array_bytes(tensor.data, chunk_bytes):
        digest.update(chunk)
    return digest.hexdigest()


def validate(
    gguf: Any,
    gsq: Any,
    output: Path,
    tensor_plan: list[dict[str, Any]],
    manifest_entries: dict[str, dict[str, Any]],
    model_dir: Path,
) -> dict[str, Any]:
    """Re-read the published candidate and check it independently."""
    from safetensors import safe_open

    candidate = gguf.GGUFReader(output)
    gsq_fields = {
        name: (field.types, field.contents())
        for name, field in gsq.fields.items() if not name.startswith("GGUF.")}
    candidate_fields = {
        name: (field.types, field.contents())
        for name, field in candidate.fields.items()
        if not name.startswith("GGUF.")}
    if gsq_fields != candidate_fields:
        raise ValueError("candidate metadata differs from authentic GSQ")
    if [tensor.name for tensor in candidate.tensors] != [
        tensor.name for tensor in gsq.tensors
    ]:
        raise ValueError("candidate tensor order differs from authentic GSQ")

    geometry = Qwen35LinearAttentionGeometry.from_model_dir(model_dir)
    retained = upgraded_native = upgraded_bf16 = 0
    for original, written, item in zip(gsq.tensors, candidate.tensors, tensor_plan):
        if [int(value) for value in written.shape] != item["gguf_shape"]:
            raise ValueError(f"candidate shape differs for {written.name}")
        if int(written.tensor_type) != item["ggml_type_id"]:
            raise ValueError(f"candidate type differs for {written.name}")
        if item["source"] == "authentic_gsq":
            if _tensor_sha256(written) != _tensor_sha256(original):
                raise ValueError(f"retained tensor bytes differ: {written.name}")
            retained += 1
            continue
        if _tensor_sha256(written) != item["sha256"]:
            raise ValueError(f"upgrade bytes differ from plan: {written.name}")
        if item["ggml_type_id"] != GGML_TYPE_BF16:
            upgraded_native += 1
            continue
        entry = manifest_entries[written.name]
        columns, rows = item["gguf_shape"]
        canonical = bf16_bytes_to_float32(written.data, (rows, columns))
        restored = restore_source_matrix(
            canonical, entry["normalized_source_name"], geometry)
        with safe_open(
            model_dir / entry["source_shard"], framework="pt", device="cpu",
        ) as handle:
            source = handle.get_tensor(entry["source_name"])
        import torch
        source = source.to(torch.float32).numpy()
        if source.shape != restored.shape or not np.array_equal(
            source.view(np.uint32), restored.view(np.uint32)
        ):
            raise ValueError(
                f"BF16 upgrade does not invert to its source: {written.name}")
        upgraded_bf16 += 1
    return {
        "metadata_fields_identical": True,
        "tensor_order_identical": True,
        "retained_tensors_byte_identical": retained,
        "native_q4_upgrades_checksum_verified": upgraded_native,
        "bf16_upgrades_inverse_layout_bit_exact": upgraded_bf16,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-report", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--gguf-python", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-chunk", type=int, default=64)
    parser.add_argument("--chunk-bytes", type=int, default=8 << 20)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output}")
    report = build(args)
    _atomic_json(args.output.resolve(), report)
    print(json.dumps({
        key: report[key] for key in ("status", "output", "validation")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
