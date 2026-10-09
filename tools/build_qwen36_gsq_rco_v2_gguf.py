#!/usr/bin/env python3
"""Phase 4 of RCO_PLAN_NEW.md: the pruned GSQ-E6 base GGUF (P24, ungated).

It starts from the authentic GSQ GGUF and the promoted Phase 3 pruning mask
(24 of 256 experts pruned in every layer), and it writes:

* ``qwen35moe.expert_count`` = 232 (``expert_used_count`` stays 8), and
  ``general.size_label`` with the same length, so the header length is
  unchanged.  All other metadata is copied as is.
* For each per-expert tensor (``ffn_{gate,up,down}_exps`` in Q2_0 and the F32
  router ``ffn_gate_inp``), the kept experts in ascending original order.
  The expert dimension is outermost, so every kept expert is one contiguous
  byte slice of authentic GSQ.  Each slice's sha256 is recorded, and
  validation compares every slice byte for byte with its source.
* ``token_embd`` as the checksummed BF16-derived Q6_K payload from Phase 1.
* Every other tensor copied byte for byte from authentic GSQ.

Nothing is decoded or requantized.  This is the zero-upgrade base for the
llama.cpp parity check, not a release artifact.

``--unpruned`` builds the Phase 4 parity control instead: GSQ-E6 with all 256
experts, which is authentic GSQ with only ``token_embd`` replaced.  No mask is
read and no metadata is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from native_gguf import (  # noqa: E402
    _atomic_json,
    _sha256_file,
    _stream_array_bytes,
    import_pinned_gguf,
)

SCHEMA = "rco.qwen36.gsq_e6_p24_gguf.v1"
LAYERS = 40
EXPERTS = 256
PRUNE_PER_LAYER = 24
KEPT = EXPERTS - PRUNE_PER_LAYER
EXPERT_TENSORS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_inp")
EXPECTED_BYTES = 10_760_624_832
UNPRUNED_EXPECTED_BYTES = 11_617_835_712
PRUNED_OVERRIDES = {
    "qwen35moe.expert_count": KEPT,
    "general.size_label": "232x2.6B",
}
WARNING = (
    "Ungated model for the Phase 4 llama.cpp parity check. It is not a release "
    "GSQ-RCO model.")


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _expert_tensor(name: str) -> int | None:
    """Return the layer of a per-expert tensor, or None."""
    parts = name.split(".")
    if len(parts) == 4 and parts[0] == "blk" and parts[2] in EXPERT_TENSORS:
        return int(parts[1])
    return None


def _copy_metadata(reader: Any, writer: Any, gguf: Any,
                   overrides: dict[str, Any]) -> None:
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        value = overrides.get(field.name, field.contents())
        writer.add_key_value(field.name, value, value_type, sub_type=sub_type)


def _sliced_shape(data: np.ndarray) -> tuple[int, ...]:
    if data.shape[0] != EXPERTS:
        raise ValueError(f"expert dimension is not outermost: {data.shape}")
    return (KEPT, *data.shape[1:])


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    gguf = import_pinned_gguf(args.gguf_python)

    pruned = not args.unpruned
    overrides = PRUNED_OVERRIDES if pruned else {}
    expected_bytes = EXPECTED_BYTES if pruned else UNPRUNED_EXPECTED_BYTES
    inputs: dict[str, Any] = {}
    mask = np.zeros((LAYERS, EXPERTS), dtype=bool)
    if pruned:
        # The promoted mask must be the one that step g replayed.
        replay = _load_json(args.replay_report)
        if replay.get("status") != "complete":
            raise ValueError("replay report is not complete")
        mask_sha256 = _sha256_file(args.mask)
        if mask_sha256 != replay["mask"]["sha256"]:
            raise ValueError("mask differs from the replayed mask")
        mask = np.load(args.mask)
        if mask.shape != (LAYERS, EXPERTS) or mask.dtype != bool:
            raise ValueError("mask must be a 40 x 256 boolean array")
        if not bool((mask.sum(axis=1) == PRUNE_PER_LAYER).all()):
            raise ValueError(f"mask must prune {PRUNE_PER_LAYER} experts per layer")
        inputs["mask"] = {"path": str(args.mask), "sha256": mask_sha256}
        inputs["replay_report"] = {"path": str(args.replay_report),
                                   "sha256": _sha256_file(args.replay_report)}
    kept = [np.flatnonzero(~mask[layer]) for layer in range(LAYERS)]

    embedding = _load_json(args.embedding_report)
    embd_path = Path(embedding["payload"]["path"])
    embd_sha256 = _sha256_file(embd_path)
    if embd_sha256 != embedding["payload"]["sha256"]:
        raise ValueError("Q6_K token_embd payload differs from its Phase 1 report")
    embd_type = gguf.GGMLQuantizationType[embedding["payload"]["ggml_type"]]
    embd_shape = [int(value) for value in embedding["payload"]["gguf_shape"]]
    embd_bytes = int(embedding["payload"]["payload_bytes"])
    if embd_path.stat().st_size != embd_bytes:
        raise ValueError("Q6_K token_embd payload size differs from its report")

    gsq_path = args.gguf.resolve(strict=True)
    gsq_sha256 = _sha256_file(gsq_path)
    if gsq_sha256 != embedding["source"]["gsq_gguf_sha256"]:
        raise ValueError("GSQ GGUF differs from the one Phase 1 was built against")
    reader = gguf.GGUFReader(gsq_path)
    tensors = list(reader.tensors)
    alignment = int(reader.alignment)

    output = args.output_model.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed GGUF: {output}")
    staging = output.with_name(f".{output.name}.partial")
    staging.unlink(missing_ok=True)

    writer = gguf.GGUFWriter(
        staging, arch=reader.get_field(gguf.Keys.General.ARCHITECTURE).contents(),
        endianess=reader.endianess)
    writer.data_alignment = alignment
    plan = []
    try:
        _copy_metadata(reader, writer, gguf, overrides)
        for tensor in tensors:
            layer = _expert_tensor(tensor.name) if pruned else None
            if tensor.name == "token_embd.weight":
                if [int(value) for value in tensor.shape] != embd_shape:
                    raise ValueError("token_embd shape differs from the Q6_K payload")
                byte_shape = gguf.quant_shape_to_byte_shape(
                    tuple(reversed(embd_shape)), embd_type)
                writer.add_tensor_info(tensor.name, byte_shape, np.dtype(np.uint8),
                                       embd_bytes, raw_dtype=embd_type)
                plan.append({"tensor": tensor.name, "source": "gsq_e6_q6_k_token_embd",
                             "ggml_type": embd_type.name, "payload_bytes": embd_bytes,
                             "replaced_gsq_type": tensor.tensor_type.name})
            elif layer is not None:
                slice_bytes = int(tensor.n_bytes) // EXPERTS
                writer.add_tensor_info(tensor.name, _sliced_shape(tensor.data),
                                       tensor.data.dtype, slice_bytes * KEPT,
                                       raw_dtype=tensor.tensor_type)
                plan.append({"tensor": tensor.name, "source": "authentic_gsq_expert_slices",
                             "ggml_type": tensor.tensor_type.name, "layer": layer,
                             "slice_bytes": slice_bytes,
                             "payload_bytes": slice_bytes * KEPT})
            else:
                writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                                       int(tensor.n_bytes), raw_dtype=tensor.tensor_type)
                plan.append({"tensor": tensor.name, "source": "authentic_gsq",
                             "ggml_type": tensor.tensor_type.name,
                             "payload_bytes": int(tensor.n_bytes)})
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        handle = writer.fout[0]
        writer.write_padding(handle, handle.tell())
        header_bytes = handle.tell()

        for index, (tensor, item) in enumerate(zip(tensors, plan)):
            digest = hashlib.sha256()
            written = 0
            if item["source"] == "gsq_e6_q6_k_token_embd":
                with embd_path.open("rb") as source:
                    while chunk := source.read(args.chunk_bytes):
                        handle.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
            elif item["source"] == "authentic_gsq_expert_slices":
                slice_hashes = []
                for expert in kept[item["layer"]]:
                    chunk = np.ascontiguousarray(tensor.data[expert]).tobytes()
                    if len(chunk) != item["slice_bytes"]:
                        raise ValueError(f"slice size differs for {tensor.name}")
                    handle.write(chunk)
                    digest.update(chunk)
                    slice_hashes.append(hashlib.sha256(chunk).hexdigest())
                    written += len(chunk)
                item["kept_slice_sha256"] = slice_hashes
            else:
                for chunk in _stream_array_bytes(tensor.data, args.chunk_bytes):
                    handle.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
            if written != item["payload_bytes"]:
                raise ValueError(f"payload size differs for {tensor.name}")
            item["sha256"] = digest.hexdigest()
            padding = (-written) % alignment
            if padding:
                handle.write(bytes(padding))
            if index % 100 == 0 or item["source"] == "gsq_e6_q6_k_token_embd":
                print(json.dumps({"tensor_index": index, "tensor": tensor.name,
                                  "source": item["source"]}), flush=True)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        writer.close()

    output_bytes = staging.stat().st_size
    if output_bytes != expected_bytes:
        raise ValueError(f"output size {output_bytes} differs from the expected {expected_bytes}")
    os.replace(staging, output)
    directory = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(json.dumps({"written": str(output), "bytes": output_bytes}), flush=True)

    validation = validate(gguf, reader, output, plan, kept, embd_path, overrides, pruned)
    return {
        "schema": SCHEMA,
        "status": "built_pending_llama_cpp_parity",
        "variant": "p24" if pruned else "unpruned_parity_control",
        "warning": WARNING,
        "inputs": {
            "gsq_gguf": {"path": str(gsq_path), "sha256": gsq_sha256},
            **inputs,
            "embedding_report": {"path": str(args.embedding_report),
                                 "sha256": _sha256_file(args.embedding_report)},
            "token_embd_q6_k": {"path": str(embd_path), "sha256": embd_sha256},
        },
        "metadata_overrides": overrides,
        "pruned_experts_per_layer": [np.flatnonzero(mask[layer]).tolist()
                                     for layer in range(LAYERS)],
        "output": {"path": str(output), "bytes": output_bytes,
                   "expected_bytes": expected_bytes, "header_bytes": header_bytes,
                   "sha256": validation.pop("output_sha256")},
        "validation": validation,
        "wall_seconds": time.monotonic() - started,
        "tensors": plan,
    }


def validate(gguf: Any, gsq: Any, output: Path, plan: list[dict[str, Any]],
             kept: list[np.ndarray], embd_path: Path, overrides: dict[str, Any],
             pruned: bool) -> dict[str, Any]:
    """Re-read the written GGUF and check it against its sources."""
    model = gguf.GGUFReader(output)
    gsq_fields = {name: (field.types, field.contents())
                  for name, field in gsq.fields.items() if not name.startswith("GGUF.")}
    model_fields = {name: (field.types, field.contents())
                    for name, field in model.fields.items() if not name.startswith("GGUF.")}
    if set(gsq_fields) != set(model_fields):
        raise ValueError("metadata keys differ from authentic GSQ")
    for name, (types, value) in gsq_fields.items():
        expected = (types, overrides.get(name, value))
        if model_fields[name] != expected:
            raise ValueError(f"metadata differs for {name}")
    if [tensor.name for tensor in model.tensors] != [tensor.name for tensor in gsq.tensors]:
        raise ValueError("tensor order differs from authentic GSQ")

    counts = {"retained_tensors_byte_identical": 0,
              "expert_tensors_slice_identical": 0,
              "expert_slices_compared": 0}
    for original, written, item in zip(gsq.tensors, model.tensors, plan):
        if written.tensor_type.name != item["ggml_type"]:
            raise ValueError(f"type differs for {written.name}")
        if int(written.n_bytes) != item["payload_bytes"]:
            raise ValueError(f"size differs for {written.name}")
        if item["source"] == "gsq_e6_q6_k_token_embd":
            on_disk = np.memmap(embd_path, dtype=np.uint8, mode="r")
            if not np.array_equal(written.data.reshape(-1), on_disk):
                raise ValueError("token_embd differs from the Q6_K payload")
            if [int(v) for v in written.shape] != [int(v) for v in original.shape]:
                raise ValueError("token_embd logical shape differs")
            continue
        if item["source"] == "authentic_gsq_expert_slices":
            expected_shape = [int(v) for v in original.shape[:-1]] + [KEPT]
            if [int(v) for v in written.shape] != expected_shape:
                raise ValueError(f"sliced shape differs for {written.name}")
            if written.data.shape[0] != KEPT:
                raise ValueError(f"expert dimension differs for {written.name}")
            for position, expert in enumerate(kept[item["layer"]]):
                if not np.array_equal(written.data[position], original.data[expert]):
                    raise ValueError(f"slice {expert} differs for {written.name}")
                counts["expert_slices_compared"] += 1
            counts["expert_tensors_slice_identical"] += 1
            continue
        if [int(v) for v in written.shape] != [int(v) for v in original.shape]:
            raise ValueError(f"shape differs for {written.name}")
        if not np.array_equal(written.data, original.data):
            raise ValueError(f"retained tensor bytes differ: {written.name}")
        counts["retained_tensors_byte_identical"] += 1
    if pruned and counts["expert_tensors_slice_identical"] != LAYERS * len(EXPERT_TENSORS):
        raise ValueError("not every per-expert tensor was sliced")
    return {"metadata_identical_except_overrides": True,
            "tensor_order_identical": True,
            "token_embd_q6_k_byte_identical": True,
            **counts,
            "output_sha256": _sha256_file(output)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mask", type=Path,
                        default=ROOT / "data/qwen36_prune24/consensus_mask.npy")
    parser.add_argument("--replay-report", type=Path,
                        default=RCO / "reports/qwen36_gsq_e6_prune24_score_replay_consensus.json")
    parser.add_argument("--embedding-report", type=Path,
                        default=RCO / "reports/qwen36_gsq_e6_embedding.json")
    parser.add_argument("--gguf", type=Path,
                        default=ROOT / "results/Qwen3.6-35B-A3B-GSQ-hybrid.gguf")
    parser.add_argument("--gguf-python", type=Path, default=ROOT / "repos/llama.cpp/gguf-py")
    parser.add_argument("--unpruned", action="store_true",
                        help="build the unpruned GSQ-E6 parity control")
    parser.add_argument("--output-model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-bytes", type=int, default=64 << 20)
    args = parser.parse_args()
    variant = "GSQ-E6" if args.unpruned else "GSQ-E6-P24"
    if args.output_model is None:
        args.output_model = ROOT / f"results/Qwen3.6-35B-A3B-{variant}-UNGATED.gguf"
    if args.output is None:
        stem = "qwen36_gsq_e6" if args.unpruned else "qwen36_gsq_e6_p24"
        args.output = RCO / f"reports/{stem}_gguf.json"
    for key in ("mask", "replay_report", "embedding_report", "gguf", "gguf_python",
                "output_model", "output"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main() -> int:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output}")
    report = build(args)
    _atomic_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "output", "validation")}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
