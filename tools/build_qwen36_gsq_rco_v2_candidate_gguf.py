#!/usr/bin/env python3
"""Phase 8 of RCO_PLAN_NEW.md: the evaluation candidate GGUF (ungated).

The Phase 4 P24 build (``build_qwen36_gsq_rco_v2_gguf.py``) with the
frequency mask, plus the 23 routed-expert tensors promoted in Phase 7
installed as Q3_K:

* ``qwen35moe.expert_count`` = 232 and the same-length ``size_label``, as in
  Phase 4; all other metadata copied as is.
* Per-expert tensors not upgraded: the kept experts' Q2_0 slices of
  authentic GSQ, as in Phase 4.  Router rows likewise.
* The 23 upgraded tensors: the Phase 6 store's 256-expert Q3_K tensor cut to
  the kept experts in ascending order.  The cut must reproduce the store
  report's ``kept_payload`` sha256.
* ``token_embd`` as the Phase 1 Q6_K payload; every other tensor byte for
  byte from authentic GSQ.

The promoted set and the mask come from the passing Phase 7 replay
(``qwen36_prune24_q3k_replay_extended.json``).  Validation re-reads the
written file: metadata, tensor order, every slice against its source, the
exact size, and the GPU-memory budget (all tensors except ``token_embd`` at
most authentic GSQ's bytes).
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

from build_qwen36_gsq_rco_v2_gguf import (  # noqa: E402
    EXPECTED_BYTES as P24_BYTES,
    EXPERTS,
    KEPT,
    LAYERS,
    PRUNE_PER_LAYER,
    PRUNED_OVERRIDES,
    _copy_metadata,
    _expert_tensor,
    _load_json,
    _sliced_shape,
)
from native_gguf import (  # noqa: E402
    _atomic_json,
    _sha256_file,
    _stream_array_bytes,
    import_pinned_gguf,
)

SCHEMA = "rco.qwen36.gsq_rco_v2_candidate_gguf.v1"
UPGRADES = 23
UPGRADE_DELTA_BYTES = 36_110_336
EXPECTED_BYTES = P24_BYTES + UPGRADES * UPGRADE_DELTA_BYTES
WARNING = ("Ungated evaluation candidate for the Phase 8 gates. It is not a release "
           "GSQ-RCO model.")


def _store_slices(store_root: Path, entry: dict[str, Any]) -> np.ndarray:
    """The store's Q3_K tensor as [experts, bytes per expert], memory-mapped."""
    payload = int(entry["payload_bytes"])
    if payload % EXPERTS:
        raise ValueError(f"{entry['path']}: payload not divisible by {EXPERTS} experts")
    data = np.memmap(store_root / entry["path"], dtype=np.uint8, mode="r",
                     offset=int(entry["offset"]), shape=(payload,))
    return data.reshape(EXPERTS, payload // EXPERTS)


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    gguf = import_pinned_gguf(args.gguf_python)

    replay = _load_json(args.replay_report)
    if replay.get("status") != "pass":
        raise ValueError("the Phase 7 replay did not pass its validation gate")
    promoted = sorted(replay["promoted_tensors"])
    if len(promoted) != UPGRADES or len(set(promoted)) != UPGRADES:
        raise ValueError(f"replay promotes {len(promoted)} tensors, not {UPGRADES}")
    store_report = _load_json(args.store_report)
    if store_report.get("status") != "complete":
        raise ValueError("the Phase 6 store is not complete")
    store_root = Path(store_report["store"]["root"])
    if _sha256_file(store_root / "native-candidate-index.json") != store_report["store"]["index_sha256"]:
        raise ValueError("Q3_K store index differs from its report")
    if replay["identity"]["q3k_store_index_sha256"] != store_report["store"]["index_sha256"]:
        raise ValueError("the replay used another Q3_K store")
    mask_path = Path(store_report["mask"]["path"])
    mask_sha256 = _sha256_file(mask_path)
    if mask_sha256 != store_report["mask"]["sha256"] or mask_sha256 != replay["identity"]["mask_sha256"]:
        raise ValueError("mask differs from the store report or the replay")
    mask = np.load(mask_path)
    if mask.shape != (LAYERS, EXPERTS) or mask.dtype != bool:
        raise ValueError("mask must be a 40 x 256 boolean array")
    if not bool((mask.sum(axis=1) == PRUNE_PER_LAYER).all()):
        raise ValueError(f"mask must prune {PRUNE_PER_LAYER} experts per layer")
    kept = [np.flatnonzero(~mask[layer]) for layer in range(LAYERS)]
    upgrades = {name: store_report["tensors"][name] for name in promoted}

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
    if gsq_sha256 != embedding["source"]["gsq_gguf_sha256"] or gsq_sha256 != store_report["gsq_gguf_sha256"]:
        raise ValueError("GSQ GGUF differs from the one Phases 1 and 6 used")
    reader = gguf.GGUFReader(gsq_path)
    tensors = list(reader.tensors)
    alignment = int(reader.alignment)
    q3_k = gguf.GGMLQuantizationType.Q3_K
    if set(promoted) - {tensor.name for tensor in tensors}:
        raise ValueError("a promoted tensor is not in authentic GSQ")

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
        _copy_metadata(reader, writer, gguf, PRUNED_OVERRIDES)
        for tensor in tensors:
            layer = _expert_tensor(tensor.name)
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
            elif tensor.name in upgrades:
                entry = upgrades[tensor.name]["store_entry"]
                if entry["ggml_type_name"] != "Q3_K":
                    raise ValueError(f"{tensor.name}: store entry is not Q3_K")
                if [int(v) for v in entry["gguf_shape"]] != [int(v) for v in tensor.shape]:
                    raise ValueError(f"{tensor.name}: store shape differs from GSQ")
                slices = _store_slices(store_root, entry)
                shape = [int(v) for v in tensor.shape[:-1]] + [KEPT]
                byte_shape = gguf.quant_shape_to_byte_shape(tuple(reversed(shape)), q3_k)
                payload = slices.shape[1] * KEPT
                if payload != upgrades[tensor.name]["kept_payload"]["bytes"]:
                    raise ValueError(f"{tensor.name}: kept Q3_K size differs from the store report")
                writer.add_tensor_info(tensor.name, byte_shape, np.dtype(np.uint8),
                                       payload, raw_dtype=q3_k)
                plan.append({"tensor": tensor.name, "source": "q3_k_store_kept_slices",
                             "ggml_type": "Q3_K", "layer": layer,
                             "slice_bytes": int(slices.shape[1]), "payload_bytes": payload,
                             "replaced_gsq_type": tensor.tensor_type.name,
                             "store_path": entry["path"], "store_sha256": entry["sha256"]})
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
            elif item["source"] in ("authentic_gsq_expert_slices", "q3_k_store_kept_slices"):
                if item["source"] == "q3_k_store_kept_slices":
                    source = _store_slices(store_root, upgrades[tensor.name]["store_entry"])
                else:
                    source = tensor.data
                slice_hashes = []
                for expert in kept[item["layer"]]:
                    chunk = np.ascontiguousarray(source[expert]).tobytes()
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
            if (item["source"] == "q3_k_store_kept_slices"
                    and item["sha256"] != upgrades[tensor.name]["kept_payload"]["sha256"]):
                raise ValueError(f"{tensor.name}: kept Q3_K payload differs from the store report")
            padding = (-written) % alignment
            if padding:
                handle.write(bytes(padding))
            if index % 100 == 0 or item["source"] != "authentic_gsq":
                if item["source"] != "authentic_gsq_expert_slices":
                    print(json.dumps({"tensor_index": index, "tensor": tensor.name,
                                      "source": item["source"]}), flush=True)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        writer.close()

    output_bytes = staging.stat().st_size
    if output_bytes != EXPECTED_BYTES:
        raise ValueError(f"output size {output_bytes} differs from the expected {EXPECTED_BYTES}")
    os.replace(staging, output)
    directory = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(json.dumps({"written": str(output), "bytes": output_bytes}), flush=True)

    validation = validate(gguf, reader, output, plan, kept, embd_path, store_root, upgrades)
    gsq_gpu_bytes = sum(int(t.n_bytes) for t in tensors if t.name != "token_embd.weight")
    gpu_bytes = sum(item["payload_bytes"] for item in plan if item["tensor"] != "token_embd.weight")
    if gpu_bytes > gsq_gpu_bytes:
        raise ValueError(f"tensors except token_embd take {gpu_bytes} bytes, "
                         f"more than authentic GSQ's {gsq_gpu_bytes}")
    return {
        "schema": SCHEMA,
        "status": "built",
        "warning": WARNING,
        "inputs": {
            "gsq_gguf": {"path": str(gsq_path), "sha256": gsq_sha256},
            "mask": {"path": str(mask_path), "sha256": mask_sha256},
            "replay_report": {"path": str(args.replay_report),
                              "sha256": _sha256_file(args.replay_report)},
            "store_report": {"path": str(args.store_report),
                             "sha256": _sha256_file(args.store_report),
                             "index_sha256": store_report["store"]["index_sha256"]},
            "embedding_report": {"path": str(args.embedding_report),
                                 "sha256": _sha256_file(args.embedding_report)},
            "token_embd_q6_k": {"path": str(embd_path), "sha256": embd_sha256},
        },
        "promoted": replay["promoted"],
        "upgraded_tensors": promoted,
        "metadata_overrides": PRUNED_OVERRIDES,
        "pruned_experts_per_layer": [np.flatnonzero(mask[layer]).tolist()
                                     for layer in range(LAYERS)],
        "budget": {"tensor_bytes_except_token_embd": gpu_bytes,
                   "authentic_gsq_tensor_bytes_except_token_embd": gsq_gpu_bytes,
                   "margin_bytes": gsq_gpu_bytes - gpu_bytes},
        "output": {"path": str(output), "bytes": output_bytes,
                   "expected_bytes": EXPECTED_BYTES, "header_bytes": header_bytes,
                   "sha256": validation.pop("output_sha256")},
        "validation": validation,
        "wall_seconds": time.monotonic() - started,
        "tensors": plan,
    }


def validate(gguf: Any, gsq: Any, output: Path, plan: list[dict[str, Any]],
             kept: list[np.ndarray], embd_path: Path, store_root: Path,
             upgrades: dict[str, Any]) -> dict[str, Any]:
    """Re-read the written GGUF and check it against its sources."""
    model = gguf.GGUFReader(output)
    gsq_fields = {name: (field.types, field.contents())
                  for name, field in gsq.fields.items() if not name.startswith("GGUF.")}
    model_fields = {name: (field.types, field.contents())
                    for name, field in model.fields.items() if not name.startswith("GGUF.")}
    if set(gsq_fields) != set(model_fields):
        raise ValueError("metadata keys differ from authentic GSQ")
    for name, (types, value) in gsq_fields.items():
        if model_fields[name] != (types, PRUNED_OVERRIDES.get(name, value)):
            raise ValueError(f"metadata differs for {name}")
    if [tensor.name for tensor in model.tensors] != [tensor.name for tensor in gsq.tensors]:
        raise ValueError("tensor order differs from authentic GSQ")

    counts = {"retained_tensors_byte_identical": 0,
              "expert_tensors_slice_identical": 0,
              "expert_slices_compared": 0,
              "q3_k_tensors_slice_identical": 0,
              "q3_k_slices_compared": 0}
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
        if item["source"] in ("authentic_gsq_expert_slices", "q3_k_store_kept_slices"):
            expected_shape = [int(v) for v in original.shape[:-1]] + [KEPT]
            if [int(v) for v in written.shape] != expected_shape:
                raise ValueError(f"sliced shape differs for {written.name}")
            if written.data.shape[0] != KEPT:
                raise ValueError(f"expert dimension differs for {written.name}")
            upgraded = item["source"] == "q3_k_store_kept_slices"
            source = (_store_slices(store_root, upgrades[written.name]["store_entry"])
                      if upgraded else original.data)
            for position, expert in enumerate(kept[item["layer"]]):
                if not np.array_equal(written.data[position].reshape(-1),
                                      np.asarray(source[expert]).reshape(-1)):
                    raise ValueError(f"slice {expert} differs for {written.name}")
                counts["q3_k_slices_compared" if upgraded else "expert_slices_compared"] += 1
            counts["q3_k_tensors_slice_identical" if upgraded
                   else "expert_tensors_slice_identical"] += 1
            continue
        if [int(v) for v in written.shape] != [int(v) for v in original.shape]:
            raise ValueError(f"shape differs for {written.name}")
        if not np.array_equal(written.data, original.data):
            raise ValueError(f"retained tensor bytes differ: {written.name}")
        counts["retained_tensors_byte_identical"] += 1
    if counts["q3_k_tensors_slice_identical"] != UPGRADES:
        raise ValueError("not every promoted tensor was installed as Q3_K")
    if (counts["expert_tensors_slice_identical"] + counts["q3_k_tensors_slice_identical"]
            != LAYERS * 4):
        raise ValueError("not every per-expert tensor was sliced")
    return {"metadata_identical_except_overrides": True,
            "tensor_order_identical": True,
            "token_embd_q6_k_byte_identical": True,
            **counts,
            "output_sha256": _sha256_file(output)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay-report", type=Path,
                        default=RCO / "reports/qwen36_prune24_q3k_replay_extended.json")
    parser.add_argument("--store-report", type=Path,
                        default=RCO / "reports/qwen36_prune24_q3k_store.json")
    parser.add_argument("--embedding-report", type=Path,
                        default=RCO / "reports/qwen36_gsq_e6_embedding.json")
    parser.add_argument("--gguf", type=Path,
                        default=ROOT / "results/Qwen3.6-35B-A3B-GSQ-hybrid.gguf")
    parser.add_argument("--gguf-python", type=Path, default=ROOT / "repos/llama.cpp/gguf-py")
    parser.add_argument("--output-model", type=Path,
                        default=ROOT / "results/Qwen3.6-35B-A3B-GSQ-RCO-v2-UNGATED.gguf")
    parser.add_argument("--output", type=Path,
                        default=RCO / "reports/qwen36_gsq_rco_v2_candidate_gguf.json")
    parser.add_argument("--chunk-bytes", type=int, default=64 << 20)
    args = parser.parse_args()
    for key in ("replay_report", "store_report", "embedding_report", "gguf", "gguf_python",
                "output_model", "output"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main() -> int:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output}")
    report = build(args)
    _atomic_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "output", "budget", "validation")}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
