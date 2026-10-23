#!/usr/bin/env python3
"""A v3 GGUF from an assignment (RCO_PLAN_V3.md).

Every tensor of authentic GSQ's GGUF, in its order, with all metadata copied
(``--name`` optionally sets ``general.name``), except:

- the 370 searched tensors: the assignment's type, the bytes taken from that
  type's candidate store (``rco_v3_common.STORE_REPORTS``); Q8_0 is the GSQ
  GGUF's own bytes;
- ``output.weight``: Q4_K from the Q4_K store;
- ``token_embd.weight``: the Phase 1 Q6_K payload (GSQ-E6).

``--center`` writes the center assignment (``gate``/``up`` IQ2_XS, ``down``
IQ3_XXS, dense Q6_K) instead of reading ``--assignment``.  The bytes on the
GPU (every tensor except ``token_embd``) must not exceed 12,000,000,000.
Validation re-reads the written file: metadata, tensor order, names, types,
shapes, the SHA-256 of every tensor's bytes against its source, and the
exact file size.
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

from build_qwen36_gsq_rco_v2_gguf import _copy_metadata  # noqa: E402
from native_gguf import _atomic_json, _sha256_file, _stream_array_bytes, import_pinned_gguf  # noqa: E402
from rco_v3_common import (  # noqa: E402
    BUDGET_GPU_BYTES,
    center_assignment,
    choices,
    load_store_reports,
    searched,
    store_entry,
    store_label,
)

SCHEMA = "rco.qwen36.rco_v3_gguf.v1"
CHUNK = 64 << 20


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    gguf = import_pinned_gguf(args.gguf_python)
    names = [n for members in searched().values() for n in members]
    if args.center:
        assignment, source = center_assignment(), "center"
    else:
        assignment, source = json.loads(args.assignment.read_text())["assignment"], str(args.assignment)
    if sorted(assignment) != sorted(names):
        raise ValueError("the assignment does not cover exactly the 370 searched tensors")
    for name, type_name in assignment.items():
        if type_name not in choices(name):
            raise ValueError(f"{name}: {type_name} is not one of {choices(name)}")
    labels = sorted({store_label(n, t) for n, t in assignment.items() if t != "Q8_0"} | {"Q4_K"})
    reports = load_store_reports(labels)

    embedding = _load(args.embedding_report)
    embd_path = Path(embedding["payload"]["path"])
    embd_sha256 = _sha256_file(embd_path)
    if embd_sha256 != embedding["payload"]["sha256"]:
        raise ValueError("Q6_K token_embd payload differs from its Phase 1 report")
    embd_type = gguf.GGMLQuantizationType[embedding["payload"]["ggml_type"]]
    embd_bytes = int(embedding["payload"]["payload_bytes"])

    gsq_path = args.gguf.resolve(strict=True)
    gsq_sha256 = _sha256_file(gsq_path)
    reader = gguf.GGUFReader(gsq_path)
    tensors = list(reader.tensors)
    alignment = int(reader.alignment)

    plan = []
    for tensor in tensors:
        name = tensor.name
        shape = [int(v) for v in tensor.shape]
        if name == "token_embd.weight":
            item = {"tensor": name, "source": "gsq_e6_q6_k_token_embd", "ggml_type": embd_type.name,
                    "payload_bytes": embd_bytes, "path": str(embd_path), "offset": 0,
                    "expected_sha256": embd_sha256}
        elif name == "output.weight" or (name in assignment and assignment[name] != "Q8_0"):
            type_name = "Q4_K" if name == "output.weight" else assignment[name]
            entry = store_entry(reports, name, type_name)
            if [int(v) for v in entry["gguf_shape"]] != shape:
                raise ValueError(f"{name}: store shape differs from GSQ")
            root = Path(reports[store_label(name, type_name)]["store"]["root"])
            item = {"tensor": name, "source": f"store_{store_label(name, type_name)}",
                    "ggml_type": type_name, "payload_bytes": int(entry["payload_bytes"]),
                    "path": str(root / entry["path"]), "offset": int(entry["offset"]),
                    "expected_sha256": entry["sha256"]}
        else:
            if name in assignment and tensor.tensor_type.name != "Q8_0":
                raise ValueError(f"{name}: GSQ tensor is not Q8_0")
            item = {"tensor": name, "source": "authentic_gsq", "ggml_type": tensor.tensor_type.name,
                    "payload_bytes": int(tensor.n_bytes)}
        item["gguf_shape"] = shape
        plan.append(item)
    gpu = sum(i["payload_bytes"] for i in plan if i["tensor"] != "token_embd.weight")
    if gpu > BUDGET_GPU_BYTES:
        raise ValueError(f"{gpu} bytes on the GPU exceed the budget of {BUDGET_GPU_BYTES}")

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    staging = output.with_name(f".{output.name}.partial")
    staging.unlink(missing_ok=True)
    writer = gguf.GGUFWriter(staging, arch=reader.get_field(gguf.Keys.General.ARCHITECTURE).contents(),
                             endianess=reader.endianess)
    writer.data_alignment = alignment
    overrides = {"general.name": args.name} if args.name else {}
    try:
        _copy_metadata(reader, writer, gguf, overrides)
        for tensor, item in zip(tensors, plan):
            if item["source"] == "authentic_gsq":
                writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                                       int(tensor.n_bytes), raw_dtype=tensor.tensor_type)
            else:
                qtype = gguf.GGMLQuantizationType[item["ggml_type"]]
                byte_shape = gguf.quant_shape_to_byte_shape(tuple(reversed(item["gguf_shape"])), qtype)
                writer.add_tensor_info(tensor.name, byte_shape, np.dtype(np.uint8),
                                       item["payload_bytes"], raw_dtype=qtype)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        handle = writer.fout[0]
        writer.write_padding(handle, handle.tell())
        header_bytes = handle.tell()
        for index, (tensor, item) in enumerate(zip(tensors, plan)):
            digest = hashlib.sha256()
            written = 0
            if item["source"] == "authentic_gsq":
                chunks = _stream_array_bytes(tensor.data, CHUNK)
            else:
                chunks = _file_chunks(Path(item["path"]), item["offset"], item["payload_bytes"])
            for chunk in chunks:
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
            if written != item["payload_bytes"]:
                raise ValueError(f"payload size differs for {tensor.name}")
            item["sha256"] = digest.hexdigest()
            if item.get("expected_sha256", item["sha256"]) != item["sha256"]:
                raise ValueError(f"{tensor.name}: payload differs from its source")
            padding = (-written) % alignment
            if padding:
                handle.write(bytes(padding))
            if index % 200 == 0:
                print(json.dumps({"tensor_index": index, "tensor": tensor.name}), flush=True)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        writer.close()
    expected_bytes = header_bytes + sum(i["payload_bytes"] + (-i["payload_bytes"]) % alignment
                                        for i in plan)
    output_bytes = staging.stat().st_size
    if output_bytes != expected_bytes:
        raise ValueError(f"output size {output_bytes} differs from the expected {expected_bytes}")
    validation = _validate(gguf, reader, staging, plan, overrides)
    if not validation["ok"]:
        raise ValueError(f"validation failed: {validation}")
    os.replace(staging, output)
    types: dict[str, int] = {}
    for name in names:
        types[assignment[name]] = types.get(assignment[name], 0) + 1
    return {
        "schema": SCHEMA,
        "status": "built",
        "assignment_source": source,
        "assignment": assignment,
        "assignment_type_counts": types,
        "inputs": {"gsq_gguf": {"path": str(gsq_path), "sha256": gsq_sha256},
                   "token_embd_q6_k": {"path": str(embd_path), "sha256": embd_sha256},
                   "store_index_sha256": {k: r["store"]["index_sha256"] for k, r in reports.items()}},
        "budget": {"gpu_bytes": gpu, "budget_gpu_bytes": BUDGET_GPU_BYTES,
                   "margin_bytes": BUDGET_GPU_BYTES - gpu,
                   "measure": "all tensor bytes except token_embd; metadata not counted"},
        "output": {"path": str(output), "bytes": output_bytes, "header_bytes": header_bytes,
                   "sha256": _sha256_file(output)},
        "validation": validation,
        "wall_seconds": time.monotonic() - started,
        "tensors": [{k: v for k, v in i.items() if k not in ("path", "offset")} for i in plan],
    }


def _file_chunks(path: Path, offset: int, size: int):
    with path.open("rb") as handle:
        handle.seek(offset)
        left = size
        while left:
            chunk = handle.read(min(CHUNK, left))
            if not chunk:
                raise ValueError(f"{path} ends early")
            yield chunk
            left -= len(chunk)


def _validate(gguf, gsq, path: Path, plan, overrides) -> dict[str, Any]:
    model = gguf.GGUFReader(path)
    before = {k: (f.types, f.contents()) for k, f in gsq.fields.items() if not k.startswith("GGUF.")}
    after = {k: (f.types, f.contents()) for k, f in model.fields.items() if not k.startswith("GGUF.")}
    metadata_ok = set(before) == set(after) and all(
        after[k] == (types, overrides.get(k, value)) for k, (types, value) in before.items())
    tensors = list(model.tensors)
    order_ok = [t.name for t in tensors] == [i["tensor"] for i in plan]
    mismatches = []
    for tensor, item in zip(tensors, plan):
        if (tensor.tensor_type.name != item["ggml_type"]
                or [int(v) for v in tensor.shape] != item["gguf_shape"]
                or int(tensor.n_bytes) != item["payload_bytes"]
                or hashlib.sha256(memoryview(tensor.data).cast("B")).hexdigest() != item["sha256"]):
            mismatches.append(tensor.name)
    return {"metadata_identical": metadata_ok, "tensor_order_identical": order_ok,
            "tensor_count": len(tensors), "tensor_mismatches": mismatches,
            "ok": metadata_ok and order_ok and not mismatches}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--center", action="store_true")
    source.add_argument("--assignment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--gguf", type=Path,
                        default=ROOT / "results/Qwen3.6-35B-A3B-GSQ-hybrid.gguf")
    parser.add_argument("--embedding-report", type=Path,
                        default=RCO / "reports/qwen36_gsq_e6_embedding.json")
    parser.add_argument("--gguf-python", type=Path, default=ROOT / "repos/llama.cpp/gguf-py")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.report.exists() and _load(args.report).get("status") == "built" and args.output.exists():
        print(f"v3 gguf: already built ({args.output})", flush=True)
        return 0
    report = build(args)
    _atomic_json(args.report, report)
    print(f"v3 gguf: built {report['output']['path']}, {report['output']['bytes']} bytes, "
          f"{report['budget']['gpu_bytes']} on the GPU (margin {report['budget']['margin_bytes']}); "
          f"types {report['assignment_type_counts']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
