#!/usr/bin/env python3
"""Audit a compressed Qwen checkpoint through a meta-device HF load."""

import argparse
import json
import os
import resource
import tempfile
import time
import traceback
from pathlib import Path


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in sorted(value, key=str)]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def audit(model_path: Path) -> dict:
    import compressed_tensors
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText

    result = {
        "transformers": transformers.__version__,
        "compressed_tensors": compressed_tensors.__version__,
        "torch": torch.__version__,
        "model_path": str(model_path.resolve()),
        "operation": (
            "AutoModelForImageTextToText.from_pretrained with device_map=meta "
            "and output_loading_info=True"),
    }
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as handle:
            result["checkpoint_index_tensor_count"] = len(
                json.load(handle)["weight_map"])
    started = time.perf_counter()
    try:
        model, loading_info = AutoModelForImageTextToText.from_pretrained(
            model_path,
            device_map="meta",
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        parameters = list(model.named_parameters())
        clean_fields = {
            name: _json_value(loading_info.get(name, []))
            for name in (
                "missing_keys", "unexpected_keys", "mismatched_keys",
                "error_msgs")
        }
        result.update({
            "result": "loaded",
            "model_class": type(model).__name__,
            "quantizer_class": (
                type(model.hf_quantizer).__name__
                if getattr(model, "hf_quantizer", None) is not None else None),
            "parameter_count": sum(value.numel() for _, value in parameters),
            "parameter_devices": sorted({
                str(value.device) for _, value in parameters}),
            "resident_parameter_bytes": sum(
                value.numel() * value.element_size()
                for _, value in parameters if value.device.type != "meta"),
            "layer_0_expert_state_keys": [
                name for name, _ in parameters
                if "model.language_model.layers.0.mlp.experts" in name
            ],
            **clean_fields,
            "clean_loading_info": all(not clean_fields[name] for name in clean_fields),
        })
    except Exception as error:
        result.update({
            "result": "failed",
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
            "clean_loading_info": False,
        })
    result["elapsed_seconds"] = time.perf_counter() - started
    result["process_peak_rss_bytes"] = (
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024)
    return result


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a local compressed checkpoint onto meta and report exact "
            "missing, unexpected, and mismatched keys."))
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(args.model)
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _atomic_json(args.output, report)
        print(f"Wrote {args.output}")
    return 0 if report.get("clean_loading_info") else 1


if __name__ == "__main__":
    raise SystemExit(main())
