#!/usr/bin/env python3
"""Audit meta-model topology and safetensors prefix compatibility."""

from __future__ import annotations

import argparse
import json
import resource
import sys
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from checkpoint_stream import SafeTensorPrefixLoader
from model_adapter import get_model_adapter
from models import load_meta_model


def _bounded(report: dict, limit: int = 8) -> dict:
    return {
        "prefix": report["prefix"],
        "expected_count": report["expected_count"],
        "checkpoint_count": report["checkpoint_count"],
        "missing_count": len(report["missing"]),
        "unexpected_count": len(report["unexpected"]),
        "missing_preview": report["missing"][:limit],
        "unexpected_preview": report["unexpected"][:limit],
        "compatible": not report["missing"] and not report["unexpected"],
    }


def audit(model_path: str) -> dict:
    import transformers

    model = load_meta_model(model_path)
    adapter = get_model_adapter(model)
    loader = SafeTensorPrefixLoader(model_path)
    prefixes = [
        *adapter.embedding_paths,
        *(f"{adapter.layers_path}.{index}"
          for index in range(len(adapter.layers))),
        *adapter.final_module_paths,
    ]
    prefix_reports = [
        _bounded(loader.validate_prefix_schema(model, prefix))
        for prefix in dict.fromkeys(prefixes)
    ]
    layer_types = Counter(
        str(getattr(layer, "layer_type", getattr(layer, "attention_type", "unknown")))
        for layer in adapter.layers)
    resident_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.device.type != "meta")
    resident_buffer_bytes = sum(
        buffer.numel() * buffer.element_size()
        for buffer in model.buffers()
        if buffer.device.type != "meta")
    compatible = all(report["compatible"] for report in prefix_reports)
    return {
        "model_path": str(model_path),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "model_class": type(model).__name__,
        "adapter_family": adapter.family,
        "layers_path": adapter.layers_path,
        "num_layers": len(adapter.layers),
        "layer_types": dict(sorted(layer_types.items())),
        "meta_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.device.type == "meta"),
        "resident_parameter_bytes": resident_parameter_bytes,
        "resident_buffer_bytes": resident_buffer_bytes,
        "process_peak_rss_bytes": (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024),
        "streaming_schema_compatible": compatible,
        "prefixes": prefix_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit(args.model)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if report["streaming_schema_compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
