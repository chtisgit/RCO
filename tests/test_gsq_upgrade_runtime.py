import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gsq_upgrade_runtime import GSQUpgradeWeightStore


class _FakeNativeStore:
    def __init__(self, names=()):
        self.entries = {name: {} for name in names}
        self.installs = []

    def install_layer_weight(self, model, name, bitwidth):
        self.installs.append((name, bitwidth))
        return {
            "installed_rows": 1,
            "max_decoded_fp32_bytes": 16,
            "max_install_bf16_bytes": 8,
        }


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
        self.fused = nn.Parameter(torch.zeros(2, 4, 3, dtype=torch.bfloat16))


def _record(name, source, *, kind="pinned_bf16_source", shard="model.safetensors"):
    upgrade = {
        "policy_status": "admitted",
        "kind": kind,
        "ggml_type": "BF16" if kind == "pinned_bf16_source" else "Q4_0",
        "payload_bytes": 24,
        "incremental_gguf_bytes": 8,
    }
    if kind == "pinned_bf16_source":
        upgrade.update({"source_tensor": source, "source_shard": shard})
    return {"destination_name": name, "upgrade": upgrade}


class GSQUpgradeRuntimeTest(unittest.TestCase):
    def _build_store(self, root, records, entries, native=()):
        index = {entry["source_name"]: "model.safetensors" for entry in entries}
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": index}), encoding="utf-8")
        policy = {
            "status": "pass",
            "inventory": {"admitted_upgrade_count": len(records)},
            "tensors": records,
        }
        return GSQUpgradeWeightStore(
            policy, {"entries": entries}, root, _FakeNativeStore(native),
            rows_per_chunk=1)

    def test_retain_choice_is_a_true_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
            save_file({"layer.weight": expected}, root / "model.safetensors")
            store = self._build_store(
                root,
                [_record("blk.0.test.weight", "layer.weight")],
                [{"destination_name": "blk.0.test.weight",
                  "source_name": "layer.weight", "source_shape": [2, 3],
                  "source_view": None}],
            )
            model = _Model()
            before = model.layer.weight.detach().clone()
            result = store.install_layer_weight(
                model, "blk.0.test.weight", 0)
            torch.testing.assert_close(model.layer.weight, before)
            self.assertEqual(result["installed_rows"], 0)

    def test_streams_exact_bf16_matrix_and_fused_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
            fused = torch.arange(24, dtype=torch.bfloat16).reshape(2, 4, 3)
            save_file({"layer.weight": matrix, "fused": fused},
                      root / "model.safetensors")
            records = [
                _record("blk.0.test.weight", "layer.weight"),
                _record("blk.0.fused_gate.weight", "fused"),
            ]
            entries = [
                {"destination_name": "blk.0.test.weight",
                 "source_name": "layer.weight", "source_shape": [2, 3],
                 "source_view": None},
                {"destination_name": "blk.0.fused_gate.weight",
                 "source_name": "fused", "source_shape": [2, 4, 3],
                 "source_view": {"axis": 1, "start": 0, "stop": 2}},
            ]
            store = self._build_store(root, records, entries)
            model = _Model()
            model.layer.weight.data.zero_()
            result_matrix = store.install_layer_weight(
                model, "blk.0.test.weight", 1)
            result_fused = store.install_layer_weight(
                model, "blk.0.fused_gate.weight", 1)
            torch.testing.assert_close(model.layer.weight, matrix)
            torch.testing.assert_close(model.fused[:, :2], fused[:, :2])
            torch.testing.assert_close(
                model.fused[:, 2:], torch.zeros_like(model.fused[:, 2:]))
            self.assertEqual(result_matrix["installed_rows"], 2)
            self.assertEqual(result_fused["installed_rows"], 4)

    def test_delegates_native_q4_upgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"layer.weight": torch.zeros(2, 3)},
                      root / "model.safetensors")
            name = "blk.0.routed.weight"
            store = self._build_store(
                root, [_record(name, "layer.weight",
                               kind="bf16_derived_native_candidate")],
                [{"destination_name": name, "source_name": "layer.weight",
                  "source_shape": [2, 3], "source_view": None}],
                native=(name,),
            )
            result = store.install_layer_weight(_Model(), name, 1)
            self.assertEqual(store.native_weight_store.installs, [(name, 4)])
            self.assertEqual(result["max_decoded_fp32_bytes"], 16)


if __name__ == "__main__":
    unittest.main()
