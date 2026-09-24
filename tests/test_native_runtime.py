import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from native_runtime import NativeManifestWeightStore
from qwen35_native import Qwen35LinearAttentionGeometry, matrix_permutations


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "PyTorch is not installed")
class NativeManifestWeightStoreTest(unittest.TestCase):
    class CandidateStore:
        def __init__(self, candidates):
            self.candidates = candidates
            self.index = {
                "tensor_count": len(candidates),
                "tensors": {name: {} for name in candidates},
            }

        def metadata(self, name, candidate_type):
            return {"payload_bytes": self.candidates[name].nbytes // 4}

        def iter_decoded_rows(
            self, name, candidate_type, *, rows_per_chunk,
        ):
            rows = self.candidates[name].reshape(
                -1, self.candidates[name].shape[-1])
            for start in range(0, len(rows), rows_per_chunk):
                yield start, rows[start:start + rows_per_chunk].copy()

    class Model(nn.Module if DEPENDENCIES_AVAILABLE else object):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.layers = nn.ModuleList([nn.Module()])
            layer = self.model.language_model.layers[0]
            layer.linear_attn = nn.Module()
            layer.linear_attn.out_proj = nn.Linear(
                8, 3, bias=False, dtype=torch.bfloat16)
            layer.linear_attn.in_proj_qkv = nn.Linear(
                5, 20, bias=False, dtype=torch.bfloat16)
            layer.mlp = nn.Module()
            layer.mlp.experts = nn.Module()
            layer.mlp.experts.register_parameter(
                "gate_up_proj",
                nn.Parameter(torch.zeros(2, 6, 4, dtype=torch.bfloat16)),
            )

    def test_installs_permuted_matrices_and_fused_expert_views_in_chunks(self):
        geometry = Qwen35LinearAttentionGeometry(2, 4, 3, 2)
        out_source = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)
        _, out_columns = matrix_permutations(
            "model.layers.0.linear_attn.out_proj.weight",
            out_source.shape,
            geometry,
        )
        out_canonical = out_source[:, out_columns]

        qkv_source = np.arange(20 * 5, dtype=np.float32).reshape(20, 5)
        qkv_rows, _ = matrix_permutations(
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            qkv_source.shape,
            geometry,
        )
        qkv_canonical = qkv_source[qkv_rows]
        expert_source = (
            np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) + 100
        )

        candidates = {
            "blk.0.ssm_out.weight": out_canonical,
            "blk.0.ssm_qkv.weight": qkv_canonical,
            "blk.0.ffn_gate_exps.weight": expert_source,
        }
        manifest = {"entries": [
            {
                "rco_search": True,
                "destination_name": "blk.0.ssm_out.weight",
                "source_name": (
                    "model.language_model.layers.0.linear_attn.out_proj.weight"),
                "normalized_source_name": (
                    "model.layers.0.linear_attn.out_proj.weight"),
                "source_shape": [3, 8],
            },
            {
                "rco_search": True,
                "destination_name": "blk.0.ssm_qkv.weight",
                "source_name": (
                    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
                "normalized_source_name": (
                    "model.layers.0.linear_attn.in_proj_qkv.weight"),
                "source_shape": [20, 5],
            },
            {
                "rco_search": True,
                "destination_name": "blk.0.ffn_gate_exps.weight",
                "source_name": (
                    "model.language_model.layers.0.mlp.experts.gate_up_proj"),
                "normalized_source_name": (
                    "model.layers.0.mlp.experts.gate_proj.weight"),
                "source_shape": [2, 6, 4],
                "candidate_source_shape": [2, 3, 4],
                "source_view": {"axis": 1, "start": 0, "stop": 3},
            },
        ]}

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            (directory / "config.json").write_text(json.dumps({
                "text_config": {
                    "linear_num_key_heads": 2,
                    "linear_num_value_heads": 4,
                    "linear_key_head_dim": 3,
                    "linear_value_head_dim": 2,
                },
            }))
            adapter = NativeManifestWeightStore(
                self.CandidateStore(candidates), manifest, directory,
                rows_per_chunk=4,
            )
            model = self.Model()
            out_stats = adapter.install_layer_weight(
                model, "blk.0.ssm_out.weight", 2)
            qkv_stats = adapter.install_layer_weight(
                model, "blk.0.ssm_qkv.weight", 4)
            expert_stats = adapter.install_layer_weight(
                model, "blk.0.ffn_gate_exps.weight", 2)

        np.testing.assert_array_equal(
            model.model.language_model.layers[0].linear_attn.out_proj.weight
            .detach().float().numpy(),
            out_source,
        )
        np.testing.assert_array_equal(
            model.model.language_model.layers[0].linear_attn.in_proj_qkv.weight
            .detach().float().numpy(),
            qkv_source,
        )
        np.testing.assert_array_equal(
            model.model.language_model.layers[0].mlp.experts.gate_up_proj[
                :, :3].detach().float().numpy(),
            expert_source,
        )
        self.assertEqual(adapter.block_index("blk.0.ssm_out.weight"), 0)
        self.assertEqual(out_stats["max_decoded_fp32_bytes"], 3 * 8 * 4)
        self.assertEqual(qkv_stats["max_install_bf16_bytes"], 4 * 5 * 2)
        self.assertEqual(expert_stats["max_decoded_fp32_bytes"], 4 * 4 * 4)
        self.assertEqual(expert_stats["max_install_bf16_bytes"], 3 * 4 * 2)


if __name__ == "__main__":
    unittest.main()
