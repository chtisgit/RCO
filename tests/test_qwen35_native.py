import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
    from safetensors.torch import save_file

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

from qwen35_native import (
    Qwen35LinearAttentionGeometry,
    SafetensorGGUFRowSource,
    _reordered_head_indices,
    matrix_permutations,
)


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "PyTorch/safetensors is not installed")
class Qwen35NativeTest(unittest.TestCase):
    def test_value_head_permutations_match_grouped_to_tiled_order(self):
        geometry = Qwen35LinearAttentionGeometry(2, 4, 3, 2)
        expected_v = np.array([0, 1, 4, 5, 2, 3, 6, 7])
        np.testing.assert_array_equal(
            _reordered_head_indices(2, 4, 2), expected_v)

        qkv_rows, qkv_columns = matrix_permutations(
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            (20, 5),
            geometry,
        )
        self.assertIsNone(qkv_columns)
        np.testing.assert_array_equal(qkv_rows[:12], np.arange(12))
        np.testing.assert_array_equal(qkv_rows[12:], 12 + expected_v)

        out_rows, out_columns = matrix_permutations(
            "model.layers.0.linear_attn.out_proj.weight",
            (5, 8),
            geometry,
        )
        self.assertIsNone(out_rows)
        np.testing.assert_array_equal(out_columns, expected_v)

    def test_row_source_reads_bounded_runs_and_reorders_columns(self):
        geometry = {
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 3,
            "linear_value_head_dim": 2,
        }
        values = torch.arange(5 * 8, dtype=torch.bfloat16).reshape(5, 8)
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            (directory / "config.json").write_text(
                __import__("json").dumps({"text_config": geometry}))
            source_name = "model.language_model.layers.0.linear_attn.out_proj.weight"
            save_file({source_name: values}, directory / "model.safetensors")
            entry = {
                "rco_search": True,
                "source_name": source_name,
                "normalized_source_name": (
                    "model.layers.0.linear_attn.out_proj.weight"),
                "source_shape": [5, 8],
                "destination_gguf_shape": [8, 5],
                "source_shard": "model.safetensors",
                "converter_transforms": [
                    "strip_language_model_namespace", "reorder_value_heads"],
                "destination_name": "blk.0.ssm_out.weight",
            }
            stats = {}
            actual = np.concatenate(list(SafetensorGGUFRowSource(
                directory).iter_rows(entry, rows_per_chunk=2, stats=stats)))
            order = torch.tensor([0, 1, 4, 5, 2, 3, 6, 7])
            expected = values.index_select(1, order).float().numpy()
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(stats["max_source_rows_per_chunk"], 2)
            self.assertEqual(stats["max_dense_chunk_bytes"], 2 * 8 * 4)


if __name__ == "__main__":
    unittest.main()
