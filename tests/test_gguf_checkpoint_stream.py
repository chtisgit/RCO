import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gguf_checkpoint_stream import _restore_conv1d, _restore_vector
from qwen35_native import Qwen35LinearAttentionGeometry, _reordered_head_indices


class GGUFCheckpointStreamTest(unittest.TestCase):
    def setUp(self):
        self.geometry = Qwen35LinearAttentionGeometry(
            num_key_heads=2,
            num_value_heads=4,
            key_head_dim=2,
            value_head_dim=2,
        )

    def test_restores_reordered_negative_exponential_vector(self):
        source = np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float32)
        order = _reordered_head_indices(2, 4, 1)
        canonical = -np.exp(source[order])
        entry = {
            "converter_transforms": [
                "strip_language_model_namespace",
                "reorder_value_heads",
                "negative_exponential",
            ]
        }
        np.testing.assert_allclose(
            _restore_vector(canonical, entry, self.geometry), source,
            rtol=1e-6, atol=1e-6)

    def test_restores_add_one_norm(self):
        source = np.array([-0.5, 0.0, 0.25], dtype=np.float32)
        entry = {"converter_transforms": [
            "strip_language_model_namespace", "add_one_to_norm"]}
        np.testing.assert_array_equal(
            _restore_vector(source + 1, entry, self.geometry), source)

    def test_restores_squeezed_reordered_conv1d(self):
        channels = (
            2 * self.geometry.num_key_heads * self.geometry.key_head_dim
            + self.geometry.num_value_heads * self.geometry.value_head_dim
        )
        source = np.arange(channels * 3, dtype=np.float32).reshape(
            channels, 1, 3)
        qk = 2 * self.geometry.num_key_heads * self.geometry.key_head_dim
        order = np.concatenate((
            np.arange(qk),
            qk + _reordered_head_indices(2, 4, 2),
        ))
        canonical = source[:, 0, :][order]
        entry = {
            "source_shape": list(source.shape),
            "converter_transforms": [
                "strip_language_model_namespace",
                "reorder_value_heads",
                "squeeze_conv1d",
            ],
        }
        np.testing.assert_array_equal(
            _restore_conv1d(canonical, entry, self.geometry), source)


if __name__ == "__main__":
    unittest.main()
