import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gguf_manifest import (
    build_gguf_manifest,
    canonical_source_name,
    parse_converter_dry_run,
)


class GGUFManifestTest(unittest.TestCase):
    def test_parses_converter_records_and_builds_exact_decision_groups(self):
        output = "\n".join((
            "INFO:hf-to-gguf:blk.0.ffn_gate.weight, torch.bfloat16 --> BF16, shape = {64, 32}",
            "INFO:hf-to-gguf:blk.0.attn_norm.weight, torch.bfloat16 --> F32, shape = {32}",
        ))
        converter = parse_converter_dry_run(output)
        identity = {
            "repo_id": "Qwen/test",
            "revision": "a" * 40,
            "category_counts": {"vision": 2, "mtp": 1},
            "text_inventory": [
                {
                    "name": "model.language_model.layers.0.mlp.gate_proj.weight",
                    "shard": "model.safetensors",
                    "category": "ordinary_text",
                    "dtype": "BF16",
                    "shape": [32, 64],
                },
                {
                    "name": "model.language_model.layers.0.input_layernorm.weight",
                    "shard": "model.safetensors",
                    "category": "normalization",
                    "dtype": "BF16",
                    "shape": [32],
                },
            ],
        }
        mapping = {
            "model.layers.0.mlp.gate_proj.weight": "blk.0.ffn_gate.weight",
            "model.layers.0.input_layernorm.weight": "blk.0.attn_norm.weight",
        }
        report = build_gguf_manifest(
            identity, converter, mapping.get, llama_cpp_revision="b" * 40)
        self.assertEqual(report["canonical_tensor_count"], 2)
        self.assertEqual(report["decision_group_count"], 1)
        self.assertEqual(report["copied_tensor_count"], 1)
        self.assertEqual(report["intentionally_omitted_tensor_count"], 3)
        self.assertEqual(
            report["entries"][0]["decision_group"], "blk.0.ffn_gate.weight")
        self.assertEqual(
            report["entries"][1]["copy_reason"], "not_a_matrix")

    def test_normalizes_delta_time_bias_for_pinned_mapping(self):
        self.assertEqual(
            canonical_source_name(
                "model.language_model.layers.2.linear_attn.dt_bias"),
            "model.layers.2.linear_attn.dt_proj.bias",
        )

    def test_rejects_converter_output_without_source(self):
        identity = {
            "repo_id": "test", "revision": "x",
            "category_counts": {}, "text_inventory": [],
        }
        converter = {
            "blk.0.extra.weight": {
                "source_dtype": "bfloat16",
                "ggml_type": "BF16",
                "gguf_shape": [64, 64],
            }
        }
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            build_gguf_manifest(
                identity, converter, lambda _: None, llama_cpp_revision="y")


if __name__ == "__main__":
    unittest.main()

