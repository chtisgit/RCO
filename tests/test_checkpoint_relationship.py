import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoint_relationship import audit_base_gsq_relationship


class CheckpointRelationshipTest(unittest.TestCase):
    def _arguments(self):
        config = {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "transformers_version": "base-version",
            "text_config": {
                "num_hidden_layers": 1,
                "hidden_size": 64,
                "num_experts": 3,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 32,
            },
        }
        gsq_config = dict(config)
        gsq_config["transformers_version"] = "gsq-version"
        gsq_config["quantization_config"] = {"quant_method": "compressed-tensors"}
        identity = {
            "status": "pass", "repo_id": "Qwen/base", "revision": "dense-rev",
        }
        entries = [
            {"destination_name": "blk.0.ffn_gate_exps.weight",
             "destination_gguf_shape": [64, 32, 3],
             "source_view": {"axis": 1, "start": 0, "stop": 32}},
            {"destination_name": "blk.0.ffn_down_exps.weight",
             "destination_gguf_shape": [32, 64, 3], "source_view": None},
        ]
        manifest = {
            "status": "pass",
            "source": {"repo_id": "Qwen/base", "revision": "dense-rev"},
            "source_text_tensor_count": 2,
            "entries": entries,
        }
        return {
            "base_identity": identity,
            "base_manifest": manifest,
            "base_config": config,
            "gsq_config": gsq_config,
            "gsq_readme": (
                "---\nbase_model: Qwen/base\n"
                "base_model_relation: quantized\n---\n# test\n"),
            "shared_asset_hashes": {"tokenizer.json": ("same", "same")},
            "gguf_tensors": {
                item["destination_name"]: item["destination_gguf_shape"]
                for item in entries
            },
            "gguf_metadata": {
                "general.architecture": "qwen35moe",
                "qwen35moe.block_count": 1,
                "qwen35moe.embedding_length": 64,
                "qwen35moe.expert_count": 3,
                "qwen35moe.expert_used_count": 2,
                "qwen35moe.expert_feed_forward_length": 32,
            },
            "gsq_revision": "gsq-rev",
        }

    def test_proves_lineage_and_exact_canonical_inventory(self):
        report = audit_base_gsq_relationship(**self._arguments())
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["config"]["normalized_exact_match"])
        self.assertEqual(report["canonical_inventory"]["tensor_count"], 2)
        self.assertFalse(report["claim_limits"]["dense_and_quantized_values_equal"])

    def test_rejects_undeclared_base(self):
        arguments = self._arguments()
        arguments["gsq_readme"] = arguments["gsq_readme"].replace(
            "Qwen/base", "Qwen/different")
        with self.assertRaisesRegex(ValueError, "GSQ declares base"):
            audit_base_gsq_relationship(**arguments)

    def test_rejects_inventory_shape_difference(self):
        arguments = self._arguments()
        arguments["gguf_tensors"]["blk.0.ffn_down_exps.weight"] = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, "shape mismatches"):
            audit_base_gsq_relationship(**arguments)


if __name__ == "__main__":
    unittest.main()
