import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_identity import audit_qwen_checkpoint_identity


class ModelIdentityTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[str, str]:
        tensors = {
            "model.language_model.embed_tokens.weight": torch.ones(8, 4),
            "model.language_model.layers.0.input_layernorm.weight": torch.ones(4),
            "model.language_model.layers.0.mlp.gate_proj.weight": torch.ones(6, 4),
            "model.visual.patch_embed.proj.weight": torch.ones(2, 2),
        }
        shard_name = "model.safetensors-00001-of-00001.safetensors"
        save_file(tensors, root / shard_name)
        total_size = sum(value.numel() * value.element_size() for value in tensors.values())
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": total_size},
            "weight_map": {name: shard_name for name in tensors},
        }))
        (root / "config.json").write_text(json.dumps({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "tie_word_embeddings": True,
            "text_config": {
                "model_type": "qwen3_5_text",
                "dtype": "bfloat16",
                "hidden_size": 4,
                "intermediate_size": 6,
                "num_hidden_layers": 1,
                "vocab_size": 8,
                "layer_types": ["linear_attention"],
            },
        }))
        revision = "a" * 40
        metadata_root = root / ".cache" / "huggingface" / "download"
        metadata_root.mkdir(parents=True)
        for path in root.iterdir():
            if path.is_file():
                (metadata_root / f"{path.name}.metadata").write_text(
                    f"{revision}\nfixture-etag\n0\n")
        weight_hash = hashlib.sha256((root / shard_name).read_bytes()).hexdigest()
        return revision, weight_hash

    def test_records_complete_text_inventory_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revision, weight_hash = self._fixture(root)
            report = audit_qwen_checkpoint_identity(
                root,
                repo_id="Qwen/test",
                expected_revision=revision,
                expected_weight_sha256=weight_hash,
            )
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["tensor_count"], 4)
        self.assertEqual(report["text_tensor_count"], 3)
        self.assertEqual(report["omitted_tensor_count"], 1)
        self.assertEqual(report["category_counts"]["vision"], 1)
        self.assertEqual(len(report["text_inventory"]), 3)
        self.assertEqual(report["config"]["layer_type_counts"], {"linear_attention": 1})

    def test_rejects_unexpected_weight_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revision, _ = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "does not match expected"):
                audit_qwen_checkpoint_identity(
                    root,
                    repo_id="Qwen/test",
                    expected_revision=revision,
                    expected_weight_sha256="0" * 64,
                )

    def test_rejects_file_that_disagrees_with_sha256_hub_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revision, _ = self._fixture(root)
            metadata = (
                root / ".cache" / "huggingface" / "download"
                / "config.json.metadata"
            )
            metadata.write_text(f"{revision}\n{'0' * 64}\n0\n")
            with self.assertRaisesRegex(ValueError, "Hub LFS/Xet identity"):
                audit_qwen_checkpoint_identity(
                    root,
                    repo_id="Qwen/test",
                    expected_revision=revision,
                )


if __name__ == "__main__":
    unittest.main()
