import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from safetensors.torch import save_file
except ModuleNotFoundError:
    torch = None
    nn = None

if torch is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from checkpoint_stream import SafeTensorPrefixLoader
    from quant.quantizer import Quantizer


@unittest.skipIf(torch is None, "PyTorch/safetensors is not installed")
class CheckpointStreamTest(unittest.TestCase):
    class Model(nn.Module if nn is not None else object):
        def __init__(self, device="meta"):
            super().__init__()
            self.embed = nn.Embedding(5, 3, device=device)
            self.layers = nn.ModuleList([
                nn.Linear(3, 4, bias=False, device=device),
                nn.Linear(4, 3, bias=False, device=device),
            ])

    def test_loads_one_prefix_and_releases_it_to_meta(self):
        tensors = {
            "embed.weight": torch.arange(15, dtype=torch.float32).reshape(5, 3),
            "layers.0.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3),
            "layers.1.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"embed.weight": tensors["embed.weight"]}, root / "a.safetensors")
            save_file({
                "layers.0.weight": tensors["layers.0.weight"],
                "layers.1.weight": tensors["layers.1.weight"],
            }, root / "b.safetensors")
            index = {
                "weight_map": {
                    "embed.weight": "a.safetensors",
                    "layers.0.weight": "b.safetensors",
                    "layers.1.weight": "b.safetensors",
                }
            }
            (root / "model.safetensors.index.json").write_text(json.dumps(index))

            model = self.Model()
            loader = SafeTensorPrefixLoader(root)
            report = loader.assert_prefix_schema(self.Model(), "layers.0")
            self.assertEqual(report["checkpoint_count"], 1)
            loaded = loader.load_prefix(model, "layers.0", device="cpu")

            self.assertEqual(loaded, tensors["layers.0.weight"].nbytes)
            self.assertEqual(model.layers[0].weight.device.type, "cpu")
            self.assertTrue(torch.equal(
                model.layers[0].weight, tensors["layers.0.weight"]))
            self.assertEqual(model.layers[1].weight.device.type, "meta")

            released = loader.release_prefix(model, "layers.0")
            self.assertEqual(released, loaded)
            self.assertEqual(model.layers[0].weight.device.type, "meta")

    def test_missing_prefix_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"weight": torch.ones(1)}, root / "model.safetensors")
            loader = SafeTensorPrefixLoader(root)
            with self.assertRaisesRegex(KeyError, "no tensors"):
                loader.load_prefix(self.Model(), "layers.9", device="cpu")

    def test_schema_preflight_rejects_mismatched_tensor_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(
                {"layers.0.weight_packed": torch.ones(1)},
                root / "model.safetensors")
            loader = SafeTensorPrefixLoader(root)
            report = loader.validate_prefix_schema(self.Model(), "layers.0")
            self.assertEqual(report["missing"], ["layers.0.weight"])
            self.assertEqual(
                report["unexpected"], ["layers.0.weight_packed"])
            with self.assertRaisesRegex(ValueError, "schema does not match"):
                loader.assert_prefix_schema(self.Model(), "layers.0")

    def test_quantizer_advances_activations_while_releasing_each_block(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(3, 3, bias=False, device="meta")

            def forward(self, hidden_states, **kwargs):
                return self.proj(hidden_states)

        class StreamingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Config", (), {"use_cache": True})()
                self.embed = nn.Embedding(5, 3, device="meta")
                self.layers = nn.ModuleList([Block(), Block()])

            def forward(self, input_ids):
                hidden = self.embed(input_ids)
                for layer in self.layers:
                    hidden = layer(hidden)
                return hidden

        tensors = {
            "embed.weight": torch.randn(5, 3),
            "layers.0.proj.weight": torch.eye(3),
            "layers.1.proj.weight": 2 * torch.eye(3),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(tensors, root / "model.safetensors")
            model = StreamingModel()
            loader = SafeTensorPrefixLoader(root)
            quantizer = Quantizer(
                model,
                data_loader=[([], {"input_ids": torch.tensor([[1, 2]])})],
                quantizable_modules=r"does-not-match",
                pre_block_modules=["embed"],
                block_modules="layers",
                save_dir=root / "output",
                device=torch.device("cpu"),
                cpu_offload_activations=True,
                checkpoint_loader=loader,
            )
            quantizer.quantize([2, 4], calibration_bitwidth=4)

        self.assertEqual(model.embed.weight.device.type, "meta")
        self.assertEqual(model.layers[0].proj.weight.device.type, "meta")
        self.assertEqual(model.layers[1].proj.weight.device.type, "meta")
        self.assertTrue(model.config.use_cache)


if __name__ == "__main__":
    unittest.main()
