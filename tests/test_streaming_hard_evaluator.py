import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors.torch import save_file
except ModuleNotFoundError:
    torch = None
    nn = None

if torch is not None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from checkpoint_stream import SafeTensorPrefixLoader
    from grouping import LayerGroup
    from search.streaming import (
        StreamingHardCausalEvaluator,
        chunked_causal_cross_entropy,
    )


@unittest.skipIf(torch is None, "PyTorch/safetensors is not installed")
class StreamingHardEvaluatorTest(unittest.TestCase):
    class Block(nn.Module if nn is not None else object):
        def __init__(self, index, device="cpu"):
            super().__init__()
            self.layer_type = "even" if index % 2 == 0 else "odd"
            self.proj = nn.Linear(3, 3, bias=False, device=device)
            # Mirrors rotary/convolution helpers constructed from config and
            # intentionally absent from the checkpoint.
            self.register_buffer(
                "runtime_scale", torch.tensor(1.0), persistent=False)

        def forward(self, hidden_states, *, block_scale, **kwargs):
            return hidden_states + block_scale * self.runtime_scale * self.proj(hidden_states)

    class TextModel(nn.Module if nn is not None else object):
        def __init__(self, device="cpu"):
            super().__init__()
            self.embed_tokens = nn.Embedding(7, 3, device=device)
            self.layers = nn.ModuleList([
                StreamingHardEvaluatorTest.Block(0, device=device),
                StreamingHardEvaluatorTest.Block(1, device=device),
            ])
            self.norm = nn.LayerNorm(3, device=device)

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            hidden = self.embed_tokens(input_ids)
            # This deliberately inspects each layer before calling it.  The
            # streaming wrapper must proxy attributes and preserve distinct
            # layer-specific kwargs from the canonical forward loop.
            for layer in self.layers:
                scale = 1.0 if layer.layer_type == "even" else 2.0
                hidden = layer(hidden, block_scale=scale)
            return SimpleNamespace(last_hidden_state=self.norm(hidden))

    class Model(nn.Module if nn is not None else object):
        def __init__(self, device="cpu"):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="llama", hidden_size=3, num_hidden_layers=2)
            self.model = StreamingHardEvaluatorTest.TextModel(device=device)
            self.lm_head = nn.Linear(3, 7, bias=False, device=device)

    class Store:
        cache = False

        def __init__(self, candidates):
            self.candidates = candidates
            self.calls = []

        def get_layer_weight(self, name, bits):
            self.calls.append((name, bits))
            return self.candidates[(name, bits)].clone()

    def test_chunked_ce_matches_full_logits_and_mask(self):
        torch.manual_seed(1)
        hidden = torch.randn(2, 5, 3)
        labels = torch.randint(0, 7, (2, 5))
        head = nn.Linear(3, 7, bias=True)
        mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 0, 1, 1],
        ], dtype=torch.bool)
        actual, count = chunked_causal_cross_entropy(
            hidden, labels, head, loss_mask=mask, vocab_chunk_size=3)
        losses = F.cross_entropy(
            head(hidden[:, :-1]).reshape(-1, 7),
            labels[:, 1:].reshape(-1), reduction="none")
        active = mask[:, 1:].reshape(-1)
        expected = losses[active].mean()
        self.assertEqual(count, int(active.sum()))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_streams_blocks_and_selected_candidates_through_canonical_forward(self):
        torch.manual_seed(2)
        dense = self.Model()
        state = {name: value.detach().clone()
                 for name, value in dense.state_dict().items()}
        candidates = {
            ("model.layers.0.proj", 2): 0.25 * torch.eye(3),
            ("model.layers.0.proj", 4): 0.50 * torch.eye(3),
            ("model.layers.1.proj", 2): 0.75 * torch.eye(3),
            ("model.layers.1.proj", 4): 1.00 * torch.eye(3),
        }
        groups = [
            LayerGroup(0, ["model.layers.0.proj"], "block0"),
            LayerGroup(1, ["model.layers.1.proj"], "block1"),
        ]
        assignment = torch.tensor([1, 0])
        input_ids = torch.tensor([[1, 2, 3, 4]])

        expected_model = self.Model()
        expected_model.load_state_dict(state)
        expected_model.model.layers[0].proj.weight.data.copy_(
            candidates[("model.layers.0.proj", 4)])
        expected_model.model.layers[1].proj.weight.data.copy_(
            candidates[("model.layers.1.proj", 2)])
        with torch.no_grad():
            hidden = expected_model.model(input_ids).last_hidden_state
            expected = F.cross_entropy(
                expected_model.lm_head(hidden[:, :-1]).reshape(-1, 7),
                input_ids[:, 1:].reshape(-1))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(state, root / "model.safetensors")
            meta_model = self.Model(device="meta")
            store = self.Store(candidates)
            evaluator = StreamingHardCausalEvaluator(
                meta_model,
                SafeTensorPrefixLoader(root),
                store,
                groups,
                [2, 4],
                device="cpu",
                vocab_chunk_size=3,
            )
            result = evaluator.evaluate(input_ids, assignment)
            repeated = evaluator.evaluate(input_ids, assignment)

        self.assertAlmostEqual(result.loss, expected.item(), places=6)
        self.assertAlmostEqual(repeated.loss, expected.item(), places=6)
        self.assertEqual(result.token_count, 3)
        self.assertEqual(result.memory.loaded_blocks, 2)
        self.assertGreater(result.memory.max_block_bytes, 0)
        self.assertEqual(
            store.calls,
            [
                ("model.layers.0.proj", 4), ("model.layers.1.proj", 2),
                ("model.layers.0.proj", 4), ("model.layers.1.proj", 2),
            ],
        )
        self.assertIsInstance(meta_model.model.layers[0], self.Block)
        self.assertEqual(meta_model.model.embed_tokens.weight.device.type, "meta")
        self.assertEqual(meta_model.model.layers[0].proj.weight.device.type, "meta")
        self.assertEqual(meta_model.model.layers[1].proj.weight.device.type, "meta")
        self.assertEqual(
            meta_model.model.layers[0].runtime_scale.device.type, "cpu")
        self.assertEqual(meta_model.model.norm.weight.device.type, "meta")
        self.assertEqual(meta_model.lm_head.weight.device.type, "meta")

    def test_copies_logical_experts_into_fused_storage(self):
        class FusedExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = nn.Parameter(torch.zeros(2, 4, 3))
                self.down_proj = nn.Parameter(torch.zeros(2, 3, 2))

        model = nn.Module()
        model.experts = FusedExperts()
        from search.streaming import _copy_candidate
        _copy_candidate(
            model, "experts.1.gate_proj", torch.full((2, 3), 1.0))
        _copy_candidate(
            model, "experts.1.up_proj", torch.full((2, 3), 2.0))
        _copy_candidate(
            model, "experts.1.down_proj", torch.full((3, 2), 3.0))
        self.assertTrue(torch.equal(
            model.experts.gate_up_proj[1, :2], torch.ones(2, 3)))
        self.assertTrue(torch.equal(
            model.experts.gate_up_proj[1, 2:], torch.full((2, 3), 2.0)))
        self.assertTrue(torch.equal(
            model.experts.down_proj[1], torch.full((3, 2), 3.0)))


if __name__ == "__main__":
    unittest.main()
