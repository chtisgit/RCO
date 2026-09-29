import tempfile
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoint_stream import SafeTensorPrefixLoader
from search.quant import WeightInterpolation
from search.relaxed import (
    CheckpointedRelaxedBlockStats,
    CheckpointedStreamingRelaxedBlock,
    DenseDeltaRowSource,
    RelaxedLinearBinding,
    StreamingRelaxedLinearStats,
    streaming_relaxed_linear,
)


class StreamingRelaxedLinearTest(unittest.TestCase):
    def test_matches_released_dense_loss_and_gradients(self):
        torch.manual_seed(29)
        dtype = torch.float64
        reference = torch.randn(7, 5, dtype=dtype)
        alternatives = [
            reference + 0.2 * torch.randn_like(reference),
            reference + 0.4 * torch.randn_like(reference),
        ]
        bias = torch.randn(7, dtype=dtype)
        target = torch.randn(2, 3, 7, dtype=dtype)
        initial_input = torch.randn(2, 3, 5, dtype=dtype)
        initial_logits = torch.tensor([0.4, -0.7, 0.2], dtype=dtype)

        dense_input = initial_input.clone().requires_grad_(True)
        dense_logits = initial_logits.clone().requires_grad_(True)
        interpolation = WeightInterpolation(
            [candidate - reference for candidate in alternatives],
            dense_logits.unsqueeze(0),
            0,
        )
        dense_output = F.linear(
            dense_input, interpolation(reference), bias)
        dense_loss = F.mse_loss(dense_output, target)
        dense_loss.backward()

        streamed_input = initial_input.clone().requires_grad_(True)
        streamed_logits = initial_logits.clone().requires_grad_(True)
        source = DenseDeltaRowSource(
            reference, alternatives, rows_per_chunk=2)
        stats = StreamingRelaxedLinearStats()
        streamed_output = streaming_relaxed_linear(
            streamed_input,
            streamed_logits,
            source,
            bias=bias,
            stats=stats,
        )
        streamed_loss = F.mse_loss(streamed_output, target)
        streamed_loss.backward()

        self.assertTrue(torch.allclose(
            streamed_output, dense_output, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(
            streamed_loss, dense_loss, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(
            streamed_input.grad, dense_input.grad, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(
            streamed_logits.grad, dense_logits.grad,
            atol=1e-12, rtol=1e-12))
        self.assertEqual(stats.forward_reference_passes, 1)
        self.assertEqual(stats.backward_reference_passes, 1)
        self.assertEqual(stats.forward_alternative_passes, 2)
        self.assertEqual(stats.backward_alternative_passes, 2)
        self.assertEqual(
            stats.max_materialized_chunk_bytes,
            2 * reference.shape[1] * reference.element_size(),
        )
        self.assertEqual(
            stats.forward_reference_bytes,
            reference.numel() * reference.element_size(),
        )
        self.assertEqual(
            stats.backward_alternative_bytes,
            len(alternatives) * reference.numel() * reference.element_size(),
        )

    def test_rejects_incomplete_row_source(self):
        class IncompleteSource:
            in_features = 3
            out_features = 4
            alternative_count = 1

            def iter_reference_rows(self):
                yield 0, torch.ones(3, 3)

            def iter_delta_rows(self, alternative_index):
                yield 0, torch.ones(4, 3)

        with self.assertRaisesRegex(ValueError, "ended at row 3"):
            streaming_relaxed_linear(
                torch.ones(2, 3),
                torch.zeros(2, requires_grad=True),
                IncompleteSource(),
            )

    def test_bias_gradient_matches_dense_linear(self):
        torch.manual_seed(31)
        reference = torch.randn(4, 3, dtype=torch.float64)
        alternative = reference + torch.randn_like(reference)
        source = DenseDeltaRowSource(
            reference, [alternative], rows_per_chunk=1)
        values = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        logits = torch.randn(2, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(4, dtype=torch.float64, requires_grad=True)

        output = streaming_relaxed_linear(
            values, logits, source, bias=bias)
        output.sum().backward()

        self.assertTrue(torch.equal(bias.grad, torch.full_like(bias, 2.0)))

    def test_checkpointed_blocks_reload_and_match_dense_model_gradients(self):
        class Block(nn.Module):
            def __init__(self, device="cpu"):
                super().__init__()
                self.norm = nn.LayerNorm(4, dtype=torch.float64, device=device)
                self.proj = nn.Linear(
                    4, 4, bias=True, dtype=torch.float64, device=device)

            def forward(self, hidden_states):
                return hidden_states + torch.tanh(
                    self.proj(self.norm(hidden_states)))

        class Model(nn.Module):
            def __init__(self, device="cpu"):
                super().__init__()
                self.layers = nn.ModuleList([
                    Block(device=device), Block(device=device)])

            def forward(self, hidden_states):
                for layer in self.layers:
                    hidden_states = layer(hidden_states)
                return hidden_states

        torch.manual_seed(43)
        checkpoint_model = Model()
        state = {
            name: value.detach().clone()
            for name, value in checkpoint_model.state_dict().items()
        }
        references = [
            state[f"layers.{index}.proj.weight"]
            + 0.1 * torch.randn_like(state[f"layers.{index}.proj.weight"])
            for index in range(2)
        ]
        alternatives = [
            reference + 0.2 * torch.randn_like(reference)
            for reference in references
        ]
        input_values = torch.randn(2, 3, 4, dtype=torch.float64)
        initial_logits = torch.tensor(
            [[0.4, -0.2], [-0.3, 0.7]], dtype=torch.float64)

        dense_model = Model()
        dense_model.load_state_dict(state)
        dense_logits = initial_logits.clone().requires_grad_(True)
        for index, layer in enumerate(dense_model.layers):
            layer.proj.weight.data.copy_(references[index])
            torch.nn.utils.parametrize.register_parametrization(
                layer.proj,
                "weight",
                WeightInterpolation(
                    [alternatives[index] - references[index]],
                    dense_logits,
                    index,
                ),
            )
        dense_input = input_values.clone().requires_grad_(True)
        dense_output = dense_model(dense_input)
        dense_loss = dense_output.square().mean()
        dense_loss.backward()

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            save_file(state, directory / "model.safetensors")
            loader = SafeTensorPrefixLoader(directory)
            streamed_model = Model(device="meta")
            streamed_logits = initial_logits.clone().requires_grad_(True)
            block_stats = []
            linear_stats = []
            originals = list(streamed_model.layers)
            for index, block in enumerate(originals):
                block_stat = CheckpointedRelaxedBlockStats()
                linear_stat = StreamingRelaxedLinearStats()
                block_stats.append(block_stat)
                linear_stats.append(linear_stat)
                streamed_model.layers[index] = (
                    CheckpointedStreamingRelaxedBlock(
                        block,
                        model=streamed_model,
                        path=f"layers.{index}",
                        checkpoint_loader=loader,
                        logits=streamed_logits,
                        bindings=[RelaxedLinearBinding(
                            module_path=f"layers.{index}.proj",
                            group_index=index,
                            source=DenseDeltaRowSource(
                                references[index],
                                [alternatives[index]],
                                rows_per_chunk=2,
                            ),
                            stats=linear_stat,
                        )],
                        device="cpu",
                        stats=block_stat,
                    )
                )

            streamed_input = input_values.clone().requires_grad_(True)
            streamed_output = streamed_model(streamed_input)
            self.assertTrue(all(
                wrapper.module.proj.weight.device.type == "meta"
                for wrapper in streamed_model.layers
            ))
            streamed_loss = streamed_output.square().mean()
            streamed_loss.backward()
            self.assertTrue(all(
                wrapper.module.proj.weight.device.type == "meta"
                for wrapper in streamed_model.layers
            ))

        self.assertTrue(torch.allclose(
            streamed_output, dense_output, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(
            streamed_loss, dense_loss, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(
            streamed_input.grad, dense_input.grad, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(
            streamed_logits.grad, dense_logits.grad,
            atol=1e-12, rtol=1e-12))
        self.assertTrue(all(stats.load_passes == 2 for stats in block_stats))
        self.assertTrue(all(stats.release_passes == 2 for stats in block_stats))
        self.assertTrue(all(
            stats.forward_reference_passes == 2
            and stats.backward_reference_passes == 1
            and stats.forward_alternative_passes == 2
            and stats.backward_alternative_passes == 1
            for stats in linear_stats
        ))


if __name__ == "__main__":
    unittest.main()
