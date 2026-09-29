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
    RelaxedExpertsBinding,
    RelaxedLinearBinding,
    RelaxedRouterBinding,
    StreamingRelaxedLinearStats,
    streaming_relaxed_causal_cross_entropy,
    streaming_relaxed_embedding,
    streaming_relaxed_linear,
)


class StreamingRelaxedLinearTest(unittest.TestCase):
    def test_sparse_embedding_matches_dense_relaxation_and_gradients(self):
        torch.manual_seed(23)
        dtype = torch.float64
        reference = torch.randn(11, 5, dtype=dtype)
        alternatives = [
            reference + 0.15 * torch.randn_like(reference),
            reference + 0.35 * torch.randn_like(reference),
        ]
        input_ids = torch.tensor([[7, 2, 7, 4], [2, 9, 4, 2]])
        upstream = torch.randn(2, 4, 5, dtype=dtype)
        initial_logits = torch.tensor([0.3, -0.6, 0.1], dtype=dtype)

        dense_logits = initial_logits.clone().requires_grad_(True)
        probabilities = torch.softmax(dense_logits, dim=0)
        dense_weight = reference.clone()
        for index, alternative in enumerate(alternatives):
            dense_weight = dense_weight + probabilities[index] * (
                alternative - reference)
        dense_output = F.embedding(input_ids, dense_weight)
        dense_loss = (dense_output * upstream).sum()
        dense_loss.backward()

        streamed_logits = initial_logits.clone().requires_grad_(True)
        source = DenseDeltaRowSource(
            reference, alternatives, rows_per_chunk=2)
        stats = StreamingRelaxedLinearStats()
        streamed_output = streaming_relaxed_embedding(
            input_ids, streamed_logits, source, stats=stats)
        streamed_loss = (streamed_output * upstream).sum()
        streamed_loss.backward()

        self.assertTrue(torch.allclose(
            streamed_output, dense_output, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(
            streamed_logits.grad, dense_logits.grad,
            atol=1e-12, rtol=1e-12))
        self.assertEqual(stats.forward_reference_passes, 1)
        self.assertEqual(stats.forward_alternative_passes, 2)
        self.assertEqual(stats.backward_reference_passes, 0)
        self.assertEqual(stats.backward_alternative_passes, 2)
        unique_rows = input_ids.unique().numel()
        expected_bytes = unique_rows * reference.shape[1] * reference.element_size()
        self.assertEqual(stats.forward_reference_bytes, expected_bytes)
        self.assertEqual(stats.forward_alternative_bytes, 2 * expected_bytes)

    def test_sparse_embedding_can_emit_model_dtype_from_float_logits(self):
        reference = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        source = DenseDeltaRowSource(
            reference, [reference + 1], rows_per_chunk=2)
        logits = torch.tensor([0.2, -0.1], dtype=torch.float32,
                              requires_grad=True)
        output = streaming_relaxed_embedding(
            torch.tensor([[1, 3]]), logits, source, dtype=torch.bfloat16)
        self.assertEqual(output.dtype, torch.bfloat16)
        output.float().sum().backward()
        self.assertEqual(logits.grad.dtype, torch.float32)

    def test_streamed_causal_ce_matches_dense_loss_and_gradients(self):
        torch.manual_seed(27)
        dtype = torch.float64
        reference = torch.randn(13, 5, dtype=dtype)
        alternatives = [
            reference + 0.1 * torch.randn_like(reference),
            reference + 0.25 * torch.randn_like(reference),
        ]
        initial_hidden = torch.randn(2, 5, 5, dtype=dtype)
        labels = torch.tensor([
            [1, 8, 2, 11, 4],
            [3, 6, 9, 5, 12],
        ])
        loss_mask = torch.tensor([
            [False, True, False, True, True],
            [False, False, True, True, False],
        ])
        initial_logits = torch.tensor([-0.2, 0.5, 0.1], dtype=dtype)

        dense_hidden = initial_hidden.clone().requires_grad_(True)
        dense_logits = initial_logits.clone().requires_grad_(True)
        probabilities = torch.softmax(dense_logits, dim=0)
        dense_weight = reference.clone()
        for index, alternative in enumerate(alternatives):
            dense_weight = dense_weight + probabilities[index] * (
                alternative - reference)
        full_logits = F.linear(dense_hidden[:, :-1], dense_weight)
        active = loss_mask[:, 1:].reshape(-1)
        dense_loss = F.cross_entropy(
            full_logits.reshape(-1, reference.shape[0])[active],
            labels[:, 1:].reshape(-1)[active],
        )
        dense_loss.backward()

        streamed_hidden = initial_hidden.clone().requires_grad_(True)
        streamed_logits = initial_logits.clone().requires_grad_(True)
        stats = StreamingRelaxedLinearStats()
        streamed_loss, token_count = streaming_relaxed_causal_cross_entropy(
            streamed_hidden,
            labels,
            streamed_logits,
            DenseDeltaRowSource(
                reference, alternatives, rows_per_chunk=3),
            loss_mask=loss_mask,
            stats=stats,
        )
        streamed_loss.backward()

        self.assertEqual(token_count, int(active.sum()))
        self.assertTrue(torch.allclose(
            streamed_loss, dense_loss, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(
            streamed_hidden.grad, dense_hidden.grad,
            atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(
            streamed_logits.grad, dense_logits.grad,
            atol=1e-12, rtol=1e-12))
        self.assertEqual(stats.forward_reference_passes, 1)
        self.assertEqual(stats.backward_reference_passes, 1)
        self.assertEqual(stats.forward_alternative_passes, 2)
        self.assertEqual(stats.backward_alternative_passes, 2)
        self.assertEqual(
            stats.max_materialized_chunk_bytes,
            3 * reference.shape[1] * reference.element_size(),
        )

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

    def test_checkpointed_fused_experts_match_dense_relaxation(self):
        class Experts(nn.Module):
            def __init__(self, device="cpu"):
                super().__init__()
                self.num_experts = 3
                self.hidden_dim = 4
                self.intermediate_dim = 5
                self.gate_up_proj = nn.Parameter(torch.randn(
                    3, 10, 4, dtype=torch.float64, device=device))
                self.down_proj = nn.Parameter(torch.randn(
                    3, 4, 5, dtype=torch.float64, device=device))
                self.act_fn = F.silu

            def forward(self, hidden_states, top_k_index, top_k_weights):
                final = torch.zeros_like(hidden_states)
                mask = F.one_hot(
                    top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
                for expert_index in range(self.num_experts):
                    top_k_pos, token_index = torch.where(mask[expert_index])
                    current = hidden_states[token_index]
                    gate, up = F.linear(
                        current, self.gate_up_proj[expert_index]).chunk(2, -1)
                    current = F.silu(gate) * up
                    current = F.linear(current, self.down_proj[expert_index])
                    current = current * top_k_weights[
                        token_index, top_k_pos, None]
                    final.index_add_(0, token_index, current)
                return final

        class Block(nn.Module):
            def __init__(self, device="cpu"):
                super().__init__()
                self.experts = Experts(device=device)

            def forward(self, hidden, *, top_k_index, top_k_weights):
                return hidden + self.experts(
                    hidden, top_k_index, top_k_weights)

        class Model(nn.Module):
            def __init__(self, device="cpu"):
                super().__init__()
                self.blocks = nn.ModuleList([Block(device=device)])

        torch.manual_seed(47)
        checkpoint_model = Model()
        state = {
            name: value.detach().clone()
            for name, value in checkpoint_model.state_dict().items()
        }
        gate_reference = torch.randn(3, 5, 4, dtype=torch.float64)
        up_reference = torch.randn(3, 5, 4, dtype=torch.float64)
        down_reference = torch.randn(3, 4, 5, dtype=torch.float64)
        references = {
            "gate": gate_reference,
            "up": up_reference,
            "down": down_reference,
        }
        alternatives = {
            name: value + 0.2 * torch.randn_like(value)
            for name, value in references.items()
        }
        input_values = torch.randn(6, 4, dtype=torch.float64)
        top_k_index = torch.tensor([
            [0, 1], [1, 2], [2, 0], [0, 2], [1, 0], [2, 1],
        ])
        top_k_weights = torch.tensor([
            [0.7, 0.3], [0.6, 0.4], [0.8, 0.2],
            [0.55, 0.45], [0.65, 0.35], [0.75, 0.25],
        ], dtype=torch.float64)
        initial_logits = torch.tensor([
            [0.2, -0.1], [-0.4, 0.3], [0.5, -0.2],
        ], dtype=torch.float64)

        dense_input = input_values.clone().requires_grad_(True)
        dense_logits = initial_logits.clone().requires_grad_(True)
        probabilities = torch.softmax(dense_logits, dim=-1)[:, 0]
        mixed = {
            name: reference + probabilities[index] * (
                alternatives[name] - reference)
            for index, (name, reference) in enumerate(references.items())
        }
        dense_expert_output = torch.zeros_like(dense_input)
        expert_mask = F.one_hot(top_k_index, num_classes=3).permute(2, 1, 0)
        for expert_index in range(3):
            top_k_pos, token_index = torch.where(expert_mask[expert_index])
            current = dense_input[token_index]
            gate = F.linear(current, mixed["gate"][expert_index])
            up = F.linear(current, mixed["up"][expert_index])
            current = F.silu(gate) * up
            current = F.linear(current, mixed["down"][expert_index])
            current = current * top_k_weights[token_index, top_k_pos, None]
            dense_expert_output.index_add_(0, token_index, current)
        dense_output = dense_input + dense_expert_output
        dense_loss = dense_output.square().mean()
        dense_loss.backward()

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            save_file(state, directory / "model.safetensors")
            loader = SafeTensorPrefixLoader(directory)
            streamed_model = Model(device="meta")
            streamed_logits = initial_logits.clone().requires_grad_(True)
            operation_stats = {
                name: StreamingRelaxedLinearStats() for name in references
            }

            def source_factory(projection, expert_index):
                return DenseDeltaRowSource(
                    references[projection][expert_index],
                    [alternatives[projection][expert_index]],
                    rows_per_chunk=2,
                )

            block_stats = CheckpointedRelaxedBlockStats()
            wrapper = CheckpointedStreamingRelaxedBlock(
                streamed_model.blocks[0],
                model=streamed_model,
                path="blocks.0",
                checkpoint_loader=loader,
                logits=streamed_logits,
                bindings=[],
                expert_bindings=[RelaxedExpertsBinding(
                    module_path="blocks.0.experts",
                    gate_group_index=0,
                    up_group_index=1,
                    down_group_index=2,
                    source_factory=source_factory,
                    gate_stats=operation_stats["gate"],
                    up_stats=operation_stats["up"],
                    down_stats=operation_stats["down"],
                )],
                device="cpu",
                stats=block_stats,
            )
            streamed_model.blocks[0] = wrapper
            streamed_input = input_values.clone().requires_grad_(True)
            streamed_output = wrapper(
                streamed_input,
                top_k_index=top_k_index,
                top_k_weights=top_k_weights,
            )
            self.assertEqual(
                wrapper.module.experts.gate_up_proj.device.type, "meta")
            streamed_loss = streamed_output.square().mean()
            streamed_loss.backward()
            self.assertEqual(
                wrapper.module.experts.gate_up_proj.device.type, "meta")

        self.assertTrue(torch.allclose(
            streamed_output, dense_output, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(
            streamed_input.grad, dense_input.grad, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(
            streamed_logits.grad, dense_logits.grad,
            atol=1e-12, rtol=1e-12))
        self.assertEqual(block_stats.load_passes, 2)
        self.assertEqual(block_stats.release_passes, 2)
        self.assertTrue(all(
            stats.forward_reference_passes == 6
            and stats.backward_reference_passes == 3
            for stats in operation_stats.values()
        ))

    def test_checkpointed_direct_weight_router_matches_dense_relaxation(self):
        class Router(nn.Module):
            def __init__(self, device="cpu"):
                super().__init__()
                self.top_k = 2
                self.num_experts = 3
                self.hidden_dim = 4
                self.weight = nn.Parameter(torch.randn(
                    3, 4, dtype=torch.float64, device=device))

            def forward(self, hidden_states):
                hidden_states = hidden_states.reshape(-1, self.hidden_dim)
                router_logits = F.linear(hidden_states, self.weight)
                probabilities = F.softmax(router_logits, dtype=torch.float, dim=-1)
                values, indices = torch.topk(probabilities, self.top_k, dim=-1)
                values /= values.sum(dim=-1, keepdim=True)
                return router_logits, values.to(router_logits.dtype), indices

        class Block(nn.Module):
            def __init__(self, device="cpu"):
                super().__init__()
                self.router = Router(device=device)

            def forward(self, hidden):
                logits, scores, _ = self.router(hidden)
                return logits + scores.sum(dim=-1, keepdim=True)

        class Model(nn.Module):
            def __init__(self, device="cpu"):
                super().__init__()
                self.blocks = nn.ModuleList([Block(device=device)])

        torch.manual_seed(53)
        checkpoint_model = Model()
        state = {
            name: value.detach().clone()
            for name, value in checkpoint_model.state_dict().items()
        }
        reference = torch.randn(3, 4, dtype=torch.float64)
        alternative = reference + 0.1 * torch.randn_like(reference)
        input_values = torch.randn(2, 3, 4, dtype=torch.float64)
        initial_logits = torch.tensor([[0.3, -0.4]], dtype=torch.float64)

        dense_input = input_values.clone().requires_grad_(True)
        dense_logits = initial_logits.clone().requires_grad_(True)
        probability = torch.softmax(dense_logits[0], dim=0)[0]
        mixed = reference + probability * (alternative - reference)
        router_logits = F.linear(dense_input.reshape(-1, 4), mixed)
        probabilities = F.softmax(router_logits, dtype=torch.float, dim=-1)
        scores, _ = torch.topk(probabilities, 2, dim=-1)
        scores /= scores.sum(dim=-1, keepdim=True)
        dense_output = router_logits + scores.to(
            router_logits.dtype).sum(dim=-1, keepdim=True)
        dense_loss = dense_output.square().mean()
        dense_loss.backward()

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            save_file(state, directory / "model.safetensors")
            streamed_model = Model(device="meta")
            streamed_logits = initial_logits.clone().requires_grad_(True)
            stats = StreamingRelaxedLinearStats()
            wrapper = CheckpointedStreamingRelaxedBlock(
                streamed_model.blocks[0],
                model=streamed_model,
                path="blocks.0",
                checkpoint_loader=SafeTensorPrefixLoader(directory),
                logits=streamed_logits,
                bindings=[],
                router_bindings=[RelaxedRouterBinding(
                    module_path="blocks.0.router",
                    group_index=0,
                    source=DenseDeltaRowSource(
                        reference, [alternative], rows_per_chunk=2),
                    stats=stats,
                )],
                device="cpu",
            )
            streamed_model.blocks[0] = wrapper
            streamed_input = input_values.clone().requires_grad_(True)
            streamed_output = wrapper(streamed_input)
            streamed_loss = streamed_output.square().mean()
            streamed_loss.backward()

        self.assertTrue(torch.allclose(
            streamed_output, dense_output, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(
            streamed_input.grad, dense_input.grad, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(
            streamed_logits.grad, dense_logits.grad,
            atol=1e-12, rtol=1e-12))
        self.assertEqual(stats.forward_reference_passes, 2)
        self.assertEqual(stats.backward_reference_passes, 1)


if __name__ == "__main__":
    unittest.main()
