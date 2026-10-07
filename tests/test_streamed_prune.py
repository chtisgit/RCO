import sys
import tempfile
from pathlib import Path
import unittest

import torch
from safetensors.torch import save_file


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoint_stream import SafeTensorPrefixLoader  # noqa: E402
from metrics import compute_kl_loss  # noqa: E402
from search.prune import install_wrappers, remove_wrappers  # noqa: E402
from search.streamed_prune import (  # noqa: E402
    StreamedPruneConfig,
    StreamedPruneObjective,
    StreamedPruneResult,
    StreamedPruneSearch,
    StreamedPruneStats,
    ste_survival,
    surrogate_routing_weights,
)


LAYERS, EXPERTS, TOP_K, VOCAB = 2, 8, 2, 64


def _tiny_config():
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )

    return Qwen3_5MoeTextConfig(
        vocab_size=VOCAB, hidden_size=16, num_hidden_layers=LAYERS,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        moe_intermediate_size=8, shared_expert_intermediate_size=8,
        num_experts=EXPERTS, num_experts_per_tok=TOP_K,
        linear_num_value_heads=2, linear_num_key_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=64, tie_word_embeddings=False)


def _tiny_model(meta=False):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForCausalLM,
    )

    config = _tiny_config()
    if meta:
        from accelerate import init_empty_weights

        with init_empty_weights(include_buffers=False):
            model = Qwen3_5MoeForCausalLM(config).double()
    else:
        model = Qwen3_5MoeForCausalLM(config).double()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "experts" in name or name.endswith("mlp.gate.weight"):
                    parameter.normal_(std=0.3)
    # The RCO adapter dispatches on the multimodal model type.
    model.config.model_type = "qwen3_5_moe"
    return model.eval()


class SurrogateRoutingTest(unittest.TestCase):
    def test_scales_each_token_by_its_variant_mask(self):
        weights = torch.tensor([[0.6, 0.4], [0.7, 0.3]])
        indices = torch.tensor([[0, 2], [1, 2]])
        survival = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        scaled = surrogate_routing_weights(
            weights, indices, survival, torch.tensor([0, 1]))
        torch.testing.assert_close(scaled, torch.tensor([[0.6, 0.0], [0.7, 0.3]]))


class SteSurvivalTest(unittest.TestCase):
    def test_hard_forward_prunes_exact_count_per_layer(self):
        alpha = torch.randn(LAYERS * EXPERTS, 2, requires_grad=True)
        noise = torch.randn(LAYERS * EXPERTS, 2)
        survival, hard = ste_survival(alpha, noise, 0.5, 3, LAYERS, EXPERTS)
        self.assertTrue(bool((hard.sum(dim=1) == 3).all()))
        torch.testing.assert_close(survival.detach(), 1.0 - hard.float())
        survival.sum().backward()
        self.assertGreater(float(alpha.grad.abs().sum()), 0.0)


class StreamedObjectiveTest(unittest.TestCase):
    def test_matches_resident_rco_wrapper_and_reloads_each_block_twice(self):
        torch.manual_seed(5)
        resident = _tiny_model()
        state = {name: value.detach().clone()
                 for name, value in resident.state_dict().items()}
        input_ids = torch.randint(0, VOCAB, (4, 9))
        teacher = torch.log_softmax(torch.randn(4, 8, VOCAB), dim=-1)
        values, indices = teacher.topk(5, dim=-1)
        values, indices = values.float(), indices.int()
        alpha = torch.randn(LAYERS * EXPERTS, 2, dtype=torch.float64)
        noises = [torch.randn(LAYERS * EXPERTS, 2, dtype=torch.float64)
                  for _ in range(2)]
        row_variant = torch.tensor([0, 0, 1, 1])

        # Resident RCO path: one compute_kl_loss per variant, scaled 1/V.
        resident_alpha = alpha.clone().requires_grad_(True)
        resident.requires_grad_(False)
        shared = {"ste_masks": None}
        wrappers = install_wrappers(resident, LAYERS, EXPERTS, shared)
        resident_losses = []
        try:
            for variant, noise in enumerate(noises):
                survival, _ = ste_survival(
                    resident_alpha, noise, 0.7, 3, LAYERS, EXPERTS)
                shared["ste_masks"] = survival
                rows = row_variant == variant
                loss = compute_kl_loss(
                    resident, input_ids[rows],
                    {"values": values[rows], "indices": indices[rows]}, topk=5)
                (loss / len(noises)).backward()
                resident_losses.append(float(loss))
        finally:
            remove_wrappers(wrappers)

        streamed_alpha = alpha.clone().requires_grad_(True)
        survival = torch.stack([
            ste_survival(streamed_alpha, noise, 0.7, 3, LAYERS, EXPERTS)[0]
            for noise in noises])
        with tempfile.TemporaryDirectory() as directory:
            save_file(state, Path(directory) / "model.safetensors")
            streamed = _tiny_model(meta=True)
            objective = StreamedPruneObjective(
                streamed, SafeTensorPrefixLoader(directory), device="cpu",
                position_chunk=5)
            result = objective.run(input_ids, survival, row_variant, values, indices)
            self.assertTrue(all(
                parameter.device.type == "meta"
                for parameter in streamed.parameters()))

        self.assertEqual(result.stats.block_loads, [2] * LAYERS)
        self.assertEqual(result.stats.block_releases, [2] * LAYERS)
        streamed_losses = [float(result.row_kl[row_variant == v].mean())
                           for v in range(2)]
        for got, want in zip(streamed_losses, resident_losses):
            self.assertAlmostEqual(got, want, places=5)
        self.assertAlmostEqual(result.objective, sum(resident_losses) / 2, places=5)
        self.assertGreater(float(resident_alpha.grad.abs().sum()), 0.0)
        torch.testing.assert_close(
            streamed_alpha.grad, resident_alpha.grad, rtol=1e-4, atol=1e-7)


class _QuadraticObjective:
    """Cheap differentiable stand-in for the streamed model."""

    def __init__(self):
        self.target = torch.linspace(-1, 1, LAYERS * EXPERTS).view(LAYERS, EXPERTS)

    def run(self, input_ids, survival, row_variant, values, indices):
        per_variant = ((survival - self.target) ** 2).sum(dim=(1, 2))
        objective = per_variant.mean()
        objective.backward()
        return StreamedPruneResult(
            per_variant.detach()[row_variant], float(objective),
            StreamedPruneStats())


class SearchTest(unittest.TestCase):
    def _search(self):
        config = StreamedPruneConfig(
            layers=LAYERS, experts=EXPERTS, prune_per_layer=3, steps=4,
            gumbel_samples=4, documents_per_sample=2, seed=11)
        alpha = torch.zeros(LAYERS * EXPERTS, 2)
        alpha[:, 1] = torch.log(torch.tensor(3 / 5))
        return StreamedPruneSearch(config, alpha)

    def test_antithetic_pairs_share_documents(self):
        search = self._search()
        _, survival, hard, documents, row_variant = search.sample(10)
        self.assertEqual(survival.shape, (4, LAYERS, EXPERTS))
        self.assertTrue(bool((hard.sum(dim=2) == 3).all()))
        self.assertEqual(row_variant.tolist(), [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(documents[0:2].tolist(), documents[2:4].tolist())
        self.assertEqual(documents[4:6].tolist(), documents[6:8].tolist())

    def test_steps_keep_budget_and_resume_exactly(self):
        inputs = torch.zeros(10, 3, dtype=torch.long)
        uninterrupted = self._search()
        for _ in range(3):
            record = uninterrupted.take_step(
                _QuadraticObjective(), inputs, inputs, inputs)
            self.assertAlmostEqual(record["expected_prune_per_layer"], 3.0, places=2)

        first = self._search()
        first.take_step(_QuadraticObjective(), inputs, inputs, inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            first.save(path)
            resumed = self._search()
            resumed.load(path)
        for _ in range(2):
            resumed.take_step(_QuadraticObjective(), inputs, inputs, inputs)
        torch.testing.assert_close(resumed.alpha, uninterrupted.alpha, rtol=0, atol=0)
        self.assertEqual(
            [item["documents"] for item in resumed.history],
            [item["documents"] for item in uninterrupted.history])

    def test_rejects_resume_with_other_config(self):
        search = self._search()
        state = search.state_dict()
        state["config"]["lr"] = 0.2
        with self.assertRaises(ValueError):
            self._search().load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
