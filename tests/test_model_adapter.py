import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grouping import build_layer_groups, build_moe_per_expert_groups
from model_adapter import (
    assert_complete_qwen35_text_coverage,
    classify_qwen35_tensor,
    get_model_adapter,
    get_text_config,
    is_qwen35_config,
)


class FakeNode:
    def get_submodule(self, path):
        node = self
        for component in path.split("."):
            node = getattr(node, component)
        return node


class FakeMlp(FakeNode):
    def __init__(self):
        self.gate = object()


class FakeLayer(FakeNode):
    def __init__(self, attention_name):
        setattr(self, attention_name, object())
        self.mlp = FakeMlp()


class FakeTextModel(FakeNode):
    def __init__(self):
        self.embed_tokens = object()
        self.layers = [
            FakeLayer("linear_attn"),
            FakeLayer("self_attn"),
        ]
        self.norm = object()


class FakeQwenWrapper(FakeNode):
    def __init__(self):
        text_config = SimpleNamespace(
            hidden_size=4,
            num_hidden_layers=2,
            num_experts=256,
            num_experts_per_tok=8,
        )
        self.config = SimpleNamespace(
            model_type="qwen3_5_moe",
            architectures=["Qwen3_5MoeForConditionalGeneration"],
            text_config=text_config,
        )
        self.model = FakeNode()
        self.model.language_model = FakeTextModel()
        self.lm_head = object()


class FakeQwenCausalLm(FakeNode):
    def __init__(self):
        self.config = SimpleNamespace(
            model_type="qwen3_5_moe",
            hidden_size=4,
            num_hidden_layers=2,
            num_experts=256,
            num_experts_per_tok=8,
        )
        self.model = FakeTextModel()
        self.lm_head = object()


class ModelAdapterTest(unittest.TestCase):
    def test_qwen35_nested_topology_and_text_config(self):
        model = FakeQwenWrapper()
        adapter = get_model_adapter(model)

        self.assertTrue(is_qwen35_config(model.config))
        self.assertIs(get_text_config(model.config), model.config.text_config)
        self.assertEqual(adapter.family, "qwen3_5")
        self.assertEqual(adapter.layers_path, "model.language_model.layers")
        self.assertEqual(len(adapter.layers), 2)
        self.assertEqual(adapter.hidden_size, 4)
        self.assertEqual(adapter.num_hidden_layers, 2)
        self.assertEqual(adapter.num_experts, 256)
        self.assertEqual(adapter.num_experts_per_tok, 8)
        self.assertIs(adapter.mlp(1), model.model.language_model.layers[1].mlp)
        self.assertEqual(adapter.attention_name(0), "linear_attn")
        self.assertEqual(adapter.attention_name(1), "self_attn")
        self.assertIs(adapter.final_norm(), model.model.language_model.norm)
        self.assertIs(adapter.lm_head(), model.lm_head)

    def test_architecture_name_also_detects_qwen35(self):
        config = SimpleNamespace(
            model_type="qwen3_5_vl",
            architectures=["Qwen3_5MoeForConditionalGeneration"],
            text_config=SimpleNamespace(),
        )
        self.assertTrue(is_qwen35_config(config))

    def test_qwen35_decoder_only_topology(self):
        adapter = get_model_adapter(FakeQwenCausalLm())
        self.assertEqual(adapter.text_model_path, "model")
        self.assertEqual(adapter.layers_path, "model.layers")
        self.assertEqual(len(adapter.layers), 2)

    def test_qwen35_tensor_classification(self):
        cases = {
            "model.language_model.layers.0.mlp.experts.7.gate_proj.weight_packed": "routed_expert",
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": "shared_expert",
            "model.language_model.layers.0.self_attn.q_proj.weight": "self_attention",
            "model.language_model.layers.1.linear_attn.in_proj_qkv.weight": "linear_attention",
            "model.language_model.layers.0.mlp.gate.weight": "router",
            "model.language_model.embed_tokens.weight": "embedding",
            "model.language_model.layers.0.input_layernorm.weight": "normalization",
            "model.language_model.norm.weight": "normalization",
            "lm_head.weight_packed": "lm_head",
            "model.language_model.layers.0.rotary_emb.inv_freq": "ordinary_text",
            "model.visual.blocks.0.attn.qkv.weight": "vision",
            "mtp.layers.0.mlp.gate_proj.weight": "mtp",
            "some.unexpected.tensor": "unknown",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(classify_qwen35_tensor(name), expected)

        assert_complete_qwen35_text_coverage(cases.keys() - {"some.unexpected.tensor"})
        with self.assertRaisesRegex(ValueError, "some.unexpected.tensor"):
            assert_complete_qwen35_text_coverage(cases.keys())

    def test_grouping_accepts_nested_language_model_paths(self):
        names = [
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.self_attn.k_proj",
            "model.language_model.layers.1.self_attn.q_proj",
        ]
        groups = build_layer_groups(names, [["q_proj", "k_proj"]])
        self.assertEqual(
            [g.layer_names for g in groups],
            [sorted(names[:2]), names[2:]],
        )

        experts = [
            "model.language_model.layers.0.mlp.experts.1.gate_proj",
            "model.language_model.layers.0.mlp.experts.1.up_proj",
            "model.language_model.layers.0.mlp.experts.2.down_proj",
        ]
        groups = build_moe_per_expert_groups(experts)
        self.assertEqual([len(g.layer_names) for g in groups], [2, 1])


if __name__ == "__main__":
    unittest.main()
