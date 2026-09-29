import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from native_runtime import (
    NativeManifestRelaxedLinearSource,
    NativeManifestWeightStore,
)
from quant.ggml_native import GGMLType
from qwen35_native import Qwen35LinearAttentionGeometry, matrix_permutations
from search.relaxed import streaming_relaxed_linear


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "PyTorch is not installed")
class NativeManifestWeightStoreTest(unittest.TestCase):
    class CandidateStore:
        def __init__(self, candidates):
            self.candidates = candidates
            self.index = {
                "tensor_count": len(candidates),
                "tensors": {name: {} for name in candidates},
            }

        def metadata(self, name, candidate_type):
            return {"payload_bytes": self.candidates[name].nbytes // 4}

        def iter_decoded_rows(
            self, name, candidate_type, *, rows_per_chunk,
        ):
            rows = self.candidates[name].reshape(
                -1, self.candidates[name].shape[-1])
            for start in range(0, len(rows), rows_per_chunk):
                yield start, rows[start:start + rows_per_chunk].copy()

    class RelaxedCandidateStore:
        def __init__(self, candidates):
            self.candidates = candidates

        def metadata(self, name, candidate_type):
            values = self.candidates[(name, GGMLType(candidate_type))]
            return {
                "payload_bytes": values.nbytes // 4,
                "gguf_shape": list(reversed(values.shape)),
            }

        def iter_decoded_rows(
            self, name, candidate_type, *, rows_per_chunk,
        ):
            rows = self.candidates[
                (name, GGMLType(candidate_type))].reshape(
                    -1, self.candidates[
                        (name, GGMLType(candidate_type))].shape[-1])
            for start in range(0, len(rows), rows_per_chunk):
                yield start, rows[start:start + rows_per_chunk].copy()

        def iter_decoded_row_indices(
            self, name, candidate_type, row_indices, *, rows_per_chunk,
        ):
            rows = self.candidates[
                (name, GGMLType(candidate_type))].reshape(
                    -1, self.candidates[
                        (name, GGMLType(candidate_type))].shape[-1])
            selected = rows[np.asarray(row_indices)]
            for start in range(0, len(selected), rows_per_chunk):
                yield start, selected[start:start + rows_per_chunk].copy()

    class Model(nn.Module if DEPENDENCIES_AVAILABLE else object):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.embed_tokens = nn.Embedding(
                7, 4, dtype=torch.bfloat16)
            self.model.language_model.layers = nn.ModuleList([nn.Module()])
            layer = self.model.language_model.layers[0]
            layer.linear_attn = nn.Module()
            layer.linear_attn.out_proj = nn.Linear(
                8, 3, bias=False, dtype=torch.bfloat16)
            layer.linear_attn.in_proj_qkv = nn.Linear(
                5, 20, bias=False, dtype=torch.bfloat16)
            layer.mlp = nn.Module()
            layer.mlp.experts = nn.Module()
            layer.mlp.experts.register_parameter(
                "gate_up_proj",
                nn.Parameter(torch.zeros(2, 6, 4, dtype=torch.bfloat16)),
            )
            self.lm_head = nn.Linear(
                4, 7, bias=False, dtype=torch.bfloat16)

    def test_installs_permuted_matrices_and_fused_expert_views_in_chunks(self):
        geometry = Qwen35LinearAttentionGeometry(2, 4, 3, 2)
        out_source = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)
        _, out_columns = matrix_permutations(
            "model.layers.0.linear_attn.out_proj.weight",
            out_source.shape,
            geometry,
        )
        out_canonical = out_source[:, out_columns]

        qkv_source = np.arange(20 * 5, dtype=np.float32).reshape(20, 5)
        qkv_rows, _ = matrix_permutations(
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            qkv_source.shape,
            geometry,
        )
        qkv_canonical = qkv_source[qkv_rows]
        expert_source = (
            np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) + 100
        )

        candidates = {
            "blk.0.ssm_out.weight": out_canonical,
            "blk.0.ssm_qkv.weight": qkv_canonical,
            "blk.0.ffn_gate_exps.weight": expert_source,
        }
        manifest = {"entries": [
            {
                "rco_search": True,
                "destination_name": "blk.0.ssm_out.weight",
                "source_name": (
                    "model.language_model.layers.0.linear_attn.out_proj.weight"),
                "normalized_source_name": (
                    "model.layers.0.linear_attn.out_proj.weight"),
                "source_shape": [3, 8],
            },
            {
                "rco_search": True,
                "destination_name": "blk.0.ssm_qkv.weight",
                "source_name": (
                    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
                "normalized_source_name": (
                    "model.layers.0.linear_attn.in_proj_qkv.weight"),
                "source_shape": [20, 5],
            },
            {
                "rco_search": True,
                "destination_name": "blk.0.ffn_gate_exps.weight",
                "source_name": (
                    "model.language_model.layers.0.mlp.experts.gate_up_proj"),
                "normalized_source_name": (
                    "model.layers.0.mlp.experts.gate_proj.weight"),
                "source_shape": [2, 6, 4],
                "candidate_source_shape": [2, 3, 4],
                "source_view": {"axis": 1, "start": 0, "stop": 3},
            },
        ]}

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            (directory / "config.json").write_text(json.dumps({
                "text_config": {
                    "linear_num_key_heads": 2,
                    "linear_num_value_heads": 4,
                    "linear_key_head_dim": 3,
                    "linear_value_head_dim": 2,
                },
            }))
            adapter = NativeManifestWeightStore(
                self.CandidateStore(candidates), manifest, directory,
                rows_per_chunk=4,
            )
            model = self.Model()
            out_stats = adapter.install_layer_weight(
                model, "blk.0.ssm_out.weight", 2)
            qkv_stats = adapter.install_layer_weight(
                model, "blk.0.ssm_qkv.weight", 4)
            expert_stats = adapter.install_layer_weight(
                model, "blk.0.ffn_gate_exps.weight", 2)

        np.testing.assert_array_equal(
            model.model.language_model.layers[0].linear_attn.out_proj.weight
            .detach().float().numpy(),
            out_source,
        )
        np.testing.assert_array_equal(
            model.model.language_model.layers[0].linear_attn.in_proj_qkv.weight
            .detach().float().numpy(),
            qkv_source,
        )
        np.testing.assert_array_equal(
            model.model.language_model.layers[0].mlp.experts.gate_up_proj[
                :, :3].detach().float().numpy(),
            expert_source,
        )
        self.assertEqual(adapter.block_index("blk.0.ssm_out.weight"), 0)
        self.assertEqual(out_stats["max_decoded_fp32_bytes"], 3 * 8 * 4)
        self.assertEqual(qkv_stats["max_install_bf16_bytes"], 4 * 5 * 2)
        self.assertEqual(expert_stats["max_decoded_fp32_bytes"], 4 * 4 * 4)
        self.assertEqual(expert_stats["max_install_bf16_bytes"], 3 * 4 * 2)

    def test_routes_and_installs_embedding_and_output_candidates(self):
        embedding = np.arange(7 * 4, dtype=np.float32).reshape(7, 4)
        output = embedding + 100
        candidates = {
            "token_embd.weight": embedding,
            "output.weight": output,
        }
        manifest = {"entries": [
            {
                "rco_search": True,
                "destination_name": "token_embd.weight",
                "source_name": "model.language_model.embed_tokens.weight",
                "normalized_source_name": "model.embed_tokens.weight",
                "source_shape": [7, 4],
            },
            {
                "rco_search": True,
                "destination_name": "output.weight",
                "source_name": "lm_head.weight",
                "normalized_source_name": "lm_head.weight",
                "source_shape": [7, 4],
            },
        ]}

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            (directory / "config.json").write_text(json.dumps({
                "text_config": {
                    "linear_num_key_heads": 2,
                    "linear_num_value_heads": 4,
                    "linear_key_head_dim": 3,
                    "linear_value_head_dim": 2,
                },
            }))
            adapter = NativeManifestWeightStore(
                self.CandidateStore(candidates), manifest, directory,
                rows_per_chunk=3,
            )
            model = self.Model()
            adapter.install_layer_weight(model, "token_embd.weight", 2)
            adapter.install_layer_weight(model, "output.weight", 4)

        self.assertEqual(
            adapter.candidate_location("token_embd.weight"), "embedding")
        self.assertEqual(
            adapter.candidate_location("output.weight"), "lm_head")
        with self.assertRaisesRegex(ValueError, "outside a canonical block"):
            adapter.block_index("output.weight")
        np.testing.assert_array_equal(
            model.model.language_model.embed_tokens.weight.detach().float(),
            embedding,
        )
        np.testing.assert_array_equal(
            model.lm_head.weight.detach().float(), output)

    def test_streams_native_relaxed_deltas_in_hf_matrix_order(self):
        geometry = Qwen35LinearAttentionGeometry(2, 4, 3, 2)
        qkv_reference = (
            np.arange(20 * 5, dtype=np.float32).reshape(20, 5) / 19)
        qkv_alternative = qkv_reference + np.linspace(
            -0.25, 0.25, qkv_reference.size, dtype=np.float32,
        ).reshape(qkv_reference.shape)
        qkv_rows, _ = matrix_permutations(
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            qkv_reference.shape,
            geometry,
        )

        out_reference = (
            np.arange(3 * 8, dtype=np.float32).reshape(3, 8) / 7)
        out_alternative = out_reference + np.linspace(
            0.3, -0.3, out_reference.size, dtype=np.float32,
        ).reshape(out_reference.shape)
        _, out_columns = matrix_permutations(
            "model.layers.0.linear_attn.out_proj.weight",
            out_reference.shape,
            geometry,
        )

        fixtures = [
            (
                "blk.0.ssm_qkv.weight",
                "model.layers.0.linear_attn.in_proj_qkv.weight",
                qkv_reference,
                qkv_alternative,
                qkv_reference[qkv_rows],
                qkv_alternative[qkv_rows],
            ),
            (
                "blk.0.ssm_out.weight",
                "model.layers.0.linear_attn.out_proj.weight",
                out_reference,
                out_alternative,
                out_reference[:, out_columns],
                out_alternative[:, out_columns],
            ),
        ]

        for (name, normalized_name, reference, alternative,
             canonical_reference, canonical_alternative) in fixtures:
            with self.subTest(name=name):
                store = self.RelaxedCandidateStore({
                    (name, GGMLType.Q4_0): canonical_reference,
                    (name, GGMLType.Q2_0): canonical_alternative,
                })
                source = NativeManifestRelaxedLinearSource(
                    store,
                    {
                        "rco_search": True,
                        "destination_name": name,
                        "normalized_source_name": normalized_name,
                        "source_shape": list(reference.shape),
                    },
                    reference_type=GGMLType.Q4_0,
                    alternative_types=(GGMLType.Q2_0,),
                    geometry=geometry,
                    rows_per_chunk=3,
                )
                restored_reference = np.concatenate([
                    rows for _, rows in source.iter_reference_rows()
                ])
                restored_delta = np.concatenate([
                    rows for _, rows in source.iter_delta_rows(0)
                ])
                np.testing.assert_array_equal(
                    restored_reference, reference)
                np.testing.assert_allclose(
                    restored_delta, alternative - reference,
                    atol=0.0, rtol=0.0)

                torch.manual_seed(41)
                values = torch.randn(
                    2, reference.shape[1], requires_grad=True)
                logits = torch.tensor(
                    [0.3, -0.2], requires_grad=True)
                output = streaming_relaxed_linear(values, logits, source)
                loss = output.square().mean()
                loss.backward()

                dense_values = values.detach().clone().requires_grad_(True)
                dense_logits = logits.detach().clone().requires_grad_(True)
                probability = torch.softmax(dense_logits, dim=0)[0]
                mixed = torch.from_numpy(reference) + probability * (
                    torch.from_numpy(alternative)
                    - torch.from_numpy(reference))
                dense_output = torch.nn.functional.linear(
                    dense_values, mixed)
                dense_loss = dense_output.square().mean()
                dense_loss.backward()

                self.assertTrue(torch.allclose(
                    output, dense_output, atol=1e-5, rtol=1e-5))
                self.assertTrue(torch.allclose(
                    values.grad, dense_values.grad, atol=1e-5, rtol=1e-5))
                self.assertTrue(torch.allclose(
                    logits.grad, dense_logits.grad, atol=1e-5, rtol=1e-5))
                self.assertGreater(source.stats.reference_payload_bytes_read, 0)
                self.assertGreater(source.stats.alternative_payload_bytes_read, 0)
                self.assertLessEqual(
                    source.stats.max_resident_decoded_bytes,
                    3 * 3 * reference.shape[1] * 4,
                )

    def test_streams_one_native_expert_view_without_decoding_the_stack(self):
        geometry = Qwen35LinearAttentionGeometry(2, 4, 3, 2)
        reference = np.arange(
            3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4) / 9
        alternative = reference + np.linspace(
            -0.2, 0.2, reference.size, dtype=np.float32,
        ).reshape(reference.shape)
        name = "blk.0.ffn_gate_exps.weight"
        store = self.RelaxedCandidateStore({
            (name, GGMLType.Q4_0): reference,
            (name, GGMLType.Q2_0): alternative,
        })
        source = NativeManifestRelaxedLinearSource(
            store,
            {
                "rco_search": True,
                "destination_name": name,
                "normalized_source_name": (
                    "model.layers.0.mlp.experts.gate_proj.weight"),
                "source_shape": [3, 4, 4],
                "candidate_source_shape": [3, 2, 4],
                "source_view": {"axis": 1, "start": 0, "stop": 2},
            },
            reference_type=GGMLType.Q4_0,
            alternative_types=(GGMLType.Q2_0,),
            geometry=geometry,
            rows_per_chunk=1,
            expert_index=1,
        )

        decoded_reference = np.concatenate([
            rows for _, rows in source.iter_reference_rows()
        ])
        decoded_delta = np.concatenate([
            rows for _, rows in source.iter_delta_rows(0)
        ])
        np.testing.assert_array_equal(decoded_reference, reference[1])
        np.testing.assert_allclose(
            decoded_delta, alternative[1] - reference[1],
            atol=0.0, rtol=0.0)
        self.assertEqual(source.in_features, 4)
        self.assertEqual(source.out_features, 2)
        self.assertEqual(source.expert_index, 1)
        self.assertLessEqual(source.stats.max_resident_decoded_bytes, 2 * 4 * 4)


if __name__ == "__main__":
    unittest.main()
