import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from quant.compressed_layout import (
        compressed_state_from_bundle,
        pack_codes_to_int32,
        require_uniform_routed_expert_bits,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CompressedLayoutTest(unittest.TestCase):
    @staticmethod
    def _unpack_words(words, bits, columns):
        unsigned = words.to(torch.int64) & 0xFFFFFFFF
        output = torch.empty(words.shape[0], columns, dtype=torch.uint8)
        mask = (1 << bits) - 1
        for column in range(columns):
            start = column * bits
            word = start // 32
            offset = start % 32
            value = unsigned[:, word] >> offset
            if offset + bits > 32:
                value |= unsigned[:, word + 1] << (32 - offset)
            output[:, column] = (value & mask).to(torch.uint8)
        return output

    def test_row_aligned_int32_packing_round_trips_every_width(self):
        torch.manual_seed(19)
        for bits in range(1, 9):
            for columns in (1, 17, 31, 32, 33, 67):
                with self.subTest(bits=bits, columns=columns):
                    codes = torch.randint(
                        0, 1 << bits, (3, columns), dtype=torch.uint8)
                    packed = pack_codes_to_int32(codes, bits)
                    self.assertEqual(packed.dtype, torch.int32)
                    self.assertEqual(
                        packed.shape,
                        (3, (columns * bits + 31) // 32),
                    )
                    self.assertTrue(torch.equal(
                        self._unpack_words(packed, bits, columns), codes))

    def test_bundle_mapping_preserves_codes_scales_and_shape(self):
        codes = torch.tensor([
            [0, 1, 2, 3, 0],
            [3, 2, 1, 0, 3],
        ], dtype=torch.uint8)
        scales = torch.tensor([[0.5, 0.25], [1.0, 2.0]])
        bundle = {
            "bits": 2,
            "shape": (2, 5),
            "qweight_shape": (2, 5),
            "qweight": codes,
            "scales": scales,
            "zeros": torch.full((2, 2), 2.0),
            "group_size": 4,
            "sym": True,
            "act_order": False,
            "perm": None,
        }
        state = compressed_state_from_bundle(bundle)
        self.assertEqual(set(state), {
            "weight_packed", "weight_scale", "weight_shape"})
        self.assertTrue(torch.equal(state["weight_scale"], scales))
        self.assertEqual(state["weight_shape"].tolist(), [2, 5])
        self.assertTrue(torch.equal(
            self._unpack_words(state["weight_packed"], 2, 5), codes))

    def test_bundle_mapping_rejects_unproven_variants(self):
        base = {
            "bits": 2,
            "shape": (1, 4),
            "qweight_shape": (1, 4),
            "qweight": torch.tensor([[0, 1, 2, 3]], dtype=torch.uint8),
            "scales": torch.ones(1, 1),
            "zeros": torch.full((1, 1), 2.0),
            "group_size": 4,
            "sym": True,
            "act_order": False,
            "perm": None,
        }
        for update, message in (
            ({"sym": False}, "requires symmetric"),
            ({"act_order": True}, "act_order=False"),
            ({"zeros": torch.ones(1, 1)}, "zero point"),
            ({"scales": torch.ones(1, 2), "zeros": torch.full((1, 2), 2.0)},
             "scale shape"),
        ):
            with self.subTest(update=update):
                bundle = {**base, **update}
                with self.assertRaisesRegex(ValueError, message):
                    compressed_state_from_bundle(bundle)

    def test_fused_qwen_experts_require_one_packed_width(self):
        uniform = {
            "model.layers.0.mlp.experts.0.gate_proj": 2,
            "model.layers.1.mlp.experts.7.down_proj": 2,
            "model.layers.0.self_attn.q_proj": 4,
        }
        self.assertEqual(require_uniform_routed_expert_bits(uniform), 2)
        self.assertIsNone(require_uniform_routed_expert_bits({
            "model.layers.0.self_attn.q_proj": 2,
            "model.layers.1.self_attn.o_proj": 4,
        }))
        mixed = dict(uniform)
        mixed["model.layers.1.mlp.experts.7.down_proj"] = 4
        with self.assertRaisesRegex(ValueError, "one packed bit width"):
            require_uniform_routed_expert_bits(mixed)


if __name__ == "__main__":
    unittest.main()
