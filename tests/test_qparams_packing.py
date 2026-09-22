import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from quant.qparams import (
        build_qparams_index,
        dequantize_from_qparams,
        load_qparams,
        pack_qweight,
        qparams_path,
        save_qparams,
        unpack_qweight,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class QparamsPackingTest(unittest.TestCase):
    def test_round_trip_every_supported_width_and_unaligned_shape(self):
        torch.manual_seed(11)
        shape = (13, 17)
        for bits in range(1, 9):
            with self.subTest(bits=bits):
                values = torch.randint(0, 1 << bits, shape, dtype=torch.uint8)
                packed = pack_qweight(values, bits, chunk_values=24)
                decoded = unpack_qweight(
                    packed, bits, shape, chunk_values=24)
                self.assertTrue(torch.equal(values, decoded))
                self.assertEqual(
                    packed.numel(),
                    (values.numel() * bits + 7) // 8,
                )

    def test_schema_three_is_atomic_verified_and_optionally_unpacked(self):
        bits = 3
        values = torch.arange(35, dtype=torch.uint8).reshape(5, 7) % (1 << bits)
        quantizer = SimpleNamespace(sym=True, perchannel=True)
        handle = SimpleNamespace(
            quantizer_dict={bits: quantizer},
            group_size=7,
            d_col=7,
            act_order=False,
            W_shape=values.shape,
            W_dtype=torch.bfloat16,
        )
        with tempfile.TemporaryDirectory() as directory:
            save_qparams(
                directory,
                bits=bits,
                qweight=values,
                scales=torch.ones(5, 1),
                zeros=torch.zeros(5, 1),
                perm=None,
                handle=handle,
            )
            packed = load_qparams(directory, bits, unpack=False)
            decoded = load_qparams(directory, bits, unpack=True)

            leftovers = list(Path(directory).glob("*.tmp"))

        self.assertEqual(packed["schema"], 3)
        self.assertEqual(leftovers, [])
        self.assertIn("checksums", packed)
        self.assertNotIn("qweight", packed)
        self.assertLess(packed["qweight_packed"].numel(), values.numel())
        self.assertTrue(torch.equal(decoded["qweight"], values))

    def test_schema_two_remains_readable(self):
        bits = 2
        values = torch.tensor([[0, 1, 2, 3]], dtype=torch.uint8)
        handle = SimpleNamespace(
            quantizer_dict={bits: SimpleNamespace(sym=True, perchannel=True)},
            group_size=4, d_col=4, act_order=False,
            W_shape=values.shape, W_dtype=torch.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            save_qparams(
                directory, bits=bits, qweight=values,
                scales=torch.ones(1, 1), zeros=torch.zeros(1, 1),
                perm=None, handle=handle)
            path = qparams_path(directory, bits)
            bundle = torch.load(path, map_location="cpu", weights_only=False)
            bundle["schema"] = 2
            bundle.pop("checksums")
            torch.save(bundle, path)
            loaded = load_qparams(directory, bits)
        self.assertTrue(torch.equal(loaded["qweight"], values))

    def test_checksum_detects_tensor_corruption(self):
        bits = 2
        values = torch.tensor([[0, 1, 2, 3]], dtype=torch.uint8)
        handle = SimpleNamespace(
            quantizer_dict={bits: SimpleNamespace(sym=False, perchannel=True)},
            group_size=4, d_col=4, act_order=False,
            W_shape=values.shape, W_dtype=torch.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            save_qparams(
                directory, bits=bits, qweight=values,
                scales=torch.ones(1, 1), zeros=torch.zeros(1, 1),
                perm=None, handle=handle)
            path = qparams_path(directory, bits)
            bundle = torch.load(path, map_location="cpu", weights_only=False)
            bundle["scales"][0, 0] = 2.0
            torch.save(bundle, path)
            with self.assertRaisesRegex(ValueError, "checksum mismatch.*scales"):
                load_qparams(directory, bits)

    def test_chunked_dequantization_restores_order_into_caller_buffer(self):
        codes = torch.tensor([
            [0, 1, 2, 3],
            [3, 2, 1, 0],
        ], dtype=torch.uint8)
        perm = torch.tensor([2, 0, 3, 1])
        bundle = {
            "qweight": codes,
            "scales": torch.tensor([[0.5, 1.0], [1.0, 2.0]]),
            "zeros": torch.tensor([[1.0, 2.0], [0.0, 1.0]]),
            "group_size": 2,
            "dtype": "float32",
            "perm": perm,
        }
        out = torch.empty(2, 4)
        actual = dequantize_from_qparams(
            bundle, out=out, restore_order=True, column_chunk_size=1)
        permuted = torch.empty(2, 4)
        for column in range(4):
            group = column // 2
            permuted[:, column] = (
                bundle["scales"][:, group]
                * (codes[:, column].float() - bundle["zeros"][:, group]))
        expected = torch.empty_like(permuted)
        expected[:, perm] = permuted
        self.assertIs(actual, out)
        self.assertTrue(torch.equal(actual, expected))

    def test_rejects_out_of_range_code(self):
        with self.assertRaisesRegex(ValueError, "outside 2-bit range"):
            pack_qweight(torch.tensor([0, 4], dtype=torch.uint8), bits=2)

    def test_candidate_index_records_paths_shapes_sizes_and_checksums(self):
        bits = 4
        values = torch.arange(12, dtype=torch.uint8).reshape(3, 4)
        handle = SimpleNamespace(
            quantizer_dict={bits: SimpleNamespace(sym=True, perchannel=True)},
            group_size=4, d_col=4, act_order=False,
            W_shape=values.shape, W_dtype=torch.bfloat16,
        )
        with tempfile.TemporaryDirectory() as directory:
            layer = Path(directory) / "model.layers.0.proj"
            layer.mkdir()
            save_qparams(
                layer, bits=bits, qweight=values,
                scales=torch.ones(3, 1), zeros=torch.zeros(3, 1),
                perm=None, handle=handle)
            output = build_qparams_index(directory)
            index = json.loads(Path(output).read_text())
        self.assertEqual(index["candidate_count"], 1)
        self.assertEqual(index["layer_count"], 1)
        entry = index["layers"][layer.name][str(bits)]
        self.assertEqual(entry["offset"], 0)
        self.assertEqual(entry["shape"], [3, 4])
        self.assertGreater(entry["file_bytes"], 0)
        self.assertEqual(len(entry["checksums"]["qweight_packed"]), 64)


if __name__ == "__main__":
    unittest.main()
