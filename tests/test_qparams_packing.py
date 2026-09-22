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
        load_qparams,
        pack_qweight,
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

    def test_schema_two_can_stay_packed_or_decode_on_load(self):
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

        self.assertEqual(packed["schema"], 2)
        self.assertNotIn("qweight", packed)
        self.assertLess(packed["qweight_packed"].numel(), values.numel())
        self.assertTrue(torch.equal(decoded["qweight"], values))

    def test_rejects_out_of_range_code(self):
        with self.assertRaisesRegex(ValueError, "outside 2-bit range"):
            pack_qweight(torch.tensor([0, 4], dtype=torch.uint8), bits=2)


if __name__ == "__main__":
    unittest.main()
