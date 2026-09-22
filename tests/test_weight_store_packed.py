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
    from quant.qparams import build_qparams_index, dequantize_from_qparams, save_qparams
    from store import LoadMode, WeightStore


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class PackedWeightStoreTest(unittest.TestCase):
    def test_indexes_and_decodes_qparams_without_dense_candidate(self):
        bits = 2
        codes = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.uint8)
        scales = torch.tensor([[0.5], [0.25]])
        zeros = torch.tensor([[1.0], [2.0]])
        handle = SimpleNamespace(
            quantizer_dict={bits: SimpleNamespace(sym=False, perchannel=True)},
            group_size=4,
            d_col=4,
            act_order=True,
            W_shape=codes.shape,
            W_dtype=torch.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            layer_path = Path(directory) / "model.layers.0.mlp.down_proj"
            layer_path.mkdir()
            perm = torch.tensor([2, 0, 3, 1])
            save_qparams(
                layer_path, bits=bits, qweight=codes, scales=scales,
                zeros=zeros, perm=perm, handle=handle)
            build_qparams_index(directory)

            store = WeightStore(
                directory, mode=LoadMode.LAZY, cache=False).load()
            self.assertEqual(store.get_layer_names(), [layer_path.name])
            self.assertEqual(store.get_available_bitwidths(layer_path.name), [bits])
            self.assertEqual(store.get_layer_numel(layer_path.name, bits), 8)
            self.assertGreater(store.get_layer_storage_bytes(layer_path.name, bits), 0)
            self.assertEqual(store._verified_qparams, set())
            actual = store.get_layer_weight(layer_path.name, bits)
            destination = torch.empty_like(actual)
            into = store.get_layer_weight_into(
                layer_path.name, bits, destination)

            expected = (scales * (codes.float() - zeros))[:, perm.argsort()]
            self.assertTrue(torch.equal(actual, expected))
            self.assertIs(into, destination)
            self.assertTrue(torch.equal(into, expected))
            self.assertEqual(
                store._verified_qparams, {(layer_path.name, bits)})
            self.assertFalse(store._index[layer_path.name][bits].loaded)


if __name__ == "__main__":
    unittest.main()
