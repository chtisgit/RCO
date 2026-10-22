import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_qwen36_post_release_q2_types as A  # noqa: E402
from audit_qwen36_q3k_viability import expert_names  # noqa: E402
from search.typed_upgrades import (  # noqa: E402
    TypedUpgradeStore, check_variant_keys, split_key, typed_key)

LIBRARY = (Path(__file__).resolve().parents[3]
           / "experiment/build-cpu/bin/libggml-base.so.0.24.0")


class FakeStore:
    def __init__(self, label):
        self.label = label
        self.installed = []

    def block_index(self, name):
        return int(name.split(".")[1])

    def install_layer_weight(self, model, name, choice):
        self.installed.append((name, choice))


class TypedUpgradesTest(unittest.TestCase):
    def test_keys(self):
        key = typed_key("IQ2_XS", "blk.3.ffn_up_exps.weight")
        self.assertEqual(split_key(key), ("IQ2_XS", "blk.3.ffn_up_exps.weight"))
        with self.assertRaises(ValueError):
            split_key("blk.3.ffn_up_exps.weight")
        with self.assertRaises(ValueError):
            check_variant_keys(["Q3_K:blk.1.x", "IQ2_XS:blk.1.x"])

    def test_dispatch(self):
        q3k, iq = FakeStore("Q3_K"), FakeStore("IQ2_XS")
        store = TypedUpgradeStore({"Q3_K": q3k, "IQ2_XS": iq})
        self.assertEqual(store.block_index("IQ2_XS:blk.7.ffn_gate_exps.weight"), 7)
        store.install_layer_weight(None, "Q3_K:blk.2.ffn_down_exps.weight", 1)
        self.assertEqual(q3k.installed, [("blk.2.ffn_down_exps.weight", 1)])
        self.assertEqual(iq.installed, [])
        with self.assertRaises(KeyError):
            store.block_index("IQ2_XXS:blk.1.ffn_up_exps.weight")


class ArmsTest(unittest.TestCase):
    def test_arms(self):
        names = expert_names()
        release = sorted(names[:23])
        arms = A.arm_assignments(release)
        self.assertEqual(len(arms["iq2_xs_gate_up"]), 80)
        self.assertEqual(len(arms["iq2_xs_down"]), 40)
        self.assertEqual(len(arms["iq2_xxs_all"]), 120)
        rest = arms["release_iq2_xs_rest"]
        self.assertEqual(sum(t == "Q3_K" for t in rest.values()), 23)
        self.assertEqual(sum(t == "IQ2_XS" for t in rest.values()), 97)
        per_type = {"Q2_0": 10, "Q3_K": 15, "IQ2_XS": 11, "IQ2_XXS": 9}
        self.assertEqual(A.kept_bytes(arms["base"], per_type), 1200)
        self.assertEqual(A.kept_bytes(arms["release"], per_type), 1200 + 23 * 5)


@unittest.skipUnless(LIBRARY.exists(), "pinned ggml library not built")
class IQCodecTest(unittest.TestCase):
    def test_round_trip(self):
        import numpy as np

        from quant.ggml_importance import quantize_rows_with_importance
        from quant.ggml_native_iq import IQ_TYPE_IDS, IQGGMLNativeCodec

        codec = IQGGMLNativeCodec(os.fspath(LIBRARY))
        self.assertEqual(codec.type_name(17), "IQ2_XS")
        self.assertEqual(codec.type_name(42), "Q2_0")
        self.assertEqual(codec.geometry(IQ_TYPE_IDS["IQ2_XS"], 2048)["row_size"], 592)
        rows = np.random.default_rng(0).standard_normal((4, 2048)).astype(np.float32)
        payload = quantize_rows_with_importance(codec, rows, 17, np.ones(2048, np.float32))
        decoded = np.empty_like(rows)
        codec.dequantize_rows_into(payload, 17, decoded)
        error = np.sqrt(np.square(decoded - rows).mean() / np.square(rows).mean())
        self.assertLess(error, 0.4)


if __name__ == "__main__":
    unittest.main()
