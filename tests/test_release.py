import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_qwen36_release_vision as V  # noqa: E402
import build_qwen36_release_gguf as B  # noqa: E402


def _field(types, value):
    return SimpleNamespace(types=types, contents=lambda: value)


def _reader(name, embd=b"\x01\x02", count=232):
    fields = {"GGUF.version": _field([4], 3), "general.name": _field([8], name),
              "qwen35moe.expert_count": _field([4], count)}
    tensors = [SimpleNamespace(name=n, tensor_type=SimpleNamespace(name=t), shape=[2],
                               data=np.frombuffer(d, dtype=np.uint8))
               for n, t, d in (("token_embd.weight", "Q6_K", embd),
                               ("blk.0.ffn_up_exps.weight", "Q3_K", b"\x03\x04"))]
    return SimpleNamespace(fields=fields, tensors=tensors)


class ReleaseCompareTest(unittest.TestCase):
    def test_only_the_name_changes(self):
        v = B.compare(_reader("GSQ-hybrid"), _reader("GSQ-RCO"), "GSQ-RCO")
        self.assertTrue(v["metadata_identical_except_name"] and v["name_set"])
        self.assertTrue(v["tensors_identical"] and v["tensor_order_identical"])
        self.assertEqual(v["name_before"], "GSQ-hybrid")
        self.assertEqual(v["tensor_types"], {"Q3_K": 1, "Q6_K": 1})

    def test_other_changes_are_caught(self):
        v = B.compare(_reader("a"), _reader("b", count=256), "b")
        self.assertEqual(v["metadata_differing"], ["qwen35moe.expert_count"])
        self.assertFalse(B.compare(_reader("a"), _reader("b", embd=b"\x01\x00"), "b")
                         ["tensors_identical"])
        self.assertFalse(B.compare(_reader("a"), _reader("b"), "c")["name_set"])


class HeadlineTest(unittest.TestCase):
    def test_reads_headline(self):
        self.assertTrue(V.reads_headline("MEN WALK ON MOON"))
        self.assertTrue(V.reads_headline('"Men Walk on Moon"'))
        self.assertTrue(V.reads_headline("**MEN WALK\nON MOON**"))
        self.assertFalse(V.reads_headline("Man walks on the Moon"))
        self.assertFalse(V.reads_headline(""))


if __name__ == "__main__":
    unittest.main()
