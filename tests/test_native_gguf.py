import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from native_gguf import import_pinned_gguf, write_selected_native_gguf
from native_store import NativeCandidateStore, NativeCandidateStoreWriter
from quant.ggml_native import GGMLNativeCodec, GGMLType


GGML_LIBRARY = os.environ.get("RCO_GGML_LIBRARY")
GGUF_PYTHON = os.environ.get("RCO_GGUF_PYTHON")


@unittest.skipUnless(
    GGML_LIBRARY and GGUF_PYTHON,
    "RCO_GGML_LIBRARY and RCO_GGUF_PYTHON are not configured",
)
class NativeGGUFTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.codec = GGMLNativeCodec(GGML_LIBRARY)
        cls.gguf = import_pinned_gguf(GGUF_PYTHON)

    def test_selected_payloads_are_framed_without_requantization(self):
        gguf = self.gguf
        ordinary = np.linspace(-1, 1, 5 * 64, dtype=np.float32).reshape(5, 64)
        experts = np.linspace(-2, 2, 3 * 7 * 64, dtype=np.float32).reshape(3, 7, 64)
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            reference = directory / "reference.gguf"
            writer = gguf.GGUFWriter(reference, arch="test")
            writer.add_tensor("blk.0.attn_q.weight", ordinary)
            writer.add_tensor("blk.0.ffn_gate_exps.weight", experts)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            store_writer = NativeCandidateStoreWriter(
                directory / "store", self.codec, source={"kind": "test"})
            q2 = store_writer.quantize_array(
                "blk.0.attn_q.weight", GGMLType.Q2_0, ordinary,
                provenance={"kind": "ordinary"})
            q4 = store_writer.quantize_array(
                "blk.0.ffn_gate_exps.weight", GGMLType.Q4_0, experts,
                provenance={"expert_order": [0, 1, 2]})
            store_writer.finalize()
            store = NativeCandidateStore(directory / "store", self.codec)
            output = directory / "mixed.gguf"
            selected = write_selected_native_gguf(
                reference,
                output,
                store,
                {
                    "blk.0.attn_q.weight": GGMLType.Q2_0,
                    "blk.0.ffn_gate_exps.weight": GGMLType.Q4_0,
                },
                gguf_python=GGUF_PYTHON,
            )
            self.assertEqual(len(selected), 2)
            loaded = gguf.GGUFReader(output)
            tensors = {tensor.name: tensor for tensor in loaded.tensors}
            for name, expected_type, metadata in (
                ("blk.0.attn_q.weight", GGMLType.Q2_0, q2),
                ("blk.0.ffn_gate_exps.weight", GGMLType.Q4_0, q4),
            ):
                tensor = tensors[name]
                self.assertEqual(int(tensor.tensor_type), int(expected_type))
                payload = tensor.data.reshape(-1).tobytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), metadata["sha256"])

    def test_unknown_assignment_is_rejected_without_output(self):
        gguf = self.gguf
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            reference = directory / "reference.gguf"
            writer = gguf.GGUFWriter(reference, arch="test")
            writer.add_tensor("known", np.zeros((2, 64), dtype=np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            store_writer = NativeCandidateStoreWriter(
                directory / "store", self.codec, source={"kind": "test"})
            store_writer.finalize()
            output = directory / "must-not-exist.gguf"
            with self.assertRaisesRegex(ValueError, "absent from reference"):
                write_selected_native_gguf(
                    reference,
                    output,
                    NativeCandidateStore(directory / "store", self.codec),
                    {"unknown": GGMLType.Q2_0},
                    gguf_python=GGUF_PYTHON,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
