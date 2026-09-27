import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from native_gguf import (
    import_pinned_gguf,
    write_resumable_selected_native_gguf,
    write_selected_native_gguf,
)
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

    def test_resumable_writer_discards_only_uncommitted_tail(self):
        gguf = self.gguf
        ordinary = np.linspace(-1, 1, 5 * 64, dtype=np.float32).reshape(5, 64)
        second = np.linspace(-2, 2, 7 * 64, dtype=np.float32).reshape(7, 64)
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            reference = directory / "reference.gguf"
            writer = gguf.GGUFWriter(reference, arch="test")
            writer.add_tensor("first", ordinary)
            writer.add_tensor("second", second)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            store_writer = NativeCandidateStoreWriter(
                directory / "store", self.codec, source={"kind": "test"})
            first = store_writer.quantize_array(
                "first", GGMLType.Q2_0, ordinary,
                provenance={"kind": "ordinary"})
            second_metadata = store_writer.quantize_array(
                "second", GGMLType.Q4_0, second,
                provenance={"kind": "ordinary"})
            store_writer.finalize()
            store = NativeCandidateStore(directory / "store", self.codec)
            assignment = {
                "first": GGMLType.Q2_0,
                "second": GGMLType.Q4_0,
            }
            resumed_output = directory / "resumed.gguf"
            interrupted = write_resumable_selected_native_gguf(
                reference,
                resumed_output,
                store,
                assignment,
                gguf_python=GGUF_PYTHON,
                chunk_bytes=31,
                stop_after_tensors=1,
            )
            self.assertEqual(interrupted["status"], "incomplete")
            self.assertEqual(interrupted["completed_tensor_count"], 1)
            staging = Path(interrupted["staging"])
            committed_size = staging.stat().st_size
            with staging.open("ab") as handle:
                handle.write(b"uncommitted crash tail")
            state = json.loads(Path(interrupted["state"]).read_text())
            corrupt_offset = state["completed_tensors"][0]["data_offset"]
            with staging.open("r+b") as handle:
                handle.seek(corrupt_offset)
                original_byte = handle.read(1)
                handle.seek(corrupt_offset)
                handle.write(bytes([original_byte[0] ^ 0xFF]))
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                write_resumable_selected_native_gguf(
                    reference,
                    resumed_output,
                    store,
                    assignment,
                    gguf_python=GGUF_PYTHON,
                    resume=True,
                    chunk_bytes=31,
                )
            with staging.open("r+b") as handle:
                handle.seek(corrupt_offset)
                handle.write(original_byte)

            completed = write_resumable_selected_native_gguf(
                reference,
                resumed_output,
                store,
                assignment,
                gguf_python=GGUF_PYTHON,
                resume=True,
                chunk_bytes=31,
            )
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(completed["tensor_count"], 2)
            self.assertEqual(completed["selected_tensor_count"], 2)
            self.assertEqual(completed["max_copy_chunk_bytes"], 31)
            self.assertGreater(completed["output_bytes"], committed_size)
            self.assertFalse(staging.exists())
            self.assertFalse(Path(interrupted["state"]).exists())

            direct_output = directory / "direct.gguf"
            direct = write_resumable_selected_native_gguf(
                reference,
                direct_output,
                store,
                assignment,
                gguf_python=GGUF_PYTHON,
                chunk_bytes=31,
            )
            self.assertEqual(completed["output_sha256"], direct["output_sha256"])
            self.assertEqual(resumed_output.read_bytes(), direct_output.read_bytes())

            loaded = gguf.GGUFReader(resumed_output)
            tensors = {tensor.name: tensor for tensor in loaded.tensors}
            for name, expected_type, metadata in (
                ("first", GGMLType.Q2_0, first),
                ("second", GGMLType.Q4_0, second_metadata),
            ):
                tensor = tensors[name]
                self.assertEqual(int(tensor.tensor_type), int(expected_type))
                self.assertEqual(
                    hashlib.sha256(tensor.data.reshape(-1).tobytes()).hexdigest(),
                    metadata["sha256"],
                )

    def test_resumable_writer_rejects_changed_plan(self):
        gguf = self.gguf
        values = np.zeros((2, 64), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            reference = directory / "reference.gguf"
            writer = gguf.GGUFWriter(reference, arch="test")
            writer.add_tensor("known", values)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            store_writer = NativeCandidateStoreWriter(
                directory / "store", self.codec, source={"kind": "test"})
            store_writer.quantize_array(
                "known", GGMLType.Q2_0, values, provenance={"kind": "test"})
            store_writer.quantize_array(
                "known", GGMLType.Q4_0, values, provenance={"kind": "test"})
            store_writer.finalize()
            store = NativeCandidateStore(directory / "store", self.codec)
            output = directory / "changed.gguf"
            write_resumable_selected_native_gguf(
                reference,
                output,
                store,
                {"known": GGMLType.Q2_0},
                gguf_python=GGUF_PYTHON,
                stop_after_tensors=0,
            )
            with self.assertRaisesRegex(ValueError, "plan differs"):
                write_resumable_selected_native_gguf(
                    reference,
                    output,
                    store,
                    {"known": GGMLType.Q4_0},
                    gguf_python=GGUF_PYTHON,
                    resume=True,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
