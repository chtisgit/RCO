import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from native_store import NativeCandidateStore, NativeCandidateStoreWriter
from quant.ggml_native import GGMLNativeCodec, GGMLType


GGML_LIBRARY = os.environ.get("RCO_GGML_LIBRARY")


@unittest.skipUnless(GGML_LIBRARY, "RCO_GGML_LIBRARY is not configured")
class NativeGGMLStoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.codec = GGMLNativeCodec(GGML_LIBRARY)

    def test_chunked_quantization_matches_single_native_call(self):
        values = np.linspace(-3.0, 3.0, 7 * 64, dtype=np.float32).reshape(7, 64)
        for ggml_type in (GGMLType.Q2_0, GGMLType.Q4_0):
            with self.subTest(ggml_type=ggml_type.name):
                whole = self.codec.quantize_rows(values, ggml_type)
                chunked = b"".join(self.codec.iter_quantized_rows(
                    values, ggml_type, rows_per_chunk=3))
                geometry = self.codec.geometry(ggml_type, 64)
                self.assertEqual(chunked, whole)
                self.assertEqual(len(whole), 7 * geometry["row_size"])

                decoded = np.empty_like(values)
                self.assertIs(
                    self.codec.dequantize_rows_into(whole, ggml_type, decoded),
                    decoded,
                )
                self.assertTrue(np.isfinite(decoded).all())

    def test_atomic_store_handles_matrix_and_aggregated_experts(self):
        ordinary = np.arange(5 * 64, dtype=np.float32).reshape(5, 64) / 31
        experts = np.linspace(
            -2.0, 2.0, 3 * 4 * 64, dtype=np.float32).reshape(3, 4, 64)
        with tempfile.TemporaryDirectory() as directory:
            writer = NativeCandidateStoreWriter(
                directory,
                self.codec,
                source={"model": "synthetic", "revision": "test-fixture"},
            )
            entries = {}
            for tensor_name, values, provenance in (
                (
                    "blk.0.attn_q.weight",
                    ordinary,
                    {"source_tensor": "model.layers.0.self_attn.q_proj.weight"},
                ),
                (
                    "blk.0.ffn_gate_exps.weight",
                    experts,
                    {
                        "source_family": "model.layers.0.mlp.experts.*.gate_proj.weight",
                        "expert_order": [0, 1, 2],
                    },
                ),
            ):
                for ggml_type in (GGMLType.Q2_0, GGMLType.Q4_0):
                    entries[(tensor_name, ggml_type)] = writer.quantize_array(
                        tensor_name, ggml_type, values,
                        rows_per_chunk=2,
                        provenance=provenance,
                    )
            index_path = writer.finalize()

            index = json.loads(index_path.read_text())
            self.assertEqual(index["schema"], 1)
            self.assertEqual(index["tensor_count"], 2)
            self.assertEqual(index["candidate_count"], 4)
            q2 = entries[("blk.0.attn_q.weight", GGMLType.Q2_0)]
            q4 = entries[("blk.0.ffn_gate_exps.weight", GGMLType.Q4_0)]
            self.assertEqual(q2["gguf_shape"], [64, 5])
            self.assertEqual(q4["gguf_shape"], [64, 4, 3])
            self.assertEqual(q2["aligned_gguf_bytes"] % 32, 0)
            self.assertEqual(q4["aligned_gguf_bytes"] % 32, 0)
            self.assertEqual(list(Path(directory).rglob("*.tmp")), [])

            store = NativeCandidateStore(directory, self.codec)
            ordinary_out = np.empty_like(ordinary)
            experts_out = np.empty_like(experts)
            self.assertIs(store.decode_into(
                "blk.0.attn_q.weight", GGMLType.Q2_0, ordinary_out,
                rows_per_chunk=1), ordinary_out)
            self.assertIs(store.decode_into(
                "blk.0.ffn_gate_exps.weight", GGMLType.Q4_0, experts_out,
                rows_per_chunk=2), experts_out)
            self.assertTrue(np.isfinite(ordinary_out).all())
            self.assertTrue(np.isfinite(experts_out).all())
            self.assertEqual(len(store._verified), 2)

            streamed = list(store.iter_decoded_rows(
                "blk.0.ffn_gate_exps.weight", GGMLType.Q4_0,
                rows_per_chunk=5))
            self.assertEqual([start for start, _ in streamed], [0, 5, 10])
            self.assertTrue(all(chunk.shape[0] <= 5 for _, chunk in streamed))
            np.testing.assert_array_equal(
                np.concatenate([chunk for _, chunk in streamed]).reshape(
                    experts.shape),
                experts_out,
            )

            copied = b"".join(store.iter_payload(
                "blk.0.ffn_gate_exps.weight", GGMLType.Q4_0,
                chunk_bytes=7))
            self.assertEqual(hashlib.sha256(copied).hexdigest(), q4["sha256"])

    def test_corruption_is_rejected_before_decode(self):
        values = np.ones((2, 64), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            writer = NativeCandidateStoreWriter(
                directory, self.codec, source={"model": "synthetic"})
            entry = writer.quantize_array(
                "blk.0.proj.weight", GGMLType.Q4_0, values,
                provenance={"kind": "test"})
            writer.finalize()
            payload = Path(directory) / entry["path"]
            contents = bytearray(payload.read_bytes())
            contents[-1] ^= 1
            payload.write_bytes(contents)

            store = NativeCandidateStore(directory, self.codec)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                store.decode_into(
                    "blk.0.proj.weight", GGMLType.Q4_0,
                    np.empty_like(values))


if __name__ == "__main__":
    unittest.main()
