import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from audit_qwen36_chat_kl_gguf import BUNDLE_MAGIC, TOP_K, scored_positions, write_bundle  # noqa: E402


class ChatKLBundleTest(unittest.TestCase):
    def test_positions_score_the_next_token(self):
        conversation = {"id": "c", "tokens": [5, 6, 7, 8, 9], "scored": [1, 0, 1, 1, 0]}
        # scored[0] never counts; position t scores token t + 1.
        self.assertEqual(scored_positions(conversation).tolist(), [1, 2])

    def test_bundle_layout(self):
        conversations = [{"id": "a", "tokens": [1, 2, 3, 4], "scored": [0, 0, 1, 1]},
                         {"id": "bb", "tokens": [7, 8], "scored": [0, 1]}]
        reference = []
        for conversation in conversations:
            count = scored_positions(conversation).size
            reference.append((np.full((count, TOP_K), -0.5, dtype=np.float32),
                              np.arange(count * TOP_K, dtype=np.int32).reshape(count, TOP_K)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle"
            write_bundle(path, conversations, reference)
            data = path.read_bytes()
        self.assertEqual(data[:8], BUNDLE_MAGIC)
        version, top_k, count = struct.unpack_from("<III", data, 8)
        self.assertEqual((version, top_k, count), (1, TOP_K, 2))
        offset = 20
        for conversation, (values, indices) in zip(conversations, reference):
            (size,) = struct.unpack_from("<I", data, offset)
            offset += 4
            self.assertEqual(data[offset:offset + size].decode(), conversation["id"])
            offset += size
            (tokens,) = struct.unpack_from("<I", data, offset)
            offset += 4
            self.assertEqual(np.frombuffer(data, np.int32, tokens, offset).tolist(),
                             conversation["tokens"])
            offset += 4 * tokens
            (scored,) = struct.unpack_from("<I", data, offset)
            offset += 4
            self.assertEqual(np.frombuffer(data, np.int32, scored, offset).tolist(),
                             scored_positions(conversation).tolist())
            offset += 4 * scored
            np.testing.assert_array_equal(
                np.frombuffer(data, np.int32, scored * TOP_K, offset), indices.reshape(-1))
            offset += 4 * scored * TOP_K
            np.testing.assert_array_equal(
                np.frombuffer(data, np.float32, scored * TOP_K, offset), values.reshape(-1))
            offset += 4 * scored * TOP_K
        self.assertEqual(offset, len(data))

    def test_reference_must_match_positions(self):
        conversation = {"id": "a", "tokens": [1, 2, 3], "scored": [0, 1, 1]}
        bad = (np.zeros((1, TOP_K), np.float32), np.zeros((1, TOP_K), np.int32))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                write_bundle(Path(directory) / "bundle", [conversation], [bad])


if __name__ == "__main__":
    unittest.main()
