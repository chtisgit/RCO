import importlib.util
import math
import struct
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "audit_qwen36_gsq_gguf_nll.py"
SPEC = importlib.util.spec_from_file_location("audit_qwen36_gsq_gguf_nll", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class GgufDocumentNllTest(unittest.TestCase):
    def test_writes_little_endian_versioned_bundle(self):
        documents = [{"id": "doc-é", "text": "hello"}]
        tokens = [[17, -2, 99]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bundle"
            MODULE._write_bundle(path, documents, tokens)
            payload = path.read_bytes()

        self.assertEqual(payload[:8], b"RCONLL1\0")
        version, count = struct.unpack_from("<II", payload, 8)
        self.assertEqual((version, count), (1, 1))
        offset = 16
        id_size, = struct.unpack_from("<I", payload, offset)
        offset += 4
        self.assertEqual(payload[offset:offset + id_size].decode(), "doc-é")
        offset += id_size
        text_size, = struct.unpack_from("<I", payload, offset)
        offset += 4
        self.assertEqual(payload[offset:offset + text_size], b"hello")
        offset += text_size
        token_count, = struct.unpack_from("<I", payload, offset)
        offset += 4
        self.assertEqual(token_count, 3)
        self.assertEqual(list(struct.unpack_from("<3i", payload, offset)), tokens[0])

    def test_aggregate_weights_documents_by_predicted_tokens(self):
        result = MODULE._aggregate([
            {"id": "a", "predicted_token_count": 1, "nll_sum": 2.0,
             "mean_nll": 2.0, "seconds": 0.25},
            {"id": "b", "predicted_token_count": 3, "nll_sum": 3.0,
             "mean_nll": 1.0, "seconds": 0.75},
        ])
        self.assertEqual(result["predicted_token_count"], 4)
        self.assertEqual(result["mean_nll"], 1.25)
        self.assertTrue(math.isclose(result["perplexity"], math.exp(1.25)))
        self.assertEqual(result["elapsed_seconds"], 1.0)


if __name__ == "__main__":
    unittest.main()
