import sys
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense_oracle import tensor_sha256, write_block_oracle


class DenseOracleTest(unittest.TestCase):
    def test_writes_atomic_checksummed_caller_tensors(self):
        values = {
            "block_input": torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4),
            "block_output": torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4) / 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oracle.safetensors"
            report = write_block_oracle(
                path, values, metadata={"revision": "test", "layer": 0})
            with safe_open(path, framework="pt", device="cpu") as handle:
                actual = {
                    name: handle.get_tensor(name)
                    for name in handle.keys() if name != "__metadata_json__"
                }
                metadata = json.loads(bytes(
                    handle.get_tensor("__metadata_json__").tolist()).decode("utf-8"))
            leftovers = list(Path(directory).glob("*.tmp"))

        self.assertEqual(leftovers, [])
        self.assertEqual(report["metadata"]["layer"], "0")
        self.assertEqual(metadata, report["metadata"])
        self.assertGreater(report["bytes"], 0)
        self.assertEqual(actual.keys(), values.keys())
        for name, expected in values.items():
            self.assertTrue(torch.equal(actual[name], expected))
            self.assertEqual(report["tensors"][name]["sha256"], tensor_sha256(expected))


if __name__ == "__main__":
    unittest.main()
