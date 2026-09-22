import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from audit_tensor_coverage import (
    build_report,
    names_from_conversion_manifest,
    names_from_safetensors_index,
)


class TensorCoverageAuditTest(unittest.TestCase):
    def test_reads_safetensors_index_without_tensor_payloads(self):
        data = {
            "weight_map": {
                "model.language_model.embed_tokens.weight": "model-1.safetensors",
                "model.visual.patch_embed.weight": "model-1.safetensors",
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors.index.json"
            path.write_text(json.dumps(data))
            names = names_from_safetensors_index(path)
            report = build_report(names, path)

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["text_tensor_count"], 1)
        self.assertEqual(report["omitted_tensor_count"], 1)

    def test_reads_unique_manifest_source_names(self):
        name = "model.language_model.layers.0.self_attn.q_proj.weight"
        data = {
            "families": [
                {"source_tensors": {name: {"shape": [4, 4]}}},
                {"source_tensors": {name: {"shape": [4, 4]}}},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(data))
            self.assertEqual(names_from_conversion_manifest(path), [name])

    def test_unknown_tensor_fails_report(self):
        report = build_report(["unexpected.weight"], Path("index.json"))
        self.assertEqual(report["status"], "FAIL_UNKNOWN_TENSORS")
        self.assertEqual(report["unknown_tensors"], ["unexpected.weight"])


if __name__ == "__main__":
    unittest.main()
