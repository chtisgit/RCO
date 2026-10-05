import hashlib
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from qwen36_runtime_provenance import (  # noqa: E402
    AUDIT_RUNTIME_IMPLEMENTATION_FILES,
    REQUIRED_RUNTIME_CONTROLS,
    SEARCH_RUNTIME_IMPLEMENTATION_FILES,
    runtime_provenance,
    runtime_report_status,
)
from audit_qwen36_gsq_upgrade_runtime import (  # noqa: E402
    _expected_authentic_loss,
)


class Qwen36RuntimeProvenanceTest(unittest.TestCase):
    def test_runtime_reports_pin_only_their_executed_driver(self):
        self.assertIn(
            "tools/audit_qwen36_gsq_upgrade_runtime.py",
            AUDIT_RUNTIME_IMPLEMENTATION_FILES)
        self.assertNotIn(
            "tools/search_qwen36_gsq_rco.py",
            AUDIT_RUNTIME_IMPLEMENTATION_FILES)
        self.assertIn(
            "tools/search_qwen36_gsq_rco.py",
            SEARCH_RUNTIME_IMPLEMENTATION_FILES)
        self.assertIn("src/search/hard.py", SEARCH_RUNTIME_IMPLEMENTATION_FILES)

    def test_authentic_baseline_shape_is_validated_before_model_work(self):
        self.assertEqual(_expected_authentic_loss({
            "results": {"authentic_gsq": {"mean_nll": 1.25}}
        }), 1.25)
        with self.assertRaisesRegex(RuntimeError, "authentic_gsq mean NLL"):
            _expected_authentic_loss({"mean_nll": 1.25})

    def test_runtime_report_is_complete_only_after_both_controls(self):
        self.assertEqual(runtime_report_status({}), "partial")
        retain = {
            "exact_mean_nll_match": True,
            "exact_logit_match": True,
            "logit_comparison_to_authentic": {"passed": True},
        }
        upgrades = {"logit_comparison_to_authentic": {"passed": True}}
        self.assertEqual(runtime_report_status({
            "authentic_retain": retain,
        }), "partial")
        self.assertEqual(runtime_report_status({
            "all_admitted_upgrades": upgrades,
        }), "partial")
        self.assertEqual(runtime_report_status({
            "authentic_retain": retain,
            "all_admitted_upgrades": upgrades,
        }), "complete")
        self.assertEqual(runtime_report_status({
            "authentic_retain": {
                "exact_mean_nll_match": False,
                "exact_logit_match": True,
                "logit_comparison_to_authentic": {"passed": True},
            }
        }), "failed")
        self.assertEqual(runtime_report_status({
            "authentic_retain": {},
            "all_admitted_upgrades": {},
        }), "failed")

    def test_runtime_provenance_is_stable_and_covers_dependencies(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gguf_python = root / "gguf-python"
            package = gguf_python / "gguf"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            reader = package / "gguf_reader.py"
            reader.write_text("reader-v1", encoding="utf-8")
            library = root / "libggml.so"
            library.write_bytes(b"ggml-v1")

            first = runtime_provenance(
                repository=repository, gguf_python=gguf_python,
                ggml_library=library,
                implementation_files=("tools/qwen36_runtime_provenance.py",))
            second = runtime_provenance(
                repository=repository, gguf_python=gguf_python,
                ggml_library=library,
                implementation_files=("tools/qwen36_runtime_provenance.py",))
            self.assertEqual(first, second)
            self.assertEqual(
                first["gguf_python_files"]["gguf/gguf_reader.py"],
                hashlib.sha256(b"reader-v1").hexdigest())
            self.assertEqual(
                first["ggml_shared_library_sha256"],
                hashlib.sha256(b"ggml-v1").hexdigest())
            self.assertTrue(first["rco_revision"])
            self.assertTrue(first["rco_implementation_sha256"])
            self.assertTrue(first["environment"]["python"])
            self.assertIn("torch", first["environment"]["packages"])

            reader.write_text("reader-v2", encoding="utf-8")
            changed = runtime_provenance(
                repository=repository, gguf_python=gguf_python,
                ggml_library=library,
                implementation_files=("tools/qwen36_runtime_provenance.py",))
            self.assertNotEqual(first["gguf_python_sha256"],
                                changed["gguf_python_sha256"])


if __name__ == "__main__":
    unittest.main()
