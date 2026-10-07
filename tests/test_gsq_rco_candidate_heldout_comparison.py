import copy
import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from compare_qwen36_gsq_rco_candidate_heldout import compare  # noqa: E402


def _report(model_sha, mean_nlls):
    documents = [
        {"id": f"code-{index}", "mean_nll": value, "nll_sum": value * 4,
         "predicted_token_count": 4}
        for index, value in enumerate(mean_nlls)
    ]
    return {
        "status": "complete",
        "documents": documents,
        "aggregate": {
            "document_ids": [document["id"] for document in documents],
            "mean_nll": sum(mean_nlls) / len(mean_nlls),
        },
        "environment": {"gpu_layers": 20},
        "identity": {
            "schema": "rco.gguf_document_nll.v1",
            "corpus_manifest": {"canonical_manifest_sha256": "c"},
            "helper": {"sha256": "h"},
            "llama_cpp": {"revision": "r"},
            "parameters": {"ubatch": 256},
            "token_sequence_sha256": "t",
            "model": {"sha256": model_sha},
        },
    }


def _bf16(document_ids):
    return {
        "problem": {"token_sequence_sha256": "t"},
        "runs": {"bf16": {"aggregate": {
            "document_ids": document_ids, "mean_nll": 0.5}}},
    }


class HeldoutComparisonTest(unittest.TestCase):
    def setUp(self):
        self.candidate = _report("cand", [1.0, 1.1, 1.2])
        self.incumbent = _report("gsq", [1.1, 1.2, 1.25])
        self.bf16 = _bf16(self.candidate["aggregate"]["document_ids"])

    def test_paired_improvement(self):
        report = compare(self.candidate, copy.deepcopy(self.candidate),
                         self.incumbent, self.bf16, samples=200, seed=1)
        quality = report["quality"]
        self.assertTrue(report["improvement_over_incumbent_passed"])
        self.assertEqual(quality["documents_improved"], 3)
        self.assertAlmostEqual(
            quality["paired_candidate_minus_incumbent_mean_nll"], -0.25 / 3)
        self.assertFalse(quality["perplexity_ratio_gate_passed"])

    def test_rejects_different_scoring_parameters(self):
        self.incumbent["identity"]["parameters"] = {"ubatch": 512}
        with self.assertRaises(RuntimeError):
            compare(self.candidate, copy.deepcopy(self.candidate),
                    self.incumbent, self.bf16, samples=10, seed=1)

    def test_rejects_nonreproducible_repeat(self):
        repeat = copy.deepcopy(self.candidate)
        repeat["documents"][0]["nll_sum"] += 1e-3
        with self.assertRaises(RuntimeError):
            compare(self.candidate, repeat, self.incumbent, self.bf16,
                    samples=10, seed=1)


if __name__ == "__main__":
    unittest.main()
