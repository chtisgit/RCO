import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.release_gate import evaluate_release_gate


def _passing_evidence():
    digest = "a" * 64
    return {
        "candidate": {
            "model_revision": "c" * 40,
            "candidate_store_index_sha256": "d" * 64,
            "assignment_sha256": "e" * 64,
            "assignment_cost_bytes": 100,
            "target_cost_bytes": 100,
            "independent_hard_replay": True,
            "search_calibration_sha256": ["b" * 64],
        },
        "heldout_corpus": {
            "sha256": digest,
            "document_count": 100,
            "predicted_token_count": 100_000,
            "strata": ["general", "knowledge", "code", "multilingual"],
        },
        "perplexity": {
            "objective": "exact full-vocabulary causal cross-entropy",
            "candidate_mean_nll": 2.01,
            "bf16_mean_nll": 2.0,
            "repeat_candidate_mean_nll": 2.01,
            "paired_candidate_minus_incumbent_ci95_upper": -0.001,
            "nonfinite_token_count": 0,
            "token_sequence_sha256": digest,
            "repeat_token_sequence_sha256": digest,
        },
        "generation": {
            "prompt_count": 32,
            "prompt_manifest_sha256": digest,
            "assertions_passed": 32,
            "assertions_total": 32,
            "deterministic_repeat": True,
            "nonfinite_run_count": 0,
            "invalid_utf8_count": 0,
            "replacement_character_count": 0,
            "empty_output_count": 0,
        },
        "runtime": {
            "cuda_kernel_matrix_pass": True,
            "cuda_one_step_pass": True,
            "cuda_multi_step_pass": True,
            "max_cuda_reserved_bytes": (9 << 30),
            "partial_cuda_offload_pass": True,
            "unmodified_llama_cpp": True,
        },
    }


class ReleaseGateTest(unittest.TestCase):
    def test_complete_evidence_authorizes_construction(self):
        result = evaluate_release_gate(_passing_evidence())
        self.assertTrue(result["authorized_for_final_gguf_construction"])
        self.assertEqual(result["failed_checks"], [])

    def test_search_calibration_cannot_be_reused_as_heldout(self):
        evidence = _passing_evidence()
        evidence["candidate"]["search_calibration_sha256"] = [
            evidence["heldout_corpus"]["sha256"]]
        result = evaluate_release_gate(evidence)
        self.assertFalse(result["authorized_for_final_gguf_construction"])
        self.assertIn("heldout_is_disjoint", result["failed_checks"])

    def test_lower_relaxed_quality_cannot_bypass_perplexity_gate(self):
        evidence = _passing_evidence()
        evidence["perplexity"]["candidate_mean_nll"] = 2.5
        evidence["perplexity"]["repeat_candidate_mean_nll"] = 2.5
        result = evaluate_release_gate(evidence)
        self.assertFalse(result["authorized_for_final_gguf_construction"])
        self.assertIn("perplexity_ratio_to_bf16", result["failed_checks"])

    def test_missing_cuda_gate_is_fail_closed(self):
        evidence = _passing_evidence()
        evidence["runtime"]["partial_cuda_offload_pass"] = False
        result = evaluate_release_gate(evidence)
        self.assertFalse(result["authorized_for_final_gguf_construction"])
        self.assertIn("partial_cuda_offload", result["failed_checks"])

    def test_generation_failure_is_fail_closed(self):
        evidence = copy.deepcopy(_passing_evidence())
        evidence["generation"]["empty_output_count"] = 1
        result = evaluate_release_gate(evidence)
        self.assertFalse(result["authorized_for_final_gguf_construction"])
        self.assertIn("generation_is_well_formed", result["failed_checks"])

    def test_incomplete_evidence_produces_explicit_failure(self):
        evidence = _passing_evidence()
        evidence["heldout_corpus"] = {
            "sha256": None,
            "document_count": 0,
            "predicted_token_count": 0,
            "strata": [],
        }
        evidence["perplexity"] = {}
        result = evaluate_release_gate(evidence)
        self.assertFalse(result["authorized_for_final_gguf_construction"])
        self.assertIsNone(result["measurements"]["candidate_perplexity"])
        self.assertIn("heldout_manifest_pinned", result["failed_checks"])
        self.assertIn(
            "finite_exact_full_vocabulary_nll", result["failed_checks"])


if __name__ == "__main__":
    unittest.main()
