import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_qwen36_calibration_corpus import (
    _choose_disjoint_segment,
    _ngram_hashes,
    _validate_canonical_manifest_sha256,
)
from release_corpus import canonical_json_bytes, sha256_bytes


class CalibrationCorpusTest(unittest.TestCase):
    def test_heldout_canonical_manifest_hash_is_recomputed(self):
        manifest = {"document_count": 1, "corpus": {"sha256": "abc"}}
        report = {
            "canonical_manifest": manifest,
            "canonical_manifest_sha256": sha256_bytes(
                canonical_json_bytes(manifest)),
        }
        self.assertEqual(
            _validate_canonical_manifest_sha256(report),
            report["canonical_manifest_sha256"])

    def test_heldout_canonical_manifest_hash_mismatch_fails(self):
        report = {
            "canonical_manifest": {"document_count": 1},
            "canonical_manifest_sha256": "0" * 64,
        }
        with self.assertRaisesRegex(RuntimeError, "canonical manifest hash"):
            _validate_canonical_manifest_sha256(report)

    def test_ngram_hashes_detect_overlap(self):
        left = _ngram_hashes([1, 2, 3, 4], 3)
        right = _ngram_hashes([0, 2, 3, 4, 5], 3)
        self.assertEqual(len(left & right), 1)

    def test_segment_selection_skips_forbidden_window(self):
        tokens = list(range(20))
        forbidden = _ngram_hashes(tokens[4:10], 3)
        start, segment = _choose_disjoint_segment(
            tokens, requested_start=4, token_count=6,
            forbidden_ngrams=forbidden, ngram_width=3, stride=2)
        self.assertEqual(start, 8)
        self.assertEqual(segment, list(range(8, 14)))

    def test_segment_selection_fails_without_room(self):
        with self.assertRaises(ValueError):
            _choose_disjoint_segment(
                [1, 2, 3], requested_start=2, token_count=2,
                forbidden_ngrams=set(), ngram_width=2, stride=1)


if __name__ == "__main__":
    unittest.main()
