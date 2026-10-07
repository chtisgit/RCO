import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_qwen36_calibration_corpus_v2 import (  # noqa: E402
    SourceCursor,
    round_robin,
    spread_starts,
    van_der_corput_order,
)
from build_qwen36_calibration_corpus import _ngram_hashes  # noqa: E402


class SpreadOrderTest(unittest.TestCase):
    def test_van_der_corput_is_a_spread_permutation(self):
        order = van_der_corput_order(8)
        self.assertEqual(sorted(order), list(range(8)))
        self.assertEqual(order[:4], [0, 4, 2, 6])

    def test_spread_starts_stay_in_range(self):
        starts = spread_starts(100, initial=10, length=20, stride=5)
        self.assertEqual(sorted(starts), list(range(10, 81, 5)))
        self.assertEqual(starts[:2], [10, 50])


class RoundRobinTest(unittest.TestCase):
    def _cursor(self, tokens, **kwargs):
        return SourceCursor(tokens, initial=1, length=4, stride=2, width=2, **kwargs)

    def test_segments_are_disjoint_non_overlapping_and_alternate(self):
        cursors = [self._cursor(list(range(0, 40))), self._cursor(list(range(100, 140)))]
        forbidden = _ngram_hashes([5, 6], 2)
        picked = list(round_robin(cursors, 4, forbidden))
        self.assertEqual([source for source, _, _ in picked], [0, 1, 0, 1])
        seen = set()
        for _, _, segment in picked:
            ngrams = _ngram_hashes(segment, 2)
            self.assertFalse(ngrams & seen)
            self.assertNotIn((5, 6), list(zip(segment, segment[1:])))
            seen |= ngrams

    def test_rejected_candidates_are_skipped_and_counted(self):
        cursor = self._cursor(list(range(20)), accept=lambda segment: segment[0] != 1)
        start, segment, _ = cursor.next_segment(set())
        self.assertNotEqual(segment[0], 1)
        self.assertEqual(cursor.rejected, 1)

    def test_exhaustion_raises(self):
        with self.assertRaises(RuntimeError):
            list(round_robin([self._cursor(list(range(8)))], 5, set()))


if __name__ == "__main__":
    unittest.main()
