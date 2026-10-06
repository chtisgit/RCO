import sys
from pathlib import Path
import unittest

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from build_qwen36_gsq_rco_candidate_gguf import (  # noqa: E402
    _rechunk,
    _selected_assignment,
    bf16_bytes_from_exact_float32,
    bf16_bytes_to_float32,
)


class BF16PackingTest(unittest.TestCase):
    def test_exact_bf16_round_trip_matches_torch(self):
        source = torch.randn(3, 8, generator=torch.Generator().manual_seed(1)).to(
            torch.bfloat16)
        payload = bf16_bytes_from_exact_float32(source.to(torch.float32).numpy())
        expected = source.view(torch.int16).numpy().astype("<i2").tobytes()
        self.assertEqual(payload, expected)
        restored = bf16_bytes_to_float32(payload, (3, 8))
        np.testing.assert_array_equal(restored, source.to(torch.float32).numpy())

    def test_rejects_values_that_need_rounding(self):
        with self.assertRaises(ValueError):
            bf16_bytes_from_exact_float32(np.array([1.0 + 2.0 ** -20], np.float32))

    def test_rechunk_preserves_bytes(self):
        chunks = [b"abc", b"defgh", b"", b"ij"]
        out = list(_rechunk(iter(chunks), 4))
        self.assertEqual(b"".join(out), b"abcdefghij")
        self.assertEqual([len(item) for item in out], [4, 4, 2])


class SelectedAssignmentTest(unittest.TestCase):
    def _reports(self, bits="0110", status="independent_replay_complete_pending_release_gates"):
        budget = {"tensor_names": ["a", "b", "c", "d"]}
        replay = {
            "status": status,
            "replay": {"passed": True},
            "selected": {"assignment_bits": bits},
            "budget": budget,
        }
        reference = {"selected": {"assignment_bits": "0110"}, "budget": budget}
        return replay, reference

    def test_selects_upgraded_names(self):
        self.assertEqual(_selected_assignment(*self._reports()), ["b", "c"])

    def test_rejects_unreplayed_or_divergent_selection(self):
        with self.assertRaises(ValueError):
            _selected_assignment(*self._reports(status="in_progress"))
        with self.assertRaises(ValueError):
            _selected_assignment(*self._reports(bits="0111"))


if __name__ == "__main__":
    unittest.main()
