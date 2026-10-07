import sys
from pathlib import Path
import unittest

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_qwen36_prune24_prelim import (  # noqa: E402
    exact_pruned_routing,
    frequency_prune_mask,
    top_k_kl,
)


class ExactPrunedRoutingTest(unittest.TestCase):
    def test_matches_router_over_physically_kept_experts(self):
        generator = torch.Generator().manual_seed(0)
        logits = torch.randn(64, 16, generator=generator)
        mask = torch.zeros(16, dtype=torch.bool)
        mask[[1, 4, 9, 15]] = True
        _, weights, indices = exact_pruned_routing(logits, mask, 4)

        kept = torch.nonzero(~mask).squeeze(-1)
        removed_probs = torch.softmax(logits[:, kept], dim=-1)
        removed_weights, removed_local = torch.topk(removed_probs, 4, dim=-1)
        removed_weights = removed_weights / removed_weights.sum(-1, keepdim=True)
        torch.testing.assert_close(indices, kept[removed_local])
        torch.testing.assert_close(weights, removed_weights)
        self.assertFalse(bool(mask[indices].any()))

    def test_no_mask_reproduces_hf_router(self):
        logits = torch.randn(8, 16, generator=torch.Generator().manual_seed(1))
        _, weights, indices = exact_pruned_routing(
            logits, torch.zeros(16, dtype=torch.bool), 4)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        expected, expected_indices = torch.topk(probs, 4, dim=-1)
        torch.testing.assert_close(indices, expected_indices)
        torch.testing.assert_close(weights, expected / expected.sum(-1, keepdim=True))


class FrequencyMaskTest(unittest.TestCase):
    def test_least_selected_with_ties_by_mass_then_index(self):
        counts = np.array([[5, 1, 1, 9, 0], [3, 3, 3, 3, 3]])
        mass = np.array([[0.5, 0.2, 0.1, 0.9, 0.0], [0.3, 0.1, 0.1, 0.2, 0.4]])
        mask = frequency_prune_mask(counts, mass, 2)
        np.testing.assert_array_equal(
            mask, [[False, False, True, False, True],
                   [False, True, True, False, False]])
        self.assertTrue(np.all(mask.sum(axis=1) == 2))

    def test_rejects_bad_count(self):
        with self.assertRaises(ValueError):
            frequency_prune_mask(np.ones((1, 4)), np.ones((1, 4)), 4)


class TopKKLTest(unittest.TestCase):
    def test_zero_for_identical_and_positive_otherwise(self):
        log_probs = torch.log_softmax(torch.randn(3, 50), dim=-1)
        values, indices = log_probs.topk(5, dim=-1)
        torch.testing.assert_close(
            top_k_kl(values, indices, log_probs), torch.zeros(3))
        shifted = torch.log_softmax(torch.randn(3, 50), dim=-1)
        manual = (values.exp() * (values - shifted.gather(-1, indices))).sum(-1)
        torch.testing.assert_close(top_k_kl(values, indices, shifted), manual)


if __name__ == "__main__":
    unittest.main()
