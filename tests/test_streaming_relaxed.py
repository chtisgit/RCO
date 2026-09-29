import unittest

import torch
import torch.nn.functional as F

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.quant import WeightInterpolation
from search.relaxed import (
    DenseDeltaRowSource,
    StreamingRelaxedLinearStats,
    streaming_relaxed_linear,
)


class StreamingRelaxedLinearTest(unittest.TestCase):
    def test_matches_released_dense_loss_and_gradients(self):
        torch.manual_seed(29)
        dtype = torch.float64
        reference = torch.randn(7, 5, dtype=dtype)
        alternatives = [
            reference + 0.2 * torch.randn_like(reference),
            reference + 0.4 * torch.randn_like(reference),
        ]
        bias = torch.randn(7, dtype=dtype)
        target = torch.randn(2, 3, 7, dtype=dtype)
        initial_input = torch.randn(2, 3, 5, dtype=dtype)
        initial_logits = torch.tensor([0.4, -0.7, 0.2], dtype=dtype)

        dense_input = initial_input.clone().requires_grad_(True)
        dense_logits = initial_logits.clone().requires_grad_(True)
        interpolation = WeightInterpolation(
            [candidate - reference for candidate in alternatives],
            dense_logits.unsqueeze(0),
            0,
        )
        dense_output = F.linear(
            dense_input, interpolation(reference), bias)
        dense_loss = F.mse_loss(dense_output, target)
        dense_loss.backward()

        streamed_input = initial_input.clone().requires_grad_(True)
        streamed_logits = initial_logits.clone().requires_grad_(True)
        source = DenseDeltaRowSource(
            reference, alternatives, rows_per_chunk=2)
        stats = StreamingRelaxedLinearStats()
        streamed_output = streaming_relaxed_linear(
            streamed_input,
            streamed_logits,
            source,
            bias=bias,
            stats=stats,
        )
        streamed_loss = F.mse_loss(streamed_output, target)
        streamed_loss.backward()

        self.assertTrue(torch.allclose(
            streamed_output, dense_output, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(
            streamed_loss, dense_loss, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(
            streamed_input.grad, dense_input.grad, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(
            streamed_logits.grad, dense_logits.grad,
            atol=1e-12, rtol=1e-12))
        self.assertEqual(stats.forward_reference_passes, 1)
        self.assertEqual(stats.backward_reference_passes, 1)
        self.assertEqual(stats.forward_alternative_passes, 2)
        self.assertEqual(stats.backward_alternative_passes, 2)
        self.assertEqual(
            stats.max_materialized_chunk_bytes,
            2 * reference.shape[1] * reference.element_size(),
        )
        self.assertEqual(
            stats.forward_reference_bytes,
            reference.numel() * reference.element_size(),
        )
        self.assertEqual(
            stats.backward_alternative_bytes,
            len(alternatives) * reference.numel() * reference.element_size(),
        )

    def test_rejects_incomplete_row_source(self):
        class IncompleteSource:
            in_features = 3
            out_features = 4
            alternative_count = 1

            def iter_reference_rows(self):
                yield 0, torch.ones(3, 3)

            def iter_delta_rows(self, alternative_index):
                yield 0, torch.ones(4, 3)

        with self.assertRaisesRegex(ValueError, "ended at row 3"):
            streaming_relaxed_linear(
                torch.ones(2, 3),
                torch.zeros(2, requires_grad=True),
                IncompleteSource(),
            )

    def test_bias_gradient_matches_dense_linear(self):
        torch.manual_seed(31)
        reference = torch.randn(4, 3, dtype=torch.float64)
        alternative = reference + torch.randn_like(reference)
        source = DenseDeltaRowSource(
            reference, [alternative], rows_per_chunk=1)
        values = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        logits = torch.randn(2, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(4, dtype=torch.float64, requires_grad=True)

        output = streaming_relaxed_linear(
            values, logits, source, bias=bias)
        output.sum().backward()

        self.assertTrue(torch.equal(bias.grad, torch.full_like(bias, 2.0)))


if __name__ == "__main__":
    unittest.main()
