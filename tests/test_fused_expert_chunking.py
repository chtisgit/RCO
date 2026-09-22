import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None

if torch is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from quant.quantizer import Quantizer


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class FusedExpertChunkingTest(unittest.TestCase):
    class Handle:
        def __init__(self, layer):
            self.layer = layer
            self.H = None
            self.num_samples = 0

        def update(self, values):
            width = values.shape[-1]
            update = values.float().reshape(-1, width).T @ values.float().reshape(-1, width)
            self.H = update if self.H is None else self.H + update
            self.num_samples += values.shape[0]

        def reset(self):
            self.H = None
            self.num_samples = 0

    class RecordingQuantizer(Quantizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.quantized = []

        def _create_handle(self, bitwidth_options, layer):
            return FusedExpertChunkingTest.Handle(layer)

        def _quantize_handle(self, name, handle, bitwidth_options,
                             calibration_bitwidth, *, replace_module):
            self.assert_handle(handle)
            self.quantized.append((name, tuple(handle.H.shape), handle.num_samples))
            result = handle.layer.weight.detach().clone() + 0.01
            handle.reset()
            return result

        @staticmethod
        def assert_handle(handle):
            if handle.H is None or handle.num_samples == 0:
                raise AssertionError("missing calibration statistics")

    class Experts(nn.Module):
        def __init__(self, n_experts=4, hidden=4, intermediate=3):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(
                n_experts, 2 * intermediate, hidden))
            self.down_proj = nn.Parameter(torch.randn(
                n_experts, hidden, intermediate))
            self.act_fn = torch.nn.functional.silu

        def forward(self, hidden, selected, weights):
            # The test exercises pre-hook collection; preserving hidden is
            # sufficient for the enclosing block.
            return hidden

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = FusedExpertChunkingTest.Experts()
            self.calls = 0

        def forward(self, hidden_states):
            self.calls += 1
            tokens = hidden_states.reshape(-1, hidden_states.shape[-1])
            selected = torch.arange(tokens.shape[0], device=tokens.device)
            selected = (selected % 4).unsqueeze(1)
            weights = torch.ones_like(selected, dtype=tokens.dtype)
            return self.experts(tokens, selected, weights).reshape_as(hidden_states)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([FusedExpertChunkingTest.Block()])

    def test_fused_experts_are_processed_in_bounded_chunks(self):
        model = self.Model()
        block = model.blocks[0]
        original_gate_up = block.experts.gate_up_proj.detach().clone()
        original_down = block.experts.down_proj.detach().clone()
        data = [(torch.randn(1, 8, 4),)]
        with tempfile.TemporaryDirectory() as directory:
            quantizer = self.RecordingQuantizer(
                model,
                data_loader=[],
                quantizable_modules=r"experts\.\d+\.(gate_proj|up_proj|down_proj)$",
                pre_block_modules=[],
                block_modules="blocks",
                save_dir=directory,
                expert_chunk_size=2,
            )
            quantizer._quantize_fused_experts(
                block, data, [{}], [2, 4], 4, torch.device("cpu"))

        self.assertEqual(block.calls, 2)
        self.assertEqual(len(quantizer.quantized), 12)
        self.assertTrue(torch.allclose(
            block.experts.gate_up_proj, original_gate_up + 0.01))
        self.assertTrue(torch.allclose(
            block.experts.down_proj, original_down + 0.01))
        self.assertTrue(all(name.startswith("blocks.0.experts.")
                            for name, _, _ in quantizer.quantized))


if __name__ == "__main__":
    unittest.main()
