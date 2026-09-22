import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    from safetensors.torch import save_file
except ModuleNotFoundError:
    torch = None

if torch is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from quant.qparams import build_qparams_index, save_qparams
    from validation import (
        SafeTensorReference,
        load_assignment,
        tensor_error,
        validate_candidates,
    )


@unittest.skipIf(torch is None, "PyTorch/safetensors is not installed")
class CandidateValidationTest(unittest.TestCase):
    @staticmethod
    def _save_candidate(root, name, codes, bits=2):
        layer = Path(root) / name
        layer.mkdir(parents=True)
        handle = SimpleNamespace(
            quantizer_dict={
                bits: SimpleNamespace(sym=False, perchannel=True)},
            group_size=codes.shape[1],
            d_col=codes.shape[1],
            act_order=False,
            W_shape=codes.shape,
            W_dtype=torch.float32,
        )
        save_qparams(
            layer,
            bits=bits,
            qweight=codes,
            scales=torch.ones(codes.shape[0], 1),
            zeros=torch.zeros(codes.shape[0], 1),
            perm=None,
            handle=handle,
        )

    def test_tensor_error_reports_requested_metrics(self):
        reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        candidate = torch.tensor([[2.0, 0.0], [3.0, 5.0]])
        metrics, sums = tensor_error(
            reference, candidate, chunk_elements=2)
        self.assertEqual(sums.elements, 4)
        self.assertEqual(metrics["max_absolute_error"], 2.0)
        self.assertEqual(metrics["mean_absolute_error"], 1.0)
        self.assertEqual(metrics["mean_signed_error"], 0.0)
        self.assertAlmostEqual(
            metrics["root_mean_square_error"], (6.0 / 4.0) ** 0.5)
        self.assertAlmostEqual(
            metrics["relative_frobenius_error"], (6.0 / 30.0) ** 0.5)

    def test_reads_direct_and_fused_expert_sources_and_validates_sequentially(self):
        direct_name = "model.layers.0.proj"
        up_name = "model.layers.0.mlp.experts.1.up_proj"
        down_name = "model.layers.0.mlp.experts.0.down_proj"
        direct_codes = torch.tensor([[0, 1], [2, 3]], dtype=torch.uint8)
        up_codes = torch.tensor([[3, 2], [1, 0]], dtype=torch.uint8)
        down_codes = torch.tensor([[1, 1], [2, 2]], dtype=torch.uint8)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "source"
            candidate_dir = root / "candidates"
            source_dir.mkdir()
            candidate_dir.mkdir()
            self._save_candidate(candidate_dir, direct_name, direct_codes)
            self._save_candidate(candidate_dir, up_name, up_codes)
            self._save_candidate(candidate_dir, down_name, down_codes)
            build_qparams_index(candidate_dir)

            gate_up = torch.zeros(2, 4, 2)
            gate_up[1, 2:] = up_codes.float()
            down = torch.zeros(2, 2, 2)
            down[0] = down_codes.float()
            direct_reference = direct_codes.float()
            direct_reference[0, 0] += 0.5
            save_file({
                f"{direct_name}.weight": direct_reference,
                "model.layers.0.mlp.experts.gate_up_proj": gate_up,
                "model.layers.0.mlp.experts.down_proj": down,
            }, source_dir / "model.safetensors")

            reader = SafeTensorReference(source_dir)
            up_reference, up_source = reader.get(up_name)
            self.assertTrue(torch.equal(up_reference, up_codes.float()))
            self.assertIn("gate_up_proj[1:up_proj]", up_source)

            details = io.StringIO()
            report = validate_candidates(
                source_dir,
                candidate_dir,
                {direct_name: 2, up_name: 2, down_name: 2},
                chunk_elements=2,
                worst_count=2,
                details_handle=details,
            )
            zero_report = validate_candidates(
                source_dir,
                candidate_dir,
                {direct_name: 0},
                chunk_elements=2,
                worst_count=1,
            )
            records = [json.loads(line) for line in details.getvalue().splitlines()]

        self.assertEqual(report["tensor_count"], 3)
        self.assertEqual(report["element_count"], 12)
        self.assertEqual(report["errors"]["max_absolute_error"], 0.5)
        self.assertAlmostEqual(
            report["errors"]["mean_absolute_error"], 0.5 / 12.0)
        self.assertAlmostEqual(
            report["errors"]["root_mean_square_error"], (0.25 / 12.0) ** 0.5)
        self.assertEqual(report["weighted_average_selected_bits"], 2.0)
        self.assertEqual(len(records), 3)
        self.assertEqual(len(report["worst_by_max_absolute_error"]), 2)
        self.assertGreater(
            report["io_and_memory"]["candidate_file_bytes_read"], 0)
        self.assertEqual(
            report["io_and_memory"]["max_active_source_plus_candidate_bytes"],
            32,
        )
        self.assertEqual(zero_report["weighted_average_selected_bits"], 0.0)
        self.assertEqual(
            zero_report["io_and_memory"]["candidate_file_bytes_read"], 0)
        self.assertEqual(zero_report["errors"]["max_absolute_error"], 3.0)

    def test_load_assignment_supports_text_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            text_path = Path(directory) / "assignment.txt"
            text_path.write_text("# header\nmodel.a: 2\nmodel.b: 4\n")
            json_path = Path(directory) / "assignment.json"
            json_path.write_text(json.dumps({
                "assignment": {"model.a": 3},
            }))
            self.assertEqual(
                load_assignment(text_path), {"model.a": 2, "model.b": 4})
            self.assertEqual(load_assignment(json_path), {"model.a": 3})


if __name__ == "__main__":
    unittest.main()
