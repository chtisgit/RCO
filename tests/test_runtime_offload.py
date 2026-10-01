import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runtime_offload import parse_partial_cuda_offload


VALID_LOG = """
llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 3060) - 11929 MiB free
load_tensors: offloading output layer to GPU
load_tensors: offloading 3 repeating layers to GPU
load_tensors: offloaded 4/41 layers to GPU
load_tensors: CPU_Mapped model buffer size = 12398.48 MiB
load_tensors: CUDA0 model buffer size = 1101.95 MiB
sched_reserve: CUDA0 compute buffer size = 148.25 MiB
slot print_timing: prompt eval time = 4357.49 ms / 20 tokens
slot print_timing: eval time = 1137.09 ms / 4 tokens
"""


class PartialCudaOffloadParserTest(unittest.TestCase):
    def test_accepts_complete_partial_offload_evidence(self):
        result = parse_partial_cuda_offload(
            VALID_LOG, requested_layers=4, expected_total_layers=41)
        self.assertTrue(result["partial_offload"])
        self.assertEqual(result["offloaded_layers"], 4)
        self.assertEqual(result["total_offloadable_layers"], 41)
        self.assertEqual(result["cuda_model_buffer_mib"], 1101.95)
        self.assertEqual(result["cuda_compute_buffer_mib"], 148.25)
        self.assertEqual(result["generated_tokens"], 4)

    def test_rejects_cuda_failure_even_with_success_like_lines(self):
        with self.assertRaisesRegex(ValueError, "CUDA failure"):
            parse_partial_cuda_offload(
                "ggml_cuda_init: failed to initialize CUDA\n" + VALID_LOG,
                requested_layers=4,
                expected_total_layers=41,
            )

    def test_rejects_ignored_layer_request(self):
        with self.assertRaisesRegex(ValueError, "CUDA failure"):
            parse_partial_cuda_offload(
                "warning: ignored --gpu-layers\n" + VALID_LOG,
                requested_layers=4,
                expected_total_layers=41,
            )

    def test_rejects_wrong_offloaded_layer_count(self):
        with self.assertRaisesRegex(ValueError, "requested 4"):
            parse_partial_cuda_offload(
                VALID_LOG.replace("offloaded 4/41", "offloaded 3/41"),
                requested_layers=4,
                expected_total_layers=41,
            )

    def test_rejects_non_partial_offload(self):
        with self.assertRaisesRegex(ValueError, "must be partial"):
            parse_partial_cuda_offload(
                VALID_LOG.replace("offloaded 4/41", "offloaded 41/41"),
                requested_layers=41,
                expected_total_layers=41,
            )

    def test_rejects_missing_cuda_compute_buffer(self):
        with self.assertRaisesRegex(ValueError, "compute buffer"):
            parse_partial_cuda_offload(
                VALID_LOG.replace(
                    "sched_reserve: CUDA0 compute buffer size = 148.25 MiB\n", ""),
                requested_layers=4,
                expected_total_layers=41,
            )


if __name__ == "__main__":
    unittest.main()
