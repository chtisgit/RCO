import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import audit_qwen36_offload as O  # noqa: E402

LOG = """load_tensors:   CPU_Mapped model buffer size = 10958.98 MiB
load_tensors:        CUDA0 model buffer size =  2023.80 MiB
llama_context:  CUDA_Host  output buffer size =     0.95 MiB
llama_kv_cache:      CUDA0 KV buffer size =    15.00 MiB
llama_memory_recurrent:      CUDA0 RS buffer size =    62.81 MiB
sched_reserve:      CUDA0 compute buffer size =   264.52 MiB
~llama_context:      CUDA0 compute buffer size is 264.5167 MiB, matches expectation of 264.5167 MiB
"""


class PlacementTest(unittest.TestCase):
    def test_flags(self):
        cpu = {"gpu_layers": 0, "device_none": True}
        self.assertEqual(O.placement_args(cpu, "server"), ["-ngl", "0", "--device", "none"])
        self.assertEqual(O.placement_args(cpu, "helper"), ["--device-none"])
        self.assertEqual(O.placement_args(cpu, "bench"), ["-ngl", "0", "-dev", "none"])
        moe = {"gpu_layers": 99, "n_cpu_moe": 12, "device_none": False}
        self.assertEqual(O.placement_args(moe, "server"), ["-ngl", "99", "--n-cpu-moe", "12"])
        self.assertEqual(O.placement_args(moe, "helper"), ["--gpu-layers", "99", "--n-cpu-moe", "12"])
        self.assertEqual(O.placement_args(moe, "bench"), ["-ngl", "99", "-ncmoe", "12"])
        self.assertEqual(O.placement_args({"gpu_layers": 20, "device_none": False}, "server"),
                         ["-ngl", "20"])

    def test_placements_follow_p3(self):
        report = {"stages": {"p3": {"n": 8, "p4_fits": False}}}
        found = O.placements(report)
        self.assertEqual(found["P3"]["n_cpu_moe"], 8)
        self.assertNotIn("P4", found)
        self.assertEqual(sorted(O.placements({"stages": {}})), ["P0", "P1", "P2"])


class BufferTest(unittest.TestCase):
    def test_buffers(self):
        sizes = O._buffers(LOG)
        self.assertEqual(sizes["CUDA0:load_tensors"], 2023.80)
        self.assertEqual(sizes["CPU_Mapped:load_tensors"], 10958.98)
        self.assertEqual(O._cuda_mib(sizes), round(2023.80 + 15.00 + 62.81 + 264.52, 2))


class AgreementChoiceTest(unittest.TestCase):
    def test_three_per_stratum(self):
        from types import SimpleNamespace
        args = SimpleNamespace(corpus=(O.RCO / "reports/qwen36_chat_corpus_v2_manifest.json"),
                               reports=O.RCO / "reports")
        try:
            conversations, reference, _ = O._agreement_conversations(args)
        except FileNotFoundError:
            self.skipTest("chat corpus v2 not present")
        self.assertEqual(len(conversations), 15)
        self.assertEqual(len(reference), 15)
        self.assertEqual(conversations[0]["id"], "chat-v2-calibration-chat-00")


if __name__ == "__main__":
    unittest.main()
