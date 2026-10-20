import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bench_qwen36_speed_factors as F  # noqa: E402


class SpeedFactorsTest(unittest.TestCase):
    def test_row_key(self):
        self.assertEqual(F.row_key({"n_threads": 16, "type_k": "q8_0", "n_prompt": 0,
                                    "n_gen": 128, "n_depth": 8192}), "t16/q8_0/tg128@d8192")
        self.assertEqual(F.row_key({"n_threads": 8, "type_k": "f16", "n_prompt": 512,
                                    "n_gen": 0, "n_depth": 0}), "t8/f16/pp512@d0")

    def test_bench_binary_falls_back_to_llama_bench_subcommand(self):
        self.assertEqual(F.bench_binary(Path("/nonexistent")), ["/nonexistent/llama", "bench"])


if __name__ == "__main__":
    unittest.main()
