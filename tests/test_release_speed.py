import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bench_qwen36_release_speed as S  # noqa: E402


class ReleaseSpeedTest(unittest.TestCase):
    def test_names_and_summary(self):
        rows = [{"n_prompt": 4096, "n_gen": 0, "n_depth": 8192, "avg_ts": 500.123, "stddev_ts": 1.234},
                {"n_prompt": 0, "n_gen": 128, "n_depth": 0, "avg_ts": 70.987}]
        self.assertEqual(S.summarize(rows), {
            "pp4096@d8192": {"avg_ts": 500.12, "stddev_ts": 1.23},
            "tg128@d0": {"avg_ts": 70.99, "stddev_ts": 0.0}})


if __name__ == "__main__":
    unittest.main()
