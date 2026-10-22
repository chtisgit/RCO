import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import compare_qwen36_heldout_nll_outside as O  # noqa: E402


def report(description, threads=8):
    identity = {key: "x" for key in ("corpus_manifest", "helper", "llama_cpp", "parameters",
                                     "token_sequence_sha256", "schema")}
    return {"identity": identity, "environment": {"description": description, "threads": threads}}


class OutsideTest(unittest.TestCase):
    def test_only_the_description_may_differ(self):
        O.require_same_scoring(report("Q2_0"), report("Q2_K - Medium"))
        with self.assertRaises(RuntimeError):
            O.require_same_scoring(report("Q2_0"), report("Q2_K - Medium", threads=16))

    def test_installed_in_the_original(self):
        self.assertIs(O.base._require_same_scoring, O.require_same_scoring)


if __name__ == "__main__":
    unittest.main()
