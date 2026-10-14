import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent / "src"))

import probe_qwen36_template_tokens as probe  # noqa: E402


class CorpusInputsTest(unittest.TestCase):
    def test_scores_are_put_back_in_corpus_order(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "score_x").mkdir()
            # Batches hold conversations longest first, not in corpus order.
            np.savez(work / "score_x/batch000.npz", members=np.asarray([2, 0]),
                     **{"0_nll": np.asarray([0.1, 0.2, 0.3]), "1_nll": np.asarray([0.5])})
            np.savez(work / "score_x/batch001.npz", members=np.asarray([1]),
                     **{"0_nll": np.asarray([0.7, 0.8])})
            rows = probe._label_log_probs(work, "x", 3)
            self.assertEqual([r.tolist() for r in rows], [[-0.5], [-0.7, -0.8], [-0.1, -0.2, -0.3]])
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                probe._label_log_probs(work, "x", 4)

    def test_reference_log_prob_is_read_from_the_top20_or_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            values = np.asarray([[-0.1, -2.0], [-0.3, -1.5], [-0.2, -4.0]], dtype=np.float32)
            indices = np.asarray([[5, 7], [9, 5], [1, 2]], dtype=np.int32)
            np.savez(work / "reference_bf16_top20.npz", values=values, indices=indices,
                     counts=np.asarray([2, 1]))
            result = probe._reference_log_probs(work, [np.asarray([7, 5]), np.asarray([3])])
        np.testing.assert_allclose(result[0], [-2.0, -1.5])
        self.assertTrue(np.isnan(result[1][0]))


class PromptsTest(unittest.TestCase):
    def test_first_token_prompts_are_tool_prompts_with_thinking_off(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "prompts.jsonl"
            rows = [{"id": f"p{i}", "stratum": stratum, "thinking": thinking, "split": "heldout",
                     "messages": [{"role": "user", "content": "x"}], "tools": []}
                    for i, (stratum, thinking) in enumerate(
                        [("tool-call", False), ("tool-call", True), ("tool-answer", False),
                         ("chat", False)])]
            corpus.write_text("".join(json.dumps(r) + "\n" for r in rows))
            args = probe.SimpleArgs(prompts=TOOLS / "qwen36_chat_generation_prompts_v2.json",
                                    corpus_prompts=corpus)
            prompts = probe._tool_prompts(args)
        self.assertEqual([(p["id"], p["source"]) for p in prompts],
                         [("tool-weather", "gate"), ("tool-flights", "gate"),
                          ("tool-currency", "gate"), ("p0", "heldout")])


class ServerCommandTest(unittest.TestCase):
    def test_every_flag_keeps_its_value(self):
        args = probe.SimpleArgs(server="llama-server", model="m.gguf", gpu_layers=99,
                                n_cpu_moe=30, threads=8, context=8192, port=18081)
        command = probe._server_command(args)
        self.assertEqual(command[command.index("-ngl") + 1], "99")
        self.assertEqual(command[command.index("--n-cpu-moe") + 1], "30")
        args.n_cpu_moe = 0
        self.assertNotIn("--n-cpu-moe", probe._server_command(args))


if __name__ == "__main__":
    unittest.main()
