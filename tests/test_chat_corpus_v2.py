import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent / "src"))

import audit_qwen36_chat_kl as chat_kl  # noqa: E402
import build_qwen36_chat_corpus_v2 as corpus_v2  # noqa: E402

IM_END, END_OF_TEXT = 900_001, 900_000


class _Tokenizer:
    """One token per character, plus the two end-of-generation ids."""

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": IM_END, "<|endoftext|>": END_OF_TEXT}[token]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _prompt(index, split, stratum="chat"):
    return {"id": f"chat-v2-{split}-{stratum}-{index:02d}", "stratum": stratum,
            "language": "en", "split": split, "thinking": index % 2 == 0,
            "source": {"key": f"src-{split}-{index}"}, "seed": 20261010 + index,
            "messages": [{"role": "user", "content": "hi"}], "tools": None}


def _generation(prompt, prompt_text, reply, stopped=True):
    generated = [ord(c) for c in reply] + ([IM_END] if stopped else [])
    return {"id": prompt["id"], "prompt_text": prompt_text,
            "prompt_tokens": [ord(c) for c in prompt_text], "generated_tokens": generated,
            "content": reply, "stop_type": "eos" if stopped else "limit",
            "tokens_predicted": len(generated)}


class FinalizeTest(unittest.TestCase):
    def _finalize(self, prompts, generations):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        model_dir, work = root / "model", root / "work"
        model_dir.mkdir()
        work.mkdir()
        (model_dir / "chat_template.jinja").write_text("template")
        (model_dir / "tokenizer.json").write_text("{}")
        (root / "documents.jsonl").write_text("{}\n")
        (work / "generation_identity.json").write_text(json.dumps({"model": "q6k"}))
        for name, rows in (("prompts.jsonl", prompts), ("generations.jsonl", generations)):
            (work / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        args = SimpleNamespace(model_dir=model_dir, work=work, output=root / "manifest.json",
                               documents=root / "documents.jsonl")
        with mock.patch.object(corpus_v2, "_tokenizer", return_value=_Tokenizer()), \
                mock.patch("builtins.print"):
            corpus_v2.run_finalize(args)
        return root, json.loads(args.output.read_text())

    def test_tokens_scores_and_summary(self):
        prompts = [_prompt(0, "calibration"), _prompt(1, "heldout")]
        generations = [_generation(prompts[0], "<p0>", "Hello."),
                       _generation(prompts[1], "<prompt 1>", "Long", stopped=False)]
        root, manifest = self._finalize(prompts, generations)
        tokens = {row["id"]: row for row in json.loads((root / "work/tokens.json").read_text())}

        first = tokens[prompts[0]["id"]]
        self.assertEqual(first["tokens"], [ord(c) for c in "<p0>Hello."] + [IM_END])
        # The generated tokens and the closing <|im_end|> are scored, the prompt is not.
        self.assertEqual(first["scored"], [0] * 4 + [1] * 7)
        second = tokens[prompts[1]["id"]]
        self.assertEqual(second["scored"], [0] * 10 + [1] * 4)

        summary = manifest["summary"]
        self.assertEqual(summary["checks"], {"prompt_retokenization_mismatches": 0,
                                             "content_decode_mismatches": 0})
        self.assertEqual((summary["calibration"]["scored_targets"],
                          summary["calibration"]["hit_token_limit"]), (7, 0))
        self.assertEqual((summary["heldout"]["scored_targets"],
                          summary["heldout"]["hit_token_limit"]), (4, 1))
        stopped = {c["id"]: c["stopped"]
                   for c in manifest["canonical_manifest"]["conversations"]}
        self.assertEqual(stopped, {prompts[0]["id"]: True, prompts[1]["id"]: False})

    def test_output_loads_in_the_chat_kl_tool(self):
        prompts = [_prompt(0, "calibration"), _prompt(1, "calibration"), _prompt(2, "heldout")]
        generations = [_generation(p, f"<{i}>", "ok" * (i + 1)) for i, p in enumerate(prompts)]
        root, _ = self._finalize(prompts, generations)
        args = SimpleNamespace(corpus=root / "manifest.json", split="calibration", limit=None)
        conversations, identity = chat_kl.load_corpus(args)
        self.assertEqual([c["id"] for c in conversations], [p["id"] for p in prompts[:2]])
        self.assertEqual(identity["conversation_count"], 2)
        self.assertEqual(conversations[1]["scored_target_count"], 5)

    def test_mismatches_are_counted_not_fatal(self):
        prompt = _prompt(0, "calibration")
        generation = _generation(prompt, "<p0>", "Hi")
        generation["prompt_tokens"] = [1, 2, 3, 4]
        generation["content"] = "Ho"
        _, manifest = self._finalize([prompt], [generation])
        self.assertEqual(manifest["check_details"], {
            "prompt_retokenization_mismatches": [prompt["id"]],
            "content_decode_mismatches": [prompt["id"]]})

    def test_rejects_inconsistent_generations(self):
        prompt = _prompt(0, "calibration")
        cases = {
            "stop type and final token disagree": {"stop_type": "limit"},
            "differs from tokens_predicted": {"tokens_predicted": 99},
        }
        for message, change in cases.items():
            generation = {**_generation(prompt, "<p0>", "Hi"), **change}
            with self.assertRaisesRegex(RuntimeError, message):
                self._finalize([prompt], [generation])
        with self.assertRaisesRegex(RuntimeError, "have no generation"):
            self._finalize([prompt, _prompt(1, "heldout")], [_generation(prompt, "<p0>", "Hi")])


if __name__ == "__main__":
    unittest.main()
