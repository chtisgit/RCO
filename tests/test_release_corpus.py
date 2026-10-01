import unittest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from release_corpus import (
    canonical_json_bytes,
    contains_subsequence,
    normalize_text,
    prepare_document,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False, truncation=False,
               max_length=None):
        if not hasattr(self, "encode_calls"):
            self.encode_calls = []
        self.encode_calls.append((add_special_tokens, truncation, max_length))
        self.last_add_special_tokens = add_special_tokens
        values = [ord(value) for value in text]
        return values[:max_length] if truncation else values

    def decode(self, tokens, skip_special_tokens=False,
               clean_up_tokenization_spaces=False):
        self.last_skip_special_tokens = skip_special_tokens
        self.last_cleanup = clean_up_tokenization_spaces
        return "".join(chr(value) for value in tokens)


class ReleaseCorpusTest(unittest.TestCase):
    def test_normalizes_unicode_newlines_and_trailing_space(self):
        self.assertEqual(normalize_text("e\u0301  \r\nnext \r"), "é\nnext")

    def test_canonical_json_is_stable_utf8_with_newline(self):
        self.assertEqual(
            canonical_json_bytes({"z": "é", "a": 1}),
            b'{"a":1,"z":"\xc3\xa9"}\n',
        )

    def test_detects_integer_subsequences(self):
        self.assertTrue(contains_subsequence([1, 2, 3, 4], [2, 3]))
        self.assertFalse(contains_subsequence([1, 2, 3, 4], [2, 4]))

    def test_prepares_an_exact_round_trip_prefix(self):
        tokenizer = CharacterTokenizer()
        text, tokens = prepare_document(
            tokenizer, "abcdef", token_count=4, forbidden_sequences=[[9, 9]])
        self.assertEqual(text, "abcd")
        self.assertEqual(tokens, [97, 98, 99, 100])
        self.assertFalse(tokenizer.last_add_special_tokens)
        self.assertEqual(tokenizer.encode_calls, [
            (False, True, 4), (False, False, None),
        ])
        self.assertFalse(tokenizer.last_skip_special_tokens)
        self.assertFalse(tokenizer.last_cleanup)

    def test_rejects_short_or_calibration_contaminated_documents(self):
        tokenizer = CharacterTokenizer()
        with self.assertRaisesRegex(ValueError, "only 3 tokens"):
            prepare_document(tokenizer, "abc", token_count=4)
        with self.assertRaisesRegex(ValueError, "calibration"):
            prepare_document(
                tokenizer, "abcdef", token_count=6,
                forbidden_sequences=[[99, 100]],
            )


if __name__ == "__main__":
    unittest.main()
