"""Canonicalization helpers for release-quality held-out corpora."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any, Iterable, Sequence


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def contains_subsequence(values: Sequence[int], query: Sequence[int]) -> bool:
    if not query:
        return True
    stop = len(values) - len(query) + 1
    return any(list(values[start:start + len(query)]) == list(query)
               for start in range(max(0, stop)))


def prepare_document(
    tokenizer: Any,
    text: str,
    *,
    token_count: int,
    forbidden_sequences: Iterable[Sequence[int]] = (),
) -> tuple[str, list[int]]:
    normalized = normalize_text(text)
    tokens = list(tokenizer.encode(
        normalized, add_special_tokens=False, truncation=True,
        max_length=token_count,
    ))
    if len(tokens) < token_count:
        raise ValueError(
            f"document has only {len(tokens)} tokens; need {token_count}")
    tokens = [int(value) for value in tokens[:token_count]]
    prepared = tokenizer.decode(
        tokens, skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    round_trip = [int(value) for value in tokenizer.encode(
        prepared, add_special_tokens=False)]
    if round_trip != tokens:
        raise ValueError("tokenizer truncation does not round-trip exactly")
    for forbidden in forbidden_sequences:
        if contains_subsequence(tokens, forbidden):
            raise ValueError("document contains a search calibration sequence")
    return prepared, tokens
