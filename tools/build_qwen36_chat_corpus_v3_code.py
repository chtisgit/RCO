#!/usr/bin/env python3
"""RCO_PLAN_V3.md: the ``code`` stratum of chat corpus v3 (v2 plus ``code``).

Per split (``calibration``, ``heldout``) 48 conversations, 24 single-turn
and 24 multi-turn, all ending in a user turn whose reply the reference model
writes (as in v2):

- ``arena-single`` (24): the first user turn of a one-turn coding session
  from ``lmarena-ai/arena-human-preference-140k`` (one shard), 16 English
  and 8 in other languages;
- ``arena-multi`` (8): a coding session of two or more turns, through its
  last user turn (the ``full_conversation`` of the session's last vote row
  in the shard).  Each earlier assistant turn is the side that won that
  turn's vote; side A on a tie or when that turn's vote row is not in the
  shard; sessions with a "both bad" vote are dropped;
- ``oasst2-multi`` (16): English ``OpenAssistant/oasst2`` trees with code
  on the top-ranked path and at least one earlier assistant turn, through
  the last user turn.  No tree of chat corpus v1 or v2 is reused.

A coding prompt (the first user turn) has a code block, at least two lines
of code syntax, or a programming language or tool named together with a
coding task word.  Contexts: no images, no chat-template markup, at most
``MAX_CONTEXT_TOKENS`` tokens with the template applied.  Thinking
alternates within each bucket as in v2; the splits alternate.

Code content (the user's rules after the first review, with guidelines taken
from the user's drops).  Code means source code in a programming, scripting
or query language, HTML and CSS included; prose, documents in a markup
language (Markdown, reST, LaTeX), data and configuration files (JSON, YAML,
XML, ...) and terminal output or logs are not code.  A conversation is kept
when one of these holds (``code_source``):

- ``user``: the first user turn has at least ``MIN_CODE_LINES`` lines of
  code;
- ``earlier-assistant`` (multi-turn): an earlier assistant turn has a code
  block of at least ``MIN_CODE_LINES`` lines;
- ``reply`` (single-turn Arena): both Arena replies to the prompt have a code
  block of at least ``MIN_CODE_LINES`` lines (the prompt asks for code, or
  code belongs in the answer).

and in addition: the first user turn has a request in words (at least
``MIN_REQUEST_WORDS`` words besides code and terminal output, so no bare
pastes: the request is the prose before the first and after the last line of
code); the last user turn of a multi-turn conversation is not an
acknowledgement ("thanks", "cool"); the first user turn is not a "hello
world" program; a first user turn at least ``DUPLICATE_RATIO`` similar to
that of an earlier pick is a duplicate; ``USER_DROPS`` (the user's review)
are excluded.

Subcommands: ``select`` writes ``prompts.jsonl`` and ``review.json`` (the
selected conversations with the candidate counts, for the user's review).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_q3k_viability import _sha256_file  # noqa: E402
from build_qwen36_chat_corpus import SOURCES, _clean, _source_path, oasst2_candidates  # noqa: E402
from build_qwen36_chat_corpus_v2 import SPLITS, _context_tokens, _tokenizer, _v1_keys  # noqa: E402

SEED = 20261024
MAX_CONTEXT_TOKENS = 4096
MIN_CODE_LINES = 5
MIN_REQUEST_WORDS = 4
DUPLICATE_RATIO = 0.8
FENCE_CODE_SHARE = 0.4
QUOTAS = {("arena-single", "en"): 16, ("arena-single", "other"): 8,
          ("arena-multi", "any"): 8, ("oasst2-multi", "en"): 16}
ARENA = {"dataset": "lmarena-ai/arena-human-preference-140k",
         "file": "data/train-00000-of-00007.parquet", "license": "CC-BY-4.0 (prompts)"}

FENCE = re.compile(r"```")
SYNTAX_LINE = re.compile(
    r"^\s*(def |class |function |import |from \S+ import|#include|public |private |"
    r"const |let |var |return\b|if\s*\(|for\s*\(|while\s*\(|SELECT\b|<\w+[ >]|\}|.*;\s*$|.*\{\s*$)",
    re.M)
LANGUAGE = re.compile(
    r"\b(python|javascript|typescript|java|kotlin|swift|rust|golang|c\+\+|c#|php|ruby|"
    r"sql|html|css|bash|shell script|powershell|matlab|react|vue|angular|django|flask|"
    r"node\.?js|express|spring|unity|godot|pandas|numpy|pytorch|tensorflow|docker|"
    r"kubernetes|git|regex|excel formula|vba|lua|haskell|scala|dart|flutter|wgsl|glsl)\b", re.I)
TASK = re.compile(
    r"\b(code|script|function|program|implement|write|fix|debug|bug|error|exception|"
    r"refactor|class|api|compile|optimi[sz]e|unit test|algorithm)\b", re.I)


# The user's reviews of the selections (2026-10-25): source keys.
USER_DROPS = {
    "32f3c041-bc2c-42b8-87a4-37db3a23a12e", "8f661c6f-0dfa-4aad-ad07-fb96fa35c24b",
    "e0564934-9e0a-48b2-b3c3-8e493086e8e0", "fed54002-577e-48f5-a91b-8abde853c90e",
    "3908dd09-1704-4b53-ab94-24ed34e23436", "493913e5-29a0-404e-a082-603105318a83",
    "d201a061-ac0e-4902-8aba-c5f3fc6fcb77", "d7feab4f-18af-4487-820f-cfd9763c70fd",
    "f988d75d-fec4-444e-9931-539a8f5822bf", "1655d235-304e-475c-a29c-0d5af8b46f2e",
    "c4d5c60b-722f-4fe5-982e-a4f979f8027b", "d401c009-4f45-4ccd-a4dd-cad54419ca06",
    "0fb52b31-304e-4000-b145-f70c2945adeb", "2abaf89c-4d78-4aa1-9c05-9c297086da6f",
    "31d6964c-dc6f-4807-875b-ed59244fa5cb", "3d896638-c1de-40fe-a452-ae8b79b2c44f",
    "4c03c680-f14e-477a-8024-b0e6c334ccf4", "c46c91ff-f3f3-46b6-9f6d-252407cd4d83",
    "eee7eb54-f94b-466b-8269-fba4fc70b825", "098b90db-f9a0-441e-9df8-9067211c838c",
    "132f569d-d8eb-40f8-91c4-48c14551a68a", "4a97475e-dfc7-4bb7-97a3-c2a297e3532e",
    "7c6f3c4d-e6b5-4e7f-8a83-f71d808ad15f", "cd0d7bac-1e6c-459b-b60d-0b210d6b820b",
    # second review
    "6547b599-867a-4432-a31d-6b8a721cd0d1", "9e3b03cc-b216-4256-90c0-a491e2cb5f12",
    "b0cc78d9-7b0e-4db7-bd0c-03316a84d7e7",
    # third review
    "b32be44b-63a0-403d-b873-d8c298f9125d", "e25856e4-3af8-49dd-acf0-562ebafda63c",
    # fourth review
    "5642151a-5310-4b96-ac5b-1a9daea482f0",
}

FENCE_BLOCK = re.compile(
    r"^[ \t]*(`{3,}|~{3,})[ \t]*([^\n`]*)\n(.*?)(?:^[ \t]*\1[ \t]*$|\Z)", re.S | re.M)
NOT_CODE_LANGUAGES = {
    "text", "txt", "plain", "plaintext", "markdown", "md", "rst", "rest", "latex", "tex", "json",
    "jsonc", "json5", "yaml", "yml", "toml", "ini", "xml", "csv", "tsv", "log", "output",
    "console", "shell-session", "none", "diff"}
TERMINAL_LINE = re.compile(
    r"^\s*([A-Z]:\\[^>]*>|PS [A-Z]:\\|\S+@\S+:[^$#]*[$#]\s|at \S+.*[(:]\d+|Traceback |"
    r"File \".*\", line \d+|(npm (ERR!|WARN)|error|warning|info|debug|fatal|ERROR|WARNING|INFO|"
    r"DEBUG)\b[:\s]|\d{4}-\d\d-\d\d[ T]\d\d:\d\d|\[\d+(\.\d+)?\]|\$ \S)")
CODE_LINE = re.compile(
    r"^\s*(def |class |function\b|async |await |import |export |from \S+ import|#include|#define|"
    r"public |private |protected |static |const |let |var |return\b|if\s*\(|else\b|elif |"
    r"for\s*\(|for \w+ in |while\s*\(|try\s*[:{]|except\b|catch\s*\(|fn |func |"
    r"package [\w.]+;?\s*$|using [\w.]+;|namespace |struct \w+|enum \w+|interface \w+|"
    r"type \w+\s*=|@\w+|</?[A-Za-z][\w.-]*[\s>/]|[}\])]+[;,)]*\s*$|"
    r"(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|FROM|WHERE|JOIN|GROUP BY|ORDER BY)\b|"
    r"local |end\b|.*\b(do|then)\s*$|--[^-\s]|-- \w|//|/\*|\*/|[\w.\[\]'\"]+\s*(\+|-|\*|/|\|\||&&|\?\?)?=(?!=)\s*\S|"
    r"[\w.$]+\(.*\)\s*[;{]?\s*$|.*[;{]\s*$)")
NEUTRAL_LINE = re.compile(r"^\s*([\[\]{}(),;]+|[-=*_#~]{3,}|\"[^\"]*\"\s*:.*)\s*$")
BULLET = re.compile(r"^\s*([●•·▪◦–—]|[-*+]\s|\d+[.)]\s)")
LEADING_WORDS = re.compile(r"^\s*([^\W\d_]+)(\s+[^\W\d_]+){3,}\s")
CODE_WORDS = {"public", "private", "protected", "static", "final", "abstract", "const", "let",
              "var", "def", "class", "return", "import", "from", "export", "async", "await",
              "int", "void", "char", "float", "double", "bool", "unsigned", "long", "select",
              "insert", "update", "delete", "create", "alter", "local", "function", "fn", "pub",
              "impl", "use", "new", "throw", "else", "case", "inline", "virtual", "override",
              "internal", "sealed", "readonly", "volatile", "extern", "typedef", "struct"}
COMMENT = re.compile(r"^\s*(//|#|--|/\*|\*)")
WORD = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]|[^\W\d_]{2,}")
ACKNOWLEDGEMENT = re.compile(
    r"^\W*(thanks|thank you|thx|ty|wow|cool|great|nice|perfect|ok|okay|awesome)\b", re.I)
HELLO_WORLD = re.compile(r"hello,?\s+world", re.I)


def _line_kind(line: str) -> str:
    """``code``, ``prose`` or ``other`` (blank, terminal output, data, a bare
    bracket)."""
    if not line.strip() or TERMINAL_LINE.match(line) or NEUTRAL_LINE.match(line):
        return "other"
    if BULLET.match(line) or not CODE_LINE.match(line):
        return "prose"
    lead = LEADING_WORDS.match(line)
    if lead and lead.group(1).lower() not in CODE_WORDS:
        return "prose"
    words = len(WORD.findall(line))
    symbols = sum(not (c.isalnum() or c.isspace()) for c in line)
    if words > 8 and symbols < 0.04 * len(line) and not COMMENT.match(line):
        return "prose"
    return "code"


def code_shape(text: str) -> dict:
    """Lines of code, the longest code block, the words of prose and the
    words of the request (prose before the first and after the last line of
    code) in a message.  A fenced block counts whole when its language is a
    programming language or, without a language, when at least
    ``FENCE_CODE_SHARE`` of its lines look like code; other lines count by
    line."""
    kinds, last = [], 0
    for match in FENCE_BLOCK.finditer(text):
        kinds += [(_line_kind(line), line) for line in text[last:match.start()].splitlines()]
        last = match.end()
        language = (match.group(2).split() or [""])[0].lower()
        body = [line for line in match.group(3).splitlines() if line.strip()]
        terminal = sum(bool(TERMINAL_LINE.match(line)) for line in body)
        by_line = [(_line_kind(line), line) for line in body]
        if language in NOT_CODE_LANGUAGES or language.startswith("`"):
            kinds += [("other", line) for line in body]
        elif language:
            kind = "other" if terminal > len(body) / 2 else "code"
            kinds += [(kind, line) for line in body]
        elif sum(k == "code" for k, _ in by_line) >= FENCE_CODE_SHARE * len(body):
            kinds += [("other" if TERMINAL_LINE.match(line) else "code", line) for line in body]
        else:
            kinds += [(k if k == "prose" else "other", line) for k, line in by_line]
        kinds.append(("break", ""))
    kinds += [(_line_kind(line), line) for line in text[last:].splitlines()]
    lines, longest, run = 0, 0, 0
    for kind, line in kinds:
        if kind == "code":
            lines += 1
            run += 1
            longest = max(longest, run)
        elif kind == "break" or kind == "prose" and len(WORD.findall(line)) > 6:
            run = 0
    code_at = [n for n, (kind, _) in enumerate(kinds) if kind == "code"]
    outer = (kinds if not code_at else kinds[:code_at[0]] + kinds[code_at[-1] + 1:])
    words = lambda part: sum(len(WORD.findall(line)) for kind, line in part if kind == "prose")
    return {"code_lines": lines, "longest_block": longest, "prose_words": words(kinds),
            "request_words": words(outer)}


def code_source(messages: list[dict], replies: list[str]) -> str | None:
    """Which part of the conversation holds the code, or None if it fails
    the code content rules (module docstring)."""
    first = code_shape(messages[0]["content"])
    if first["request_words"] < MIN_REQUEST_WORDS or HELLO_WORLD.search(messages[0]["content"]):
        return None
    final = messages[-1]["content"]
    if len(messages) > 1 and (ACKNOWLEDGEMENT.match(final) and "?" not in final
                              or code_shape(final)["prose_words"] < MIN_REQUEST_WORDS - 1):
        return None
    if first["code_lines"] >= MIN_CODE_LINES:
        return "user"
    if any(code_shape(m["content"])["longest_block"] >= MIN_CODE_LINES
           for m in messages if m["role"] == "assistant"):
        return "earlier-assistant"
    if len(messages) == 1 and replies and all(
            code_shape(reply)["longest_block"] >= MIN_CODE_LINES for reply in replies):
        return "reply"
    return None


def _duplicate(text: str, taken: list[str]) -> bool:
    from difflib import SequenceMatcher

    return any(SequenceMatcher(None, text[:2000], other[:2000]).ratio() >= DUPLICATE_RATIO
               for other in taken)


def coding_prompt(text: str) -> bool:
    return bool(FENCE.search(text) or len(SYNTAX_LINE.findall(text)) >= 2
                or (LANGUAGE.search(text) and TASK.search(text)))


def _text(message: dict | None) -> str:
    content = (message or {}).get("content") or []
    return "\n".join(part.get("text") or "" for part in content if part.get("type") == "text")


def _has_image(message: dict | None) -> bool:
    return any(part.get("image") for part in (message or {}).get("content") or [])


def arena_sessions(path: Path):
    """Yield (session id, language, messages, turns, models, replies) per
    coding session whose last vote row in the shard has ``is_code``; replies
    are both sides' answers to the last user turn."""
    import pyarrow.parquet as pq

    sessions: dict[str, list[dict]] = defaultdict(list)
    for row in pq.read_table(path).to_pylist():
        sessions[row["evaluation_session_id"]].append(row)
    for session, rows in sorted(sessions.items()):
        votes = {r["evaluation_order"]: r for r in rows}
        last = votes[max(votes)]
        turns = last["full_conversation"]
        if not last["is_code"] or any(_has_image(t["user"]) for t in turns):
            continue
        first = _text(turns[0]["user"])
        if not (80 <= len(first) <= 12_000 and coding_prompt(first)):
            continue
        messages, models = [], []
        for number, turn in enumerate(turns, start=1):
            messages.append({"role": "user", "content": _text(turn["user"])})
            if number == len(turns):
                break
            vote = votes.get(number)
            if vote is not None and vote["winner"] == "both_bad":
                messages = None
                break
            side = "b" if vote is not None and vote["winner"] == "model_b" else "a"
            messages.append({"role": "assistant", "content": _text(turn[f"model_side_{side}"])})
            models.append(vote[f"model_{side}"] if vote is not None else f"side {side}, no vote row")
        if messages and _clean(*(m["content"] for m in messages)):
            replies = [_text(turns[-1][f"model_side_{side}"]) for side in "ab"]
            yield session, last["language"], messages, len(turns), models, replies


def oasst2_code(path: Path, used: set[str]):
    for key, language, messages, _ in oasst2_candidates(path):
        if key in used or language != "en":
            continue
        while messages and messages[-1]["role"] != "user":
            messages.pop()
        code = any(coding_prompt(m["content"]) for m in messages) and coding_prompt(
            messages[0]["content"])
        if len(messages) >= 3 and code:
            yield key, messages


def run_select(args) -> None:
    tokenizer = _tokenizer(args.model_dir)
    used = _v1_keys(args.v1_corpus)["chat"]
    with args.v2_prompts.open(encoding="utf-8") as handle:
        used |= {str(json.loads(line)["source"]["key"]) for line in handle
                 if json.loads(line)["stratum"] == "chat"}
    arena_sha256 = _sha256_file(args.arena)
    sessions = list(arena_sessions(args.arena))
    oasst2 = list(oasst2_code(_source_path(args.sources, "chat"), used))
    pools = {
        "arena-single": [(s, "en" if lang == "en" else "other", m,
                          {**ARENA, "sha256": arena_sha256, "key": s, "language": lang,
                           "replies": replies})
                         for s, lang, m, turns, _, replies in sessions
                         if turns == 1 and lang not in ("und", "")],
        "arena-multi": [(s, "any", m, {**ARENA, "sha256": arena_sha256, "key": s,
                                       "language": lang, "earlier_turn_models": models})
                        for s, lang, m, turns, models, _ in sessions if turns >= 2],
        "oasst2-multi": [(k, "en", m, {**SOURCES["chat"], "key": k}) for k, m in oasst2],
    }
    records, counts, firsts = [], {}, []
    for source, pool in pools.items():
        needed = {b: 2 * q for (s, b), q in QUOTAS.items() if s == source}
        taken = {b: 0 for b in needed}
        rejected = 0
        excluded = {"user_drop": 0, "code_rules": 0, "duplicate": 0}
        pool = sorted(pool, key=lambda item: item[0])
        random.Random(f"{SEED}-v3-code-{source}").shuffle(pool)
        for key, bucket, messages, meta in pool:
            if all(taken[b] >= needed[b] for b in needed):
                break
            if taken.get(bucket, 0) >= needed.get(bucket, 0):
                continue
            if key in USER_DROPS:
                excluded["user_drop"] += 1
                continue
            where = code_source(messages, meta.pop("replies", []))
            if where is None:
                excluded["code_rules"] += 1
                continue
            if _duplicate(messages[0]["content"], firsts):
                excluded["duplicate"] += 1
                continue
            thinking = (taken[bucket] // 2) % 2 == 0
            tokens = _context_tokens(tokenizer, messages, None, thinking)
            if tokens > MAX_CONTEXT_TOKENS:
                rejected += 1
                continue
            records.append({"stratum": "code", "language": meta.get("language", "en"),
                            "source_kind": source, "code_source": where,
                            "split": SPLITS[taken[bucket] % 2],
                            "thinking": thinking, "messages": messages, "tools": None,
                            "source": meta, "context_tokens": tokens})
            firsts.append(messages[0]["content"])
            taken[bucket] += 1
        counts[source] = {"candidates": len(pool), "taken": taken, "needed": needed,
                          "over_context_limit": rejected, **excluded}
        print(f"{source}: {taken} of {needed} from {len(pool)} candidates "
              f"({rejected} over {MAX_CONTEXT_TOKENS} tokens, excluded {excluded})", flush=True)
        if any(taken[b] < needed[b] for b in needed):
            print(f"{source}: NOT ENOUGH candidates", flush=True)
    records.sort(key=lambda r: (SPLITS.index(r["split"]), r["source_kind"], str(r["source"]["key"])))
    counters: dict[str, int] = defaultdict(int)
    for index, record in enumerate(records):
        record["id"] = f"chat-v3-{record['split']}-code-{counters[record['split']]:02d}"
        counters[record["split"]] += 1
        record["seed"] = SEED + index
    args.work.mkdir(parents=True, exist_ok=True)
    with (args.work / "prompts.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.work / "review.json").write_text(json.dumps(
        {"counts": counts, "max_context_tokens": MAX_CONTEXT_TOKENS, "records": records},
        ensure_ascii=False), encoding="utf-8")
    print(f"{len(records)} prompts", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("select",))
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/qwen36_35b_base")
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_chat_corpus_v3_code")
    parser.add_argument("--sources", type=Path, default=ROOT / "data/qwen36_chat_corpus_v1/sources")
    parser.add_argument("--v1-corpus", type=Path,
                        default=ROOT / "data/qwen36_chat_corpus_v1/corpus.jsonl")
    parser.add_argument("--v2-prompts", type=Path,
                        default=ROOT / "data/qwen36_chat_corpus_v2/prompts.jsonl")
    parser.add_argument("--arena", type=Path,
                        default=ROOT / "data/arena_140k_shard/train-00000-of-00007.parquet")
    args = parser.parse_args()
    run_select(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
