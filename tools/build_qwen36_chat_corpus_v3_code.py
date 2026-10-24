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


def coding_prompt(text: str) -> bool:
    return bool(FENCE.search(text) or len(SYNTAX_LINE.findall(text)) >= 2
                or (LANGUAGE.search(text) and TASK.search(text)))


def _text(message: dict | None) -> str:
    content = (message or {}).get("content") or []
    return "\n".join(part.get("text") or "" for part in content if part.get("type") == "text")


def _has_image(message: dict | None) -> bool:
    return any(part.get("image") for part in (message or {}).get("content") or [])


def arena_sessions(path: Path):
    """Yield (session id, language, messages, turns, models) per coding session
    whose last vote row in the shard has ``is_code``."""
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
            yield session, last["language"], messages, len(turns), models


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
                          {**ARENA, "sha256": arena_sha256, "key": s, "language": lang})
                         for s, lang, m, turns, _ in sessions
                         if turns == 1 and lang not in ("und", "")],
        "arena-multi": [(s, "any", m, {**ARENA, "sha256": arena_sha256, "key": s,
                                       "language": lang, "earlier_turn_models": models})
                        for s, lang, m, turns, models in sessions if turns >= 2],
        "oasst2-multi": [(k, "en", m, {**SOURCES["chat"], "key": k}) for k, m in oasst2],
    }
    records, counts = [], {}
    for source, pool in pools.items():
        needed = {b: 2 * q for (s, b), q in QUOTAS.items() if s == source}
        taken = {b: 0 for b in needed}
        rejected = 0
        pool = sorted(pool, key=lambda item: item[0])
        random.Random(f"{SEED}-v3-code-{source}").shuffle(pool)
        for key, bucket, messages, meta in pool:
            if all(taken[b] >= needed[b] for b in needed):
                break
            if taken.get(bucket, 0) >= needed.get(bucket, 0):
                continue
            thinking = (taken[bucket] // 2) % 2 == 0
            tokens = _context_tokens(tokenizer, messages, None, thinking)
            if tokens > MAX_CONTEXT_TOKENS:
                rejected += 1
                continue
            records.append({"stratum": "code", "language": meta.get("language", "en"),
                            "source_kind": source, "split": SPLITS[taken[bucket] % 2],
                            "thinking": thinking, "messages": messages, "tools": None,
                            "source": meta, "context_tokens": tokens})
            taken[bucket] += 1
        counts[source] = {"candidates": len(pool), "taken": taken, "needed": needed,
                          "over_context_limit": rejected}
        print(f"{source}: {taken} of {needed} from {len(pool)} candidates "
              f"({rejected} over {MAX_CONTEXT_TOKENS} tokens)", flush=True)
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
