#!/usr/bin/env python3
"""Phase 4 chat checks of RCO_PLAN_NEW.md: build chat corpus v1.

Three strata, each from an Apache-2.0 Hugging Face dataset at a pinned
revision:

``chat``
    ``OpenAssistant/oasst2``: the top-ranked path of each human-written
    conversation tree (24 English, and 4 each of de/es/fr/it, per split).
``tool``
    ``glaiveai/glaive-function-calling-v2``: conversations with at least one
    function call.  The function schemas become the template's ``tools`` and
    each ``<functioncall>`` becomes a ``tool_calls`` entry, so the template
    renders it in its own ``<tool_call>`` XML format.  Function responses
    become ``tool`` messages.
``reasoning``
    ``open-thoughts/OpenThoughts-114k`` (shard 5 of 6): DeepSeek-R1 traces.
    The thought becomes ``reasoning_content`` and the solution the content,
    so the template renders a ``<think>`` block.

Every conversation is rendered with the model's own ``chat_template.jinja``
and tokenized with its tokenizer.  Kept: 192 to 1,024 tokens, and no
template markup inside the source text.  Candidates are taken in a seeded
shuffle; each stratum and language bucket takes twice its quota, alternating
between the ``calibration`` and ``heldout`` splits.

Scored tokens are the assistant's: from after ``<|im_start|>assistant\\n``
through ``<|im_end|>``.  An opening ``<think>\\n``, and an empty
``<think>\\n\\n</think>\\n\\n`` block, are skipped: llama-server prefills
those as part of the generation prompt, so the model never predicts them.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterator

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_q3k_viability import _atomic_json, _sha256_file  # noqa: E402
from release_corpus import canonical_json_bytes, sha256_bytes  # noqa: E402

SCHEMA = "rco.qwen36.chat_corpus_v1"
SEED = 20261010
MIN_TOKENS, MAX_TOKENS = 192, 1024
SPLITS = ("calibration", "heldout")
SOURCES = {
    "chat": {"dataset": "OpenAssistant/oasst2",
             "revision": "179dd21fc55192153d94adb0e0ce8f69e222bf75",
             "file": "2023-11-05_oasst2_ready.trees.jsonl.gz", "license": "Apache-2.0"},
    "tool": {"dataset": "glaiveai/glaive-function-calling-v2",
             "revision": "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac",
             "file": "glaive-function-calling-v2.json", "license": "Apache-2.0"},
    "reasoning": {"dataset": "open-thoughts/OpenThoughts-114k",
                  "revision": "bd093c3994fd54d2390985b66988ddf282a55eb6",
                  "file": "data/train-00005-of-00006.parquet", "license": "Apache-2.0"},
}
# Per split; each bucket draws twice this many.
QUOTAS = {("chat", "en"): 24, ("chat", "de"): 4, ("chat", "es"): 4, ("chat", "fr"): 4,
          ("chat", "it"): 4, ("tool", "en"): 40, ("reasoning", "en"): 40}
MARKUP = re.compile(r"<\|[a-z_]+\|>|</?think>|</?tool_call>|</?tool_response>|"
                    r"<function=|<parameter=|</?tools>")
ASSISTANT_TURN = re.compile(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.S)
EMPTY_THINK = "<think>\n\n</think>\n\n"
OPEN_THINK = "<think>\n"


def _source_path(sources: Path, stratum: str) -> Path:
    source = SOURCES[stratum]
    return sources / source["dataset"].replace("/", "__") / source["file"]


def _clean(*texts: str) -> bool:
    return all(text and text.strip() and not MARKUP.search(text) for text in texts)


def oasst2_candidates(path: Path) -> Iterator[tuple[str, str, list[dict], None]]:
    def usable(message: dict) -> bool:
        return not message.get("deleted") and message.get("review_result") is not False

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            tree = json.loads(line)
            node, messages = tree["prompt"], []
            while node is not None and usable(node):
                role = "user" if node["role"] == "prompter" else "assistant"
                messages.append({"role": role, "content": node["text"]})
                replies = [reply for reply in node.get("replies", []) if usable(reply)]
                replies.sort(key=lambda reply: (reply.get("rank") is None,
                                                reply.get("rank") or 0))
                node = replies[0] if replies else None
            while messages and messages[-1]["role"] != "assistant":
                messages.pop()
            if len(messages) >= 2 and _clean(*(m["content"] for m in messages)):
                yield tree["message_tree_id"], tree["prompt"]["lang"], messages, None


GLAIVE_TURN = re.compile(r"(USER|ASSISTANT|FUNCTION RESPONSE): ")
GLAIVE_CALL = re.compile(r"<functioncall>\s*(\{.*\})\s*$", re.S)


def _glaive_tools(system: str) -> list[dict] | None:
    marker = "Use them if required -"
    if marker not in system:
        return None
    text, decoder, tools = system.split(marker, 1)[1].strip(), json.JSONDecoder(), []
    while text:
        try:
            function, end = decoder.raw_decode(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(function, dict) or "name" not in function:
            return None
        tools.append({"type": "function", "function": function})
        text = text[end:].strip()
    return tools or None


def _glaive_call(text: str, names: set[str]) -> dict | None:
    match = GLAIVE_CALL.search(text)
    if match is None:
        return None
    raw = match.group(1)
    # The arguments are a JSON object quoted as a single-quoted string.
    raw = re.sub(r"\"arguments\":\s*'(.*)'\s*}$", r'"arguments": \1}', raw, flags=re.S)
    try:
        call = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if (not isinstance(call, dict) or call.get("name") not in names
            or not isinstance(call.get("arguments"), dict)):
        return None
    return {"type": "function",
            "function": {"name": call["name"], "arguments": call["arguments"]}}


def glaive_candidates(path: Path) -> Iterator[tuple[str, str, list[dict], list[dict]]]:
    for index, item in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        if "<functioncall>" not in item["chat"]:
            continue
        tools = _glaive_tools(item["system"])
        if tools is None:
            continue
        names = {tool["function"]["name"] for tool in tools}
        parts = GLAIVE_TURN.split(item["chat"])
        if parts[0].strip():
            continue
        messages, ok = [], True
        for speaker, text in zip(parts[1::2], parts[2::2]):
            text = text.strip()
            if text.endswith("<|endoftext|>"):
                text = text[:-len("<|endoftext|>")].rstrip()
            if speaker == "USER":
                messages.append({"role": "user", "content": text})
            elif speaker == "FUNCTION RESPONSE":
                messages.append({"role": "tool", "content": text})
            elif "<functioncall>" in text:
                call = _glaive_call(text, names)
                prefix = text.split("<functioncall>", 1)[0].strip()
                if call is None:
                    ok = False
                    break
                previous = messages[-1] if messages else {}
                if (previous.get("role") == "assistant" and "tool_calls" not in previous
                        and not prefix):
                    # A text turn followed by a call turn is one assistant
                    # message in the template: text, then the tool call.
                    previous["tool_calls"] = [call]
                else:
                    messages.append({"role": "assistant", "content": prefix,
                                     "tool_calls": [call]})
            else:
                messages.append({"role": "assistant", "content": text})
        if (not ok or len(messages) < 2 or messages[0]["role"] != "user"
                or messages[-1]["role"] != "assistant"):
            continue
        texts = [m["content"] for m in messages if m["content"]]
        texts += [json.dumps(m["tool_calls"]) for m in messages if "tool_calls" in m]
        if _clean(*texts, json.dumps(tools)):
            yield str(index), "en", messages, tools


THOUGHT = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>\s*"
                     r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>\s*$", re.S)


def openthoughts_candidates(path: Path) -> Iterator[tuple[str, str, list[dict], None]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["conversations"])
    for index, row in enumerate(table.column("conversations").to_pylist()):
        if [turn["from"] for turn in row] != ["user", "assistant"]:
            continue
        match = THOUGHT.match(row[1]["value"].strip())
        if match is None:
            continue
        thought, solution = match.group(1).strip(), match.group(2).strip()
        question = row[0]["value"].strip()
        if _clean(question, thought, solution):
            yield str(index), "en", [
                {"role": "user", "content": question},
                {"role": "assistant", "content": solution, "reasoning_content": thought},
            ], None


def scored_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for match in ASSISTANT_TURN.finditer(text):
        start = match.start(1)
        body = match.group(1)
        if body.startswith(EMPTY_THINK):
            start += len(EMPTY_THINK)
        elif body.startswith(OPEN_THINK):
            start += len(OPEN_THINK)
        spans.append((start, match.end()))
    return spans


def render(tokenizer, messages: list[dict], tools: list[dict] | None):
    text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = scored_spans(text)
    scored = [int(any(lo <= start < hi for lo, hi in spans))
              for start, _ in encoded["offset_mapping"]]
    return text, list(encoded["input_ids"]), scored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/qwen36_35b_base")
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_chat_corpus_v1")
    parser.add_argument("--output", type=Path,
                        default=RCO / "reports/qwen36_chat_corpus_v1_manifest.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    sources = args.work / "sources"
    generators = {"chat": oasst2_candidates, "tool": glaive_candidates,
                  "reasoning": openthoughts_candidates}
    records = []
    for stratum, generator in generators.items():
        path = _source_path(sources, stratum)
        candidates = sorted(generator(path), key=lambda item: item[0])
        random.Random(f"{SEED}-{stratum}").shuffle(candidates)
        needed = {lang: 2 * quota for (s, lang), quota in QUOTAS.items() if s == stratum}
        taken = {lang: 0 for lang in needed}
        for key, language, messages, tools in candidates:
            if taken.get(language, 0) >= needed.get(language, 0):
                if all(taken[lang] >= needed[lang] for lang in needed):
                    break
                continue
            text, tokens, scored = render(tokenizer, messages, tools)
            if not MIN_TOKENS <= len(tokens) <= MAX_TOKENS or sum(scored[1:]) < 16:
                continue
            split = SPLITS[taken[language] % 2]
            taken[language] += 1
            records.append({
                "stratum": stratum, "language": language, "split": split,
                "source": {**SOURCES[stratum], "key": key},
                "messages": messages, "tools": tools, "text": text,
                "_tokens": tokens, "_scored": scored})
        if any(taken[lang] < needed[lang] for lang in needed):
            raise RuntimeError(f"{stratum}: not enough candidates {taken} for {needed}")
        print(f"{stratum}: {taken} from {len(candidates)} candidates", flush=True)

    records.sort(key=lambda r: (SPLITS.index(r["split"]), list(generators).index(r["stratum"])))
    counters: dict[tuple[str, str], int] = {}
    for record in records:
        bucket = (record["split"], record["stratum"])
        counters[bucket] = counters.get(bucket, 0) + 1
        record["id"] = f"chat-v1-{record['split']}-{record['stratum']}-{counters[bucket] - 1:02d}"

    corpus_path = args.work / "corpus.jsonl"
    tokens_path = args.work / "tokens.json"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({key: value for key, value in record.items()
                                     if not key.startswith("_")}, ensure_ascii=False) + "\n")
    tokens_path.write_text(json.dumps(
        [{"id": r["id"], "tokens": r["_tokens"], "scored": r["_scored"]} for r in records]),
        encoding="utf-8")
    template = args.model_dir / "chat_template.jinja"
    canonical = {
        "schema": SCHEMA,
        "seed": SEED,
        "token_range": [MIN_TOKENS, MAX_TOKENS],
        "quotas_per_split": {f"{s}/{lang}": q for (s, lang), q in QUOTAS.items()},
        "scoring": ("assistant tokens from after '<|im_start|>assistant\\n' through "
                    "'<|im_end|>', skipping a leading '<think>\\n' or empty think block"),
        "sources": SOURCES,
        "chat_template": {"path": str(template), "sha256": _sha256_file(template)},
        "tokenizer_sha256": _sha256_file(args.model_dir / "tokenizer.json"),
        "corpus": {"path": str(corpus_path), "sha256": _sha256_file(corpus_path)},
        "tokens": {"path": str(tokens_path), "sha256": _sha256_file(tokens_path)},
        "source_files": {stratum: {"path": str(_source_path(sources, stratum)),
                                   "sha256": _sha256_file(_source_path(sources, stratum))}
                         for stratum in SOURCES},
        "conversations": [{
            "id": r["id"], "stratum": r["stratum"], "language": r["language"],
            "split": r["split"], "source_key": r["source"]["key"],
            "token_count": len(r["_tokens"]), "scored_target_count": sum(r["_scored"][1:]),
            "content_sha256": sha256_bytes(r["text"].encode("utf-8")),
        } for r in records],
    }
    summary = {}
    for split in SPLITS:
        rows = [r for r in records if r["split"] == split]
        summary[split] = {
            "conversations": len(rows),
            "tokens": sum(len(r["_tokens"]) for r in rows),
            "scored_targets": sum(sum(r["_scored"][1:]) for r in rows),
        }
    _atomic_json(args.output, {
        "schema": SCHEMA + ".manifest",
        "status": "complete",
        "canonical_manifest": canonical,
        "canonical_manifest_sha256": sha256_bytes(canonical_json_bytes(canonical)),
        "summary": summary,
    })
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
