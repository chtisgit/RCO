#!/usr/bin/env python3
"""``compare_qwen36_gsq_rco_candidate_heldout.py`` for a GGUF of another file type.

The original requires the two reports' ``environment`` to be identical; that
includes llama.cpp's model description (``qwen35moe 35B.A3B Q2_0``), which
names the file type and so differs for an outside GGUF such as Unsloth's
UD-Q2_K_XL.  This wrapper ignores only ``environment.description``; every
identity key and every other environment key must still match.  Same
arguments and output as the original.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_qwen36_gsq_rco_candidate_heldout as base  # noqa: E402

IGNORED = ("description",)


def require_same_scoring(left: dict, right: dict) -> None:
    for key in ("corpus_manifest", "helper", "llama_cpp", "parameters",
                "token_sequence_sha256", "schema"):
        if left["identity"][key] != right["identity"][key]:
            raise RuntimeError(f"held-out scoring identity differs: {key}")
    strip = lambda env: {k: v for k, v in env.items() if k not in IGNORED}  # noqa: E731
    if strip(left["environment"]) != strip(right["environment"]):
        raise RuntimeError("held-out scoring environment differs")


base._require_same_scoring = require_same_scoring

if __name__ == "__main__":
    base.main()
