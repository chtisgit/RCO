#!/usr/bin/env python3
"""Vision check of a text-model GGUF with the GSQ vision projector.

The GSQ-hybrid model card's check, rerun for the release: read the headline
of llama.cpp's newspaper test image (``MEN WALK ON MOON``).  Every
generation uses the model card's sampling per mode (generation manifest v2),
thinking off and on, with the manifest's 3 seeds, at P3 of the offload
checks (``-ngl 99 --n-cpu-moe 4``, ``--fit off``).  This is a load and
short-generation check, not a vision benchmark.

    audit_qwen36_release_vision.py --model GGUF --label LABEL
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_chat_generation import _post, _wait_ready  # noqa: E402
from audit_qwen36_q3k_viability import _atomic_json, _load_json, _sha256_file  # noqa: E402

SCHEMA = "rco.qwen36.release_vision.v1"
QUESTION = "What is the main headline of this newspaper? Reply with the headline only."
EXPECTED = "MEN WALK ON MOON"


def reads_headline(answer: str) -> bool:
    words = re.sub(r"[^A-Z]+", " ", answer.upper()).split()
    return EXPECTED in " ".join(words)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--mmproj", type=Path, default=ROOT / "results/mmproj-BF16.gguf")
    parser.add_argument("--image", type=Path, default=ROOT / "repos/llama.cpp/tools/mtmd/test-1.jpeg")
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--server", type=Path,
                        default=ROOT / "experiment/build-cuda-mmqfix/bin/llama-server")
    parser.add_argument("--manifest", type=Path,
                        default=RCO / "reports/qwen36_35b_generation_prompt_manifest_v2.json")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    args = parser.parse_args()
    decoding = _load_json(args.manifest)["canonical_manifest"]["decoding"]
    image = "data:image/jpeg;base64," + base64.b64encode(args.image.read_bytes()).decode()
    output = args.reports / f"qwen36_release_vision_{args.label}.json"
    report = {"schema": SCHEMA, "label": args.label, "question": QUESTION, "expected": EXPECTED,
              "identity": {"model": {"path": str(args.model), "sha256": _sha256_file(args.model)},
                           "mmproj": {"path": str(args.mmproj), "sha256": _sha256_file(args.mmproj)},
                           "image": {"path": str(args.image), "sha256": _sha256_file(args.image)},
                           "server": {"path": str(args.server), "sha256": _sha256_file(args.server)},
                           "parameters": {"gpu_layers": 99, "n_cpu_moe": 4,
                                          "context": args.context, "fit": "off"}},
              "generations": []}
    base = f"http://127.0.0.1:{args.port}"
    log = (ROOT / f"qwen36_release_vision_{args.label}.server.log").open("w")
    command = [str(args.server), "-m", str(args.model), "--mmproj", str(args.mmproj), "--jinja",
               "-ngl", "99", "--n-cpu-moe", "4", "-c", str(args.context), "-np", "1",
               "--fit", "off", "--host", "127.0.0.1", "--port", str(args.port), "--no-webui"]
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        _wait_ready(base, process, timeout=1800)
        for thinking in (False, True):
            settings = decoding["thinking_on" if thinking else "thinking_off"]
            for seed in decoding["seeds"]:
                body = {"messages": [{"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": image}},
                            {"type": "text", "text": QUESTION}]}],
                        "chat_template_kwargs": {"enable_thinking": thinking},
                        **settings, "max_tokens": decoding["maximum_generated_tokens"],
                        "seed": seed, "verbose": True}
                response = _post(base + "/v1/chat/completions", body)
                message = response["choices"][0]["message"]
                verbose = response.get("__verbose", {})
                answer = message.get("content") or ""
                item = {"thinking": thinking, "seed": seed, "content": answer,
                        "reasoning_content": message.get("reasoning_content") or "",
                        "stop_type": verbose.get("stop_type"),
                        "tokens_predicted": verbose.get("tokens_predicted"),
                        "reads_headline": reads_headline(answer)}
                report["generations"].append(item)
                print(f"vision {args.label}: thinking {'on' if thinking else 'off'} seed {seed}: "
                      f"{item['reads_headline']} {answer.strip()[:80]!r}", flush=True)
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
        log.close()
    passed = sum(g["reads_headline"] and g["stop_type"] == "eos" for g in report["generations"])
    report["summary"] = {"generations": len(report["generations"]), "passed": passed}
    report["status"] = "pass" if passed == len(report["generations"]) else "fail"
    _atomic_json(output, report)
    print(f"vision {args.label}: {report['status']}, {passed} of "
          f"{len(report['generations'])}", flush=True)
    return 0 if report["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
