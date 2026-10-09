#!/usr/bin/env python3
"""Phase 3 step f of RCO_PLAN_NEW.md: cross-seed consensus pruning mask.

The rule was fixed before any candidate was scored:

1. Every expert that both seeds' final masks prune stays pruned.
2. Each layer's remaining slots, up to 24, are filled from the experts that
   exactly one seed prunes.  They are ranked by the mean of the two seeds'
   final prune advantage ``alpha[:, 1] - alpha[:, 0]``, the quantity whose
   per-layer top 24 gives each seed's own final mask.

Experts that neither seed prunes are never chosen.  The mask is written to
``data/qwen36_prune24/consensus_mask.npy`` and its provenance to
``reports/qwen36_gsq_e6_prune24_consensus.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

RCO = Path(__file__).resolve().parents[1]
ROOT = RCO.parents[1]
sys.path.insert(0, str(RCO / "tools"))
sys.path.insert(0, str(RCO / "src"))

from audit_qwen36_prune24_prelim import EXPERTS, LAYERS, PRUNE_PER_LAYER  # noqa: E402
from audit_qwen36_q3k_viability import _atomic_json, _sha256_file  # noqa: E402

SCHEMA = "rco.qwen36.prune24_consensus.v1"
SEEDS = (0, 1)


def _advantage(state_path: Path, steps: int) -> np.ndarray:
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state["step"] != steps:
        raise ValueError(f"{state_path}: step {state['step']}, expected {steps}")
    logits = state["alpha"].detach().float().reshape(LAYERS * EXPERTS, 2)
    return (logits[:, 1] - logits[:, 0]).numpy().reshape(LAYERS, EXPERTS)


def _top_per_layer(score: np.ndarray, k: int) -> np.ndarray:
    mask = np.zeros(score.shape, dtype=bool)
    np.put_along_axis(mask, np.argsort(-score, axis=1, kind="stable")[:, :k], True, axis=1)
    return mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, default=ROOT / "data/qwen36_prune24")
    parser.add_argument("--reports", type=Path, default=RCO / "reports")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    inputs, finals, advantages = {}, [], []
    for seed in SEEDS:
        run = args.work / f"search_seed{seed}"
        final = np.load(run / "final_mask.npy")
        advantage = _advantage(run / "state.pt", args.steps)
        # Each final mask must be the per-layer top 24 of its own advantage.
        if not np.array_equal(_top_per_layer(advantage, PRUNE_PER_LAYER), final):
            raise ValueError(f"seed {seed}: final mask is not the top-24 of its alpha")
        finals.append(final)
        advantages.append(advantage)
        inputs[f"seed{seed}"] = {
            "final_mask": {"path": str(run / "final_mask.npy"),
                           "sha256": _sha256_file(run / "final_mask.npy")},
            "state": {"path": str(run / "state.pt"),
                      "sha256": _sha256_file(run / "state.pt")}}

    both = finals[0] & finals[1]
    one = finals[0] ^ finals[1]
    mean_advantage = (advantages[0] + advantages[1]) / 2
    # Agreed experts rank first, single-seed experts by mean advantage, the
    # rest are excluded.
    score = np.where(both, np.inf, np.where(one, mean_advantage, -np.inf))
    mask = _top_per_layer(score, PRUNE_PER_LAYER)
    assert bool((mask.sum(axis=1) == PRUNE_PER_LAYER).all())
    assert bool((mask & both).sum() == both.sum())
    assert not bool((mask & ~(finals[0] | finals[1])).any())

    output = args.work / "consensus_mask.npy"
    np.save(output, mask)
    agreed = both.sum(axis=1)
    summary = {
        "agreed_by_both_seeds": int(both.sum()),
        "agreed_per_layer": {"min": int(agreed.min()), "median": float(np.median(agreed)),
                             "max": int(agreed.max())},
        "filled_from_one_seed": int((mask & one).sum()),
        "shared_with_seed0_final": int((mask & finals[0]).sum()),
        "shared_with_seed1_final": int((mask & finals[1]).sum()),
    }
    _atomic_json(args.reports / "qwen36_gsq_e6_prune24_consensus.json", {
        "schema": SCHEMA,
        "status": "complete",
        "rule": (
            "prune every expert both seeds' final masks prune; fill each layer to "
            f"{PRUNE_PER_LAYER} from experts exactly one seed prunes, ranked by the mean "
            "of the two seeds' final alpha[:,1]-alpha[:,0]; never choose an expert "
            "neither seed prunes"),
        "inputs": inputs,
        "mask": {"path": str(output), "sha256": _sha256_file(output),
                 "pruned_per_layer": PRUNE_PER_LAYER},
        **summary,
    })
    print(f"consensus mask: {output}", summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
