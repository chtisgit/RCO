# Qwen3.6-35B-A3B: GSQ-E6, expert pruning and Q3_K upgrades

This is the current line of work in this fork. It is called "plan v2" in
the tool docstrings, which refer to it as `RCO_PLAN_NEW.md`, the experiment
plan; that plan is not published yet. This page summarizes its method,
decision rules and results so far. The evidence for each result is in
`reports/`.

The earlier relaxed and hard bit-width searches and the streaming
foundations are documented in [LOW_MEMORY.md](LOW_MEMORY.md).

## Goal

Build a Qwen3.6-35B-A3B GGUF of the same size as the authentic 2-bit GSQ
release (at most 12,217,776,832 bytes) that is measurably better than GSQ
as a chat model.

- **Primary measure (since 2026-10-10):** on the held-out half of chat
  corpus v2, the paired per-conversation difference in top-20 KL to BF16
  (candidate minus GSQ) must have a bootstrap 95% upper bound below 0.
- **Guard:** raw-text held-out NLL against GSQ.
- **Reported, secondary:** perplexity within 1.15 × BF16.

RCO decides where the bytes go. The bytes come from two places:
1. **GSQ-E6:** the BF16 `token_embd` is downgraded to Q6_K.
2. **P24:** 24 of the 256 routed experts are removed in each of the 40 MoE
   layers.

They are spent on upgrading GSQ Q2_0 routed-expert tensors to Q3_K, which
is quantized from the BF16 source, never from decoded GSQ.

Fixed decisions:
- Q8_0 tensors stay as authentic GSQ bytes.
- Routers stay F32.
- Pruning removes the same count in every layer (232 kept).
- Upgrades are Q2_0 → Q3_K only.

### Budget (exact, in bytes)

| Item | Bytes |
| --- | ---: |
| Authentic GSQ file | 12,217,776,832 |
| Prune 24 experts × 40 layers × 3 Q2_0 expert tensors | −849,346,560 |
| Prune 24 router rows × 40 layers (F32) | −7,864,320 |
| `token_embd` BF16 → Q6_K | −599,941,120 |
| **Freed** | **−1,457,152,000** |
| One Q2_0 → Q3_K upgrade of a 232-expert tensor | +36,110,336 |
| **Affordable upgrades** | **40 of the 120 routed-expert tensors** |
| Resulting file | 12,205,038,272 |

All 120 routed-expert tensors (gate, up and down × 40 layers) cost the
same to upgrade, so the quant search is a cardinality problem: choose
exactly 40 of 120.

## Concepts

- **Streaming.** The 35B model never fits in memory. Every forward or
  backward pass loads one decoder block at a time and then releases it. The
  block comes either from the GGUF (GSQ Q2_0, decoded on load by
  `src/gguf_parallel_stream.py`) or from the BF16 safetensors.
- **Reference.** Objectives and scores compare against BF16 through cached
  top-20 log-probabilities, never against GSQ. Using GSQ as the teacher would
  preserve its own 2-bit errors.
- **Two pruning semantics.**
  - The search uses RCO's surrogate (`MoEPruneWrapper`): top-8 is taken
    over all 256 experts, and a pruned expert's slot is scaled by its STE
    survival, without renormalization.
  - Every score and comparison uses exact routing: pruned router logits are
    set to −inf, then softmax, top-8 and renormalization. This equals
    llama.cpp with the experts physically removed, which Phase 4 checks end
    to end.
- **Chat is the yardstick (since 2026-10-10).** The model is only used
  through its chat template, so the search, the imatrix and every score use
  chat-formatted data. Only the final assistant turn is scored, and it is
  the model's own reply, generated with the model card's sampling.
- **Held-out data is never used for search, imatrix, selection or
  tuning.** That covers the raw-text held-out corpus and the `heldout`
  halves of both chat corpora.

## Phases

| Phase | What | Status |
| --- | --- | --- |
| 0 | Is Q3_K a useful upgrade over GSQ Q2_0? | Go |
| 1 | GSQ-E6: Q6_K `token_embd` | Pass |
| 2 | Calibration corpus v2 (raw text) | Done |
| 3 | RCO pruning search on raw text: P24 mask | Done (superseded by 4b) |
| 4 | P24 GGUF, llama.cpp parity, chat checks | Done |
| 4b | Chat corpus v2 and the pruning search rerun on it | In progress |
| 5 | Per-expert imatrix on the pruned model | Not started |
| 6 | Q3_K candidate store | Not started |
| 7 | RCO quant search: 40 of 120 tensors | Not started |
| 8 | Evaluation candidate and held-out gate | Not started |
| 9 | Remaining gates (parity, generation, IFEval, offload) | Not started |

### Phase 0: Q3_K viability

- **Method.**
  - A streamed BF16 forward over the 50-document calibration set collects
    a per-expert imatrix: Σx² of the expert inputs, following llama.cpp's
    `MUL_MAT_ID` convention.
  - Tensor-level errors are measured on 15 sampled tensors for GSQ Q2_0,
    Q3_K with and without the imatrix, and Q4_K.
  - Exact calibration NLL is measured for GSQ and for three upgrade arms.
- **Gate.**
  - imatrix-Q3_K beats GSQ Q2_0 on weighted error on all sampled tensors,
    with a median reduction of at least 30%.
  - The imatrix beats plain Q3_K.
  - The down-family ΔNLL has a 95% upper bound below 0.
  - If Q3_K's gain per GB is below Q4_0's, the choice of upgrade type goes
    back to the user.
- **Result: go.**
  - Q3_K on the 40 `ffn_down_exps` tensors: −0.131 NLL for 1.59 GB.
  - Q4_0 on the same tensors: −0.139 NLL for 3.02 GB. So Q3_K gives 94% of
    Q4_0's gain for 53% of the bytes.
  - Q3_K on all 120 tensors closes 89% of the GSQ-to-BF16 gap.
- **Tool and reports.** `audit_qwen36_q3k_viability.py`, writing
  `qwen36_q3k_viability{,_imatrix,_tensors,_evaluation}.json`.

### Phase 1: GSQ-E6

- **Method.** Replace GSQ's BF16 `token_embd`, which is bit-identical to
  the source, with Q6_K quantized from BF16 without an imatrix, since the
  embedding is a lookup table. Then score calibration NLL.
- **Gate.** Stop if |ΔNLL| > 0.002.
- **Result: pass.** ΔNLL +0.0000124, 95% CI [−0.0018, +0.0018], while
  saving 599,941,120 bytes. GSQ-E6 is the base model from here on.
- **Tool and reports.** `audit_qwen36_gsq_e6.py`, writing
  `qwen36_gsq_e6_{embedding,calibration,gguf}.json`.

### Phase 2: Calibration corpus v2

- 250 documents × 512 tokens (127,750 predicted tokens) from the same
  licensed sources and strata as the original calibration set.
- Disjoint from the held-out corpus and from calibration v1; the tokens are
  pinned by a sha256 manifest.
- **Tool and report.** `build_qwen36_calibration_corpus_v2.py`, writing
  `qwen36_35b_calibration_corpus_v2_manifest.json`.

### Phase 3: Pruning search on raw text

- **Model and objective.** GSQ-E6 is pruned with RCO's projected
  Gumbel-STE search, using a per-layer budget of exactly 24. The objective
  is the top-20 KL to BF16 on calibration v2.
- **Search defaults** (RCO's `run_search_prune.sh`):
  - 300 steps, lr 0.1, τ from 1.0 to 0.05;
  - 4 Gumbel samples in 2 antithetic pairs;
  - `alpha` initialized from GSQ-E6 router probability sums, spread 5.0.
- **Streaming.** `src/search/streamed_prune.py` runs every block under
  activation checkpointing, so backward reloads it. All Gumbel samples
  share one pass, and every step is checkpointed and resumable. It matches
  RCO's resident `MoEPruneWrapper` path exactly on a tiny model
  (`tests/test_streamed_prune.py`).
- **Steps.**

  | Step | What | Tool |
  | --- | --- | --- |
  | a–c | BF16 reference, router statistics, frequency baseline (each layer's 24 least-selected experts) | `audit_qwen36_prune24_prelim.py` |
  | d | Timing pilot (gate: at most 30 min per step) | `search_qwen36_prune24.py pilot` |
  | e | Two searches, seeds 0 and 1, 12.9 h each | `search_qwen36_prune24.py search` |
  | f | Exact scoring of both final masks and their consensus | `build_qwen36_prune24_consensus.py`, `search_qwen36_prune24.py score` |
  | g | Independent replay of the promoted mask | `search_qwen36_prune24.py score` |

- **Promotion rule, fixed before scoring.**
  - The candidates are each seed's final mask and the consensus mask: both
    seeds' shared prunes first, then the remaining slots filled by mean
    prune advantage.
  - The lowest exact KL wins, provided it beats the frequency baseline.
- **Result: the consensus mask is promoted (P24).**
  - KL 0.47984 against 0.48994 unpruned, Δ −0.0101 [−0.0124, −0.0079].
  - Against the frequency baseline: Δ −0.0186.
  - NLL improves by 0.0129 over the unpruned model while freeing 849 MB.
  - The two seeds score the same although their masks share only 58% of
    pruned experts, which points to a flat landscape.
  - The replay reproduces the scores.
- **Reports.**
  `qwen36_gsq_e6_prune24_{reference,router_stats,frequency_baseline,pilot,search_seed0,search_seed1,consensus,score_*}.json`.

### Phase 4: Pruned GGUF, llama.cpp parity and chat checks

- **P24 GGUF** (`build_qwen36_gsq_rco_v2_gguf.py`, report
  `qwen36_gsq_e6_p24_gguf.json`).
  - `expert_count` is set to 232.
  - The kept experts' Q2_0 slices and router rows are byte ranges of
    authentic GSQ, checksummed per slice.
  - `token_embd` is Q6_K.
  - The file is 10,760,624,832 bytes.
- **Differential logit parity** (`audit_qwen36_p24_logit_parity.py`).
  - Native evaluator against llama.cpp, run once for the unpruned control
    and once for P24.
  - Pass: P24's logit RMSE is at most 1.2 × the control's, its top-1
    agreement is at least the control's − 3 points, and its per-layer error
    is within margin.
  - **Pass:** RMSE ratio 0.96, top-1 agreement 97.8% against 96.5%. Both
    runtimes reproduce the same pruning effect.
- **llama.cpp document NLL** (`audit_qwen36_gsq_gguf_nll.py`,
  `compare_qwen36_p24_v2_llama_nll.py`).
  - The pruning gain is −0.01289 in llama.cpp against −0.01294 in the
    evaluator.
  - Per-document r is 0.99996.
- **Chat check 1: generation in llama-server** with the GGUF's own Jinja
  template (`audit_qwen36_chat_generation.py`,
  `compare_qwen36_chat_generation.py`).
  - **Prompts:** 12 prompts with machine-checked assertions: thinking off,
    thinking on, tool calls and multi-turn.
  - **Runs:** P24 and the unpruned control, each on CPU and on GPU.
  - **Revision v1, greedy:** five P24-only failures. They did not reproduce
    under the model card's sampling, which is consistent with
    greedy-decoding artefacts.
  - **Revision v2, the model card's sampling per mode, 3 seeds:**
    - The sampling is temperature 1.0, top-p 0.95, top-k 20 and presence
      penalty 1.5 with thinking on; temperature 0.7 and top-p 0.8 with it
      off.
    - The gate fails a check only if P24 fails it on more seeds than the
      control.
    - **Pass:** 0 of 66 checks failed for either model on either device.
  - Prompts: `tools/qwen36_chat_generation_prompts{,_v2}.json`. Reports:
    `qwen36_chat_generation_*{,_v2}.{json,md}`.
- **Chat check 2: chat KL to BF16** (`audit_qwen36_chat_kl.py`, chat
  corpus v1 from `build_qwen36_chat_corpus.py`). Not run: chat corpus v2
  scoring superseded it.

### Phase 4b: Chat corpus v2 and the search rerun

The P24 mask was chosen on raw text. Chat became the yardstick, so the
search is rerun on chat data. Raw text stays as content inside user turns:
pasted documents, files and tool results.

- **Chat corpus v2** (`build_qwen36_chat_corpus_v2.py`, manifest
  `qwen36_chat_corpus_v2_manifest.json`).
  - 223 conversations per split, `calibration` and `heldout`, with disjoint
    source items and none reused from chat corpus v1.
  - Each conversation is rendered with the model's own template, and only
    the final assistant turn is scored.
  - **Replies are on-policy.** They are generated by a Q6_K of the pinned
    BF16 checkpoint in llama-server, with the model card's sampling for the
    mode, at most 6,144 new tokens, and one seed per prompt.
  - **Tokens** are exactly what the server saw and wrote: the prompt as
    tokenized, the generated ids and the closing `<|im_end|>`.

  | Stratum | Per split | Source | Thinking |
  | --- | ---: | --- | --- |
  | chat | 64 (en 24; de, es, fr, it, pt 8 each) | `OpenAssistant/oasst2` | alternating |
  | tool-call | 24 | `glaiveai/glaive-function-calling-v2` | alternating |
  | tool-answer | 24 | same, different conversations | alternating |
  | reasoning | 48 | `open-thoughts/OpenThoughts-114k` questions | on |
  | document | 63 | calibration v2 documents plus an instruction | alternating |

- **Search rerun** (`search_qwen36_prune24_chat.py`, objective
  `src/search/streamed_prune_chat.py`).
  - Phase 3's search, defaults and promotion rule, on the calibration half,
    with the masked per-conversation top-20 KL.
  - Conversations run in length-sorted micro-batches. Each block is one
    autograd function loaded once per direction, so a step's token count
    does not multiply the block loads.
  - It matches RCO's resident path row by row on padded, partly scored rows
    (`tests/test_streamed_prune_chat.py`).
  - The initial `alpha` and the frequency baseline come from router
    statistics over the chat calibration half
    (`audit_qwen36_chat_kl.py score --router-stats`).
  - Exact scores and the BF16 reference come from `audit_qwen36_chat_kl.py`
    with `--corpus` set to the v2 manifest.
  - The old P24 mask is scored alongside for comparison.

### Phases 5–9 (planned)

- **5. imatrix.** Per-expert imatrix from a streamed BF16 forward over the
  chat v2 calibration half (all tokens), restricted to the kept experts.
- **6. Q3_K candidate store.** All 120 tensors, quantized from BF16 rows
  with their experts' imatrix rows, with checksums and error statistics.
- **7. Quant search.** The exact-cost hard search (`search.hard`) chooses 40
  of 120 upgrades to minimize masked top-20 KL on the chat v2 calibration
  half.
  - It must beat two controls: all 40 down tensors, and the top 40 by
    imatrix-weighted error reduction.
  - The ceiling with all 120 upgraded is reported.
- **8. Evaluation candidate.** The GGUF at 12,205,038,272 bytes, validated
  byte by byte against GSQ. It must pass the primary gate on the chat v2
  held-out half, with the raw-text held-out NLL as a guard.
- **9. Remaining gates.**
  - Cross-runtime logit parity.
  - At least 32 generation prompts with machine-checked answers, with the
    model card's sampling, 3 seeds, thinking off and on
    (`build_qwen36_generation_manifest.py`, manifest v2).
  - IFEval against unpruned GSQ.
  - CUDA offload checks.

Evaluation-only GGUFs are named `*-UNGATED.gguf` and are never published.
Building a release GGUF needs explicit authorization.

## Shared code

- `src/search/streamed_prune.py`: block-streamed Gumbel-STE pruning on
  fixed-length rows, and `StreamedPruneSearch`, which is RCO's `optimize`
  with a per-layer budget, resumable checkpoints and all samples in one
  pass.
- `src/search/streamed_prune_chat.py`: the same objective for
  variable-length conversations.
- `src/gguf_parallel_stream.py`: multi-threaded native decoding of GGUF
  blocks. It is bit-identical to the base loader.
- `tools/audit_qwen36_prune24_prelim.py`: `RouterHooks`, which applies exact
  pruning and collects router statistics, optionally skipping padding; also
  `exact_pruned_routing` and `frequency_prune_mask`.
- `tools/audit_qwen36_q3k_viability.py`: model skeletons and corpus loaders
  reused by later phases.

## Practical notes

- **Attention implementation.** Chat corpus v2 has conversations of up to
  8,192 tokens. Eager attention materializes the full attention matrix and
  runs out of memory on them in 12 GiB, even forward-only. All v2
  measurements use SDPA (`--attn-implementation sdpa`), and a score refuses
  a reference made with another implementation. The v1 tools default to
  eager, as they were run.
- **Report naming.** The long-running tools (references, scores, searches)
  skip a report that is already complete; they print "already complete"
  and exit. Reruns use a new label or suffix. Comparison tools are cheap
  and rewrite their report.
- **Resumability.** Long steps are resumable: per batch for reference and
  scoring passes, and per optimizer step for searches.
- **Data outside the repository.** Masks, reference caches, router
  statistics and corpora are written to a `data/` directory outside the
  repository and identified in the reports by path and sha256. Arrays and
  model files are not committed.
- **Hardware assumed.** One RTX 3060 (12 GiB), 31 GiB of RAM and 8 CPU
  cores. Defaults such as micro-batch sizes and decode workers are tuned to
  that machine.
