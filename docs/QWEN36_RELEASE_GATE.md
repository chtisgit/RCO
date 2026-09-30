# Qwen3.6-35B-A3B release gate

The four-token search loss is a deterministic engineering oracle, not a
release-quality measurement. A relaxed projection may authorize construction
of a final GGUF only when `tools/evaluate_qwen36_release_gate.py` returns
`authorized_for_final_gguf_construction: true`. There are no manual waivers.

## Held-out perplexity

The evaluation corpus must be immutable, locally licensed, and identified by
the SHA-256 digest of its canonical manifest. It must be disjoint from every
search/calibration prompt and contain at least 100 documents, 100,000 predicted
tokens, and four declared strata covering general text, knowledge, code, and
multilingual text. Candidate, BF16, and incumbent losses use identical tokens
and exact full-vocabulary causal cross-entropy; top-k KL is not accepted.

The candidate must satisfy both of these quality bounds:

- perplexity is no more than 1.15 times the pinned BF16 base perplexity; and
- the upper endpoint of a document-paired 95% bootstrap confidence interval
  for candidate-minus-incumbent mean NLL is at most zero.

An independent repeat must use the identical token-sequence digest and match
mean NLL within `1e-6`. Any non-finite token loss fails the gate.

## Deterministic generation

Use an immutable manifest of at least 32 prompts spanning instruction
following, extraction, arithmetic/reasoning, code, knowledge, and multilingual
behavior. Each prompt declares a machine-checkable exact, regular-expression,
JSON-schema, or numeric-tolerance assertion. Every required assertion must
pass at temperature zero in both deterministic runs. Empty output, invalid
UTF-8, Unicode replacement characters, or non-finite runtime state fails the
gate. The manifest digest and complete token outputs belong in the evidence.

## Runtime and memory

The same assignment must have an independent exact hard replay at its exact
serialized-byte target. The Q2_0/Q4_0 CUDA kernel matrix, one-step CUDA memory
gate, multi-step CUDA memory gate, and partial llama.cpp CUDA offload must all
pass. Streamed CUDA reserved memory must remain below 10 GiB, and the runtime
must be the pinned unmodified llama.cpp revision.

## Construction boundary

The gate consumes measured assignment evidence and produces a decision report;
it does not invoke the GGUF writer. A failing or incomplete report preserves
the existing incumbent and forbids final GGUF construction. After a passing
preconstruction report, final construction and byte-for-byte/load/generation
verification remain separate Phase 6 actions.
