# Bounded-memory Qwen3.6 work

This fork is adding a bounded-memory path for Qwen3.5/Qwen3.6 MoE models. The
target machine is an RTX 3060 with 12 GiB VRAM and about 32 GiB RAM. The design
keeps inactive weights and candidates packed on disk and limits dense device
state to one transformer block and a small routed-expert group.

## Implemented foundations

- `model_adapter.py` resolves nested `text_config`,
  `model.language_model.layers`, hybrid attention, and decoder-only variants.
- `tools/audit_tensor_coverage.py` classifies names directly from a safetensors
  index and fails when any source tensor remains unknown.
- `SafeTensorPrefixLoader` loads one checkpoint prefix from sharded
  safetensors into a meta-initialized model and releases it back to `meta`.
- Fused routed experts are calibrated in configurable chunks without unfusing
  all experts into persistent `nn.Linear` modules.
- Each expert shares one collected input Hessian between its gate and up
  projections; they consume copies sequentially during GPTQ.
- Qparams schema 2 stores integer codes at their actual bit width. Dense
  fake-quant candidates are optional and disabled by default.
- `WeightStore` indexes qparams-only databases and decodes one requested tensor
  without retaining it when its cache is off.
- KL teacher caches can store only top-k FP16 log-probabilities and int32 token
  indices. Candidate normalization uses `logsumexp` without constructing a
  full FP32 log-softmax tensor.
- The hard SPSA search core evaluates exact-budget two-choice assignments and
  retains no candidate deltas or model-weight autograd graph.
- Candidate generation checks free space before each write. The CLI default
  stops at a 50 GiB free-space floor.

The tensor-name audit of the local published checkpoint covers all 93,625
source tensors: 93,275 text tensors, 333 vision tensors, and 17 MTP tensors,
with zero unknown names. The recorded audit is
`reports/qwen35_source_coverage.json`.

## Streaming candidate generation

The streaming path requires a dense or otherwise trusted higher-precision base
checkpoint. The published two-bit GSQ experts can supply a two-bit candidate,
but they cannot reconstruct valid higher-bit candidates.

A representative two-choice routed-expert run is:

```bash
torchrun --nproc-per-node=1 run_quantize.py \
  --model_name_or_path /path/to/dense-qwen3.6-35b-a3b \
  --quantizable_modules '.*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$' \
  --calibration_data fineweb_edu \
  --calibration_tokens 8192 \
  --calibration_sequence_length 512 \
  --bitwidth_options 2 4 \
  --calibration_bitwidth 4 \
  --group_size 128 \
  --expert_chunk_size 8 \
  --stream_blocks \
  --cpu_offload_activations \
  --min_free_gb 50 \
  --save_dir /path/to/rco-database
```

The model adapter supplies embedding, block, and final-module paths unless the
corresponding command-line options override them. Increase the expert chunk
size only after measuring the Cholesky peak. `--save_fake_quant` re-enables the
legacy dense candidate files and should remain off for the low-memory path.

The block streamer has unit coverage with a meta model and sharded
safetensors. A full Qwen run still needs validation against the dense base
checkpoint and its installed Transformers version before it can be considered
production-ready.

## Compact reference caches

Both search drivers accept `--kl-topk`; the default is 20. With compact mode,
the cache contains `[batch, tokens, k]` values and indices rather than
`[batch, tokens, vocabulary]` probabilities. `--kl-topk 0` retains the original
full-vocabulary behavior.

The compact implementation reproduces the previous top-k objective on a test
model and retains a nonzero gradient through the candidate model. It preserves
the existing top-k objective semantics; it is not equivalent to full-vocabulary
KL.

## Hard assignment search

`rco_search_quant.py --search-mode hard-spsa` avoids persistent BF16 candidate
deltas. The current hard path supports exactly two bitwidths and equal-size
groups. That matches routed-expert groups when every group contains one
expert's gate, up, and down projections:

```text
--search-mode hard-spsa --moe-per-expert --bitwidths 2,4
```

Each SPSA step evaluates a plus/minus assignment pair on the same calibration
batch. Every assignment selects an exact number of high-bit groups, and the
driver rejects targets that the group count cannot represent exactly.

This removes candidate-delta memory, but the current driver still relies on a
loaded Hugging Face model during end-to-end search. The remaining major task is
to connect hard assignment evaluation to a training-aware block streamer so a
35B dense model is never resident in RAM. Until then, the hard path is useful
for small-model validation and for models that fit through supported offload.

## Remaining work

1. Validate meta initialization and a complete streamed block on the pinned
   Qwen3.6 Transformers implementation and a dense base checkpoint.
2. Add a streamed end-to-end executor for hard assignment and pruning loss,
   including reverse recomputation if gradients are needed.
3. Remove the remaining full-model dependency from final checkpoint building
   by writing selected tensors directly into bounded-size output shards.
4. Validate one complete optimization step, then a multi-step run, while
   recording CUDA allocated/reserved peaks, process RSS, disk growth, and I/O.
5. Load the materialized checkpoint and run numerical layer comparisons and a
   short generation test.

The standard test suite uses only tiny synthetic tensors and produces no model
artifacts:

```bash
python -m unittest discover -s tests -v
```
