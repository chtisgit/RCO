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
- Qparams schema 3 stores integer codes at their actual bit width, records a
  SHA-256 digest for each tensor payload, and publishes each candidate with an
  atomic same-directory rename. Schema 1 and 2 files remain readable.
- Candidate generation writes an atomic `candidate-index.json` with paths,
  sizes, shapes, quantization metadata, and component checksums. WeightStore
  uses this index for parameter counts without decoding candidate payloads.
- Candidate dequantization works in column chunks, can target a caller-provided
  output tensor, and restores activation-order columns without a second dense
  candidate allocation. Dense fake-quant candidates remain optional and are
  disabled by default.
- `WeightStore` indexes qparams-only databases and decodes one requested tensor
  without retaining it when its cache is off.
- KL teacher caches can store only top-k FP16 log-probabilities and int32 token
  indices. Candidate normalization uses `logsumexp` without constructing a
  full FP32 log-softmax tensor.
- The hard-search core offers paired SPSA and antithetic REINFORCE estimators.
  Both evaluate exact-budget two-choice assignments and retain no candidate
  deltas or model-weight autograd graph.
- `--stream-hard-eval` runs hard-search cross-entropy evaluation through a meta
  model. Lightweight block wrappers preserve the canonical Transformers
  forward loop, including its per-layer hybrid-attention masks and rotary
  inputs, while loading and releasing each decoder block on demand.
- Streamed evaluation copies logical gate/up/down candidates directly into
  fused expert slices and reverses GPTQ activation-order column permutations
  before installing a candidate in an ordinary linear weight.
- Exact causal cross-entropy streams the output vocabulary in configurable
  row chunks instead of materializing `[batch, sequence, vocabulary]` logits.
- Candidate generation checks free space before each write. The CLI default
  stops at a 50 GiB free-space floor.
- `GGMLNativeCodec` calls the pinned GGML shared library directly for native
  row quantization and bounded caller-buffer dequantization. The schema-1
  `NativeCandidateStore` atomically publishes exact payloads and its index,
  recording canonical GGUF shapes, native type geometry, exact payload and
  aligned costs, source provenance, and SHA-256 integrity data without a
  persistent decoded cache.

The deterministic native-store proof covers Q2_0 and Q4_0 on both an ordinary
matrix and an aggregated three-expert tensor. Configure the exact pinned
library explicitly so type IDs and bytes cannot silently come from another
GGML build:

```bash
python tools/audit_native_ggml_store.py \
  --ggml-library /path/to/pinned/llama.cpp/build/bin/libggml-base.so \
  --output reports/native_ggml_store_audit.json
```

This small proof establishes byte-stable chunking, geometry/cost accounting,
atomic storage, checksummed streaming reads, and bounded decoding. It does not
replace the pending reference-GGUF load test, real Qwen3.5-2B candidate run,
genuine Qwen3.6-MoE block gate, or CUDA memory gates.

The reference-GGUF and synthetic-MoE load gate is now complete as a separate
audit. `tools/audit_tiny_native_gguf.py` constructs a deterministic, complete
one-layer Qwen3.5-MoE checkpoint with three routed experts, converts it through
the pinned llama.cpp converter, and verifies that gate/up/down aggregation
preserves source expert order `[0, 1, 2]` exactly. It then creates both Q2_0
and Q4_0 native candidates for an ordinary attention matrix and all three
aggregated routed projections. A mixed assignment selects Q2_0 for attention-q
and routed-up, and Q4_0 for routed-gate and routed-down.

`src/native_gguf.py` substitutes those selected payloads into the complete
reference model without decode or requantization. It rejects names or shapes
that disagree with the reference schema, copies all unselected tensors
unchanged, writes to a same-directory temporary file, and atomically publishes
only a complete GGUF. The audit reloads the result with pinned `gguf-py`, proves
all four selected SHA-256 hashes byte-for-byte, proves store decoding equals
decoding the bytes in the GGUF, and verifies all fourteen unselected tensors
remain exact. Finally, the external `tools/llama_model_probe.cpp` harness loads
the complete model through the unmodified pinned
`llama_model_load_from_file(..., check_tensors=true)` API on CPU.

The recorded result is `reports/qwen35_tiny_native_gguf.json`: 18 complete
model tensors, four decision groups, eight stored alternatives, two selected
native types, exact expert ordering, and a successful stock llama.cpp model
load. Peak audit RSS is approximately 2.30 GiB. This closes the tiny synthetic
MoE and reference-GGUF byte-preservation gates; it is not evidence for genuine
35B weights, a full 2B candidate block, generation quality, or CUDA memory.

Build the small external loader against the same pinned llama.cpp build and
reproduce the audit with:

```bash
g++ -std=c++17 -O2 tools/llama_model_probe.cpp \
  -I /path/to/llama.cpp/include -I /path/to/llama.cpp/ggml/include \
  -L /path/to/llama-build/bin \
  -Wl,-rpath,/path/to/llama-build/bin -lllama \
  -o /tmp/rco-llama-model-probe

python tools/audit_tiny_native_gguf.py \
  --tokenizer-source /path/to/Qwen3.5-2B-Base \
  --llama-cpp /path/to/pinned/llama.cpp \
  --llama-revision 911f6cdc8ab8a530b2bee09ee61471a6f3178eeb \
  --ggml-library /path/to/llama-build/bin/libggml-base.so \
  --llama-probe /tmp/rco-llama-model-probe \
  --output reports/qwen35_tiny_native_gguf.json
```

## Qwen3.5-2B dense oracle

The primary development checkpoint is pinned to
`Qwen/Qwen3.5-2B-Base` revision
`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`. The single safetensors shard is
4,548,221,488 bytes and has SHA-256
`928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d`.
All twelve downloaded files carry that same Hub revision in their local
metadata, and the independently calculated shard hash matches its published
LFS identity.

`reports/qwen35_2b_identity.json` is the authoritative metadata-only inventory.
It verifies the index against the shard and records every text tensor's name,
category, dtype, shape, element count, logical bytes, and source shard. The
checkpoint contains 632 tensors and 4,548,144,832 logical bytes. The text-only
scope contains 320 tensors and 3,763,655,360 logical bytes; 297 vision and 15
MTP tensors are explicitly omitted. The text configuration has 24 decoder
layers—18 linear-attention and 6 full-attention layers—and a 248,320-token
vocabulary.

Reproduce the identity gate without materializing any tensor payload:

```bash
python tools/audit_model_identity.py \
  --model-dir /path/to/Qwen3.5-2B-Base \
  --repo-id Qwen/Qwen3.5-2B-Base \
  --revision b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
  --weight-sha256 928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d \
  --output reports/qwen35_2b_identity.json
```

This passes the acquisition and immutable-inventory portion of the small-oracle
gate. Canonical GGUF mapping, dense calibration-loss reproduction, and retained
block numerical comparison remain separate milestones.

The canonical mapping portion is recorded in
`reports/qwen35_2b_gguf_manifest.json`. It pins llama.cpp revision
`911f6cdc8ab8a530b2bee09ee61471a6f3178eeb`, maps all 320 text sources through
its QWEN35 `TensorNameMap`, and cross-checks the result against an actual
`convert_hf_to_gguf.py --no-mtp --dry-run`. The converter reports exactly 320
unique final tensors, with no unmapped sources, duplicate destinations, extra
outputs, or missing outputs.

The initial Q2_0/Q4_0 policy creates 187 one-tensor decision groups. The other
133 text tensors are copied: 79 are not matrices, 36 are non-weight state or
bias tensors, and 18 have a row width that is not Q2_0-aligned. Every manifest
entry records source and destination shapes/types plus required converter
semantics such as DeltaNet value-head reordering, A-log transformation,
convolution squeezing, and normalization offsets. The 297 vision and 15 MTP
tensors remain intentionally omitted.

Reproduce the canonical mapping gate with:

```bash
python tools/audit_gguf_mapping.py \
  --identity-report reports/qwen35_2b_identity.json \
  --model-dir /path/to/Qwen3.5-2B-Base \
  --llama-cpp /path/to/pinned/llama.cpp \
  --llama-revision 911f6cdc8ab8a530b2bee09ee61471a6f3178eeb \
  --output reports/qwen35_2b_gguf_manifest.json
```

This establishes canonical names and initial decision groups; it does not yet
prove Q2_0/Q4_0 CPU and CUDA kernel support for every selected geometry or the
byte-level transformed candidate path.

The dense numerical portion of the small-oracle gate is also established on
the Transformers torch fallback for Gated DeltaNet. With the fixed 32-token
sequence recorded in `reports/qwen35_2b_dense_oracle.json`, two consecutive
full-model passes produce an exactly equal causal loss of
`5.655152320861816`. A second fresh process reproduces the same loss and every
tensor digest.

`reports/qwen35_2b_block0_oracle.safetensors` retains the exact BF16 layer-0
input and output, each shaped `[1, 32, 2048]`, plus the input token IDs and
canonical metadata. Its byte-reproducible SHA-256 is
`63bce45d5a6e0e38adeef3ff7a32a48a2e5c772ce93b91dcb92d4fb1af156545`.
The audit ran with Transformers 5.13.1 and a CUDA-12.4 PyTorch 2.6.0 build on
CPU because NVIDIA device nodes disappeared before execution. It peaked at
approximately 4.23 GiB RSS. This is valid dense-loss/block numerical evidence,
but it is not an RTX VRAM gate or a fast-kernel reference.

Reproduce it with a compatible Transformers environment:

```bash
python tools/audit_dense_oracle.py \
  --model-dir /path/to/Qwen3.5-2B-Base \
  --revision b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
  --device cpu \
  --sequence-length 32 \
  --oracle-output reports/qwen35_2b_block0_oracle.safetensors \
  --report-output reports/qwen35_2b_dense_oracle.json
```

## Complete Qwen3.5-2B native candidate block

Block 0 now has a complete canonical native-candidate database. The
transform-aware row source in `src/qwen35_native.py` reads only the safetensors
row runs required for the next output chunk and implements the pinned
converter's grouped-to-tiled value-head order for QKV, gate, alpha/beta, and
output projections. Its unequal-key/value-head unit test exercises the
nontrivial permutation even though the 2B checkpoint's equal head counts make
that transform an identity for this particular model.

`reports/qwen35_2b_block0_native_candidates.json` accounts for all 14 canonical
block tensors: eight searched matrices and six explicit copy tensors. Each
searched matrix has exact Q2_0 and Q4_0 alternatives, for 16 candidates and
49,600,512 payload bytes. The retained local store is
`data/qwen35_2b_block0_native` at the workspace root and is intentionally not
versioned.

The audit independently creates a block-only safetensors checkpoint and runs
the pinned llama.cpp converter over it in F32. For every decision group, the
streamed transformed rows equal that reference tensor exactly, and chunked
native quantization produces the same bytes as a single native quantization of
the converter output. Every payload is then checksum-verified and decoded into
a caller-owned buffer with recorded maximum, mean, RMSE, and relative
Frobenius errors. The streaming generator uses 16-row chunks and never exceeds
393,216 dense chunk bytes. The audit's approximately 1.69 GiB peak RSS includes
the intentionally non-streaming full-block reference construction and must not
be attributed to candidate generation or counted as a CUDA gate.

Reproduce this milestone into a new, nonexistent store directory with:

```bash
python tools/audit_qwen35_2b_native_block.py \
  --model-dir /path/to/Qwen3.5-2B-Base \
  --manifest reports/qwen35_2b_gguf_manifest.json \
  --identity reports/qwen35_2b_identity.json \
  --llama-cpp /path/to/pinned/llama.cpp \
  --ggml-library /path/to/llama-build/bin/libggml-base.so \
  --store-output /path/to/new/qwen35_2b_block0_native \
  --temporary-parent /path/to/nvme/staging \
  --rows-per-chunk 16 \
  --output reports/qwen35_2b_block0_native_candidates.json
```

This proves complete block candidate generation, transformation fidelity, and
bounded source reads. The next numerical gate must install selected candidates
into the retained block-0 oracle and measure block-output error. CPU/CUDA
matmul-kernel coverage, a genuine routed 35B block, and full-model generation
remain separate requirements.

That 2B numerical block gate is now recorded in
`reports/qwen35_2b_block0_native_output.json`. A meta-instantiated Transformers
block loads exactly 117,629,248 BF16 checkpoint bytes, and its direct dense
forward reproduces the retained block-0 oracle bit-for-bit. Native candidate
evaluation decodes every selected canonical GGUF payload through pinned GGML,
undoes value-head ordering back into the Hugging Face layout, installs all
eight decision groups, and retains the six copy-only tensors from the dense
block.

Three complete assignments run twice with bit-identical, finite BF16 outputs:

- uniform Q4_0: relative block-output Frobenius error `0.15615076165037167`;
- uniform Q2_0: relative error `1.2237737652528997`; and
- alternating Q2_0/Q4_0: relative error `0.8788811960769464`.

The report records maximum, mean, RMSE, signed-mean, relative-Frobenius, and
cosine metrics plus exact payload/aligned costs and every installed candidate
hash. Decoding has no persistent cache; the largest decoded and inverse-
transformed FP32 buffers are 50,331,648 bytes each, and the largest BF16 install
buffer is 25,165,824 bytes. The complete CPU audit peaks at 763,424,768 bytes
RSS and releases the block back to `meta` afterward.

Reproduce it with:

```bash
python tools/audit_qwen35_2b_native_block_output.py \
  --model-dir /path/to/Qwen3.5-2B-Base \
  --oracle reports/qwen35_2b_block0_oracle.safetensors \
  --manifest reports/qwen35_2b_gguf_manifest.json \
  --identity reports/qwen35_2b_identity.json \
  --store /path/to/qwen35_2b_block0_native \
  --ggml-library /path/to/llama-build/bin/libggml-base.so \
  --device cpu \
  --output reports/qwen35_2b_block0_native_output.json
```

This completes the dense 2B candidate-block numerical oracle. It does not
establish an acceptable Q2 quality threshold, end-to-end loss, CUDA kernels,
or the genuine routed-expert 35B gate.

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
`[batch, tokens, vocabulary]` probabilities. Schema 2 also stores one FP16
retained-probability value per token, adding two bytes per position so the
omitted teacher mass is measurable. `--kl-topk 0` retains the original
full-vocabulary behavior.

The compact implementation reproduces the previous top-k objective on a test
model and retains a nonzero gradient through the candidate model. It preserves
the existing top-k objective semantics; it is not equivalent to full-vocabulary
KL. Cache construction logs mean/minimum retained mass and mean/maximum omitted
mass. Search JSON records the same statistics over loss-selected token
positions. Quantization cache filenames include `massv2`, preventing older
compact caches without mass measurements from being silently reused.

## Hard assignment search

`rco_search_quant.py` provides `hard-spsa` and `hard-reinforce` modes that avoid
persistent BF16 candidate deltas. The current hard path supports exactly two
bitwidths and equal-size groups. That matches routed-expert groups when every
group contains one expert's gate, up, and down projections:

```text
--search-mode hard-spsa --moe-per-expert --bitwidths 2,4
```

Each SPSA step evaluates a plus/minus assignment pair on the same calibration
batch. Every assignment selects an exact number of high-bit groups, and the
driver rejects targets that the group count cannot represent exactly.

The REINFORCE alternative samples an ordered Plackett-Luce draw with Gumbel
top-k, so every stochastic assignment also has the exact required number of
high-bit groups. It evaluates antithetic `u` and `1-u` samples on the same
batch, uses an exponential moving loss baseline, and backpropagates only
through the small vector of group scores. Select it with:

```text
--search-mode hard-reinforce --reinforce-baseline-decay 0.9
```

Neither estimator has yet been compared on the real 35B calibration loss.
Record convergence and loss variance for both with the same seed and batches
before choosing a default for a production run.

For the inference-only cross-entropy bring-up, the full-model RAM floor is
removed with `--stream-hard-eval`:

```bash
python rco_search_quant.py \
  --model /path/to/dense-qwen3.6-35b-a3b \
  --layer-dir /path/to/rco-database \
  --bitwidths 2,4 \
  --target-avg-bits 3.0 \
  --search-mode hard-spsa \
  --stream-hard-eval \
  --objective ce \
  --moe-per-expert \
  --calibration-data fineweb_edu \
  --calibration-samples 8 \
  --calibration-seq-length 256 \
  --batch-size 1 \
  --n-steps 1 \
  --vocab-chunk-size 8192 \
  --weightstore-cache off \
  --save-json
```

The dense checkpoint remains the source for embeddings, attention, router,
normalization, and output weights. The candidate database supplies only the
selected searched weights. One dense decoder block and one decoded candidate
are active at a time; the output head is resident while exact CE streams its
vocabulary rows. Each run records the maximum materialized block and candidate,
CUDA allocated/reserved peaks, process peak RSS, and logical checkpoint and
candidate bytes read per evaluation. It also records wall time and separate
CUDA allocated/reserved high-water marks for checkpoint loading, candidate
decode/installation, block forward, chunked loss, and checkpoint release. CUDA
timings synchronize at phase boundaries for measurement accuracy; this adds
profiling overhead and should be considered when interpreting throughput.

This mode is intentionally inference-only. Both hard estimators obtain
assignment updates from paired scalar losses, so neither needs an activation or
weight backward graph. Pruning masks and relaxed assignment methods still
require a real training-aware streamed backward path.

## Bounded numerical candidate validation

`tools/validate_candidates.py` compares a selected assignment with its dense
safetensors source while keeping only one source tensor and one decoded
candidate resident. For fused Qwen experts it uses safetensors slicing to read
only the requested expert's gate, up, or down projection rather than the full
expert bank. Zero-bit assignments are compared with an implicit zero tensor.

```bash
python tools/validate_candidates.py \
  --source-model /path/to/dense-qwen3.6-35b-a3b \
  --layer-dir /path/to/rco-database \
  --assignment /path/to/search-result.json \
  --chunk-elements 1048576 \
  --output candidate-validation.json
```

The bounded report includes aggregate maximum absolute error, mean absolute
error, mean signed error, RMSE, relative Frobenius error, realized average
selected bits, effective candidate-file bits per weight, logical bytes read,
peak process RSS, elapsed time, and a configurable number of worst tensors.
Error reduction uses bounded chunks, and the default report does not retain all
per-tensor records. `--details-jsonl` explicitly enables exhaustive records;
`--tensor` and `--max-tensors` support small bring-up runs before a complete
scan. Candidate checksum verification remains active through `WeightStore`.

This tool requires the original dense or otherwise trusted higher-precision
checkpoint. The published two-bit GSQ checkpoint is not a substitute for that
reference and cannot validate generated higher-bit candidates.

## Packed output compatibility gate

The released `run_build_checkpoint.py --format compressed-tensors` is not a
bounded-memory writer: it loads or decompresses the complete base model, applies
quantization wrappers in memory, and then calls `save_pretrained`. The version
pair originally pinned for that path also does not support this Qwen schema.

With the former Transformers 5.7.0 and `compressed-tensors` 0.15.0.1 pins, the
published GSQ checkpoint fails during model preprocessing, before tensor
payloads are loaded:

```text
ValueError: Quantization of module type Qwen3_5MoeGatedDeltaNet is not supported
```

Stable `compressed-tensors` 0.18.0 adds arbitrary-module configuration support
and successfully applies the published quantization config to a meta skeleton.
That change alone did not establish checkpoint compatibility under pinned
Transformers 5.7.0: the skeleton retains fused expert state keys
`experts.gate_up_proj` and `experts.down_proj`, whereas the published packed
checkpoint contains per-expert `gate_proj`, `up_proj`, and `down_proj` tensors.

An isolated Transformers 5.13.1 plus `compressed-tensors` 0.18.0 environment
passes the same gate. The normal `AutoModelForImageTextToText.from_pretrained`
path loads the exact 93,625-tensor published checkpoint onto `meta` as
`Qwen3_5MoeForConditionalGeneration` with `CompressedTensorsHfQuantizer`.
Loading information contains zero missing, unexpected, mismatched, or error
entries; all 33,683,169,638 resulting parameters remain on `meta`, resident
parameter bytes are zero, and the audit peaks at 894,894,080 bytes RSS. This
proves the loader's logical packed-expert to fused-runtime schema mapping for
that version pair without patching third-party code.

`tools/audit_compressed_compatibility.py` reproduces the clean meta-load check,
and the version history is recorded in
`reports/qwen35_compressed_tensors_compatibility.json`. The input/runtime schema
gate is now passed for the newer pair. The bounded writer should target the
published logical packed layout, first on a tiny synthetic checkpoint, and must
pass this clean meta-load round trip before repository pins change or a full
materialized load is attempted.

The candidate-to-output mapping is also proven for symmetric candidates with
activation ordering disabled, which is the initial low-memory configuration.
RCO stores unsigned codes with zero point `2^(bits-1)` and reconstructs
`scale * (code - zero)`. Compressed-tensors interprets signed codes, adds the
same offset before packing, and reconstructs `scale * signed_code`. Codes and
scales therefore transfer without requantization. The only storage transform
restarts the little-endian bitstream at each row's INT32 boundary:

```text
RCO qweight + symmetric zero  ->  compressed-tensors weight_packed (INT32)
RCO scales                    ->  compressed-tensors weight_scale
RCO logical shape             ->  compressed-tensors weight_shape (INT64)
```

`src/quant/compressed_layout.py` implements this mapping and rejects asymmetric
zero points or activation-order permutations rather than guessing a target
layout. `tools/audit_compressed_packing.py` compares its output byte-for-byte
with compressed-tensors 0.18.0. All 48 combinations spanning bit widths 1–8
and aligned/unaligned row lengths match; the evidence is in
`reports/qwen35_packing_compatibility.json`.

The tiny writer gate now passes as well. `tools/audit_tiny_packed_roundtrip.py`
constructs a one-layer, two-expert `Qwen3_5MoeForCausalLM`, maps all six
logical expert projections from synthetic RCO bundles through the same
`compressed_state_from_bundle` path, and writes a deliberately small indexed
checkpoint. The 51,025-byte artifact has three safetensors shards, 34 indexed
tensors, and six Q2 `weight_packed` tensors. It exists only in a temporary
directory and is deleted when the audit finishes.

Under Transformers 5.13.1 and `compressed-tensors` 0.18.0, both supported load
paths succeed with no missing, unexpected, mismatched, or error entries. The
meta load leaves every parameter on `meta`. A materialized CPU load dequantizes
and merges the logical `gate_proj` and `up_proj` tensors into runtime
`gate_up_proj` and stacks `down_proj` across experts. Both fused tensors equal
the values reconstructed directly from the source RCO codes, zero points, and
scales with maximum absolute error `0.0`. The complete compact evidence is in
`reports/qwen35_tiny_packed_roundtrip.json`; peak process RSS was 325,640,192
bytes and the retained report is 2.4 KiB.

This proves the logical packed-expert schema, index, sharding, and loader
conversion needed by a bounded writer. It does not prove the old full-model
`run_build_checkpoint.py` path is memory-safe, nor does it validate a full
mixed-bit assignment. The repository now pins the verified Transformers
5.13.1 and `compressed-tensors` 0.18.0 pair.

### Mixed routed-expert output incompatibility

The production writer cannot yet preserve an RCO assignment that selects
different packed widths for different routed experts. Qwen exposes its routed
experts as two fused runtime parameters rather than individual `nn.Linear`
modules. Transformers 5.13.1 therefore runs `DecompressExperts` before stacking
the logical checkpoint tensors, and that conversion takes the first
compressed-tensors config group as the quantization scheme for every expert.

`tools/audit_mixed_expert_compatibility.py` proves the consequence with the
same one-layer, two-expert model used by the positive writer audit. Expert 0 is
packed at Q2 and expert 1 at Q4. It tests Q2-first and Q4-first config order,
each through both meta and materialized CPU loading. All four loads fail during
expert conversion. With Q2 selected globally, the Q4 packed rows decode at
twice the expected logical width (`32` versus `64` in the gate/up probe). With
Q4 selected globally, the Q2 rows decode at half width (`16` versus `32`). The
51,842-byte checkpoint is automatically deleted; the 4.3 KiB evidence report
is `reports/qwen35_mixed_expert_compatibility.json`.

The current upstream
[Transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/compressed_tensors.py)
improves selection between an expert-wide scheme and schemes for other modules,
but still resolves one scheme for the complete expert collection. The current
[vLLM compressed-tensors integration](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py)
similarly selects one quantization method for a fused `RoutedExperts` layer.
The compressed-tensors format can describe non-uniform module schemes in
general, but these fused-MoE integration paths do not expose an independently
selectable module per expert.

The apparent alternatives do not satisfy the project objective:

- constraining every routed expert to one width removes the per-expert mixed
  allocation that RCO is meant to optimize;
- storing Q2 choices in Q4 containers preserves values but forfeits their
  storage and memory savings and misstates the selected-bit budget;
- emitting dense BF16 fake-quant experts gives up the required model-memory
  reduction;
- converting the GPTQ-style RCO candidates to GGUF block types requantizes the
  weights and no longer materializes the searched candidates; and
- patching Transformers, adding custom model code, or designing a new runtime
  format is a substantial compatibility workaround.

`require_uniform_routed_expert_bits` now provides a fail-fast preflight for any
packed writer built on the currently verified loader. Production writer work
is paused at this boundary rather than emitting a checkpoint that cannot load
or silently expanding lower-bit choices. A supported per-expert scheme lookup
and execution path, or an explicitly accepted change to the output objective,
is required before that milestone can continue.

## Remaining work

The earlier Transformers 5.7.0 meta initialization was checked against the
local Qwen3.6 config. It creates `Qwen3_5MoeForConditionalGeneration`, resolves
40 text layers with a 30 linear-attention / 10 full-attention split, keeps all
35,107,181,936 parameters on meta, and retained only 164 bytes of runtime
buffers during the audit. The process peaked at about 341 MiB RSS.

`tools/audit_streaming_schema.py` performs this check and compares every
required text prefix against the safetensors index without reading payloads.
The recorded local GSQ audit is `reports/qwen35_streaming_schema.json`. As
expected, that already-compressed checkpoint is not a valid dense streaming
source: only the embedding and final-norm prefixes match the dense model
schema, while packed/scale/shape tensors replace ordinary dense weights. The
driver now rejects such a mismatch before starting an optimization pass.

1. Validate one complete streamed block and then all blocks against a dense
   base checkpoint using the pinned Qwen3.6 Transformers implementation.
2. Add a training-aware streamed backward path for pruning masks if full-model
   pruning remains in scope.
3. Resolve the fused-Qwen per-expert mixed-bit runtime incompatibility, then
   implement the production writer that streams copied and selected tensors
   directly into bounded-size output shards.
4. Validate one complete optimization step, then a multi-step run, while
   recording CUDA allocated/reserved peaks, process RSS, disk growth, and I/O.
5. Load the materialized checkpoint and run numerical layer comparisons and a
   short generation test.

The standard test suite uses only tiny synthetic tensors and produces no model
artifacts:

```bash
python -m unittest discover -s tests -v
```
