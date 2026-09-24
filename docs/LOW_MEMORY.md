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
bounded source reads. Candidate installation, runtime execution, the genuine
routed 35B block, and CUDA remain separate evidence below.

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

The CPU runtime half of the 2B gate is retained in
`reports/qwen35_2b_block0_native_gguf_runtime.json`. Pinned llama.cpp converts
the immutable dense checkpoint to a 3,775,708,864-byte BF16 text GGUF with
SHA-256
`caa3c359d2dbf6f63639b98a5018d469358b88b23982ae74a85378b1d585be95`.
The generalized native-GGUF audit then writes two complete temporary models:
one selects Q2_0 for all eight eligible block-0 tensors and the other selects
Q4_0 for all eight. It verifies the 16,533,504 and 33,067,008 selected payload
bytes directly against the store after serialization.

Both 320-tensor outputs pass the strict pinned model loader and complete stock
llama.cpp CPU generation. Their hashes are
`74a6adf1e4d66cc2dfd275a6ca72c1b81d75b7d090fcc60993e53b02631a63e6`
and `7215ea4779cb0077eea984d25ebbebaff189be8a8c3573ae4608ca237d7766de`.
The Q4_0 run generates `Hello.`; the deliberately low-quality uniform-Q2_0
assignment generates different text but completes inference successfully.
This is kernel/format evidence, not an assertion that uniform Q2_0 meets a
quality threshold. The two-run audit takes 85.49 seconds and peaks at
4,640,272,384 bytes RSS.

Use the same audit tool with explicit small-model geometry:

```bash
python tools/audit_qwen36_35b_native_gguf_runtime.py \
  --reference /path/to/Qwen3.5-2B-Base-BF16-text.gguf \
  --reference-sha256 \
    caa3c359d2dbf6f63639b98a5018d469358b88b23982ae74a85378b1d585be95 \
  --store /path/to/qwen35_2b_block0_native \
  --llama-cpp /path/to/pinned/llama.cpp \
  --ggml-library /path/to/cpu/libggml-base.so \
  --llama-probe /path/to/llama-model-probe \
  --llama-executable /path/to/llama \
  --temporary-parent /path/to/nvme/staging \
  --output reports/qwen35_2b_block0_native_gguf_runtime.json \
  --expected-candidate-tensors 8 \
  --expected-layers 24 \
  --expected-embedding 2048 \
  --scope-label 'complete Qwen3.5-2B-Base block 0'
```

Together with the genuine 35B runtime audit below, this completes CPU execution
coverage for the initial Q2_0/Q4_0 candidate set. CUDA coverage remains open.

## Qwen3.6-35B-A3B dense source and canonical relationship

The production BF16 source is now pinned to `Qwen/Qwen3.6-35B-A3B` revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`. The complete local snapshot has
26 safetensors shards and 40 top-level files. The identity audit hashes every
file, verifies all 27 SHA-256 Hub identities (the 26 Xet/LFS shards plus one
large tokenizer asset), checks the shard index against the physical tensors,
and validates the declared logical size exactly.

`reports/qwen36_35b_base_identity.json` records 1,045 source tensors and
71,903,645,408 logical bytes. The text-only scope has 693 tensors and
69,321,221,376 logical bytes; 333 vision and 19 MTP tensors are explicitly
omitted. The inventory includes all 40 layers, with 80 fused routed-expert
source tensors, 160 shared-expert tensors, 40 routers, and the complete hybrid
attention stack. Reproduce the fail-closed identity audit with:

```bash
python tools/audit_model_identity.py \
  --model-dir /path/to/Qwen3.6-35B-A3B \
  --repo-id Qwen/Qwen3.6-35B-A3B \
  --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output reports/qwen36_35b_base_identity.json
```

The official dense MoE checkpoint stores one fused `gate_up_proj` tensor and
one suffixless fused `down_proj` tensor per layer. The canonical mapper models
each gate/up half as an explicit source view and normalizes the down projection
to the `.weight` name expected by pinned llama.cpp. It then cross-checks every
result against an actual BF16 `convert_hf_to_gguf.py --no-mtp --dry-run` at
llama.cpp revision `911f6cdc8ab8a530b2bee09ee61471a6f3178eeb`.

`reports/qwen36_35b_base_gguf_manifest.json` proves that all 693 text sources
map to exactly 733 unique canonical QWEN35MOE tensors. The 40 fused gate/up
sources expand to 80 canonical source views. The initial native Q2_0/Q4_0
policy defines 512 one-tensor decision groups and 221 copy tensors, with no
unmapped source, extra converter output, missing output, or duplicate
destination. Reproduce it with:

```bash
python tools/audit_gguf_mapping.py \
  --identity-report reports/qwen36_35b_base_identity.json \
  --model-dir /path/to/Qwen3.6-35B-A3B \
  --llama-cpp /path/to/pinned/llama.cpp \
  --llama-revision 911f6cdc8ab8a530b2bee09ee61471a6f3178eeb \
  --output reports/qwen36_35b_base_gguf_manifest.json
```

Finally, `reports/qwen36_35b_base_gsq_relationship.json` links this exact
dense source to the published GSQ release without claiming that their weight
values are equal. It verifies the GSQ model card's `base_model` and
`base_model_relation: quantized` declarations, exact normalized model-config
equality, and exact hashes for seven shared text/tokenizer assets. It compares
all 733 dense canonical destination names and shapes with the actual proven
GSQ hybrid GGUF and validates its 40-layer, 256-expert, 8-active-expert
metadata. The comparison has zero missing, extra, or shape-mismatched tensors;
the hybrid GGUF SHA-256 is
`8e50912f5d0703401ca21b09deb2ee8a84d03536ade6fb44815fcb40a0b03596`.

```bash
python tools/audit_qwen36_base_gsq_relationship.py \
  --base-identity reports/qwen36_35b_base_identity.json \
  --base-manifest reports/qwen36_35b_base_gguf_manifest.json \
  --base-dir /path/to/Qwen3.6-35B-A3B \
  --gsq-dir /path/to/Qwen3.6-35B-A3B-GSQ \
  --gsq-audit /path/to/results/full_model_metrics.json \
  --gsq-gguf /path/to/results/Qwen3.6-35B-A3B-GSQ-hybrid.gguf \
  --llama-cpp /path/to/pinned/llama.cpp \
  --output reports/qwen36_35b_base_gsq_relationship.json
```

This completes the production-source identity, full canonical mapping,
decision-group definition, and BF16/GSQ relationship portions of the Phase 1
gate. It does not yet establish a genuine 35B block numerical oracle, native
candidate bytes for every routed-expert geometry, CUDA execution, or the
full-model output gate. BF16 remains the authoritative source for new native
higher-precision candidates; the GSQ checkpoint is only a quantized lineage
and low-bit provenance source.

The final Phase 1 numerical rung is now retained separately in
`reports/qwen36_35b_block0_oracle.safetensors` and
`reports/qwen36_35b_dense_block_oracle.json`. The audit instantiates the
complete 40-layer, 256-expert model on `meta`, materializes the real embedding
and block 0 sequentially, and releases each prefix back to `meta`. Embedding
and block weights never coexist as resident streamed prefixes.

The exact dense embedding payload is 1,017,118,720 bytes. The genuine routed
block contains 18 source tensors and 1,685,401,984 BF16 bytes. A fixed 16-token
sequence produces BF16 input and output tensors shaped `[1, 16, 2048]`; two
passes are bit-identical and finite. A second fresh process reproduced the
complete 131,797-byte oracle byte-for-byte at SHA-256
`cdfa89b09c5eb6001ec0cedcfaf7a49c1a2798cc0cee0b6178ef58c77bf12f5e`.
The retained output tensor hash is
`b7c2467f0bd2c3abb8c3fabf69dad2dcf0ef20885baaea80cd1bf9eecc006485`.
The CPU run used the Transformers torch fallback, peaked at 1,267,003,392
bytes RSS, and released the complete block afterward.

```bash
python tools/audit_qwen36_35b_dense_block.py \
  --model-dir /path/to/Qwen3.6-35B-A3B \
  --identity reports/qwen36_35b_base_identity.json \
  --device cpu \
  --sequence-length 16 \
  --expected-oracle-sha256 \
    cdfa89b09c5eb6001ec0cedcfaf7a49c1a2798cc0cee0b6178ef58c77bf12f5e \
  --oracle-output reports/qwen36_35b_block0_oracle.safetensors \
  --report-output reports/qwen36_35b_dense_block_oracle.json
```

This closes the Phase 1 one-block dense path for the genuine target. It is not
candidate-generation evidence, and the CPU fallback does not pass any CUDA
kernel or VRAM gate.

## Complete Qwen3.6-35B native candidate block

The BF16-derived native candidate rung now covers every eligible tensor in the
genuine first routed-expert block. `SafetensorGGUFRowSource` streams official
three-dimensional fused expert tensors in canonical expert-major row order,
including explicit gate/up views into `gate_up_proj` and the complete
suffixless `down_proj`. Synthetic tests independently verify gate, up, and
down values and ordering without materializing an expert stack.

`reports/qwen36_35b_block0_native_candidates.json` accounts for all 19
canonical block tensors: 13 searched tensors and six explicit copy tensors.
Each searched tensor has exact Q2_0 and Q4_0 alternatives, yielding 26
candidates and 710,997,696 payload bytes. The retained local store is
`data/qwen36_35b_block0_native` at the workspace root and is intentionally not
versioned.

Generation uses 16-row chunks with a maximum 262,144-byte dense FP32 chunk.
Every published payload is then checksum-verified, read independently in
bounded row chunks, compared byte-for-byte with a fresh native quantization,
decoded through pinned GGML, and evaluated against its BF16 source with full
maximum, mean, signed-mean, RMSE, and relative-Frobenius metrics. Validation
scratch never exceeds 2,097,152 bytes. The complete audit reads the two exact
source shards recorded in the report, finishes below 895 MiB peak RSS, and
publishes its index atomically only after all 26 validations pass.

The audit also inventories all 512 model-wide decision groups. Their eleven
distinct GGUF shapes use only three row widths (512, 2048, and 4096); pinned
GGML CPU quantize/dequantize round trips pass for Q2_0 and Q4_0 at every width,
and exact payload costs are recorded for every complete shape. This is codec
and storage coverage, not a matrix-multiplication or CUDA-kernel claim. CUDA
remains explicitly `not_run_cuda_initialization_failed` in the report.

```bash
python tools/audit_qwen36_35b_native_block.py \
  --model-dir /path/to/Qwen3.6-35B-A3B \
  --identity reports/qwen36_35b_base_identity.json \
  --manifest reports/qwen36_35b_base_gguf_manifest.json \
  --llama-cpp /path/to/pinned/llama.cpp \
  --ggml-library /path/to/llama-build/bin/libggml-base.so \
  --store-output /path/to/new/qwen36_35b_block0_native \
  --temporary-parent /path/to/nvme/staging \
  --rows-per-chunk 16 \
  --validation-rows-per-chunk 16 \
  --output reports/qwen36_35b_block0_native_candidates.json
```

This passes the BF16-derived, genuine-block candidate generation and bounded
CPU codec gates. Installing complete assignments into the retained dense
oracle, importing the authentic GSQ-derived low-bit candidate, exercising
llama.cpp matrix multiplication, and CUDA validation remain separate work.

Complete candidate installation and numerical comparison now pass as recorded
in `reports/qwen36_35b_block0_native_output.json`. The native store exposes a
checksummed decoded-row iterator, so no complete FP32 candidate is allocated.
Ordinary value-head permutations are restored directly into their target rows
or columns, while fused gate/up/down chunks overwrite only their canonical
expert/source views. The largest decoded FP32 chunk is 262,144 bytes and the
largest BF16 install buffer is 131,072 bytes.

Before each assignment, the audit reloads the exact 1,685,401,984-byte dense
block, retains all six copy-only tensors, and overwrites all 13 decision
groups. The direct dense block first reproduces the retained oracle exactly.
Three complete assignments then run twice with bit-identical finite outputs,
and a second process reproduces all three output hashes:

- uniform Q4_0: 473,998,464 payload bytes and relative block-output Frobenius
  error `0.12647789524236974`;
- uniform Q2_0: 236,999,232 payload bytes and relative error
  `0.5741061967995075`; and
- alternating Q2_0/Q4_0: 390,961,728 payload bytes and relative error
  `0.5045100533778869`.

The report retains exact assignments, payload hashes and costs, output hashes,
maximum/mean/signed-mean/RMSE/relative-Frobenius/cosine metrics, prefix schema,
and release evidence. The CPU audit peaks at 2,359,668,736 bytes RSS and
releases the complete block back to `meta`.

```bash
python tools/audit_qwen36_35b_native_block_output.py \
  --model-dir /path/to/Qwen3.6-35B-A3B \
  --identity reports/qwen36_35b_base_identity.json \
  --manifest reports/qwen36_35b_base_gguf_manifest.json \
  --oracle reports/qwen36_35b_block0_oracle.safetensors \
  --store /path/to/qwen36_35b_block0_native \
  --ggml-library /path/to/llama-build/bin/libggml-base.so \
  --device cpu \
  --rows-per-chunk 16 \
  --expected-output-sha256 \
    uniform_q2_0=48fdf3a0b904ccf392198ed93773ea10d46baae163dd84264d2ab9be722c5f24 \
  --expected-output-sha256 \
    uniform_q4_0=97f5229bafcdb1c46dad2755afc98ab535400332885d4852e4426c57066884f6 \
  --expected-output-sha256 \
    alternating_q2_0_q4_0=c28824f4dae3d53eb3e751d0a4a48856787161ca52ea6d38fea2fc19dda3cbfb \
  --output reports/qwen36_35b_block0_native_output.json
```

This completes the BF16-derived genuine-block numerical rung. It does not set
an acceptable Q2 quality threshold, import the authentic GSQ candidate,
exercise llama.cpp matrix multiplication, or pass CUDA/VRAM gates.

## Authentic GSQ-derived routed Q2_0 candidates

The production block policy now has a separate direct-import path for the
published low-bit source. `src/gsq_q2.py` reads one logical expert triplet at a
time, unpacks the exact two-bit integer lanes, and emits stock Q2_0 blocks via
`q_q2 = 3 - q_gsq` and `d = FP16(-scale)`, duplicating each group-128 scale for
the two group-64 Q2_0 blocks. It never invokes a floating-point weight
quantizer.

A GGUF tensor can expose only one Q2_0 payload. The production-policy store at
`data/qwen36_35b_block0_native_gsq` therefore uses authentic GSQ-derived Q2_0
for the three routed projection tensors, retains BF16-derived Q2_0 for the ten
other searched tensors, and retains BF16-derived Q4_0 for all thirteen. The
original all-BF16-derived store remains intact as comparison evidence.

`reports/qwen36_35b_block0_gsq_import.json` audits all 768 layer-0 expert
projections and all 6,291,456 source scale groups. Every stock Q2_0 decode is
bit-exact against the mapped GSQ value. The observed maximum error against the
published source is exactly the accepted
`5.960464477539063e-8` bound. All three aggregated routed payloads are also
byte-for-byte identical to the corresponding tensors in the proven
`Qwen3.6-35B-A3B-GSQ-hybrid.gguf`; their hashes are retained individually.
Every non-imported candidate is an exact hash-preserving copy from the
BF16-derived store. The resulting 26-candidate store remains 710,997,696
payload bytes and the import audit peaks at 1,504,808,960 bytes RSS.

```bash
python tools/audit_qwen36_35b_gsq_block_import.py \
  --gsq-dir /path/to/Qwen3.6-35B-A3B-GSQ \
  --gsq-audit /path/to/results/full_model_metrics.json \
  --acceptance /path/to/results/scale_error_acceptance.json \
  --manifest reports/qwen36_35b_base_gguf_manifest.json \
  --relationship reports/qwen36_35b_base_gsq_relationship.json \
  --base-store /path/to/qwen36_35b_block0_native \
  --store-output /path/to/new/qwen36_35b_block0_native_gsq \
  --gsq-gguf /path/to/results/Qwen3.6-35B-A3B-GSQ-hybrid.gguf \
  --llama-cpp /path/to/pinned/llama.cpp \
  --ggml-library /path/to/llama-build/bin/libggml-base.so \
  --temporary-parent /path/to/nvme/staging \
  --output reports/qwen36_35b_block0_gsq_import.json
```

`reports/qwen36_35b_block0_gsq_native_output.json` repeats complete block
evaluation with that production-policy store. Uniform authentic/derived Q2_0
has relative output error `0.5607835252206417`, the alternating assignment has
error `0.5033222782579743`, and unchanged uniform Q4_0 retains
`0.12647789524236974`. All three output hashes reproduce across fresh
processes; peak CPU RSS is 2,370,060,288 bytes. This closes the authentic GSQ
import and genuine-block numerical portions of the native-candidate gate. It
does not prove llama.cpp matrix multiplication, CUDA execution, or acceptable
end-to-end model quality.

## Genuine-block native GGUF CPU runtime

`reports/qwen36_35b_block0_native_gguf_runtime.json` closes the CPU execution
part of that remaining kernel gate. The audit starts from the immutable proven
hybrid GGUF, writes one complete temporary model with all 13 eligible layer-0
tensors selected as Q2_0, and repeats with all 13 selected as Q4_0. It copies
candidate payloads directly from the production-policy store through
`write_selected_native_gguf`; neither run decodes or requantizes a selected
payload while writing.

After each write, the pinned GGUF reader verifies every selected type, shape,
byte count, alignment, and SHA-256 against the store. The resulting 733-tensor
models then pass the strict external llama model loader and generate `Hello`
through the unmodified pinned `llama cli` CPU runtime. The Q2_0 and Q4_0 runs
preserve 236,999,232 and 473,998,464 selected payload bytes respectively. Their
complete temporary-model hashes are
`3211ba512c5fdfc2fcf3a6c7de373e26bea0210a1c469d071ffed03a61a6e6fb`
and `d2934f1012be289b492c9123257744aa6287ec259b91aa27829ce24436c078ea`.

```bash
python tools/audit_qwen36_35b_native_gguf_runtime.py \
  --reference /path/to/Qwen3.6-35B-A3B-GSQ-hybrid.gguf \
  --reference-sha256 \
    8e50912f5d0703401ca21b09deb2ee8a84d03536ade6fb44815fcb40a0b03596 \
  --store /path/to/qwen36_35b_block0_native_gsq \
  --llama-cpp /path/to/pinned/llama.cpp \
  --ggml-library /path/to/cpu/libggml-base.so \
  --llama-probe /path/to/llama-model-probe \
  --llama-executable /path/to/llama \
  --temporary-parent /path/to/nvme/staging \
  --output reports/qwen36_35b_block0_native_gguf_runtime.json
```

The audit took 424.59 seconds. Peak process and child RSS were
12,930,052,096 and 13,118,967,808 bytes because the current correctness-first
GGUF writer and validators memory-map and touch the complete reference/output.
That stays within the available host RAM but is not the Phase 6 streaming
writer result; bounded-copy output construction remains open. Both CPU runs
also recorded the current system-wide CUDA initialization failure, so this
report makes no CUDA or RTX memory-gate claim.

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

The native-GGUF path cannot use equal group counts as a proxy for its budget:
tensor sizes differ, and Q2_0/Q4_0 costs include their actual block scales and
alignment. `search.hard.exact_cost_assignment` therefore solves a sparse exact
multiple-choice knapsack over per-candidate serialized byte costs.
`sample_exact_cost_assignment` uses a suffix log-partition dynamic program to
sample only exactly feasible assignments and returns their differentiable log
probabilities. `optimize_cost_spsa` and `optimize_cost_reinforce` expose these
operations to paired hard search. Every evaluation and final assignment is
checked against the exact integer-byte target; an unreachable target fails
before model evaluation. The older CLI path above remains the equal-size
legacy interface until the native candidate store is wired into the full-model
driver.

`reports/qwen36_35b_block0_hard_search.json` supplies the controlled genuine-
model comparison. It streams four independently tokenized 16-token calibration
texts through the real dense embedding and block 0; the first batch reproduces
the retained oracle bit-for-bit. At an exact 315,169,344-byte selected-candidate
target, there are 36 feasible assignments. The audit evaluates all 36, records
their complete block normalized-MSE landscape, and reproduces the global
optimum after traversing every other assignment. Its optimum has normalized
MSE `0.11889813207786598` (relative Frobenius error
`0.344816084424532`) and output SHA-256
`a3de42a5c55d127f74d4fe2a594ceefa5108e1b5ada2a3a6109b68d24fe6dbfa`.

SPSA and antithetic REINFORCE then run for 100 steps over matched seeds 0–19
against that identical cached real-model landscape. Every one of their 8,000
scalar evaluations (4,000 pairs) meets the exact integer-byte target. SPSA
reaches the
exhaustive optimum in 7/20 runs; REINFORCE reaches it in 19/20. Their returned
best-incumbent median losses are `0.1196484408321559` and
`0.11889813207786598`, and population variances are
`9.149102632887156e-7` and `2.0791051533637335e-8`. REINFORCE explores a wider
raw loss distribution, but converges to markedly better and less variable
incumbents. It is therefore the default estimator for the next production
trials; this block result is not an end-to-end quality claim.

```bash
python tools/audit_qwen36_35b_hard_search.py \
  --model-dir /path/to/Qwen3.6-35B-A3B \
  --identity reports/qwen36_35b_base_identity.json \
  --manifest reports/qwen36_35b_base_gguf_manifest.json \
  --oracle reports/qwen36_35b_block0_oracle.safetensors \
  --store /path/to/qwen36_35b_block0_native_gsq \
  --ggml-library /path/to/cpu/libggml-base.so \
  --target-cost 315169344 \
  --steps 100 --seed-start 0 --seed-count 20 \
  --learning-rate 0.1 --perturbation 0.25 --baseline-decay 0.9 \
  --output reports/qwen36_35b_block0_hard_search.json
```

The exhaustive model portion takes 100.25 seconds; the complete audit takes
134.16 seconds and peaks at 2,565,668,864 bytes RSS. Candidate installation
retains the established 262,144-byte decoded and 131,072-byte BF16 chunk
bounds. The exact-cost full-model driver integration and its one-step memory
gate remain open.

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

## Native full-model exact-byte evaluation

`src/native_runtime.py` maps canonical GGUF candidate names back to their
checkpoint parameters and installs Q2_0/Q4_0 rows directly into the currently
resident Transformers block. It restores the pinned converter's linear-
attention head permutations and writes routed gate/up views into fused expert
storage without materializing a complete dense candidate. The streamed hard
evaluator now routes canonical names by block, accounts for selected packed
bytes and bounded decode/install chunks, and releases partially loaded prefixes
if an I/O or device transfer fails. Synthetic matrix, expert-view, and complete
streaming tests cover this path.

`reports/qwen36_35b_native_full_step.json` records the first end-to-end use of
that adapter on the genuine Qwen3.6-35B-A3B text stack. Two same-seed,
one-step antithetic REINFORCE runs each evaluate both samples, for four complete
40-layer forward passes. Only the 13 block-0 decisions have native candidates
at this stage; the other blocks remain streamed BF16, so this is the required
full-model execution proof but not a substitute for the future 512-group
database.

Every evaluation uses exact full-vocabulary causal cross-entropy over seven
next-token targets and realizes the exact 315,169,344-byte aligned candidate
budget. The paired assignments `0000100001011` and `1000000110100` produce
losses `10.090235710144043` and `9.915270805358887`. Both assignments, losses,
updated scores, and the selected incumbent reproduce exactly in the second
run. Each evaluation streams all 40 blocks and reads 69,321,221,376 checkpoint
tensor bytes. The maximum resident block is 1,685,401,984 bytes, the largest
decoded/install chunk is 262,144 bytes, and process peak RSS is 2,455,117,824
bytes. The four evaluations complete in 747.4 seconds on CPU.

CUDA remains an environment gate, not evidence inferred from the CPU run. The
host driver can enumerate the idle RTX 3060 through `nvidia-smi` after restoring
its missing device nodes, but the CUDA driver API still returns error 999 from
`cuInit`; PyTorch 2.6.0+cu124 therefore exposes no usable CUDA device. The
retained report records zero CUDA allocation/reservation and does not claim the
RTX memory gate.

### Short full-model search and Phase 5 decision

`reports/qwen36_35b_native_short_search.json` extends the same seed and
objective to three REINFORCE steps (six complete model evaluations).
`reports/qwen36_35b_native_short_search_repeat.json` is an independent replay
that names and hashes the first report as its comparison reference. The audit
requires identical assignments, scalar losses, final score vector, and selected
incumbent; the entire trajectory reproduces exactly.

All three steps have nonzero gradients and all six evaluations remain at the
exact 315,169,344-byte budget. Five distinct assignments are exercised. The
best observed loss improves on step 1's incumbent from `9.915270805358887` to
`9.846053123474121` on step 2 and `9.695976257324219` on step 3, a 2.21%
reduction from the first incumbent. The earlier block-MSE global optimum,
`0001000010101`, is sampled twice and reproducibly gives the worse end-to-end
loss `10.079442024230957`; this confirms that the complete causal-LM objective
provides information not available from the block surrogate alone.

Each run reads 415,927,328,256 checkpoint tensor bytes and 1,891,016,064
selected candidate bytes. The original and replay take 853.4 and 889.4 seconds
wall time, with median evaluation times 145.8 and 147.5 seconds. Peak RSS is
2,435,145,728 and 2,453,704,704 bytes, the resident-block maximum remains
1,685,401,984 bytes, and the candidate-chunk maximum remains 262,144 bytes.
The six-loss population variance is `0.021618132055790638`; this is assignment
quality variation, not calibration noise, because every loss reproduces.

These results, together with the genuine-block landscape where REINFORCE found
the global optimum in 19/20 seeds, satisfy the current hard-search quality
criterion: feasible assignments respond to loss, the incumbent improves at
each full-model step, memory is bounded, and the trajectory is deterministic.
Phase 5 therefore does not implement the substantially more complex relaxed
streamed backward now. Revisit that fallback only if the complete 512-group
search stalls or fails a later quality target. The immediate next production
work is full candidate-database generation.

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

1. Generate the complete native candidate database for all eligible blocks;
   the current full-model proof intentionally varies only block 0.
2. Run the one-step and multi-step memory gates on the RTX 3060 after the CUDA
   driver API can initialize successfully.
3. Implement the bounded GGUF writer, copy every selected native payload
   without requantization, and validate its assignment manifest byte-for-byte.
4. Load the completed text-only GGUF in unmodified pinned llama.cpp, run
   numerical layer checks and short CPU generation, then repeat with partial
   CUDA offload.

The standard test suite uses only tiny synthetic tensors and produces no model
artifacts:

```bash
python -m unittest discover -s tests -v
```
