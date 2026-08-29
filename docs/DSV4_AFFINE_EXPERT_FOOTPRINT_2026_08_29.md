# DSV4 Affine Expert Footprint — 2026-08-29

Status: **FAILED PERFORMANCE GATE / PARTIAL INVESTIGATION**. The causal source
mechanism and exact-region implementation have source-test coverage, and the
real bundle now loads and produces coherent multi-turn output without
materializing the complete routed bank. However, the exact-region strategy is
not production-viable: live decode fell to 0.08--0.75 token/s versus 13.68
token/s on the resident path. No production/readiness claim is accepted.

## Before evidence

The prior PR-head OsaurusEvals row completed two children through safe
serialization (`[1,1]`, limited by `engineCapacity`) at 13.51 token/s and
185 ms TTFT, but reported 104,993 MB peak physical footprint. Behavioral
success does not waive that RAM failure.

Header-only inspection of
`dealign.ai/DeepSeek-V4-Flash-0731-JANG-CRACK` found:

- 43 routed layers and 256 experts per layer;
- 99,072 affine routed tensors across all `w1`, `w2`, and `w3`
  weight/scale/bias components;
- 93,952,409,600 routed bytes (87.5 GiB) across 102 shards;
- mixed per-projection widths, including 3-bit `w1` tensors.

Current `DeepseekV4Model.sanitize` renames every per-expert tensor and then
constructs one lazy full-bank `stacked(tensors)` value for every
layer/projection/suffix. The first routed evaluation can therefore materialize
or fault essentially the complete 87.5 GiB expert bank. This is separate from
the DSV4 one-sequence architecture cap and separate from cache size.

## Candidate correction

- Configure a model-owned header catalog before generic safetensors loading.
- Exclude only exact DSV4 affine routed keys; shared experts, router,
  attention, cache, MTP, and JANGTQ tensors remain on their existing paths.
- Map only the selected experts' exact file regions.
- Preserve the MLX-native mixed 2/3/4/5/6/8-bit affine geometry per
  projection. Reject 1-bit descriptors fail-closed; MLX's native affine
  kernels do not support that width.
- Preserve DSV4's required score-before-quantized-down ordering and fp32
  limited-SwiGLU operation.
- Bound the mapping cache per layer; do not install a permanent stacked
  overlay or enable JangPress.
- Disable the whole-tail compiled region only for this dynamic expert path,
  because CPU-resolved route IDs cannot be captured as constants. The speed
  cost must be measured and is a release gate.

## Current source verification

- Xcode-toolchain build with tests: RC 0.
- Affine streaming suite: 9/9, including mixed 3/4/2-bit scored-before-down
  numerical parity, exact loader exclusion, production loader ordering, and
  fail-closed rejection of an incomplete per-expert catalog.
- Real-bundle descriptor validation: all 43 x 256 expert layouts accepted;
  first and last expert regions mapped successfully.
- DSV4 + affine neighboring suites: 31/31.

The exact-head lifecycle correction also replaces the routed modules through
MLXNN's supported `update(modules:)` tree operation. The first live attempt
found that direct mutation of the `@ModuleInfo` property process-fataled at
load; the corrected head loads the real bundle without that fatal. Focused
tests remain 9/9 after the correction.

## Live result on head `c09c7f28`

Bundle:
`/Users/eric/models/dealign.ai/DeepSeek-V4-Flash-0731-JANG-CRACK`

- Catalog accepted all 43 x 256 expert layouts and excluded 99,072 routed
  tensors from generic hydration.
- Model load completed in 4.02 seconds.
- Three consecutive chat turns completed coherently; the previous MLXNN
  module-mutation fatal did not recur.
- Decode rates were 0.08, 0.75, and 0.14 token/s.
- The matched current resident-path OsaurusEvals row measured 13.6766 token/s
  and 169.37 ms TTFT, but failed its independent RAM gate at 104,971.37 MB
  peak physical footprint.

Verdict: the exact-region prototype proves that the full-bank materialization
is avoidable, but **fails the speed gate**. It synchronizes routed IDs back to
the CPU in every routed layer, constructs compact projection banks dynamically,
and forces evaluation before proceeding. With 43 routed layers, that destroys
decode throughput. Increasing the per-layer mmap cache cannot remove those
43 CPU/GPU synchronization boundaries.

Artifact: `/private/tmp/dsv4-338-live-chat2.log`.

## Layout constraint and next implementation boundary

The 256 per-expert tensors for a layer/projection are not one contiguous
safetensors region. They span 3--4 shards and have irregular offsets within
those shards. Consequently, `mlx_array_new_mmap_file_region` cannot expose
them as one ordinary `[experts, out, in]` array without copying or writing a
prestacked overlay. A permanent overlay and per-token active SSD streaming are
both outside the accepted product methodology.

The next viable design must retain GPU-side routing and fused affine matmul.
The concrete boundary under investigation is a shard-span/offset-addressed
affine kernel: map a bounded number of file-backed spans, provide per-expert
shard IDs and element offsets, and let Metal touch only the selected expert
pages. It must prove that Metal/MLX does not pin or fault the complete backing
spans. If that cannot be shown, this lane stays blocked rather than shipping
the 0.08--0.75 token/s prototype.

Artifacts:

- `/private/tmp/dsv4-streaming-build-xcode.log`
- `/private/tmp/dsv4-streaming-build-final.log`
- `/private/tmp/dsv4-streaming-focused-final.log`
- `/private/tmp/dsv4-streaming-loader-ordering.log`
- `/private/tmp/dsv4-streaming-real-descriptor-tests3.log`
- `/private/tmp/dsv4-streaming-broad-tests.log`
- `/private/tmp/dsv4-338-live-chat2.log`

## Mandatory live gates

Run only with no competing model/build process and one exact binary:

1. Matched base vs branch prompt, generation config, cache settings, and
   output limit.
2. Load, TTFT, token/s, coherent output, stop reason, current/peak
   `phys_footprint`, swap, compressor, and process survival.
3. At least 10 turns with tool call -> tool result -> continuation.
4. Stop mid-stream -> continue; no stale cache or endless Thinking state.
5. Restart and disk restore with prefix/paged/L2 plus DSV4 companion cache
   counters.
6. Safe-serialized two-child eval (`[1,1]`, `engineCapacity`) with both child
   outputs and parent final response.
7. Release Osaurus app built from the exact downstream pin, visually driven
   with expected permissions set to Always Allow.

Fail the lane if footprint remains near full model size, token/s becomes
unusable, output becomes incoherent, a length cap masks failure, or any cache
or tool continuation result is missing.
