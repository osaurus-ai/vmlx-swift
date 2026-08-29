# DSV4 Affine Expert Footprint — 2026-08-29

Status: **PARTIAL**. The causal source mechanism and a bounded exact-region
implementation have source-test coverage. No production/live performance or
footprint claim is accepted until the real DSV4 bundle completes the matrix
below on the exact branch head.

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
- Preserve mixed 1/2/3/4/5/6/8-bit affine geometry per projection.
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

Artifacts:

- `/private/tmp/dsv4-streaming-build-xcode.log`
- `/private/tmp/dsv4-streaming-build-final.log`
- `/private/tmp/dsv4-streaming-focused-final.log`
- `/private/tmp/dsv4-streaming-loader-ordering.log`
- `/private/tmp/dsv4-streaming-real-descriptor-tests3.log`
- `/private/tmp/dsv4-streaming-broad-tests.log`

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
