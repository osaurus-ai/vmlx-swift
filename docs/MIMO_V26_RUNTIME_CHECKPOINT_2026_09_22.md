# MiMo V2.6 converted runtime checkpoint

Status: partial; not merge-ready. No release or tag is authorized.

The target is the fused-QKV, mixed affine/MXFP4 representation of
MiMo-V2.6-Flash-RL-JANG_2L. Dispatch uses the representation in `config.json`,
not a model repository name. The older MiMo implementation remains separate.

## Implemented and tested

- Text backbone preserves checkpoint dtypes, fused QKV, partial rotary
  positions, asymmetric key/value dimensions, attention sinks, and FP32
  router/weighted-expert accumulation. Native cache is nine full KV layers
  plus 39 rotating layers, with no TurboQuant layers.
- Generic loading excludes auxiliary media and MTP tensors before mapping.
  Selected text weights use exact tensor mappings. Mixed quantization keeps
  each module's explicit, shape-valid mode.
- Vision and input-audio component math has deterministic reference fixtures.
  These components are not yet connected to a production multimodal factory.
- The local mmap-enabled component run passed 68 tests in seven suites.
  The separate 97-test loader matrix has 30 existing issues; an unchanged
  baseline reproduces the same normalized issues. That matrix is not green.

## Local app evidence

All current execution is on the user's selected M5 Max Mac with 128 GiB RAM.
The private Osaurus build uses an isolated profile and this Swift worktree.

Osaurus's tool-capability rejection was caused by treating an unrecognized
descriptive XML format string as an explicit unsupported format, while
ignoring the bundle's `tool_parser` and `dialect`. The companion Osaurus
change distinguishes unknown metadata from explicit unsupported capability.
Its 15 policy tests and 16 diagnostics tests pass. Actual app requests now
pass that gate and reach model loading and prefill. Successful tool execution
has not yet been shown.

The original Strict/mmap app row produced no response before cancellation,
read about 923 GiB, and peaked at 1.58 GiB process physical footprint. That is
a failed performance row, not a successful low-memory result. Exact tensor
mappings reduce mapped allocation by about 3 GB, but another cold run still
read 300 GiB without producing a visible response before cancellation.

A controlled diagnostic changed only the existing app's custom physical
memory fraction from 0.60 to 0.80, retaining Strict mode, the same prompt,
native sampling, and cache topology. This places the measured mapped MLX
allocation below the allocator limit. It also failed: 1,119 GiB read and no
visible response before cancellation; peak physical footprint was 2.52 GiB.

Further investigation found that `loadArraysAndMetadata` ignored the exact
mapping option when its exclusion set was empty. This affected 21 of the
target's 24 shards, so the previous loader log overstated mapping coverage.
A new regression reproduced a 1,048,836-byte whole-shard mapping where the
test requires less than 262,144 bytes. The corrected option handling passes
that regression and all 12 save/load tests. The rebuilt R5 app was measured and still failed throughput, as detailed below.

The pinned Metal allocator initializes explicit wired residency to zero
(`backend/metal/allocator.h`, `resident.h`). Swift comments describing a
75-percent default are stale. The current residency policy declines to wire
this bundle because it exceeds the physical-memory reserve policy. Neither
the residency policy nor the kernel working-set limit has been changed.

Detailed private receipts, binary/source identities, process memory samples,
and runtime API snapshots are in
`~/vmlx-private-evidence/mimo26-swift-2026-09-22/STATUS.md`.
Private paths must not enter the final package dependency pin.

## Remaining gates

- Coherent visible text, native reasoning on/off, real tool continuation,
  multi-turn output, emitted token rates, and acceptable read pressure.
- Production multimodal factory/processor, ordered image/video/audio
  expansion, lazy auxiliary loading, and real media app requests.
- Correct media and geometry cache identity; cold/warm/restart restoration
  with actual BF16 KV and all topology-dependent state.
- Saved/relaunched settings, API kwargs, applicable agent-loop evaluations,
  and measured optimizations without prompt or sampler coercion.
- Complete dev-app/CLI rebuild, Swift PR and merge, then Osaurus dependency
  pin to that real merged commit and companion PR/merge. No release.

## Local integration update

The fresh VLM factory now wraps the fused text runtime and loads native vision/audio towers lazily. The processor reads the authoritative nested settings, expands media tokens before cache lookup, retains per-clip audio and visual geometry, and carries ordered content through Osaurus. Tiny checkpoint, media preprocessing, mixed quantization, and scatter tests execute on the local M5 Max. This remains partial until full-bundle multimodal app turns and cache restore are proven.

Default-policy component results:77SwiftTesting tests with9attributed TF32 chunk-comparison issues and12/12SaveTests. The pinned backend defaults multi-rowF32GEMM toTF32 onM5; GEMV staysF32. The same text matrix passes7/7withMLX_ENABLE_TF32=0; a separate media-prefill test passes all6chunk sizes under that strict policy. No application precision policy or sampling default was changed. The F32vision fixture currently records default-M5TF32 results, so strict-F32/other-device fixture qualification remains open. Strict-F32 real-bundle audio proof now matches both WAV clips exactly through mel preprocessing and all RVQ codes; default-TF32 code parity and full media app proof remain open.

The local R5 app no longer rejects tool capability metadata. A short text-only API request naturally stopped with the correct answer4, but first text took123.621seconds and throughput was0.0317tokens/s. Long tool prompts produced no response and were canceled. Exact-tensor mappings and a larger prefill chunk did not eliminate pathological weight rereads. These are failed performance rows, not usable low-RAM or completed tool/UI proof. Osaurus policy and ordered-content mapping suites pass28/28; the latest media integration still needs a fresh Releaseapp and full live/eval gates before merge. No release/tag action is authorized.


Actual local auxiliary loading now passes with the real bundle: vision produces
8x4096 BF16 embeddings; the two locally synthesized speech clips produce 46x20 and 42x20
codes, then 12x4096 and 11x4096 BF16 embeddings. The independent Python reference
matches both mel arrays and every code exactly under strict F32. One audio
embedding is exact; the other differs by at most 0.00048828125. Vision rotary
CPU/Metal trig differs at approximately 1e-7; providing identical rotary values
makes all 28 blocks and final embeddings bit-exact. These are bounded component
diagnostics, not full multimodal generation proof.

A private exact-expert mapping diagnostic produced bit-identical outputs for
all 141 routed projections versus whole-bank gather, preserving MXFP4 and affine
packing and companion dtypes. The first whole-bank traversal read 80.5 GiB;
its later warm traversal took 0.136 seconds. Exact mappings expose about 2.82 GiB
for the selected experts, with warm traversals around 0.1 seconds. Cache warmth
and diagnostic dispatch differ, so these numbers are not end-to-end token rates
or a qualified production speedup. A bounded full text-generation diagnostic
is the next gate before integrating any alternative mapping path.


## Latest bounded local diagnostics

The audio tokenizer now scopes precise encoder/RVQ math to the CPU. With the
normal TF32 backend setting, both real clips match strict Swift codes and
embeddings exactly. This does not change global GPU precision. Full audio app
requests and optimized frontend timing remain unproven.

A private optimized exact-region forward completed three coherent natural-stop
turns at 12.89, 14.97, and 13.25 tokens/s with 128 retained expert mappings per
layer. Sampled physical footprint peaked at 635 MB. A separate 32 GiB process
residency experiment did not reliably improve throughput and restored the
previous process limit. These are private diagnostic results, not shipped code,
Osaurus UI proof, or completion of the approximately 45 tokens/s target.

A compiled attention/router experiment failed its cache-wrap parity check;
it remains disabled pending strict correctness proof. Production mapping/kernel
integration, full app/tool/media/cache proof, evaluation gates, and both merges
remain outstanding. No release or tag is authorized.


## Production exact-region checkpoint

Indexed MiMo loads now validate and map exact expert slices during the throwing
load phase, exclude routed banks from generic loading, and use package-owned
native affine/MXFP4 Metal kernels for eight-expert decode. Seven focused tests
pass, including bit-exact kernel and tiny full-model/cache-wrap comparisons and
mapping-error propagation. The host-routed path declines generic whole-forward
compilation; no sampler or prompt behavior changes.

The actual local production factory and forward completed three coherent,
natural-stop turns. Load took2.061seconds. Single-token answers4and7 took1.992
and0.928seconds total; these do not establish a sustained decode rate. The
longer answer measured9.205tokens/s with peak sampled physical footprint712MB.
This is below the approximately45tokens/s target.

The canonical R6 Osaurus app/CLI build completed on this Mac. In the isolated
dev app, two consecutive `osaurus_help` calls executed with valid arguments,
grounded visible answers, preserved history, and natural stops. Final-answer
rates were 13.4 and 14.0 tokens/s; complete tool loops took 82.078 and 50.516
seconds. Native thinking was then enabled through the picker: a probability
question produced a separate closed reasoning panel and the correct visible
answer at 14.5 tokens/s. Two-tool physical-footprint sampling peaked at 7.8 GB;
the interval read approximately 184 GB from disk, so read pressure remains a
performance concern. Receipts: `local-app-r6-source-identity.json`,
`local-app-r6-tool-conversation.json`, `local-app-r6-ui-actions.jsonl`, and
`local-app-r6-two-tools-memory-summary.json` under the private evidence root.

During the follow-up, disk L2 reported a hit, with 9 ordinary KV layers and
39 rotating layers, disk-backed restore, paged RAM off, and zero TurboQuant
layers. The configured 30-second idle policy unloads the model and resets
per-model counters. This is text/tool cache evidence only.

The first real image attachment failed before generation. Unified runtime logs
and a new TokenIterator regression trace this to stable-boundary capture placing
media on a text prefix that ends before its placeholders. The split now keeps
media with the half containing the complete placeholder span; boundaries inside
that span remain unsplit. The fix and a related absolute-boundary offset
correction now pass eight focused text/media tests, four existing hybrid
stable-capture tests, and the disk-restore progress regression. The R7 app
contains the fix: the same image retry correctly identified the red circle,
then a second image was correctly identified as a blue square and contrasted
with the first. A third turn without a new attachment correctly recalled which
image had corners. All three ended naturally with separate closed reasoning
and unlocked input. Observed rates were 8.5, 5.3, and 8.6 tokens/s; first-token
times were 80.01, 55.45, and 7.49 seconds. Builds were concurrent during the
latter turns; these are operational measurements, not isolated benchmarks.

The follow-up cache receipt reports one disk L2 hit, 37 misses, and five stores,
with the same 9 KV plus 39 rotating layers, paged RAM off, and zero TurboQuant
layers. Receipts: `local-app-r7-source-identity.json`,
`local-app-r7-multimodal-history-conversation.json`, and
`local-app-r7-media-followup-admin-cache-stats.json`.

The tiny audio tokenizer reference was regenerated on CPU F32 to match the
production tokenizer's scoped device. All weights, input mels, and RVQ codes
remain bit-exact; only the two reference feature arrays changed. The same
four audio tests now pass under both strict F32 and default TF32 backend
settings, including device restoration after success and failure.

Osaurus native audio/video attachment capability evidence now uses the
representation, native processor configuration, and installed component
weights. The 52-test Osaurus metadata/policy/mapping matrix passes; the actual
bundle probe reports image, audio, and video support from 1,477 root tensors.
The fresh app rebuild is still pending. Audio/video app turns, restart cache reuse, affected full evaluations,
performance, and both merges remain incomplete. No release or tag is authorized.


### R8 local app and media accuracy checkpoint

The fresh R8 app completed three audio turns with Thinking off, using neutral
attachment names. It transcribed the first clip as “The access code is blue
seven,” identified the second as “green nine,” compared the changed color and
number, and recalled the first clip on a follow-up without a new attachment.
All three ended naturally with empty reasoning fields and unlocked input.
Observed rates were 6.9, 10.2, and 12.0 tokens/s. These WAV fixtures were
synthesized locally with macOS speech, not human recordings.

Video accuracy failed. The attached four-frame fixture contains a red circle
followed by a blue square on white. The model invented additional shapes,
split/merge events, and a black background; its follow-up repeated the wrong
background and shape count. Both turns stopped naturally, at 10.3 and 11.5
tokens/s, but neither is an accuracy pass. Decoded frames were inspected using
both FFmpeg and AVFoundation. The processor's nominal-FPS frame-count estimate
selects only two frames from this four-frame asset. Independent sRGB pixel
checks also reproduced a double tone-curve conversion in image and video
preparation; a regression fails both modalities before the correction.
Correction and refreshed live proof are in progress.

Model-free ToolEnvelope and ToolResultGrounding suites passed 10/10 and 15/15.
The full local AgentLoop, AgentLoopFrontier, ReasoningChannel, and CacheProof
run is in progress. No external judge key is configured; fallback self-judge
rubrics require manual review and are not independent scores. This local run
does not establish a frontier-provider model result.

R8 app SHA256:
`58d63d073a87516f979d24ee9f64af79671af62f7c9fa5f4e994d28d1ed96c3c`.
Runtime source for that app: `3a927fee1aacc8b6ea41fcac02543c124ca77b82`.
Receipts are `local-app-r8-source-identity.json`,
`local-app-r8-audio-multiturn-conversation.json`,
`local-app-r8-video-multiturn-conversation.json`,
`local-media-colors-r1.log`, and `evals-r8-deterministic-r2/` in the private
evidence directory. The current processor correction is newer than this app.
Performance remains below the approximately 45 tokens/s target. Both merges
remain incomplete; no release or tag is authorized.


The color and timestamp corrections now pass five processor tests, including
an encoded sRGB fixture and an H.264 file generated with variable presentation
times. R9 app proof remains pending. The vision fixture now retains all 62
original tensors unchanged and adds a CPU-F32 golden from the independent
reference. The strict matrix passes 23/23 with zero issues. The default matrix
runs 23 tests with the same nine documented TF32 chunk differences; no comparison
tolerance or production precision setting changed. A separate parameterized
MiMo disk round-trip test passes all three wrapped-window cases with bit-exact
BF16 state and continuation logits.

The older R8 full-eval baseline was interrupted after eight completed cases:
five passed and three failed. The failures include incorrect file content and
garbled post-tool replies. All three had disk L2 hits, but that association is
not a diagnosis. A focused memory-only comparison errored on all three cases:
the first generated one tool call, then a reload was refused; the remaining
cases failed admission. The harness had omitted the app's saved server-runtime
profile, including its mmap-loading and prefill policy. The comparison is
inconclusive. The Osaurus eval bootstrap correction now preserves that profile
while redirecting writable KV directories into isolated storage; its tests and
refreshed matched-profile runs are pending. Full eval gates remain incomplete.

Additional receipts: `local-media-video-timing-r1.log`,
`vision-dual-precision-comparison.json`, `local-media-strict-r26.log`,
`local-media-default-r27.log`, `local-cache-roundtrip-r1.log`,
`evals-r8-baseline-interruption.json`, and `evals-r8-memory-only-ab/AgentLoop.json`.

### R9 video retry and warm-cache boundary correction

The rebuilt local app now describes the actual video correctly: red circle to
blue square, on white, with the transition at one second. A second turn correctly
identifies the final blue square on white. Both stop naturally with thinking off.
The rows generated 116 tokens at 4.8 tokens/s (90.95 s TTFT plus 2 s load), then
17 tokens at 5.8 tokens/s (34.05 s TTFT). Compiler activity was concurrent; these
are operational observations, not an isolated performance comparison. Peak app
physical footprint was 7,830,115,632 bytes, with 369,837,142,016 bytes of read I/O
over the sampled run. The approximately 45 tokens/s target remains unmet.
The cache receipt records one L2 hit, 14 misses, five stores, nine full-KV and
39 rotating layers, paged RAM off, and TurboQuant layer count zero.

R9 app SHA256 is
`421d0d5fec5afe9a5e7e70367f12bdc79ae2949889ebe854a4ad97991b4c3eac`,
built from runtime `2cefe12fc3a68bdb91b3f56f4836ad2f14d57c38` and the recorded
Osaurus worktree. Receipts: `local-app-r9-source-identity.json`,
`local-app-r9-video-multiturn-conversation.json`,
`local-app-r9-video-followup-cache.json`, and `local-app-r9-memory-summary.json`.

A subsequent real TokenIterator reproduction found a warm-prefill bookkeeping
defect: a snapshot at absolute token 59 was labeled token 23 after restoring 36
tokens. Capture keys and stable-boundary traversal now include that restored
prefix. The regression failed before the correction and passes afterward,
including a third-turn disk restore whose continuation agrees with fresh prefill.
The strict matrix passes 19 tests in three suites with zero issues; the default
matrix passes 13 tests with four previously attributed TF32 chunk differences.
The strict invocation must set `TEST_RUNNER_MLX_ENABLE_TF32=0`, since a bare shell
`MLX_ENABLE_TF32` is not forwarded by Xcode. Receipts: `local-warm-boundary-r1.log`,
`local-warm-boundary-r3.log`, and `local-warm-boundary-r4-strict.log`.
This confirmed defect is not yet established as the cause of the full-model
post-tool failures. The correction is newer than the R9 app; refreshed app and
full eval proof are still required before merge.

### Resident packed banks and controlled decode comparison

The current candidate keeps the original packed expert banks in owned MLX memory
and passes router indices directly to native GPU gather-quantized matmul. Gate,
up and down projections retain their individual affine/MXFP4 modes, group sizes,
scales and biases. No quantization conversion or system memory-limit change is
part of this correction. Explicit mapped/host-routing diagnostics remain opt-in.
The model contract is shared with load-policy inspection so hosts account for a
full resident load. Caller allocator budgets remain unchanged.

Two loader defects were caught before promotion: Foundation read buffers needed
per-tensor autorelease pools, and integer indexing built deferred gather copies
instead of zero-copy expert views. Range slicing fixes the diagnostic views;
the production resident path consumes full banks without per-expert slicing.

Matched 128-step replay rounds on the local M5 Max 128 GiB measured:

| Path | Round 1 steps/s | Round 2 steps/s | Median latency, ms | p95, ms | Maximum, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native, unwired A2 | 39.93 | 38.02 | 25.07 / 26.03 | 26.04 / 28.69 | 26.38 / 29.31 |
| Native, wired B1 | 40.55 | 39.35 | 24.72 / 25.23 | 25.57 / 26.98 | 26.39 / 28.17 |
| Compiled MoE diagnostic C1 | 40.26 | 39.60 | 24.94 / 25.04 | 26.42 / 26.84 | 29.59 / 28.59 |

All these rounds had zero pageins and 0–20 KiB process read I/O per round.
There is one explicit sampled-token host read per step, no CPU routing read.
Graph building takes approximately 1.6–1.8 ms median; evaluation/wait takes
approximately 23–24 ms, including CPU encoding, scheduling and GPU work.
Process CPU time is approximately 18–20 ms per step. The original Swift receipt
fields named `cpu_*_ns` contain Mach ticks; the comparison receipt corrects them
using the independently calibrated 125/3 ns-per-tick host timebase.

The 904 ms historical first-decode stall did not recur in the boundary probes;
its original cause remains undetermined. An independent Python 0.32.2 reference
on the same bundle/input trace measured 38.90 steps/s in its warm round. Its
cold round included a 1.17 s outlier and substantial prefill reads. These rows
are controlled teacher-forced throughput diagnostics, not natural-answer proof.
The separate natural capture finished 596 tokens at 39.98 tokens/s.

Receipts: `controlled-comparison-r1.json`, `controlled-workload-r1.json`,
`cpu-counter-unit-calibration.json`, `controlled-python-reference-p1.json`, and
matching process-memory/summary files. Earlier `resident-gpu-compiled-r7` was
actually eager because the global compile diagnostic flag was absent; C1 sets
both flags. No production compile-policy gate was relaxed.

The approximately 45 tokens/s target is still open. The app must be rebuilt and
re-proven against this candidate; old R9/R10 mmap performance is not the current
engine ceiling. Full app/eval/merge gates remain outstanding.

The resident admission/catalog/runtime matrix passes 17/17, and the VLM wrapper
plus LoadConfiguration matrix passes 47/47 with TF32 disabled for numerical
qualification. The wrapper explicitly forwards its text model's owned-weight
requirement. Receipts: `local-resident-admission-build-r1.log` and
`local-resident-vlm-build-r1.log`. App R11 is building from the recorded source
inputs; no R11 runtime result is claimed here. The controlled replay used a
synchronous model/sample/item loop; production TokenIterator overlaps the next
GPU submission with the previous token's host read, so actual app throughput
still needs its own measurement.
