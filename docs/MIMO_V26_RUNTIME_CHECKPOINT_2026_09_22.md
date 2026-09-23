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

### Resident app R12 and command batching diagnostic

The Release dev app now runs the resident path on this Mac. Its first two
natural-stop text answers measured **43.1 tokens/s over 482 tokens** and
**39.0 tokens/s over 421 tokens** using final engine-backed UI counters. The
temporary streaming estimate of 56.7 tokens/s was superseded by 43.1 and is
not acceptance evidence. Both answers preserve the conversation without loops
or protocol leakage; their rainbow explanations contain factual imprecision,
so this is not a perfect factual-quality result.

The unchanged strict 80% profile admitted 109,071,847,320 estimated bytes against
109,951,162,777 budget bytes, after replacing the generic 25% weight multiplier
with validated MiMo payload and KV accounting in the companion Osaurus tree.
Actual peak physical footprint was **111,369,596,384 bytes**, above that admission
estimate and nominal budget. The load budget is not a hard physical-footprint
guarantee. Full media peak qualification remains open. Cache telemetry reports
nine full and 39 rotating layers, paged RAM off, TurboQuant layers zero, one
disk-L2 hit and seven stores across the text conversation.

App SHA256: `e7cc53f960845586bc586d14f3e6b475497860b3926443a3e5afaf0e2ef342d2`.
Receipts: `local-app-r12-build-inputs.json`,
`local-app-r12-rainbow-ui-observations.json`,
`local-app-r12-rainbow-conversation.json`, `local-app-r12-rainbow-cache.json`,
and `local-app-r12-memory-summary.json`; actual CUA controls and screenshots
were exercised and inspected.

A subsequent same-binary 128-step A/B/A comparison of native command-buffer
batching versus `MLX_MAX_MB_PER_BUFFER=8192` measured 39.6–41.0 steps/s for the
default and 40.1–41.2 for the override. CPU median fell from 18–19 ms to about
13 ms per step, but wall-clock improvement was small. The override is not
promoted. This changes command batching, not a system memory ceiling.
Receipts: `controlled-command-buffer-r1-identity.json`,
`controlled-command-buffer-r1-results.json` and matching raw replay files.

The paired gate/up dispatch passed strict native parity, but its A/B/A speed
bracket was unstable: native 41.76/41.82, paired 40.29/42.69, closing native
31.34/33.32 steps/s. It is not promoted on this evidence.

### Grouped prefill, TensorOps and fused activation investigation

With identical 3,595 prompt tokens, native unsorted A/B/A baselines measured
96.7–98.9 tokens/s. Grouped routing (`VMLX_MIMO_SORT_PREFILL=1`) measured
408.3/474.1 tokens/s. It uses the existing gatherSort/scatterUnsort helpers and
native sorted-index gathered quantized GEMM without changing packed weights.
Strict cache/model parity passed 17/17. Grouped prefill is now enabled by default, with an explicit zero-valued
opt-out for diagnostics; updated app proof is building. Receipts: `grouped-prefill-r1-identity.json`,
`grouped-prefill-r1-results.json`, `local-grouped-prefill-tests-r1.log`.

The local stack sample actually entered `gather_qmm_rhs_nax` (frames at lines
326/333 of `prefill-sorted-profile-c1-stacks.txt`). This verifies existing
MLX NAX/TensorOps dispatch on this M5, rather than inferring it solely from a
source predicate. The sampled run is diagnostic, not a clean speed benchmark.
First-decode stalls remain; unprofiled sorted rounds took 0.63–1.05 seconds
with zero disk reads/pageins and predominantly system CPU time. The aggregate
sample contains substantial Metal command submission, but does not isolate
those stalls or establish the cause of the historical 904 ms event.

Apple's [TensorOps session](https://developer.apple.com/videos/play/wwdc2026/330/)
places native FP4/FP8/2-bit tensor types and E8M0 scale planes in macOS 27.
This host runs 26.4. Existing NAX gathered GEMM is available here; the new native
quantized APIs must not be described as available or benchmarked here.

The CoreML device probe sees ANE/GPU/CPU, but `AccelerationRuntime` has no
validated MiMo CoreML island. This is capability evidence only, not inference
or speed proof (`coreml-capability-r1/identity.json`, `result.txt`). Apple's
[linear quantization guide](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html)
documents 4/8-bit weights; it does not establish a direct replacement for this
bundle's mixed 2-bit affine/MXFP4 execution. No CoreML conversion is promoted.

A new opt-in `VMLX_MIMO_FUSED_GATE_UP=1` combines both packed dot products,
BF16 projection rounding, SiLU and multiplication in one kernel/output.
Strict parity passes 18/18, including the production 4096-to-2048 shape,
both gate formats, both up groups, duplicate routes, strided input, and
zero/normal/wide activations (`local-fused-gate-up-tests-r2.log`). Full-model
A/B/A measured native 42.17/42.07, fused 40.53/42.66, then native 41.84/41.83
steps/s. The fused cold round includes a 194.87 ms first-step stall; the warm
improvement is about 1.6%, with CPU median about 14.2 ms versus 17 ms.
It remains opt-in and is not proof of 45 tokens/s. Receipts:
`fused-gate-up-r1-identity.json`, `fused-gate-up-r1-results.json`. Sustained 45 tokens/s,
refreshed full media/eval gates, and merges remain open.


### R13 actual app with grouped prefill and fused gate/up

The fresh Release app SHA256
`2efb2022852d2e7dd9822cb59065fa3467fc4768613e3836eea6b486d97a1f2d`
ran grouped prefill at its new default and fused gate/up by explicit opt-in.
Final settled UI counters measured **45.3 tokens/s for 548 tokens**, followed
by **33.8 tokens/s for 422 tokens**. This is not sustained 45 across turns.
Both outputs stopped naturally without loops or protocol leakage. The first
answer contains a secondary-color-order contradiction and incorrectly calls
the reflection total; the follow-up fixes the color order but retains other
scientific imprecision. These are not clean factual-quality passes.

First TTFT was 1.52 seconds plus 38.8 seconds model load, with disk restore
at boundary 3234 and only 106 remaining prompt tokens. The follow-up had
TTFT 8.07 seconds without reloading, restoring boundary 3340 and prefilling
663 remaining tokens. Do not attribute the first short TTFT solely to the
sorting optimization. Peak physical footprint was 109,489,154,856 bytes.
Cache topology remains nine full/39 rotating, TurboQuant layers zero, paged
RAM off, two disk-L2 hits and six stores.

Receipts: `local-app-r13-build-inputs.json`, `local-app-r13-process.json`,
`local-app-r13-rainbow-ui-observations.json`,
`local-app-r13-rainbow-conversation.json`, `local-app-r13-rainbow-cache.json`,
`local-app-r13-memory-summary.json`, plus inspected actual CUA screenshots.
The current strict focused matrix passes 18 kernel/cache tests and 61
media/loading tests; full current app/eval qualification remains open.

After saving Keep Model Loaded in the actual Settings UI and restarting the
same binary, two further turns measured 45.2 tokens/s (449 tokens) and
44.8 tokens/s (532 tokens). The model stayed resident through 88.825 seconds
of idle time; the next turn had 1.74 seconds TTFT without another load.
Both stopped naturally, but factual inaccuracies remain. Receipts:
`local-app-r13-keep-ui-observations.json`,
`local-app-r13-keep-conversation.json`,
`local-app-r13-keep-after-idle-cache.json`.
The second run peaked at 109,236,890,792 bytes physical footprint. A
9.059-second interior decode window of the 44.8 tokens/s turn had zero
process disk reads (19 samples, excluding the first/last second):
`local-app-r13-keep-decode-read-window.json`. This is a bounded app window,
not a claim that loading or cache restore performs no reads.

### Post-answer boundary correctness

The R13 live trace showed a saved key ending in token 436 where the next
chat template contained stop token 151645. The iterator had forwarded the
stop into KV, then replaced `y` with the next prediction; the store mistakenly
used that unforwarded prediction to label the snapshot. A real iterator/disk
regression reproduced both a missed correct boundary and a false hit on the
lookahead key (`post-answer-boundary-red-r2.log`).

The fix retains the token read already required by `next()` and uses it when
the cache is exactly one token beyond the visible answer. It adds no extra
per-token scalar synchronization. Cache policy v5 isolates old mislabeled
rows, including snapshots promoted to resume boundaries. Length-stop and
legacy-row isolation regressions are included. All four iterator/cache tests
pass, as do the mixed-expert parity suite and the 61 media/loading tests
(`post-answer-boundary-green-r1.log`, `post-answer-media-regression-r1.log`).
R14 app proof failed during a host restart; this is not a measured TTFT improvement
and does not explain the variation in decode throughput.

A private down-projection/weighting/reduction fusion experiment also passed
exact native parity at the production 2048-to-4096 shape, including distinct
and duplicate routes and zero/normal/wide BF16 inputs
(`down-reduce-parity-r1.log`). The actual-bank
128-route A/B/A microbenchmark passed exact parity and measured median
0.413 / 0.229 / 0.403 ms for native / fused / native, with p95
0.553 / 0.344 / 0.598 ms (`down-reduce-actual-micro-r1.json`). These are
synchronized single-layer timings, not full-model token throughput.

The candidate is now wired through `MixedQuantizedSwitchGLU` and
`MiMoV26MoE` behind default-off `VMLX_MIMO_FUSED_DOWN_REDUCE=1`. It preserves
BF16 projection rounding, FP32 route weighting/reduction, and GPU-resident
indices. Unsupported shapes/formats, mapped weights and prefill fall back to
the ordinary path. The integrated 11-test suite passed for fused gate/up,
native gate/up, paired gate/up, and all-default dispatch, with strict TF32
disabled (`down-reduce-integrated-*-r1.log`). Production-shape parity includes
strided inputs, duplicate routes, and zero/normal/wide activations. No
full-model throughput claim or default promotion follows from these tests.

A second actual-bank microbenchmark exercised the integrated helper, rather
than the private prototype: all 128 route traces matched native output
exactly. Native / fused / native median times were 0.363 / 0.205 / 0.368 ms,
with p95 0.416 / 0.255 / 0.440 ms
(`down-reduce-actual-micro-r2.json`, corresponding identity and summary).
Only one resident layer was loaded; full-model throughput remains unmeasured.

### R14 host watchdog restart — live proof failed

The R14 app cold-prefill run did not complete. The Mac restarted at
18:24:48 local time; the panic reports no watchdogd check-ins for 93 seconds.
The last runtime trace was a 3,332-token cold prefill under cache policy v5,
with all cache tiers missing. There is no completed answer or token/s result.
The post-answer fix remains unit-tested, not verified in the updated app.

Peak sampled app physical footprint was 110,792,845,640 bytes, below the
112 GiB process guard. That guard did not protect whole-host liveness.
The panic reported compressor and swap limits OK; these records do not
establish a specific OOM or GPU-kernel cause. Large-model retries are held
while the saved incident is investigated. Evidence:
`r14-watchdog-restart/incident.json`, `local-app-r14-runtime.log`,
`local-app-r14-memory.jsonl`, `LOCAL-RESIDENT-RUN-HOLD.json`.
No release, merge-readiness or complete-runtime claim follows from R14.

The full retired panic snapshot additionally reports 881 free 16 KiB pages
(about 14 MiB) and 42.91 GiB of compressor memory, despite a normal pressure
flag. `r14-watchdog-restart/full-panic-memory-receipt.json` preserves the
source identity and snapshot. This is evidence of inadequate headroom, not
a proven causal stack for the watchdog.

Private run supervision now checks whole-host reclaimable capacity before
launch and throughout the run, with an 8 GiB diagnostic reserve. It aborts
its own child on pressure, excessive compressor/swap growth, stale or missing
telemetry, or a missed deadline. Ten bounded tests pass, including a real
tiny child with live samples, refusal before spawn, and terminating only the
owned child while another remains alive. A tenth regression also verifies
that a failed receipt write cannot prevent abort (`host-guard-tests-r2.log`,
`host-guard-tiny-child-r2.jsonl`). No model was loaded in these tests.
The old large-model launchers remain held. These safeguards reduce risk;
they cannot guarantee that a kernel or driver will never stall.

Osaurus materialized-load admission also removes the extra 10% reclaim
credit, accounts for working state and concurrent loads, and shares the
existing handoff OS headroom policy. Its updated 31-test focused suite passes,
including refusal telemetry (`resident-host-admission-tests-r3.log`). The fresh
R15 Release dev app built successfully with source manifest verification
(`local-app-r15-build-outputs.json`), but has not been launched. Actual-app
proof remains required; R14 predates these changes.
