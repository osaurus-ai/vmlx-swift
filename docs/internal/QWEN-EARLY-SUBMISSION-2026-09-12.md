# Qwen4Exp early AR submission checkpoint — PARTIAL

Final-default executable, UI, adverse rows and merge evidence are now recorded
in [the default-app checkpoint](../QWEN-AR-CHECKPOINT-2026-09-12.md).
It supersedes the pending-adoption statements in the historical records below;
it does not turn factual failures or interrupted sessions into passes.

## Current 2L/4S checkpoint and default policy

The source now enables eligible early submission, forward-local rotary reuse
and exact HC residual combination when their environment variables are absent.
`1` enables; explicit `0`, empty or malformed values disable. The QSA decode
mask and parallel PLE-row path retain their existing defaults. No model-name,
quant-label, sampler, cache-format or memory-limit override is introduced.
The narrower explicit-AR/shape/trace guards below are unchanged.

This decision follows the actual local M5 Max/128 GiB app observations below.
They were collected on engine `b2bd05027ae534d0edd598fe0de02fadaf3a964e`,
app `d1f65a9e0b235ffc704f1d6f79ffc056114e1f4b`, unchanged MLX core
`73312d3e9ad0bd2e1bdf9a08d91e25571e255964`, binary SHA256
`6a5b93f2f0e1e3c299520517a0a0e6a4e73f970f7d7de44998554163ab3d48aa`,
UUID `7A6962F1-6327-3D4C-A57C-CDCD7E58E8DD`. That binary used explicit
diagnostic flags: it does NOT prove absent-variable behavior of this revision.
A fresh app with six consistent dependency pins and no optimization overrides
is the remaining adoption gate; its exact-head receipts belong in PR evidence.

The adopted policy's focused run `SWIFTTEST_QwenARDefaults0912__212412.log`
executed43 Swift Testing cases across seven suites plus one native-governor
XCTest, zero failures. Suites cover default/opt-out, current/legacy environment
lookup, activity lifetime,72 exact rotary factor configurations, text/media
rotary, exact QSA/HC and mixed-format GDN/QSA/PLE disk continuation. The
governor fixture retained240 expected token IDs; it is not a real-model MTP
speed result. Supervisor exited0 at21:32:42, peak3.92GiB, swap0.49GiB unchanged,
cleanup0/0/0. The optimized unit build uses DEBUG only for an existing test
hook; the forthcoming optimized development app must not include that flag.

Each cell is the mean of two measured 891-token, naturally stopped counts
after an excluded warmup. All paired request bytes, outputs, actual input
counts, seed829 and executed bundle sampler (temperature1/top-p.95/top-k20)
match. Thinking is explicitly off for this diagnostic, MTP is off. Grouped
arms are not randomized thermal controls, a prefill gain or a universal floor.

| Quant / changed component | 34 input: control -> candidate | 8339 input | 34939 input |
|---|---:|---:|---:|
| 2L: rotary reuse only; other components on | 45.1187 -> 46.2789 | 41.2650 -> 44.3199 | 39.7389 -> 41.8813 |
| 4S: early + QSA + HC + rotary off -> on | 39.6710 -> 47.8775 | 36.0027 -> 43.6431 | 33.4495 -> 39.8239 |

Do not attribute the whole 4S gain to rotary or compare the two different
component controls as if they were the same experiment. Earlier 2L early-only
app pairs are retained below. 2L candidate rolling one-second iterator windows
were42..49 short,42..46 at8K and38..44 at35K; 4S41..51,38..46 and34..44.
Every row retains one-/five-second minima/peaks and longest token gaps; client
SSE chunk arrivals are recorded separately and are not token-rate samples.

Three frozen reference outputs and two connected image outputs per quant also
match their controls byte-for-byte. Reference parity is not factual grading:
2L reference case2 incorrectly expands AR. Both quants' image answers remain
factually wrong (OCR/scattering,0/2 each); this is a pre-existing quality
failure, NOT a vision-quality pass. The 2L image comparison spans binaries and
does not isolate rotary speed. No prompt, sampler or output filter hides this.

Private evidence root: `/Users/eric/vmlx-private-evidence/post-1653-qwen38-audit`.
Comparisons: `checkpoint-2l-rotary-{short,8k,35k}-0912.json`,
`checkpoint-2l-rotary-reference-{0,1,2}-0912.json`,
`checkpoint-2l-vision-comparison-{0,1}-0912.json`,
`checkpoint-4s-full-{short,8k,35k}-0912.json`,
`checkpoint-4s-reference-{0,1,2}-0912.json`, `checkpoint-4s-vision-{0,1}-0912.json`.
These bind full request/SSE/output/iterator/app/engine/cache receipts.
Supervised app logs under `mtp-swift-2026-09-04/logs`:
`QwenRotaryControlApp0912a__204446`, `QwenRotaryEnabledApp0912a__205108`,
`Qwen4SControlApp0912b__210324`, `Qwen4SEnabledApp0912a__211239` (all prefixed
`SWIFTTEST_`, suffixed `.log/.mem/.procs`). Peak tracked footprint55GiB for
2L and63GiB for4S; all four exited0 after AX Quit, swap0.49GiB unchanged,
owned process/watchdog cleanup0/0/0. The earlier4S control-a supervisor aborted
at a16GiB free-page floor during35K prefill and remains FAILED/INCOMPLETE.
Both final4S arms used the same8GiB test floor/72GiB physical-footprint cap;
no model memory setting or allocation throttle changed.

The installed4S config SHA256 is
`8a61c099737025f6053e277c694e15defc4ebe90df069f7c65d2f4def435060f`.
Header receipt `checkpoint-4s-header-receipt-0912.json` records actual mixed
2/3/4/8-bit groups32/64, not a whole-model top-level8-bit claim. The unchanged
loader materializes4S BF16 (`mmap=false`) but preserves2L affine metadata
(`mmap=true`); do not claim identical dtype policy across these bundles.

Python references `b8dac820`/`171e4788` retain2L56.53/4S44.09/4M43.17/6S45.85
tok/s for their specific essay workloads. Python2L and Swift2L have matching
config, tensor-index and generation-config hashes despite different folder
names; full shard payload identity has not been established. Do not infer a
different quant from the CRACK/JANG folder names. Sampling, contexts, streaming,
cache policy and4S materialization still require matched attribution.
Those receipts motivate further AR work but do not supply Swift speed proof.
The user keeps AR improvement ahead of sustained MTP. No release/tag/install
is authorized. Remaining final-default UI continuation/Stop/load-cancel and
PR/merge evidence is explicit, not inferred from these opt-in rows.

## Historical shared-AR app measurements (75ce953e)

App `05ab691fc`, engine `75ce953e`, unchanged core `73312d3e`, binary
SHA256 `8fca0c9c43f58373d7f760526c907f10a2c4850d3bc5dd2acebb1f0f6307ef67`
executed the new environment lookup and separately controlled QSA/HC paths.
All rows below use early submission, unchanged bundle sampler settings,
explicit test seed829/thinking-off, MTP off, and 891 naturally stopped output
tokens. Two measured rows per cell exclude the initial warmup. Values are
app `genTps`; independent iterator and SSE timings are retained separately.

| Actual input | QSA/HC off | QSA only | QSA + HC | QSA + HC, serial PLE |
|---|---:|---:|---:|---:|
| 34 | 45.9133 | 45.0724 | 46.3210 | 38.5121 |
| 8339 | 39.3775 | 42.9496 | 42.0010 | 35.3378 |

Serial PLE scheduling is rejected by these same-binary observations. Do not
copy the reference C/Python small-row scheduling threshold into Swift as a
default. The isolated HC effect is small/noisy here; it is still opt-in.

At34953 actual input, a new-prefix request took63924ms to prepare and decoded
41.7517tok/s. Its exact-byte repeat restored the prefix in219ms and decoded
42.2351tok/s. Both outputs matched. Rolling one-second iterator rates were
39–43 and41–44. The earlier `host-qsa-32k-cold-0912` directory is misnamed:
its own log records a disk prefix hit, so it is not cold-prefill evidence.

Private receipts: `host-all-short-components-0912.json`,
`host-all-8k-components-0912.json`, `host-35k-prefix-comparison-0912.json`,
and `QwenHostQSAApp0912a__200627.log`. Last app supervisor exited0 after an
observed AX Quit at20:21:12; peak56GiB, flat0.49GiB swap, cleanup0/0/0.
**Above45 at short AND long context is still not met.**

## Historical forward-local rotary qualification (b2bd0502)

`VMLX_QWEN4_AR_ROTARY_REUSE=1` is opt-in. A fresh `Qwen4ExpRotaryContext`
is owned by one explicit single-token AR forward. Equivalent layers share
the original position-product, cos/sin and dtype-cast graph, keyed by the
complete rotary configuration, output dtype, start/end and stride. This is
not a new approximation or a persistent KV/SSM/prefix-cache entry. The same
AR-only call-site exclusions as early submission apply. Explicit media
position arrays bypass factor reuse; real sequential post-media offsets
remain part of the key. No global model table or compiled weight capture.

The motivating native sample `host-qsa-hotpath-decode-sample-0912.txt`
observed repeated QSA position-factor construction and2497/3896 inclusive
samples in per-layer MLX submission,1076 in layer construction. These are
CPU observations, not additive GPU costs or a predicted application gain.
`QwenRotaryReuse0912__202136` completed25 Swift Testing cases and the
native-governor XCTest with zero failures, peak4.28GiB, flat0.49GiB swap,
cleanup0/0/0. This includes72 exact-factor configurations, evaluated
cross-layer reuse, key/lifetime and phase exclusions, plus QSA/HC and
mixed-cache regressions. `QwenRotaryMedia0912__202832` completed all five
existing text/batch/explicit-three-channel-media/resumed-rotary tests with
zero failures and cleanup0/0/0. The optimized unit build uses DEBUG only for
an existing unrelated host-read test hook; the dev app does not.
Real-app comparison is pending; these tests are not a speed result.

## Historical early-only app evidence (fe8b231d)

The isolated optimized local dev app at app `61e8e6d0`, engine `fe8b231d`,
core `73312d3e` has now executed the guarded path. Binary SHA256
`3b5c9f2ab86494b65f13f61e9768187931d70d9b2971b54829676ea18fc51638`,
UUID `7BF7323C-373F-3203-9CEA-280A624F0F72`; no DEBUG test hooks.
This is not a published release or an installed-app replacement.

Two measured app API rows per arm, following excluded warmups, retained the
same 891-token complete count, natural stop, request bytes, bundle sampler,
seed 829 and MTP-off route:

| Actual input tokens | Early off mean tok/s | Early on mean tok/s | Change |
|---|---:|---:|---:|
| 34 | 36.3948 | 45.3106 | +24.497% |
| 8339 | 32.8596 | 39.0530 | +18.848% |

On rolling one-second windows were 42–47 short and 36–41 at 8K. Reverse off
controls were 34.5004/30.0555; they are retained, not used to inflate the gain.
These app rows supersede the older runner-only timing for app performance,
not its source attribution. **The user's above-45 tok/s bar at both short and
long context is NOT met.** Repeated long rows restore a prefix, so their small
prepare durations are not full-prefill throughput. Separate nonce-prefill
rows are retained but are not an exact-wire comparison.

The visible first answer completed coherently at 37.6832 tok/s, but a raw-free
supervisor abort occurred before the GUI follow-up. Subsequent runs use the
existing kernel-free metric with unchanged pressure/swap/footprint bounds.
All API timing rows preceded that abort; it is not silently counted as a
successful whole-app session. Connected image rows had identical off/on
outputs including wrong OCR/science; they establish neither vision quality
nor a fresh vision-encoder pass.

Private receipts: `early-app-short-comparison-0912.json`,
`early-app-8k-comparison-0912.json`, reverse-control comparisons, the app build
log `QwenEarlySubmitAppRelease0912__182608`, and per-token stream traces.

## Historical shared-AR candidate qualification (75ce953e)

A live native sample of the same app during 8K AR decode captured 391/6002
observations rebuilding the entire Foundation environment dictionary in two
GDN policy gates. `RuntimeEnvironment.value` now copies only the requested C
environment value and preserves current/legacy precedence, explicit snapshot
lookups and dynamic changes. Inclusive stack observations are not additive
GPU timings or a predicted speedup.

`Qwen4ExpQSA` retains score math and argPartition, specializing only the B1/S1
fully-causal boolean membership/tail mask. Future-key, prefill and batch paths
remain generic; `VMLX_QSA_DECODE_MASK=0` provides a same-binary control.

`Qwen4ExpHCCombine` is opt-in via `VMLX_QWEN4_EXACT_HC_COMBINE=1`. It derives
stream count, hidden width and dtype from actual tensors, preserves the
low-precision multiply rounding before addition, and disables FP contraction
and reassociation. No weight quant, norm, sigmoid or recurrent-state math is
replaced. Non-single-row and compile-trace calls retain the original graph.

The first QSA shader test failed compilation on an invalid scalar/metadata
interface; the failure is retained and the interface corrected. Combined
generated regression `QwenHostQSAHC0912b__193404` exited 0: 27 Swift Testing
cases plus the native-governor XCTest, zero failures, peak 4.09 GiB, unchanged
0.49 GiB swap and cleanup 0/0/0. QSA tests cover F16/BF16/F32, ratios 1/4/32,
ties, threshold/tail crossings through 32769 keys, batch and future-key fallback.
HC tests compare exact bit patterns for 54 shape/dtype/stride configurations
and three rounding counterexamples that distinguish multiplication-then-add
from FMA. Connected mixed-format cache and 240-token governor tests also ran
with both candidates enabled. The test semaphore recovered the previous
crashed runner's abandoned lock using its existing 90-second timeout.

In synchronized generated-mask diagnostics, the 8339-key full-mask call
averaged 0.5060 ms generic versus 0.3789 ms specialized; at32769 keys,
0.6660 versus 0.4857 ms. The environment diagnostic measured approximately
18.87 microseconds per snapshot lookup versus 0.317 microseconds direct.
These are optimized unit-run diagnostics, not app throughput predictions or
isolated GPU timings. The new app-performance, default-adoption and merge
gates remain pending.

## Scope

The default-enabled `VMLX_QWEN4_EXP_EARLY_SUBMIT` path submits each completed trunk
layer with `asyncEval(hidden)` on the caller's stream. The CPU can assemble
later layers while previously submitted GPU work executes. No new stream,
worker, arithmetic, quantization, sampler or cache format is introduced.

The implementation is model-family based, not a JANG filename or bit-depth
special case. Existing per-projection bit/group/dtype dispatch remains intact.
This does not enable the rejected outer compiled-decode path.

Eligibility requires B=1/S=1 and an explicit AR call site. Prefill (including
a one-token prepare tail), native seed/re-entry, prefix capture, external-PLE
compiled forward, and compile tracing do not use this specialization.
The optional `NativeMTPAutoregressiveBackboneModel` capability lets only
`NativeMTPTokenIterator.generateAutoregressiveToken` use the same scheduling
as ordinary AR. Existing models without that capability retain their forward.
Hidden states remain the pre-mixer trunk states required by native MTP.

Absent settings enable the eligible path; explicit invalid and non-1 settings
disable it. Four-layer grouping was
experimentally slower at longer context and is not an accepted setting.

## Retained diagnostic evidence, before the call-site guards

Local M5 Max / 128 GiB, Release `-O -whole-module-optimization`, engine base
`6be76cc24be917cee811da60a7250043aa289ea7`, core
`73312d3e9ad0bd2e1bdf9a08d91e25571e255964`.
Runner SHA256 `68d088db16da1c1455f1818e7c75052a534d1f43ac34024df9c4e1b3484cf73d`;
diagnostic Qwen4Exp.swift SHA256
`543deebdf6bcb5044d070224f9dbfd7b8a807debd768867b092ff2ea5e1eac23`.

Actual Qwen3.8-Flash-Next-JANG_2L; bundle temperature 1/top-p 0.95/top-k 20/min-p 0,
explicit seed 829 and thinking-off. Both contexts produce the same complete
891-token count from 1 through 250, ending naturally at stop token 248046.
Two measured runs per arm follow one separately excluded warmup. These are
iterator first-to-last delivery timings, not UI refresh rates or GPU occupancy.

| Actual input tokens | Off median tok/s | Every layer median tok/s | Reverse off median tok/s | Four-layer diagnostic |
|---|---:|---:|---:|---:|
| 34 | 35.8496 | 39.7670 | 35.4182 | 40.8086 |
| 8339 | 32.2516 | 32.0621 | 31.4960 | 29.4297 |

Every-layer observed short-context improvement is 10.93%; longer-context
throughput is approximately neutral against these controls. Four-layer
grouping is rejected for its longer-context loss, despite a better short row.
Grouped measurements are not randomized thermal controls or a throughput floor.

Short every-layer 1-second rolling windows range 35–44 tok/s, 5-second windows
37.4–42.4; longer-context windows 26–37 and 31.2–34.0 respectively. Longest
iterator gaps are 40.87 ms and 44.58 ms. Complete token timestamps, outputs,
requests, sampler receipts, binary hashes and comparisons are retained in
private `early-short-abba-0912.json` and `early-8k-scheduling-0912.json` evidence.
The former's actual arm order is off/every-layer/four-layer/off, not strict ABBA.

Physical footprint peaks were 48.09 GiB short and 50.09 GiB longer-context; swap
remained 0.49 GiB and each owned process group/watchdog exited. No model files
or prefix checkpoints were deleted. Initial missing-metallib launch and the
first wrapper's stop-string assertion failure are retained as failed harness
rows; neither is silently relabeled as a successful invocation.

The measured host trace motivating this change has 23 complete token intervals,
27.557 ms average wall time and 8.562 ms mean largest gap between completed GPU
buffer spans, predominantly before the next host commit. Buffer-span union is
not useful GPU occupancy and does not attribute all remaining cost to the CPU.
Python/reference runtime per-layer submission is a source comparison, not Swift proof. The
separate Python 6S 37.2713→45.8489 result holds its HC/QSA/PLE composition fixed;
its different bundle, sampling and runtime cannot supply Swift's missing rows.

## Revised-source generated tests

`QwenEarlySubmissionDisk0912__182055` executed three Swift Testing cases and
the XCTest governor handoff case, with zero failures. The mixed-format case
covered dense F32 and routed 2/3/4-bit group 32, 4/4/4-bit group 64, and
6/4/6-bit group 64. Off/on logits, dtypes, full native cache state and offsets
matched exactly across connected prefill/decode, the fixture QSA boundary,
and a safetensors round-trip through `TQDiskSerializer`/`restoreFromDiskArrays`.
This tiny matrix does not exercise every installed fused-kernel geometry.

Executed counters distinguished ordinary/explicit AR from native seed and
one-token prepare. The real governor iterator retained all 240 expected token
IDs and cache contents, and its specialized call count equaled its AR-fallback
count. Fixture throughput is not an installed-model performance result.

The initial raw-state restore in the new test was invalid: it omitted Mamba
offsets and did not provide independent disk-loaded storage. Its 16 failures
occurred with submission off and on and are retained; the test now exercises
the actual production serialization/restore functions, with no tolerance change.

Tests used an optimized unit-only binary with DEBUG hooks for an unrelated
existing evaluation-lock test. Initial compile and missing-Metal-library
failures are retained. The final runner used the colocated Metal library from
the same core revision (SHA256 `24d4cfcd3ca8b15ead691e46219f35adabbea64c9f8de4eae9bf293fd8d5eb7b`).
Peak tracked footprint was 3.96 GiB, swap remained 0.49 GiB, cleanup 0/0/0.

## Historical remaining gates before b2bd0502 app qualification

- Qualify the additional candidates, then rebuild the optimized local dev app
  without DEBUG test hooks; repeat short and long app/API rows with exact pins,
  executable UUID/hash, actual dispatch markers and rolling stream rates.
- Connected app history, real image/video and explicit MTP fallback/re-entry
  must retain their original output/cache contracts.
- Longer contexts, other installed quants, cancellation and the default-on
  decision remain unqualified. This is not a 40–50 tok/s family-wide claim.
- No new PR/merge or user-facing default enablement is claimed by this document.

The first Release test invocation failed before these tests because an existing
MLXTests test references a DEBUG-only evaluation-lock hook. A subsequent
optimized unit-only invocation enables that hook; it is not a performance
binary and must not be used as production Release evidence.
