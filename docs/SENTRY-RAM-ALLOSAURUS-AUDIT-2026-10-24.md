# Sentry, RAM parity and Allosaurus investigation — 2026-10-24

Status: **Scoped diagnostic and mixer repairs are locally qualified on the final
app pin. Overall audit remains PARTIAL: other crashes, the new Metal interactivity
report, long-reasoning speed and unavailable older-hardware rows are not cleared.**

The investigation starts at vMLX `b9a1afaeb` and Osaurus `c5fabeea4`.
All builds, loads, reproductions, tests and visual proof now run locally on
`erics-m5-max2`, as instructed by the user. Earlier remote diagnostic rows are
retained only for provenance; that machine's isolated audit apps, data, KV
cache, preferences and temporary files have been removed and its audit
processes/listener verified absent.
The user's reported M5 Pro and M3 Max are unavailable for reproduction. Tests
on another GPU cannot certify their speed or hardware-specific behavior.

## Crash inventory

Read-only, paginated Sentry queries cover all statuses over the last 90 days,
including ignored and resolved issues. The broad snapshot contains 2,676 issue
groups. A separate fatal query contains 114 groups (82 unresolved, 19 resolved,
13 ignored), with all 1,358 retained full events collected by exhausting each group cursor.
Sixty groups span multiple releases. The 90-day fatal snapshot has 1,359 matching
events; 1,358 full events were returned by exhausted group pagination. Z6 has
29 matching events in that snapshot and 28 returned records. These windowed
counts must not be labeled lifetime counts: the no-period issue query reports
2,038 aggregate count-field events across the same groups, and older records
can fall outside retention. For example, 7M reports 56 in issue detail but only
23 retained records are available. Query-filtered lastSeen values can also differ
from unfiltered issue detail. Counts are issue groups, not unique defects.
The project API's no-period query was also paginated to exhaustion and returned
the same 2,676 IDs. This covers the accessible inventory; it does not establish
availability of historical events beyond Sentry's retained data.

A per-group report records retained releases, exception mechanisms, source
assessment and proof gaps for every fatal group. Full-event triage is not
reproduction of every event in a group. Each group
keeps an explicit unproven status until its failure mechanism, affected source,
reproduction, regression tests and applicable live app rows are established.
Old-client events must not be attributed to current source without evidence.

### Current-release loader diagnostic crash

`APPLE-MACOS-3DN` occurred on Osaurus 0.25.26. An invalid stderr pipe caused
`ModelResidency.recompute` to raise `NSFileHandleOperationException` while
reporting a residency budget. This is a diagnostic-sink failure, not evidence
that the residency arithmetic or a model allocation failed.

The existing `LoadDiagnostics` helper already avoids stderr on Darwin. Route
residency, adjacent factory diagnostics and optional stage timings through that
helper. Changing only residency leaves subsequent factory writes exposed to
the same pipe. No signal disposition, descriptor flags, model setting or
generation behavior needs to change.

Subprocess regression exercises the actual residency registry with a broken
pipe and default SIGPIPE, a broken pipe and ignored SIGPIPE, and closed stderr.
Acquire and final lease release must complete. The original code fails all
three cases; the corrected registry passes all eight residency cases. The two
existing diagnostic tests also pass, including thread signal-mask and descriptor
flag preservation. Commit `1804f03ab` contains the fix. The exact committed runtime was subsequently consumed through its remote
revision by a fresh local Release Osaurus build. Local normal loading, strict
model handoff, idle unloading, cancellation and successful retry were exercised. This does not claim the entire runtime is safe with broken stderr:
other direct stderr diagnostics remain outside this scoped fix.

### Other priority crash families

- MLX host reads following failed evaluation: latest `1ZE`, `37H`, `319`,
  `2PW`, `2EX` events include pre-fix clients. Audit the checked-evaluation and
  request-failure propagation paths and rerun their real error-injection tests.
  A historical null-pointer signature alone does not establish a fresh defect.
- SQLite WAL reads: `388` and `2Z7` require connection ownership, concurrent
  readers/writers, cancellation, clear/unload and database-lifetime reproduction.
  Both retained call chains reach the vMLX DiskCache WAL: one through quota
  accounting, the other candidate metadata. Instance locking and writer
  ownership are present; the exact connection-lifetime failure is unproven.
  Twenty write-behind and four budget cases pass; that is not crash reproduction.
- `384`: the actual failing call chain is Nemotron Mamba2 / `ssmUpdate` /
  `ssmAttn` / `shape4`. Allosaurus appears elsewhere in the event; that alone
  does not attribute this crash to Allosaurus. The exact tensor/cache boundary
  remains unproven. No tensor-shape workaround is cleared.
- Plugin lifecycle/native calls, model inventory, Vision OCR continuation and
  SwiftUI/AppKit failures retain separate rows in the private ledger. Resolved
  or ignored Sentry state does not count as current-source proof.

## RAM parity audit

Measure process `phys_footprint`, MLX active/cache/peak, loaded base/draft bytes,
GPU-resident bytes and SSD-only embedding-table bytes separately. Record load,
prefill, decode, checkpoint, writer completion, idle and unload. Reusing a disk
checkpoint does not establish that its writer released the arrays.

Current source already includes owned safetensor reads, shared fused GDN rows,
early hybrid checkpoint publication, append-only checkpoint lifetime changes,
idle allocator release and process-wide asynchronous writer reservations.
Their existence is not a new RAM qualification.

`Qwen4ExpGatedResidual.fusedProjection` retained concatenated dense/affine
down+injection tensors alongside member projections. Commit `725ef1a0d` gates
fusion with `WeightBacking.copyWouldReplace` and rebinds members to row views
of the fused bank, retaining the same numerical path and packed parameter
bytes. It does not disable fusion. The f16-metadata compiled path remains a
nonfused control with its small F32 injection projection.

Synthetic tests exercise the actual mixer at its real 10240-wide geometry.
With compiled mixing disabled to exercise all three fusion routes, the
original dense, BF16-affine and F16-affine cases retain an extra 6,658,982,
4,219,046 and 3,627,936 active bytes respectively (3/3 fail). The candidate
passes 3/3 with that ablation and 3/3 with default compilation, including
parameter-byte preservation and repeated bit-identical output. An earlier
default dense baseline measured 6,674,718 extra bytes. These are synthetic
allocation regressions, not full-model speed or footprint measurements.

The v0.1 bundle's down/injection payload totals 663,388,160 bytes; v0.2 totals
347,468,800. Header totals do not establish exact full-model savings. Both
bundles have 57,600,276,736 n-gram tensor bytes served separately from GPU
weights; counting those SSD-only bytes as GPU-resident model memory is wrong.

Remaining checks include failed/cancelled loads, repeated reloads, draft-model
ownership, overlapping SSD writes across models, writer failures, growing chat,
media companion state and cold/fresh-process restore. No new flat-RAM claim.

## Allosaurus performance diagnosis

- JANGHT has fused gate/up+SwiGLU and weighted-down decode kernels, but its H128
  input/hidden rotations are separate dispatches. Fusing rotation into a dot
  kernel must be measured: recomputing a rotation per output tile can increase
  work. A separate opt-in prototype computes each rotation once per threadgroup
  and shares it across 32 output rows. Its 120 deterministic parity
  configurations are bit-identical and hot-bank microbenchmarks show
  1.05–1.20x ratios. Full-model benefit is unproven; it remains off by default
  and outside this crash/RAM change.
- JANGT sorted matrix prefill is available only through NAX. On other GPUs it
  falls back to decode kernels for every prompt batch. This is a concrete
  missing optimization for older hardware, not a measured explanation of all
  reported 11 tok/s rows.
- JANGH already has a Steel matrix prefill backend. Its fusion/rotation and
  sparse-expert tile costs differ from NAX. Test that backend explicitly;
  M5-only measurements do not qualify M3 performance.
- NAX selection checks macOS availability and GPU architecture, matching the
  vendored MLX capability rule. Determine the reporter's effective backend;
  do not infer it from the chip's product name alone.
- Native MTP already primes the head and uses an adaptive governor. Measure
  acceptance by depth, verification cost, AR pauses/probes and matched AR vs
  adaptive wall time. Acceptance alone does not establish a speedup.
- Adequate installed RAM does not establish adequate wired headroom. Capture
  wire-limit rejections, residency coverage, page-ins, SSD n-gram read latency,
  allocator limits, clocks and thermal/power state.
- Record prompt processing, runtime decode and visible chat completion rates
  independently. The current app has separate prompt and generation telemetry;
  the reported equal numbers need actual request/log evidence.

## Proof and merge gate

Keep sanitized source/test conclusions here. Raw Sentry events, per-issue
ledger, pagination receipts, crash reproduction logs, app builds, screenshots,
responses and memory samples stay in the private campaign directory. Each fix
requires the relevant fresh Release Osaurus Chat/Settings flow, coherent
multi-turn generation with token/s, cache restore and lifecycle/resource proof.
Do not merge this investigation based on inventory, source inspection or unit
tests alone.

The focused baseline passes 108 cases, covering the residency/sink
fix, checked host access following MLX failure, failure transport, resident
reader validation, weight backing, writer reservations and write-behind
failure/ownership, append-only checkpoints, native-MTP priming/boundaries,
JANGH NAX/Steel prefill and low-bit JANGT decode/prefill geometry. The three new mixer
storage cases pass on the candidate (111 distinct focused cases total). GPU kernel tests ran on M5
Max; they do not constitute M3/M5 Pro hardware proof.

## Exact-pin local qualification

Release artifact: Osaurus source `200c73539c8c588e40728bd43367f5daadd0a88b`,
vMLX revision `2a147e3cb4eb1dbd7b3cdbc04c56bec813247c4d`, executable SHA-256
`b6d40ff3bac534da42b95da4428f2e64752ac7c88b10bd23a040d95c93583f4e`.
Built and run on erics-m5-max2, Apple M5 Max (40 GPU cores), 128 GiB.
The isolated proof app uses its own data root and bundle ID, no Sentry DSN,
and the real Chat/Settings/runtime path. No hidden sampler or memory throttle.

Easy-code observations, Extra High reasoning, natural stop and closed reasoning:

| Bundle | Mode | Task | Generation tok/s | Generated tokens | Prompt tok/s |
| --- | --- | --- | ---: | ---: | ---: |
| v0.1 JANGH2 | Adaptive MTP | 100 Python print statements | 48.66 | 2691 | 806.67 |
| v0.1 JANGH2 | AR | same request | 49.67 | 959 | 499.31 |
| v0.2 JANGHT1.8 | Adaptive MTP | same request | 70.69 | 965 | 251.84 |
| v0.2 JANGHT1.8 | AR | same request | 51.50 | 1182 | 581.78 |
| v0.2 JANGHT1.8 | Adaptive MTP | JavaScript array 1..150 | 95.79 | 1234 | 662.47 |
| v0.1 JANGH2 | Adaptive MTP | same array request | 81.91 | 1438 | 775.67 |

All six visible code outputs validate exactly. Growing follow-ups return
100/1/100 and 150/151 correctly. The AR-to-adaptive transition was exercised
inside the same conversation. These UI samples are unseeded, with different
reasoning lengths; they are not a matched-token speedup experiment. Prompt
rates are effective runtime prompt telemetry and must be read with cache
receipts; short prompts do not establish long-context prefill throughput.
The 50–100 tok/s target is observed for both array rows, not every workload.
No current universal 11 tok/s cap was reproduced on this M5 Max.

Physical footprint is sampled at 50 Hz using `proc_pid_rusage`, with actual
Activity Monitor screenshots and PID identity. v0.1 settles around 38.5 GB
(decimal), v0.2 around 30.5 GB. Activity Monitor shows v0.2 at 28.30 GiB.
Strict v0.1→v0.2→v0.1 text handoffs peak at 41.11 GB in the
exact-pin process, rather than retaining both weight banks. The later real-image
prefill raises the full-process lifetime peak to 46.10 GB at 21:39:20 UTC,
settling back to about 38.8 GB. That media transient remains a separate RAM
row; the text/handoff peak must not be presented as the full-run maximum. Idle unload drops
the still-running app to 936.8 MiB in Activity Monitor (about 0.982 GB sampled
physical footprint). Cancelling a subsequent load returns below 1 GB and
unlocks the composer; retry loads successfully and answers 65 for 37+28.
These observations support parity with GPU weight payload plus runtime state;
they do not claim that total process footprint equals file size exactly.

Real-image fresh-process history restore answers correctly. Effective cache
mode is fp16 hybrid KV with disk-backed restore, 48 layers (12 KV,
36 Mamba), prefix and disk L2 enabled, paged RAM disabled, and zero TurboQuant
KV layers. The latest topology reports 12 ordinary KV layers and zero
rotating KV/wrapper layers; do not substitute an earlier load topology. Media follow-up restored boundary 101 and reprocessed the image;
this is not a full-image checkpoint hit or an SSM-pool hit. Separate SSM-pool
counters remained zero hits. No video row has been cleared.

Native picker cancellation, reopening and selection were re-exercised on this
exact artifact. Keep Model Loaded persists through relaunch; changing it to
30-second idle unload applies to the existing resident without a new request,
and restoring it saves successfully. Raw receipts stay private.

Remaining independent crash finding: 1ZM still has a ChatPanel event on
0.25.23 with an approximately 56,386-point width. The existing AppKit-owned
controller fix covers management windows, while ChatWindowManager still sets
`.minSize`. A settings regression pass cannot qualify this chat variant.
No speculative chat-sizing fix is included without a baseline reproduction.

CI distinction: Linux SwiftPM (both architectures) and four Linux CMake builds
pass. Runtime lint is advisory in the existing workflow and reports broad
format drift; it is not green. The self-hosted macOS and CUDA jobs remain
queued. Local runtime regression and fresh Release app evidence are recorded
separately from those unavailable runners. App required checks include
`test-core`, `test-cli`, `swiftlint`, and `shellcheck`; do not merge the app
until those pass.

Growing tool history accepted disk-backed restore boundaries 4318, 4412 and
4621, with all 48 cache layers restored and correct continuation to 1200.
These are full hybrid checkpoint restores; the separate SSM-pool counters
still show zero hits and must be reported separately.

## Later full-app invalid-stderr finding and expanded candidate

The narrower runtime2a147/app200c artifact was launched with the read end of
its stderr pipe closed. An actual v0.2 load failed after 4.006 seconds in
Qwen4ExpNGramTable.log/shouldWarm, raising the same broken-pipe
NSFileHandleOperationException. No generation began and token/s was absent;
this is a failed row. Passing the registry subprocess and normal-stderr app
flows does not clear the full app. Promotion was held when this surfaced.

The expanded candidate adds public `RuntimeDiagnostics` for runtime/host callers:
on Darwin it emits private-field unified logs and never writes stderr or
changes signals/descriptors; non-Darwin keeps its existing stderr behavior.
The initial 242 call replacements in 63 runtime files are mechanically verified; the
image-model target declares its already-transitive MLXLMCommon dependency
explicitly. Six host diagnostic writers are being routed through it too.
The tokenizer integration macro also emits 16 runtime diagnostic writes;
those now call the qualified shared sink. Linux logging compatibility keeps
its existing stderr behavior. This is diagnostic routing, not suppression of typed errors.

After the expansion, 112 distinct focused cases pass, including three sink
cases (errno preservation adds one), actual-registry disconnected-stderr
subprocesses, failure transport, checked host reads, resident readers,
writer/write-behind ownership, cache checkpoints, MTP boundaries and low-bit
kernels. Full Release load/generation/cache/lifecycle with disconnected stderr
remains pending. Prior exact-pin timing/RAM/UI receipts stay tied to their
original source, not to this unqualified candidate.

The shared-memory fused-rotation prototype failed its full-model AR control:
four seeded rows have byte-identical answer AND reasoning with equal token
counts, but fusion/original speed ratios 0.938–0.957 (4.3–6.2 percent slower).
It remains default-off and is rejected for promotion. Adaptive rows differ in
reasoning length even at the same seed and do not support a matched speedup
claim. A second SIMD-register candidate retains 64-thread dot geometry, passes
120 bit-exact parity configurations, and is under separate app qualification.

The second SIMD-register prototype also fails promotion: a counterbalanced
control/SIMD/SIMD/control series has identical answer, reasoning and token
counts in all 16 rows. Combined generation-time ratios per case are about
0.93, 0.98, 0.99 and 1.01; the unchanged control itself drifts substantially.
There is no reliable full-model gain. Both prototypes stay default-off and
outside the crash/RAM PR. Raw epoch outputs and usage stay private.

Apple explicitly lists Neural Accelerators on both M5 Pro and M5 Max in its
[Metal documentation](https://developer.apple.com/metal/whats-new/). Treating
M5 Pro as inherently lacking NAX would be incorrect. The current runtime
also checks OS availability and its actual architecture string; the reported
M5 Pro's effective backend remains unobserved. The missing non-NAX JANGT
matrix prefill is relevant to M3, but is not a measured root cause of the
reported equal 11 tok/s figures.

## Expanded-source local requalification and C stdio sweep

App `4edaaf816b` / runtime `5a77992b5f` was launched locally with the read
end of its stderr pipe closed, without changing the SIGPIPE policy. The
original failing API request now loads v0.2 and answers 65 at 56.72 tok/s.
Both bundle loads, real-image reading and a growing image follow-up,
picker cancellation/reopening/selection, load cancellation and successful
retry survive in the same process. The cancelled load drains to 888.9 MiB
in Activity Monitor (about 0.932 GB physical footprint). Its full-process
peak is 46.52 GB during image/handoff work. These are source-specific receipts,
not a claim about every Sentry mechanism.

The v0.1 four-function Python module completes naturally, closes reasoning,
and passes all six generated unittest methods. Its 34.96 tok/s is **below the
50–100 target**: 7,393 of 7,596 output tokens use AR fallback, with the last
lost probe measuring 32.9 ms/token against live AR at 30.6 ms/token. The
reasoning channel spends 175.8 seconds reconsidering valid implementation
choices before the complete answer. No thinking closure or sampler mask was
introduced. The growing code follow-up correctly returns 55, 6 and duplicate
index 1 at 47.26 tok/s. Real-image rows run at 57.12 and 50.22 tok/s. The v0.2
retry produces all 150 requested array values correctly at 68.10 tok/s, with
1,503 output tokens, natural stop and closed reasoning. Short/cache-assisted
prompt rates remain distinct from raw long-context prefill.

The final C stdio sweep found the default storage-healer logger still using
`fputs(stderr)`. A Foundation-only probe compiled from the actual production
sources exits with SIGPIPE (-13) under the default broken-pipe condition even
when repair is disabled; ignored-SIGPIPE and closed-stderr modes survive.
Runtime `d1c07b4df2` routes this logger through the same diagnostic sink and
survives all three conditions. The registry subprocess also invokes the
actual default-disabled healer path now. No model storage is rewritten by
this probe. App `51f63c22d` consistently pins that runtime in all three
lockfiles, its manifest and both revision contracts. Its fresh local Release
qualification remains pending at this checkpoint.

Current runtime regression: 194 distinct cases across 25 suites pass, including
10 storage-repair cases for payload/config fidelity, disk failure,
interruption, symlinks and authorization. An additional 39 distinct adaptive
MTP cases pass under their applicable fixture controls: default governor /
policy / queue checks give 36 passes and 3 explicit skips; isolated dispatch
and recovery runs execute those 3 successfully. The initial optional invocations
were invalid controls: copy proposals bypass the head-only width assumption,
and the production ceiling exceeds the simulated three-depth ladder. Those
failed logs remain retained. The corrected fixtures disable copy drafts and
cap only that simulated ladder at depth 3; production app settings are unchanged.
Fixture token/s is not real-model speed proof.

The long reasoning-to-code transition remains a performance lead: repeated
lost probes can back off to 4,096 AR tokens, potentially missing a later easier
region. No scheduling constant is changed without a controlled performance /
coherency / cache comparison. Neither fusion prototype establishes a gain.

## Final pinned local qualification and RAM Safety contract

Release app `51f63c22d9349f5e552a984c609af6630ba52c8f` consumes runtime
`d1c07b4df26af22ca4756d33d3b669f5fd483895`. The isolated, ad-hoc-signed binary
SHA-256 is `e7c655e144814140af5940b0fe262be139e16cc47ce34cce504a868aed334304`.
It has no Sentry DSN and runs on Max2 with stderr connected to a pipe whose
read end is already closed. Earlier held checkpoints above are superseded
by the following exact-code receipts:

- Both bundles load, generate coherent growing turns and stop naturally with
  closed reasoning. v0.1 adaptive array generation is 76.99 tok/s for 2,010
  tokens; all 150 values are correct. v0.2 real code review is 50.42 tok/s for
  300 tokens, with its generated unittest passing. No bundle sampler is changed.
  The earlier v0.1 full-module row at 34.96 tok/s remains below target; its
  six generated tests and 8,110 independent edge/oracle cases pass. These are
  workload observations, not a matched AR/adaptive speedup claim.
- The v0.2 longer code prompt has 3,799 tokens, with 101 restored and 3,698
  newly processed. Runtime prompt time is 5.662 seconds: about 653 newly
  processed tokens/s, versus the cache-inclusive headline 670.94. Actual UI
  prefill progress and the complete answer are captured. This is warm-model,
  cold-user-prompt evidence, not a cold-SSD or older-GPU benchmark.
- Real calculator calls carry `65 * 17` → `1105` and `1105 + 95` → `1200` in
  their actual JSON arguments/results. Four engine segments run at
  59.47 / 64.47 / 72.30 / 51.15 tok/s. Growing prefix restore is observed;
  cache topology remains fp16, 12 ordinary KV + 36 Mamba, disk-backed hybrid
  restore, prefix/L2 on, paged RAM off and TurboQuant layers 0.
- Cancelled v0.1 image loading unlocks input and drains to 262.1 MiB in Activity
  Monitor. Retried history retains the real image and answers MARIGOLD 47,
  blue square left / red circle right. The growing follow-up and a fresh-process
  image-history restore also answer correctly. Measured image/restart rows
  are 59.53 and 56.96 tok/s; every generation has a recorded rate.
- Activity Monitor shows v0.1 at 35.71 GiB (about 38.34 GB decimal) after image
  work. The first final-pin process peaks at 46.41 GB physical footprint during
  media work. Saving 30-second idle unload drops the same process to 820.7 MiB;
  Keep Loaded is restored, saves and persists through restart. Native picker
  cancel/reopen/select passes. Fresh-process media restoration reprocesses the
  image; this does not qualify a full-image checkpoint hit or video.

RAM Safety was checked against the actual Settings panel, persistence, API
and next model load. Safe Auto level 2 resolves a 70% budget; selecting Strict
level 3 and saving unloads residents. The next v0.1 load applies Strict's
60% budget (82,463,372,083 bytes on this 128 GiB host), 16,384 KV plan and
owned weights. The answer remains correct. A user-explicit 64 MiB allocator
maximum then saves, unloads and reloads; the actual post-generation MLX
cache limit is exactly 67,108,864 bytes. Finally Safe Auto level 2 is restored,
the custom override is removed, persistence/API agree and residents unload.
The UI's offscreen accessibility text lagged until its summary was scrolled
onscreen; the visible Strict summary matches the API. No speculative UI fix
was made for that stale accessibility observation.

The profile's 128 MiB allocator value is a **load-time allowance**. Existing
host policy deliberately uses a bounded, model-scaled persistent reuse pool
(8 GiB ceiling for this bundle/host) and a separately bounded active MTP pool.
The ceiling is not allocated usage. Explicit user maxima constrain both.
Silently replacing the dynamic pool with the profile's 128 MiB value would
reintroduce a documented deep-context allocation-churn slowdown. This audit
changes no RAM Safety mode, admission percentage, sampler or wire policy.
The mixer repair releases duplicate backing allocations instead.

Regression receipts on d1c07b4df2: 194 focused cases across 25 suites; 39 adaptive
MTP cases under their applicable controls; 58 memory/settings-policy cases;
12 invalid-array propagation + 3 preparation methods and 2 batch failure rows
all pass. The batch rows preserve the originating error and recover the same
engine after a shader-compilation failure; they do not reproduce a real GPU
watchdog failure. App CI run `40141342943` passes all required checks and the
additional eval/package/statspack jobs. Its XCTest component reports 442 tests,
7 skips and zero failures; the separate Swift Testing component also succeeds.
Runtime run `40141326826` passes six Linux builds; lint is advisory and failing
on broad existing style drift, while self-hosted macOS/CUDA remain queued.
Local Release/GPU evidence is separate from those unavailable runners.

The 1ZM long-word/zoom test and native 36F permission-menu test do not reproduce
those historical crashes. No chat-sizing or eager-menu patch is included.

## Added report: Metal Impacting Interactivity

The user's exact `0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity`
message matches Sentry 7M: unfiltered issue detail reports 56 events, with
23 retained full events, all matching. Its ignored state is not a fix. Retained
hosts span 16/32/64/192 GiB and several OS versions. Qwen/Gemma load breadcrumbs
are present but do not establish the failing model, context length, actual
headroom or selected RAM Safety settings for the new report. Those facts remain
unknown. The file/line in `transforms.cpp` is the generic eval error bridge.

Upstream [MLX #3267](https://github.com/ml-explore/mlx/issues/3267) reports a
display-active training case; [#3302](https://github.com/ml-explore/mlx/issues/3302)
reports long-context SDPA. The latter was closed as a duplicate, which does not
prove that a chunked-attention fix shipped. These are investigative leads,
not an attribution to this reporter or proof of OOM. No watchdog-disabling
environment variable, blind retry, reduced RAM budget or sampler clamp is
installed. Pinned attention/command-buffer behavior and bounded reproduction
remain under investigation; current transport tests do not clear this row.

### Follow-up diagnostics on the merged code

Both scoped fixes are now merged: runtime #647 (`5055455f8`) and app #3185
(`2f2e1158b`). The app merge tree equals the tested Release source; the runtime
merge contains the tested pin. Subsequent diagnostics still use that exact
Release artifact, locally on Max2.

The explicit allocator maximum was also checked **during** native MTP, rather
than inferred from the post-generation value. Under Safe Auto with a temporary
64 MiB custom maximum, all 137 active-window polls report exactly 67,108,864
bytes. The array response contains all 150 correct values, stops naturally,
and generates 1,720 tokens at 55.34 tok/s. Activity Monitor shows 36.18 GiB in
the active request. This is a custom-cap policy diagnostic, not default-speed
qualification. The override is removed through Settings afterward; Save
unloads the resident model, and Safe Auto/custom-nil is restored.

A pinned-MLX attention diagnostic completes 81 combinations: FP16/BF16/FP32,
32,769/65,537/131,075 keys, 8/64/512 query rows, and unmasked/causal/last-1,024
masks, with 20 query heads, two KV heads and head dimension 128. No reported
Metal error occurs. The largest measured warm forward is 36.83 ms and the MLX
peak is 408,092,688 bytes. This is a bounded kernel-only diagnostic with no
model or memory-policy override, not a reproduction of the unknown reporter's
workload or clearance for older hardware. Six FP32 eight-query cases fail
the original strict numerical oracle; those failures remain recorded. In
particular, eight queries times GQA factor ten exceeds the vector-fused
eligibility limit of 32, so those rows take the unfused fallback. Do not
attribute their precision differences to the vector kernel without evidence.
The explicit `MLX_ENABLE_TF32=0` diagnostic repeats all 81 rows on the same
binary with the same oracle: all pass. The failing default rows use unfused
softmax/matmul, where TF32 is enabled by default. This isolates a precision
control dependency; it is not a reproduction of the interactivity failure
and does not justify changing the app's precision defaults.

Read-only Sentry Logs queries for the exact interactivity message (90 days)
and command-buffer failure text (30 days) return no records; a general query
does return logs, so this is not evidence that Logs is entirely unavailable.
Current source records generation begin/submit breadcrumbs, but the mapper's
handled `generationFailure` path preserves the error to the caller without
capturing a dedicated Sentry event. Absence of a recent fatal group therefore
does not establish absence of handled GPU failures. No guessed RAM policy or
blind GPU retry is introduced to compensate for the missing workload evidence.

### Controlled greedy observations, with rejected rows retained

The current Release app also runs explicit greedy AR/adaptive controls. These
are diagnostics, distinct from the bundle-default sampler rows above. Both
arms receive identical prompts, seed and temperature-zero overrides; each
arm is primed separately and the full answer, reasoning and token count are
compared. Settings changes go through the real panel and unload residents.

| Bundle / task | AR tok/s | Adaptive tok/s | Comparison result |
|---|---:|---:|---|
| v0.1 / 150-value array | 30.91 | 51.41 | Correct arrays; reasoning/counts differ |
| v0.1 / 100 individual prints | 28.44 | 29.77 | Same answer/reasoning/611 tokens, but both violate the requested statement format |
| v0.2 / 150-value array | 29.25 | 53.63 | Correct arrays; formatting/reasoning/counts differ |
| v0.2 / 100 individual prints | 26.80 | 46.04 | Correct identical answer; reasoning/counts differ |

All stop naturally, but no row establishes a quality-qualified exact-output
speedup: the only fully identical pair fails its instruction requirement.
Do not weaken the output validator, apply prompt/sampler masking, or present
these as universal 50–100 tok/s qualification. The no-custom-cap Safe Auto
profile and adaptive mode are restored after the controls.
