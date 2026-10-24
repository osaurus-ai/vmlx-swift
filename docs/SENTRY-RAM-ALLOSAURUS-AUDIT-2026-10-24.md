# Sentry, RAM parity and Allosaurus investigation — 2026-10-24

Status: **Audit PARTIAL. Full-app broken-stderr requalification is required
before diagnostic promotion. Remaining crash and hardware-speed rows are not cleared.**

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
Sixty groups span multiple releases. Group Z6 reports 29 lifetime events but
retains 28; the missing event is not silently counted as inspected. Severity can change
between the two queries. Counts are issue groups, not unique defects; Sentry's
per-group event totals are lifetime totals, not necessarily 90-day totals.
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
The 242 call replacements in 63 runtime files are mechanically verified; the
image-model target declares its already-transitive MLXLMCommon dependency
explicitly. Six host diagnostic writers are being routed through it too.
Compiler-plugin output and Linux logging compatibility are outside the macOS
host migration. This is diagnostic routing, not suppression of typed errors.

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
