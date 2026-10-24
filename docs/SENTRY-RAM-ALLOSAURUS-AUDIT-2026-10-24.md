# Sentry, RAM parity and Allosaurus investigation — 2026-10-24

Status: **PARTIAL. No merge or model-performance clearance.**

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
13 ignored), with their latest full events collected. Severity can change
between the two queries. Counts are issue groups, not unique defects; Sentry's
per-group event totals are lifetime totals, not necessarily 90-day totals.
The project API's no-period query was also paginated to exhaustion and returned
the same 2,676 IDs. This covers the accessible inventory; it does not establish
availability of historical events beyond Sentry's retained data.

Latest-event triage is not reproduction of every event in a group. Each group
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
flag preservation. Commit `1804f03ab` contains the fix. Normal fresh Release
load/unload was exercised in earlier diagnostics; local requalification is in
progress. This does not claim the entire runtime is safe with broken stderr:
other direct stderr diagnostics remain outside this scoped fix.

### Other priority crash families

- MLX host reads following failed evaluation: latest `1ZE`, `37H`, `319`,
  `2PW`, `2EX` events include pre-fix clients. Audit the checked-evaluation and
  request-failure propagation paths and rerun their real error-injection tests.
  A historical null-pointer signature alone does not establish a fresh defect.
- SQLite WAL reads: `388` and `2Z7` require connection ownership, concurrent
  readers/writers, cancellation, clear/unload and database-lifetime reproduction.
  Their similar signatures do not prove the same owning database or root cause.
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
  work. No such optimization has been proven here.
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

The focused baseline currently passes 108 cases, covering the residency/sink
fix, checked host access following MLX failure, failure transport, resident
reader validation, weight backing, writer reservations and write-behind
failure/ownership, append-only checkpoints, native-MTP priming/boundaries,
JANGH NAX/Steel prefill and low-bit JANGT decode/prefill geometry. The three new mixer
storage cases pass on the candidate (111 distinct focused cases total). GPU kernel tests ran on M5
Max; they do not constitute M3/M5 Pro hardware proof.
