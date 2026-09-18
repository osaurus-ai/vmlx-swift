# Bonsai2 packed-load follow-up — component proof only

This is separate from merged runtime PR480 and app PR2808. The app pin and
30-second residency default are unchanged. Do not block the team's merged
integration on this follow-up or call this an app load-time fix yet.

## Change

`JangTernaryPacked.swift` directly expands packed bytes into affine2 words on
Metal and reports invalid bytes from the same kernel. This removes the large
intermediate trit tensors and reduces per-chunk scalar readbacks from two to
one. Shape metadata avoids a different shader specialization per module size.
CPU/non-Metal keeps the original integer implementation. Chunk size, scale
identity/F16 precision, biases, storage contract, and rejection rules remain.
No sampler, model forward, parser, cache, media, or setting change.

SOURCE EVIDENCE: base main b05cd978d75db588c8ad438540c04273c96c9de0.
Tested Git blobs: production `5ca3b62846a2456cbdd030aacd2f2a026315a2be`,
runtime tests `6f6cd184bb23cbb545b47469578ceb1552154699`, profile test
`d46b37c65a9919a2afe9639683aad7bcdee4fb88`.

## Executed evidence

LIVE EVIDENCE: M5 Max2 component tests, not a model run. Private evidence:
`/Users/eric/vmlx-private-evidence/bonsai2-swift-2026-09-17/packed-load/`.

- `SWIFTTEST_bonsai2_packed_baseline__232751.log`: original production
  expansion, five calls per shape, every output word checked. Warm iterations
  1–4: 4096x5120 took29.948–32.321ms; 2048x17408 took50.424–53.310ms.
- `SWIFTTEST_bonsai2_packed_candidate__233106.log`: same shapes/checks,
  warm iterations1–4 took6.331–7.548ms and10.878–11.535ms respectively.
  About4.4–4.6x by warm medians, not a full-model load or decode-speed claim.
  Candidate profiling followed correctness tests, so do not compare its first
  invocation with baseline as a cold shader-compilation A/B.
- `candidate-swift-testing.xml`: 24test functions,5suites,0failures/0skips.
  Includes6102 canonical byte/position cases on each of GPU and CPU,554
  malformed-byte cases in a later chunk, strided invalid padding, shape/dtype
  checks, Hadamard routing and both-storage tiny hybrid disk reconstruction.
- Same1200s/12GiB independent supervisor; normal pressure, free24GiB floor,
  swap4GiB/growth1GiB limits. Candidate peak tracked1.63GiB, unchanged1.79GiB
  swap, exit0/zero owned survivors23:33:37PDT. No full weights or generation.

Reproduction: existing isolated `.build-bonsai2` SwiftPM graph and exact vendor
pins; `BONSAI2_PACKED_PROFILE=1`, filter
`JangTernaryPackedProfileTests|JangHadamardRuntimeTests|JangHadamardContractTests|Qwen35HadamardCacheTests|Qwen35HadamardRoutingTests`.
Full command/guards are retained in the private `PLAN.md` and prior unit RUN.md.

## Promotion boundary

PARTIAL: no repinned app build or actual-model comparison for this change.
Before merging: preserve exact-array parity, measure packed load phases and
whole load in the dev app, complete a tool/image/history continuation with
both storages one at a time, and retain the existing resource guard. Do not
infer a 4x app speedup from this component test. Long-image prefill remains
the separately documented failed memory row.

Eric's23:31PDT instruction prioritizes finishing the merged integration and
forbids stretching into additional test scope. This bounded run is closed;
no further app rebuild or model campaign was started for this candidate.

## Resumed 2026-09-18 after 0.25.7

The user explicitly resumed loading/prefill, Gemma and handoff improvements.
Current engine main87a686e9 was merged into this branch as357d22a3, preserving
Gemma required-tool history and the merged Bonsai runtime. The production
expansion blob remains5ca3b62846a2456cbdd030aacd2f2a026315a2be; there is no
additional kernel or codec change in this merge. Whitespace checks pass.
This is not a substitute for rerunning the combined regression graph.

The PR remains a draft until current-head combined tests and repinned native
app evidence exist. Record cold and warm weight-loading time separately from
media preparation, cache restoration, remaining prefill and decode. Retain
the old long-media memory failure as a separate row; this change only affects
the packed storage expansion during load, not the native affine2 bundle.

### Current-main component rerun

SOURCE EVIDENCE:34026e3f84952ad7a144843fc602649c5cfbd02e, including
engine main87a686e9. Production expansion blob remains5ca3b628; the last
change only formatted the new profile fixture. All three changed Swift files
pass local strict swift-format. Repository-wide advisory CI lint still fails
on unrelated files; no CI rule was weakened.

LIVE EVIDENCE: bounded Max2 component run03:29:50–03:31:28PDT on2026-09-18,
`/Users/eric/vmlx-private-evidence/runtime-followup-2026-09-18/`:

- `SWIFTTEST_BonsaiPacked34026Components1__032950.log`:35test functions in
  8suites,0failures/0skips; selected tests include both actual bundle
  tokenizers/media processors (no full model weights), fragmented
  tool/reasoning streams, Gemma required-history, codec rejection, Hadamard
  routing and hybrid disk checkpoint reopen/continuation.
- `packed-34026e3f84952ad7a144843fc602649c5cfbd02e-one-swift-testing.xml`
  retains individual test records. `run-engine-components.sh packed
  34026e3f84952ad7a144843fc602649c5cfbd02e one` records the exact filter,
  vendor pins, lockfile and Metal artifact hashes; run through the bounded
  supervisor, not directly.
- Profile warm iterations1–4:4096x5120,7.000–8.722ms;
  2048x17408,12.531–14.791ms. These are current component times, not a new
  paired baseline or app load/decode speed claim.
- Guard exit0, peak tracked physical footprint1.42GiB, swap1.67GiB unchanged;
  final cleanup verified0group/0tracked/0watchdog survivors.

PARTIAL: current native model-loading/continuation proof still requires the
named follow-up authorization. No full model was loaded in this component run.
