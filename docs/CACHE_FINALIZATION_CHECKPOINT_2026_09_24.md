# Cache finalization attribution — 2026-09-24

Baseline c3227f4c, separate from unpromoted Spark dual-projection candidate. No release/tag. This checkpoint initially adds opt-in wall-clock phase traces only; it changes no cache validation, serialization, eviction or synchronization behavior.

R33 live growing-chat tails were4.745–5.549s. Existing disk store timers begin after non-finite validation and locks, so they cannot attribute the entire tail. The current validation performs72separate scalar reductions for the137array/463MB retained Raptor checkpoint. Standalone read-only probe: reference first479.7ms, warmmedian22.62ms; batched finite predicates first52.3ms,warmmedian7.93ms. First-call order is not a controlled cold comparison. Float16/BF16/Float32 NaN/Inf counts and sorted-name limits0/1/4/8 matched. This suggests an optimization but does not explain the multi-second tail or constitute an app speedup.

`CacheFinalizationTrace` observes entry-total, coordinator geometry/companion/paged/serialization, and disk validation/locks/materialization/publication/index/metadata phases under VMLX_CACHE_FETCH_TRACE. It never evaluates/synchronizes tensors itself. Remaining native phase proof is required before changing behavior.

Review item: store-side nonFiniteTensorNames currently executes before MLXDiskCacheIOLock, despite submitting MLX work; fetch-side validation is inside its documented lock. Preserve the guard against poisoned cache data and verify safe lock ordering if batching/serialization changes are introduced. Do not use concurrent full-model stress as proof given prior host crashes.

Private evidence: `/Users/eric/vmlx-private-evidence/raptor06-speed-2026-09-23/cache-validation-r34/`, especially `fixture.json` and `r1/`. No finalization root cause, cache speedup, dev-app proof or merge readiness claim yet.

## Instrumented baseline and next candidate
Exact trace-only source050474db built400.8s with unchanged files. Binary04e2ab252c6708dab771c90b1c8ca63854e595f46cfd72b822e0d64239ffe075. Native two turns complete with expected answer and stops, tails3356/3173ms. First boundary replays10030/10057tokens with reused0 cost2307/2318ms; later boundaries reuse9728tokens and cost102–114ms. Disk validation is tens of milliseconds, so it is secondary. The earlier4.7–5.5s batch likewise attributes3.9–4.0s to the same cold replay, not quota eviction.

Candidate: capture one sealed existing chunk after ordinary batched prepare(), before forwarding the remaining chunk; do not split/change the forward schedule. Only direct rotating/full caches, native unmasked text, exact matching token suffix/offsets, eligible persistence and budget permit capture. Finalization owns the snapshot and releases/replaces it under the existing shorter-boundary policy. No second retained seed is kept when that policy replaces it. Warm restored prefixes not on the required chunk boundary keep the existing fallback; durable chunk reuse across restarts remains a separate requirement, not claimed fixed here.

Tests extend the exact token-recording fixture with distant and multiple stable prefixes from rejectedR26; expected work drops by the already-computed chunk and persisted/restored tensor/continuation checks remain. Build, tests, native exact-state/latency/footprint and dev-app proof still required before promotion.

## Captured chunk qualification

Source `4d81de5caa995b64ce0b51ad51922a8de73ba6fc` passed the production build in200.77s with no source changes; binary SHA256 `4d83e76641cc141a4c81cc15c916e3ee19ddbfc07b9b78e68c290db5762eda20`. Five focused tests passed in0.321s, covering eight chunk-edge lengths, exact cache/continuation, shorter stable-prefix replay counts, and release across shared slot aliases. The earlier ownership-test compilation failure is retained; the operation is now explicitly named `takeSnapshot`.

Guarded baseline/candidate ABBA completed eight native turns. All full outputs and685retained tensors per comparison matched exactly. Cold-turn finalization: baseline3530.648/4365.560ms, candidate1271.966/985.357ms. Candidate traces reuse9728tokens for the10030token boundary instead of replaying from zero. This removes an observed redundant replay; n=2 per arm and visible temporal drift limit the performance estimate. Candidate first-turn whole-turn rates37.10/38.47tok/s versus baseline27.34/22.80; these include finalization and are not decode rates. Peak sampled physical footprint3.58–4.23GB candidate versus3.87–4.00GB baseline; no low-RAM-family claim.

Warm-turn finalization remains3191.854/3288.992ms candidate and3141.077/3968.581ms baseline, with reused0 on the first warm boundary. Durable chunk restoration is still open. No dev-app proof or merge-readiness claim for this candidate yet. Evidence: `capture-build/build-receipt.json`, `replay-tests/candidate-owned-r2-test.log`, `capture-abba-review.json`, `capture-abba.noindex/*/comparison.json` under the private evidence directory above.
