# Cache finalization attribution — 2026-09-24

Baseline c3227f4c, separate from unpromoted Spark dual-projection candidate. No release/tag. This checkpoint initially adds opt-in wall-clock phase traces only; it changes no cache validation, serialization, eviction or synchronization behavior.

R33 live growing-chat tails were4.745–5.549s. Existing disk store timers begin after non-finite validation and locks, so they cannot attribute the entire tail. The current validation performs72separate scalar reductions for the137array/463MB retained Raptor checkpoint. Standalone read-only probe: reference first479.7ms, warmmedian22.62ms; batched finite predicates first52.3ms,warmmedian7.93ms. First-call order is not a controlled cold comparison. Float16/BF16/Float32 NaN/Inf counts and sorted-name limits0/1/4/8 matched. This suggests an optimization but does not explain the multi-second tail or constitute an app speedup.

`CacheFinalizationTrace` observes entry-total, coordinator geometry/companion/paged/serialization, and disk validation/locks/materialization/publication/index/metadata phases under VMLX_CACHE_FETCH_TRACE. It never evaluates/synchronizes tensors itself. Remaining native phase proof is required before changing behavior.

Review item: store-side nonFiniteTensorNames currently executes before MLXDiskCacheIOLock, despite submitting MLX work; fetch-side validation is inside its documented lock. Preserve the guard against poisoned cache data and verify safe lock ordering if batching/serialization changes are introduced. Do not use concurrent full-model stress as proof given prior host crashes.

Private evidence: `/Users/eric/vmlx-private-evidence/raptor06-speed-2026-09-23/cache-validation-r34/`, especially `fixture.json` and `r1/`. No finalization root cause, cache speedup, dev-app proof or merge readiness claim yet.
