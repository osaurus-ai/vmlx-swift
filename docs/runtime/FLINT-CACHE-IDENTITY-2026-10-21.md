# Flint cache identity at the automatic KV threshold

A growing real-app H2 chat restored 63,538 tokens at a 129,022-token prompt. At 134,915 tokens, Safe Auto added maxKVSize=65,536 and the salt changed from ba0800ff443a to 99e3a45b37d4. Every old boundary missed and all 134,915 tokens were prefilled again.

Flint's cache construction does not use maxKVSize: full attention uses learned-sink BF16 caches and sliding attention uses the architecture's fixed 513-row window. The written cache still had 28 BF16 key/value arrays shaped [1,4,134915,128], despite the requested cap. This was an identity change without a state/attention change.

The model now vouches for maximumKVSizeIsCacheStateInvariant. Its default is false for all other models. Model-aware cache salting omits only this provably inert identity field for Flint; it does not change the request parameters, coordinator limits, attention geometry, sampler or physical memory admission. The same computation is wired into solo, batch, speculative and media-boundary salt paths. Legacy callers without a model keep conservative keys. No global cache-format version bump invalidates other families.

Flint has no implemented optional rolling cap on its full learned-sink attention. This patch does not pretend to enforce one or silently introduce truncation. Its architectural sliding windows and all physical memory-safety limits remain as before. Old pre-threshold rows remain usable; old post-threshold keys are not migrated by this change.

Validation: all 11 Flint tests pass, including exact numerical/cache-state equality with and without a small requested maximum; automatic cap still appears at 131,073 tokens; request parameters remain unchanged; unrelated models, existing invariant adopters and other cache policy fields keep distinct keys. The strengthened conservative-default case passes separately.

Before artifacts: resident-weight-ownership-20261021/turns/F20-h2-growing-segment-C and F21-h2-over-131k-followup, flint-128k/inert-cap-cache-header.json and FLINT-131K-CACHE-IDENTITY.md. F22 beyond 134k makes a real new calculate call returning4056 and recalls all three unique markers. Two earlier turns reused the previous identical calculator result; that does not establish a parser failure or strict required-tool enforcement.

Current-source post-fix Release-app replay is pending. No release/tag, seed/sampler/depth override or fallback.
