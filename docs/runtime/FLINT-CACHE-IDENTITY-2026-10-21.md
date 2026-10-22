# Flint cache identity at the automatic KV threshold

A growing real-app H2 chat restored 63,538 tokens at a 129,022-token prompt. At 134,915 tokens, Safe Auto added maxKVSize=65,536 and the salt changed from ba0800ff443a to 99e3a45b37d4. Every old boundary missed and all 134,915 tokens were prefilled again.

Flint's cache construction does not use maxKVSize: full attention uses learned-sink BF16 caches and sliding attention uses the architecture's fixed 513-row window. The written cache still had 28 BF16 key/value arrays shaped [1,4,134915,128], despite the requested cap. This was an identity change without a state/attention change.

The model now vouches for maximumKVSizeIsCacheStateInvariant. Its default is false for all other models. Model-aware cache salting omits only this provably inert identity field for Flint; it does not change the request parameters, coordinator limits, attention geometry, sampler or physical memory admission. The same computation is wired into solo, batch, speculative and media-boundary salt paths. Legacy callers without a model keep conservative keys. No global cache-format version bump invalidates other families.

Flint has no implemented optional rolling cap on its full learned-sink attention. This patch does not pretend to enforce one or silently introduce truncation. Its architectural sliding windows and all physical memory-safety limits remain as before. Old pre-threshold rows remain usable; old post-threshold keys are not migrated by this change.

Validation: all 11 Flint tests pass, including exact numerical/cache-state equality with and without a small requested maximum; automatic cap still appears at 131,073 tokens; request parameters remain unchanged; unrelated models, existing invariant adopters and other cache policy fields keep distinct keys. The strengthened conservative-default case passes separately.

Before artifacts: resident-weight-ownership-20261021/turns/F20-h2-growing-segment-C and F21-h2-over-131k-followup, flint-128k/inert-cap-cache-header.json and FLINT-131K-CACHE-IDENTITY.md. F22 beyond 134k makes a real new calculate call returning4056 and recalls all three unique markers. Two earlier turns reused the previous identical calculator result; that does not establish a parser failure or strict required-tool enforcement.

SOURCE EVIDENCE: engine48e1222737e8abe5984ce4109a300c12c5241775, Flint.swift:201, MediaSalt.swift:252–294 and PrefixCacheInvariants.swift:49–53. App65038e1481ef3534339a2d43ecee7c1b56b2e321 pins that engine at all six sites; core169ba206. Release build and strict code-signature verification passed. Actual binary SHA256 d08785572c399b043f138202beff43960d2c764cdc3a19273eff8c1475de86b9. Eleven focused Flint tests and the strengthened conservative-default case pass. Other production models retain defaultfalse, so their key/controller/kernel behavior is unchanged by this key fix.

LIVE EVIDENCE: actual isolated Release-app PID68610, final Flint JANGH2, normal UI Regenerate of saved F21 request and a connected fresh send. Captures under resident-weight-ownership-20261021/flint-threshold-fixed/ and turns/F23-h2-threshold-replay/F24-h2-threshold-followup include launch identity, raw cache/prefill logs, process rusage, SQLite turn receipts and visually inspected answer screenshots. Actual Activity Monitor was inspected: settled UI15.79GB; separate process sampling is reported below. App quit normally after completion.

F23 has the same134915-token prompt as the before row. RequestedmaxKV65536 remains in the trace; salt staysba0800ff443a, and disk restores129022tokens, prefilling only5893. The original before row missed all tiers and prefilled134915. Recorded first-token time148.8668s before versus17.4511s after. This is a cache-reuse reproduction, not a perfectly identical generation benchmark: normal Regenerate preserves original injected context but selects toolChoice.auto, whereas original send was required, and sampled output differs. No seed/sampler override was applied.

F23 actually calls calculate420-391→29, then restores134915/135247 on the tool-result step. All three markers and totals are correct, visible, normal stops;306/178tokens at42.122/43.1505tok/s. Peak/end physical footprint28.917/16.958decimalGB.

F24 fresh send uses toolChoice.required and actually calls the novel218*31-77→6681; all markers correct, normal stops. Disk restores135247/136630, then136630/136790 on the tool-result step.130/134tokens at42.5604/42.0397tok/s; peak/end28.931/17.007GB. Large~3.9GB async checkpoints settle in813–839ms in this row. These are background writer timings, not terminal response delay or zero-RAM storage. No full-history replay occurs in these steps.

The bounded cache-key defect is qualified by this before/source/after evidence. It does not close long-context checkpoint/queued-writer RAM amplification (separately documented), all-family speculative128k/media, LiLiCorr, full-quota eviction or actual48GB hardware. No release/tag; app#3161 remains gated on its fresh required CI.

No release/tag, seed/sampler/depth override or fallback.
