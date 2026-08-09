# DSV4 Flash — Findings & Cross-Family Backlog (2026-08-09)

Status ledger for the DSV4 speed campaign at merge time, plus the follow-up
sweep plan for other families (Laguna bases, Ornith, Bonsai, Qwen bases,
Gemma 4). Companion docs: `DSV4-PORT-STATUS.md`, `DSV4-SPEED-CAMPAIGN.md`,
`DSV4-SPEED-HANDOFF-2026-08-08.md`.

## 1. Landed in this branch (work/dsv4-fixes → main)

| Commit | What |
|---|---|
| 096b44ef | DSV4 decode fastpaths: fused qmm lm_head, shared RoPE tables, Metal live-buffer guard |
| e56571e4 | Loader limit resolution mirrored in memory-safety plan; fastpaths pinned |
| fd93476a | Fine-grained compiled decode regions behind trusted-compile entry (`VMLX_ENABLE_UNSAFE_COMPILE`) |
| 71ef54d4 | Prefill fastpaths: heads16 indexed attention kernel, compiled compressor front, strided RoPE tables |
| d117eac5..5b4ce915 | Reasoning-floor enforcement: low-effort preface injection, visible think block on low rail, decode-side 16-token close mask, floor on every effort rail |
| 3aaab225 | **CompiledFunction self-retain cycle fix** — cached trampoline captured `self` via the C closure payload → `deinit`/`mlx_detail_compile_erase` never ran → C++ `compiler_cache_` retained traced constants (= all weights) after unload; 94–96 GB stayed dirty until process restart. Fix binds `f`/`outputs` to locals before the trampoline. Regression: `Tests/MLXTests/CompiledFunctionLifetimeTests.swift` (sabotage-verified red without fix). |

## 2. Cache-tier ground truth (verified file:line, live-traced)

- **Two tiers only.** Tier 1 paged/GPU (`PagedCacheManager`, block-chain hash +
  longest-partial-prefix, `PagedCacheManager.swift:320-376`). Tier 2 disk/L2
  (`DiskCache`, SQLite index + SHA256-keyed safetensors).
- **DSV4 is paged-incompatible** (`_isPagedIncompatible`,
  `CacheCoordinator.swift:107-117`): fetch and store skip the paged tier
  entirely; `TQDiskSerializer` emits `.deepseekV4` LayerKind which
  `extractLayerData()` skips. Live trace confirms `effectiveKVLayers=0 blocks=0
  payload=false`. **All DSV4 reuse comes from disk L2.**
- **Disk tier is a genuine longest-prefix search over stored counts** —
  `candidateTokenCounts` (`DiskCache.swift:453-477`) enumerates DISTINCT
  token_counts descending; `diskHit(boundary:)`
  (`CacheCoordinator.swift:557-592`) hashes `tokens.prefix(boundary)` per
  candidate. There is no enumeration bug: observed misses are content
  divergence (see §3).
- **`skipExactDiskBoundary` is a correctness requirement** for path-dependent
  caches (`cacheRequiresDiskBackedCoordinatorRestore`,
  `CacheHelpers.swift:159-170`: HybridPoolCache/DSV4, RotatingKVCache(+Wrapper),
  TurboQuantKVCache, QuantizedKVCache, Mamba/ArraysCache/ZayaCCA). Restores at
  N-1 instead of the exact boundary; the regression it fixed: Qwen 3.5 /
  Ornith GDN restored an exact boundary, failed to find the N-1 seed, and
  discarded the restore into a full prefill.
- **Generated-suffix stripping** (`sharedPromptStripBoundary`,
  `BatchEngine.swift:~2900`, gate `VMLX_HYBRID_STRIPPED_STORE`): for hybrid
  chat prompts the one canonical reusable boundary is the prompt with the
  trailing generation scaffold removed.

## 3. The multi-turn prefill tax (root cause, unfixed — top backlog item)

Live trace on DSV4: `HIT disk boundary=3464 remaining=1367` and
`MISS all tiers tokens=2644`. The stored post-answer snapshot (count=4830) can
**never** be a prefix of the next turn's prompt because history rendering
strips think/reasoning blocks — the divergence point is always the end of the
previous prompt's history. Consequence: **every multi-turn send re-prefills
the entire previous assistant reply** (~1367 tok ≈ 3.4 s at 400 pp/s), and the
tax grows with conversation length.

Candidate fix (follow-up PR): **speculative next-turn cache warming** — after
the response is delivered, background-prefill the re-rendered (think-stripped)
assistant reply so the next turn's prompt hits at its true boundary.

## 4. Live measurements at merge time (dev osaurus GUI, M5 Max)

- Decode: `Stream completed: 602 deltas in 39.65s`, `contentLen=2208`; delta
  gaps 0.037 s early → 0.068 s late ⇒ ~27 tok/s early degrading to ~15 tok/s
  (≈13.9 tok/s effective on that long-context turn). **Below the 33 tok/s
  floor — short-context clean re-measure + sustained-hot pass deferred to the
  follow-up PR.** (Rule: speedups must hold hot AND cool; ABAB only for
  thermal attribution.)
- Memory: `footprint` 96 GB loaded / 101 GB peak; reload-after-unload no
  longer fails ("not enough free memory" was the compile-cache leak, §1).
- GPU RAM cache toggle: default OFF honored; toggling ON works end-to-end
  (`pagedKV.enabled=true` persisted, model loaded, 602-delta generation clean).

## 5. Deferred to the follow-up PR (DSV4)

1. 33 tok/s decode floor / 35+ target on short context; sustained-hot proof.
2. ~400 pp/s prefill target measurement.
3. Long/extreme-context speed rows (short + long tables).
4. 16 s post-output hang after stream completion.
5. Pre-merge trio re-proof on the merged build: first-turn reasoning present,
   no triple prefill, tools on DSV4.
6. SSD longest-prefix live proof matrix (exact / partial / divergent).
7. Disk-tier partial-block reuse (paged tier has longest-partial-prefix;
   disk tier is boundary-granular only).
8. Speculative next-turn cache warming (§3).

## 6. Cross-family sweep plan (after DSV4 merge)

Families: **Laguna bases, Ornith (GDN), Bonsai, Qwen bases, Gemma 4.**
For each family, audit the same three axes with live traces:

1. **Exactness** — cache-on vs cache-off byte-identical (or documented benign
   divergence, as with the hybrid stripped-store path).
2. **Prefix/suffix best-match block reuse** — does the family hit the paged
   tier (block-level partial reuse) or only disk L2 (boundary-level)? Which
   caches trip `cacheRequiresDiskBackedCoordinatorRestore` and pay the N-1
   restore? Does history re-rendering (think-strip, tool-call re-rendering)
   break prefix continuity the way it does for DSV4 (§3)?
3. **Decode + prefill overhead** — per-family fastpath parity with DSV4's
   ports (fused qmm lm_head, shared/strided RoPE tables, compiled decode
   regions, indexed attention): which apply, which need family-specific
   equivalents (SSM/conv state for Ornith/Laguna hybrids).

Known prior art to reuse: hybrid gen-suffix-stripped store (vmlx#125),
LFM2 conv-offset advancement (#211), Ornith SSM disk-L2 (0d3444ca),
compile-region wrapper env-read tax (false "compiled slower" verdicts),
`VMLX_ENABLE_UNSAFE_COMPILE` gating.
