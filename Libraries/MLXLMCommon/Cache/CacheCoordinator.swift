// Copyright © 2025 Apple Inc. All rights reserved.

import Foundation
@preconcurrency import MLX
import os

/// Serializes process-wide combined disk-quota reconciliation. Individual KV
/// and companion stores already own their IO locks; this lock only protects
/// the cross-store snapshot/eviction decision.
private enum CombinedDiskCacheQuotaLock {
    static let shared = OSAllocatedUnfairLock()
}

// MARK: - CacheDetail

/// Identifies which cache tier satisfied a lookup.
public enum CacheDetail: String, Sendable {
    /// The in-memory paged KV cache.
    case paged
    /// The on-disk L2 cache.
    case disk
    /// No cache tier had a match.
    case miss
}

// MARK: - CacheFetchResult

/// The result of a unified cache lookup across all tiers.
///
/// This carries `MLXArray` cache payloads restored from disk/SSM tiers. The
/// coordinator serializes the cache lookup/store boundaries, but MLX arrays do
/// not advertise a static `Sendable` conformance.
public enum CacheFetchResult: @unchecked Sendable {
    /// A cache hit with the matched prefix data.
    ///
    /// - Parameters:
    ///   - matchedTokens: Number of tokens matched from the cache.
    ///   - remainingTokens: Tokens that still need to be computed.
    ///   - detail: Which cache tier provided the hit.
    ///   - blocks: Paged cache blocks covering the matched prefix (empty for disk hits).
    ///   - ssmStates: Companion SSM states for hybrid models, if available.
    case hit(
        matchedTokens: Int,
        remainingTokens: [Int],
        detail: CacheDetail,
        blocks: [CacheBlock],
        ssmStates: [MLXArray]?,
        diskArrays: [String: MLXArray]? = nil
    )

    /// No cache tier had a match for the given tokens.
    case miss
}

// MARK: - CacheCoordinatorStatsSnapshot

/// Snapshot of the unified cache stack for UI and server telemetry.
public struct CacheCoordinatorStatsSnapshot: Sendable {
    public let pagedEnabled: Bool
    public let pagedStats: CacheStats?
    public let diskEnabled: Bool
    public let diskStats: DiskCacheStats?
    public let ssmStats: SSMStateCacheStats
    public let isHybrid: Bool
    public let isPagedIncompatible: Bool
    public let requiresPagedBoundaryCompanion: Bool
}

// MARK: - CacheCoordinator

/// Unified cache coordinator that cascades lookups across paged (L1),
/// disk (L2), and SSM companion caches.
///
/// The coordinator implements a tiered fetch strategy:
/// 1. Try the in-memory paged cache first (fastest).
/// 2. Fall back to the on-disk cache if the paged cache misses.
/// 3. For hybrid models (with SSM layers), also fetch companion SSM state.
///
/// Thread safety for the `_isHybrid` flag is provided by `OSAllocatedUnfairLock`.
/// Individual sub-caches handle their own internal locking.
public final class CacheCoordinator: @unchecked Sendable {

    // MARK: - Properties

    /// The configuration used to create this coordinator.
    public let config: CacheCoordinatorConfig

    /// The in-memory paged KV cache, or `nil` if disabled.
    public let pagedCache: PagedCacheManager?

    /// The on-disk L2 cache, or `nil` if disabled.
    public let diskCache: DiskCache?

    /// The SSM state companion cache for hybrid models.
    public let ssmStateCache: SSMStateCache

    /// Whether the model has hybrid (attention + SSM) layers.
    private var _isHybrid: Bool = false

    /// Whether a disk hit must have the recurrent SSM/GDN companion sidecar.
    /// Mamba and ArraysCache tensors may be path-dependent even when a v2
    /// payload also contains ordinary KV layer tags. ZAYA CCA is the one
    /// hybrid topology that owns its companion tensors inside the v2 layer
    /// payload and therefore sets this false.
    private var _requiresRecurrentSSMCompanion: Bool = false

    /// 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
    /// Whether the model has hybrid pool caches (DeepseekV4 SWA+CSA+HSA)
    /// that the paged cache can't represent. When true, fetch + store
    /// skip the paged tier entirely; the disk tier (which understands
    /// `LayerKind.deepseekV4`) handles prefix-cache reuse instead. This
    /// closes a silent regression where the paged tier reported a hit
    /// for DSV4 prompts (because token-id hashes match) but the blocks
    /// had no per-layer data (because `extractLayerData` returns nil
    /// for hybrid layers), so the `restoreLayerData` short-circuit
    /// suppressed the disk-tier lookup that WOULD have hit.
    private var _isPagedIncompatible: Bool = false

    /// Whether a paged hit is valid only at a leaf carrying typed rotating
    /// boundary state. Gemma 4's mixed SWA/full-attention cache uses paged KV
    /// for the full-attention layers and this companion for the rotating ring.
    private var _requiresPagedBoundaryCompanion: Bool = false

    /// The chat template's generation-prompt suffix token sequence — the
    /// tokens `add_generation_prompt=true` appends (e.g. `<|im_start|>assistant\n`
    /// + channel/think scaffold). Used to store a cross-turn-reusable cache
    /// boundary stripped back to the user turn, before the gen prompt that the
    /// NEXT turn replaces with the assistant reply. Empty = unknown/non-chat
    /// (stripped-boundary store skipped; safe). See Evaluate.storeCacheAfterGeneration.
    private var _genPromptSuffixTokens: [Int] = []

    /// Lock protecting `_isHybrid`, `_requiresRecurrentSSMCompanion`,
    /// `_isPagedIncompatible`, `_requiresPagedBoundaryCompanion`, and
    /// `_genPromptSuffixTokens`.
    private let lock = OSAllocatedUnfairLock()

    private struct PostPrepareCacheKeyAlias: Hashable {
        let rawTokenHash: String
        let mediaSalt: String
    }

    /// Maps a raw/pre-prepare media prompt to the model-derived token stream
    /// that actually describes the prompt-boundary KV cache.
    ///
    /// Nemotron Omni video EVS is the motivating case: the tokenizer emits a
    /// full run of video placeholders, `prepare` prunes that run after media
    /// embeddings exist, and cache storage must use the post-pruned token
    /// stream. A later identical or growing prompt only has the raw token
    /// stream before `prepare`; this alias lets it safely fetch the existing
    /// post-pruned cache entry without re-running the media path first.
    private var postPrepareAliases: [PostPrepareCacheKeyAlias: [Int]] = [:]
    private var postPrepareAliasCountsBySalt: [String: Set<Int>] = [:]

    // MARK: - Initialization

    /// Creates a new cache coordinator.
    ///
    /// Sub-caches are instantiated based on the configuration flags.
    ///
    /// - Parameter config: The cache configuration to use.
    public init(config: CacheCoordinatorConfig = CacheCoordinatorConfig()) {
        self.config = config

        if config.usePagedCache {
            self.pagedCache = PagedCacheManager(
                blockSize: config.pagedBlockSize,
                maxBlocks: config.maxCacheBlocks,
                modelKey: config.modelKey
            )
        } else {
            self.pagedCache = nil
        }

        if config.enableDiskCache {
            let dir = config.diskCacheDir
                ?? FileManager.default.temporaryDirectory
                    .appendingPathComponent("vmlx_disk_cache")
            self.diskCache = DiskCache(cacheDir: dir, maxSizeGB: config.diskCacheMaxGB, modelKey: config.modelKey)
        } else {
            self.diskCache = nil
        }

        self.ssmStateCache = SSMStateCache(
            maxEntries: config.ssmMaxEntries,
            modelKey: config.modelKey)

        if config.enableDiskCache {
            let baseDir = config.diskCacheDir
                ?? FileManager.default.temporaryDirectory
                    .appendingPathComponent("vmlx_disk_cache")
            let ssmDir = baseDir.appendingPathComponent("ssm_companion")
            let ssmMaxBytes = max(
                1,
                Int(config.diskCacheMaxGB * 1_073_741_824))
            self.ssmStateCache.diskStore = try? SSMCompanionDiskStore(
                cacheDir: ssmDir,
                modelKey: config.modelKey,
                maxBytes: ssmMaxBytes)
        }

        enforceCombinedDiskQuota()
    }

    // MARK: - Hybrid Flag

    /// Set whether the model is hybrid (has both attention and SSM layers).
    ///
    /// When hybrid mode is active, the coordinator will also fetch/store
    /// SSM companion states alongside the KV cache data.
    ///
    /// - Parameters:
    ///   - isHybrid: `true` for hybrid models.
    ///   - requiresRecurrentSSMCompanion: Exact topology contract. Omit only
    ///     when the caller has no cache topology; the conservative fallback
    ///     then requires a sidecar rather than accepting a false disk hit.
    public func setHybrid(
        _ isHybrid: Bool,
        requiresRecurrentSSMCompanion: Bool? = nil
    ) {
        lock.withLock {
            _isHybrid = isHybrid
            _requiresRecurrentSSMCompanion = isHybrid
                ? (requiresRecurrentSSMCompanion ?? true)
                : false
        }
    }

    /// Whether the model is hybrid (has both attention and SSM layers).
    public var isHybrid: Bool {
        lock.withLock { _isHybrid }
    }

    /// Whether disk admission requires a separately persisted recurrent
    /// SSM/GDN snapshot at the exact matched prompt boundary.
    public var requiresRecurrentSSMCompanion: Bool {
        lock.withLock { _requiresRecurrentSSMCompanion }
    }

    /// Whether a prompt-boundary store has any tier to land in. With both tiers
    /// disabled every store is discarded, so callers must not pay to produce a
    /// boundary snapshot — the hybrid stripped boundary in particular costs a
    /// retained cache copy or, failing that, a whole extra prefill.
    public var canPersistBoundaries: Bool {
        (pagedCache != nil && !isPagedIncompatible) || diskCache != nil
    }

    /// 2026-05-04: mark the model as paged-incompatible (DSV4 hybrid pool
    /// caches). Forces the coordinator's fetch + store paths to skip the
    /// paged tier so the disk tier (`TQDiskSerializer`) is the only
    /// prefix-reuse mechanism — which is correct for DSV4, where the
    /// cache state can't be reduced to per-token KV blocks.
    public func setPagedIncompatible(_ incompatible: Bool) {
        lock.withLock {
            _isPagedIncompatible = incompatible
            if incompatible {
                _requiresPagedBoundaryCompanion = false
            }
        }
    }

    /// Require an exact-boundary typed companion beside paged KV blocks.
    /// Enabling this contract makes the topology paged-compatible; callers
    /// must only set it after ``cacheCanUsePagedWithRotatingCompanion(_:)``.
    public func setPagedBoundaryCompanionRequired(_ required: Bool) {
        lock.withLock {
            _requiresPagedBoundaryCompanion = required
            if required {
                _isPagedIncompatible = false
            }
        }
    }

    /// Set the chat template's generation-prompt suffix tokens (computed once
    /// at model load by diffing a dummy chat render with vs. without
    /// `add_generation_prompt`).
    public func setGenPromptSuffixTokens(_ tokens: [Int]) {
        lock.withLock { _genPromptSuffixTokens = tokens }
    }

    /// The chat template's generation-prompt suffix tokens (may be empty).
    public var genPromptSuffixTokens: [Int] {
        lock.withLock { _genPromptSuffixTokens }
    }

    /// Whether the model is paged-incompatible (hybrid pool caches).
    public var isPagedIncompatible: Bool {
        lock.withLock { _isPagedIncompatible }
    }

    /// Whether paged hits require typed state on the exact matched leaf.
    public var requiresPagedBoundaryCompanion: Bool {
        lock.withLock { _requiresPagedBoundaryCompanion }
    }

    /// Thread-safe snapshot for diagnostics, UI status, and admin routes.
    public func snapshotStats() -> CacheCoordinatorStatsSnapshot {
        let pagedIsEffective = pagedCache != nil && !isPagedIncompatible
        return CacheCoordinatorStatsSnapshot(
            pagedEnabled: pagedIsEffective,
            pagedStats: pagedIsEffective ? pagedCache?.snapshotStats() : nil,
            diskEnabled: diskCache != nil,
            diskStats: diskCache?.snapshotStats(),
            ssmStats: ssmStateCache.snapshotStats(),
            isHybrid: isHybrid,
            isPagedIncompatible: isPagedIncompatible,
            requiresPagedBoundaryCompanion: requiresPagedBoundaryCompanion)
    }

    /// Release paged-cache blocks returned by ``fetch(tokens:mediaSalt:)``.
    ///
    /// Paged hits pin blocks while restore reads `cacheData`; callers must
    /// release those pins as soon as restore has copied tensors into the
    /// live model cache. Disk hits return an empty block list, so this is
    /// a no-op for non-paged tiers.
    public func release(blocks: [CacheBlock]) {
        guard let pagedCache, !blocks.isEmpty else { return }
        // Release leaves before roots. A child cannot be restored after its
        // parent is evicted, so roots must remain the newest LRU candidates.
        for block in blocks.reversed() {
            pagedCache.freeBlock(block)
        }
    }

    // MARK: - Post-Prepare Cache-Key Aliases

    /// Record a raw-to-effective prompt-token mapping for media prompts whose
    /// final cache key is only known after model preparation.
    ///
    /// The alias is deliberately scoped by `mediaSalt`. This prevents a prompt
    /// with the same text but different video/audio/image bytes, reasoning
    /// scope, or KV policy from reusing an incompatible post-prepare key.
    public func recordPostPrepareCacheKeyAlias(
        rawTokens: [Int],
        effectiveTokens: [Int],
        mediaSalt: String?
    ) {
        guard let mediaSalt, !rawTokens.isEmpty, !effectiveTokens.isEmpty else {
            return
        }
        let key = postPrepareAliasKey(rawTokens: rawTokens, mediaSalt: mediaSalt)
        lock.withLock {
            postPrepareAliases[key] = effectiveTokens
            var counts = postPrepareAliasCountsBySalt[mediaSalt] ?? []
            counts.insert(rawTokens.count)
            postPrepareAliasCountsBySalt[mediaSalt] = counts
        }
    }

    /// Resolve a pre-prepare media prompt to the effective token sequence used
    /// by cache storage, if this coordinator has already seen that raw prompt.
    ///
    /// Exact repeats return the recorded effective sequence. Growing turns use
    /// the longest recorded raw prefix and append the raw suffix, which is safe
    /// only inside the same `mediaSalt` namespace.
    public func resolvePostPrepareCacheKeyAlias(
        rawTokens: [Int],
        mediaSalt: String?
    ) -> [Int]? {
        guard let mediaSalt, !rawTokens.isEmpty else {
            return nil
        }
        return lock.withLock {
            let counts = postPrepareAliasCountsBySalt[mediaSalt] ?? []
            for count in counts
                .filter({ $0 <= rawTokens.count })
                .sorted(by: >)
            {
                let prefix = count == rawTokens.count
                    ? rawTokens
                    : Array(rawTokens.prefix(count))
                let key = postPrepareAliasKey(rawTokens: prefix, mediaSalt: mediaSalt)
                guard let effectiveTokens = postPrepareAliases[key] else {
                    continue
                }
                if count == rawTokens.count {
                    return effectiveTokens
                }
                return effectiveTokens + Array(rawTokens.dropFirst(count))
            }
            return nil
        }
    }

    private func postPrepareAliasKey(
        rawTokens: [Int],
        mediaSalt: String
    ) -> PostPrepareCacheKeyAlias {
        PostPrepareCacheKeyAlias(
            rawTokenHash: DiskCache.hashTokens(
                rawTokens,
                modelKey: config.modelKey,
                mediaSalt: mediaSalt),
            mediaSalt: mediaSalt)
    }

    // MARK: - Fetch

    /// Perform a tiered cache lookup for the given token sequence.
    ///
    /// The lookup cascades through cache tiers in order:
    /// 1. **Paged cache** (in-memory, block-aligned prefix matching).
    /// 2. **Disk cache** (exact match on full token sequence, then with one fewer token).
    /// 3. If all tiers miss, returns `.miss`.
    ///
    /// For hybrid models, SSM companion states are fetched alongside paged cache hits.
    ///
    /// The `mediaSalt` argument is a stable fingerprint of any VLM image or
    /// video content associated with the prompt (see ``computeMediaSalt(for:)``).
    /// When non-`nil` it is mixed into every tier's hash so VLM inputs with
    /// the same text prefix but different media don't alias. Pass `nil` for
    /// text-only inputs to preserve the exact pre-existing hash.
    ///
    /// - Parameters:
    ///   - tokens: The full token sequence to look up.
    ///   - mediaSalt: Optional VLM media fingerprint; `nil` for text-only.
    /// - Returns: A ``CacheFetchResult`` describing the outcome.
    public func fetch(
        tokens: [Int],
        mediaSalt: String? = nil,
        skipExactDiskBoundary: Bool = false,
        preferredDiskBoundaries: [Int] = []
    ) -> CacheFetchResult {
        func ftrace(_ msg: String) {
            if ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1" {
                FileHandle.standardError.write(Data(
                    "[vmlx][cache/fetch] \(msg) tokens=\(tokens.count) skipExactDisk=\(skipExactDiskBoundary)\n".utf8))
            }
        }
        func hasRequiredHybridSSM(
            _ states: [MLXArray]?,
            diskArrays: [String: MLXArray]? = nil
        ) -> Bool {
            if !isHybrid {
                return true
            }
            if !(states?.isEmpty ?? true) {
                return true
            }
            // Mamba/ArraysCache topologies require the separately keyed
            // prompt-boundary sidecar. A generic format-v2 marker is not
            // evidence that every recurrent layer is complete (ArraysCache
            // is intentionally serialized as `.skip`). ZAYA CCA is topology-
            // classified with this flag false because its v2 layer payload
            // atomically owns KV + conv + previous-hidden state.
            return !requiresRecurrentSSMCompanion
                && diskArrays.map { TQDiskSerializer.formatVersion(of: $0) >= 2 } == true
        }

        // 2026-05-04: skip the paged tier entirely for paged-incompatible
        // models (DSV4 hybrid pool caches). Without this short-circuit,
        // paged would report a hit on the token-id hash but `restoreLayerData`
        // would silently restore zero tokens (DSV4 layers aren't KV-bearing
        // in the paged taxonomy), and the disk tier — which DOES handle
        // DSV4 via `LayerKind.deepseekV4` — would never get consulted.
        let skipPaged = isPagedIncompatible

        // Tier 1: Paged cache (in-memory)
        if !skipPaged,
           let pagedCache,
           let result = pagedCache.fetchPrefix(tokens: tokens, mediaSalt: mediaSalt)
        {
            var matchedBlocks = result.blocks
            var matchedTokens = result.matchedTokens
            var remainingTokens = result.remainingTokens
            var ssmStates: [MLXArray]? = nil
            var canUsePagedHit = true

            if requiresPagedBoundaryCompanion {
                if let companionLeaf = matchedBlocks.lastIndex(where: {
                    $0.boundaryCompanionData != nil
                }) {
                    if companionLeaf + 1 < matchedBlocks.count {
                        let trailing = Array(matchedBlocks[(companionLeaf + 1)...])
                        release(blocks: trailing)
                        matchedBlocks = Array(matchedBlocks[...companionLeaf])
                        matchedTokens = matchedBlocks.reduce(0) { $0 + $1.tokenCount }
                        remainingTokens = Array(tokens.dropFirst(matchedTokens))
                    }
                } else {
                    release(blocks: matchedBlocks)
                    matchedBlocks = []
                    canUsePagedHit = false
                }
            }

            if canUsePagedHit, isHybrid {
                ssmStates = fetchCompleteSSMStates(
                    tokens: tokens,
                    boundary: matchedTokens,
                    mediaSalt: mediaSalt
                )
                if ssmStates?.isEmpty ?? true {
                    release(blocks: matchedBlocks)
                    canUsePagedHit = false
                }
            }

            if canUsePagedHit {
                return .hit(
                    matchedTokens: matchedTokens,
                    remainingTokens: remainingTokens,
                    detail: .paged,
                    blocks: matchedBlocks,
                    ssmStates: ssmStates,
                    diskArrays: nil
                )
            }
        }

        // Tier 2: Disk cache.
        //
        // Disk entries are stored at prompt boundaries. Exact hits cover
        // resumed identical prompts, but normal chat turns grow by many tokens
        // at a time. After an app-side unload the in-memory paged tier is gone,
        // so exact-or-one-shorter probing makes the L2 cache effectively miss
        // every growing turn. Probe indexed prompt-boundary lengths from
        // longest to shortest; each candidate is still content-address verified
        // by `DiskCache.fetch(tokens:)`, so same-length entries from other
        // models/media/prompts remain false-positive safe.
        if let diskCache {
            func diskHit(boundary: Int) -> CacheFetchResult? {
                guard boundary > 0, boundary <= tokens.count else { return nil }
                let prefix = boundary == tokens.count ? tokens : Array(tokens.prefix(boundary))
                guard let arrays = diskCache.fetch(tokens: prefix, mediaSalt: mediaSalt) else {
                    return nil
                }
                let ssmStates = resolveSSMStates(
                    forTokens: prefix,
                    boundary: boundary,
                    diskArrays: arrays,
                    mediaSalt: mediaSalt)
                if hasRequiredHybridSSM(ssmStates, diskArrays: arrays) {
                    ftrace("HIT disk boundary=\(boundary) remaining=\(tokens.count - boundary) ssm=\(ssmStates?.count ?? -1) fmtV=\(TQDiskSerializer.formatVersion(of: arrays))")
                    return .hit(
                        matchedTokens: boundary,
                        remainingTokens: Array(tokens.dropFirst(boundary)),
                        detail: .disk,
                        blocks: [],
                        ssmStates: ssmStates,
                        diskArrays: arrays
                    )
                }
                return nil
            }

            var tried = Set<Int>()
            let preferredBoundaries = skipExactDiskBoundary
                ? [tokens.count - 1]
                : [tokens.count, tokens.count - 1]
            for boundary in preferredBoundaries where boundary > 0 {
                tried.insert(boundary)
                if let hit = diskHit(boundary: boundary) {
                    return hit
                }
            }

            // Indexed lookup is intentionally bounded, so a frequently reused
            // stable system/tool boundary can eventually fall below the 128
            // largest historical prompt lengths. Merge processor-proven
            // boundaries back into the probe set before sorting; each remains
            // content-address checked against this request's exact token prefix.
            let candidateBoundaries = Set(
                diskCache.candidateTokenCounts(maxTokens: tokens.count)
                    + preferredDiskBoundaries
            ).sorted(by: >)
            for boundary in candidateBoundaries {
                // `skipExactDiskBoundary` is a correctness requirement for
                // path-dependent hybrid caches, not merely a preference for
                // the first two probes above. The indexed fallback used to
                // re-admit `tokens.count`, so Qwen 3.5 / Ornith restored an
                // exact GDN boundary, failed to find the N-1 seed state, and
                // discarded the restore for a full prompt prefill. Keep the
                // exact boundary excluded across every probe source so the
                // longest safe partial boundary is selected instead.
                guard !skipExactDiskBoundary || boundary != tokens.count else {
                    continue
                }
                guard tried.insert(boundary).inserted else { continue }
                if let hit = diskHit(boundary: boundary) {
                    return hit
                }
            }
        }

        // All tiers missed
        ftrace("MISS all tiers")
        return .miss
    }

    /// True only after the current process has deserialized or written the
    /// exact L2 entry and its on-disk fingerprint still matches the index.
    public func hasValidatedDiskEntry(
        tokens: [Int],
        mediaSalt: String? = nil
    ) -> Bool {
        guard diskCache?.hasValidatedEntry(
            tokens: tokens, mediaSalt: mediaSalt) == true
        else {
            return false
        }
        if isHybrid, requiresRecurrentSSMCompanion {
            return ssmStateCache.hasValidatedCompleteDiskEntry(
                tokens: tokens,
                boundary: tokens.count,
                mediaSalt: mediaSalt)
        }
        return true
    }

    /// Resolve SSM companion state for a disk-cache hit on a hybrid model.
    ///
    /// The in-memory SSM cache is tried first. If it misses, the unified
    /// disk payload may carry folded `__ssm_count__` / `ssm_N` entries;
    /// those are rehydrated and written back into the L1 SSM cache.
    private func resolveSSMStates(
        forTokens tokens: [Int],
        boundary: Int,
        diskArrays: [String: MLXArray],
        mediaSalt: String? = nil
    ) -> [MLXArray]? {
        guard isHybrid else { return nil }
        if let l1 = fetchCompleteSSMStates(
            tokens: tokens,
            boundary: boundary,
            mediaSalt: mediaSalt)
        {
            return l1
        }
        guard let folded = TQDiskSerializer.ssmStates(from: diskArrays) else {
            return nil
        }
        ssmStateCache.store(
            ssmStates: folded,
            tokens: tokens,
            boundary: boundary,
            mediaSalt: mediaSalt)
        enforceCombinedDiskQuota()
        return folded
    }

    /// Fetch companion SSM state only when the stored boundary is safe to
    /// extend. Partial entries represent mid-prefill snapshots and must not
    /// satisfy prefix reuse for a later growing turn.
    private func fetchCompleteSSMStates(
        tokens: [Int],
        boundary: Int,
        mediaSalt: String? = nil
    ) -> [MLXArray]? {
        guard let entry = ssmStateCache.fetchEntry(
            tokens: tokens,
            boundary: boundary,
            mediaSalt: mediaSalt)
        else {
            return nil
        }
        return entry.isComplete ? entry.states : nil
    }

    // MARK: - Store

    /// Store cache data after generation completes.
    ///
    /// Distributes the data to each enabled cache tier:
    /// 1. Paged cache receives the token sequence and per-block layer data.
    /// 2. Disk cache receives serialized cache state keyed by token hash.
    ///    ``TQDiskSerializer`` preserves each layer's real cache kind:
    ///    TurboQuant layers stay compressed, and DSV4 `HybridPoolCache`
    ///    layers store their SWA window plus CSA/HSA pool state instead of
    ///    being flattened into generic paged KV blocks.
    /// 3. SSM companion cache receives states for hybrid models.
    ///
    /// The `perLayerData` is the full-sequence per-layer output from
    /// ``extractLayerData(from:)``. This method splits it into block-sized
    /// chunks internally before passing to the paged cache.
    ///
    /// - Parameters:
    ///   - promptTokens: The full prompt token sequence.
    ///   - perLayerData: Per-layer KV tensors covering the entire prompt sequence.
    ///     Layers without KV data (SSM layers) are `nil`.
    ///   - ssmStates: SSM layer states for hybrid models, or `nil`.
    ///   - cache: The raw per-layer KV cache array from the model. When provided
    ///     and any layer is a TurboQuant cache in compressed phase, the disk tier
    ///     stores the compressed representation. Pass `nil` (default) to use the
    ///     standard float16 disk path.
    public func storeAfterGeneration(
        promptTokens: [Int],
        perLayerData: [(keys: MLXArray, values: MLXArray)?],
        ssmStates: [MLXArray]?,
        cache: [any KVCache]? = nil,
        mediaSalt: String? = nil
    ) {
        let totalTokens = promptTokens.count
        let blockSize = config.pagedBlockSize

        // Older generation call sites intentionally supplied an empty paged
        // payload for every cache that also needed typed disk persistence.
        // That was correct for rotating/CCA/pool layouts, but it silently
        // prevented Mamba/Arrays + ordinary/TurboQuant attention topologies
        // from ever populating their otherwise-compatible paged KV tier.
        // Derive the payload here, where the coordinator knows that paged RAM
        // caching is actually enabled. Keeping this conditional avoids
        // decompressing TurboQuant state when the user left paged caching off.
        let effectivePerLayerData: [(keys: MLXArray, values: MLXArray)?]
        if perLayerData.isEmpty,
           pagedCache != nil,
           !isPagedIncompatible,
           let cache,
           (!cacheCannotUsePagedCoordinatorRestore(cache)
                || cacheCanUsePagedWithRotatingCompanion(cache))
        {
            effectivePerLayerData = extractLayerData(from: cache)
        } else {
            effectivePerLayerData = perLayerData
        }

        // Split per-layer full-sequence data into per-block chunks.
        let blockLayerData = splitLayerDataIntoBlocks(
            effectivePerLayerData, blockSize: blockSize, totalTokens: totalTokens)
        let hasPagedKVPayload = blockLayerData.contains { !$0.isEmpty }
        let pagedBoundaryCompanion: [String: MLXArray]?
        if requiresPagedBoundaryCompanion, let cache {
            pagedBoundaryCompanion = TQDiskSerializer.serializePagedRotatingCompanion(
                cache: cache,
                expectedOffset: totalTokens)
        } else {
            pagedBoundaryCompanion = nil
        }
        let hasRequiredPagedCompanion = !requiresPagedBoundaryCompanion
            || pagedBoundaryCompanion != nil

        // Store in paged cache (skip when the model is paged-incompatible —
        // see `isPagedIncompatible` above). Recurrent-only/state-only caches
        // must not publish token hashes without any restorable KV payload;
        // doing so would suppress the valid typed disk fallback on fetch.
        if !isPagedIncompatible,
           hasPagedKVPayload,
           hasRequiredPagedCompanion,
           let pagedCache
        {
            pagedCache.storeTokenSequence(
                tokens: promptTokens,
                layerData: blockLayerData,
                boundaryCompanionData: pagedBoundaryCompanion,
                mediaSalt: mediaSalt)
        }

        // Store in disk cache.
        //
        // SLIDING-1: when the raw cache is available, use the v2
        // `TQDiskSerializer.serialize(cache:)` path unconditionally. The
        // v2 schema tags every layer with its `LayerKind` (kvSimple,
        // tqCompressed, qkv, mamba, rotating, kv) so RotatingKVCache,
        // MambaCache and QuantizedKVCache layers all round-trip to disk
        // — previously only the standard KV layers reached disk because
        // the legacy path filtered everything else via
        // `splitLayerDataIntoBlocks`. This is what enables full L2
        // disk persistence for sliding-window models (Gemma3/Gemma4
        // SWA layers, Mistral4 with maxKVSize, MiMoV2Flash, BaichuanM1,
        // Qwen3.5-VL inherited sliding layers).
        if let diskCache {
            if let cache {
                let arrays = TQDiskSerializer.serialize(
                    cache: cache,
                    ssmStates: isHybrid ? ssmStates : nil)
                if !arrays.isEmpty {
                    diskCache.store(
                        tokens: promptTokens, arrays: arrays, mediaSalt: mediaSalt)
                }
            } else {
                // Legacy fallback when the caller didn't pass the raw cache:
                // use the per-block flatten path so existing call sites
                // don't regress. Only standard KV layers reach disk on
                // this path; sliding/mamba/qkv layers are silently
                // dropped, same as before SLIDING-1.
                var arrays: [String: MLXArray] = [:]
                for (blockIdx, block) in blockLayerData.enumerated() {
                    for (layerIdx, kv) in block.enumerated() {
                        arrays["b\(blockIdx)_l\(layerIdx)_keys"] = kv.keys
                        arrays["b\(blockIdx)_l\(layerIdx)_values"] = kv.values
                    }
                }
                if !arrays.isEmpty {
                    diskCache.store(
                        tokens: promptTokens, arrays: arrays, mediaSalt: mediaSalt)
                }
            }
        }

        // Store SSM companion states for hybrid models
        if isHybrid, let ssmStates, !ssmStates.isEmpty {
            let boundary = totalTokens
            ssmStateCache.store(
                ssmStates: ssmStates,
                tokens: promptTokens,
                boundary: boundary,
                mediaSalt: mediaSalt
            )
        }

        enforceCombinedDiskQuota()
    }

    /// Enforce `diskCacheMaxGB` across the whole persistent cache root, not
    /// once for KV payloads and again for recurrent companion payloads.
    ///
    /// New companion sidecars record their matching KV hash, allowing an old
    /// hybrid entry to be evicted as a unit. Legacy sidecars remain readable;
    /// under quota pressure they retire before indexed KV because they cannot
    /// prove which durable KV payload can still reach them. Companions whose
    /// recorded KV payload is already gone are removed immediately.
    func enforceCombinedDiskQuota() {
        guard config.enableDiskCache,
              let diskCache,
              let companionStore = ssmStateCache.diskStore
        else { return }

        let maxBytes = Int64(max(1, Int(config.diskCacheMaxGB * 1_073_741_824)))
        CombinedDiskCacheQuotaLock.shared.lock()
        defer { CombinedDiskCacheQuotaLock.shared.unlock() }

        let kvEntries = diskCache.quotaEntries()
        let kvHashes = Set(kvEntries.map(\.hash))
        var companionEntries = companionStore.quotaEntries()

        let orphaned = companionEntries.filter {
            guard let kvHash = $0.kvHash else { return false }
            return !kvHashes.contains(kvHash)
        }
        if !orphaned.isEmpty {
            companionStore.removeQuotaEntries(hashes: Set(orphaned.map(\.hash)))
            let orphanHashes = Set(orphaned.map(\.hash))
            companionEntries.removeAll { orphanHashes.contains($0.hash) }
        }

        struct EvictionGroup {
            let sortKey: String
            let kvHashes: Set<String>
            let companionHashes: Set<String>
            let bytes: Int64
            let createdAt: Date
            /// Legacy companions predate the KV-link sidecar. They cannot
            /// prove that an indexed KV payload can still reach them, so quota
            /// pressure retires them before directly addressable KV groups.
            let priority: Int
        }

        let companionsByKVHash = Dictionary(grouping: companionEntries.compactMap { entry in
            entry.kvHash.map { ($0, entry) }
        }, by: { $0.0 })
        var groupedCompanionHashes = Set<String>()
        var groups: [EvictionGroup] = []

        for kv in kvEntries {
            let companions = companionsByKVHash[kv.hash]?.map(\.1) ?? []
            groupedCompanionHashes.formUnion(companions.map(\.hash))
            groups.append(EvictionGroup(
                sortKey: "kv:\(kv.hash)",
                kvHashes: [kv.hash],
                companionHashes: Set(companions.map(\.hash)),
                bytes: kv.bytes + companions.reduce(0) { $0 + $1.bytes },
                createdAt: companions.reduce(kv.createdAt) {
                    min($0, $1.modifiedAt)
                },
                priority: 1))
        }

        for companion in companionEntries
        where !groupedCompanionHashes.contains(companion.hash)
        {
            groups.append(EvictionGroup(
                sortKey: "ssm:\(companion.hash)",
                kvHashes: [],
                companionHashes: [companion.hash],
                bytes: companion.bytes,
                createdAt: companion.modifiedAt,
                priority: companion.kvHash == nil ? 0 : 1))
        }

        let totalBefore = groups.reduce(Int64(0)) { $0 + $1.bytes }
        guard totalBefore > maxBytes else { return }

        var remaining = totalBefore
        var evictKV = Set<String>()
        var evictCompanion = Set<String>()
        for group in groups.sorted(by: {
            if $0.priority != $1.priority { return $0.priority < $1.priority }
            if $0.createdAt == $1.createdAt { return $0.sortKey < $1.sortKey }
            return $0.createdAt < $1.createdAt
        }) where remaining > maxBytes {
            evictKV.formUnion(group.kvHashes)
            evictCompanion.formUnion(group.companionHashes)
            remaining -= group.bytes
        }

        diskCache.removeQuotaEntries(hashes: evictKV)
        companionStore.removeQuotaEntries(hashes: evictCompanion)

        let legacyCompanionEvicted = companionEntries.reduce(into: 0) { count, entry in
            if entry.kvHash == nil, evictCompanion.contains(entry.hash) {
                count += 1
            }
        }

        if ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1" {
            FileHandle.standardError.write(Data(
                "[vmlx][cache/disk-quota] before=\(totalBefore) after=\(max(0, remaining)) max=\(maxBytes) kvEvicted=\(evictKV.count) companionEvicted=\(evictCompanion.count) legacyCompanionEvicted=\(legacyCompanionEvicted) orphanCompanionEvicted=\(orphaned.count)\n".utf8))
        }
    }

    /// Split full-sequence per-layer KV data into block-sized chunks.
    ///
    /// Each block spans `blockSize` tokens along the sequence dimension (axis 2
    /// for the standard `[B, H, T, D]` layout). The last block may be shorter
    /// if `totalTokens` is not a multiple of `blockSize`.
    ///
    /// Layers that are `nil` (SSM layers without KV data) are skipped in
    /// the output — only layers with actual KV data are included.
    ///
    /// - Parameters:
    ///   - layerData: Per-layer `(keys, values)` for the full sequence, from ``extractLayerData(from:)``.
    ///   - blockSize: Number of tokens per block.
    ///   - totalTokens: Total number of tokens in the sequence.
    /// - Returns: Per-block array of per-layer `(keys, values)` tuples (non-optional, nil layers filtered out).
    private func splitLayerDataIntoBlocks(
        _ layerData: [(keys: MLXArray, values: MLXArray)?],
        blockSize: Int,
        totalTokens: Int
    ) -> [[(keys: MLXArray, values: MLXArray)]] {
        guard totalTokens > 0, !layerData.isEmpty else { return [] }

        var blocks: [[(keys: MLXArray, values: MLXArray)]] = []
        var offset = 0

        while offset < totalTokens {
            let end = min(offset + blockSize, totalTokens)
            var blockData: [(keys: MLXArray, values: MLXArray)] = []

            for kv in layerData {
                guard let kv else { continue }
                // KV tensors are [B, H, T, D] — slice along axis 2 (sequence dim)
                let slicedKeys = kv.keys[.ellipsis, offset ..< end, 0...]
                let slicedValues = kv.values[.ellipsis, offset ..< end, 0...]
                blockData.append((keys: slicedKeys, values: slicedValues))
            }

            blocks.append(blockData)
            offset = end
        }

        return blocks
    }

    // MARK: - Clear

    /// Release only volatile cache tiers for model unload.
    ///
    /// Unloading a model should drop in-memory paged/companion state, but it
    /// must not delete the persistent L2 disk entries. Hosts rely on those
    /// entries to survive model eviction and app restarts for prefix reuse.
    public func releaseVolatile() {
        pagedCache?.clear()
        ssmStateCache.clear()
    }

    /// Clear all cache tiers, releasing all cached data.
    public func clear() {
        releaseVolatile()
        diskCache?.clear()
        ssmStateCache.diskStore?.clear()
    }
}
