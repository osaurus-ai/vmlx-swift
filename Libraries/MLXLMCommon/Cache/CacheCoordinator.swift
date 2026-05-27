// Copyright © 2025 Apple Inc. All rights reserved.

import Foundation
@preconcurrency import MLX
import os

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

    /// Lock protecting `_isHybrid` and `_isPagedIncompatible`.
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
    }

    // MARK: - Hybrid Flag

    /// Set whether the model is hybrid (has both attention and SSM layers).
    ///
    /// When hybrid mode is active, the coordinator will also fetch/store
    /// SSM companion states alongside the KV cache data.
    ///
    /// - Parameter isHybrid: `true` for hybrid models.
    public func setHybrid(_ isHybrid: Bool) {
        lock.withLock { _isHybrid = isHybrid }
    }

    /// Whether the model is hybrid (has both attention and SSM layers).
    public var isHybrid: Bool {
        lock.withLock { _isHybrid }
    }

    /// 2026-05-04: mark the model as paged-incompatible (DSV4 hybrid pool
    /// caches). Forces the coordinator's fetch + store paths to skip the
    /// paged tier so the disk tier (`TQDiskSerializer`) is the only
    /// prefix-reuse mechanism — which is correct for DSV4, where the
    /// cache state can't be reduced to per-token KV blocks.
    public func setPagedIncompatible(_ incompatible: Bool) {
        lock.withLock { _isPagedIncompatible = incompatible }
    }

    /// Whether the model is paged-incompatible (hybrid pool caches).
    public var isPagedIncompatible: Bool {
        lock.withLock { _isPagedIncompatible }
    }

    /// Thread-safe snapshot for diagnostics, UI status, and admin routes.
    public func snapshotStats() -> CacheCoordinatorStatsSnapshot {
        CacheCoordinatorStatsSnapshot(
            pagedEnabled: pagedCache != nil,
            pagedStats: pagedCache?.snapshotStats(),
            diskEnabled: diskCache != nil,
            diskStats: diskCache?.snapshotStats(),
            ssmStats: ssmStateCache.snapshotStats(),
            isHybrid: isHybrid,
            isPagedIncompatible: isPagedIncompatible)
    }

    /// Release paged-cache blocks returned by ``fetch(tokens:mediaSalt:)``.
    ///
    /// Paged hits pin blocks while restore reads `cacheData`; callers must
    /// release those pins as soon as restore has copied tensors into the
    /// live model cache. Disk hits return an empty block list, so this is
    /// a no-op for non-paged tiers.
    public func release(blocks: [CacheBlock]) {
        guard let pagedCache, !blocks.isEmpty else { return }
        for block in blocks {
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
    public func fetch(tokens: [Int], mediaSalt: String? = nil) -> CacheFetchResult {
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
            // Format-v2 disk payloads carry first-class layer state for
            // most path-dependent caches. ZAYA CCA is intentionally excluded
            // until its disk-restore path has separate live proof: restoring
            // a ZAYA prefix from disk without a proven companion boundary can
            // corrupt follow-up required-tool arguments even when the XML
            // wrapper still parses as a tool call. Prefer a full prefill over
            // a false cache hit.
            if let diskArrays, TQDiskSerializer.formatVersion(of: diskArrays) >= 2 {
                return !Self.containsZayaCCADiskPayload(diskArrays)
            }
            return false
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
            var ssmStates: [MLXArray]? = nil
            var canUsePagedHit = true

            if isHybrid {
                ssmStates = fetchCompleteSSMStates(
                    tokens: tokens,
                    boundary: result.matchedTokens,
                    mediaSalt: mediaSalt
                )
                if ssmStates?.isEmpty ?? true {
                    release(blocks: result.blocks)
                    canUsePagedHit = false
                }
            }

            if canUsePagedHit {
                return .hit(
                    matchedTokens: result.matchedTokens,
                    remainingTokens: result.remainingTokens,
                    detail: .paged,
                    blocks: result.blocks,
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
            let preferredBoundaries = [tokens.count, tokens.count - 1]
            for boundary in preferredBoundaries where boundary > 0 {
                tried.insert(boundary)
                if let hit = diskHit(boundary: boundary) {
                    return hit
                }
            }

            for boundary in diskCache.candidateTokenCounts(maxTokens: tokens.count) {
                guard tried.insert(boundary).inserted else { continue }
                if let hit = diskHit(boundary: boundary) {
                    return hit
                }
            }
        }

        // All tiers missed
        return .miss
    }

    private static func containsZayaCCADiskPayload(_ arrays: [String: MLXArray]) -> Bool {
        arrays.keys.contains { key in
            key.hasPrefix("zaya_") || key.contains("_zaya_")
        }
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

        // Split per-layer full-sequence data into per-block chunks.
        let blockLayerData = splitLayerDataIntoBlocks(
            perLayerData, blockSize: blockSize, totalTokens: totalTokens)

        // Store in paged cache (skip when the model is paged-incompatible —
        // see `isPagedIncompatible` above).
        if !isPagedIncompatible, let pagedCache {
            pagedCache.storeTokenSequence(
                tokens: promptTokens, layerData: blockLayerData, mediaSalt: mediaSalt)
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

    /// Clear all cache tiers, releasing all cached data.
    public func clear() {
        pagedCache?.clear()
        diskCache?.clear()
        ssmStateCache.clear()
    }
}
