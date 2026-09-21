// Copyright © 2025 Apple Inc. All rights reserved.

import Foundation
@preconcurrency import MLX
import os

/// Serializes process-wide combined disk-quota reconciliation. Individual KV
/// and companion stores already own their IO locks; this lock only protects
/// the cross-store snapshot/eviction decision.
private enum CombinedDiskCacheQuotaLock {
    static let shared = OSAllocatedUnfairLock()

    /// Cache roots whose companion directory this process has imported into
    /// the index with a COMMITTED import. A root whose import was skipped is
    /// not in here, so the next coordinator on it — and the quota pass of
    /// every coordinator already on it — tries again. Read and written only
    /// while `shared` is held.
    nonisolated(unsafe) static var importedRoots = Set<String>()

    /// When an import of a root last failed to commit. Only roots that are
    /// not in `importedRoots` have an entry. Paces the quota pass's retry;
    /// same lock.
    nonisolated(unsafe) static var lastUncommittedImport: [String: Date] = [:]
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

    private let initialConfig: CacheCoordinatorConfig

    /// Creation settings with the current shared disk quota. A live size
    /// change must be visible to callers reporting the active runtime policy.
    public var config: CacheCoordinatorConfig {
        var current = initialConfig
        if let diskCache {
            current.diskCacheMaxGB = Float(Double(diskCache.maxSizeBytes) / 1_073_741_824)
        }
        return current
    }

    /// Applies to every resident coordinator on this root without unloading
    /// models. Raises take effect immediately; decreases are enforced on the
    /// next serialized store, so a settings edit never deletes synchronously.
    public func updateDiskCap(bytes: Int) {
        CombinedDiskCacheQuotaLock.shared.lock()
        defer { CombinedDiskCacheQuotaLock.shared.unlock() }
        diskCache?.updateMaxSizeBytes(bytes)
    }

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
    private var _requiresSeparateRecurrentPayload: Bool = false

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

    /// Pacing of the import retry; see ``retryCompanionImportIfDueLocked``.
    private let importRetryInterval: TimeInterval
    private let now: @Sendable () -> Date

    /// The disk cache root as `importedRoots` keys it, computed once: the
    /// quota pass looks it up on every store.
    private let diskRootKey: String?

    // MARK: - Initialization

    /// Creates a new cache coordinator.
    ///
    /// Sub-caches are instantiated based on the configuration flags.
    ///
    /// - Parameter config: The cache configuration to use.
    public convenience init(config: CacheCoordinatorConfig = CacheCoordinatorConfig()) {
        self.init(config: config, diskIndexBusyTimeoutMs: DiskCache.defaultIndexBusyTimeoutMs)
    }

    /// How long a quota pass waits before it tries an import that did not
    /// commit again. See ``retryCompanionImportIfDueLocked``.
    static let defaultImportRetryInterval: TimeInterval = 60

    /// `diskIndexBusyTimeoutMs` is how long the disk index waits for another
    /// connection's write lock, and `diskIndexMigrationBusyTimeoutMs` how
    /// long its migration does. Production always uses the defaults; a test
    /// that holds the lock on purpose passes short ones. `importRetryInterval`
    /// and `now` pace the retry of an import that did not commit; a test
    /// passes its own clock instead of sleeping.
    init(
        config: CacheCoordinatorConfig, diskIndexBusyTimeoutMs: Int32,
        diskIndexMigrationBusyTimeoutMs: Int32 = DiskCacheIndexSchema.defaultBusyTimeoutMs,
        importRetryInterval: TimeInterval = CacheCoordinator.defaultImportRetryInterval,
        now: @escaping @Sendable () -> Date = { Date() }
    ) {
        self.initialConfig = config
        self.importRetryInterval = importRetryInterval
        self.now = now

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
            self.diskCache = DiskCache(
                cacheDir: dir,
                maxSizeBytes: DiskCacheCapPolicy.byteLimit(gigabytes: config.diskCacheMaxGB),
                modelKey: config.modelKey,
                indexMigrationBusyTimeoutMs: diskIndexMigrationBusyTimeoutMs,
                indexBusyTimeoutMs: diskIndexBusyTimeoutMs,
                now: now)
        } else {
            self.diskCache = nil
        }
        self.diskRootKey = self.diskCache?.cacheDir.standardizedFileURL.path

        self.ssmStateCache = SSMStateCache(
            maxEntries: config.ssmMaxEntries,
            modelKey: config.modelKey)

        if config.enableDiskCache {
            let baseDir = config.diskCacheDir
                ?? FileManager.default.temporaryDirectory
                    .appendingPathComponent("vmlx_disk_cache")
            let ssmDir = baseDir.appendingPathComponent(DiskCache.companionDirectoryName)
            let ssmMaxBytes = max(
                1,
                DiskCacheCapPolicy.byteLimit(gigabytes: config.diskCacheMaxGB))
            self.ssmStateCache.diskStore = try? SSMCompanionDiskStore(
                cacheDir: ssmDir,
                modelKey: config.modelKey,
                maxBytes: ssmMaxBytes,
                sweepUnpublishedAtOpen: !(self.diskCache?.indexIsFromANewerBuild ?? false),
                rootIndexIsFromANewerBuild: self.diskCache?.indexIsFromANewerBuild ?? false,
                sharedQuotaRoot: baseDir)
        }

        importCompanionAccountingOncePerRoot()
        enforceCombinedDiskQuota()
    }

    /// Whether quota and stats read companion bytes from the index. False on
    /// an index without the v2 columns, where both keep walking the directory.
    private var companionBytesAreIndexed: Bool {
        diskCache?.indexHasV2Columns == true && ssmStateCache.diskStore != nil
    }

    /// Companions are files outside the index, so the first coordinator to
    /// open a root in this process walks the companion directory once and
    /// writes what it finds into the index; after that the index is kept in
    /// step by the stores themselves and nothing walks the directory again.
    /// The same pass repairs an index an older build has written to since.
    ///
    /// "Once" means once COMMITTED. An import that could not take the index
    /// write lock (another connection held it past the busy timeout) leaves
    /// the root unmarked. The next coordinator to open it tries again, and so
    /// does the quota pass of this one (``retryCompanionImportIfDueLocked``);
    /// until one of them commits, an upgraded directory's companions are not
    /// counted.
    private func importCompanionAccountingOncePerRoot() {
        guard let diskCache, diskCache.indexHasV2Columns,
              let companionStore = ssmStateCache.diskStore, let root = diskRootKey
        else { return }
        companionStore.attachLedger(diskCache)

        CombinedDiskCacheQuotaLock.shared.lock()
        defer { CombinedDiskCacheQuotaLock.shared.unlock() }
        guard !CombinedDiskCacheQuotaLock.importedRoots.contains(root) else { return }
        attemptCompanionImportLocked(
            root: root, diskCache: diskCache, companionStore: companionStore, reason: "open")
    }

    /// One import attempt and its bookkeeping: a committed import marks the
    /// root done; one that did not commit unmarks it and notes when, so the
    /// quota pass knows how long to leave it. Returns whether it committed.
    /// Caller holds ``CombinedDiskCacheQuotaLock``.
    @discardableResult
    private func attemptCompanionImportLocked(
        root: String, diskCache: DiskCache, companionStore: SSMCompanionDiskStore,
        reason: String
    ) -> Bool {
        let committed = importCompanionAccountingLocked(
            diskCache: diskCache, companionStore: companionStore, reason: reason)
        if committed {
            CombinedDiskCacheQuotaLock.importedRoots.insert(root)
            CombinedDiskCacheQuotaLock.lastUncommittedImport[root] = nil
        } else {
            CombinedDiskCacheQuotaLock.importedRoots.remove(root)
            CombinedDiskCacheQuotaLock.lastUncommittedImport[root] = now()
        }
        return committed
    }

    /// An import that did not commit leaves this root's pre-existing
    /// companions uncounted, and in a session with one model no other
    /// coordinator will ever open the root to try again. So the quota pass —
    /// which runs on every store, already under the combined lock — retries
    /// it, at most once per `importRetryInterval` per root: an index that
    /// stays locked costs one directory walk and one busy wait per interval,
    /// not per store. For an imported root this is one set lookup.
    /// Caller holds ``CombinedDiskCacheQuotaLock``.
    private func retryCompanionImportIfDueLocked(
        diskCache: DiskCache, companionStore: SSMCompanionDiskStore
    ) {
        guard let root = diskRootKey,
              !CombinedDiskCacheQuotaLock.importedRoots.contains(root)
        else { return }
        if let last = CombinedDiskCacheQuotaLock.lastUncommittedImport[root] {
            let elapsed = now().timeIntervalSince(last)
            // A clock that went backwards must not postpone the retry.
            guard elapsed < 0 || elapsed >= importRetryInterval else { return }
        }
        attemptCompanionImportLocked(
            root: root, diskCache: diskCache, companionStore: companionStore, reason: "retry")
    }

    /// One walk of the companion directory, reconciled into the index.
    /// Returns whether it committed. Caller holds
    /// ``CombinedDiskCacheQuotaLock``.
    private func importCompanionAccountingLocked(
        diskCache: DiskCache, companionStore: SSMCompanionDiskStore, reason: String
    ) -> Bool {
        // A directory that cannot be listed is not an empty one: importing
        // "no companions" would commit, mark the root done, and leave every
        // companion in it uncounted for good. Not committed; retried later.
        guard let companions = companionStore.listedQuotaEntries() else {
            FileHandle.standardError.write(Data(
                ("[vmlx][cache/disk-index] companion import reason=\(reason) committed=false "
                    + "companion directory could not be listed or examined\n").utf8))
            return false
        }
        // The directory goes in with the entries that were listed from it:
        // `ssmStateCache.diskStore` is a public var, and the re-check inside
        // the transaction must look where this walk looked.
        let summary = diskCache.reconcileCompanionAccounting(
            companions: companions, companionDirectory: companionStore.directory)
        // A skipped import is always reported; a committed one only under
        // the trace flag, and only when it changed something.
        let traced = ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1"
        if summary == nil || (traced && summary?.changedAnything == true) {
            let counts = summary ?? DiskCacheCompanionImportSummary()
            FileHandle.standardError.write(Data(
                ("[vmlx][cache/disk-index] companion import reason=\(reason) "
                    + "committed=\(summary != nil) "
                    + "rowsDeletedForMissingPayload=\(counts.rowsDeletedForMissingPayload) "
                    + "linksWritten=\(counts.linksWritten) linksCleared=\(counts.linksCleared) "
                    + "legacyUpserted=\(counts.legacyUpserted) "
                    + "legacyDeleted=\(counts.legacyDeleted) "
                    + "rowsDroppedForInvalidHash=\(counts.rowsDroppedForInvalidHash) "
                    + "legacyDroppedForInvalidKey=\(counts.legacyDroppedForInvalidKey) "
                    + "unindexedPayloadsRemoved=\(counts.unindexedPayloadsRemoved)\n").utf8))
        }
        return summary != nil
    }

    /// Bring the disk accounting back in line with the directory after
    /// something outside this package changed it — the host's "Clear SSD
    /// Cache", which deletes payloads, companion files and `cache_entries`
    /// rows with its own SQL and knows nothing about `legacy_companions`.
    /// Without this the index keeps counting companions whose files are gone,
    /// and does not count companion files that lost their row, until the
    /// next launch.
    ///
    /// Walks the companion directory once (the same import that runs at
    /// open), drops rows whose payload is gone, removes payloads that have
    /// no row and were last modified at least ten minutes ago, and forgets
    /// which files this process had validated. The ten minutes are the
    /// file's age, not how long it has been without a row: a week-old
    /// payload whose row was deleted a second ago goes at once. Only regular
    /// files named like this cache's payloads are ever removed, and none at
    /// all in a root that holds `config.json` / `jang_config.json` or under
    /// an index a newer build has claimed.
    ///
    /// Deleting `cache_index.db` by hand therefore does not keep the
    /// payloads: every one of them is then without a row — not served, and
    /// removed by the next import once it is old enough.
    ///
    /// Returns false when the index write lock could not be taken, or when
    /// a directory or a file the import has to look at could not be examined
    /// (which is never read as "not there"). The index
    /// is then unchanged — the validated sets HAVE been cleared, which only
    /// costs a re-validation — and the call can be repeated. It does not
    /// have to be: the root stops counting as imported, so this
    /// coordinator's quota pass retries within a minute of its next store.
    /// On an index without the companion columns there is nothing to
    /// reconcile — quota and stats walk the directory there — and the result
    /// is true.
    ///
    /// Call it off the main thread, and once per resident coordinator: it
    /// takes the lock every store holds for its whole write, so it waits
    /// behind an in-flight store; it lists the companion directory and the
    /// payload directory and reads every sidecar; and it can wait out the
    /// index busy timeout.
    @discardableResult
    public func reconcileDiskAccounting() -> Bool {
        guard let diskCache else { return true }
        CombinedDiskCacheQuotaLock.shared.lock()
        defer { CombinedDiskCacheQuotaLock.shared.unlock() }

        diskCache.forgetValidatedFiles()
        ssmStateCache.diskStore?.forgetValidatedEntries()
        guard diskCache.indexHasV2Columns, let companionStore = ssmStateCache.diskStore,
              let root = diskRootKey
        else { return true }

        return attemptCompanionImportLocked(
            root: root, diskCache: diskCache, companionStore: companionStore,
            reason: "on-demand")
    }

    /// Forget which roots were imported, so a test can stand in for a new
    /// process opening the same directory.
    static func resetImportedRootsForTesting() {
        CombinedDiskCacheQuotaLock.shared.lock()
        CombinedDiskCacheQuotaLock.importedRoots.removeAll()
        CombinedDiskCacheQuotaLock.lastUncommittedImport.removeAll()
        CombinedDiskCacheQuotaLock.shared.unlock()
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
        requiresRecurrentSSMCompanion: Bool? = nil,
        requiresSeparateRecurrentPayload: Bool? = nil
    ) {
        lock.withLock {
            _isHybrid = isHybrid
            _requiresRecurrentSSMCompanion = isHybrid
                ? (requiresRecurrentSSMCompanion ?? true)
                : false
            // Callers that don't know the payload topology inherit the
            // companion flag — the conservative pre-split behavior (persist
            // and require the separate recurrent payload).
            _requiresSeparateRecurrentPayload = isHybrid
                ? (requiresSeparateRecurrentPayload
                    ?? _requiresRecurrentSSMCompanion)
                : false
        }
    }

    /// Whether the model is hybrid (has both attention and SSM layers).
    public var isHybrid: Bool {
        lock.withLock { _isHybrid }
    }

    /// Whether exact-boundary persistence must preserve recurrent SSM/GDN
    /// state. Native disk payloads may carry it without a separate sidecar.
    public var requiresRecurrentSSMCompanion: Bool {
        lock.withLock { _requiresRecurrentSSMCompanion }
    }

    /// Whether stores must persist recurrent state as a separate payload
    /// (folded `ssm_*` + companion sidecar) because the v2 layer
    /// serialization cannot round-trip it natively (ArraysCache/GDN).
    /// MambaCache state round-trips in-file, so topologies without an
    /// ArraysCache layer skip both extra copies. See
    /// `ModelCacheTopologySnapshot.requiresSeparateRecurrentPayloadState`.
    public var requiresSeparateRecurrentPayload: Bool {
        lock.withLock { _requiresSeparateRecurrentPayload }
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
            diskStats: combinedDiskStatsSnapshot(),
            ssmStats: ssmStateCache.snapshotStats(),
            isHybrid: isHybrid,
            isPagedIncompatible: isPagedIncompatible,
            requiresPagedBoundaryCompanion: requiresPagedBoundaryCompanion)
    }

    /// Snapshot the complete L2 quota footprint. `DiskCache` owns the public
    /// counter surface, while the coordinator folds linked recurrent sidecars
    /// into byte usage and treats each KV + companion group as one logical
    /// entry. Holding the combined lock keeps this view coherent with atomic
    /// linked stores and evictions.
    ///
    /// Hosts poll this every few seconds per idle window, under the lock every
    /// store needs. With companion bytes in the index it is two SQL
    /// aggregates; only an index without the v2 columns still walks the
    /// companion directory.
    private func combinedDiskStatsSnapshot() -> DiskCacheStats? {
        guard let diskCache else { return nil }

        CombinedDiskCacheQuotaLock.shared.lock()
        defer { CombinedDiskCacheQuotaLock.shared.unlock() }

        if companionBytesAreIndexed {
            return diskCache.snapshotStatsIncludingCompanions()
        }

        let base = diskCache.snapshotStats()
        guard let companionStore = ssmStateCache.diskStore else { return base }

        // A poll is a read: it retires nothing, so with a record that names
        // nothing in the index it opens no write transaction — every few
        // seconds, per window, under the lock every store needs.
        let kvHashes = Set(diskCache.quotaEntries(retiringInvalidRecords: false).map(\.hash))
        let companionEntries = companionStore.quotaEntries()
        let companionBytes = IndexedBytes.total(companionEntries.lazy.map(\.bytes))
        let unlinkedCompanionCount = companionEntries.reduce(into: 0) { count, entry in
            if entry.kvHash.map({ !kvHashes.contains($0) }) ?? true {
                count += 1
            }
        }

        return base.replacingUsage(
            currentPayloadBytes: IndexedBytes.asInt(
                IndexedBytes.sum(Int64(base.currentPayloadBytes), companionBytes)),
            currentEntryCount: base.currentEntryCount + unlinkedCompanionCount)
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
    /// `chainId` is the conversation asking (see
    /// ``GenerateParameters/cacheChainId``): a disk hit on an unowned history
    /// row hands that row to it. The id never affects which entry is chosen.
    public func fetch(
        tokens: [Int],
        mediaSalt: String? = nil,
        skipExactDiskBoundary: Bool = false,
        preferredDiskBoundaries: [Int] = [],
        chainId: String? = nil
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
            // ArraysCache topologies require the separately keyed
            // prompt-boundary sidecar: v2 has no LayerKind for GDN state, so
            // a format-v2 marker is not evidence those layers are complete.
            // MambaCache state round-trips in the v2 payload itself
            // (`mamba_{i}_state0/1`, applied by deserializeV2), so those
            // topologies accept a v2 entry without any companion — matching
            // the store side, which no longer persists the redundant copies
            // for them. ZAYA CCA is topology-classified with this flag false
            // because its v2 layer payload atomically owns KV + conv +
            // previous-hidden state.
            return !requiresSeparateRecurrentPayload
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
                // A deserializable KV payload is only a candidate until
                // architecture-specific companion state is validated below.
                // Do not make rejected hybrid candidates hot: no recency
                // touch and no hit until one is accepted.
                let arrays: [String: MLXArray]
                switch diskCache.fetchCandidate(tokens: prefix, mediaSalt: mediaSalt) {
                case .arrays(let found):
                    arrays = found
                case .miss:
                    // A miss here means the content-addressed key over this
                    // prefix found no row; a rejection below means the row
                    // existed but its companion state was refused. Only the
                    // aggregate "MISS all tiers" was ever traced, which cannot
                    // tell those apart — and a silent companion veto has
                    // already cost a whole family (LFM2.5) its cache once.
                    ftrace("probe boundary=\(boundary) noRow")
                    return nil
                case .restoreRejectedEarlier:
                    // See `reportDiskRestoreRejected`: serving it again would
                    // end in the same full prefill, with a shorter entry that
                    // does restore still waiting behind it.
                    ftrace("probe boundary=\(boundary) skipped: restore was rejected earlier")
                    return nil
                }
                let ssmStates = resolveSSMStates(
                    forTokens: prefix,
                    boundary: boundary,
                    diskArrays: arrays,
                    mediaSalt: mediaSalt,
                    chainId: chainId)
                if hasRequiredHybridSSM(ssmStates, diskArrays: arrays) {
                    touchSuccessfulDiskRestore(
                        matchedTokens: prefix,
                        matchedBoundary: boundary,
                        mediaSalt: mediaSalt,
                        chainId: chainId)
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
                ftrace(
                    "probe boundary=\(boundary) rowFound but companion REJECTED "
                        + "ssm=\(ssmStates?.count ?? -1)")
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

            // Read the index in bounded keyset pages, then sort the complete
            // candidate set before probing. A single 128-length window can hide
            // the longest valid prefix below unrelated larger entries from
            // other prompts, models, or media. `maxTokens` is an inclusive
            // cursor, so advancing to the smallest returned count minus one
            // avoids OFFSET scans while guaranteeing forward progress.
            let candidatePageSize = 128
            var indexedBoundaries: [Int] = []
            var candidateMaximum = tokens.count
            while candidateMaximum > 0 {
                let page = diskCache.candidateTokenCounts(
                    maxTokens: candidateMaximum,
                    limit: candidatePageSize)
                guard !page.isEmpty else { break }
                indexedBoundaries.append(contentsOf: page)

                guard page.count == candidatePageSize,
                      let smallest = page.last,
                      smallest > 1
                else { break }
                let nextMaximum = smallest - 1
                guard nextMaximum < candidateMaximum else { break }
                candidateMaximum = nextMaximum
            }

            // Merge processor-proven boundaries back into the globally sorted
            // probe set; each remains content-address checked against this
            // request's exact model/media/token prefix.
            // A path-dependent hybrid restore cannot consume an exact prompt
            // boundary without an N-1 recurrent seed. Stable system/tool
            // checkpoints are therefore persisted one token short. Keep that
            // processor-proven seed in the probe set even if its index row is
            // missing or has not yet appeared to this connection.
            let preferredSafeSeeds = skipExactDiskBoundary
                ? preferredDiskBoundaries.compactMap { boundary in
                    boundary > 1 ? boundary - 1 : nil
                }
                : []
            let candidateBoundaries = Set(
                indexedBoundaries
                    + preferredDiskBoundaries
                    + preferredSafeSeeds
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

    /// Tell the coordinator that a disk hit it served could not be restored.
    ///
    /// `fetch` accepts a disk candidate as soon as its payload deserializes
    /// and its companion state is present: that is the moment the hit is
    /// counted and the entry's recency refreshed. Whether the payload FITS
    /// the running model's cache is only known to the engine, afterwards —
    /// `restoreFromDiskArrays` restores 0 tokens for a payload whose layers
    /// do not match the cache, and `validateRestoredCacheBoundary` refuses
    /// offsets that disagree with the boundary. The engine then prefills the
    /// whole prompt. Left untold, the coordinator serves the same entry on
    /// every later turn, ahead of every shorter entry that would restore,
    /// and — the fetch having validated the file — treats the boundary as
    /// durable, so nothing ever writes it again.
    ///
    /// Call this for a STRUCTURAL rejection of a `.disk` hit only: the entry
    /// cannot be used by this model as it runs now. Do not call it when the
    /// entry was fine and the request could not use it (a missing seed state
    /// for an exact hit, media placeholders left in the suffix).
    ///
    /// Everything it does is in memory, for this coordinator, until the
    /// process ends; nothing is deleted and recency is left as it is:
    ///
    /// - `fetch` passes over the entry, so the longest entry that does
    ///   restore wins — from the NEXT fetch on: nothing is fetched again in
    ///   the turn that found out, which prefills the whole prompt;
    /// - the entry counts as neither validated nor durable, so a store of
    ///   that boundary writes it again, and that store lifts the mark (as
    ///   does a payload another process has replaced). Whether such a store
    ///   comes depends on the turn: only a boundary in this turn's store set
    ///   is written — the prompt itself (an exact re-send), a stable prefix,
    ///   a ladder rung. A refused boundary from earlier in the history is in
    ///   no later turn's set, and just stays passed over until the process
    ///   ends;
    /// - that rewrite happens once per entry per process. If what was
    ///   written is refused as well, the entry stays passed over and counts
    ///   as durable, so it is not written again
    ///   (``DiskCacheStats/rejectedRewritesSuppressed``);
    /// - the hit is taken back out of ``DiskCacheStats/hits`` and counted in
    ///   ``DiskCacheStats/rejectedDiskRestores``.
    ///
    /// A report for an entry that is not there, or whose payload is no
    /// longer the one `fetch` served, changes nothing.
    ///
    /// - Parameters:
    ///   - tokens: The token sequence that was fetched.
    ///   - boundary: `matchedTokens` of the hit: the entry is keyed by
    ///     `tokens.prefix(boundary)`.
    ///   - mediaSalt: The media salt of the fetch.
    ///   - reason: What the engine found; traced, not interpreted.
    public func reportDiskRestoreRejected(
        tokens: [Int],
        boundary: Int,
        mediaSalt: String?,
        reason: String
    ) {
        guard let diskCache, boundary > 0, boundary <= tokens.count else { return }
        let prefix = boundary == tokens.count ? tokens : Array(tokens.prefix(boundary))
        let marked = diskCache.markRestoreRejected(tokens: prefix, mediaSalt: mediaSalt)
        if marked || ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1" {
            // Said once per payload: it explains a full prefill after a hit.
            FileHandle.standardError.write(Data(
                ("[vmlx][cache/restore] disk restore rejected boundary=\(boundary) "
                    + "tokens=\(tokens.count) marked=\(marked) reason=\(reason)\n").utf8))
        }
    }

    /// True only after the current process has deserialized or written the
    /// exact L2 entry and its on-disk fingerprint still matches the index.
    public func hasValidatedDiskEntry(
        tokens: [Int],
        mediaSalt: String? = nil
    ) -> Bool {
        let needsSeparate = isHybrid && requiresSeparateRecurrentPayload
        let needsNative = isHybrid && requiresRecurrentSSMCompanion && !needsSeparate
        guard diskCache?.hasValidatedEntry(
            tokens: tokens, mediaSalt: mediaSalt,
            requireNativeRecurrent: needsNative) == true
        else {
            return false
        }
        if needsSeparate {
            return ssmStateCache.hasValidatedCompleteDiskEntry(
                tokens: tokens,
                boundary: tokens.count,
                mediaSalt: mediaSalt)
        }
        return true
    }

    /// Whether this boundary is already durable on disk, whoever wrote it.
    ///
    /// Used to decide whether a boundary needs producing at all, as opposed to
    /// whether a rewrite can be skipped. See `DiskCache.hasDurableEntry`.
    public func hasDurableDiskEntry(
        tokens: [Int],
        mediaSalt: String? = nil
    ) -> Bool {
        // Match the disk writer's ownership contract. Mamba state is already
        // native in-file; ArraysCache and unknown topologies still need their
        // complete sidecar. A paged hit has its own companion admission gate:
        // these predicates certify only the durable disk boundary.
        let needsSeparate = isHybrid && requiresSeparateRecurrentPayload
        let needsNative = isHybrid && requiresRecurrentSSMCompanion && !needsSeparate
        guard diskCache?.hasDurableEntry(
            tokens: tokens, mediaSalt: mediaSalt,
            requireNativeRecurrent: needsNative) == true
        else { return false }
        if needsSeparate {
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
        mediaSalt: String? = nil,
        chainId: String? = nil
    ) -> [MLXArray]? {
        guard isHybrid else { return nil }
        if let l1 = fetchCompleteSSMStates(
            tokens: tokens,
            boundary: boundary,
            mediaSalt: mediaSalt,
            touchDiskRecency: false)
        {
            return l1
        }
        guard let folded = TQDiskSerializer.ssmStates(from: diskArrays) else {
            return nil
        }
        // A read is not the moment to spend the reader's own boundaries. This
        // write-back runs a quota pass like any store, and without the reading
        // conversation named it would spend the COLDEST chain's superseded
        // rows — which is the chat being read precisely when the user has just
        // returned to an older one.
        storePersistentBoundary(
            tokens: tokens,
            diskArrays: nil,
            ssmStates: folded,
            mediaSalt: mediaSalt,
            chainId: chainId)
        return folded
    }

    /// Fetch companion SSM state only when the stored boundary is safe to
    /// extend. Partial entries represent mid-prefill snapshots and must not
    /// satisfy prefix reuse for a later growing turn.
    private func fetchCompleteSSMStates(
        tokens: [Int],
        boundary: Int,
        mediaSalt: String? = nil,
        touchDiskRecency: Bool = true
    ) -> [MLXArray]? {
        guard let entry = ssmStateCache.fetchEntry(
            tokens: tokens,
            boundary: boundary,
            mediaSalt: mediaSalt,
            touchDiskRecency: touchDiskRecency,
            requireComplete: true)
        else {
            return nil
        }
        return entry.isComplete ? entry.states : nil
    }

    /// Refresh an accepted disk restore's eviction recency.
    ///
    /// Candidate KV payloads are deliberately fetched without a recency touch
    /// because hybrid validation may reject them for missing or incomplete
    /// companion state. Dense restores touch KV only after acceptance; hybrid
    /// restores touch KV and their linked recurrent companion with one
    /// timestamp so quota observes the pair atomically. Stable checkpoints are
    /// deliberately excluded here because the runtime has not yet proved that
    /// it retained the fetched arrays.
    private func touchSuccessfulDiskRestore(
        matchedTokens: [Int],
        matchedBoundary: Int,
        mediaSalt: String?,
        chainId: String? = nil
    ) {
        guard let diskCache else { return }
        let recency = Date()
        CombinedDiskCacheQuotaLock.shared.lock()
        defer { CombinedDiskCacheQuotaLock.shared.unlock() }

        _ = diskCache.touchRecency(
            tokens: matchedTokens,
            mediaSalt: mediaSalt,
            at: recency)
        if let chainId {
            diskCache.assignChain(tokens: matchedTokens, mediaSalt: mediaSalt, chainId: chainId)
        }
        diskCache.recordAcceptedHit()
        if isHybrid {
            _ = ssmStateCache.diskStore?.touchRecency(
                tokens: matchedTokens,
                boundary: matchedBoundary,
                mediaSalt: mediaSalt,
                at: recency)
        }
    }

    /// Refresh only processor-proven stable checkpoints after a caller has
    /// retained a disk restore. Calling this before architecture-specific
    /// restore validation can keep an entry hot even when the runtime rolls
    /// back to full prefill, so every production caller invokes it only from
    /// its retained `diskArrays` branch.
    ///
    /// Path-dependent hybrid caches persist the stable checkpoint one token
    /// short, matching the safe-seed rule used by fetch/store. Every touch is
    /// content-addressed with the active model/media namespace and updates only
    /// an existing entry; no missing or incompatible checkpoint is invented.
    func touchStableDiskCheckpointsAfterRetainedRestore(
        requestTokens: [Int],
        matchedTokenCount: Int,
        preferredDiskBoundaries: [Int],
        skipExactDiskBoundary: Bool,
        mediaSalt: String?
    ) {
        guard let diskCache,
              matchedTokenCount > 0,
              matchedTokenCount <= requestTokens.count
        else { return }
        let recency = Date()
        CombinedDiskCacheQuotaLock.shared.lock()
        defer { CombinedDiskCacheQuotaLock.shared.unlock() }

        let stableBoundaries = Set(preferredDiskBoundaries.compactMap { boundary in
            let storedBoundary = skipExactDiskBoundary ? boundary - 1 : boundary
            return storedBoundary > 0 && storedBoundary <= matchedTokenCount
                ? storedBoundary
                : nil
        })
        for boundary in stableBoundaries {
            let tokens = boundary == requestTokens.count
                ? requestTokens
                : Array(requestTokens.prefix(boundary))
            let touchedKV = diskCache.touchRecency(
                tokens: tokens,
                mediaSalt: mediaSalt,
                at: recency)
            if touchedKV, isHybrid {
                _ = ssmStateCache.diskStore?.touchRecency(
                    tokens: tokens,
                    boundary: boundary,
                    mediaSalt: mediaSalt,
                    at: recency)
            }
        }
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
    /// `chainId` / `isStableRoot` are the row's owner and kind for the quota
    /// planner (see ``GenerateParameters/cacheChainId``); they never enter
    /// the content key.
    public func storeAfterGeneration(
        promptTokens: [Int],
        perLayerData: [(keys: MLXArray, values: MLXArray)?],
        ssmStates: [MLXArray]?,
        cache: [any KVCache]? = nil,
        mediaSalt: String? = nil,
        chainId: String? = nil,
        isStableRoot: Bool = false,
        isResumeBoundary: Bool = false
    ) {
        let totalTokens = promptTokens.count
        let blockSize = config.pagedBlockSize
        let traceCacheStore =
            ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1"

        // A boundary entry asserts "this cache covers exactly these tokens".
        // The paged tier already enforces that for its rotating companion —
        // `serializePagedRotatingCompanion(expectedOffset:)` returns nil on a
        // mismatch, observed live as `companion=false rotatingOffsets=[1832]`
        // under `tokens=1831`. The disk tier had no equivalent guard, so the
        // very same poisoned snapshot was persisted to SSD.
        //
        // A later fetch content-matches the key, restores one more token of
        // state than the key claims, and the engine re-feeds the boundary
        // token the cache already contains. Every subsequent position shifts
        // by one, attention is silently wrong for the whole continuation, and
        // each following turn stores a new seed derived FROM that corrupted
        // state — compounding per turn. Live signature: fluent DSV4 agent
        // turns degenerating into non-converging loops, only with disk L2
        // enabled.
        //
        // Refuse the whole store instead: a skipped entry costs one prefill,
        // a poisoned entry costs correctness for every later turn.
        if let cache {
            let offsets = Set(cache.map(\.offset))
            if offsets.contains(where: { $0 != totalTokens }) {
                if traceCacheStore {
                    FileHandle.standardError.write(Data(
                        ("[vmlx][cache/store] REFUSED offset/key mismatch"
                            + " tokens=\(totalTokens) offsets=\(offsets.sorted())\n").utf8))
                }
                return
            }
        }

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

        if traceCacheStore {
            let rotatingOffsets = cache?.compactMap {
                ($0 as? RotatingKVCache)?.offset
            } ?? []
            let uniqueRotatingOffsets = Array(Set(rotatingOffsets)).sorted()
            let effectiveKVLayerCount = effectivePerLayerData.compactMap { $0 }.count
            let message = "[vmlx][cache/paged-store] tokens=\(totalTokens) "
                + "requiredCompanion=\(requiresPagedBoundaryCompanion) "
                + "effectiveKVLayers=\(effectiveKVLayerCount) "
                + "blocks=\(blockLayerData.count) payload=\(hasPagedKVPayload) "
                + "companion=\(pagedBoundaryCompanion != nil) "
                + "rotatingOffsets=\(uniqueRotatingOffsets)\n"
            FileHandle.standardError.write(Data(message.utf8))
        }

        // Store in paged cache (skip when the model is paged-incompatible —
        // see `isPagedIncompatible` above). Recurrent-only/state-only caches
        // must not publish token hashes without any restorable KV payload;
        // doing so would suppress the valid typed disk fallback on fetch.
        var publishedPagedPayload = false
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
            publishedPagedPayload = true
        }

        // Build the disk payload before entering the linked-store transaction.
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
        // Persist the separate recurrent payload (folded `ssm_*` + companion
        // sidecar) only when something actually consumes it:
        //  - ArraysCache/GDN topologies — the v2 layer serialization has no
        //    LayerKind for that state, so the companion is its only carrier;
        //  - a hybrid boundary that just published a PAGED payload — paged
        //    blocks carry KV only, and the tier-1 hit path rejects hybrid
        //    hits without `fetchCompleteSSMStates`.
        // For disk-only MambaCache hybrids, the state round-trips in-file as
        // `mamba_{i}_state0/1`; the extra copies tripled the recurrent bytes
        // per stored boundary (~+300MB each on Qwen3.8-27B) without the
        // restore path ever applying them.
        let persistSeparateRecurrentPayload =
            isHybrid && (requiresSeparateRecurrentPayload || publishedPagedPayload)
        var diskArrays: [String: MLXArray]?
        if diskCache != nil {
            if let cache {
                let arrays = TQDiskSerializer.serialize(
                    cache: cache,
                    ssmStates: persistSeparateRecurrentPayload ? ssmStates : nil,
                    preserveStandardKVStorageDType: config.preserveStandardKVStorageDType)
                if !arrays.isEmpty {
                    diskArrays = arrays
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
                    diskArrays = arrays
                }
            }
        }

        storePersistentBoundary(
            tokens: promptTokens,
            diskArrays: diskArrays,
            ssmStates: persistSeparateRecurrentPayload ? ssmStates : nil,
            mediaSalt: mediaSalt,
            chainId: chainId,
            isStableRoot: isStableRoot,
            isResumeBoundary: isResumeBoundary)
    }

    /// Persist one reusable prompt boundary as a linked transaction.
    ///
    /// `DiskCache` and `SSMCompanionDiskStore` remain independently usable,
    /// but applying each store's full quota while a hybrid boundary is only
    /// half-written can evict an older usable prefix, admit a newer KV/SSM
    /// half, then have the combined pass evict that oversized new pair too.
    /// Holding the cross-store lock and deferring both standalone quota passes
    /// ensures the final eviction decision sees complete linked groups.
    func storePersistentBoundary(
        tokens: [Int],
        diskArrays: [String: MLXArray]?,
        ssmStates: [MLXArray]?,
        mediaSalt: String? = nil,
        chainId: String? = nil,
        isStableRoot: Bool = false,
        isResumeBoundary: Bool = false
    ) {
        let usesCombinedQuota = config.enableDiskCache
            && diskCache != nil
            && ssmStateCache.diskStore != nil
        if usesCombinedQuota {
            CombinedDiskCacheQuotaLock.shared.lock()
        }
        defer {
            if usesCombinedQuota {
                CombinedDiskCacheQuotaLock.shared.unlock()
            }
        }

        let storesCompanion = isHybrid && !(ssmStates?.isEmpty ?? true)
        if let diskArrays, !diskArrays.isEmpty {
            diskCache?.store(
                tokens: tokens,
                arrays: diskArrays,
                mediaSalt: mediaSalt,
                enforceQuota: !usesCombinedQuota,
                chainId: chainId,
                isStableRoot: isStableRoot,
                isResumeBoundary: isResumeBoundary)
            if !storesCompanion {
                adoptEarlyCompanion(tokens: tokens, mediaSalt: mediaSalt)
            }
        }

        // KV first: the companion store reports what it wrote to the index
        // before it returns (still inside this critical section), and links
        // onto the row the KV store just wrote. With no row to link to —
        // `resolveSSMStates` passes no KV payload, and its row normally exists
        // already — the companion is counted as unlinked instead.
        if storesCompanion, let ssmStates {
            ssmStateCache.store(
                ssmStates: ssmStates,
                tokens: tokens,
                boundary: tokens.count,
                mediaSalt: mediaSalt,
                enforceDiskQuota: !usesCombinedQuota)
        }

        // The storing conversation is the one in progress: the pass that its
        // own store triggers must not take its rows while cold ones remain.
        if usesCombinedQuota {
            enforceCombinedDiskQuotaLocked(activeChain: chainId)
        } else {
            enforceCombinedDiskQuota(activeChain: chainId)
        }
        diskCache?.reconcileCapacityPressure(chainId: chainId, requiresCompanion: isHybrid)
    }

    /// The direct companion writers (`maybeReDeriveSSMState`, and
    /// `SSMStateCache.store` with its default `persistToDisk`) can put a
    /// companion on disk before its KV row exists; it is then counted as
    /// unlinked, and unlinked companions are evicted FIRST. This call has
    /// just written the row and writes no companion of its own, so nothing
    /// else would ever join the two: the hottest companion would be retired
    /// ahead of every older group, leaving its KV payload unusable.
    ///
    /// One `LIMIT 1` SELECT when no companion is counted as unlinked; the
    /// companion key is only hashed when there is something it could match.
    private func adoptEarlyCompanion(tokens: [Int], mediaSalt: String?) {
        guard companionBytesAreIndexed, isHybrid, let diskCache,
              diskCache.hasLegacyCompanions()
        else { return }
        diskCache.adoptLegacyCompanion(
            kvHash: DiskCache.hashTokens(
                tokens, modelKey: diskCache.modelKey, mediaSalt: mediaSalt),
            companionKey: SSMCompanionDiskStore.keyFor(
                tokens: tokens, boundary: tokens.count,
                mediaSalt: mediaSalt, modelKey: config.modelKey))
    }

    /// Enforce `diskCacheMaxGB` across the whole persistent cache root, not
    /// once for KV payloads and again for recurrent companion payloads.
    ///
    /// New companion sidecars record their matching KV hash, allowing an old
    /// hybrid entry to be evicted as a unit. Legacy sidecars remain readable;
    /// under quota pressure they retire before indexed KV because they cannot
    /// prove which durable KV payload can still reach them.
    ///
    /// A companion whose recorded KV payload is already gone is treated
    /// differently by the two passes. The directory-walk pass (v1 index)
    /// removes it immediately, on every call. The index pass does not: such a
    /// companion is counted as unlinked, so it stays on disk while the total
    /// fits and is the first thing evicted once it does not.
    ///
    /// `activeChain` is the conversation in progress, which the index pass
    /// protects (see ``DiskQuotaPlanner``). Nothing assigns chain ids yet, so
    /// every production call passes nil; the parameter is how a test stands
    /// in for the store path that will.
    func enforceCombinedDiskQuota(activeChain: String? = nil) {
        guard config.enableDiskCache,
              diskCache != nil,
              ssmStateCache.diskStore != nil
        else { return }

        CombinedDiskCacheQuotaLock.shared.lock()
        defer { CombinedDiskCacheQuotaLock.shared.unlock() }
        enforceCombinedDiskQuotaLocked(activeChain: activeChain)
    }

    /// One unit of combined-quota eviction: a KV payload with its linked
    /// companion, or a companion on its own.
    private struct EvictionGroup {
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

    /// The directory-walk pass's eviction policy (an index without the v2
    /// columns; the index pass selects with ``DiskQuotaPlanner``): none when
    /// the total fits; otherwise every group that can never fit, then
    /// unlinked companions, then oldest recency, until the total fits.
    private static func groupsToEvict(
        from groups: [EvictionGroup], maxBytes: Int64
    ) -> (evicted: [EvictionGroup], totalBefore: Int64, remaining: Int64) {
        let totalBefore = IndexedBytes.total(groups.lazy.map(\.bytes))
        guard totalBefore > maxBytes else { return ([], totalBefore, totalBefore) }

        var remaining = totalBefore
        var evicted: [EvictionGroup] = []

        // Reject any group that can never fit before applying LRU. Stable
        // boundaries are normally written shortest-to-longest; evicting the
        // older fitting boundary first and only then discovering that the
        // newest group is individually oversized leaves the cache empty.
        // Pre-eviction preserves the best prior prefix that actually fits.
        let oversized = groups.filter { $0.bytes > maxBytes }
        for group in oversized {
            evicted.append(group)
            remaining = IndexedBytes.difference(remaining, group.bytes)
        }
        if totalBefore == .max {
            // Saturated on the way up (a `file_size` from the index can be
            // anything): what is left is counted, not subtracted.
            remaining = IndexedBytes.total(
                groups.lazy.filter { $0.bytes <= maxBytes }.map(\.bytes))
        }
        let oversizedKeys = Set(oversized.map(\.sortKey))

        for group in groups.sorted(by: {
            if $0.priority != $1.priority { return $0.priority < $1.priority }
            if $0.createdAt == $1.createdAt { return $0.sortKey < $1.sortKey }
            return $0.createdAt < $1.createdAt
        }) where remaining > maxBytes && !oversizedKeys.contains(group.sortKey) {
            evicted.append(group)
            remaining = IndexedBytes.difference(remaining, group.bytes)
        }
        return (evicted, totalBefore, remaining)
    }

    /// Reconcile the linked KV + recurrent quota. Caller must hold
    /// ``CombinedDiskCacheQuotaLock``.
    private func enforceCombinedDiskQuotaLocked(activeChain: String? = nil) {
        guard config.enableDiskCache,
              let diskCache,
              let companionStore = ssmStateCache.diskStore
        else { return }

        let maxBytes = Int64(max(1, diskCache.maxSizeBytes))

        if companionBytesAreIndexed {
            retryCompanionImportIfDueLocked(diskCache: diskCache, companionStore: companionStore)
            enforceIndexedQuotaLocked(
                diskCache: diskCache, companionStore: companionStore, maxBytes: maxBytes,
                activeChain: activeChain)
        } else {
            enforceDirectoryWalkQuotaLocked(
                diskCache: diskCache, companionStore: companionStore, maxBytes: maxBytes)
        }
    }

    /// Where the last over-cap index pass spent its time. For the cost probe;
    /// read and written only while ``CombinedDiskCacheQuotaLock`` is held.
    struct QuotaPassTiming: Sendable, Equatable {
        /// Reading the rows out of the index and building the planner's input.
        var rowsMs = 0.0
        /// ``DiskQuotaPlanner/plan(rows:capBytes:activeChain:)`` alone.
        var selectMs = 0.0
        /// Deleting the victims' files and then their rows.
        var deleteMs = 0.0
        /// The whole pass, the usage aggregate included.
        var totalMs = 0.0
        var evictedGroups = 0
    }
    private var _lastQuotaPassTiming = QuotaPassTiming()

    var lastQuotaPassTimingForTesting: QuotaPassTiming {
        CombinedDiskCacheQuotaLock.shared.lock()
        defer { CombinedDiskCacheQuotaLock.shared.unlock() }
        return _lastQuotaPassTiming
    }

    /// Rows come from the index alone: no directory listing, no file stat, no
    /// sidecar read. Below the cap this is one SQL aggregate — no row is read
    /// and nothing is built for the planner.
    ///
    /// Over the cap the victims are chosen by ``DiskQuotaPlanner``: one row per
    /// KV boundary (its linked companion's bytes included — they are evicted
    /// as a unit) and one per unlinked companion. While every `chain_id` is
    /// NULL and no row is a stable root that is the old order (oversized,
    /// unlinked companions, oldest recency), except that unlinked companions
    /// are taken down to the low watermark rather than to the cap.
    private func enforceIndexedQuotaLocked(
        diskCache: DiskCache, companionStore: SSMCompanionDiskStore, maxBytes: Int64,
        activeChain: String?
    ) {
        let passStart = DispatchTime.now().uptimeNanoseconds
        guard diskCache.usageBytes() > maxBytes else { return }
        func msSince(_ start: UInt64) -> Double {
            Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
        }

        let kvEntries = diskCache.quotaEntries()
        let legacy = diskCache.legacyCompanions()
        var rows: [QuotaRow] = []
        rows.reserveCapacity(kvEntries.count + legacy.count)
        var kvByID: [String: DiskCacheQuotaEntry] = [:]
        kvByID.reserveCapacity(kvEntries.count)
        for kv in kvEntries {
            kvByID[kv.hash] = kv
            rows.append(QuotaRow(
                id: kv.hash,
                tokenCount: kv.tokenCount,
                bytes: IndexedBytes.sum(kv.bytes, kv.companionBytes),
                // A hit refreshes the row and its companion's files with one
                // timestamp, so the row's recency is the group's — with one
                // exception. The directory walk uses min(row, companion file
                // mtimes). A KV row that is re-stored or skip-touched in a
                // call that writes no companion moves `created_at` forward
                // and leaves the companion's files alone, so here the group
                // is as recent as its row, where the walk would have kept it
                // as old as its companion. The group is evicted later than
                // before, never earlier.
                recency: kv.createdAt.timeIntervalSince1970,
                isStableRoot: kv.isStableRoot,
                isResumeBoundary: kv.isResumeBoundary,
                chainId: kv.chainId,
                isLegacyCompanion: false))
        }
        // A KV hash is hex, so the prefix keeps the two id spaces apart.
        let legacyPrefix = "legacy:"
        for companion in legacy {
            rows.append(QuotaRow(
                id: legacyPrefix + companion.key,
                tokenCount: 0,
                bytes: companion.bytes,
                recency: companion.modifiedAt.timeIntervalSince1970,
                isStableRoot: false,
                chainId: nil,
                isLegacyCompanion: true))
        }
        let rowsMs = msSince(passStart)

        // The planner sees the rows this build understands. Under a newer
        // build's index the others are opaque — counted, never offered — so
        // what they hold comes off the cap the understood rows share. (Under
        // the current schema a record that names nothing has just been
        // retired; if that could not be written, nothing real pays for it.)
        //
        // Once the opaque bytes reach the cap, `planCap` is 0: every row is
        // oversized, and each store is evicted by the pass that follows it.
        // That is the decision, and it stands; it is said once per root and
        // reported in ``DiskCacheStats/opaqueBytes``.
        var planCap = maxBytes
        if diskCache.indexIsFromANewerBuild {
            let understood = IndexedBytes.total(rows.lazy.map(\.bytes))
            let opaque = IndexedBytes.difference(diskCache.usageBytes(), understood)
            planCap = IndexedBytes.difference(maxBytes, opaque)
            diskCache.noteOpaqueBytes(opaque, capBytes: maxBytes)
        }

        let selectStart = DispatchTime.now().uptimeNanoseconds
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: planCap, activeChain: activeChain)
        let selectMs = msSince(selectStart)
        guard !plan.evict.isEmpty else { return }

        var evictKV = Set<String>()
        var evictCompanion = Set<String>()
        for id in plan.evict {
            if let kv = kvByID[id] {
                evictKV.insert(kv.hash)
                if let key = kv.companionKey { evictCompanion.insert(key) }
            } else {
                evictCompanion.insert(String(id.dropFirst(legacyPrefix.count)))
            }
        }
        let legacyBytes = Dictionary(
            legacy.map { ($0.key, $0.bytes) }, uniquingKeysWith: { first, _ in first })

        // Order: every FILE goes before the ROW that counts it. If the
        // process dies part-way, what is left is a row naming files that are
        // gone — an over-count, which the next import (or the next fetch of
        // that row) clears. The other order leaves files the index has
        // stopped counting, which nothing would ever evict. So: companion
        // files first, since the KV call below drops a row together with its
        // companion bytes; then each KV payload followed by its row; then the
        // unlinked companions' rows.
        //
        // The same rule covers a file that could not be deleted: it keeps
        // its record, for the bytes that are left, and the next pass selects
        // it again. Each file is tried once per pass, and the plan is not
        // redone to make up for it, so one undeletable file costs at most its
        // own bytes over the cap and never takes the rest of the cache.
        let deleteStart = DispatchTime.now().uptimeNanoseconds
        let companionsStillOnDisk = companionStore.removeQuotaEntries(hashes: evictCompanion)
        let removedCompanions = evictCompanion.subtracting(companionsStillOnDisk)
        let removedKV = diskCache.removeQuotaEntries(
            hashes: evictKV, removedCompanions: removedCompanions)
        let removedLegacy = removedCompanions.intersection(legacyBytes.keys)
        diskCache.forgetLegacyCompanions(keys: removedLegacy)
        for key in companionsStillOnDisk {
            if let current = SSMCompanionDiskStore.publishedEntry(
                key: key, in: companionStore.directory)
            {
                diskCache.correctCompanionBytes(key: key, bytes: current.bytes)
            }
        }
        let deleteMs = msSince(deleteStart)

        // A group is evicted once every file of it is gone.
        var evictedGroups = 0
        var evictedBytes: Int64 = 0
        for id in plan.evict {
            if let kv = kvByID[id] {
                guard removedKV.contains(kv.hash),
                      kv.companionKey.map(removedCompanions.contains) ?? true
                else { continue }
                evictedGroups += 1
                evictedBytes = IndexedBytes.sum(
                    evictedBytes, IndexedBytes.sum(kv.bytes, kv.companionBytes))
            } else {
                let key = String(id.dropFirst(legacyPrefix.count))
                guard removedLegacy.contains(key) else { continue }
                evictedGroups += 1
                evictedBytes = IndexedBytes.sum(evictedBytes, legacyBytes[key] ?? 0)
            }
        }
        let totalMs = msSince(passStart)
        let lostRows = removedKV.union(kvEntries.compactMap { kv in
            kv.companionKey.map(removedCompanions.contains) == true ? kv.hash : nil
        })
        let confirmedEvent = plan.confirmedEvent(rows: rows, lostRows: lostRows)
        diskCache.recordQuotaPass(
            evictedGroups: evictedGroups, evictedBytes: evictedBytes, milliseconds: totalMs,
            event: confirmedEvent,
            tipTokenCount: rows.lazy.filter { $0.chainId == activeChain && !$0.isStableRoot }
                .map(\.tokenCount).max() ?? 0)
        _lastQuotaPassTiming = QuotaPassTiming(
            rowsMs: rowsMs, selectMs: selectMs, deleteMs: deleteMs, totalMs: totalMs,
            evictedGroups: evictedGroups)

        if ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1" {
            // Same leading keys as the directory-walk line; `ms` and `event`
            // are appended. An index pass never removes an orphan on sight, so
            // that count is always 0 here.
            FileHandle.standardError.write(Data(
                "[vmlx][cache/disk-quota] before=\(plan.totalBefore) after=\(max(0, plan.totalAfter)) max=\(maxBytes) logicalEvictions=\(evictedGroups) kvEvicted=\(removedKV.count) companionEvicted=\(removedCompanions.count) legacyCompanionEvicted=\(removedLegacy.count) orphanCompanionEvicted=0 deleteFailures=\(evictKV.count - removedKV.count + companionsStillOnDisk.count) source=index ms=\(String(format: "%.3f", totalMs)) event=\(confirmedEvent?.kind.rawValue ?? "none") chain=\(activeChain ?? "none")\n".utf8))
        }
    }

    /// The pre-v2 pass, kept for an index without the companion columns:
    /// lists the companion directory and reads every sidecar on every call.
    private func enforceDirectoryWalkQuotaLocked(
        diskCache: DiskCache, companionStore: SSMCompanionDiskStore, maxBytes: Int64
    ) {
        let passStart = DispatchTime.now().uptimeNanoseconds
        let kvEntries = diskCache.quotaEntries()
        let kvHashes = Set(kvEntries.map(\.hash))
        var companionEntries = companionStore.quotaEntries()

        // Under a newer build's index a row this build cannot read is not
        // in `kvEntries`, so "no such row" proves nothing about a companion:
        // none is removed on sight, and the opaque rows' payload bytes come
        // off the cap the understood groups share.
        let rowsAreOpaque = diskCache.indexIsFromANewerBuild
        let opaqueBytes = rowsAreOpaque
            ? IndexedBytes.difference(
                Int64(diskCache.snapshotStats().currentPayloadBytes),
                IndexedBytes.total(kvEntries.lazy.map(\.bytes)))
            : 0
        if rowsAreOpaque { diskCache.noteOpaqueBytes(opaqueBytes, capBytes: maxBytes) }
        let orphaned = rowsAreOpaque ? [] : companionEntries.filter {
            guard let kvHash = $0.kvHash else { return false }
            return !kvHashes.contains(kvHash)
        }
        if !orphaned.isEmpty {
            companionStore.removeQuotaEntries(hashes: Set(orphaned.map(\.hash)))
            let orphanHashes = Set(orphaned.map(\.hash))
            companionEntries.removeAll { orphanHashes.contains($0.hash) }
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
                bytes: IndexedBytes.sum(
                    kv.bytes, IndexedBytes.total(companions.lazy.map(\.bytes))),
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

        let plan = Self.groupsToEvict(
            from: groups, maxBytes: IndexedBytes.difference(maxBytes, opaqueBytes))
        guard !plan.evicted.isEmpty else { return }

        let evictKV = plan.evicted.reduce(into: Set<String>()) { $0.formUnion($1.kvHashes) }
        let evictCompanion = plan.evicted.reduce(into: Set<String>()) {
            $0.formUnion($1.companionHashes)
        }

        let removedKV = diskCache.removeQuotaEntries(hashes: evictKV)
        let removedCompanions = evictCompanion.subtracting(
            companionStore.removeQuotaEntries(hashes: evictCompanion))
        // A group is evicted once every file of it is gone.
        let evicted = plan.evicted.filter {
            $0.kvHashes.isSubset(of: removedKV) && $0.companionHashes.isSubset(of: removedCompanions)
        }
        let evictedGroups = evicted.count
        // No chain ids on this index, so no conversation to raise an event for.
        diskCache.recordQuotaPass(
            evictedGroups: evictedGroups,
            evictedBytes: IndexedBytes.total(evicted.lazy.map(\.bytes)),
            milliseconds: Double(DispatchTime.now().uptimeNanoseconds - passStart) / 1_000_000,
            event: nil)

        let legacyCompanionEvicted = companionEntries.reduce(into: 0) { count, entry in
            if entry.kvHash == nil, removedCompanions.contains(entry.hash) {
                count += 1
            }
        }

        if ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1" {
            FileHandle.standardError.write(Data(
                "[vmlx][cache/disk-quota] before=\(plan.totalBefore) after=\(max(0, plan.remaining)) max=\(maxBytes) logicalEvictions=\(evictedGroups) kvEvicted=\(removedKV.count) companionEvicted=\(removedCompanions.count) legacyCompanionEvicted=\(legacyCompanionEvicted) orphanCompanionEvicted=\(orphaned.count) deleteFailures=\(evictKV.count - removedKV.count + evictCompanion.count - removedCompanions.count) source=walk\n".utf8))
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
