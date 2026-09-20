import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// How the disk tier of `CacheCoordinator.fetch` chooses a stored prefix.
///
/// Entries are whole-prefix snapshots keyed by a hash over (model key, media
/// salt, the exact token prefix). A fetch probes `[N, N-1]`, then every
/// distinct indexed `token_count <= N`, longest first, and the first candidate
/// that deserializes (and, for a hybrid, has its companion state) wins.
///
/// Every test asserts a non-empty baseline before it compares anything, and
/// no token length is a multiple of 64, 128 or 256.
@Suite(.serialized)
struct DiskPrefixMatchSelectionTests {

    // MARK: - Fixtures

    private typealias Support = DiskCacheAccountingTestSupport

    private static func makeRoot(_ label: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-prefix-selection-\(label)-\(UUID().uuidString)")
    }

    /// One conversation: every prefix of it is a prefix of every longer one.
    private static func chain(_ count: Int, seed: Int = 7) -> [Int] {
        (0 ..< count).map { seed * 100_000 + $0 }
    }

    private static func payload(_ elements: Int = 13) -> [String: MLXArray] {
        ["data": MLXArray.ones([elements], dtype: .float32)]
    }

    private static func coordinator(
        root: URL, modelKey: String, capBytes: Int64 = 1 << 30
    ) -> CacheCoordinator {
        CacheCoordinator(
            config: CacheCoordinatorConfig(
                usePagedCache: false,
                enableDiskCache: true,
                diskCacheMaxGB: Float(capBytes) / 1_073_741_824,
                diskCacheDir: root,
                modelKey: modelKey))
    }

    private static func diskStats(_ coordinator: CacheCoordinator) throws -> DiskCacheStats {
        try #require(coordinator.snapshotStats().diskStats)
    }

    /// The matched length of a disk hit; nil for a miss. A paged hit fails
    /// the test: every coordinator here has the paged tier off.
    private static func diskMatch(
        _ result: CacheFetchResult, sourceLocation: SourceLocation = #_sourceLocation
    ) -> (matched: Int, remaining: [Int], arrays: [String: MLXArray])? {
        guard case .hit(let matched, let remaining, let detail, _, _, let arrays) = result
        else { return nil }
        #expect(detail == .disk, sourceLocation: sourceLocation)
        return (matched, remaining, arrays ?? [:])
    }

    private static func indexedTokenCounts(_ root: URL) throws -> [Int] {
        try Support.RawDB(root: root)
            .rows("SELECT token_count FROM cache_entries ORDER BY token_count")
            .map { Int($0[0] ?? "") ?? -1 }
    }

    /// `layers` plain attention layers holding `tokens` tokens each.
    private static func attentionCache(layers: Int, tokens: Int, fill: Float) -> [any KVCache] {
        let cache: [any KVCache] = (0 ..< layers).map { _ in KVCacheSimple() }
        for (index, layer) in cache.enumerated() {
            let keys = MLXArray.ones([1, 1, tokens, 4]) * (fill + Float(index))
            _ = layer.update(keys: keys, values: keys + 1)
        }
        MLX.eval(cache)
        return cache
    }

    /// What identifies one published payload file: a rewrite publishes with
    /// `rename`, so the name then carries a different inode.
    private struct FileIdentity: Equatable {
        let inode: UInt64
        let modified: Date
    }

    private static func identity(_ url: URL) throws -> FileIdentity {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return FileIdentity(
            inode: try #require((attributes[.systemFileNumber] as? NSNumber)?.uint64Value),
            modified: try #require(attributes[.modificationDate] as? Date))
    }

    // MARK: - 1. Longest stored prefix

    @Test func longestStoredPrefixWins() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("longest")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "prefix-longest"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let prompt = Self.chain(200)
            for length in [37, 101, 149] {
                coordinator.storePersistentBoundary(
                    tokens: Array(prompt.prefix(length)), diskArrays: Self.payload(),
                    ssmStates: nil)
            }
            try #require(try Self.indexedTokenCounts(root) == [37, 101, 149])

            let first = try #require(Self.diskMatch(coordinator.fetch(tokens: prompt)))
            #expect(first.matched == 149)
            #expect(first.remaining.count == 51)
            #expect(first.remaining == Array(prompt.dropFirst(149)))

            // The payload goes; its row is still there and still the longest
            // candidate. The probe that finds the file missing drops the row.
            let longest = DiskCache.hashTokens(Array(prompt.prefix(149)), modelKey: modelKey)
            let longestURL = Support.payloadURL(root, longest)
            try #require(Support.fileBytes(longestURL) > 0)
            try FileManager.default.removeItem(at: longestURL)
            try #require(try Self.indexedTokenCounts(root) == [37, 101, 149])

            let second = try #require(Self.diskMatch(coordinator.fetch(tokens: prompt)))
            #expect(second.matched == 101)
            #expect(second.remaining.count == 99)
            #expect(try Self.indexedTokenCounts(root) == [37, 101])
        }
    }

    // MARK: - 2. Shared prefix, different suffix

    @Test func sharedPrefixDifferentSuffixUsesTheSharedBoundary() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("diverge")
            defer { try? FileManager.default.removeItem(at: root) }
            let coordinator = Self.coordinator(root: root, modelKey: "prefix-diverge")
            let stored = Self.chain(149)
            for length in [101, 149] {
                coordinator.storePersistentBoundary(
                    tokens: Array(stored.prefix(length)), diskArrays: Self.payload(),
                    ssmStates: nil)
            }
            try #require(try Self.indexedTokenCounts(root) == [101, 149])

            // Same first 120 tokens, then another conversation: the 149 entry
            // is a candidate by length and must not match by content.
            let prompt = Array(stored.prefix(120)) + Self.chain(81, seed: 9)
            try #require(prompt.count == 201)
            try #require(Array(prompt.prefix(101)) == Array(stored.prefix(101)))
            try #require(Array(prompt.prefix(149)) != stored)

            let hit = try #require(Self.diskMatch(coordinator.fetch(tokens: prompt)))
            #expect(hit.matched == 101)
            #expect(hit.remaining == Array(prompt.dropFirst(101)))
            // The 149 entry was probed and missed; it is somebody's valid
            // entry and stays.
            #expect(try Self.indexedTokenCounts(root) == [101, 149])
        }
    }

    // MARK: - 3. Exact boundary excluded for disk-backed topologies

    @Test func exactBoundaryIsExcludedForDiskBackedTopologies() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("exact")
            defer { try? FileManager.default.removeItem(at: root) }
            let coordinator = Self.coordinator(root: root, modelKey: "prefix-exact")
            let prompt = Self.chain(101)
            coordinator.storePersistentBoundary(
                tokens: prompt, diskArrays: Self.payload(), ssmStates: nil)
            try #require(try Self.indexedTokenCounts(root) == [101])

            // Control: the same entry IS served when the exact boundary is
            // allowed, so the miss below is the flag and not a broken fixture.
            let exact = try #require(Self.diskMatch(coordinator.fetch(tokens: prompt)))
            #expect(exact.matched == 101)
            #expect(exact.remaining.isEmpty)

            guard case .miss = coordinator.fetch(tokens: prompt, skipExactDiskBoundary: true)
            else {
                Issue.record("a stored N was served although the exact boundary is excluded")
                return
            }

            coordinator.storePersistentBoundary(
                tokens: Array(prompt.prefix(100)), diskArrays: Self.payload(), ssmStates: nil)
            let seed = try #require(
                Self.diskMatch(coordinator.fetch(tokens: prompt, skipExactDiskBoundary: true)))
            #expect(seed.matched == 100)
            #expect(seed.remaining == [prompt[100]])
        }
    }

    // MARK: - 4. Another model's rows

    /// Probes are counted as the `misses` delta of the fetching cache: every
    /// probe that finds no payload under its hash is one `DiskCache.fetch`
    /// miss, and the accepted candidate is not a miss.
    ///
    /// The candidate lengths come from the index without regard to which
    /// model wrote the row, so another model's rows are hashed and probed,
    /// one by one, before the only row that can match.
    @Test func foreignModelRowsCostProbesButNeverHit() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("foreign")
            defer { try? FileManager.default.removeItem(at: root) }
            let ours = Self.coordinator(root: root, modelKey: "A")
            let theirs = Self.coordinator(root: root, modelKey: "B")
            let prompt = Self.chain(6_001)

            ours.storePersistentBoundary(
                tokens: Array(prompt.prefix(5_003)), diskArrays: Self.payload(), ssmStates: nil)

            // Model B has the SAME conversation at many other lengths: same
            // tokens, another model key, so another hash.
            let foreignLengths = stride(from: 5_005, through: 5_999, by: 3)
                .filter { $0 % 64 != 0 }
            try #require(foreignLengths.count >= 300)
            let theirDisk = try #require(theirs.diskCache)
            for length in foreignLengths {
                theirDisk.store(
                    tokens: Array(prompt.prefix(length)), arrays: Self.payload(3),
                    enforceQuota: false)
            }
            try #require(try Self.indexedTokenCounts(root).count == foreignLengths.count + 1)

            let before = try Self.diskStats(ours)
            let hit = try #require(Self.diskMatch(ours.fetch(tokens: prompt)))
            let after = try Self.diskStats(ours)

            #expect(hit.matched == 5_003)
            #expect(hit.remaining.count == 998)
            #expect(after.hits - before.hits == 1)

            let probes = after.misses - before.misses
            // [N, N-1], then every foreign length: none is a multiple of the
            // page size, so this also shows that paging goes past page one.
            #expect(probes == foreignLengths.count + 2)
            #expect(probes > 128)
        }
    }

    // MARK: - 5. Rows an older build wrote

    /// An older build writes three columns, so its rows carry no model key;
    /// `INSERT OR REPLACE` also drops the key of a row this build wrote.
    @Test func legacyRowsWithNullModelKeyStayReachable() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("legacy")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "prefix-legacy"
            let prompt = Self.chain(149)
            let stored = Array(prompt.prefix(37))
            let hash = DiskCache.hashTokens(stored, modelKey: modelKey)
            do {
                let writer = Self.coordinator(root: root, modelKey: modelKey)
                writer.storePersistentBoundary(
                    tokens: stored, diskArrays: Self.payload(), ssmStates: nil)
            }
            let bytes = Support.fileBytes(Support.payloadURL(root, hash))
            try #require(bytes > 0)

            // The last v1 build's insert, verbatim.
            let raw = try Support.RawDB(root: root)
            try raw.require(
                """
                INSERT OR REPLACE INTO cache_entries (hash, token_count, file_size)
                VALUES ('\(hash)', \(stored.count), \(bytes))
                """)
            let rows = try raw.rows("SELECT hash, model_key FROM cache_entries")
            try #require(rows.count == 1)
            try #require(rows[0][0] == hash)
            try #require(rows[0][1] == nil, "the fixture row still carries a model key")

            let reader = Self.coordinator(root: root, modelKey: modelKey)
            let hit = try #require(Self.diskMatch(reader.fetch(tokens: prompt)))
            #expect(hit.matched == 37)
            #expect(hit.remaining.count == 112)
        }
    }

    // MARK: - 6. An entry the engine cannot restore

    /// The coordinator accepts a candidate as soon as it deserializes; whether
    /// it fits the running model's cache is only known to the engine, after
    /// `restoreFromDiskArrays`. Here the longest entry describes ONE layer and
    /// the runtime cache has two, so the restore is refused (0 tokens) and the
    /// engine prefills everything — while a shorter, restorable entry exists.
    ///
    /// Pinned as it is today: nothing tells the coordinator. The next fetch
    /// serves the same entry and counts another hit, the entry counts as
    /// durable, and a store of the same boundary with the same layout is
    /// skipped, so nothing replaces it. (A store whose layout differs is
    /// written even today; `hasDurableDiskEntry` does not look at the layout.)
    @Test func acceptedThenRejectedEntryShadowsShorterOne() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("shadow")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "prefix-shadow"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let prompt = Self.chain(37)
            let longTokens = Array(prompt.prefix(11))
            let shortTokens = Array(prompt.prefix(5))

            let oneLayer = Self.attentionCache(layers: 1, tokens: 11, fill: 1)
            let twoLayers = Self.attentionCache(layers: 2, tokens: 5, fill: 3)
            coordinator.storeAfterGeneration(
                promptTokens: longTokens, perLayerData: [], ssmStates: nil, cache: oneLayer)
            coordinator.storeAfterGeneration(
                promptTokens: shortTokens, perLayerData: [], ssmStates: nil, cache: twoLayers)
            try #require(try Self.indexedTokenCounts(root) == [5, 11])

            // Control: the short entry restores into a two-layer cache, so
            // the refusal below is the long entry's and not the fixture's.
            let disk = try #require(coordinator.diskCache)
            var control: [any KVCache] = [KVCacheSimple(), KVCacheSimple()]
            let shortArrays = try #require(
                disk.fetch(tokens: shortTokens, touchRecency: false, countHit: false))
            try #require(
                restoreFromDiskArrays(shortArrays, into: &control, requirePromptBoundary: true) == 5
            )

            let before = try Self.diskStats(coordinator)
            let first = try #require(Self.diskMatch(coordinator.fetch(tokens: prompt)))
            #expect(first.matched == 11)
            var runtime: [any KVCache] = [KVCacheSimple(), KVCacheSimple()]
            try #require(!first.arrays.isEmpty)
            #expect(
                restoreFromDiskArrays(first.arrays, into: &runtime, requirePromptBoundary: true)
                    == 0)
            #expect(runtime.allSatisfy { $0.offset == 0 })

            // TODAY: served again, counted again.
            let second = try #require(Self.diskMatch(coordinator.fetch(tokens: prompt)))
            #expect(second.matched == 11)
            #expect(try Self.diskStats(coordinator).hits - before.hits == 2)

            // TODAY: durable, and the same boundary is not rewritten.
            #expect(coordinator.hasDurableDiskEntry(tokens: longTokens))
            let longURL = Support.payloadURL(
                root, DiskCache.hashTokens(longTokens, modelKey: modelKey))
            let identityBefore = try Self.identity(longURL)
            let skipsBefore = try Self.diskStats(coordinator).storeSkips
            coordinator.storeAfterGeneration(
                promptTokens: longTokens, perLayerData: [], ssmStates: nil,
                cache: Self.attentionCache(layers: 1, tokens: 11, fill: 1))
            #expect(try Self.diskStats(coordinator).storeSkips - skipsBefore == 1)
            #expect(try Self.identity(longURL) == identityBefore)
        }
    }

    // MARK: - 7. A hybrid veto comes after the payload is loaded

    /// For a hybrid that needs a separate recurrent payload, a candidate with
    /// no companion state is refused — after its KV payload has been opened
    /// and deserialized. The refusal is a miss to the caller and is counted
    /// neither as a hit nor as a miss.
    @Test func hybridCompanionVetoLoadsThePayloadFirst() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("veto")
            defer { try? FileManager.default.removeItem(at: root) }
            let coordinator = Self.coordinator(root: root, modelKey: "prefix-veto")
            coordinator.setHybrid(true, requiresRecurrentSSMCompanion: true)
            let prompt = Self.chain(149)
            let stored = Array(prompt.prefix(37))
            let disk = try #require(coordinator.diskCache)
            disk.store(tokens: stored, arrays: Self.payload(), enforceQuota: false)
            try #require(try Self.indexedTokenCounts(root) == [37])

            // A second instance has validated nothing yet.
            let reader = Self.coordinator(root: root, modelKey: "prefix-veto")
            reader.setHybrid(true, requiresRecurrentSSMCompanion: true)
            let readerDisk = try #require(reader.diskCache)
            try #require(!readerDisk.hasValidatedEntry(tokens: stored))
            let before = try Self.diskStats(reader)

            guard case .miss = reader.fetch(tokens: prompt) else {
                Issue.record("a hybrid hit was served without companion state")
                return
            }
            let after = try Self.diskStats(reader)
            #expect(readerDisk.hasValidatedEntry(tokens: stored), "the payload was not loaded")
            #expect(after.hits == before.hits)
            // [N, N-1] found nothing; the vetoed candidate is not a miss.
            #expect(after.misses - before.misses == 2)
        }
    }

    // MARK: - 8. A fetch can write

    /// A hybrid hit whose companion is missing but whose payload carries the
    /// folded recurrent state re-publishes that state as a companion — a
    /// write, followed by a quota pass, inside `fetch`.
    @Test func fetchCanWriteACompanionAndRunAQuotaPass() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("fetch-writes")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "prefix-fetch-writes"
            let prompt = Self.chain(149)
            let stored = Array(prompt.prefix(37))
            let folded = TQDiskSerializer.serialize(
                cache: Self.attentionCache(layers: 1, tokens: 37, fill: 1),
                ssmStates: [MLXArray.ones([3_001], dtype: .float32)])
            try #require(TQDiskSerializer.ssmStates(from: folded)?.count == 1)

            let decoy = Self.chain(11, seed: 3)
            let kvBytes: Int64
            do {
                let writer = Self.coordinator(root: root, modelKey: modelKey)
                let disk = try #require(writer.diskCache)
                disk.store(tokens: decoy, arrays: Self.payload(5_003), enforceQuota: false)
                _ = disk.touchRecency(
                    tokens: decoy, mediaSalt: nil, at: Date(timeIntervalSinceNow: -3_600))
                disk.store(tokens: stored, arrays: folded, enforceQuota: false)
                kvBytes = disk.usageBytes()
            }
            try #require(kvBytes > 0)
            let companionDir = Support.companionDir(root)
            let companionsBefore =
                (try? FileManager.default.contentsOfDirectory(atPath: companionDir.path)) ?? []
            try #require(companionsBefore.isEmpty)

            // Everything fits at open; the companion the fetch writes does not.
            let reader = Self.coordinator(root: root, modelKey: modelKey, capBytes: kvBytes + 101)
            reader.setHybrid(true, requiresRecurrentSSMCompanion: true)
            let before = try Self.diskStats(reader)
            try #require(before.evictions == 0)
            try #require(before.currentEntryCount == 2)

            let hit = try #require(Self.diskMatch(reader.fetch(tokens: prompt)))
            #expect(hit.matched == 37)

            let companionsAfter = try FileManager.default.contentsOfDirectory(
                atPath: companionDir.path)
            #expect(companionsAfter.count == 2, "tensor file and sidecar")
            let after = try Self.diskStats(reader)
            #expect(after.evictions - before.evictions == 1)
            #expect(after.quotaPasses - before.quotaPasses == 1)
            #expect(try Self.indexedTokenCounts(root) == [37], "the older decoy paid")
        }
    }
}
