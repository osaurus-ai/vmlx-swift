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
    /// `rename` while the older file still holds its inode, so the name then
    /// carries a different one.
    private struct FileIdentity: Equatable {
        let inode: UInt64
        let modified: Date
        let size: UInt64
    }

    private static func identity(_ url: URL) throws -> FileIdentity {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return FileIdentity(
            inode: try #require((attributes[.systemFileNumber] as? NSNumber)?.uint64Value),
            modified: try #require(attributes[.modificationDate] as? Date),
            size: try #require((attributes[.size] as? NSNumber)?.uint64Value))
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
    /// Another model's rows hash to other keys by construction, so their
    /// lengths are not candidates: the fetch probes `[N, N-1]` and then the
    /// one row that can match. Without the filter this was one probe per
    /// foreign length (329 here).
    ///
    /// Rows of the SAME model that cannot match — other conversations, other
    /// media salts — are still probed: neither is a column of the index.
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
            #expect(probes <= 3)
            // Not because nothing was probed: [N, N-1] always are.
            #expect(probes == 2)

            // The foreign rows are all still there, and still theirs.
            let theirHit = try #require(Self.diskMatch(theirs.fetch(tokens: prompt)))
            #expect(theirHit.matched == foreignLengths.last)
        }
    }

    /// The same-model control for the test above: the filter removes only
    /// what another model wrote.
    @Test func sameModelRowsFromOtherConversationsAreStillProbed() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("same-model")
            defer { try? FileManager.default.removeItem(at: root) }
            let coordinator = Self.coordinator(root: root, modelKey: "A")
            let disk = try #require(coordinator.diskCache)
            let prompt = Self.chain(301)
            let other = Self.chain(301, seed: 9)
            disk.store(
                tokens: Array(prompt.prefix(37)), arrays: Self.payload(), enforceQuota: false)
            let decoys = stride(from: 41, through: 299, by: 2).map { $0 }
            try #require(decoys.count > 128)
            for length in decoys {
                disk.store(
                    tokens: Array(other.prefix(length)), arrays: Self.payload(3),
                    enforceQuota: false)
            }

            let before = try Self.diskStats(coordinator)
            let hit = try #require(Self.diskMatch(coordinator.fetch(tokens: prompt)))
            let after = try Self.diskStats(coordinator)
            #expect(hit.matched == 37)
            // [301, 300], then every decoy length (299 is one of them).
            #expect(after.misses - before.misses == decoys.count + 2)
        }
    }

    /// The string `store` writes into `model_key` and the string the
    /// candidate filter binds must be the same bytes, or the filter hides
    /// this model's own rows. Across a close and a reopen, for keys that a
    /// careless comparison would mangle, and with every key in ONE root.
    @Test func modelKeyWrittenByStoreIsTheOneTheFilterBinds() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("key-round-trip")
            defer { try? FileManager.default.removeItem(at: root) }
            let prompt = Self.chain(149)
            let keys: [(key: String?, length: Int)] = [
                ("A", 37), ("a", 41), ("org/Modèle 4-bit 'q' \"x\" %_\\ ", 43), ("", 47),
                (nil, 53),
            ]
            for entry in keys {
                let writer = CacheCoordinator(
                    config: CacheCoordinatorConfig(
                        usePagedCache: false, enableDiskCache: true, diskCacheMaxGB: 1,
                        diskCacheDir: root, modelKey: entry.key))
                writer.storePersistentBoundary(
                    tokens: Array(prompt.prefix(entry.length)), diskArrays: Self.payload(),
                    ssmStates: nil)
            }

            // What is in the column, byte for byte.
            let raw = try Support.RawDB(root: root)
            let stored = try raw.rows(
                "SELECT token_count, typeof(model_key), hex(model_key) FROM cache_entries ORDER BY token_count"
            )
            try #require(stored.count == keys.count)
            for (row, entry) in zip(stored, keys) {
                #expect(row[0] == "\(entry.length)")
                if let key = entry.key {
                    #expect(row[1] == "text")
                    #expect(row[2] == key.utf8.map { String(format: "%02X", $0) }.joined())
                } else {
                    #expect(row[1] == "null")
                }
            }

            // Every key finds its own row after a reopen, plus the unkeyed
            // one, and nobody else's. No key and the empty key are ONE
            // namespace — they hash alike — so each of the two must be
            // offered the other's row as well, and is served the longer.
            for entry in keys {
                let reader = CacheCoordinator(
                    config: CacheCoordinatorConfig(
                        usePagedCache: false, enableDiskCache: true, diskCacheMaxGB: 1,
                        diskCacheDir: root, modelKey: entry.key))
                let disk = try #require(reader.diskCache)
                try #require(disk.indexHasV2Columns)
                let unkeyed = (entry.key ?? "").isEmpty
                let expected: Set<Int> = unkeyed ? [47, 53] : [entry.length, 53]
                #expect(
                    Set(disk.candidateTokenCounts(maxTokens: prompt.count)) == expected,
                    "model key \(entry.key ?? "nil")")
                let hit = try #require(
                    Self.diskMatch(reader.fetch(tokens: prompt)),
                    "model key \(entry.key ?? "nil") lost its own row")
                #expect(hit.matched == (unkeyed ? 53 : entry.length))
            }

            // The shorter of the two shared rows is reachable from both too.
            let shorter = Array(prompt.prefix(48))
            for key in [String?.none, ""] {
                let reader = CacheCoordinator(
                    config: CacheCoordinatorConfig(
                        usePagedCache: false, enableDiskCache: true, diskCacheMaxGB: 1,
                        diskCacheDir: root, modelKey: key))
                let hit = try #require(Self.diskMatch(reader.fetch(tokens: shorter)))
                #expect(hit.matched == 47)
            }
        }
    }

    /// Without the model column there is nothing to filter on, and under a
    /// newer build's schema the column may not mean what it means here: both
    /// keep offering every length.
    @Test func candidatesAreNotFilteredWithoutTheModelColumnOrUnderANewerSchema() throws {
        try MLXMetalTestLock.withLock {
            for newerWithColumns in [false, true] {
                let root = Self.makeRoot("unfiltered-\(newerWithColumns)")
                defer { try? FileManager.default.removeItem(at: root) }
                if newerWithColumns {
                    do {
                        _ = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: "A")
                    }
                    try Support.RawDB(root: root).require("PRAGMA user_version = 99")
                } else {
                    try Support.makeV1OnlyIndex(in: root)
                }
                let ours = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: "A")
                let theirs = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: "B")
                try #require(ours.indexIsFromANewerBuild)
                try #require(ours.indexHasV2Columns == newerWithColumns)
                let prompt = Self.chain(149)
                ours.store(tokens: Array(prompt.prefix(37)), arrays: Self.payload())
                theirs.store(tokens: Array(prompt.prefix(101)), arrays: Self.payload())

                #expect(ours.candidateTokenCounts(maxTokens: 149) == [101, 37])
                #expect(ours.fetch(tokens: Array(prompt.prefix(37))) != nil)
            }
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
    /// Until the engine says so, nothing changes: the entry is served again
    /// and counted again (that half is the control). Once it has reported the
    /// rejection, the shorter entry wins, the refused hit is taken back, and
    /// the next store of the boundary replaces the payload and lifts the mark.
    @Test func rejectedRestoreLetsTheShorterEntryWin() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("shadow")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "prefix-shadow"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let fixture = try Self.storeShadowingFixture(coordinator)
            let prompt = fixture.prompt
            let longTokens = Array(prompt.prefix(11))

            let before = try Self.diskStats(coordinator)
            let first = try #require(Self.diskMatch(coordinator.fetch(tokens: prompt)))
            #expect(first.matched == 11)
            var runtime: [any KVCache] = [KVCacheSimple(), KVCacheSimple()]
            try #require(!first.arrays.isEmpty)
            try #require(
                restoreFromDiskArrays(first.arrays, into: &runtime, requirePromptBoundary: true)
                    == 0)

            // Control — nothing reported yet: served again, counted again.
            let unreported = try #require(Self.diskMatch(coordinator.fetch(tokens: prompt)))
            #expect(unreported.matched == 11)
            #expect(try Self.diskStats(coordinator).hits - before.hits == 2)
            #expect(coordinator.hasDurableDiskEntry(tokens: longTokens))

            // The engine reports both refused restores.
            for _ in 0 ..< 2 {
                coordinator.reportDiskRestoreRejected(
                    tokens: prompt, boundary: 11, mediaSalt: nil, reason: "test")
            }
            let reported = try Self.diskStats(coordinator)
            // One payload, one mark: the second report changes nothing.
            #expect(reported.rejectedDiskRestores - before.rejectedDiskRestores == 1)
            #expect(reported.hits - before.hits == 1)
            #expect(!coordinator.hasDurableDiskEntry(tokens: longTokens))
            #expect(!coordinator.hasValidatedDiskEntry(tokens: longTokens))
            #expect(coordinator.hasDurableDiskEntry(tokens: Array(prompt.prefix(5))))

            let second = try #require(Self.diskMatch(coordinator.fetch(tokens: prompt)))
            #expect(second.matched == 5)
            #expect(second.remaining == Array(prompt.dropFirst(5)))
            var restoredShort: [any KVCache] = [KVCacheSimple(), KVCacheSimple()]
            #expect(
                restoreFromDiskArrays(
                    second.arrays, into: &restoredShort, requirePromptBoundary: true) == 5)
            let afterShort = try Self.diskStats(coordinator)
            #expect(afterShort.hits - reported.hits == 1)
            // Nothing was deleted, and the skipped probe is not a miss.
            #expect(try Self.indexedTokenCounts(root) == [5, 11])
            #expect(afterShort.misses - reported.misses == 2, "[N, N-1] only")

            // The turn ends: the engine stores the boundary with the cache it
            // really has. The payload is replaced and the mark is lifted.
            let longURL = Support.payloadURL(
                root, DiskCache.hashTokens(longTokens, modelKey: modelKey))
            let identityBefore = try Self.identity(longURL)
            coordinator.storeAfterGeneration(
                promptTokens: longTokens, perLayerData: [], ssmStates: nil,
                cache: Self.attentionCache(layers: 2, tokens: 11, fill: 5))
            #expect(try Self.identity(longURL) != identityBefore)
            #expect(coordinator.hasDurableDiskEntry(tokens: longTokens))

            let third = try #require(Self.diskMatch(coordinator.fetch(tokens: prompt)))
            #expect(third.matched == 11)
            var restoredLong: [any KVCache] = [KVCacheSimple(), KVCacheSimple()]
            #expect(
                restoreFromDiskArrays(
                    third.arrays, into: &restoredLong, requirePromptBoundary: true) == 11)
            #expect(restoredLong.allSatisfy { $0.offset == 11 })
        }
    }

    /// One layer at 11 tokens, two layers at 5, both prefixes of one prompt;
    /// and proof that the short one restores into a two-layer cache, so a
    /// refusal of the long one is the long one's.
    private static func storeShadowingFixture(
        _ coordinator: CacheCoordinator
    ) throws -> (prompt: [Int], oneLayer: [any KVCache]) {
        let prompt = Self.chain(37)
        let oneLayer = Self.attentionCache(layers: 1, tokens: 11, fill: 1)
        coordinator.storeAfterGeneration(
            promptTokens: Array(prompt.prefix(11)), perLayerData: [], ssmStates: nil,
            cache: oneLayer)
        coordinator.storeAfterGeneration(
            promptTokens: Array(prompt.prefix(5)), perLayerData: [], ssmStates: nil,
            cache: Self.attentionCache(layers: 2, tokens: 5, fill: 3))
        let disk = try #require(coordinator.diskCache)
        try #require(disk.candidateTokenCounts(maxTokens: prompt.count) == [11, 5])

        var control: [any KVCache] = [KVCacheSimple(), KVCacheSimple()]
        let shortArrays = try #require(
            disk.fetch(tokens: Array(prompt.prefix(5)), touchRecency: false, countHit: false))
        try #require(
            restoreFromDiskArrays(shortArrays, into: &control, requirePromptBoundary: true) == 5)
        return (prompt, oneLayer)
    }

    /// A store of a boundary whose payload this process has validated, with
    /// the same layout, is skipped — that is what keeps a warm turn from
    /// rewriting hundreds of megabytes. For a payload whose restore was
    /// refused the same store must write. The un-rejected entry, stored the
    /// same way in the same test, is the control: it is still skipped.
    @Test func rejectedEntryIsRewrittenEvenWhenTheLayoutMatches() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("rewrite")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "prefix-rewrite"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let fixture = try Self.storeShadowingFixture(coordinator)
            let prompt = fixture.prompt
            let longTokens = Array(prompt.prefix(11))
            let shortTokens = Array(prompt.prefix(5))
            let longURL = Support.payloadURL(
                root, DiskCache.hashTokens(longTokens, modelKey: modelKey))
            let shortURL = Support.payloadURL(
                root, DiskCache.hashTokens(shortTokens, modelKey: modelKey))

            let hit = try #require(Self.diskMatch(coordinator.fetch(tokens: prompt)))
            try #require(hit.matched == 11)
            coordinator.reportDiskRestoreRejected(
                tokens: prompt, boundary: 11, mediaSalt: nil, reason: "test")

            // A direct read validates the file again for this process; the
            // mark has to hold on its own.
            let disk = try #require(coordinator.diskCache)
            try #require(
                disk.fetch(tokens: longTokens, touchRecency: false, countHit: false) != nil)
            #expect(!disk.hasValidatedEntry(tokens: longTokens))
            #expect(!coordinator.hasDurableDiskEntry(tokens: longTokens))

            let longBefore = try Self.identity(longURL)
            let shortBefore = try Self.identity(shortURL)
            let statsBefore = try Self.diskStats(coordinator)

            coordinator.storeAfterGeneration(
                promptTokens: shortTokens, perLayerData: [], ssmStates: nil,
                cache: Self.attentionCache(layers: 2, tokens: 5, fill: 3))
            let afterControl = try Self.diskStats(coordinator)
            #expect(afterControl.storeSkips - statsBefore.storeSkips == 1)
            #expect(try Self.identity(shortURL) == shortBefore)

            coordinator.storeAfterGeneration(
                promptTokens: longTokens, perLayerData: [], ssmStates: nil,
                cache: Self.attentionCache(layers: 1, tokens: 11, fill: 1))
            let afterRewrite = try Self.diskStats(coordinator)
            #expect(afterRewrite.storeSkips == afterControl.storeSkips)
            #expect(try Self.identity(longURL) != longBefore)

            // The mark is lifted, so the same store is skipped from now on.
            coordinator.storeAfterGeneration(
                promptTokens: longTokens, perLayerData: [], ssmStates: nil,
                cache: Self.attentionCache(layers: 1, tokens: 11, fill: 1))
            #expect(try Self.diskStats(coordinator).storeSkips - afterRewrite.storeSkips == 1)
        }
    }

    @Test func rejectionReportedForAnEntryThatIsNotThereChangesNothing() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("no-entry")
            defer { try? FileManager.default.removeItem(at: root) }
            let coordinator = Self.coordinator(root: root, modelKey: "prefix-no-entry")
            let prompt = Self.chain(37)
            coordinator.storePersistentBoundary(
                tokens: Array(prompt.prefix(11)), diskArrays: Self.payload(), ssmStates: nil)
            let hit = try #require(Self.diskMatch(coordinator.fetch(tokens: prompt)))
            try #require(hit.matched == 11)
            let before = try Self.diskStats(coordinator)
            try #require(before.hits == 1)

            // No row at 13; no such boundary at all at 0, -1 and 38; another
            // salt is another entry.
            for boundary in [13, 0, -1, 38] {
                coordinator.reportDiskRestoreRejected(
                    tokens: prompt, boundary: boundary, mediaSalt: nil, reason: "test")
            }
            coordinator.reportDiskRestoreRejected(
                tokens: prompt, boundary: 11, mediaSalt: "other-media", reason: "test")

            let after = try Self.diskStats(coordinator)
            #expect(after.hits == before.hits)
            #expect(after.rejectedDiskRestores == 0)
            #expect(coordinator.hasDurableDiskEntry(tokens: Array(prompt.prefix(11))))
            let again = try #require(Self.diskMatch(coordinator.fetch(tokens: prompt)))
            #expect(again.matched == 11)
            #expect(try Self.indexedTokenCounts(root) == [11])
        }
    }

    /// The hit count never goes below zero: a payload can be served without
    /// a hit having been counted for it (a candidate the coordinator then
    /// vetoes, a counter that was reset).
    @Test func rejectionNeverTakesTheHitCountBelowZero() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("floor")
            defer { try? FileManager.default.removeItem(at: root) }
            let coordinator = Self.coordinator(root: root, modelKey: "prefix-floor")
            let prompt = Self.chain(37)
            let stored = Array(prompt.prefix(11))
            coordinator.storePersistentBoundary(
                tokens: stored, diskArrays: Self.payload(), ssmStates: nil)
            // Served, and no hit counted: the coordinator does the counting.
            let disk = try #require(coordinator.diskCache)
            guard case .arrays = disk.fetchCandidate(tokens: stored, mediaSalt: nil) else {
                Issue.record("the stored entry was not served")
                return
            }
            try #require(try Self.diskStats(coordinator).hits == 0)

            coordinator.reportDiskRestoreRejected(
                tokens: prompt, boundary: 11, mediaSalt: nil, reason: "test")
            let after = try Self.diskStats(coordinator)
            #expect(after.hits == 0)
            #expect(after.rejectedDiskRestores == 1)
            guard case .miss = coordinator.fetch(tokens: prompt) else {
                Issue.record("the rejected entry was served")
                return
            }
        }
    }

    /// A rejection is about the payload the fetch handed out. Reported for
    /// an entry this cache never served, it says nothing about the file that
    /// is there: nothing is marked and nothing is counted.
    @Test func aRejectionOfAnEntryThatWasNeverServedChangesNothing() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("never-served")
            defer { try? FileManager.default.removeItem(at: root) }
            let coordinator = Self.coordinator(root: root, modelKey: "prefix-never-served")
            let prompt = Self.chain(37)
            coordinator.storePersistentBoundary(
                tokens: Array(prompt.prefix(11)), diskArrays: Self.payload(), ssmStates: nil)
            try #require(try Self.indexedTokenCounts(root) == [11])

            coordinator.reportDiskRestoreRejected(
                tokens: prompt, boundary: 11, mediaSalt: nil, reason: "test")
            #expect(try Self.diskStats(coordinator).rejectedDiskRestores == 0)
            #expect(coordinator.hasDurableDiskEntry(tokens: Array(prompt.prefix(11))))
            #expect(Self.diskMatch(coordinator.fetch(tokens: prompt))?.matched == 11)
        }
    }

    /// Between the fetch and the engine's report another writer replaces the
    /// payload. What was refused is gone; what is under the name now has
    /// been refused by nobody, and marking it would hide a good entry (and
    /// have this process write it again). The same sequence without the
    /// replacement is the control: there the report marks.
    @Test func aRejectionMarksOnlyThePayloadThatWasServed() throws {
        try MLXMetalTestLock.withLock {
            for replaced in [false, true] {
                let root = Self.makeRoot("served-\(replaced)")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "prefix-served"
                let told = Self.coordinator(root: root, modelKey: modelKey)
                let prompt = try Self.storeShadowingFixture(told).prompt
                let longTokens = Array(prompt.prefix(11))
                let longURL = Support.payloadURL(
                    root, DiskCache.hashTokens(longTokens, modelKey: modelKey))

                try #require(Self.diskMatch(told.fetch(tokens: prompt))?.matched == 11)
                let before = try Self.diskStats(told)
                try #require(before.hits == 1)
                if replaced {
                    let servedFile = try Self.identity(longURL)
                    let elsewhere = try #require(
                        Self.coordinator(root: root, modelKey: modelKey).diskCache)
                    elsewhere.store(
                        tokens: longTokens,
                        arrays: TQDiskSerializer.serialize(
                            cache: Self.attentionCache(layers: 2, tokens: 11, fill: 5)),
                        enforceQuota: false)
                    let now = try Self.identity(longURL)
                    try #require(now.inode != servedFile.inode)
                    try #require(now.size != servedFile.size, "the fingerprint cannot tell them apart")
                }

                told.reportDiskRestoreRejected(
                    tokens: prompt, boundary: 11, mediaSalt: nil, reason: "test")
                let after = try Self.diskStats(told)
                #expect(after.rejectedDiskRestores == (replaced ? 0 : 1), "replaced=\(replaced)")
                #expect(after.hits == (replaced ? 1 : 0), "replaced=\(replaced)")
                #expect(told.hasDurableDiskEntry(tokens: longTokens) == replaced)
                let next = try #require(Self.diskMatch(told.fetch(tokens: prompt)))
                #expect(next.matched == (replaced ? 11 : 5), "replaced=\(replaced)")
            }
        }
    }

    /// The served payloads are remembered for the most recent 64 entries
    /// only. A report that arrives after 64 other entries have been served
    /// finds nothing to compare with and is dropped — the entry is simply
    /// served once more. One fewer in between, and the report still marks.
    @Test func servedPayloadsAreRememberedForABoundedNumberOfEntries() throws {
        try MLXMetalTestLock.withLock {
            for others in [63, 64] {
                let root = Self.makeRoot("served-bound-\(others)")
                defer { try? FileManager.default.removeItem(at: root) }
                let coordinator = Self.coordinator(root: root, modelKey: "prefix-served-bound")
                let disk = try #require(coordinator.diskCache)
                let prompt = Self.chain(37)
                let stored = Array(prompt.prefix(11))
                disk.store(tokens: stored, arrays: Self.payload(), enforceQuota: false)
                for other in 0 ..< others {
                    disk.store(
                        tokens: Self.chain(5, seed: 11 + other), arrays: Self.payload(),
                        enforceQuota: false)
                }
                try #require(try Self.indexedTokenCounts(root).count == others + 1)

                try #require(Self.diskMatch(coordinator.fetch(tokens: prompt))?.matched == 11)
                var servedOthers = 0
                for other in 0 ..< others {
                    if case .arrays = disk.fetchCandidate(
                        tokens: Self.chain(5, seed: 11 + other), mediaSalt: nil)
                    {
                        servedOthers += 1
                    }
                }
                try #require(servedOthers == others)

                coordinator.reportDiskRestoreRejected(
                    tokens: prompt, boundary: 11, mediaSalt: nil, reason: "test")
                let marked = others < 64
                #expect(
                    try Self.diskStats(coordinator).rejectedDiskRestores == (marked ? 1 : 0),
                    "others=\(others)")
                if marked {
                    guard case .miss = coordinator.fetch(tokens: prompt) else {
                        Issue.record("others=\(others): the rejected entry was served")
                        continue
                    }
                } else {
                    #expect(Self.diskMatch(coordinator.fetch(tokens: prompt))?.matched == 11)
                }
            }
        }
    }

    /// A mark belongs to the coordinator that was told. Another model on the
    /// same root has other entries; another coordinator for the SAME model (a
    /// reload) starts without marks and is refused once more before it knows.
    @Test func rejectedMarksStayWithTheCoordinatorThatWasTold() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("per-coordinator")
            defer { try? FileManager.default.removeItem(at: root) }
            let told = Self.coordinator(root: root, modelKey: "A")
            let other = Self.coordinator(root: root, modelKey: "B")
            let prompt = try Self.storeShadowingFixture(told).prompt
            _ = try Self.storeShadowingFixture(other)

            try #require(Self.diskMatch(told.fetch(tokens: prompt))?.matched == 11)
            told.reportDiskRestoreRejected(
                tokens: prompt, boundary: 11, mediaSalt: nil, reason: "test")
            #expect(Self.diskMatch(told.fetch(tokens: prompt))?.matched == 5)

            #expect(Self.diskMatch(other.fetch(tokens: prompt))?.matched == 11)
            #expect(try Self.diskStats(other).rejectedDiskRestores == 0)
            #expect(other.hasDurableDiskEntry(tokens: Array(prompt.prefix(11))))

            let reloaded = Self.coordinator(root: root, modelKey: "A")
            #expect(Self.diskMatch(reloaded.fetch(tokens: prompt))?.matched == 11)
            #expect(try Self.diskStats(reloaded).rejectedDiskRestores == 0)
            // The one that was told still knows.
            #expect(Self.diskMatch(told.fetch(tokens: prompt))?.matched == 5)
        }
    }

    /// A mark is for the payload that was refused. When another writer — a
    /// second process, or a reloaded model — has replaced that file, the
    /// entry is a candidate again without this coordinator storing anything.
    @Test func aPayloadReplacedBySomebodyElseLiftsTheMark() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("replaced")
            defer { try? FileManager.default.removeItem(at: root) }
            let told = Self.coordinator(root: root, modelKey: "A")
            let prompt = try Self.storeShadowingFixture(told).prompt
            try #require(Self.diskMatch(told.fetch(tokens: prompt))?.matched == 11)
            told.reportDiskRestoreRejected(
                tokens: prompt, boundary: 11, mediaSalt: nil, reason: "test")
            try #require(Self.diskMatch(told.fetch(tokens: prompt))?.matched == 5)

            let elsewhere = Self.coordinator(root: root, modelKey: "A")
            elsewhere.storeAfterGeneration(
                promptTokens: Array(prompt.prefix(11)), perLayerData: [], ssmStates: nil,
                cache: Self.attentionCache(layers: 2, tokens: 11, fill: 5))

            let hit = try #require(Self.diskMatch(told.fetch(tokens: prompt)))
            #expect(hit.matched == 11)
            var runtime: [any KVCache] = [KVCacheSimple(), KVCacheSimple()]
            #expect(
                restoreFromDiskArrays(hit.arrays, into: &runtime, requirePromptBoundary: true)
                    == 11)
            #expect(told.hasDurableDiskEntry(tokens: Array(prompt.prefix(11))))
        }
    }

    /// When what this process stores for a boundary does not restore either
    /// — a serializer that does not round-trip, a caller-supplied cache of
    /// another topology — every turn would fetch, be refused, write the whole
    /// boundary again and lift the mark with it. One rewrite per entry per
    /// process is the bound: the second rejection keeps the entry passed over
    /// and leaves it durable, so the store is skipped and the shorter entry
    /// keeps winning.
    @Test func anEntryRejectedAgainAfterItsRewriteIsNotWrittenASecondTime() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("rewrite-once")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "prefix-rewrite-once"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let prompt = try Self.storeShadowingFixture(coordinator).prompt
            let longTokens = Array(prompt.prefix(11))
            let longURL = Support.payloadURL(
                root, DiskCache.hashTokens(longTokens, modelKey: modelKey))
            // The store this process keeps making: one layer, as before.
            func storeLong() {
                coordinator.storeAfterGeneration(
                    promptTokens: longTokens, perLayerData: [], ssmStates: nil,
                    cache: Self.attentionCache(layers: 1, tokens: 11, fill: 1))
            }

            try #require(Self.diskMatch(coordinator.fetch(tokens: prompt))?.matched == 11)
            coordinator.reportDiskRestoreRejected(
                tokens: prompt, boundary: 11, mediaSalt: nil, reason: "test")
            try #require(Self.diskMatch(coordinator.fetch(tokens: prompt))?.matched == 5)

            // First rejection: the boundary is written again, once.
            let original = try Self.identity(longURL)
            try #require(original.size > 0)
            storeLong()
            let rewritten = try Self.identity(longURL)
            #expect(rewritten.inode != original.inode)
            #expect(rewritten != original)
            let served = try #require(Self.diskMatch(coordinator.fetch(tokens: prompt)))
            #expect(served.matched == 11, "the rewrite lifts the first mark")
            let beforeSecond = try Self.diskStats(coordinator)
            try #require(beforeSecond.rejectedDiskRestores == 1)
            try #require(beforeSecond.rejectedRewritesSuppressed == 0)

            // Second rejection, of the payload this process wrote itself.
            coordinator.reportDiskRestoreRejected(
                tokens: prompt, boundary: 11, mediaSalt: nil, reason: "test")
            let afterSecond = try Self.diskStats(coordinator)
            #expect(afterSecond.rejectedDiskRestores == 2)
            #expect(afterSecond.rejectedRewritesSuppressed == 1)
            #expect(afterSecond.hits == beforeSecond.hits - 1)
            #expect(coordinator.hasDurableDiskEntry(tokens: longTokens))
            #expect(coordinator.hasValidatedDiskEntry(tokens: longTokens))

            for turn in 0 ..< 3 {
                storeLong()
                #expect(try Self.identity(longURL) == rewritten, "turn \(turn): written again")
                #expect(
                    try Self.diskStats(coordinator).storeSkips - afterSecond.storeSkips
                        == turn + 1)
                #expect(Self.diskMatch(coordinator.fetch(tokens: prompt))?.matched == 5)
            }
            #expect(try Self.diskStats(coordinator).rejectedDiskRestores == 2)
            #expect(try Self.indexedTokenCounts(root) == [5, 11])
        }
    }

    /// The bound must not outlive the payload it is about. Once another
    /// writer has replaced the file, the entry is served again, and this
    /// process has not rewritten THAT payload: a rejection of it is a first
    /// rejection, and earns its one rewrite.
    @Test func aPayloadReplacedBySomebodyElseAlsoForgetsTheRewrite() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("rewrite-forgotten")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "prefix-rewrite-forgotten"
            let told = Self.coordinator(root: root, modelKey: modelKey)
            let prompt = try Self.storeShadowingFixture(told).prompt
            let longTokens = Array(prompt.prefix(11))
            let longURL = Support.payloadURL(
                root, DiskCache.hashTokens(longTokens, modelKey: modelKey))
            func reject() {
                told.reportDiskRestoreRejected(
                    tokens: prompt, boundary: 11, mediaSalt: nil, reason: "test")
            }
            func storeLong(layers: Int, through coordinator: CacheCoordinator) {
                coordinator.storeAfterGeneration(
                    promptTokens: longTokens, perLayerData: [], ssmStates: nil,
                    cache: Self.attentionCache(layers: layers, tokens: 11, fill: 1))
            }

            try #require(Self.diskMatch(told.fetch(tokens: prompt))?.matched == 11)
            reject()
            storeLong(layers: 1, through: told)
            try #require(Self.diskMatch(told.fetch(tokens: prompt))?.matched == 11)
            reject()
            try #require(try Self.diskStats(told).rejectedRewritesSuppressed == 1)
            try #require(Self.diskMatch(told.fetch(tokens: prompt))?.matched == 5)
            try #require(told.hasDurableDiskEntry(tokens: longTokens))

            // Somebody else publishes a payload that does restore.
            let elsewhere = Self.coordinator(root: root, modelKey: modelKey)
            storeLong(layers: 2, through: elsewhere)
            let hit = try #require(Self.diskMatch(told.fetch(tokens: prompt)))
            #expect(hit.matched == 11)
            var runtime: [any KVCache] = [KVCacheSimple(), KVCacheSimple()]
            #expect(
                restoreFromDiskArrays(hit.arrays, into: &runtime, requirePromptBoundary: true)
                    == 11)

            // A rejection of the new payload is a first rejection again.
            reject()
            let stats = try Self.diskStats(told)
            #expect(stats.rejectedDiskRestores == 3)
            #expect(stats.rejectedRewritesSuppressed == 1)
            #expect(!told.hasDurableDiskEntry(tokens: longTokens))
            let replaced = try Self.identity(longURL)
            storeLong(layers: 2, through: told)
            #expect(try Self.identity(longURL).inode != replaced.inode)
            #expect(Self.diskMatch(told.fetch(tokens: prompt))?.matched == 11)
        }
    }

    /// Both engines report a structural rejection of a disk hit, and only
    /// that: the two reports sit on the two refusals (`restoreFromDiskArrays`
    /// restored nothing; the restored offsets disagree with the boundary),
    /// each behind `detail == .disk`, and before the contextual roll-backs
    /// (media placeholders in the suffix, a missing seed state), which say
    /// nothing about the entry.
    @Test func bothEnginesReportStructuralRejectionsOfDiskHitsOnly() throws {
        for path in [
            "Libraries/MLXLMCommon/Evaluate.swift",
            "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
        ] {
            let source = try String(contentsOfFile: path, encoding: .utf8)
            let calls = source.components(separatedBy: "coordinator.reportDiskRestoreRejected(")
            try #require(calls.count - 1 == 2, "\(path): \(calls.count - 1) reports")
            for before in calls.dropLast() {
                #expect(
                    before.suffix(120).contains("detail == .disk"),
                    "\(path): a report that is not behind `detail == .disk`")
            }
            let reasons = [
                "payload does not fit the runtime cache",
                "restored offsets do not match the boundary",
            ]
            for reason in reasons {
                #expect(source.components(separatedBy: reason).count - 1 == 1, "\(path)")
            }
            let contextual = try #require(source.range(of: "let unsafePartial ="))
            let lastReport = try #require(
                source.range(of: "coordinator.reportDiskRestoreRejected(", options: .backwards))
            #expect(lastReport.upperBound < contextual.lowerBound, "\(path)")
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
