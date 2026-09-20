import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// `DiskQuotaPlanner` is a pure function; these tests are about the pass that
/// feeds it and carries its answer out: the index-sourced combined quota pass
/// of `CacheCoordinator`, on real files, plus what that pass reports in
/// `DiskCacheStats`.
///
/// Every storing test asserts the accounting invariant directly
/// (`usageBytes()` == bytes on disk of what the index names, and the index
/// names every published file), not just which rows survived.
///
/// Token counts are deliberately not multiples of 64 or 256. Caps are computed
/// from the measured bytes of the fixture, and every test requires the
/// inequalities its cap was chosen for before it looks at an outcome.
@Suite(.serialized)
struct DiskQuotaPlannerWiringTests {

    private typealias Support = DiskCacheAccountingTestSupport

    // MARK: - Fixtures

    private static func makeRoot(_ label: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-quota-wiring-\(label)-\(UUID().uuidString)")
    }

    private static func tokens(_ count: Int, seed: Int) -> [Int] {
        (0 ..< count).map { seed * 100_000 + $0 }
    }

    private static func kv(_ elements: Int = 1_024) -> [String: MLXArray] {
        ["data": MLXArray.ones([elements], dtype: .float32)]
    }

    private static func recurrent(_ elements: Int = 1_024) -> [MLXArray] {
        [MLXArray.ones([elements], dtype: .float32)]
    }

    private final class TestClock: @unchecked Sendable {
        private let lock = NSLock()
        private var value = Date()
        var now: Date {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
        func advance(_ seconds: TimeInterval) {
            lock.lock()
            value += seconds
            lock.unlock()
        }
    }

    private static func coordinator(
        root: URL, capBytes: Int64 = 1 << 30, modelKey: String, clock: TestClock? = nil
    ) throws -> CacheCoordinator {
        try #require(
            capBytes == 1 << 30 || capBytes < 1 << 24,
            "cap must survive the Float GiB round trip exactly")
        let config = CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheMaxGB: Float(capBytes) / 1_073_741_824,
            diskCacheDir: root,
            modelKey: modelKey)
        let coordinator =
            clock.map { clock in
                CacheCoordinator(
                    config: config, diskIndexBusyTimeoutMs: DiskCache.defaultIndexBusyTimeoutMs,
                    importRetryInterval: 60, now: { clock.now })
            } ?? CacheCoordinator(config: config)
        coordinator.setHybrid(true, requiresRecurrentSSMCompanion: true)
        try #require(Int64(try #require(coordinator.diskCache).maxSizeBytes) == capBytes)
        return coordinator
    }

    private static func kvHash(_ tokens: [Int], _ modelKey: String) -> String {
        DiskCache.hashTokens(tokens, modelKey: modelKey)
    }

    private static func ssmKey(_ tokens: [Int], _ modelKey: String) -> String {
        SSMCompanionDiskStore.keyFor(tokens: tokens, boundary: tokens.count, modelKey: modelKey)
    }

    /// Payload + companion files of one boundary, as they are on disk now.
    private static func groupBytes(_ root: URL, _ tokens: [Int], _ modelKey: String) -> Int64 {
        Support.fileBytes(Support.payloadURL(root, kvHash(tokens, modelKey)))
            + Support.companionBytes(root, ssmKey(tokens, modelKey))
    }

    private static func survivingHashes(_ root: URL) throws -> Set<String> {
        Set(
            try Support.RawDB(root: root).rows("SELECT hash FROM cache_entries").compactMap {
                $0[0]
            })
    }

    /// One row of a fixture the planner can tell apart: which conversation it
    /// belongs to, whether it is a stable root, and how recently it was used.
    private struct Boundary {
        let label: String
        let tokens: [Int]
        var kvElements = 1_024
        let recency: TimeInterval
        var chain: String? = nil
        var stable = false
    }

    /// Writes `boundaries` through the coordinator's own two stores with the
    /// per-store pass deferred (what `storePersistentBoundary` does, minus the
    /// inline quota pass, which would run with no active chain), then sets the
    /// columns nothing assigns yet — `chain_id`, `kind` — by raw SQL. Returns
    /// each boundary's bytes on disk.
    private static func populate(
        _ coordinator: CacheCoordinator, root: URL, modelKey: String, _ boundaries: [Boundary]
    ) throws -> [String: Int64] {
        let disk = try #require(coordinator.diskCache)
        let companion = try #require(coordinator.ssmStateCache.diskStore)
        try #require(disk.indexHasV2Columns)
        let raw = try Support.RawDB(root: root)
        var bytes: [String: Int64] = [:]
        for boundary in boundaries {
            disk.store(
                tokens: boundary.tokens, arrays: kv(boundary.kvElements), enforceQuota: false)
            try companion.store(
                ssmStates: recurrent(), tokens: boundary.tokens, boundary: boundary.tokens.count,
                enforceQuota: false)
            try #require(
                disk.touchRecency(
                    tokens: boundary.tokens, at: Date(timeIntervalSince1970: boundary.recency)))
            let chain = boundary.chain.map { "'\($0)'" } ?? "NULL"
            try raw.require(
                "UPDATE cache_entries SET chain_id = \(chain), kind = \(boundary.stable ? 1 : 0) "
                    + "WHERE hash = '\(kvHash(boundary.tokens, modelKey))'")
            bytes[boundary.label] = groupBytes(root, boundary.tokens, modelKey)
            try #require(bytes[boundary.label]! > 0)
        }
        // Fail closed: every boundary is one linked group in the index.
        let rows = try Support.indexedRows(root)
        try #require(rows.count == boundaries.count)
        try #require(rows.allSatisfy { $0.companionKey != nil && $0.companionBytes > 0 })
        return bytes
    }

    /// The fixture's bytes, measured in a throw-away root under a cap nothing
    /// reaches. Payloads and sidecars are a function of the tokens, the shapes
    /// and the model key alone, so the real run gets the same bytes — and
    /// requires that it did.
    private static func measure(modelKey: String, _ boundaries: [Boundary]) throws -> [String:
        Int64]
    {
        let scratch = makeRoot("measure")
        defer { try? FileManager.default.removeItem(at: scratch) }
        let roomy = try coordinator(root: scratch, modelKey: modelKey)
        return try populate(roomy, root: scratch, modelKey: modelKey, boundaries)
    }

    private static func labels(
        _ hashes: Set<String>, _ boundaries: [Boundary], _ modelKey: String
    ) -> Set<String> {
        let byHash = Dictionary(
            uniqueKeysWithValues: boundaries.map { (kvHash($0.tokens, modelKey), $0.label) })
        return Set(hashes.map { byHash[$0] ?? "unknown:\($0)" })
    }

    // MARK: - 1

    /// The no-regression pin: the six-group fixture of
    /// `DiskCacheCompanionAccountingTests.quotaOrderIsUnchanged`, built file
    /// for file the same way, with no chain ids anywhere — which is every
    /// cache that exists today. The planner-driven pass must leave exactly
    /// the survivors the old order left, and report what it did.
    @Test func indexPassUsesThePlannerAndMatchesTheOldOrderWithoutChains() throws {
        try MLXMetalTestLock.withLock {
            let modelKey = "wiring-order"
            let root = Self.makeRoot("order")
            defer { try? FileManager.default.removeItem(at: root) }
            let g1 = Self.tokens(301, seed: 31)
            let g2 = Self.tokens(517, seed: 32)
            let g3 = Self.tokens(1_003, seed: 33)
            let g4 = Self.tokens(307, seed: 34)
            let oversized = Self.tokens(1_291, seed: 35)
            let legacy = Self.tokens(311, seed: 36)
            // Recency is deliberately not insertion order, and the two groups
            // that go first regardless of recency are the two NEWEST.
            let recency: [(tokens: [Int], at: TimeInterval)] = [
                (g1, 40_000), (g2, 10_000), (g3, 20_000), (g4, 30_000),
                (oversized, 60_000), (legacy, 50_000),
            ]

            // Built with the standalone stores; the coordinator only ever sees
            // a finished directory, and its first pass is the one under test.
            do {
                let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                try #require(disk.indexHasV2Columns)
                let companion = try SSMCompanionDiskStore(
                    cacheDir: Support.companionDir(root), modelKey: modelKey, maxBytes: 0)
                for tokens in [g1, g2, g3, g4] {
                    disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
                    try companion.store(
                        ssmStates: Self.recurrent(), tokens: tokens, boundary: tokens.count,
                        enforceQuota: false)
                }
                disk.store(tokens: oversized, arrays: Self.kv(65_537), enforceQuota: false)
                try companion.store(
                    ssmStates: Self.recurrent(), tokens: oversized, boundary: oversized.count,
                    enforceQuota: false)
                // A companion from before sidecars carried `kv_hash`, with no
                // KV payload of its own.
                try companion.store(
                    ssmStates: Self.recurrent(), tokens: legacy, boundary: legacy.count,
                    enforceQuota: false)
                let sidecarURL = Support.companionURLs(root, Self.ssmKey(legacy, modelKey))[1]
                var sidecar = try #require(
                    JSONSerialization.jsonObject(with: Data(contentsOf: sidecarURL))
                        as? [String: Any])
                sidecar.removeValue(forKey: "kv_hash")
                sidecar.removeValue(forKey: "boundary")
                try JSONSerialization.data(withJSONObject: sidecar, options: [.sortedKeys])
                    .write(to: sidecarURL, options: [.atomic])

                for (tokens, at) in recency {
                    let date = Date(timeIntervalSince1970: at)
                    if tokens != legacy {
                        try #require(disk.touchRecency(tokens: tokens, at: date))
                    }
                    for url in Support.companionURLs(root, Self.ssmKey(tokens, modelKey)) {
                        try FileManager.default.setAttributes(
                            [.modificationDate: date], ofItemAtPath: url.path)
                    }
                }
            }
            let bytes = [g1, g2, g3, g4].map { Self.groupBytes(root, $0, modelKey) }
            let oversizedBytes = Self.groupBytes(root, oversized, modelKey)
            let legacyBytes = Support.companionBytes(root, Self.ssmKey(legacy, modelKey))
            try #require(bytes.allSatisfy { $0 > 0 } && legacyBytes > 0)

            // Room for the two newest ordinary groups and half of another.
            let cap = bytes[0] + bytes[3] + bytes.min()! / 2
            try #require(oversizedBytes > cap)
            try #require(bytes[0] + bytes[2] + bytes[3] > cap)

            CacheCoordinator.resetImportedRootsForTesting()
            let coordinator = try Self.coordinator(root: root, capBytes: cap, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            try #require(disk.indexHasV2Columns)

            // Evicted: `oversized` (can never fit, newest of all), `legacy`
            // (unlinked, second newest), then g2 (t=10 000) and g3 (t=20 000).
            // g4 + g1 fit, so the pass stops there.
            #expect(
                try Self.survivingHashes(root)
                    == [Self.kvHash(g1, modelKey), Self.kvHash(g4, modelKey)])
            #expect(try Support.legacyRows(root).isEmpty)
            for gone in [g2, g3, oversized, legacy] {
                #expect(Self.groupBytes(root, gone, modelKey) == 0)
            }
            try Support.expectUsageMatchesDisk(disk, root: root)

            let stats = try #require(coordinator.snapshotStats().diskStats)
            #expect(stats.evictions == 4)
            #expect(stats.quotaPasses == 1)
            #expect(stats.evictedBytes == oversizedBytes + legacyBytes + bytes[1] + bytes[2])
            #expect(stats.currentPayloadBytes == Int(bytes[0] + bytes[3]))
            #expect(stats.lastQuotaPassMs > 0)
            // No conversation is in progress as far as this pass knows.
            #expect(stats.pressureEventSeq == 0)
            #expect(stats.lastPressureEvent == nil)
        }
    }

    // MARK: - 2

    /// The active conversation holds the OLDEST rows in the cache, so the old
    /// order would eat it first. With chain ids in the index the pass spends
    /// the cold conversation's superseded rows, then the active one's, then
    /// the cold resume point — and the active resume point stays.
    @Test func activeChainIsProtectedWhenChainsExist() throws {
        try MLXMetalTestLock.withLock {
            let modelKey = "wiring-chains"
            let fixture = [
                Boundary(
                    label: "root", tokens: Self.tokens(137, seed: 40), recency: 70_000, stable: true
                ),
                Boundary(
                    label: "cold:301", tokens: Self.tokens(301, seed: 41), recency: 40_000,
                    chain: "cold"),
                Boundary(
                    label: "cold:517", tokens: Self.tokens(517, seed: 42), recency: 50_000,
                    chain: "cold"),
                Boundary(
                    label: "cold:1003", tokens: Self.tokens(1_003, seed: 43), recency: 60_000,
                    chain: "cold"),
                Boundary(
                    label: "act:307", tokens: Self.tokens(307, seed: 44), recency: 10_000,
                    chain: "act"),
                Boundary(
                    label: "act:521", tokens: Self.tokens(521, seed: 45), recency: 20_000,
                    chain: "act"),
                Boundary(
                    label: "act:1009", tokens: Self.tokens(1_009, seed: 46), recency: 30_000,
                    chain: "act"),
            ]
            let size = try Self.measure(modelKey: modelKey, fixture)
            let smallest = try #require(size.values.min())
            let all = size.values.reduce(0, +)

            struct Outcome {
                let survivors: Set<String>
                let stats: DiskCacheStats
            }
            func run(cap: Int64, activeChain: String?) throws -> Outcome {
                let root = Self.makeRoot("chains")
                defer { try? FileManager.default.removeItem(at: root) }
                let coordinator = try Self.coordinator(
                    root: root, capBytes: cap, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                try #require(
                    try Self.populate(coordinator, root: root, modelKey: modelKey, fixture) == size,
                    "INVALID: the fixture is not the one the cap was computed for")
                try #require(disk.usageBytes() == all && all > cap)

                coordinator.enforceCombinedDiskQuota(activeChain: activeChain)

                try Support.expectUsageMatchesDisk(disk, root: root)
                #expect(disk.usageBytes() <= cap)
                return Outcome(
                    survivors: Self.labels(try Self.survivingHashes(root), fixture, modelKey),
                    stats: try #require(coordinator.snapshotStats().diskStats))
            }

            // Light pressure: room for everything but the cold chain's two
            // superseded rows. They go; nothing of the active chain does.
            let lightKeeps: Set = ["root", "cold:1003", "act:307", "act:521", "act:1009"]
            let lightCap = lightKeeps.reduce(Int64(0)) { $0 + size[$1]! } + smallest / 2
            let light = try run(cap: lightCap, activeChain: "act")
            #expect(light.survivors == lightKeeps)
            #expect(light.stats.evictions == 2)
            #expect(light.stats.evictedBytes == size["cold:301"]! + size["cold:517"]!)
            #expect(light.stats.pressureEventSeq == 0)
            #expect(light.stats.lastPressureEvent == nil)

            // Hard pressure: room for the root and one resume point. Cold
            // non-tips, active non-tips, then the COLD tip; the active tip —
            // the third-oldest row of seven — survives.
            let hardCap = size["root"]! + size["act:1009"]! + smallest / 2
            try #require(size["root"]! + size["act:1009"]! + size["cold:1003"]! > hardCap)
            let hard = try run(cap: hardCap, activeChain: "act")
            #expect(hard.survivors == ["root", "act:1009"])
            #expect(hard.stats.evictions == 5)
            #expect(hard.stats.quotaPasses == 1)
            #expect(hard.stats.pressureEventSeq == 1)
            #expect(
                hard.stats.lastPressureEvent
                    == DiskCachePressureEvent(
                        kind: .activeChainTrimmed, chainId: "act", tipBytes: size["act:1009"]!,
                        capBytes: hardCap))

            // The control: the same cap with no conversation in progress.
            // "act" is then just the coldest chain and loses its tip instead.
            let nobody = try run(cap: hardCap, activeChain: nil)
            #expect(nobody.survivors == ["root", "cold:1003"])
            #expect(nobody.stats.pressureEventSeq == 0)
        }
    }

    // MARK: - 3

    /// `pressureEventSeq` moves once per pass that produced an event: not on a
    /// pass that had nothing to do, not on a poll, and again when a later pass
    /// produces another event.
    @Test func oversizedActiveTipRaisesThePressureEventInStats() throws {
        try MLXMetalTestLock.withLock {
            let modelKey = "wiring-pressure"
            let first = [
                Boundary(
                    label: "root", tokens: Self.tokens(137, seed: 50), recency: 20_000, stable: true
                ),
                Boundary(
                    label: "act:301", tokens: Self.tokens(301, seed: 51), recency: 10_000,
                    chain: "act"),
                Boundary(
                    label: "act:1291", tokens: Self.tokens(1_291, seed: 52), kvElements: 65_537,
                    recency: 30_000, chain: "act"),
            ]
            let second = [
                Boundary(
                    label: "act:1301", tokens: Self.tokens(1_301, seed: 53), kvElements: 70_001,
                    recency: 40_000, chain: "act")
            ]
            let size = try Self.measure(modelKey: modelKey, first + second)
            let cap = size["root"]! + size["act:301"]! + min(size["root"]!, size["act:301"]!) / 2
            try #require(size["act:1291"]! > cap && size["act:1301"]! > cap)

            let root = Self.makeRoot("pressure")
            defer { try? FileManager.default.removeItem(at: root) }
            let coordinator = try Self.coordinator(root: root, capBytes: cap, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            func stats() throws -> DiskCacheStats {
                try #require(coordinator.snapshotStats().diskStats)
            }

            _ = try Self.populate(coordinator, root: root, modelKey: modelKey, first)
            #expect(try stats().pressureEventSeq == 0)
            #expect(try stats().lastPressureEvent == nil)

            coordinator.enforceCombinedDiskQuota(activeChain: "act")
            let dropped = DiskCachePressureEvent(
                kind: .activeTipDropped, chainId: "act", tipBytes: size["act:1291"]!, capBytes: cap)
            #expect(
                Self.labels(try Self.survivingHashes(root), first, modelKey) == ["root", "act:301"])
            #expect(try stats().pressureEventSeq == 1)
            #expect(try stats().lastPressureEvent == dropped)
            #expect(try stats().lastPressureEvent?.kind == .activeTipDropped)
            #expect(try stats().evictedBytes == size["act:1291"]!)
            #expect(try stats().quotaPasses == 1)
            try Support.expectUsageMatchesDisk(disk, root: root)

            // The same pass again has nothing to do: no new event, and the
            // last one is still there to be read.
            let passMs = try stats().lastQuotaPassMs
            coordinator.enforceCombinedDiskQuota(activeChain: "act")
            #expect(try stats().pressureEventSeq == 1)
            #expect(try stats().lastPressureEvent == dropped)
            #expect(try stats().quotaPasses == 1)
            #expect(try stats().lastQuotaPassMs == passMs)

            // A new oversized tip, a new event.
            let rowsBefore = try Support.indexedRows(root).count
            disk.store(
                tokens: second[0].tokens, arrays: Self.kv(second[0].kvElements), enforceQuota: false
            )
            try #require(coordinator.ssmStateCache.diskStore).store(
                ssmStates: Self.recurrent(), tokens: second[0].tokens,
                boundary: second[0].tokens.count, enforceQuota: false)
            try Support.RawDB(root: root).require(
                "UPDATE cache_entries SET chain_id = 'act' "
                    + "WHERE hash = '\(Self.kvHash(second[0].tokens, modelKey))'")
            try #require(try Support.indexedRows(root).count == rowsBefore + 1)
            try #require(Self.groupBytes(root, second[0].tokens, modelKey) == size["act:1301"]!)

            coordinator.enforceCombinedDiskQuota(activeChain: "act")
            #expect(try stats().pressureEventSeq == 2)
            #expect(
                try stats().lastPressureEvent
                    == DiskCachePressureEvent(
                        kind: .activeTipDropped, chainId: "act", tipBytes: size["act:1301"]!,
                        capBytes: cap))
            #expect(try stats().quotaPasses == 2)
            #expect(try stats().evictedBytes == size["act:1291"]! + size["act:1301"]!)
            try Support.expectUsageMatchesDisk(disk, root: root)
        }
    }

    // MARK: - 4

    /// Through the production path: `storePersistentBoundary`, whose pass runs
    /// inline. Two of four stores push the cache over its cap.
    @Test func statsCountEvictedBytesAndPasses() throws {
        try MLXMetalTestLock.withLock {
            let modelKey = "wiring-stats"
            let boundaries = [301, 517, 1_003, 307].enumerated().map { index, count in
                Boundary(
                    label: "g\(index + 1)", tokens: Self.tokens(count, seed: 60 + index),
                    recency: 10_000 * Double(index + 1))
            }
            let size = try Self.measure(modelKey: modelKey, boundaries)
            // Room for two groups and half of another, whichever two they are.
            let cap = size.values.sorted().suffix(2).reduce(0, +) + size.values.min()! / 2
            try #require(size.values.sorted().prefix(3).reduce(0, +) > cap)

            let root = Self.makeRoot("stats")
            defer { try? FileManager.default.removeItem(at: root) }
            let coordinator = try Self.coordinator(root: root, capBytes: cap, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            func stats() throws -> DiskCacheStats {
                try #require(coordinator.snapshotStats().diskStats)
            }

            var passesSeen: [Int] = []
            for boundary in boundaries {
                coordinator.storePersistentBoundary(
                    tokens: boundary.tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
                // The row just written is "now"; give it its place in the
                // fixture's order before the next store's pass reads it.
                try #require(
                    disk.touchRecency(
                        tokens: boundary.tokens, at: Date(timeIntervalSince1970: boundary.recency)))
                passesSeen.append(try stats().quotaPasses)
            }
            // Stores 1 and 2 fit; 3 and 4 each evict the oldest group.
            #expect(passesSeen == [0, 0, 1, 2])
            #expect(
                Self.labels(try Self.survivingHashes(root), boundaries, modelKey) == ["g3", "g4"])
            let after = try stats()
            #expect(after.evictions == 2)
            #expect(after.quotaPasses == 2)
            #expect(after.evictedBytes == size["g1"]! + size["g2"]!)
            #expect(after.currentPayloadBytes == Int(size["g3"]! + size["g4"]!))
            #expect(after.lastQuotaPassMs > 0)
            #expect(after.pressureEventSeq == 0)
            try Support.expectUsageMatchesDisk(disk, root: root)

            // A pass below the cap is not a counted pass and is not timed.
            coordinator.enforceCombinedDiskQuota()
            let idle = try stats()
            #expect(idle.quotaPasses == 2)
            #expect(idle.evictedBytes == after.evictedBytes)
            #expect(idle.lastQuotaPassMs == after.lastQuotaPassMs)
        }
    }

    // MARK: - 5

    /// An index without the companion columns has no chain ids to read: the
    /// directory-walk pass keeps the old selection. The one observable
    /// difference between the two selections on a chain-less cache is
    /// hysteresis on unlinked legacy companions — the old order stops at the
    /// cap, the planner goes on to the low watermark — so the same files are
    /// evicted once under each index, and the v2 run is the control that the
    /// fixture can tell them apart.
    @Test func v1IndexStillUsesTheOldSelection() throws {
        try MLXMetalTestLock.withLock {
            let modelKey = "wiring-v1"
            let g1 = Self.tokens(301, seed: 71)
            let g2 = Self.tokens(1_003, seed: 72)
            let legacyOld = Self.tokens(311, seed: 73)
            let legacyNew = Self.tokens(523, seed: 74)

            struct Outcome: Equatable {
                var survivingKV: Set<String>
                var survivingCompanions: Set<String>
                var evictions: Int
                var quotaPasses: Int
            }
            func run(v2: Bool) throws -> Outcome {
                let root = Self.makeRoot(v2 ? "v2-control" : "v1")
                defer { try? FileManager.default.removeItem(at: root) }
                if !v2 { try Support.makeV1OnlyIndex(in: root) }
                do {
                    let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                    try #require(disk.indexHasV2Columns == v2)
                    let companion = try SSMCompanionDiskStore(
                        cacheDir: Support.companionDir(root), modelKey: modelKey, maxBytes: 0)
                    for tokens in [g1, g2] {
                        disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
                    }
                    for tokens in [g1, g2, legacyOld, legacyNew] {
                        try companion.store(
                            ssmStates: Self.recurrent(), tokens: tokens, boundary: tokens.count,
                            enforceQuota: false)
                    }
                    for tokens in [legacyOld, legacyNew] {
                        let sidecarURL = Support.companionURLs(root, Self.ssmKey(tokens, modelKey))[
                            1]
                        var sidecar = try #require(
                            JSONSerialization.jsonObject(with: Data(contentsOf: sidecarURL))
                                as? [String: Any])
                        sidecar.removeValue(forKey: "kv_hash")
                        sidecar.removeValue(forKey: "boundary")
                        try JSONSerialization.data(withJSONObject: sidecar, options: [.sortedKeys])
                            .write(to: sidecarURL, options: [.atomic])
                    }
                    for (tokens, at) in [
                        (legacyOld, 10_000.0), (legacyNew, 20_000), (g1, 30_000), (g2, 40_000),
                    ] {
                        let date = Date(timeIntervalSince1970: at)
                        if tokens == g1 || tokens == g2 {
                            try #require(disk.touchRecency(tokens: tokens, at: date))
                        }
                        for url in Support.companionURLs(root, Self.ssmKey(tokens, modelKey)) {
                            try FileManager.default.setAttributes(
                                [.modificationDate: date], ofItemAtPath: url.path)
                        }
                    }
                }
                let groups =
                    Self.groupBytes(root, g1, modelKey) + Self.groupBytes(root, g2, modelKey)
                let oldBytes = Support.companionBytes(root, Self.ssmKey(legacyOld, modelKey))
                let newBytes = Support.companionBytes(root, Self.ssmKey(legacyNew, modelKey))
                try #require(groups > 0 && oldBytes > 0 && newBytes > 0)
                // Losing the older legacy companion restores the cap, with a
                // tenth of it to spare, but not the low watermark.
                let cap = groups + newBytes + oldBytes / 10
                try #require(groups + newBytes + oldBytes > cap)
                try #require(groups + newBytes <= cap)
                try #require(
                    groups + newBytes > Int64(Double(cap) * DiskQuotaPlanner.lowWatermarkFraction),
                    "INVALID: the fixture cannot tell the two selections apart")

                CacheCoordinator.resetImportedRootsForTesting()
                let coordinator = try Self.coordinator(
                    root: root, capBytes: cap, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                try #require(disk.indexHasV2Columns == v2)

                let companionNames = try FileManager.default
                    .contentsOfDirectory(atPath: Support.companionDir(root).path)
                let survivingCompanions = Set(
                    companionNames.compactMap { name -> String? in
                        guard name.hasPrefix("ssm-"), name.hasSuffix(".safetensors") else {
                            return nil
                        }
                        return String(name.dropFirst(4).dropLast(".safetensors".count))
                    })
                #expect(companionNames.count == survivingCompanions.count * 2)
                if v2 { try Support.expectUsageMatchesDisk(disk, root: root) }
                let stats = try #require(coordinator.snapshotStats().diskStats)
                #expect(stats.pressureEventSeq == 0)
                return Outcome(
                    survivingKV: try Self.survivingHashes(root),
                    survivingCompanions: survivingCompanions,
                    evictions: stats.evictions, quotaPasses: stats.quotaPasses)
            }

            let bothGroups: Set = [Self.kvHash(g1, modelKey), Self.kvHash(g2, modelKey)]
            let linked: Set = [Self.ssmKey(g1, modelKey), Self.ssmKey(g2, modelKey)]
            // v1: the old selection — the older legacy companion, and stop.
            let walked = try run(v2: false)
            #expect(
                walked
                    == Outcome(
                        survivingKV: bothGroups,
                        survivingCompanions: linked.union([Self.ssmKey(legacyNew, modelKey)]),
                        evictions: 1, quotaPasses: 1))
            // v2, the control: the planner — both legacy companions, for low.
            let indexed = try run(v2: true)
            #expect(
                indexed
                    == Outcome(
                        survivingKV: bothGroups, survivingCompanions: linked,
                        evictions: 2, quotaPasses: 1))
        }
    }

    // MARK: - 6

    /// A companion directory that cannot be listed is not an empty one. An
    /// import over it must not commit: it would mark the root imported with
    /// every companion in it uncounted, and nothing would ever count them.
    @Test func unlistableCompanionDirectoryMakesTheImportNotCommit() throws {
        try MLXMetalTestLock.withLock {
            let modelKey = "wiring-unlistable"
            let root = Self.makeRoot("unlistable")
            let dir = Support.companionDir(root)
            defer {
                try? FileManager.default.setAttributes(
                    [.posixPermissions: 0o755], ofItemAtPath: dir.path)
                try? FileManager.default.removeItem(at: root)
            }

            // An upgraded directory: companions on disk, none counted.
            let populated: [Support.IndexedRow]
            do {
                let writer = try Self.coordinator(root: root, modelKey: modelKey)
                for boundary in [Self.tokens(301, seed: 81), Self.tokens(1_003, seed: 82)] {
                    writer.storePersistentBoundary(
                        tokens: boundary, diskArrays: Self.kv(), ssmStates: Self.recurrent())
                }
                populated = try Support.indexedRows(root)
                try #require(populated.count == 2)
                try #require(
                    populated.allSatisfy { $0.companionKey != nil && $0.companionBytes > 0 })
            }
            try Support.RawDB(root: root).require(
                "UPDATE cache_entries SET companion_key = NULL, companion_bytes = 0")
            CacheCoordinator.resetImportedRootsForTesting()

            try FileManager.default.setAttributes(
                [.posixPermissions: 0o000], ofItemAtPath: dir.path)
            try Support.requireUnlistable(dir)

            // The import at open, and an on-demand one: neither may commit.
            let clock = TestClock()
            let reopened = try Self.coordinator(root: root, modelKey: modelKey, clock: clock)
            let disk = try #require(reopened.diskCache)
            try #require(reopened.ssmStateCache.diskStore != nil)
            #expect(reopened.reconcileDiskAccounting() == false)
            #expect(try Support.indexedRows(root).allSatisfy { $0.companionKey == nil })

            // Readable again. Because nothing committed, the root still counts
            // as not imported, and the quota pass of the next store — once the
            // retry interval has passed — imports it.
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o755], ofItemAtPath: dir.path)
            clock.advance(61)
            let extra = Self.tokens(517, seed: 83)
            reopened.storePersistentBoundary(tokens: extra, diskArrays: Self.kv(), ssmStates: nil)
            let extraHash = Self.kvHash(extra, modelKey)
            #expect(try Support.indexedRows(root).filter { $0.hash != extraHash } == populated)
            try Support.expectUsageMatchesDisk(disk, root: root)

            // An ABSENT directory is not an error: there is nothing to import,
            // and saying so commits.
            try FileManager.default.removeItem(at: dir)
            #expect(reopened.reconcileDiskAccounting() == true)
            #expect(try Support.indexedRows(root).allSatisfy { $0.companionKey == nil })
            #expect(try Support.legacyRows(root).isEmpty)
        }
    }
}
