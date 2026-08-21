import Foundation
import MLX
@testable import MLXLMCommon
import Testing

// MARK: - Disk-cache quota enforcement
//
// These exercise the REAL DiskCache against a real directory: real safetensors
// payloads, the real SQLite index, the real quota pass. Nothing is mocked.
//
// They exist because the questions that matter here cannot be answered by a
// screenshot of the cache readout. A screenshot shows one number at one moment.
// It cannot show that eviction fires as the cache grows, that the OLDEST entry
// goes first, that lowering the cap re-enforces at the new value, or that a
// purge reclaims orphaned files. Those are sequence properties, so they need a
// test that drives the sequence.

/// Store a payload of roughly `bytes` under a synthetic token prefix.
private func store(
    _ cache: DiskCache,
    tokens: [Int],
    approximateBytes: Int
) {
    // float32 → 4 bytes per element.
    let count = Swift.max(1, approximateBytes / 4)
    cache.store(tokens: tokens, arrays: ["k": MLXArray.zeros([count])])
}

private func makeCacheDir() -> URL {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("quota-\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

@Test func quotaEvictsAsTheCacheGrowsPastTheCap() throws {
    try MLXMetalTestLock.withLock {
        let dir = makeCacheDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let cap = 400_000
        let cache = DiskCache(cacheDir: dir, maxSizeBytes: cap, modelKey: "grow")

        var series: [(entries: Int, bytes: Int)] = []
        for i in 0..<12 {
            store(cache, tokens: [i, i + 1, i + 2], approximateBytes: 80_000)
            let s = cache.snapshotStats()
            series.append((s.currentEntryCount, s.currentPayloadBytes))
        }

        let final = cache.snapshotStats()

        // The cap is enforced after each store, so the resting size must be at
        // or under it even though 12 x 80KB were written into a 400KB budget.
        #expect(
            final.currentPayloadBytes <= cap,
            "payload \(final.currentPayloadBytes) exceeded cap \(cap); series=\(series)")

        // The janitor must actually have run — a cache that simply refused to
        // store anything would also satisfy the assertion above.
        #expect(final.evictions > 0, "no eviction recorded; series=\(series)")
        #expect(final.currentEntryCount > 0, "everything was evicted; series=\(series)")

        // And it must have grown before it plateaued, otherwise the writes were
        // being dropped rather than the cache being trimmed.
        let peak = series.map(\.bytes).max() ?? 0
        #expect(peak > 0, "cache never grew; series=\(series)")
    }
}

@Test func quotaEvictsTheOldestEntryFirst() throws {
    try MLXMetalTestLock.withLock {
        let dir = makeCacheDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        // Room for about two entries.
        let cache = DiskCache(cacheDir: dir, maxSizeBytes: 250_000, modelKey: "lru")

        store(cache, tokens: [1, 1, 1], approximateBytes: 100_000)
        #expect(cache.hasDurableEntry(tokens: [1, 1, 1]))
        // createdAt has second resolution in the index, so separate the writes.
        Thread.sleep(forTimeInterval: 1.1)
        store(cache, tokens: [2, 2, 2], approximateBytes: 100_000)
        Thread.sleep(forTimeInterval: 1.1)
        // This third write must push the FIRST one out, not the second.
        store(cache, tokens: [3, 3, 3], approximateBytes: 100_000)

        #expect(
            !cache.hasDurableEntry(tokens: [1, 1, 1]),
            "the oldest entry survived while newer ones were written")
        #expect(cache.hasDurableEntry(tokens: [3, 3, 3]), "the newest entry was evicted")
    }
}

@Test func loweringTheCapIsEnforcedOnTheNextStore() throws {
    try MLXMetalTestLock.withLock {
        let dir = makeCacheDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        // Fill comfortably under a roomy cap.
        let roomy = DiskCache(cacheDir: dir, maxSizeBytes: 2_000_000, modelKey: "recap")
        for i in 0..<6 { store(roomy, tokens: [i, 9, 9], approximateBytes: 100_000) }
        let before = roomy.snapshotStats()
        #expect(before.currentPayloadBytes > 300_000, "setup did not fill the cache")

        // Reopen the SAME directory with a much smaller cap — this is what
        // changing Disk Cache Size and reloading does.
        let tightened = DiskCache(cacheDir: dir, maxSizeBytes: 250_000, modelKey: "recap")
        store(tightened, tokens: [42, 42, 42], approximateBytes: 100_000)

        let after = tightened.snapshotStats()
        #expect(
            after.currentPayloadBytes <= 250_000,
            "the lowered cap was not enforced: \(after.currentPayloadBytes) > 250000")
        #expect(after.currentPayloadBytes < before.currentPayloadBytes)
    }
}

@Test func raisingTheCapStopsEvicting() throws {
    try MLXMetalTestLock.withLock {
        let dir = makeCacheDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let tight = DiskCache(cacheDir: dir, maxSizeBytes: 250_000, modelKey: "raise")
        for i in 0..<5 { store(tight, tokens: [i, 7, 7], approximateBytes: 100_000) }
        #expect(tight.snapshotStats().evictions > 0, "setup did not trigger eviction")

        // Same directory, generous cap: further stores must accumulate.
        let roomy = DiskCache(cacheDir: dir, maxSizeBytes: 5_000_000, modelKey: "raise")
        let baseline = roomy.snapshotStats().currentPayloadBytes
        for i in 10..<15 { store(roomy, tokens: [i, 7, 7], approximateBytes: 100_000) }
        let grown = roomy.snapshotStats()

        #expect(
            grown.currentPayloadBytes > baseline,
            "cache did not grow after the cap was raised")
        #expect(grown.evictions == 0, "evicted despite a cap with plenty of room")
    }
}

@Test func clearRemovesEveryPayloadIncludingOrphans() throws {
    try MLXMetalTestLock.withLock {
        let dir = makeCacheDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let cache = DiskCache(cacheDir: dir, maxSizeBytes: 5_000_000, modelKey: "clear")
        for i in 0..<4 { store(cache, tokens: [i, 5, 5], approximateBytes: 60_000) }
        #expect(cache.snapshotStats().currentPayloadBytes > 0)

        // An orphan: a payload on disk that the index does not know about,
        // exactly what a crash between the file write and the insert leaves.
        // Quota accounting reads only cache_entries, so this is invisible to
        // eviction and would otherwise occupy disk forever.
        let orphan = dir.appendingPathComponent("orphaned-by-crash.safetensors")
        try Data(repeating: 0xAB, count: 50_000).write(to: orphan)
        #expect(FileManager.default.fileExists(atPath: orphan.path))

        cache.clear()

        let after = cache.snapshotStats()
        #expect(after.currentPayloadBytes == 0)
        #expect(after.currentEntryCount == 0)
        #expect(
            !FileManager.default.fileExists(atPath: orphan.path),
            "clear left an orphaned payload behind — the one path that reclaims it")

        let remaining = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        #expect(
            remaining.allSatisfy { !$0.hasSuffix(".safetensors") },
            "safetensors survived clear: \(remaining)")
    }
}

@Test func quotaIsSharedAcrossModelsAndEvictsTheOtherModelsOldestEntry() throws {
    try MLXMetalTestLock.withLock {
        let dir = makeCacheDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        // Two models, ONE cache root. The quota pass reads cache_entries with no
        // model_key predicate, so model B's writes must be able to reclaim model
        // A's stale entries rather than each model getting its own budget.
        let modelA = DiskCache(cacheDir: dir, maxSizeBytes: 250_000, modelKey: "model-a")
        store(modelA, tokens: [1, 2, 3], approximateBytes: 100_000)
        #expect(modelA.hasDurableEntry(tokens: [1, 2, 3]))

        Thread.sleep(forTimeInterval: 1.1)

        let modelB = DiskCache(cacheDir: dir, maxSizeBytes: 250_000, modelKey: "model-b")
        store(modelB, tokens: [4, 5, 6], approximateBytes: 100_000)
        Thread.sleep(forTimeInterval: 1.1)
        store(modelB, tokens: [7, 8, 9], approximateBytes: 100_000)

        // Model A's older entry is the one that should have gone.
        #expect(
            !modelA.hasDurableEntry(tokens: [1, 2, 3]),
            "a stale OTHER model's entry survived while a hot model was capped")
        #expect(modelB.snapshotStats().currentPayloadBytes <= 250_000)
    }
}

@Test func warningThresholdTracksRealCacheGrowth() throws {
    try MLXMetalTestLock.withLock {
        let dir = makeCacheDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        // Ties the UI's 75% warning to a real filling cache rather than to
        // hand-made numbers: below the threshold before, at or above it after.
        let cap = 500_000
        let cache = DiskCache(cacheDir: dir, maxSizeBytes: cap, modelKey: "warn")

        store(cache, tokens: [1, 1, 1], approximateBytes: 100_000)
        let early = Double(cache.snapshotStats().currentPayloadBytes) / Double(cap)
        #expect(early < 0.75, "cache was already past the warning threshold at setup")

        for i in 2..<6 { store(cache, tokens: [i, 1, 1], approximateBytes: 100_000) }
        let late = Double(cache.snapshotStats().currentPayloadBytes) / Double(cap)
        #expect(late >= 0.75, "cache never reached the warning threshold; got \(late)")
    }
}
