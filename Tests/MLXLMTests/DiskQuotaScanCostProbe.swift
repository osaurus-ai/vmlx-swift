import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Measurement probe, not a regression test: times the combined KV +
/// recurrent-companion disk quota scan at fixed entry counts.
///
/// Gated on `VMLX_QUOTA_PROBE=1` so an ordinary suite run never pays for it.
/// The cap is far above the fixture size, so no sample evicts anything and
/// each one measures the scan alone — except the last step, which lowers the
/// cap once and times the one evicting pass that follows (`QUOTA_PROBE_OVERCAP`).
@Suite(.serialized)
struct DiskQuotaScanCostProbe {

    /// Nanoseconds for one call of `body`.
    private static func sampleNanos(_ body: () -> Void) -> UInt64 {
        let start = DispatchTime.now().uptimeNanoseconds
        body()
        return DispatchTime.now().uptimeNanoseconds - start
    }

    /// Six consecutive calls, the first discarded, the remaining five sorted
    /// ascending and converted to milliseconds.
    private static func sortedSteadyMillis(_ body: () -> Void) -> [Double] {
        var samples: [UInt64] = []
        for _ in 0 ..< 6 {
            samples.append(sampleNanos(body))
        }
        return samples.dropFirst().sorted().map { Double($0) / 1_000_000 }
    }

    private static func ms(_ value: Double) -> String {
        String(format: "%.3f", value)
    }

    /// The numbers differ several-fold between configurations, so every
    /// result line says which one produced it.
    private static var buildConfiguration: String {
        #if DEBUG
            return "debug"
        #else
            return "release"
        #endif
    }

    @Test(
        .enabled(if: ProcessInfo.processInfo.environment["VMLX_QUOTA_PROBE"] == "1"),
        arguments: [100, 1_000, 5_003])  // 5003 deliberately not round
    func scanCost(entries: Int) throws {
        let lockRequested = DispatchTime.now().uptimeNanoseconds
        try MLXMetalTestLock.withLock {
            let wallStart = DispatchTime.now().uptimeNanoseconds
            let lockWaitSeconds = Double(wallStart - lockRequested) / 1_000_000_000
            let root = FileManager.default.temporaryDirectory
                .appendingPathComponent("vmlx-quota-scan-probe-\(entries)-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: root) }

            let modelKey = "quota-scan-probe-model"
            let coordinator = CacheCoordinator(
                config: CacheCoordinatorConfig(
                    usePagedCache: false,
                    enableDiskCache: true,
                    diskCacheMaxGB: 64,
                    diskCacheDir: root,
                    modelKey: modelKey))
            coordinator.setHybrid(true, requiresRecurrentSSMCompanion: true)
            let disk = try #require(coordinator.diskCache)
            let companion = try #require(coordinator.ssmStateCache.diskStore)

            // Populate through the two stores' own write paths, exactly as
            // `storePersistentBoundary` does, but with the per-store quota pass
            // deferred and the combined pass not run per entry: running it per
            // entry would make fixture construction quadratic in the very scan
            // being measured.
            let kv = ["data": MLXArray.ones([16], dtype: .float32)]
            let recurrent = [MLXArray.ones([16], dtype: .float32)]
            let populateStart = DispatchTime.now().uptimeNanoseconds
            for index in 0 ..< entries {
                let tokens = [1_000_000 + index, 1, 2, 3]
                disk.store(tokens: tokens, arrays: kv, enforceQuota: false)
                try companion.store(
                    ssmStates: recurrent,
                    tokens: tokens,
                    boundary: tokens.count,
                    enforceQuota: false)
            }
            let populateSeconds =
                Double(DispatchTime.now().uptimeNanoseconds - populateStart) / 1_000_000_000

            // Fail closed: an empty or half-written fixture must fail here,
            // before any timing, rather than report a fast scan of nothing.
            let kvBefore = disk.quotaEntries()
            let companionBefore = companion.quotaEntries()
            let kvHashes = Set(kvBefore.map(\.hash))
            #expect(kvBefore.count == entries)
            #expect(companionBefore.count == entries)
            #expect(kvHashes.count == entries)
            // Hard requirements: an unlinked or empty fixture measures a
            // different scan, so it must never reach a QUOTA_PROBE line.
            try #require(
                companionBefore.allSatisfy { entry in
                    entry.kvHash.map(kvHashes.contains) ?? false
                })
            try #require(kvBefore.allSatisfy { $0.bytes > 0 })
            try #require(companionBefore.allSatisfy { $0.bytes > 0 })
            let fixtureBytes =
                kvBefore.reduce(Int64(0)) { $0 + $1.bytes }
                + companionBefore.reduce(Int64(0)) { $0 + $1.bytes }
            #expect(fixtureBytes < Int64(disk.maxSizeBytes) / 100)
            try #require(kvBefore.count == entries && companionBefore.count == entries)

            let full = Self.sortedSteadyMillis {
                coordinator.enforceCombinedDiskQuota()
            }

            // The scan must not have evicted anything, or the samples above
            // measured a shrinking fixture.
            let kvAfter = disk.quotaEntries().count
            let companionAfter = companion.quotaEntries().count
            #expect(kvAfter == entries)
            #expect(companionAfter == entries)

            print(
                "QUOTA_PROBE entries=\(entries) build=\(Self.buildConfiguration) median_ms=\(Self.ms(full[2])) "
                    + "min_ms=\(Self.ms(full[0])) max_ms=\(Self.ms(full[4])) "
                    + "kv_rows=\(kvAfter) companion_entries=\(companionAfter)")

            // The two halves, on the same store instances the coordinator scans.
            var kvSeen = 0
            let kvPart = Self.sortedSteadyMillis {
                kvSeen = disk.quotaEntries().count
            }
            var companionSeen = 0
            let companionPart = Self.sortedSteadyMillis {
                companionSeen = companion.quotaEntries().count
            }
            #expect(kvSeen == entries)
            #expect(companionSeen == entries)

            print(
                "QUOTA_PROBE_PARTS entries=\(entries) build=\(Self.buildConfiguration) "
                    + "kv_quotaEntries_median_ms=\(Self.ms(kvPart[2])) "
                    + "kv_quotaEntries_min_ms=\(Self.ms(kvPart[0])) "
                    + "kv_quotaEntries_max_ms=\(Self.ms(kvPart[4])) "
                    + "companion_quotaEntries_median_ms=\(Self.ms(companionPart[2])) "
                    + "companion_quotaEntries_min_ms=\(Self.ms(companionPart[0])) "
                    + "companion_quotaEntries_max_ms=\(Self.ms(companionPart[4]))")

            // What the host's idle stats poll pays. Fail closed on the value it
            // returns: a fast poll that lost the companion bytes is not a result.
            var polledBytes = -1
            var polledEntries = -1
            let statsPoll = Self.sortedSteadyMillis {
                let stats = coordinator.snapshotStats().diskStats
                polledBytes = stats?.currentPayloadBytes ?? -1
                polledEntries = stats?.currentEntryCount ?? -1
            }
            try #require(Int64(polledBytes) == fixtureBytes)
            try #require(polledEntries == entries)

            print(
                "QUOTA_PROBE_STATS entries=\(entries) build=\(Self.buildConfiguration) "
                    + "snapshotStats_median_ms=\(Self.ms(statsPoll[2])) "
                    + "snapshotStats_min_ms=\(Self.ms(statsPoll[0])) "
                    + "snapshotStats_max_ms=\(Self.ms(statsPoll[4])) "
                    + "polled_bytes=\(polledBytes) polled_entries=\(polledEntries)")

            // Over the cap: the pass that actually evicts. A second coordinator
            // on the same root with a cap a tenth below the fixture runs ONE
            // pass at open, and that pass has to evict about a tenth of the
            // entries. Destructive, so it is a single sample — the line says
            // so — and it runs after every measurement above.
            if entries >= 1_000 {
                let cap = fixtureBytes - fixtureBytes / 10
                let tight = CacheCoordinator(
                    config: CacheCoordinatorConfig(
                        usePagedCache: false,
                        enableDiskCache: true,
                        diskCacheMaxGB: Float(cap) / 1_073_741_824,
                        diskCacheDir: root,
                        modelKey: modelKey))
                let tightDisk = try #require(tight.diskCache)
                let timing = tight.lastQuotaPassTimingForTesting
                let remaining = tightDisk.quotaEntries().count
                let stats = try #require(tight.snapshotStats().diskStats)
                // Fail closed: a pass that evicted nothing, or everything,
                // timed something else.
                try #require(timing.evictedGroups == entries - remaining)
                try #require(timing.evictedGroups >= entries * 8 / 100)
                try #require(timing.evictedGroups <= entries * 12 / 100)
                try #require(stats.quotaPasses == 1 && stats.evictions == timing.evictedGroups)
                try #require(stats.currentPayloadBytes <= tightDisk.maxSizeBytes)
                try #require(companion.quotaEntries().count == remaining)
                try #require(timing.totalMs > 0 && stats.lastQuotaPassMs == timing.totalMs)
                print(
                    "QUOTA_PROBE_OVERCAP entries=\(entries) build=\(Self.buildConfiguration) "
                        + "evicted=\(timing.evictedGroups) pass_ms=\(Self.ms(timing.totalMs)) "
                        + "select_ms=\(Self.ms(timing.selectMs)) delete_ms=\(Self.ms(timing.deleteMs)) "
                        + "rows_ms=\(Self.ms(timing.rowsMs)) samples=1")
            }

            // Timed explicitly so the wall line accounts for fixture teardown;
            // the `defer` above remains the cleanup on every failing path.
            let cleanupStart = DispatchTime.now().uptimeNanoseconds
            try? FileManager.default.removeItem(at: root)
            let now = DispatchTime.now().uptimeNanoseconds
            let cleanupSeconds = Double(now - cleanupStart) / 1_000_000_000
            let wallSeconds = Double(now - wallStart) / 1_000_000_000
            print(
                "QUOTA_PROBE_WALL entries=\(entries) "
                    + "lock_wait_s=\(String(format: "%.2f", lockWaitSeconds)) "
                    + "populate_s=\(String(format: "%.2f", populateSeconds)) "
                    + "cleanup_s=\(String(format: "%.2f", cleanupSeconds)) "
                    + "total_s=\(String(format: "%.2f", wallSeconds)) "
                    + "fixture_bytes=\(fixtureBytes)")
        }
    }
}
