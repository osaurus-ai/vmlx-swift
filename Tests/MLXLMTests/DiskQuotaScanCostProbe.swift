import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Measurement probe, not a regression test: times the combined KV +
/// recurrent-companion disk quota scan at fixed entry counts.
///
/// Gated on `VMLX_QUOTA_PROBE=1` so an ordinary suite run never pays for it.
/// The cap is far above the fixture size, so no sample evicts anything and
/// each one measures the scan alone.
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
        for _ in 0..<6 {
            samples.append(sampleNanos(body))
        }
        return samples.dropFirst().sorted().map { Double($0) / 1_000_000 }
    }

    private static func ms(_ value: Double) -> String {
        String(format: "%.3f", value)
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
            let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
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
            for index in 0..<entries {
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
            #expect(companionBefore.allSatisfy { entry in
                entry.kvHash.map(kvHashes.contains) ?? false
            })
            #expect(kvBefore.allSatisfy { $0.bytes > 0 })
            #expect(companionBefore.allSatisfy { $0.bytes > 0 })
            let fixtureBytes = kvBefore.reduce(Int64(0)) { $0 + $1.bytes }
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
                "QUOTA_PROBE entries=\(entries) median_ms=\(Self.ms(full[2])) "
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
                "QUOTA_PROBE_PARTS entries=\(entries) "
                    + "kv_quotaEntries_median_ms=\(Self.ms(kvPart[2])) "
                    + "companion_quotaEntries_median_ms=\(Self.ms(companionPart[2]))")

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
