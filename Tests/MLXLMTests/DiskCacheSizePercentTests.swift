import Foundation
import MLX
@testable import MLXLMCommon
import Testing

@Suite("Disk cache percentages and saved choices")
struct DiskCacheSizePercentTests {
    private func tempDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("cache-percent-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    @Test func percentWinsOverLegacyWithoutChangingItsUnits() {
        let result = DiskCacheCapPolicy.resolve(
            percent: 0.5, legacyGB: 99, totalBytes: 256_000_000_000,
            freeBytes: 100_000_000_000, ownBytes: 0)
        #expect(result.rule == .explicitPercent)
        #expect(result.capBytes == 1_280_000_000)
    }

    @Test func legacySizeSurvivesWhenNoPercentIsSet() {
        #expect(VMLXServerRuntimeSettings.resolveDiskCacheMaxGB(
            percent: nil, legacyGB: 42, directory: nil) == 42)
    }

    @Test func unknownVolumeDoesNotPretendToResolveAPercentage() {
        #expect(VMLXServerRuntimeSettings.resolveDiskCacheMaxGB(
            percent: 10, legacyGB: 42, directory: nil) == 10)
    }

    @Test(arguments: [0.0, -5, 1000, Double.infinity, Double.nan])
    func invalidPercentIsReported(percent: Double) {
        var settings = VMLXServerRuntimeSettings()
        settings.cache.blockDisk.maxSizePercent = percent
        #expect(settings.validationIssues().contains { $0.field == "cache.blockDisk.maxSizePercent" })
        let resolved = settings.cacheCoordinatorConfig(modelKey: "invalid-cap").diskCacheMaxGB
        #expect(resolved.isFinite && resolved > 0)
    }

    @Test(arguments: [Double?.none, 10, 1, 250])
    func migrationKeepsExistingGigabytes(previousGB: Double?) {
        var settings = VMLXServerRuntimeSettings()
        settings.schemaVersion = 2
        settings.cache.blockDisk.maxSizeGB = previousGB
        settings.migrateToCurrentSchema()
        #expect(settings.cache.blockDisk.maxSizeGB == previousGB)
        #expect(settings.cache.blockDisk.maxSizePercent == nil)
    }

    @Test(arguments: [Double?.none, 0.005, 10, 33])
    func migrationKeepsExistingPercentages(percent: Double?) {
        var settings = VMLXServerRuntimeSettings()
        settings.schemaVersion = 3
        settings.cache.blockDisk.maxSizePercent = percent
        for _ in 0..<3 { settings.migrateToCurrentSchema() }
        #expect(settings.cache.blockDisk.maxSizePercent == percent)
    }

    @Test func explicitTinySharesReachTheCoordinator() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let capacity = try #require(VMLXServerRuntimeSettings.cacheVolumeCapacityGB(for: dir))
        var settings = VMLXServerRuntimeSettings()
        settings.cache.blockDisk.directory = dir.path
        settings.cache.blockDisk.maxSizePercent = 0.0001
        let small = settings.cacheCoordinatorConfig(modelKey: "small").diskCacheMaxGB
        #expect(abs(Double(small) - capacity * 0.000001) < 0.0001)
        settings.cache.blockDisk.maxSizePercent = 0.0002
        let larger = settings.cacheCoordinatorConfig(modelKey: "large").diskCacheMaxGB
        #expect(larger > small)
    }

    // MARK: - End-to-end enforcement
    //
    // The wiring tests above prove the number ARRIVES at
    // `CacheCoordinatorConfig`. This proves the arriving number is actually
    // ENFORCED: a real DiskCache, real payloads written past the cap, and the
    // janitor trimming them. A cap that is plumbed correctly and never acted
    // on looks identical from the settings side.

    @Test("a percent-derived cap is enforced by the disk janitor")
    func percentDerivedCapIsEnforced() throws {
        try MLXMetalTestLock.withLock {
            let dir = tempDir()
            defer { try? FileManager.default.removeItem(at: dir) }

            // Choose a share that lands on a small, fillable cap for THIS
            // volume, so the test works on any machine rather than assuming a
            // disk size. Possible only because an explicit share is not
            // floored — the whole point of that fix.
            let capacity = try #require(
                VMLXServerRuntimeSettings.cacheVolumeCapacityGB(for: dir))
            let targetCapBytes = 400_000.0
            let percent = (targetCapBytes / 1_073_741_824.0) / capacity * 100.0

            var settings = VMLXServerRuntimeSettings()
            settings.schemaVersion = VMLXServerRuntimeSettings.contractVersion
            settings.cache.blockDisk.enabled = true
            settings.cache.blockDisk.directory = dir.path
            settings.cache.blockDisk.maxSizePercent = percent

            // Through the real config path the engine uses.
            let config = settings.cacheCoordinatorConfig(modelKey: "percent-enforced")
            let capBytes = Int(Double(config.diskCacheMaxGB) * 1_073_741_824.0)
            #expect(
                abs(Double(capBytes) - targetCapBytes) < 50_000,
                "settings resolved to \(capBytes) bytes, expected ~\(Int(targetCapBytes))")

            let cache = DiskCache(
                cacheDir: dir, maxSizeBytes: capBytes, modelKey: "percent-enforced")

            for i in 0..<12 {
                let count = Swift.max(1, 80_000 / 4)
                cache.store(
                    tokens: [i, i + 1, i + 2], arrays: ["k": MLXArray.zeros([count])])
            }

            let final = cache.snapshotStats()
            #expect(
                final.currentPayloadBytes <= capBytes,
                "payload \(final.currentPayloadBytes) exceeded the percent-derived cap \(capBytes)")
            // A cache that refused every write would also satisfy the line
            // above, so the janitor has to have actually run.
            #expect(final.evictions > 0, "nothing was evicted — the cap was not enforced")
            #expect(final.currentEntryCount > 0, "everything was evicted")
        }
    }

    /// Changing the share changes what gets evicted. This is the "toggle the
    /// setting and watch it take effect" case, end to end.
    @Test("raising the share lets the cache keep more, lowering it evicts")
    func changingTheShareChangesWhatSurvives() throws {
        try MLXMetalTestLock.withLock {
            let dir = tempDir()
            defer { try? FileManager.default.removeItem(at: dir) }
            let capacity = try #require(
                VMLXServerRuntimeSettings.cacheVolumeCapacityGB(for: dir))

            func capBytes(forTargetBytes target: Double) -> Int {
                var s = VMLXServerRuntimeSettings()
                s.schemaVersion = VMLXServerRuntimeSettings.contractVersion
                s.cache.blockDisk.enabled = true
                s.cache.blockDisk.directory = dir.path
                s.cache.blockDisk.maxSizePercent =
                    (target / 1_073_741_824.0) / capacity * 100.0
                return Int(
                    Double(s.cacheCoordinatorConfig(modelKey: "resize").diskCacheMaxGB)
                        * 1_073_741_824.0)
            }

            // Fill under a roomy share first.
            let roomy = DiskCache(
                cacheDir: dir, maxSizeBytes: capBytes(forTargetBytes: 2_000_000),
                modelKey: "resize")
            for i in 0..<8 {
                roomy.store(tokens: [i, i + 1], arrays: ["k": MLXArray.zeros([20_000])])
            }
            let beforeBytes = roomy.snapshotStats().currentPayloadBytes
            #expect(beforeBytes > 400_000, "setup did not fill the cache")

            // Now tighten the share. The next store must enforce the new cap.
            let tightCap = capBytes(forTargetBytes: 250_000)
            roomy.updateMaxSizeBytes(tightCap)
            let tightened = roomy
            tightened.store(tokens: [99, 100], arrays: ["k": MLXArray.zeros([20_000])])

            let after = tightened.snapshotStats()
            #expect(
                after.currentPayloadBytes <= tightCap,
                "lowering the share did not take effect: \(after.currentPayloadBytes) > \(tightCap)")
            #expect(after.currentPayloadBytes < beforeBytes, "nothing was trimmed")
            #expect(after.evictions > 0)
        }
    }

    @Test func capacityIsUnknownWithoutADirectory() {
        #expect(VMLXServerRuntimeSettings.cacheVolumeCapacityGB(for: nil) == nil)
    }
}
