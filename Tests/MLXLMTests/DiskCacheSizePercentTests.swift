// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The disk-cache cap is a PERCENT of the volume, not a byte count.
//
// A GB figure is the wrong unit for this setting. KV cost scales with the
// model — a 27B stores ~256 KiB per token, so a 222k window needs ~54 GB —
// and the cap is shared across every model in the cache root. So any single
// number is simultaneously too small on a 4 TB machine and too large on a
// 256 GB one. Worse, shipping BOTH a percent and a GB control asks the user
// to reconcile two units that describe the same thing.
//
// One control, in percent. GB survives only so an install that already chose
// an explicit number keeps exactly that cap until migration converts it.
//
// The migration invariant is the important one: converting the unit must not
// change how much disk the user actually gets.

import Foundation
import MLX

@testable import MLXLMCommon
import Testing

@Suite("Disk cache size is a percent of the volume")
struct DiskCacheSizePercentTests {

    private func tempDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("cache-percent-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - Resolution order

    @Test("percent wins over a legacy GB value")
    func percentTakesPrecedenceOverLegacyGB() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let capacity = try #require(VMLXServerRuntimeSettings.cacheVolumeCapacityGB(for: dir))

        let resolved = VMLXServerRuntimeSettings.resolveDiskCacheMaxGB(
            percent: 25, legacyGB: 1.0, directory: dir)

        #expect(resolved > 1.0, "the stale GB value must not win")
        #expect(resolved == Swift.max(10.0, capacity * 0.25))
    }

    @Test("a legacy GB value is still honoured when no percent is set")
    func legacyGBStillHonoured() {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let resolved = VMLXServerRuntimeSettings.resolveDiskCacheMaxGB(
            percent: nil, legacyGB: 42.0, directory: dir)
        #expect(resolved == 42.0, "an install that chose a size must keep it")
    }

    @Test("neither set falls back to the default share")
    func unsetUsesTheDefaultShare() {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let resolved = VMLXServerRuntimeSettings.resolveDiskCacheMaxGB(
            percent: nil, legacyGB: nil, directory: dir)
        #expect(resolved == VMLXServerRuntimeSettings.autoDiskCacheMaxGB(for: dir))
    }

    /// A percent of a disk we cannot measure must not become a guess. The
    /// floor is the historical default, so this can only ever be generous.
    @Test("an unmeasurable volume falls back to the floor, never a small guess")
    func unmeasurableVolumeUsesFloor() {
        let resolved = VMLXServerRuntimeSettings.resolveDiskCacheMaxGB(
            percent: 10, legacyGB: nil, directory: nil)
        #expect(resolved == VMLXServerRuntimeSettings.autoDiskCacheFloorGB)
    }

    @Test("a non-positive percent is ignored rather than producing a zero cap")
    func nonPositivePercentIsIgnored() {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        for bad in [0.0, -5.0] {
            let resolved = VMLXServerRuntimeSettings.resolveDiskCacheMaxGB(
                percent: bad, legacyGB: nil, directory: dir)
            #expect(resolved > 0, "percent \(bad) produced a zero/negative cap")
        }
    }

    /// The floor guards AUTO, and only auto.
    ///
    /// Resolving an unset value must never lower anyone's cap — that is what
    /// the floor is for. But clamping a share the USER typed is an invented
    /// limit overriding an explicit choice: 1% of a 500 GB disk means 5 GB,
    /// and silently making it 10 GB is us deciding we know better.
    @Test("the floor applies to auto, never to a share the user chose")
    func floorAppliesToAutoNotToAnExplicitShare() {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        // Auto: floored.
        let auto = VMLXServerRuntimeSettings.resolveDiskCacheMaxGB(
            percent: nil, legacyGB: nil, directory: dir)
        #expect(auto >= VMLXServerRuntimeSettings.autoDiskCacheFloorGB)

        // Explicit tiny share: honoured exactly, not clamped up.
        let explicit = VMLXServerRuntimeSettings.resolveDiskCacheMaxGB(
            percent: 0.0001, legacyGB: nil, directory: dir)
        #expect(
            explicit < VMLXServerRuntimeSettings.autoDiskCacheFloorGB,
            "an explicit share was clamped up to the floor — that overrides the user")
        let capacity = try! #require(VMLXServerRuntimeSettings.cacheVolumeCapacityGB(for: dir))
        #expect(abs(explicit - capacity * 0.000001) < 0.0001)
    }

    // MARK: - Migration

    /// THE requirement: every updating install lands on 10%, once.
    ///
    /// Parameterised over the shapes real installs are actually in, because
    /// "it worked for the case I happened to try" is how the flat-10-GB
    /// default survived six releases.
    @Test(
        "every updating install is reset to ten percent, whatever it had before",
        arguments: [
            Double?.none,  // never set anything
            Double?.some(10.0),  // the old flat default
            Double?.some(1.0),  // hand-set small
            Double?.some(250.0),  // hand-set large
        ])
    func everyUpdatingInstallLandsOnTenPercent(previousGB: Double?) throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        var settings = VMLXServerRuntimeSettings()
        settings.schemaVersion = 2  // an install from before this change
        settings.cache.blockDisk.maxSizeGB = previousGB
        settings.cache.blockDisk.directory = dir.path

        settings.migrateToCurrentSchema()

        #expect(
            settings.cache.blockDisk.maxSizePercent == 10.0,
            "previous GB \(String(describing: previousGB)) did not land on 10%")
        #expect(settings.cache.blockDisk.maxSizeGB == nil, "left a stale GB value behind")
        #expect(settings.cache.legacyDisk.maxSizeGB == nil, "left a stale legacy GB value behind")

        // And it resolves to a real tenth of THIS volume, not a constant.
        let capacity = try #require(VMLXServerRuntimeSettings.cacheVolumeCapacityGB(for: dir))
        let resolved = VMLXServerRuntimeSettings.resolveDiskCacheMaxGB(
            percent: settings.cache.blockDisk.maxSizePercent,
            legacyGB: settings.cache.blockDisk.maxSizeGB,
            directory: dir)
        #expect(abs(resolved - Swift.max(10.0, capacity * 0.10)) < 0.001)
    }

    /// A brand-new install must agree with a migrated one. Two code paths
    /// producing two different caps is the drift this catches.
    @Test("a fresh install and a migrated install resolve to the same cap")
    func freshAndMigratedAgree() {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        var migrated = VMLXServerRuntimeSettings()
        migrated.schemaVersion = 1
        migrated.cache.blockDisk.maxSizeGB = 10.0
        migrated.cache.blockDisk.directory = dir.path
        migrated.migrateToCurrentSchema()

        var fresh = VMLXServerRuntimeSettings()
        fresh.cache.blockDisk.directory = dir.path
        fresh.migrateToCurrentSchema()

        let migratedGB = VMLXServerRuntimeSettings.resolveDiskCacheMaxGB(
            percent: migrated.cache.blockDisk.maxSizePercent,
            legacyGB: migrated.cache.blockDisk.maxSizeGB, directory: dir)
        let freshGB = VMLXServerRuntimeSettings.resolveDiskCacheMaxGB(
            percent: fresh.cache.blockDisk.maxSizePercent,
            legacyGB: fresh.cache.blockDisk.maxSizeGB, directory: dir)

        #expect(abs(migratedGB - freshGB) < 0.001, "\(migratedGB) vs \(freshGB)")
    }

    /// The reset raises the cap for anyone on the old flat default. On any
    /// disk larger than 100 GB, 10% beats 10 GB.
    @Test("the reset raises the cap for an install on the old flat default")
    func resetRaisesTheOldFlatDefault() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let capacity = try #require(VMLXServerRuntimeSettings.cacheVolumeCapacityGB(for: dir))
        try #require(capacity > 100, "this machine's volume is too small to show the difference")

        var settings = VMLXServerRuntimeSettings()
        settings.schemaVersion = 2
        settings.cache.blockDisk.maxSizeGB = 10.0
        settings.cache.blockDisk.directory = dir.path
        settings.migrateToCurrentSchema()

        let after = VMLXServerRuntimeSettings.resolveDiskCacheMaxGB(
            percent: settings.cache.blockDisk.maxSizePercent,
            legacyGB: settings.cache.blockDisk.maxSizeGB, directory: dir)
        #expect(after > 10.0, "the whole point was that 10 GB was too small")
    }

    // MARK: - Wiring
    //
    // A setting that is stored correctly and never reaches the engine is the
    // exact defect shape this codebase has shipped before. These assert the
    // value arrives at the object that actually enforces the quota.

    @Test("the migrated percent reaches CacheCoordinatorConfig")
    func percentReachesTheCoordinatorConfig() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        var settings = VMLXServerRuntimeSettings()
        settings.schemaVersion = 2
        settings.cache.blockDisk.maxSizeGB = 10.0
        settings.cache.blockDisk.enabled = true
        settings.cache.blockDisk.directory = dir.path
        settings.migrateToCurrentSchema()

        let config = settings.cacheCoordinatorConfig(modelKey: "wiring-probe")
        let capacity = try #require(VMLXServerRuntimeSettings.cacheVolumeCapacityGB(for: dir))

        #expect(
            abs(Double(config.diskCacheMaxGB) - Swift.max(10.0, capacity * 0.10)) < 0.5,
            "the coordinator got \(config.diskCacheMaxGB) GB, not 10% of the volume")
    }

    /// Changing the percent must change what the engine enforces — otherwise
    /// the control is decorative.
    @Test("editing the percent changes the cap the coordinator enforces")
    func editingPercentChangesTheEnforcedCap() {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        func enforcedGB(percent: Double) -> Double {
            var s = VMLXServerRuntimeSettings()
            s.schemaVersion = VMLXServerRuntimeSettings.contractVersion
            s.cache.blockDisk.enabled = true
            s.cache.blockDisk.directory = dir.path
            s.cache.blockDisk.maxSizePercent = percent
            return Double(s.cacheCoordinatorConfig(modelKey: "probe").diskCacheMaxGB)
        }

        let small = enforcedGB(percent: 10)
        let large = enforcedGB(percent: 40)
        #expect(large > small, "40% enforced \(large) GB, 10% enforced \(small) GB")
    }

    @Test("migration is version-gated and does not re-run")
    func migrationDoesNotReRun() {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        var settings = VMLXServerRuntimeSettings()
        settings.schemaVersion = VMLXServerRuntimeSettings.contractVersion
        settings.cache.blockDisk.maxSizeGB = 7.0
        settings.cache.blockDisk.directory = dir.path
        settings.migrateToCurrentSchema()

        #expect(
            settings.cache.blockDisk.maxSizeGB == 7.0,
            "a deliberate choice made after migrating must survive")
        #expect(settings.cache.blockDisk.maxSizePercent == nil)
    }

    /// A percent the user chooses AFTER updating is theirs and must survive
    /// every later launch.
    ///
    /// This is the case that matters. A percent on a pre-v3 record is not a
    /// real state — the field did not exist then — so asserting that such a
    /// value survives would be testing a condition no install can be in.
    @Test("a size chosen after updating survives later launches")
    func userChoiceAfterMigrationSurvives() {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        var settings = VMLXServerRuntimeSettings()
        settings.schemaVersion = 2
        settings.cache.blockDisk.maxSizeGB = 10.0
        settings.cache.blockDisk.directory = dir.path
        settings.migrateToCurrentSchema()
        #expect(settings.cache.blockDisk.maxSizePercent == 10.0, "the one-time reset")

        // The user then picks their own share.
        settings.cache.blockDisk.maxSizePercent = 33

        // Every subsequent launch re-runs migration; it must be a no-op now.
        for _ in 0..<3 { settings.migrateToCurrentSchema() }
        #expect(
            settings.cache.blockDisk.maxSizePercent == 33,
            "the one-time reset ran again and overwrote a deliberate choice")
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
            let tightened = DiskCache(
                cacheDir: dir, maxSizeBytes: tightCap, modelKey: "resize")
            tightened.store(tokens: [99, 100], arrays: ["k": MLXArray.zeros([20_000])])

            let after = tightened.snapshotStats()
            #expect(
                after.currentPayloadBytes <= tightCap,
                "lowering the share did not take effect: \(after.currentPayloadBytes) > \(tightCap)")
            #expect(after.currentPayloadBytes < beforeBytes, "nothing was trimmed")
            #expect(after.evictions > 0)
        }
    }

    // MARK: - Capacity readout

    /// The UI shows "N% of <disk> ≈ X GB" using this, so it must agree with
    /// what the runtime enforces rather than being a second estimate.
    @Test("capacity readout matches the number the resolver uses")
    func capacityReadoutMatchesResolver() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let capacity = try #require(VMLXServerRuntimeSettings.cacheVolumeCapacityGB(for: dir))
        #expect(capacity > 0)

        let resolved = VMLXServerRuntimeSettings.resolveDiskCacheMaxGB(
            percent: 50, legacyGB: nil, directory: dir)
        #expect(abs(resolved - capacity * 0.5) < 0.001)
    }

    @Test("capacity is nil rather than a fabricated number when unreadable")
    func capacityIsNilWhenUnreadable() {
        #expect(VMLXServerRuntimeSettings.cacheVolumeCapacityGB(for: nil) == nil)
    }

    // MARK: - Default

    @Test("the shipped default is ten percent")
    func defaultShareIsTenPercent() {
        #expect(VMLXServerRuntimeSettings.autoDiskCacheFraction == 0.10)
    }
}
