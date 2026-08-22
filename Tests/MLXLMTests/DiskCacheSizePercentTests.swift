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

    /// Advises upward, never downward — the rule the whole auto-sizing change
    /// was built on.
    @Test("the resolved cap never drops below the historical default")
    func neverResolvesBelowTheFloor() {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        // 0.0001% of any real volume is far under the floor.
        let resolved = VMLXServerRuntimeSettings.resolveDiskCacheMaxGB(
            percent: 0.0001, legacyGB: nil, directory: dir)
        #expect(resolved >= VMLXServerRuntimeSettings.autoDiskCacheFloorGB)
    }

    // MARK: - Migration

    /// THE invariant: changing the unit must not change the amount of disk.
    @Test("migrating a GB install preserves the cap it actually had")
    func migrationPreservesTheEffectiveCap() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let capacity = try #require(VMLXServerRuntimeSettings.cacheVolumeCapacityGB(for: dir))
        // Comfortably above the floor so the clamp cannot mask a bad conversion.
        let chosenGB = Swift.max(50.0, capacity * 0.2)

        var settings = VMLXServerRuntimeSettings()
        settings.schemaVersion = 2
        settings.cache.blockDisk.maxSizeGB = chosenGB
        settings.cache.blockDisk.directory = dir.path

        let before = VMLXServerRuntimeSettings.resolveDiskCacheMaxGB(
            percent: nil, legacyGB: chosenGB, directory: dir)
        settings.migrateToCurrentSchema()
        let after = VMLXServerRuntimeSettings.resolveDiskCacheMaxGB(
            percent: settings.cache.blockDisk.maxSizePercent,
            legacyGB: settings.cache.blockDisk.maxSizeGB,
            directory: dir)

        #expect(settings.cache.blockDisk.maxSizePercent != nil, "did not convert to a percent")
        #expect(settings.cache.blockDisk.maxSizeGB == nil, "left a stale GB value behind")
        #expect(
            abs(after - before) < 0.5,
            "cap changed on update: \(before) GB -> \(after) GB")
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

    /// A user who already set a percent is not overwritten by a stale GB.
    @Test("an existing percent survives migration untouched")
    func existingPercentIsNotOverwritten() {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        var settings = VMLXServerRuntimeSettings()
        settings.schemaVersion = 2
        settings.cache.blockDisk.maxSizePercent = 33
        settings.cache.blockDisk.maxSizeGB = 5
        settings.cache.blockDisk.directory = dir.path
        settings.migrateToCurrentSchema()

        #expect(settings.cache.blockDisk.maxSizePercent == 33)
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
