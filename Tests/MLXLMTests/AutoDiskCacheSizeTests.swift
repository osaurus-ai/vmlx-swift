import Foundation
@testable import MLXLMCommon
import Testing

// MARK: - Auto disk-cache size
//
// The disk-cache cap used to be a flat `?? 10.0` GB. That could not hold ONE
// full-context conversation of a 27B: at 64 layers / 4 KV heads / head_dim 256
// the KV cost is 256 KiB per token at bf16, so a 222k window needs ~54 GB — and
// the cap is shared across every model in the cache root, not per bundle. Past
// roughly 18% of a single window every store had to evict earlier boundaries of
// the SAME conversation, so reuse collapsed exactly as context grew.
//
// "Auto" resolves to a share of the cache volume instead. These tests pin the
// two properties that make that safe to ship: it can only ever RAISE the cap,
// and an unreadable volume degrades to the historical default rather than to
// something smaller.

@Test func autoDiskCacheSizeIsAtLeastTheHistoricalFlatDefault() {
    // The floor is the old default, so no existing user can lose cache by
    // upgrading. This is the "advises upward, never refuses" property.
    #expect(VMLXServerRuntimeSettings.autoDiskCacheFloorGB == 10.0)

    let tmp = FileManager.default.temporaryDirectory
    let resolved = VMLXServerRuntimeSettings.autoDiskCacheMaxGB(for: tmp)
    #expect(
        resolved >= VMLXServerRuntimeSettings.autoDiskCacheFloorGB,
        "auto must never resolve below the historical flat default; got \(resolved)")
}

@Test func autoDiskCacheSizeTracksTenPercentOfVolumeCapacity() throws {
    let tmp = FileManager.default.temporaryDirectory
    let capacity = try tmp.resourceValues(forKeys: [.volumeTotalCapacityKey])
        .volumeTotalCapacity
    let capacityGB = Double(try #require(capacity)) / 1_073_741_824.0
    let expected = Swift.max(
        VMLXServerRuntimeSettings.autoDiskCacheFloorGB,
        capacityGB * VMLXServerRuntimeSettings.autoDiskCacheFraction)

    let resolved = VMLXServerRuntimeSettings.autoDiskCacheMaxGB(for: tmp)
    #expect(abs(resolved - expected) < 0.5)
    #expect(VMLXServerRuntimeSettings.autoDiskCacheFraction == 0.10)
}

@Test func autoDiskCacheSizeResolvesForADirectoryThatDoesNotExistYet() {
    // First run: the cache directory has not been created. Resolution walks up
    // to the nearest existing ancestor rather than failing to the floor, so a
    // fresh install still gets the full auto size.
    let missing = FileManager.default.temporaryDirectory
        .appendingPathComponent("auto-disk-\(UUID().uuidString)")
        .appendingPathComponent("nested")
        .appendingPathComponent("deeper")
    #expect(!FileManager.default.fileExists(atPath: missing.path))

    let resolved = VMLXServerRuntimeSettings.autoDiskCacheMaxGB(for: missing)
    let viaExistingParent = VMLXServerRuntimeSettings.autoDiskCacheMaxGB(
        for: FileManager.default.temporaryDirectory)
    #expect(abs(resolved - viaExistingParent) < 0.5)
}

@Test func autoDiskCacheSizeFallsBackToFloorWhenThereIsNoDirectory() {
    // Detection failure must degrade to the old behaviour, never to a smaller
    // cap — a failed probe must not become a user-facing restriction.
    #expect(
        VMLXServerRuntimeSettings.autoDiskCacheMaxGB(for: nil)
            == VMLXServerRuntimeSettings.autoDiskCacheFloorGB)
}

@Test func explicitUserSizeOverridesAutoAndReachesTheCoordinatorConfig() {
    // The setting must still win — auto only fills in for nil. Verified through
    // `cacheCoordinatorConfig`, the real mapping the runtime uses, not by
    // re-reading the settings struct.
    var settings = VMLXServerRuntimeSettings()
    settings.cache.blockDisk.enabled = true
    settings.cache.blockDisk.maxSizeGB = 3.5

    let config = settings.cacheCoordinatorConfig(modelKey: "auto-disk-explicit")
    #expect(config.diskCacheMaxGB == 3.5)
}

@Test func unsetSizeResolvesToAutoInTheCoordinatorConfig() {
    // The end-to-end property: leaving the field blank yields the auto size, and
    // that size is at least the floor. This is the mapping that was previously
    // hardcoded to 10.0.
    var settings = VMLXServerRuntimeSettings()
    settings.cache.blockDisk.enabled = true
    settings.cache.blockDisk.maxSizeGB = nil

    let config = settings.cacheCoordinatorConfig(modelKey: "auto-disk-unset")
    #expect(Double(config.diskCacheMaxGB) >= VMLXServerRuntimeSettings.autoDiskCacheFloorGB)
}

// MARK: - Existing-install migration
//
// Auto only fills in for a NIL size. Every install that had already persisted
// the literal 10.0 would therefore have kept 10 GB forever after updating, and
// only fresh installs would have benefited. These pin the one-shot that moves
// those users to auto — and, just as importantly, that it does not run twice.

@Test func legacyTenGigabyteSettingMigratesToAuto() {
    var settings = VMLXServerRuntimeSettings()
    settings.cache.blockDisk.maxSizeGB = 10.0
    settings.schemaVersion = nil  // written before versioning existed

    settings.migrateToCurrentSchema()

    #expect(settings.cache.blockDisk.maxSizeGB == nil, "legacy 10 GB must become auto")
    #expect(settings.schemaVersion == VMLXServerRuntimeSettings.contractVersion)
}

/// BEHAVIOUR CHANGE, schema v3: the one-time reset moves EVERY updating
/// install onto 10% of its own disk, including one that had hand-set a GB
/// figure. Previously such a value was left alone.
///
/// That is the intent, not an accident. The setting is a share of the disk
/// now, and a stale absolute number is exactly what the change exists to
/// retire — the old flat 10 GB could not hold one full-context conversation
/// of a 27B (~54 GB at a 222k window), and any GB figure is wrong for some
/// machine. It happens once; a size chosen after updating survives.
@Test func migrationResetsADeliberateGigabyteSizeToTheShare() {
    var settings = VMLXServerRuntimeSettings()
    settings.cache.blockDisk.maxSizeGB = 40.0
    settings.schemaVersion = nil

    settings.migrateToCurrentSchema()

    #expect(settings.cache.blockDisk.maxSizeGB == nil, "stale absolute size must be cleared")
    #expect(settings.cache.blockDisk.maxSizePercent == 10.0)
}

@Test func migrationDoesNotRunTwiceOnADeliberateTenGigabyteChoice() {
    // The dangerous case: a user who has ALREADY been migrated and then
    // deliberately picks 10 GB. A blanket "rewrite any 10.0" would silently
    // undo that on every launch.
    var settings = VMLXServerRuntimeSettings()
    settings.cache.blockDisk.maxSizeGB = 10.0
    settings.schemaVersion = VMLXServerRuntimeSettings.contractVersion

    settings.migrateToCurrentSchema()

    #expect(
        settings.cache.blockDisk.maxSizeGB == 10.0,
        "a post-migration choice of 10 GB belongs to the user and must survive")
}

@Test func migrationIsIdempotent() {
    var settings = VMLXServerRuntimeSettings()
    settings.cache.blockDisk.maxSizeGB = 10.0
    settings.schemaVersion = nil

    settings.migrateToCurrentSchema()
    let afterFirst = settings
    settings.migrateToCurrentSchema()

    #expect(settings == afterFirst)
}

@Test func migratedSettingsResolveToTheAutoSizeThroughTheCoordinator() {
    // End to end: a legacy install ends up with a cap at least the floor,
    // rather than the 10 GB it was pinned to before.
    var settings = VMLXServerRuntimeSettings()
    settings.cache.blockDisk.enabled = true
    settings.cache.blockDisk.maxSizeGB = 10.0
    settings.schemaVersion = nil

    settings.migrateToCurrentSchema()
    let config = settings.cacheCoordinatorConfig(modelKey: "auto-disk-migrated")

    #expect(Double(config.diskCacheMaxGB) >= VMLXServerRuntimeSettings.autoDiskCacheFloorGB)
}

@Test func aFreshSettingsValueIsAlreadyAtTheCurrentSchema() {
    var settings = VMLXServerRuntimeSettings()
    settings.migrateToCurrentSchema()
    // Against `contractVersion` rather than a literal: the assertion is that a
    // fresh value needs no migration, which stays true at every schema bump.
    // Pinning the number made adding a migration look like a regression.
    #expect(settings.schemaVersion == VMLXServerRuntimeSettings.contractVersion)
    #expect(settings.cache.blockDisk.maxSizeGB == nil)
    // A fresh install lands on the same explicit 10% an updating install gets,
    // so both paths resolve to one cap rather than one going through "unset →
    // auto" and the other through a stored percent.
    #expect(settings.cache.blockDisk.maxSizePercent == 10.0)
}
