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
