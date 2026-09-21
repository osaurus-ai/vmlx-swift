import Foundation
@testable import MLXLMCommon
import Testing

@Test func automaticCacheSizeTracksAvailableSpace() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let volume = DiskCacheVolumeSnapshot.read(directory: root)
    let free = try #require(volume.freeBytes)
    let resolved = VMLXServerRuntimeSettings.autoDiskCacheMaxGB(for: root)
    // This integration check reads a live volume twice; unrelated disk writes
    // can move it. Exact arithmetic is covered by DiskCacheCapPolicyTests.
    #expect(abs(resolved - Double(free) * 0.30 / 1_073_741_824) < 0.5)
    #expect(VMLXServerRuntimeSettings.autoDiskCacheFraction == 0.30)
    #expect(!FileManager.default.fileExists(atPath: root.path))
}

@Test func unknownVolumeRetainsTheHistoricalFallback() {
    #expect(VMLXServerRuntimeSettings.autoDiskCacheMaxGB(for: nil) == 10)
}

@Test func explicitUserSizeReachesTheCoordinatorConfig() {
    var settings = VMLXServerRuntimeSettings()
    settings.cache.blockDisk.maxSizeGB = 3.5
    #expect(settings.cacheCoordinatorConfig(modelKey: "explicit").diskCacheMaxGB == 3.5)
}

@Test(arguments: [Int?.none, 1, 2, 3, 4])
func migratingPreservesExplicitSizes(version: Int?) {
    var settings = VMLXServerRuntimeSettings()
    settings.schemaVersion = version
    settings.cache.blockDisk.maxSizePercent = 10
    settings.cache.blockDisk.maxSizeGB = 40
    settings.cache.legacyDisk.maxSizeGB = 7
    settings.migrateToCurrentSchema()
    #expect(settings.schemaVersion == VMLXServerRuntimeSettings.contractVersion)
    #expect(settings.cache.blockDisk.maxSizePercent == 10)
    #expect(settings.cache.blockDisk.maxSizeGB == 40)
    #expect(settings.cache.legacyDisk.maxSizeGB == 7)
    let first = settings
    settings.migrateToCurrentSchema()
    #expect(settings == first)
}

@Test func freshInstallRemainsAutomatic() {
    var settings = VMLXServerRuntimeSettings()
    settings.migrateToCurrentSchema()
    #expect(settings.schemaVersion == VMLXServerRuntimeSettings.contractVersion)
    #expect(settings.cache.blockDisk.maxSizePercent == nil)
    #expect(settings.cache.blockDisk.maxSizeGB == nil)
}

@Test func migrationDoesNotDowngradeANewerSettingsSchema() {
    var settings = VMLXServerRuntimeSettings()
    settings.schemaVersion = VMLXServerRuntimeSettings.contractVersion + 1
    let newer = settings.schemaVersion
    settings.migrateToCurrentSchema()
    #expect(settings.schemaVersion == newer)
}
