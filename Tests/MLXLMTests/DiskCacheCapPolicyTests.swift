import Foundation
import SQLite3
import Testing

@testable import MLXLMCommon

@Suite struct DiskCacheCapPolicyTests {
    private let gb: Int64 = 1_073_741_824

    @Test func unreadableIndexPreservesAnotherModelsLiveRootCap() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let liveLimit = SharedDiskCacheLimit.forRoot(root, initialBytes: 42 * Int(gb))
        try Data("not a sqlite database".utf8).write(to: root.appendingPathComponent("cache_index.db"))
        let resolved = DiskCacheCapPolicy.resolve(percent: nil, legacyGB: nil, directory: root)
        #expect(resolved.rule == .unknownVolume)
        #expect(resolved.capBytes == Int64(liveLimit.bytes))
        let second = SharedDiskCacheLimit.forRoot(root, initialBytes: Int(resolved.capBytes))
        #expect(second.bytes == 42 * Int(gb))
    }

    @Test(arguments: [Double?.none, 10.0, 0.005])
    func cacheGrowthDoesNotRatchetTheCap(percent: Double?) {
        let empty = DiskCacheCapPolicy.resolve(
            percent: percent, legacyGB: nil, totalBytes: 256 * gb,
            freeBytes: 40 * gb, ownBytes: 0)
        let filled = DiskCacheCapPolicy.resolve(
            percent: percent, legacyGB: nil, totalBytes: 256 * gb,
            freeBytes: 30 * gb, ownBytes: 10 * gb)
        #expect(empty.capBytes == filled.capBytes)
        #expect(empty.rule == filled.rule)
    }

    @Test func automaticUsesThirtyPercentWithoutASecondCeiling() {
        let result = DiskCacheCapPolicy.resolve(
            percent: nil, legacyGB: nil, totalBytes: 256 * gb,
            freeBytes: 30 * gb, ownBytes: 10 * gb)
        #expect(result.capBytes == 12 * gb)
        #expect(result.rule == .automatic)
        #expect(!result.limitedByHost)
    }

    @Test func explicitPercentKeepsItsUnitsAndWinsOverLegacy() {
        let result = DiskCacheCapPolicy.resolve(
            percent: 0.5, legacyGB: 99, totalBytes: 256 * gb,
            freeBytes: 100 * gb, ownBytes: 0)
        #expect(result.capBytes == Int64(Double(256 * gb) * 0.005))
        #expect(result.rule == .explicitPercent)
        #expect(!result.limitedByHost)
    }

    @Test func explicitCeilingNamesTheRequestedAndEffectiveCap() {
        let result = DiskCacheCapPolicy.resolve(
            percent: 10, legacyGB: nil, totalBytes: 1000 * gb,
            freeBytes: 60 * gb, ownBytes: 20 * gb)
        #expect(result.requestedBytes == 100 * gb)
        #expect(result.capBytes == 20 * gb)
        #expect(result.limitedByHost)
    }

    @Test(arguments: [0, 1, 2 * 1_073_741_824] as [Int64])
    func lowDiskIsAdvisoryAndDoesNotInventATenGBFloor(free: Int64) {
        let result = DiskCacheCapPolicy.resolve(
            percent: nil, legacyGB: nil, totalBytes: 256 * gb,
            freeBytes: free, ownBytes: 0)
        #expect(result.lowFreeSpace)
        #expect(result.capBytes == max(1, Int64(Double(free) * 0.30)))
    }

    @Test func unreadableIndexIsNotAnEmptyIndex() {
        let result = DiskCacheCapPolicy.resolve(
            percent: nil, legacyGB: nil, totalBytes: 256 * gb,
            freeBytes: 30 * gb, ownBytes: nil, previousCapBytes: 20 * gb)
        #expect(result.rule == .unknownVolume)
        #expect(result.capBytes == 20 * gb)
    }

    @Test func unknownVolumeUsesHistoricalFallback() {
        let result = DiskCacheCapPolicy.resolve(
            percent: nil, legacyGB: nil, totalBytes: nil, freeBytes: nil, ownBytes: 0)
        #expect(result.rule == .unknownVolume)
        #expect(result.capBytes == 10 * gb)
        #expect(!result.lowFreeSpace)
    }

    @Test func extremeCountsSaturateWithoutWrapping() {
        let result = DiskCacheCapPolicy.resolve(
            percent: 100, legacyGB: nil, totalBytes: .max,
            freeBytes: .max, ownBytes: .max)
        #expect(result.capBytes > 0)
        #expect(result.capBytes <= .max)
        #expect(
            DiskCacheCapPolicy.byteLimit(gigabytes: Float(Double(Int.max) / 1_073_741_824)) == .max)
        #expect(DiskCacheCapPolicy.byteLimit(gigabytes: .infinity) == .max)
        #expect(DiskCacheCapPolicy.byteLimit(gigabytes: .nan) == 0)
    }

    @Test func indexMeasurementCountsCompanionsAndDoesNotCreateAMissingDirectory() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let missing = DiskCacheVolumeSnapshot.read(directory: root)
        #expect(missing.ownBytes == 0)
        #expect(missing.totalBytes != nil)
        #expect(missing.freeBytes != nil)
        #expect(!FileManager.default.fileExists(atPath: root.path))

        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var db: OpaquePointer?
        #expect(sqlite3_open(root.appendingPathComponent("cache_index.db").path, &db) == SQLITE_OK)
        let connection = try #require(db)
        defer { sqlite3_close(connection) }
        #expect(
            sqlite3_exec(
                connection,
                """
                CREATE TABLE cache_entries (file_size INTEGER, companion_bytes INTEGER);
                CREATE TABLE legacy_companions (bytes INTEGER);
                INSERT INTO cache_entries VALUES (101, 203), (NULL, 307), (-41, 409);
                INSERT INTO legacy_companions VALUES (503), (-47);
                """, nil, nil, nil) == SQLITE_OK)
        #expect(DiskCacheVolumeSnapshot.read(directory: root).ownBytes == 1523)
    }

    @Test func invalidExistingIndexIsUnknownInsteadOfZero() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("not a database".utf8).write(to: root.appendingPathComponent("cache_index.db"))
        #expect(DiskCacheVolumeSnapshot.read(directory: root).ownBytes == nil)
    }

    @Test func onlyDiskSizeChangesAvoidModelReload() {
        let initial = VMLXServerCacheSettings()
        var size = initial
        size.blockDisk.maxSizePercent = 0.005
        size.blockDisk.maxSizeGB = 7
        size.legacyDisk.maxSizeGB = 9
        #expect(!initial.requiresModelReload(comparedTo: size))
        var toggle = size
        toggle.prefix.enabled.toggle()
        #expect(initial.requiresModelReload(comparedTo: toggle))
        toggle = size
        toggle.blockDisk.enabled.toggle()
        #expect(initial.requiresModelReload(comparedTo: toggle))
        toggle = size
        toggle.pagedKV.enabled.toggle()
        #expect(initial.requiresModelReload(comparedTo: toggle))
        toggle = size
        toggle.blockDisk.directory = "/another/cache"
        #expect(initial.requiresModelReload(comparedTo: toggle))
        toggle = size
        toggle.defaultMaxKVSize = 16384
        #expect(initial.requiresModelReload(comparedTo: toggle))
    }

    @Test func increasingTheCapRetiresAnOutdatedOversizedNotice() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let disk = DiskCache(cacheDir: root, maxSizeBytes: 100)
        disk.recordQuotaPass(
            evictedGroups: 1, evictedBytes: 151, milliseconds: 1,
            event: DiskCachePressureEvent(
                kind: .activeTipDropped, chainId: "chat", tipBytes: 151, capBytes: 100))
        #expect(disk.snapshotStats().lastPressureEvent?.kind == .activeTipDropped)
        disk.updateMaxSizeBytes(200)
        #expect(disk.snapshotStats().lastPressureEvent == nil)
        #expect(disk.snapshotStats().pressureEventSeq == 1, "historical counters do not reset")
    }

    @Test func failedDeletionDoesNotReportLostProgress() {
        let rows = [
            QuotaRow(id: "old", tokenCount: 10, bytes: 151, recency: 1, isStableRoot: false, chainId: "chat", isLegacyCompanion: false),
            QuotaRow(id: "tip", tokenCount: 20, bytes: 151, recency: 2, isStableRoot: false, chainId: "chat", isLegacyCompanion: false),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 100, activeChain: "chat")
        #expect(plan.event?.kind == .activeTipDropped)
        #expect(plan.confirmedEvent(rows: rows, lostRows: []) == nil)
        #expect(plan.confirmedEvent(rows: rows, lostRows: ["old"]) == nil)
        #expect(plan.confirmedEvent(rows: rows, lostRows: ["tip"])?.kind == .activeTipDropped)
    }

    @Test func eventOrderAcrossModelsDoesNotDependOnTheirLocalCounters() {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let first = DiskCache(cacheDir: root, maxSizeBytes: 100, modelKey: "a")
        let second = DiskCache(cacheDir: root, maxSizeBytes: 100, modelKey: "b")
        let event = DiskCachePressureEvent(
            kind: .activeTipDropped, chainId: "chat", tipBytes: 151, capBytes: 100)
        for _ in 0 ..< 3 {
            first.recordQuotaPass(
                evictedGroups: 1, evictedBytes: 151, milliseconds: 20, event: event)
        }
        second.recordQuotaPass(evictedGroups: 1, evictedBytes: 151, milliseconds: 2, event: event)
        let a = first.snapshotStats()
        let b = second.snapshotStats()
        #expect(a.pressureEventSeq > b.pressureEventSeq)
        #expect(a.lastQuotaPassMs > b.lastQuotaPassMs)
        #expect(a.lastPressureEventTick < b.lastPressureEventTick)
        #expect(a.lastQuotaPassTick < b.lastQuotaPassTick)
    }
}
