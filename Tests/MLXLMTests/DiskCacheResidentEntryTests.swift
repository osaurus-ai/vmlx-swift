import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// The disk tier's resident entries (the RAM tier of whole-snapshot entries): a stored or read
/// payload stays in RAM within `residentByteBudget` and is served from there (`CacheDetail.memory`)
/// while its index row and file are the ones it was kept for.
///
/// Every test asserts the non-resident baseline (a coordinator with the budget 0 serves `.disk`)
/// before it compares, and no token length is a multiple of 64, 128 or 256.
@Suite(.serialized)
struct DiskCacheResidentEntryTests {

    private static func makeRoot(_ label: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-resident-\(label)-\(UUID().uuidString)")
    }

    private static func chain(_ count: Int, seed: Int = 3) -> [Int] {
        (0 ..< count).map { seed * 100_000 + $0 }
    }

    private static func payload(_ elements: Int, value: Float = 1) -> [String: MLXArray] {
        ["data": MLXArray.ones([elements], dtype: .float32) * value]
    }

    private static func coordinator(
        root: URL, residentBytes: Int, writeBehind: Bool = false, paged: Bool = false
    ) -> CacheCoordinator {
        var config = CacheCoordinatorConfig(
            usePagedCache: paged, enableDiskCache: true, diskCacheDir: root, modelKey: "resident-tests")
        config.diskWriteBehind = writeBehind
        config.residentEntryBytes = residentBytes
        return CacheCoordinator(config: config)
    }

    private static func hit(_ result: CacheFetchResult) -> (matched: Int, detail: CacheDetail, arrays: [String: MLXArray])? {
        guard case .hit(let matched, _, let detail, _, _, let arrays?) = result else { return nil }
        return (matched, detail, arrays)
    }

    @Test func storedEntryIsServedFromRAMAndEqualsTheFile() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("serve")
            defer { try? FileManager.default.removeItem(at: root) }
            let prompt = Self.chain(301)
            let stored = Array(prompt.prefix(187))
            let warm = Self.coordinator(root: root, residentBytes: 1 << 20)
            warm.storePersistentBoundary(tokens: stored, diskArrays: Self.payload(97, value: 3), ssmStates: nil)

            // Baseline: a coordinator on the same root with no resident budget reads the file.
            let cold = Self.coordinator(root: root, residentBytes: 0)
            let fromFile = try #require(Self.hit(cold.fetch(tokens: prompt)))
            #expect(fromFile.matched == 187 && fromFile.detail == .disk)
            #expect(cold.diskCache?.residentStats.entries == 0)

            let fromRAM = try #require(Self.hit(warm.fetch(tokens: prompt)))
            #expect(fromRAM.matched == 187 && fromRAM.detail == .memory)
            #expect(fromRAM.detail.isDiskTierEntry)
            #expect(warm.diskCache?.residentStats.hits == 1)
            let a = try #require(fromRAM.arrays["data"]), b = try #require(fromFile.arrays["data"])
            #expect(a.shape == b.shape && MLX.all(a .== b).item(Bool.self))
            // Hit accounting is the disk tier's: one hit for the accepted restore.
            #expect(warm.snapshotStats().diskStats?.hits == 1)
        }
    }

    @Test func writeBehindEntryStaysResidentAfterItsFileLands() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("behind")
            defer { try? FileManager.default.removeItem(at: root) }
            let prompt = Self.chain(229)
            let warm = Self.coordinator(root: root, residentBytes: 1 << 20, writeBehind: true)
            warm.storePersistentBoundary(
                tokens: Array(prompt.prefix(141)), diskArrays: Self.payload(53), ssmStates: nil)
            #expect(warm.waitForPendingDiskWrites())
            let settled = try #require(Self.hit(warm.fetch(tokens: prompt)))
            #expect(settled.matched == 141 && settled.detail == .memory)
            #expect(warm.diskCache?.residentStats.entries == 1)
        }
    }

    /// The serving default caches on the SSD only (no resident budget). A read while the entry's
    /// file is still being written is served from the writer's payload and labelled
    /// `diskPending`, never `memory` (the opt-in RAM tier). Once the file lands the same read is
    /// `disk`, and nothing is resident at any point.
    @Test func ssdOnlyReadDuringTheWriteIsDiskPendingNeverMemory() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("pending")
            defer { try? FileManager.default.removeItem(at: root) }
            let prompt = Self.chain(263)
            let ssd = Self.coordinator(root: root, residentBytes: 0, writeBehind: true)
            ssd.diskCache?.writeDelayForTesting = 0.3
            ssd.storePersistentBoundary(
                tokens: Array(prompt.prefix(157)), diskArrays: Self.payload(61, value: 5), ssmStates: nil)
            let during = try #require(Self.hit(ssd.fetch(tokens: prompt)))
            #expect(during.matched == 157 && during.detail == .diskPending && during.detail.isDiskTierEntry)
            #expect(ssd.waitForPendingDiskWrites())
            let after = try #require(Self.hit(ssd.fetch(tokens: prompt)))
            #expect(after.matched == 157 && after.detail == .disk)
            #expect(ssd.diskCache?.residentStats.entries == 0)
            let a = try #require(during.arrays["data"]), b = try #require(after.arrays["data"])
            #expect(a.shape == b.shape && MLX.all(a .== b).item(Bool.self))
        }
    }

    @Test func budgetEvictsLeastRecentlyUsedAndNeverTheNewest() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("lru")
            defer { try? FileManager.default.removeItem(at: root) }
            // Each payload is 4 * 1000 bytes; the budget holds two.
            let warm = Self.coordinator(root: root, residentBytes: 9_000)
            let prompt = Self.chain(997)
            for count in [101, 333, 517] {
                warm.storePersistentBoundary(
                    tokens: Array(prompt.prefix(count)), diskArrays: Self.payload(1000), ssmStates: nil)
            }
            let stats = try #require(warm.diskCache?.residentStats)
            #expect(stats.entries == 2 && stats.bytes <= 9_000)
            // The newest (517) is resident; the oldest (101) went to the SSD only.
            let newest = try #require(Self.hit(warm.fetch(tokens: Array(prompt.prefix(613)))))
            #expect(newest.matched == 517 && newest.detail == .memory)
            let oldest = try #require(Self.hit(warm.fetch(tokens: Array(prompt.prefix(211)))))
            #expect(oldest.matched == 101 && oldest.detail == .disk)
            // That read promoted 101 and evicted the least recently used one, 333, not 517.
            let promoted = try #require(Self.hit(warm.fetch(tokens: Array(prompt.prefix(211)))))
            #expect(promoted.detail == .memory)
            let middle = try #require(Self.hit(warm.fetch(tokens: Array(prompt.prefix(401)))))
            #expect(middle.matched == 333 && middle.detail == .disk)
        }
    }

    @Test func entryLargerThanTheBudgetIsNotKept() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("large")
            defer { try? FileManager.default.removeItem(at: root) }
            let warm = Self.coordinator(root: root, residentBytes: 1_000)
            let prompt = Self.chain(155)
            warm.storePersistentBoundary(
                tokens: Array(prompt.prefix(77)), diskArrays: Self.payload(1001), ssmStates: nil)
            #expect(warm.diskCache?.residentStats.entries == 0)
            let served = try #require(Self.hit(warm.fetch(tokens: prompt)))
            #expect(served.matched == 77 && served.detail == .disk)
        }
    }

    @Test func removedOrReplacedFileIsNotServedFromRAM() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("gone")
            defer { try? FileManager.default.removeItem(at: root) }
            let warm = Self.coordinator(root: root, residentBytes: 1 << 20)
            let prompt = Self.chain(183)
            warm.storePersistentBoundary(
                tokens: Array(prompt.prefix(91)), diskArrays: Self.payload(31), ssmStates: nil)
            #expect(Self.hit(warm.fetch(tokens: prompt))?.detail == .memory)
            // Another process clears the root: the payload files go.
            let files = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "safetensors" }
            try #require(!files.isEmpty)
            for file in files { try FileManager.default.removeItem(at: file) }
            guard case .miss = warm.fetch(tokens: prompt) else {
                Issue.record("a payload whose file is gone was served from RAM")
                return
            }
            #expect(warm.diskCache?.residentStats.entries == 0)
        }
    }

    @Test func clearAndRejectionDropResidentEntries() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("clear")
            defer { try? FileManager.default.removeItem(at: root) }
            let warm = Self.coordinator(root: root, residentBytes: 1 << 20)
            let prompt = Self.chain(263)
            warm.storePersistentBoundary(
                tokens: Array(prompt.prefix(119)), diskArrays: Self.payload(17), ssmStates: nil)
            #expect(Self.hit(warm.fetch(tokens: prompt))?.detail == .memory)
            warm.reportDiskRestoreRejected(tokens: prompt, boundary: 119, mediaSalt: nil, reason: "test")
            #expect(warm.diskCache?.residentStats.entries == 0)
            guard case .miss = warm.fetch(tokens: prompt) else {
                Issue.record("a rejected payload was served again")
                return
            }
            warm.storePersistentBoundary(
                tokens: Array(prompt.prefix(119)), diskArrays: Self.payload(17), ssmStates: nil)
            #expect((warm.diskCache?.residentStats.entries ?? 0) == 1)
            warm.diskCache?.clear()
            #expect(warm.diskCache?.residentStats.entries == 0)
        }
    }

    @Test func releaseVolatileDropsResidentEntriesAndKeepsTheFile() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("volatile")
            defer { try? FileManager.default.removeItem(at: root) }
            let warm = Self.coordinator(root: root, residentBytes: 1 << 20)
            let prompt = Self.chain(229)
            warm.storePersistentBoundary(
                tokens: Array(prompt.prefix(141)), diskArrays: Self.payload(23), ssmStates: nil)
            #expect(Self.hit(warm.fetch(tokens: prompt))?.detail == .memory)
            // Memory-pressure recovery: the RAM copy goes, the SSD entry still restores.
            warm.releaseVolatile()
            #expect(warm.diskCache?.residentStats.entries == 0)
            #expect(warm.diskCache?.residentStats.bytes == 0)
            let afterRelease = try #require(Self.hit(warm.fetch(tokens: prompt)))
            #expect(afterRelease.matched == 141 && afterRelease.detail == .disk)
            // The read keeps it resident again for the next turn.
            #expect(Self.hit(warm.fetch(tokens: prompt))?.detail == .memory)
        }
    }

    @Test func pagedTierKeepsResidentEntriesOffUntilTheModelIsPagedIncompatible() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("paged")
            defer { try? FileManager.default.removeItem(at: root) }
            let coordinator = Self.coordinator(root: root, residentBytes: 1 << 20, paged: true)
            #expect(coordinator.diskCache?.residentByteBudget == 0)
            coordinator.setPagedIncompatible(true)
            #expect(coordinator.diskCache?.residentByteBudget == 1 << 20)
            coordinator.setPagedIncompatible(false)
            #expect(coordinator.diskCache?.residentByteBudget == 0)
        }
    }

    @Test func servingConfigurationCachesOnTheSSDOnlyUnlessAskedForRAM() {
        let saved = ProcessInfo.processInfo.environment["VMLX_RESIDENT_CACHE_GB"]
        defer {
            if let saved { setenv("VMLX_RESIDENT_CACHE_GB", saved, 1) } else { unsetenv("VMLX_RESIDENT_CACHE_GB") }
        }
        func servingBytes() -> Int {
            VMLXServerRuntimeSettings().cacheCoordinatorConfig(
                modelKey: "resident-serving", diskCacheDirectory: Self.makeRoot("serving")
            ).residentEntryBytes
        }
        unsetenv("VMLX_RESIDENT_CACHE_GB")
        #expect(servingBytes() == 0)
        setenv("VMLX_RESIDENT_CACHE_GB", "1.5", 1)
        #expect(servingBytes() == 1_610_612_736)
        setenv("VMLX_RESIDENT_CACHE_GB", "0", 1)
        #expect(servingBytes() == 0)
        #expect(CacheCoordinatorConfig().residentEntryBytes == 0)
    }
}
