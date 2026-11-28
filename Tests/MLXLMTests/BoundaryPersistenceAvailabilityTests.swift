import Foundation
import MLXLMCommon
import XCTest

/// Capability checks must not request snapshots for a sentinel-only RAM pool.
final class BoundaryPersistenceAvailabilityTests: XCTestCase {
    private func coordinator(paged: Bool = true, blocks: Int = 2, disk: URL? = nil) -> CacheCoordinator {
        CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: paged, enableDiskCache: disk != nil,
            pagedBlockSize: 4, maxCacheBlocks: blocks,
            diskCacheMaxGB: 0.01, diskCacheDir: disk,
            modelKey: "boundary-persistence-availability"))
    }

    func testSentinelOnlyPoolCannotPersistBoundaryPayloads() {
        XCTAssertFalse(coordinator(blocks: 1).canPersistBoundaries)
    }

    func testOneDataBlockMakesPagedPersistenceAvailable() {
        XCTAssertTrue(coordinator(blocks: 2).canPersistBoundaries)
    }

    func testDisabledPagedTierHasNoPersistenceWithoutDisk() {
        XCTAssertFalse(coordinator(paged: false).canPersistBoundaries)
    }

    func testIncompatiblePagedTopologyHasNoPersistenceWithoutDisk() {
        let c = coordinator()
        c.setPagedIncompatible(true)
        XCTAssertFalse(c.canPersistBoundaries)
    }

    func testDiskTierRemainsAvailableForSentinelOnlyOrIncompatiblePool() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let c = coordinator(blocks: 1, disk: directory)
        XCTAssertTrue(c.canPersistBoundaries)
        c.setPagedIncompatible(true)
        XCTAssertTrue(c.canPersistBoundaries)
    }
}
