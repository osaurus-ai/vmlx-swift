import Foundation
import MLX
@testable import MLXLMCommon
import XCTest

final class HybridBoundaryEarlySpillTests: XCTestCase {
    private var roots: [URL] = []

    override func tearDown() {
        roots.forEach { try? FileManager.default.removeItem(at: $0) }
        roots.removeAll()
        super.tearDown()
    }

    private func coordinator(disk: Bool = true, paged: Bool = false) -> CacheCoordinator {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        roots.append(root)
        let result = CacheCoordinator(config: .init(
            usePagedCache: paged, enableDiskCache: disk, diskCacheDir: root,
            modelKey: "early-spill-test"))
        result.setHybrid(true, requiresRecurrentSSMCompanion: true,
                         requiresSeparateRecurrentPayload: false)
        return result
    }

    private func snapshot() -> [KVCache] {
        let attention = KVCacheSimple()
        let rows = MLXArray((0..<32).map(Float.init)).reshaped(1, 1, 4, 8).asType(.bfloat16)
        _ = attention.update(keys: rows, values: rows + 1)
        let recurrent = MambaCache()
        recurrent.offset = 4
        recurrent.state = [MLXArray([Float(73), Float(4)]).reshaped(1, 1, 2)]
        return makePromptBoundaryCacheSnapshot(from: [attention, recurrent])
    }

    private func spill(_ snapshot: [KVCache], into coordinator: CacheCoordinator,
                       tokens: [Int] = [1, 2, 3, 4], salt: String = "image-A") -> Bool {
        storeCapturedHybridBoundary(
            snapshot, tokens: tokens, coordinator: coordinator, mediaSalt: salt,
            chainId: "chat-A", kvMode: .none, kvGroupSize: 64, quantizedKVStart: 0)
    }

    func testAcceptedSpillRestoresExactKVAndRecurrentStateFromDisk() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let disk = coordinator()
        let captured = snapshot()
        let expected = captured.map { $0.state.map { $0.asArray(Float.self) } }
        XCTAssertTrue(spill(captured, into: disk))
        // A different media identity must never inherit this checkpoint.
        XCTAssertFalse(disk.hasValidatedDiskEntry(tokens: [1, 2, 3, 4], mediaSalt: "image-B"))
        // Mutating caller-owned objects after synchronous publication must not
        // change the serialized recurrent state or attention values.
        (captured[1] as? MambaCache)?.state = [MLXArray.zeros([1, 1, 2])]
        let fresh = CacheCoordinator(config: .init(
            usePagedCache: false, enableDiskCache: true, diskCacheDir: roots[0],
            modelKey: "early-spill-test"))
        let arrays = try XCTUnwrap(fresh.diskCache?.fetch(tokens: [1, 2, 3, 4], mediaSalt: "image-A"))
        var restored: [KVCache] = [KVCacheSimple(), MambaCache()]
        XCTAssertEqual(restoreFromDiskArrays(arrays, into: &restored), 4)
        XCTAssertEqual(restored.map { $0.state.map { $0.asArray(Float.self) } }, expected)
    }

    func testRefusedSpillsLeaveSnapshotAvailableForTerminalStore() {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        for (disk, paged, tokens) in [
            (false, false, [1, 2, 3, 4]),
            (true, true, [1, 2, 3, 4]),
            (true, false, [1, 2, 3]),
        ] {
            let coordinator = coordinator(disk: disk, paged: paged)
            let captured = snapshot()
            let before = captured.map { $0.state.map { $0.asArray(Float.self) } }
            XCTAssertFalse(spill(captured, into: coordinator, tokens: tokens))
            XCTAssertEqual(captured.map { $0.state.map { $0.asArray(Float.self) } }, before)
            XCTAssertEqual(coordinator.snapshotStats().diskStats?.stores ?? 0, 0)
        }
    }

    func testQueuedWriterOwnsCheckpointAfterCallerReleasesIt() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let coordinator = coordinator()
        let disk = try XCTUnwrap(coordinator.diskCache)
        disk.writeBehind = true
        disk.writeDelayForTesting = 0.2
        var captured = snapshot()
        let expected = captured.map { $0.state.map { $0.asArray(Float.self) } }
        XCTAssertTrue(spill(captured, into: coordinator))
        XCTAssertEqual(disk.pendingWriteCount, 1)
        captured.removeAll()
        XCTAssertTrue(disk.waitForPendingWrites(timeout: 10))
        let reopened = DiskCache(cacheDir: roots[0], maxSizeBytes: 50_000_000,
                                 modelKey: "early-spill-test")
        let arrays = try XCTUnwrap(reopened.fetch(tokens: [1, 2, 3, 4], mediaSalt: "image-A"))
        var restored: [KVCache] = [KVCacheSimple(), MambaCache()]
        XCTAssertEqual(restoreFromDiskArrays(arrays, into: &restored), 4)
        XCTAssertEqual(restored.map { $0.state.map { $0.asArray(Float.self) } }, expected)
    }

    func testPublicationFailureDoesNotClaimSpillSucceeded() throws {
        let lock = lockSerializedMLXTest(); defer { lock.unlock() }
        let coordinator = coordinator()
        let disk = try XCTUnwrap(coordinator.diskCache)
        disk.publishFaultForTesting = { throw CocoaError(.fileWriteUnknown) }
        let captured = snapshot()
        XCTAssertFalse(spill(captured, into: coordinator))
        XCTAssertEqual(cacheBoundaryLeafOffsets(captured), [4, 4])
        XCTAssertFalse(coordinator.hasValidatedDiskEntry(
            tokens: [1, 2, 3, 4], mediaSalt: "image-A"))
        disk.publishFaultForTesting = nil
        XCTAssertTrue(spill(captured, into: coordinator),
                      "a failed early store must leave a usable checkpoint for retry")
    }
}
