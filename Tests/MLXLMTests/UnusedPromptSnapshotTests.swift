import MLX
@testable import MLXLMCommon
import XCTest

final class UnusedPromptSnapshotTests: XCTestCase {
    private final class CountingCache: BaseKVCache {
        var copyCount = 0

        override func copy() -> any KVCache {
            copyCount += 1
            return CountingCache()
        }
    }

    func testDisabledStorageDoesNotCopyAnyLayer() {
        let caches = [CountingCache(), CountingCache()]
        XCTAssertNil(makeRetainedExactPromptSnapshot(from: caches, storageEnabled: false))
        XCTAssertEqual(caches.map(\.copyCount), [0, 0])
    }

    func testEnabledStorageStillSealsAnIndependentSnapshot() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let cache = KVCacheSimple()
        let rows = MLXArray((0..<16).map(Float.init)).reshaped(1, 1, 2, 8)
        _ = cache.update(keys: rows, values: rows + 10)
        let snapshot = try XCTUnwrap(makeRetainedExactPromptSnapshot(from: [cache]))
        let before = snapshot[0].state.map { $0.asArray(Float.self) }
        _ = cache.update(keys: rows + 20, values: rows + 30)
        MLX.eval(cache)
        XCTAssertEqual(snapshot[0].offset, 2)
        XCTAssertEqual(cache.offset, 4)
        XCTAssertEqual(snapshot[0].state.map { $0.asArray(Float.self) }, before)
    }
}
