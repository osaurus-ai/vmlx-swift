import Foundation
import MLX
import MLXNN
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

private final class SnapshotProbeModel: Module, LanguageModel, @unchecked Sendable {
    var vocabularySize: Int { 8 }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [KVCacheSimple()] }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(LMInput.Text(tokens: input.text.tokens.reshaped([-1])))
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let rows = inputs.reshaped(1, 1, -1, 1).asType(.float32)
        for layer in cache ?? [] { _ = layer.update(keys: rows, values: rows) }
        return MLXArray([Float(0), 0, 0, 0, 0, 0, 10, 0]).reshaped(1, 1, 8)
    }
}

extension UnusedPromptSnapshotTests {
    func testCoordinatorFreeIteratorKeepsLiveCacheWithoutCheckpoint() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
            model: SnapshotProbeModel(), parameters: GenerateParameters(maxTokens: 2, temperature: 0))
        XCTAssertNil(iterator.promptCacheSnapshot)
        XCTAssertEqual(iterator.next(), 6)
        XCTAssertEqual(iterator.next(), 6)
    }

    func testInstalledCoordinatorWithBothTiersOffDoesNotRetainCheckpoint() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false, enableDiskCache: false))
        XCTAssertFalse(coordinator.canPersistBoundaries)
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray([Int32(1), 2, 3])), model: SnapshotProbeModel(),
            parameters: GenerateParameters(maxTokens: 2, temperature: 0), cacheCoordinator: coordinator)
        XCTAssertNil(iterator.promptCacheSnapshot)
        XCTAssertEqual(iterator.next(), 6)
    }

    func testAuxiliaryIteratorDoesNotRetainAnUnpublishablePrompt() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false, enableDiskCache: true, diskCacheDir: directory, modelKey: "auxiliary-snapshot"))
        let input = LMInput(text: LMInput.Text(tokens: MLXArray([Int32(1), 2, 3])), cachePromptIntent: .auxiliary)
        var iterator = try TokenIterator(input: input, model: SnapshotProbeModel(),
            parameters: GenerateParameters(maxTokens: 2, temperature: 0), cacheCoordinator: coordinator)
        XCTAssertNil(iterator.promptCacheSnapshot)
        XCTAssertEqual(iterator.next(), 6)
        iterator.storeCacheAfterGeneration(generatedTokenIds: [6], includeGeneratedBoundary: true)
        XCTAssertNil(iterator.promptCacheSnapshot)
    }
}
