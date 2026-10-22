import Foundation
import MLX
@testable import MLXLMCommon
import XCTest

final class AppendOnlyPromptCheckpointTests: XCTestCase {
    func testOnlyMutableLayersRetainStateAndBothBoundariesRemainExact() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let full = AttentionSinkKVCache(headCount: 2, headDimension: 8)
        let ring = RotatingKVCache(maxSize: 5, keep: 0)
        let cache: [KVCache] = [full, ring]
        func append(_ n: Int) {
            let rows = MLXArray.full([1, 2, 1, 8], values: MLXArray(Float(n))).asType(.bfloat16)
            for layer in cache { _ = layer.update(keys: rows, values: rows + 1) }
            MLX.eval(cache)
        }
        for n in 0..<12 { append(n) }
        let seed = try XCTUnwrap(AppendOnlyPromptCheckpoint.capture(cache: cache, tokenCount: 12))
        let seedExpected = cache.map { $0.state.map { $0.asType(.float32).asArray(Float.self) } }
        append(12)
        let prompt = try XCTUnwrap(AppendOnlyPromptCheckpoint.capture(cache: cache, tokenCount: 13))
        let promptExpected = cache.map { $0.state.map { $0.asType(.float32).asArray(Float.self) } }
        for n in 13..<19 { append(n) }
        let restoredSeed = try XCTUnwrap(seed.materialize(from: cache))
        let restoredPrompt = try XCTUnwrap(prompt.materialize(from: cache))
        XCTAssertEqual(restoredSeed.map(\.offset), [12, 12])
        XCTAssertEqual(restoredPrompt.map(\.offset), [13, 13])
        XCTAssertTrue(restoredPrompt[0] is AttentionSinkKVCache)
        XCTAssertEqual(restoredSeed.map { $0.state.map { $0.asType(.float32).asArray(Float.self) } }, seedExpected)
        XCTAssertEqual(restoredPrompt.map { $0.state.map { $0.asType(.float32).asArray(Float.self) } }, promptExpected)
        let bytes = TQDiskSerializer.serialize(cache: restoredPrompt, ssmStates: nil,
                                               preserveStandardKVStorageDType: true)
        var destination: [KVCache] = [AttentionSinkKVCache(headCount: 2, headDimension: 8),
                                      RotatingKVCache(maxSize: 5, keep: 0)]
        XCTAssertEqual(restoreFromDiskArrays(bytes, into: &destination, requirePromptBoundary: true), 13)
        XCTAssertEqual(destination.map { $0.state.map { $0.asType(.float32).asArray(Float.self) } }, promptExpected)
    }

    func testCheckpointDoesNotOwnAppendOnlyCache() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        var source: KVCacheSimple? = KVCacheSimple()
        weak var weakSource = source
        let rows = MLXArray.ones([1, 1, 2, 8])
        _ = source!.update(keys: rows, values: rows)
        let checkpoint = try XCTUnwrap(AppendOnlyPromptCheckpoint.capture(cache: [source!], tokenCount: 2))
        source = nil
        XCTAssertNil(weakSource)
        XCTAssertNil(checkpoint.materialize(from: []))
    }

    func testSliceBeforeCopySealsOnlyRequestedPrefix() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let source = KVCacheSimple()
        let rows = MLXArray((0..<800).map(Float.init)).reshaped(1, 1, 100, 8)
        _ = source.update(keys: rows, values: rows + 1)
        let prefix = try XCTUnwrap(compactPromptPrefix([source], count: 3))
        XCTAssertEqual(prefix[0].offset, 3)
        XCTAssertEqual(prefix[0].innerState().map(\.shape), [[1, 1, 3, 8], [1, 1, 3, 8]])
        XCTAssertEqual(prefix[0].state[0].asArray(Float.self), Array(0..<24).map(Float.init))
        _ = source.update(keys: rows, values: rows)
        MLX.eval(source)
        XCTAssertEqual(prefix[0].state[0].asArray(Float.self), Array(0..<24).map(Float.init))
    }

    func testUnprovenCompanionTopologyCannotUseDeferredAppendOnlyState() {
        XCTAssertNil(AppendOnlyPromptCheckpoint.capture(cache: [MambaCache()], tokenCount: 1))
    }
}
