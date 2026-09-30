import Foundation
import MLX
import MLXLMCommon
import XCTest
@testable import MLXVLM

final class Qwen25VLTextPrefillTests: XCTestCase {
    private func model() throws -> Qwen25VL {
        let json = #"""
        {"model_type":"qwen2_5_vl","vocab_size":32,"hidden_size":16,"tie_word_embeddings":true,
         "num_hidden_layers":1,"intermediate_size":32,"num_attention_heads":2,"num_key_value_heads":1,
         "rope_scaling":{"mrope_section":[1,1,2]},"sliding_window":32,"use_sliding_window":false,"max_window_layers":1,
         "image_token_id":25,"video_token_id":26,"vision_start_token_id":27,"vision_end_token_id":28,"vision_token_id":29,
         "vision_config":{"depth":1,"hidden_size":16,"intermediate_size":32,"num_heads":2,"patch_size":2,"out_hidden_size":16,"spatial_merge_size":2,"spatial_patch_size":2,"temporal_patch_size":1,"window_size":8,"fullatt_block_indexes":[0],"tokens_per_second":2}}
        """#
        return Qwen25VL(try JSONDecoder().decode(Qwen25VLConfiguration.self, from: Data(json.utf8)))
    }
    private func logits(_ result: PrepareResult) throws -> MLXArray {
        guard case .logits(let output) = result else { throw NSError(domain: "fixture", code: 1) }
        return output.logits
    }
    func testSingleBatchAndFlatTextAgree() throws {
        try MLXMetalTestLock.withLock {
            let m = try model(), ids = MLXArray([Int32(1),2,3,4])
            let a = [KVCacheSimple()], b = [KVCacheSimple()]
            let x = try logits(m.prepare(LMInput(tokens: ids), cache: a, windowSize: 4))
            let y = try logits(m.prepare(LMInput(tokens: ids.reshaped(1,4)), cache: b, windowSize: 4))
            XCTAssertEqual(x.asArray(Float.self), y.asArray(Float.self))
            XCTAssertEqual(a[0].offset, 4); XCTAssertEqual(b[0].offset, 4)
            XCTAssertEqual(a[0].state.count, b[0].state.count)
            for (x, y) in zip(a[0].state, b[0].state) {
                XCTAssertEqual(x.shape, y.shape)
                XCTAssertEqual(x.asArray(Float.self), y.asArray(Float.self))
            }
        }
    }
    func testMalformedRanksRefuseBeforeCacheMutation() throws {
        try MLXMetalTestLock.withLock {
            let m = try model()
            for ids in [MLXArray(Int32(1)), MLXArray([Int32(1),2]).reshaped(2,1), MLXArray([Int32(1),2]).reshaped(1,1,2)] {
                let cache = [KVCacheSimple()]
                XCTAssertThrowsError(try m.prepare(LMInput(tokens: ids), cache: cache, windowSize: 2))
                XCTAssertEqual(cache[0].offset, 0)
            }
        }
    }
    func testLongTextChunkingMatchesLastLogitsAndReportsProgress() throws {
        try MLXMetalTestLock.withLock {
            let m = try model(), ids = MLXArray((0..<33).map { Int32($0 % 30 + 1) })
            let a = [KVCacheSimple()], b = [KVCacheSimple()]
            let progress = Progress()
            let chunked = try PrefillProgressReporter.withHandler({ progress.add($0) }) {
                try logits(m.prepare(LMInput(tokens: ids), cache: a, windowSize: 8))
            }
            let full = try logits(m.prepare(LMInput(tokens: ids), cache: b, windowSize: 64))
            let delta = abs(chunked[0..., -1, 0...] - full[0..., -1, 0...]).max().item(Float.self)
            XCTAssertLessThan(delta, 0.001)
            XCTAssertEqual(progress.values, [8,16,24,32,33]); XCTAssertEqual(a[0].offset, 33)
            XCTAssertEqual(b[0].offset, 33)
            XCTAssertEqual(a[0].state.count, b[0].state.count)
            for (x,y) in zip(a[0].state,b[0].state) {
                XCTAssertEqual(x.shape,y.shape)
                XCTAssertLessThan(abs(x-y).max().item(Float.self), 0.001)
            }
            let next = MLXArray([Int32(3)]).reshaped(1,1)
            XCTAssertLessThan(abs(m(next, cache: a) - m(next, cache: b)).max().item(Float.self), 0.001)
        }
    }
    func testUncachedPrepareKeepsFullContextAndAllTrueMask() throws {
        try MLXMetalTestLock.withLock {
            let m = try model(), ids = MLXArray([Int32(1),2,3,4]).reshaped(1,4)
            let input = LMInput(text: .init(tokens: ids, mask: MLXArray.ones([1,4], dtype: .bool)))
            let actual = try logits(m.prepare(input, cache: [], windowSize: 1))
            let expected = m(ids, cache: nil)
            XCTAssertEqual(actual.shape, [1,4,32])
            XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
        }
    }
    func testIntegerAllOneMasksMatchUnmaskedText() throws {
        try MLXMetalTestLock.withLock {
            let m = try model(), ids = MLXArray([Int32(1),2,3,4])
            let referenceCache = [KVCacheSimple()]
            let expected = try logits(m.prepare(LMInput(tokens: ids), cache: referenceCache, windowSize: 2))
            for dtype: DType in [.int8, .uint8, .int32, .int64] {
                let cache = [KVCacheSimple()]
                let input = LMInput(text: .init(tokens: ids, mask: MLXArray.ones([4], dtype: dtype)))
                let actual = try logits(m.prepare(input, cache: cache, windowSize: 2))
                XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
                XCTAssertEqual(cache[0].offset, 4)
                for (a, b) in zip(cache[0].state, referenceCache[0].state) {
                    XCTAssertEqual(a.asArray(Float.self), b.asArray(Float.self))
                }
            }
        }
    }
    func testInvalidStepEmptyTokensAndPaddingRejectBeforeMutation() throws {
        try MLXMetalTestLock.withLock {
            let m = try model(), ids = MLXArray([Int32(1),2])
            let cache = [KVCacheSimple()]
            for step in [0,-1] {
                XCTAssertThrowsError(try m.prepare(LMInput(tokens: ids), cache: cache, windowSize: step))
            }
            XCTAssertThrowsError(try m.prepare(LMInput(tokens: MLXArray([Int32]())), cache: cache, windowSize: 2))
            let masked = LMInput(text: .init(tokens: ids, mask: MLXArray([true,false])))
            XCTAssertThrowsError(try m.prepare(masked, cache: cache, windowSize: 2))
            XCTAssertThrowsError(try m.prepare(LMInput(tokens: MLXArray([Float(1),2])), cache: cache, windowSize: 2))
            XCTAssertThrowsError(try m.prepare(LMInput(tokens: ids), cache: [KVCacheSimple(), KVCacheSimple()], windowSize: 2))
            let wrongMask = LMInput(text: .init(tokens: ids, mask: MLXArray.ones([1,2], dtype: .bool)))
            XCTAssertThrowsError(try m.prepare(wrongMask, cache: cache, windowSize: 2))
            for mask in [MLXArray([Int8(1),0]), MLXArray([Int8(1),2]),
                         MLXArray([Int8(1),-1]), MLXArray.ones([2], dtype: .float32)] {
                XCTAssertThrowsError(try m.prepare(LMInput(text: .init(tokens: ids, mask: mask)),
                    cache: cache, windowSize: 2))
                XCTAssertEqual(cache[0].offset, 0); XCTAssertTrue(cache[0].state.isEmpty)
            }
            XCTAssertEqual(cache[0].offset, 0); XCTAssertTrue(cache[0].state.isEmpty)
            _ = try m.prepare(LMInput(tokens: ids), cache: cache, windowSize: 2)
            let retained = cache[0].state.map { $0.asArray(Float.self) }
            for values: [Int8] in [[1,0], [1,2], [1,-1]] {
                XCTAssertThrowsError(try m.prepare(LMInput(text: .init(tokens: ids,
                    mask: MLXArray(values))), cache: cache, windowSize: 2))
                XCTAssertEqual(cache[0].offset, 2)
                XCTAssertEqual(cache[0].state.map { $0.asArray(Float.self) }, retained)
            }
        }
    }
    func testOwnedBaselineRankTwoFatalFixture() throws {
        guard ProcessInfo.processInfo.environment["VMLX_QWEN25VL_BASELINE_FATAL_FIXTURE"] == "1" else {
            throw XCTSkip("Owned subprocess diagnostic only")
        }
        try MLXMetalTestLock.withLock {
            let m = try model(), cache = [KVCacheSimple()]
            print("QWEN25VL_RANK2_FIXTURE_BEGIN tokens=2")
            let output = try logits(m.prepare(LMInput(tokens: MLXArray([Int32(1),2]).reshaped(1,2)),
                cache: cache, windowSize: 2))
            eval(output)
            XCTAssertEqual(output.shape, [1,2,32])
            XCTAssertEqual(cache[0].offset, 2)
            print("QWEN25VL_RANK2_FIXTURE_PASS")
        }
    }

    private final class Progress: @unchecked Sendable {
        let lock = NSLock(); var values: [Int] = []
        func add(_ value: Int) { lock.lock(); defer { lock.unlock() }; values.append(value) }
    }
}
