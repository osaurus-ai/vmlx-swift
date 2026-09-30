import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class Glm4vTextPrefillTests: XCTestCase {
    private func model() throws -> Glm4v {
        let json = #"""
            {"model_type":"glm4v","vocab_size":32,"hidden_size":16,"tie_word_embeddings":true,
             "text_config":{"hidden_size":16,"num_hidden_layers":1,"intermediate_size":32,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"vocab_size":32,"rope_parameters":{"mrope_section":[1,1,2],"partial_rotary_factor":1.0,"rope_theta":10000}},
             "vision_config":{"depth":1,"hidden_size":16,"intermediate_size":32,"num_heads":2,"patch_size":2,"out_hidden_size":16,"spatial_merge_size":2,"temporal_patch_size":1}}
            """#
        return Glm4v(try JSONDecoder().decode(Glm4vConfiguration.self, from: Data(json.utf8)))
    }
    private func logits(_ result: PrepareResult) throws -> MLXArray {
        guard case .logits(let output) = result else { throw NSError(domain: "fixture", code: 1) }
        return output.logits
    }
    func testSingleBatchAndFlatTextAgree() throws {
        try MLXMetalTestLock.withLock {
            let m = try model()
            let ids = MLXArray([Int32(1), 2, 3, 4])
            let a = [KVCacheSimple()]
            let b = [KVCacheSimple()]
            let x = try logits(m.prepare(LMInput(tokens: ids), cache: a, windowSize: 4))
            let y = try logits(
                m.prepare(LMInput(tokens: ids.reshaped(1, 4)), cache: b, windowSize: 4))
            XCTAssertEqual(x.asArray(Float.self), y.asArray(Float.self))
            XCTAssertEqual(a[0].offset, 4)
            XCTAssertEqual(b[0].offset, 4)
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
            for ids in [
                MLXArray(Int32(1)), MLXArray([Int32(1), 2]).reshaped(2, 1),
                MLXArray([Int32(1), 2]).reshaped(1, 1, 2),
            ] {
                let cache = [KVCacheSimple()]
                XCTAssertThrowsError(
                    try m.prepare(LMInput(tokens: ids), cache: cache, windowSize: 2))
                XCTAssertEqual(cache[0].offset, 0)
            }
        }
    }
    func testLongFlatAndBatchedTextKeepExactMultiTurnCacheParity() throws {
        try MLXMetalTestLock.withLock {
            let m = try model()
            let ids = MLXArray((0 ..< 33).map { Int32($0 % 30 + 1) })
            let a = [KVCacheSimple()]
            let b = [KVCacheSimple()]
            for turn in [ids, MLXArray([Int32(3), 4, 5]), MLXArray([Int32(6)])] {
                let flat = try logits(m.prepare(LMInput(tokens: turn), cache: a, windowSize: 8))
                let batched = try logits(
                    m.prepare(
                        LMInput(tokens: turn.reshaped(1, -1)),
                        cache: b, windowSize: 8))
                XCTAssertEqual(flat.shape, batched.shape)
                XCTAssertEqual(flat.asArray(Float.self), batched.asArray(Float.self))
                XCTAssertEqual(a[0].offset, b[0].offset)
                XCTAssertEqual(a[0].state.count, b[0].state.count)
                for (x, y) in zip(a[0].state, b[0].state) {
                    XCTAssertEqual(x.shape, y.shape)
                    XCTAssertEqual(x.asArray(Float.self), y.asArray(Float.self))
                }
            }
            XCTAssertEqual(a[0].offset, 37)
            XCTAssertEqual(b[0].offset, 37)
        }
    }
    func testUncachedPrepareKeepsFullContextAndAllTrueMask() throws {
        try MLXMetalTestLock.withLock {
            let m = try model()
            let ids = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)
            let input = LMInput(
                text: .init(tokens: ids, mask: MLXArray.ones([1, 4], dtype: .bool)))
            let actual = try logits(m.prepare(input, cache: [], windowSize: 1))
            let expected = m(ids, cache: nil)
            XCTAssertEqual(actual.shape, [1, 4, 32])
            XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
        }
    }
    func testInvalidStepEmptyTokensAndPaddingRejectBeforeMutation() throws {
        try MLXMetalTestLock.withLock {
            let m = try model()
            let ids = MLXArray([Int32(1), 2])
            let cache = [KVCacheSimple()]
            for step in [0, -1] {
                XCTAssertThrowsError(
                    try m.prepare(LMInput(tokens: ids), cache: cache, windowSize: step))
            }
            XCTAssertThrowsError(
                try m.prepare(LMInput(tokens: MLXArray([Int32]())), cache: cache, windowSize: 2))
            let masked = LMInput(text: .init(tokens: ids, mask: MLXArray([true, false])))
            XCTAssertThrowsError(try m.prepare(masked, cache: cache, windowSize: 2))
            XCTAssertEqual(cache[0].offset, 0)
            XCTAssertTrue(cache[0].state.isEmpty)
        }
    }
    func testProcessorIntegralMasksAcceptOnlyExactOnes() throws {
        try MLXMetalTestLock.withLock {
            let m = try model()
            let ids = MLXArray([Int32(1), 2]).reshaped(1, 2)
            for dtype in [DType.int8, .uint8, .int16, .uint16, .int32, .uint32, .int64, .uint64] {
                let cache = [KVCacheSimple()]
                let accepted = LMInput(tokens: ids, mask: MLXArray.ones([1, 2], dtype: dtype))
                _ = try m.prepare(accepted, cache: cache, windowSize: 2)
                XCTAssertEqual(cache[0].offset, 2)
                let snapshot = cache[0].state.map { $0.asArray(Float.self) }
                for values in [[0, 0], [1, 0], [1, 2]] {
                    let mask = MLXArray(values.map(Int32.init)).reshaped(1, 2).asType(dtype)
                    XCTAssertThrowsError(
                        try m.prepare(LMInput(tokens: ids, mask: mask), cache: cache, windowSize: 2)
                    )
                    XCTAssertEqual(cache[0].offset, 2)
                    XCTAssertEqual(cache[0].state.map { $0.asArray(Float.self) }, snapshot)
                }
            }
        }
    }
    func testIteratorStableBoundariesKeepExactFlatAndBatchParity() throws {
        try MLXMetalTestLock.withLock {
            let m = try model()
            let ids = MLXArray([Int32(1), 2, 3, 4, 5, 6, 7, 8])
            let parameters = GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: 3)
            let plainCache = [KVCacheSimple()]
            let splitCache = [KVCacheSimple()]
            var plain = try TokenIterator(
                input: LMInput(
                    tokens: ids,
                    cachePrefixTokenCounts: [4, 6], cacheStablePrefixTokenCounts: [4, 6]), model: m,
                cache: plainCache, parameters: parameters)
            // The production iterator slices stable boundaries into [1,T]
            // inputs before calling this model's prepare implementation.
            var split = try TokenIterator(
                input: LMInput(
                    tokens: ids.reshaped(1, -1),
                    cachePrefixTokenCounts: [4, 6], cacheStablePrefixTokenCounts: [4, 6]),
                model: m, cache: splitCache, parameters: parameters)
            XCTAssertEqual(plainCache[0].offset, splitCache[0].offset)
            XCTAssertGreaterThanOrEqual(splitCache[0].offset, 8)
            XCTAssertEqual(plainCache[0].state.count, splitCache[0].state.count)
            for (a, b) in zip(plainCache[0].state, splitCache[0].state) {
                XCTAssertEqual(a.shape, b.shape)
                XCTAssertEqual(a.asArray(Float.self), b.asArray(Float.self))
            }
            let started = Date()
            let expected = plain.next()
            let actual = split.next()
            XCTAssertNotNil(expected)
            XCTAssertEqual(actual, expected)
            XCTAssertNil(plain.next())
            XCTAssertNil(split.next())
            print(
                "VLM_ITERATOR_FIXTURE tokens=2 tokens_per_second=\(2 / max(Date().timeIntervalSince(started), 0.000001))"
            )
        }
    }

}
