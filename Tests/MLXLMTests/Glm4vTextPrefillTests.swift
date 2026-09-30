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
            let m = try model(), ids = MLXArray([Int32(1),2,3,4])
            let a = [KVCacheSimple()], b = [KVCacheSimple()]
            let x = try logits(m.prepare(LMInput(tokens: ids), cache: a, windowSize: 4))
            let y = try logits(m.prepare(LMInput(tokens: ids.reshaped(1,4)), cache: b, windowSize: 4))
            XCTAssertEqual(x.asArray(Float.self), y.asArray(Float.self))
            XCTAssertEqual(a[0].offset, 4); XCTAssertEqual(b[0].offset, 4)
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
            let next = MLXArray([Int32(3)]).reshaped(1,1)
            XCTAssertLessThan(abs(m(next, cache: a) - m(next, cache: b)).max().item(Float.self), 0.001)
        }
    }
    private final class Progress: @unchecked Sendable {
        let lock = NSLock(); var values: [Int] = []
        func add(_ value: Int) { lock.lock(); defer { lock.unlock() }; values.append(value) }
    }
}
