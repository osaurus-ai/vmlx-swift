import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class LLMPrefillErrorBoundaryTests: XCTestCase {
    private final class Progress: @unchecked Sendable {
        let lock = NSLock()
        var completed: [Int] = []
        func append(_ count: Int) { lock.withLock { completed.append(count) } }
    }

    private struct Receipt {
        let completed: [Int]
        let offsets: [Int]
        let tail: [Int32]?
        let mask: [Bool]?
        let error: String?
    }

    private func makeModel() throws -> FlintModel {
        MLXRandom.seed(47)
        let config = try JSONDecoder().decode(
            FlintConfiguration.self, from: Data(Self.configuration.utf8))
        let model = try FlintModel(config)
        // Small untrained fixture, not a checkpoint/dtype qualification.
        let weights = model.parameters().flattened().map { ($0.0, $0.1.asType(.float32)) }
        model.update(parameters: .unflattened(weights))
        return model
    }

    private func exercise(
        errorAfter: Int? = nil, restored: Int = 0,
        rankTwo: Bool = false, masked: Bool = false
    ) throws -> Receipt {
        try MLXMetalTestLock.withLock {
            try Device.withDefaultDevice(.cpu) {
                let model = try makeModel()
                let cache = model.newCache(parameters: nil)
                if restored > 0 {
                    try withError {
                        let prefix = try model.prepare(
                            LMInput(tokens: MLXArray(Array(repeating: Int32(3), count: restored))),
                            cache: cache, windowSize: 4)
                        if case .tokens(let tail) = prefix {
                            _ = model(.init(tokens: tail.tokens[.newAxis]), cache: cache, state: nil)
                        }
                        MLX.eval(cache)
                    }
                }
                let progress = Progress()
                let ids = MLXArray(Array(Int32(1) ... Int32(9)))
                let mask = masked ? MLXArray(Array(repeating: true, count: 8) + [false]) : nil
                let input = LMInput(text: .init(
                    tokens: rankTwo ? ids.reshaped(1, 9) : ids,
                    mask: rankTwo ? mask?.reshaped(1, 9) : mask))
                var remaining: [Int32]?
                var remainingMask: [Bool]?
                var caught: String?
                do {
                    try withError {
                        try PrefillProgressReporter.withHandler(
                            { count in
                                progress.append(count)
                                if count == errorAfter {
                                    _ = MLXArray.zeros([2, 3]) + MLXArray.zeros([4, 5])
                                }
                            },
                            operation: {
                                let result = try model.prepare(input, cache: cache, windowSize: 2)
                                if case .tokens(let tail) = result {
                                    remaining = tail.tokens.asArray(Int32.self)
                                    remainingMask = tail.mask?.asArray(Bool.self)
                                }
                                MLX.eval(cache)
                            })
                    }
                } catch { caught = String(describing: error) }
                return Receipt(
                    completed: progress.completed, offsets: cache.map(\.offset),
                    tail: remaining, mask: remainingMask, error: caught)
            }
        }
    }

    func testSuccessfulFlatInputReturnsOnlyTheUnconsumedTail() throws {
        let result = try exercise()
        XCTAssertNil(result.error)
        XCTAssertEqual(result.completed, [2, 4, 6, 8])
        XCTAssertEqual(result.offsets, [8, 8, 8])
        XCTAssertEqual(result.tail, [9])
    }

    func testSuccessfulRankTwoInputRetainsTailAndMask() throws {
        let result = try exercise(rankTwo: true, masked: true)
        XCTAssertNil(result.error)
        XCTAssertEqual(result.completed, [2, 4, 6, 8])
        XCTAssertEqual(result.offsets, [8, 8, 8])
        XCTAssertEqual(result.tail, [9])
        XCTAssertEqual(result.mask, [false])
    }

    func testCapturedErrorStopsBeforeAnotherChunk() throws {
        let result = try exercise(errorAfter: 2)
        XCTAssertTrue(result.error?.contains("[broadcast_shapes]") == true)
        XCTAssertEqual(result.completed, [2])
        XCTAssertEqual(result.offsets, [2, 2, 2])
        XCTAssertNil(result.tail)
    }

    func testCapturedErrorDoesNotReturnATailForFurtherEvaluation() throws {
        let result = try exercise(errorAfter: 8)
        XCTAssertTrue(result.error?.contains("[broadcast_shapes]") == true)
        XCTAssertEqual(result.offsets, [8, 8, 8])
        XCTAssertNil(result.tail)
    }

    func testRestoredPrefixStopsAtTheFirstFailedSuffix() throws {
        let result = try exercise(errorAfter: 2, restored: 17, rankTwo: true)
        XCTAssertTrue(result.error?.contains("[broadcast_shapes]") == true)
        XCTAssertEqual(result.completed, [2])
        XCTAssertEqual(result.offsets, [19, 19, 19])
        XCTAssertNil(result.tail)
    }

    private static let configuration = #"""
        {"model_type":"flint","hidden_size":32,"num_hidden_layers":3,
        "intermediate_size":64,"moe_intermediate_size":32,"shared_expert_intermediate_size":32,
        "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"num_experts":4,
        "num_experts_per_tok":2,"vocab_size":64,"rms_norm_eps":0.000001,"rope_theta":10000,
        "sliding_window":5,"layer_types":["full_attention","sliding_attention","sliding_attention"],
        "no_rope_layer_indices":[0,2],"mlp_only_layers":[0],"norm_topk_prob":true,
        "router_score_function":"sigmoid","softmax_type":"learnable","tie_word_embeddings":false}
        """#
}
