import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class Qwen4ExpPrefillErrorBoundaryTests: XCTestCase {
    private final class Progress: @unchecked Sendable {
        let lock = NSLock()
        var completed: [Int] = []
        func append(_ count: Int) { lock.withLock { completed.append(count) } }
    }

    private struct Receipt {
        let completed: [Int]
        let offsets: [Int]
        let headOffsets: [Int]
        let error: String?
    }

    private func exercise(
        errorAfter: Int? = nil, restored: Int = 0, primeHead: Bool = false
    ) throws -> Receipt {
        try MLXMetalTestLock.withLock {
            try Device.withDefaultDevice(.cpu) {
                MLXRandom.seed(47)
                let config = try JSONDecoder().decode(
                    Qwen4ExpConfiguration.self, from: Data(Self.configuration.utf8))
                let model = try Qwen4Exp(config, requesting: [.text])
                let cache = model.newCache(parameters: nil)
                if restored > 0 {
                    try withError {
                        _ = try model.prepare(
                            LMInput(tokens: MLXArray(Array(repeating: Int32(3), count: restored))),
                            cache: cache, windowSize: 4)
                        MLX.eval(cache)
                    }
                }
                let head = primeHead ? model.makeNativeMTPCache() : []
                let primer = primeHead ? NativeMTPPromptPrimer(model: model, cache: head) : nil
                let progress = Progress()
                var caught: String?
                do {
                    try withError {
                        try NativeMTPPromptPrimer.$current.withValue(primer) {
                            try PrefillProgressReporter.withHandler({ count in
                                progress.append(count)
                                if count == errorAfter {
                                    // A real mlx-c error exercises the same capture/check boundary
                                    // without poisoning Metal or requiring a GPU watchdog failure.
                                    _ = MLXArray.zeros([2, 3]) + MLXArray.zeros([4, 5])
                                }
                            }, operation: {
                                _ = try model.prepare(
                                    LMInput(tokens: MLXArray(Array(Int32(1)...Int32(9)))),
                                    cache: cache, windowSize: 2)
                                MLX.eval(cache)
                            })
                        }
                    }
                } catch { caught = String(describing: error) }
                return Receipt(completed: progress.completed, offsets: cache.map(\.offset),
                               headOffsets: head.map(\.offset), error: caught)
            }
        }
    }

    func testSuccessfulChunksAndTailRemainComplete() throws {
        let result = try exercise()
        XCTAssertNil(result.error)
        XCTAssertEqual(result.completed, [2, 4, 6, 8])
        XCTAssertEqual(result.offsets, [9, 9])
    }

    func testCapturedErrorStopsBeforeAnotherChunk() throws {
        let result = try exercise(errorAfter: 2)
        XCTAssertTrue(result.error?.contains("[broadcast_shapes]") == true)
        XCTAssertEqual(result.completed, [2])
        XCTAssertEqual(result.offsets, [2, 2])
    }

    func testCapturedErrorStopsBeforeTheUnevaluatedTail() throws {
        let result = try exercise(errorAfter: 8)
        XCTAssertTrue(result.error?.contains("[broadcast_shapes]") == true)
        XCTAssertEqual(result.completed, [2, 4, 6, 8])
        XCTAssertEqual(result.offsets, [8, 8])
    }

    func testRestoredPrefixDoesNotAdvancePastTheFailedSuffix() throws {
        let result = try exercise(errorAfter: 2, restored: 17)
        XCTAssertTrue(result.error?.contains("[broadcast_shapes]") == true)
        XCTAssertEqual(result.completed, [2])
        XCTAssertEqual(result.offsets, [19, 19])
    }

    func testNativeHeadStopsWithTheCapturedError() throws {
        let result = try exercise(errorAfter: 2, primeHead: true)
        XCTAssertTrue(result.error?.contains("[broadcast_shapes]") == true)
        XCTAssertEqual(result.completed, [2])
        XCTAssertEqual(result.offsets, [2, 2])
        XCTAssertEqual(result.headOffsets, [1])
    }

    func testSuccessfulNativeHeadStillPrimesEveryPromptPair() throws {
        let result = try exercise(primeHead: true)
        XCTAssertNil(result.error)
        XCTAssertEqual(result.completed, [2, 4, 6, 8])
        XCTAssertEqual(result.offsets, [9, 9])
        XCTAssertEqual(result.headOffsets, [8])
    }

    private static let configuration = #"""
    {"model_type":"qwen4_exp","text_config":{
      "model_type":"qwen4_exp_text","dtype":"float32","mtp_num_hidden_layers":1,
      "hidden_size":64,"num_hidden_layers":2,"intermediate_size":64,
      "num_attention_heads":4,"num_key_value_heads":1,"head_dim":16,
      "linear_num_value_heads":4,"linear_num_key_heads":1,"linear_key_head_dim":16,
      "linear_value_head_dim":16,"linear_conv_kernel_dim":4,"vocab_size":128,
      "num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":32,
      "shared_expert_intermediate_size":32,"layer_types":["full_attention","full_attention"],
      "full_attention_interval":2,"hc_count":4,"hc_lowrank":8,"ple_layer_ids":[],
      "indexer_n_heads":2,"indexer_kv_heads":1,"indexer_head_dim":8,
      "indexer_budget":32,"indexer_compress_ratio":4,"mrope_section":[1,1,0]}}
    """#
}
