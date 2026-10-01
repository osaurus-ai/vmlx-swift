import Foundation
import MLX
import MLXLMCommon
import XCTest
@testable import MLXLLM

final class NaiveN05AllowedMaskGPUArangeTests: XCTestCase {
    private func configuration() throws -> NaiveN05ArchitectureContract {
        let values: [String: Any] = [
            "model_type": "naive_n05_flash", "hidden_size": 8, "intermediate_size": 12,
            "num_hidden_layers": 2, "vocab_size": 16, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 192, "v_head_dim": 128,
            "swa_num_attention_heads": 2, "swa_num_key_value_heads": 1,
            "swa_head_dim": 192, "swa_v_head_dim": 128, "partial_rotary_factor": 0.5,
            "n_routed_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 6,
            "hybrid_layer_pattern": [0, 1], "moe_layer_freq": [0, 1],
            "index_n_heads": 2, "index_head_dim": 192, "index_top_k": 3,
            "sliding_window": 3, "attention_value_scale": 0.707,
        ]
        return try JSONDecoder().decode(NaiveN05ArchitectureContract.self,
            from: JSONSerialization.data(withJSONObject: values))
    }

    func testPolicyDefaultsOnIsImmutableAndKeepsNumericalCacheIdentity() throws {
        XCTAssertTrue(NaiveN05FlashMath.allowedMaskGPUArangeRequested(environment: [:]))
        for environment in [["VMLX_NAIVE_ALLOWED_MASK_GPU_ARANGE": "0"],
                            ["VMLX_NAIVE_ALLOWED_MASK_GPU_ARANGE": "true"],
                            ["VMLX_NAIVE_ALLOWED_MASK_GPU_ARANGE": "false"],
                            ["VMLX_NAIVE_ALLOWED_MASK_GPU_ARANGE": "yes"],
                            ["VMLX_NAIVE_ALLOWED_MASK_GPU_ARANGE": " 1 "],
                            ["VMLX_NAIVE_ALLOWED_MASK_GPU_ARANGE": ""],
                            ["VMLX_NAIVE_ALLOWED_MASK_GPU_ARANGE": "2"]] {
            XCTAssertFalse(NaiveN05FlashMath.allowedMaskGPUArangeRequested(environment: environment))
        }
        XCTAssertTrue(NaiveN05FlashMath.allowedMaskGPUArangeRequested(
            environment: ["VMLX_NAIVE_ALLOWED_MASK_GPU_ARANGE": "1"]))
        try MLXMetalTestLock.withLock {
            let baseline = try NaiveN05FlashModel(configuration(),
                                                allowedMaskGPUArange: false)
            let candidate = try NaiveN05FlashModel(configuration(),
                                                 allowedMaskGPUArange: true)
            let modelDefault = try NaiveN05FlashModel(configuration())
            let expectedDefault = NaiveN05FlashMath.allowedMaskGPUArangeRequested(
                environment: ProcessInfo.processInfo.environment)
            XCTAssertEqual(modelDefault.allowedMaskGPUArange, expectedDefault)
            XCTAssertTrue(modelDefault.model.layers.allSatisfy {
                $0.attention.allowedMaskGPUArange == expectedDefault
            })
            XCTAssertFalse(baseline.allowedMaskGPUArange)
            XCTAssertTrue(candidate.allowedMaskGPUArange)
            XCTAssertTrue(baseline.model.layers.allSatisfy { !$0.attention.allowedMaskGPUArange })
            XCTAssertTrue(candidate.model.layers.allSatisfy { $0.attention.allowedMaskGPUArange })
            XCTAssertEqual(baseline.cacheStorageDTypeIdentity, "naive-n05-paired-v1")
            XCTAssertEqual(candidate.cacheStorageDTypeIdentity, baseline.cacheStorageDTypeIdentity)
            XCTAssertEqual(modelDefault.cacheStorageDTypeIdentity, baseline.cacheStorageDTypeIdentity)
            XCTAssertFalse(NaiveN05FlashMath.allowedMaskGPUArangeRequested(
                environment: ["VMLX_NAIVE_ALLOWED_MASK_GPU_ARANGE": "0"]))
            XCTAssertTrue(candidate.allowedMaskGPUArange)
        }
    }

    func testPositionRangesAreExactlyInt32IncludingLargeAndSignedOffsets() {
        MLXMetalTestLock.withLock {
            for (start, count) in [(0, 1), (0, 8192), (0, 10290), (9480, 128),
                                   (-129, 257), (16_777_210, 19),
                                   (Int(Int32.max) - 4, 5), (Int(Int32.max), 1),
                                   (Int(Int32.min), 5), (23, 0)] {
                let baseline = NaiveN05FlashMath.maskPositionRange(start: start, count: count,
                                                                   gpuArange: false)
                let candidate = NaiveN05FlashMath.maskPositionRange(start: start, count: count,
                                                                    gpuArange: true)
                XCTAssertEqual(baseline.dtype, .int32)
                XCTAssertEqual(candidate.dtype, .int32)
                XCTAssertEqual(candidate.shape, [count])
                XCTAssertEqual(candidate.asArray(Int32.self), baseline.asArray(Int32.self))
            }
        }
    }

    private func assertMask(queryOffset: Int, length: Int, keyOffset: Int, keyLength: Int,
                            window: Int?, paddingValues: [[Bool]],
                            file: StaticString = #filePath, line: UInt = #line) {
        let total = paddingValues[0].count
        let padding = MLXArray(paddingValues.flatMap { $0 }).reshaped(paddingValues.count, total)
        let originalPadding = padding.asArray(Bool.self)
        let baseline = NaiveN05FlashMath.allowedMask(padding: padding, queryOffset: queryOffset,
            length: length, keyOffset: keyOffset, keyLength: keyLength, window: window)
        let candidate = NaiveN05FlashMath.allowedMask(padding: padding, queryOffset: queryOffset,
            length: length, keyOffset: keyOffset, keyLength: keyLength, window: window,
            gpuArange: true)
        let expected = paddingValues.flatMap { row in
            (0 ..< length).flatMap { query in
                (0 ..< keyLength).map { key in
                    let q = queryOffset + query, k = keyOffset + key
                    return q >= k && (window.map { q - k < $0 } ?? true) && row[k]
                }
            }
        }
        XCTAssertEqual(candidate.dtype, .bool, file: file, line: line)
        XCTAssertEqual(candidate.shape, [paddingValues.count, length, keyLength], file: file, line: line)
        XCTAssertEqual(baseline.asArray(Bool.self), expected, file: file, line: line)
        XCTAssertEqual(candidate.asArray(Bool.self), expected, file: file, line: line)
        XCTAssertEqual(padding.asArray(Bool.self), originalPadding, file: file, line: line)
    }

    func testProduction8k10kOneRowMasksPreservePaddingCausalityAndSlidingOffsets() {
        MLXMetalTestLock.withLock {
            for history in [8192, 9542, 10290] {
                let padding = (0 ..< history).map { $0 != 0 && $0 != 17 && $0 != history - 2 }
                assertMask(queryOffset: history - 1, length: 1, keyOffset: 0,
                           keyLength: history, window: nil, paddingValues: [padding])
                assertMask(queryOffset: history - 1, length: 1, keyOffset: history - 128,
                           keyLength: 128, window: 128, paddingValues: [padding])
                assertMask(queryOffset: history - 1, length: 1, keyOffset: history - 128,
                           keyLength: 128, window: 17, paddingValues: [padding])
            }
        }
    }

    func testMultiqueryBatchedNegativePaddingFutureKeysAndAllMaskedRows() {
        MLXMetalTestLock.withLock {
            let total = 8200
            let mixed = (0 ..< total).map { $0 > 5 && $0 % 7 != 0 }
            let allFalse = [Bool](repeating: false, count: total)
            for window in [nil, 128, 1] as [Int?] {
                assertMask(queryOffset: 8189, length: 8, keyOffset: 0,
                           keyLength: total, window: window, paddingValues: [mixed, allFalse])
                assertMask(queryOffset: 8189, length: 8, keyOffset: 8062,
                           keyLength: 138, window: window, paddingValues: [mixed, allFalse])
            }
        }
    }

    func testSparseSelectionPreservesStableTiesAndMaskedEntriesAtProductionHistory() {
        MLXMetalTestLock.withLock {
            let history = 10290
            let scores = MLXArray((0 ..< history).map { Float(($0 * 13) % 11) }).reshaped(1, 1, history)
            for valid in [(0 ..< history).map { $0 % 19 != 0 },
                          [Bool](repeating: false, count: history)] {
                let padding = MLXArray(valid).reshaped(1, history)
                let baseline = NaiveN05FlashMath.allowedMask(padding: padding, queryOffset: history - 1,
                    length: 1, keyOffset: 0, keyLength: history, window: nil)
                let candidate = NaiveN05FlashMath.allowedMask(padding: padding, queryOffset: history - 1,
                    length: 1, keyOffset: 0, keyLength: history, window: nil, gpuArange: true)
                let expected = NaiveN05FlashMath.sparseMask(scores: scores, allowed: baseline, topK: 2048)
                let actual = NaiveN05FlashMath.sparseMask(scores: scores, allowed: candidate, topK: 2048)
                XCTAssertEqual(actual.asArray(Bool.self), expected.asArray(Bool.self))
                XCTAssertEqual(actual.asArray(Bool.self).filter { $0 }.count, valid.contains(true) ? 2048 : 0)
            }
        }
    }

    func testProductionAttentionAndCompanionCachesRemainBitExactAcrossPrefillAndDecode() throws {
        try MLXMetalTestLock.withLock {
            let config = try configuration()
            for layer in [0, 1] {
                let baseline = NaiveN05FlashAttention(config, layer: layer, allowedMaskGPUArange: false)
                let candidate = NaiveN05FlashAttention(config, layer: layer, allowedMaskGPUArange: true)
                let parameters = baseline.parameters().mapValues { $0.asType(.bfloat16) }
                try baseline.update(parameters: parameters, verify: [.all])
                try candidate.update(parameters: parameters, verify: [.all])
                let baselineCache = NaiveN05FlashCache(window: layer == 1 ? 3 : nil, requiresIndexer: layer == 0)
                let candidateCache = NaiveN05FlashCache(window: layer == 1 ? 3 : nil, requiresIndexer: layer == 0)
                for (offset, tokens) in [(0, 7), (7, 1), (8, 4), (12, 1)] {
                    let states = (MLXArray(0 ..< tokens * 8).asType(.float32) / 32 - 0.25)
                        .reshaped(1, tokens, 8).asType(.bfloat16)
                    let positions = MLXArray(offset ..< offset + tokens).reshaped(1, tokens)
                    let padding = MLXArray((0 ..< offset + tokens).map { $0 != 0 && $0 != 8 })
                        .reshaped(1, offset + tokens)
                    let expected = try baseline(states, positions: positions, padding: padding, cache: baselineCache)
                    let actual = try candidate(states, positions: positions, padding: padding, cache: candidateCache)
                    XCTAssertEqual(actual.asType(.float32).asArray(Float.self), expected.asType(.float32).asArray(Float.self))
                    XCTAssertEqual(candidateCache.offset, baselineCache.offset)
                    XCTAssertEqual(candidateCache.keyOffset, baselineCache.keyOffset)
                    XCTAssertEqual(candidateCache.metaState, baselineCache.metaState)
                    XCTAssertEqual(candidateCache.state.count, baselineCache.state.count)
                    for (a, b) in zip(candidateCache.state, baselineCache.state) {
                        XCTAssertEqual(a.dtype, b.dtype)
                        XCTAssertEqual(a.shape, b.shape)
                        XCTAssertEqual(a.asType(.float32).asArray(Float.self), b.asType(.float32).asArray(Float.self))
                    }
                }
            }
        }
    }
}

#if canImport(Darwin)
import Darwin

/// Optional host-only diagnostic, separate from the six correctness methods.
/// No eval/synchronize/readback occurs inside the timed region. Live heap deltas
/// do not measure cumulative transient allocations; report that limit explicitly.
final class NaiveN05MaskHostConstructionDiagnosticTests: XCTestCase {
    func testHostRangeConstructionTimeAndRetainedAllocationDiagnostic() throws {
        guard ProcessInfo.processInfo.environment["VMLX_NAIVE_MASK_HOST_DIAGNOSTIC"] == "1" else {
            throw XCTSkip("Set VMLX_NAIVE_MASK_HOST_DIAGNOSTIC=1 for host-only allocation/timing diagnostic")
        }
        MLXMetalTestLock.withLock {
            func heap() -> malloc_statistics_t {
                var statistics = malloc_statistics_t()
                malloc_zone_statistics(nil, &statistics)
                return statistics
            }
            for history in [8192, 10290] {
                for enabled in [false, true] {
                    // Warm the constructor and arange kernel outside all timers.
                    NaiveN05FlashMath.maskPositionRange(start: 0, count: history, gpuArange: enabled).eval()
                    let before = heap()
                    let start = DispatchTime.now().uptimeNanoseconds
                    let steps = 20
                    var positions: [MLXArray] = []
                    positions.reserveCapacity(steps * 96)
                    for _ in 0 ..< steps {
                        for layer in 0 ..< 48 {
                            positions.append(NaiveN05FlashMath.maskPositionRange(
                                start: history - 1, count: 1, gpuArange: enabled))
                            positions.append(NaiveN05FlashMath.maskPositionRange(
                                start: layer < 9 ? 0 : history - 128,
                                count: layer < 9 ? history : 128, gpuArange: enabled))
                        }
                    }
                    let elapsed = DispatchTime.now().uptimeNanoseconds - start
                    let after = heap()
                    withExtendedLifetime(positions) {
                        let report: [String: Any] = [
                            "scope": "range-graph-construction-only-no-timed-eval-not-end-to-end",
                            "history": history, "gpu_arange": enabled, "steps": steps,
                            "arrays_retained": positions.count,
                            "milliseconds_per_step": Double(elapsed) / 1_000_000 / Double(steps),
                            "live_malloc_bytes_delta": Int64(after.size_in_use) - Int64(before.size_in_use),
                            "live_malloc_blocks_delta": Int64(after.blocks_in_use) - Int64(before.blocks_in_use),
                            "baseline_transient_host_payload_bytes_per_step_source_estimate":
                                (9 * (history + 1) + 39 * 129) * (MemoryLayout<Int>.stride + MemoryLayout<Int32>.stride),
                            "limit": "live heap delta excludes freed transient arrays, MLX/Metal allocations and driver memory; diagnostic perturbs allocation lifetimes",
                        ]
                        let data = try! JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
                        print("NAIVE_MASK_HOST_CONSTRUCTION " + String(decoding: data, as: UTF8.self))
                    }
                }
            }
        }
    }
}
#endif
