// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Testing

@testable import MLXVLM

@Suite("Qwen dense router preserves loaded weight precision", .serialized)
struct Qwen4ExpDenseRouterTests {
    private func sample(_ shape: [Int], seed: UInt64, magnitude: Float, dtype: DType) -> MLXArray {
        (MLXRandom.uniform(low: -1, high: 1, shape, key: MLXRandom.key(seed)) * magnitude)
            .asType(dtype)
    }

    private func reference(_ x: MLXArray, _ weight: MLXArray, k: Int, normalize: Bool)
        -> (MLXArray, MLXArray)
    {
        // Match SparseMoeBlock's ordinary graph, including precise softmax
        // and normalization after selection. No sampler or quantization edits.
        let gates = softmax(matmul(x, weight.transposed()), axis: -1, precise: true)
        let kth = weight.dim(0) - k
        let indices = argPartition(gates, kth: kth, axis: -1)[.ellipsis, kth...]
        let scores = takeAlong(gates, indices, axis: -1)
        return (indices, normalize ? scores / scores.sum(axis: -1, keepDims: true) : scores)
    }

    private func bits(_ x: MLXArray) -> [UInt32] {
        x.asType(.float32).asArray(Float.self).map(\.bitPattern)
    }

    @Test("F32 loaded router and BF16 router retain exact IDs, scores, and source tensors")
    func loadedDTypeAndQueuedWeights() throws {
        try MLXMetalTestLock.withLock {
            var cases = 0
            for dtype in [DType.float32, .bfloat16] {
                for magnitude: Float in [0.001, 0.1, 1, 16] {
                    for normalize in [false, true] {
                        for k in [1, 10, 512] {
                            var pending: [(MLXArray, MLXArray, MLXArray, MLXArray)] = []
                            var sources: [([MLXArray], [[UInt32]])] = []
                            // Multiple layers share one compiled region. Queue
                            // fresh weights before evaluation; sequential eval
                            // cannot detect stale captured-weight reuse.
                            for layer in 0 ..< 3 {
                                let x = sample(
                                    [1, 1, 2560], seed: UInt64(800 + layer),
                                    magnitude: magnitude, dtype: .bfloat16)
                                let w = sample(
                                    [512, 2560], seed: UInt64(900 + layer),
                                    magnitude: 0.1, dtype: dtype)
                                let expected = reference(x, w, k: k, normalize: normalize)
                                let actual = try #require(
                                    Qwen4ExpCompiledMoE.denseRouter(
                                        x, weight: w, topK: k, normTopK: normalize))
                                sources.append(([x, w], [bits(x), bits(w)]))
                                pending.append(
                                    (expected.0, expected.1, actual.indices, actual.scores))
                            }
                            MLX.eval(pending.flatMap { [$0.0, $0.1, $0.2, $0.3] })
                            for (expectedIDs, expectedScores, ids, scores) in pending {
                                #expect(ids.dtype == expectedIDs.dtype)
                                #expect(ids.shape == expectedIDs.shape)
                                #expect(ids.asArray(Int32.self) == expectedIDs.asArray(Int32.self))
                                #expect(scores.dtype == expectedScores.dtype)
                                #expect(scores.shape == expectedScores.shape)
                                #expect(bits(scores) == bits(expectedScores))
                                #expect(isFinite(scores).all().item(Bool.self))
                                cases += 1
                            }
                            for (arrays, originals) in sources {
                                #expect(arrays.map(bits) == originals)
                            }
                        }
                    }
                }
            }
            #expect(cases == 144)
            print("[DenseRouter] exact_cases=\(cases) generation_tps=NA reason=no_generation")
        }
    }

    @Test("ties, strided inputs, batch shape changes, and non-square dimensions remain exact")
    func layoutsAndTies() throws {
        try MLXMetalTestLock.withLock {
            for batch in [1, 2] {
                for width in [64, 192, 2560] {
                    let x = sample(
                        [batch, 1, width * 2], seed: 311,
                        magnitude: 1, dtype: .bfloat16)[.ellipsis, .stride(by: 2)]
                    for zero in [false, true] {
                        let w =
                            zero
                            ? MLXArray.zeros([64, width], dtype: .float32)
                            : sample([64, width * 2], seed: 512, magnitude: 1, dtype: .float32)[
                                .ellipsis, .stride(by: 2)]
                        let expected = reference(x, w, k: 10, normalize: true)
                        let actual = try #require(
                            Qwen4ExpCompiledMoE.denseRouter(
                                x, weight: w, topK: 10, normTopK: true))
                        MLX.eval(expected.0, expected.1, actual.indices, actual.scores)
                        #expect(
                            actual.indices.asArray(Int32.self) == expected.0.asArray(Int32.self))
                        #expect(bits(actual.scores) == bits(expected.1))
                    }
                }
            }
        }
    }

    /// Verification preserves the actual singleton-shaped AR operation graph,
    /// including score bits near BF16 rounding boundaries and live weights.
    @Test("verify-window router rows equal the 1-row router bit for bit")
    func verifyRowsMatchOneRowRouter() throws {
        try MLXMetalTestLock.withLock {
            var cases = 0
            for rows in [2, 3, 4, 6, 7, 8, 9, 13] {
                for seed in UInt64(0) ..< 4 {
                    for normalize in [false, true] {
                        for layout in 0 ..< 3 {
                            let window: MLXArray
                            if layout == 1 {
                                window =
                                    sample(
                                        [1, rows * 2, 2560], seed: 1_000 + seed,
                                        magnitude: 4, dtype: .bfloat16)[0..., .stride(by: 2), 0...]
                            } else if layout == 2 {
                                window =
                                    sample(
                                        [1, rows, 5120], seed: 1_000 + seed,
                                        magnitude: 4, dtype: .bfloat16)[0..., 0..., .stride(by: 2)]
                            } else {
                                window = sample(
                                    [1, rows, 2560], seed: 1_000 + seed,
                                    magnitude: 4, dtype: .bfloat16)
                            }
                            let weight = sample(
                                [512, 2560], seed: 2_000 + seed,
                                magnitude: 0.05, dtype: .bfloat16)
                            let originals = [bits(window), bits(weight)]
                            let batched = try #require(
                                Qwen4ExpCompiledMoE.denseRouterRows(
                                    window, weight: weight, topK: 10, normTopK: normalize))
                            var rowIDs: [MLXArray] = []
                            var rowScores: [MLXArray] = []
                            for row in 0 ..< rows {
                                let one = try #require(
                                    Qwen4ExpCompiledMoE.denseRouter(
                                        window[0..., row ..< (row + 1)], weight: weight,
                                        topK: 10, normTopK: normalize))
                                rowIDs.append(one.indices)
                                rowScores.append(one.scores)
                            }
                            let expectedIDs = concatenated(rowIDs, axis: 1)
                            let expectedScores = concatenated(rowScores, axis: 1)
                            MLX.eval(batched.indices, batched.scores, expectedIDs, expectedScores)
                            #expect(batched.indices.shape == [1, rows, 10])
                            #expect(
                                batched.indices.asArray(Int32.self)
                                    == expectedIDs.asArray(Int32.self))
                            #expect(
                                bits(batched.scores) == bits(expectedScores),
                                "rows=\(rows), seed=\(seed), normalized=\(normalize), layout=\(layout)"
                            )
                            #expect([bits(window), bits(weight)] == originals)
                            cases += 1
                        }
                    }
                }
                let f32Weight = sample([512, 2560], seed: 42, magnitude: 0.05, dtype: .float32)
                #expect(
                    Qwen4ExpCompiledMoE.denseRouterRows(
                        .zeros([1, rows, 2560], dtype: .bfloat16), weight: f32Weight,
                        topK: 10, normTopK: true) == nil)
            }
            #expect(cases == 192)
            print("[DenseRouterRows] exact_cases=\(cases) generation_tps=NA reason=no_generation")
        }
    }

    @Test(
        "Captured router diagnostic compares actual batched math against each AR row",
        .enabled(if: ProcessInfo.processInfo.environment["VMLX_ROUTER_CAPTURE_FIXTURE"] != nil))
    func capturedRouterDiagnostic() throws {
        guard let path = ProcessInfo.processInfo.environment["VMLX_ROUTER_CAPTURE_FIXTURE"] else {
            Issue.record("Captured diagnostic enabled without a fixture")
            return
        }
        try MLXMetalTestLock.withLock {
            let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: path))
            let window = try #require(arrays["verify_input"])
            let ar = try #require(arrays["ar_input"])
            let weight = try #require(arrays["weight"])
            let rows = window.dim(1)
            let hidden = window.dim(2)
            #expect(rows == 7 && hidden == 2560)
            #expect(bits(window[0..., 2 ..< 3]) == bits(ar))
            let view = asStrided(window, [rows, 1, hidden], strides: [hidden, 1, 1])
            let batched = matmul(view, weight.transposed())
            let oneRows = (0 ..< rows).map {
                matmul(window[0..., $0 ..< ($0 + 1)], weight.transposed())
            }
            let one = concatenated(oneRows, axis: 0)
            let batchSoftmax = softmax(batched, axis: -1, precise: true)
            let oneSoftmax = concatenated(
                oneRows.map { softmax($0, axis: -1, precise: true) }, axis: 0)
            MLX.eval(batched, one, batchSoftmax, oneSoftmax)
            print(
                "[ROUTER-CAPTURE] input_dtype=\(window.dtype) weight_dtype=\(weight.dtype) matmul_equal=\(bits(batched) == bits(one)) softmax_equal=\(bits(batchSoftmax) == bits(oneSoftmax))"
            )
            for normalize in [false, true] {
                let actual = try #require(
                    Qwen4ExpCompiledMoE.denseRouterRows(
                        window, weight: weight, topK: 10, normTopK: normalize))
                let references = try (0 ..< rows).map { row in
                    try #require(
                        Qwen4ExpCompiledMoE.denseRouter(
                            window[0..., row ..< (row + 1)], weight: weight, topK: 10,
                            normTopK: normalize))
                }
                let ids = concatenated(references.map(\.indices), axis: 1)
                let scores = concatenated(references.map(\.scores), axis: 1)
                MLX.eval(actual.indices, actual.scores, ids, scores)
                print(
                    "[ROUTER-CAPTURE] normalized=\(normalize) ids_equal=\(bits(actual.indices) == bits(ids)) scores_equal=\(bits(actual.scores) == bits(scores))"
                )
                #expect(bits(actual.indices) == bits(ids))
                #expect(bits(actual.scores) == bits(scores))
            }
        }
    }

    @Test(
        "Opt-in router timing retains alternating correct-reference controls",
        .enabled(if: ProcessInfo.processInfo.environment["VMLX_ROUTER_TIMING"] == "1"))
    func alternatingRouterTiming() throws {
        try MLXMetalTestLock.withLock {
            let path = try #require(
                ProcessInfo.processInfo.environment["VMLX_ROUTER_CAPTURE_FIXTURE"])
            let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: path))
            let captured = try #require(arrays["verify_input"])
            let weight = try #require(arrays["weight"])
            for rows in [2, 4, 6, 8] {
                let window = concatenated([captured, captured], axis: 1)[0..., 0 ..< rows]
                func run(_ candidate: Bool) throws -> (indices: MLXArray, scores: MLXArray) {
                    if candidate {
                        return try #require(
                            Qwen4ExpCompiledMoE.denseRouterRows(
                                window, weight: weight, topK: 10, normTopK: true))
                    }
                    let outputs = try (0 ..< rows).map { row in
                        try #require(
                            Qwen4ExpCompiledMoE.denseRouter(
                                window[0..., row ..< (row + 1)], weight: weight,
                                topK: 10, normTopK: true))
                    }
                    return (
                        concatenated(outputs.map(\.indices), axis: 1),
                        concatenated(outputs.map(\.scores), axis: 1)
                    )
                }
                let reference = try run(false)
                let candidate = try run(true)
                MLX.eval(reference.indices, reference.scores, candidate.indices, candidate.scores)
                #expect(reference.scores.size > 0)
                #expect(bits(reference.indices) == bits(candidate.indices))
                #expect(bits(reference.scores) == bits(candidate.scores))
                for probe in 0 ..< 8 {
                    for candidate in (probe.isMultiple(of: 2) ? [false, true] : [true, false]) {
                        let start = DispatchTime.now().uptimeNanoseconds
                        for _ in 0 ..< 200 {
                            let output = try run(candidate)
                            MLX.eval(output.indices, output.scores)
                        }
                        let nanos = DispatchTime.now().uptimeNanoseconds - start
                        print(
                            "[ROUTER-TIMING] rows=\(rows) probe=\(probe) candidate=\(candidate) iterations=200 ns=\(nanos) discard=\(probe == 0) generation_tps=NA"
                        )
                    }
                }
            }
        }
    }

    @Test("prefill, verification, outer traces, and malformed signatures use the ordinary path")
    func eligibility() throws {
        try MLXMetalTestLock.withLock {
            let weight = MLXArray.zeros([16, 64], dtype: .float32)
            for shape in [[64], [1, 64], [1, 2, 64], [1, 3, 64], [1, 4, 64], [1, 1, 32]] {
                #expect(
                    Qwen4ExpCompiledMoE.denseRouter(
                        .zeros(shape, dtype: .bfloat16), weight: weight, topK: 4, normTopK: true)
                        == nil)
            }
            let x = MLXArray.zeros([1, 1, 64], dtype: .bfloat16)
            for k in [0, -1, 17] {
                #expect(
                    Qwen4ExpCompiledMoE.denseRouter(x, weight: weight, topK: k, normTopK: true)
                        == nil)
            }
            #expect(
                Qwen4ExpCompiledMoE.denseRouter(
                    x.asType(.float32), weight: weight, topK: 4, normTopK: true) == nil)
            #expect(
                Qwen4ExpCompiledMoE.denseRouter(
                    x, weight: weight.asType(.float16), topK: 4, normTopK: true) == nil)
            #expect(
                Qwen4ExpCompiledMoE.denseRouter(
                    x, weight: weight.reshaped(-1), topK: 4, normTopK: true) == nil)
            CompiledDecodeTrace.withActive {
                #expect(
                    Qwen4ExpCompiledMoE.denseRouter(x, weight: weight, topK: 4, normTopK: true)
                        == nil)
            }
        }
    }
}
