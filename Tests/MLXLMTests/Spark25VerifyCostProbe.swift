// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Diagnostic: what one target forward of S rows costs on a Spark2.5 bundle,
// the quantity that decides which DFlash block width pays. Measures the
// whole forward and its parts (projections alone, attention alone) at a
// context past the sliding window.
//
//   VMLX_SPARK25_DFLASH_BUNDLE=<bundle> swift test --filter Spark25VerifyCostProbe

import Foundation
import MLX
import MLXNN
@preconcurrency import VMLXTokenizers
import XCTest

@testable import MLXHuggingFace
@testable import MLXLLM
@testable import MLXLMCommon

final class Spark25VerifyCostProbe: XCTestCase {

    func testForwardCostByRows() async throws {
        guard let path = ProcessInfo.processInfo.environment["VMLX_SPARK25_DFLASH_BUNDLE"] else {
            throw XCTSkip("Set VMLX_SPARK25_DFLASH_BUNDLE")
        }
        let context = try await MLXLMCommon.loadModel(
            from: URL(fileURLWithPath: path), using: #huggingFaceTokenizerLoader())
        nonisolated(unsafe) let ctx = context
        let model = try XCTUnwrap(ctx.model as? Spark25Model)
        let contextLength = Int(ProcessInfo.processInfo.environment["VMLX_PROBE_CONTEXT"] ?? "700")!
        let rowsList = [1, 2, 3, 4, 5, 6, 8, 12, 16]
        let repeats = 25

        func median(_ xs: [Double]) -> Double { xs.sorted()[xs.count / 2] }

        let cache = model.newCache()
        let prompt = MLXArray((0 ..< contextLength).map { Int32(1000 + $0 % 5000) }).reshaped(1, -1)
        eval(model(prompt, cache: cache))

        // Whole forward + rollback, as a verify cycle does it.
        var line = "[verify-cost] ctx \(contextLength) forward+trim ms:"
        for rows in rowsList {
            let input = MLXArray((0 ..< rows).map { Int32(2000 + $0) }).reshaped(1, -1)
            var times: [Double] = []
            for i in 0 ... repeats {
                let start = Date.timeIntervalSinceReferenceDate
                let logits = model(input, cache: cache)
                eval(logits)
                let elapsed = Date.timeIntervalSinceReferenceDate - start
                // One row is an in-place ring write, which cannot roll back;
                // the iterator never trims after one. Let the context grow.
                if rows > 1 { XCTAssertTrue(DFlash2TokenIterator.trimVerifiedRows(cache, rows)) }
                if i > 0 { times.append(elapsed * 1000) }
            }
            line += String(format: " S%d=%.2f", rows, median(times))
        }
        print(line)

        // Projections only: every Linear in the stack applied to S rows.
        let layers = model.model.layers
        line = "[verify-cost] linears-only ms:"
        for rows in rowsList {
            let x = MLXRandom.normal([1, rows, 2560]).asType(.bfloat16)
            var times: [Double] = []
            for i in 0 ... repeats {
                let start = Date.timeIntervalSinceReferenceDate
                var outs: [MLXArray] = []
                for layer in layers {
                    outs.append(layer.attention.qkv(x))
                    if let g = layer.attention.gate { outs.append(g(x)) }
                    outs.append(layer.mlp.gate(x))
                    outs.append(layer.mlp.up(x))
                }
                outs.append(model.projectToLogits(x))
                eval(outs)
                let elapsed = Date.timeIntervalSinceReferenceDate - start
                if i > 0 { times.append(elapsed * 1000) }
            }
            line += String(format: " S%d=%.2f", rows, median(times))
        }
        print(line)

        // Down/out projections take the wide input.
        line = "[verify-cost] down+out ms:"
        for rows in rowsList {
            let wide = MLXRandom.normal([1, rows, model.config.intermediateSize]).asType(.bfloat16)
            let q = MLXRandom.normal([1, rows, 4096]).asType(.bfloat16)
            var times: [Double] = []
            for i in 0 ... repeats {
                let start = Date.timeIntervalSinceReferenceDate
                var outs: [MLXArray] = []
                for layer in layers {
                    outs.append(layer.mlp.down(wide))
                    outs.append(layer.attention.output(q))
                }
                eval(outs)
                let elapsed = Date.timeIntervalSinceReferenceDate - start
                if i > 0 { times.append(elapsed * 1000) }
            }
            line += String(format: " S%d=%.2f", rows, median(times))
        }
        print(line)
        print(
            "[verify-cost] shapes qkv \(layers[0].attention.qkv.weight.shape) mlp.gate \(layers[0].mlp.gate.weight.shape) down \(layers[0].mlp.down.weight.shape) types \(type(of: layers[0].mlp.gate))"
        )
    }
}
