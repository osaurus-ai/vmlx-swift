import Foundation
import MLX
import MLXNN
import MLXVLM
import XCTest

@testable import MLXLMCommon

final class Glm5NextRoutedActivationTests: XCTestCase {
    private func config(limit: Float?, inputs: Int = 64, hidden: Int = 32, routes: Int = 2) throws
        -> Glm5NextTextConfiguration
    {
        var root =
            try JSONSerialization.jsonObject(
                with: Data(Glm5NextConstructionTests.tinyJSON.utf8)) as! [String: Any]
        var text = root["text_config"] as! [String: Any]
        text["hidden_size"] = inputs
        text["moe_intermediate_size"] = hidden
        text["num_experts_per_tok"] = routes
        if let limit {
            text["swiglu_limit"] = limit
        } else {
            text.removeValue(forKey: "swiglu_limit")
        }
        root["text_config"] = text
        return try JSONDecoder().decode(
            Glm5NextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: root)
        ).textConfig
    }

    private func routedFixture(limit: Float?, dtype: DType) throws -> SwitchGLU {
        let config = try config(limit: limit)
        let routed = Glm5NextMoE(config).switchMLP
        let inputs = config.hiddenSize
        let hidden = config.moeIntermediateSize
        let experts = config.nRoutedExperts
        let gate: [Float] = [25, -25, 1, -1]
        let up: [Float] = [1, 1, 400, -400]
        var gw = [Float](repeating: 0, count: experts * hidden * inputs)
        var uw = gw
        var dw = [Float](repeating: 0, count: experts * inputs * hidden)
        for expert in 0 ..< experts {
            for row in 0 ..< hidden {
                gw[(expert * hidden + row) * inputs] = gate[row % 4]
                uw[(expert * hidden + row) * inputs] = up[row % 4]
                dw[(expert * inputs + row) * hidden + row] = 1
            }
        }
        let gateProjection = SwitchLinear(
            inputDims: inputs, outputDims: hidden, numExperts: experts,
            weight: MLXArray(gw, [experts, hidden, inputs]).asType(dtype))
        let upProjection = SwitchLinear(
            inputDims: inputs, outputDims: hidden, numExperts: experts,
            weight: MLXArray(uw, [experts, hidden, inputs]).asType(dtype))
        let downProjection = SwitchLinear(
            inputDims: hidden, outputDims: inputs, numExperts: experts,
            weight: MLXArray(dw, [experts, inputs, hidden]).asType(dtype))
        try routed.update(modules: ModuleChildren.unflattened([
            ("gate_proj", gateProjection as Module), ("up_proj", upProjection as Module),
            ("down_proj", downProjection as Module),
        ]), verify: .all)
        return routed
    }

    private func check(limit: Float?) throws {
        for dtype in [DType.float32, .float16, .bfloat16] {
            let routed = try routedFixture(limit: limit, dtype: dtype)
            for tokens in [1, 32] {
                var values = [Float](repeating: 0, count: tokens * 64)
                for token in 0 ..< tokens { values[token * 64] = 1 }
                let input = MLXArray(values, [1, tokens, 64]).asType(dtype)
                let indices = MLXArray(
                    (0 ..< (tokens * 2)).map { UInt32(($0 * 3 + 1) % 8) }, [1, tokens, 2])
                let output = routed(input, indices)
                XCTAssertEqual(output.dtype, dtype)
                let actual = output.asType(.float32).asArray(Float.self)
                let gate: [Float] = [25, -25, 1, -1]
                let up: [Float] = [1, 1, 400, -400]
                // Independent scalar activation oracle; use the real down projection
                // only after that boundary so sorted F32 TF32 rounding is not mistaken
                // for an activation-contract error. No runtime precision override.
                let hidden = routed.hiddenDims
                var activated = [Float](repeating: 0, count: tokens * 2 * hidden)
                for route in 0 ..< tokens * 2 {
                    for column in 0 ..< hidden {
                        let g = limit.map { min(gate[column % 4], $0) } ?? gate[column % 4]
                        let u = limit.map { max(-$0, min(up[column % 4], $0)) } ?? up[column % 4]
                        activated[route * hidden + column] = g / (1 + exp(-g)) * u
                    }
                }
                let oracleRows = MLXArray(activated, [tokens * 2, 1, hidden]).asType(dtype)
                let oracleOutput: MLXArray
                if indices.size >= 64 {
                    let order = argSort(indices.flattened())
                    let down = routed.downProj(oracleRows[order], indices.flattened()[order], sortedIndices: true)
                    oracleOutput = scatterUnsort(x: down, invOrder: argSort(order), shape: indices.shape).squeezed(axis: -2)
                } else {
                    oracleOutput = routed.downProj(
                        oracleRows.reshaped([1, tokens, 2, 1, hidden]), indices).squeezed(axis: -2)
                }
                let expectedRows = oracleOutput.asType(.float32).asArray(Float.self)
                for route in 0 ..< (tokens * 2) {
                    for column in 0 ..< 64 {
                        let scalar: Float = column < hidden ? activated[route * hidden + column] : 0
                        let tolerance: Float = dtype == .float32 ? 0.0001 : max(0.002, abs(scalar) * 0.015)
                        XCTAssertEqual(actual[route * 64 + column], expectedRows[route * 64 + column],
                                       accuracy: tolerance, "native down dtype=\(dtype) tokens=\(tokens) column=\(column)")
                        if tokens == 1 || ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0" {
                            XCTAssertEqual(actual[route * 64 + column], scalar, accuracy: tolerance,
                                           "scalar dtype=\(dtype) tokens=\(tokens) column=\(column)")
                        }
                    }
                }
            }
        }
    }

    func testActualMoERoutedDecodeAndSortedPrefillClampBothProjections() throws {
        try MLXMetalTestLock.withLock { try check(limit: 10) }
    }

    func testAbsentLimitRemainsPlainSwiGLU() throws {
        try MLXMetalTestLock.withLock { try check(limit: nil) }
    }
    func testEligibleFusedReducerMatchesModelGlueAtProjectionOutliers() throws {
        try MLXMetalTestLock.withLock {
            let inputs = 4096
            let hidden = 2048
            let experts = 8
            let group = 128
            let moe = Glm5NextMoE(
                try config(limit: 10, inputs: inputs, hidden: hidden, routes: experts))
            let glue = try XCTUnwrap(moe.switchMLP.glue)
            func projection(input: Int, output: Int, bias: Float) -> QuantizedSwitchLinear {
                QuantizedSwitchLinear(
                    inputDims: input, outputDims: output, numExperts: experts,
                    weight: MLXArray.zeros([experts, output, input / 16], dtype: .uint32),
                    scales: MLXArray.zeros([experts, output, input / group], dtype: .bfloat16),
                    biases: MLXArray.full([experts, output, input / group], values: MLXArray(bias))
                        .asType(.bfloat16),
                    groupSize: group, bits: 2, mode: .affine)
            }
            let gate = projection(input: inputs, output: hidden, bias: 25)
            let up = projection(input: inputs, output: hidden, bias: 400)
            let down = projection(input: hidden, output: inputs, bias: 1 / Float(hidden))
            try moe.switchMLP.update(modules: ModuleChildren.unflattened([
                ("gate_proj", gate as Module), ("up_proj", up as Module), ("down_proj", down as Module),
            ]), verify: .all)
            var values = [Float](repeating: 0, count: inputs)
            values[0] = 1
            let input = MLXArray(values, [1, 1, inputs]).asType(.bfloat16)
            let indices = MLXArray((0 ..< experts).map(UInt32.init), [1, 1, experts])
            let scores = MLXArray.full(
                [1, 1, experts], values: MLXArray(Float(1) / Float(experts))
            ).asType(.bfloat16)
            // This is the same eligibility entry point Glm5NextMoE calls, not a
            // direct forced kernel dispatch. Nil must fail the regression.
            let fused = try XCTUnwrap(
                moe.switchMLP.qwen4ExpReduced(input, indices: indices, scores: scores))
            let expanded = expandedDimensions(input, axes: [-2, -3])
            let activated = glue(gate(expanded, indices), up(expanded, indices))
            let routed = down(activated, indices).squeezed(axis: -2)
            let reference = (routed * expandedDimensions(scores, axis: -1)).sum(axis: -2)
            let eager = (moe.switchMLP(input, indices) * expandedDimensions(scores, axis: -1)).sum(
                axis: -2)
            XCTAssertEqual(fused.dtype, .bfloat16)
            XCTAssertEqual(eager.dtype, .bfloat16)
            let actual = fused.asType(.float32).asArray(Float.self)
            let expected = eager.asType(.float32).asArray(Float.self)
            let contract = reference.asType(.float32).asArray(Float.self)
            XCTAssertEqual(actual.count, inputs)
            for index in actual.indices {
                XCTAssertEqual(actual[index], 100, accuracy: 1)
                XCTAssertEqual(actual[index], expected[index], accuracy: 1)
                XCTAssertEqual(expected[index], contract[index], accuracy: 1)
            }
        }
    }

}
