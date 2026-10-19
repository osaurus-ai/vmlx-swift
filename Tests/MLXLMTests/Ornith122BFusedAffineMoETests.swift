import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

final class Ornith122BFusedAffineMoETests: XCTestCase {
    private let hidden = 3072
    private let intermediate = 1024
    private let experts = 16
    private let routes = 8

    private func projection(_ input: Int, _ output: Int, seed: UInt64)
        -> (QuantizedSwitchLinear, MLXArray)
    {
        let source = MLXRandom.uniform(
            low: -0.125, high: 0.125, [experts, output, input],
            key: MLXRandom.key(seed)
        ).asType(.bfloat16)
        let (weight, scales, biases) = MLX.quantized(
            source, groupSize: 64, bits: 4, mode: .affine)
        let layer = QuantizedSwitchLinear(
            inputDims: input, outputDims: output, numExperts: experts,
            weight: weight, scales: scales, biases: biases,
            groupSize: 64, bits: 4, mode: .affine)
        let exact = dequantized(
            weight, scales: scales, biases: biases,
            groupSize: 64, bits: 4, mode: .affine, dtype: .float32)
        return (layer, exact)
    }

    func testQ4BF16ShapeAgainstIndependentMathAndSerialRows() throws {
        let (gate, g) = projection(hidden, intermediate, seed: 122001)
        let (up, u) = projection(hidden, intermediate, seed: 122002)
        let (down, d) = projection(intermediate, hidden, seed: 122003)
        let reducer = try XCTUnwrap(
            Qwen4ExpFusedAffineMoE.makeReducer(
                gate: gate, up: up, down: down))
        // The extension does not qualify F16 metadata at this new geometry.
        let halfGate = QuantizedSwitchLinear(
            inputDims: hidden, outputDims: intermediate, numExperts: experts,
            weight: gate.weight, scales: gate.scales.asType(.float16),
            biases: gate.biases?.asType(.float16), groupSize: 64, bits: 4, mode: .affine)
        XCTAssertNil(Qwen4ExpFusedAffineMoE.makeReducer(gate: halfGate, up: up, down: down))
        let backing = MLXRandom.uniform(
            low: -1, high: 1, [1, 8, hidden],
            key: MLXRandom.key(122004)
        ).asType(.bfloat16)
        let input = backing[0..., .stride(by: 2), 0...]
        let indexValues = (0 ..< (4 * routes * 2)).map {
            UInt32(($0 * 3 + $0 / (routes * 2)) % experts)
        }
        let indices = MLXArray(indexValues, [1, 4, routes * 2])[0..., 0..., .stride(by: 2)]
        let scoreBacking = MLXRandom.uniform(
            low: 0, high: 1, [1, 4, routes * 2], key: MLXRandom.key(122005))
        let scoreRows = softmax(scoreBacking[0..., 0..., .stride(by: 2)], axis: -1)
        for scoreType in [DType.bfloat16, .float32] {
            let scores = scoreRows.asType(scoreType)
            var serial: [MLXArray] = []
            for row in 0 ..< 4 {
                let x = input[0..., row ..< (row + 1), 0...]
                let i = indices[0..., row ..< (row + 1), 0...]
                let s = scores[0..., row ..< (row + 1), 0...]
                let actual = try XCTUnwrap(reducer(x, i, s))
                let ids = i.reshaped(-1).asArray(UInt32.self)
                let xf = x.asType(.float32).reshaped([hidden, 1])
                var expected = MLXArray.zeros([hidden], dtype: .float32)
                for route in 0 ..< routes {
                    let expert = Int(ids[route])
                    let gateValue = matmul(g[expert], xf)
                    let upValue = matmul(u[expert], xf)
                    let activation = gateValue * sigmoid(gateValue) * upValue
                    expected =
                        expected + s.reshaped(-1)[route].asType(.float32)
                        * matmul(d[expert], activation).reshaped(-1)
                }
                let delta = actual.asType(.float32).reshaped(-1) - expected
                let relative =
                    (sqrt((delta * delta).sum())
                    / sqrt((expected * expected).sum())).item(Float.self)
                print(
                    "[Ornith122BFused] score=\(scoreType) row=\(row) relativeF32Error=\(relative)")
                XCTAssertLessThan(relative, 0.02)
                let expanded = expandedDimensions(x, axes: [-2, -3])
                let eagerGate = gate(expanded, i)
                let eagerUp = up(expanded, i)
                let eagerDown = squeezed(
                    down((eagerGate * sigmoid(eagerGate)) * eagerUp, i), axis: -2)
                let eager = (eagerDown * s[.ellipsis, .newAxis]).sum(axis: -2)
                let eagerDelta = eager.asType(.float32).reshaped(-1) - expected
                let eagerError =
                    (sqrt((eagerDelta * eagerDelta).sum())
                    / sqrt((expected * expected).sum())).item(Float.self)
                print("[Ornith122BFused] genericRelativeF32Error=\(eagerError)")
                XCTAssertLessThan(eagerError, 0.02)
                XCTAssertLessThan(relative, max(4 * eagerError, 0.01))
                MLX.eval(actual)
                serial.append(MLXArray(data: actual.asData(access: .copy)))
            }
            for width in [2, 4] {
                let actual = try XCTUnwrap(
                    reducer(
                        input[0..., ..<width, 0...], indices[0..., ..<width, 0...],
                        scores[0..., ..<width, 0...]))
                let expected = concatenated(Array(serial.prefix(width)), axis: 1)
                MLX.eval(actual, expected)
                XCTAssertEqual(actual.shape, [1, width, hidden])
                XCTAssertEqual(actual.dtype, .bfloat16)
                XCTAssertEqual(
                    actual.asData(access: .copy).data,
                    expected.asData(access: .copy).data,
                    "W\(width) must preserve fused single-row arithmetic")
            }
        }
        // Normal prefill is not admitted by the new shape contract.
        XCTAssertNil(
            reducer(
                MLXArray.zeros([1, 5, hidden], dtype: .bfloat16),
                MLXArray.zeros([1, 5, routes], dtype: .uint32),
                MLXArray.ones([1, 5, routes], dtype: .float32)))
    }
}
