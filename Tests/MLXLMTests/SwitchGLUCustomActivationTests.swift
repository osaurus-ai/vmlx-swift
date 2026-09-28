import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

enum SwitchGLUActivationFixture {
    static func projections(_ routed: SwitchGLU, quantized: Bool) throws {
        let weight = broadcast(MLXArray.eye(64), to: [2, 64, 64])
        func projection() -> SwitchLinear {
            let linear = SwitchLinear(inputDims: 64, outputDims: 64, numExperts: 2, weight: weight)
            return quantized
                ? QuantizedSwitchLinear(linear, groupSize: 64, bits: 4, mode: .affine) : linear
        }
        try routed.update(modules: ModuleChildren.unflattened([
            ("gate_proj", projection() as Module), ("up_proj", projection() as Module),
            ("down_proj", projection() as Module),
        ]), verify: .all)
    }

    static func check(
        _ routed: SwitchGLU, quantized: Bool,
        scored: Bool = false, expected: (Float, Float) -> Float,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        try projections(routed, quantized: quantized)
        for tokens in [1, 32] {
            let values = (0 ..< (tokens * 64)).map { Float($0 % 2 == 0 ? -2 : 3) }
            let input = MLXArray(values, [1, tokens, 64])
            let indices = MLXArray((0 ..< (tokens * 2)).map { UInt32($0 % 2) }, [1, tokens, 2])
            let scores = MLXArray(
                (0 ..< (tokens * 2)).map { Float($0 % 2 == 0 ? 0.25 : 0.75) }, [1, tokens, 2])
            let output = routed(input, indices, preDownScores: scored ? scores : nil)
            let actual = output.asArray(Float.self)
            XCTAssertEqual(output.shape, [1, tokens, 2, 64], file: file, line: line)
            for route in 0 ..< (tokens * 2) {
                for column in 0 ..< 64 {
                    let value: Float = column % 2 == 0 ? -2 : 3
                    let score: Float = route % 2 == 0 ? 0.25 : 0.75
                    XCTAssertEqual(
                        actual[route * 64 + column], expected(value, score), accuracy: 0.0002,
                        "quantized=\(quantized) tokens=\(tokens) route=\(route) column=\(column)",
                        file: file, line: line)
                }
            }
        }
    }

    static func silu(_ x: Float) -> Float { x / (1 + exp(-x)) }
    static func gelu(_ x: Float) -> Float {
        0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
    }
}

final class SwitchGLUCustomActivationTests: XCTestCase {
    func testCustomActivationMatchingSiluAtOneIsPreserved() throws {
        try MLXMetalTestLock.withLock {
            for quantized in [false, true] {
                let routed = SwitchGLU(
                    inputDims: 64, hiddenDims: 64, numExperts: 2,
                    activation: { MLXNN.silu($0) + ($0 - 1) * ($0 - 1) })
                try SwitchGLUActivationFixture.check(routed, quantized: quantized) { x, _ in
                    (SwitchGLUActivationFixture.silu(x) + (x - 1) * (x - 1)) * x
                }
            }
        }
    }

    func testCustomActivationMatchingGeluAtOneIsPreserved() throws {
        try MLXMetalTestLock.withLock {
            for quantized in [false, true] {
                let routed = SwitchGLU(
                    inputDims: 64, hiddenDims: 64, numExperts: 2,
                    activation: { safeGeluApproximate($0) + ($0 - 1) * ($0 - 1) })
                try SwitchGLUActivationFixture.check(routed, quantized: quantized) { x, _ in
                    (SwitchGLUActivationFixture.gelu(x) + (x - 1) * (x - 1)) * x
                }
            }
        }
    }

    func testGlueAndScoredGluePrecedence() throws {
        try MLXMetalTestLock.withLock {
            for quantized in [false, true] {
                for scored in [false, true] {
                    let routed = SwitchGLU(
                        inputDims: 64, hiddenDims: 64, numExperts: 2,
                        activation: { $0 * 100 }, glue: { gate, up in gate + up + 7 },
                        scoredGlue: { gate, up, scores in
                            gate - up + scores[.ellipsis, .newAxis, .newAxis] * 11
                        })
                    try SwitchGLUActivationFixture.check(routed, quantized: quantized, scored: scored) {
                        x, score in
                        scored ? score * 11 : 2 * x + 7
                    }
                }
            }
        }
    }

    func testConstructorDoesNotExecuteCustomActivation() {
        MLXMetalTestLock.withLock {
            var calls = 0
            _ = SwitchGLU(
                inputDims: 64, hiddenDims: 64, numExperts: 2,
                activation: { x in
                    calls += 1
                    return MLXNN.silu(x)
                })
            XCTAssertEqual(calls, 0)
        }
    }
}
