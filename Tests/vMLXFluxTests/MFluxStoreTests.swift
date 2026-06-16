import MLX
import XCTest
@testable import vMLXFluxKit
@testable import vMLXFluxModels

final class MFluxStoreTests: XCTestCase {
    func testLinearPrefixFallbackUsesFirstAvailableWeight() throws {
        let weight = MLXArray([0, 1, 2, 3, 4, 5], [2, 3]).asType(.float32)
        let bias = MLXArray([1, 2]).asType(.float32)
        let loaded = LoadedWeights(
            weights: [:],
            componentWeights: [
                "transformer": [
                    "fallback.weight": weight,
                    "fallback.bias": bias,
                ]
            ])
        let store = MFluxStore(loaded)

        let linear = try store.linear(
            "transformer",
            prefixes: ["missing", "fallback"],
            inputDimensions: 3,
            outputDimensions: 2,
            bias: true)
        let output = linear(MLXArray([1, 1, 1], [1, 3]).asType(.float32))
        eval(output)

        XCTAssertEqual(output[0, 0].item(Float.self), 4, accuracy: 0.001)
        XCTAssertEqual(output[0, 1].item(Float.self), 14, accuracy: 0.001)
    }

    func testLinearUsesFp8WeightScaleWhenPresent() throws {
        let rawWeight = MLXArray(Array(UInt8(1) ... UInt8(6)), [2, 3])
        let weightScale = MLXArray([Float(0.5), Float(2.0)], [2])
        let bias = MLXArray([Float(0.25), Float(-0.5)], [2])
        let loaded = LoadedWeights(
            weights: [:],
            componentWeights: [
                "transformer": [
                    "fp8.weight": rawWeight,
                    "fp8.weight_scale": weightScale,
                    "fp8.bias": bias,
                ]
            ])
        let store = MFluxStore(loaded)

        let linear = try store.linear(
            "transformer",
            "fp8",
            inputDimensions: 3,
            outputDimensions: 2,
            bias: true)
        let x = MLXArray([Float(1), Float(2), Float(3)], [1, 3])
        let output = linear(x)
        let decoded = MLX.fromFP8(rawWeight, dtype: .float32) * weightScale[0..., .newAxis]
        let expected = matmul(x, decoded.T) + bias
        eval(output, expected)

        XCTAssertEqual(output[0, 0].item(Float.self), expected[0, 0].item(Float.self), accuracy: 0.001)
        XCTAssertEqual(output[0, 1].item(Float.self), expected[0, 1].item(Float.self), accuracy: 0.001)
    }
}
