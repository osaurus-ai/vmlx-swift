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
        let decoded = mfluxFromFP8(rawWeight, dtype: .float32) * weightScale[0..., .newAxis]
        let expected = matmul(x, decoded.T) + bias
        eval(output, expected)

        XCTAssertEqual(output[0, 0].item(Float.self), expected[0, 0].item(Float.self), accuracy: 0.001)
        XCTAssertEqual(output[0, 1].item(Float.self), expected[0, 1].item(Float.self), accuracy: 0.001)
    }

    func testLinearUsesBitsAndBytesNF4HighNibbleFirst() throws {
        var packed = Array(repeating: UInt8(0), count: 32)
        packed[0] = 0x12
        let quantMap = MLXArray((0 ..< 16).map { Float($0) }, [16])
        let absmax = MLXArray([Float(1)], [1])
        let loaded = LoadedWeights(
            weights: [:],
            componentWeights: [
                "transformer": [
                    "nf4.weight": MLXArray(packed, [32, 1]),
                    "nf4.weight.absmax": absmax,
                    "nf4.weight.quant_map": quantMap,
                    "nf4.weight.quant_state.bitsandbytes__nf4": MLXArray(Array(UInt8(0) ..< UInt8(4)), [4]),
                ]
            ])
        let store = MFluxStore(loaded)

        let linear = try store.linear(
            "transformer",
            "nf4",
            inputDimensions: 64,
            outputDimensions: 1)
        var firstInput = Array(repeating: Float(0), count: 64)
        firstInput[0] = 1
        var secondInput = Array(repeating: Float(0), count: 64)
        secondInput[1] = 1

        let firstOutput = linear(MLXArray(firstInput, [1, 64]))
        let secondOutput = linear(MLXArray(secondInput, [1, 64]))
        eval(firstOutput, secondOutput)

        XCTAssertEqual(firstOutput[0, 0].item(Float.self), 1, accuracy: 0.001)
        XCTAssertEqual(secondOutput[0, 0].item(Float.self), 2, accuracy: 0.001)
    }
}
