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
}
