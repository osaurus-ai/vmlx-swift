import Foundation
import MLX
import XCTest
@testable import vMLXFluxModels
import vMLXFluxKit

/// Opt-in: these tests execute Metal. The host-only contracts never load weights.
final class Flux3KleinTensorTests: XCTestCase {
    private func requireGPU() throws {
        guard ProcessInfo.processInfo.environment["FLUX3_GPU_TESTS"] == "1" else {
            throw XCTSkip("Set FLUX3_GPU_TESTS=1 in a Metal-capable session")
        }
    }

    func testVAEChannelFirstPatchificationAndInverse() throws {
        try requireGPU()
        let x = MLXArray((0..<32).map(Float.init), [1, 2, 4, 4])
        let packed = Flux3VAE.patchify(x)
        XCTAssertEqual(packed.shape, [1, 8, 2, 2])
        XCTAssertEqual(Array(packed.asArray(Float.self).prefix(8)), [0, 2, 8, 10, 1, 3, 9, 11])
        XCTAssertEqual(Flux3VAE.unpatchify(packed).asArray(Float.self), x.asArray(Float.self))
        Memory.clearCache()
    }

    func testAdjacentPairRotary() throws {
        try requireGPU()
        let x = MLXArray([Float(1), 2, 3, 4], [1, 1, 1, 4])
        let rotated = Flux3Rotary.apply(x, cos: MLXArray([Float(0), 1], [1, 2]),
            sin: MLXArray([Float(1), 0], [1, 2]))
        XCTAssertEqual(rotated.asArray(Float.self), [-2, 1, 3, 4])
        Memory.clearCache()
    }

}
