import MLX
import XCTest

@testable import MLXLMCommon

final class SwitchGLUKnownActivationTests: XCTestCase {
    func testKnownSiluKeepsFastPath() throws {
        try MLXMetalTestLock.withLock {
            for quantized in [false, true] {
                let routed = SwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 2)
                XCTAssertTrue(routed.isSiluActivation)
                XCTAssertFalse(routed.isGeluActivation)
                try SwitchGLUActivationFixture.check(routed, quantized: quantized) { x, _ in
                    SwitchGLUActivationFixture.silu(x) * x
                }
            }
        }
    }

    func testKnownGeluKeepsFastPath() throws {
        try MLXMetalTestLock.withLock {
            for quantized in [false, true] {
                let routed = SwitchGLU(
                    inputDims: 64, hiddenDims: 64, numExperts: 2,
                    activation: .geluApproximate)
                XCTAssertFalse(routed.isSiluActivation)
                XCTAssertTrue(routed.isGeluActivation)
                try SwitchGLUActivationFixture.check(routed, quantized: quantized) { x, _ in
                    SwitchGLUActivationFixture.gelu(x) * x
                }
            }
        }
    }
    func testExplicitCustomIdentityCannotSelectKnownFastPath() throws {
        try MLXMetalTestLock.withLock {
            let routed = SwitchGLU(
                inputDims: 64, hiddenDims: 64, numExperts: 2,
                activation: .custom { $0 * 2 })
            XCTAssertFalse(routed.isSiluActivation)
            XCTAssertFalse(routed.isGeluActivation)
            try SwitchGLUActivationFixture.check(routed, quantized: false) { x, _ in 2 * x * x }
        }
    }

}
