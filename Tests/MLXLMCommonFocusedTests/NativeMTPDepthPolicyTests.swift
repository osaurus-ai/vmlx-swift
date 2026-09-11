import XCTest
@testable import MLXLMCommon

final class NativeMTPDepthPolicyTests: XCTestCase {
    func testManualRecommendationDoesNotClaimSamplerCoercion() throws {
        let status = MTPBundleStatus(
            bundleHasMTP: true, configuredLayers: 1, tensorCount: 57,
            mode: .preservedEnabled, nativeMTPTuning: nil)
        for depth in 1...3 {
            let recommendation = try XCTUnwrap(NativeMTPAutoDecodePolicy.manualRecommendation(
                depth: depth, configData: Data("{\"model_type\":\"qwen4_exp\"}".utf8),
                jangConfig: nil, status: status))
            XCTAssertEqual(recommendation.depth, depth)
            XCTAssertTrue(recommendation.reason.contains("request sampling remains in effect"))
            XCTAssertFalse(recommendation.reason.contains("greedy"))
        }
    }

    func testFixedRequestsCannotPromoteAboveSelectedDepth() throws {
        for depth in 1...3 {
            let result = try NativeMTPDepthPolicy.fixed.resolve(requestedDepth: depth, runtimeCap: 5)
            XCTAssertEqual(result.initialDepth, depth)
            XCTAssertEqual(result.maximumDepth, depth)
        }
    }

    func testAdaptiveExplorationIsExplicitAndBounded() throws {
        let result = try NativeMTPDepthPolicy.adaptive(maximumDepth: 3)
            .resolve(requestedDepth: 1, runtimeCap: 5)
        XCTAssertEqual(result.initialDepth, 1)
        XCTAssertEqual(result.maximumDepth, 3)
        let capped = try NativeMTPDepthPolicy.adaptive(maximumDepth: 5)
            .resolve(requestedDepth: 3, runtimeCap: 2)
        XCTAssertEqual(capped.initialDepth, 2)
        XCTAssertEqual(capped.maximumDepth, 2)
    }

    func testRuntimeCapCanOnlyLowerFixedDepth() throws {
        let result = try NativeMTPDepthPolicy.fixed.resolve(requestedDepth: 3, runtimeCap: 2)
        XCTAssertEqual(result.initialDepth, 2)
        XCTAssertEqual(result.maximumDepth, 2)
    }

    func testInvalidPolicyInputsThrow() {
        XCTAssertThrowsError(try NativeMTPDepthPolicy.fixed.resolve(requestedDepth: 0, runtimeCap: 5))
        XCTAssertThrowsError(try NativeMTPDepthPolicy.fixed.resolve(requestedDepth: 3, runtimeCap: 0))
        XCTAssertThrowsError(try NativeMTPDepthPolicy.adaptive(maximumDepth: -1)
            .resolve(requestedDepth: 3, runtimeCap: 5))
    }

    func testParametersDefaultToFixedAndCopiesPreserveExplicitPolicy() {
        var parameters = GenerateParameters()
        XCTAssertEqual(parameters.nativeMTPDepthPolicy, .fixed)
        parameters.nativeMTPDepthPolicy = .adaptive(maximumDepth: 3)
        let copy = parameters
        XCTAssertEqual(copy.nativeMTPDepthPolicy, .adaptive(maximumDepth: 3))
    }
}
