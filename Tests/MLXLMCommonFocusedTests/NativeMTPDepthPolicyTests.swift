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

    func testNemotronManualControlsUseArchitectureAndRequirePhysicalHead() throws {
        for modelType in ["nemotron_h", "nemotronh", "nemotron_h_text", "nemotronh_text"] {
            let data = try JSONSerialization.data(withJSONObject: ["model_type": modelType])
            XCTAssertTrue(NativeMTPAutoDecodePolicy.supportsModel(configData: data))
            XCTAssertNil(NativeMTPAutoDecodePolicy.manualRecommendation(
                depth: 2, configData: data, jangConfig: nil, status: nil))
            let status = MTPBundleStatus(
                bundleHasMTP: true, configuredLayers: 1, tensorCount: 40,
                mode: .preservedEnabled, nativeMTPTuning: nil)
            let manual = try XCTUnwrap(NativeMTPAutoDecodePolicy.manualRecommendation(
                depth: 2, configData: data, jangConfig: nil, status: status))
            XCTAssertEqual(manual.depth, 2)
            // Hardware presence makes the manual control available. It does
            // not invent performance tuning or change the family default.
            XCTAssertNil(NativeMTPAutoDecodePolicy.recommendation(
                configData: data, jangConfig: nil, status: status))
        }
        XCTAssertFalse(NativeMTPAutoDecodePolicy.supportsModel(
            configData: Data("{\"model_type\":\"unsupported\"}".utf8)))
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

    /// Adaptive is the default; `.fixed` is an explicit choice. A `.fixed` default made any resolved
    /// depth (bundle `recommended_num_drafts`, tuning sidecar, family default) a hard ceiling — the vmlx
    /// Python bug of 2026-10-20 that capped Ornith 1.5 122B's Adaptive at its stamped D1.
    func testParametersDefaultToAdaptiveAndCopiesPreserveExplicitPolicy() {
        var parameters = GenerateParameters()
        XCTAssertEqual(
            parameters.nativeMTPDepthPolicy,
            .adaptive(maximumDepth: GenerateParameters.defaultAdaptiveDepthCeiling))
        parameters.nativeMTPDepthPolicy = .fixed
        let copy = parameters
        XCTAssertEqual(copy.nativeMTPDepthPolicy, .fixed)
    }

    /// A stamped / sidecar / family-default depth only sets where the default policy STARTS.
    func testResolvedDepthIsAStartNotACeilingByDefault() throws {
        for stamped in [1, 2, 3] {
            let resolution = try GenerateParameters().nativeMTPDepthPolicy.resolve(
                requestedDepth: stamped, runtimeCap: 5)
            XCTAssertEqual(resolution.initialDepth, stamped)
            XCTAssertEqual(resolution.maximumDepth, GenerateParameters.defaultAdaptiveDepthCeiling)
        }
        XCTAssertGreaterThanOrEqual(GenerateParameters.defaultAdaptiveDepthCeiling, 3)
    }
}
