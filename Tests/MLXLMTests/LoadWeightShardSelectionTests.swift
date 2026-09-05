import XCTest

@testable import MLXLMCommon

final class LoadWeightShardSelectionTests: XCTestCase {
    func testCalibrationArtifactsAreExcluded() {
        XCTAssertTrue(isAuxiliaryCalibrationSafetensor("awq-calibration.safetensors"))
        XCTAssertTrue(isAuxiliaryCalibrationSafetensor("jang_imatrix.safetensors"))
        XCTAssertTrue(isAuxiliaryCalibrationSafetensor("awq_activations.safetensors"))
    }

    func testInferenceArtifactsRemainEligible() {
        XCTAssertFalse(isAuxiliaryCalibrationSafetensor("model-00001-of-00102.safetensors"))
        XCTAssertFalse(isAuxiliaryCalibrationSafetensor("model.safetensors"))
        XCTAssertFalse(isAuxiliaryCalibrationSafetensor("jangtq_stacked.safetensors"))
        XCTAssertFalse(isAuxiliaryCalibrationSafetensor("jangpress-prestacked.safetensors"))
    }
}
