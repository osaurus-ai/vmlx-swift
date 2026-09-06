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

    /// osaurus#2652: JANGQ-AI/Ling-2.6-flash-JANGTQ ships a complete 29-shard
    /// family and an index naming 31 files from an earlier layout. Only a
    /// COMPLETE replacement family may stand in for a stale index; an empty
    /// or partial download, unrelated sidecars, two families, or a partially
    /// present indexed set all stay fail-closed.
    func testStaleIndexNeedsOneCompleteReplacementFamily() {
        let indexed = ["model-00001-of-00031.safetensors", "model-00002-of-00031.safetensors"]
        let family = (1...29).map { String(format: "model-%05d-of-00029.safetensors", $0) }

        XCTAssertEqual(
            indexManifestDecision(indexedNames: indexed, presentNames: family + ["jangtq_runtime.safetensors"]),
            .staleIndex(missing: indexed, replacement: family))
        XCTAssertEqual(
            indexManifestDecision(indexedNames: indexed, presentNames: indexed),
            .manifest(indexed))
        XCTAssertEqual(
            indexManifestDecision(indexedNames: indexed, presentNames: ["model-00001-of-00031.safetensors"]),
            .truncated(missing: ["model-00002-of-00031.safetensors"]))
        // Empty download: nothing present.
        XCTAssertEqual(
            indexManifestDecision(indexedNames: indexed, presentNames: []),
            .incomplete(missing: indexed))
        // One replacement shard missing.
        XCTAssertEqual(
            indexManifestDecision(indexedNames: indexed, presentNames: Array(family.dropLast())),
            .incomplete(missing: indexed))
        // Unrelated sidecar only.
        XCTAssertEqual(
            indexManifestDecision(indexedNames: indexed, presentNames: ["jangtq_runtime.safetensors"]),
            .incomplete(missing: indexed))
        // Two families: ambiguous, not a replacement.
        XCTAssertEqual(
            indexManifestDecision(indexedNames: indexed, presentNames: family + ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]),
            .incomplete(missing: indexed))
        XCTAssertEqual(indexManifestDecision(indexedNames: [], presentNames: family), .manifest([]))
        XCTAssertNil(completeNumberedShardFamily(in: ["model-00002-of-00002.safetensors"]))
        XCTAssertEqual(completeNumberedShardFamily(in: ["model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors"]),
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    }
}
