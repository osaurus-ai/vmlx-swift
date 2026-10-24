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

    /// osaurus#2652: under mmap a JANGTQ-native bundle keeps its f16 affine
    /// scales unless its family is listed here; Ling 2.6 flash (bailing_hybrid)
    /// produced all-NaN logits without the bf16 materialisation and answered
    /// coherently with it (single-variable control, 2026-09-06).
    func testJANGTQMmapBFloat16FamiliesIncludeBailingHybrid() {
        XCTAssertTrue(requiresJANGTQMmapBFloat16(config: ["model_type": "bailing_hybrid"]))
        XCTAssertTrue(requiresJANGTQMmapBFloat16(config: ["model_type": "Bailing_Hybrid"]))
        XCTAssertTrue(requiresJANGTQMmapBFloat16(config: ["model_type": "nemotron_h"]))
        XCTAssertTrue(requiresJANGTQMmapBFloat16(config: ["model_type": "qwen3_5_moe", "layer_types": ["linear_attention", "full_attention"]]))
        XCTAssertFalse(requiresJANGTQMmapBFloat16(config: ["model_type": "qwen3_5_moe", "layer_types": ["full_attention"]]))
        XCTAssertFalse(requiresJANGTQMmapBFloat16(config: ["model_type": "llama"]))
        XCTAssertFalse(requiresJANGTQMmapBFloat16(config: [:]))
    }

    /// bottlecapai/ThinkingCap-Qwen3.8-27B-MLX-4bit-DWQ ships a separate MTP
    /// drafter in `mtp-drafter/`, with its own config.json. The recursive glob
    /// merged its prefix-free keys (`fc`, `layers`, `norm`, `pre_fc_norm_*`)
    /// into the target, and `model.update()` rejected them as unhandled. A
    /// directory with its own config.json is another model; a weights-only
    /// subdirectory, and anything the root index names, stay eligible.
    func testNestedModelBundleShardsAreExcluded() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("nested-bundle-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        func touch(_ path: String) throws {
            let url = root.appendingPathComponent(path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: url)
        }
        for path in [
            "config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
            "model-vision.safetensors", "mtp.safetensors",
            "mtp-drafter/config.json", "mtp-drafter/model.safetensors",
            "weights/extra.safetensors",
            "drafters/config.json",
            "awq-calibration.safetensors",
        ] {
            try touch(path)
        }
        // A Hugging Face cache stores every snapshot file as a symlink into `blobs/`, outside the
        // bundle, so resolving the FILE would lose its place in the bundle.
        let blob = root.appendingPathExtension("blob")
        defer { try? FileManager.default.removeItem(at: blob) }
        try Data().write(to: blob)
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("drafters/b"), withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("drafters/b/model.safetensors"), withDestinationURL: blob)
        let base = root.standardizedFileURL.resolvingSymlinksInPath().path
        func candidates() -> [String] {
            candidateShardURLs(in: root, skipDSV4Sidecar: false)
                .map { url in
                    let parent = url.deletingLastPathComponent().standardizedFileURL
                        .resolvingSymlinksInPath().path
                    return String((parent + "/" + url.lastPathComponent).dropFirst(base.count + 1))
                }
                .sorted()
        }

        XCTAssertEqual(candidates(), [
            "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
            "model-vision.safetensors", "mtp.safetensors", "weights/extra.safetensors",
        ])

        // A root index that names the nested file keeps it eligible.
        try JSONSerialization.data(withJSONObject: [
            "weight_map": ["a": "model-00001-of-00002.safetensors", "b": "mtp-drafter/model.safetensors"]
        ]).write(to: root.appendingPathComponent("model.safetensors.index.json"))
        XCTAssertEqual(candidates(), [
            "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
            "model-vision.safetensors", "mtp-drafter/model.safetensors", "mtp.safetensors",
            "weights/extra.safetensors",
        ])
    }
}
