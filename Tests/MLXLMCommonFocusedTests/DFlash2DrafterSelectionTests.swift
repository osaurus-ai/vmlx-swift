// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The host-facing half of DFlash 2: which strategy a request gets, and
// why. These run without any weights, so they gate every build.
//
// The behaviour under test is the one that is easy to get wrong and
// impossible to see: a drafter is trained against ONE target, it borrows
// that target's embedding and LM head, and pointing it at a different
// model produces confident tokens from the wrong vocabulary rather than
// an error. Everything here exists to make sure that combination never
// reaches the runtime.

import Foundation
import MLXLMCommon
import XCTest

final class DFlash2DrafterSelectionTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("dflash2-selection-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    /// Writes a drafter folder shaped like `z-lab/<model>-DFlash2`.
    @discardableResult
    private func makeDrafter(
        name: String = "drafter",
        vocabularySize: Int = 248_320,
        hiddenSize: Int = 5120,
        targetLayerIDs: [Int] = [5, 19, 33, 47, 61],
        numTargetLayers: Int = 64,
        blockSize: Int = 8,
        selectorTopK: Int = 16,
        withWeights: Bool = true
    ) throws -> URL {
        let dir = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let config: [String: Any] = [
            "architectures": ["DFlash2DraftModel"],
            "model_type": "qwen3",
            "hidden_size": hiddenSize,
            "num_hidden_layers": 5,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 17408,
            "vocab_size": vocabularySize,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262_144,
            "num_target_layers": numTargetLayers,
            "sliding_window": 2048,
            "is_causal": false,
            "layer_types": Array(repeating: "sliding_attention", count: 5),
            "rope_parameters": ["rope_theta": 10_000_000, "rope_type": "default"],
            "dflash_config": [
                "block_size": blockSize,
                "conv_group_size": 16,
                "conv_kernel_size": 2,
                "mask_token_id": 248_070,
                "selector_rank": 256,
                "selector_top_k": selectorTopK,
                "target_layer_ids": targetLayerIDs,
            ],
        ]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: dir.appendingPathComponent("config.json"))
        if withWeights {
            try Data(repeating: 0, count: 2048)
                .write(to: dir.appendingPathComponent("model.safetensors"))
        }
        return dir
    }

    /// A target `config.json` in the nested `text_config` shape Qwen 3.x
    /// bundles use.
    private func targetConfig(
        vocabularySize: Int = 248_320, layers: Int = 64, hiddenSize: Int = 5120
    ) -> Data {
        let root: [String: Any] = [
            "model_type": "qwen3_5",
            "text_config": [
                "vocab_size": vocabularySize,
                "num_hidden_layers": layers,
                "hidden_size": hiddenSize,
            ],
        ]
        return try! JSONSerialization.data(withJSONObject: root)
    }

    private func settings(drafter: URL?, mode: VMLXMTPServerMode = .auto)
        -> VMLXServerRuntimeSettings
    {
        var s = VMLXServerRuntimeSettings()
        s.mtp = VMLXServerMTPSettings(mode: mode, dflash2DrafterPath: drafter?.path)
        return s
    }

    // MARK: - Reading a drafter folder

    func testReadsDrafterMetadata() throws {
        let dir = try makeDrafter()
        let info = try XCTUnwrap(VMLXDFlash2DrafterInfo.read(at: dir))
        XCTAssertEqual(info.blockSize, 8)
        XCTAssertEqual(info.vocabularySize, 248_320)
        XCTAssertEqual(info.targetLayerIDs, [5, 19, 33, 47, 61])
        XCTAssertEqual(info.targetLayerCount, 64)
        XCTAssertTrue(info.summary.contains("7 tokens drafted per step"))
    }

    func testDFlash1CheckpointIsNotMistakenForDFlash2() throws {
        // A DFlash 1 drafter has `dflash_config` too — the selector is what
        // separates them, and reading `dflash_config` alone would happily
        // half-load a checkpoint this runtime cannot drive.
        let dir = try makeDrafter(name: "v1", selectorTopK: 0)
        XCTAssertNil(VMLXDFlash2DrafterInfo.read(at: dir))
        XCTAssertFalse(DFlash2Loader.looksLikeDFlash2Drafter(at: dir))
    }

    func testRejectionReasonNamesTheProblem() throws {
        let v1 = try makeDrafter(name: "v1", selectorTopK: 0)
        XCTAssertEqual(
            DFlash2Loader.rejectionReason(at: v1),
            "This is a DFlash 1 drafter; DFlash 2 is required")

        let noWeights = try makeDrafter(name: "bare", withWeights: false)
        XCTAssertEqual(
            DFlash2Loader.rejectionReason(at: noWeights),
            "No .safetensors weights in bare")

        let good = try makeDrafter(name: "good")
        XCTAssertNil(DFlash2Loader.rejectionReason(at: good))
    }

    // MARK: - Matching a drafter to a model

    func testMatchingDrafterSupersedesNativeMTP() throws {
        let dir = try makeDrafter()
        let strategy = settings(drafter: dir).resolvedMTPDraftStrategy(
            configData: targetConfig(), jangConfig: nil, status: nil)
        guard case .dflash2(let path, let blockSize)? = strategy else {
            return XCTFail("expected .dflash2, got \(String(describing: strategy?.kindName))")
        }
        XCTAssertEqual(path.path, dir.path)
        XCTAssertNil(blockSize, "block size should default to the checkpoint's own")
    }

    func testDrafterIsUsedEvenWhenMTPModeIsOff() throws {
        // Downloading a drafter and pointing the runtime at it IS the
        // request for speculation. Requiring a second switch would mean a
        // user who did the hard part still gets no speedup and no reason
        // why.
        let dir = try makeDrafter()
        let strategy = settings(drafter: dir, mode: .off).resolvedMTPDraftStrategy(
            configData: targetConfig(), jangConfig: nil, status: nil)
        XCTAssertTrue(strategy?.usesDFlash2 == true)
    }

    func testVocabularyMismatchFallsBackInsteadOfEngaging() throws {
        let dir = try makeDrafter()
        let settings = settings(drafter: dir)
        let mismatched = targetConfig(vocabularySize: 151_936)

        XCTAssertNil(settings.resolvedDFlash2Selection(configData: mismatched))
        XCTAssertFalse(
            settings.resolvedMTPDraftStrategy(
                configData: mismatched, jangConfig: nil, status: nil)?.usesDFlash2 == true)
        XCTAssertEqual(
            settings.dflash2RejectionReason(configData: mismatched),
            "Drafter was trained for a 248320-token vocabulary; this model has 151936.")
    }

    func testTooShallowTargetFallsBack() throws {
        let dir = try makeDrafter()
        let shallow = targetConfig(layers: 40)
        XCTAssertNil(settings(drafter: dir).resolvedDFlash2Selection(configData: shallow))
        XCTAssertEqual(
            settings(drafter: dir).dflash2RejectionReason(configData: shallow),
            "Drafter reads layer 61 of its target; this model has 40 layers.")
    }

    func testHiddenSizeMismatchFallsBack() throws {
        let dir = try makeDrafter()
        let narrow = targetConfig(hiddenSize: 4096)
        XCTAssertNil(settings(drafter: dir).resolvedDFlash2Selection(configData: narrow))
        XCTAssertEqual(
            settings(drafter: dir).dflash2RejectionReason(configData: narrow),
            "Drafter expects a hidden size of 5120; this model uses 4096.")
    }

    func testDeletedDrafterFolderDegradesQuietly() throws {
        let dir = try makeDrafter()
        let settings = settings(drafter: dir)
        try FileManager.default.removeItem(at: dir)

        XCTAssertNil(settings.resolvedDFlash2Selection(configData: targetConfig()))
        XCTAssertEqual(
            settings.dflash2RejectionReason(configData: targetConfig()),
            "Drafter folder no longer exists at \(dir.path).")
    }

    func testNoDrafterSelectedLeavesMTPResolutionUntouched() throws {
        let settings = settings(drafter: nil)
        XCTAssertNil(settings.resolvedDFlash2Selection(configData: targetConfig()))
        XCTAssertNil(settings.dflash2RejectionReason(configData: targetConfig()))
        // Without MTP evidence there is no strategy at all — the point is
        // that adding DFlash 2 did not change that answer.
        XCTAssertNil(
            settings.resolvedMTPDraftStrategy(
                configData: targetConfig(), jangConfig: nil, status: nil))
    }

    // MARK: - Settings round-trip

    func testSettingsWrittenBeforeDFlash2StillDecode() throws {
        // Older settings files have neither key. Decoding must default
        // them rather than throwing, or upgrading would wipe every other
        // runtime setting in the file.
        let legacy = """
            {"mode":"auto","keepDraftCacheSeparate":true,"acceptedTokensOnlyEnterBaseCache":true}
            """
        let decoded = try JSONDecoder().decode(
            VMLXServerMTPSettings.self, from: Data(legacy.utf8))
        XCTAssertNil(decoded.dflash2DrafterPath)
        XCTAssertNil(decoded.dflash2BlockSize)
        XCTAssertEqual(decoded.mode, .auto)
    }

    func testSelectedDrafterSurvivesEncodeDecode() throws {
        let dir = try makeDrafter()
        let original = VMLXServerMTPSettings(dflash2DrafterPath: dir.path, dflash2BlockSize: 6)
        let restored = try JSONDecoder().decode(
            VMLXServerMTPSettings.self, from: JSONEncoder().encode(original))
        XCTAssertEqual(restored.dflash2DrafterPath, dir.path)
        XCTAssertEqual(restored.dflash2BlockSize, 6)
    }

    func testExplicitBlockSizeReachesTheStrategy() throws {
        let dir = try makeDrafter()
        var s = settings(drafter: dir)
        s.mtp.dflash2BlockSize = 4
        guard case .dflash2(_, let blockSize)? = s.resolvedMTPDraftStrategy(
            configData: targetConfig(), jangConfig: nil, status: nil)
        else {
            return XCTFail("expected .dflash2")
        }
        XCTAssertEqual(blockSize, 4)
    }
}
