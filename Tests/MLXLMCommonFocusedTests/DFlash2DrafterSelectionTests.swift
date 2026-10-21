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
        if withWeights && selectorTopK > 0 {
            try writeCompleteWeights(at: dir)

        }
        return dir
    }

    func testExtremeDimensionsRejectWithoutOverflowOrHugeAllocation() throws {
        let directory = try makeDrafter(withWeights: false)
        let url = directory.appendingPathComponent("config.json")
        let baseline =
            try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        for key in [
            "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "head_dim", "intermediate_size", "vocab_size",
        ] {
            var config = baseline
            config[key] = Int.max
            XCTAssertThrowsError(
                try DFlash2ArtifactMetadata.requiredShapes(
                    configData: JSONSerialization.data(withJSONObject: config)), key)
        }
        for key in ["conv_kernel_size", "selector_rank"] {
            var config = baseline
            var inner = config["dflash_config"] as! [String: Any]
            inner[key] = Int.max
            config["dflash_config"] = inner
            XCTAssertThrowsError(
                try DFlash2ArtifactMetadata.requiredShapes(
                    configData: JSONSerialization.data(withJSONObject: config)), key)
        }
    }

    private func writeCompleteWeights(at dir: URL, omit: String? = nil, packed: Bool = false) throws
    {
        let configURL = dir.appendingPathComponent("config.json")
        var config =
            try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as! [String: Any]
        if packed { config["quantization"] = ["bits": 4, "group_size": 64, "mode": "affine"] }
        let configData = try JSONSerialization.data(withJSONObject: config)
        try configData.write(to: configURL)
        let shapes = try DFlash2ArtifactMetadata.requiredShapes(configData: configData)
        var header: [String: Any] = [:]
        var offset = 0
        func add(_ name: String, _ shape: [Int], _ dtype: String, _ bytes: Int) {
            let end = offset + shape.reduce(bytes, *)
            header[name] = ["dtype": dtype, "shape": shape, "data_offsets": [offset, end]]
            offset = end
        }
        for (name, shape) in shapes.sorted(by: { $0.key < $1.key }) where name != omit {
            if packed && shape.count == 2 && shape[1] % 64 == 0 {
                add(name, [shape[0], shape[1] / 8], "U32", 4)
                let prefix = String(name.dropLast(7))
                add(prefix + ".scales", [shape[0], shape[1] / 64], "BF16", 2)
                add(prefix + ".biases", [shape[0], shape[1] / 64], "BF16", 2)
            } else {
                let key =
                    name.hasPrefix("candidate_selector.") && name.contains("codebook")
                    ? String(name.dropLast(7)) : name
                add(key, shape, "BF16", 2)
            }
        }
        let bytes = try JSONSerialization.data(withJSONObject: header)
        var length = UInt64(bytes.count).littleEndian
        var prefix = withUnsafeBytes(of: &length) { Data($0) }
        prefix.append(bytes)
        let file = dir.appendingPathComponent("model.safetensors")
        try prefix.write(to: file)
        let handle = try FileHandle(forWritingTo: file)
        defer { try? handle.close() }
        // Sparse fixture: valid header/length, never allocate/read model payload.
        try handle.truncate(atOffset: UInt64(prefix.count + offset))
    }

    func testMissingRequiredTensorRejectsOtherwiseValidArtifact() throws {
        let dir = try makeDrafter()
        XCTAssertNil(DFlash2ArtifactMetadata.rejectionReason(at: dir))
        try writeCompleteWeights(at: dir, omit: "layers.0.self_attn.q_proj.weight")
        XCTAssertNotNil(DFlash2ArtifactMetadata.rejectionReason(at: dir))
        XCTAssertNil(VMLXDFlash2DrafterInfo.read(at: dir))
    }

    func testValidPayloadWithWrongProjectionShapeIsRejected() throws {
        let dir = try makeDrafter()
        let file = dir.appendingPathComponent("model.safetensors")
        let reader = try FileHandle(forReadingFrom: file)
        let lengthBytes = try XCTUnwrap(reader.read(upToCount: 8))
        let oldLength = lengthBytes.enumerated().reduce(UInt64(0)) {
            $0 | UInt64($1.element) << UInt64(8 * $1.offset)
        }
        var header =
            try JSONSerialization.jsonObject(
                with: XCTUnwrap(reader.read(upToCount: Int(oldLength)))) as! [String: Any]
        let oldSize = try reader.seekToEnd()
        try reader.close()
        var projection = header["layers.0.self_attn.q_proj.weight"] as! [String: Any]
        projection["shape"] = [5120, 4096]  // same byte count, wrong orientation
        header["layers.0.self_attn.q_proj.weight"] = projection
        let bytes = try JSONSerialization.data(withJSONObject: header)
        var length = UInt64(bytes.count).littleEndian
        var prefix = withUnsafeBytes(of: &length) { Data($0) }
        prefix.append(bytes)
        try prefix.write(to: file)
        let writer = try FileHandle(forWritingTo: file)
        try writer.truncate(atOffset: oldSize - oldLength + UInt64(bytes.count))
        try writer.close()
        XCTAssertNotNil(DFlash2ArtifactMetadata.rejectionReason(at: dir))
        XCTAssertNil(VMLXDFlash2DrafterInfo.read(at: dir))
    }

    func testPackedAffineAndCodebookAliasesAreRecognized() throws {
        let dir = try makeDrafter()
        XCTAssertNotNil(VMLXDFlash2DrafterInfo.read(at: dir))
        try writeCompleteWeights(at: dir, packed: true)
        XCTAssertNil(DFlash2ArtifactMetadata.rejectionReason(at: dir))
        XCTAssertNotNil(VMLXDFlash2DrafterInfo.read(at: dir))
    }

    func testInstalledDrafterHeadersWithoutLoadingWeights() throws {
        guard let paths = ProcessInfo.processInfo.environment["VMLX_DFLASH_METADATA_LOCAL_PATHS"]
        else {
            throw XCTSkip("Set explicit local drafter paths for header-only inventory proof")
        }
        let directories = try JSONDecoder().decode([String].self, from: Data(paths.utf8))
        XCTAssertFalse(directories.isEmpty)
        for path in directories {
            let dir = URL(fileURLWithPath: path)
            XCTAssertNil(DFlash2ArtifactMetadata.rejectionReason(at: dir), path)
            XCTAssertNotNil(VMLXDFlash2DrafterInfo.read(at: dir), path)
        }
    }

    func testOptionalDrafterPreservesCancellation() async throws {
        let path = root.appendingPathComponent("absent")
        let task = Task.detached { () -> Bool in
            withUnsafeCurrentTask { $0?.cancel() }
            do {
                _ = try DFlash2DrafterResolver().optionalDrafter(at: path)
                return false
            } catch is CancellationError { return true } catch { return false }
        }
        let cancelled = await task.value
        XCTAssertTrue(cancelled)
    }

    func testOptionalMissingDrafterReturnsARWithoutLoading() throws {
        XCTAssertNil(
            try DFlash2DrafterResolver().optionalDrafter(at: root.appendingPathComponent("absent")))
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

    func testBundledDrafterDoesNotAlsoLoadUnusedNativeHead() throws {
        var config = try JSONSerialization.jsonObject(with: targetConfig()) as! [String: Any]
        config["model_type"] = "qwen4_exp"
        config["mtp_num_hidden_layers"] = 1
        config["quantization"] = ["bits": 4, "group_size": 64]
        let data = try JSONSerialization.data(withJSONObject: config)
        let status = MTPBundleStatus(
            bundleHasMTP: true, configuredLayers: 1, tensorCount: 57,
            mode: .preservedEnabled, measuredFamilyAutoDepth: 3)
        var settings = VMLXServerRuntimeSettings(mtp: .init(mode: .auto))
        XCTAssertTrue(settings.resolvedLoadConfiguration(
            configData: data, jangConfig: nil, status: status, bundleDirectory: root).nativeMTP)

        let drafter = try makeDrafter(name: "dflash2")
        guard case .dflash2(let path, let width)? = settings.resolvedMTPDraftStrategy(
            configData: data, jangConfig: nil, status: status, bundleDirectory: root)
        else { return XCTFail("compatible bundled drafter must win over native MTP") }
        XCTAssertEqual(path.standardizedFileURL.path, drafter.standardizedFileURL.path)
        XCTAssertNil(width, "default remains adaptive")
        XCTAssertFalse(settings.resolvedLoadConfiguration(
            configData: data, jangConfig: nil, status: status, bundleDirectory: root).nativeMTP)

        settings.mtp.mode = .off
        XCTAssertNil(settings.resolvedMTPDraftStrategy(
            configData: data, jangConfig: nil, status: status, bundleDirectory: root))
        XCTAssertFalse(settings.resolvedLoadConfiguration(
            configData: data, jangConfig: nil, status: status, bundleDirectory: root).nativeMTP)

        settings.mtp.mode = .auto
        try FileManager.default.removeItem(at: drafter)
        XCTAssertTrue(settings.resolvedLoadConfiguration(
            configData: data, jangConfig: nil, status: status, bundleDirectory: root).nativeMTP,
            "a removed drafter must restore native loading on the next load")
    }

    func testOffDisablesSelectedDrafterWithoutDeletingSelection() throws {
        let dir = try makeDrafter()
        var settings = settings(drafter: dir, mode: .off)
        XCTAssertNil(
            settings.resolvedMTPDraftStrategy(
                configData: targetConfig(), jangConfig: nil, status: nil))
        XCTAssertEqual(settings.mtp.dflash2DrafterPath, dir.path)
        settings.mtp.mode = .auto
        XCTAssertTrue(
            settings.resolvedMTPDraftStrategy(
                configData: targetConfig(), jangConfig: nil, status: nil)?.usesDFlash2 == true)
    }

    func testIncompleteAndTruncatedDraftersAreNotSelectable() throws {
        XCTAssertNil(
            VMLXDFlash2DrafterInfo.read(at: try makeDrafter(name: "bare", withWeights: false)))
        let dir = try makeDrafter()
        let file = dir.appendingPathComponent("model.safetensors")
        let handle = try FileHandle(forWritingTo: file)
        let size = try handle.seekToEnd()
        try handle.truncate(atOffset: size - 1)
        try handle.close()
        XCTAssertNil(VMLXDFlash2DrafterInfo.read(at: dir))
    }

    func testNestedWeightsAndSymlinksMatchLoaderLayout() throws {
        let dir = try makeDrafter()
        let original = dir.appendingPathComponent("model.safetensors")
        let nested = dir.appendingPathComponent("weights")
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        let moved = nested.appendingPathComponent("part.safetensors")
        try FileManager.default.moveItem(at: original, to: moved)
        XCTAssertNotNil(VMLXDFlash2DrafterInfo.read(at: dir))
        let backing = root.appendingPathComponent("backing.bin")
        try FileManager.default.moveItem(at: moved, to: backing)
        try FileManager.default.createSymbolicLink(at: moved, withDestinationURL: backing)
        XCTAssertNotNil(VMLXDFlash2DrafterInfo.read(at: dir))
    }

    func testMalformedHeadersAndNegativeLayerIDsAreRejected() throws {
        let invalidLayers = try makeDrafter(name: "negative")
        let cfg = invalidLayers.appendingPathComponent("config.json")
        let text = try String(contentsOf: cfg, encoding: .utf8).replacingOccurrences(
            of: "[5,19,33,47,61]", with: "[-1,4]")
        try text.write(to: cfg, atomically: true, encoding: .utf8)
        XCTAssertNil(VMLXDFlash2DrafterInfo.read(at: invalidLayers))
        let invalidFile = try makeDrafter(name: "malformed")
        try Data(repeating: 0, count: 32).write(
            to: invalidFile.appendingPathComponent("model.safetensors"))
        XCTAssertNil(VMLXDFlash2DrafterInfo.read(at: invalidFile))
    }

    func testIndexMustReferencePresentTensorAndShard() throws {
        let dir = try makeDrafter()
        try Data(#"{"weight_map":{"fixture.weight":"missing.safetensors"}}"#.utf8)
            .write(to: dir.appendingPathComponent("model.safetensors.index.json"))
        XCTAssertNil(VMLXDFlash2DrafterInfo.read(at: dir))
    }

    func testMissingDimensionsAndDifferentTargetLayerCountDoNotFit() throws {
        let info = try XCTUnwrap(VMLXDFlash2DrafterInfo.read(at: try makeDrafter()))
        XCTAssertNotNil(info.mismatchReason(configData: nil))
        XCTAssertNotNil(info.mismatchReason(configData: Data(#"{"vocab_size":248320}"#.utf8)))
        XCTAssertNotNil(info.mismatchReason(configData: targetConfig(layers: 65)))
        XCTAssertNil(info.mismatchReason(configData: targetConfig()))
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
        guard
            case .dflash2(_, let blockSize)? = s.resolvedMTPDraftStrategy(
                configData: targetConfig(), jangConfig: nil, status: nil)
        else {
            return XCTFail("expected .dflash2")
        }
        XCTAssertEqual(blockSize, 4)
    }
}
