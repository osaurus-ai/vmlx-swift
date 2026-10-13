// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DSpark drafter (MiniCPM-V 4.7 `dspark/`): config contract, module topology and draft-row mapping.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class DSparkDrafterTests: XCTestCase {

    /// The MiniCPM-V 4.7 1B DSpark config as shipped (contract split between the root and `dflash_config`).
    private func dsparkConfig(markovHeadType: String = "vanilla", extra: [String: Any] = [:]) -> [String: Any] {
        var root: [String: Any] = [
            "architectures": ["Qwen3DSparkModel"], "block_size": 15, "hidden_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "intermediate_size": 128, "vocab_size": 300, "rms_norm_eps": 1e-6,
            "rope_parameters": ["rope_theta": 10_000_000, "rope_type": "default"],
            "layer_types": ["full_attention", "full_attention"], "sliding_window": NSNull(),
            "markov_rank": 8, "markov_head_type": markovHeadType, "enable_confidence_head": true,
            "dflash_config": [
                "mask_token_id": 299, "target_layer_ids": [1, 3], "projector_type": "dspark",
                "markov_rank": 8, "enable_confidence_head": true,
            ] as [String: Any],
        ]
        for (k, v) in extra { root[k] = v }
        return root
    }

    func testDSparkContractNormalizesBlockToVerifyWidth() throws {
        let c = try DFlash2Configuration(json: dsparkConfig())
        XCTAssertTrue(c.isDSpark)
        XCTAssertFalse(c.isDFlash2)
        XCTAssertTrue(c.isRunnable)
        XCTAssertTrue(c.anchorRowDrafts)
        // Declared 15 counts drafts (the anchor row drafts too); the lane speaks verify width.
        XCTAssertEqual(c.blockSize, 16)
        XCTAssertEqual(c.markovRank, 8)
        XCTAssertEqual(c.ropeTheta, 10_000_000)
        XCTAssertFalse(c.ropeTraditional, "MiniCPM-V 4.7 DSpark is neox")
        XCTAssertTrue(c.hasConfidenceHead)
        XCTAssertNil(c.slidingWindow)
    }

    func testInterleavedRopeIsReadFromTheStamp() throws {
        let c = try DFlash2Configuration(json: dsparkConfig(extra: ["rope_is_neox_style": false]))
        XCTAssertTrue(c.ropeTraditional)
    }

    func testUnportedMarkovHeadIsRefusedByName() {
        XCTAssertThrowsError(try DFlash2Configuration(json: dsparkConfig(markovHeadType: "gated"))) { error in
            guard case DFlash2LoadError.unsupportedMarkovHead(let kind) = error else {
                return XCTFail("unexpected \(error)")
            }
            XCTAssertEqual(kind, "gated")
        }
    }

    func testDSparkModelHasMarkovAndConfidenceHeadsAndNoDFlash2Modules() throws {
        let model = DFlash2DraftModel(try DFlash2Configuration(json: dsparkConfig()))
        let keys = Set(model.parameters().flattened().map(\.0))
        XCTAssertTrue(keys.contains("markov_head.markov_w1.weight"))
        XCTAssertTrue(keys.contains("markov_head.markov_w2.weight"))
        XCTAssertTrue(keys.contains("confidence_head.proj.weight"))
        XCTAssertTrue(keys.contains("fc.weight"))
        XCTAssertFalse(keys.contains { $0.hasPrefix("candidate_selector.") })
        XCTAssertFalse(keys.contains { $0.contains("attention_conv") || $0.contains("mlp_conv") })
        XCTAssertFalse(model.supportsLattice, "a DSpark drafter drafts one chain, no tree lattice")
    }

    func testDSparkExtremeBlockRejectsWithoutOverflow() throws {
        let config = dsparkConfig(extra: ["block_size": Int.max])
        XCTAssertThrowsError(try DFlash2Configuration(json: config))
    }

    func testDSparkExtremeConfidenceRankRejectsWithoutOverflow() throws {
        var config = dsparkConfig()
        var inner = config["dflash_config"] as! [String: Any]
        inner["markov_rank"] = Int.max
        config["dflash_config"] = inner
        XCTAssertThrowsError(try DFlash2ArtifactMetadata.requiredShapes(
            configData: JSONSerialization.data(withJSONObject: config)))
    }

    func testDSparkLadderStaysWithinTheTrainedBlock() {
        XCTAssertEqual(DFlash2WidthChooser.dsparkLadder(trained: 16), [3, 5, 8, 16])
        XCTAssertEqual(DFlash2WidthChooser.dsparkLadder(trained: 6), [3, 5, 6])
    }
}
