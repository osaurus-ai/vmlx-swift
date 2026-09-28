// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
import MLX
import MLXRandom
import XCTest

@testable import MLXLMCommon

final class DFlash2SamplingReadbackTests: XCTestCase {
    private func deviceAcceptance(
        draftTokens: MLXArray, targetProbabilities: MLXArray,
        draftProbabilities: MLXArray, draftIndices: MLXArray
    ) -> DFlash2Sampling.Acceptance {
        let result = DFlash2Sampling.sampledAcceptance(
            draftTokens: draftTokens, targetProbabilities: targetProbabilities,
            draftProbabilities: draftProbabilities, draftIndices: draftIndices
        ).asArray(Int32.self)
        return .init(accepted: Int(result[0]), bonus: Int(result[1]))
    }

    func testFullAcceptanceUsesBonusRow() {
        let result = deviceAcceptance(
            draftTokens: MLXArray([Int32(0), 1]).reshaped(1, 2),
            targetProbabilities: MLXArray([Float(1), 0, 0, 0, 1, 0, 0, 0, 1])
                .reshaped(1, 3, 3),
            draftProbabilities: MLXArray([Float(1), 1]).reshaped(1, 2, 1),
            draftIndices: MLXArray([Int32(0), 1]).reshaped(1, 2, 1))
        XCTAssertEqual(result.accepted, 2)
        XCTAssertEqual(result.bonus, 2)
    }

    func testRejectionGathersTheCorrectRowAndResidual() {
        // Row zero agrees; row one rejects with probability one. The target
        // replacement is outside the draft candidate set, not its last row.
        let result = deviceAcceptance(
            draftTokens: MLXArray([Int32(0), 1, 2]).reshaped(1, 3),
            targetProbabilities: MLXArray([
                Float(1), 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
            ]).reshaped(1, 4, 3),
            draftProbabilities: MLXArray([Float(1), 1, 1]).reshaped(1, 3, 1),
            draftIndices: MLXArray([Int32(0), 1, 2]).reshaped(1, 3, 1))
        XCTAssertEqual(result.accepted, 1)
        XCTAssertEqual(result.bonus, 2)
    }

    func testSampledOutputPreservesTargetDistribution() {
        MLXRandom.seed(417)
        // q deliberately differs from p and has no support at token 2.
        // Counting accepted proposals AND replacements tests the complete
        // rejection construction against p, not just its implementation.
        let q = MLXArray([Float(0.75), 0.25]).reshaped(1, 1, 2)
        let indices = MLXArray([Int32(0), 1]).reshaped(1, 1, 2)
        let p = MLXArray([Float(0.15), 0.25, 0.60, 0.15, 0.25, 0.60])
            .reshaped(1, 2, 3)
        var counts = [0, 0, 0]
        let trials = 4000
        for _ in 0 ..< trials {
            let draft = DFlash2Sampling.sample(probabilities: q).asType(.int32)
            let decision = DFlash2Sampling.sampledAcceptance(
                draftTokens: draft, targetProbabilities: p,
                draftProbabilities: q, draftIndices: indices)
            let result = concatenated([draft.reshaped(-1), decision]).asArray(Int32.self)
            let emitted = result[1] == 1 ? result[0] : result[2]
            XCTAssertTrue((0 ... 2).contains(emitted))
            counts[Int(emitted)] += 1
        }
        for (token, expected) in [0.15, 0.25, 0.60].enumerated() {
            XCTAssertEqual(Double(counts[token]) / Double(trials), expected, accuracy: 0.035)
        }
    }

    func testSeededDecisionMatchesExistingRejectionSampler() {
        let draft = MLXArray([Int32(0), 1, 0, 1]).reshaped(1, 4)
        let p = broadcast(MLXArray([Float(0.4), 0.3, 0.2, 0.1]), to: [1, 5, 4])
        let q = broadcast(MLXArray([Float(0.2), 0.8]), to: [1, 4, 2])
        let indices = broadcast(MLXArray([Int32(0), 1]), to: [1, 4, 2])
        for seed in UInt64(0) ..< 128 {
            MLXRandom.seed(seed)
            let expected = DFlash2Sampling.acceptSampled(
                draftTokens: draft, targetProbabilities: p,
                draftProbabilities: q, draftIndices: indices)
            MLXRandom.seed(seed)
            let actual = deviceAcceptance(
                draftTokens: draft, targetProbabilities: p,
                draftProbabilities: q, draftIndices: indices)
            XCTAssertEqual(actual.accepted, expected.accepted, "seed=\(seed)")
            XCTAssertEqual(actual.bonus, expected.bonus, "seed=\(seed)")
        }
    }
}
