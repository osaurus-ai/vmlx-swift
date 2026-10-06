import MLX
import XCTest

@testable import MLXLMCommon

/// Exact rejection-sampling boundaries, using the actual accept/correct owner.
/// Tiny probability arrays only; no model or generated-text evidence.
final class SpeculativeAcceptanceBoundaryTests: XCTestCase {
    private func decision(
        target: [Float], draft: [Float], roll: Float, draws: inout Int
    ) -> SpeculativeSamplingController.AcceptanceDecision {
        let controller = SpeculativeSamplingController(
            parameters: GenerateParameters(temperature: 1, randomSeed: 13))
        return controller.acceptOrCorrect(
            draftToken: MLXArray([UInt32(0)]),
            targetProbabilities: MLXArray(target)[.newAxis, .ellipsis],
            draftProbabilities: MLXArray(draft)[.newAxis, .ellipsis],
            acceptanceRoll: {
                draws += 1
                return roll
            })
    }

    func testZeroProbabilityRejectsZeroDrawAndUsesResidual() {
        var draws = 0
        let result = decision(target: [0, 1], draft: [1, 0], roll: 0, draws: &draws)
        XCTAssertEqual(draws, 1, "zero acceptance still consumes the original RNG draw")
        XCTAssertEqual(result.acceptanceProbability, 0)
        XCTAssertFalse(result.accepted)
        XCTAssertEqual(
            result.correction?.item(Int.self), 1,
            "target-impossible proposal0 must be replaced by residual token1")
    }

    func testEqualityAtInteriorThresholdRejects() {
        var draws = 0
        let result = decision(
            target: [0.25, 0.75], draft: [0.5, 0.5],
            roll: 0.5, draws: &draws)
        XCTAssertEqual(draws, 1)
        XCTAssertEqual(result.acceptanceProbability, 0.5)
        XCTAssertFalse(result.accepted)
        XCTAssertEqual(result.correction?.item(Int.self), 1)
    }

    func testImmediatelyBelowThresholdAccepts() {
        var draws = 0
        let result = decision(
            target: [0.25, 0.75], draft: [0.5, 0.5],
            roll: Float(0.5).nextDown, draws: &draws)
        XCTAssertEqual(draws, 1)
        XCTAssertEqual(result.acceptanceProbability, 0.5)
        XCTAssertTrue(result.accepted)
        XCTAssertNil(result.correction)
    }

    func testImmediatelyAboveThresholdRejects() {
        var draws = 0
        let result = decision(
            target: [0.25, 0.75], draft: [0.5, 0.5],
            roll: Float(0.5).nextUp, draws: &draws)
        XCTAssertEqual(draws, 1)
        XCTAssertFalse(result.accepted)
        XCTAssertEqual(result.correction?.item(Int.self), 1)
    }

    func testCertainAcceptanceDoesNotConsumeAcceptanceDraw() {
        let targets: [[Float]] = [[0.5, 0.5], [0.75, 0.25]]
        for target in targets {
            var draws = 0
            let result = decision(target: target, draft: [0.5, 0.5], roll: 0, draws: &draws)
            XCTAssertEqual(draws, 0, "alpha>=1 retains the original no-draw fast path")
            XCTAssertEqual(result.acceptanceProbability, 1)
            XCTAssertTrue(result.accepted)
            XCTAssertNil(result.correction)
        }
    }
}
