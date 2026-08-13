import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// Covers the explicit reasoning ceiling added for the DSV4 Flash runaway:
/// a spec-heavy prompt on the Low rail produced a 38,494-char think block over
/// 10,554 tokens, hit `stop=length`, and returned no usable answer.
///
/// The properties that matter are (a) it is inert unless asked for, and
/// (b) while budget remains it does not touch the distribution at all — those
/// are what separate it from the automatic close bias that
/// `NoHiddenReasoningCloseBiasFocusedTests` bans.
class ReasoningBudgetTests: XCTestCase {

    private func logits(_ values: [Float]) -> MLXArray {
        MLXArray(values).reshaped(1, values.count)
    }

    // MARK: - Inert by default

    func testParametersDoNotBuildAProcessorWithoutABudget() {
        var p = GenerateParameters()
        p.reasoningBudgetCloseTokenID = 7
        XCTAssertNil(p.reasoningBudgetTokens)
        XCTAssertNil(p.processor(), "No budget means no processor at all.")
    }

    func testParametersDoNotBuildAProcessorWithoutACloseToken() {
        var p = GenerateParameters()
        p.reasoningBudgetTokens = 4
        XCTAssertNil(p.processor(), "A budget without a resolved close token stays inert.")
    }

    func testAZeroBudgetIsInert() {
        var p = GenerateParameters()
        p.reasoningBudgetTokens = 0
        p.reasoningBudgetCloseTokenID = 3
        XCTAssertNil(p.processor())
    }

    // MARK: - Pass-through while budget remains

    func testLogitsAreUntouchedWhileBudgetRemains() {
        var proc = ReasoningBudgetProcessor(closeTokenID: 2, tokenCount: 3)
        let input = logits([1.0, 5.0, 0.5, -2.0])
        for step in 0 ..< 3 {
            let out = proc.process(logits: input)
            XCTAssertEqual(
                out.reshaped(-1).asArray(Float.self), [1.0, 5.0, 0.5, -2.0],
                "Step \(step) must pass logits through unchanged.")
            XCTAssertFalse(proc.didForceClose)
            proc.didSample(token: MLXArray(Int32(1)))
        }
    }

    // MARK: - Requires the close exactly once, then disarms

    func testCloseIsRequiredOnceTheBudgetIsSpent() {
        var proc = ReasoningBudgetProcessor(closeTokenID: 2, tokenCount: 1)
        proc.didSample(token: MLXArray(Int32(9)))  // spends the single token

        let out = proc.process(logits: logits([1.0, 5.0, 0.5, -2.0]))
        let vals = out.reshaped(-1).asArray(Float.self)
        XCTAssertEqual(vals[2], 0.5, "The close token keeps its own logit.")
        for i in [0, 1, 3] {
            XCTAssertTrue(vals[i] == -Float.infinity, "Token \(i) must be masked.")
        }
        XCTAssertEqual(
            vals.firstIndex(where: { $0 > -Float.infinity }), 2,
            "Only the close token may survive, so argmax must pick it.")
    }

    func testBudgetDisarmsAfterForcingSoTheAnswerSamplesNormally() {
        var proc = ReasoningBudgetProcessor(closeTokenID: 2, tokenCount: 1)
        proc.didSample(token: MLXArray(Int32(9)))
        _ = proc.process(logits: logits([1.0, 5.0, 0.5, -2.0]))
        proc.didSample(token: MLXArray(Int32(2)))  // the forced close
        XCTAssertTrue(proc.didForceClose)

        let after = proc.process(logits: logits([1.0, 5.0, 0.5, -2.0]))
        XCTAssertEqual(
            after.reshaped(-1).asArray(Float.self), [1.0, 5.0, 0.5, -2.0],
            "Once spent, the answer and any tool call sample untouched.")
    }

    func testAnOutOfRangeCloseTokenIsIgnoredRatherThanCorrupting() {
        var proc = ReasoningBudgetProcessor(closeTokenID: 99, tokenCount: 1)
        proc.didSample(token: MLXArray(Int32(0)))
        let out = proc.process(logits: logits([1.0, 5.0, 0.5, -2.0]))
        XCTAssertEqual(
            out.reshaped(-1).asArray(Float.self), [1.0, 5.0, 0.5, -2.0],
            "A close id outside the vocab must not blank the distribution.")
    }

    // MARK: - Arming conditions

    func testOnlyAnOpenReasoningTailArms() {
        XCTAssertTrue(ReasoningBudget.promptTailOpensReasoning("…assistant<think>"))
        XCTAssertTrue(ReasoningBudget.promptTailOpensReasoning("<thinking>"))
        XCTAssertFalse(
            ReasoningBudget.promptTailOpensReasoning("…</think>"),
            "The chat rail (already-closed tail) must never arm a budget.")
        XCTAssertFalse(ReasoningBudget.promptTailOpensReasoning("plain user text"))
    }

    func testMTPIsRefusedWhileABudgetIsActive() {
        var p = GenerateParameters(temperature: 0, topP: 1.0, topK: 0)
        XCTAssertTrue(p.isNativeMTPLosslessGreedyEligible)
        p.reasoningBudgetTokens = 32
        XCTAssertFalse(
            p.isNativeMTPLosslessGreedyEligible,
            "A drafted token could sail past the ceiling unchecked.")
    }
}
