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
        // Qwen 3.x primes with a trailing newline; the strict suffix check
        // classified that as un-primed and left the ceiling silently inert.
        XCTAssertTrue(ReasoningBudget.promptTailOpensReasoning("…assistant\n<think>\n"))
        XCTAssertTrue(ReasoningBudget.promptTailOpensReasoning("<think>  \n"))
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

    func testMTPIsRefusedWhileARequestedBudgetIsPending() {
        var p = GenerateParameters(temperature: 0, topP: 1.0, topK: 0)
        XCTAssertTrue(p.isNativeMTPLosslessGreedyEligible)
        p.requestedReasoningBudgetTokens = 32
        XCTAssertFalse(
            p.isNativeMTPLosslessGreedyEligible,
            "The requested form arms at submit — after this check — so it must disqualify MTP here."
        )
    }

    // MARK: - Per-request arming (requestedReasoningBudgetTokens)

    /// Fixed-vocab stub whose think tags genuinely round-trip, unlike the
    /// random-vocab `TestTokenizer`.
    private struct ThinkVocabTokenizer: MLXLMCommon.Tokenizer {
        let vocab: [String: Int] = ["<think>": 10, "</think>": 11, "hello": 12]
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [12] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "hello" }
        func convertTokenToId(_ token: String) -> Int? { vocab[token] }
        func convertIdToToken(_ id: Int) -> String? {
            vocab.first { $0.value == id }?.key
        }
        var bosToken: String? { nil }
        var eosToken: String? { "</s>" }
        var unknownToken: String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [12] }
    }

    func testRequestedCountArmsOnAPrimedTail() {
        let armed = ReasoningBudget.arm(
            tokenizer: ThinkVocabTokenizer(),
            promptTail: "…assistant<think>",
            tokenCount: 96
        )
        XCTAssertEqual(armed?.tokenCount, 96)
        XCTAssertEqual(armed?.closeTokenID, 11)
        XCTAssertEqual(
            armed?.startTokenIDs, [],
            "A primed tail counts immediately — waiting for an open tag that was already rendered would leave the ceiling inert."
        )
        XCTAssertEqual(armed?.openTokenIDs, [10])
    }

    func testRequestedCountArmsForASelfOpeningFamily() {
        let armed = ReasoningBudget.arm(
            tokenizer: ThinkVocabTokenizer(),
            promptTail: "…assistant\n",
            tokenCount: 64
        )
        XCTAssertEqual(
            armed?.startTokenIDs, [10],
            "Un-primed families start the count only when the model opens reasoning itself."
        )
    }

    func testANonPositiveRequestedCountStaysInert() {
        XCTAssertNil(
            ReasoningBudget.arm(
                tokenizer: ThinkVocabTokenizer(), promptTail: "<think>", tokenCount: 0))
        XCTAssertNil(
            ReasoningBudget.arm(
                tokenizer: ThinkVocabTokenizer(), promptTail: "<think>", tokenCount: -5))
    }

    func testEnvArmedPathStillDelegatesToTheSameResolution() {
        // Without VMLX_REASONING_BUDGET in the environment the legacy entry
        // point must stay nil even though the vocab could arm.
        XCTAssertNil(
            ReasoningBudget.armIfNeeded(
                tokenizer: ThinkVocabTokenizer(), promptTail: "<think>"))
    }
}
