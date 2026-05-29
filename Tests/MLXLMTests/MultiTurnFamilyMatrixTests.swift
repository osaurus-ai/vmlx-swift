// Copyright © 2026 osaurus.
//
// Multi-turn × cross-family contract matrix. Locks the per-family
// dispatch + parser invariants every shipping bundle relies on:
//
//   1. Reasoning stamp resolution           — every family × case/suffix variant
//   2. Tool call format dispatch            — every family × case/suffix variant
//   3. Multi-turn parser state isolation    — turn-N parser doesn't bleed into N+1
//   4. Reasoning ON → OFF → ON toggling     — chat client can flip per-turn
//   5. Empty / truncated reasoning per fam  — every think_xml family handles it
//   6. Cross-turn marker bleed              — `<think>` from prior assistant turn
//                                             does NOT trigger spurious events
//                                             when echoed back as history input
//
// Pure unit tests over the parser + factory APIs. Covers every LLM
// model_type registered in `LLMModelFactory` and every VLM model_type
// in `VLMModelFactory` plus the JANG-stamped capability shapes.
//
// Companion to the engine-side stamp / format suites:
//   - Mistral3LagunaCoverageTests       (Mistral3 + Laguna only)
//   - InterleavedReasoningLeakTests     (boundary leak audit)
//   - ReasoningStampMCDCTests           (MC/DC coverage)
// This file is the *family matrix* — one row per model_type, exhaustive
// across the dispatch surface so a routing regression in any one row
// can't slip through.

import Foundation
@testable import MLXLMCommon
import Testing

// =====================================================================
// MARK: - Section A — Reasoning stamp matrix (every family × variants)
// =====================================================================

@Suite("Reasoning stamp — full family × case/suffix matrix")
struct ReasoningStampFamilyMatrixTests {

    // think_xml families: native `<think>...</think>` chat templates.
    // Every prefix here MUST resolve to "think_xml" regardless of suffix
    // or case so a pin bump that adds e.g. `qwen3_7` doesn't silently
    // demote it to "none".
    @Test(arguments: [
        // Qwen 3 family
        "qwen3", "qwen3_5", "qwen3_5_moe", "qwen3_5_text",
        "qwen3_6", "qwen3_moe", "qwen3_next", "qwen3_next_moe", "qwen3_vl",
        // DeepSeek
        "deepseek", "deepseek_v3", "deepseek_v4", "deepseek_v4_flash", "deepseek_r1",
        // GLM-4 MoE (glm4 base is .glm4 dispatch; only glm4_moe* gets think_xml)
        "glm4_moe", "glm4_moe_lite", "glm5",
        // MiniMax — user's looping report
        "minimax", "minimax_m2", "minimax_m3",
        // Kimi K2 family
        "kimi", "kimi_k2", "kimi_k25", "kimi_k26",
        // Nemotron H
        "nemotron_h", "nemotron_h_omni",
        // Holo3
        "holo", "holo3",
        // Ling / Bailing
        "bailing_moe", "bailing_hybrid", "bailing_moe_v2_5",
        // Laguna (Poolside)
        "laguna", "laguna_xs", "laguna_s", "laguna_m",
        // ZAYA
        "zaya", "zaya1", "zaya2",
        // MiMo
        "mimo", "mimo_v2_flash",
    ])
    func thinkXmlFamilies(_ modelType: String) {
        #expect(reasoningStampFromModelType(modelType) == "think_xml",
            "\(modelType) must resolve to think_xml")
        #expect(reasoningStampFromModelType(modelType.uppercased()) == "think_xml",
            "\(modelType.uppercased()) must be case-insensitive")
        #expect(reasoningStampFromModelType("\(modelType)_v2") == "think_xml",
            "\(modelType)_v2 (suffix variant) must still match")
    }

    // harmony families: `<|channel>thought\n…<channel|>` (Gemma 4 style).
    @Test(arguments: ["gemma4", "gemma4_text", "gemma4_moe"])
    func harmonyFamilies(_ modelType: String) {
        #expect(reasoningStampFromModelType(modelType) == "harmony")
        #expect(reasoningStampFromModelType(modelType.uppercased()) == "harmony")
    }

    // none families: no reasoning side-channel — stream raw content only.
    // Critical that these stay "none" — adding them to think_xml would
    // make a non-reasoning model loop forever waiting for </think>.
    @Test(arguments: [
        "llama", "mistral", "mistral3", "mistral3_text", "ministral3", "mistral4",
        "phi", "phi3", "phimoe",
        "gemma", "gemma2", "gemma3", "gemma3_text", "gemma3n",
        "qwen2", "qwen2_vl", "qwen2_5_vl",
        "starcoder2", "cohere", "openelm", "internlm2",
        "granite", "granitemoehybrid",
        "bitnet", "falcon_h1",
        // VLM-only that don't route through think_xml
        "paligemma", "idefics3", "smolvlm", "fastvlm", "llava_qwen2",
        "pixtral", "lfm2_vl", "glm_ocr", "zaya1_vl",
        // LFM2/LFM2-MoE remain no-reasoning by model_type alone. LFM2.5
        // thinking is template-declared and covered by
        // ReasoningStampFromModelTypeTests.
        "lfm2", "lfm2_moe",
        // Empty/nil
        "", "unknown_model_type",
    ])
    func noneFamilies(_ modelType: String) {
        #expect(reasoningStampFromModelType(modelType) == "none",
            "\(modelType) must NOT route to a reasoning parser")
    }

    @Test func nilModelTypeIsNone() {
        #expect(reasoningStampFromModelType(nil) == "none")
    }

    /// Defensive-parsing semantics for whitespace in `model_type`: raw
    /// config values are trimmed before family matching so pasted leading
    /// or trailing whitespace does not silently demote a reasoning family
    /// to "none".
    @Test func whitespacedStampSemantics() {
        #expect(reasoningStampFromModelType(" qwen3") == "think_xml",
            "leading whitespace is trimmed before prefix match")
        #expect(reasoningStampFromModelType("qwen3 ") == "think_xml",
            "trailing whitespace is tolerated — still matches `qwen3` prefix")
        #expect(reasoningStampFromModelType("\nqwen3") == "think_xml",
            "leading newline is trimmed before prefix match")
    }
}

// =====================================================================
// MARK: - Section B — Tool format dispatch matrix
// =====================================================================

@Suite("Tool format dispatch — full family × case/suffix matrix")
struct ToolFormatFamilyMatrixTests {

    @Test(arguments: [
        // (modelType, expectedFormat)
        ("glm4", ToolCallFormat.glm4),
        ("glm4_moe", .glm4),
        ("glm4_moe_lite", .glm4),
        ("gemma3", .gemma),
        ("gemma3_text", .gemma),
        ("gemma", .gemma),
        ("gemma4", .gemma4),
        ("gemma4_text", .gemma4),
        ("minimax", .minimaxM2),
        ("minimax_m2", .minimaxM2),
        ("minimax_m3", .minimaxM2),
        ("nemotron_h", .nemotron),
        ("nemotron_h_omni", .nemotron),
        ("qwen3_5", .xmlFunction),
        ("qwen3_5_moe", .xmlFunction),
        ("qwen3_next", .xmlFunction),
        ("qwen3_next_moe", .xmlFunction),
        ("mistral3", .mistral),
        ("mistral3_text", .mistral),
        ("ministral3", .mistral),
        ("laguna", .glm4),
        ("laguna_xs", .glm4),
        ("kimi", .kimiK2),
        ("kimi_k2", .kimiK2),
        ("kimi_k25", .kimiK2),
    ])
    func formatPerFamily(_ pair: (String, ToolCallFormat)) {
        let (modelType, expected) = pair
        #expect(ToolCallFormat.infer(from: modelType) == expected,
            "\(modelType) must dispatch to \(expected.rawValue)")
        // Case-insensitive: same dispatch for upper-case input.
        #expect(ToolCallFormat.infer(from: modelType.uppercased()) == expected,
            "\(modelType.uppercased()) must dispatch identically")
    }

    /// Every shipping format MUST round-trip through the parser
    /// factory without crashing — `createParser()` is exhaustive over
    /// the enum, but if a new case is added without a matching arm,
    /// this test catches it via `allCases`.
    @Test func everyFormatHasParser() {
        for fmt in ToolCallFormat.allCases {
            _ = fmt.createParser()
        }
    }

    @Test func unknownModelTypeReturnsNil() {
        #expect(ToolCallFormat.infer(from: "totally_unknown_v999") == nil)
        #expect(ToolCallFormat.infer(from: "") == nil)
    }
}

// =====================================================================
// MARK: - Section C — Multi-turn parser state isolation
// =====================================================================

@Suite("Multi-turn — parser state isolated across turns")
struct MultiTurnParserStateIsolationTests {

    /// Helper: drive a fresh parser for one turn with the supplied
    /// token stream.
    private func runTurn(
        startInReasoning: Bool = false,
        tokens: [String]
    ) -> (content: String, reasoning: String) {
        var parser = ReasoningParser(startInReasoning: startInReasoning)
        var c = ""
        var r = ""
        for t in tokens {
            for seg in parser.feed(t) {
                switch seg {
                case .content(let s): c += s
                case .reasoning(let s): r += s
                }
            }
        }
        for seg in parser.flush() {
            switch seg {
            case .content(let s): c += s
            case .reasoning(let s): r += s
            }
        }
        return (c, r)
    }

    /// Three back-to-back turns: each instantiates a fresh parser.
    /// No turn's state must leak into another turn's output.
    @Test func threeTurnsAllReasoningClean() {
        let t1 = runTurn(tokens: ["<think>", "r1", "</think>", "c1"])
        let t2 = runTurn(tokens: ["<think>", "r2", "</think>", "c2"])
        let t3 = runTurn(tokens: ["<think>", "r3", "</think>", "c3"])
        #expect(t1.reasoning == "r1" && t1.content == "c1")
        #expect(t2.reasoning == "r2" && t2.content == "c2")
        #expect(t3.reasoning == "r3" && t3.content == "c3")
    }

    /// Mixed: turn 1 reasoning, turn 2 plain, turn 3 reasoning. No
    /// open `<think>` from t1 must put t2 into reasoning mode.
    @Test func mixedReasoningPlainReasoning() {
        let t1 = runTurn(tokens: ["<think>", "r1", "</think>", "c1"])
        let t2 = runTurn(tokens: ["c2 only no markers"])
        let t3 = runTurn(tokens: ["<think>", "r3", "</think>", "c3"])
        #expect(t1.reasoning == "r1")
        #expect(t2.reasoning.isEmpty, "plain turn must not synthesise reasoning")
        #expect(t2.content == "c2 only no markers")
        #expect(t3.reasoning == "r3")
    }

    /// Turn 1 with mid-stream truncation (max_tokens hit inside the
    /// `<think>` block — never emits the closer). The leftover buffered
    /// tokens must flush as `.reasoning`. Turn 2 is plain content;
    /// without correct flushing on turn 1, leftover state could splice
    /// reasoning markers into turn 2's content.
    @Test func midReasoningTruncationFlushesAsReasoning() {
        let t1 = runTurn(tokens: ["<think>", "still ", "thinking ", "more"])
        // Truncated mid-think: every token after the opener was
        // reasoning. Flushed buffer should land as reasoning, not content.
        #expect(t1.reasoning.contains("still"))
        #expect(t1.reasoning.contains("thinking"))
        #expect(t1.content.isEmpty,
            "no content tokens were emitted, content must be empty")
    }
}

// =====================================================================
// MARK: - Section D — Reasoning ON → OFF → ON toggling
// =====================================================================

@Suite("Reasoning ON/OFF — per-turn toggle via startInReasoning")
struct ReasoningToggleTests {

    private func runTurn(
        startInReasoning: Bool,
        tokens: [String]
    ) -> (content: String, reasoning: String) {
        var p = ReasoningParser(startInReasoning: startInReasoning)
        var c = ""
        var r = ""
        for t in tokens {
            for seg in p.feed(t) {
                switch seg {
                case .content(let s): c += s
                case .reasoning(let s): r += s
                }
            }
        }
        for seg in p.flush() {
            switch seg {
            case .content(let s): c += s
            case .reasoning(let s): r += s
            }
        }
        return (c, r)
    }

    /// Turn 1 with reasoning ON (template prefilled `<think>`):
    /// parser starts in reasoning, model emits thought, then closer,
    /// then content.
    @Test func turn1_startInReasoning_emitsReasoningThenContent() {
        let t1 = runTurn(
            startInReasoning: true,
            tokens: ["thought1 ", "</think>", "answer1"])
        #expect(t1.reasoning.contains("thought1"))
        #expect(t1.content.contains("answer1"))
        #expect(!t1.content.contains("</think>"))
    }

    /// Turn 2 with reasoning OFF (template did NOT prefill `<think>`):
    /// parser starts in content, no `<think>` opener arrives — pure
    /// content stream.
    @Test func turn2_startInContent_pureContent() {
        let t2 = runTurn(
            startInReasoning: false,
            tokens: ["just content here"])
        #expect(t2.content == "just content here")
        #expect(t2.reasoning.isEmpty)
    }

    /// Turn 3: reasoning ON again; the parser must not carry any state
    /// from turn 1 (since each turn instantiates fresh). Content from
    /// turn 1 must NOT leak into turn 3.
    @Test func turn3_startInReasoning_clean() {
        // Simulate three full turns; each turn is its own parser.
        let t1 = runTurn(startInReasoning: true,
            tokens: ["thought1 ", "</think>", "answer1"])
        let t2 = runTurn(startInReasoning: false,
            tokens: ["pure content"])
        let t3 = runTurn(startInReasoning: true,
            tokens: ["thought3 ", "</think>", "answer3"])
        #expect(!t3.reasoning.contains("thought1"))
        #expect(!t3.reasoning.contains("pure content"))
        #expect(t3.reasoning.contains("thought3"))
        #expect(t3.content.contains("answer3"))
        // And t1's state hasn't been mutated by t3.
        #expect(t1.content.contains("answer1"))
        #expect(t2.content == "pure content")
    }

    /// Edge: prompt template prefilled `<think>` but the model has
    /// nothing to think about and immediately emits `</think>` — must
    /// produce zero reasoning bytes and full content.
    @Test func startInReasoning_immediateClose_noReasoningBytes() {
        let t = runTurn(
            startInReasoning: true,
            tokens: ["</think>", "answer"])
        #expect(t.reasoning.isEmpty)
        #expect(t.content == "answer")
    }
}

// =====================================================================
// MARK: - Section E — forPrompt auto-detection (multi-turn safety)
// =====================================================================

@Suite("ReasoningParser.forPrompt — turn-end auto-detection")
struct ForPromptAutoDetectionTests {

    /// Tail of the prompt has no think markers → parser starts in content.
    /// Models the case where the chat history ends with a user turn (no
    /// open `<think>`), so the next assistant turn begins outside of
    /// reasoning.
    @Test func tailWithoutThinkTags_startsInContent() {
        var parser = ReasoningParser.forPrompt(
            stampName: "think_xml",
            promptTail: "user: hello\nassistant: hi\nuser: what's up?")
        #expect(parser != nil, "think_xml stamp must produce a parser")
        // Drive without an opener: pure content.
        var c = ""
        var r = ""
        for seg in parser!.feed("plain content") {
            if case .content(let s) = seg { c += s }
            if case .reasoning(let s) = seg { r += s }
        }
        for seg in parser!.flush() {
            if case .content(let s) = seg { c += s }
            if case .reasoning(let s) = seg { r += s }
        }
        #expect(c == "plain content")
        #expect(r.isEmpty)
    }

    /// Tail of the prompt has an open `<think>` → parser starts in
    /// reasoning. Models the Qwen template that prefills `<think>` at
    /// the start of every assistant turn.
    @Test func tailWithOpenThink_startsInReasoning() {
        var parser = ReasoningParser.forPrompt(
            stampName: "think_xml",
            promptTail: "<|im_start|>assistant\n<think>\n")
        #expect(parser != nil)
        var r = ""
        for seg in parser!.feed("internal thought") {
            if case .reasoning(let s) = seg { r += s }
        }
        for seg in parser!.flush() {
            if case .reasoning(let s) = seg { r += s }
        }
        #expect(r.contains("internal thought"))
    }

    /// Tail of the prompt has a CLOSED `<think>...</think>` → parser
    /// starts in content (the prefilled marker is from the system or
    /// a prior turn that already closed). Locks the B1 edge case:
    /// `</think>` in last 64 tokens → parser opens in content mode.
    @Test func tailWithClosedThink_startsInContent() {
        var parser = ReasoningParser.forPrompt(
            stampName: "think_xml",
            promptTail: "<think>old thought</think>\nassistant: ")
        #expect(parser != nil)
        var c = ""
        var r = ""
        for seg in parser!.feed("answer") {
            if case .content(let s) = seg { c += s }
            if case .reasoning(let s) = seg { r += s }
        }
        for seg in parser!.flush() {
            if case .content(let s) = seg { c += s }
            if case .reasoning(let s) = seg { r += s }
        }
        #expect(c == "answer")
        #expect(r.isEmpty,
            "closed think in tail means assistant starts in content mode")
    }

    /// `none` stamp returns nil parser regardless of tail content —
    /// non-reasoning models always stream raw, even if the user pastes
    /// `<think>` into the prompt.
    @Test func noneStampAlwaysReturnsNil() {
        #expect(ReasoningParser.forPrompt(stampName: "none", promptTail: "") == nil)
        #expect(ReasoningParser.forPrompt(stampName: "none",
            promptTail: "<think>weird</think>") == nil)
    }

    /// Cross-turn marker bleed: a prior assistant turn's `<think>`
    /// echoed back as part of history (chat templates render assistant
    /// turn content into the prompt) must NOT cause the next turn to
    /// start inside reasoning unless the literal LAST think tag is
    /// `<think>` without a closer.
    @Test func multiTurnHistoryEcho_noFalseReasoningStart() {
        // History: turn 1 had reasoning, turn 1 closed it, turn 2 is
        // about to start. The prompt tail contains `</think>` last.
        var parser = ReasoningParser.forPrompt(
            stampName: "think_xml",
            promptTail: "...assistant: <think>r1</think>answer1\nuser: q2\nassistant: ")
        #expect(parser != nil)
        var c = ""
        for seg in parser!.feed("turn-2 answer") {
            if case .content(let s) = seg { c += s }
        }
        for seg in parser!.flush() {
            if case .content(let s) = seg { c += s }
        }
        #expect(c == "turn-2 answer",
            "closed-think history must not put turn 2 into reasoning")
    }
}

// =====================================================================
// MARK: - Section F — Per-family multi-turn reasoning streams
// =====================================================================

@Suite("Per-family — multi-turn reasoning + content streams")
struct PerFamilyMultiTurnReasoningTests {

    /// Drive 3 turns of (reasoning ON → reasoning OFF → reasoning ON)
    /// for every think_xml family. Each turn re-resolves the parser
    /// from `forPrompt(stampName:promptTail:)` so the test exercises
    /// the *real* osaurus-side resolution path, not just direct
    /// constructor calls.
    @Test(arguments: [
        "qwen3", "qwen3_5", "qwen3_5_moe", "qwen3_6", "qwen3_moe",
        "qwen3_next", "qwen3_next_moe",
        "deepseek_v3", "deepseek_v4", "deepseek_v4_flash",
        "glm4_moe", "glm4_moe_lite",
        "minimax", "minimax_m2", "minimax_m3",
        "kimi_k2", "kimi_k25",
        "nemotron_h", "nemotron_h_omni",
        "holo3",
        "laguna", "laguna_xs",
        "zaya", "zaya1",
    ])
    func threeTurnReasoningToggle(_ modelType: String) {
        let stamp = reasoningStampFromModelType(modelType)
        #expect(stamp == "think_xml",
            "preflight: \(modelType) must be think_xml for this test")

        // Turn 1 — reasoning ON. Prompt tail mimics chat template
        // prefilling `<think>` immediately before the assistant turn.
        var p1 = ReasoningParser.forPrompt(
            stampName: stamp,
            promptTail: "user: q1\nassistant: <think>\n")
        #expect(p1 != nil)
        var r1 = ""
        var c1 = ""
        for seg in p1!.feed("inner1 ") {
            if case .reasoning(let s) = seg { r1 += s }
            if case .content(let s) = seg { c1 += s }
        }
        for seg in p1!.feed("</think>answer1") {
            if case .reasoning(let s) = seg { r1 += s }
            if case .content(let s) = seg { c1 += s }
        }
        for seg in p1!.flush() {
            if case .content(let s) = seg { c1 += s }
        }
        #expect(r1.contains("inner1"), "[\(modelType)] turn 1 reasoning")
        #expect(c1.contains("answer1"), "[\(modelType)] turn 1 content")

        // Turn 2 — reasoning OFF. The chat client toggled enable_thinking
        // to false; chat template did NOT prefill `<think>`. Prompt tail
        // contains the closed-think marker from turn 1 history.
        var p2 = ReasoningParser.forPrompt(
            stampName: stamp,
            promptTail: "<think>inner1</think>answer1\nuser: q2\nassistant: ")
        #expect(p2 != nil)
        var c2 = ""
        for seg in p2!.feed("just answer2") {
            if case .content(let s) = seg { c2 += s }
        }
        for seg in p2!.flush() {
            if case .content(let s) = seg { c2 += s }
        }
        #expect(c2 == "just answer2",
            "[\(modelType)] turn 2 (reasoning OFF) must not synthesise reasoning")

        // Turn 3 — reasoning ON again. Re-resolve parser for the new
        // turn; prompt tail again includes a fresh prefilled `<think>`.
        var p3 = ReasoningParser.forPrompt(
            stampName: stamp,
            promptTail: "user: q3\nassistant: <think>\n")
        #expect(p3 != nil)
        var r3 = ""
        var c3 = ""
        for seg in p3!.feed("inner3</think>answer3") {
            if case .reasoning(let s) = seg { r3 += s }
            if case .content(let s) = seg { c3 += s }
        }
        for seg in p3!.flush() {
            if case .content(let s) = seg { c3 += s }
        }
        #expect(r3.contains("inner3"), "[\(modelType)] turn 3 reasoning")
        #expect(c3.contains("answer3"), "[\(modelType)] turn 3 content")
        // Cross-turn isolation: turn 1's reasoning bytes must NOT
        // appear in turn 3's reasoning bytes.
        #expect(!r3.contains("inner1"),
            "[\(modelType)] turn 1 reasoning must not bleed into turn 3")
    }

    /// Non-reasoning families with the same multi-turn shape: every
    /// turn must stream raw content, never producing `.reasoning`
    /// segments even if `<think>` happens to appear in the input
    /// (e.g. user typed `<think>` literally).
    @Test(arguments: [
        "llama", "mistral", "mistral3", "ministral3", "mistral4",
        "phi3", "phimoe", "gemma2", "gemma3",
        "qwen2", "qwen2_5_vl",
        "starcoder2", "cohere",
    ])
    func nonReasoningFamiliesNeverProduceReasoningEvents(_ modelType: String) {
        let stamp = reasoningStampFromModelType(modelType)
        #expect(stamp == "none", "preflight: \(modelType) is non-reasoning")
        // forPrompt with stamp=none returns nil — caller streams raw.
        #expect(ReasoningParser.forPrompt(stampName: stamp, promptTail: "") == nil)
        #expect(
            ReasoningParser.forPrompt(stampName: stamp,
                promptTail: "<think>weird user input</think>") == nil,
            "[\(modelType)] stray <think> in prompt must NOT engage parser")
    }
}

// =====================================================================
// MARK: - Section G — Tool format multi-turn dispatch consistency
// =====================================================================

@Suite("Tool format — multi-turn dispatch is stable")
struct ToolFormatMultiTurnConsistencyTests {

    /// `infer(from:)` is a pure function — must return identical
    /// dispatch on repeated invocation across turns. Locks against a
    /// future regression that introduces hidden state in the resolver.
    @Test(arguments: [
        "qwen3_5_moe", "minimax_m2", "kimi_k25", "nemotron_h_omni",
        "deepseek_v4_flash", "glm4_moe", "mistral3", "ministral3",
        "laguna", "gemma4_text",
    ])
    func dispatchStableAcrossInvocations(_ modelType: String) {
        let f1 = ToolCallFormat.infer(from: modelType)
        let f2 = ToolCallFormat.infer(from: modelType)
        let f3 = ToolCallFormat.infer(from: modelType.uppercased())
        #expect(f1 == f2, "[\(modelType)] dispatch must be deterministic")
        #expect(f1 == f3, "[\(modelType)] dispatch must be case-insensitive")
    }
}
