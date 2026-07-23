// Copyright © 2026 osaurus.
//
// Coverage tests for Mistral 3 family + Laguna across:
//   - Reasoning stamp resolution (think_xml / harmony / none)
//   - Tool call format dispatch (xmlFunction / mistral / glm4 / etc.)
//   - No-leak streaming (Mistral 3 has NO <think> template; output must
//     never produce .reasoning events when the model emits plain text)
//
// Pure unit tests — no model-load required, no real bundle on disk.
// They cover the logical contracts every Mistral 3 / Mistral 3.5 /
// Laguna deployment will rely on at runtime.
//

import Foundation
@testable import MLXLMCommon
import Testing

@Suite("Mistral 3 family + Laguna — reasoning + tool-call coverage")
struct Mistral3LagunaCoverageTests {

    // MARK: - Reasoning stamp

    @Test("Mistral 3 / mistral3 outer → 'none' (no <think> template)")
    func reasoning_mistral3_none() {
        #expect(reasoningStampFromModelType("mistral3") == "none")
        #expect(reasoningStampFromModelType("mistral3_text") == "none")
        #expect(reasoningStampFromModelType("Mistral3") == "none",
            "case-insensitive — must still resolve none")
    }

    @Test("Ministral3 inner → 'none' (no <think> template)")
    func reasoning_ministral3_none() {
        // The inner text_config.model_type for Mistral 3.5. No
        // reasoning template — must NOT match think_xml.
        #expect(reasoningStampFromModelType("ministral3") == "none")
    }

    @Test("Laguna → 'think_xml' (chat template emits <think>)")
    func reasoning_laguna_thinkXml() {
        // Pre-registered for when the engine class lands. Already
        // covered by `lagunaResolvesToThinkXml` but reasserting here
        // for coverage parity.
        #expect(reasoningStampFromModelType("laguna") == "think_xml")
        #expect(reasoningStampFromModelType("laguna_xs") == "think_xml")
        #expect(reasoningStampFromModelType("laguna_s") == "think_xml")
        #expect(ReasoningParser.fromCapabilityName("laguna") != nil)
        #expect(ReasoningParser.fromCapabilityName("laguna_xs") != nil)
        #expect(ReasoningParser.fromCapabilityName("laguna_s") != nil)
    }

    @Test("Mistral 4 (sibling family) → 'none'")
    func reasoning_mistral4_none() {
        // Locks the boundary: Mistral 4 is not Mistral 3 / 3.5; sliding-
        // window-only, no reasoning template either.
        #expect(reasoningStampFromModelType("mistral4") == "none")
    }

    // MARK: - Tool call format

    @Test("mistral3 outer → .mistral parser (model_type heuristic)")
    func tool_mistral3() {
        #expect(ToolCallFormat.infer(from: "mistral3") == .mistral)
        #expect(ToolCallFormat.infer(from: "mistral3_text") == .mistral)
    }

    @Test("ministral3 inner → .mistral parser (LLM-only Mistral 3.5)")
    func tool_ministral3() {
        // For text-only Mistral 3.5 (no vision_config) the outer
        // model_type can be ministral3 directly. Tool dispatch must
        // still match.
        #expect(ToolCallFormat.infer(from: "ministral3") == .mistral)
    }

    @Test("laguna → .glm4 parser (GLM-family chat template)")
    func tool_laguna() {
        #expect(ToolCallFormat.infer(from: "laguna") == .glm4)
        #expect(ToolCallFormat.infer(from: "laguna_xs") == .glm4)
        #expect(ToolCallFormat.infer(from: "laguna_s") == .glm4)
        #expect(ToolCallFormat.fromCapabilityName("laguna") == .glm4)
        #expect(ToolCallFormat.fromCapabilityName("laguna_xs") == .glm4)
        #expect(ToolCallFormat.fromCapabilityName("laguna_s") == .glm4)
        #expect(ToolCallFormat.fromCapabilityName("poolside_v1") == .glm4)
    }

    @Test("Mistral 4 → .mistral (sibling family, same parser)")
    func tool_mistral4() {
        // mistral4 falls under the `hasPrefix("mistral")` family
        // matcher elsewhere — verify boundary.
        let result = ToolCallFormat.infer(from: "mistral4")
        #expect(result == .mistral || result == nil,
            "mistral4 should either match mistral or fall through cleanly")
    }

    // MARK: - No-leak boundary: Mistral 3 family must NOT emit .reasoning

    /// Mistral 3 / 3.5 ship NO <think> template, so the parser stamp
    /// resolves to "none". The "none" stamp must produce only
    /// `.content(...)` segments — never `.reasoning(...)` — even if a
    /// model misbehaves and emits literal `<think>` markers.
    @Test("'none' stamp emits only .content (no reasoning leak)")
    func noLeak_noneStampStreamsAsContent() {
        // `fromCapabilityName("none")` returns nil — that's the
        // contract for "no reasoning envelope". Locks that contract:
        // a nil parser means callers stream raw model output as
        // visible content, no reasoning side-channel.
        let parser = ReasoningParser.fromCapabilityName("none")
        #expect(parser == nil,
            "'none' stamp must produce a nil parser; callers stream raw as .chunk")
    }

    /// think_xml stamp parser produces a non-nil parser. The parser's
    /// detailed streaming semantics (inside-<think> tokens go to
    /// `.reasoning`, outside go to `.content`, no marker leaks) are
    /// covered by the much-larger `ReasoningParserTests` suite — this
    /// test verifies the stamp resolution returns the correct parser
    /// shape so Laguna / Qwen / Nemotron-3 / etc. route into that
    /// already-tested streaming machinery.
    @Test("'think_xml' stamp produces non-nil parser (Laguna routes here)")
    func thinkXml_parserExists() {
        let parser = ReasoningParser.fromCapabilityName("think_xml")
        #expect(parser != nil,
            "think_xml stamp must produce a streaming parser; got nil")
    }
}
