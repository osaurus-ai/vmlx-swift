// Pin the Hy3 / Tencent Hunyuan v3 reasoning-no-leak contract.
//
// JANG handoff doc (`docs/runtime/2026-05-09-hy3-runtime-handoff-vmlx-python-swift.md`)
// mandates:
//
//   "ensure `no_think` emits closed `<think:opensource></think:opensource>` prefill and does
//    not leak reasoning into content"
//
// The Swift wiring is:
//   - JANG capability stamp `reasoning_parser=hunyuan|tencent|hy3|hy_v3|hy-v3`
//     resolves to the same `think_xml` parser as Qwen3 (per
//     `ReasoningParser.fromCapabilityName` line 295-303).
//   - `ReasoningParser.forPrompt(stampName:promptTail:)` then inspects the
//     decoded prompt tail and overrides `startInReasoning` based on which
//     of `<think:opensource>` / `</think:opensource>` appears LAST in the prompt.
//   - `Evaluate.swift:2033,2145` and `BatchEngine.swift:410` both call
//     `forPrompt` so the live request path uses the prompt-aware parser.
//
// Existing tests cover DSV4 + qwen3_6 prompt-tail patterns. Hy3 was NOT
// covered. This file pins three Hy3-specific contracts:
//
//   1. `reasoning_effort=no_think` (closed `<think:opensource>\n\n</think:opensource>\n\n` in
//       prompt) → parser starts in CONTENT and does NOT route output
//       into `.reasoning`.
//   2. `reasoning_effort=high|low` (open `<think:opensource>\n` in prompt) → parser
//      starts in REASONING and routes pre-`</think:opensource>` output into
//      `.reasoning`, post-`</think:opensource>` into `.content`.
//   3. Mid-stream stray `<think:opensource>` tag from a misbehaving model is
//      latched correctly — content collected before the tag stays in
//      `.content`, reasoning after stays in `.reasoning`.
//
// Source-coverage style — no MLX runtime needed.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Hy3 reasoning-parser no-leak contract (think_xml family)")
struct Hy3ReasoningNoLeakTests {

    /// Drains a parser into `(content, reasoning)` strings.
    private static func drain(
        _ parser: inout ReasoningParser, _ feed: String
    ) -> (content: String, reasoning: String) {
        var content = ""
        var reasoning = ""
        for segment in parser.feed(feed) {
            switch segment {
            case .content(let text): content += text
            case .reasoning(let text): reasoning += text
            }
        }
        for segment in parser.flush() {
            switch segment {
            case .content(let text): content += text
            case .reasoning(let text): reasoning += text
            }
        }
        return (content, reasoning)
    }

    @Test("Hy3 stamps resolve through fromCapabilityName")
    func capabilityResolves() throws {
        // All five Hy3 stamps must produce a parser. Hy3ParserDispatchTests
        // already covers != nil; this pins the stamp's policy details.
        for stamp in ["hy3", "hy_v3", "hy-v3", "hunyuan", "tencent"] {
            guard let parser = ReasoningParser.fromCapabilityName(stamp) else {
                Issue.record("stamp \(stamp) did not resolve to a parser")
                continue
            }
            // Hy3 uses qwen3-style `<think:opensource>...</think:opensource>`.
            #expect(parser.startTag == "<think:opensource>")
            #expect(parser.endTag == "</think:opensource>")
            // Strays on think_xml family must be stripped, not leaked.
            #expect(parser.stripStrayTags)
        }
    }

    @Test("Hy3 no_think prompt (closed <think:opensource></think:opensource> prefill) does NOT leak content into reasoning")
    func noThinkPromptKeepsContentInContent() throws {
        // The Hy3 chat template renders a closed empty think block when
        // `reasoning_effort=no_think` is set:
        //   `<think:opensource>\n\n</think:opensource>\n\nThe answer is...`
        // `forPrompt` must inspect the tail, see `</think:opensource>` is the last
        // tag, and start in CONTENT.
        let promptTail = "<｜hy_Assistant｜><think:opensource>\n\n</think:opensource>\n\n"
        var parser = ReasoningParser.forPrompt(
            stampName: "hy_v3", promptTail: promptTail)!

        let (content, reasoning) = Self.drain(
            &parser, "The capital of France is Paris.")

        // No reasoning leak — the entire output is content.
        #expect(reasoning.isEmpty,
            "no_think Hy3 must NOT leak content into .reasoning, got: \(reasoning)")
        #expect(content == "The capital of France is Paris.")
    }

    @Test("Hy3 reasoning_effort=high prompt (open <think:opensource> prefill) routes pre-</think:opensource> into reasoning")
    func highEffortPromptRoutesReasoningCorrectly() throws {
        // The Hy3 chat template renders an OPEN think block when
        // `reasoning_effort=high|low` is set:
        //   `<｜hy_Assistant｜><think:opensource>\n` (no closer)
        // Parser starts in REASONING and emits the pre-`</think:opensource>` text
        // into `.reasoning`, post-`</think:opensource>` into `.content`.
        let promptTail = "<｜hy_Assistant｜><think:opensource>\n"
        var parser = ReasoningParser.forPrompt(
            stampName: "hy_v3", promptTail: promptTail)!

        let (content, reasoning) = Self.drain(
            &parser, "Let me work this out...\n</think:opensource>\nThe answer is 42.")

        // Pre-closer text is reasoning; post-closer is content.
        #expect(reasoning.contains("Let me work this out..."))
        #expect(content.contains("The answer is 42."))
        // No content bleed into reasoning AFTER the closer.
        #expect(!reasoning.contains("The answer is 42."),
            "Post-</think:opensource> content must not appear in reasoning: \(reasoning)")
        // No reasoning bleed into content BEFORE the closer.
        #expect(!content.contains("Let me work this out..."),
            "Pre-</think:opensource> reasoning must not appear in content: \(content)")
    }

    @Test("Hy3 misbehaving model emitting stray <think:opensource> mid-content latches correctly")
    func midStreamStrayThinkLatches() throws {
        // Some prompts have NO think prefill (e.g. caller didn't set
        // `reasoning_effort` and the template default is no_think).
        // If the model misbehaves and emits `<think:opensource>...</think:opensource>` mid-stream
        // anyway, the parser should latch on the opener and route the
        // inner block into reasoning, then return to content.
        let promptTail = "<｜hy_Assistant｜>"
        var parser = ReasoningParser.forPrompt(
            stampName: "hy_v3", promptTail: promptTail)!

        let (content, reasoning) = Self.drain(
            &parser,
            "Sure! <think:opensource>brief check</think:opensource>The result is correct.")

        #expect(reasoning.contains("brief check"))
        #expect(content.contains("Sure!"))
        #expect(content.contains("The result is correct."))
        // Content must NOT contain the inner reasoning text.
        #expect(!content.contains("brief check"),
            "Mid-stream stray reasoning must NOT leak into content: \(content)")
    }

    @Test("Hy3 strips stray closing </think:opensource> when no opener was seen")
    func straysStripped() throws {
        // think_xml family has stripStrayTags=true. If the model emits
        // a stray `</think:opensource>` without a preceding `<think:opensource>` AND the
        // prompt didn't open one, the tag should be STRIPPED from
        // content (not appear as literal text in the user-visible
        // output). This mirrors the `qwen3_6` family contract.
        let promptTail = "<｜hy_Assistant｜><think:opensource>\n\n</think:opensource>\n\n"
        var parser = ReasoningParser.forPrompt(
            stampName: "hy_v3", promptTail: promptTail)!

        let (content, _) = Self.drain(&parser, "The answer is </think:opensource>42.")

        // Stray closer should be stripped from content.
        #expect(!content.contains("</think:opensource>"),
            "Stray </think:opensource> must be stripped, got content: \(content)")
        #expect(content.contains("The answer is"))
        #expect(content.contains("42."))
    }
}
