// Pin the Apertus 1.5 reasoning-no-leak contract.
//
// Apertus 1.5 does not use `<think>` XML. Its chat template defines two dedicated
// vocabulary tokens (chat_template.jinja lines 152-153):
//
//     {%- set inner_token = '<|inner_prefix|>' -%}
//     {%- set outer_token = '<|inner_suffix|>' -%}
//
// and the generation prompt tail is a bare `<|assistant|>` — the template prefills
// NO opener — so the model emits `<|inner_prefix|>` itself when it decides to think.
//
// THE DEFECT THIS PINS. `reasoningStampFromModelType("apertus1p5")` returned "none",
// so `JangLoader` selected NO parser, so the monologue and the raw marker streamed
// straight into the user-visible channel. Measured on
// cubiculum/Apertus-v1.5-8B-Jang_6M before the fix, gsm8k, 3 turns:
//
//     ── GEN channels: text=3998ch reasoning=0ch finish=stop tokens=1033
//
// Zero reasoning characters on every turn, with the actual answer buried 3.7 kB
// deep behind the monologue. This is the same shape as the `gptoss` Harmony marker
// leak and the `muse_glimmer` fall-through that the neighbouring stamps exist to
// prevent — and, as the `muse_glimmer` comment records, the parser is the half
// people fix while the stamp is the half that makes it reachable. BOTH halves are
// required: a test that only checked `fromCapabilityName("apertus")` would pass
// against a build where real Apertus bundles still leak.
//
// Source-coverage style — no MLX runtime needed.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Apertus 1.5 reasoning-parser no-leak contract")
struct ApertusReasoningNoLeakTests {

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

    /// THE REACHABILITY HALF. Without this the parser below is dead code for every
    /// real bundle, because `model_type` is what the loader actually has in hand.
    @Test("apertus1p5 stamps to its own parser while apertus 1.0 stays none")
    func stampResolves() {
        // `apertus1p5` is the model_type in the shipped config.json, in the spellings
        // `normalizedReasoningAlias` / `compactReasoningAlias` can produce.
        for modelType in ["apertus1p5", "Apertus1p5", "apertus_1p5", "apertus-1p5"] {
            let stamp = reasoningStampFromModelType(modelType)
            let why = "model_type \(modelType) stamped \(stamp) — \"none\" means no parser "
                + "is selected and the monologue leaks into the visible answer"
            #expect(stamp == "apertus1p5", "\(why)")
        }

        // AND THE OTHER HALF OF THE CONTRACT. Plain `apertus` is Apertus 1.0, which has
        // no inner monologue; `testPlainFamiliesGetNone` enumerates it as a non-thinking
        // family. A prefix match broad enough to catch it would hand 1.0 a parser it must
        // not have — the first draft of this fix did exactly that and turned that suite
        // red. Pin both directions so the scoping cannot silently widen later.
        #expect(reasoningStampFromModelType("apertus") == "none",
            "Apertus 1.0 must keep its non-thinking stamp")
        #expect(ReasoningParser.fromCapabilityName("apertus") == nil,
            "the 1.0 capability name must not resolve to a monologue parser")
    }

    @Test("the apertus stamp resolves to a parser with the template's own markers")
    func capabilityResolves() throws {
        let parser = try #require(ReasoningParser.fromCapabilityName("apertus1p5"),
            "apertus1p5 stamp did not resolve to a parser")
        #expect(parser.startTag == "<|inner_prefix|>")
        #expect(parser.endTag == "<|inner_suffix|>")
        // Both markers are single special tokens that cannot occur in prose, so a
        // stray must be stripped rather than shown to the user as literal text.
        #expect(parser.stripStrayTags)
    }

    /// THE DEFECT ITSELF, replayed from a real generation.
    ///
    /// Text is taken verbatim from cubiculum/Apertus-v1.5-8B-Jang_6M answering a gsm8k
    /// item (Laurel's baby outfits, gold answer 87) — the monologue's opening words and
    /// the post-closer answer are the model's own, not a hand-written imitation. Before
    /// the fix ALL of this arrived as content.
    @Test("a real Apertus turn routes the monologue to reasoning and the answer to content")
    func realTurnDoesNotLeak() throws {
        var parser = try #require(
            ReasoningParser.forPrompt(stampName: "apertus1p5", promptTail: "<|assistant|>"))

        let (content, reasoning) = Self.drain(&parser,
            "<|inner_prefix|>We need to parse the question. It's a straightforward "
            + "arithmetic problem. The user says: \"Laurel's friend gave her 24 baby "
            + "outfits\". So 24, then twice as many, then 15 more."
            + "<|inner_suffix|>The friend gave \\(24\\) outfits.\n"
            + "Total outfits:\n\\[\n24 + 48 + 15 = 87\n\\]\n\n\\[\n\\boxed{87}\n\\]")

        // The answer, and ONLY the answer, is visible.
        #expect(content.contains("\\boxed{87}"), "the answer must reach the user")
        #expect(!content.contains("We need to parse the question"),
            "the monologue must NOT be visible; got content: \(content.prefix(200))")
        #expect(!content.contains("<|inner_prefix|>") && !content.contains("<|inner_suffix|>"),
            "raw markers must never be shown to the user; got: \(content.prefix(200))")

        // And the monologue is preserved on the reasoning rail rather than discarded —
        // osaurus renders it in the think pane.
        #expect(reasoning.contains("We need to parse the question"))
        #expect(!reasoning.contains("\\boxed{87}"),
            "the answer must not be swallowed into the think pane: \(reasoning.suffix(120))")
    }

    /// A turn with NO thinking must be untouched. Apertus answers directly for easy
    /// prompts, and a parser that invented a reasoning block — or ate the first
    /// characters waiting for one — would break the common case to fix the rare one.
    @Test("a turn with no monologue is passed through unchanged")
    func nonThinkingTurnUnchanged() throws {
        var parser = try #require(
            ReasoningParser.forPrompt(stampName: "apertus1p5", promptTail: "<|assistant|>"))
        let (content, reasoning) = Self.drain(&parser, "The capital of France is Paris.")
        #expect(content == "The capital of France is Paris.")
        #expect(reasoning.isEmpty, "no opener was emitted, so nothing is reasoning")
    }

    /// THE FAILURE MODE THAT WOULD BE WORSE THAN THE BUG. If a turn opens a block and
    /// never closes it, everything after the opener is reasoning and the visible reply
    /// is EMPTY — the "trapped thinking" the Hunyuan entry documents.
    ///
    /// This is pinned rather than asserted-away because it is a real risk of the fix,
    /// not a hypothetical: 3/3 observed turns closed their block, but 3 turns is not a
    /// proof. `flush()` must surface the unterminated text rather than dropping it, so
    /// a future regression shows up as misplaced text and never as silence.
    @Test("an unclosed monologue still surfaces its text rather than vanishing")
    func unclosedBlockIsNotSilentlyDropped() throws {
        var parser = try #require(
            ReasoningParser.forPrompt(stampName: "apertus1p5", promptTail: "<|assistant|>"))
        let (content, reasoning) = Self.drain(&parser,
            "<|inner_prefix|>thinking that never terminates")
        #expect(!(content + reasoning).isEmpty,
            "an unterminated block must not swallow the turn into nothing")
        #expect((content + reasoning).contains("never terminates"))
    }
}
