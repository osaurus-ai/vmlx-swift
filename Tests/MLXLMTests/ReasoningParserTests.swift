// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
import MLXLMCommon
import Testing

/// Mini driver that mirrors the post-detokenizer pipeline shared by
/// Evaluate.swift / BatchEngine.swift / SpecDecStream.swift (ReasoningParser →
/// ToolCallProcessor → emit). Used only by the `Generation.reasoning` event
/// regression tests below — keeps them free of model-loading machinery.
private func driveGenerationPipeline(
    chunks: [String],
    reasoningParserStartTag: String = "<think>",
    reasoningParserEndTag: String = "</think>",
    toolCallFormat: ToolCallFormat = .json
) -> [Generation] {
    var reasoningParser: ReasoningParser? = ReasoningParser(
        startTag: reasoningParserStartTag, endTag: reasoningParserEndTag)
    let toolCallProcessor = ToolCallProcessor(format: toolCallFormat)
    var events: [Generation] = []

    func pump(_ raw: String) {
        let pieces: [String]
        if var parser = reasoningParser {
            var kept: [String] = []
            for segment in parser.feed(raw) {
                switch segment {
                case .content(let c):
                    kept.append(c)
                case .reasoning(let r):
                    events.append(
                        contentsOf: routeGenerationText(
                            r, channel: .reasoning, through: toolCallProcessor))
                }
            }
            reasoningParser = parser
            pieces = kept
        } else {
            pieces = [raw]
        }
        for piece in pieces {
            events.append(
                contentsOf: routeGenerationText(
                    piece, channel: .content, through: toolCallProcessor))
        }
    }

    func flush() {
        if var parser = reasoningParser {
            for segment in parser.flush() {
                switch segment {
                case .content(let c):
                    events.append(
                        contentsOf: routeGenerationText(
                            c, channel: .content, through: toolCallProcessor))
                case .reasoning(let r):
                    events.append(
                        contentsOf: routeGenerationText(
                            r, channel: .reasoning, through: toolCallProcessor))
                }
            }
            reasoningParser = parser
        }
        toolCallProcessor.processEOS()
        events.append(contentsOf: drainToolCallEvents(from: toolCallProcessor))
    }

    for chunk in chunks { pump(chunk) }
    flush()
    return events
}

@Suite("Generation.reasoning event")
struct GenerationReasoningEventTests {

    @Test("reasoning block surfaces as .reasoning events, answer as .chunk")
    func testReasoningAndContentSeparated() {
        let events = driveGenerationPipeline(chunks: [
            "<think>weighing options</think>final answer"
        ])

        let reasoningPieces = events.compactMap { $0.reasoning }
        let contentPieces = events.compactMap { $0.chunk }
        #expect(reasoningPieces.joined() == "weighing options")
        #expect(contentPieces.joined() == "final answer")
    }

    @Test("reasoning streams char-by-char across chunk boundaries")
    func testReasoningStreamsAcrossChunks() {
        // Simulate the token-by-token delivery osaurus sees from the
        // detokenizer. The parser must still emit a coherent .reasoning
        // stream even when the start/end tags straddle chunks.
        let prompt = "pre<think>A B C</think>post"
        let chunks = prompt.map { String($0) }
        let events = driveGenerationPipeline(chunks: chunks)

        let reasoning = events.compactMap { $0.reasoning }.joined()
        let content = events.compactMap { $0.chunk }.joined()
        #expect(reasoning == "A B C")
        #expect(content == "prepost")
    }

    @Test("unclosed <think> flushes trailing reasoning at end-of-stream")
    func testUnclosedReasoningFlushesOnEOS() {
        let events = driveGenerationPipeline(chunks: [
            "answer<think>truncated mid-thought"
        ])
        let reasoning = events.compactMap { $0.reasoning }.joined()
        let content = events.compactMap { $0.chunk }.joined()
        // Matches ReasoningParser.split semantics: post-start bytes are
        // reasoning, even if no closing tag arrived.
        #expect(reasoning == "truncated mid-thought")
        #expect(content == "answer")
    }

    @Test("no reasoning tag → zero .reasoning events")
    func testNoReasoningEmitsNoReasoningEvents() {
        let events = driveGenerationPipeline(chunks: ["just an answer"])
        let reasoningCount = events.filter { $0.reasoning != nil }.count
        #expect(reasoningCount == 0)
        #expect(events.compactMap { $0.chunk }.joined() == "just an answer")
    }

    @Test("reasoning + content coexist cleanly — neither bleeds into the other")
    func testReasoningAndContentChannelsIsolated() {
        // Pin the channel contract: reasoning segments never appear in
        // the content stream, and content never appears in the reasoning
        // stream. Tool-call extraction format specifics are covered by
        // ToolCallEdgeCasesTests / ToolTests — here we only care that
        // the two CHANNELS are clean.
        let events = driveGenerationPipeline(chunks: [
            "<think>deliberating</think>",
            "the actual answer goes here.",
        ])
        let reasoning = events.compactMap { $0.reasoning }.joined()
        let content = events.compactMap { $0.chunk }.joined()

        #expect(reasoning == "deliberating")
        #expect(content == "the actual answer goes here.")
        #expect(!content.contains("<think>"))
        #expect(!content.contains("</think>"))
        #expect(!reasoning.contains("actual answer"))
    }

    @Test("MiniMax tool calls inside reasoning extract as tool calls and do not leak")
    func testMiniMaxToolCallInsideReasoningDoesNotLeak() throws {
        let events = driveGenerationPipeline(
            chunks: [
                """
                <think>Need a lookup.
                <minimax:tool_call><invoke name="search"><parameter name="query">swift</parameter></invoke></minimax:tool_call>
                Continue after tool.</think>Done.
                """
            ],
            toolCallFormat: .minimaxM2)

        let reasoning = events.compactMap { $0.reasoning }.joined()
        let content = events.compactMap { $0.chunk }.joined()
        let calls = events.compactMap { $0.toolCall }

        #expect(reasoning.contains("Need a lookup."))
        #expect(reasoning.contains("Continue after tool."))
        #expect(content.contains("Done."))
        #expect(!reasoning.contains("minimax:tool_call"))
        #expect(!content.contains("minimax:tool_call"))
        #expect(calls.count == 1)
        let call = try #require(calls.first)
        #expect(call.function.name == "search")
        #expect(call.function.arguments["query"] == .string("swift"))
    }

    @Test("Generation enum has .reasoning case + reasoning computed property")
    func testGenerationEnumSurface() {
        // Pure enum smoke test — pin the public API surface osaurus
        // depends on.
        let r: Generation = .reasoning("thinking out loud")
        #expect(r.reasoning == "thinking out loud")
        #expect(r.chunk == nil)
        #expect(r.toolCall == nil)
        #expect(r.info == nil)

        let c: Generation = .chunk("answer")
        #expect(c.reasoning == nil)
        #expect(c.chunk == "answer")
    }
}

@Suite("ReasoningParser")
struct ReasoningParserTests {

    // MARK: - Whole-string split

    @Test func splitEmpty() {
        let (r, c) = ReasoningParser.split("")
        #expect(r.isEmpty)
        #expect(c.isEmpty)
    }

    @Test func splitNoTags() {
        let (r, c) = ReasoningParser.split("hello world")
        #expect(r.isEmpty)
        #expect(c == "hello world")
    }

    @Test func splitSingleReasoningBlock() {
        let (r, c) = ReasoningParser.split("<think>weighing options</think>final answer")
        #expect(r == "weighing options")
        #expect(c == "final answer")
    }

    @Test func splitReasoningOnly() {
        let (r, c) = ReasoningParser.split("prefix<think>only thinking</think>")
        #expect(r == "only thinking")
        #expect(c == "prefix")
    }

    @Test func splitMultipleReasoningBlocks() {
        // Two interleaved think blocks — accumulate.
        let (r, c) = ReasoningParser.split(
            "first<think>r1</think>middle<think>r2</think>last")
        #expect(r == "r1r2")
        #expect(c == "firstmiddlelast")
    }

    @Test func splitUnclosedReasoning() {
        // No `</think>` → everything after `<think>` is reasoning.
        let (r, c) = ReasoningParser.split("answer<think>truncated...")
        #expect(r == "truncated...")
        #expect(c == "answer")
    }

    // MARK: - Streaming

    @Test func streamCharByCharSimpleBlock() {
        // Drip-feed every character to verify partial-tag holdback works.
        var parser = ReasoningParser()
        let input = "<think>hi</think>ok"
        var reasoning = ""
        var content = ""
        for ch in input {
            for seg in parser.feed(String(ch)) {
                switch seg {
                case .reasoning(let r): reasoning.append(r)
                case .content(let c): content.append(c)
                }
            }
        }
        for seg in parser.flush() {
            switch seg {
            case .reasoning(let r): reasoning.append(r)
            case .content(let c): content.append(c)
            }
        }
        #expect(reasoning == "hi")
        #expect(content == "ok")
    }

    @Test func streamTagSplitAcrossChunks() {
        // The opening tag `<think>` arrives as `<thi` then `nk>`.
        var parser = ReasoningParser()
        let chunks = ["pre<thi", "nk>thoughts</thi", "nk>post"]
        var reasoning = ""
        var content = ""
        for ch in chunks {
            for seg in parser.feed(ch) {
                switch seg {
                case .reasoning(let r): reasoning.append(r)
                case .content(let c): content.append(c)
                }
            }
        }
        for seg in parser.flush() {
            switch seg {
            case .reasoning(let r): reasoning.append(r)
            case .content(let c): content.append(c)
            }
        }
        #expect(reasoning == "thoughts")
        #expect(content == "prepost")
    }

    @Test func streamAdjacentBlocks() {
        // `</think>` immediately followed by `<think>` — back-to-back.
        var parser = ReasoningParser()
        let input = "<think>a</think><think>b</think>tail"
        var reasoning = ""
        var content = ""
        for seg in parser.feed(input) {
            switch seg {
            case .reasoning(let r): reasoning.append(r)
            case .content(let c): content.append(c)
            }
        }
        for seg in parser.flush() {
            switch seg {
            case .reasoning(let r): reasoning.append(r)
            case .content(let c): content.append(c)
            }
        }
        #expect(reasoning == "ab")
        #expect(content == "tail")
    }

    @Test("Stray </think> in content mode is stripped (interleaved-thinking leak)")
    func streamStrayCloserStrippedInContent() {
        // 2026-04-25 osaurus repro: MiniMax-Small JANGTQ chat where the
        // model emitted `</think>` THREE times across one assistant
        // turn after the first `</think>` had already flipped state to
        // content. Old parser only looked for `<think>` when in content
        // mode, so the stray `</think>` markers leaked into `.chunk`.
        // With stripStrayTags=true (default for think_xml family) the
        // strays are consumed silently.
        var parser = ReasoningParser(startInReasoning: true)
        let input =
            "first reasoning</think>"        // legitimate close → content
            + "first content</think>"        // STRAY — must not leak
            + "more content</think>"         // STRAY — must not leak
            + "tail"
        var reasoning = ""
        var content = ""
        for seg in parser.feed(input) {
            switch seg {
            case .reasoning(let r): reasoning.append(r)
            case .content(let c): content.append(c)
            }
        }
        for seg in parser.flush() {
            switch seg {
            case .reasoning(let r): reasoning.append(r)
            case .content(let c): content.append(c)
            }
        }
        #expect(reasoning == "first reasoning")
        #expect(content == "first contentmore contenttail")
        #expect(!content.contains("</think>"),
            "stray </think> markers must not leak into the visible chunk stream")
    }

    @Test("Stray <think> in reasoning mode is stripped (symmetric)")
    func streamStrayOpenerStrippedInReasoning() {
        // Symmetric: model emits `<think>` while parser is already in
        // reasoning. Should be stripped without flipping state.
        var parser = ReasoningParser(startInReasoning: true)
        let input =
            "outer <think>nested opener</think>"  // STRAY <think>, real </think>
            + "tail"
        var reasoning = ""
        var content = ""
        for seg in parser.feed(input) {
            switch seg {
            case .reasoning(let r): reasoning.append(r)
            case .content(let c): content.append(c)
            }
        }
        for seg in parser.flush() {
            switch seg {
            case .reasoning(let r): reasoning.append(r)
            case .content(let c): content.append(c)
            }
        }
        #expect(reasoning == "outer nested opener")
        #expect(content == "tail")
        #expect(!reasoning.contains("<think>"),
            "stray <think> markers must not leak into reasoning text")
    }

    @Test func streamFlushDrainsBufferedPartial() {
        // Stream ends mid-pretag — flush must still emit the buffered text
        // as content rather than dropping it.
        var parser = ReasoningParser()
        var content = ""
        for seg in parser.feed("hello<thi") {
            if case .content(let c) = seg { content.append(c) }
        }
        for seg in parser.flush() {
            if case .content(let c) = seg { content.append(c) }
        }
        #expect(content == "hello<thi")
    }

    // MARK: - Custom tags

    @Test func customTags() {
        let (r, c) = ReasoningParser.split(
            "[REASON]inner[/REASON]visible",
            startTag: "[REASON]", endTag: "[/REASON]")
        #expect(r == "inner")
        #expect(c == "visible")
    }

    // MARK: - Capability-name resolution

    @Test func capabilityAliasesQwen3() {
        for name in ["qwen3", "qwen3_5", "qwen3_6", "think_xml", "deepseek_r1", "poolside_v1"] {
            #expect(
                ReasoningParser.fromCapabilityName(name) != nil,
                "\(name) should resolve to a parser")
        }
    }

    @Test func capabilityNoneAliases() {
        // `gemma4` used to resolve to nil but now routes to the harmony
        // parser (see capabilityGemma4Harmony below) — it's no longer
        // a "no reasoning" stamp because Gemma-4 DOES emit reasoning
        // inside `<|channel>thought\n…<channel|>` envelopes.
        for name in ["none", "off", "disabled", "mistral", "gemma"] {
            #expect(
                ReasoningParser.fromCapabilityName(name) == nil,
                "\(name) should resolve to no parser")
        }
    }

    @Test("Gemma-4 stamp resolves to a harmony-tag ReasoningParser")
    func capabilityGemma4Harmony() {
        // Verifies the 2026-04-20 fix for tpae's report that Gemma-4
        // was emitting `<|channel>thought\n…` directly in `.chunk`.
        // All three alias names for the harmony family must resolve.
        for name in ["harmony", "harmony_channel", "gemma4_channel", "gemma4"] {
            guard let parser = ReasoningParser.fromCapabilityName(name) else {
                Issue.record("\(name) must resolve to a non-nil parser")
                continue
            }
            // Start tag is bare `<|channel>` so the parser latches on
            // any channel name (thought / analysis / final / action /
            // …) — not just `thought`. This fixes the 2026-04-20
            // 3:54 PM report where Gemma-4 emitted a JSON
            // `<|channel> {"action": …}<channel|>` block that leaked
            // into `.chunk` when the start tag required `thought\n`.
            #expect(parser.startTag == "<|channel>")
            #expect(parser.endTag == "<channel|>")
        }
    }

    @Test("Qwen 3.x stamps resolve to a parser that starts in reasoning")
    func capabilityQwen3StartsInReasoning() {
        // Qwen 3.x chat templates prefill `<think>\n` at prompt tail
        // when `enable_thinking=true` (the default), so the model's
        // first byte is already inside a think block. The parser MUST
        // start in reasoning state or pre-</think> text leaks into
        // `.chunk` — see TPAE-2026-04-20-TRIAGE.md Bug B.
        for name in ["think_xml", "qwen3", "qwen3_5", "qwen3_6"] {
            guard var parser = ReasoningParser.fromCapabilityName(name) else {
                Issue.record("\(name) must resolve to a parser")
                continue
            }
            // Feed a single closer tag — if the parser starts in
            // reasoning mode, the tag flips state to content with NO
            // prior content emitted. Call flush() after to drain the
            // tail that the streaming parser holds back against a
            // potential split tag.
            var segs = parser.feed("</think>hello")
            segs.append(contentsOf: parser.flush())
            let reasoning = segs.compactMap { s -> String? in
                if case .reasoning(let r) = s { return r } else { return nil }
            }.joined()
            let content = segs.compactMap { s -> String? in
                if case .content(let c) = s { return c } else { return nil }
            }.joined()
            #expect(reasoning.isEmpty, "\(name): no text before </think>")
            #expect(content == "hello", "\(name): post-</think> is content")
        }
    }

    @Test func capabilityUnknownReturnsNil() {
        #expect(ReasoningParser.fromCapabilityName(nil) == nil)
        #expect(ReasoningParser.fromCapabilityName("") == nil)
        #expect(ReasoningParser.fromCapabilityName("madeup") == nil)
    }

    // MARK: - ParserResolution precedence

    @Test func reasoningStampedQwenWinsOverHeuristic() {
        let cap = JangCapabilities(reasoningParser: "qwen3")
        let (parser, source) = ParserResolution.reasoning(
            capabilities: cap, modelType: "mistral4")
        #expect(parser != nil, "stamp must override mistral heuristic")
        #expect(source == .jangStamped)
    }

    @Test func reasoningStampedNoneWinsOverHeuristic() {
        let cap = JangCapabilities(reasoningParser: "none")
        let (parser, source) = ParserResolution.reasoning(
            capabilities: cap, modelType: "qwen3_5_moe")
        #expect(parser == nil, "stamp `none` must suppress qwen heuristic")
        #expect(source == .jangStamped)
    }

    @Test func reasoningHeuristicQwenFallback() {
        let (parser, source) = ParserResolution.reasoning(
            capabilities: nil, modelType: "qwen3_5_moe")
        #expect(parser != nil)
        #expect(source == .modelTypeHeuristic)
    }

    @Test func reasoningHeuristicQwen36TextConfigVariant() {
        // Qwen 3.6 sometimes surfaces model_type=qwen3_5_moe_text from text_config.
        // Heuristic must still return a reasoning parser.
        let (parser, source) = ParserResolution.reasoning(
            capabilities: nil, modelType: "qwen3_5_moe_text")
        #expect(parser != nil)
        #expect(source == .modelTypeHeuristic)
    }

    @Test func reasoningHeuristicMistralReturnsNone() {
        let (parser, source) = ParserResolution.reasoning(
            capabilities: nil, modelType: "mistral4")
        #expect(parser == nil)
        #expect(source == .modelTypeHeuristic)
    }

    @Test func reasoningEmptyInputsReturnNone() {
        let (parser, source) = ParserResolution.reasoning(
            capabilities: nil, modelType: nil)
        #expect(parser == nil)
        #expect(source == .none)
    }
}

// MARK: - ToolCallFormat capability resolution

@Suite("ToolCallFormat capability")
struct ToolCallFormatCapabilityTests {

    @Test func directRawValueWins() {
        #expect(ToolCallFormat.fromCapabilityName("xml_function") == .xmlFunction)
        #expect(ToolCallFormat.fromCapabilityName("minimax_m2") == .minimaxM2)
        #expect(ToolCallFormat.fromCapabilityName("kimi_k2") == .kimiK2)
    }

    @Test func qwenAliases() {
        for name in ["qwen", "qwen3", "qwen3_5", "qwen3_6", "qwen3_coder"] {
            #expect(
                ToolCallFormat.fromCapabilityName(name) == .xmlFunction,
                "\(name) should map to xml_function")
        }
    }

    @Test func minimaxAlias() {
        #expect(ToolCallFormat.fromCapabilityName("minimax") == .minimaxM2)
    }

    @Test func glmAndDeepseekAliases() {
        #expect(ToolCallFormat.fromCapabilityName("glm47") == .glm4)
        #expect(ToolCallFormat.fromCapabilityName("glm4_moe") == .glm4)
        #expect(ToolCallFormat.fromCapabilityName("deepseek") == .glm4)
    }

    @Test func nemotronAlias() {
        #expect(ToolCallFormat.fromCapabilityName("nemotron") == .nemotron)
        #expect(ToolCallFormat.fromCapabilityName("nemotron_h") == .nemotron)
    }

    /// Locks in `laguna` model_type → `think_xml` reasoning stamp.
    /// Poolside's `laguna_glm_thinking_v5/chat_template.jinja` emits
    /// `<think>...</think>` blocks when `enable_thinking=true`. Without
    /// this stamp, when (later) the vmlx `laguna` model class lands, the
    /// CoT output would land as plain `.chunk` events instead of
    /// `.reasoning` deltas — silently breaking the host's reasoning UI.
    /// The pre-registration here means stamp resolution Just Works the
    /// day the model class lands; no follow-up edit needed.
    @Test func lagunaResolvesToThinkXml() {
        #expect(reasoningStampFromModelType("laguna") == "think_xml")
        #expect(reasoningStampFromModelType("laguna_xs") == "think_xml")
        #expect(reasoningStampFromModelType("laguna_s") == "think_xml")
        #expect(reasoningStampFromModelType("Laguna") == "think_xml",
            "case-insensitive match")
    }

    @Test func unknownReturnsNil() {
        #expect(ToolCallFormat.fromCapabilityName(nil) == nil)
        #expect(ToolCallFormat.fromCapabilityName("") == nil)
        #expect(ToolCallFormat.fromCapabilityName("zzzunknown") == nil)
    }

    @Test func resolutionStampedWinsOverHeuristic() {
        let cap = JangCapabilities(toolParser: "qwen3_coder")
        let (fmt, src) = ParserResolution.toolCall(
            capabilities: cap, modelType: "mistral3")
        #expect(fmt == .xmlFunction, "stamp must override mistral heuristic")
        #expect(src == .jangStamped)
    }

    @Test func resolutionHeuristicFallback() {
        let (fmt, src) = ParserResolution.toolCall(
            capabilities: nil, modelType: "qwen3_5_moe")
        #expect(fmt == .xmlFunction)
        #expect(src == .modelTypeHeuristic)
    }

    @Test func resolutionEmptyReturnsNone() {
        let (fmt, src) = ParserResolution.toolCall(
            capabilities: nil, modelType: nil)
        #expect(fmt == nil)
        #expect(src == .none)
    }
}

// MARK: - Harmony (Gemma-4) + prefilled <think> (Qwen 3.6) regressions
//
// Covers the two bugs reported by tpae on 2026-04-20:
//
//   A) Gemma-4-26B emitting `<|channel>thought\n…<channel|>` directly
//      in `.chunk(String)`. Root cause: `reasoningParserName = "none"`
//      for Gemma-4 → no parser. Fix: route `gemma4` → harmony parser.
//
//   B) Qwen3.6 `.chunk` showing `R</think>` because the chat template
//      prefills `<think>\n` at prompt tail (enable_thinking=true
//      default), so the model's first generated byte is already inside
//      a think block. Fix: `startInReasoning=true` for the `<think>`
//      family stamps.

@Suite("Harmony (Gemma-4) parser — streaming")
struct HarmonyParserStreamingTests {
    // Bare `<|channel>` start tag matches ANY channel name — not just
    // `thought`. Gemma-4 at inference emits channels like
    // `<|channel>thought\n…<channel|>` for CoT but also
    // `<|channel> {...json...}<channel|>` for ReAct-style tool hints.
    // See 2026-04-20 3:54 PM tpae screenshot / TRIAGE 3:54 PM addendum.
    private var harmonyParser: ReasoningParser {
        ReasoningParser(
            startTag: "<|channel>",
            endTag: "<channel|>",
            startInReasoning: false,
            // Match the production harmony factory in
            // ReasoningParser.fromCapabilityName(_:) — A2/A3 expect
            // stray markers to pass through as literal content.
            stripStrayTags: false)
    }

    @Test("full harmony block (thought channel) in a single feed")
    func testSingleFeedHarmonyBlock() {
        var p = harmonyParser
        var segs = p.feed(
            "prefix<|channel>thought\ninner reasoning<channel|>after")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning == "inner reasoning")
        #expect(content == "prefixafter")
    }

    @Test("char-by-char streaming of a harmony block")
    func testCharByCharHarmony() {
        var p = harmonyParser
        let input = "pre<|channel>thought\nthinking<channel|>answer"
        var reasoning = ""
        var content = ""
        for ch in input {
            for s in p.feed(String(ch)) {
                switch s {
                case .reasoning(let r): reasoning += r
                case .content(let c): content += c
                }
            }
        }
        for s in p.flush() {
            switch s {
            case .reasoning(let r): reasoning += r
            case .content(let c): content += c
            }
        }
        #expect(reasoning == "thinking")
        #expect(content == "preanswer")
    }

    @Test("multiple adjacent harmony blocks accumulate")
    func testMultipleHarmonyBlocks() {
        var p = harmonyParser
        var segs = p.feed(
            "<|channel>thought\na<channel|>mid<|channel>thought\nb<channel|>end")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning == "ab")
        #expect(content == "midend")
    }

    @Test("2026-04-20 3:54 PM regression — JSON action channel with no `thought` name")
    func testJsonActionChannelRoutesToReasoning() {
        // Exact byte sequence from tpae's 3:54 PM screenshot:
        //   <|channel> {\n "action": "google_search",\n ...\n}\n<channel|>I don't...
        // The old parser (start = "<|channel>thought\n") didn't match
        // because byte 11 is ' ', not 't', so the whole envelope leaked
        // into `.chunk`. With bare `<|channel>` start, it latches.
        var p = harmonyParser
        let input =
            "<|channel> {\n" +
            " \"action\": \"google_search\",\n" +
            " \"action_input\": \"weather in Irvine\"\n" +
            "}\n" +
            "<channel|>I don't have real-time weather data."
        var segs = p.feed(input)
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning.contains("google_search"),
            "JSON action block must be routed to reasoning, not content")
        #expect(content == "I don't have real-time weather data.")
        // Absolutely no harmony markers anywhere in content.
        #expect(!content.contains("<|channel>"))
        #expect(!content.contains("<channel|>"))
    }

    @Test("analysis / final / custom channel names all route to reasoning")
    func testCustomChannelNames() {
        for (name, innerText) in [
            ("analysis", "weighing options"),
            ("final", "pre-answer thoughts"),
            ("tool", "{\"name\":\"f\"}"),
            ("", "no-name channel"),
        ] {
            var p = harmonyParser
            let input = "a<|channel>\(name)\n\(innerText)<channel|>b"
            var segs = p.feed(input)
            segs.append(contentsOf: p.flush())
            let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
            let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
            #expect(reasoning == innerText,
                "channel name `\(name)`: full envelope should route to reasoning")
            #expect(content == "ab",
                "channel name `\(name)`: only pre/post envelope should remain as content")
        }
    }

    @Test("unclosed harmony block flushes as reasoning on EOS")
    func testUnclosedHarmonyFlush() {
        var p = harmonyParser
        var segs = p.feed("visible<|channel>thought\ntruncated mid-thought")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning == "truncated mid-thought")
        #expect(content == "visible")
    }

    @Test("JSON action body without channel name stays intact")
    func testJsonActionBodyWithoutChannelNameStaysIntact() {
        var p = harmonyParser
        let input =
            "<|channel>{\n" +
            " \"action\": \"google_search\"\n" +
            "}\n<channel|>done"
        var segs = p.feed(input)
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning.hasPrefix("{\n"))
        #expect(reasoning.contains("google_search"))
        #expect(content == "done")
    }

    @Test("identifier-only channel header at EOS is stripped")
    func testIdentifierOnlyChannelHeaderAtEOSIsStripped() {
        var p = harmonyParser
        var segs = p.feed("visible<|channel>thought")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning.isEmpty)
        #expect(content == "visible")
    }

    @Test("single-line JSON action body at EOS stays intact")
    func testSingleLineJsonActionBodyAtEOSStaysIntact() {
        var p = harmonyParser
        var segs = p.feed("<|channel>{\"action\":\"google_search\"}")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning == "{\"action\":\"google_search\"}")
        #expect(content.isEmpty)
    }

    @Test("no harmony markers → all content, zero reasoning")
    func testPassThrough() {
        var p = harmonyParser
        let segs = p.feed("plain answer, no channel markers")
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning.isEmpty)
        // Up to (maxTagLen-1) chars may still be buffered — so drain with flush.
        let tail = p.flush()
        let tailContent = tail.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(content + tailContent == "plain answer, no channel markers")
    }

    // MARK: - Category A (edge-case audit 2026-04-20)

    @Test("A1: empty channel body — zero-byte reasoning")
    func testA1EmptyChannelBody() {
        // Model emits `<|channel><channel|>` immediately — e.g., Gemma-4
        // when `enable_thinking=false` and the template prefills an
        // empty thought block (chat_template.jinja line 344).
        var p = harmonyParser
        var segs = p.feed("before<|channel><channel|>after")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning.isEmpty)
        #expect(content == "beforeafter")
    }

    @Test("A2: nested `<|channel>` inside reasoning body — first closer wins")
    func testA2NestedOpenerInBody() {
        // State machine is toggle-based: once inside reasoning, a
        // second `<|channel>` is just bytes until we see `<channel|>`.
        var p = harmonyParser
        var segs = p.feed("<|channel>outer<|channel>nested<channel|>after")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning == "outer<|channel>nested")
        #expect(content == "after")
    }

    @Test("A3: closer before opener is literal content")
    func testA3OrphanCloser() {
        // `<channel|>` before any opener should NOT flip state — it's
        // just bytes in the content stream.
        var p = harmonyParser
        var segs = p.feed("foo<channel|>bar<|channel>inner<channel|>tail")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning == "inner")
        #expect(content == "foo<channel|>bartail")
    }

    @Test("A4: closer split across feeds — no leak")
    func testA4CloserSplitAcrossFeeds() {
        var p = harmonyParser
        var segs: [ReasoningSegment] = []
        segs += p.feed("<|channel>thought\ninner reasoning<chan")
        // Partial closer — must not flip state yet.
        segs += p.feed("nel|>visible answer")
        segs += p.flush()
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning == "inner reasoning")
        #expect(content == "visible answer")
    }

    @Test("A5: opener split across feeds — no leak")
    func testA5OpenerSplitAcrossFeeds() {
        var p = harmonyParser
        var segs: [ReasoningSegment] = []
        segs += p.feed("prefix<|cha")
        segs += p.feed("nnel>thought\ninner<channel|>after")
        segs += p.flush()
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning == "inner")
        #expect(content == "prefixafter")
    }

    @Test("A7: maxTokens truncates mid-opener — partial opener is content")
    func testA7TruncatedMidOpener() {
        // Model started to emit `<|cha` then hit max_tokens. State
        // never flipped to reasoning, so the partial opener bytes are
        // plain content.
        var p = harmonyParser
        var segs = p.feed("the answer is<|cha")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning.isEmpty)
        #expect(content == "the answer is<|cha")
    }

    @Test("A8: maxTokens truncates mid-closer inside reasoning — tail is reasoning")
    func testA8TruncatedMidCloser() {
        // Inside a reasoning block, maxTokens hits mid-closer. Flush
        // must emit the held bytes as reasoning (not content).
        var p = harmonyParser
        var segs = p.feed("<|channel>thought\nlots of reasoning<chann")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning == "lots of reasoning<chann")
        #expect(content.isEmpty)
    }
}

@Suite("startInReasoning=true (Qwen 3.6 enable_thinking prefill)")
struct StartInReasoningTests {

    @Test("first bytes are reasoning until first </think>")
    func testReasoningFirstFlipsAtCloser() {
        var p = ReasoningParser(startInReasoning: true)
        var segs = p.feed("thinking about the problem</think>final answer here.")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning == "thinking about the problem")
        #expect(content == "final answer here.")
    }

    @Test("char-by-char streaming of pre-</think> reasoning")
    func testCharByCharPrefilled() {
        var p = ReasoningParser(startInReasoning: true)
        let input = "abc</think>xyz"
        var reasoning = ""
        var content = ""
        for ch in input {
            for s in p.feed(String(ch)) {
                switch s {
                case .reasoning(let r): reasoning += r
                case .content(let c): content += c
                }
            }
        }
        for s in p.flush() {
            switch s {
            case .reasoning(let r): reasoning += r
            case .content(let c): content += c
                }
        }
        #expect(reasoning == "abc")
        #expect(content == "xyz")
    }

    @Test("unclosed pre-</think> on EOS flushes to reasoning")
    func testUnclosedPrefilledFlushesToReasoning() {
        var p = ReasoningParser(startInReasoning: true)
        var segs = p.feed("model ran out of tokens mid-thought")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning == "model ran out of tokens mid-thought")
        #expect(content.isEmpty)
    }

    @Test("default (startInReasoning=false) unchanged — byte-compat")
    func testDefaultUnchanged() {
        // The existing byte-identical behaviour for callers that don't
        // opt into startInReasoning.
        var p = ReasoningParser()
        var segs = p.feed("visible<think>hidden</think>more visible")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning == "hidden")
        #expect(content == "visiblemore visible")
    }

    // MARK: - Category B (edge-case audit 2026-04-20)

    @Test("B1: enable_thinking=false with think_xml stamp — no leak into reasoning")
    func testB1EnableThinkingFalse() {
        // When the caller passes `enable_thinking=false` to the Qwen 3.x
        // chat template, the prompt body already contains
        // `<think>\n\n</think>\n\n` so the model output starts in
        // CONTENT mode (no opener, no closer in the output stream).
        //
        // `fromCapabilityName("think_xml")` returns startInReasoning=true
        // because the DEFAULT template branch prefills `<think>\n`. That
        // default is wrong for enable_thinking=false callers — the
        // parser would route the entire output to `.reasoning`.
        //
        // Correct API to use in that case: `ReasoningParser.forPrompt`
        // with the decoded prompt tail. If the tail contains an
        // un-matched `</think>` (meaning the prompt already closed a
        // think block), start in content mode.
        let promptTail = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        guard let parser = ReasoningParser.forPrompt(
            stampName: "think_xml",
            promptTail: promptTail)
        else {
            Issue.record("forPrompt should return a non-nil parser for think_xml")
            return
        }
        var p = parser
        // Model output: just plain content (no tags).
        var segs = p.feed("the answer is 4")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning.isEmpty,
            "enable_thinking=false prompt must not force all output into .reasoning")
        #expect(content == "the answer is 4")
    }

    @Test("B1: enable_thinking=true (default) still auto-detects startInReasoning=true")
    func testB1EnableThinkingTrueAutoDetects() {
        // Default Qwen template branch: prompt ends with `<think>\n`.
        // `forPrompt` must detect the open think block and start the
        // parser in reasoning state.
        let promptTail = "<|im_start|>assistant\n<think>\n"
        guard let parser = ReasoningParser.forPrompt(
            stampName: "think_xml",
            promptTail: promptTail)
        else {
            Issue.record("forPrompt should return a non-nil parser for think_xml")
            return
        }
        var p = parser
        // Model output: reasoning text then closer then answer.
        var segs = p.feed("thinking hard</think>answer is 4")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning == "thinking hard")
        #expect(content == "answer is 4")
    }

    @Test("ZAYA model_type fallback routes open-thinking prompt to reasoning")
    func testZayaModelTypeFallbackRoutesOpenThinkingPrompt() {
        let stamp = reasoningStampFromModelType("zaya")
        #expect(stamp == "think_xml")
        let promptTail = "<|im_start|>assistant\n<think>\n"
        guard let parser = ReasoningParser.forPrompt(
            stampName: stamp,
            promptTail: promptTail)
        else {
            Issue.record("ZAYA fallback stamp should create a think_xml parser")
            return
        }
        var p = parser
        var segs = p.feed("zaya internal note</think>visible answer")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning == "zaya internal note")
        #expect(content == "visible answer")
    }

    @Test("B1: prompt tail with NEITHER <think> nor </think> → start in content (mis-stamp safety net)")
    func testB1NoTagsInTailStartsInContent() {
        // Defense-in-depth against mis-stamped bundles. If a JANG
        // bundle or caller passes `stampName="think_xml"` but the
        // prompt tail contains no `<think>` or `</think>` at all, the
        // chat template almost certainly did NOT prefill a think
        // block — so starting in reasoning would route the entire
        // response into `.reasoning`. Reported 2026-04-24 for LFM2
        // models that some older JANG converters stamped as
        // `reasoning_parser=think_xml`.
        //
        // Safer default: start in content. The parser still latches
        // on `<think>` mid-stream if the model actually emits one,
        // so Qwen 3.6 interleaved thinking remains correct.
        let promptTail = "<|im_start|>assistant\n"  // no <think>, no </think>
        guard let parser = ReasoningParser.forPrompt(
            stampName: "think_xml",
            promptTail: promptTail)
        else {
            Issue.record("forPrompt should return a non-nil parser for think_xml")
            return
        }
        var p = parser
        // Model output: plain content, never emits any tag.
        var segs = p.feed("the answer is 4")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(
            reasoning.isEmpty,
            "Mis-stamped model with no <think> in prompt tail must not route content into .reasoning")
        #expect(content == "the answer is 4")
    }

    @Test("B1: prompt tail has no tags but model later emits <think> — still latches correctly")
    func testB1NoTagsInTailButModelEmitsThink() {
        // Follow-up to the mis-stamp safety net: even when we force
        // startInReasoning=false because of no tail tags, a model that
        // legitimately emits `<think>…</think>` mid-stream must still
        // have those segments routed to `.reasoning`.
        let promptTail = "<|im_start|>assistant\n"
        guard let parser = ReasoningParser.forPrompt(
            stampName: "think_xml",
            promptTail: promptTail)
        else {
            Issue.record("forPrompt should return a non-nil parser")
            return
        }
        var p = parser
        var segs = p.feed("prefix <think>mid-stream thought</think>answer")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning == "mid-stream thought")
        #expect(content == "prefix answer")
    }

    @Test("B1: no prompt tail given → falls back to stamp default")
    func testB1NoPromptTailFallsBackToStampDefault() {
        // forPrompt without promptTail should behave exactly like
        // fromCapabilityName.
        guard let parser = ReasoningParser.forPrompt(
            stampName: "think_xml",
            promptTail: nil)
        else {
            Issue.record("forPrompt should return a non-nil parser")
            return
        }
        var p = parser
        // If startInReasoning=true default fires, `</think>answer`
        // flips to content at the closer.
        var segs = p.feed("thinking</think>answer")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning == "thinking")
        #expect(content == "answer")
    }

    @Test("B2: interleaved thinking — multiple <think> blocks mid-response")
    func testB2InterleavedThinking() {
        // Qwen 3.6's interleaved-thinking family can emit multiple
        // <think>…</think> blocks mid-response. With startInReasoning
        // either way, the state-machine toggle must correctly split
        // every block.
        var p = ReasoningParser(startInReasoning: true)
        var segs = p.feed(
            "first thought</think>answer part 1<think>second thought</think>answer part 2")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning == "first thoughtsecond thought")
        #expect(content == "answer part 1answer part 2")
    }

    @Test("B4: partial </think> at EOS flushes as reasoning")
    func testB4PartialCloserAtEOS() {
        var p = ReasoningParser(startInReasoning: true)
        var segs = p.feed("reasoning text</thi")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning == "reasoning text</thi")
        #expect(content.isEmpty)
    }

    @Test("B5: entire output is reasoning when no closer arrives")
    func testB5NoCloserAllReasoning() {
        var p = ReasoningParser(startInReasoning: true)
        var segs = p.feed("model ran out mid-thought without closing the tag")
        segs.append(contentsOf: p.flush())
        let reasoning = segs.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let content = segs.compactMap { if case .content(let c) = $0 { return c } else { return nil } }.joined()
        #expect(reasoning == "model ran out mid-thought without closing the tag")
        #expect(content.isEmpty)
    }
}
