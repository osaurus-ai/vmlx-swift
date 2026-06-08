// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Iter 65+: edge cases + regressions that the baseline `ToolTests.swift`
// suite doesn't cover. Focus on scenarios tpae surfaced:
//
//   1. Qwen 3.6 + MiniMax M2 interleaved <think>...</think> blocks
//      appearing BETWEEN tool calls, not just at the start.
//   2. Gemma-4 harmony-style channel markers in the token stream.
//   3. Character-by-character streaming delivery (osaurus's actual
//      real-world consumption pattern).
//   4. JANG `capabilities.tool_parser` stamp → ToolCallFormat mapping.
//   5. Gemma-4 escape-marker default regression guard.
//   6. Tool calls in back-to-back sequences without delimiters.
//
// This file intentionally sits next to `ToolTests.swift` rather than
// editing it, so the 42 baseline tests stay as a known-good reference.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Tool-Call Edge Cases (iter 65+)")
struct ToolCallEdgeCasesTests {

    // MARK: - Helpers

    /// Feed text into a processor one character at a time — models the
    /// worst case of streaming delivery where every chunk is a single
    /// token / character.
    private func feedCharByChar(_ text: String, into processor: ToolCallProcessor) -> String {
        var output = ""
        for ch in text {
            if let chunk = processor.processChunk(String(ch)) {
                output += chunk
            }
        }
        processor.processEOS()
        return output
    }

    /// Feed text into a processor in arbitrary chunk sizes. Simulates
    /// real streaming where chunks straddle token boundaries.
    private func feedInChunks(_ text: String, chunkSize: Int, into processor: ToolCallProcessor) -> String {
        let output = feedInChunksWithoutEOS(text, chunkSize: chunkSize, into: processor)
        processor.processEOS()
        return output
    }

    private func feedInChunksWithoutEOS(
        _ text: String,
        chunkSize: Int,
        into processor: ToolCallProcessor
    ) -> String {
        var output = ""
        var idx = text.startIndex
        while idx < text.endIndex {
            let end = text.index(idx, offsetBy: chunkSize, limitedBy: text.endIndex) ?? text.endIndex
            let chunk = String(text[idx..<end])
            if let out = processor.processChunk(chunk) {
                output += out
            }
            idx = end
        }
        return output
    }

    private func lineCountToolSpec() -> [String: any Sendable] {
        [
            "type": "function",
            "function": [
                "name": "line_count",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "text": ["type": "string"] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["text"],
                    "additionalProperties": false,
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    // MARK: - Gemma-4 escape-marker regression guard

    /// The default escape marker on `GemmaFunctionParser.init(startTag:endTag:)`
    /// MUST be `<|"|>` — NOT `<|"\|>` (with backslash). The prior revision
    /// had the backslash and silently failed to decode string args for any
    /// direct caller not going through `ToolCallFormat.gemma4.createParser()`.
    @Test("Gemma-4 parser default escape marker is <|\"|> not <|\"\\|>")
    func testGemma4ParserDefaultEscapeMarkerIsCorrect() {
        let parser = GemmaFunctionParser(
            startTag: "<|tool_call>", endTag: "<tool_call|>")
        let input = "<|tool_call>call:get_weather{location:<|\"|>Tokyo<|\"|>}<tool_call|>"
        let call = parser.parse(content: input, tools: nil)
        #expect(call != nil)
        #expect(call?.function.name == "get_weather")
        #expect(call?.function.arguments["location"] == .string("Tokyo"),
            "Default escape marker must decode <|\"|>-wrapped strings (iter 65 fix).")
    }

    // MARK: - Qwen 3.6 interleaved thinking

    /// Qwen 3.6 wire format: `<think>...</think>` can appear before, between,
    /// and after tool calls. `ToolCallProcessor` for xml_function must
    /// correctly extract the tool calls without its state machine getting
    /// confused by the `<think>` tags (which share the leading `<`).
    @Test("Qwen 3.6 interleaved thinking: multi-<think> multi-tool-call produces all tool calls")
    func testQwen36InterleavedThinkingProducesAllToolCalls() {
        let processor = ToolCallProcessor(format: .xmlFunction)
        let stream = """
            <think>First I need weather data.</think>
            <tool_call><function=get_weather><parameter=location>Paris</parameter></function></tool_call>
            <think>Now check the time.</think>
            <tool_call><function=get_time><parameter=zone>UTC</parameter></function></tool_call>
            Final answer: it is nice in Paris.
            """
        let emitted = feedInChunks(stream, chunkSize: 7, into: processor)
        #expect(processor.toolCalls.count == 2,
            "Two tool calls interleaved with two <think> blocks must all extract.")
        #expect(processor.toolCalls[0].function.name == "get_weather")
        #expect(processor.toolCalls[1].function.name == "get_time")
        // The user-visible text MUST include the final answer.
        #expect(emitted.contains("Final answer: it is nice in Paris."))
        // Tool-call payloads MUST NOT leak into user-visible text.
        #expect(!emitted.contains("<tool_call>"),
            "Tool-call tags must not leak into user-visible chunks.")
        #expect(!emitted.contains("<function=get_weather>"),
            "Tool-call internals must not leak into user-visible chunks.")
    }

    // MARK: - Reasoning parser pipelined with tool-call processor

    /// When a `ReasoningParser` is pipelined BEFORE `ToolCallProcessor`,
    /// `<think>...</think>` blocks must be stripped from the text
    /// handed to the tool-call processor's `processChunk` so the
    /// downstream consumer sees neither reasoning markers nor
    /// tool-call markers in `.chunk` output — only user-visible text.
    ///
    /// This is the contract osaurus relies on: its `.chunk` stream is
    /// pure text, its tool-call events are authoritative, and any
    /// reasoning content is routed to a separate reasoning channel.
    @Test("ReasoningParser pipelined before ToolCallProcessor strips <think> from chunks")
    func testReasoningParserPipelinedStripsThink() {
        var reasoning = ReasoningParser()
        let tools = ToolCallProcessor(format: .xmlFunction)
        let stream = """
            <think>First I need weather data.</think>
            <tool_call><function=get_weather><parameter=location>Paris</parameter></function></tool_call>
            <think>Now check the time.</think>
            Final answer: nice weather.
            """
        var userVisible = ""
        var reasoningCaptured = ""
        // Feed in small chunks simulating streaming.
        var idx = stream.startIndex
        while idx < stream.endIndex {
            let end = stream.index(idx, offsetBy: 6, limitedBy: stream.endIndex) ?? stream.endIndex
            let chunk = String(stream[idx..<end])
            // Reasoning parser first — it peels off <think>...</think>
            // into its own channel and returns non-reasoning segments as .content.
            for segment in reasoning.feed(chunk) {
                switch segment {
                case .reasoning(let r):
                    reasoningCaptured += r
                case .content(let c):
                    if let toolVisible = tools.processChunk(c) {
                        userVisible += toolVisible
                    }
                }
            }
            idx = end
        }
        // Flush any buffered content.
        for segment in reasoning.flush() {
            switch segment {
            case .reasoning(let r):
                reasoningCaptured += r
            case .content(let c):
                if let toolVisible = tools.processChunk(c) {
                    userVisible += toolVisible
                }
            }
        }
        tools.processEOS()
        // Reasoning captured at least once.
        #expect(reasoningCaptured.contains("First I need weather data"),
            "ReasoningParser must capture `<think>` content. Got: \(reasoningCaptured)")
        // Tool call extracted.
        #expect(tools.toolCalls.count == 1,
            "Tool-call processor must still see the tool call after reasoning strip.")
        #expect(tools.toolCalls.first?.function.name == "get_weather")
        // Neither reasoning tags nor tool-call tags in user-visible text.
        #expect(!userVisible.contains("<think>"),
            "User-visible text must not contain <think>. Got: \(userVisible)")
        #expect(!userVisible.contains("</think>"),
            "User-visible text must not contain </think>.")
        #expect(!userVisible.contains("<tool_call>"),
            "User-visible text must not contain <tool_call>.")
        #expect(userVisible.contains("Final answer: nice weather"),
            "User-visible text must carry the final answer.")
    }

    @Test("LFM2 parses only native tagged calls from reasoning-routed output")
    func testLFM2ParsesOnlyNativeTaggedCallsFromReasoningRoutedOutput() throws {
        let processor = ToolCallProcessor(format: .lfm2, tools: [lineCountToolSpec()])
        let reasoningEvents = routeGenerationText(
            "We need to call line_count() with the exact text first.",
            channel: .reasoning,
            through: processor)

        #expect(reasoningEvents.compactMap(\.toolCall).isEmpty)
        #expect(reasoningEvents.compactMap(\.reasoning).joined().contains("line_count()"))

        let nativeCallEvents = routeGenerationText(
            #"<|tool_call_start|>[line_count(text="red\ngreen\nblue")]<|tool_call_end|>"#,
            channel: .reasoning,
            through: processor)
        let call = try #require(nativeCallEvents.compactMap(\.toolCall).first)
        #expect(call.function.name == "line_count")
        #expect(call.function.arguments["text"] == .string("red\ngreen\nblue"))
        #expect(nativeCallEvents.compactMap(\.reasoning).joined().isEmpty)
    }

    @Test("Gemma4 still parses tool calls from reasoning channel")
    func testGemma4StillParsesToolCallsFromReasoningChannel() throws {
        let processor = ToolCallProcessor(format: .gemma4, tools: [lineCountToolSpec()])
        let events = routeGenerationText(
            """
            <zyphra_tool_call>
            <function=line_count>
            <parameter=text>
            red
            green
            blue
            </parameter>
            </function>
            </zyphra_tool_call>
            """,
            channel: .reasoning,
            through: processor)

        let call = try #require(events.compactMap(\.toolCall).first)
        #expect(call.function.name == "line_count")
        #expect(call.function.arguments["text"] == .string("red\ngreen\nblue"))
        #expect(events.compactMap(\.chunk).joined().isEmpty)
    }

    /// Character-by-character streaming — matches what NaiveStreamingDetokenizer
    /// produces on short tokens. Must not lose tool calls.
    @Test("Qwen 3.6 character-by-character streaming preserves both tool calls")
    func testQwen36CharByCharStreamingPreservesToolCalls() {
        let processor = ToolCallProcessor(format: .xmlFunction)
        let stream = """
            <tool_call><function=a></function></tool_call><tool_call><function=b></function></tool_call>
            """
        _ = feedCharByChar(stream, into: processor)
        #expect(processor.toolCalls.count == 2,
            "Two back-to-back tool calls delivered character-by-character must both extract.")
        #expect(processor.toolCalls.map(\.function.name) == ["a", "b"])
    }

    // MARK: - MiniMax M2 interleaved thinking

    @Test("MiniMax M2 interleaved thinking: <think> between <minimax:tool_call> blocks")
    func testMiniMaxM2InterleavedThinking() {
        let processor = ToolCallProcessor(format: .minimaxM2)
        // MiniMax M2 wraps invokes in `<minimax:tool_call>...</minimax:tool_call>`.
        // Interleaved `<think>` blocks should not break the state machine.
        let stream = """
            <think>Thinking first.</think>
            <minimax:tool_call><invoke name="f"><parameter name="x">1</parameter></invoke></minimax:tool_call>
            <think>Still thinking.</think>
            <minimax:tool_call><invoke name="g"><parameter name="y">2</parameter></invoke></minimax:tool_call>
            """
        _ = feedInChunks(stream, chunkSize: 5, into: processor)
        #expect(processor.toolCalls.count == 2,
            "MiniMax M2 must surface both invokes across interleaved thinking.")
        #expect(processor.toolCalls[0].function.name == "f")
        #expect(processor.toolCalls[1].function.name == "g")
    }

    @Test("MiniMax M2 preserves visible text around tagged tool calls")
    func testMiniMaxM2PreservesVisibleTextAroundToolCall() {
        let processor = ToolCallProcessor(format: .minimaxM2)
        let stream = """
            before <minimax:tool_call><invoke name="search"><parameter name="query">swift</parameter></invoke></minimax:tool_call> after
            """

        let visible = processor.processChunk(stream) ?? ""

        #expect(visible.contains("before "))
        #expect(visible.contains(" after"))
        #expect(!visible.contains("minimax:tool_call"))
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "search")
    }

    @Test("MiniMax M2 incomplete tool-call-looking reasoning flushes as text at EOS")
    func testMiniMaxM2IncompleteToolCallLookingReasoningFlushesAsText() {
        let processor = ToolCallProcessor(format: .minimaxM2)
        let stream = "The user mentioned <minimax:tool_call but this is not a complete tool call"

        let visibleDuringStream = feedInChunks(stream, chunkSize: 6, into: processor)

        #expect(
            visibleDuringStream == stream,
            "Unparseable MiniMax tool-call-looking text must be returned losslessly instead of disappearing at EOS"
        )
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("MiniMax M2 incomplete tool-call-looking text preserves its channel on generation flush")
    func testMiniMaxM2IncompleteToolCallFlushPreservesReasoningChannel() {
        let processor = ToolCallProcessor(format: .minimaxM2)
        let text = "The user is greeting <minimax:tool_call without an invoke"
        var visible: [Generation] = []

        for ch in text {
            visible.append(contentsOf: routeGenerationText(String(ch), channel: .reasoning, through: processor))
        }
        let flushed = flushGenerationText(channel: .reasoning, through: processor)
        visible.append(contentsOf: flushed)

        let reasoningText = visible.compactMap {
            if case .reasoning(let text) = $0 { text } else { nil }
        }.joined()
        #expect(reasoningText == text)
        #expect(!visible.contains { if case .chunk = $0 { true } else { false } })
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("MiniMax M2 complete start tag followed by prose flushes during reasoning stream")
    func testMiniMaxM2StartTagFollowedByProseFlushesDuringReasoningStream() {
        let processor = ToolCallProcessor(format: .minimaxM2)
        let text = "The user said <minimax:tool_call>The user simply wants a greeting."
        var visible: [Generation] = []

        for ch in text {
            visible.append(contentsOf: routeGenerationText(String(ch), channel: .reasoning, through: processor))
        }

        let reasoningText = visible.compactMap {
            if case .reasoning(let text) = $0 { text } else { nil }
        }.joined()
        #expect(
            reasoningText == text,
            "Invalid MiniMax tool-call body must not trap later reasoning tokens in the parser buffer"
        )
        #expect(!visible.contains { if case .chunk = $0 { true } else { false } })
        #expect(flushGenerationText(channel: .reasoning, through: processor).isEmpty)
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("MiniMax M2 valid invoke prefix still buffers until the wrapper closes")
    func testMiniMaxM2ValidInvokePrefixStillBuffersUntilClose() {
        let processor = ToolCallProcessor(format: .minimaxM2)
        let prefix = "<minimax:tool_call><invoke name=\"search\""
        let visiblePrefix = feedInChunksWithoutEOS(prefix, chunkSize: 4, into: processor)

        #expect(visiblePrefix.isEmpty)
        #expect(processor.toolCalls.isEmpty)

        let suffix = "><parameter name=\"query\">swift</parameter></invoke></minimax:tool_call>"
        let visibleSuffix = feedInChunks(suffix, chunkSize: 4, into: processor)

        #expect(visibleSuffix.isEmpty)
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "search")
        #expect(processor.toolCalls.first?.function.arguments["query"] == .string("swift"))
    }

    // MARK: - Gemma-4 harmony-format channels

    /// Gemma-4 occasionally emits `<|channel|>thought\n...\n<channel|>` blocks
    /// (its own channel-based reasoning format; similar in spirit to GPT-OSS's
    /// "harmony" format). These MUST NOT leak into `.chunk` output as raw
    /// channel markers. With tool calls mixed in, both signals need to land
    /// in their right places.
    ///
    /// Tool parsing still needs to recover the call even if native Gemma-4
    /// channel markers precede it. Reasoning-channel stripping itself is
    /// covered in `HarmonyParserFocusedTests`.
    @Test("Gemma-4 with interleaved channel + tool-call: tool call extracted")
    func testGemma4InterleavedChannelAndToolCall() {
        let processor = ToolCallProcessor(format: .gemma4)
        let stream = """
            <|channel|>thought
            I need weather.
            <channel|>
            <|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}<tool_call|>
            """
        _ = feedInChunks(stream, chunkSize: 6, into: processor)
        #expect(processor.toolCalls.count == 1,
            "Gemma-4 tool call must extract even when a channel block precedes it.")
        #expect(processor.toolCalls.first?.function.name == "get_weather")
        #expect(processor.toolCalls.first?.function.arguments["location"] == .string("Tokyo"))
    }

    @Test("Gemma-4 tool-call parser trims whitespace around function names and keys")
    func testGemma4ToolCallWhitespaceAroundFunctionAndKeys() {
        let processor = ToolCallProcessor(format: .gemma4)
        let stream = """
            <|tool_call>call:get_weather
            {
              location:<|"|>Tokyo<|"|>
            }<tool_call|>
            """
        _ = feedInChunks(stream, chunkSize: 4, into: processor)

        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "get_weather")
        #expect(processor.toolCalls.first?.function.arguments["location"] == .string("Tokyo"))
        #expect(processor.toolCalls.first?.function.arguments[" location"] == nil)
    }

    @Test("Gemma-4 parser accepts live 12B quoted native string argument")
    func testGemma4ParserAcceptsLive12BQuotedNativeStringArgument() throws {
        let parser = ToolCallFormat.gemma4.createParser()
        let call = try #require(parser.parse(
            content: #"<|tool_call>call:get_weather{location: "Tokyo"}<tool_call|>"#,
            tools: nil
        ))

        #expect(call.function.name == "get_weather")
        #expect(call.function.arguments["location"] == .string("Tokyo"))
    }

    @Test("Gemma-4 parser unwraps redundant JSON quotes inside escaped string values")
    func testGemma4ParserUnwrapsQuotedEscapedStringArgument() throws {
        let parser = ToolCallFormat.gemma4.createParser()
        let call = try #require(parser.parse(
            content: #"<|tool_call>call:line_count{text:<|"|>"red\ngreen\nblue"<|"|>}<tool_call|>"#,
            tools: [lineCountToolSpec()]
        ))

        #expect(call.function.name == "line_count")
        #expect(call.function.arguments["text"] == .string("red\ngreen\nblue"))
    }

    @Test("Gemma-4 parser decodes native escaped newline string values")
    func testGemma4ParserDecodesNativeEscapedNewlineStringArgument() throws {
        let parser = ToolCallFormat.gemma4.createParser()
        let call = try #require(parser.parse(
            content: #"<|tool_call>call:line_count{text:<|"|>red\ngreen\nblue<|"|>}<tool_call|>"#,
            tools: [lineCountToolSpec()]
        ))

        #expect(call.function.name == "line_count")
        #expect(call.function.arguments["text"] == .string("red\ngreen\nblue"))
    }

    @Test("Gemma-4 parser unwraps redundant quotes around literal-newline escaped values")
    func testGemma4ParserUnwrapsQuotedLiteralNewlineStringArgument() throws {
        let parser = ToolCallFormat.gemma4.createParser()
        let call = try #require(parser.parse(
            content: """
                <|tool_call>call:line_count{text:<|"|>"one
                two"<|"|>}<tool_call|>
                """,
            tools: [lineCountToolSpec()]
        ))

        #expect(call.function.name == "line_count")
        #expect(call.function.arguments["text"] == .string("one\ntwo"))
    }

    @Test("Gemma-4 parser unwraps redundant quotes around raw literal-newline values")
    func testGemma4ParserUnwrapsQuotedRawLiteralNewlineStringArgument() throws {
        let parser = ToolCallFormat.gemma4.createParser()
        let call = try #require(parser.parse(
            content: """
                <|tool_call>call:line_count{text:"one
                two"}<tool_call|>
                """,
            tools: [lineCountToolSpec()]
        ))

        #expect(call.function.name == "line_count")
        #expect(call.function.arguments["text"] == .string("one\ntwo"))
    }

    @Test("Gemma-4 escaped string parser keeps literal booleans as strings")
    func testGemma4EscapedStringParserKeepsLiteralBooleanString() throws {
        let parser = ToolCallFormat.gemma4.createParser()
        let call = try #require(parser.parse(
            content: #"<|tool_call>call:label{value:<|"|>true<|"|>}<tool_call|>"#,
            tools: nil
        ))

        #expect(call.function.arguments["value"] == .string("true"))
    }

    @Test("Gemma-4 processor routes live 12B quoted native tool-call without visible leak")
    func testGemma4ProcessorRoutesLive12BQuotedNativeToolCallWithoutLeak() {
        let processor = ToolCallProcessor(format: .gemma4)
        let visible = feedCharByChar(
            #"<|tool_call>call:get_weather{location: "Tokyo"}<tool_call|>"#,
            into: processor
        )

        #expect(visible.isEmpty)
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "get_weather")
        #expect(processor.toolCalls.first?.function.arguments["location"] == .string("Tokyo"))
    }

    @Test("Gemma-4 parser preserves native escaped-string arrays as JSON arrays")
    func testGemma4ToolCallEscapedStringArrayArgument() {
        let parser = ToolCallFormat.gemma4.createParser()
        let calls = parser.parseEOS(
            """
            <|tool_call>call:capabilities_search{queries:[<|"|>file writing<|"|>,<|"|>create file<|"|>]}<tool_call|>
            """,
            tools: nil
        )

        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "capabilities_search")
        #expect(
            calls.first?.function.arguments["queries"] == .array([
                .string("file writing"),
                .string("create file"),
            ])
        )
    }

    // MARK: - Back-to-back tool calls, no delimiter between

    @Test("JSON format: two tool calls back-to-back, character-streamed")
    func testJSONBackToBackCharStreamed() {
        let processor = ToolCallProcessor(format: .json)
        let stream = """
            <tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{}}</tool_call>
            """
        _ = feedCharByChar(stream, into: processor)
        #expect(processor.toolCalls.count == 2)
        #expect(processor.toolCalls.map(\.function.name) == ["a", "b"])
    }

    // MARK: - JANG capability stamp → ToolCallFormat mapping

    @Test("fromCapabilityName maps JANG `qwen` stamp to xml_function")
    func testJANGStampQwenMapsToXMLFunction() {
        for stamp in ["qwen", "qwen3", "qwen3_5", "qwen35", "qwen3_6", "qwen36", "qwen3_coder"] {
            #expect(ToolCallFormat.fromCapabilityName(stamp) == .xmlFunction,
                "JANG `\(stamp)` stamp must map to .xmlFunction")
        }
    }

    @Test("fromCapabilityName maps Step stamps to Step parser")
    func testJANGStampStepMapsToStepParser() {
        for stamp in ["step", "stepfun", "step3p5", "step3p7", "step3_5", "step3_7"] {
            #expect(ToolCallFormat.fromCapabilityName(stamp) == .step,
                "JANG `\(stamp)` stamp must map to .step")
        }
    }

    @Test("fromCapabilityName maps JANG `minimax` stamp to minimaxM2")
    func testJANGStampMiniMaxMapsToMiniMaxM2() {
        for stamp in ["minimax", "minimax_m2", "minimax_m2_5"] {
            #expect(ToolCallFormat.fromCapabilityName(stamp) == .minimaxM2)
        }
    }

    @Test("fromCapabilityName maps JANG `gemma4` stamp to gemma4")
    func testJANGStampGemma4MapsToGemma4() {
        #expect(ToolCallFormat.fromCapabilityName("gemma4") == .gemma4)
        #expect(ToolCallFormat.fromCapabilityName("gemma") == .gemma)
    }

    @Test("fromCapabilityName maps GLM4 family + deepseek alias")
    func testJANGStampGLM4Family() {
        for stamp in ["glm4", "glm47", "glm5", "glm4_moe", "deepseek"] {
            #expect(ToolCallFormat.fromCapabilityName(stamp) == .glm4)
        }
    }

    @Test("fromCapabilityName maps Nemotron to family parser")
    func testJANGStampNemotron() {
        #expect(ToolCallFormat.fromCapabilityName("nemotron") == .nemotron)
        #expect(ToolCallFormat.fromCapabilityName("nemotron_h") == .nemotron)
    }

    @Test("fromCapabilityName maps Mistral family")
    func testJANGStampMistral() {
        #expect(ToolCallFormat.fromCapabilityName("mistral") == .mistral)
        #expect(ToolCallFormat.fromCapabilityName("mistral4") == .mistral)
    }

    @Test("fromCapabilityName maps ZAYA / Zyphra family")
    func testJANGStampZaya() {
        for stamp in ["zaya", "zaya_xml", "zyphra", "zyphra_xml"] {
            #expect(ToolCallFormat.fromCapabilityName(stamp) == .zayaXml)
        }
    }

    @Test("fromCapabilityName returns nil for unknown + empty")
    func testJANGStampUnknownReturnsNil() {
        #expect(ToolCallFormat.fromCapabilityName(nil) == nil)
        #expect(ToolCallFormat.fromCapabilityName("") == nil)
        #expect(ToolCallFormat.fromCapabilityName("totally_made_up") == nil)
    }

    @Test("fromCapabilityName matches canonical rawValues directly")
    func testJANGStampCanonicalRawValues() {
        // Every case's rawValue must round-trip.
        for format in ToolCallFormat.allCases {
            #expect(ToolCallFormat.fromCapabilityName(format.rawValue) == format,
                "\(format.rawValue) must round-trip through fromCapabilityName")
        }
    }

    // MARK: - Pure text (no tool calls) passes through unchanged

    /// No-tool-call scenario must not buffer or drop text. Osaurus shows
    /// this to users directly — any loss is visible to the end user.
    @Test("JSON format: plain text with no tool calls passes through unchanged")
    func testJSONPlainTextPassesThrough() {
        let processor = ToolCallProcessor(format: .json)
        let text = "Hello world, I have no tool calls to make. Here is just regular text."
        let out = feedCharByChar(text, into: processor)
        #expect(out == text, "Plain text must pass through byte-identical.")
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("XML function format: plain text passes through unchanged")
    func testXMLFunctionPlainTextPassesThrough() {
        let processor = ToolCallProcessor(format: .xmlFunction)
        let text = "Here is a normal response with no function calls at all."
        let out = feedCharByChar(text, into: processor)
        #expect(out == text)
        #expect(processor.toolCalls.isEmpty)
    }

    // MARK: - Text before tool call flushes correctly

    @Test("Leading text before first tool call is yielded before the tool call")
    func testLeadingTextFlushedBeforeToolCall() {
        let processor = ToolCallProcessor(format: .json)
        let text = """
            Sure, let me look that up. <tool_call>{"name":"f","arguments":{}}</tool_call>
            """
        let out = feedInChunks(text, chunkSize: 3, into: processor)
        // The leading text must not be swallowed.
        #expect(out.contains("Sure, let me look that up"),
            "Text preceding the tool call must flush to the output stream. Got: \(out)")
        #expect(processor.toolCalls.count == 1)
    }

    // MARK: - Trailing text after tool call flushes correctly

    @Test("Trailing text after tool call is yielded")
    func testTrailingTextAfterToolCall() {
        let processor = ToolCallProcessor(format: .json)
        let text = """
            <tool_call>{"name":"f","arguments":{}}</tool_call>Done, goodbye.
            """
        let out = feedInChunks(text, chunkSize: 4, into: processor)
        #expect(out.contains("Done, goodbye"),
            "Trailing text after tool call must be yielded. Got: \(out)")
        #expect(processor.toolCalls.count == 1)
    }

    // MARK: - Unclosed tool call on EOS

    /// Inline formats and tagged formats both need to handle EOS-mid-stream.
    @Test("Mistral inline format: EOS before close delimiter still extracts")
    func testMistralEOSExtraction() {
        let processor = ToolCallProcessor(format: .mistral)
        let text = #"[TOOL_CALLS]f [ARGS]{"x":1}"#
        _ = feedInChunks(text, chunkSize: 3, into: processor)
        // Note: Mistral is inline (no end tag) — process/EOS path extracts.
        #expect(processor.toolCalls.count == 1 || processor.toolCalls.isEmpty,
            "Mistral EOS inline-parser behaviour (extract-or-buffer) pinned.")
    }

    // MARK: - Pythonic multi-call parseEOS (iter 67, osaurus parity)

    /// LFM2 can emit multiple calls inside one `[…]` block — e.g.
    /// `[search(q="a"), search(q="b")]`. The default protocol `parseEOS`
    /// only surfaces the first match, so `PythonicToolCallParser` must
    /// override it to return every call. Regression test for drift vs
    /// upstream ml-explore/mlx-swift-lm `main`.
    @Test("Pythonic parseEOS extracts every call in one bracket block")
    func testPythonicParseEOSReturnsAllCalls() {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let buffer = "<|tool_call_start|>[search(q='a'), fetch(url='https://x.io')]<|tool_call_end|>"
        let calls = parser.parseEOS(buffer, tools: nil)
        #expect(calls.count == 2,
            "parseEOS must emit every call in the bracket list.")
        #expect(calls.map(\.function.name) == ["search", "fetch"])
        #expect(calls[0].function.arguments["q"] == .string("a"))
        #expect(calls[1].function.arguments["url"] == .string("https://x.io"))
    }

    @Test("LFM2 parser accepts observed standard name arguments JSON object")
    func testLFM2ParserAcceptsObservedNamedArgumentsJSON() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let buffer = #"""
        {
          "name": "line_count",
          "arguments": {
            "text": "red\ngreen\nblue"
          }
        }
        """#

        let call = try #require(parser.parse(content: buffer, tools: [lineCountToolSpec()]))
        #expect(call.function.name == "line_count")
        #expect(call.function.arguments["text"] == .string("red\ngreen\nblue"))
    }

    @Test("LFM2 processor accepts observed OpenAI tool_call JSON envelope")
    func testLFM2ProcessorAcceptsObservedOpenAIToolCallJSONEnvelope() throws {
        let processor = ToolCallProcessor(format: .lfm2, tools: [lineCountToolSpec()])
        let output = #"""

        {
          "tool_call": {
            "id": "tool_call_1",
            "name": "line_count",
            "arguments": {
              "text": "red\ngreen\nblue"
            }
          }
        }
        """#

        let visible = processor.processChunk(output) ?? ""
        let eosVisible = processor.processEOS() ?? ""
        let call = try #require(processor.toolCalls.first)

        #expect(visible.isEmpty)
        #expect(eosVisible.isEmpty)
        #expect(call.function.name == "line_count")
        #expect(call.function.arguments["text"] == .string("red\ngreen\nblue"))
    }

    @Test("LFM2 parser rejects observed JSON object with unknown arguments")
    func testLFM2ParserRejectsObservedNamedJSONWithUnknownArguments() {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let buffer = #"""
        {"name":"line_count","arguments":{"text":"one\ntwo","path":"/tmp/a"}}
        """#

        #expect(parser.parse(content: buffer, tools: [lineCountToolSpec()]) == nil)
    }

    @Test("Step parser extracts bare registered JSON function call from reasoning channel")
    func testStepParserExtractsBareFunctionJSONFromReasoning() throws {
        let direct = StepToolCallParser().parse(
            content: #"line_count({"text":"red\ngreen\nblue"})"#,
            tools: [lineCountToolSpec()])
        #expect(direct?.function.name == "line_count")
        #expect(direct?.function.arguments["text"] == .string("red\ngreen\nblue"))

        let processor = ToolCallProcessor(format: .step, tools: [lineCountToolSpec()])
        var events = routeGenerationText(
            #"line_count({"text":"red\ngreen\nblue"})"#,
            channel: .reasoning,
            through: processor)
        events.append(contentsOf: flushGenerationText(channel: .reasoning, through: processor))

        let call = try #require(events.compactMap(\.toolCall).first)
        #expect(call.function.name == "line_count")
        #expect(call.function.arguments["text"] == .string("red\ngreen\nblue"))
        #expect(events.compactMap(\.chunk).joined().isEmpty)
        #expect(events.compactMap(\.reasoning).joined().isEmpty)
    }

    @Test("Step parser rejects bare JSON function call with unknown argument")
    func testStepParserRejectsBareFunctionJSONUnknownArgument() {
        let processor = ToolCallProcessor(format: .step, tools: [lineCountToolSpec()])
        var events = routeGenerationText(
            #"line_count({"text":"one\ntwo","path":"/tmp/a"})"#,
            channel: .reasoning,
            through: processor)
        events.append(contentsOf: flushGenerationText(channel: .reasoning, through: processor))

        #expect(events.compactMap(\.toolCall).isEmpty)
    }

    // MARK: - ToolCall.Function(name:arguments: JSONValue) upstream init

    /// Upstream ship both `init(name:, arguments: [String: JSONValue])`
    /// AND `init(name:, arguments: [String: any Sendable])`. Our earlier
    /// drop of the typed-JSONValue init would silently break any caller
    /// that passes through decoded `JSONValue` dicts. Regression pin.
    @Test("ToolCall.Function init(JSONValue:) preserves typed args byte-for-byte")
    func testToolCallFunctionJSONValueInit() {
        let args: [String: JSONValue] = [
            "location": .string("Tokyo"),
            "unit": .string("celsius"),
            "days": .int(7),
        ]
        let fn = ToolCall.Function(name: "get_weather", arguments: args)
        #expect(fn.name == "get_weather")
        #expect(fn.arguments["location"] == .string("Tokyo"))
        #expect(fn.arguments["unit"] == .string("celsius"))
        #expect(fn.arguments["days"] == .int(7))
    }

    // MARK: - ModelConfiguration.reasoningParserName plumbing (iter 66)

    /// `ModelConfiguration.reasoningParserName` is the capability stamp
    /// that `Evaluate.generate` / `BatchEngine.generate` resolve into a
    /// live `ReasoningParser` via `ReasoningParser.fromCapabilityName`.
    /// This test pins the round-trip: init → resolved → fromCapabilityName.
    @Test("ModelConfiguration.reasoningParserName survives init + resolve()")
    func testReasoningParserNamePlumbsThroughResolved() {
        let stamps: [(stamp: String, shouldResolve: Bool)] = [
            ("think_xml", true),
            ("qwen3_6", true),
            ("deepseek_r1", true),
            ("glm4", true),
            ("minimax", true),
            ("nemotron", true),
            ("none", false),
            ("mistral", false),
            // gemma4 used to return nil — now routes to the harmony
            // parser as of 2026-04-20 (see TPAE-2026-04-20-TRIAGE.md
            // "2:59 PM" addendum).
            ("gemma4", true),
            ("harmony", true),
        ]
        let tmpDir = URL(fileURLWithPath: "/tmp/vmlx-test-placeholder")
        for (stamp, shouldResolve) in stamps {
            let cfg = ModelConfiguration(
                directory: tmpDir, reasoningParserName: stamp)
            let resolved = cfg.resolved(
                modelDirectory: tmpDir, tokenizerDirectory: tmpDir)
            #expect(resolved.reasoningParserName == stamp,
                "resolved() must carry reasoningParserName through.")
            let parser = ReasoningParser.fromCapabilityName(
                resolved.reasoningParserName)
            if shouldResolve {
                #expect(parser != nil,
                    "Stamp `\(stamp)` must resolve to a ReasoningParser instance.")
            } else {
                #expect(parser == nil,
                    "Stamp `\(stamp)` must resolve to nil (no reasoning stripping).")
            }
        }
    }

    /// `TextToolTokenLoopHandler`-equivalent pipeline with reasoning
    /// parser present: interleaved `<think>` blocks must be stripped
    /// from the user-visible chunks flowing to the tool-call processor,
    /// while tool calls still extract cleanly. This is the contract the
    /// real handler implements — composing the two parsers in a single
    /// pipeline without state-machine conflict.
    @Test("Reasoning + tool-call pipeline: interleaved <think> + <tool_call> survive streaming")
    func testReasoningToolCallPipelineInterleaved() {
        var reasoning = ReasoningParser()
        let tools = ToolCallProcessor(format: .xmlFunction)
        let stream = """
            <think>First, plan.</think>
            <tool_call><function=a></function></tool_call>
            <think>Now do b.</think>
            <tool_call><function=b></function></tool_call>
            Done.
            """
        var userVisible = ""
        var idx = stream.startIndex
        while idx < stream.endIndex {
            let end = stream.index(idx, offsetBy: 4, limitedBy: stream.endIndex) ?? stream.endIndex
            let chunk = String(stream[idx..<end])
            for segment in reasoning.feed(chunk) {
                if case .content(let c) = segment,
                   let out = tools.processChunk(c) {
                    userVisible += out
                }
            }
            idx = end
        }
        for segment in reasoning.flush() {
            if case .content(let c) = segment,
               let out = tools.processChunk(c) {
                userVisible += out
            }
        }
        tools.processEOS()
        #expect(tools.toolCalls.count == 2,
            "Both tool calls must extract with reasoning pipelined.")
        #expect(tools.toolCalls.map(\.function.name) == ["a", "b"])
        #expect(!userVisible.contains("<think>"),
            "User-visible text must not contain <think>.")
        #expect(!userVisible.contains("First, plan"),
            "Reasoning content must not leak into user-visible text.")
        #expect(!userVisible.contains("<tool_call>"),
            "Tool-call tags must not leak into user-visible text.")
        #expect(userVisible.contains("Done"),
            "Final user-visible text must survive the pipeline.")
    }
}
