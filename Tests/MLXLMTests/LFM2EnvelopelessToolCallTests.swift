// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import MLXLMCommon

/// LFM2.5-VL-3B emits the tool-call body its own chat template defines and omits
/// the envelope around it:
///
///     wanted: <|tool_call_start|>[lookup_launch_status(launch_id='LAGUNA-77')]<|tool_call_end|>
///     got:    lookup_launch_status(launch_id='LAGUNA-77')
///
/// Measured live on ALL FOUR 3B quants — JANG_2L, JANG_4M, JANG_6M and MXFP8 —
/// via `BENCH_AGENTIC_TOOL`, each reporting `toolCalls=0` with the call leaking
/// to the user as prose. So it is the bundle's behaviour, not quantization
/// damage.
///
/// These drive the real `ToolCallProcessor` (not the parser in isolation) and
/// they PASS on today's tree: given the format and the offered tool schemas, the
/// processor recovers both live-observed shapes — the pythonic one above and the
/// bare JSON object `BENCH_BATCH_TOOLCALL` produced from the same bundle — in
/// one chunk, in six, and one character at a time.
///
/// That is the point of the file. The tool-call layer is NOT what is wrong for
/// this bundle, so anyone chasing the live `toolCalls=0` should look at what the
/// generation path hands the processor rather than at parsing. Equally, if a
/// future change breaks the inline fallbacks these go red, which is coverage the
/// shapes did not have before.
@Suite("LFM2 envelopeless pythonic tool call")
struct LFM2EnvelopelessToolCallTests {

    private static var tools: [[String: any Sendable]] {
        [[
            "type": "function",
            "function": [
                "name": "lookup_launch_status",
                "description": "Look up a launch",
                "parameters": [
                    "type": "object",
                    "properties": ["launch_id": ["type": "string"]],
                    "required": ["launch_id"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]]
    }

    /// Feed `text` through the processor the way generation does, then close the
    /// stream, and report what the user would have seen plus what was parsed.
    private func run(_ text: String, chunks: Int = 1) -> (visible: String, calls: [ToolCall]) {
        let processor = ToolCallProcessor(format: .lfm2, tools: Self.tools)
        var visible = ""
        let pieces: [String]
        if chunks <= 1 {
            pieces = [text]
        } else {
            // Split into roughly equal pieces to exercise partial buffering.
            let size = max(1, text.count / chunks)
            pieces = stride(from: 0, to: text.count, by: size).map { offset in
                let start = text.index(text.startIndex, offsetBy: offset)
                let end = text.index(start, offsetBy: size, limitedBy: text.endIndex)
                    ?? text.endIndex
                return String(text[start ..< end])
            }
        }
        for piece in pieces {
            if let out = processor.processChunk(piece) { visible += out }
        }
        if let tail = processor.processEOS() { visible += tail }
        return (visible, processor.toolCalls)
    }

    // MARK: - The live failure

    @Test("the live single-quoted call is parsed, not leaked as prose")
    func recoversSingleQuoted() {
        let (visible, calls) = run("lookup_launch_status(launch_id='LAGUNA-77')")
        #expect(calls.count == 1, "the call leaked to the user as text: \(visible)")
        #expect(calls.first?.function.name == "lookup_launch_status")
        #expect(calls.first?.function.arguments["launch_id"]?.anyValue as? String == "LAGUNA-77")
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    /// MXFP8 and JANG_2L emitted the double-quoted spelling of the same call.
    @Test("the live double-quoted call is parsed")
    func recoversDoubleQuoted() {
        let (_, calls) = run("lookup_launch_status(launch_id=\"LAGUNA-77\")")
        #expect(calls.first?.function.arguments["launch_id"]?.anyValue as? String == "LAGUNA-77")
    }

    /// Real generation arrives token by token, not as one string.
    @Test("the call is parsed when it arrives across several chunks")
    func recoversStreamed() {
        let (visible, calls) = run("lookup_launch_status(launch_id='LAGUNA-77')", chunks: 6)
        #expect(calls.count == 1, "streamed call leaked as text: \(visible)")
        #expect(calls.first?.function.arguments["launch_id"]?.anyValue as? String == "LAGUNA-77")
    }

    /// The harshest realistic shape: one character per chunk, which is closer to
    /// token-by-token delivery than an even split and is where a buffering gate
    /// that only arms after the first chunk would show up.
    @Test("the call is parsed when it arrives one character at a time")
    func recoversCharByChar() {
        let text = "lookup_launch_status(launch_id='LAGUNA-77')"
        let (visible, calls) = run(text, chunks: text.count)
        #expect(calls.count == 1, "char-streamed call leaked as text: \(visible)")
        #expect(calls.first?.function.arguments["launch_id"]?.anyValue as? String == "LAGUNA-77")
    }

    /// The native envelope must keep working exactly as before.
    @Test("the native envelope still parses")
    func nativeEnvelopeStillParses() {
        let (_, calls) = run(
            "<|tool_call_start|>[lookup_launch_status(launch_id='LAGUNA-77')]<|tool_call_end|>")
        #expect(calls.count == 1)
        #expect(calls.first?.function.arguments["launch_id"]?.anyValue as? String == "LAGUNA-77")
    }

    /// The OTHER shape this bundle emits. `BENCH_BATCH_TOOLCALL` on the same
    /// model produced a bare JSON object rather than the pythonic form:
    ///
    ///     {"name": "get_weather", "arguments": {"location": "Tokyo"}}
    ///
    /// also with `toolCalls: 0`. Pinning it here says whether the inline-JSON
    /// fallback covers it or whether that is a second, separate gap.
    @Test("the bare JSON shape the same bundle also emits is parsed")
    func recoversBareJSONShape() {
        let weatherTools: [[String: any Sendable]] = [[
            "type": "function",
            "function": [
                "name": "get_weather",
                "parameters": [
                    "type": "object",
                    "properties": ["location": ["type": "string"]],
                    "required": ["location"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]]
        let processor = ToolCallProcessor(format: .lfm2, tools: weatherTools)
        let text = "{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}"
        var visible = ""
        for ch in text { if let out = processor.processChunk(String(ch)) { visible += out } }
        if let tail = processor.processEOS() { visible += tail }
        #expect(processor.toolCalls.count == 1, "bare JSON call leaked as text: \(visible)")
        #expect(processor.toolCalls.first?.function.name == "get_weather")
    }

    /// The spelling the OTHER three quants emit for the identical call. JANG_2L
    /// produced `{"name": …, "arguments": …}` while JANG_4M, JANG_6M and MXFP8
    /// produced `{"function": …, "parameters": …}` — same intent, different
    /// keys, and only the first was recovered.
    @Test("the function/parameters spelling is parsed like name/arguments")
    func recoversFunctionParametersSpelling() {
        let weatherTools: [[String: any Sendable]] = [[
            "type": "function",
            "function": [
                "name": "get_weather",
                "parameters": [
                    "type": "object",
                    "properties": ["location": ["type": "string"]],
                    "required": ["location"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]]
        for text in [
            "{\"function\": \"get_weather\", \"parameters\": {\"location\": \"Tokyo\"}}",
            "{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}",
        ] {
            let processor = ToolCallProcessor(format: .lfm2, tools: weatherTools)
            var visible = ""
            for ch in text { if let out = processor.processChunk(String(ch)) { visible += out } }
            if let tail = processor.processEOS() { visible += tail }
            #expect(processor.toolCalls.count == 1, "leaked as text: \(visible)")
            #expect(processor.toolCalls.first?.function.name == "get_weather")
            #expect(
                processor.toolCalls.first?.function.arguments["location"]?.anyValue as? String
                    == "Tokyo")
        }
    }

    /// The alias must not loosen the schema gate: an unoffered name in the
    /// `function` slot is still not a call.
    @Test("the function spelling still requires an offered tool")
    func functionSpellingStillChecksSchema() {
        let (_, calls) = run("{\"function\": \"rm_rf\", \"parameters\": {\"path\": \"/\"}}")
        #expect(calls.isEmpty)
    }

    // MARK: - Must stay prose

    /// An answer that merely mentions a tool must not become a call. This is the
    /// hazard `ToolCallFormat.usesTaggedOnlyReasoningExtraction` documents for
    /// LFM2 ("may mention `line_count()` while deliberating").
    @Test("ordinary prose is not turned into a tool call")
    func proseStaysProse() {
        for text in [
            "The launch was scrubbed due to weather.",
            "I do not have access to launch data.",
        ] {
            let (visible, calls) = run(text)
            #expect(calls.isEmpty, "prose became a tool call: \(text)")
            #expect(visible.contains(text.prefix(10)))
        }
    }

    /// A name that was never offered is not a tool call, however well-formed.
    @Test("an unoffered function name is not called")
    func unknownToolStaysProse() {
        let (_, calls) = run("rm_rf(path='/')")
        #expect(calls.isEmpty)
    }
}
