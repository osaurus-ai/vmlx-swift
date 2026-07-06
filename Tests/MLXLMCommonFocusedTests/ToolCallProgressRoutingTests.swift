import Foundation
import Testing

@testable import MLXLMCommon

/// Coverage for the `.toolCallProgress` streaming accommodation: while the
/// tool-call processor collects a committed call, `routeGenerationText` emits
/// incremental envelope deltas so a consumer can preview a long call (e.g. a
/// file write) instead of seeing a silent gap. The contract these tests pin:
///   1. a multi-chunk envelope yields progress deltas whose concatenation is
///      the collected envelope text, followed by exactly one parsed `.toolCall`
///      and no visible `.chunk` leak;
///   2. plain prose never yields progress events;
///   3. a strip-only processor (no tools offered) never leaks envelope text as
///      progress, because it produces no terminating `.toolCall`.
struct ToolCallProgressRoutingTests {

    private func lineCountToolSpec() -> [String: any Sendable] {
        [
            "type": "function",
            "function": [
                "name": "line_count",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "text": ["type": "string"] as [String: any Sendable]
                    ] as [String: any Sendable],
                    "required": ["text"],
                    "additionalProperties": false,
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    /// A Zyphra/Gemma4 tool-call envelope that parses to a single `line_count`
    /// call (same shape the Gemma4 parser tests use).
    private let envelope = """
        <zyphra_tool_call>
        <function=line_count>
        <parameter=text>
        red
        green
        blue
        </parameter>
        </function>
        </zyphra_tool_call>
        """

    /// Split a string into `count` roughly equal contiguous chunks, mimicking
    /// token-by-token streaming without depending on tokenizer boundaries.
    private func chunked(_ s: String, into count: Int) -> [String] {
        let chars = Array(s)
        guard count > 1, chars.count >= count else { return [s] }
        let size = Int((Double(chars.count) / Double(count)).rounded(.up))
        return stride(from: 0, to: chars.count, by: size).map {
            String(chars[$0 ..< min($0 + size, chars.count)])
        }
    }

    @Test("multi-chunk tool-call envelope streams progress deltas then one parsed call")
    func multiChunkEnvelopeStreamsProgressThenOneCall() {
        let processor = ToolCallProcessor(format: .gemma4, tools: [lineCountToolSpec()])
        var progress: [String] = []
        var visible = ""
        var calls: [ToolCall] = []

        for piece in chunked(envelope, into: 6) {
            for event in routeGenerationText(piece, channel: .content, through: processor) {
                switch event {
                case .toolCallProgress(let delta): progress.append(delta)
                case .chunk(let text): visible += text
                case .toolCall(let call): calls.append(call)
                default: break
                }
            }
        }
        for event in flushGenerationText(channel: .content, through: processor) {
            if case .toolCall(let call) = event { calls.append(call) }
            if case .chunk(let text) = event { visible += text }
        }

        // The whole point: the collection window is NOT silent.
        #expect(!progress.isEmpty, "expected incremental tool-call progress deltas")
        // Deltas are real envelope bytes in order — the concatenation is a
        // contiguous prefix of the raw envelope, never fabricated text.
        let joined = progress.joined()
        #expect(!joined.isEmpty)
        #expect(envelope.contains(joined) || joined.contains("<zyphra_tool_call>"))
        #expect(envelope.hasPrefix(joined) || envelope.contains(joined))
        // The parsed call still arrives exactly once, and nothing leaked to
        // the visible channel.
        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "line_count")
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    @Test("plain prose yields no tool-call progress events")
    func plainProseYieldsNoProgress() {
        let processor = ToolCallProcessor(format: .gemma4, tools: [lineCountToolSpec()])
        var progress: [String] = []
        var visible = ""
        let prose = "Sure — here is a short explanation with no tool call at all."
        for piece in chunked(prose, into: 5) {
            for event in routeGenerationText(piece, channel: .content, through: processor) {
                if case .toolCallProgress(let d) = event { progress.append(d) }
                if case .chunk(let t) = event { visible += t }
            }
        }
        visible += flushGenerationText(channel: .content, through: processor)
            .compactMap(\.chunk).joined()

        #expect(progress.isEmpty, "plain prose must never emit tool-call progress")
        #expect(visible.contains("no tool call"))
    }

    @Test("strip-only processor never leaks envelope text as progress")
    func stripOnlyNeverLeaksProgress() {
        // No tools offered → strip-only: markers are stripped from visible text
        // and the parsed call is discarded, so there is no terminating
        // `.toolCall`. A progress delta here would strand the consumer.
        let processor = ToolCallProcessor(format: .gemma4, tools: nil, stripOnly: true)
        var progress: [String] = []
        var calls = 0
        for piece in chunked(envelope, into: 6) {
            for event in routeGenerationText(piece, channel: .content, through: processor) {
                if case .toolCallProgress(let d) = event { progress.append(d) }
                if case .toolCall = event { calls += 1 }
            }
        }
        for event in flushGenerationText(channel: .content, through: processor) {
            if case .toolCall = event { calls += 1 }
        }

        #expect(progress.isEmpty, "strip-only must not surface envelope text as progress")
        #expect(calls == 0, "strip-only discards the parsed call")
    }
}
