// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
import MLXLMCommon
import Testing

@Suite("Tool-call progress routing contracts")
struct ToolCallProgressRoutingTests {

    private let weatherTool: [[String: any Sendable]] = [
        [
            "type": "function",
            "function": [
                "name": "get_weather",
                "parameters": [
                    "type": "object",
                    "properties": ["city": ["type": "string"]],
                ],
            ] as [String: any Sendable],
        ]
    ]

    /// Streaming a tagged JSON envelope in small chunks must surface
    /// `.toolCallProgress` deltas that concatenate to the collected envelope
    /// text, followed by exactly one parsed `.toolCall`.
    @Test("progress deltas stream while a call is collected")
    func progressDeltasWhileCollecting() {
        let processor = ToolCallProcessor(format: .json, tools: weatherTool)
        let chunks = [
            "<tool_call>", "{\"name\": \"get_w", "eather\", \"argu",
            "ments\": {\"city\": \"Par", "is\"}}", "</tool_call>",
        ]

        var progress = ""
        var progressEvents = 0
        var calls: [ToolCall] = []
        var visible = ""
        for chunk in chunks {
            for event in routeGenerationText(chunk, channel: .content, through: processor) {
                switch event {
                case .toolCallProgress(let delta):
                    #expect(!delta.isEmpty)
                    progress += delta
                    progressEvents += 1
                case .toolCall(let call):
                    calls.append(call)
                case .chunk(let text):
                    visible += text
                default:
                    break
                }
            }
        }
        for event in flushGenerationText(channel: .content, through: processor) {
            if case .toolCall(let call) = event { calls.append(call) }
        }

        #expect(progressEvents > 1, "multi-chunk envelope should yield multiple deltas")
        #expect(progress.contains("\"city\": \"Par"))
        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "get_weather")
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    /// Plain prose through the processor must never produce progress events.
    @Test("no progress events for plain text")
    func noProgressForPlainText() {
        let processor = ToolCallProcessor(format: .json, tools: weatherTool)
        var progressEvents = 0
        var visible = ""
        for chunk in ["Hello ", "world, no ", "tools here."] {
            for event in routeGenerationText(chunk, channel: .content, through: processor) {
                switch event {
                case .toolCallProgress: progressEvents += 1
                case .chunk(let text): visible += text
                default: break
                }
            }
        }
        #expect(progressEvents == 0)
        #expect(visible == "Hello world, no tools here.")
    }

    /// Strip-only processors (no tools offered) must not leak envelope text
    /// as progress — the call itself is discarded, so no preview either.
    @Test("strip-only processors emit no progress")
    func stripOnlyEmitsNoProgress() {
        let processor = ToolCallProcessor(format: .json, tools: nil, stripOnly: true)
        var progressEvents = 0
        for chunk in ["<tool_call>", "{\"name\": \"x\", \"arguments\": {}}", "</tool_call>"] {
            for event in routeGenerationText(chunk, channel: .content, through: processor) {
                if case .toolCallProgress = event { progressEvents += 1 }
            }
        }
        #expect(progressEvents == 0)
    }
}
