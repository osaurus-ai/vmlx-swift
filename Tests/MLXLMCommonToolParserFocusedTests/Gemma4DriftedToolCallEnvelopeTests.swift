// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing
@testable import MLXLMCommon

/// Regression: live Gemma-4 JANG/CRACK rows occasionally drift the native
/// `<|tool_call>call:name{...}<tool_call|>` envelope to
/// `<|tool_call|>call:name({...})<tool_call|>` — an extra `|` in the open tag
/// plus JSON arguments wrapped in parens. The streaming processor must still
/// recognize this as a tool call and must NOT leak the raw markup into visible
/// text (observed live: the whole envelope printed in the chat transcript).
@Suite("Gemma4 drifted tool-call envelope")
struct Gemma4DriftedToolCallEnvelopeTests {
    private var capabilitiesTool: [[String: any Sendable]] {
        let idsSchema: [String: any Sendable] = [
            "type": "array",
            "items": ["type": "string"] as [String: any Sendable],
        ]
        let parameters: [String: any Sendable] = [
            "type": "object",
            "properties": ["ids": idsSchema] as [String: any Sendable],
            "required": ["ids"],
        ]
        let function: [String: any Sendable] = [
            "name": "capabilities_load",
            "description": "Load capability tools by id.",
            "parameters": parameters,
        ]
        return [
            [
                "type": "function",
                "function": function,
            ] as [String: any Sendable]
        ]
    }

    private func feed(_ envelope: String, chunkSize: Int, tools: [[String: any Sendable]]?)
        -> (visible: String, processor: ToolCallProcessor)
    {
        let processor = ToolCallProcessor(format: .gemma4, tools: tools)
        var visible = ""
        var idx = envelope.startIndex
        while idx < envelope.endIndex {
            let end = envelope.index(idx, offsetBy: chunkSize, limitedBy: envelope.endIndex)
                ?? envelope.endIndex
            if let out = processor.processChunk(String(envelope[idx..<end])) { visible += out }
            idx = end
        }
        if let tail = processor.processEOS() { visible += tail }
        return (visible, processor)
    }

    @Test("drifted <|tool_call|> + paren-JSON envelope parses and does not leak (whole-string)")
    func parsesDriftedEnvelopeWholeString() throws {
        let envelope =
            #"<|tool_call|>call:capabilities_load({"ids":["tool/browser_navigate","tool/browser_click","tool/browser_type"]})<tool_call|>"#
        let (visible, processor) = feed(envelope, chunkSize: envelope.count, tools: capabilitiesTool)

        #expect(!visible.contains("tool_call"), "leaked envelope markup: \(visible)")
        #expect(!visible.contains("call:"), "leaked bare-call marker: \(visible)")
        let call = try #require(processor.toolCalls.first)
        #expect(processor.toolCalls.count == 1)
        #expect(call.function.name == "capabilities_load")
        #expect(call.function.arguments["ids"] != nil, "ids missing: \(call.function.arguments)")
    }

    @Test("drifted envelope parses and does not leak under tiny streaming chunks")
    func parsesDriftedEnvelopeStreamed() throws {
        let envelope =
            #"<|tool_call|>call:capabilities_load({"ids":["tool/browser_navigate"]})<tool_call|>"#
        for size in [1, 2, 3, 5, 7] {
            let (visible, processor) = feed(envelope, chunkSize: size, tools: capabilitiesTool)
            #expect(!visible.contains("tool_call"), "size \(size) leaked markup: \(visible)")
            #expect(!visible.contains("call:"), "size \(size) leaked bare-call: \(visible)")
            #expect(processor.toolCalls.count == 1, "size \(size) missed tool call: \(visible)")
            #expect(processor.toolCalls.first?.function.name == "capabilities_load")
        }
    }

    @Test("native envelope still parses (no regression)")
    func nativeEnvelopeStillParses() throws {
        let envelope = #"<|tool_call>call:capabilities_load{ids:["tool/browser_navigate"]}<tool_call|>"#
        let (visible, processor) = feed(envelope, chunkSize: 4, tools: capabilitiesTool)
        #expect(!visible.contains("tool_call"), "leaked native markup: \(visible)")
        #expect(processor.toolCalls.first?.function.name == "capabilities_load")
    }
}
