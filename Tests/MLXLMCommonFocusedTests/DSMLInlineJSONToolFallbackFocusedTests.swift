// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon
import Testing

@Suite("DSV4 DSML inline JSON fallback focused contracts")
struct DSMLInlineJSONToolFallbackFocusedTests {
    @Test("registered top-level JSON tool fallback is parsed without visible leakage")
    func registeredTopLevelJSONToolFallbackIsParsedWithoutVisibleLeakage() {
        let output = """
            {"tool":"file_read","r":"np.clip(esc * 4.0 - 1.0, 0.0, 1.0)","g":"np.clip(1.0 - np.abs(esc * 2.0 - 1.0), 0.0, 1.0)","b":"np.clip(1.0 - esc * 2.0, 0.0, 1.0)"}
            """
        let processor = ToolCallProcessor(format: .dsml, tools: fileReadToolSchema())
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "file_read")
        #expect(processor.toolCalls.first?.function.arguments["path"] == nil)
        #expect(
            processor.toolCalls.first?.function.arguments["r"]
                == .string("np.clip(esc * 4.0 - 1.0, 0.0, 1.0)")
        )
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!visible.contains("\"tool\":\"file_read\""))
        #expect(!visible.contains("np.clip"))
    }

    @Test("unknown top-level JSON tool fallback remains visible")
    func unknownTopLevelJSONToolFallbackRemainsVisible() {
        let output = #"{"tool":"not_registered","path":"mandelbrot.py"}"#
        let processor = ToolCallProcessor(format: .dsml, tools: fileReadToolSchema())
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.isEmpty)
        #expect(visible.contains(#""tool":"not_registered""#))
    }

    private func fileReadToolSchema() -> [[String: any Sendable]] {
        let parameters: [String: any Sendable] = [
            "type": "object",
            "properties": [
                "path": ["type": "string"] as [String: any Sendable]
            ] as [String: any Sendable],
            "required": ["path"],
        ]
        let function: [String: any Sendable] = [
            "name": "file_read",
            "parameters": parameters,
        ]
        return [
            [
                "type": "function",
                "function": function,
            ] as [String: any Sendable]
        ]
    }
}
