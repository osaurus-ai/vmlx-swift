// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon
import Testing

@Suite("DSV4 DSML inline JSON fallback focused contracts")
struct DSMLInlineJSONToolFallbackFocusedTests {
    @Test("decode loop receives prepared tool schemas for schema-aware fallback")
    func decodeLoopReceivesPreparedToolSchemasForSchemaAwareFallback() throws {
        let lmInput = try String(
            contentsOfFile: "Libraries/MLXLMCommon/LanguageModel.swift",
            encoding: .utf8)
        let batchEngine = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)
        let evaluate = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)

        #expect(lmInput.contains("public let toolSchemas: [ToolSpec]?"))
        #expect(lmInput.contains("public func withToolSchemas"))
        #expect(batchEngine.contains("let toolSchemas = input.toolSchemas"))
        #expect(batchEngine.contains("ToolCallProcessor(format: toolCallFormat, tools: toolSchemas)"))
        #expect(evaluate.contains("toolCallProcessor = ToolCallProcessor(format: format, tools: tools)"))
    }

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
        #expect(processor.toolCalls.first?.function.arguments["_error"] == .string("invalid_tool_arguments"))
        #expect(processor.toolCalls.first?.function.arguments["_field"] == .string("path"))
        #expect(processor.toolCalls.first?.function.arguments["r"] == nil)
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!visible.contains("\"tool\":\"file_read\""))
        #expect(!visible.contains("np.clip"))
    }

    @Test("schema-less DSV4 JSON tool intent is captured without visible leakage")
    func schemaLessDSV4JSONToolIntentIsCapturedWithoutVisibleLeakage() {
        let output = """
            {"tool":"file_read","r":"np.clip(esc * 4.0 - 1.0, 0.0, 1.0)","g":"np.clip(1.0 - np.abs(esc * 2.0 - 1.0), 0.0, 1.0)","b":"np.clip(1.0 - esc * 2.0, 0.0, 1.0)"}
            """
        let processor = ToolCallProcessor(format: .dsml)
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

    @Test("schema-less DSV4 JSON tool intent covers built-in tool names")
    func schemaLessDSV4JSONToolIntentCoversBuiltInToolNames() {
        for toolName in [
            "file_tree",
            "file_read",
            "file_write",
            "file_edit",
            "file_search",
            "shell_run",
            "git_status",
            "git_diff",
            "git_commit",
        ] {
            let output = #"{"tool":"\#(toolName)","path":"mandelbrot.py"}"#
            let processor = ToolCallProcessor(format: .dsml)
            var visible = ""
            for ch in output {
                visible += processor.processChunk(String(ch)) ?? ""
            }
            visible += processor.processEOS() ?? ""

            #expect(processor.toolCalls.count == 1, "\(toolName) should emit a tool attempt")
            #expect(processor.toolCalls.first?.function.name == toolName)
            #expect(processor.toolCalls.first?.function.arguments["path"] == .string("mandelbrot.py"))
            #expect(
                visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "\(toolName) JSON tool attempt leaked visible text: \(visible)"
            )
        }
    }

    @Test("live DSV4 python-style file_read attempt is captured without visible leakage")
    func liveDSV4PythonStyleFileReadAttemptIsCapturedWithoutVisibleLeakage() {
        let output = #"""
            file_read("path": "/Users/eric/Desktop/testmandel/mandelbrot.py", "start_line": 33, "end_line": 39)???
            Wait, I will read precisely.
            """#
        let processor = ToolCallProcessor(format: .dsml, tools: fileReadToolSchema())
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.count == 1)
        let call = processor.toolCalls.first
        #expect(call?.function.name == "file_read")
        #expect(
            call?.function.arguments["path"]
                == .string("/Users/eric/Desktop/testmandel/mandelbrot.py")
        )
        #expect(call?.function.arguments["start_line"] == .int(33))
        #expect(call?.function.arguments["end_line"] == .int(39))
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!visible.contains("file_read"))
        #expect(!visible.contains("Wait, I will read"))
    }

    @Test("live DSV4 bare-name JSON file_read attempt is captured without visible leakage")
    func liveDSV4BareNameJSONFileReadAttemptIsCapturedWithoutVisibleLeakage() {
        let output = #"""
            file_read
            {"path":"/Users/eric/Desktop/testmandel/mandelbrot.py","start_line":33,"end_line":39}
            DSV4_UI_TOUT_OK: post-tool prose should not leak before tool execution.
            """#
        let processor = ToolCallProcessor(format: .dsml, tools: fileReadToolSchema())
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.count == 1)
        let call = processor.toolCalls.first
        #expect(call?.function.name == "file_read")
        #expect(
            call?.function.arguments["path"]
                == .string("/Users/eric/Desktop/testmandel/mandelbrot.py")
        )
        #expect(call?.function.arguments["start_line"] == .int(33))
        #expect(call?.function.arguments["end_line"] == .int(39))
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!visible.contains("file_read"))
        #expect(!visible.contains(#""path":"#))
        #expect(!visible.contains("DSV4_UI_TOUT_OK"))
    }

    @Test("live DSV4 bare-name fenced JSON file_read attempt is captured without visible leakage")
    func liveDSV4BareNameFencedJSONFileReadAttemptIsCapturedWithoutVisibleLeakage() {
        let output = #"""
            file_read
            ```json
            {"path": "/Users/eric/Desktop/testmandel/mandelbrot.py", "start_line": 33, "end_line": 39}
            ```
            DSV4_UI_TOUT_OK: post-tool prose should not leak before tool execution.
            """#
        let processor = ToolCallProcessor(format: .dsml, tools: fileReadToolSchema())
        let visible = processor.processChunk(output) ?? ""

        #expect(processor.toolCalls.count == 1)
        let call = processor.toolCalls.first
        #expect(call?.function.name == "file_read")
        #expect(
            call?.function.arguments["path"]
                == .string("/Users/eric/Desktop/testmandel/mandelbrot.py")
        )
        #expect(call?.function.arguments["start_line"] == .int(33))
        #expect(call?.function.arguments["end_line"] == .int(39))
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(processor.processEOS()?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
        #expect(!visible.contains("file_read"))
        #expect(!visible.contains("```json"))
        #expect(!visible.contains("DSV4_UI_TOUT_OK"))
    }

    @Test("split DSV4 bare-name fenced JSON file_read attempt stays buffered")
    func splitDSV4BareNameFencedJSONFileReadAttemptStaysBuffered() {
        let processor = ToolCallProcessor(format: .dsml, tools: fileReadToolSchema())
        var visible = ""
        for chunk in [
            "file_read\n```json\n",
            #"{"path": "/Users/eric/Desktop/testmandel/mandelbrot.py", "start_line": 33"#,
            #", "end_line": 39}"#,
            "\n```\nDSV4_UI_TOUT_OK: post-tool prose should not leak.",
        ] {
            visible += processor.processChunk(chunk) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.count == 1)
        let call = processor.toolCalls.first
        #expect(call?.function.name == "file_read")
        #expect(
            call?.function.arguments["path"]
                == .string("/Users/eric/Desktop/testmandel/mandelbrot.py")
        )
        #expect(call?.function.arguments["start_line"] == .int(33))
        #expect(call?.function.arguments["end_line"] == .int(39))
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!visible.contains("file_read"))
        #expect(!visible.contains("```json"))
        #expect(!visible.contains("DSV4_UI_TOUT_OK"))
    }

    @Test("truncated schema-less DSV4 JSON tool intent is quarantined without visible leakage")
    func truncatedSchemaLessDSV4JSONToolIntentIsQuarantinedWithoutVisibleLeakage() {
        let output = """
            {"tool":"file_read","r":"np.clip(esc * 4.0 - 1.0, 0.0, 1.0)","g":"np.clip(1.0 - np.abs(esc * 2.0 - 1.0), 0.0, 1.0)","b":"np.clip(1.0 - esc * 2.0, 0.0, 1.
            """
        let processor = ToolCallProcessor(format: .dsml)
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.isEmpty)
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!visible.contains("\"tool\":\"file_read\""))
        #expect(!visible.contains("np.clip"))
    }

    @Test("schema-less bare name JSON remains visible")
    func schemaLessBareNameJSONRemainsVisible() {
        let output = #"{"name":"file_read","path":"mandelbrot.py"}"#
        let processor = ToolCallProcessor(format: .dsml)
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.isEmpty)
        #expect(visible.contains(#""name":"file_read""#))
    }

    @Test("bare tool name followed by prose remains visible")
    func bareToolNameFollowedByProseRemainsVisible() {
        let output = "file_read is available, but this sentence is not a call."
        let processor = ToolCallProcessor(format: .dsml, tools: fileReadToolSchema())
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.isEmpty)
        #expect(visible == output)
    }

    @Test("bare tool name alone does not emit invalid args")
    func bareToolNameAloneDoesNotEmitInvalidArgs() {
        let processor = ToolCallProcessor(format: .dsml, tools: fileReadToolSchema())
        var visible = ""
        visible += processor.processChunk("file") ?? ""
        visible += processor.processChunk("_read") ?? ""
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.isEmpty)
        #expect(visible == "file_read")
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
