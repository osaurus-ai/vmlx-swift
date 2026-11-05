// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon
import Testing

@Suite("DSV4 strict DSML transport contracts")
struct DSMLInlineJSONToolFallbackFocusedTests {
    @Test("decode loop receives prepared tool schemas")
    func decodeLoopReceivesPreparedToolSchemas() throws {
        let lmInput = try String(
            contentsOf: FocusedMLXTestSupport.sourceURL(
                "Libraries/MLXLMCommon/LanguageModel.swift"),
            encoding: .utf8)
        let batchEngine = try String(
            contentsOf: FocusedMLXTestSupport.sourceURL(
                "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift"),
            encoding: .utf8)
        let evaluate = try String(
            contentsOf: FocusedMLXTestSupport.sourceURL("Libraries/MLXLMCommon/Evaluate.swift"),
            encoding: .utf8)

        #expect(lmInput.contains("public let toolSchemas: [ToolSpec]?"))
        #expect(lmInput.contains("public func withToolSchemas"))
        #expect(batchEngine.contains("let toolSchemas = input.toolSchemas"))
        #expect(
            batchEngine.contains(
                "let activeToolSchemas = toolSchemas?.isEmpty == false ? toolSchemas : nil"))
        #expect(batchEngine.contains("if let activeToolSchemas"))
        #expect(
            batchEngine.contains(
                "ToolCallProcessor(format: toolCallFormat, tools: activeToolSchemas)"))
        #expect(evaluate.contains("let activeTools = tools?.isEmpty == false ? tools : nil"))
        #expect(evaluate.contains("if let activeTools"))
        #expect(evaluate.contains("ToolCallProcessor(format: format, tools: activeTools)"))
    }

    @Test("decode loop disables tool parser without active schemas")
    func decodeLoopDisablesToolParserWithoutActiveSchemas() throws {
        let routing = try String(
            contentsOf: FocusedMLXTestSupport.sourceURL(
                "Libraries/MLXLMCommon/GenerationStreamRouting.swift"),
            encoding: .utf8)
        let batchEngine = try String(
            contentsOf: FocusedMLXTestSupport.sourceURL(
                "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift"),
            encoding: .utf8)
        let evaluate = try String(
            contentsOf: FocusedMLXTestSupport.sourceURL("Libraries/MLXLMCommon/Evaluate.swift"),
            encoding: .utf8)
        let specDec = try String(
            contentsOf: FocusedMLXTestSupport.sourceURL(
                "Libraries/MLXLMCommon/SpecDec/SpecDecStream.swift"),
            encoding: .utf8)

        #expect(routing.contains("through toolCallProcessor: ToolCallProcessor?"))
        #expect(routing.contains("guard let toolCallProcessor else"))
        #expect(
            batchEngine.contains(
                "let activeToolSchemas = toolSchemas?.isEmpty == false ? toolSchemas : nil"))
        #expect(evaluate.contains("let activeTools = tools?.isEmpty == false ? tools : nil"))
        #expect(
            specDec.contains(
                "let activeToolSchemas = toolSchemas?.isEmpty == false ? toolSchemas : nil"))
        #expect(
            !batchEngine.contains("ToolCallProcessor(format: toolCallFormat, tools: toolSchemas)"))
        #expect(
            !evaluate.contains(
                "toolCallProcessor = ToolCallProcessor(format: format, tools: tools)"))
    }

    @Test("native DSML opts out of inline and reasoning-channel tool parsing")
    func nativeDSMLOptsOutOfFallbackRails() {
        #expect(DSMLToolCallParser().supportsInlineJSONToolFallback == false)
        #expect(ToolCallFormat.dsml.parsesToolCallsFromReasoningChannel == false)
    }

    @Test("Nemotron retains its independent legacy fallback compatibility")
    func nemotronRetainsLegacyFallbackCompatibility() {
        let parser = ToolCallFormat.nemotron.createParser()
        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "file_read",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "path": ["type": "string"] as [String: any Sendable]
                        ] as [String: any Sendable],
                        "required": ["path"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable]
        ]

        #expect(parser.supportsInlineJSONToolFallback == true)
        let call = parser.parse(
            content: #"{"tool":"file_read","path":"probe.txt"}"#,
            tools: tools)
        #expect(call?.function.name == "file_read")
        #expect(call?.function.arguments["path"] == .string("probe.txt"))
    }
}
