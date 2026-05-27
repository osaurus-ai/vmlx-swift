// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import VMLXJinja
import MLXLMCommon
import Testing

private extension Template {
    func renderDSV4(_ context: [String: any Sendable]) throws -> String {
        var values: [String: Value] = [:]
        for (key, value) in context {
            values[key] = try Value(any: value)
        }
        return try render(values)
    }
}

private func repositoryFile(_ relativePath: String) throws -> String {
    var search = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    for _ in 0..<8 {
        let package = search.appendingPathComponent("Package.swift")
        let candidate = search.appendingPathComponent(relativePath)
        if FileManager.default.fileExists(atPath: package.path),
           FileManager.default.fileExists(atPath: candidate.path) {
            return try String(contentsOf: candidate, encoding: .utf8)
        }
        search = search.deletingLastPathComponent()
    }
    throw CocoaError(.fileNoSuchFile)
}

@Suite("DeepseekV4 chat-template fallback")
struct DeepseekV4ChatTemplateFallbackFocusedTests {
    @Test("reasoning effort max reaches DSV4 fallback preface")
    func reasoningEffortMaxReachesFallbackPreface() throws {
        let template = try Template(ChatTemplateFallbacks.dsv4Minimal)
        let rendered = try template.renderDSV4([
            "messages": [
                ["role": "user", "content": "Prove max reasoning reaches the template."],
            ],
            "add_generation_prompt": true,
            "enable_thinking": true,
            "reasoning_effort": "max",
        ])

        #expect(rendered.contains("Reasoning Effort: Absolute maximum with no shortcuts permitted."))
        #expect(rendered.hasSuffix("<\u{FF5C}Assistant\u{FF5C}><think>"))
    }

    @Test("top-level OpenAI tools render in DSV4 DSML schema block")
    func topLevelOpenAIToolsRenderInDSMLSchemaBlock() throws {
        let template = try Template(ChatTemplateFallbacks.dsv4Minimal)
        let rendered = try template.renderDSV4([
            "messages": [
                ["role": "system", "content": "You are a local Osaurus engine."],
                ["role": "user", "content": "What can you do with tools?"],
                ["role": "assistant", "content": "I can call tools when needed."],
                ["role": "user", "content": "Use the available tool if helpful."],
            ],
            "tools": [
                [
                    "type": "function",
                    "function": [
                        "name": "osaurus_probe_tool_0",
                        "description": "Probe live tool-schema rendering.",
                        "parameters": [
                            "type": "object",
                            "properties": [
                                "query": [
                                    "type": "string",
                                    "description": "Probe query.",
                                ] as [String: any Sendable],
                            ] as [String: any Sendable],
                            "required": ["query"],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ],
            "add_generation_prompt": true,
            "enable_thinking": true,
            "reasoning_effort": "high",
        ])

        #expect(rendered.contains("## Tools"))
        #expect(rendered.contains("### Available Tool Schemas"))
        #expect(rendered.contains("osaurus_probe_tool_0"))
        #expect(rendered.contains("<\u{FF5C}DSML\u{FF5C}tool_calls>"))
        #expect(rendered.hasSuffix(
            "<\u{FF5C}User\u{FF5C}>Use the available tool if helpful.<\u{FF5C}Assistant\u{FF5C}><think>"))
    }

    @Test("compiled DSV4 fallback renders tools without a system message")
    func compiledDSV4FallbackRendersToolsWithoutSystemMessage() throws {
        let template = try Template(ChatTemplateFallbacks.dsv4Minimal)
        let rendered = try template.renderDSV4(noSystemToolProbeContext())

        assertNoSystemToolsRenderBetweenUserAndAssistant(rendered)
    }

    @Test("DSV4 required tool choice reaches DSML protocol block")
    func dsv4RequiredToolChoiceReachesDSMLProtocolBlock() throws {
        var context = noSystemToolProbeContext()
        context["tool_choice"] = "required"

        let compiled = try Template(ChatTemplateFallbacks.dsv4Minimal).renderDSV4(context)
        assertRequiredToolChoiceDirective(compiled)

        let source = try repositoryFile("Libraries/MLXLMCommon/ChatTemplates/DSV4Minimal.jinja")
        let standalone = try Template(source).renderDSV4(context)
        assertRequiredToolChoiceDirective(standalone)

        let swiftRendered = DeepseekV4ChatEncoder().encode(
            messages: [
                .init(
                    role: .system,
                    content: "You are a local agent.",
                    tools: [fileReadToolSpec()]
                ),
                .init(role: .user, content: "Use file_read."),
            ],
            thinkingMode: .chat,
            toolChoiceRequired: true
        )
        assertRequiredToolChoiceDirective(swiftRendered)
    }

    @Test("Swift DSV4 required tool choice appends latest reminder after conflicting no-tool history")
    func swiftDSV4RequiredToolChoiceAppendsLatestReminderAfterNoToolHistory() {
        let rendered = DeepseekV4ChatEncoder().encode(
            messages: [
                .init(role: .system, content: "", tools: [lineCountToolSpec()]),
                .init(role: .user, content: "Use line_count on red\ngreen\nblue."),
                .init(
                    role: .assistant,
                    toolCalls: [
                        .init(
                            id: "call_lines",
                            name: "line_count",
                            arguments: #"{"text":"red\ngreen\nblue"}"#)
                    ]),
                .init(role: .tool, content: #"{"lines":3}"#, toolCallId: "call_lines"),
                .init(role: .user, content: "How many lines? Do not call another tool."),
                .init(role: .assistant, content: "Three lines were counted."),
                .init(role: .user, content: "Now use line_count on one\ntwo.", task: "action"),
            ],
            thinkingMode: .chat,
            toolChoiceRequired: true
        )

        let action = "<\u{FF5C}action\u{FF5C}>"
        let reminder = "<\u{FF5C}latest_reminder\u{FF5C}>"
        let tail = "<\u{FF5C}Assistant\u{FF5C}></think><\u{FF5C}action\u{FF5C}>"
        let actionRange = rendered.range(of: action)
        let reminderRange = rendered.range(of: reminder)
        #expect(actionRange != nil)
        #expect(reminderRange != nil)
        #expect(reminderRange!.lowerBound < actionRange!.lowerBound)
        #expect(rendered.contains("The active API tool_choice is required"))
        #expect(rendered.contains("<\u{FF5C}DSML\u{FF5C}tool_calls> block"))
        #expect(rendered.hasSuffix(tail))
    }

    @Test("Swift DSV4 required tool choice preserves assistant tail after plain no-tool history")
    func swiftDSV4RequiredToolChoicePreservesAssistantTailAfterPlainNoToolHistory() {
        let rendered = DeepseekV4ChatEncoder().encode(
            messages: [
                .init(role: .system, content: "", tools: [lineCountToolSpec()]),
                .init(role: .user, content: "Use line_count on red\ngreen\nblue."),
                .init(
                    role: .assistant,
                    toolCalls: [
                        .init(
                            id: "call_lines",
                            name: "line_count",
                            arguments: #"{"text":"red\ngreen\nblue"}"#)
                    ]),
                .init(role: .tool, content: #"{"lines":3}"#, toolCallId: "call_lines"),
                .init(role: .user, content: "How many lines? Do not call another tool."),
                .init(role: .assistant, content: "Three lines were counted."),
                .init(role: .user, content: "Now use line_count on one\ntwo."),
            ],
            thinkingMode: .chat,
            toolChoiceRequired: true
        )

        let finalUser = "Now use line_count on one\ntwo."
        let reminder = "<\u{FF5C}latest_reminder\u{FF5C}>"
        let tail = "<\u{FF5C}Assistant\u{FF5C}></think>"
        let finalUserRange = rendered.range(of: finalUser)
        let reminderRange = rendered.range(of: reminder)
        #expect(finalUserRange != nil)
        #expect(reminderRange != nil)
        #expect(reminderRange!.lowerBound < finalUserRange!.lowerBound)
        #expect(rendered.contains("The active API tool_choice is required"))
        #expect(rendered.contains("<\u{FF5C}DSML\u{FF5C}tool_calls> block"))
        #expect(rendered.hasSuffix(tail))
    }

    @Test("Nemotron required tool choice keeps native XML tool contract")
    func nemotronRequiredToolChoiceKeepsNativeXMLToolContract() throws {
        let template = try Template(ChatTemplateFallbacks.nemotronMinimal)
        let rendered = try template.renderDSV4([
            "messages": [
                ["role": "user", "content": "Use line_count on alpha\nbeta."],
            ],
            "tools": [
                [
                    "type": "function",
                    "function": [
                        "name": "line_count",
                        "description": "Count newline-separated lines in text.",
                        "parameters": [
                            "type": "object",
                            "properties": [
                                "text": [
                                    "type": "string",
                                    "description": "Text to count.",
                                ] as [String: any Sendable],
                            ] as [String: any Sendable],
                            "required": ["text"],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ],
            "tool_choice": "required",
            "add_generation_prompt": true,
            "enable_thinking": false,
        ])

        #expect(rendered.contains("# Tools"))
        #expect(rendered.contains("<tools>"))
        #expect(rendered.contains("<function>"))
        #expect(rendered.contains("<name>line_count</name>"))
        #expect(rendered.contains("<tool_call>"))
        #expect(rendered.contains("<function=example_function_name>"))
        #expect(rendered.contains("MUST be a tool call"))
        #expect(rendered.contains("one available tool and no prose before the tool result"))
        #expect(!rendered.contains("[AVAILABLE_TOOLS]"))
        #expect(!rendered.contains("<｜DSML｜"))
        #expect(rendered.hasSuffix("<|im_start|>assistant\n<think></think>"))
    }

    @Test("Nemotron required tool choice repeats contract after no-tool history")
    func nemotronRequiredToolChoiceRepeatsContractAfterNoToolHistory() throws {
        let template = try Template(ChatTemplateFallbacks.nemotronMinimal)
        let rendered = try template.renderDSV4([
            "messages": [
                ["role": "user", "content": "Use line_count on red\ngreen\nblue."],
                [
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        [
                            "id": "call_lines",
                            "type": "function",
                            "name": "line_count",
                            "arguments": ["text": "red\ngreen\nblue"],
                            "function": [
                                "name": "line_count",
                                "arguments": ["text": "red\ngreen\nblue"],
                            ] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ],
                ] as [String: any Sendable],
                ["role": "tool", "tool_call_id": "call_lines", "content": #"{"lines":3}"#],
                ["role": "user", "content": "How many lines? Do not call another tool."],
                ["role": "assistant", "content": "Three lines were counted."],
                ["role": "user", "content": "Now use line_count on one\ntwo."],
            ],
            "tools": [
                [
                    "type": "function",
                    "function": [
                        "name": "line_count",
                        "description": "Count newline-separated lines in text.",
                        "parameters": [
                            "type": "object",
                            "properties": [
                                "text": ["type": "string"] as [String: any Sendable],
                            ] as [String: any Sendable],
                            "required": ["text"],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ],
            "tool_choice": "required",
            "add_generation_prompt": true,
            "enable_thinking": false,
        ])

        let finalUser = "Now use line_count on one\ntwo."
        let tailDirective = "The current assistant response MUST be a tool call."
        let finalUserRange = try #require(rendered.range(of: finalUser))
        let tailDirectiveRange = try #require(
            rendered.range(of: tailDirective, options: .backwards))
        #expect(tailDirectiveRange.lowerBound > finalUserRange.upperBound)
        #expect(rendered.contains("<parameter=text>\nred\ngreen\nblue\n</parameter>"))
        #expect(rendered.contains("<tool_response>\n{\"lines\":3}\n</tool_response>"))
        #expect(rendered.hasSuffix("<|im_start|>assistant\n<think></think>"))
    }

    @Test("compiled DSV4 fallback renders assistant DSML tool history and tool results")
    func compiledDSV4FallbackRendersAssistantToolHistory() throws {
        let template = try Template(ChatTemplateFallbacks.dsv4Minimal)
        let rendered = try template.renderDSV4([
            "messages": [
                ["role": "user", "content": "Check Paris weather."],
                [
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        [
                            "id": "call_weather",
                            "type": "function",
                            "name": "get_weather",
                            "arguments": ["city": "Paris", "units": "metric"],
                            "function": [
                                "name": "get_weather",
                                "arguments": ["city": "Paris", "units": "metric"],
                            ] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ],
                ] as [String: any Sendable],
                ["role": "tool", "tool_call_id": "call_weather", "content": "{\"temp_c\":18}"],
                ["role": "user", "content": "Summarize."],
            ],
            "add_generation_prompt": true,
            "enable_thinking": false,
        ])

        #expect(rendered.contains("<\u{FF5C}DSML\u{FF5C}tool_calls>"))
        #expect(rendered.contains("<\u{FF5C}DSML\u{FF5C}invoke name=\"get_weather\">"))
        #expect(rendered.contains("<\u{FF5C}DSML\u{FF5C}parameter name=\"city\" string=\"true\">Paris</\u{FF5C}DSML\u{FF5C}parameter>"))
        #expect(rendered.contains("<\u{FF5C}DSML\u{FF5C}parameter name=\"units\" string=\"true\">metric</\u{FF5C}DSML\u{FF5C}parameter>"))
        #expect(rendered.contains("<tool_result>{\"temp_c\":18}</tool_result>"))
        #expect(rendered.contains("<tool_result>{\"temp_c\":18}</tool_result>\n\nSummarize."))
        #expect(!rendered.contains("<tool_result>{\"temp_c\":18}</tool_result><\u{FF5C}User\u{FF5C}>Summarize."))
        #expect(rendered.hasSuffix("Summarize.<\u{FF5C}Assistant\u{FF5C}></think>"))
    }

    @Test("default message generator preserves explicit tool-call ids")
    func defaultMessageGeneratorPreservesExplicitToolCallIDs() throws {
        let call = ToolCall(
            id: "call_weather",
            function: .init(name: "get_weather", arguments: ["city": .string("Paris")])
        )
        let raw = DefaultMessageGenerator().generate(
            message: .assistant("", toolCalls: [call])
        )
        let calls = try #require(raw["tool_calls"] as? [[String: any Sendable]])

        #expect(calls.first?["id"] as? String == "call_weather")
    }

    @Test("standalone DSV4 template renders tools without a system message")
    func standaloneDSV4TemplateRendersToolsWithoutSystemMessage() throws {
        let source = try repositoryFile("Libraries/MLXLMCommon/ChatTemplates/DSV4Minimal.jinja")
        let template = try Template(source)
        let rendered = try template.renderDSV4(noSystemToolProbeContext())

        assertNoSystemToolsRenderBetweenUserAndAssistant(rendered)
    }

    @Test("DSV4 tool instructions include no-argument invoke protocol")
    func dsv4ToolInstructionsIncludeNoArgumentInvokeProtocol() throws {
        let compiled = try Template(ChatTemplateFallbacks.dsv4Minimal)
            .renderDSV4(noArgumentToolProbeContext())
        assertNoArgumentToolProtocol(compiled)

        let source = try repositoryFile("Libraries/MLXLMCommon/ChatTemplates/DSV4Minimal.jinja")
        let standalone = try Template(source).renderDSV4(noArgumentToolProbeContext())
        assertNoArgumentToolProtocol(standalone)

        let swiftEncoder = DeepseekV4ChatEncoder()
        let swiftRendered = swiftEncoder.encode(
            messages: [
                .init(role: .system, content: "You are a local agent.", tools: [gitStatusToolSpec()]),
                .init(role: .user, content: "Check git status."),
            ],
            thinkingMode: .chat
        )
        assertNoArgumentToolProtocol(swiftRendered)
    }

    @Test("compiled DSV4 fallback separates system preface from first user turn")
    func compiledDSV4FallbackSeparatesSystemFromFirstUser() throws {
        let template = try Template(ChatTemplateFallbacks.dsv4Minimal)
        let rendered = try template.renderDSV4(systemThenUserContext())

        assertSystemSeparatedFromUser(rendered)
    }

    @Test("standalone DSV4 template separates system preface from first user turn")
    func standaloneDSV4TemplateSeparatesSystemFromFirstUser() throws {
        let source = try repositoryFile("Libraries/MLXLMCommon/ChatTemplates/DSV4Minimal.jinja")
        let template = try Template(source)
        let rendered = try template.renderDSV4(systemThenUserContext())

        assertSystemSeparatedFromUser(rendered)
    }

    @Test("Swift DSV4 encoder separates system preface from first user turn")
    func swiftDSV4EncoderSeparatesSystemFromFirstUser() {
        let encoder = DeepseekV4ChatEncoder()
        let rendered = encoder.encode(
            messages: [
                .init(role: .system, content: "You are concise."),
                .init(role: .user, content: "Remember sapphire-42."),
            ],
            thinkingMode: .chat)

        assertSystemSeparatedFromUser(rendered)
    }

    private func noSystemToolProbeContext() -> [String: any Sendable] {
        [
            "messages": [
                ["role": "user", "content": "Use a tool if needed."],
            ],
            "tools": [
                [
                    "type": "function",
                    "function": [
                        "name": "osaurus_no_system_probe",
                        "description": "Probe no-system DSV4 tool rendering.",
                        "parameters": [
                            "type": "object",
                            "properties": [
                                "query": ["type": "string"] as [String: any Sendable],
                            ] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ],
            "add_generation_prompt": true,
            "enable_thinking": false,
        ]
    }

    private func systemThenUserContext() -> [String: any Sendable] {
        [
            "messages": [
                ["role": "system", "content": "You are concise."],
                ["role": "user", "content": "Remember sapphire-42."],
            ],
            "add_generation_prompt": true,
            "enable_thinking": false,
        ]
    }

    private func noArgumentToolProbeContext() -> [String: any Sendable] {
        [
            "messages": [
                ["role": "user", "content": "Check git status."],
            ],
            "tools": [gitStatusToolSpec()],
            "add_generation_prompt": true,
            "enable_thinking": false,
        ]
    }

    private func gitStatusToolSpec() -> [String: any Sendable] {
        [
            "type": "function",
            "function": [
                "name": "git_status",
                "description": "Show working tree status.",
                "parameters": [
                    "type": "object",
                    "properties": [:] as [String: any Sendable],
                    "required": [] as [String],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ] as [String: any Sendable]
    }

    private func fileReadToolSpec() -> [String: any Sendable] {
        [
            "type": "function",
            "function": [
                "name": "file_read",
                "description": "Read a file.",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "path": ["type": "string"] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["path"] as [String],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ] as [String: any Sendable]
    }

    private func lineCountToolSpec() -> [String: any Sendable] {
        [
            "type": "function",
            "function": [
                "name": "line_count",
                "description": "Count newline-separated text lines.",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "text": ["type": "string"] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["text"] as [String],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ] as [String: any Sendable]
    }

    private func assertNoSystemToolsRenderBetweenUserAndAssistant(_ rendered: String) {
        #expect(rendered.contains("## Tools"))
        #expect(rendered.contains("osaurus_no_system_probe"))
        #expect(rendered.contains("<\u{FF5C}DSML\u{FF5C}tool_calls>"))
        #expect(rendered.contains("<\u{FF5C}User\u{FF5C}>Use a tool if needed.\n\n## Tools"))
        if let toolsIndex = rendered.range(of: "## Tools")?.lowerBound,
           let userIndex = rendered.range(of: "<\u{FF5C}User\u{FF5C}>")?.lowerBound,
           let assistantIndex = rendered.range(of: "<\u{FF5C}Assistant\u{FF5C}>")?.lowerBound {
            #expect(userIndex < toolsIndex)
            #expect(toolsIndex < assistantIndex)
        } else {
            Issue.record("DSV4 no-system tool probe is missing expected turn markers")
        }
    }

    private func assertNoArgumentToolProtocol(_ rendered: String) {
        #expect(rendered.contains("git_status"))
        #expect(rendered.contains("For tools with no parameters"))
        #expect(
            rendered.contains(
                "<\u{FF5C}DSML\u{FF5C}invoke name=\"$TOOL_NAME_WITHOUT_PARAMETERS\">\n</\u{FF5C}DSML\u{FF5C}invoke>"
            )
        )
        #expect(rendered.contains("Do not emit JSON objects for tool calls"))
    }

    private func assertRequiredToolChoiceDirective(_ rendered: String) {
        #expect(rendered.contains("file_read") || rendered.contains("osaurus_no_system_probe"))
        #expect(rendered.contains("The current assistant response MUST be a tool call"))
        #expect(rendered.contains("Start with a \"<\u{FF5C}DSML\u{FF5C}tool_calls>\" block"))
        #expect(rendered.contains("do not answer in prose before the tool result"))
    }

    private func assertSystemSeparatedFromUser(_ rendered: String) {
        let separated = "You are concise.\n\(DeepseekV4Tokens.user)Remember sapphire-42."
        let glued = "You are concise.\(DeepseekV4Tokens.user)Remember sapphire-42."
        #expect(rendered.contains(separated))
        #expect(!rendered.contains(glued))
    }

    @Test("Swift Jinja tojson accepts Python separators kwarg used by Kimi tools")
    func tojsonAcceptsPythonSeparatorsKwarg() throws {
        let template = try Template("{{ tools | tojson(separators=(',', ':')) }}")
        let rendered = try template.renderDSV4([
            "tools": [
                [
                    "type": "function",
                    "function": [
                        "name": "kimi_probe_tool",
                        "parameters": [
                            "type": "object",
                            "properties": [
                                "city": ["type": "string"] as [String: any Sendable],
                            ] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ],
        ])

        #expect(rendered.contains("\"name\":\"kimi_probe_tool\""))
        #expect(rendered.contains("\"city\""))
        #expect(!rendered.contains("\": "))
    }

    @Test("Swift Jinja for-loop iterable accepts binary expressions")
    func forLoopIterableAcceptsBinaryExpression() throws {
        let template = try Template("{% for item in left + right %}{{ item }}{% endfor %}")
        let rendered = try template.renderDSV4([
            "left": ["A"],
            "right": ["B", "C"],
        ])

        #expect(rendered == "ABC")
    }

    @Test("Swift Jinja for-loop if clause remains a loop filter")
    func forLoopIfClauseRemainsLoopFilter() throws {
        let template = try Template("{% for item in values if item != 'B' %}{{ item }}{% endfor %}")
        let rendered = try template.renderDSV4([
            "values": ["A", "B", "C"],
        ])

        #expect(rendered == "AC")
    }

    @Test("ZAYA1-VL fallback preserves vision placeholders and ZAYA XML tools")
    func zayaVLFallbackPreservesVisionAndTools() throws {
        let template = try Template(ChatTemplateFallbacks.zayaVLVisionToolMinimal)
        let rendered = try template.renderDSV4([
            "messages": [
                [
                    "role": "user",
                    "content": [
                        ["type": "image"],
                        ["type": "text", "text": "Describe the image, then call a tool if needed."],
                    ],
                ] as [String: any Sendable],
            ],
            "tools": [
                [
                    "type": "function",
                    "function": [
                        "name": "osaurus_probe_tool_0",
                        "description": "Probe ZAYA1-VL tool rendering.",
                        "parameters": [
                            "type": "object",
                            "properties": [
                                "query": [
                                    "type": "string",
                                    "description": "Search query text.",
                                ] as [String: any Sendable],
                            ] as [String: any Sendable],
                            "required": ["query"],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ],
            "bos_token": "<bos>",
            "add_generation_prompt": true,
            "enable_thinking": false,
            "tool_choice": "required",
            "tool_choice_name": "osaurus_probe_tool_0",
        ])

        #expect(rendered.contains("<|vision_start|><image><|vision_end|>"))
        #expect(rendered.contains("Describe the image"))
        #expect(rendered.contains("<name>osaurus_probe_tool_0</name>"))
        #expect(rendered.contains("<name>query</name>"))
        #expect(rendered.contains("<type>string</type>"))
        #expect(rendered.contains("<description>Search query text.</description>"))
        #expect(rendered.contains("<required>[\"query\"]</required>"))
        #expect(rendered.contains("<zyphra_tool_call>"))
        #expect(rendered.contains("The current assistant response MUST be a tool call"))
        #expect(rendered.contains("Use the `osaurus_probe_tool_0` function."))
        #expect(rendered.contains("Required parameters for `osaurus_probe_tool_0`: query."))
        #expect(rendered.contains("<function=osaurus_probe_tool_0>"))
        #expect(rendered.contains("<parameter=PARAMETER_NAME>\nACTUAL_ARGUMENT_VALUE\n</parameter>"))
        #expect(rendered.contains("fill each parameter with the actual argument value"))
        #expect(!rendered.contains("VALUE_FOR_query"))
        #expect(!rendered.contains("VALUE_FOR_*"))
        #expect(rendered.contains("Do not wrap the parameter value in JSON quotes"))
        #expect(rendered.hasSuffix("<|im_start|>assistant\n"))
        #expect(!rendered.contains("<think>"))
        #expect(!rendered.contains("enable_thinking"))
    }

    @Test("ZAYA1-VL required tool choice repeats at current turn after no-tool history")
    func zayaVLRequiredToolChoiceRepeatsAfterNoToolHistory() throws {
        let template = try Template(ChatTemplateFallbacks.zayaVLVisionToolMinimal)
        let rendered = try template.renderDSV4([
            "messages": [
                ["role": "user", "content": "Use line_count on red\ngreen\nblue."],
                [
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        [
                            "id": "call_lines",
                            "type": "function",
                            "function": [
                                "name": "line_count",
                                "arguments": ["text": "red\ngreen\nblue"],
                            ] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ],
                ] as [String: any Sendable],
                ["role": "tool", "tool_call_id": "call_lines", "content": #"{"lines":3}"#],
                ["role": "user", "content": "How many lines? Do not call another tool."],
                ["role": "assistant", "content": "Three lines were counted."],
                ["role": "user", "content": "Now use line_count on one\ntwo."],
            ],
            "tools": [
                [
                    "type": "function",
                    "function": [
                        "name": "line_count",
                        "description": "Count newline-separated lines in text.",
                        "parameters": [
                            "type": "object",
                            "properties": [
                                "text": ["type": "string"] as [String: any Sendable],
                            ] as [String: any Sendable],
                            "required": ["text"],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ],
            "bos_token": "<bos>",
            "add_generation_prompt": true,
            "enable_thinking": false,
            "tool_choice": "required",
        ])

        let finalUser = "Now use line_count on one\ntwo."
        let currentReminder = "The current assistant response MUST be a tool call."
        let tail = "<|im_start|>assistant\n"
        let finalUserRange = rendered.range(of: finalUser)
        let reminderRange = rendered.range(of: currentReminder, options: .backwards)
        #expect(finalUserRange != nil)
        #expect(reminderRange != nil)
        #expect(finalUserRange!.lowerBound < reminderRange!.lowerBound)
        let afterFinalUser = rendered[finalUserRange!.upperBound...]
        #expect(afterFinalUser.contains(currentReminder))
        #expect(!afterFinalUser.contains("<|im_start|>system\n" + currentReminder))
        let turnRoles = rendered.components(separatedBy: "<|im_start|>")
            .dropFirst()
            .compactMap { segment in
                segment.split(separator: "\n", maxSplits: 1).first.map(String.init)
            }
        #expect(turnRoles == ["system", "user", "assistant"])
        #expect(!rendered.contains("Use line_count on red\ngreen\nblue."))
        #expect(!rendered.contains("How many lines? Do not call another tool."))
        #expect(!rendered.contains("Three lines were counted."))
        #expect(!rendered.contains("Previous tool result available."))
        #expect(!rendered.contains("<zyphra_tool_response>\n{\"lines\":3}"))
        #expect(!rendered.contains("<function=line_count>\n<parameter=text>\nred\ngreen\nblue\n</parameter>\n</function>"))
        #expect(rendered.contains("Return the call in this form:\n<zyphra_tool_call>\n<function=line_count>"))
        #expect(rendered.contains("<parameter=PARAMETER_NAME>\nACTUAL_ARGUMENT_VALUE\n</parameter>"))
        #expect(rendered.contains("include these required parameter block(s) exactly once"))
        #expect(rendered.contains("<parameter=text>\nACTUAL_ARGUMENT_VALUE\n</parameter>"))
        #expect(rendered.contains("Do not omit required parameters."))
        #expect(rendered.contains("copy that exact text into the string parameter body"))
        #expect(!rendered.contains("VALUE_FOR_text"))
        #expect(rendered.contains("For string parameters, write the raw string value only."))
        #expect(rendered.hasSuffix(tail))
    }

    @Test("ZAYA XML parser decodes live HTML line breaks in string parameters")
    func zayaXMLParserDecodesLiveHTMLLineBreaksInStringParameters() throws {
        let output = #"""
            <zyphra_tool_call>
            <function=line_count>
            <parameter=text>one<br>two</parameter>
            </function>
            </zyphra_tool_call>
            """#
        let call = try #require(
            ToolCallFormat.zayaXml.createParser().parse(
                content: output,
                tools: [lineCountToolSpec()]
            )
        )

        #expect(call.function.name == "line_count")
        #expect(call.function.arguments["text"] == .string("one\ntwo"))
    }

    @Test("ZAYA XML parser unwraps accidental JSON-quoted string parameters")
    func zayaXMLParserUnwrapsJSONQuotedStringParameters() throws {
        let content = #"""
        <zyphra_tool_call>
        <function=line_count>
        <parameter=text>
        "red\ngreen\nblue"
        </parameter>
        </function>
        </zyphra_tool_call>
        """#
        let call = try #require(
            ToolCallFormat.zayaXml.createParser().parse(
                content: content,
                tools: [lineCountToolSpec()]))

        #expect(call.function.name == "line_count")
        #expect(call.function.arguments["text"] == .string("red\ngreen\nblue"))
    }

    @Test("ZAYA XML parser unwraps accidental raw multiline quoted string parameters")
    func zayaXMLParserUnwrapsRawMultilineQuotedStringParameters() throws {
        let content = """
        <zyphra_tool_call>
        <function=line_count>
        <parameter=text>
        "red
        green
        blue"
        </parameter>
        </function>
        </zyphra_tool_call>
        """
        let call = try #require(
            ToolCallFormat.zayaXml.createParser().parse(
                content: content,
                tools: [lineCountToolSpec()]))

        #expect(call.function.name == "line_count")
        #expect(call.function.arguments["text"] == .string("red\ngreen\nblue"))
    }

    @Test("ZAYA1-VL sidecar shim rewrites every loader-visible template source")
    func zayaVLSidecarShimRewritesTemplateSources() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(
            "zaya-vl-sidecar-shim-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let tokenizerConfig: [String: Any] = [
            "bos_token": "<bos>",
            "eos_token": "<|im_end|>",
            "chat_template": "user: {{ messages[0]['content'] }}",
        ]
        let sidecarConfig: [String: Any] = [
            "chat_template": "{{ '<|vision_start|><image><|vision_end|>\\n' }}",
        ]
        let jangConfig: [String: Any] = [
            "capabilities": [
                "family": "zaya1_vl",
                "tool_parser": "zaya_xml",
                "think_in_template": false,
                "supports_tools": true,
            ],
        ]
        try JSONSerialization.data(withJSONObject: tokenizerConfig).write(
            to: root.appendingPathComponent("tokenizer_config.json"))
        try JSONSerialization.data(withJSONObject: sidecarConfig).write(
            to: root.appendingPathComponent("chat_template.json"))
        try JSONSerialization.data(withJSONObject: jangConfig).write(
            to: root.appendingPathComponent("jang_config.json"))

        let shim = JangLoader.resolveChatTemplateSidecarSubstitution(for: root)
        #expect(shim != root)

        let rewrittenTokenizerData = try Data(
            contentsOf: shim.appendingPathComponent("tokenizer_config.json"))
        let rewrittenTokenizer = try #require(
            JSONSerialization.jsonObject(with: rewrittenTokenizerData) as? [String: Any])
        let tokenizerTemplate = try #require(rewrittenTokenizer["chat_template"] as? String)
        #expect(tokenizerTemplate.contains("zyphra_tool_call"))
        #expect(tokenizerTemplate.contains("<|vision_start|><image><|vision_end|>"))

        let rewrittenSidecarData = try Data(
            contentsOf: shim.appendingPathComponent("chat_template.json"))
        let rewrittenSidecar = try #require(
            JSONSerialization.jsonObject(with: rewrittenSidecarData) as? [String: Any])
        let sidecarTemplate = try #require(rewrittenSidecar["chat_template"] as? String)
        #expect(sidecarTemplate == tokenizerTemplate)

        let jinjaTemplate = try String(
            contentsOf: shim.appendingPathComponent("chat_template.jinja"),
            encoding: .utf8)
        #expect(jinjaTemplate == tokenizerTemplate)
    }

    @Test("ZAYA1-VL metadata shim engages without sidecar template")
    func zayaVLMetadataShimEngagesWithoutSidecarTemplate() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(
            "zaya-vl-metadata-shim-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let tokenizerConfig: [String: Any] = [
            "bos_token": "<bos>",
            "eos_token": "<|im_end|>",
            "chat_template": "user: {{ messages[0]['content'] }}\nassistant: ",
        ]
        let jangConfig: [String: Any] = [
            "capabilities": [
                "family": "zaya1_vl",
                "tool_parser": "zaya_xml",
                "think_in_template": false,
                "supports_tools": true,
                "supports_thinking": true,
            ],
        ]
        try JSONSerialization.data(withJSONObject: tokenizerConfig).write(
            to: root.appendingPathComponent("tokenizer_config.json"))
        try JSONSerialization.data(withJSONObject: jangConfig).write(
            to: root.appendingPathComponent("jang_config.json"))

        let shim = JangLoader.resolveChatTemplateSidecarSubstitution(for: root)
        #expect(shim != root)

        let rewrittenTokenizerData = try Data(
            contentsOf: shim.appendingPathComponent("tokenizer_config.json"))
        let rewrittenTokenizer = try #require(
            JSONSerialization.jsonObject(with: rewrittenTokenizerData) as? [String: Any])
        let tokenizerTemplate = try #require(rewrittenTokenizer["chat_template"] as? String)
        #expect(tokenizerTemplate.contains("zyphra_tool_call"))
        #expect(tokenizerTemplate.contains("<|vision_start|><image><|vision_end|>"))
        #expect(!tokenizerTemplate.contains("enable_thinking"))
        #expect(!tokenizerTemplate.contains("<think>"))
    }

    @Test("ZAYA1-VL metadata shim tolerates missing supports_tools key")
    func zayaVLMetadataShimToleratesMissingSupportsToolsKey() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(
            "zaya-vl-missing-supports-tools-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let tokenizerConfig: [String: Any] = [
            "bos_token": "<bos>",
            "eos_token": "<|im_end|>",
            "chat_template": "user: {{ messages[0]['content'] }}\nassistant: ",
        ]
        let jangConfig: [String: Any] = [
            "capabilities": [
                "family": "zaya1_vl",
                "tool_parser": "zaya_xml",
                "think_in_template": false,
                "supports_thinking": true,
            ],
        ]
        try JSONSerialization.data(withJSONObject: tokenizerConfig).write(
            to: root.appendingPathComponent("tokenizer_config.json"))
        try JSONSerialization.data(withJSONObject: jangConfig).write(
            to: root.appendingPathComponent("jang_config.json"))

        let shim = JangLoader.resolveChatTemplateSidecarSubstitution(for: root)
        #expect(shim != root)

        let rewrittenTokenizerData = try Data(
            contentsOf: shim.appendingPathComponent("tokenizer_config.json"))
        let rewrittenTokenizer = try #require(
            JSONSerialization.jsonObject(with: rewrittenTokenizerData) as? [String: Any])
        let tokenizerTemplate = try #require(rewrittenTokenizer["chat_template"] as? String)
        #expect(tokenizerTemplate.contains("zyphra_tool_call"))
        #expect(tokenizerTemplate.contains("<|vision_start|><image><|vision_end|>"))
    }
}
