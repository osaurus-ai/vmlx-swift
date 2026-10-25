import Foundation
import MLXLMCommon
import Testing
import VMLXJinja

private extension Template {
    func renderGemma4(_ context: [String: any Sendable]) throws -> String {
        var values: [String: Value] = [:]
        for (key, value) in context {
            values[key] = try Value(any: value)
        }
        return try render(values)
    }
}

struct Gemma4TemplateFallbackSourceTests {
    @Test
    func gemmaTemplateRuntimeErrorsRemainFallbackEligible() throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let sourceURL = root.appending(
            path: "Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift"
        )
        let source = try String(contentsOf: sourceURL, encoding: .utf8)

        #expect(
            !source.contains("if isGemma {\n                                throw error\n                            }"),
            "Gemma native Jinja runtime errors must not bypass built-in Gemma fallbacks."
        )
        #expect(!source.contains("let gemmaToolSchemasPresent"),
            "Offering tool schemas must not bypass a valid native bundle template")
        #expect(!source.contains("chat-template tools -> Gemma4WithTools fallback engaged"))
        #expect(source.contains(#""Gemma4WithTools", MLXLMCommon.ChatTemplateFallbacks.gemma4WithTools"#))
        #expect(source.contains(#""Gemma4Minimal",   MLXLMCommon.ChatTemplateFallbacks.gemma4Minimal"#))
    }

    @Test
    func gemmaToolFallbackCoversAutoToolChoiceWithoutRequiredInstruction() throws {
        let template = try Template(ChatTemplateFallbacks.gemma4WithTools)
        let rendered = try template.renderGemma4([
            "messages": [
                [
                    "role": "user",
                    "content": "Answer plainly in one short sentence: what color is grass?",
                ],
            ],
            "tools": [
                [
                    "type": "function",
                    "function": [
                        "name": "line_count",
                        "description": "Count lines in text.",
                        "parameters": [
                            "type": "object",
                            "properties": [
                                "text": [
                                    "type": "string",
                                ] as [String: any Sendable],
                            ] as [String: any Sendable],
                            "required": ["text"],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ],
            "tool_choice": "auto",
            "add_generation_prompt": true,
        ])

        #expect(rendered.contains("<|tool>declaration:line_count"))
        #expect(!rendered.contains("Tool use is REQUIRED for this assistant turn."))
        #expect(!rendered.contains("Required call shape for the current request"))
        #expect(rendered.contains("Answer plainly in one short sentence: what color is grass?"))
        #expect(rendered.hasSuffix("<|turn>model\n"))
    }

    @Test
    func gemmaRequiredToolFallbackKeepsUserTurnContract() throws {
        let template = try Template(ChatTemplateFallbacks.gemma4WithTools)
        var context: [String: any Sendable] = [
            "messages": [
                [
                    "role": "user",
                    "content": "Use the line_count tool on this exact text: red\ngreen\nblue",
                ],
            ],
            "tools": [
                [
                    "type": "function",
                    "function": [
                        "name": "line_count",
                        "parameters": [
                            "type": "object",
                            "properties": [
                                "text": [
                                    "type": "string",
                                ] as [String: any Sendable],
                            ] as [String: any Sendable],
                            "required": ["text"],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ],
            "tool_choice": "required",
            "tool_choice_name": "line_count",
            "add_generation_prompt": true,
        ]
        let rendered = try template.renderGemma4(context)
        context["tool_choice"] = "auto"
        context.removeValue(forKey: "tool_choice_name")
        let automatic = try template.renderGemma4(context)

        #expect(rendered.contains("<|tool>declaration:line_count"))
        #expect(rendered == automatic, "Tool choice is validated at response completion, not by rewriting prompts")
        for injected in ["Tool use is REQUIRED", "Use exactly this function name", "Required arguments:",
            "Required call shape for the current request", "Copy the `text` value exactly"] {
            #expect(!rendered.contains(injected))
        }
        #expect(!rendered.contains("<|tool_call>call:line_count{"),
            "The runtime must not invent an assistant call from user text")
        #expect(!rendered.contains("FUNCTION_NAME"))
        #expect(!rendered.contains("ARGUMENT_NAME"))
        #expect(!rendered.contains("VALUE_FOR_text"))
        #expect(rendered.contains("Use the line_count tool on this exact text: red\ngreen\nblue"))
        #expect(rendered.hasSuffix("<|turn>model\n"))
    }

    @Test
    func fallbackVariantsKeepMediaAndCompletedToolHistoryAcrossChoices() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let sidecar = try String(contentsOf: root.appendingPathComponent(
            "Libraries/MLXLMCommon/ChatTemplates/Gemma4WithTools.jinja"), encoding: .utf8)
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": [
                ["type": "text", "text": "Original media question."],
                ["type": "image"], ["type": "audio"], ["type": "video"],
            ]],
            ["role": "assistant", "content": "Earlier narration.", "tool_calls": [[
                "function": ["name": "line_count", "arguments": ["text": "earlier\ninput"]] as [String: any Sendable],
            ]]],
            ["role": "tool", "name": "line_count", "content": "Earlier result: 2."],
            ["role": "user", "content": "Use line_count on this exact text: current\ninput"],
        ]
        let tools: [[String: any Sendable]] = [[
            "function": ["name": "line_count", "parameters": ["type": "object",
                "properties": ["text": ["type": "string"]], "required": ["text"]] as [String: any Sendable]] as [String: any Sendable],
        ]]
        var baseline: String?
        for source in [ChatTemplateFallbacks.gemma4WithTools, sidecar] {
            let template = try Template(source)
            for choice in ["auto", "required"] {
                for named in [false, true] where choice == "required" || !named {
                    var context: [String: any Sendable] = ["bos_token": "<bos>", "messages": messages,
                        "tools": tools, "tool_choice": choice, "add_generation_prompt": true]
                    if named { context["tool_choice_name"] = "line_count" }
                    let rendered = try template.renderGemma4(context)
                    if let baseline { #expect(rendered == baseline) } else { baseline = rendered }
                    for original in ["Original media question.", "Earlier narration.", "earlier\ninput",
                        "Earlier result: 2.", "Use line_count on this exact text: current\ninput",
                        "<|image|>", "<|audio|>", "<|video|>", "<|tool_call>call:line_count{"] {
                        #expect(rendered.contains(original))
                    }
                    #expect(!rendered.contains("Tool use is REQUIRED"))
                    #expect(!rendered.contains("Required call shape for the current request"))
                }
            }
        }
    }
}
