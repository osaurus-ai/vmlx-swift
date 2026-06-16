import Foundation
import MLXLMCommon
import Testing
import VMLXJinja

private extension Template {
    func renderGemma4Context(_ context: [String: any Sendable]) throws -> String {
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
        #expect(source.contains(#""Gemma4WithTools", MLXLMCommon.ChatTemplateFallbacks.gemma4WithTools"#))
        #expect(source.contains(#""Gemma4Minimal",   MLXLMCommon.ChatTemplateFallbacks.gemma4Minimal"#))
    }

    @Test
    func gemma4WithToolsHonorsRequiredNamedToolChoice() throws {
        let template = try Template(ChatTemplateFallbacks.gemma4WithTools)
        let rendered = try template.renderGemma4Context([
            "messages": [
                [
                    "role": "user",
                    "content": "Use line_count on this exact text: red\ngreen\nblue",
                ] as [String: any Sendable],
            ],
            "tools": [
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
                            "required": ["text"],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ],
            "bos_token": "<bos>",
            "add_generation_prompt": true,
            "tool_choice": "required",
            "tool_choice_name": "line_count",
        ])

        #expect(rendered.contains("<|tool>declaration:line_count"))
        #expect(rendered.contains("Tool use is REQUIRED for this turn:"))
        #expect(rendered.contains("call `line_count` exactly once."))
        #expect(rendered.contains("Output only <|tool_call>call:name{args}<tool_call|>; do not answer in prose."))
        #expect(rendered.hasSuffix("<|turn>model\n"))
    }
}
