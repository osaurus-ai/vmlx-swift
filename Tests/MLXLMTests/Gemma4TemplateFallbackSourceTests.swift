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
        #expect(source.contains("let gemmaRequiredToolChoice"))
        #expect(source.contains("chat-template required tools -> Gemma4WithTools fallback engaged"))
        #expect(source.contains(#""Gemma4WithTools", MLXLMCommon.ChatTemplateFallbacks.gemma4WithTools"#))
        #expect(source.contains(#""Gemma4Minimal",   MLXLMCommon.ChatTemplateFallbacks.gemma4Minimal"#))
    }

    @Test
    func gemmaRequiredToolFallbackEmitsExactTextNativeCallShape() throws {
        let template = try Template(ChatTemplateFallbacks.gemma4WithTools)
        let rendered = try template.renderGemma4([
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
        ])

        #expect(rendered.contains("<|tool>declaration:line_count"))
        #expect(rendered.contains("The current assistant response MUST be a function call."))
        #expect(rendered.contains("Use the `line_count` function."))
        #expect(rendered.contains("Required parameters for `line_count`: text."))
        #expect(rendered.contains("<|tool_call>call:FUNCTION_NAME{ARGUMENT_NAME:<|\"|>ARGUMENT_VALUE<|\"|>}<tool_call|>"))
        #expect(rendered.contains("Do not wrap the argument value in quote characters"))
        #expect(rendered.contains("do not add or remove whitespace or spaces after newlines"))
        #expect(rendered.contains("<|tool_call>call:line_count{text:<|\"|>red\ngreen\nblue<|\"|>}<tool_call|>"))
        #expect(rendered.hasSuffix("<|turn>model\n"))
    }
}
