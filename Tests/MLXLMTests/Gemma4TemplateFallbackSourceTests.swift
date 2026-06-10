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
    func gemmaRequiredToolFallbackKeepsUserTurnContract() throws {
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
        #expect(rendered.contains("Tool use is REQUIRED for this assistant turn."))
        #expect(rendered.contains("Use exactly this function name: line_count"))
        #expect(rendered.contains("Required arguments: text"))
        #expect(rendered.contains("Required call shape for the current request"))
        #expect(rendered.components(separatedBy: "Tool use is REQUIRED for this assistant turn.").count == 2)
        #expect(rendered.contains("""
            <|tool_call>call:line_count{text:<|"|>red
            green
            blue<|"|>}<tool_call|>
            """))
        #expect(!rendered.contains(#"Do not replace \n with a physical newline"#))
        let currentShapeRange = try #require(rendered.range(of: "Required call shape for the current request"))
        let userContentRange = try #require(rendered.range(of: "Use the line_count tool on this exact text: red\ngreen\nblue"))
        #expect(userContentRange.lowerBound < currentShapeRange.lowerBound)
        #expect(!rendered.contains("FUNCTION_NAME"))
        #expect(!rendered.contains("ARGUMENT_NAME"))
        #expect(!rendered.contains("VALUE_FOR_text"))
        #expect(rendered.contains("Use the line_count tool on this exact text: red\ngreen\nblue"))
        #expect(rendered.hasSuffix("<|turn>model\n"))
    }
}
