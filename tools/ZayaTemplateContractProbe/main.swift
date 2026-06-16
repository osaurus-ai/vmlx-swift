import Foundation
import MLXLMCommon
import VMLXJinja

private extension Template {
    func renderProbe(_ context: [String: any Sendable]) throws -> String {
        var values: [String: Value] = [:]
        for (key, value) in context {
            values[key] = try Value(any: value)
        }
        return try render(values)
    }
}

private func fail(_ message: String, rendered: String? = nil) -> Never {
    FileHandle.standardError.write(Data(("FAIL \(message)\n").utf8))
    if let rendered {
        FileHandle.standardError.write(Data(("--- rendered ---\n\(rendered)\n").utf8))
    }
    Foundation.exit(1)
}

private func require(_ condition: @autoclosure () -> Bool, _ message: String, rendered: String) {
    if !condition() {
        fail(message, rendered: rendered)
    }
}

private let lineCountTool: [String: any Sendable] = [
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
]

do {
    let template = try Template(ChatTemplateFallbacks.zayaVLVisionToolMinimal)

    let optionalTools = try template.renderProbe([
        "messages": [
            [
                "role": "user",
                "content": "Use line_count on this exact text: one\ntwo",
            ] as [String: any Sendable],
        ],
        "tools": [lineCountTool],
        "bos_token": "<bos>",
        "add_generation_prompt": true,
        "enable_thinking": false,
    ])

    require(optionalTools.contains("<zyphra_tool_call>"), "optional-tools contract must show ZAYA tool wrapper", rendered: optionalTools)
    require(!optionalTools.contains("The current assistant response MUST be a function call."), "optional tools must not force a required tool call", rendered: optionalTools)
    require(!optionalTools.contains("Use the `line_count` function."), "optional tools must not inject named required tool text", rendered: optionalTools)
    require(optionalTools.hasSuffix("<|im_start|>assistant\n"), "optional tools must preserve assistant generation rail", rendered: optionalTools)

    let requiredTools = try template.renderProbe([
        "messages": [
            [
                "role": "user",
                "content": "Use line_count on this exact text: one\ntwo",
            ] as [String: any Sendable],
        ],
        "tools": [lineCountTool],
        "bos_token": "<bos>",
        "add_generation_prompt": true,
        "enable_thinking": false,
        "tool_choice": "required",
        "tool_choice_name": "line_count",
    ])

    require(requiredTools.contains("<zyphra_tool_call>"), "required contract must show ZAYA tool wrapper", rendered: requiredTools)
    require(requiredTools.contains("The current assistant response MUST be a function call."), "required contract must include must-call instruction", rendered: requiredTools)
    require(requiredTools.contains("Use the `line_count` function."), "required named contract must include selected function", rendered: requiredTools)
    require(!requiredTools.contains("<think>"), "required contract must not synthesize thinking tags", rendered: requiredTools)
    require(!requiredTools.contains("enable_thinking"), "required contract must not leak template kwargs", rendered: requiredTools)
    require(requiredTools.hasSuffix("<|im_start|>assistant\n"), "required contract must preserve assistant generation rail", rendered: requiredTools)

    print("PASS ZAYA required/named tool-choice template contract")
} catch {
    fail("ZAYA template probe threw \(error)")
}
