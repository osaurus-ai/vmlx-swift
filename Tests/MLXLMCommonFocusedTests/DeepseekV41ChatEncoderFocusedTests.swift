// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 (1017) prompt-contract tests, plus guards that DSV4-0731
// prompts are untouched by the V4.1 work. The golden strings were produced by
// the bundle's authoritative `encoding/encoding.py` (`encode_messages`) for
// DeepSeek-V4.1-Flash-1017; regenerate them from that file if it changes.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("DeepSeek-V4.1 chat encoder contract")
struct DeepseekV41ChatEncoderFocusedTests {
    typealias Msg = DeepseekV4ChatEncoder.Message

    /// `encoding.py encode_messages(...)` outputs, keyed by case name.
    static let golden: [String: String] = {
        let json = #"""
{"user_low": "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>Say hello.<｜Assistant｜><think>", "user_max": "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 100 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>Say hello.<｜Assistant｜><think>", "user_num30": "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 30 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>Hi<｜Assistant｜><think>", "sys_user_high": "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\nYou are helpful.<｜User｜>Hi<｜Assistant｜><think>", "sys_user_chat": "<｜begin▁of▁sentence｜><｜System｜>You are helpful.<｜User｜>Hi<｜Assistant｜></think>", "mid_system": "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)\n\nS0<｜User｜>Hi<｜Assistant｜></think>Hello!<｜end▁of▁sentence｜><｜System｜>Now answer in French.<｜Assistant｜><think>", "tools_history": "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)\n\n\n\n## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜ calls>\" block like the following:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"$TOOL_NAME\">\n<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n...\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{\"name\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n<｜User｜>Weather in Paris?<｜Assistant｜><think>call tool</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜><｜User｜><tool_result>sunny</tool_result><｜Assistant｜><think>"}
"""#
        return (try? JSONDecoder().decode([String: String].self, from: Data(json.utf8))) ?? [:]
    }()

    static let v41 = DeepseekV4ChatEncoder(family: .deepseekV41)

    static let weatherTool: [String: any Sendable] = [
        "type": "function",
        "function": [
            "name": "get_weather",
            "description": "Get weather",
            "parameters": [
                "type": "object",
                "properties": ["city": ["type": "string"] as [String: any Sendable]] as [String: any Sendable],
                "required": ["city"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    @Test("numeric effort ladder anchors render inside the System token")
    func numericEffortAnchors() {
        let user = [Msg(role: .user, content: "Say hello.")]
        #expect(Self.v41.encode(messages: user, reasoningEffort: .low) == Self.golden["user_low"])
        #expect(Self.v41.encode(messages: user, reasoningEffort: .max) == Self.golden["user_max"])
        let hi = [Msg(role: .user, content: "Hi")]
        #expect(Self.v41.encode(messages: hi, reasoningBudget: 30) == Self.golden["user_num30"])
    }

    @Test("leading system message shares the System token with the effort prefix")
    func systemMessageRendering() {
        let msgs = [Msg(role: .system, content: "You are helpful."), Msg(role: .user, content: "Hi")]
        #expect(Self.v41.encode(messages: msgs, reasoningEffort: .high) == Self.golden["sys_user_high"])
        #expect(Self.v41.encode(messages: msgs, thinkingMode: .chat) == Self.golden["sys_user_chat"])
    }

    @Test("mid-conversation system message behaves like a user turn")
    func midConversationSystem() {
        let msgs = [
            Msg(role: .system, content: "S0"),
            Msg(role: .user, content: "Hi"),
            Msg(role: .assistant, content: "Hello!", reasoningContent: "greet"),
            Msg(role: .system, content: "Now answer in French."),
        ]
        #expect(Self.v41.encode(messages: msgs, reasoningEffort: .low) == Self.golden["mid_system"])
    }

    @Test("tool schemas and tool-call history use the leading-space DSML tags")
    func toolHistory() {
        let msgs = [
            Msg(role: .system, content: "", tools: [Self.weatherTool]),
            Msg(role: .user, content: "Weather in Paris?"),
            Msg(
                role: .assistant, content: "", reasoningContent: "call tool",
                toolCalls: [.init(id: "c1", name: "get_weather", arguments: #"{"city": "Paris"}"#)]),
            Msg(role: .tool, content: "sunny", toolCallId: "c1"),
        ]
        let prompt = Self.v41.encode(messages: msgs, reasoningEffort: .low)
        #expect(prompt == Self.golden["tools_history"])
        #expect(!prompt.contains("\u{FF5C}DSML\u{FF5C}tool_calls"))
    }

    @Test("OpenAI adapter maps the host effort ladder to numeric budgets")
    func adapterEffortLadder() throws {
        let messages: [[String: any Sendable]] = [["role": "user", "content": "Hi"]]
        func render(_ effort: (any Sendable)?) throws -> String {
            var ctx: [String: any Sendable] = ["enable_thinking": true]
            if let effort { ctx["reasoning_effort"] = effort }
            return try DeepseekV4ChatEncoder.renderOpenAIChat(
                messages: messages, tools: nil, additionalContext: ctx,
                addGenerationPrompt: true, family: .deepseekV41)
        }
        let expected: [(String, Int)] = [
            ("none", 25), ("minimal", 25), ("low", 50), ("medium", 62),
            ("high", 75), ("xhigh", 88), ("max", 100), ("7", 7), ("250", 100),
        ]
        for (effort, budget) in expected {
            #expect(try render(effort).contains("<\u{FF5C}System\u{FF5C}>Reasoning Effort: \(budget) (range"))
        }
        #expect(try render(42).contains("Reasoning Effort: 42 (range"))
        // Documented vmlx default: absent effort renders low=50 (reference default is high=75).
        #expect(try render(nil).contains(
            "Reasoning Effort: \(DeepseekV4ChatEncoder.deepseekV41DefaultReasoningBudget) (range"))
        #expect(DeepseekV4ChatEncoder.deepseekV41DefaultReasoningBudget == 50)
    }

    @Test("family resolves only from an explicit V4.1 model_type")
    func familyDetection() {
        #expect(DeepseekV4ChatFamily(modelType: "deepseek_v41") == .deepseekV41)
        #expect(DeepseekV4ChatFamily(modelType: "deepseek_v41_text") == .deepseekV41)
        #expect(DeepseekV4ChatFamily(modelType: "deepseek_v4") == .deepseekV4)
        #expect(DeepseekV4ChatFamily(modelType: nil) == .deepseekV4)
        let nested = Data(#"{"model_type":"x","text_config":{"model_type":"deepseek_v41_text"}}"#.utf8)
        #expect(DeepseekV4ChatFamily.detect(configData: nested) == .deepseekV41)
        let v4 = Data(#"{"model_type":"deepseek_v4"}"#.utf8)
        #expect(DeepseekV4ChatFamily.detect(configData: v4) == .deepseekV4)
    }

    // MARK: - DSV4-0731 isolation

    @Test("default encoder keeps the 0731 text prefaces and never emits the System token")
    func default0731Unchanged() throws {
        let enc = DeepseekV4ChatEncoder()
        #expect(enc.family == .deepseekV4)
        let msgs = [Msg(role: .system, content: "You are helpful."), Msg(role: .user, content: "Hi")]
        let low = enc.encode(messages: msgs)
        #expect(low == DeepseekV4Tokens.bos + DeepseekV4ChatEncoder.reasoningEffortLowPreface
            + "You are helpful." + DeepseekV4Tokens.user + "Hi"
            + DeepseekV4Tokens.assistant + DeepseekV4Tokens.thinkStart)
        #expect(enc.encode(messages: msgs, reasoningEffort: .high)
            .contains(DeepseekV4ChatEncoder.reasoningEffortHighPreface))
        #expect(enc.encode(messages: msgs, reasoningEffort: .max)
            .contains(DeepseekV4ChatEncoder.reasoningEffortMaxPreface))
        // A numeric budget is a V4.1 concept and must not leak into 0731.
        #expect(enc.encode(messages: msgs, reasoningBudget: 30) == low)
        let adapted = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: [["role": "system", "content": "S"], ["role": "user", "content": "Hi"]],
            tools: nil, additionalContext: ["reasoning_effort": "xhigh"], addGenerationPrompt: true)
        #expect(!adapted.contains("\u{FF5C}System\u{FF5C}"))
        #expect(adapted.contains(DeepseekV4ChatEncoder.reasoningEffortLowPreface))
        let tools = DeepseekV4ChatEncoder.renderTools([Self.weatherTool])
        #expect(tools.contains("<\u{FF5C}DSML\u{FF5C}tool_calls>"))
        #expect(!tools.contains("\u{FF5C}DSML\u{FF5C} calls"))
    }

    @Test("reasoning floor arms on 0731 prefaces and never on the V4.1 numeric head")
    func floorGating() {
        let tail = DeepseekV4Tokens.assistant + DeepseekV4Tokens.thinkStart
        let v4Head = DeepseekV4Tokens.bos + DeepseekV4ChatEncoder.reasoningEffortLowPreface
        #expect(MinimumReasoningFloor.railRequiresFloor(promptHead: v4Head, promptTail: tail))
        let v41Head = Self.v41.encode(messages: [Msg(role: .user, content: "Hi")])
        #expect(!MinimumReasoningFloor.railRequiresFloor(promptHead: v41Head, promptTail: tail))
    }
}
