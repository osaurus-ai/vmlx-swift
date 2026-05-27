// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Tests for `DeepseekV4ChatEncoder` — Swift port of
// `encoding_dsv4.py` (744 LOC Python reference). Covers the §4
// "Chat template + reasoning modes" surface of the DSV4 runtime
// architecture guide.

import Foundation
import MLXLLM
import MLXLMCommon
import Testing

@Suite("DeepseekV4 Chat Encoder")
struct DeepseekV4ChatEncoderTests {

    typealias Msg = DeepseekV4ChatEncoder.Message
    typealias Role = DeepseekV4ChatEncoder.MessageRole
    typealias TC = DeepseekV4ChatEncoder.ToolCall
    typealias Block = DeepseekV4ChatEncoder.MessageContentBlock

    // MARK: - Basic prompt shapes

    @Test("chat mode ends with closed </think> tail")
    func chatMode() {
        let encoder = DeepseekV4ChatEncoder()
        let prompt = encoder.encode(
            messages: [Msg(role: .user, content: "What is 2+2?")],
            thinkingMode: .chat)
        #expect(prompt.hasPrefix(DeepseekV4Tokens.bos))
        #expect(prompt.contains(DeepseekV4Tokens.user + "What is 2+2?"))
        #expect(prompt.hasSuffix(DeepseekV4Tokens.assistant + DeepseekV4Tokens.thinkEnd),
            "chat mode must end with <｜Assistant｜></think> for closed empty block")
    }

    @Test("thinking mode ends with open <think> tail")
    func thinkingMode() {
        let encoder = DeepseekV4ChatEncoder()
        let prompt = encoder.encode(
            messages: [Msg(role: .user, content: "Solve x^2=4.")],
            thinkingMode: .thinking)
        #expect(prompt.hasPrefix(DeepseekV4Tokens.bos))
        #expect(prompt.hasSuffix(DeepseekV4Tokens.assistant + DeepseekV4Tokens.thinkStart),
            "thinking mode must end with <｜Assistant｜><think> so model generates reasoning")
    }

    @Test("reasoning_effort=max emits the full reasoning-effort preface before turn 0")
    func reasoningEffortMaxPreface() {
        let encoder = DeepseekV4ChatEncoder()
        let prompt = encoder.encode(
            messages: [Msg(role: .user, content: "hi")],
            thinkingMode: .thinking,
            reasoningEffort: .max)
        #expect(prompt.contains("Reasoning Effort: Absolute maximum"),
            "max effort must prepend the preface at index 0")
    }

    @Test("reasoning_effort=high or nil does NOT emit the max preface")
    func highEffortNoPreface() {
        let encoder = DeepseekV4ChatEncoder()
        let promptHigh = encoder.encode(
            messages: [Msg(role: .user, content: "hi")],
            thinkingMode: .thinking,
            reasoningEffort: .high)
        let promptNil = encoder.encode(
            messages: [Msg(role: .user, content: "hi")],
            thinkingMode: .thinking)
        #expect(!promptHigh.contains("Reasoning Effort: Absolute"))
        #expect(!promptNil.contains("Reasoning Effort: Absolute"))
    }

    // MARK: - Multi-turn drop_earlier_reasoning

    @Test("drop_earlier_reasoning=true strips prior assistant's thinking block")
    func dropEarlierReasoningOn() {
        let encoder = DeepseekV4ChatEncoder()
        let prompt = encoder.encode(
            messages: [
                Msg(role: .user, content: "Turn 1 question?"),
                Msg(
                    role: .assistant,
                    content: "Answer 1.",
                    reasoningContent: "Thinking about turn 1..."),
                Msg(role: .user, content: "Turn 2 question?"),
            ],
            thinkingMode: .thinking,
            dropEarlierReasoning: true)
        // Prior turn's thinking must NOT appear verbatim in the prompt.
        #expect(!prompt.contains("Thinking about turn 1..."),
            "prior turn reasoning must be stripped under drop_earlier_reasoning")
        // But the final answer MUST survive.
        #expect(prompt.contains("Answer 1."))
        // Prompt must end with open <think> since we're in thinking mode
        // and the last message is user.
        #expect(prompt.hasSuffix(DeepseekV4Tokens.assistant + DeepseekV4Tokens.thinkStart))
    }

    @Test("drop_earlier_reasoning=false keeps prior reasoning")
    func dropEarlierReasoningOff() {
        let encoder = DeepseekV4ChatEncoder()
        let prompt = encoder.encode(
            messages: [
                Msg(role: .user, content: "Turn 1 question?"),
                Msg(
                    role: .assistant,
                    content: "Answer 1.",
                    reasoningContent: "Thinking about turn 1..."),
                Msg(role: .user, content: "Turn 2 question?"),
            ],
            thinkingMode: .thinking,
            dropEarlierReasoning: false)
        #expect(prompt.contains("Thinking about turn 1..."),
            "drop_earlier_reasoning=false must preserve prior reasoning blocks")
    }

    @Test("tools present forces drop_earlier_reasoning=false (mid-agent trajectory)")
    func toolsForceKeepReasoning() {
        let encoder = DeepseekV4ChatEncoder()
        let prompt = encoder.encode(
            messages: [
                Msg(
                    role: .system,
                    content: "You are helpful.",
                    tools: [[
                        "type": "function",
                        "function": [
                            "name": "get_weather",
                            "description": "Get the weather",
                            "parameters": ["location": "string"] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable]]),
                Msg(role: .user, content: "Turn 1"),
                Msg(
                    role: .assistant,
                    content: "Answer 1.",
                    reasoningContent: "Reasoning 1"),
                Msg(role: .user, content: "Turn 2"),
            ],
            thinkingMode: .thinking,
            dropEarlierReasoning: true)
        // With tools present, drop_earlier_reasoning becomes effective-false.
        #expect(prompt.contains("Reasoning 1"),
            "tools in any message must override drop_earlier_reasoning=true")
    }

    // MARK: - Tool call rendering (DSML)

    @Test("assistant tool_calls render as DSML invoke block")
    func assistantToolCall() {
        let encoder = DeepseekV4ChatEncoder()
        let prompt = encoder.encode(
            messages: [
                Msg(role: .user, content: "What's the weather?"),
                Msg(
                    role: .assistant,
                    content: "Checking.",
                    reasoningContent: "",
                    toolCalls: [
                        TC(name: "get_weather", arguments: "{\"location\":\"Paris\"}")
                    ]),
                Msg(
                    role: .tool,
                    content: "{\"temp\":22}",
                    toolCallId: "call_1"),
                Msg(role: .user, content: "Thanks, now for Tokyo."),
            ],
            thinkingMode: .thinking,
            dropEarlierReasoning: false)
        let dsml = DeepseekV4Tokens.dsml
        #expect(prompt.contains("<\(dsml)tool_calls>"), "must emit DSML tool_calls opener")
        #expect(prompt.contains("<\(dsml)invoke name=\"get_weather\">"),
            "must emit DSML invoke with function name")
        #expect(prompt.contains("<\(dsml)parameter name=\"location\" string=\"true\">Paris"),
            "string params must carry string=\"true\"")
        #expect(prompt.contains("<tool_result>{\"temp\":22}</tool_result>"),
            "tool result must merge into next user turn as <tool_result>")
    }

    @Test("required tool choice survives tool-result merge into next user turn")
    func requiredToolChoiceSurvivesToolResultMerge() {
        let encoder = DeepseekV4ChatEncoder()
        let prompt = encoder.encode(
            messages: [
                Msg(role: .user, content: "Count lines in alpha beta gamma."),
                Msg(
                    role: .assistant,
                    content: "",
                    toolCalls: [
                        TC(
                            id: "call_line_count",
                            name: "line_count",
                            arguments: "{\"text\":\"alpha\\nbeta\\ngamma\"}")
                    ]),
                Msg(
                    role: .tool,
                    content: "{\"lines\":3}",
                    toolCallId: "call_line_count"),
                Msg(
                    role: .user,
                    content: "Now read /Users/eric/Desktop/testmandel/mandelbrot.py.",
                    task: "action"),
            ],
            thinkingMode: .chat,
            toolChoiceRequired: true,
            toolChoiceName: "line_count")

        #expect(prompt.contains("<tool_result>{\"lines\":3}</tool_result>"))
        #expect(prompt.contains(
            "<tool_result>{\"lines\":3}</tool_result>\n\nNow read /Users/eric/Desktop/testmandel/mandelbrot.py."))
        #expect(prompt.hasSuffix(
            DeepseekV4Tokens.assistant
                + DeepseekV4Tokens.thinkEnd
                + DeepseekV4Tokens.taskSPTokens["action"]!),
            "the latest user action task must survive merging a preceding tool result")
    }

    @Test("required tool choice reminder stays after latest user turn without action rail")
    func requiredToolChoiceReminderFollowsLatestUserWithoutActionRail() throws {
        let encoder = DeepseekV4ChatEncoder()
        let finalUser = "Now use line_count on this exact text: one\ntwo"
        let prompt = encoder.encode(
            messages: [
                Msg(
                    role: .system,
                    content: "Helpful assistant.",
                    tools: [lineCountToolSchema()]),
                Msg(role: .user, content: "Use line_count on this exact text: red\ngreen\nblue"),
                Msg(
                    role: .assistant,
                    content: "",
                    toolCalls: [
                        TC(
                            id: "call_lines_1",
                            name: "line_count",
                            arguments: "{\"text\":\"red\\ngreen\\nblue\"}")
                    ]),
                Msg(role: .tool, content: "{\"lines\":3}", toolCallId: "call_lines_1"),
                Msg(role: .user, content: "How many lines were counted? Do not call another tool."),
                Msg(role: .assistant, content: "Three lines were counted."),
                Msg(role: .user, content: finalUser),
            ],
            thinkingMode: .chat,
            toolChoiceRequired: true)

        let finalUserRange = try #require(prompt.range(of: finalUser))
        let reminderRange = try #require(prompt.range(of: DeepseekV4Tokens.latestReminder))
        let assistantTail = DeepseekV4Tokens.assistant + DeepseekV4Tokens.thinkEnd
        let tailRange = try #require(prompt.range(of: assistantTail, options: .backwards))

        #expect(finalUserRange.upperBound <= reminderRange.lowerBound, "Prompt: \(prompt)")
        #expect(reminderRange.upperBound <= tailRange.lowerBound, "Prompt: \(prompt)")
        #expect(prompt.hasSuffix(assistantTail), "Prompt: \(prompt)")
        #expect(prompt.contains("Use the `line_count` function."), "Prompt: \(prompt)")
        #expect(!prompt.contains(DeepseekV4Tokens.taskSPTokens["action"]!), "Prompt: \(prompt)")
        #expect(
            prompt.contains("<tool_result>{\"lines\":3}</tool_result>\n\n\(finalUser)"),
            "Required DSV4 tool turns after a completed prose answer must compact back to tool_result + latest user. Prompt: \(prompt)"
        )
        #expect(
            !prompt.contains("How many lines were counted? Do not call another tool."),
            "Completed prose-only answer exchange should be dropped from this required tool prompt. Prompt: \(prompt)"
        )
        #expect(
            !prompt.contains("Three lines were counted."),
            "Completed prose-only assistant answer should not sit before the latest required DSML turn. Prompt: \(prompt)"
        )
    }

    @Test("tool_call arguments with non-string values render string=\"false\" + JSON")
    func toolCallNumericArgs() {
        let prompt = DeepseekV4ChatEncoder.renderToolCallInvoke(
            name: "set_config", arguments: "{\"retries\":3,\"enabled\":true}")
        #expect(prompt.contains("name=\"retries\" string=\"false\">3"))
        #expect(prompt.contains("name=\"enabled\" string=\"false\">true"))
    }

    // MARK: - Role handling

    @Test("system message with tools injects tools schema after content")
    func systemWithTools() {
        let encoder = DeepseekV4ChatEncoder()
        let prompt = encoder.encode(
            messages: [
                Msg(
                    role: .system,
                    content: "Helpful assistant.",
                    tools: [[
                        "function": [
                            "name": "search",
                            "description": "search the web",
                            "parameters": ["q": "string"] as [String: any Sendable],
                        ] as [String: any Sendable]
                    ] as [String: any Sendable]]),
                Msg(role: .user, content: "hi"),
            ],
            thinkingMode: .chat)
        #expect(prompt.contains("## Tools"))
        #expect(prompt.contains("\"name\": \"search\"") || prompt.contains("\"name\":\"search\""))
    }

    @Test("developer role opens with <｜User｜> like user")
    func developerRole() {
        let encoder = DeepseekV4ChatEncoder()
        let prompt = encoder.encode(
            messages: [
                Msg(role: .developer, content: "Internal instruction"),
                Msg(role: .user, content: "go"),
            ],
            thinkingMode: .chat)
        #expect(prompt.contains(DeepseekV4Tokens.user + "Internal instruction"))
    }

    @Test("latest_reminder role emits its own marker token")
    func latestReminderRole() {
        let encoder = DeepseekV4ChatEncoder()
        let prompt = encoder.encode(
            messages: [
                Msg(role: .user, content: "main query"),
                Msg(role: .latestReminder, content: "reminder text"),
            ],
            thinkingMode: .chat)
        #expect(prompt.contains(DeepseekV4Tokens.latestReminder + "reminder text"))
    }

    private func lineCountToolSchema() -> [String: any Sendable] {
        [
            "function": [
                "name": "line_count",
                "description": "Count newline-separated text lines.",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "text": ["type": "string"] as [String: any Sendable]
                    ] as [String: any Sendable],
                    "required": ["text"],
                ] as [String: any Sendable],
            ] as [String: any Sendable]
        ] as [String: any Sendable]
    }

    // MARK: - BOS handling

    @Test("addBOS=false omits BOS; with context, BOS also omitted")
    func bosToggling() {
        let encoder = DeepseekV4ChatEncoder()
        let noBOS = encoder.encode(
            messages: [Msg(role: .user, content: "hi")],
            thinkingMode: .chat,
            addBOS: false)
        #expect(!noBOS.hasPrefix(DeepseekV4Tokens.bos))

        let withContext = encoder.encode(
            messages: [Msg(role: .user, content: "hi")],
            thinkingMode: .chat,
            context: [Msg(role: .user, content: "prior")],
            addBOS: true)
        #expect(!withContext.hasPrefix(DeepseekV4Tokens.bos),
            "context non-empty must skip BOS even when addBOS=true")
    }

    // MARK: - Configuration round-trip

    @Test("DeepseekV4Configuration round-trips from a minimal config.json")
    func configRoundTrip() throws {
        let json = """
            {
              "vocab_size": 129280,
              "hidden_size": 4096,
              "num_hidden_layers": 43,
              "num_attention_heads": 64,
              "num_key_value_heads": 1,
              "head_dim": 512,
              "qk_rope_head_dim": 64,
              "q_lora_rank": 1024,
              "o_groups": 8,
              "o_lora_rank": 1024,
              "n_routed_experts": 256,
              "n_shared_experts": 1,
              "num_experts_per_tok": 6,
              "moe_intermediate_size": 2048,
              "num_hash_layers": 3,
              "scoring_func": "sqrtsoftplus",
              "norm_topk_prob": true,
              "routed_scaling_factor": 1.5,
              "swiglu_limit": 10.0,
              "hc_mult": 4,
              "hc_sinkhorn_iters": 20,
              "hc_eps": 1.0e-6,
              "rope_theta": 10000.0,
              "compress_rope_theta": 160000.0,
              "sliding_window": 128,
              "compress_ratios": [0, 4, 128, 0, 4, 128],
              "use_attn_sink": true
            }
            """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(DeepseekV4Configuration.self, from: json)
        #expect(cfg.vocabSize == 129280)
        #expect(cfg.numHiddenLayers == 43)
        #expect(cfg.headDim == 512)
        #expect(cfg.swigluLimit == 10.0)
        #expect(cfg.hcMult == 4)
        #expect(cfg.compressRatios == [0, 4, 128, 0, 4, 128])
        #expect(cfg.useAttnSink == true)
        // Per-layer helpers
        #expect(cfg.isHashLayer(0))
        #expect(cfg.isHashLayer(2))
        #expect(!cfg.isHashLayer(3))
        #expect(!cfg.hasCompressor(layer: 0))
        #expect(cfg.hasCompressor(layer: 1))
        #expect(cfg.hasCompressor(layer: 2))
        #expect(cfg.ropeTheta(forLayer: 0) == 10000.0)
        #expect(cfg.ropeTheta(forLayer: 1) == 160000.0)
    }

    @Test("DeepseekV4Configuration fills defaults when fields are absent")
    func configDefaults() throws {
        let json = "{}".data(using: .utf8)!
        let cfg = try JSONDecoder().decode(DeepseekV4Configuration.self, from: json)
        #expect(cfg.vocabSize == 129280, "default vocab must match DSV4-Flash")
        #expect(cfg.headDim == 512)
        #expect(cfg.swigluLimit == 10.0)
        #expect(cfg.hcMult == 4)
    }

    @Test("DeepseekV4Configuration preserves routed expert bit plan")
    func configurationRoutedExpertBitPlan() throws {
        let json = """
            {
              "model_type": "deepseek_v4",
              "num_hidden_layers": 43,
              "mxtq_bits": {
                "routed_expert": 2,
                "attention": 8,
                "shared_expert": 8
              },
              "routed_expert_bit_plan": {
                "default_bits": 2,
                "codec": "mxtq",
                "routed_layer_bits": {
                  "23": 4,
                  "25": 4,
                  "28": 4,
                  "34": 4,
                  "36": 4
                }
              }
            }
            """.data(using: .utf8)!

        let cfg = try JSONDecoder().decode(DeepseekV4Configuration.self, from: json)

        #expect(cfg.routedExpertBits(forLayer: 0) == 2)
        #expect(cfg.routedExpertBits(forLayer: 22) == 2)
        #expect(cfg.routedExpertBits(forLayer: 23) == 4)
        #expect(cfg.routedExpertBits(forLayer: 25) == 4)
        #expect(cfg.routedExpertBits(forLayer: 28) == 4)
        #expect(cfg.routedExpertBits(forLayer: 34) == 4)
        #expect(cfg.routedExpertBits(forLayer: 36) == 4)
        #expect(cfg.routedExpertBits(forLayer: 42) == 2)

        let model = DeepseekV4JANGTQModel(cfg)
        #expect(model.routedExpertBitsByLayer[23] == 4)
        #expect(model.routedExpertBitsByLayer[42] == 2)

        let roundTripped = try JSONDecoder().decode(
            DeepseekV4Configuration.self,
            from: JSONEncoder().encode(cfg))
        #expect(roundTripped.routedExpertBits(forLayer: 23) == 4)
        #expect(roundTripped.routedExpertBits(forLayer: 42) == 2)
    }
}
