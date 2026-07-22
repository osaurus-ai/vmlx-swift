// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Tests the DSV4-era `jang_config.json -> chat` schema that
// `JangLoader.parseConfig` now reads. Covers:
//   - nested reasoning / tool_calling / sampling_defaults parsing
//   - `reasoning_effort_levels` JSON-null → Swift-nil mapping
//   - `model_family` top-level hint + capabilities fallback
//   - graceful absence when bundle is pre-DSV4 (chat block missing).

import Foundation
import MLXLMCommon
import Testing

@Suite("JangLoader — DSV4 chat schema")
struct JangConfigChatSchemaTests {

    @Test("parseConfig reads chat.reasoning / tool_calling / sampling_defaults")
    func parsesFullChatBlock() throws {
        let json: [String: Any] = [
            "format": "jang",
            "format_version": "2.0",
            "model_family": "deepseek_v4",
            "chat": [
                "encoder": "dsv4",
                "has_tokenizer_chat_template": true,
                "bos_token": "<\u{FF5C}begin\u{FF5C}>",
                "bos_token_id": 1,
                "eos_token": "<\u{FF5C}end\u{FF5C}>",
                "eos_token_id": 2,
                "role_tokens": ["user": "<\u{FF5C}user\u{FF5C}>", "assistant": "<\u{FF5C}assistant\u{FF5C}>"],
                "reasoning": [
                    "supported": true,
                    "modes": ["chat", "thinking"],
                    "default_mode": "chat",
                    "thinking_start": "<think>",
                    "thinking_end": "</think>",
                    "reasoning_effort_levels": ["max", "high", NSNull()],
                    "drop_earlier_reasoning": true,
                ],
                "tool_calling": [
                    "supported": true,
                    "parser": "dsml",
                    "dsml_token": "<\u{FF5C}DSML\u{FF5C}>",
                    "tool_calls_block": "<\u{FF5C}DSML\u{FF5C}tool_calls>",
                    "invoke_block": "<\u{FF5C}DSML\u{FF5C}invoke>",
                    "parameter_block": "<\u{FF5C}DSML\u{FF5C}parameter>",
                    "tool_output_tag": "<tool_result>",
                ],
                "sampling_defaults": [
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "max_new_tokens": 300,
                ] as [String: Any],
                "template_kwargs_defaults": [
                    "enable_thinking": true
                ],
            ] as [String: Any],
        ]

        let cfg = try JangLoader.parseConfig(from: json)
        #expect(cfg.modelFamily == "deepseek_v4")

        let chat = try #require(cfg.chat)
        #expect(chat.encoder == "dsv4")
        #expect(chat.hasTokenizerChatTemplate == true)
        #expect(chat.bosTokenId == 1)
        #expect(chat.eosTokenId == 2)
        #expect(chat.roleTokens?["user"] == "<\u{FF5C}user\u{FF5C}>")

        let reasoning = try #require(chat.reasoning)
        #expect(reasoning.supported == true)
        #expect(reasoning.modes == ["chat", "thinking"])
        #expect(reasoning.defaultMode == "chat")
        #expect(reasoning.thinkingStart == "<think>")
        #expect(reasoning.thinkingEnd == "</think>")
        #expect(reasoning.dropEarlierReasoning == true)
        let levels = try #require(reasoning.reasoningEffortLevels)
        #expect(levels.count == 3)
        #expect(levels[0] == "max")
        #expect(levels[1] == "high")
        #expect(levels[2] == nil)  // JSON null → Swift nil

        let tc = try #require(chat.toolCalling)
        #expect(tc.supported == true)
        #expect(tc.parser == "dsml")
        #expect(tc.dsmlToken == "<\u{FF5C}DSML\u{FF5C}>")
        #expect(tc.toolCallsBlock == "<\u{FF5C}DSML\u{FF5C}tool_calls>")

        let sd = try #require(chat.samplingDefaults)
        #expect(sd.temperature == 0.6)
        #expect(sd.topP == 0.95)
        #expect(sd.maxNewTokens == 300)
        #expect(chat.templateKwargsDefaults?.enableThinking == true)
    }

    @Test("Missing chat block leaves JangConfig.chat nil (pre-DSV4 bundle)")
    func absentChatBlock() throws {
        let json: [String: Any] = [
            "format": "jang",
            "format_version": "1.0",
            "capabilities": [
                "reasoning_parser": "think_xml",
                "tool_parser": "xml_function",
                "family": "qwen3",
            ] as [String: Any],
        ]
        let cfg = try JangLoader.parseConfig(from: json)
        #expect(cfg.chat == nil)
        // model_family falls back to capabilities.family when not
        // explicitly stamped at the top level.
        #expect(cfg.modelFamily == "qwen3")
        #expect(cfg.capabilities?.reasoningParser == "think_xml")
    }

    @Test("Partial chat block parses only the fields present")
    func partialChatBlock() throws {
        let json: [String: Any] = [
            "format": "jang",
            "chat": [
                "reasoning": [
                    "supported": true
                ]
            ] as [String: Any],
        ]
        let cfg = try JangLoader.parseConfig(from: json)
        let chat = try #require(cfg.chat)
        #expect(chat.reasoning?.supported == true)
        #expect(chat.reasoning?.modes == nil)
        #expect(chat.toolCalling == nil)
        #expect(chat.samplingDefaults == nil)
    }

    @Test("chat.tool_calling.parser = \"dsml\" resolves to ToolCallFormat.dsml")
    func dsmlParserRoundtrip() throws {
        let json: [String: Any] = [
            "chat": [
                "tool_calling": ["parser": "dsml"]
            ] as [String: Any]
        ]
        let cfg = try JangLoader.parseConfig(from: json)
        let parser = try #require(cfg.chat?.toolCalling?.parser)
        #expect(ToolCallFormat.fromCapabilityName(parser) == .dsml)
    }

    @Test("Explicit model_family overrides capabilities.family")
    func modelFamilyTopLevelWins() throws {
        let json: [String: Any] = [
            "model_family": "deepseek_v4",
            "capabilities": [
                "family": "deepseek"
            ] as [String: Any],
        ]
        let cfg = try JangLoader.parseConfig(from: json)
        #expect(cfg.modelFamily == "deepseek_v4")
        #expect(cfg.capabilities?.family == "deepseek")
    }

    @Test("ZAYA capability stamps are trusted instead of runtime-rewritten")
    func zayaCapabilityStampsAreTrusted() throws {
        let json: [String: Any] = [
            "format": "jang",
            "format_version": "2.0",
            "capabilities": [
                "family": "zaya",
                "reasoning_parser": "qwen3",
                "tool_parser": "zaya_xml",
                "think_in_template": true,
                "supports_tools": true,
                "supports_thinking": false,
                "cache_type": "hybrid",
                "modality": "text",
            ] as [String: Any],
        ]

        let cfg = try JangLoader.parseConfig(from: json)

        #expect(cfg.capabilities?.family == "zaya")
        #expect(cfg.capabilities?.supportsThinking == false)
        #expect(cfg.capabilities?.thinkInTemplate == true)
    }
}
