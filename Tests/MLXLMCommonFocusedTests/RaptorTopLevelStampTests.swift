//
//  RaptorTopLevelStampTests.swift
//  MLXLMCommonFocusedTests
//
//  Raptor-v0.5-8B-A1B (Ling 3 / KDA) ships a `jang_config.json` whose stamp
//  lives in top-level `reasoning` / `tools` / `chat` blocks and has NO
//  `capabilities` block. Before the top-level fallback, `JangLoader.parseConfig`
//  read nothing from it: the parsers fell back to the model_type heuristic,
//  `chat.stop_token_ids` (`<|role_end|>` = 156895) never reached the stop set,
//  and the bundle-authored "reasoning on" default was lost.
//
//  These tests pin:
//    - the synthesized stamps resolve through `fromCapabilityName` to the SAME
//      parsers the `bailing_hybrid` model_type heuristic produced (no
//      behaviour change for the parsers, only a change of source);
//    - 156895 lands in the resolved EOS set alongside config.json's eos id;
//    - `reasoning.default = "on"` becomes the `enable_thinking` default the
//      factories already read from `chat`.
//

import Foundation
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class RaptorTopLevelStampTests: XCTestCase {

    /// Verbatim copy of
    /// `Raptor-v0.5-8B-A1B-JANG_6M/jang_config.json` (2026-09-04).
    private static let raptorJangConfig = #"""
    {
      "chat": {
        "sampling_defaults": {
          "temperature": 0.7,
          "top_p": 0.95,
          "top_k": 20,
          "repetition_penalty": 1.05,
          "source": "bundle generation_config.json (source-validated profile)",
          "mode": "default"
        },
        "stop_token_ids": [
          156895
        ],
        "context_length": {
          "native": 131072
        },
        "role_framing": {
          "style": "bailing_v3",
          "note": "NOT ChatML: turns are <role>SYSTEM|HUMAN|ASSISTANT</role> ... <|role_end|>."
        }
      },
      "reasoning": {
        "supported": true,
        "parser": "bailing_v3",
        "default": "on",
        "think_in_template": true,
        "enable_kwarg": "enable_thinking",
        "think_open_id": 156903,
        "think_close_id": 156904,
        "off_is_prefilled_closed_block": true,
        "off_prefill": "<think></think>",
        "preserve_thinking_supported": true,
        "preserve_thinking_default": true,
        "preserve_thinking_transport": "template_constant",
        "note": "Reasoning is ON by default UPSTREAM: the template sets thinking_option='on' when neither enable_thinking nor thinking_option is passed, and injects 'detailed thinking on' into the SYSTEM turn. Verified empirically. preserved_thinking is HARDCODED true in the template (line 16) — it is not a caller kwarg, so history <think> blocks are always retained, which also improves prefix-cache reuse."
      },
      "tools": {
        "supported": true,
        "parser": "bailing_v3_xml_arg",
        "dialect": "xml_arg",
        "mlx_lm_autodetected": false,
        "tool_open_id": 156896,
        "tool_close_id": 156897,
        "schema_direction": "json_in_xml_out",
        "call_format": "<tool_call>{function-name}\\n<arg_key>{k}</arg_key>\\n<arg_value>{v}</arg_value>\\n...\\n</tool_call>",
        "note": "ASYMMETRIC: tool SCHEMAS are injected as JSON inside <tools>, but tool CALLS come back as XML key/value pairs with a BARE function name on the first line — not a JSON object. No Hermes/Qwen-style JSON parser matches. Values are emitted raw when the argument is a string and tojson otherwise, so a parser must round-trip both. Ship a Ling3 parser or tool calling silently does not work."
      },
      "vision": {
        "supported": false
      },
      "audio": {
        "supported": false
      },
      "runtime": {
        "model_type": "bailing_hybrid",
        "architecture": "BailingMoeV3",
        "hybrid_attention": {
          "full_attention_layers": [
            3,
            7,
            11,
            15,
            19,
            23
          ],
          "linear_attention": "kda",
          "kda_conv_kernel": 4,
          "kda_lower_bound": -5,
          "note": "KDA (Kimi Delta Attention) — NOT the Ling-2.6 Lightning/GLA linear attention. Per-layer decode state is a recurrent [H,128,128] tensor PLUS three short-conv ring buffers; it does not grow with context."
        },
        "bundle_has_mtp": false,
        "jang_profile": "JANG_6M"
      }
    }
    """#

    private static let modelType = "bailing_hybrid"

    private func parseRaptor() throws -> JangConfig {
        let data = try XCTUnwrap(Self.raptorJangConfig.data(using: .utf8))
        let json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        return try JangLoader.parseConfig(from: json)
    }

    /// Drive a parser through the same stream and flatten what it emits so
    /// two parsers can be compared by behaviour (`insideReasoning` is
    /// private, so equality of the start state is observed, not read).
    private func transcript(_ parser: ReasoningParser?, feeding text: String) -> [String] {
        guard var parser else { return ["<nil parser>"] }
        var out: [String] = []
        var segments = parser.feed(text)
        segments += parser.flush()
        for segment in segments {
            switch segment {
            case .reasoning(let s): out.append("reasoning:\(s)")
            case .content(let s): out.append("content:\(s)")
            }
        }
        return out
    }

    // MARK: - Stamps

    func testTopLevelBlocksSynthesizeCapabilities() throws {
        let config = try parseRaptor()
        let caps = try XCTUnwrap(config.capabilities,
            "top-level reasoning/tools must synthesize a capabilities stamp")
        XCTAssertEqual(caps.reasoningParser, "bailing_v3")
        XCTAssertEqual(caps.toolParser, "bailing_v3_xml_arg")
        XCTAssertEqual(caps.thinkInTemplate, true)
        XCTAssertEqual(caps.supportsThinking, true)
        XCTAssertEqual(caps.supportsTools, true)
        XCTAssertEqual(caps.supportsVision, false)
        XCTAssertEqual(caps.supportsAudio, false)
        // `think_in_template = true` keeps the stamp honoured (the ignore
        // path only fires on `think_in_template == false`).
        XCTAssertFalse(ParserResolution.shouldIgnoreReasoningStamp(
            capabilities: caps, modelType: Self.modelType))
    }

    func testStampedReasoningParserMatchesHeuristic() throws {
        let config = try parseRaptor()
        let stamped = ParserResolution.reasoning(
            capabilities: config.capabilities, modelType: Self.modelType)
        let heuristic = ParserResolution.reasoning(
            capabilities: nil, modelType: Self.modelType)

        XCTAssertEqual(stamped.source, .jangStamped)
        XCTAssertEqual(heuristic.source, .modelTypeHeuristic)
        XCTAssertNotNil(stamped.parser)
        XCTAssertNotNil(heuristic.parser)

        // Same tag pair, same start state (both start inside `<think>`).
        XCTAssertEqual(stamped.parser?.startTag, heuristic.parser?.startTag)
        XCTAssertEqual(stamped.parser?.endTag, heuristic.parser?.endTag)
        let stream = "let me think</think>The answer is 42. <think>again</think> done"
        let stampedOut = transcript(stamped.parser, feeding: stream)
        let heuristicOut = transcript(heuristic.parser, feeding: stream)
        XCTAssertEqual(stampedOut, heuristicOut)
        // Sanity: the stream really exercised the start-in-reasoning state
        // (the first emitted segment is reasoning, not content).
        XCTAssertEqual(stampedOut.first?.hasPrefix("reasoning:"), true)
        XCTAssertTrue(stampedOut.contains { $0.hasPrefix("content:") })
    }

    func testStampedToolParserMatchesHeuristic() throws {
        let config = try parseRaptor()
        let stamped = ParserResolution.toolCall(
            capabilities: config.capabilities, modelType: Self.modelType)
        let heuristic = ParserResolution.toolCall(
            capabilities: nil, modelType: Self.modelType)
        XCTAssertEqual(stamped.source, .jangStamped)
        XCTAssertEqual(heuristic.source, .modelTypeHeuristic)
        XCTAssertEqual(stamped.format, .glm4)
        XCTAssertEqual(stamped.format, heuristic.format)
        // The factories consult `chat.tool_calling.parser` first; it must
        // resolve to the same format too.
        XCTAssertEqual(config.chat?.toolCalling?.parser, "bailing_v3_xml_arg")
        XCTAssertEqual(
            ToolCallFormat.fromCapabilityName(config.chat?.toolCalling?.parser), .glm4)
    }

    // MARK: - Stop tokens

    func testChatStopTokenIdsReachTheResolvedEOSSet() throws {
        let config = try parseRaptor()
        XCTAssertEqual(config.chat?.stopTokenIds, [156895])

        let configJSON = #"{"model_type": "bailing_hybrid", "eos_token_id": 156892}"#
        let configData = try XCTUnwrap(configJSON.data(using: .utf8))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: configData)

        let resolved = ModelTokenConfigurationResolver.resolvedEOSTokenIds(
            baseConfig: base,
            configurationData: configData,
            generationConfig: nil,
            jangStopTokenIds: config.chat?.stopTokenIds)
        XCTAssertTrue(resolved.contains(156895), "chat.stop_token_ids must be in the stop set")
        XCTAssertTrue(resolved.contains(156892), "config.json eos must be kept, not replaced")

        // Without the JANG ids the set is exactly the config declaration.
        let withoutJang = ModelTokenConfigurationResolver.resolvedEOSTokenIds(
            baseConfig: base,
            configurationData: configData,
            generationConfig: nil,
            jangStopTokenIds: nil)
        XCTAssertEqual(withoutJang, [156892])
    }

    // MARK: - Reasoning default

    func testReasoningDefaultOnBecomesEnableThinkingDefault() throws {
        let config = try parseRaptor()
        XCTAssertEqual(config.chat?.templateKwargsDefaults?.enableThinking, true)
        XCTAssertEqual(config.chat?.reasoning?.defaultMode, "thinking")
        XCTAssertEqual(config.chat?.reasoning?.supported, true)

        let context = llmDefaultAdditionalContext(
            modelType: Self.modelType,
            capabilities: config.capabilities,
            generationConfig: nil,
            chatConfig: config.chat)
        XCTAssertEqual(context?["enable_thinking"] as? Bool, true)
    }

    func testReasoningDefaultOffMapsToChatMode() throws {
        var json = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: try XCTUnwrap(Self.raptorJangConfig.data(using: .utf8)))
                as? [String: Any])
        var reasoning = try XCTUnwrap(json["reasoning"] as? [String: Any])
        reasoning["default"] = "off"
        json["reasoning"] = reasoning
        let config = try JangLoader.parseConfig(from: json)
        XCTAssertEqual(config.chat?.templateKwargsDefaults?.enableThinking, false)
        XCTAssertEqual(config.chat?.reasoning?.defaultMode, "chat")
    }

    // MARK: - Precedence / no-op guarantees

    func testExplicitCapabilitiesBlockWinsOverTopLevelBlocks() throws {
        var json = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: try XCTUnwrap(Self.raptorJangConfig.data(using: .utf8)))
                as? [String: Any])
        json["capabilities"] = [
            "reasoning_parser": "qwen3",
            "tool_parser": "xml_function",
        ] as [String: Any]
        let config = try JangLoader.parseConfig(from: json)
        XCTAssertEqual(config.capabilities?.reasoningParser, "qwen3")
        XCTAssertEqual(config.capabilities?.toolParser, "xml_function")
        // The chat sub-blocks still derive from the top-level blocks where
        // `chat` itself says nothing.
        XCTAssertEqual(config.chat?.stopTokenIds, [156895])
        XCTAssertEqual(config.chat?.templateKwargsDefaults?.enableThinking, true)
    }

    func testExplicitChatSubBlocksWinOverTopLevelBlocks() throws {
        var json = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: try XCTUnwrap(Self.raptorJangConfig.data(using: .utf8)))
                as? [String: Any])
        var chat = try XCTUnwrap(json["chat"] as? [String: Any])
        chat["reasoning"] = ["default_mode": "chat"] as [String: Any]
        chat["tool_calling"] = ["parser": "json"] as [String: Any]
        chat["template_kwargs_defaults"] = ["enable_thinking": false] as [String: Any]
        json["chat"] = chat
        let config = try JangLoader.parseConfig(from: json)
        XCTAssertEqual(config.chat?.reasoning?.defaultMode, "chat")
        XCTAssertEqual(config.chat?.toolCalling?.parser, "json")
        XCTAssertEqual(config.chat?.templateKwargsDefaults?.enableThinking, false)
    }

    func testPreStampBundleStaysUnstamped() throws {
        let json: [String: Any] = [
            "format": "jang",
            "format_version": "2.0",
        ]
        let config = try JangLoader.parseConfig(from: json)
        XCTAssertNil(config.capabilities)
        XCTAssertNil(config.chat)
        let heuristic = ParserResolution.reasoning(
            capabilities: config.capabilities, modelType: Self.modelType)
        XCTAssertEqual(heuristic.source, .modelTypeHeuristic)
    }
}
