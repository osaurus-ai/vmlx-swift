// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Regression for the osaurus-reported LFM2 bug (2026-04-24): both
// LLMModelFactory and VLMModelFactory historically defaulted any
// model_type outside {gemma4, gemma, mistral} to `"think_xml"`. That
// stamp resolves to a `ReasoningParser(startInReasoning: true)` —
// matching Qwen's `<think>`-prefilled prompt tail — so every decoded
// chunk from LFM2 (which never emits `<think>` markers) came out as
// `Generation.reasoning(_)` and osaurus rendered the entire answer
// into the thinking block.
//
// Fixed by `reasoningStampFromModelType(_:)` — explicit allowlist of
// model_types that ACTUALLY emit a `<think>` envelope. Every other
// family falls through to `"none"`. This test suite locks the
// allowlist in place so a future edit can't silently flip the
// default back to think_xml.

import Foundation
import XCTest

@testable import MLXLMCommon

final class ReasoningStampFromModelTypeTests: XCTestCase {

    // MARK: - Models that MUST emit .chunk only (stamp = "none")

    func testLFM2DoesNotGetThinkXml() {
        // The exact bug reported by osaurus — LFM2 was incorrectly
        // getting `think_xml` and leaking every chunk as reasoning.
        XCTAssertEqual(reasoningStampFromModelType("lfm2"), "none")
        XCTAssertEqual(reasoningStampFromModelType("lfm2_moe"), "none")
    }

    func testLFM25TemplateDeclaresThinkingWithoutPromotingPlainLFM2MoE() {
        let template = """
        {%- macro render_tool_calls(tool_calls) -%}
        {{- "<|tool_call_start|>[" + "line_count(text='a')" + "]<|tool_call_end|>" -}}
        {%- endmacro -%}
        {%- if message.thinking is defined -%}
        {{- "<think>" + message.thinking + "</think>" -}}
        {%- endif -%}
        """

        let plain = ParserResolution.reasoning(
            capabilities: nil,
            modelType: "lfm2_moe",
            chatTemplate: nil)
        XCTAssertNil(plain.parser)
        XCTAssertEqual(plain.source, .modelTypeHeuristic)

        let lfm25 = ParserResolution.reasoning(
            capabilities: nil,
            modelType: "lfm2_moe",
            chatTemplate: template)
        XCTAssertNotNil(lfm25.parser)
        XCTAssertEqual(lfm25.source, .chatTemplate)
    }

    func testLFM25JangStampWithoutThinkingTemplateDoesNotForceQwenReasoning() {
        let capabilities = JangCapabilities(
            reasoningParser: "qwen3",
            toolParser: "lfm2",
            thinkInTemplate: false,
            supportsTools: true,
            supportsThinking: true,
            family: "lfm2_moe",
            cacheType: "hybrid")

        XCTAssertTrue(ParserResolution.shouldIgnoreReasoningStamp(
            capabilities: capabilities,
            modelType: "lfm2_moe"))

        let resolved = ParserResolution.reasoning(
            capabilities: capabilities,
            modelType: "lfm2_moe",
            chatTemplate: """
            {%- if message.thinking is defined -%}
            {{- "<think>" + message.thinking + "</think>" -}}
            {%- endif -%}
            {{- "<|tool_call_start|>[line_count(text='a')]<|tool_call_end|>" -}}
            """)
        XCTAssertNil(resolved.parser)
        XCTAssertEqual(resolved.source, .modelTypeHeuristic)
    }

    func testPlainFamiliesGetNone() {
        // Every non-reasoning family vmlx supports must emit plain
        // .chunk, not .reasoning. Enumerate explicitly so a
        // model-type rename in the factory can't hide a regression.
        let plainModels = [
            "llama", "mistral", "mistral3", "mistral4",
            "phi", "phi3", "phimoe",
            "gemma", "gemma2", "gemma3", "gemma3_text", "gemma3n",
            "starcoder2", "cohere", "openelm", "internlm2", "granite",
            "granitemoehybrid",
            "minicpm", "nanochat", "olmoe", "olmo2", "olmo3",
            "smollm3", "ernie4_5", "baichuan_m1",
            "exaone4", "lille-130m", "apertus", "jamba_3b",
            "falcon_h1", "bitnet", "afmoe",
        ]
        for modelType in plainModels {
            XCTAssertEqual(
                reasoningStampFromModelType(modelType), "none",
                "model_type `\(modelType)` must NOT get a think-parser stamp")
        }
    }

    func testEmptyAndNilModelTypeReturnNone() {
        XCTAssertEqual(reasoningStampFromModelType(nil), "none")
        XCTAssertEqual(reasoningStampFromModelType(""), "none")
    }

    // MARK: - Models that MUST emit .reasoning for <think> blocks

    func testQwen3FamilyGetsThinkXml() {
        for modelType in [
            "qwen3", "qwen3_5", "qwen3_6", "qwen3_moe", "qwen3_next",
            "qwen3_5_moe", "qwen3_5_text", "qwen3_next_moe",
        ] {
            XCTAssertEqual(
                reasoningStampFromModelType(modelType), "think_xml",
                "Qwen 3.x `\(modelType)` must get the think_xml stamp")
        }
    }

    func testDeepseekFamilyGetsThinkXml() {
        // DSV3 / DSV4 / R1 all use `<think>` envelope.
        for modelType in ["deepseek_v3", "deepseek_v4", "deepseek_r1"] {
            XCTAssertEqual(reasoningStampFromModelType(modelType), "think_xml")
        }
    }

    func testBailingFamilyGetsThinkXml() {
        // Ling/Bailing templates use the DeepSeek-R1-style `<think>` envelope.
        for modelType in ["bailing_moe", "bailing_hybrid", "bailing_moe_v2_5"] {
            XCTAssertEqual(reasoningStampFromModelType(modelType), "think_xml")
        }
    }

    func testKimiFamilyGetsThinkXml() {
        // Kimi K2 / K2.5 / K2.6 are always-thinking models — chat
        // template unconditionally prefills `<think>`.
        for modelType in ["kimi_k2", "kimi_k25", "kimi_k26"] {
            XCTAssertEqual(reasoningStampFromModelType(modelType), "think_xml")
        }
    }

    func testGLMMoEGetsThinkXml() {
        for modelType in ["glm4_moe", "glm4_moe_lite", "glm5", "glm5_moe"] {
            XCTAssertEqual(reasoningStampFromModelType(modelType), "think_xml")
        }
    }

    func testMiniMaxGetsThinkXml() {
        for modelType in ["minimax", "minimax_m2", "minimax_m3"] {
            XCTAssertEqual(reasoningStampFromModelType(modelType), "think_xml")
        }
    }

    func testZayaGetsThinkXml() {
        for modelType in ["zaya", "zaya1", "zaya2"] {
            XCTAssertEqual(reasoningStampFromModelType(modelType), "think_xml")
        }
    }

    func testMiMoFamilyGetsThinkXml() {
        for modelType in ["mimo", "mimo_v2_flash"] {
            XCTAssertEqual(reasoningStampFromModelType(modelType), "think_xml")
        }
    }

    func testNemotronHGetsThinkXml() {
        XCTAssertEqual(reasoningStampFromModelType("nemotron_h"), "think_xml")
    }

    func testHoloGetsThinkXml() {
        XCTAssertEqual(reasoningStampFromModelType("holo3"), "think_xml")
    }

    // MARK: - Gemma-4 harmony channel

    func testGemma4GetsHarmony() {
        // Gemma-4 uses the `<|channel>thought\n…<channel|>` envelope,
        // NOT `<think>`. Must resolve to the harmony stamp.
        for modelType in ["gemma4", "gemma4_text"] {
            XCTAssertEqual(reasoningStampFromModelType(modelType), "harmony")
        }
    }

    func testGptOssGetsHarmony() {
        // GPT-OSS emits the native Harmony envelope (`<|start|>`,
        // `<|channel|>`, `<|end|>`, etc.). Without the harmony stamp
        // the markers leak into the user-visible chunk stream.
        for modelType in ["gpt_oss", "gpt_oss_120b", "gpt_oss_20b"] {
            XCTAssertEqual(reasoningStampFromModelType(modelType), "harmony")
        }
    }

    // MARK: - Parser round-trip — stamp must produce expected parser

    /// Critical invariant: whatever stamp we return, it MUST resolve
    /// via `ReasoningParser.fromCapabilityName`. If someone adds a
    /// stamp here without updating the parser resolver, every model
    /// in that family silently falls through to no parsing — which
    /// is safer than the old bug but still a regression.
    func testEveryEmittedStampResolves() {
        let stamps = Set([
            reasoningStampFromModelType("lfm2"),
            reasoningStampFromModelType("qwen3_6"),
            reasoningStampFromModelType("deepseek_v4"),
            reasoningStampFromModelType("kimi_k25"),
            reasoningStampFromModelType("gemma4"),
            reasoningStampFromModelType("mistral"),
            reasoningStampFromModelType("holo3"),
            reasoningStampFromModelType("zaya"),
        ])
        for stamp in stamps {
            // `"none"` intentionally returns nil from the parser
            // factory. Everything else must produce a non-nil
            // parser.
            let parser = ReasoningParser.fromCapabilityName(stamp)
            if stamp == "none" {
                XCTAssertNil(parser,
                    "stamp `none` must resolve to nil parser")
            } else {
                XCTAssertNotNil(parser,
                    "stamp `\(stamp)` must resolve to a non-nil parser")
            }
        }
    }

    /// Invariant: no model in the plain-families list should get a
    /// parser that initializes with `startInReasoning: true`. The old
    /// bug was exactly this — LFM2's stamp resolved to a
    /// `startInReasoning: true` parser and everything leaked into
    /// reasoning.
    func testLFM2ParserDoesNotStartInReasoning() {
        let stamp = reasoningStampFromModelType("lfm2")
        let parser = ReasoningParser.fromCapabilityName(stamp)
        // `"none"` → nil parser → no reasoning parsing happens at all.
        // That's the contract — pipeline passes chunks through
        // untouched.
        XCTAssertNil(parser,
            "LFM2 must get a nil reasoning parser — any parser that starts in reasoning mode would leak chunks")
    }
}
