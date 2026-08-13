// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

/// Pins the Gemma-4 tool-template routing condition.
///
/// The fallback that swaps a bundle onto `gemma4WithTools` must key on Gemma-4's OWN tool
/// sentinels, never on `bosToken == "<bos>"`. `<bos>` is not a Gemma marker:
///
/// | bundle | bos | `<\|tool_call>` | `<\|turn>` |
/// |---|---|---|---|
/// | `gemma-4-12B-it-MXFP8` | `<bos>` | yes | yes |
/// | `gemma-4-26B-A4B-it-qat-JANG_4M` | `<bos>` | yes | yes |
/// | `ZAYA1-VL-8B-JANGTQ_K` | `<bos>` | **no** | **no** |
///
/// ZAYA1-VL is ChatML (`<bos><|im_start|>system …`). Routing on `<bos>` alone swapped it onto the
/// Gemma turn-marker template the moment tools were offered, so its prompt arrived as
/// `<|turn>model … <turn|>` instead of `<|im_start|>assistant`. The model echoed that alien format
/// back as literal text (`<|turn>user`) and emitted no tool call — on both shipped quants, and only
/// ever on tool turns. Verified live through `VMLX_CHAT_TEMPLATE_FALLBACK_LOG=1`:
/// `chat-template tools -> Gemma4WithTools fallback engaged` on Zaya before the fix, absent after,
/// while gemma-4-12B still engages it and passes the full variating pattern including
/// image+tools in one turn.
@Suite("Gemma-4 tool fallback routes on sentinels, not <bos>")
struct Gemma4ToolFallbackRoutingSourceTests {

    private func macroSource() throws -> String {
        try String(
            contentsOfFile: "Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift",
            encoding: .utf8)
    }

    @Test("routing requires the Gemma-4 sentinels")
    func routingRequiresSentinels() throws {
        let source = try macroSource()
        #expect(
            source.contains(
                """
                !(chatTemplateTools?.isEmpty ?? true)
                                            && hasGemma4NativeToolSentinels
                """),
            "the Gemma-4 tool fallback must gate on hasGemma4NativeToolSentinels")
    }

    @Test("<bos> alone can never select the Gemma-4 tool template")
    func bosAloneCannotSelectGemmaTools() throws {
        let source = try macroSource()
        #expect(
            !source.contains(#"(upstream.bosToken == "<bos>" || hasGemma4NativeToolSentinels)"#),
            """
            `<bos>` was re-added as a sufficient condition. It is shared by ChatML bundles such as \
            ZAYA1-VL-8B, which then receive Gemma's <|turn>/<turn|> markers on every tool turn and \
            stop emitting tool calls entirely.
            """)
    }

    /// The sentinel check itself must stay a round-trip, not a bare id lookup — `convertTokenToId`
    /// returns the unk id for unknown tokens, so `!= nil` would be true for every vocabulary and
    /// re-break exactly what this routing fix repaired.
    @Test("the sentinel check round-trips both markers")
    func sentinelCheckRoundTrips() throws {
        let source = try macroSource()
        #expect(
            source.contains(
                #"(upstream.convertTokenToId("<|tool_call>").flatMap { upstream.convertIdToToken($0) } == "<|tool_call>")"#
            ))
        #expect(
            source.contains(
                #"(upstream.convertTokenToId("<|turn>").flatMap { upstream.convertIdToToken($0) } == "<|turn>")"#
            ))
    }
}
