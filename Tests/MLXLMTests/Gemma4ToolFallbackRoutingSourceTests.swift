// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

/// A valid bundle template owns the prompt even on tool turns. Failure-only
/// fallback selection must still require exact Gemma markers: ZAYA also uses
/// `<bos>` but has a different ChatML/template contract.
@Suite("Gemma-4 native-first tool template routing")
struct Gemma4ToolFallbackRoutingSourceTests {

    private func macroSource() throws -> String {
        try String(
            contentsOfFile: "Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift",
            encoding: .utf8)
    }

    @Test("tool availability never selects a fallback before native rendering")
    func validNativeTemplateHasPriority() throws {
        let source = try macroSource()
        #expect(!source.contains("gemmaToolSchemasPresent"))
        #expect(!source.contains("chat-template tools -> Gemma4WithTools fallback engaged"))
        #expect(
            source.components(separatedBy: "let isGemma = hasGemma4NativeToolSentinels").count == 3,
            "Both generation-prompt variants must use exact markers only inside failure handling")
        #expect(
            source.components(separatedBy: "if hasGemma4NativeToolSentinels {").count == 3,
            "Both missing-template handlers must retain family-specific structural fallbacks")
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
