// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Testing

@Suite("No hidden reasoning close bias")
struct NoHiddenReasoningCloseBiasFocusedTests {
    @Test("decode does not bias or force reasoning close tokens")
    func decodeDoesNotBiasOrForceReasoningCloseTokens() throws {
        let evaluate = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)
        let engine = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)

        #expect(!evaluate.contains("ReasoningCloseBiasConfig"))
        #expect(!evaluate.contains("ReasoningCloseBiasProcessor"))
        #expect(!evaluate.contains("reasoningCloseBias"))
        #expect(!evaluate.contains("forceAfterTokens"))
        #expect(!evaluate.contains("parametersWithAutomaticReasoningCloseBias"))
        #expect(!evaluate.contains("_parametersWithAutomaticReasoningCloseBias"))
        #expect(!evaluate.contains("_specialTokenID(\"</think>\", tokenizer: tokenizer)"))
        #expect(!evaluate.contains("reasoningCloseBias active"))
        #expect(!engine.contains("parametersWithAutomaticReasoningCloseBias"))
        #expect(!engine.contains("_parametersWithAutomaticReasoningCloseBias"))
    }
}
