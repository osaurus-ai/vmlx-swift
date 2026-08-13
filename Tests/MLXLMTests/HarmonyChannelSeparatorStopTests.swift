// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import MLXLMCommon

/// Pins the one narrow exception to the `commonEndTokenStrings` blanket: `<|end|>` is a turn end
/// for Phi-3/Phi-4 but a CHANNEL SEPARATOR in the OpenAI harmony format (gpt-oss), where the model
/// closes its `analysis` channel with `<|end|>` and then opens `final`. Stopping there truncates
/// the turn at the end of reasoning and the visible answer is empty.
///
/// Diagnosis from vmlx-swift#91 (rcfa). That PR fixed it by skipping the blanket for ANY bundle
/// declaring an eos in `generation_config.json`, which is too broad — see
/// `UnderDeclaredEOSStopTests`, which fails against it because VibeThinker-3B declares only
/// `<|endoftext|>` while its template ends turns with `<|im_end|>`. The suppression here instead
/// requires two independent pieces of positive evidence, so exactly one family is affected.
@Suite("harmony <|end|> is a channel separator, not a stop")
struct HarmonyChannelSeparatorStopTests {

    private static let endToken = 200_007  // <|end|>  — channel separator
    private static let returnToken = 200_002  // <|return|> — real stop
    private static let callToken = 200_012  // <|call|>   — real stop
    private static let endOfText = 199_999
    private static let imEnd = 151_645

    /// Resolves exactly the tokens named in `map`, so a stop that survives survives because it
    /// resolved — not because the tokenizer said yes to everything.
    private struct MapTokenizer: MLXLMCommon.Tokenizer {
        let map: [String: Int]
        let eos: String?

        var bosToken: String? { nil }
        var eosToken: String? { eos }
        var unknownToken: String? { nil }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { map[token] }
        func convertIdToToken(_ id: Int) -> String? { map.first { $0.value == id }?.key }
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    /// A harmony vocabulary: carries the three channel markers plus `<|end|>`.
    private static func harmonyTokenizer() -> MapTokenizer {
        MapTokenizer(
            map: [
                "<|channel|>": 200_005, "<|message|>": 200_008, "<|return|>": returnToken,
                "<|end|>": endToken, "<|call|>": callToken, "<|endoftext|>": endOfText,
            ],
            eos: "<|return|>")
    }

    /// A Phi-shaped vocabulary: `<|end|>` present, but NO harmony channel markers.
    private static func phiTokenizer() -> MapTokenizer {
        MapTokenizer(map: ["<|end|>": endToken, "<|endoftext|>": endOfText], eos: "<|endoftext|>")
    }

    /// Mirrors the real load path: `ModelTokenConfigurationResolver.resolvedEOSTokenIds` copies a
    /// declared `generation_config.json` eos into `eosTokenIds`, so a fixture that sets only
    /// `generationDefaults` would understate the stop set and test a configuration that never ships.
    private func configuration(declaring declared: [Int]) -> ModelConfiguration {
        var configuration = ModelConfiguration(id: "stop-set-fixture")
        if !declared.isEmpty {
            configuration.generationDefaults = GenerationConfigFile(
                eosTokenIds: IntOrIntArray(declared))
            configuration.eosTokenIds = Set(declared)
        }
        return configuration
    }

    /// The reported bug. Harmony vocab + a declaration that omits `<|end|>` → it must not be a stop.
    @Test func harmonyEndIsNotAStop() {
        let resolved = resolveStopSequences(
            modelConfiguration: configuration(declaring: [Self.returnToken, Self.endOfText, Self.callToken]),
            tokenizer: Self.harmonyTokenizer())

        #expect(
            !resolved.tokenIDs.contains(Self.endToken),
            """
            <|end|> became a stop for a harmony bundle: generation halts at the end of the \
            analysis channel and the visible answer is empty.
            """)
        #expect(!resolved.textStopStrings.contains("<|end|>"))
        // The model's own declared stops must still be honoured.
        #expect(resolved.tokenIDs.contains(Self.returnToken))
        #expect(resolved.tokenIDs.contains(Self.callToken))
    }

    /// The non-regression #91's blanket gate did not preserve: a Phi-shaped bundle that declares
    /// an eos must KEEP `<|end|>`, because there it really is the turn end.
    @Test func phiKeepsEndAsAStopEvenWhenItDeclaresEOS() {
        let resolved = resolveStopSequences(
            modelConfiguration: configuration(declaring: [Self.endOfText]),
            tokenizer: Self.phiTokenizer())

        #expect(
            resolved.tokenIDs.contains(Self.endToken),
            "a Phi-style bundle lost <|end|>, the token its template ends turns with")
    }

    /// Suppression needs the model's own declaration. A harmony-shaped bundle shipping no
    /// `generation_config.json` eos falls back to today's blanket rather than guessing.
    @Test func harmonyWithoutDeclarationKeepsBlanket() {
        let resolved = resolveStopSequences(
            modelConfiguration: configuration(declaring: []),
            tokenizer: Self.harmonyTokenizer())

        #expect(resolved.tokenIDs.contains(Self.endToken))
    }

    /// And if a harmony bundle ever DOES declare `<|end|>` as eos, that declaration wins.
    @Test func harmonyDeclaringEndKeepsIt() {
        let resolved = resolveStopSequences(
            modelConfiguration: configuration(declaring: [Self.returnToken, Self.endToken]),
            tokenizer: Self.harmonyTokenizer())

        #expect(resolved.tokenIDs.contains(Self.endToken))
    }

    /// The blast radius, stated as a test: a Qwen-shaped bundle (no harmony markers) that
    /// under-declares its eos still gets `<|im_end|>` from the blanket. This is the VibeThinker
    /// shape that the broader gate broke.
    @Test func qwenUnderDeclarationUnaffected() {
        let tokenizer = MapTokenizer(
            map: ["<|endoftext|>": 151_643, "<|im_end|>": Self.imEnd], eos: "<|endoftext|>")
        let resolved = resolveStopSequences(
            modelConfiguration: configuration(declaring: [151_643]),
            tokenizer: tokenizer)

        #expect(resolved.tokenIDs.contains(Self.imEnd))
    }
}
