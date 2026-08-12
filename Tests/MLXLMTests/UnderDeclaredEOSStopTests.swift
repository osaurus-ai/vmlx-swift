// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import MLXLMCommon

/// Pins the stop set for bundles whose `generation_config.json` **under-declares** its eos —
/// it names one end token while the model's own chat template ends turns with a different one.
///
/// This shape is not hypothetical. Scanning the local model library (78 bundles shipping a
/// `generation_config.json` with `eos_token_id`, resolving each `commonEndTokenStrings` entry to
/// its id and checking membership) found three:
///
/// | bundle | declared eos | template emits |
/// |---|---|---|
/// | `OsaurusAI/VibeThinker-3B-MXFP8`   | `151643` (`<\|endoftext\|>`) | `<\|im_end\|>` (`151645`) |
/// | `OsaurusAI/VibeThinker-3B-JANG_4M` | `151643`                     | `<\|im_end\|>`            |
/// | `JANGQ-AI/LFM2.5-230M-MXFP8`       | `124900`                     | `<\|im_end\|>`            |
///
/// Today the hard-coded `commonEndTokenStrings` blanket rescues them. That blanket is under
/// discussion (vmlx-swift#91 proposes skipping it whenever `generation_config.json` declares an
/// eos, which fixes gpt-oss/harmony halting at `<|end|>`), so this test exists to make the cost of
/// that trade **executable** rather than a paragraph in a review: if the blanket stops applying to
/// a declaring-but-incomplete bundle, these turns lose their only turn boundary and run to the
/// token cap.
///
/// `extraEOSTokens` is not a safety net here — it is populated only for registry entries
/// (`LLMModelFactory` / `VLMModelFactory`), and these bundles are loaded from a local directory,
/// where it defaults to empty.
@Suite("Under-declared generation_config eos still stops at the template's turn end")
struct UnderDeclaredEOSStopTests {

    /// Qwen-style ids, matching the VibeThinker bundles above.
    private static let endOfText = 151_643
    private static let imEnd = 151_645

    /// Resolves only the tokens a Qwen2.5-family vocabulary actually carries, so a stop that
    /// survives here survives because it was resolved — not because the tokenizer said yes to
    /// everything.
    private struct QwenishTokenizer: MLXLMCommon.Tokenizer {
        static let map: [String: Int] = [
            "<|endoftext|>": endOfText,
            "<|im_end|>": imEnd,
        ]

        var bosToken: String? { nil }
        var eosToken: String? { "<|endoftext|>" }
        var eosTokenId: Int? { endOfText }
        var unknownToken: String? { nil }
        var unknownTokenId: Int? { nil }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { Self.map[token] }
        func convertIdToToken(_ id: Int) -> String? {
            Self.map.first { $0.value == id }?.key
        }
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    private func configuration(declaringEOS declared: Int?) -> ModelConfiguration {
        var configuration = ModelConfiguration(id: "under-declared-eos-fixture")
        if let declared {
            configuration.generationDefaults = GenerationConfigFile(
                eosTokenIds: IntOrIntArray(declared))
        }
        return configuration
    }

    /// The regression this file exists for: a bundle declaring only `<|endoftext|>` must still stop
    /// at `<|im_end|>`, because that is what its chat template emits at the end of every turn.
    @Test func underDeclaredEOSStillStopsAtImEnd() {
        let resolved = resolveStopSequences(
            modelConfiguration: configuration(declaringEOS: Self.endOfText),
            tokenizer: QwenishTokenizer())

        #expect(
            resolved.tokenIDs.contains(Self.imEnd),
            """
            a bundle whose generation_config.json names only <|endoftext|> lost <|im_end|>, \
            the token its chat template actually ends turns with — this turn now runs to the \
            token cap. Affects VibeThinker-3B (both quants) and LFM2.5-230M in the local library.
            """)
        // The declared one obviously has to survive too.
        #expect(resolved.tokenIDs.contains(Self.endOfText))
    }

    /// A bundle that declares nothing must be unaffected — that path never depended on the gate,
    /// so a change to the gate must not disturb it.
    @Test func undeclaredBundleKeepsBlanketStops() {
        let resolved = resolveStopSequences(
            modelConfiguration: configuration(declaringEOS: nil),
            tokenizer: QwenishTokenizer())

        #expect(resolved.tokenIDs.contains(Self.imEnd))
        #expect(resolved.tokenIDs.contains(Self.endOfText))
    }

    /// The tokenizer's own eos is inserted regardless of the blanket, so this stop is the one
    /// guarantee that holds no matter how the gate is settled.
    @Test func tokenizerEOSIsAlwaysPresent() {
        for declared in [Self.endOfText, Self.imEnd, nil] as [Int?] {
            let resolved = resolveStopSequences(
                modelConfiguration: configuration(declaringEOS: declared),
                tokenizer: QwenishTokenizer())
            #expect(resolved.tokenIDs.contains(Self.endOfText))
        }
    }
}
