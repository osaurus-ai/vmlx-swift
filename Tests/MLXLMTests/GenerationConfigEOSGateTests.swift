import Foundation
import Testing

@testable import MLXLMCommon

/// Pins the `generation_config.json` eos-gate in `resolveStopSequences`: the hard-coded
/// `commonEndTokenStrings` blanket is applied ONLY when the model did not authoritatively declare its
/// stop set. A model that ships `generation_config.json` with an `eos_token_id` gets its declaration
/// trusted verbatim — the blanket is skipped, so it can't inject stops the model deliberately omitted.
///
/// Own file (not appended to `StopStringMatcherTests`) on purpose: this and the gemma `<end_of_turn>`
/// tests are two independent upstream PRs, so each patch must apply to a pristine tree by itself. See
/// the note in `Gemma4StaleEOSTests`.
@Suite("generation_config.json eos-gate")
struct GenerationConfigEOSGateTests {

    /// Resolves a fixed token→id map; everything else is empty. `eosToken` (optional) lets a test give
    /// the tokenizer its own eos so `resolveStopSequences` inserts it (the model's real stop).
    private struct MapTokenizer: Tokenizer {
        var map: [String: Int]
        var eosToken: String?
        var bosToken: String? { nil }
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

    private static let endToken = 200007      // harmony `<|end|>` — a channel separator, NOT a stop
    private static let imEndToken = 151645    // Qwen `<|im_end|>` — a genuine turn-end stop
    private static let tokenizer = MapTokenizer(map: ["<|end|>": endToken, "<|im_end|>": imEndToken])

    private func config(name: String, declaredEOS: [Int]?) -> ModelConfiguration {
        ModelConfiguration(
            id: name,
            generationDefaults: declaredEOS.map { GenerationConfigFile(eosTokenIds: IntOrIntArray($0)) })
    }

    /// The motivating case (gpt-oss / OpenAI harmony). `generation_config.json` declares eos WITHOUT
    /// `<|end|>` (200007); the blanket would otherwise inject it, halting at the channel separator.
    /// With the gate, `<|end|>` must NOT appear in the resolved stops.
    @Test("harmony: declared eos excluding <|end|> → blanket skipped, 200007 not a stop")
    func harmonyDeclaredEOSSkipsBlanket() {
        let cfg = config(name: "gpt-oss-harmony", declaredEOS: [199_999, 200_002, 200_012])
        let resolved = resolveStopSequences(modelConfiguration: cfg, tokenizer: Self.tokenizer)
        #expect(!resolved.tokenIDs.contains(Self.endToken))
    }

    /// The before/after: an UNDER-declared model (no `generation_config.json` eos) still gets the
    /// blanket, so `<|end|>` IS resolved as a stop. This is exactly the state the gate flips.
    @Test("no declaration → blanket runs, <|end|> IS a stop")
    func underDeclaredKeepsBlanket() {
        let cfg = config(name: "under-declared", declaredEOS: nil)
        let resolved = resolveStopSequences(modelConfiguration: cfg, tokenizer: Self.tokenizer)
        #expect(resolved.tokenIDs.contains(Self.endToken))
    }

    /// Regression on the other side of the gate: a model whose declared eos IS `<|im_end|>` must still
    /// stop on it. Skipping the blanket must not drop the model's own declared stop.
    @Test("Qwen: declared <|im_end|> eos still stops")
    func declaredQwenEOSStillStops() {
        let cfg = config(name: "qwen-imend", declaredEOS: [Self.imEndToken])
        let tok = MapTokenizer(map: ["<|im_end|>": Self.imEndToken], eosToken: "<|im_end|>")
        let resolved = resolveStopSequences(modelConfiguration: cfg, tokenizer: tok)
        #expect(resolved.tokenIDs.contains(Self.imEndToken))
    }
}
