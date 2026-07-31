import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Tests for `Weights.stripLanguageModelPrefix`, the single home for absorbing the
/// `language_model.` converter artifact.
///
/// Tensors are tagged by SHAPE rather than by value: this function routes KEYS and never inspects
/// tensor contents, so a distinct extent per tensor is enough to catch a mis-bind, and no assertion
/// has to force an evaluation.
@Suite("Weights.stripLanguageModelPrefix")
struct WeightsTests {

    /// Distinct shape per tensor, so a mis-bind is visible as the wrong extent.
    private static func tagged(_ tag: Int) -> MLXArray { MLXArray.zeros([tag]) }

    // MARK: - The motivating case

    @Test("a prefixed head lands on the top-level key while the body is untouched")
    func stripsPrefixedHeadOntoTopLevelKey() {
        let weights: [String: MLXArray] = [
            "language_model.lm_head.weight": Self.tagged(7),
            "model.embed_tokens.weight": Self.tagged(3),
            "model.layers.0.self_attn.q_proj.weight": Self.tagged(4),
        ]

        let result = Weights.stripLanguageModelPrefix(weights, only: ["lm_head."])

        #expect(result["lm_head.weight"]?.shape == [7])
        #expect(result["language_model.lm_head.weight"] == nil)
        // the body is not this function's business
        #expect(result["model.embed_tokens.weight"]?.shape == [3])
        #expect(result["model.layers.0.self_attn.q_proj.weight"]?.shape == [4])
        #expect(result.count == weights.count)
    }

    @Test("quant sidecars travel with the head")
    func stripsQuantSidecars() {
        let weights: [String: MLXArray] = [
            "language_model.lm_head.weight": Self.tagged(1),
            "language_model.lm_head.scales": Self.tagged(2),
            "language_model.lm_head.biases": Self.tagged(3),
        ]

        let result = Weights.stripLanguageModelPrefix(weights, only: ["lm_head."])

        #expect(result["lm_head.weight"]?.shape == [1])
        #expect(result["lm_head.scales"]?.shape == [2])
        #expect(result["lm_head.biases"]?.shape == [3])
    }

    // MARK: - The collision guard (the nondeterminism this function exists to prevent)

    @Test("both spellings present: the unprefixed key wins, deterministically")
    func collisionPrefersUnprefixedKey() {
        let weights: [String: MLXArray] = [
            "lm_head.weight": Self.tagged(11),  // the one that must win
            "language_model.lm_head.weight": Self.tagged(22),
            "model.embed_tokens.weight": Self.tagged(3),
        ]

        let result = Weights.stripLanguageModelPrefix(weights, only: ["lm_head."])

        #expect(result["lm_head.weight"]?.shape == [11])
        #expect(result["language_model.lm_head.weight"] == nil)
    }

    /// The stronger case the naive `guard weights[stripped] == nil` misses: TWO prefixed spellings
    /// contend for one destination and NEITHER is present unprefixed, so there is no existing key to
    /// guard against. Last-write-wins would pick by dictionary iteration order here.
    @Test("two prefixed spellings contend: resolved by a fixed rule, not iteration order")
    func collisionBetweenTwoPrefixedSpellings() {
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": Self.tagged(11),
            "model.language_model.embed_tokens.weight": Self.tagged(22),
        ]

        let result = Weights.stripLanguageModelPrefix(weights)

        // Both want `model.embed_tokens.weight`; the rule is lexicographically smallest source.
        #expect(result["model.embed_tokens.weight"]?.shape == [11])
        #expect(result.count == 1)
    }

    /// The real guard against order-dependence, and the reason it is shaped like this.
    ///
    /// An order-dependent bug cannot be caught reliably by ONE colliding key set: for a fixed set of
    /// keys, `Dictionary`'s iteration order is decided by the per-process hash seed, so a
    /// last-write-wins implementation lands on the right answer roughly half the time and the test is
    /// a coin flip. (Varying *insertion* order does not help either — same keys, same seed, same
    /// iteration order.)
    ///
    /// So vary the KEY SETS instead. Each one hashes independently, so a broken implementation has to
    /// win every coin flip to pass. Measured against the `guard weights[stripped] == nil` form, a
    /// single set passed 14 of 25 runs; across 200 sets that survives with probability ~2^-200.
    @Test("collision resolution holds across many independently-hashing key sets")
    func collisionResolutionIsStableAcrossKeySets() {
        for i in 0 ..< 200 {
            let tail = "layers.\(i).mlp.down_proj.weight"
            // Both spellings want `model.<tail>`, and NEITHER exists unprefixed — so there is no
            // pre-existing key to guard against and only a fixed rule can decide this.
            let weights: [String: MLXArray] = [
                "language_model.model." + tail: Self.tagged(11),
                "model.language_model." + tail: Self.tagged(22),
            ]

            let result = Weights.stripLanguageModelPrefix(weights)

            // Documented rule: lexicographically smallest source wins ("language_model." < "model.").
            #expect(
                result["model." + tail]?.shape == [11],
                "key set \(i) bound the wrong spelling — resolution is order-dependent")
        }
    }

    // MARK: - `only:` scoping

    @Test("only: leaves unrelated language_model keys alone")
    func onlyFilterLeavesOtherKeysAlone() {
        let weights: [String: MLXArray] = [
            "language_model.lm_head.weight": Self.tagged(7),
            "language_model.model.embed_tokens.weight": Self.tagged(3),
        ]

        let result = Weights.stripLanguageModelPrefix(weights, only: ["lm_head."])

        #expect(result["lm_head.weight"]?.shape == [7])
        // NOT ours to touch: a genuine nested body key stays exactly as it was.
        #expect(result["language_model.model.embed_tokens.weight"]?.shape == [3])
        #expect(result["model.embed_tokens.weight"] == nil)
    }

    @Test("nil only: strips both spellings wherever they appear")
    func nilOnlyStripsEverywhere() {
        let weights: [String: MLXArray] = [
            "language_model.lm_head.weight": Self.tagged(7),
            "language_model.model.embed_tokens.weight": Self.tagged(3),
            "model.language_model.layers.0.mlp.gate.weight": Self.tagged(4),
            "model.layers.0.self_attn.q_proj.weight": Self.tagged(5),
        ]

        let result = Weights.stripLanguageModelPrefix(weights)

        #expect(result["lm_head.weight"]?.shape == [7])
        #expect(result["model.embed_tokens.weight"]?.shape == [3])
        #expect(result["model.layers.0.mlp.gate.weight"]?.shape == [4])
        // a key with no prefix is passed through untouched
        #expect(result["model.layers.0.self_attn.q_proj.weight"]?.shape == [5])
        #expect(result.keys.allSatisfy { !$0.contains("language_model.") })
    }

    @Test("a checkpoint with no prefix at all is returned unchanged")
    func noPrefixIsIdentity() {
        let weights: [String: MLXArray] = [
            "lm_head.weight": Self.tagged(1),
            "model.embed_tokens.weight": Self.tagged(2),
        ]

        let result = Weights.stripLanguageModelPrefix(weights)

        #expect(result.count == 2)
        #expect(result["lm_head.weight"]?.shape == [1])
        #expect(result["model.embed_tokens.weight"]?.shape == [2])
    }
}
