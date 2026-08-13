// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Covers `ApertusModel.sanitize`'s absorption of a stray `language_model.lm_head.` prefix.
///
/// Mirrors `LlamaSanitizeLMHeadPrefixTests` deliberately, case for case. `stripLanguageModelPrefix`
/// has its own tests, but those pin the HELPER; what a user hits is the CALL SITE, and nothing else
/// in-tree exercises this wiring — a refactor could drop the call and every other test would still
/// pass. The `only:` scoping is the part that most needs pinning: widening it to strip a nested
/// `language_model.*` BODY would silently drop the real weights instead of failing loudly.
///
/// Observed on `mlx-community/Apertus-8B-Instruct-2509-Jang_6M`: 775 tensors, of which exactly three
/// are `language_model.lm_head.{weight,scales,biases}` with no un-prefixed twin, so the head is
/// unreachable and the bundle fails with `unhandledKeys(modules: ["ApertusModel"], keys:
/// ["language_model"])`.
@Suite("ApertusModel.sanitize absorbs a prefixed lm_head")
struct ApertusSanitizeLMHeadPrefixTests {

    /// Smallest config that constructs; `sanitize` only routes keys, so dims are irrelevant beyond
    /// being valid.
    private func makeModel(tieWordEmbeddings: Bool = false) -> ApertusModel {
        ApertusModel(
            ApertusConfiguration(
                hiddenSize: 8,
                intermediateSize: 16,
                numHiddenLayers: 1,
                numAttentionHeads: 2,
                numKeyValueHeads: 1,
                rmsNormEps: 1e-5,
                vocabSize: 32,
                tieWordEmbeddings: tieWordEmbeddings
            ))
    }

    /// Distinct extent per tensor so a mis-bind shows up as the wrong shape rather than passing.
    private func tagged(_ tag: Int) -> MLXArray { MLXArray.zeros([tag]) }

    @Test("a `language_model.lm_head.*` head lands on the top-level `lm_head.*`")
    func stripsPrefixedHead() {
        let sanitized = makeModel().sanitize(weights: [
            "language_model.lm_head.weight": tagged(7),
            "model.embed_tokens.weight": tagged(3),
        ])
        #expect(sanitized["lm_head.weight"]?.shape == [7])
        #expect(sanitized["language_model.lm_head.weight"] == nil)
        #expect(sanitized["model.embed_tokens.weight"]?.shape == [3])
    }

    /// The quantised bundle that motivated this carries all three keys; dropping the sidecars would
    /// load a head with no scales and produce garbage rather than an error.
    @Test("quant sidecars travel with the head")
    func stripsQuantSidecars() {
        let sanitized = makeModel().sanitize(weights: [
            "language_model.lm_head.weight": tagged(1),
            "language_model.lm_head.scales": tagged(2),
            "language_model.lm_head.biases": tagged(3),
        ])
        #expect(sanitized["lm_head.weight"]?.shape == [1])
        #expect(sanitized["lm_head.scales"]?.shape == [2])
        #expect(sanitized["lm_head.biases"]?.shape == [3])
    }

    /// THE scoping test. A genuine nested body must be left alone: stripping it would rebind real
    /// weights onto the wrong keys and silently drop them, which is far worse than the load failure
    /// this patch fixes.
    @Test("a nested language_model body is left alone")
    func leavesNestedBodyAlone() {
        let sanitized = makeModel().sanitize(weights: [
            "language_model.model.embed_tokens.weight": tagged(5)
        ])
        #expect(sanitized["language_model.model.embed_tokens.weight"]?.shape == [5])
        #expect(sanitized["model.embed_tokens.weight"] == nil)
    }

    /// The pre-existing behaviour must survive the new strip: precomputed rotary frequencies are
    /// still dropped, since they are recomputed at load.
    @Test("precomputed rotary frequencies are still dropped")
    func stillDropsRotaryInvFreq() {
        let sanitized = makeModel().sanitize(weights: [
            "model.layers.0.self_attn.rotary_emb.inv_freq": tagged(9),
            "model.embed_tokens.weight": tagged(3),
        ])
        #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["model.embed_tokens.weight"]?.shape == [3])
    }

    /// The overwhelmingly common case: an ordinary checkpoint must be returned untouched, so the
    /// patch cannot change behaviour for any bundle that loads today.
    @Test("unprefixed weights pass through unchanged")
    func passesThroughUnprefixed() {
        let sanitized = makeModel().sanitize(weights: [
            "lm_head.weight": tagged(4),
            "model.embed_tokens.weight": tagged(3),
        ])
        #expect(sanitized["lm_head.weight"]?.shape == [4])
        #expect(sanitized["model.embed_tokens.weight"]?.shape == [3])
    }
}
