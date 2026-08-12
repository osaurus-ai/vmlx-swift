// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Covers `LlamaModel.sanitize`'s absorption of a stray `language_model.lm_head.` prefix.
///
/// `Weights.stripLanguageModelPrefix` already has its own tests, but those pin the *helper*.
/// The behaviour a user actually hits is the **call site** — a checkpoint whose head is prefixed
/// while its body is not, which otherwise fails to load with `Unhandled keys [language_model]`.
/// Nothing exercised that wiring, so a refactor could drop the call and every existing test would
/// still pass.
@Suite("LlamaModel.sanitize absorbs a prefixed lm_head")
struct LlamaSanitizeLMHeadPrefixTests {

    /// Smallest config that constructs; `sanitize` only routes keys, so the dims are irrelevant
    /// beyond being valid.
    private func makeModel(tieWordEmbeddings: Bool = false) -> LlamaModel {
        LlamaModel(
            LlamaConfiguration(
                hiddenSize: 8,
                hiddenLayers: 1,
                intermediateSize: 16,
                attentionHeads: 2,
                rmsNormEps: 1e-5,
                vocabularySize: 32,
                kvHeads: 1,
                tieWordEmbeddings: tieWordEmbeddings
            ))
    }

    /// Distinct extent per tensor so a mis-bind shows up as the wrong shape.
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

    /// Quantized bundles carry the sidecars beside the head; they have to travel with it or the
    /// head binds without its scales.
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

    /// The strip is scoped to the head. A genuinely nested `language_model.*` body is another
    /// model's layout and must pass through untouched rather than being flattened into this one.
    @Test("a nested language_model body is left alone")
    func leavesNestedBodyAlone() {
        let sanitized = makeModel().sanitize(weights: [
            "language_model.model.embed_tokens.weight": tagged(5),
            "language_model.model.layers.0.self_attn.q_proj.weight": tagged(6),
        ])

        #expect(sanitized["language_model.model.embed_tokens.weight"]?.shape == [5])
        #expect(sanitized["model.embed_tokens.weight"] == nil)
    }

    /// The pre-existing filter must survive the rewrite: rotary inverse frequencies are
    /// precomputed buffers, not parameters, and binding them fails the load.
    @Test("precomputed rotary frequencies are still dropped")
    func stillDropsRotaryInvFreq() {
        let sanitized = makeModel().sanitize(weights: [
            "model.layers.0.self_attn.rotary_emb.inv_freq": tagged(9),
            "model.embed_tokens.weight": tagged(3),
        ])

        #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["model.embed_tokens.weight"]?.shape == [3])
    }

    /// An ordinary unprefixed checkpoint must be untouched — the common case.
    @Test("unprefixed weights pass through unchanged")
    func passesThroughUnprefixed() {
        let weights: [String: MLXArray] = [
            "lm_head.weight": tagged(4),
            "model.embed_tokens.weight": tagged(3),
        ]
        let sanitized = makeModel().sanitize(weights: weights)

        #expect(sanitized.count == weights.count)
        #expect(sanitized["lm_head.weight"]?.shape == [4])
        #expect(sanitized["model.embed_tokens.weight"]?.shape == [3])
    }
}
