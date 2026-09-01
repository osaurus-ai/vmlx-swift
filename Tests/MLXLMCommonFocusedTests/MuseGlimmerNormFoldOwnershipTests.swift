// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The centered-norm `+1` fold has one owner. These tests exist because the failure mode of it
// having two is SILENT: a tower loaded with unfolded gains against a forward that no longer adds
// the `+1` sits at zero gain, still loads, and still emits text.

import Foundation
import MLX
import MLXLLM
import MLXVLM
import Testing

// Weights are built `as [Float]` deliberately: a bare Swift float literal
// array makes a float64 MLXArray, and float64 traps on the GPU.

@Suite("Muse Glimmer centered-norm fold ownership")
struct MuseGlimmerNormFoldOwnershipTests {

    private static func textOnlyConfig() throws -> MuseGlimmerConfiguration {
        let json = """
            {"model_type":"muse_glimmer","hidden_size":128,"num_hidden_layers":2,
             "intermediate_size":256,"num_attention_heads":4,"num_key_value_heads":2,
             "rms_norm_eps":1e-6,"vocab_size":1000,"rope_theta":10000.0}
            """
        return try JSONDecoder.json5().decode(
            MuseGlimmerConfiguration.self, from: Data(json.utf8))
    }

    /// Both key shapes fold: the bare one the text model sees, and the `language_model.`-prefixed
    /// one the VLM sees. One predicate, one loop, both callers.
    @Test("the shared fold covers both key shapes")
    func foldCoversBothShapes() {
        let folded = MuseGlimmerTextModel.foldCenteredNorms([
            "norm.weight": MLXArray([0.0, 0.5] as [Float]),
            "language_model.model.norm.weight": MLXArray([0.0, 0.5] as [Float]),
            "layers.0.input_layernorm.weight": MLXArray([0.0] as [Float]),
            // NOT a centered norm — must pass through untouched.
            "layers.0.self_attn.q_proj.weight": MLXArray([0.0, 0.5] as [Float]),
        ])
        #expect(folded["norm.weight"]!.asArray(Float.self) == [1.0, 1.5])
        #expect(folded["language_model.model.norm.weight"]!.asArray(Float.self) == [1.0, 1.5])
        #expect(folded["layers.0.input_layernorm.weight"]!.asArray(Float.self) == [1.0])
        #expect(folded["layers.0.self_attn.q_proj.weight"]!.asArray(Float.self) == [0.0, 0.5])
    }

    /// The guard that matters: the VLM's own `sanitize` must reach the shared fold. If someone
    /// removes the call, this fails loudly instead of shipping a zero-gain tower.
    @Test("the VLM sanitize path folds, it does not skip")
    func vlmSanitizeFolds() throws {
        let model = MuseGlimmer(try Self.textOnlyConfig())
        let out = model.sanitize(weights: [
            "language_model.model.norm.weight": MLXArray([0.0, 0.25] as [Float])
        ])
        #expect(out["language_model.model.norm.weight"]!.asArray(Float.self) == [1.0, 1.25])
    }

    /// Same input, same fold, whichever path massaged it — which is the property that used to
    /// depend on two loops staying in step by hand.
    @Test("both paths agree on a norm weight they both carry")
    func bothPathsAgree() throws {
        let raw = MLXArray([0.0, 0.5, -0.25] as [Float])
        let viaShared = MuseGlimmerTextModel.foldCenteredNorms(["norm.weight": raw])
        let viaVLM = MuseGlimmer(try Self.textOnlyConfig())
            .sanitize(weights: ["language_model.model.norm.weight": raw])
        #expect(
            viaShared["norm.weight"]!.asArray(Float.self)
                == viaVLM["language_model.model.norm.weight"]!.asArray(Float.self))
    }
    // MARK: - Caller-level parity (requested on #312)

    /// A fixture broad enough that a wrong predicate shows up: every centered spelling, plus
    /// near-misses that must NOT fold (`q_norm.weight` ends in `norm.weight` but is not centered).
    private static func fixture(prefix: String) -> [String: MLXArray] {
        [
            "\(prefix)layers.0.input_layernorm.weight": MLXArray([0.25, -0.5] as [Float]),
            "\(prefix)layers.0.post_attention_layernorm.weight": MLXArray([1.5] as [Float]),
            "\(prefix)layers.0.pre_feedforward_layernorm.weight": MLXArray([0.0] as [Float]),
            "\(prefix)layers.0.post_feedforward_layernorm.weight": MLXArray([-1.0] as [Float]),
            "\(prefix)norm.weight": MLXArray([2.0] as [Float]),
            "\(prefix)layers.0.self_attn.q_proj.weight": MLXArray([7.0, 8.0] as [Float]),
            "\(prefix)embed_tokens.weight": MLXArray([9.0] as [Float]),
        ]
    }

    /// Which keys SHOULD fold, stated here rather than read from the implementation — otherwise
    /// this asserts that the code agrees with itself.
    private static func shouldFold(_ key: String) -> Bool {
        ["input_layernorm.weight", "post_attention_layernorm.weight",
         "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", ".norm.weight"]
            .contains { key.hasSuffix($0) }
    }

    /// Exhaustive parity through the REAL caller, not the helper: exact key set, dtype, shape and
    /// every value — including the ones that must not move.
    @Test("the VLM caller folds exactly the centered norms, exactly once, and changes nothing else")
    func vlmCallerExhaustiveParity() throws {
        let input = Self.fixture(prefix: "language_model.model.")
        let out = MuseGlimmer(try Self.textOnlyConfig()).sanitize(weights: input)

        #expect(Set(out.keys) == Set(input.keys), "key set changed: \(Set(out.keys).symmetricDifference(Set(input.keys)))")
        for (key, before) in input {
            let after = try #require(out[key], "\(key) vanished")
            #expect(after.dtype == before.dtype, "\(key): dtype changed")
            #expect(after.shape == before.shape, "\(key): shape changed")
            let want = before.asArray(Float.self).map { Self.shouldFold(key) ? $0 + 1 : $0 }
            #expect(
                after.asArray(Float.self) == want,
                "\(key): got \(after.asArray(Float.self)) want \(want) (folds=\(Self.shouldFold(key)))")
        }
    }

    /// The same fixture through the shared owner, so the two callers are compared to each other
    /// rather than each to its own expectation.
    @Test("both callers produce the same fold for the same weights")
    func callersAgreeExhaustively() throws {
        let bare = Self.fixture(prefix: "")
        let viaText = MuseGlimmerTextModel.foldCenteredNorms(bare)
        let viaVLM = MuseGlimmer(try Self.textOnlyConfig())
            .sanitize(weights: Self.fixture(prefix: "language_model.model."))
        for (key, folded) in viaText {
            let vlmKey = "language_model.model.\(key)"
            let other = try #require(viaVLM[vlmKey], "\(vlmKey) missing from the VLM path")
            #expect(
                folded.asArray(Float.self) == other.asArray(Float.self),
                "\(key): text \(folded.asArray(Float.self)) vs VLM \(other.asArray(Float.self))")
        }
    }

    /// A shared helper makes one failure possible that two copies did not: calling it twice. The
    /// fold is deliberately NOT idempotent, so a double call is observable — which is the only
    /// reason this test can exist.
    @Test("a double fold is observable, and neither caller performs one")
    func doubleFoldIsObservableAndAbsent() throws {
        let once = MuseGlimmerTextModel.foldCenteredNorms(["norm.weight": MLXArray([2.0] as [Float])])
        let twice = MuseGlimmerTextModel.foldCenteredNorms(once)
        #expect(once["norm.weight"]!.asArray(Float.self) == [3.0])
        #expect(twice["norm.weight"]!.asArray(Float.self) == [4.0], "not idempotent, so detectable")

        // And the real callers land on the single-fold value.
        let text = MuseGlimmerTextModel.foldCenteredNorms(["norm.weight": MLXArray([2.0] as [Float])])
        let vlm = MuseGlimmer(try Self.textOnlyConfig())
            .sanitize(weights: ["language_model.model.norm.weight": MLXArray([2.0] as [Float])])
        #expect(text["norm.weight"]!.asArray(Float.self) == [3.0])
        #expect(vlm["language_model.model.norm.weight"]!.asArray(Float.self) == [3.0])
    }

}
