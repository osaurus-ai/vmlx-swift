// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The orientation of a translated LoRA factor cannot be checked by loading successfully. Where the
// dimensions happen to agree, a transposed factor loads fine and computes a DIFFERENT function —
// the model generates fluent, plausible, wrong text. So the round trip is checked numerically
// against the reference PEFT formula.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@Suite("PEFT adapter translation")
struct PEFTAdapterTranslationTests {

    /// The claim, stated as arithmetic.
    ///
    ///   PEFT  : y = (alpha/r) * (x @ Aᵀ @ Bᵀ)      A=(r, in)  B=(out, r)
    ///   MLX   : y = scale     * (x @ a  @ b )      a=(in, r)  b=(r, out)
    ///
    /// so `a = Aᵀ`, `b = Bᵀ`, `scale = alpha/r`. If either transpose is dropped the two disagree.
    @Test("the translated factors compute the same update as the PEFT reference")
    func roundTripMatchesReference() {
        let (inDim, outDim, rank) = (16, 12, 4)
        MLXRandom.seed(7)
        let A = MLXRandom.normal([rank, inDim])      // PEFT orientation
        let B = MLXRandom.normal([outDim, rank])
        let x = MLXRandom.normal([3, inDim])
        let alpha: Float = 8
        let scale = alpha / Float(rank)

        // Reference: exactly what PEFT computes.
        let reference = scale * matmul(matmul(x, A.transposed(1, 0)), B.transposed(1, 0))

        // Translated, then fed through MLX's own forward shape.
        let t = PEFTAdapterTestBridge.translate([
            "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": A,
            "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": B,
        ])
        let a = t["model.layers.0.mlp.down_proj.lora_a"]!
        let b = t["model.layers.0.mlp.down_proj.lora_b"]!
        #expect(a.shape == [inDim, rank], "lora_a must be (in, rank)")
        #expect(b.shape == [rank, outDim], "lora_b must be (rank, out)")

        let got = scale * matmul(matmul(x, a), b)
        let maxDiff = abs(got - reference).max().item(Float.self)
        #expect(maxDiff < 1e-4, "translated update differs from the PEFT reference by \(maxDiff)")
    }

    /// The negative control. Without it the test above proves only that some arrangement works,
    /// not that THIS one is required — and a test that passes for the wrong reason is the failure
    /// mode this whole exercise keeps hitting.
    @Test("skipping either transpose gives a DIFFERENT function")
    func untransposedDisagrees() {
        let (inDim, outDim, rank) = (16, 12, 4)
        MLXRandom.seed(11)
        let A = MLXRandom.normal([rank, inDim])
        let B = MLXRandom.normal([outDim, rank])
        let x = MLXRandom.normal([3, inDim])
        let reference = matmul(matmul(x, A.transposed(1, 0)), B.transposed(1, 0))

        // A left un-transposed is (r, in): x @ A is not even conformable, so the wrong-ness shows
        // as a shape error rather than a number. Assert the shapes differ, which is the same claim.
        #expect(A.shape != [inDim, rank])
        #expect(B.shape != [rank, outDim])

        // Both transposed the WRONG way round (a=Bᵀ, b=Aᵀ) is conformable here only if in == out,
        // so use the square case to make the numerical disagreement visible.
        let sq = 8
        MLXRandom.seed(13)
        let A2 = MLXRandom.normal([rank, sq])
        let B2 = MLXRandom.normal([sq, rank])
        let x2 = MLXRandom.normal([3, sq])
        let right = matmul(matmul(x2, A2.transposed(1, 0)), B2.transposed(1, 0))
        let swapped = matmul(matmul(x2, B2), A2)      // factors exchanged
        let diff = abs(right - swapped).max().item(Float.self)
        #expect(diff > 1e-3, "swapping the factors should NOT agree, but differed by only \(diff)")
    }

    /// Scale is as easy to get wrong as orientation, and fails just as quietly: the model loads,
    /// generates, and applies the adaptation at the wrong strength.
    @Test("scale is alpha/r, not MLX's default")
    func scaleFollowsPEFT() throws {
        let json = #"{"peft_type":"LORA","r":32,"lora_alpha":32,"target_modules":["down_proj"]}"#
        let cfg = try #require(PEFTAdapterTestBridge.detect(Data(json.utf8)))
        let weights = ["model.layers.0.mlp.down_proj.lora_a": MLXArray.zeros([4, 2])]
        let lora = PEFTAdapterTestBridge.configuration(cfg, weights)
        #expect(lora.loraParameters.rank == 32)
        #expect(lora.loraParameters.scale == 1.0)     // 32/32 — NOT the 10–20 MLX defaults to
        #expect(lora.loraParameters.keys == ["down_proj"])
        #expect(lora.numLayers == 1)
    }

    /// PEFT writes `target_modules` two different ways, and both appear in the wild — in two
    /// adapters from the SAME author. Qwen3.8-absolute-heresy uses leaf names; gemma-4-SOMPOA uses
    /// full checkpoint paths, all 60 of them. Whichever way it is spelled, what has to come out is
    /// the MODULE path, because that is what `replaceLayers` matches against.
    ///
    /// Asserted against a live module tree rather than against string handling, so the test fails
    /// if the expansion stops resolving real modules — not merely if the parsing changes.
    @Test("target_modules resolves to module paths from leaf names OR from full paths")
    func targetModulesBothConventions() {
        let layers: [Module] = [ProbeLayer(), ProbeLayer()]
        let expected = ["mlp.down_proj", "self_attn.o_proj"]

        let fromLeaves = PEFTAdapterTestBridge.expandTargetKeys(["o_proj", "down_proj"], in: layers)
        #expect(fromLeaves == expected)

        // The same two projections, spelled as checkpoint paths from a DIFFERENT model's naming.
        // The layer indices in them are noise: which layers get LoRA is carried by `numLayers` and
        // by which weights the adapter ships, never by these strings.
        let fromPaths = PEFTAdapterTestBridge.expandTargetKeys(
            [
                "model.language_model.layers.16.mlp.down_proj",
                "model.language_model.layers.7.self_attn.o_proj",
            ], in: layers)
        #expect(fromPaths == expected, "full-path targets resolved to \(fromPaths)")
        #expect(fromPaths == fromLeaves, "the two spellings must be interchangeable")
    }

    /// A target the model does not have must NOT silently become a module path.
    @Test("an unknown target expands to nothing invented")
    func unknownTargetIsNotInvented() {
        let expanded = PEFTAdapterTestBridge.expandTargetKeys(["not_a_projection"], in: [ProbeLayer()])
        #expect(!expanded.contains { $0.hasPrefix("mlp.") || $0.hasPrefix("self_attn.") })
    }

    /// An MLX-native adapter must keep loading exactly as before.
    @Test("an MLX adapter is not mistaken for a PEFT one")
    func mlxAdapterUnaffected() {
        let json = #"{"fine_tune_type":"lora","num_layers":28,"lora_parameters":{"rank":16,"scale":20.0}}"#
        #expect(PEFTAdapterTestBridge.detect(Data(json.utf8)) == nil)
    }
}

/// A two-deep stand-in for a decoder layer: the projections PEFT targets do not sit at the top of a
/// layer, they sit under `self_attn` / `mlp`, which is the whole reason leaf names need expanding.
private class ProbeLayer: Module {
    private class Attention: Module {
        @ModuleInfo(key: "o_proj") var oProj: Linear
        override init() { self._oProj.wrappedValue = Linear(4, 4); super.init() }
    }
    private class Feedforward: Module {
        @ModuleInfo(key: "down_proj") var downProj: Linear
        override init() { self._downProj.wrappedValue = Linear(4, 4); super.init() }
    }
    @ModuleInfo(key: "self_attn") private var attention: Attention
    @ModuleInfo(key: "mlp") private var feedforward: Feedforward
    override init() {
        self._attention.wrappedValue = Attention()
        self._feedforward.wrappedValue = Feedforward()
        super.init()
    }
}
