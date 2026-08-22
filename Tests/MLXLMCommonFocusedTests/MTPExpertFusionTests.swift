// Copyright © 2026 Osaurus AI. All rights reserved.
//
// The MTP block's MoE experts ship UNFUSED while the main layers ship fused.
//
// Measured on Ornith-1.5-35B-A3B-JANG_4M:
//
//   language_model.model.layers.N.mlp.switch_mlp.gate_proj.weight  360 keys, fused
//   mtp.layers.N.mlp.experts.N.gate_proj.weight                    2304 keys, NOT fused
//
// `Qwen35SparseMoeBlock` only has a `switch_mlp` (SwitchGLU) slot, so the
// unfused keys had nowhere to land and every load of that bundle's MTP head
// died with `unhandledKeys(... "SparseMoeBlock", keys: ["experts"])`. The
// weights were present and correct; nothing could map them — which is why all
// four Ornith-1.5-35B bundles ship with no MTP tuning file. The head could not
// load, so it could never be measured, so it could never be enabled.

import Foundation
import MLX
import Testing

@testable import MLXLLM

@Suite("MTP expert fusion")
struct MTPExpertFusionTests {

    private func expertWeights(experts: Int, layers: Int = 1) -> [String: MLXArray] {
        var w: [String: MLXArray] = [:]
        for l in 0..<layers {
            for e in 0..<experts {
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    for comp in ["weight", "scales", "biases"] {
                        // Distinct values so a mis-ordered stack is detectable.
                        w["mtp.layers.\(l).mlp.experts.\(e).\(proj).\(comp)"] =
                            MLXArray([Float(e * 100 + l)]).reshaped([1, 1])
                    }
                }
            }
        }
        return w
    }

    @Test("unfused MTP experts are stacked into switch_mlp")
    func unfusedExpertsAreStacked() {
        let fused = Qwen35TextModel.fusingMTPExpertWeights(expertWeights(experts: 4), numExperts: 4)

        #expect(fused["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"] != nil)
        #expect(fused["mtp.layers.0.mlp.switch_mlp.gate_proj.scales"] != nil)
        #expect(fused["mtp.layers.0.mlp.switch_mlp.gate_proj.biases"] != nil)

        // The per-expert keys must be GONE, or the loader still sees `experts`
        // and fails exactly as before.
        #expect(fused.keys.contains { $0.contains(".mlp.experts.") } == false)

        // Leading axis is the expert axis, in expert order.
        let w = fused["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"]!
        #expect(w.shape[0] == 4)
        MLX.eval(w)
        let flat = w.reshaped([4]).asArray(Float.self)
        #expect(flat == [0, 100, 200, 300])
    }

    /// A bundle that already ships fused must be untouched — this runs on every
    /// Qwen3.6/3.8 load too.
    @Test("already-fused weights pass through unchanged")
    func fusedWeightsAreUntouched() {
        let already: [String: MLXArray] = [
            "mtp.layers.0.mlp.switch_mlp.gate_proj.weight": MLXArray([Float(1)]),
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": MLXArray([Float(2)]),
        ]
        let out = Qwen35TextModel.fusingMTPExpertWeights(already, numExperts: 4)
        #expect(out.count == already.count)
        #expect(out["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"] != nil)
    }

    /// The MAIN layers must never be rewritten. They are already fused, and a
    /// stray rewrite there would corrupt the model that currently works.
    @Test("non-MTP expert keys are left alone")
    func mainLayerExpertsAreNotTouched() {
        let main: [String: MLXArray] = [
            "language_model.model.layers.0.mlp.experts.0.gate_proj.weight": MLXArray([Float(1)])
        ]
        let out = Qwen35TextModel.fusingMTPExpertWeights(main, numExperts: 1)
        #expect(out["language_model.model.layers.0.mlp.experts.0.gate_proj.weight"] != nil)
        #expect(out["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"] == nil)
    }

    /// An incomplete expert set must be dropped rather than stacked. Stacking
    /// 3 of 4 experts would silently encode the wrong routing — a fast, wrong
    /// model is worse than one that refuses to load.
    @Test("an incomplete expert set is not fused")
    func incompleteExpertSetIsRefused() {
        var partial = expertWeights(experts: 4)
        for k in partial.keys where k.contains(".experts.2.") { partial[k] = nil }

        let out = Qwen35TextModel.fusingMTPExpertWeights(partial, numExperts: 4)
        #expect(out["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"] == nil)
    }

    /// Multiple MTP layers each fuse independently.
    @Test("each MTP layer fuses separately")
    func multipleLayersFuseIndependently() {
        let out = Qwen35TextModel.fusingMTPExpertWeights(
            expertWeights(experts: 2, layers: 3), numExperts: 2)
        for l in 0..<3 {
            let w = out["mtp.layers.\(l).mlp.switch_mlp.up_proj.weight"]
            #expect(w != nil, "layer \(l) did not fuse")
            #expect(w?.shape[0] == 2)
        }
    }
}
