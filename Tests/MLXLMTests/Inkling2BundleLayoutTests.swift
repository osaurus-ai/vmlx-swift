// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
import MLX
@testable import MLXLLM
import MLXNN
import Testing

/// Community Inkling-2 bundles store the same model in other trees and encodings. These check that each
/// one is brought to exactly the tensors the official checkpoint produces.
@Suite("Inkling-2 bundle layouts", .serialized)
struct Inkling2BundleLayoutTests {
    private let E = 3, I = 8, H = 64

    /// Distinct, recognisable rows: row r of expert e holds e * 1000 + r in every word.
    private func rows(_ count: Int, cols: Int, offset: Int) -> MLXArray {
        let values = (0 ..< E * count * cols).map { i -> UInt32 in
            let e = i / (count * cols), r = (i / cols) % count
            return UInt32(e * 1000 + offset + r)
        }
        return MLXArray(values, [E, count, cols])
    }

    @Test("mlx-lm gate_up_proj (w13 order) and mlx-vlm gate/up give the official interleaved rows")
    func expertRowsMatchOfficialInterleave() {
        let gate = rows(I, cols: 4, offset: 0)
        let up = rows(I, cols: 4, offset: 500)
        // The official w13: row 2i = gate_i, row 2i+1 = up_i.
        let official = MLX.stacked([gate, up], axis: 2).reshaped([E, 2 * I, 4])
        let scales = MLXArray.zeros([E, 2 * I, 1], dtype: .uint8)

        let hf = Inkling2LoopModel.canonicalExperts([
            "model.layers.0.mlp.|hf_gate_up_proj": [
                // mlx-lm renames w13 and keeps its interleaved rows.
                "weight": official, "scales": scales,
            ]
        ])
        let codesHF = try! #require(hf["model.layers.0.mlp.switch_mlp.gate_up_proj.codes"])
        #expect(MLX.arrayEqual(codesHF, official).item(Bool.self))

        let vlm = Inkling2LoopModel.canonicalExperts([
            "model.layers.0.mlp.|vlm_gate_proj": ["weight": gate, "scales": scales[0..., ..<I, 0...]],
            "model.layers.0.mlp.|vlm_up_proj": ["weight": up, "scales": scales[0..., ..<I, 0...]],
        ])
        let codesVLM = try! #require(vlm["model.layers.0.mlp.switch_mlp.gate_up_proj.codes"])
        #expect(MLX.arrayEqual(codesVLM, official).item(Bool.self))
    }

    @Test("per-expert factors: MLX global_scale / 2688, ModelOpt weight_scale_2 as is")
    func expertFactors() {
        let w = MLXArray.zeros([E, H, H / 8], dtype: .uint32)
        let s = MLXArray.zeros([E, H, H / 16], dtype: .uint8)
        let g = MLXArray([Float(2688), 1344, 5376])
        let a = Inkling2LoopModel.canonicalExperts([
            "model.layers.1.mlp.|hf_down_proj": ["weight": w, "scales": s, "global_scale": g]
        ])
        let fa = try! #require(a["model.layers.1.mlp.switch_mlp.down_proj.expert_scale"])
        #expect(fa.asArray(Float.self) == [1, 0.5, 2])
        let b = Inkling2LoopModel.canonicalExperts([
            "model.layers.1.mlp.|vlm_down_proj": [
                "weight": w, "scales": s, "weight_scale_2": MLXArray([Float(0.25), 0.5, 1]),
            ]
        ])
        let fb = try! #require(b["model.layers.1.mlp.switch_mlp.down_proj.expert_scale"])
        #expect(fb.asArray(Float.self) == [0.25, 0.5, 1])
        // No factor: ones.
        let c = Inkling2LoopModel.canonicalExperts([
            "model.layers.1.mlp.|hf_down_proj": ["weight": w, "scales": s]
        ])
        #expect(c["model.layers.1.mlp.switch_mlp.down_proj.expert_scale"]!.asArray(Float.self) == [1, 1, 1])
    }

    @Test("gate and up with different per-expert factors are refused, not fused wrong")
    func mismatchedGateUpFactorsAreRefused() {
        let w = MLXArray.zeros([E, I, H / 8], dtype: .uint32)
        let s = MLXArray.zeros([E, I, H / 16], dtype: .uint8)
        let out = Inkling2LoopModel.canonicalExperts([
            "model.layers.2.mlp.|vlm_gate_proj": ["weight": w, "scales": s, "weight_scale_2": MLXArray([Float(1), 1, 1])],
            "model.layers.2.mlp.|vlm_up_proj": ["weight": w, "scales": s, "weight_scale_2": MLXArray([Float(1), 2, 1])],
        ])
        #expect(out["model.layers.2.mlp.switch_mlp.gate_up_proj.codes"] == nil)
    }

    @Test("expert format is read off the tensors")
    func formatInference() {
        typealias L = Inkling2NativeSwitchLinear
        let u8 = { (c: Int) in MLXArray.zeros([1, 4, c], dtype: .uint8) }
        let bf = { (c: Int) in MLXArray.zeros([1, 4, c], dtype: .bfloat16) }
        let codes = { (c: Int) in MLXArray.zeros([1, 4, c], dtype: .uint32) }
        #expect(L.format(codes: codes(256 / 8), scales: u8(256 / 16), inputDims: 256).mode == .nvfp4)
        #expect(L.format(codes: codes(256 / 4), scales: u8(256 / 32), inputDims: 256).mode == .mxfp8)
        #expect(L.format(codes: codes(256 / 8), scales: u8(256 / 32), inputDims: 256).mode == .mxfp4)
        let affine = L.format(codes: codes(256 * 3 / 32), scales: bf(256 / 64), inputDims: 256)
        #expect(affine == .init(mode: .affine, bits: 3, groupSize: 64))
        // Same packed width as MXFP8, told apart by the scale dtype.
        let affine4 = L.format(codes: codes(256 / 8), scales: bf(256 / 64), inputDims: 256)
        #expect(affine4 == .init(mode: .affine, bits: 4, groupSize: 64))
    }

    @Test("affine routed experts run through the gather and match the dense product")
    func affineExpertGatherMatchesDense() {
        MLXRandom.seed(7)
        let dense = MLXRandom.normal([E, 2 * I, H]).asType(.bfloat16)
        let (q, sc, bi) = MLX.quantized(dense, groupSize: 32, bits: 4)
        let lin = Inkling2NativeSwitchLinear(inputDims: H, outputDims: 2 * I, experts: E)
        try! lin.update(
            parameters: ModuleParameters.unflattened([
                "codes": q, "block_scales": sc, "affine_biases": bi!,
                "expert_scale": MLXArray.ones([E], dtype: .float32),
            ]), verify: .none)
        let x = MLXRandom.normal([1, 1, 1, H])
        let idx = MLXArray([UInt32(2)])
        let got = lin(x, idx, sorted: false).reshaped([2 * I])
        // The gather runs affine experts in the scale dtype (bf16 activations, bf16 output).
        let ref = MLX.matmul(
            x.asType(.bfloat16).asType(.float32).reshaped([1, H]),
            MLX.dequantized(q, scales: sc, biases: bi, groupSize: 32, bits: 4)[2].asType(.float32).T
        ).reshaped([2 * I])
        let err = MLX.abs(got.asType(.float32) - ref).max().item(Float.self)
        let mag = MLX.abs(ref).max().item(Float.self)
        #expect(mag > 0.1, "INVALID: reference is ~zero")
        #expect(err <= mag * 8e-3)  // one bf16 ulp of the output
    }

    @Test("quantized dense layers are dequantized bit-exactly; widths come from the model's shape")
    func denseDequantization() {
        MLXRandom.seed(3)
        let w = MLXRandom.normal([8, H]).asType(.bfloat16)
        let (q, sc, bi) = MLX.quantized(w, groupSize: 32, bits: 4)
        let key = "model.language_model.layers.0.self_attn.sink_proj"
        let out = Inkling2LoopModel.dequantizingDenseLayers(
            [key + ".weight": q, key + ".scales": sc, key + ".biases": bi!],
            expected: ["model.layers.0.self_attn.sink_proj.weight": [8, H]])
        let got = try! #require(out[key + ".weight"])
        let ref = MLX.dequantized(q, scales: sc, biases: bi, groupSize: 32, bits: 4, dtype: .bfloat16)
        #expect(got.dtype == .bfloat16)
        #expect(MLX.arrayEqual(got, ref).item(Bool.self))
        #expect(out[key + ".scales"] == nil && out[key + ".biases"] == nil)
        // Experts are never touched by the dense pass.
        let e = Inkling2LoopModel.dequantizingDenseLayers(
            ["model.layers.0.mlp.experts.down_proj.weight": q, "model.layers.0.mlp.experts.down_proj.scales": sc],
            expected: [:])
        #expect(e["model.layers.0.mlp.experts.down_proj.scales"] != nil)
    }

    @Test("mlx-vlm tree names map onto this model's tree")
    func mlxVLMNames() {
        let n = Inkling2LoopModel.normalizingLayoutKeys([
            "language_model.model.layers.3.mlp.gate_weight": MLXArray(0),
            "language_model.lm_head.weight": MLXArray(0),
        ])
        #expect(Set(n.keys) == ["model.layers.3.mlp.gate_weight", "lm_head.weight"])
        let m = Inkling2LoopModel.mapKey
        #expect(m("model.layers.3.mlp.gate_weight") == "model.layers.3.mlp.gate.weight")
        #expect(m("model.layers.3.mlp.e_score_correction_bias") == "model.layers.3.mlp.gate.e_score_correction_bias")
        #expect(m("model.layers.3.self_attn.q_scale") == "model.layers.3.self_attn.q_l2_clip.scale")
        #expect(m("model.layers.3.self_attn.rel_scale") == "model.layers.3.self_attn.rel_l2_clip.scale")
        #expect(m("model.layers.3.self_attn.rel_proj") == "model.layers.3.self_attn.rel_logits_proj.proj")
        #expect(m("model.layers.3.self_attn.k_sconv.conv.weight") == "model.layers.3.self_attn.k_sconv.weight")
        #expect(m("model.layers.3.post_mlp_convolution.conv.weight") == "model.layers.3.post_mlp_convolution.weight")
        #expect(m("model.message_queries") == "model.message_input.query")
        // Official and HF names still map as before.
        #expect(m("model.llm.layers.3.mlp.gate.bias") == "model.layers.3.mlp.gate.e_score_correction_bias")
        #expect(m("model.language_model.layers.3.mlp.gate.weight") == "model.layers.3.mlp.gate.weight")
        #expect(Inkling2LoopModel.communityExpertKey("model.layers.3.mlp.switch_mlp.down_proj.tq2_packed") == nil)
        #expect(Inkling2LoopModel.communityExpertKey("model.layers.3.mlp.switch_mlp.down_proj.codes") == nil)
    }

    @Test("fused-projection packing (in_proj, loop controllers, [C, K] taps) converts back exactly")
    func fusedProjectionRoundTrip() {
        MLXRandom.seed(11)
        let (Hd, r, K) = (16, 4, 4)
        let pre = "model.layers.7."
        let sa = pre + "self_attn."
        // Source tensors.
        let q = MLXRandom.normal([6, Hd]), k = MLXRandom.normal([2, Hd]), v = MLXRandom.normal([2, Hd])
        let rr = MLXRandom.normal([3, Hd]), g = MLXRandom.normal([5, Hd]), sink = MLXRandom.normal([1, Hd])
        let downs = (0 ..< 4).map { _ in MLXRandom.normal([r, 2 * Hd]) }
        let gains = (0 ..< 4).map { _ in MLXRandom.normal([Hd, r]) }
        let forgets = (0 ..< 4).map { _ in MLXRandom.normal([Hd, r]) }
        let conv = MLXRandom.normal([K, 2])  // this tree: [K, C]
        // The fused-projection packing.
        let loopDown = MLX.stacked(downs.map { $0.reshaped([r, 2, Hd]).transposed(1, 2, 0) }, axis: 0)
        let loopGF = MLX.stacked(zip(gains, forgets).map { MLX.concatenated([$0.T, $1.T], axis: 1) }, axis: 0)
        let packed: [String: MLXArray] = [
            pre + "attn.in_proj.weight": MLX.concatenated([q, k, v, rr, g, sink], axis: 0),
            pre + "loop_down": loopDown, pre + "loop_gain_forget": loopGF,
            pre + "attn.k_conv": conv.T, pre + "mlp.router_weight": MLXArray.ones([3, Hd]),
            pre + "mlp.experts.w13.weight": MLXArray.zeros([1, 2, 1], dtype: .uint32),
            pre + "loop_message.query": MLXArray.ones([4, Hd]),
        ]
        var expected: [String: [Int]] = [:]
        for (n, m) in [("q_proj", q), ("k_proj", k), ("v_proj", v), ("r_proj", rr), ("gate_proj", g), ("sink_proj", sink)] {
            expected[sa + n + ".weight"] = m.shape
        }
        expected[sa + "k_sconv.weight"] = [K, 2]
        let out = Inkling2LoopModel.convertingFusedProjectionLayout(packed, expected: expected, loopCore: 7 ..< 8)
        for (n, m) in [("q_proj", q), ("k_proj", k), ("v_proj", v), ("r_proj", rr), ("gate_proj", g), ("sink_proj", sink)] {
            #expect(MLX.arrayEqual(out[sa + n + ".weight"]!, m).item(Bool.self), "\(n)")
        }
        for (i, gname) in ["q", "k", "v", "mlp"].enumerated() {
            #expect(MLX.arrayEqual(out[pre + "\(gname)_message_gate.down_proj.weight"]!, downs[i]).item(Bool.self))
            #expect(MLX.arrayEqual(out[pre + "\(gname)_message_gate.message_gate_proj.weight"]!, gains[i]).item(Bool.self))
            #expect(MLX.arrayEqual(out[pre + "\(gname)_message_gate.forget_gate_proj.weight"]!, forgets[i]).item(Bool.self))
        }
        #expect(MLX.arrayEqual(out[sa + "k_sconv.weight"]!, conv).item(Bool.self))
        #expect(out[pre + "mlp.gate.weight"] != nil)
        #expect(out[pre + "mlp.experts.gate_up_proj.weight"] != nil)
        #expect(out["model.message_input.query"]?.shape == [1, 4, Hd])
    }

    @Test("timing: affine expert gather by activation dtype (diagnostic)",
          .enabled(if: ProcessInfo.processInfo.environment["INKLING2_GATHER_BENCH"] == "1"))
    func affineGatherTiming() {
        let (E, H, O, k) = (128, 2304, 2304, 8)
        let w = MLXRandom.normal([E, O, H]).asType(.bfloat16)
        let (q, sc, bi) = MLX.quantized(w, groupSize: 32, bits: 4)
        MLX.eval(q, sc, bi!)
        let idx = MLXArray((0 ..< k).map { UInt32($0 * 13 % E) }).reshaped([1, k])
        for (label, x, s, b) in [
            ("x f32, scales bf16", MLXRandom.normal([1, 1, 1, H]), sc, bi!),
            ("x bf16, scales bf16", MLXRandom.normal([1, 1, 1, H]).asType(.bfloat16), sc, bi!),
            ("x f32, scales f32", MLXRandom.normal([1, 1, 1, H]), sc.asType(.float32), bi!.asType(.float32)),
        ] {
            MLX.eval(x, s, b)
            var best = Double.infinity
            for _ in 0 ..< 30 {
                let t0 = Date()
                let y = MLX.gatherQuantizedMM(x, q, scales: s, biases: b, rhsIndices: idx, transpose: true,
                    groupSize: 32, bits: 4, mode: .affine, sortedIndices: false)
                MLX.eval(y)
                best = min(best, Date().timeIntervalSince(t0))
            }
            print("[gather-bench] \(label): \(String(format: "%.1f", best * 1e6)) us")
        }
    }
}
