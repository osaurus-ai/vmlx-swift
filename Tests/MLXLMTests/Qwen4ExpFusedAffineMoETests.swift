import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// Numerical parity coverage for `Qwen4ExpFusedAffineMoE`.
///
/// Every combination below ships in a real JANG bundle (source:
/// /Users/eric/models/Logs/q38fn-maps). The fused kernel and the eager
/// gather-QMM fallback are both compared against the exact f32 result computed
/// from dequantized weights. A packing/unpack defect produces O(1) relative
/// error; legitimate accumulation-order drift stays at bf16 rounding scale in
/// both paths.
@Suite("Qwen4Exp fused affine MoE kernel", .serialized)
struct Qwen4ExpFusedAffineMoETests {

    private static let inputDims = 2560
    private static let expertDims = 640
    private static let experts = 16
    private static let topK = 10

    private struct Combo {
        let label: String
        let gate: (bits: Int, group: Int)
        let up: (bits: Int, group: Int)
        let down: (bits: Int, group: Int)
    }

    /// Shipped layouts: 4M uniform, the three 4S mixes incl. the layer-44
    /// g32 outlier, the 2L down g32 outlier, and the two 6S q6 mixes.
    private static let shippedCombos: [Combo] = [
        Combo(label: "4M_uniform_q4g64", gate: (4, 64), up: (4, 64), down: (4, 64)),
        Combo(label: "4S_common_q2q3q3", gate: (2, 64), up: (3, 64), down: (3, 64)),
        Combo(label: "4S_upper_q3q3q4", gate: (3, 64), up: (3, 64), down: (4, 64)),
        Combo(label: "4S_L44_gate_g32", gate: (2, 32), up: (2, 64), down: (4, 64)),
        Combo(label: "2L_down_g32", gate: (2, 64), up: (2, 64), down: (2, 32)),
        Combo(label: "6S_down_q6", gate: (4, 64), up: (4, 64), down: (6, 64)),
        Combo(label: "6S_up_down_q6", gate: (4, 64), up: (6, 64), down: (6, 64)),
        Combo(label: "Ornith_late_gate_up_q5", gate: (5, 64), up: (5, 64), down: (4, 64)),
        Combo(label: "q5_all_projections", gate: (5, 64), up: (5, 64), down: (5, 64)),
    ]

    private static func makeProjection(
        inputDims: Int, outputDims: Int, bits: Int, groupSize: Int, seed: UInt64
    ) -> (module: QuantizedSwitchLinear, dequantized: MLXArray) {
        let source = MLXRandom.uniform(
            low: -0.5, high: 0.5, [experts, outputDims, inputDims],
            key: MLXRandom.key(seed)
        ).asType(.float16)
        let (weight, scales, biases) = MLX.quantized(
            source, groupSize: groupSize, bits: bits, mode: .affine)
        let module = QuantizedSwitchLinear(
            inputDims: inputDims,
            outputDims: outputDims,
            numExperts: experts,
            weight: weight,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: .affine)
        let exact = dequantized(
            weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, mode: .affine, dtype: .float32)
        return (module, exact)
    }

    private static func relativeError(_ candidate: MLXArray, _ reference: MLXArray) -> Float {
        let c = candidate.asType(.float32).reshaped(-1)
        let r = reference.asType(.float32).reshaped(-1)
        let num = MLX.sqrt(((c - r) * (c - r)).sum())
        let den = MLX.sqrt((r * r).sum())
        return (num / den).item(Float.self)
    }

    @Test("fused kernel matches exact math for every shipped bit/group layout")
    func shippedLayoutParity() throws {
        for (comboIndex, combo) in Self.shippedCombos.enumerated() {
            let seedBase = UInt64(1000 + comboIndex * 17)
            let (gate, gateExact) = Self.makeProjection(
                inputDims: Self.inputDims, outputDims: Self.expertDims,
                bits: combo.gate.bits, groupSize: combo.gate.group, seed: seedBase)
            let (up, upExact) = Self.makeProjection(
                inputDims: Self.inputDims, outputDims: Self.expertDims,
                bits: combo.up.bits, groupSize: combo.up.group, seed: seedBase + 1)
            let (down, downExact) = Self.makeProjection(
                inputDims: Self.expertDims, outputDims: Self.inputDims,
                bits: combo.down.bits, groupSize: combo.down.group, seed: seedBase + 2)

            guard
                let reducer = Qwen4ExpFusedAffineMoE.makeReducer(
                    gate: gate, up: up, down: down)
            else {
                Issue.record("construction rejected for \(combo.label)")
                continue
            }

            let x = MLXRandom.uniform(
                low: -1.0, high: 1.0, [1, 1, Self.inputDims],
                key: MLXRandom.key(seedBase + 3)
            ).asType(.bfloat16)
            let indexValues: [UInt32] = [0, 3, 5, 7, 8, 9, 11, 12, 14, 15]
            let indices = MLXArray(indexValues, [1, 1, Self.topK])
            let scores = MLX.softmax(
                MLXRandom.uniform(
                    low: 0.0, high: 1.0, [1, 1, Self.topK],
                    key: MLXRandom.key(seedBase + 4)
                ).asType(.float32), axis: -1)

            guard let fused = reducer(x, indices, scores) else {
                Issue.record("invocation rejected for \(combo.label)")
                continue
            }

            // Exact f32 reference from dequantized weights.
            let xf = x.asType(.float32).reshaped([Self.inputDims, 1])
            var exact = MLXArray.zeros([Self.inputDims], dtype: .float32)
            // Production-like eager reference: per-expert quantized matmul in
            // bf16, expert outputs rounded to bf16 before the weighted sum —
            // the same rounding points as the SwitchGLU fallback path.
            let xb = x.reshaped([1, Self.inputDims])
            var eager = MLXArray.zeros([Self.inputDims], dtype: .float32)
            for (k, expert) in indexValues.enumerated() {
                let e = Int(expert)
                let score = scores.reshaped(-1)[k].asType(.float32)

                let gExact = MLX.matmul(gateExact[e], xf).reshaped(-1)
                let uExact = MLX.matmul(upExact[e], xf).reshaped(-1)
                let actExact = (gExact * MLX.sigmoid(gExact)) * uExact
                let dExact = MLX.matmul(downExact[e], actExact.reshaped([Self.expertDims, 1]))
                exact = exact + score * dExact.reshaped(-1)

                let gQ = quantizedMM(
                    xb, gate.weight[e], scales: gate.scales[e], biases: gate.biases?[e],
                    transpose: true, groupSize: combo.gate.group, bits: combo.gate.bits,
                    mode: .affine)
                let uQ = quantizedMM(
                    xb, up.weight[e], scales: up.scales[e], biases: up.biases?[e],
                    transpose: true, groupSize: combo.up.group, bits: combo.up.bits,
                    mode: .affine)
                let actQ = (gQ * MLX.sigmoid(gQ)) * uQ
                let dQ = quantizedMM(
                    actQ, down.weight[e], scales: down.scales[e], biases: down.biases?[e],
                    transpose: true, groupSize: combo.down.group, bits: combo.down.bits,
                    mode: .affine)
                eager = eager + score * dQ.asType(.float32).reshaped(-1)
            }

            let fusedError = Self.relativeError(fused, exact)
            let eagerError = Self.relativeError(eager, exact)
            print(
                "[FusedMoEParity] combo=\(combo.label) fused_rel=\(fusedError)"
                    + " eager_rel=\(eagerError)")

            // A broken unpack is O(1); rounding drift is bf16-scale. The fused
            // path must stay in the same error regime as the eager fallback.
            #expect(fusedError < 0.02, "combo \(combo.label) fused error \(fusedError)")
            #expect(eagerError < 0.02, "combo \(combo.label) eager error \(eagerError)")
            #expect(
                fusedError < max(4 * eagerError, 0.01),
                "combo \(combo.label): fused \(fusedError) far above eager \(eagerError)")
        }
    }

    @Test("Ornith 2048x512 top-8 q5 contract matches eager quantized math")
    func ornith35ShapeParity() throws {
        let inputDims = 2048
        let expertDims = 512
        let topK = 8
        let seed: UInt64 = 9001
        let (gate, _) = Self.makeProjection(
            inputDims: inputDims, outputDims: expertDims,
            bits: 5, groupSize: 64, seed: seed)
        let (up, _) = Self.makeProjection(
            inputDims: inputDims, outputDims: expertDims,
            bits: 5, groupSize: 64, seed: seed + 1)
        let (down, _) = Self.makeProjection(
            inputDims: expertDims, outputDims: inputDims,
            bits: 4, groupSize: 64, seed: seed + 2)

        let reducer = try #require(
            Qwen4ExpFusedAffineMoE.makeReducer(
                gate: gate, up: up, down: down))
        let x = MLXRandom.uniform(
            low: -1.0, high: 1.0, [1, 1, inputDims],
            key: MLXRandom.key(seed + 3)
        ).asType(.bfloat16)
        let indexValues: [UInt32] = [0, 2, 4, 6, 9, 11, 13, 15]
        let indices = MLXArray(indexValues, [1, 1, topK])
        let scores = MLX.softmax(
            MLXRandom.uniform(
                low: 0.0, high: 1.0, [1, 1, topK],
                key: MLXRandom.key(seed + 4)
            ).asType(.float32), axis: -1)
        let fused = try #require(reducer(x, indices, scores))

        let xb = x.reshaped([1, inputDims])
        var eager = MLXArray.zeros([inputDims], dtype: .float32)
        for (k, expert) in indexValues.enumerated() {
            let e = Int(expert)
            let g = quantizedMM(
                xb, gate.weight[e], scales: gate.scales[e], biases: gate.biases?[e],
                transpose: true, groupSize: 64, bits: 5, mode: .affine)
            let u = quantizedMM(
                xb, up.weight[e], scales: up.scales[e], biases: up.biases?[e],
                transpose: true, groupSize: 64, bits: 5, mode: .affine)
            let activation = (g * MLX.sigmoid(g)) * u
            let projected = quantizedMM(
                activation, down.weight[e], scales: down.scales[e],
                biases: down.biases?[e], transpose: true,
                groupSize: 64, bits: 4, mode: .affine)
            eager =
                eager
                + scores.reshaped(-1)[k].asType(.float32)
                * projected.asType(.float32).reshaped(-1)
        }

        let error = Self.relativeError(fused, eager)
        print("[FusedMoEParity] combo=ornith35_q5q5q4_g64_top8 fused_vs_eager=\(error)")
        #expect(error < 0.01, "Ornith fused/eager relative error \(error)")
    }

    @Test("construction rejects unsupported metadata and group sizes")
    func constructionRejection() throws {
        // f32 affine metadata must be rejected: the kernels only accept
        // bf16/f16 scales with matching bias dtype.
        let f32Source = MLXRandom.uniform(
            low: -0.5, high: 0.5, [Self.experts, Self.expertDims, Self.inputDims],
            key: MLXRandom.key(7)
        ).asType(.float32)
        let (w32, s32, b32) = MLX.quantized(f32Source, groupSize: 64, bits: 4, mode: .affine)
        let f32Projection = QuantizedSwitchLinear(
            inputDims: Self.inputDims, outputDims: Self.expertDims,
            numExperts: Self.experts,
            weight: w32, scales: s32, biases: b32, groupSize: 64, bits: 4, mode: .affine)

        let (good, _) = Self.makeProjection(
            inputDims: Self.inputDims, outputDims: Self.expertDims,
            bits: 4, groupSize: 64, seed: 11)
        let (goodDown, _) = Self.makeProjection(
            inputDims: Self.expertDims, outputDims: Self.inputDims,
            bits: 4, groupSize: 64, seed: 12)

        #expect(
            Qwen4ExpFusedAffineMoE.makeReducer(
                gate: f32Projection, up: good, down: goodDown) == nil)

        // Group size 128 is outside the supported set.
        let g128Source = MLXRandom.uniform(
            low: -0.5, high: 0.5, [Self.experts, Self.expertDims, Self.inputDims],
            key: MLXRandom.key(8)
        ).asType(.float16)
        let (w128, s128, b128) = MLX.quantized(g128Source, groupSize: 128, bits: 4, mode: .affine)
        let g128Projection = QuantizedSwitchLinear(
            inputDims: Self.inputDims, outputDims: Self.expertDims,
            numExperts: Self.experts,
            weight: w128, scales: s128, biases: b128, groupSize: 128, bits: 4, mode: .affine)

        #expect(
            Qwen4ExpFusedAffineMoE.makeReducer(
                gate: g128Projection, up: good, down: goodDown) == nil)
    }
}
