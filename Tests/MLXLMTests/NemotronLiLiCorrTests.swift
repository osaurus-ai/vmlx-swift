// Unit tests for the LiLiCorr drafter port with tiny synthetic random weights (no bundle, no
// real model). Every NVFP4 module is built with `quantized(mode: .nvfp4)` from a random bf16
// matrix so the exact production code path (quantizedMM nvfp4 + weight_scale_2) runs.

import Foundation
import MLX
@testable import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class NemotronLiLiCorrTests: XCTestCase {

    // MARK: - Synthetic checkpoint

    /// hidden 64, heads 4/1, head_dim 16, block 8, topk 4, lilicorr hidden 32, window 16,
    /// conv taps 2 / group 16, vocab 128, two draft layers, taps from two target layers.
    static func tinyConfiguration(window: Int = 16, lilicorr: Bool = true)
        -> NemotronLiLiCorrConfiguration
    {
        let spec = NemotronLiLiCorrConfiguration.QuantizationSpec(
            mode: "nvfp4", groupSize: 16, bits: 4, globalScale: nil)
        var quant: [String: NemotronLiLiCorrConfiguration.QuantizationSpec] = ["fc": spec]
        for i in 0 ..< 2 {
            for m in ["gate_proj", "up_proj", "down_proj"] { quant["layers.\(i).mlp.\(m)"] = spec }
        }
        for name in ["lilicorr.context_proj", "lilicorr.pass_hidden_proj", "lilicorr.token_proj"] {
            quant[name] = spec
        }
        for i in 0 ..< 2 {
            quant["lilicorr.layers.\(i).attn.out_proj"] = spec
            quant["lilicorr.layers.\(i).mlp.up_proj"] = spec
            quant["lilicorr.layers.\(i).mlp.down_proj"] = spec
        }
        return NemotronLiLiCorrConfiguration(
            blockSize: 8, hiddenSize: 64, numHiddenLayers: 2, numAttentionHeads: 4,
            numKeyValueHeads: 1, headDim: 16, intermediateSize: 96, rmsNormEps: 1e-6,
            ropeTheta: 10000, vocabSize: 128, window: window, maskTokenId: 5,
            targetLayerIds: [0, 1], convKernelSize: 2, convGroupSize: 16,
            lilicorrEnabled: lilicorr, lilicorrCandidateTopK: 4, lilicorrHiddenSize: 32,
            lilicorrNumLayers: 2, lilicorrNumHeads: 2, lilicorrVectorEps: 1e-4,
            lilicorrLogitScale: 8.0, quantization: quant, defaultQuantization: spec)
    }

    static let targetHidden = 64

    static func randomBF16(_ shape: [Int], scale: Float = 0.1) -> MLXArray {
        (MLXRandom.normal(shape) * scale).asType(.bfloat16)
    }

    /// Builds every tensor of the checkpoint: NVFP4 (weight/scales/weight_scale_2) where the
    /// configuration says so, bf16 elsewhere.
    static func tinyWeights(_ c: NemotronLiLiCorrConfiguration, seed: UInt64 = 7) -> [String: MLXArray] {
        MLXRandom.seed(seed)
        var w: [String: MLXArray] = [:]
        func linear(_ name: String, _ out: Int, _ inp: Int, bias: Bool = false) {
            let full = randomBF16([out, inp], scale: 0.2)
            if c.quantization[name] != nil {
                let (wq, scales, _) = quantized(full, groupSize: 16, bits: 4, mode: .nvfp4)
                w["\(name).weight"] = wq
                w["\(name).scales"] = scales
                w["\(name).weight_scale_2"] = MLXArray([Float(0.75)])
            } else {
                w["\(name).weight"] = full
            }
            if bias { w["\(name).bias"] = randomBF16([out]) }
        }
        func norm(_ name: String, _ d: Int) {
            w["\(name).weight"] = (MLXArray.ones([d]) + MLXRandom.normal([d]) * 0.05).asType(.bfloat16)
        }
        let D = c.hiddenSize
        let taps = c.targetLayerIds.count * targetHidden
        linear("fc", D, taps)
        norm("hidden_norm", D)
        norm("norm", D)
        let G = D / c.convGroupSize
        for i in 0 ..< c.numHiddenLayers {
            let p = "layers.\(i)"
            norm("\(p).input_layernorm", D)
            norm("\(p).post_attention_layernorm", D)
            linear("\(p).self_attn.q_proj", c.numAttentionHeads * c.headDim, D)
            linear("\(p).self_attn.k_proj", c.numKeyValueHeads * c.headDim, D)
            linear("\(p).self_attn.v_proj", c.numKeyValueHeads * c.headDim, D)
            linear("\(p).self_attn.o_proj", D, c.numAttentionHeads * c.headDim)
            norm("\(p).self_attn.q_norm", c.headDim)
            norm("\(p).self_attn.k_norm", c.headDim)
            w["\(p).self_attn.attention_sink_bias"] = randomBF16([c.numAttentionHeads], scale: 0.5)
            linear("\(p).mlp.gate_proj", c.intermediateSize, D)
            linear("\(p).mlp.up_proj", c.intermediateSize, D)
            linear("\(p).mlp.down_proj", D, c.intermediateSize)
            for conv in ["attention_conv", "mlp_conv"] {
                w["\(p).\(conv).base_kernel"] = randomBF16([2, c.convKernelSize, D], scale: 0.3)
                linear("\(p).\(conv).kernel_projection", 2 * c.convKernelSize * G, D)
            }
        }
        if c.lilicorrEnabled {
            let H = c.lilicorrHiddenSize
            let K = c.lilicorrCandidateTopK
            let p = "lilicorr"
            linear("\(p).token_proj", H, targetHidden, bias: true)
            linear("\(p).pass_hidden_proj", H, D, bias: true)
            w["\(p).feature_norm.weight"] = (MLXArray.ones([5])).asType(.bfloat16)
            w["\(p).feature_norm.bias"] = MLXArray.zeros([5]).asType(.bfloat16)
            linear("\(p).feature_mlp.up_proj", H, 5, bias: true)
            linear("\(p).feature_mlp.down_proj", H, H, bias: true)
            w["\(p).slot_embedding"] = randomBF16([1, 1, c.blockSize - 1, 1, H])
            w["\(p).rank_embedding"] = randomBF16([1, 1, 1, K, H])
            linear("\(p).context_proj", H, D, bias: true)
            for i in 0 ..< c.lilicorrNumLayers {
                let q = "\(p).layers.\(i)"
                norm("\(q).attn_norm", H)
                norm("\(q).mlp_norm", H)
                w["\(q).attn.in_proj_weight"] = randomBF16([3 * H, H], scale: 0.2)
                w["\(q).attn.in_proj_bias"] = randomBF16([3 * H])
                linear("\(q).attn.out_proj", H, H, bias: true)
                linear("\(q).mlp.up_proj", 2 * H, H, bias: true)
                linear("\(q).mlp.down_proj", H, 2 * H, bias: true)
            }
            norm("\(p).output_norm", H)
            norm("\(p).anchor_norm", H)
            w["\(p).factor_input_proj.weight"] = randomBF16([H, 3 * H], scale: 0.2)
            w["\(p).factor_input_proj.bias"] = randomBF16([H])
            linear("\(p).out_head", H, H, bias: true)
            linear("\(p).in_head", H, H, bias: true)
            linear("\(p).anchor_out_head", H, H, bias: true)
            w["\(p).relative_slot_bias"] = randomBF16([c.lilicorrNumHeads, 2 * c.blockSize - 1], scale: 0.5)
            w["\(p).same_slot_bias"] = randomBF16([c.lilicorrNumHeads], scale: 0.5)
        }
        eval(Array(w.values))
        return w
    }

    static func makeDrafter(window: Int = 16, lilicorr: Bool = true, seed: UInt64 = 7) throws
        -> NemotronLiLiCorrDrafter
    {
        let c = tinyConfiguration(window: window, lilicorr: lilicorr)
        return try NemotronLiLiCorrDrafter(configuration: c, weights: tinyWeights(c, seed: seed))
    }

    /// Shared "target": a random embedding table and an lm_head over vocab 128.
    struct Target {
        let embedding: MLXArray  // (V, Dm) bf16
        let lmHead: MLXArray  // (V, D) bf16
        init(vocab: Int, embedDim: Int, hidden: Int, seed: UInt64 = 11) {
            MLXRandom.seed(seed)
            embedding = (MLXRandom.normal([vocab, embedDim]) * 0.3).asType(.bfloat16)
            lmHead = (MLXRandom.normal([vocab, hidden]) * 0.3).asType(.bfloat16)
            eval(embedding, lmHead)
        }
        func embed(_ ids: MLXArray) -> MLXArray { embedding[ids] }
        func head(_ h: MLXArray) -> MLXArray { matmul(h, lmHead.T) }
    }

    static func randomTaps(_ n: Int, dim: Int, seed: UInt64) -> MLXArray {
        MLXRandom.seed(seed)
        let t = (MLXRandom.normal([n, dim]) * 0.5).asType(.bfloat16)
        eval(t)
        return t
    }

    static func blockEmbeddings(_ target: Target, anchor: Int32, drafter: NemotronLiLiCorrDrafter)
        -> MLXArray
    {
        let ids = [anchor] + [Int32](repeating: Int32(drafter.maskTokenId), count: drafter.blockSize - 1)
        return target.embed(MLXArray(ids).reshaped(1, drafter.blockSize))
    }

    // MARK: - Shapes

    func testShapesAndContextWindow() throws {
        let drafter = try Self.makeDrafter(window: 16)
        XCTAssertEqual(drafter.tapDimension, 2 * Self.targetHidden)
        XCTAssertEqual(drafter.blockSize, 8)
        let target = Target(vocab: 128, embedDim: Self.targetHidden, hidden: 64)

        try drafter.addContext(taps: Self.randomTaps(12, dim: drafter.tapDimension, seed: 1), position: 0)
        XCTAssertEqual(drafter.contextEnd, 12)
        XCTAssertEqual(drafter.contextRows, 12)
        XCTAssertEqual(drafter.contextK[0]?.shape, [1, 1, 12, 16])
        XCTAssertEqual(drafter.contextV[1]?.shape, [1, 1, 12, 16])

        // Appending past the window keeps only the last `window` positions.
        try drafter.addContext(taps: Self.randomTaps(9, dim: drafter.tapDimension, seed: 2), position: 12)
        XCTAssertEqual(drafter.contextEnd, 21)
        XCTAssertEqual(drafter.contextRows, 16)
        XCTAssertEqual(drafter.contextK[0]?.shape, [1, 1, 16, 16])
        XCTAssertEqual(drafter.lastContext?.shape, [64])

        let h = try drafter.draftHidden(
            blockEmbeddings: Self.blockEmbeddings(target, anchor: 3, drafter: drafter), position: 21)
        XCTAssertEqual(h.shape, [1, 8, 64])
        XCTAssertEqual(h.dtype, .bfloat16)
        let (path, top1) = try drafter.propose(hidden: h, lmHead: target.head, embed: target.embed)
        let p = try XCTUnwrap(path)
        eval(p, top1)
        XCTAssertEqual(p.shape, [7])
        XCTAssertEqual(top1.shape, [7])
        XCTAssertEqual(p.dtype, .int32)
        XCTAssertTrue(p.asArray(Int32.self).allSatisfy { (0 ..< 128).contains(Int($0)) })
        XCTAssertTrue(top1.asArray(Int32.self).allSatisfy { (0 ..< 128).contains(Int($0)) })

        // Position contract: a non-contiguous append and a draft at the wrong position throw.
        XCTAssertThrowsError(
            try drafter.addContext(taps: Self.randomTaps(1, dim: drafter.tapDimension, seed: 3), position: 20))
        XCTAssertThrowsError(
            try drafter.draftHidden(
                blockEmbeddings: Self.blockEmbeddings(target, anchor: 3, drafter: drafter), position: 20))
        // Head disabled -> no path, top-1 only.
        let plain = try Self.makeDrafter(lilicorr: false)
        try plain.addContext(taps: Self.randomTaps(4, dim: plain.tapDimension, seed: 1), position: 0)
        let h2 = try plain.draftHidden(
            blockEmbeddings: Self.blockEmbeddings(target, anchor: 3, drafter: plain), position: 4)
        let (none, t1) = try plain.propose(hidden: h2, lmHead: target.head, embed: target.embed)
        XCTAssertNil(none)
        XCTAssertEqual(t1.shape, [7])
    }

    // MARK: - Grouped conv block-position mask

    func testGroupedConvBlockPositionMask() throws {
        // taps 2, zero per-token delta, base_kernel[prepare] = [tap0 = 0, tap1 = 1]:
        // out[p] = x[p-1] when p mod block >= 1, else 0 (block starts never read the previous block).
        let D = 32
        let gs = 16
        let block = 4
        let N = 10
        var base = MLXArray.zeros([2, 2, D])
        base[0, 1] = MLXArray.ones([D])
        base = base.asType(.bfloat16)
        let kp = Linear(weight: MLXArray.zeros([2 * 2 * (D / gs), D]).asType(.bfloat16), bias: nil)
        let c = Self.tinyConfiguration()
        let weights = LiLiCorrWeights(
            tensors: ["conv.base_kernel": base, "conv.kernel_projection.weight": kp.weight],
            configuration: c)
        let conv = try LiLiCorrGroupedConv(
            weights: weights, name: "conv", blockSize: block, taps: 2, groupSize: gs)
        MLXRandom.seed(3)
        let x = MLXRandom.normal([N, D]).asType(.bfloat16)
        let (out, finishCoeff) = conv.prepare(x)
        eval(out, finishCoeff)
        XCTAssertEqual(finishCoeff.shape, [N, 2, D / gs])
        let xs = x.asType(.float32)
        let os = out.asType(.float32)
        for p in 0 ..< N {
            if p % block == 0 {
                XCTAssertEqual(abs(os[p]).max().item(Float.self), 0, "block start row \(p) must be zero")
            } else {
                XCTAssertTrue(
                    allClose(os[p], xs[p - 1], rtol: 1e-3, atol: 1e-3).item(Bool.self),
                    "row \(p) must read row \(p - 1)")
            }
        }
        // The finish side (base_kernel[1] = 0, delta 0) is identically zero.
        let fin = conv.finish(x, coefficients: finishCoeff)
        XCTAssertEqual(abs(fin.asType(.float32)).max().item(Float.self), 0)

        // A per-token delta on tap 1 scales the shifted row group-wise: delta = 1 -> 2 * x[p-1].
        let delta = MLXArray.ones([N, 2, D / gs]).asType(.bfloat16)
        let out2 = conv.convolve(x, delta: delta, side: 0).asType(.float32)
        eval(out2)
        // coefficient tap0 = 0 + 1 = 1 (reads x[p]), tap1 = 1 + 1 = 2 (reads x[p-1] where allowed)
        for p in 0 ..< N {
            let expected = p % block == 0 ? xs[p] : xs[p] + 2 * xs[p - 1]
            XCTAssertTrue(allClose(out2[p], expected, rtol: 2e-2, atol: 2e-2).item(Bool.self), "row \(p)")
        }
    }

    // MARK: - Attention window / sinks

    func testAttentionMaskWindowAndBlockNonCausal() throws {
        // context rows at positions 4..<10 (n = 6), block at 10..<14, window 5:
        // key visible iff kpos > qpos - 5.
        let mask = NemotronLiLiCorrDrafter.attentionMask(contextRows: 6, position: 10, blockLength: 4, window: 5)
        XCTAssertEqual(mask.shape, [1, 1, 4, 10])
        XCTAssertEqual(mask.dtype, .bool)
        let m = mask.asArray(Bool.self)
        let kpos = Array(4 ..< 10) + Array(10 ..< 14)
        for q in 0 ..< 4 {
            let qpos = 10 + q
            for (j, k) in kpos.enumerated() {
                XCTAssertEqual(m[q * 10 + j], k > qpos - 5, "q=\(qpos) k=\(k)")
            }
        }
        // Inside the block every row sees every block position (non-causal), including later ones.
        for q in 0 ..< 4 { for j in 6 ..< 10 { XCTAssertTrue(m[q * 10 + j]) } }
        // No context: mask is all-true over the block.
        let m2 = NemotronLiLiCorrDrafter.attentionMask(contextRows: 0, position: 0, blockLength: 8, window: 16)
        XCTAssertEqual(m2.shape, [1, 1, 8, 8])
        XCTAssertTrue(m2.asArray(Bool.self).allSatisfy { $0 })
    }

    func testAttentionWindowDropsOldContextAndSinksAbsorbMass() throws {
        let c = Self.tinyConfiguration()
        let H = c.numAttentionHeads
        let hd = c.headDim
        let D = c.hiddenSize
        func attention(sinks: MLXArray) -> LiLiCorrDraftAttention {
            MLXRandom.seed(5)
            let mk = { (o: Int, i: Int) -> Module in
                Linear(weight: (MLXRandom.normal([o, i]) * 0.2).asType(.bfloat16), bias: nil)
            }
            return LiLiCorrDraftAttention(
                numHeads: H, numKVHeads: 1, headDim: hd, qProj: mk(H * hd, D), kProj: mk(hd, D),
                vProj: mk(hd, D), oProj: mk(D, H * hd),
                qNorm: LiLiCorrRMSNorm(weight: MLXArray.ones([hd]).asType(.bfloat16), eps: 1e-6),
                kNorm: LiLiCorrRMSNorm(weight: MLXArray.ones([hd]).asType(.bfloat16), eps: 1e-6),
                sinks: sinks, theta: 10000)
        }
        let attn = attention(sinks: MLXArray.zeros([H]).asType(.bfloat16))
        MLXRandom.seed(9)
        let ctx = (MLXRandom.normal([1, 12, D]) * 0.5).asType(.bfloat16)
        let block = (MLXRandom.normal([1, 4, D]) * 0.5).asType(.bfloat16)
        let (k, v) = attn.contextKV(ctx, position: 0)
        XCTAssertEqual(k.shape, [1, 1, 12, hd])

        // Window 6: block at 12..<16 sees context positions > qpos - 6, i.e. at least 7..<12.
        // Replacing the masked-out context rows (0..<6) with garbage must not change the output.
        let mask = NemotronLiLiCorrDrafter.attentionMask(contextRows: 12, position: 12, blockLength: 4, window: 6)
        let out = attn(block, contextK: k, contextV: v, position: 12, mask: mask)
        var kGarbage = k
        var vGarbage = v
        kGarbage[0..., 0..., 0 ..< 6] = (MLXRandom.normal([1, 1, 6, hd]) * 10).asType(.bfloat16)
        vGarbage[0..., 0..., 0 ..< 6] = (MLXRandom.normal([1, 1, 6, hd]) * 10).asType(.bfloat16)
        let out2 = attn(block, contextK: kGarbage, contextV: vGarbage, position: 12, mask: mask)
        eval(out, out2)
        XCTAssertTrue(allClose(out.asType(.float32), out2.asType(.float32), rtol: 1e-3, atol: 1e-3).item(Bool.self))
        // ...while garbage inside the window does change it.
        var kIn = k
        kIn[0..., 0..., 8 ..< 12] = (MLXRandom.normal([1, 1, 4, hd]) * 10).asType(.bfloat16)
        let out3 = attn(block, contextK: kIn, contextV: v, position: 12, mask: mask)
        eval(out3)
        XCTAssertFalse(allClose(out.asType(.float32), out3.asType(.float32), rtol: 1e-3, atol: 1e-3).item(Bool.self))

        // Sinks: a huge per-head sink bias takes all the softmax mass -> attention output ~ 0.
        let sunk = attention(sinks: MLXArray([Float](repeating: 60, count: H)).asType(.bfloat16))
        let outSunk = sunk(block, contextK: k, contextV: v, position: 12, mask: mask)
        eval(outSunk)
        XCTAssertLessThan(abs(outSunk.asType(.float32)).max().item(Float.self), 1e-2)
        // A very negative sink is a no-op relative to the sink-less reference.
        let unsunk = attention(sinks: MLXArray([Float](repeating: -60, count: H)).asType(.bfloat16))
        let outUnsunk = unsunk(block, contextK: k, contextV: v, position: 12, mask: mask)
        eval(outUnsunk)
        XCTAssertFalse(allClose(outUnsunk.asType(.float32), outSunk.asType(.float32), rtol: 1e-2, atol: 1e-2).item(Bool.self))
    }

    // MARK: - Lattice walk

    func testGreedyWalkMatchesHostReference() throws {
        for seed: UInt64 in 1 ... 5 {
            MLXRandom.seed(seed)
            let S = 7
            let K = 8
            let start = MLXRandom.normal([K])
            let pair = MLXRandom.normal([S - 1, K, K])
            let cand = MLXRandom.randInt(0 ..< 131072, [S, K]).asType(.int32)
            let path = LiLiCorrLattice.greedyWalk(start: start, pair: pair, candidates: cand)
            eval(path)
            XCTAssertEqual(path.shape, [S])
            let pairHost = pair.asArray(Float.self)
            var pairNested: [[[Float]]] = []
            for s in 0 ..< S - 1 {
                var rows: [[Float]] = []
                for i in 0 ..< K {
                    rows.append(Array(pairHost[((s * K) + i) * K ..< ((s * K) + i + 1) * K]))
                }
                pairNested.append(rows)
            }
            let candHost = cand.asArray(Int32.self)
            let candNested = (0 ..< S).map { Array(candHost[$0 * K ..< ($0 + 1) * K]) }
            let ref = LiLiCorrLattice.greedyWalkHost(
                start: start.asArray(Float.self), pair: pairNested, candidates: candNested)
            XCTAssertEqual(path.asArray(Int32.self), ref, "seed \(seed)")
        }
    }

    func testTopKCandidatesAreSortedDescending() throws {
        MLXRandom.seed(21)
        let logits = MLXRandom.normal([7, 128]) * 3
        let logp = logits - logits.logSumExp(axis: -1, keepDims: true)
        let (ids, lp) = LiLiCorrLattice.topKCandidates(logProbs: logp, k: 4)
        eval(ids, lp)
        XCTAssertEqual(ids.shape, [7, 4])
        let full = argSort(-logp, axis: -1)[0..., ..<4]
        XCTAssertEqual(ids.asType(.int32).asArray(Int32.self), full.asType(.int32).asArray(Int32.self))
        let lpHost = lp.asArray(Float.self)
        for s in 0 ..< 7 {
            for i in 1 ..< 4 { XCTAssertGreaterThanOrEqual(lpHost[s * 4 + i - 1], lpHost[s * 4 + i]) }
        }
        XCTAssertTrue(allClose(lp, takeAlong(logp, full, axis: -1)).item(Bool.self))
    }

    // MARK: - NVFP4 linear

    func testNVFP4LinearMatchesDequantizedReference() throws {
        MLXRandom.seed(31)
        for (out, inp) in [(64, 128), (96, 64), (32, 32)] {
            let full = (MLXRandom.normal([out, inp]) * 0.3).asType(.bfloat16)
            let (wq, scales, biases) = quantized(full, groupSize: 16, bits: 4, mode: .nvfp4)
            XCTAssertNil(biases)
            let s2 = MLXArray([Float(0.6)])
            let bias = (MLXRandom.normal([out]) * 0.1).asType(.bfloat16)
            let lin = NVFP4Linear(weight: wq, scales: scales, weightScale2: s2, bias: bias)
            XCTAssertEqual(lin.inputDimensions, inp)
            XCTAssertEqual(lin.outputDimensions, out)
            let x = (MLXRandom.normal([5, inp])).asType(.bfloat16)
            let y = lin(x)
            let deq = dequantized(wq, scales: scales, biases: nil, groupSize: 16, bits: 4, mode: .nvfp4)
            let ref = (matmul(x.asType(.float32), deq.asType(.float32).T) * 0.6 + bias.asType(.float32))
            eval(y, ref)
            XCTAssertEqual(y.dtype, .bfloat16)
            XCTAssertEqual(y.shape, [5, out])
            // bf16 kernel output vs fp32 dequantized reference: bf16 rounding of the result only.
            let diff = abs(y.asType(.float32) - ref)
            let tol = abs(ref).max().item(Float.self) * 1e-2 + 1e-2
            XCTAssertLessThan(diff.max().item(Float.self), tol, "shape \(out)x\(inp)")
            // The global scale is applied AFTER the matmul: scaling it rescales the pre-bias output.
            let lin2 = NVFP4Linear(weight: wq, scales: scales, weightScale2: MLXArray([Float(1.2)]), bias: nil)
            let lin1 = NVFP4Linear(weight: wq, scales: scales, weightScale2: MLXArray([Float(0.6)]), bias: nil)
            let r = (lin2(x).asType(.float32) / lin1(x).asType(.float32))
            eval(r)
            let finite = which(abs(lin1(x).asType(.float32)) .> 1e-2, r, MLXArray(Float(2)))
            XCTAssertTrue(allClose(finite, MLXArray(Float(2)), rtol: 2e-2, atol: 2e-2).item(Bool.self))
        }
    }

    func testLoaderRejectsMalformedQuantization() throws {
        let c = Self.tinyConfiguration()
        var w = Self.tinyWeights(c)
        w.removeValue(forKey: "fc.weight_scale_2")
        XCTAssertThrowsError(try NemotronLiLiCorrDrafter(configuration: c, weights: w)) { error in
            guard case NemotronLiLiCorrError.missingTensor(let name) = error else {
                return XCTFail("unexpected error \(error)")
            }
            XCTAssertEqual(name, "fc.weight_scale_2")
        }
        var c2 = c
        c2.quantization["fc"] = .init(mode: "affine", groupSize: 64, bits: 4, globalScale: nil)
        XCTAssertThrowsError(try NemotronLiLiCorrDrafter(configuration: c2, weights: Self.tinyWeights(c))) { error in
            guard case NemotronLiLiCorrError.unsupportedQuantization = error else {
                return XCTFail("unexpected error \(error)")
            }
        }
    }

    // MARK: - Truncate / rollback

    func testTruncateMatchesFreshDrafterOnSameContext() throws {
        let target = Target(vocab: 128, embedDim: Self.targetHidden, hidden: 64)
        let a = try Self.makeDrafter(window: 16, seed: 7)
        let b = try Self.makeDrafter(window: 16, seed: 7)
        let taps = Self.randomTaps(16, dim: a.tapDimension, seed: 41)

        // A: 12 rows, then 4 more (a speculative block the verifier later rejects past position 10).
        try a.addContext(taps: taps[0 ..< 12], position: 0)
        try a.addContext(taps: taps[12 ..< 16], position: 12)
        XCTAssertEqual(a.contextEnd, 16)
        a.truncate(to: 10)
        XCTAssertEqual(a.contextEnd, 10)
        XCTAssertEqual(a.contextRows, 10)
        XCTAssertEqual(a.contextK[0]?.shape, [1, 1, 10, 16])

        // B: fresh, given exactly the committed context.
        try b.addContext(taps: taps[0 ..< 10], position: 0)

        let ha = try a.draftHidden(blockEmbeddings: Self.blockEmbeddings(target, anchor: 17, drafter: a), position: 10)
        let hb = try b.draftHidden(blockEmbeddings: Self.blockEmbeddings(target, anchor: 17, drafter: b), position: 10)
        eval(ha, hb)
        XCTAssertTrue(allClose(ha.asType(.float32), hb.asType(.float32), rtol: 1e-2, atol: 1e-2).item(Bool.self))
        XCTAssertTrue(allClose(a.lastContext!.asType(.float32), b.lastContext!.asType(.float32), rtol: 1e-3, atol: 1e-3).item(Bool.self))
        let (pa, ta) = try a.propose(hidden: ha, lmHead: target.head, embed: target.embed)
        let (pb, tb) = try b.proposeTokens(hidden: hb, lmHead: target.head, embed: target.embed)
        eval(pa!, ta)
        XCTAssertEqual(ta.asArray(Int32.self), tb)
        XCTAssertEqual(pa!.asArray(Int32.self), pb!)

        // Both continue identically after the committed position.
        try a.addContext(taps: taps[10 ..< 11], position: 10)
        try b.addContext(taps: taps[10 ..< 11], position: 10)
        XCTAssertEqual(a.contextEnd, 11)
        let ha2 = try a.draftHidden(blockEmbeddings: Self.blockEmbeddings(target, anchor: 9, drafter: a), position: 11)
        let hb2 = try b.draftHidden(blockEmbeddings: Self.blockEmbeddings(target, anchor: 9, drafter: b), position: 11)
        eval(ha2, hb2)
        XCTAssertTrue(allClose(ha2.asType(.float32), hb2.asType(.float32), rtol: 1e-2, atol: 1e-2).item(Bool.self))

        // Truncating below the kept window empties the context at that position; the lane then
        // re-adds taps from its recorder and the position contract holds.
        let d = try Self.makeDrafter(window: 8, seed: 7)
        try d.addContext(taps: taps[0 ..< 16], position: 0)
        XCTAssertEqual(d.contextRows, 8)
        d.truncate(to: 6)
        XCTAssertEqual(d.contextEnd, 6)
        XCTAssertEqual(d.contextRows, 0)
        XCTAssertNil(d.lastContext)
        let hd = try d.draftHidden(blockEmbeddings: Self.blockEmbeddings(target, anchor: 1, drafter: d), position: 6)
        XCTAssertThrowsError(try d.propose(hidden: hd, lmHead: target.head, embed: target.embed)) { error in
            guard case NemotronLiLiCorrError.noAnchorContext = error else { return XCTFail("\(error)") }
        }
        try d.addContext(taps: taps[0 ..< 6], position: 6)  // recorder rows for positions 6..<12 (synthetic)
        XCTAssertEqual(d.contextEnd, 12)
        // truncate to a position at/after the end is a no-op
        d.truncate(to: 12)
        d.truncate(to: 20)
        XCTAssertEqual(d.contextEnd, 12)
        XCTAssertEqual(d.contextRows, 6)
    }

    // MARK: - Config parsing

    func testConfigurationParsesRealLayout() throws {
        let json: [String: Any] = [
            "block_size": 8, "hidden_size": 4096, "num_hidden_layers": 5, "num_attention_heads": 32,
            "num_key_value_heads": 2, "head_dim": 128, "intermediate_size": 12288,
            "rms_norm_eps": 1e-6, "vocab_size": 131072, "rope_theta": 10000.0, "sliding_window": 1024,
            "dflash_config": [
                "mask_token_id": 990, "target_layer_ids": [1, 18, 35, 51, 68, 85],
                "swa_window_size": 1024, "lilicorr_enabled": true, "lilicorr_candidate_topk": 8,
                "lilicorr_hidden_size": 1024, "lilicorr_num_layers": 2, "lilicorr_num_heads": 8,
                "lilicorr_vector_eps": 0.0001, "lilicorr_logit_scale": 8.0, "conv_kernel_size": 2,
                "conv_group_size": 16,
            ],
            "quantization": [
                "mode": "nvfp4", "group_size": 16, "bits": 4,
                "fc": ["mode": "nvfp4", "group_size": 16, "bits": 4, "global_scale": "fc.weight_scale_2"],
            ],
        ]
        let c = try NemotronLiLiCorrConfiguration(json: json)
        XCTAssertEqual(c.window, 1024)
        XCTAssertEqual(c.targetLayerIds, [1, 18, 35, 51, 68, 85])
        XCTAssertEqual(c.maskTokenId, 990)
        XCTAssertEqual(c.quantization["fc"]?.globalScale, "fc.weight_scale_2")
        XCTAssertEqual(c.defaultQuantization?.mode, "nvfp4")
        XCTAssertEqual(c.defaultQuantization?.groupSize, 16)
        XCTAssertEqual(c.lilicorrLogitScale, 8)
        XCTAssertEqual(c.convKernelSize, 2)
        XCTAssertThrowsError(try NemotronLiLiCorrConfiguration(json: ["hidden_size": 4]))
    }

    // MARK: - Real bundle (gated; never runs in CI or without the bundle)

    func testRealDrafterLoadsFromBundle() throws {
        guard let bundle = ProcessInfo.processInfo.environment["VMLX_BOLT_BUNDLE"] else {
            throw XCTSkip("VMLX_BOLT_BUNDLE not set")
        }
        let drafter = try NemotronLiLiCorrDrafter.load(
            directory: URL(fileURLWithPath: bundle).appendingPathComponent("lilicorr"))
        XCTAssertEqual(drafter.blockSize, 8)
        XCTAssertEqual(drafter.tapDimension, 6 * 4096)
        XCTAssertEqual(drafter.lilicorr?.topK, 8)
        XCTAssertEqual(drafter.lilicorr?.attnBias.shape, [1, 8, 56, 56])
    }
}
