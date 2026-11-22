// Copyright © 2026 Jinho Jang. All rights reserved.
//
// Inkling-2-Mini native drafter: DSpark (DFlash block drafter + `rnn_hidden` Markov head).
//
// Reference: SGLang PR #51926 — models/inkling2_mtp.py, models/dflash.py, models/dspark.py,
// kernels/ops/speculative/inkling2_mtp.py.
//
// One round, committed length P, anchor = the token at position P (not yet in the target KV):
//   draft   block [embed_norm(embed(anchor)), mask_embed x 7] at positions P..P+7 through 3
//           bidirectional Qwen3-style layers whose keys are [context KV of committed positions]
//           + the 8 block keys; h = norm(...)[1:8]; base = h @ W_U^T; a 7-step Markov walk adds
//           lm_head(hidden_out(latent)) (folded at load) and picks d1..d7.
//   verify  the target runs [anchor, d1..d7] causally; greedy accept n leading matches; bonus.
//   commit  target keeps rows 0..n; the draft context KV gets rows 0..n from the target's captured
//           hidden states (layer-visit inputs 5 / 23 / 41 + the final normed hidden); f_rnn from
//           row n.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private let f32 = DType.float32

final class Inkling2DSparkAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "k_proj_ctx") var kProjCtx: Linear
    @ModuleInfo(key: "v_proj_ctx") var vProjCtx: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "k_norm_ctx") var kNormCtx: RMSNorm
    let nq: Int
    let nkv: Int
    let hd: Int
    let rope: RoPE

    init(hidden: Int, nq: Int, nkv: Int, hd: Int, eps: Float, theta: Float) {
        self.nq = nq
        self.nkv = nkv
        self.hd = hd
        _qProj.wrappedValue = Linear(hidden, nq * hd, bias: false)
        _kProj.wrappedValue = Linear(hidden, nkv * hd, bias: false)
        _vProj.wrappedValue = Linear(hidden, nkv * hd, bias: false)
        _oProj.wrappedValue = Linear(nq * hd, hidden, bias: false)
        _kProjCtx.wrappedValue = Linear(hidden, nkv * hd, bias: false)
        _vProjCtx.wrappedValue = Linear(hidden, nkv * hd, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: hd, eps: eps)
        _kNorm.wrappedValue = RMSNorm(dimensions: hd, eps: eps)
        _kNormCtx.wrappedValue = RMSNorm(dimensions: hd, eps: eps)
        // neox (half-split) rotary over the full head dim.
        rope = RoPE(dimensions: hd, traditional: false, base: theta)
    }
}

final class Inkling2DSparkMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hidden: Int, inter: Int) {
        _gate.wrappedValue = Linear(hidden, inter, bias: false)
        _up.wrappedValue = Linear(hidden, inter, bias: false)
        _down.wrappedValue = Linear(inter, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class Inkling2DSparkLayer: Module {
    @ModuleInfo(key: "self_attn") var attn: Inkling2DSparkAttention
    @ModuleInfo(key: "mlp") var mlp: Inkling2DSparkMLP
    @ModuleInfo(key: "input_layernorm") var inputLN: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postLN: RMSNorm
    @ModuleInfo(key: "dflare_norm") var dflareNorm: RMSNorm
    let isLocal: Bool

    init(hidden: Int, inter: Int, nq: Int, nkv: Int, hd: Int, eps: Float, theta: Float, local: Bool)
    {
        _attn.wrappedValue = Inkling2DSparkAttention(
            hidden: hidden, nq: nq, nkv: nkv, hd: hd, eps: eps, theta: theta)
        _mlp.wrappedValue = Inkling2DSparkMLP(hidden: hidden, inter: inter)
        _inputLN.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        _postLN.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        _dflareNorm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        isLocal = local
    }
}

final class Inkling2DSparkMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var markovW1: Embedding
    @ModuleInfo(key: "rnn_proj") var rnnProj: Linear
    @ModuleInfo(key: "hidden_out") var hiddenOut: Linear
    @ModuleInfo(key: "dflare_norm") var dflareNorm: RMSNorm
    let rank: Int

    init(vocab: Int, rank: Int, hidden: Int, eps: Float) {
        _markovW1.wrappedValue = Embedding(embeddingCount: vocab, dimensions: rank)
        _rnnProj.wrappedValue = Linear(2 * rank + 2 * hidden, 3 * rank, bias: true)
        _hiddenOut.wrappedValue = Linear(rank, hidden, bias: true)
        _dflareNorm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        self.rank = rank
    }
}

/// Load-time derived tensors (outside the parameter tree).
final class Inkling2DSparkDerived: @unchecked Sendable {
    /// softmax(fuse_weight) over the 4 target features, fp32 math, stored BF16 like the reference.
    var fuseAlpha: MLXArray?
    /// `W_U @ hidden_out.weight` [V, rank] and `W_U @ hidden_out.bias` [V], BF16.
    var foldWeight: MLXArray?
    var foldBias: MLXArray?
}

public struct Inkling2DSparkConfig: Sendable {
    var hidden = 2304
    var layers = 3
    var nq = 24
    var nkv = 8
    var hd = 128
    var inter = 9216
    var eps: Float = 1e-6
    var theta: Float = 1_000_000
    var localLayers: [Int] = [0, 2]
    var window = 1024
    var blockSize = 8
    var markovRank = 256
    var vocab = 201024
    var unpaddedVocab = 200058
    var softcap: Float = 30
    var targetLayerIds: [Int] = [5, 23, 41, 48]
}

public final class Inkling2DSparkDrafter: Module {
    let cfg: Inkling2DSparkConfig
    let layers: [Inkling2DSparkLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "markov_head") var markovHead: Inkling2DSparkMarkovHead
    @ParameterInfo(key: "mask_embed") var maskEmbed: MLXArray
    @ParameterInfo(key: "fuse_weight") var fuseWeight: MLXArray
    let derived = Inkling2DSparkDerived()

    init(_ cfg: Inkling2DSparkConfig) {
        self.cfg = cfg
        layers = (0 ..< cfg.layers).map {
            Inkling2DSparkLayer(
                hidden: cfg.hidden, inter: cfg.inter, nq: cfg.nq, nkv: cfg.nkv, hd: cfg.hd,
                eps: cfg.eps, theta: cfg.theta, local: cfg.localLayers.contains($0))
        }
        _norm.wrappedValue = RMSNorm(dimensions: cfg.hidden, eps: cfg.eps)
        _markovHead.wrappedValue = Inkling2DSparkMarkovHead(
            vocab: cfg.vocab, rank: cfg.markovRank, hidden: cfg.hidden, eps: cfg.eps)
        _maskEmbed.wrappedValue = MLXArray.zeros([cfg.hidden])
        _fuseWeight.wrappedValue = MLXArray.zeros([cfg.layers + 1, cfg.targetLayerIds.count])
    }

    /// Build the folded Markov output head against the target's unembedding (once).
    /// `lmHeadWeight` is evaluated only on the first call (a quantized head dequantizes once, transiently).
    func prepare(lmHeadWeight: @autoclosure () -> MLXArray) {
        guard derived.foldWeight == nil else { return }
        derived.fuseAlpha = MLX.softmax(fuseWeight.asType(f32), axis: -1).asType(.bfloat16)
        let wu = lmHeadWeight().asType(f32)
        derived.foldWeight = MLX.matmul(wu, markovHead.hiddenOut.weight.asType(f32)).asType(
            .bfloat16)
        derived.foldBias = MLX.matmul(wu, markovHead.hiddenOut.bias!.asType(f32)).asType(.bfloat16)
        eval(derived.fuseAlpha!, derived.foldWeight!, derived.foldBias!)
    }

    /// `mix_rmsnorm`: `sum_t fp32(T[:, t]) * fp32(alpha[row, t])`, RMS-normalized, times `w`, BF16.
    /// `targets` [n, 4, H] BF16.
    func mix(_ targets: MLXArray, row: Int, weight: MLXArray) -> MLXArray {
        let alpha = derived.fuseAlpha![row].asType(f32).reshaped([1, -1, 1])
        let mixed = MLX.sum(targets.asType(f32) * alpha, axis: 1)  // [n, H]
        return MLXFast.rmsNorm(mixed, weight: weight.asType(f32), eps: cfg.eps).asType(.bfloat16)
    }

    /// Draft-side context KV for committed positions `pos0 ..< pos0 + n` from their captured target
    /// tensors `[n, 4, H]`, appended to each layer's context cache.
    func writeContext(_ targets: MLXArray, pos0: Int, caches: [KVCacheSimple]) {
        let n = targets.dim(0)
        for (i, layer) in layers.enumerated() {
            let a = layer.attn
            let c = mix(targets, row: i, weight: layer.dflareNorm.weight).reshaped([
                1, n, cfg.hidden,
            ])
            var k = a.kProjCtx(c).reshaped([1, n, a.nkv, a.hd])
            let v = a.vProjCtx(c).reshaped([1, n, a.nkv, a.hd]).transposed(0, 2, 1, 3)
            k = a.kNormCtx(k).transposed(0, 2, 1, 3)
            k = a.rope(k, offset: pos0)
            precondition(caches[i].offset == pos0, "dspark: context cache out of sync")
            _ = caches[i].update(keys: k, values: v)
        }
    }

    /// The Markov head's per-block feature from the last committed position's captures `[4, H]`.
    func fRnn(_ lastTargets: MLXArray) -> MLXArray {
        mix(
            lastTargets.reshaped([1, -1, cfg.hidden]), row: cfg.layers,
            weight: markovHead.dflareNorm.weight
        )
        .reshaped([cfg.hidden])
    }

    /// Draft hidden for one block at positions `P ..< P + B`: `[B - 1, H]` (anchor row dropped).
    /// `anchorEmbedding` = the TARGET's `embed_norm(embed(anchor))` `[H]`.
    func blockHidden(anchorEmbedding: MLXArray, P: Int, caches: [KVCacheSimple]) -> MLXArray {
        let B = cfg.blockSize
        let H = cfg.hidden
        let dtype = maskEmbed.dtype
        let x = MLX.concatenated(
            [
                anchorEmbedding.asType(dtype).reshaped([1, H]),
                MLX.broadcast(maskEmbed.reshaped([1, H]), to: [B - 1, H]),
            ], axis: 0
        ).reshaped([1, B, H])
        var res = x
        var mlpOut: MLXArray? = nil
        for (i, layer) in layers.enumerated() {
            if let m = mlpOut { res = (res.asType(f32) + m.asType(f32)).asType(dtype) }
            let a0 = layer.inputLN(res)
            let at = layer.attn
            var q = at.qNorm(at.qProj(a0).reshaped([1, B, at.nq, at.hd])).transposed(0, 2, 1, 3)
            var k = at.kNorm(at.kProj(a0).reshaped([1, B, at.nkv, at.hd])).transposed(0, 2, 1, 3)
            let v = at.vProj(a0).reshaped([1, B, at.nkv, at.hd]).transposed(0, 2, 1, 3)
            q = at.rope(q, offset: P)
            k = at.rope(k, offset: P)
            var keys = k
            var values = v
            let ctx = caches[i]
            if ctx.offset > 0, let st = ctx.state.first, ctx.state.count == 2 {
                let lo = layer.isLocal ? max(0, P - cfg.window) : 0
                let ck = st[0..., 0..., lo ..< P, 0...]
                let cv = ctx.state[1][0..., 0..., lo ..< P, 0...]
                keys = MLX.concatenated([ck, k], axis: 2)
                values = MLX.concatenated([cv, v], axis: 2)
            }
            let o = MLXFast.scaledDotProductAttention(
                queries: q, keys: keys, values: values, scale: 1 / Float(at.hd).squareRoot(),
                mask: .none
            ).transposed(0, 2, 1, 3).reshaped([1, B, at.nq * at.hd])
            res = (res.asType(f32) + at.oProj(o).asType(f32)).asType(dtype)
            mlpOut = layer.mlp(layer.postLN(res))
        }
        res = (res.asType(f32) + mlpOut!.asType(f32)).asType(dtype)
        return norm(res)[0, 1...]  // [B-1, H]
    }

    /// The Markov walk over `steps` (<= block-1) positions. `h` [B-1, H] post-norm draft hidden,
    /// `fRnn` [H]. Greedy (argmax) unless `noise` [>= steps, unpadded vocab] is given: then step i
    /// proposes argmax(v * inverseTemperature + noise[i]) (coupled sampled drafting).
    func walk(
        h: MLXArray, unembed: (MLXArray) -> MLXArray, anchor: Int, fRnn: MLXArray, steps: Int? = nil,
        noise: MLXArray? = nil, inverseTemperature: Float = 1
    ) -> [Int] {
        let mh = markovHead
        let R = cfg.markovRank
        let n = min(steps ?? (cfg.blockSize - 1), cfg.blockSize - 1)
        let base = unembed(h[0 ..< n])  // [n, V] bf16 (the target's head: plain or affine)
        var state = MLXArray.zeros([R], dtype: .bfloat16)
        // `prev` stays on the GPU: argmax -> embedding gather, one evaluation for the whole walk.
        var prev = MLXArray([Int32(anchor)])
        var toks: [MLXArray] = []
        let fw = derived.foldWeight!
        let fb = derived.foldBias!
        let w1 = mh.markovW1.weight
        for s in 0 ..< n {
            let emb = MLX.take(w1, prev, axis: 0).reshaped([R])
            let xin = MLX.concatenated(
                [state, emb.asType(.bfloat16), h[s].asType(.bfloat16), fRnn.asType(.bfloat16)],
                axis: 0)
            let p = mh.rnnProj(xin.reshaped([1, -1])).reshaped([3 * R]).asType(f32)
            func bf(_ a: MLXArray) -> MLXArray { a.asType(.bfloat16).asType(f32) }
            let gate = bf(MLX.sigmoid(p[0 ..< R]))
            let kept = bf(gate * state.asType(f32))
            let blend = bf(bf(1 - gate) * bf(MLX.tanh(p[R ..< (2 * R)])))
            state = (kept + blend).asType(.bfloat16)
            let latent = MLX.tanh(p[(2 * R)...]).asType(.bfloat16)
            let bias = (fb + MLX.matmul(fw, latent)).asType(.bfloat16)  // [V]
            var v = (base[s].asType(f32) + bias.asType(f32)).asType(.bfloat16).asType(f32)
            v = (MLX.tanh(v / cfg.softcap) * cfg.softcap).asType(.bfloat16).asType(f32)
            v = v[..<cfg.unpaddedVocab]
            if let noise { v = v * inverseTemperature + noise[s][..<cfg.unpaddedVocab].asType(f32) }
            prev = v.argMax(keepDims: true).asType(.int32).reshaped([1])
            toks.append(prev)
        }
        let out = MLX.concatenated(toks).asArray(Int32.self).map(Int.init)
        return out
    }
}
