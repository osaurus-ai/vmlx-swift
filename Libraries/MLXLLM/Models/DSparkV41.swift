// DeepSeek-V4.1 dSpark (MTP) draft head — Swift port of jang_tools/dsv41/dspark.py.
//
// Three MTP stages appended after the 40-layer backbone. Each stage mirrors a backbone Block
// (latent MLA attention over a 128-token window + a 128-expert top-3 MoE + hyper-connections),
// but its attention window is SEEDED from the backbone's hidden state (mean-over-hc of the target
// layers 37/38/39), not from its own tokens. Stage 0 projects the concatenated target-layer hidden
// down to one hidden (main_proj/main_norm); the last stage carries a final norm, a Markov bias head
// and a confidence head.
//
// This file is the drafter MODEL + loader only. The accept/reject speculative loop, verify rollback,
// adaptive depth and AR-safety are the existing SpecDec/NativeMTP framework (see DSparkV41Lane +
// DeepseekV4Model's NativeMTPBlockDraftModel conformance). Verification is exact, so draft precision
// only moves the acceptance rate (speed), never the output.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Config

struct DSparkV41Config {
    let hiddenSize: Int
    let nHeads: Int
    let headDim: Int
    let qkRopeHeadDim: Int
    let qLoraRank: Int
    let oLoraRank: Int
    let oGroups: Int
    let slidingWindow: Int
    let ropeTheta: Float
    let rmsNormEps: Float
    let hcMult: Int
    let hcSinkhornIters: Int
    let hcEps: Float
    let swigluLimit: Float
    let routedScalingFactor: Float
    let normTopkProb: Bool
    // dSpark head
    let blockSize: Int
    let noiseTokenId: Int
    let targetLayerIds: [Int]
    let nStages: Int
    let markovRank: Int
    let nRoutedExperts: Int
    let numExpertsPerTok: Int
    let moeInterDim: Int
    let vocabSize: Int

    /// Built from the backbone config plus the dSpark-head hyperparameters (parsed from the bundle
    /// config.json by the loader; defaults are the DSV4.1-Flash-1017 values).
    init(_ c: DeepseekV4Configuration,
         blockSize: Int = 5, noiseTokenId: Int = 128799, targetLayerIds: [Int] = [37, 38, 39],
         nStages: Int = 3, markovRank: Int = 256, nRoutedExperts: Int = 128,
         numExpertsPerTok: Int = 3, moeInterDim: Int? = nil) {
        self.hiddenSize = c.hiddenSize
        self.nHeads = c.numAttentionHeads
        self.headDim = c.headDim
        self.qkRopeHeadDim = c.qkRopeHeadDim
        self.qLoraRank = c.qLoraRank
        self.oLoraRank = c.oLoraRank
        self.oGroups = c.oGroups
        self.slidingWindow = c.slidingWindow
        self.ropeTheta = c.ropeTheta
        self.rmsNormEps = c.rmsNormEps
        self.hcMult = c.hcMult
        self.hcSinkhornIters = c.hcSinkhornIters
        self.hcEps = c.hcEps
        self.swigluLimit = c.swigluLimit
        self.routedScalingFactor = c.routedScalingFactor
        self.normTopkProb = c.normTopkProb
        self.blockSize = blockSize
        self.noiseTokenId = noiseTokenId
        self.targetLayerIds = targetLayerIds
        self.nStages = nStages
        self.markovRank = markovRank
        self.nRoutedExperts = nRoutedExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.moeInterDim = moeInterDim ?? c.moeIntermediateSize
        self.vocabSize = c.vocabSize
    }
}

// MARK: - Draft attention (latent MLA, main-hidden-seeded window)

final class DSparkV41Attention: Module {
    let nHeads: Int, headDim: Int, rd: Int, nGroups: Int, oLoraRank: Int, win: Int
    let scale: Float
    @ModuleInfo(key: "wq_a") var wqA: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
    @ModuleInfo(key: "wo_b") var woB: Linear
    @ParameterInfo(key: "wo_a") var woA: MLXArray      // [groups, o_lora_rank, (H/groups)*headDim]
    @ParameterInfo(key: "attn_sink") var attnSink: MLXArray
    let ropeTheta: Float
    // committed KV window (rope'd), [1, N, headDim], oldest first
    var window: MLXArray?

    init(_ cfg: DSparkV41Config) {
        self.nHeads = cfg.nHeads
        self.headDim = cfg.headDim
        self.rd = cfg.qkRopeHeadDim
        self.nGroups = cfg.oGroups
        self.oLoraRank = cfg.oLoraRank
        self.win = cfg.slidingWindow
        self.scale = pow(Float(cfg.headDim), -0.5)
        self._wqA.wrappedValue = Linear(cfg.hiddenSize, cfg.qLoraRank, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: cfg.qLoraRank, eps: cfg.rmsNormEps)
        self._wqB.wrappedValue = Linear(cfg.qLoraRank, cfg.nHeads * cfg.headDim, bias: false)
        self._wkv.wrappedValue = Linear(cfg.hiddenSize, cfg.headDim, bias: false)
        self._kvNorm.wrappedValue = RMSNorm(dimensions: cfg.headDim, eps: cfg.rmsNormEps)
        self._woB.wrappedValue = Linear(cfg.oGroups * cfg.oLoraRank, cfg.hiddenSize, bias: false)
        self._woA.wrappedValue = zeros([cfg.oGroups, cfg.oLoraRank, cfg.nHeads * cfg.headDim / cfg.oGroups])
        self._attnSink.wrappedValue = zeros([cfg.nHeads])
        self.ropeTheta = cfg.ropeTheta
        super.init()
    }

    /// Append committed positions' main-derived KV to the window (rope'd).
    func seed(mainX: MLXArray, positions: MLXArray) {
        var kv = kvNorm(wkv(mainX))                               // [1, S, headDim]
        kv = ropeApply(kv, positions: positions, inverse: false)  // [1, S, headDim]
        kv = kv.asType(.bfloat16)
        if let w = window {
            window = concatenated([w, kv], axis: 1)
        } else {
            window = kv
        }
    }

    func resetWindow() { window = nil }

    /// Draft block attention: queries x [1, bs, d] attend to (last-`win` window ++ own KV).
    func attend(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let bs = x.dim(1)
        let qr = qNorm(wqA(x))
        var q = wqB(qr).reshaped(1, bs, nHeads, headDim)
        q = ropeApply(q, positions: positions, inverse: false)    // [1,bs,H,headDim]
        var kv = kvNorm(wkv(x))                                    // [1,bs,headDim]
        kv = ropeApply(kv, positions: positions, inverse: false)
        var full = kv
        if var w = window {
            if w.dim(1) > win { w = w[0..., (w.dim(1) - win)...] }
            full = concatenated([w, kv], axis: 1)
        }
        let L = full.dim(1)
        // q: [1,H,bs,headDim]; k/v: [1,H,L,headDim] (single latent KV broadcast to H heads).
        let qh = q.transposed(0, 2, 1, 3)
        let kh = full.reshaped(1, 1, L, headDim).broadcast(to: [1, nHeads, L, headDim])
        // every query sees all keys (non-causal within the speculative block) → mask nil.
        var o = MLXFast.scaledDotProductAttention(
            queries: qh, keys: kh, values: kh, scale: scale, mask: nil,
            sinks: attnSink.asType(qh.dtype))                     // [1,H,bs,headDim]
        o = o.transposed(0, 2, 1, 3)                              // [1,bs,H,headDim]
        o = ropeApply(o, positions: positions, inverse: true)
        o = o.asType(.bfloat16).reshaped(bs, nGroups, -1)         // [bs, groups, (H/groups)*headDim]
        // o[bs,g,d] × woA[g,r,d] -> [bs,g,r]
        let ol = MLX.einsum("sgd,grd->sgr", o, woA)
        return woB(ol.reshaped(1, bs, -1))                        // [1,bs,d]
    }

    private func ropeApply(_ x: MLXArray, positions: MLXArray, inverse: Bool) -> MLXArray {
        // Interleaved-pair rotary on the LAST `rd` features (MLA partial rope), matching
        // jang_tools/dsv41/model.py Rope. x is [...,(H,)headDim].
        return DSparkRope.apply(x, positions: positions, invFreqDim: rd, theta: ropeTheta, inverse: inverse, ndim: x.ndim)
    }
}

/// Standalone interleaved-pair partial rope matching the Python reference exactly.
enum DSparkRope {
    static func apply(_ x: MLXArray, positions: MLXArray, invFreqDim dim: Int, theta: Float, inverse: Bool, ndim: Int) -> MLXArray {
        let half = dim / 2
        let idx = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) }) / Float(dim)
        let invFreq = MLX.pow(MLXArray(theta), -idx)             // [half]
        let pos = positions.asType(.float32).reshaped(-1, 1)     // [L,1]
        var ang = pos * invFreq.reshaped(1, half)                // [L, half]
        var cos = MLX.cos(ang), sin = MLX.sin(ang)
        if inverse { sin = -sin }
        if ndim == 4 { cos = cos.reshaped(-1, 1, half); sin = sin.reshaped(-1, 1, half) }
        // tail = last `dim`; split into pairs
        let headPart = x[.ellipsis, 0 ..< (x.dim(-1) - dim)]
        let tail = x[.ellipsis, (x.dim(-1) - dim)...]
        var tshape = tail.shape; tshape[tshape.count - 1] = half; tshape.append(2)
        let t = tail.asType(.float32).reshaped(tshape)
        let a = t[.ellipsis, 0], b = t[.ellipsis, 1]
        let rot = MLX.stacked([a * cos - b * sin, a * sin + b * cos], axis: -1).reshaped(tail.shape)
        _ = ang
        return concatenated([headPart, rot.asType(x.dtype)], axis: -1)
    }
}

// MARK: - Draft MoE (128-expert top-3, 4-bit routed + shared, sqrtsoftplus gate)

final class DSparkV41MoE: Module {
    let topk: Int, routeScale: Float, normTopk: Bool, lim: Float
    @ParameterInfo(key: "gate_w") var gateW: MLXArray      // [nExperts, d] bf16
    @ParameterInfo(key: "gate_bias") var gateBias: MLXArray
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "sw1") var sw1: Linear
    @ModuleInfo(key: "sw3") var sw3: Linear
    @ModuleInfo(key: "sw2") var sw2: Linear

    init(_ cfg: DSparkV41Config) {
        self.topk = cfg.numExpertsPerTok
        self.routeScale = cfg.routedScalingFactor
        self.normTopk = cfg.normTopkProb
        self.lim = cfg.swigluLimit
        let d = cfg.hiddenSize, inter = cfg.moeInterDim
        self._gateW.wrappedValue = zeros([cfg.nRoutedExperts, d])
        self._gateBias.wrappedValue = zeros([cfg.nRoutedExperts])
        let limit = cfg.swigluLimit
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: d, hiddenDims: inter, numExperts: cfg.nRoutedExperts,
            glue: { g, u in DeepseekV4Math.dsv4SwiGLU(gate: g, up: u, limit: limit) })
        self._sw1.wrappedValue = Linear(d, inter, bias: false)
        self._sw3.wrappedValue = Linear(d, inter, bias: false)
        self._sw2.wrappedValue = Linear(inter, d, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), S = x.dim(1), D = x.dim(2)
        let xf = x.reshaped(-1, D)
        let scores = MLX.sqrt(MLXNN.softPlus(matmul(xf.asType(.float32), gateW.asType(.float32).transposed())))
        let idx = MLX.argPartition(-(scores + gateBias), kth: topk - 1, axis: -1)[0..., 0 ..< topk]
        var w = takeAlong(scores, idx, axis: -1)
        if normTopk && topk > 1 { w = w / (w.sum(axis: -1, keepDims: true) + 1e-20) }
        w = w * routeScale
        // routed experts via SwitchGLU (expects scores for the pre-down scale)
        let routed = switchMLP(xf.reshaped(B, S, D), idx.reshaped(B, S, topk), preDownScores: w.reshaped(B, S, topk))
        var y = DeepseekV4Math.reduceRoutedExpertsFP32(routed).asType(.float32).reshaped(-1, D)
        // shared expert
        var sg = sw1(xf).asType(.float32)
        var su = sw3(xf).asType(.float32)
        if lim > 0 { su = clip(su, min: -lim, max: lim); sg = MLX.minimum(sg, MLXArray(lim)) }
        y = y + sw2((MLXNN.silu(sg) * su).asType(.bfloat16)).asType(.float32)
        return y.asType(x.dtype).reshaped(B, S, D)
    }
}

// MARK: - One dSpark stage

final class DSparkV41Stage: Module {
    let cfg: DSparkV41Config
    let isFirst: Bool, isLast: Bool
    @ModuleInfo(key: "attn") var attn: DSparkV41Attention
    @ModuleInfo(key: "moe") var moe: DSparkV41MoE
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ParameterInfo(key: "hc_attn_fn") var hcAttnFn: MLXArray
    @ParameterInfo(key: "hc_attn_base") var hcAttnBase: MLXArray
    @ParameterInfo(key: "hc_attn_scale") var hcAttnScale: MLXArray
    @ParameterInfo(key: "hc_ffn_fn") var hcFfnFn: MLXArray
    @ParameterInfo(key: "hc_ffn_base") var hcFfnBase: MLXArray
    @ParameterInfo(key: "hc_ffn_scale") var hcFfnScale: MLXArray
    // first stage
    var mainProj: Linear?
    var mainNorm: RMSNorm?
    // last stage
    var norm: RMSNorm?
    @ParameterInfo(key: "markov_embed") var markovEmbed: MLXArray
    @ParameterInfo(key: "markov_head") var markovHead: MLXArray
    @ParameterInfo(key: "conf_proj") var confProj: MLXArray

    init(_ cfg: DSparkV41Config, isFirst: Bool, isLast: Bool) {
        self.cfg = cfg
        self.isFirst = isFirst
        self.isLast = isLast
        let d = cfg.hiddenSize, mix = (2 + cfg.hcMult) * cfg.hcMult
        self._attn.wrappedValue = DSparkV41Attention(cfg)
        self._moe.wrappedValue = DSparkV41MoE(cfg)
        self._attnNorm.wrappedValue = RMSNorm(dimensions: d, eps: cfg.rmsNormEps)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: d, eps: cfg.rmsNormEps)
        self._hcAttnFn.wrappedValue = zeros([mix, cfg.hcMult * d])
        self._hcAttnBase.wrappedValue = zeros([mix])
        self._hcAttnScale.wrappedValue = zeros([3])
        self._hcFfnFn.wrappedValue = zeros([mix, cfg.hcMult * d])
        self._hcFfnBase.wrappedValue = zeros([mix])
        self._hcFfnScale.wrappedValue = zeros([3])
        self._markovEmbed.wrappedValue = zeros([1])
        self._markovHead.wrappedValue = zeros([1])
        self._confProj.wrappedValue = zeros([1])
        if isFirst {
            self.mainProj = Linear(d * cfg.targetLayerIds.count, d, bias: false)
            self.mainNorm = RMSNorm(dimensions: d, eps: cfg.rmsNormEps)
        }
        if isLast {
            self.norm = RMSNorm(dimensions: d, eps: cfg.rmsNormEps)
        }
        super.init()
    }

    private func mixes(_ h: MLXArray, which: Int) -> (pre: MLXArray, post: MLXArray, comb: MLXArray) {
        let fn = which == 0 ? hcAttnFn : hcFfnFn
        let base = which == 0 ? hcAttnBase : hcFfnBase
        let scale = which == 0 ? hcAttnScale : hcFfnScale
        return DeepseekV4Math.hcMixes(h, fn: fn, scale: scale, base: base,
            hcMult: cfg.hcMult, hiddenSize: cfg.hiddenSize, iters: cfg.hcSinkhornIters,
            eps: cfg.hcEps, normEps: cfg.rmsNormEps)
    }

    /// x [1,bs,hc,d]; preMix [1,bs,hc]. Window must already hold committed KV.
    func forward(_ x: MLXArray, preMix: MLXArray, positions: MLXArray) -> (MLXArray, MLXArray) {
        var residual = x
        let (aPre, aPost, aComb) = mixes(x, which: 0)
        var h = attn.attend(attnNorm(DeepseekV4Math.hcCollapse(x, pre: preMix)), positions: positions)
        var xx = DeepseekV4Math.hcPost(h, residual: residual, post: aPost, comb: aComb)
        residual = xx
        let (fPre, fPost, fComb) = mixes(xx, which: 1)
        h = moe(ffnNorm(DeepseekV4Math.hcCollapse(xx, pre: aPre)))
        xx = DeepseekV4Math.hcPost(h, residual: residual, post: fPost, comb: fComb)
        return (xx, fPre)
    }
}

// MARK: - dSpark drafter

final class DSparkV41Drafter: Module {
    let cfg: DSparkV41Config
    var stages: [DSparkV41Stage]
    let embed: (MLXArray) -> MLXArray      // backbone token embedding
    let head: (MLXArray) -> MLXArray       // backbone lm_head

    init(_ cfg: DSparkV41Config, embed: @escaping (MLXArray) -> MLXArray, head: @escaping (MLXArray) -> MLXArray) {
        self.cfg = cfg
        self.embed = embed
        self.head = head
        self.stages = (0 ..< cfg.nStages).map {
            DSparkV41Stage(cfg, isFirst: $0 == 0, isLast: $0 == cfg.nStages - 1)
        }
        super.init()
    }

    func resetWindows() { for s in stages { s.attn.resetWindow() } }

    /// Seed every stage window with committed positions' KV. mainHidden [1,S,d*nTargets].
    func seedWindows(mainHidden: MLXArray, positions: MLXArray) {
        guard let mp = stages[0].mainProj, let mn = stages[0].mainNorm else { return }
        let mainX = mn(mp(mainHidden))         // [1,S,d]
        for s in stages { s.attn.seed(mainX: mainX, positions: positions) }
        eval(stages.map { $0.attn.window ?? MLXArray(0) })
    }

    func windowCheckpoint() -> [MLXArray?] { stages.map { $0.attn.window } }
    func windowRestore(_ ck: [MLXArray?]) { for (s, w) in zip(stages, ck) { s.attn.window = w } }

    /// Draft from `seedTok` (the backbone's token for position startPos+1). Windows hold committed
    /// KV for [0..startPos]. Returns output ids [bs+1] = [seedTok, draft@+2, ..., draft@+bs+1].
    func draft(seedTok: Int, startPos: Int) -> [Int] {
        let bs = cfg.blockSize
        var ids = [Int32](repeating: Int32(cfg.noiseTokenId), count: bs)
        ids[0] = Int32(seedTok)
        var x = embed(MLXArray(ids).reshaped(1, bs))                 // [1,bs,d]
        x = x.expandedDimensions(axis: 2)
        x = MLX.broadcast(x, to: [1, bs, cfg.hcMult, cfg.hiddenSize])  // [1,bs,hc,d]
        var preMix = zeros([1, bs, cfg.hcMult]).asType(.float32)
        preMix[0..., 0..., 0] = MLXArray(Float(1.0))
        let positions = MLXArray(Array(stride(from: Int32(startPos + 1), to: Int32(startPos + 1 + bs), by: 1)))
        for s in stages { let (nx, np) = s.forward(x, preMix: preMix, positions: positions); x = nx; preMix = np }
        let last = stages[stages.count - 1]
        let xc = DeepseekV4Math.hcCollapse(x, pre: preMix)           // [1,bs,d]
        let logits = head(last.norm!(xc)).asType(.float32)[0]        // [bs, vocab]
        var out = [Int](); out.append(seedTok)
        for i in 0 ..< bs {
            let me = last.markovEmbed[out[i]].asType(.float32)       // [rank]
            let bias = matmul(me.reshaped(1, -1), last.markovHead.asType(.float32).transposed()).reshaped(-1) // [vocab]
            out.append(Int(MLX.argMax(logits[i] + bias).item(Int32.self)))
        }
        return out
    }
}
