// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 dSpark (MTP) draft head: port of `jang_tools/dsv41/dspark.py` (`DSpark`, `load_dspark`).
//
// Three MTP stages appended after the 40-layer backbone. Each stage mirrors a backbone block (latent
// attention over a 128-token window, a 128-expert top-3 MoE, hyper-connections), but its attention
// window is seeded from the backbone's hidden state: the mean-over-hc hidden at the INPUT of target
// layers 37/38/39, concatenated. Stage 0 projects that down to one hidden (main_proj/main_norm); the
// last stage carries a final norm, a Markov bias head and a confidence head. Embedding and output
// head are the backbone's.
//
// The loop that uses the drafts is `MLXLMCommon/SpecDec/DSpark.swift`. Verification is exact, so the
// draft's precision only moves acceptance (speed), never the output.
//
// Per-request state (the per-stage windows) lives in `DSparkV41Session`, never in the shared head.
// The windows are rebuilt from the committed hidden rows the target cache carries
// (`DeepseekV4Cache.auxiliaryHiddenTail`), keyed by absolute position: an in-place restore, a
// prefix-cache hit or a stretch of AR steps all leave the cache as the single source of truth.

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
    let vocabSize: Int
    // dSpark head (`dspark_*` keys of the bundle's text config; defaults are DeepSeek-V4.1-Flash-1017's)
    let blockSize: Int
    let noiseTokenId: Int
    let targetLayerIds: [Int]
    let nStages: Int
    let markovRank: Int
    let nRoutedExperts: Int
    let numExpertsPerTok: Int
    let moeInterDim: Int

    init(_ c: DeepseekV4Configuration, raw: [String: Any] = [:]) {
        func int(_ key: String) -> Int? {
            (raw[key] as? Int) ?? (raw[key] as? NSNumber)?.intValue
        }
        hiddenSize = c.hiddenSize
        nHeads = c.numAttentionHeads
        headDim = c.headDim
        qkRopeHeadDim = c.qkRopeHeadDim
        qLoraRank = c.qLoraRank
        oLoraRank = c.oLoraRank
        oGroups = c.oGroups
        slidingWindow = c.slidingWindow
        ropeTheta = c.ropeTheta
        rmsNormEps = c.rmsNormEps
        hcMult = c.hcMult
        hcSinkhornIters = c.hcSinkhornIters
        hcEps = c.hcEps
        swigluLimit = c.swigluLimit
        routedScalingFactor = c.routedScalingFactor
        normTopkProb = c.normTopkProb
        vocabSize = c.vocabSize
        blockSize = int("dspark_block_size") ?? 5
        noiseTokenId = int("dspark_noise_token_id") ?? 128_799
        targetLayerIds =
            (raw["dspark_target_layer_ids"] as? [Any])?.compactMap {
                ($0 as? Int) ?? ($0 as? NSNumber)?.intValue
            } ?? [37, 38, 39]
        nStages = int("n_mtp_layers") ?? int("num_nextn_predict_layers") ?? 3
        markovRank = int("dspark_markov_rank") ?? 256
        nRoutedExperts = int("dspark_n_routed_experts") ?? 128
        numExpertsPerTok = int("dspark_num_experts_per_tok") ?? 3
        moeInterDim = int("dspark_moe_inter_dim") ?? c.moeIntermediateSize
    }

    /// The `dspark_*` keys of a bundle config (top level or `text_config`).
    static func rawKeys(bundleConfig url: URL) -> [String: Any] {
        guard let data = try? Data(contentsOf: url),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return [:] }
        var out = json
        if let text = json["text_config"] as? [String: Any] {
            out.merge(text) { _, t in t }
        }
        return out
    }
}

// MARK: - Rope (interleaved pairs on the last `dim` features; Python `Rope`)

enum DSparkRope {
    static func invFreq(dim: Int, theta: Float) -> [Float] {
        stride(from: 0, to: dim, by: 2).map { 1 / powf(theta, Float($0) / Float(dim)) }
    }

    /// x `[..., L, (H,) D]` with the L axis matching `positions` `[L]`.
    static func apply(
        _ x: MLXArray, positions: MLXArray, invFreq: MLXArray, dim: Int, inverse: Bool
    ) -> MLXArray {
        let half = dim / 2
        let ang = positions.asType(.float32).reshaped(-1, 1) * invFreq.reshaped(1, half)  // [L, half]
        var cos = MLX.cos(ang)
        var sin = MLX.sin(ang)
        if inverse { sin = -sin }
        if x.ndim == 4 {
            cos = cos.reshaped(-1, 1, half)
            sin = sin.reshaped(-1, 1, half)
        }
        let width = x.dim(-1)
        let head = x[.ellipsis, 0 ..< (width - dim)]
        let tail = x[.ellipsis, (width - dim)...]
        var pairShape = tail.shape
        pairShape[pairShape.count - 1] = half
        pairShape.append(2)
        let t = tail.asType(.float32).reshaped(pairShape)
        let a = t[.ellipsis, 0]
        let b = t[.ellipsis, 1]
        let rot = MLX.stacked([a * cos - b * sin, a * sin + b * cos], axis: -1).reshaped(tail.shape)
        return concatenated([head, rot.asType(x.dtype)], axis: -1)
    }
}

// MARK: - Draft attention

final class DSparkV41Attention: Module {
    let nHeads: Int, headDim: Int, rd: Int, nGroups: Int, oLoraRank: Int, win: Int
    let scale: Float
    /// Plain values, not an `MLXArray` (a stored array would be reflected as a parameter).
    let invFreq: [Float]
    @ModuleInfo(key: "wq_a") var wqA: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
    @ModuleInfo(key: "wo_b") var woB: Linear
    /// `[groups, o_lora_rank, (H / groups) * headDim]` bf16, or (JANGHT bundles) the mxfp8 packed weight
    /// `[groups, o_lora_rank, (H / groups) * headDim / 4]` uint32 with `woAScales` uint8 E8M0 per 32 inputs.
    @ParameterInfo(key: "wo_a") var woA: MLXArray
    @ParameterInfo(key: "wo_a_scales") var woAScales: MLXArray?
    @ParameterInfo(key: "attn_sink") var attnSink: MLXArray

    init(_ cfg: DSparkV41Config, mxfp8: Bool = false) {
        nHeads = cfg.nHeads
        headDim = cfg.headDim
        rd = cfg.qkRopeHeadDim
        nGroups = cfg.oGroups
        oLoraRank = cfg.oLoraRank
        win = cfg.slidingWindow
        scale = powf(Float(cfg.headDim), -0.5)
        invFreq = DSparkRope.invFreq(dim: cfg.qkRopeHeadDim, theta: cfg.ropeTheta)
        _wqA.wrappedValue = Linear(cfg.hiddenSize, cfg.qLoraRank, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: cfg.qLoraRank, eps: cfg.rmsNormEps)
        _wqB.wrappedValue = Linear(cfg.qLoraRank, cfg.nHeads * cfg.headDim, bias: false)
        _wkv.wrappedValue = Linear(cfg.hiddenSize, cfg.headDim, bias: false)
        _kvNorm.wrappedValue = RMSNorm(dimensions: cfg.headDim, eps: cfg.rmsNormEps)
        _woB.wrappedValue = Linear(cfg.oGroups * cfg.oLoraRank, cfg.hiddenSize, bias: false)
        let groupIn = cfg.nHeads * cfg.headDim / cfg.oGroups
        if mxfp8 {
            _woA.wrappedValue = MLXArray.zeros([cfg.oGroups, cfg.oLoraRank, groupIn / 4], dtype: .uint32)
            _woAScales.wrappedValue = MLXArray.zeros([cfg.oGroups, cfg.oLoraRank, groupIn / 32], dtype: .uint8)
        } else {
            _woA.wrappedValue = MLXArray.zeros([cfg.oGroups, cfg.oLoraRank, groupIn], dtype: .bfloat16)
        }
        _attnSink.wrappedValue = MLXArray.zeros([cfg.nHeads], dtype: .float32)
        super.init()
    }

    private func rope(_ x: MLXArray, _ positions: MLXArray, inverse: Bool = false) -> MLXArray {
        DSparkRope.apply(x, positions: positions, invFreq: MLXArray(invFreq), dim: rd, inverse: inverse)
    }

    /// Window rows for committed positions: rope(kv_norm(wkv(main_x))) `[1, S, headDim]` bf16
    /// (Python `DSparkAttention.seed`).
    func windowRows(mainX: MLXArray, positions: MLXArray) -> MLXArray {
        rope(kvNorm(wkv(mainX)), positions).asType(.bfloat16)
    }

    /// Queries x `[1, bs, d]` attend to (window ++ own KV), every query to every key (Python
    /// `attend` + `sparse_attention`, fp32 scores and accumulation with the learned sink).
    func attend(_ x: MLXArray, positions: MLXArray, window: MLXArray?) -> MLXArray {
        let bs = x.dim(1)
        let q = rope(wqB(qNorm(wqA(x))).reshaped(1, bs, nHeads, headDim), positions)
        let kv = rope(kvNorm(wkv(x)), positions)  // [1, bs, headDim]
        let full = window.map { concatenated([$0, kv.asType($0.dtype)], axis: 1) } ?? kv
        let qf = q[0].asType(.float32)  // [bs, H, D]
        let kf = full[0].asType(.float32)  // [L, D]
        let scores = matmul(qf, kf.transposed()) * scale  // [bs, H, L]
        let sink = attnSink.asType(.float32).reshaped(1, nHeads, 1)
        let m = maximum(scores.max(axis: -1, keepDims: true), sink)
        let p = exp(scores - m)
        let denom = p.sum(axis: -1, keepDims: true) + exp(sink - m)
        var o = matmul(p, kf) / denom  // [bs, H, D] f32
        o = rope(o.expandedDimensions(axis: 0), positions, inverse: true).asType(.bfloat16)
        // [bs, groups, (H/groups)*D] x wo_a [groups, r, (H/groups)*D] -> [bs, groups, r]
        let og = o.reshaped(bs, nGroups, -1).transposed(1, 0, 2)  // [g, bs, gd]
        let lowRank: MLXArray
        if let woAScales {  // mxfp8: one quantized matmul per group, weights read once for the bs rows
            lowRank = quantizedMM(og, woA, scales: woAScales, biases: nil, transpose: true,
                                  groupSize: 32, bits: 8, mode: .mxfp8).transposed(1, 0, 2)
        } else {
            lowRank = matmul(og, woA.transposed(0, 2, 1)).transposed(1, 0, 2)  // [bs, g, r]
        }
        return woB(lowRank.reshaped(1, bs, -1))
    }
}

// MARK: - Draft MoE (128 experts, top-3, routed 4-bit; Python `DSparkMoE`)

final class DSparkV41MoE: Module {
    let topk: Int, routeScale: Float, normTopk: Bool, lim: Float
    @ParameterInfo(key: "gate_w") var gateW: MLXArray  // [nExperts, d] f32
    @ParameterInfo(key: "gate_bias") var gateBias: MLXArray  // [nExperts] f32
    /// SwitchGLU (4-bit affine draft) or a JANGT routed-expert layer (JANGHT bundles).
    @ModuleInfo(key: "switch_mlp") var switchMLP: Module
    @ModuleInfo(key: "sw1") var sw1: Linear
    @ModuleInfo(key: "sw3") var sw3: Linear
    @ModuleInfo(key: "sw2") var sw2: Linear

    init(_ cfg: DSparkV41Config, routed: (any WeightedRoutedExpertLayer)? = nil) {
        topk = cfg.numExpertsPerTok
        routeScale = cfg.routedScalingFactor
        normTopk = cfg.normTopkProb
        lim = cfg.swigluLimit
        let d = cfg.hiddenSize
        let inter = cfg.moeInterDim
        let limit = cfg.swigluLimit
        _gateW.wrappedValue = MLXArray.zeros([cfg.nRoutedExperts, d], dtype: .float32)
        _gateBias.wrappedValue = MLXArray.zeros([cfg.nRoutedExperts], dtype: .float32)
        // Python order: gate/up in fp32, clip, silu(gate) * up, times the route weight, then bf16
        // into the down projection.
        if let routed {
            _switchMLP.wrappedValue = routed
        } else {
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: d, hiddenDims: inter, numExperts: cfg.nRoutedExperts,
            scoredGlue: { gate, up, scores in
                var g = gate.asType(.float32)
                var u = up.asType(.float32)
                if limit > 0 {
                    u = clip(u, min: -limit, max: limit)
                    g = minimum(g, MLXArray(limit))
                }
                let w = scores.asType(.float32)[.ellipsis, .newAxis, .newAxis]
                return (MLXNN.silu(g) * u * w).asType(.bfloat16)
            },
            allowFusedGateUpCache: false)
        }
        _sw1.wrappedValue = Linear(d, inter, bias: false)
        _sw3.wrappedValue = Linear(d, inter, bias: false)
        _sw2.wrappedValue = Linear(inter, d, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), S = x.dim(1), D = x.dim(2)
        let xf = x.reshaped(-1, D)
        let scores = MLX.sqrt(MLXNN.softplus(matmul(xf.asType(.float32), gateW.transposed())))
        let idx = MLX.argPartition(-(scores + gateBias), kth: topk - 1, axis: -1)[0..., 0 ..< topk]
        var w = takeAlong(scores, idx, axis: -1)
        if normTopk && topk > 1 { w = w / (w.sum(axis: -1, keepDims: true) + 1e-20) }
        w = w * routeScale
        var y: MLXArray
        if let affine = switchMLP as? SwitchGLU {
            y = affine(xf, idx, preDownScores: w).asType(.float32).sum(axis: -2)  // [N, topk, D] -> [N, D]
        } else {  // JANGT: the kernels apply the route weights and reduce (input dtype)
            y = (switchMLP as! any WeightedRoutedExpertLayer).callRouted(xf, indices: idx, scores: w).asType(.float32)
        }
        var sg = sw1(xf).asType(.float32)
        var su = sw3(xf).asType(.float32)
        if lim > 0 {
            su = clip(su, min: -lim, max: lim)
            sg = minimum(sg, MLXArray(lim))
        }
        y = y + sw2((MLXNN.silu(sg) * su).asType(.bfloat16)).asType(.float32)
        return y.asType(x.dtype).reshaped(B, S, D)
    }
}

// MARK: - One stage

final class DSparkV41Stage: Module {
    let cfg: DSparkV41Config
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
    /// Stage 0 only.
    @ModuleInfo(key: "main_proj") var mainProj: Linear?
    @ModuleInfo(key: "main_norm") var mainNorm: RMSNorm?
    /// Last stage only.
    @ModuleInfo(key: "norm") var norm: RMSNorm?

    init(_ cfg: DSparkV41Config, isFirst: Bool, isLast: Bool, mxfp8: Bool = false,
         routed: (any WeightedRoutedExpertLayer)? = nil) {
        self.cfg = cfg
        let d = cfg.hiddenSize
        let mix = (2 + cfg.hcMult) * cfg.hcMult
        _attn.wrappedValue = DSparkV41Attention(cfg, mxfp8: mxfp8)
        _moe.wrappedValue = DSparkV41MoE(cfg, routed: routed)
        _attnNorm.wrappedValue = RMSNorm(dimensions: d, eps: cfg.rmsNormEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: d, eps: cfg.rmsNormEps)
        _hcAttnFn.wrappedValue = MLXArray.zeros([mix, cfg.hcMult * d], dtype: .float32)
        _hcAttnBase.wrappedValue = MLXArray.zeros([mix], dtype: .float32)
        _hcAttnScale.wrappedValue = MLXArray.zeros([3], dtype: .float32)
        _hcFfnFn.wrappedValue = MLXArray.zeros([mix, cfg.hcMult * d], dtype: .float32)
        _hcFfnBase.wrappedValue = MLXArray.zeros([mix], dtype: .float32)
        _hcFfnScale.wrappedValue = MLXArray.zeros([3], dtype: .float32)
        _mainProj.wrappedValue = isFirst
            ? Linear(d * cfg.targetLayerIds.count, d, bias: false) : nil
        _mainNorm.wrappedValue = isFirst ? RMSNorm(dimensions: d, eps: cfg.rmsNormEps) : nil
        _norm.wrappedValue = isLast ? RMSNorm(dimensions: d, eps: cfg.rmsNormEps) : nil
        super.init()
    }

    private func mixes(_ x: MLXArray, attention: Bool) -> (pre: MLXArray, post: MLXArray, comb: MLXArray) {
        DeepseekV4Math.hcMixes(
            x, fn: attention ? hcAttnFn : hcFfnFn,
            scale: attention ? hcAttnScale : hcFfnScale,
            base: attention ? hcAttnBase : hcFfnBase,
            hcMult: cfg.hcMult, hiddenSize: cfg.hiddenSize, iters: cfg.hcSinkhornIters,
            eps: cfg.hcEps, normEps: cfg.rmsNormEps)
    }

    /// x `[1, bs, hc, d]`, preMix `[1, bs, hc]` (Python `DSparkStage.forward`).
    func forward(
        _ x: MLXArray, preMix: MLXArray, positions: MLXArray, window: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let residualA = x
        let (aPre, aPost, aComb) = mixes(x, attention: true)
        var h = attn.attend(
            attnNorm(DeepseekV4Math.hcCollapse(x, pre: preMix)), positions: positions,
            window: window)
        let xa = DeepseekV4Math.hcPost(blockOut: h, residual: residualA, post: aPost, comb: aComb)
        let (fPre, fPost, fComb) = mixes(xa, attention: false)
        h = moe(ffnNorm(DeepseekV4Math.hcCollapse(xa, pre: aPre)))
        let xf = DeepseekV4Math.hcPost(blockOut: h, residual: xa, post: fPost, comb: fComb)
        return (xf, fPre)
    }
}

// MARK: - Drafter module

final class DSparkV41Drafter: Module {
    @ModuleInfo(key: "stages") var stages: [DSparkV41Stage]

    /// `routed` (JANGHT bundles): per-stage JANGT expert layers; the stages' projections are then mxfp8.
    init(_ cfg: DSparkV41Config, routed: [any WeightedRoutedExpertLayer]? = nil) {
        _stages.wrappedValue = (0 ..< cfg.nStages).map {
            DSparkV41Stage(cfg, isFirst: $0 == 0, isLast: $0 == cfg.nStages - 1, mxfp8: routed != nil,
                           routed: routed?[$0])
        }
        super.init()
    }
}

// MARK: - Head (weights shared by every request)

/// The loaded draft head: stage weights, the Markov/confidence head tensors, and the backbone's
/// embedding and output head. Immutable after load; per-request state is in `DSparkV41Session`.
final class DSparkV41Head: @unchecked Sendable {
    let config: DSparkV41Config
    let drafter: DSparkV41Drafter
    /// Markov head embedding `[vocab, rank]` bf16.
    let markovEmbed: MLXArray
    /// Markov head output, cast to f32 and transposed ONCE at load: `[rank, vocab]`.
    let markovHeadT: MLXArray
    let embed: (MLXArray) -> MLXArray
    let head: (MLXArray) -> MLXArray

    init(
        config: DSparkV41Config, drafter: DSparkV41Drafter, markovEmbed: MLXArray,
        markovHead: MLXArray,
        embed: @escaping (MLXArray) -> MLXArray, head: @escaping (MLXArray) -> MLXArray
    ) {
        self.config = config
        self.drafter = drafter
        self.markovEmbed = markovEmbed
        let t = markovHead.asType(.float32).transposed()
        eval(t)
        self.markovHeadT = t
        self.embed = embed
        self.head = head
    }

    /// main_norm(main_proj(rows)) for committed target-layer rows `[S, d * nTargets]` -> `[1, S, d]`.
    func mainX(_ rows: MLXArray) -> MLXArray {
        let first = drafter.stages[0]
        return first.mainNorm!(first.mainProj!(rows.expandedDimensions(axis: 0).asType(.bfloat16)))
    }

    /// Draft from `seed` (the backbone's token at `startPos + 1`), windows holding committed KV for
    /// positions up to `startPos`. Returns the drafts for `startPos + 2 ... startPos + 1 + bs`
    /// (Python `DSpark.draft` without the seed). One host sync.
    func draft(seed: Int, startPos: Int, windows: [MLXArray?]) -> [Int] {
        let cfg = config
        let bs = cfg.blockSize
        var ids = [Int32](repeating: Int32(cfg.noiseTokenId), count: bs)
        ids[0] = Int32(seed)
        var x = embed(MLXArray(ids).reshaped(1, bs))  // [1, bs, d]
        x = repeated(x.expandedDimensions(axis: 2), count: cfg.hcMult, axis: 2)  // [1, bs, hc, d]
        var preMix = concatenated(
            [MLXArray.ones([1, bs, 1], dtype: .float32),
             MLXArray.zeros([1, bs, cfg.hcMult - 1], dtype: .float32)], axis: -1)
        let positions = MLXArray(Array(Int32(startPos + 1) ..< Int32(startPos + 1 + bs)))
        for (i, stage) in drafter.stages.enumerated() {
            (x, preMix) = stage.forward(
                x, preMix: preMix, positions: positions, window: i < windows.count ? windows[i] : nil)
        }
        let last = drafter.stages[drafter.stages.count - 1]
        let xc = DeepseekV4Math.hcCollapse(x, pre: preMix)  // [1, bs, d]
        let logits = head(last.norm!(xc)).asType(.float32)[0]  // [bs, vocab]
        let drafted = Self.markovChain(
            logits: logits, markovEmbed: markovEmbed, headT: markovHeadT, seed: seed)
        eval(drafted)
        return drafted.asArray(Int32.self).map(Int.init)
    }

    /// Markov-biased greedy chain over a drafted block, kept on the device (Python
    /// `markov_chain`): step i adds the Markov bias of the previous token (the seed for i = 0).
    /// logits `[bs, vocab]` f32, headT `[rank, vocab]` f32. Returns `[bs]` int32, unevaluated.
    static func markovChain(
        logits: MLXArray, markovEmbed: MLXArray, headT: MLXArray, seed: Int
    ) -> MLXArray {
        var tok = MLXArray([Int32(seed)])
        var toks: [MLXArray] = []
        for i in 0 ..< logits.dim(0) {
            let me = take(markovEmbed, tok, axis: 0).asType(.float32)  // [1, rank]
            tok = argMax(logits[i] + matmul(me, headT)[0], axis: -1, keepDims: true).asType(.int32)
            toks.append(tok)
        }
        return concatenated(toks, axis: 0)
    }
}

// MARK: - Loader

extension DSparkV41Head {
    enum LoadError: Error, CustomStringConvertible {
        case missingFile(URL)
        case missingTensor(String)
        case unexpected(String)

        var description: String {
            switch self {
            case .missingFile(let url): "dSpark draft not found at \(url.path)"
            case .missingTensor(let name): "dSpark draft is missing tensor \(name)"
            case .unexpected(let detail): "dSpark draft does not fit: \(detail)"
            }
        }
    }

    /// `<dir>/dspark.safetensors` (the bundle's `dspark/` component).
    static func file(in directory: URL) -> URL {
        directory.appendingPathComponent("dspark.safetensors")
    }

    /// Load the prestacked draft (routed experts 4-bit g64 in the SwitchGLU layout; attention,
    /// norms, shared expert and Markov head bf16; small heads f32). Verifies every module parameter
    /// is set with its declared shape and no tensor is left over (`bias_vl`, the image-token routing
    /// bias, and the confidence head are not used by text drafting).
    static func load(
        directory: URL, config cfg: DSparkV41Config,
        embed: @escaping (MLXArray) -> MLXArray, head: @escaping (MLXArray) -> MLXArray
    ) throws -> DSparkV41Head {
        let url = file(in: directory)
        guard FileManager.default.fileExists(atPath: url.path) else { throw LoadError.missingFile(url) }
        let raw = try loadArrays(url: url)
        func tensor(_ key: String) throws -> MLXArray {
            guard let t = raw[key] else { throw LoadError.missingTensor(key) }
            return t
        }
        // JANGHT bundles: JANGT trellis routed experts + the release FP8 projections as mxfp8 (loaded as stored).
        let jangt = raw["mtp.0.ffn.experts.gate_proj.jt_packed"] != nil
        let drafter: DSparkV41Drafter
        if jangt {
            let kbits = DSparkV41Config.rawKeys(bundleConfig: directory.deletingLastPathComponent()
                .appendingPathComponent("config.json"))
            let quant = kbits["quantization"] as? [String: Any] ?? [:]
            func proj(_ stage: Int, _ name: String) throws -> JANGTProjection {
                let pre = "mtp.\(stage).ffn.experts.\(name)"
                guard let entry = quant[pre] as? [String: Any], let k = (entry["kbits"] as? NSNumber)?.doubleValue
                else { throw LoadError.unexpected("JANGHT dSpark: no jangt entry for \(pre)") }
                return try JANGTProjection(
                    k: k, packed: try tensor("\(pre).jt_packed"), scale: try tensor("\(pre).jt_scale"),
                    signs: try tensor("\(pre).jt_su"), outlierColumnIndices: try tensor("\(pre).jt_ocol"),
                    outlierWeights: try tensor("\(pre).jt_wout"),
                    columnScales: name == "down_proj" ? try tensor("\(pre).jt_cs") : nil)
            }
            let routed: [any WeightedRoutedExpertLayer] = try (0 ..< cfg.nStages).map { i in
                try JANGTRoutedExpertLayer(
                    gate: try proj(i, "gate_proj"), up: try proj(i, "up_proj"), down: try proj(i, "down_proj"),
                    hiddenSize: cfg.hiddenSize, intermediateSize: cfg.moeInterDim,
                    activationLimit: cfg.swigluLimit > 0 ? cfg.swigluLimit : nil)
            }
            drafter = DSparkV41Drafter(cfg, routed: routed)
            quantize(model: drafter) { path, module in
                guard module is Linear, !path.contains(".switch_mlp.") else { return nil }
                return (groupSize: 32, bits: 8, mode: .mxfp8)
            }
        } else {
            drafter = DSparkV41Drafter(cfg)
            quantize(model: drafter, groupSize: 64, bits: 4) { path, module in
                path.contains(".switch_mlp.") && module is SwitchLinear
            }
        }
        var p: [String: MLXArray] = [:]
        /// bf16 dequantized weight (affine draft) or the stored mxfp8 weight + scales (JANGHT).
        func linear(_ dst: String, _ src: String) throws {
            if jangt {
                p["\(dst).weight"] = try tensor("\(src).weight")
                p["\(dst).scales"] = try tensor("\(src).scales")
            } else {
                p["\(dst).weight"] = try tensor("\(src).weight").asType(.bfloat16)
            }
        }
        for i in 0 ..< cfg.nStages {
            let src = "mtp.\(i)"
            let dst = "stages.\(i)"
            for name in ["wq_a", "wq_b", "wkv", "wo_b"] {
                try linear("\(dst).attn.\(name)", "\(src).attn.\(name)")
            }
            for name in ["q_norm", "kv_norm"] {
                p["\(dst).attn.\(name).weight"] = try tensor("\(src).attn.\(name).weight").asType(.bfloat16)
            }
            p["\(dst).attn.attn_sink"] = try tensor("\(src).attn.attn_sink").asType(.float32)
            if jangt {
                p["\(dst).attn.wo_a"] = try tensor("\(src).attn.wo_a.weight").reshaped(cfg.oGroups, cfg.oLoraRank, -1)
                p["\(dst).attn.wo_a_scales"] = try tensor("\(src).attn.wo_a.scales").reshaped(cfg.oGroups, cfg.oLoraRank, -1)
            } else {
                p["\(dst).attn.wo_a"] = try tensor("\(src).attn.wo_a.weight").asType(.bfloat16)
                    .reshaped(cfg.oGroups, cfg.oLoraRank, cfg.nHeads * cfg.headDim / cfg.oGroups)
            }
            p["\(dst).attn_norm.weight"] = try tensor("\(src).attn_norm.weight").asType(.bfloat16)
            p["\(dst).ffn_norm.weight"] = try tensor("\(src).ffn_norm.weight").asType(.bfloat16)
            for name in [
                "hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale",
            ] {
                p["\(dst).\(name)"] = try tensor("\(src).\(name)").asType(.float32)
            }
            p["\(dst).moe.gate_w"] = try tensor("\(src).ffn.gate.weight").asType(.float32)
            p["\(dst).moe.gate_bias"] = try tensor("\(src).ffn.gate.bias").asType(.float32)
            for (name, w) in [("sw1", "w1"), ("sw2", "w2"), ("sw3", "w3")] {
                try linear("\(dst).moe.\(name)", "\(src).ffn.shared_experts.\(w)")
            }
            for proj in jangt ? [] : ["gate_proj", "up_proj", "down_proj"] {
                p["\(dst).moe.switch_mlp.\(proj).weight"] = try tensor("\(src).ffn.experts.\(proj).weight")
                p["\(dst).moe.switch_mlp.\(proj).scales"] = try tensor("\(src).ffn.experts.\(proj).scales")
                p["\(dst).moe.switch_mlp.\(proj).biases"] = try tensor("\(src).ffn.experts.\(proj).biases")
            }
            if i == 0 {
                try linear("\(dst).main_proj", "\(src).main_proj")
                p["\(dst).main_norm.weight"] = try tensor("\(src).main_norm.weight").asType(.bfloat16)
            }
            if i == cfg.nStages - 1 {
                p["\(dst).norm.weight"] = try tensor("\(src).norm.weight").asType(.bfloat16)
            }
        }
        let lastSrc = "mtp.\(cfg.nStages - 1)"
        let markovEmbed = try tensor("\(lastSrc).markov_head.embed.weight").asType(.bfloat16)
        let markovHead = try tensor("\(lastSrc).markov_head.head.weight").asType(.bfloat16)
        guard markovEmbed.shape == [cfg.vocabSize, cfg.markovRank],
            markovHead.shape == [cfg.vocabSize, cfg.markovRank]
        else {
            throw LoadError.unexpected(
                "markov head \(markovEmbed.shape)/\(markovHead.shape), expected [\(cfg.vocabSize), \(cfg.markovRank)]")
        }
        do {
            try drafter.update(parameters: ModuleParameters.unflattened(p), verify: .all)
        } catch {
            throw LoadError.unexpected("\(error)")
        }
        eval(drafter)
        eval(markovEmbed)
        return DSparkV41Head(
            config: cfg, drafter: drafter, markovEmbed: markovEmbed, markovHead: markovHead,
            embed: embed, head: head)
    }
}

// MARK: - Verify capture (in-place partial-accept commit)

/// Per-layer cache inputs of one captured forward (Python `capture_verify`: `verify_kv` and the
/// compressor input). Lives on the sequence's holder cache between the verify forward and the
/// commit; holds references only.
final class DSV41VerifyCapture {
    struct Layer {
        /// Cache offset before this forward.
        let offset: Int
        /// Post-RoPE local KV rows `[B, 1, L, headDim]` (what the local window appended).
        let kv: MLXArray
        /// Attention input `[B, L, d]` on compressor (kv-source) layers, else nil.
        let x: MLXArray?
    }

    private(set) var tokens = 0
    private(set) var layers: [Int: Layer] = [:]
    /// False once a layer ran more than once in the forward (sub-chunked attention) or the forward
    /// count is wrong: such a capture cannot be replayed and the caller refeeds.
    private(set) var valid = true
    private var forwards = 0

    func begin(tokens: Int) {
        forwards += 1
        self.tokens = tokens
        layers.removeAll(keepingCapacity: true)
        if forwards > 1 { valid = false }
    }

    func record(layer: Int, offset: Int, kv: MLXArray, x: MLXArray?) {
        if layers[layer] != nil { valid = false }
        layers[layer] = Layer(offset: offset, kv: kv, x: x)
    }
}

// MARK: - Session (per-request draft windows)

/// Per-request draft windows, synced from the cache's hidden tail by absolute position.
final class DSparkV41Session: DSparkDraftSession {
    let head: DSparkV41Head
    /// The target, for the in-place prefix commit (nil in head-only tests: refeed).
    weak var model: DeepseekV4Model?
    /// Per-stage window rows `[1, n <= win, headDim]` for positions `winEnd - n ..< winEnd`.
    private(set) var windows: [MLXArray?]
    private(set) var winEnd = 0
    private var held = 0

    init(head: DSparkV41Head) {
        self.head = head
        self.windows = Array(repeating: nil, count: head.config.nStages)
    }

    static func tailHolder(_ cache: [KVCache]) -> DeepseekV4Cache? {
        cache.first(where: { ($0 as? DeepseekV4Cache)?.requiresAuxiliaryTokenHistory == true })
            as? DeepseekV4Cache
    }

    func reset() {
        windows = Array(repeating: nil, count: head.config.nStages)
        winEnd = 0
        held = 0
    }

    /// Bring the windows to committed end `end` (exclusive) from `tail`. Positions already held
    /// stay; only the new rows are projected (Python seeds at each commit; this seeds the same rows
    /// lazily). A tail that does not describe the cache up to `end` empties the windows.
    func sync(end: Int, tail: HybridPoolHiddenTail?) {
        let win = head.config.slidingWindow
        guard let tail, tail.endPosition == end, end > 0 else {
            reset()
            winEnd = end
            return
        }
        if winEnd > end || winEnd < tail.startPosition || end - winEnd > win || held == 0 {
            reset()
            winEnd = Swift.max(tail.startPosition, end - win)
        }
        guard winEnd < end, let rows = tail.rows(from: winEnd, to: end) else { return }
        let positions = MLXArray(Array(Int32(winEnd) ..< Int32(end)))
        let mainX = head.mainX(rows)
        for (i, stage) in head.drafter.stages.enumerated() {
            let fresh = stage.attn.windowRows(mainX: mainX, positions: positions)
            var w = windows[i].map { concatenated([$0, fresh], axis: 1) } ?? fresh
            if w.dim(1) > win { w = w[0..., (w.dim(1) - win)...] }
            windows[i] = w
        }
        held = Swift.min(win, held + (end - winEnd))
        winEnd = end
        asyncEval(windows.compactMap { $0 })
    }

    func beginVerifyCapture(cache: [KVCache]) -> Bool {
        guard model != nil, let holder = Self.tailHolder(cache) else { return false }
        holder.dsparkVerifyCapture = DSV41VerifyCapture()
        return true
    }

    func commitVerifiedPrefix(
        cache: [KVCache], checkpoint: SpeculativeCacheListCheckpoint, keep: Int
    ) -> Bool {
        model?.dsparkCommitPrefix(cache: cache, checkpoint: checkpoint, keep: keep) ?? false
    }

    func endVerifyCapture(cache: [KVCache]) {
        Self.tailHolder(cache)?.dsparkVerifyCapture = nil
    }

    func draft(seed: Int, cache: [KVCache]) -> [Int] {
        let holder = Self.tailHolder(cache)
        let end = holder?.offset ?? (cache.first?.offset ?? 0)
        sync(end: end, tail: holder?.auxiliaryHiddenTail)
        return head.draft(seed: seed, startPos: end - 1, windows: windows)
    }
}

// MARK: - DeepseekV4Model conformance

extension DeepseekV4Model: DSparkSpeculativeModel {
    public var dsparkBlockSize: Int? {
        dsparkAttachLock.lock()
        defer { dsparkAttachLock.unlock() }
        return dsparkHead?.config.blockSize
    }

    /// Load `<directory>/dspark.safetensors` once. DeepSeek-V4.1 only: a DSV4-0731 model returns
    /// false and never records a tail.
    public func attachDSparkDraft(directory: URL) throws -> Bool {
        guard config.isDeepseekV41 else { return false }
        dsparkAttachLock.lock()
        defer { dsparkAttachLock.unlock() }
        if dsparkHead != nil { return true }
        let start = Date()
        let raw = DSparkV41Config.rawKeys(
            bundleConfig: directory.deletingLastPathComponent().appendingPathComponent("config.json"))
        let cfg = DSparkV41Config(config, raw: raw)
        let head = try DSparkV41Head.load(
            directory: directory, config: cfg,
            embed: { [unowned self] ids in self.model.embedTokens(ids) },
            // Same fp32-input quantized head as the backbone's logits (reference `F.linear(x.float(), w)`).
            head: { [unowned self] x in DeepseekV4Math.lmHeadFp32(x, lmHead: self.lmHead) })
        dsparkHead = head
        model.dsparkTailCapacity = cfg.slidingWindow
        model.dsparkTapLayers = cfg.targetLayerIds.sorted()
        NSLog(
            "VMLX DSV4.1 dSpark head attached from \(directory.path): \(cfg.nStages) stages, block "
                + "\(cfg.blockSize), targets \(cfg.targetLayerIds), "
                + String(format: "%.1f s", Date().timeIntervalSince(start)))
        return true
    }

    public func makeDSparkDraftSession() -> (any DSparkDraftSession)? {
        dsparkAttachLock.lock()
        defer { dsparkAttachLock.unlock() }
        return dsparkHead.map {
            let session = DSparkV41Session(head: $0)
            session.model = self
            return session
        }
    }

    /// Python `DSV41.commit_prefix`: after a captured verify forward of n tokens that followed
    /// `checkpoint`, leave `cache` holding exactly its first `keep` tokens, the state a forward of
    /// only those tokens leaves, with no second forward. Restores in place, then per layer re-appends
    /// the kept local-window rows and replays the compressor on the kept inputs (pending tail,
    /// completed groups, compressed and index pool rows at the same RoPE positions). The Engram
    /// n-gram history and the hidden tail keep their verify records up to start + keep. No cache
    /// object is replaced. Returns false, with the cache untouched, when the capture is incomplete.
    func dsparkCommitPrefix(
        cache: [KVCache], checkpoint: SpeculativeCacheListCheckpoint, keep: Int
    ) -> Bool {
        guard config.isDeepseekV41,
            let holder = DSparkV41Session.tailHolder(cache),
            let capture = holder.dsparkVerifyCapture, capture.valid,
            keep >= 1, keep <= capture.tokens,
            cache.count == model.layers.count,
            capture.layers.count == model.layers.count
        else { return false }
        let records = (0 ..< model.layers.count).compactMap { capture.layers[$0] }
        guard records.count == model.layers.count else { return false }
        let start = records[0].offset
        guard records.allSatisfy({ $0.offset == start }) else { return false }
        // The verify forward's history and tail cover its own tokens; keep their first `keep`.
        let history = holder.auxiliaryTokenHistory
        let tail = holder.auxiliaryHiddenTail
        checkpoint.restore()
        guard holder.offset == start else {
            preconditionFailure("dSpark commit: checkpoint offset \(holder.offset) != capture \(start)")
        }
        for (i, layer) in model.layers.enumerated() {
            layer.selfAttn.commitCapturedPrefix(records[i], keep: keep, cache: cache[i])
        }
        let end = start + keep
        if var history, history.endPosition >= end, history.startPosition <= end {
            history.tokens.removeLast(history.endPosition - end)
            history.endPosition = end
            holder.auxiliaryTokenHistory = history
        } else if history != nil {
            preconditionFailure("dSpark commit: n-gram history does not cover the kept tokens")
        }
        if var tail, tail.endPosition >= end {
            tail.truncate(toEnd: end)
            holder.auxiliaryHiddenTail = tail.isEmpty ? nil : tail
        } else {
            holder.auxiliaryHiddenTail = tail
        }
        holder.dsparkVerifyCapture = nil
        return true
    }

    public func dsparkCacheUnsupportedReason(_ cache: [KVCache]) -> String? {
        guard DSparkV41Session.tailHolder(cache) != nil else {
            return "the cache has no DeepSeek-V4.1 holder layer (DSV4_KV_MODE override)"
        }
        guard SpeculativeCacheListCheckpoint.supports(cache) else {
            return "a cache layer cannot be checkpointed in place"
        }
        return nil
    }
}
