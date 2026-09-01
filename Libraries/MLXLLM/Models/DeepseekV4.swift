// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4 (DSV4-Flash / DSV4-Pro) — full model forward.
//
// Reference:
//   - jang/research/DSV4-RUNTIME-ARCHITECTURE.md §1-14
//   - jang/research/DSV-EXHAUSTIVE-VARIABLES-GUIDE.md §1 (all 13 bug fixes)
//   - jang-tools/jang_tools/dsv4_prune/mlx_model.py (1128 LOC Python ref)
//
// Architecture vs DSV3 (all new, all non-negotiable):
//   • mHC residual stream (hc_mult=4 parallel copies, collapse/expand
//     per block using a Sinkhorn-normalized mixing matrix)
//   • MLA with head_dim=512, num_kv_heads=1 (single latent KV head
//     broadcast to all 64 Q heads via GQA), RoPE only on last
//     qk_rope_head_dim=64 dims
//   • Learned per-head `attn_sink` logit prepended pre-softmax
//   • Inverse RoPE on attention OUTPUT (strips positional info before
//     residual add-back)
//   • Grouped low-rank O projection: `bsgd,grd→bsgr` einsum with
//     o_groups=8, o_lora_rank=1024, then wo_b to hidden_size
//   • MoE routing via sqrtsoftplus instead of softmax
//   • Hash routing for first num_hash_layers=3 layers (tid2eid lookup)
//   • DSV4 SwiGLU with swiglu_limit=10.0 (clamp gate + up)
//   • Per-layer rope_theta: 10000 for compress_ratio=0 (no YaRN),
//     160000 for compress_ratio>0 (with YaRN)
//   • HyperHead reduce at the top of the model (mHC copies → hidden)
//
// Compressor + Indexer (for long-context attention with compress_ratio>0)
// are wired for the canonical DSV4-Flash SWA+CSA+HSA path. Layers with
// cr>0 use DeepseekV4Cache to preserve the local sliding window plus
// pooled global context across turns and disk-cache restores.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - RoPE

/// DSV4 RoPE: YaRN scaling with `high = min(..., dim-1)` clamp (bug #10).
/// Per-layer theta — the layer chooses between `rope_theta=10000` (no
/// YaRN when compress_ratio=0) and `compress_rope_theta=160000` (with
/// YaRN scaling when compress_ratio>0).
class DeepseekV4RoPE: Module {
    let dim: Int
    let base: Float
    let factor: Float
    let origMaxPos: Int
    let betaFast: Float
    let betaSlow: Float
    // Precomputed half-dim inv-freq table.
    let invFreq: MLXArray

    init(
        dim: Int,
        base: Float,
        factor: Float = 1.0,
        origMaxPos: Int = 65536,
        betaFast: Float = 32,
        betaSlow: Float = 1
    ) {
        self.dim = dim
        self.base = base
        self.factor = factor
        self.origMaxPos = origMaxPos
        self.betaFast = betaFast
        self.betaSlow = betaSlow
        self.invFreq = DeepseekV4Math.yarnInvFreq(
            dim: dim, base: base, maxPos: 0,
            origMaxPos: origMaxPos, factor: factor,
            betaFast: betaFast, betaSlow: betaSlow)
    }

    /// Compute cos/sin tables for positions `[offset, offset+L)`.
    /// Returned shape: `(L, dim/2)`.
    ///
    /// Memoized across equal-frequency instances: every layer with the
    /// same (dim, base, factor, origMaxPos, betaFast, betaSlow) asks for
    /// the same (offset, length) within a forward pass, but each call
    /// would otherwise build its own positions→angles→cos/sin subgraph
    /// (~61 identical subgraphs per decode token). The shared single-entry
    /// table collapses them to one (Python vmlx 4472fe876).
    func cosSin(offset: Int, length: Int) -> (cos: MLXArray, sin: MLXArray) {
        let key = DeepseekV4RoPE.TableKey(
            dim: dim, base: base, factor: factor, origMaxPos: origMaxPos,
            betaFast: betaFast, betaSlow: betaSlow)
        return DeepseekV4RoPE.sharedCosSin(
            key: key, invFreq: invFreq, offset: offset, length: length)
    }

    struct TableKey: Hashable {
        let dim: Int
        let base: Float
        let factor: Float
        let origMaxPos: Int
        let betaFast: Float
        let betaSlow: Float
    }

    /// Strided variant for the compressor's pooled-row positions
    /// (`base + k*stride`, k in 0..<count). Every same-ratio compressor
    /// instance asks for the identical strided window within a forward
    /// pass, so the memo collapses ~dozens of angle→cos/sin subgraphs per
    /// prefill chunk to one per (rope key, stride).
    func cosSin(base: Int, count: Int, stride: Int) -> (cos: MLXArray, sin: MLXArray) {
        let key = DeepseekV4RoPE.TableKey(
            dim: dim, base: self.base, factor: factor, origMaxPos: origMaxPos,
            betaFast: betaFast, betaSlow: betaSlow)
        return DeepseekV4RoPE.sharedStridedCosSin(
            key: key, invFreq: invFreq, base: base, count: count, stride: stride)
    }

    private static let tableLock = NSLock()
    nonisolated(unsafe) private static var tables:
        [TableKey: (offset: Int, length: Int, cos: MLXArray, sin: MLXArray)] = [:]

    private struct StridedKey: Hashable {
        let table: TableKey
        let stride: Int
    }
    nonisolated(unsafe) private static var stridedTables:
        [StridedKey: (base: Int, count: Int, cos: MLXArray, sin: MLXArray)] = [:]

    private static func sharedCosSin(
        key: TableKey, invFreq: MLXArray, offset: Int, length: Int
    ) -> (cos: MLXArray, sin: MLXArray) {
        tableLock.lock()
        defer { tableLock.unlock() }
        if let e = tables[key], e.offset == offset, e.length == length {
            return (e.cos, e.sin)
        }
        let positions = MLXArray(Int32(offset)..<Int32(offset + length)).asType(.float32)
        // positions: (L,), invFreq: (dim/2,) → angles: (L, dim/2)
        let angles = positions.expandedDimensions(axis: -1) * invFreq.expandedDimensions(axis: 0)
        let c = cos(angles)
        let s = sin(angles)
        // Materialize before publishing. A pending graph node handed to two
        // requests lets both drive `eval` on one array_desc, and a half-written
        // position table destroys the model's positional geometry rather than
        // crashing (Osaurus dispatches chat warmup alongside the visible turn).
        MLX.eval(c, s)
        tables[key] = (offset, length, c, s)
        return (c, s)
    }

    private static func sharedStridedCosSin(
        key: TableKey, invFreq: MLXArray, base: Int, count: Int, stride: Int
    ) -> (cos: MLXArray, sin: MLXArray) {
        tableLock.lock()
        defer { tableLock.unlock() }
        let sKey = StridedKey(table: key, stride: stride)
        if let e = stridedTables[sKey], e.base == base, e.count == count {
            return (e.cos, e.sin)
        }
        let positions =
            MLXArray(Int32(0)..<Int32(count)).asType(.float32) * Float(stride)
            + Float(base)
        let angles = positions.expandedDimensions(axis: -1) * invFreq.expandedDimensions(axis: 0)
        let c = cos(angles)
        let s = sin(angles)
        MLX.eval(c, s)
        stridedTables[sKey] = (base, count, c, s)
        return (c, s)
    }
}

// MARK: - Attention (MLA with sinks + inverse RoPE + grouped O)

class DeepseekV4Attention: Module {
    let config: DeepseekV4Configuration
    let layerIdx: Int
    let numHeads: Int
    let headDim: Int
    let ropeDim: Int
    let qLoraRank: Int
    let oGroups: Int
    let oLoraRank: Int
    /// Per-layer compress_ratio ∈ {0, 4, 128}. 0 = no compressor, plain
    /// sliding-window attention. 4 or 128 = Compressor (+ Indexer at 4)
    /// augments local KV with pooled global context.
    let compressRatio: Int
    let scale: Float

    @ModuleInfo(key: "wq_a") var wqA: Linear
    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wkv") var wkv: Linear
    // wo_a operates on PER-GROUP features (numHeads*headDim // oGroups),
    // mapping them to oGroups*oLoraRank via einsum bsgd,grd→bsgr.
    // Python: Linear(n_heads*head_dim // o_groups, o_groups*o_lora_rank).
    @ModuleInfo(key: "wo_a") var woA: Linear
    @ModuleInfo(key: "wo_b") var woB: Linear
    /// q_norm is on `q_lora_rank` (1024), NOT head_dim. Applied BEFORE wq_b.
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
    /// Shape (num_heads,) — one learned sink logit per head.
    @ParameterInfo(key: "attn_sink") var attnSink: MLXArray

    let rope: DeepseekV4RoPE

    /// Python `_decode_pre_compiled`/`_decode_out_compiled` parity: per-layer
    /// compiled regions for the decode projections (3 qmm + norms + RoPE +
    /// KV QAT kernel in one dispatch) and the decode output path (inverse
    /// RoPE + grouped o-proj + wo_b). Weights are trace constants.
    private let regionLock = NSLock()
    private var preDecodeRegion: DeepseekV4CompiledRegion? = nil
    private var outDecodeRegion: DeepseekV4CompiledRegion? = nil

    // Compressor + Indexer (instantiated only when compressRatio > 0).
    // Swift can't have conditionally-present @ModuleInfo properties
    // cleanly, so we instantiate always and null the pooled path inside
    // forward when compressRatio == 0.
    @ModuleInfo(key: "compressor") var compressor: DeepseekV4Compressor?
    @ModuleInfo(key: "indexer") var indexer: DeepseekV4Indexer?

    init(config: DeepseekV4Configuration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.ropeDim = config.qkRopeHeadDim
        self.qLoraRank = config.qLoraRank
        self.oGroups = config.oGroups
        self.oLoraRank = config.oLoraRank
        self.scale = 1.0 / sqrt(Float(headDim))

        // Resolve per-layer compress_ratio. If config.compressRatios is
        // populated use it directly; otherwise fall back to the default
        // DSV4-Flash pattern (layer 0 and last → 0; middle: odd → 4,
        // even → 128 per layer index after accounting for layer 0).
        if !config.compressRatios.isEmpty && layerIdx < config.compressRatios.count {
            self.compressRatio = config.compressRatios[layerIdx]
        } else {
            let n = config.numHiddenLayers
            if layerIdx == 0 || layerIdx == n - 1 {
                self.compressRatio = 0
            } else {
                let i = layerIdx - 1
                self.compressRatio = (i % 2 == 1) ? 4 : 128
            }
        }

        self._wqA.wrappedValue = Linear(config.hiddenSize, qLoraRank, bias: false)
        self._wqB.wrappedValue = Linear(qLoraRank, numHeads * headDim, bias: false)
        self._wkv.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        // wo_a: per-group features (n_heads*head_dim // o_groups) →
        // o_groups * o_lora_rank. For DSV4-Flash: 4096 → 8192.
        self._woA.wrappedValue = Linear(
            numHeads * headDim / oGroups, oGroups * oLoraRank, bias: false)
        self._woB.wrappedValue = Linear(
            oGroups * oLoraRank, config.hiddenSize, bias: false)
        // q_norm operates on q_lora_rank (1024), not head_dim.
        self._qNorm.wrappedValue = RMSNorm(
            dimensions: qLoraRank, eps: config.rmsNormEps)
        self._kvNorm.wrappedValue = RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        self._attnSink.wrappedValue = zeros([numHeads])

        // RoPE: compressRatio>0 → compress_rope_theta (160000) + YaRN.
        // compressRatio==0 → rope_theta (10000), NO YaRN.
        let theta =
            compressRatio > 0 ? config.compressRopeTheta : config.ropeTheta
        let factor: Float =
            compressRatio > 0
            ? Float((config.ropeScaling?["factor"]?.asFloat()) ?? 16.0)
            : 1.0
        let origMax =
            Int(
                (config.ropeScaling?["original_max_position_embeddings"]?.asInt()) ?? 65536)
        self.rope = DeepseekV4RoPE(
            dim: ropeDim, base: theta, factor: factor,
            origMaxPos: origMax, betaFast: 32, betaSlow: 1)

        // Compressor + Indexer are attached ONLY on layers with a
        // non-zero compress_ratio — matches bundle weight keys.
        if compressRatio > 0 {
            self._compressor.wrappedValue = DeepseekV4Compressor(
                config: config, compressRatio: compressRatio, headDim: headDim)
            if compressRatio == 4 {
                self._indexer.wrappedValue = DeepseekV4Indexer(
                    config: config, compressRatio: compressRatio)
            }
        }
    }

    /// Python `_decode_pre_region` math: 3 projections + norms + partial
    /// RoPE + KV QAT kernel. Traced into the compiled pre region at decode;
    /// called raw at prefill. `cosT`/`sinT` arrive as arrays so per-token
    /// position changes never retrace.
    private func preMath(
        _ x: MLXArray, cosT: MLXArray, sinT: MLXArray
    ) -> (qResidual: MLXArray, q: MLXArray, kv: MLXArray) {
        let B = x.dim(0)
        let L = x.dim(1)

        // --- Q projection ---
        // wq_a(x): (B, L, qLoraRank) → q_norm on qLoraRank → wq_b:
        // (B, L, numHeads*headDim). Keep the post-qnorm residual — the
        // Indexer uses it as its own Q source.
        let qResidual = qNorm(wqA(x))
        var q = wqB(qResidual)
        q = q.reshaped(B, L, numHeads, headDim)
        // Per-head unit-weight RMSNorm via the fused `MLXFast.rmsNorm`
        // kernel (1 dispatch vs 3 ops for the manual rsqrt path).
        // Mirrors Python `mx.fast.rms_norm(q, weight=_get_q_norm_ones(...),
        // eps=...)` — reuses a `(headDim, dtype)`-cached ones tensor so
        // the 64 heads × 43 layers don't reallocate per token.
        q = MLXFast.rmsNorm(
            q,
            weight: DeepseekV4Math.qNormOnes(headDim: headDim, dtype: q.dtype),
            eps: config.rmsNormEps)
        q = q.transposed(0, 2, 1, 3)

        // --- KV projection (single latent head) ---
        var kv = kvNorm(wkv(x))
        kv = kv.reshaped(B, L, 1, headDim).transposed(0, 2, 1, 3)

        // --- Partial RoPE on last ropeDim dims of Q and K ---
        let cosQ = cosT.expandedDimensions(axes: [0, 1])
        let sinQ = sinT.expandedDimensions(axes: [0, 1])
        q = DeepseekV4Math.applyPartialRoPE(q, cos: cosQ, sin: sinQ, ropeDim: ropeDim)
        kv = DeepseekV4Math.applyPartialRoPE(kv, cos: cosQ, sin: sinQ, ropeDim: ropeDim)

        // Pinned 0731 graph contract: fake-quantize the 448 non-RoPE KV
        // dimensions in block-64 E4M3FN immediately after RoPE and before the
        // row enters any local or disk-backed cache. Tiny synthetic configs
        // use sub-64 heads and intentionally skip this production-only op.
        let nopeDim = headDim - ropeDim
        if config.activationQATEnabled && nopeDim >= 64 {
            kv = DeepseekV4Math.e4m3KVActivationRoundTrip(kv, ropeDim: ropeDim)
        }
        return (qResidual, q, kv)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let offset = cache?.offset ?? 0

        let (cosT, sinT) = rope.cosSin(offset: offset, length: L)
        // Python `_decode_pre_region`/`_decode_out_region` parity gates:
        // decode-only, off under the stage profiler (which wants eager
        // stage boundaries), never nested inside another region's trace.
        let useRegions =
            DeepseekV4Math.compileRegionsEnabled && L == 1
            && !deepseekV4StageProfileEnabled && !CompiledDecodeTrace.isActive

        var qResidual: MLXArray
        var q: MLXArray
        var kv: MLXArray
        if useRegions {
            let region = deepseekV4CachedRegion(regionLock, &preDecodeRegion) {
                vmlxTrustedCompile { [unowned self] (args: [MLXArray]) -> [MLXArray] in
                    CompiledDecodeTrace.withActive {
                        let out = self.preMath(args[0], cosT: args[1], sinT: args[2])
                        return [out.qResidual, out.q, out.kv]
                    }
                }
            }
            let outs = region([x, cosT, sinT])
            if outs.count == 3 {
                qResidual = outs[0]
                q = outs[1]
                kv = outs[2]
            } else {
                (qResidual, q, kv) = preMath(x, cosT: cosT, sinT: sinT)
            }
        } else {
            (qResidual, q, kv) = preMath(x, cosT: cosT, sinT: sinT)
        }

        // --- Cache update (sliding-window local) ---
        var keys = kv
        if let cache = cache {
            (keys, _) = cache.update(keys: kv, values: kv)
        }
        var fullKV = keys
        let windowLen = fullKV.dim(2)

        // --- Compressor + Indexer global context (compressRatio > 0 layers) ---
        //
        // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
        // Two paths now distinguished by query length:
        //
        //   * decode (L == 1): build `(B, 1, k, D)` of selected pool
        //     rows for the single query (or the whole pool if no topk
        //     gating fires) and concat onto `full_kv`. No mask needed —
        //     the only query is causally OK against every selected row
        //     because the indexer enforces `(k_idx + 1) * ratio <= q + 1`
        //     in scoring, and the compressor only emits pool rows whose
        //     summarized window has fully ended.
        //
        //   * prefill (L > 1): keep the pool flat at `(B, 1, P, D)` and
        //     build a 2-segment mask `[window_visibility | comp_visibility]`
        //     so each query sees only the local window keys it should
        //     AND only the pool rows whose summarized window ended at
        //     or before that query's position. ANDed with the indexer's
        //     selection mask on `cr=4` layers. The previous implementation
        //     padded the mask with all-ones, allowing query `q` to see
        //     pool rows summarizing tokens with positions > q, AND
        //     gathered `(B, 1, L*k, D)` — leaking query `i`'s selected
        //     rows into query `j`'s attention.
        var dsv4PrefillMask: MLXArray? = nil
        var poolEntries: Int = 0
        var tiledPoolOut: MLXArray? = nil
        if compressRatio > 0 {
            let v4Cache = cache as? DeepseekV4Cache
            if v4Cache != nil || L >= compressRatio {
                if let comp = compressor {
                    let pooledView = comp(
                        x, rope: rope, v4Cache: v4Cache, startPos: offset)
                    // pooled shape: (B, W, headDim) where W = pooled count.
                    let W = pooledView.rowCount
                    var topK: MLXArray? = nil
                    if compressRatio == 4, let idx = indexer {
                        // The indexer's learned compressor owns a separate
                        // cache branch. Advance it even before either branch
                        // emits its first complete row, so partial windows and
                        // all pre-topK history are present when learned
                        // selection first activates.
                        topK = idx(
                            x, qResidual: qResidual, rope: rope,
                            positionRope: rope, v4Cache: v4Cache, startPos: offset)
                    }
                    if W > 0, pooledView.denseArray == nil {
                        // Segmented q8 pool, or an oversized BF16 hot tier
                        // during prefill: tiled online-softmax attention
                        // replaces SDPA entirely for this layer, so the
                        // pool is never materialized (Python mirror:
                        // `_dsv4_tiled_pool_attention`).
                        tiledPoolOut = DeepseekV4Math.tiledPoolAttention(
                            queries: q,
                            localKV: fullKV,
                            pooled: pooledView,
                            offset: offset,
                            window: config.slidingWindow,
                            ratio: compressRatio,
                            scale: scale,
                            sinks: config.useAttnSink ? attnSink : nil,
                            topK: topK)
                    } else if W > 0, var pooled = pooledView.denseArray {

                        if L == 1 {
                            // DECODE FAST PATH — gather only the topk
                            // rows for the single query (or all rows
                            // when topk == nil / W <= topK), shape
                            // `(B, 1, k, D)`.
                            if let tk = topK {
                                let k = tk.dim(-1)
                                // pooled: (B, W, D) → (B, 1, 1, W, D)
                                let expanded = pooled.expandedDimensions(axes: [1, 2])
                                let pooledBroad = broadcast(
                                    expanded, to: [B, 1, L, W, headDim])
                                // tk: (B, L=1, k) → (B, 1, L, k, 1)
                                let idxExp = tk.expandedDimensions(axes: [1, 4])
                                let idxBroad = broadcast(
                                    idxExp, to: [B, 1, L, k, headDim])
                                let gathered = takeAlong(
                                    pooledBroad, idxBroad, axis: 3)
                                // (B, 1, k, D)
                                pooled = gathered.reshaped(B, 1, k, headDim)
                            } else {
                                pooled = pooled.expandedDimensions(axis: 1)
                            }
                        } else {
                            // PREFILL PATH — with indexer topk, the
                            // heads-16 Metal kernel attends each query to
                            // only its window + selected pool rows and
                            // bypasses SDPA entirely (Python
                            // `dsv4_heads16_prefill_attention`). Otherwise
                            // flat pool, mask carries visibility.
                            var heads16Out: MLXArray? = nil
                            if let tk = topK {
                                heads16Out =
                                    DeepseekV4Math.heads16PrefillAttention(
                                        queries: q,
                                        localKV: fullKV,
                                        pooled: pooled,
                                        topK: tk,
                                        offset: offset,
                                        window: config.slidingWindow,
                                        ratio: compressRatio,
                                        scale: scale,
                                        sinks: config.useAttnSink
                                            ? attnSink : nil)
                            }
                            if let heads16Out {
                                tiledPoolOut = heads16Out
                            } else {
                                pooled = pooled.expandedDimensions(axis: 1)
                                // local sliding-window visibility
                                // (B,1,L,windowLen)
                                let localMask = DeepseekV4Math.buildWindowMask(
                                    batch: B, queryLen: L, offset: offset,
                                    window: config.slidingWindow,
                                    windowLen: windowLen)
                                // compressed-pool causal visibility (B,1,L,W)
                                var compMask =
                                    DeepseekV4Math.compressedVisibility(
                                        batch: B, queryLen: L, offset: offset,
                                        compressedLen: W, ratio: compressRatio)
                                if let tk = topK {
                                    let sel =
                                        DeepseekV4Math.indexerSelectionMask(
                                            topk: tk, compressedLen: W)
                                    compMask = MLX.logicalAnd(compMask, sel)
                                }
                                // Pre-broadcast both halves to the same query
                                // dim (already done by helpers); concat along
                                // last axis.
                                dsv4PrefillMask = concatenated(
                                    [localMask, compMask], axis: -1)
                            }
                        }

                        // Kernel path already consumed the pool; `pooled`
                        // is still (B, W, D) there, so dim(2) would be the
                        // head dim — the tiledPoolOut guard is load-bearing.
                        if tiledPoolOut == nil, pooled.dim(2) > 0 {
                            poolEntries = pooled.dim(2)
                            fullKV = concatenated([fullKV, pooled], axis: 2)
                        }
                    }
                }
            }
        }

        // --- Resolve final attention mask ---
        // Three cases:
        //   (a) DSV4-built prefill mask present → use it directly.
        //   (b) Caller-provided array mask → trim/pad to `fullKV.dim(2)`
        //       (legacy code path, also triggered for `cr == 0` SWA-only
        //        layers that bypass DSV4 mask construction).
        //   (c) Bool-causal sentinel from `createAttentionMask` → leave it
        //       alone; SDPA will compute the causal mask itself.
        var adjustedMask = mask
        if let dsv4 = dsv4PrefillMask {
            adjustedMask = .array(dsv4)
        } else if case .array(let maskArr) = mask,
            poolEntries > 0
        {
            // Decode path: extend the mask with all-ones for the pool
            // entries (every selected row is causally valid for the
            // single query — see above).
            let padShape =
                Array(maskArr.shape.dropLast()) + [fullKV.dim(2) - maskArr.dim(-1)]
            let pad = MLXArray.ones(padShape, dtype: maskArr.dtype)
            adjustedMask = .array(concatenated([maskArr, pad], axis: -1))
        } else if case .array(let maskArr) = mask,
            fullKV.dim(2) != maskArr.dim(-1)
        {
            // Defensive: align array mask to actual key length.
            if maskArr.dim(-1) > fullKV.dim(2) {
                let trimmed = maskArr[.ellipsis, (-fullKV.dim(2))...]
                adjustedMask = .array(trimmed)
            } else {
                let padShape =
                    Array(maskArr.shape.dropLast()) + [fullKV.dim(2) - maskArr.dim(-1)]
                let pad = MLXArray.zeros(padShape, dtype: maskArr.dtype)
                adjustedMask = .array(concatenated([maskArr, pad], axis: -1))
            }
        }

        // --- SDPA with attention sinks ---
        // Pool-view layers already produced the full local+pool attention
        // output via the tiled online softmax; everything else runs the
        // fused SDPA over `fullKV`.
        let sdpaOut: MLXArray
        if let tiled = tiledPoolOut {
            sdpaOut = tiled
        } else {
            sdpaOut = MLXFast.scaledDotProductAttention(
                queries: q, keys: fullKV, values: fullKV,
                scale: scale, mask: adjustedMask,
                sinks: config.useAttnSink ? attnSink.asType(q.dtype) : nil)
        }
        // sdpaOut shape: (B, numHeads, L, headDim)

        if useRegions {
            let region = deepseekV4CachedRegion(regionLock, &outDecodeRegion) {
                vmlxTrustedCompile { [unowned self] (args: [MLXArray]) -> [MLXArray] in
                    CompiledDecodeTrace.withActive {
                        [self.outMath(args[0], cosT: args[1], sinT: args[2])]
                    }
                }
            }
            if let out = region([sdpaOut, cosT, sinT]).first {
                return out
            }
            // Failed compiled evaluation — fall through to eager.
        }
        return outMath(sdpaOut, cosT: cosT, sinT: sinT)
    }

    /// Python `_decode_out_math`: inverse RoPE + grouped o-proj + wo_b.
    /// Traced into the compiled out region at decode; raw at prefill.
    private func outMath(
        _ sdpaOut: MLXArray, cosT: MLXArray, sinT: MLXArray
    ) -> MLXArray {
        let B = sdpaOut.dim(0)
        let L = sdpaOut.dim(2)
        var output = sdpaOut

        // --- Inverse RoPE on the output's head-major layout ---
        let cosI = cosT.expandedDimensions(axes: [0, 1])
        let sinI = sinT.expandedDimensions(axes: [0, 1])
        output = DeepseekV4Math.applyPartialRoPE(
            output, cos: cosI, sin: sinI, ropeDim: ropeDim, inverse: true)
        output = output.transposed(0, 2, 1, 3)  // (B, L, numHeads, headDim)
            .reshaped(B, L, numHeads * headDim)

        // --- Grouped low-rank O projection ---
        // Reshape to (B, L, oGroups, groupFeat) then per-group matmul
        // through `wo_a`, producing (B, L, oGroups, oLoraRank) → concat
        // groups → wo_b. Mirrors Python `_grouped_output_projection`
        // (mlx_model.py:700) — separate dispatch for QuantizedLinear vs
        // plain Linear because the quantized packed weight cannot be
        // reshaped element-wise.
        let groupFeat = (numHeads * headDim) / oGroups
        let oReshape = output.reshaped(B, L, oGroups, groupFeat)
        let oA: MLXArray
        if let qwo = woA as? QuantizedLinear {
            // Python ref:
            //   out = out.transpose(2, 0, 1, 3)               # (oGroups, B, L, gf)
            //   weight  = wo_a.weight.reshape(oGroups, oLoraRank, -1)[:, None]
            //   scales  = wo_a.scales.reshape(oGroups, oLoraRank, -1)[:, None]
            //   biases  = wo_a.biases.reshape(oGroups, oLoraRank, -1)[:, None]
            //   out = mx.quantized_matmul(out, weight, scales, biases,
            //                              transpose=True, group_size, bits, mode)
            //   out = out.transpose(1, 2, 0, 3).reshape(B, L, oGroups*oLoraRank)
            let xT = oReshape.transposed(2, 0, 1, 3)
            // Each per-group weight slab keeps its original packed-in
            // dim (last axis) — `-1` lets MLX work out 1024 → 128 for
            // 8-bit g=32 packing.
            let wPacked = qwo.weight.reshaped(oGroups, oLoraRank, -1)
                .expandedDimensions(axis: 1)
            let wScales = qwo.scales.reshaped(oGroups, oLoraRank, -1)
                .expandedDimensions(axis: 1)
            let wBiases = qwo.biases?.reshaped(oGroups, oLoraRank, -1)
                .expandedDimensions(axis: 1)
            let outQ = MLX.quantizedMM(
                xT, wPacked, scales: wScales, biases: wBiases,
                transpose: true, groupSize: qwo.groupSize, bits: qwo.bits)
            oA = outQ.transposed(1, 2, 0, 3).reshaped(
                B, L, oGroups * oLoraRank)
        } else {
            // Non-quantized path: keep the einsum.
            // wo_a.weight has shape (oGroups*oLoraRank, groupFeat) per
            // MLX Linear convention (out, in).
            let woaW = woA.weight.reshaped(oGroups, oLoraRank, groupFeat)
            oA = einsum("bsgd,grd->bsgr", oReshape, woaW)
                .reshaped(B, L, oGroups * oLoraRank)
        }
        return woB(oA)
    }
}

// `ProcessInfo.environment` rebuilds the whole dictionary per access; reading
// it inside per-layer forward paths costs milliseconds per token across 43
// layers. The flag is a process-lifetime constant.
private let deepseekV4StageProfileEnabled =
    ProcessInfo.processInfo.environment["VMLX_DSV4_STAGE_PROFILE"] == "1"

// MARK: - MoE gate (sqrtsoftplus + hash routing)

private struct DeepseekV4SelectorKey: Hashable {
    let topK: Int
    let normalize: Bool
}

private enum DeepseekV4CompiledSelectorCache {
    typealias Selector = @Sendable ([MLXArray]) -> [MLXArray]

    private static let lock = NSLock()
    nonisolated(unsafe) private static var selectors: [DeepseekV4SelectorKey: Selector] = [:]

    static func selector(topK: Int, normalize: Bool) -> Selector {
        let key = DeepseekV4SelectorKey(topK: topK, normalize: normalize)
        lock.lock()
        defer { lock.unlock() }
        if let cached = selectors[key] { return cached }

        // Mirrors the authoritative global `@mx.compile`
        // `sqrtsoftplus_select`. Parameters that alter graph structure are
        // captured in the cache key; the scaling factor remains an array input
        // so bundles with the same selector shape can share the safe stateless
        // trace without sharing model or cache state.
        let body: Selector = { args in
            let logits = args[0]
            let bias = args[1]
            let scalingFactor = args[2]
            let originalScores = sqrt(log1p(exp(logits)))
            let biasedScores = originalScores + bias
            let indices = argPartition(
                -biasedScores, kth: topK - 1, axis: -1)[.ellipsis, 0..<topK]
                .asType(.int32)
            var weights = takeAlong(originalScores, indices, axis: -1)
            if topK > 1 && normalize {
                weights = weights / weights.sum(axis: -1, keepDims: true)
            }
            weights = weights * scalingFactor
            return [indices, weights]
        }
        let compiled = vmlxTrustedCompile(body)
        let nestedSafe: Selector = { args in
            if CompiledDecodeTrace.isActive { return body(args) }
            let result = compiled(args)
            return result.count == 2 ? result : body(args)
        }
        selectors[key] = nestedSafe
        return nestedSafe
    }
}

class DeepseekV4MoEGate: Module {
    let config: DeepseekV4Configuration
    let topK: Int
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool
    let isHashLayer: Bool
    fileprivate let compiledSelector: DeepseekV4CompiledSelectorCache.Selector
    let scalingFactorArray: MLXArray
    /// Gate projection weight: (nRoutedExperts, hiddenSize). Stored as a
    /// raw parameter (loaded via sanitize) rather than a Linear to allow
    /// the matmul to run in fp32 per the authoritative reference.
    @ParameterInfo(key: "weight") var weight: MLXArray
    /// Optional noaux bias added to scores for selection only. When
    /// absent the bias term is skipped.
    @ParameterInfo(key: "bias") var bias: MLXArray
    /// Hash routing lookup table (token_id → expert_id), shape (vocab,).
    /// Only populated for hash layers.
    @ParameterInfo(key: "tid2eid") var tid2eid: MLXArray

    init(config: DeepseekV4Configuration, layerIdx: Int) {
        self.config = config
        self.topK = config.numExpertsPerTok
        self.nRoutedExperts = config.nRoutedExperts
        self.routedScalingFactor = config.routedScalingFactor
        self.normTopkProb = config.normTopkProb
        self.isHashLayer = config.isHashLayer(layerIdx)
        self.compiledSelector = DeepseekV4CompiledSelectorCache.selector(
            topK: config.numExpertsPerTok,
            normalize: config.normTopkProb)
        self.scalingFactorArray = MLXArray([config.routedScalingFactor])
        self._weight.wrappedValue = zeros([nRoutedExperts, config.hiddenSize])
        self._bias.wrappedValue = zeros([nRoutedExperts])
        // Hash routing table: bundle ships (vocab, topK) — already
        // pre-stamped with which `topK` experts each token id should
        // route to, so the gate just gathers without computing scores.
        // Non-hash layers don't have this tensor, so we still allocate
        // a placeholder slot.
        self._tid2eid.wrappedValue =
            zeros([isHashLayer ? config.vocabSize : 1, isHashLayer ? topK : 1])
    }

    /// Returns (indices, weights) where indices has shape (B, L, topK)
    /// and weights has shape (B, L, topK).
    ///
    /// 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
    /// Hash layers now match the Python `Gate.__call__` reference
    /// (`jang_tools.dsv4.mlx_model.Gate.__call__`) — they gather the
    /// PER-TOKEN gate scores at the hash-selected expert ids instead
    /// of returning a synthetic uniform `routedScalingFactor / topK`.
    /// Without this fix every hash-routed layer collapsed all six
    /// selected experts to the same weight, throwing away the
    /// information the gate matmul + sqrtsoftplus produced and
    /// flattening the routing geometry the model was trained with.
    func callAsFunction(_ x: MLXArray, inputIds: MLXArray?) -> (MLXArray, MLXArray) {
        // Compute the gate logits in fp32 even on hash layers — the
        // hash path needs them to score the (deterministic) selected
        // experts.
        let xF32 = x.asType(.float32)
        let wF32 = weight.asType(.float32)
        let logits = xF32.matmul(wF32.transposed())
        if isHashLayer, let ids = inputIds {
            let scores = DeepseekV4Math.sqrtSoftplus(logits)
            // Hash routing: tid2eid is (vocab, topK) — pre-stamped at
            // convert time with which topK experts each token id
            // routes to. `tid2eid[ids]` for ids shape (B, L) returns
            // (B, L, topK) directly via fancy index.
            let indices = tid2eid[ids].asType(.int32)  // (B, L, topK)
            // Gate the experts using their actual sqrtsoftplus score
            // (mirror Python `mx.take_along_axis(scores, inds, axis=-1)`).
            var weights = takeAlong(scores, indices, axis: -1)
            if normTopkProb {
                let denom = weights.sum(axis: -1, keepDims: true) + 1e-20
                weights = weights / denom
            }
            weights = weights * routedScalingFactor
            return (indices.asType(.uint32), weights)
        }

        // Non-hash: the same stateless compiled sqrtsoftplus + noaux-biased
        // top-k microfunction used by the authoritative affine runtime.
        let selected = compiledSelector([logits, bias, scalingFactorArray])
        let indices = selected[0]
        let weights = selected[1]
        return (indices.asType(.uint32), weights)
    }
}

// MARK: - MoE (SwitchGLU routed + shared expert)

class DeepseekV4MoE: Module, UnaryLayer {
    let config: DeepseekV4Configuration
    let layerIdx: Int
    let topK: Int
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    var gate: DeepseekV4MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV4MLP
    /// Python `_decode_moe_region` parity: one compiled region per layer
    /// covering gate → routed experts → shared expert → fp32 accumulate,
    /// with weights riding as trace constants. Returns `[y, indices]` so
    /// expert-advisor observation sees real (non-tracer) indices outside
    /// the trace. The first forward stays eager (`moeRegionWarm`) so
    /// SwitchGLU's fused gate+up cache — which evals — materializes
    /// before any trace records.
    private let regionLock = NSLock()
    private var moeDecodeRegion: DeepseekV4CompiledRegion? = nil
    fileprivate var moeRegionWarm = false
    init(config: DeepseekV4Configuration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.topK = config.numExpertsPerTok
        let limit = config.swigluLimit
        // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
        // Symmetric DSV4 limited-SwiGLU — `silu(min(gate, limit)) *
        // clip(up, -limit, +limit)`. We pass a 2-arg `glue` closure to
        // SwitchGLU (instead of a 1-arg `activation`) so BOTH gate and
        // up get clamped before the multiply. The Python reference
        // (`jang_tools.dsv4.mlx_model._dsv4_swiglu`) also runs the
        // multiply in fp32 before casting back to gate.dtype to avoid
        // per-layer precision drift across the 43 MoE layers; we mirror
        // that here.
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            activation: MLXNN.silu,
            glue: { gate, up in
                DeepseekV4Math.dsv4SwiGLU(gate: gate, up: up, limit: limit)
            },
            scoredGlue: { gate, up, scores in
                DeepseekV4Math.dsv4ScoredSwiGLU(
                    gate: gate, up: up, scores: scores, limit: limit)
            })
        self.gate = DeepseekV4MoEGate(config: config, layerIdx: layerIdx)
        self._sharedExperts.wrappedValue = DeepseekV4MLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize * config.nSharedExperts,
            swigluLimit: limit)
    }

    fileprivate func moeMath(_ x: MLXArray, inputIds: MLXArray?) -> (y: MLXArray, indices: MLXArray) {
        let (indices, scores) = gate(x, inputIds: inputIds)
        let routed = switchMLP(x, indices, preDownScores: scores)
        let y = DeepseekV4Math.addSharedExpertFP32(
            DeepseekV4Math.reduceRoutedExpertsFP32(routed),
            shared: sharedExperts(x), outputDType: x.dtype)
        return (y, indices)
    }

    // `UnaryLayer` conformance. Hash layers need the token ids, so anything
    // routed through this shim gets the non-hash gate.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction(x, inputIds: nil)
    }

    // `inputIds` is request-scoped and MUST stay an argument: this module is
    // shared across every concurrent request through the model, so parking the
    // ids on `self` lets one request route another's tokens.
    func callAsFunction(_ x: MLXArray, inputIds: MLXArray?) -> MLXArray {
        let profileStages = deepseekV4StageProfileEnabled && x.dim(1) == 1

        // Decode (L==1) routes through the per-layer compiled MoE region.
        // Hash layers without threaded ids fall back to eager (mirrors the
        // Python guard) so the trace never bakes the wrong gate variant.
        let ids = gate.isHashLayer ? inputIds : nil
        if DeepseekV4Math.compileRegionsEnabled, !profileStages,
            x.dim(1) == 1, moeRegionWarm, !(gate.isHashLayer && ids == nil)
        {
            let region = deepseekV4CachedRegion(regionLock, &moeDecodeRegion) {
                vmlxTrustedCompile { [unowned self] (args: [MLXArray]) -> [MLXArray] in
                    // Nested-compile guard: inner compiled microfunctions
                    // (selector, SwiGLU) inline their raw bodies while this
                    // region traces — and on every retrace.
                    CompiledDecodeTrace.withActive {
                        let out = self.moeMath(
                            args[0], inputIds: args.count > 1 ? args[1] : nil)
                        return [out.y, out.indices]
                    }
                }
            }
            let outs = ids.map { region([x, $0]) } ?? region([x])
            if outs.count == 2 {
                JangPressCanonicalExpertAdvisor.shared.observe(
                    layer: layerIdx, indices: outs[1])
                return outs[0]
            }
            // Failed compiled evaluation returns empty — fall through to eager.
        }

        var stageStart = profileStages ? CFAbsoluteTimeGetCurrent() : 0
        func finishStage(_ name: String, _ arrays: [MLXArray]) {
            guard profileStages else { return }
            MLX.eval(arrays)
            let now = CFAbsoluteTimeGetCurrent()
            FileHandle.standardError.write(Data(String(format:
                "[DSV4MoEProfile] layer=%d stage=%@ ms=%.3f\n",
                layerIdx, name, (now - stageStart) * 1_000).utf8))
            stageStart = now
        }

        let (indices, scores) = gate(x, inputIds: ids)
        finishStage("gate", [indices, scores])
        JangPressCanonicalExpertAdvisor.shared.observe(layer: layerIdx, indices: indices)
        let routed = switchMLP(x, indices, preDownScores: scores)
        var y = DeepseekV4Math.reduceRoutedExpertsFP32(routed)
        finishStage("routed", [y])
        finishStage("route_reduce", [y])
        let shared = sharedExperts(x)
        finishStage("shared", [shared])
        y = DeepseekV4Math.addSharedExpertFP32(
            y, shared: shared, outputDType: x.dtype)
        finishStage("add", [y])
        moeRegionWarm = true
        return y
    }
}

// MARK: - Dense MLP (shared expert) with DSV4 SwiGLU clamp

class DeepseekV4MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let swigluLimit: Float

    init(hiddenSize: Int, intermediateSize: Int, swigluLimit: Float) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        self.swigluLimit = swigluLimit
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = gateProj(x)
        let u = upProj(x)
        return downProj(DeepseekV4Math.dsv4SwiGLU(gate: g, up: u, limit: swigluLimit))
    }
}

// MARK: - mHC Hyper-Connection (per-block collapse + expand)

public class DeepseekV4HyperConnection: Module {
    let hcMult: Int
    let hcIters: Int
    let hcEps: Float
    /// `rms_norm_eps` — the UNWEIGHTED input norm's epsilon, which the reference reads from a
    /// different config field than `hc_eps`. DSV4 ships both at 1e-6 so the distinction is
    /// invisible for this family; GLM-5.3 ships 1e-5 and 1e-6, and reusing `hc_eps` there shifts
    /// every mHC output. Kept explicit rather than defaulted so a new family has to say which it
    /// means.
    let hcNormEps: Float
    let hiddenSize: Int
    let mixHc: Int  // (2 + hcMult) * hcMult — bundle stores params at this width
    /// `hc_{attn,ffn}_fn`: shape `((2+hc)*hc, hc*hidden)`. Bundle stores
    /// `(24, 16384)` for hc=4, hidden=4096.
    @ParameterInfo(key: "fn") var fn: MLXArray
    /// `hc_{attn,ffn}_scale`: shape `(3,)` per-field scalar.
    @ParameterInfo(key: "scale") var scale: MLXArray
    /// `hc_{attn,ffn}_base`: shape `((2+hc)*hc,)` per-field bias.
    @ParameterInfo(key: "base") var base: MLXArray
    convenience init(config: DeepseekV4Configuration) {
        // The norm epsilon comes from `rms_norm_eps`, NOT `hc_eps`. DSV4 ships both at 1e-6, so for
        // this family the two are interchangeable and nothing here can tell them apart; a family
        // that ships them differently gets a different mHC output for every token.
        self.init(
            hcMult: config.hcMult, sinkhornIterations: config.hcSinkhornIters,
            eps: config.hcEps, normEps: config.rmsNormEps, hiddenSize: config.hiddenSize)
    }

    /// The designated initialiser, in explicit quantities rather than one family's configuration.
    /// Taking the two epsilons separately is what lets a test set them to DIFFERENT values — with a
    /// DSV4 config alone they are always equal, so a conflation of the two is unobservable.
    public init(hcMult: Int, sinkhornIterations: Int, eps: Float, normEps: Float, hiddenSize: Int) {
        self.hcMult = hcMult
        self.hcIters = sinkhornIterations
        self.hcEps = eps
        self.hcNormEps = normEps
        self.hiddenSize = hiddenSize
        self.mixHc = (2 + hcMult) * hcMult
        self._fn.wrappedValue = zeros([mixHc, hcMult * hiddenSize])
        self._scale.wrappedValue = zeros([3])
        self._base.wrappedValue = zeros([mixHc])
    }

    /// Collapse: `h` shape (B, L, hcMult, hiddenSize) → collapsed x
    /// (B, L, hiddenSize) plus `post` (B, L, hcMult) and `comb`
    /// (B, L, hcMult, hcMult) for the expand step.
    ///
    /// Mirrors Python `DeepseekV4DecoderLayer._hc_pre`:
    ///   x_flat   = flatten(h, axis=2)              # (B, L, hc*hidden)
    ///   x_normed = rms_norm(x_flat, ones, eps)
    ///   mixes    = x_normed @ fn.T                 # (B, L, mix_hc)
    ///   pre, post, comb = hc_split_sinkhorn(mixes, scale, base, hc, iters, eps)
    ///   y = sum(pre[..., None] * x_flat.reshape(B,L,hc,D), axis=2)
    public func collapse(_ h: MLXArray) -> (x: MLXArray, post: MLXArray, comb: MLXArray) {
        // Decode (L==1) routes through a shared compiled region — the graph is
        // identical for every layer/step, so weights ride along as inputs and
        // the host-side graph build is amortized (Python `_get_hc_pre_compiled`).
        // Inside another region's trace, inline the raw graph instead.
        if DeepseekV4Math.compileRegionsEnabled, h.dim(1) == 1,
            !CompiledDecodeTrace.isActive
        {
            return DeepseekV4Math.hcPreCompiled(
                h, fn: fn, scale: scale, base: base,
                hcMult: hcMult, hiddenSize: hiddenSize, iters: hcIters, eps: hcEps,
                normEps: hcNormEps)
        }
        return DeepseekV4Math.hcPreGraph(
            h, fn: fn, scale: scale, base: base,
            hcMult: hcMult, hiddenSize: hiddenSize, iters: hcIters, eps: hcEps,
            normEps: hcNormEps)
    }

    /// Expand: given attn/ffn output `blockOut` (B, L, hiddenSize),
    /// residual (B, L, hcMult, hiddenSize), and the (post, comb) from
    /// the matching collapse, return new h (B, L, hcMult, hiddenSize).
    ///
    /// Mirrors Python `_hc_post`:
    ///   y = post[..., None] * x[..., None, :] + matmul(comb, residual)
    public func expand(
        blockOut: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray
    ) -> MLXArray {
        DeepseekV4Math.hcPost(
            blockOut: blockOut, residual: residual, post: post, comb: comb)
    }
}

// MARK: - HyperHead (top-of-model mHC reduce)

class DeepseekV4HyperHead: Module {
    let hcMult: Int
    let hiddenSize: Int
    let hcEps: Float
    /// Bundle stores `hc_head_fn` at `(hcMult, hcMult*hiddenSize)`.
    @ParameterInfo(key: "hc_head_fn") var fn: MLXArray
    @ParameterInfo(key: "hc_head_base") var base: MLXArray
    @ParameterInfo(key: "hc_head_scale") var scale: MLXArray
    /// Constant ones-vector for the RMS norm in `_hc_head_reduce`.
    let hcHeadRMSOnes: MLXArray

    init(config: DeepseekV4Configuration) {
        self.hcMult = config.hcMult
        self.hiddenSize = config.hiddenSize
        self.hcEps = config.rmsNormEps
        self._fn.wrappedValue = zeros([hcMult, hcMult * hiddenSize])
        self._base.wrappedValue = zeros([hcMult])
        self._scale.wrappedValue = zeros([1])
        self.hcHeadRMSOnes = MLXArray.ones([config.hcMult * config.hiddenSize])
    }

    /// Reduce (B, L, hcMult, hiddenSize) → (B, L, hiddenSize). Mirrors
    /// Python `_hc_head_reduce`:
    ///   x_flat   = flatten(x, axis=2)            # (B, L, hc*hidden)
    ///   x_normed = rms_norm(x_flat, ones, eps)
    ///   mixes    = x_normed @ hc_head_fn.T       # (B, L, hc)
    ///   pre      = sigmoid(mixes * scale + base) + hc_eps
    ///   y        = sum(pre[..., None] * x_flat.reshape(B,L,hc,D), axis=2)
    /// NO sum-to-1 normalization — match the Python reference exactly.
    func reduce(_ h: MLXArray) -> MLXArray {
        let dtype = h.dtype
        let B = h.dim(0)
        let L = h.dim(1)
        let xFlat = h.reshaped(B, L, hcMult * hiddenSize)
        // Same dtype rule as `_hc_pre`: this RMS reduction spans
        // hcMult*hiddenSize (≈16K for DSV4-Flash). Apple GPUs differ in
        // implicit bf16 accumulation behavior, so keep the reduction and
        // the tiny gate projection in fp32, then cast the final mixed
        // residual back to the model dtype. This mirrors the jang-tools
        // HyperHead fix documented in DSV4-HC-PRE-FP32-CAST-FIX.
        let xNormed = MLXFast.rmsNorm(
            xFlat.asType(.float32),
            weight: hcHeadRMSOnes.asType(.float32),
            eps: hcEps)
        let mixes = xNormed.matmul(fn.asType(.float32).transposed())  // (B, L, hcMult)
        let pre = sigmoid(mixes * scale.asType(.float32) + base.asType(.float32))
            + MLXArray(hcEps)
        let xReshape = xFlat.reshaped(B, L, hcMult, hiddenSize)
        return (pre.asType(dtype).expandedDimensions(axis: -1) * xReshape).sum(axis: -2)
            .asType(dtype)
    }
}

// MARK: - Decoder layer (mHC wrap over attn + MoE)

class DeepseekV4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV4Attention
    @ModuleInfo(key: "mlp") var mlp: DeepseekV4MoE
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attnHC: DeepseekV4HyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHC: DeepseekV4HyperConnection

    let layerIdx: Int

    /// Python `_decode_tail_region` parity: one compiled region per layer for
    /// the whole decode tail — hc_post(attn) + hc_pre(ffn) + post-LN + MoE +
    /// hc_post(ffn). Merging these compiled calls + raw norms into a single
    /// region removes the per-call host dispatch floors between them. Weights
    /// ride as trace constants (per-layer closure). The attention output
    /// projection stays outside (Swift attention does not defer it).
    /// Returns `[hF, indices]` so expert-advisor observation runs on real
    /// arrays outside the trace.
    private let regionLock = NSLock()
    private var tailDecodeRegion: DeepseekV4CompiledRegion? = nil

    init(config: DeepseekV4Configuration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self._selfAttn.wrappedValue = DeepseekV4Attention(config: config, layerIdx: layerIdx)
        self._mlp.wrappedValue = DeepseekV4MoE(config: config, layerIdx: layerIdx)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._attnHC.wrappedValue = DeepseekV4HyperConnection(config: config)
        self._ffnHC.wrappedValue = DeepseekV4HyperConnection(config: config)
    }

    /// Forward. `h` shape: (B, L, hcMult, hiddenSize).
    func callAsFunction(
        _ h: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        inputIds: MLXArray?,
        statsSink: ((String, MLXArray)) -> Void = { _ in }
    ) -> MLXArray {
        // Explicit Release diagnostic only. Splitting the lazy graph at each
        // block boundary perturbs total throughput, but attributes the
        // remaining affine DSV4 gap without changing the production graph
        // when the variable is absent. Restrict to single-token decode so a
        // prompt prefill does not emit thousands of misleading stage rows.
        let profileStages = deepseekV4StageProfileEnabled && h.dim(1) == 1
        var stageStart = profileStages ? CFAbsoluteTimeGetCurrent() : 0
        func finishStage(_ name: String, _ arrays: [MLXArray]) {
            guard profileStages else { return }
            MLX.eval(arrays)
            let now = CFAbsoluteTimeGetCurrent()
            FileHandle.standardError.write(Data(String(format:
                "[DSV4StageProfile] layer=%d stage=%@ ms=%.3f\n",
                layerIdx, name, (now - stageStart) * 1_000).utf8))
            stageStart = now
        }
        // ---- Attention HC ----
        let residualA = h
        let (xA, postA, combA) = attnHC.collapse(h)
        finishStage("attn_hc_pre", [xA, postA, combA])
        let normedA = inputLayerNorm(xA)
        finishStage("attn_norm", [normedA])
        // Python parity (mlx_model.py:3613-3628): sub-chunked attention.
        // Each 512-token slice sees the cache offset advanced by the previous
        // slice's update, so the per-slice mask from the layer's own cache is
        // exact — numerically identical to top-level 512-chunk prefill. MoE
        // below still consumes the full chunk for gather_qmm batch throughput.
        let attnOut: MLXArray
        let sub = DeepseekV4Math.attnSubchunkTokens
        if let cache, sub > 0, normedA.dim(1) >= 2 * sub {
            var outs: [MLXArray] = []
            var s = 0
            let chunkLen = normedA.dim(1)
            while s < chunkLen {
                let e = min(s + sub, chunkLen)
                let xs = normedA[0..., s ..< e]
                // Same mask construction as the proven top-level 512-chunk
                // path: RotatingKVCache.makeMask defaults its window to
                // maxCacheSize (the sliding window), from THIS layer's own
                // advanced offset.
                let subMask = createAttentionMask(h: xs, cache: cache)
                outs.append(selfAttn(xs, mask: subMask, cache: cache))
                s = e
            }
            attnOut = concatenated(outs, axis: 1)
        } else {
            attnOut = selfAttn(normedA, mask: mask, cache: cache)
        }
        finishStage("attention", [attnOut])
        // Sub-block stats split the layer-output trace into "attention block"
        // vs "MoE block" so a magnitude injection names its half. Raw path
        // only — the compiled decode region fuses past these seams, but the
        // corruption under study already shows in the first prefill chunk,
        // which always takes the raw path.
        DeepseekV4Math.layerStat("layer\(layerIdx).attn", attnOut).map(statsSink)

        // Decode tail region (Python `_decode_tail_region`): everything after
        // SDPA fuses into one compiled dispatch per layer. Gated on the MoE
        // having run eager once (`moeRegionWarm`) so SwitchGLU's fused
        // gate+up cache — which evals — materializes before any trace.
        let ids = mlp.gate.isHashLayer ? inputIds : nil
        if DeepseekV4Math.compileRegionsEnabled, !profileStages,
            h.dim(1) == 1, mlp.moeRegionWarm,
            !(mlp.gate.isHashLayer && ids == nil)
        {
            let region = deepseekV4CachedRegion(regionLock, &tailDecodeRegion) {
                vmlxTrustedCompile { [unowned self] (args: [MLXArray]) -> [MLXArray] in
                    // Nested-compile guard: inner compiled microfunctions
                    // (hcPre region, selector, SwiGLU) inline raw bodies
                    // while this region traces — and on every retrace.
                    CompiledDecodeTrace.withActive {
                        let hA = self.attnHC.expand(
                            blockOut: args[0], residual: args[1],
                            post: args[2], comb: args[3])
                        let (xF, postF, combF) = DeepseekV4Math.hcPreGraph(
                            hA, fn: self.ffnHC.fn, scale: self.ffnHC.scale,
                            base: self.ffnHC.base, hcMult: self.ffnHC.hcMult,
                            hiddenSize: self.ffnHC.hiddenSize,
                            iters: self.ffnHC.hcIters, eps: self.ffnHC.hcEps,
                            normEps: self.ffnHC.hcNormEps)
                        let normedF = self.postAttentionLayerNorm(xF)
                        let out = self.mlp.moeMath(
                            normedF, inputIds: args.count > 4 ? args[4] : nil)
                        let hF = self.ffnHC.expand(
                            blockOut: out.y, residual: hA,
                            post: postF, comb: combF)
                        return [hF, out.indices]
                    }
                }
            }
            let regionArgs = [attnOut, residualA, postA, combA] + (ids.map { [$0] } ?? [])
            let outs = region(regionArgs)
            if outs.count == 2 {
                JangPressCanonicalExpertAdvisor.shared.observe(
                    layer: layerIdx, indices: outs[1])
                return outs[0]
            }
            // Failed compiled evaluation returns empty — fall through to eager.
        }

        let hA = attnHC.expand(
            blockOut: attnOut, residual: residualA, post: postA, comb: combA)
        finishStage("attn_hc_post", [hA])

        // ---- FFN HC ----
        let residualF = hA
        let (xF, postF, combF) = ffnHC.collapse(hA)
        finishStage("ffn_hc_pre", [xF, postF, combF])
        let normedF = postAttentionLayerNorm(xF)
        finishStage("ffn_norm", [normedF])
        let ffnOut = mlp(normedF, inputIds: inputIds)
        finishStage("moe", [ffnOut])
        DeepseekV4Math.layerStat("layer\(layerIdx).moe", ffnOut).map(statsSink)
        let hF = ffnHC.expand(
            blockOut: ffnOut, residual: residualF, post: postF, comb: combF)
        finishStage("ffn_hc_post", [hF])
        return hF
    }
}

// MARK: - Inner model

public class DeepseekV4ModelInner: Module {
    let config: DeepseekV4Configuration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV4DecoderLayer]
    @ModuleInfo(key: "hc_head") var hcHead: DeepseekV4HyperHead
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: DeepseekV4Configuration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0..<config.numHiddenLayers).map {
            DeepseekV4DecoderLayer(config: config, layerIdx: $0)
        }
        self._hcHead.wrappedValue = DeepseekV4HyperHead(config: config)
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // embed: (B, L) → (B, L, hiddenSize)
        var h = embedTokens(inputs)
        // Tile to mHC copies: (B, L, hiddenSize) → (B, L, hcMult, hiddenSize).
        // Python tiles via broadcast; Swift uses `repeated` along axis -2.
        h = h.expandedDimensions(axis: -2)  // (B, L, 1, H)
        h = repeated(h, count: config.hcMult, axis: -2)  // (B, L, hcMult, H)

        let firstCache = cache?.first
        let hFlat2 = h.reshaped(h.dim(0), h.dim(1), -1)  // for createAttentionMask
        let mask = createAttentionMask(h: hFlat2, cache: firstCache)

        // Python `_layerwise_prefill_materialization_enabled` parity: bound
        // the lazy cross-layer CSA/HCA graph on deep-context prefill. The
        // barrier costs 25-30% prefill throughput, so it engages only once
        // final context exceeds the safe curve (default 24,576 tokens).
        let chunkTokens = h.dim(1)
        let layerwisePrefill =
            chunkTokens > 1
            && DeepseekV4Math.layerwisePrefillEnabled(
                chunkTokens: chunkTokens,
                finalContextTokens: (firstCache?.offset ?? 0) + chunkTokens)

        var layerStats: [(String, MLXArray)] = []
        layerStats.append(contentsOf: DeepseekV4Math.layerStat("embed", h).map { [$0] } ?? [])
        for (i, layer) in layers.enumerated() {
            h = layer(
                h,
                mask: mask,
                cache: cache?[i],
                inputIds: inputs,
                statsSink: { layerStats.append($0) })
            if layerwisePrefill {
                MLX.eval(h)
            }
            layerStats.append(
                contentsOf: DeepseekV4Math.layerStat("layer\(i)", h).map { [$0] } ?? [])
        }

        // HyperHead reduce: (B, L, hcMult, H) → (B, L, H)
        var out = hcHead.reduce(h)
        layerStats.append(contentsOf: DeepseekV4Math.layerStat("hcReduce", out).map { [$0] } ?? [])
        out = norm(out)
        layerStats.append(contentsOf: DeepseekV4Math.layerStat("norm", out).map { [$0] } ?? [])
        DeepseekV4Math.flushLayerStats(layerStats, length: out.dim(1))
        // Weight checksums AFTER the first forward on purpose: forcing every
        // layer-15 weight to materialize BEFORE the racy first-touch window
        // would serialize the load and could suppress the very corruption
        // being fingerprinted. Post-forward, the boot's fate is already
        // decided; the checksum then separates "weights corrupted at load"
        // from "weights fine, compute path corrupted".
        if DeepseekV4Math.weightChecksumEnabled {
            DeepseekV4Math.runWeightChecksumOnce {
                for i in [14, 15, 16] where i < layers.count {
                    let flat = layers[i].parameters().flattened()
                    let stats = flat.map {
                        ($0.0, MLX.abs($0.1.asType(.float32)).mean())
                    }
                    MLX.eval(stats.map(\.1))
                    for (name, v) in stats {
                        print(
                            "[vmlx][dsv4/wsum] layer\(i).\(name) absmean=\(v.item(Float.self))"
                        )
                    }
                }
            }
        }
        return out
    }
}

// MARK: - Outer model

public class DeepseekV4Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    /// The canonical SWA + CSA/HSA `DeepseekV4Cache` pool is path-dependent
    /// and has no B-wide wrapper. Preserve concurrent request queuing while
    /// decoding one DSV4 sequence at a time.
    public var maximumSupportedDecodeBatchSize: Int? { 1 }

    public var kvHeads: [Int]
    var config: DeepseekV4Configuration
    public var model: DeepseekV4ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: DeepseekV4Configuration) {
        self.config = config
        // Single latent KV head per layer — report kvHeads as [1]*L so
        // the cache allocator sizes per-layer caches correctly.
        self.kvHeads = Array(repeating: 1, count: config.numHiddenLayers)
        self.model = DeepseekV4ModelInner(config: config)
        self._lmHead.wrappedValue = Linear(
            config.hiddenSize, config.vocabSize, bias: false)
    }

    /// Build per-layer caches.
    ///
    /// 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass — pure long-context):
    /// DSV4-Flash IS a hybrid SWA+CSA+HSA architecture by definition.
    /// The previous "long-ctx off" fallback (a plain `RotatingKVCache`
    /// for every layer) was producing measurably degraded output on
    /// any chat that exceeded `sliding_window=128` tokens, because the
    /// `cr>0` layers lost their compressed/indexed global context after
    /// the local window rotated. The toggle is removed; every layer
    /// now allocates the canonical cache for its `compress_ratio`:
    ///   - `cr == 0` (layers 0 and n-1) → `RotatingKVCache(window=128)`
    ///   - `cr > 0`  (every other layer) → `DeepseekV4Cache(window=128, cr=cr)`
    ///
    /// `DSV4_KV_MODE` env override is preserved for diagnostics so the
    /// host can deliberately pick the local KV sizing tradeoff:
    ///   - default (unset / "sliding"): rotating window + DeepseekV4Cache pool
    ///   - "full"  : plain KVCacheSimple on every layer (no compression,
    ///               no pool — for memory-permits long-reasoning runs
    ///               that don't need the hybrid path)
    ///   - "tq"    : KVCacheSimple, BatchEngine swaps to TurboQuantKVCache
    ///               once offset > min-tokens (caller must also set
    ///               `GenerateParameters.kvMode = .turboQuant(...)`)
    ///
    /// Caller-level `GenerateParameters.kvMode = .turboQuant` is
    /// intentionally NOT enough to switch DSV4 into `"tq"` mode. Osaurus
    /// can set global TQ defaults for ordinary KV models; DSV4 must keep
    /// its SWA+CSA+HSA hybrid cache unless the operator explicitly opts
    /// into the diagnostic/simple-cache override via `DSV4_KV_MODE=tq`.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let env = ProcessInfo.processInfo.environment
        let envMode = env["DSV4_KV_MODE"]?.lowercased()
        let mode: String = envMode ?? "sliding"

        return (0..<config.numHiddenLayers).map { layerIdx in
            switch mode {
            case "full", "tq":
                return KVCacheSimple()
            default:
                let cr =
                    config.compressRatios.count > layerIdx
                    ? config.compressRatios[layerIdx]
                    : Self.defaultCompressRatio(
                        layerIdx: layerIdx,
                        numLayers: config.numHiddenLayers)
                if cr > 0 {
                    return DeepseekV4Cache(
                        slidingWindow: config.slidingWindow,
                        compressRatio: cr)
                }
                return RotatingKVCache(
                    maxSize: config.slidingWindow, keep: 0)
            }
        }
    }

    /// Mirror of the per-layer compress_ratio default in
    /// `DeepseekV4Attention.init` for bundles whose
    /// `config.compressRatios` array isn't populated. Layers 0 and n-1
    /// are pure SWA (`cr=0`); middle layers alternate `4` (HSA+CSA)
    /// and `128` (CSA only).
    static func defaultCompressRatio(layerIdx: Int, numLayers: Int) -> Int {
        if layerIdx == 0 || layerIdx == numLayers - 1 { return 0 }
        let i = layerIdx - 1
        return (i % 2 == 1) ? 4 : 128
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let h = model(inputs, cache: cache)
        let logits = DeepseekV4Math.lmHeadFp32(h, lmHead: lmHead)
        // Separated from the hidden-state stats on purpose: a normal `norm`
        // followed by an abnormal `logits` isolates the fused quantized head
        // (VMLX_DSV4_LM_HEAD_MODE) rather than the transformer stack.
        DeepseekV4Math.flushLayerStats(
            [DeepseekV4Math.layerStat("logits", logits)].compactMap { $0 },
            length: logits.dim(1))
        return logits
    }

    /// DSV4 prefill has opposed optima (Python `_dsv4_attn_subchunk_tokens`):
    /// MoE gather_qmm throughput grows with chunk width while attention cost
    /// grows super-linearly. With in-layer 512-token attention sub-chunking
    /// bounding the attention side, the OUTER chunk can widen to 2048 so MoE
    /// sees 4x the batch (Python reference: 319 → 452 pp/s at 8k). Without
    /// sub-chunking the default 512 outer chunk is preserved.
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let prefillStepSize = windowSize ?? 512
        let sub = DeepseekV4Math.attnSubchunkTokens
        let outerStep: Int
        if sub > 0 {
            let env = ProcessInfo.processInfo.environment
            let requested = env["DSV4_PREFILL_OUTER_CHUNK"].flatMap(Int.init) ?? 2048
            outerStep = max(prefillStepSize, requested)
        } else {
            outerStep = prefillStepSize
        }

        let tokensShape = input.text.tokens.shape
        if tokensShape.count >= 2 && tokensShape[0] != 1 {
            fatalError(
                "DeepseekV4Model.prepare expects single-sequence input (batch=1), "
                    + "got shape \(tokensShape).")
        }

        var flatTokens = input.text.tokens.reshaped([-1])
        var flatMask: MLXArray? = nil
        if let m = input.text.mask {
            flatMask = m.ndim >= 2 ? m.reshaped([-1]) : m
        }

        while flatTokens.size > prefillStepSize {
            try Task.checkCancellation()
            // Always leave >= 1 token so the caller's final forward yields
            // the sampling logits (same contract as the LLMModel default).
            let step = min(outerStep, flatTokens.size - 1)
            let chunkTokens = flatTokens[..<step][.newAxis, 0...]
            let chunkMask = flatMask.map { $0[..<step] }
            let chunkText = LMInput.Text(tokens: chunkTokens, mask: chunkMask)
            _ = self(chunkText, cache: cache.isEmpty ? nil : cache, state: nil)
            MLX.eval(cache)
            flatTokens = flatTokens[step...]
            if let m = flatMask { flatMask = m[step...] }
            PrefillProgressReporter.reportCompletedUnits(input.text.tokens.size - flatTokens.size)
            Memory.clearCache()
        }

        return .tokens(LMInput.Text(tokens: flatTokens, mask: flatMask))
    }

    /// Weight sanitize — remap DSV4 bundle key names to match module
    /// attribute paths, stack per-expert weights, drop MTP + unused
    /// compressor/indexer keys.
    ///
    /// Remap rules (from §G of RUNTIME-ARCHITECTURE):
    ///   model.embed.weight            → model.embed_tokens.weight
    ///   layers.{L}.attn.*             → model.layers.{L}.self_attn.*
    ///   layers.{L}.ffn.*              → model.layers.{L}.mlp.*
    ///   layers.{L}.attn_norm.weight   → model.layers.{L}.input_layernorm.weight
    ///   layers.{L}.ffn_norm.weight    → model.layers.{L}.post_attention_layernorm.weight
    ///   layers.{L}.hc_attn_*          → model.layers.{L}.attn_hc.{fn,scale,base}
    ///   layers.{L}.hc_ffn_*           → model.layers.{L}.ffn_hc.{fn,scale,base}
    ///   hc_head_*                     → model.hc_head.{hc_head_fn,hc_head_base,hc_head_scale}
    ///   norm.weight                   → model.norm.weight
    ///   head.weight                   → lm_head.weight
    ///   ffn.experts.{E}.{w1|w2|w3}.*  → mlp.switch_mlp.{gate|down|up}_proj.* (stacked)
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        // First pass: direct rename + drop MTP (training head only).
        // Compressor + Indexer weights are KEPT — they're wired into
        // DeepseekV4Attention for long-context (L > sliding_window)
        // attention. Layers with compress_ratio == 0 carry no such
        // weights; layers with >0 carry `self_attn.compressor.*` and
        // (for ratio=4) `self_attn.indexer.*`.
        // Mirrors `Model.sanitize` in
        // jang-tools/jang_tools/dsv4/mlx_model.py:1124. Per-prefix
        // structural matching — avoids over-broad string replace bugs
        // (e.g. ".w1." colliding outside MLP contexts).
        let projForW = ["w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"]
        for (rawKey, value) in weights {
            if rawKey.hasPrefix("mtp.") { continue }

            // Top-level (no `layers.N.` prefix).
            if rawKey == "embed.weight" || rawKey == "embed.scales"
                || rawKey == "embed.biases"
            {
                let suffix = String(rawKey.dropFirst("embed.".count))
                out["model.embed_tokens.\(suffix)"] = value
                continue
            }
            if rawKey.hasPrefix("head.") {
                // head.{weight,scales,biases} → lm_head.*
                let suffix = String(rawKey.dropFirst("head.".count))
                out["lm_head.\(suffix)"] = value
                continue
            }
            if rawKey == "norm.weight" {
                out["model.norm.weight"] = value
                continue
            }
            if rawKey == "hc_head_fn" || rawKey == "hc_head_base"
                || rawKey == "hc_head_scale"
            {
                // `@ParameterInfo(key: "hc_head_*")` lives at
                // `model.hc_head.hc_head_*`.
                out["model.hc_head.\(rawKey)"] = value
                continue
            }

            // layers.N.{...} branch
            guard rawKey.hasPrefix("layers.") else {
                out["model.\(rawKey)"] = value
                continue
            }
            let afterLayers = rawKey.dropFirst("layers.".count)
            guard let dotIdx = afterLayers.firstIndex(of: ".") else { continue }
            let layerStr = String(afterLayers[..<dotIdx])
            guard Int(layerStr) != nil else { continue }
            let rest = String(afterLayers[afterLayers.index(after: dotIdx)...])
            let pfx = "model.layers.\(layerStr)"

            // Norms
            if rest == "attn_norm.weight" {
                out["\(pfx).input_layernorm.weight"] = value
                continue
            }
            if rest == "ffn_norm.weight" {
                out["\(pfx).post_attention_layernorm.weight"] = value
                continue
            }

            // mHC per-layer (hc_attn_*, hc_ffn_*).
            if rest.hasPrefix("hc_attn_") {
                let field = String(rest.dropFirst("hc_attn_".count))
                out["\(pfx).attn_hc.\(field)"] = value
                continue
            }
            if rest.hasPrefix("hc_ffn_") {
                let field = String(rest.dropFirst("hc_ffn_".count))
                out["\(pfx).ffn_hc.\(field)"] = value
                continue
            }

            // Attention subtree (q_norm / kv_norm / wq_a / wq_b / wkv /
            // wo_a / wo_b / attn_sink / compressor.* / indexer.*).
            if rest.hasPrefix("attn.") {
                let inner = String(rest.dropFirst("attn.".count))
                out["\(pfx).self_attn.\(inner)"] = value
                continue
            }

            // FFN subtree.
            if rest.hasPrefix("ffn.") {
                let inner = String(rest.dropFirst("ffn.".count))
                if inner.hasPrefix("gate.") {
                    let f = String(inner.dropFirst("gate.".count))
                    out["\(pfx).mlp.gate.\(f)"] = value
                    continue
                }
                if inner.hasPrefix("shared_experts.") {
                    let f = String(inner.dropFirst("shared_experts.".count))
                    if let firstDot = f.firstIndex(of: "."),
                        let proj = projForW[String(f[..<firstDot])]
                    {
                        let suffix = String(f[f.index(after: firstDot)...])
                        out["\(pfx).mlp.shared_experts.\(proj).\(suffix)"] = value
                        continue
                    }
                    out["\(pfx).mlp.shared_experts.\(f)"] = value
                    continue
                }
                if inner.hasPrefix("experts.") {
                    let after = String(inner.dropFirst("experts.".count))
                    guard let eDot = after.firstIndex(of: ".") else { continue }
                    let eStr = String(after[..<eDot])
                    let tail = String(after[after.index(after: eDot)...])
                    if let firstDot = tail.firstIndex(of: "."),
                        let proj = projForW[String(tail[..<firstDot])]
                    {
                        let suffix = String(tail[tail.index(after: firstDot)...])
                        out["\(pfx).mlp.experts.\(eStr).\(proj).\(suffix)"] = value
                        continue
                    }
                    out["\(pfx).mlp.experts.\(eStr).\(tail)"] = value
                    continue
                }
                out["\(pfx).mlp.\(inner)"] = value
                continue
            }

            out["\(pfx).\(rest)"] = value
        }

        // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
        // DSV4-Flash JANGTQ bundles ship a pre-stacked
        // `jangtq_stacked.safetensors` overlay where the routed-expert
        // weights live at
        // `layers.{L}.mlp.switch_mlp.{gate,down,up}_proj.{packed,norms}`
        // — note the missing `tq_` prefix. Older Swift JANGTQ bundles
        // and the in-tree `TurboQuantSwitchLinear` use `tq_packed` /
        // `tq_norms`. Rewrite the un-prefixed names so the
        // `@ParameterInfo` keys match. Layout-preserving rename only.
        for layerIdx in 0..<config.numHiddenLayers {
            for projName in ["gate_proj", "down_proj", "up_proj"] {
                for (src, dst) in [("packed", "tq_packed"), ("norms", "tq_norms")] {
                    let from = "model.layers.\(layerIdx).mlp.switch_mlp.\(projName).\(src)"
                    let to = "model.layers.\(layerIdx).mlp.switch_mlp.\(projName).\(dst)"
                    if let v = out.removeValue(forKey: from), out[to] == nil {
                        out[to] = v
                    }
                }
            }
        }

        // Second pass: stack per-expert weights into switch_mlp.{gate,
        // up,down}_proj.*. Two formats supported:
        //
        // Affine (JANG_2L / JANG4): suffixes weight / scales / biases.
        //   Source per expert: (out, in) [+ (out, in/group)] [+ (out, in/group)]
        //   Stacked shape: (n_experts, ...).
        //
        // JANGTQ (JANGTQ2 / JANGTQ4 routed experts): suffixes
        // tq_packed / tq_norms. tq_bits is a per-tensor int constant
        // — we drop it (TurboQuantSwitchLinear configures bits at
        // construction time from the model_factory).
        //   Source per expert: tq_packed (out, packed_cols), tq_norms (out,)
        //   Stacked shape: (n_experts, out, packed_cols) / (n_experts, out)
        // The first pass already rewrote `.w1.` → `.gate_proj.` (etc.)
        // globally, so per-expert keys live at
        // `model.layers.L.mlp.experts.E.gate_proj.{suffix}`. Stack into
        // `model.layers.L.mlp.switch_mlp.{gate,down,up}_proj.{suffix}`.
        let suffixes = ["weight", "scales", "biases", "tq_packed", "tq_norms"]
        let streamJANGTQExperts = JANGTQStreamingExperts.isEnabled
        for layerIdx in 0..<config.numHiddenLayers {
            let prefix = "model.layers.\(layerIdx).mlp.experts"
            for projName in ["gate_proj", "down_proj", "up_proj"] {
                for suffix in suffixes {
                    let first = "\(prefix).0.\(projName).\(suffix)"
                    guard out[first] != nil else { continue }
                    if streamJANGTQExperts && (suffix == "tq_packed" || suffix == "tq_norms") {
                        for e in 0..<config.nRoutedExperts {
                            out.removeValue(
                                forKey: "\(prefix).\(e).\(projName).\(suffix)")
                        }
                        continue
                    }
                    var tensors: [MLXArray] = []
                    for e in 0..<config.nRoutedExperts {
                        let key = "\(prefix).\(e).\(projName).\(suffix)"
                        guard let t = out[key] else {
                            tensors = []
                            break
                        }
                        tensors.append(t)
                    }
                    if tensors.count == config.nRoutedExperts {
                        let stackedKey =
                            "model.layers.\(layerIdx).mlp.switch_mlp.\(projName).\(suffix)"
                        if out[stackedKey] == nil {
                            out[stackedKey] = stacked(tensors)
                        }
                        for e in 0..<config.nRoutedExperts {
                            out.removeValue(
                                forKey: "\(prefix).\(e).\(projName).\(suffix)")
                        }
                    }
                }
                // Drop per-expert + prestacked-switch_mlp tq_bits scalars
                // — TurboQuantSwitchLinear gets the bit width from the
                // JANGTQ config (`mxtq_bits.routed_expert`), not weights.
                // Legacy bundles ship `mlp.experts.{e}.{proj}.tq_bits`;
                // prestacked bundles (DSV4-Flash JANGTQ_K, etc.) ship a
                // single `mlp.switch_mlp.{proj}.tq_bits` per layer.
                // Drop both regardless of which layout.
                for e in 0..<config.nRoutedExperts {
                    out.removeValue(
                        forKey: "\(prefix).\(e).\(projName).tq_bits")
                }
                out.removeValue(
                    forKey: "model.layers.\(layerIdx).mlp.switch_mlp.\(projName).tq_bits")
            }
        }
        return out
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
