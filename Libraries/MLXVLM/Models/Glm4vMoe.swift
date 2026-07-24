//
//  Glm4vMoe.swift
//  mlx-swift-lm
//
//  port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/glm4v_moe
//
//  GLM-4.5V family (e.g. GLM-4.5V-3bit). model_type "glm4v_moe",
//  architecture "Glm4vMoeForConditionalGeneration".
//
//  FUSION of two existing ports:
//    * VISION tower + M-RoPE machinery + image-merge + top-level model structure are
//      structurally REUSED from Glm4v.swift (glm4v / GLM-4.6V). The glm4v_moe vision
//      tower is byte-for-byte identical to glm4v's (confirmed against
//      mlx_vlm/models/glm4v_moe/vision.py), so the vision half is copied verbatim.
//    * The LANGUAGE decoder is a DeepSeek-V3-style MoE (gate + SwitchGLU experts +
//      shared expert + first_k_dense_replace), COPIED from GLM4MOE.swift
//      (mlx_lm/glm4_moe). GLM4MOE's types live in MLXLLM (not importable here), so the
//      gate / MoE / dense-MLP / decoder pieces are re-implemented in this file.
//
//  THE KEY DIFFERENCE vs Glm4v.swift (the #1 integration risk):
//    glm4v (dense) applies M-RoPE with style="sectioned_even_odd"
//      → rotate_half_even_odd + repeat_interleave sectioning  (Glm4v.swift's
//        rotateHalfInterleaved + repeatInterleave).
//    glm4v_moe applies M-RoPE with style="sectioned_half_split"
//      → plain rotate_half (split-at-midpoint, concat([-x2, x1])) and NO
//        repeat_interleave; the cos/sin are sectioned by `mrope_section * 2`
//        (6 chunks, take chunk[i % 3]) then concatenated.
//    Both share the SAME GLM4VRotaryEmbedding (inv_freq over dim = head_dim *
//    partial_rotary_factor; emb = concat(freqs, freqs); cos/sin scaled by
//    attention_scaling=1.0). Partial rotary is handled inside the apply step:
//    rotary_dim = cos.shape[-1] < head_dim, so only q[..., :rotary_dim] is rotated and
//    q[..., rotary_dim:] passes through. See `Glm4vMoeLanguage` below.
//
//  Verified against (raw.githubusercontent.com/Blaizzy/mlx-vlm/main/mlx_vlm/models):
//    glm4v_moe/config.py, glm4v_moe/language.py, glm4v_moe/vision.py,
//    rope_utils.py (_apply_mrope / _section_frequency_layout / rotate_half /
//    _apply_rotary_embedding), and glm4v/language.py (for the even_odd contrast).
//

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Language

private enum Glm4vMoeLanguage {

    // MARK: M-RoPE helpers (sectioned_half_split style — NOT glm4v's even_odd)

    /// Standard half-split rotate_half: x1 = first half, x2 = second half,
    /// return concat([-x2, x1]). (rope_utils.rotate_half)
    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let x1 = x[.ellipsis, ..<half]
        let x2 = x[.ellipsis, half...]
        return concatenated([-x2, x1], axis: -1)
    }

    /// Section cos/sin per mrope_section using the "half_split" layout:
    /// split `values` (last axis) by the cumulative splits of `mrope_section * 2`
    /// (i.e. 6 split points → 6 chunks), then for chunk i take chunk[i % 3] and
    /// concatenate. Mirrors rope_utils._section_frequency_layout (minus its trailing
    /// `[:, None, :, :]`, which the caller adds via the `heads` axis).
    /// `values` shape on entry: (3, batch, seq, rotaryFullDim).
    static func sectionFrequencyLayout(_ values: MLXArray, mropeSection: [Int]) -> MLXArray {
        precondition(mropeSection.count == 3, "sectioned MRoPE expects exactly 3 sections")
        let doubled = mropeSection + mropeSection
        var splitIndices = [Int]()
        var cum = 0
        for s in doubled.dropLast() {
            cum += s
            splitIndices.append(cum)
        }
        let chunks = split(values, indices: splitIndices, axis: -1)
        let selected = chunks.enumerated().map { i, chunk in chunk[i % 3] }
        return concatenated(selected, axis: -1)
    }

    /// Apply sectioned_half_split M-RoPE.
    /// q, k:    (B, heads, L, headDim)
    /// cos,sin: (3, B, L, rotaryFullDim)   from Glm4vMoeRotaryEmbedding
    /// Sections cos/sin (→ (B, L, rotaryDim)), unsqueezes a heads axis (→ (B,1,L,rotaryDim)),
    /// then partial-rotary applies over q[..., :rotaryDim] (rotaryDim = cos.dim(-1)).
    static func applyMultimodalRotaryPosEmb(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray, mropeSection: [Int]
    ) -> (MLXArray, MLXArray) {
        // Section (drops the leading "3" axis to a single (B, L, rotaryDim) layout)…
        var cosS = sectionFrequencyLayout(cos, mropeSection: mropeSection)  // (B, L, rotaryDim)
        var sinS = sectionFrequencyLayout(sin, mropeSection: mropeSection)
        // …unsqueeze the heads axis (axis=1), matching rope_utils' [:, None, :, :].
        cosS = expandedDimensions(cosS, axis: 1)  // (B, 1, L, rotaryDim)
        sinS = expandedDimensions(sinS, axis: 1)

        let rotaryDim = cosS.dim(-1)
        let qRot = q[.ellipsis, ..<rotaryDim]
        let qPass = q[.ellipsis, rotaryDim...]
        let kRot = k[.ellipsis, ..<rotaryDim]
        let kPass = k[.ellipsis, rotaryDim...]

        let qEmbed = (qRot * cosS) + (rotateHalf(qRot) * sinS)
        let kEmbed = (kRot * cosS) + (rotateHalf(kRot) * sinS)

        if qPass.dim(-1) == 0 && kPass.dim(-1) == 0 {
            return (qEmbed.asType(q.dtype), kEmbed.asType(k.dtype))
        }
        return (
            concatenated([qEmbed, qPass], axis: -1).asType(q.dtype),
            concatenated([kEmbed, kPass], axis: -1).asType(k.dtype)
        )
    }

    // MARK: M-RoPE Rotary Embedding (identical math to Glm4v's, half-split sectioning
    //       happens later in applyMultimodalRotaryPosEmb).

    fileprivate class Glm4vMoeRotaryEmbedding {
        let invFreq: MLXArray
        let attentionScaling: Float

        init(_ config: Glm4vMoeConfiguration.TextConfiguration) {
            // dim = int(head_dim * partial_rotary_factor); inv_freq over arange(0,dim,2).
            let dim = Int(Float(config.headDim) * config.partialRotaryFactor)
            let base = config.ropeTheta
            self.attentionScaling = 1.0

            let p =
                MLXArray(stride(from: 0, to: dim, by: 2)).asType(.int64).asType(.float32)
                / Float(dim)
            self.invFreq = 1.0 / pow(base, p)
        }

        /// positionIds: (3, batch, seq). Returns cos/sin of shape (3, batch, seq, dim)
        /// where dim = 2 * invFreq.count (emb = concat(freqs, freqs)).
        func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
            let batchSize = positionIds.dim(1)

            var invFreqExpanded = invFreq[.newAxis, .newAxis, 0..., .newAxis].asType(.float32)
            invFreqExpanded = broadcast(
                invFreqExpanded,
                to: [3, batchSize, invFreq.dim(0), 1])

            let positionIdsExpanded = positionIds[0..., 0..., .newAxis, 0...].asType(.float32)

            let freqs = matmul(invFreqExpanded, positionIdsExpanded).transposed(0, 1, 3, 2)
            let emb = concatenated([freqs, freqs], axis: -1)
            let cos = MLX.cos(emb) * attentionScaling
            let sin = MLX.sin(emb) * attentionScaling

            return (cos.asType(x.dtype), sin.asType(x.dtype))
        }
    }

    // MARK: Attention (MoE text decoder — head_dim explicit, q/k/v bias, half-split M-RoPE)

    fileprivate class Attention: Module {

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let mropeSection: [Int]

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        public init(_ args: Glm4vMoeConfiguration.TextConfiguration) {
            let dim = args.hiddenSize
            self.heads = args.attentionHeads
            self.kvHeads = args.kvHeads
            self.headDim = args.headDim  // EXPLICIT — 96*128=12288 ≠ hidden 4096.
            self.scale = pow(Float(headDim), -0.5)
            self.mropeSection = args.mropeSection

            // glm4v_moe: attention_bias=true for q/k/v; o_proj always bias-free.
            // NOTE: head_dim is explicit, so q_proj maps 4096 -> 96*128 = 12288.
            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)
            // use_qk_norm is FALSE for GLM-4.5V — no q_norm/k_norm.
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
            positionEmbeddings: (MLXArray, MLXArray)
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            let (cos, sin) = positionEmbeddings
            (queries, keys) = applyMultimodalRotaryPosEmb(
                q: queries, k: keys, cos: cos, sin: sin, mropeSection: mropeSection)

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    // MARK: Dense MLP (used for the first_k_dense_replace layers + shared expert)

    fileprivate class MLP: Module, UnaryLayer {

        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        public init(hiddenSize: Int, intermediateSize: Int) {
            self._gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
            self._up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
            self._down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    // MARK: MoE Gate (sigmoid + noaux_tc group routing) — copied from GLM4MOE.swift

    fileprivate class MoEGate: Module {
        let topK: Int
        let normTopkProb: Bool
        let nRoutedExperts: Int
        let routedScalingFactor: Float
        let nGroup: Int
        let topkGroup: Int
        let scoringFunc: String

        @ParameterInfo(key: "weight") var weight: MLXArray
        @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

        init(_ config: Glm4vMoeConfiguration.TextConfiguration) {
            precondition(config.topkMethod == "noaux_tc", "Unsupported topk method.")

            self.topK = config.numExpertsPerTok
            self.normTopkProb = config.normTopkProb
            self.nRoutedExperts = config.nRoutedExperts
            self.routedScalingFactor = config.routedScalingFactor
            self.nGroup = config.nGroup
            self.topkGroup = config.topkGroup
            self.scoringFunc = config.scoringFunc

            self._weight.wrappedValue = zeros([config.nRoutedExperts, config.hiddenSize])
            self._eScoreCorrectionBias.wrappedValue = zeros([config.nRoutedExperts])

            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
            let hiddenStates = x.matmul(weight.T)
            var scores: MLXArray
            if scoringFunc == "sigmoid" {
                scores = sigmoid(hiddenStates.asType(.float32))
            } else {
                scores = softmax(hiddenStates, axis: -1, precise: true)
            }

            let originalScores = scores
            var selectionScores = scores + eScoreCorrectionBias

            if nGroup > 1 {
                selectionScores = unflatten(selectionScores, axis: -1, shape: [nGroup, -1])
                let groupScores = top(selectionScores, k: 2, axis: -1)
                    .sum(axis: -1, keepDims: true)
                let k = nGroup - topkGroup
                let groupIdx = argPartition(groupScores, kth: k - 1, axis: -2)[
                    .ellipsis, ..<k, 0...]
                selectionScores = putAlong(
                    selectionScores, stopGradient(groupIdx),
                    values: MLXArray(0.0, dtype: selectionScores.dtype), axis: -2)
                selectionScores = flattened(selectionScores, start: -2, end: -1)
            }

            let k = topK
            let inds = argPartition(-selectionScores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
            var selectedScores = takeAlong(originalScores, inds, axis: -1)

            if topK > 1, normTopkProb {
                let denominator = selectedScores.sum(axis: -1, keepDims: true)
                selectedScores = selectedScores / denominator
            }
            selectedScores = selectedScores * routedScalingFactor

            return (inds, selectedScores)
        }
    }

    // MARK: MoE block (SwitchGLU experts + shared expert) — copied from GLM4MOE.swift

    fileprivate class MoE: Module, UnaryLayer {
        let layerIdx: Int
        let numExpertsPerTok: Int
        let gate: MoEGate

        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
        @ModuleInfo(key: "shared_experts") var sharedExperts: MLP?

        init(_ config: Glm4vMoeConfiguration.TextConfiguration, layerIdx: Int) {
            self.layerIdx = layerIdx
            self.numExpertsPerTok = config.numExpertsPerTok
            self.gate = MoEGate(config)

            self._switchMLP.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.nRoutedExperts
            )

            if config.nSharedExperts > 0 {
                let intermediateSize = config.moeIntermediateSize * config.nSharedExperts
                self._sharedExperts.wrappedValue = MLP(
                    hiddenSize: config.hiddenSize, intermediateSize: intermediateSize)
            }

            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (inds, scores) = gate(x)
            JangPressCanonicalExpertAdvisor.shared.observe(layer: layerIdx, indices: inds)
            var y = switchMLP(x, inds)
            y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
            if let sharedExperts {
                y = y + sharedExperts(x)
            }
            return y
        }
    }

    // MARK: Decoder Layer (first_k_dense_replace boundary)

    fileprivate class Glm4vMoeDecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: UnaryLayer

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        public init(_ args: Glm4vMoeConfiguration.TextConfiguration, layerIdx: Int) {
            self._attention.wrappedValue = Attention(args)

            // first_k_dense_replace: layers [0, k) are dense, layers [k, …) are MoE.
            if args.nRoutedExperts > 0 && layerIdx >= args.firstKDenseReplace {
                self.mlp = MoE(args, layerIdx: layerIdx)
            } else {
                self.mlp = MLP(
                    hiddenSize: args.hiddenSize, intermediateSize: args.intermediateSize)
            }

            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
            positionEmbeddings: (MLXArray, MLXArray)
        ) -> MLXArray {
            let r = attention(
                inputLayerNorm(x), mask: mask, cache: cache,
                positionEmbeddings: positionEmbeddings)
            let h = x + r
            let r2 = mlp(postAttentionLayerNorm(h))
            return h + r2
        }
    }

    // MARK: Text Model

    fileprivate class Glm4vMoeTextModel: Module {

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        fileprivate let layers: [Glm4vMoeDecoderLayer]
        fileprivate let norm: RMSNorm
        let rotaryEmb: Glm4vMoeRotaryEmbedding

        public init(_ args: Glm4vMoeConfiguration.TextConfiguration) {
            precondition(args.vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

            self.layers = (0 ..< args.hiddenLayers)
                .map { idx in Glm4vMoeDecoderLayer(args, layerIdx: idx) }
            self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self.rotaryEmb = Glm4vMoeRotaryEmbedding(args)
        }

        public func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil,
            positionIds: MLXArray? = nil
        ) -> MLXArray {
            var h: MLXArray
            if let inputEmbedding {
                h = inputEmbedding
            } else if let inputs {
                h = embedTokens(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            var posIds: MLXArray
            if let positionIds {
                posIds = positionIds
            } else {
                let offset = cache?.first?.offset ?? 0
                let seqLen = h.dim(h.ndim - 2)
                let positions = MLXArray(Int32(offset) ..< Int32(offset + seqLen))
                    .expandedDimensions(axis: 0)
                posIds = tiled(positions, repetitions: [3, 1, 1])
            }

            let positionEmbeddings = rotaryEmb(h, positionIds: posIds)
            let mask = createAttentionMask(h: h, cache: cache?.first)

            for (i, layer) in layers.enumerated() {
                h = layer(
                    h, mask: mask, cache: cache?[i],
                    positionEmbeddings: positionEmbeddings)
            }

            return norm(h)
        }
    }

    // MARK: Language Model

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: Glm4vMoeTextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int]
        var _positionIds: MLXArray?
        var _ropeDeltas: MLXArray?

        public init(_ args: Glm4vMoeConfiguration.TextConfiguration) {
            self.model = Glm4vMoeTextModel(args)

            if !args.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    args.hiddenSize, args.vocabularySize, bias: false)
            }

            self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        }

        public func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> LMOutput {
            var positionIds: MLXArray? = nil

            let cacheOffset: Int
            if let cache = cache, let first = cache.first {
                cacheOffset = first.offset
            } else {
                cacheOffset = 0
            }

            if let storedPositionIds = _positionIds {
                let seqLen: Int
                if let inputEmbedding {
                    seqLen = inputEmbedding.dim(inputEmbedding.ndim - 2)
                } else if let inputs {
                    seqLen = inputs.dim(inputs.ndim - 1)
                } else {
                    seqLen = 0
                }

                let storedLen = storedPositionIds.dim(2)
                if cacheOffset + seqLen <= storedLen {
                    // Prefill: use stored M-RoPE position IDs
                    positionIds =
                        storedPositionIds[
                            0..., 0..., cacheOffset ..< (cacheOffset + seqLen)]
                } else {
                    // Autoregressive: compute sequential positions using rope_deltas
                    let delta = _ropeDeltas ?? MLXArray(Int32(0))
                    let batchSize = inputEmbedding?.dim(0) ?? inputs?.dim(0) ?? 1
                    var posArrays = [MLXArray]()
                    for _ in 0 ..< 3 {
                        let pos = MLXArray(Int32(cacheOffset) ..< Int32(cacheOffset + seqLen))
                            .expandedDimensions(axis: 0)
                        let tiledPos = tiled(pos, repetitions: [batchSize, 1])
                        posArrays.append((tiledPos + delta).expandedDimensions(axis: 0))
                    }
                    positionIds = concatenated(posArrays, axis: 0)
                }
            }

            var out = model(
                inputs, cache: cache, inputEmbedding: inputEmbedding,
                positionIds: positionIds)
            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }
            return LMOutput(logits: out)
        }
    }
}

// MARK: - Vision
//
// IDENTICAL to Glm4v.swift's Glm4vVision (confirmed against glm4v_moe/vision.py). Copied
// verbatim, renamed to Glm4vMoeVision to avoid symbol collision with Glm4v in the module.

private enum Glm4vMoeVision {

    /// Pure-MLX bilinear grid_sample, matching mlx_vlm kernels._grid_sample_mlx.

    static fileprivate func applyRotaryPosEmbVision(
        _ tensor: MLXArray, freqs: MLXArray
    ) -> MLXArray {
        var cosVal = MLX.cos(freqs)
        var sinVal = MLX.sin(freqs)

        cosVal = expandedDimensions(cosVal, axis: 1)
        cosVal = tiled(cosVal, repetitions: [1, 1, 2])

        sinVal = expandedDimensions(sinVal, axis: 1)
        sinVal = tiled(sinVal, repetitions: [1, 1, 2])

        let output = (tensor * cosVal) + (QwenVL.rotateHalf(tensor) * sinVal)
        return output.asType(tensor.dtype)
    }

    fileprivate class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo var proj: Conv3d

        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let embedDim: Int

        init(_ config: Glm4vMoeConfiguration.VisionConfiguration) {
            self.patchSize = config.patchSize
            self.temporalPatchSize = config.temporalPatchSize
            self.inChannels = config.inChannels
            self.embedDim = config.hiddenSize

            let kernelSize = IntOrTriple(
                [config.temporalPatchSize, config.patchSize, config.patchSize])
            self._proj.wrappedValue = Conv3d(
                inputChannels: config.inChannels,
                outputChannels: config.hiddenSize,
                kernelSize: kernelSize,
                stride: kernelSize,
                bias: true
            )
        }

        public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            var h = hiddenStates.reshaped(
                -1, inChannels, temporalPatchSize, patchSize, patchSize
            ).movedAxis(source: 1, destination: 4)

            h = proj(h)
            h = h.reshaped(-1, embedDim)
            return h
        }
    }

    fileprivate class VisionEmbeddings: Module {
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        let embedDim: Int
        let imageSize: Int
        let patchSize: Int
        let numPositions: Int

        init(_ config: Glm4vMoeConfiguration.VisionConfiguration) {
            self.embedDim = config.hiddenSize
            self.imageSize = config.imageSize
            self.patchSize = config.patchSize
            let numPatches = (config.imageSize / config.patchSize)
                * (config.imageSize / config.patchSize)
            self.numPositions = numPatches
            self._positionEmbedding.wrappedValue = Embedding(
                embeddingCount: numPatches, dimensions: config.hiddenSize)
        }

        func callAsFunction(
            _ embeddings: MLXArray, frames: [THW], hCoords: MLXArray, wCoords: MLXArray
        ) -> MLXArray {
            let totalSeq = hCoords.dim(0)
            if totalSeq == 0 {
                return embeddings
            }

            let posWeight = positionEmbedding.weight
            let hiddenSize = posWeight.dim(1)
            let origSizeSq = posWeight.dim(0)
            let origSize = Int(Double(origSizeSq).squareRoot().rounded())

            let pos2d = posWeight.reshaped(origSize, origSize, hiddenSize)
                .transposed(2, 0, 1)[.newAxis, 0..., 0..., 0...]
                .asType(.float32)
            let pos2dNHWC = pos2d.transposed(0, 2, 3, 1)

            var targetH = [Float]()
            var targetW = [Float]()
            targetH.reserveCapacity(totalSeq)
            targetW.reserveCapacity(totalSeq)
            for frame in frames {
                let count = frame.t * frame.h * frame.w
                for _ in 0 ..< count {
                    targetH.append(Float(frame.h))
                    targetW.append(Float(frame.w))
                }
            }
            let tH = MLXArray(targetH)
            let tW = MLXArray(targetW)

            let hC = hCoords.asType(.float32)
            let wC = wCoords.asType(.float32)
            let normW = ((wC + 0.5) / tW) * 2 - 1
            let normH = ((hC + 0.5) / tH) * 2 - 1

            let grid = MLX.stacked([normW, normH], axis: -1)[.newAxis, 0..., .newAxis, 0...]

            let interp = Glm4SharedVision.gridSampleBicubic(pos2dNHWC, grid: grid)
            let adapted = interp.squeezed(axis: 0).squeezed(axis: 1).asType(posWeight.dtype)

            return embeddings + adapted
        }
    }

    fileprivate class Attention: Module {

        let numHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear

        public init(_ config: Glm4vMoeConfiguration.VisionConfiguration) {
            self.numHeads = config.numHeads
            self.headDim = config.hiddenSize / config.numHeads
            self.scale = pow(Float(headDim), -0.5)

            self._qkv.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSize * 3, bias: config.attentionBias)
            self._proj.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSize, bias: config.attentionBias)
        }

        public func callAsFunction(
            _ x: MLXArray, cuSeqlens: [Int], rotaryPositionEmbedding: MLXArray
        ) -> MLXArray {
            let sequenceLength = x.dim(0)

            let qkvOut = qkv(x)
            let qkvReshaped = qkvOut.reshaped(sequenceLength, 3, numHeads, -1)
                .transposed(1, 0, 2, 3)
            let parts = split(qkvReshaped, parts: 3, axis: 0)
            var q = parts[0].squeezed(axis: 0)
            var k = parts[1].squeezed(axis: 0)
            let v = parts[2].squeezed(axis: 0)

            q = applyRotaryPosEmbVision(q, freqs: rotaryPositionEmbedding)
            k = applyRotaryPosEmbVision(k, freqs: rotaryPositionEmbedding)

            let qT = q.transposed(1, 0, 2).expandedDimensions(axis: 0)
            let kT = k.transposed(1, 0, 2).expandedDimensions(axis: 0)
            let vT = v.transposed(1, 0, 2).expandedDimensions(axis: 0)

            var attnOutputs = [MLXArray]()
            for i in 0 ..< (cuSeqlens.count - 1) {
                let start = cuSeqlens[i]
                let end = cuSeqlens[i + 1]
                let qChunk = qT[0..., 0..., start ..< end, 0...]
                let kChunk = kT[0..., 0..., start ..< end, 0...]
                let vChunk = vT[0..., 0..., start ..< end, 0...]
                let output = MLXFast.scaledDotProductAttention(
                    queries: qChunk, keys: kChunk, values: vChunk,
                    scale: scale, mask: .none)
                attnOutputs.append(output)
            }

            let attnOutput = concatenated(attnOutputs, axis: 2)
                .transposed(0, 2, 1, 3)
                .reshaped(sequenceLength, -1)

            return proj(attnOutput)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {

        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        public init(dim: Int, hiddenDim: Int) {
            self._gate.wrappedValue = Linear(dim, hiddenDim, bias: false)
            self._up.wrappedValue = Linear(dim, hiddenDim, bias: false)
            self._down.wrappedValue = Linear(hiddenDim, dim, bias: false)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate class Glm4vMoeVisionBlock: Module {

        @ModuleInfo var norm1: RMSNorm
        @ModuleInfo var norm2: RMSNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo var mlp: MLP

        public init(_ config: Glm4vMoeConfiguration.VisionConfiguration) {
            self.norm1 = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
            self.norm2 = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
            self._attention.wrappedValue = Attention(config)
            self.mlp = MLP(dim: config.hiddenSize, hiddenDim: config.outHiddenSize)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray, cuSeqlens: [Int], rotaryPositionEmbedding: MLXArray
        ) -> MLXArray {
            var hiddenStates =
                hiddenStates
                + attention(
                    norm1(hiddenStates),
                    cuSeqlens: cuSeqlens,
                    rotaryPositionEmbedding: rotaryPositionEmbedding
                )
            hiddenStates = hiddenStates + mlp(norm2(hiddenStates))
            return hiddenStates
        }
    }

    fileprivate class PatchMerger: Module, UnaryLayer {

        @ModuleInfo var proj: Linear
        @ModuleInfo(key: "post_projection_norm") var postProjectionNorm: LayerNorm
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(dim: Int, contextDim: Int) {
            self._proj.wrappedValue = Linear(dim, dim, bias: false)
            self._postProjectionNorm.wrappedValue = LayerNorm(dimensions: dim)
            self._gate.wrappedValue = Linear(dim, contextDim, bias: false)
            self._up.wrappedValue = Linear(dim, contextDim, bias: false)
            self._down.wrappedValue = Linear(contextDim, dim, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = proj(x)
            h = gelu(postProjectionNorm(h))
            return down(silu(gate(h)) * up(h))
        }
    }

    fileprivate class VisionModel: Module {

        @ModuleInfo(key: "embeddings") var embeddings: VisionEmbeddings
        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        let rotaryPosEmb: QwenVL.VisionRotaryEmbedding
        @ModuleInfo(key: "blocks") var blocks: [Glm4vMoeVisionBlock]
        @ModuleInfo(key: "post_conv_layernorm") var postConvLayernorm: RMSNorm
        @ModuleInfo var downsample: Conv2d
        @ModuleInfo var merger: PatchMerger
        @ModuleInfo(key: "post_layernorm") var postLayernorm: RMSNorm

        let spatialMergeSize: Int

        public init(_ config: Glm4vMoeConfiguration.VisionConfiguration) {
            self.spatialMergeSize = config.spatialMergeSize

            self._embeddings.wrappedValue = VisionEmbeddings(config)
            self._patchEmbed.wrappedValue = PatchEmbed(config)

            let headDim = config.hiddenSize / config.numHeads
            self.rotaryPosEmb = QwenVL.VisionRotaryEmbedding(
                dimensions: headDim / 2, theta: 10_000)

            self._blocks.wrappedValue = (0 ..< config.depth).map { _ in
                Glm4vMoeVisionBlock(config)
            }

            self._postConvLayernorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)

            self._downsample.wrappedValue = Conv2d(
                inputChannels: config.hiddenSize,
                outputChannels: config.outHiddenSize,
                kernelSize: IntOrPair(config.spatialMergeSize),
                stride: IntOrPair(config.spatialMergeSize),
                bias: true)

            self._merger.wrappedValue = PatchMerger(
                dim: config.outHiddenSize,
                contextDim: config.intermediateSize)

            self._postLayernorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func rotaryPositionEmbedding(_ frames: [THW]) -> (MLXArray, MLXArray) {
            var positionIds = [MLXArray]()

            for row in frames {
                let (t, h, w) = row.values

                var hposIds = expandedDimensions(MLXArray(0 ..< h), axis: 1)
                hposIds = repeated(hposIds, count: w, axis: 1)
                hposIds =
                    hposIds
                    .reshaped(
                        h / spatialMergeSize, spatialMergeSize,
                        w / spatialMergeSize, spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                var wposIds = expandedDimensions(MLXArray(0 ..< w), axis: 0)
                wposIds = repeated(wposIds, count: h, axis: 0)
                wposIds =
                    wposIds
                    .reshaped(
                        h / spatialMergeSize, spatialMergeSize,
                        w / spatialMergeSize, spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                let stackedPosIds = stacked([hposIds, wposIds], axis: -1)
                positionIds.append(tiled(stackedPosIds, repetitions: [t, 1]))
            }

            let indices = concatenated(positionIds, axis: 0)
            let maxFrameSize = frames.lazy.map { max($0.h, $0.w) }.max() ?? 0
            let rotaryPosEmbFull = rotaryPosEmb(sequenceLength: maxFrameSize)[indices]

            return (rotaryPosEmbFull.reshaped(indices.dim(0), -1), indices)
        }

        public func callAsFunction(_ hiddenStates: MLXArray, frames: [THW]) -> MLXArray {
            var hiddenStates = patchEmbed(hiddenStates)
            hiddenStates = postConvLayernorm(hiddenStates)

            let (rotaryPosEmbedding, posIds) = rotaryPositionEmbedding(frames)

            var cuSeqlens = [0]
            var cumsum = 0
            for frame in frames {
                for _ in 0 ..< frame.t {
                    cumsum += frame.h * frame.w
                    cuSeqlens.append(cumsum)
                }
            }

            let hCoords = posIds[0..., 0]
            let wCoords = posIds[0..., 1]
            hiddenStates = embeddings(
                hiddenStates, frames: frames, hCoords: hCoords, wCoords: wCoords)

            for block in blocks {
                hiddenStates = block(
                    hiddenStates, cuSeqlens: cuSeqlens,
                    rotaryPositionEmbedding: rotaryPosEmbedding)
            }

            hiddenStates = postLayernorm(hiddenStates)

            let hiddenDim = hiddenStates.dim(-1)
            hiddenStates = hiddenStates.reshaped(
                -1, spatialMergeSize, spatialMergeSize, hiddenDim)
            hiddenStates = downsample(hiddenStates).reshaped(
                -1, downsample.weight.dim(0))

            hiddenStates = merger(hiddenStates)
            return hiddenStates
        }

        private func isMLXWeight(_ array: MLXArray) -> Bool {
            if array.ndim == 4 {
                let (outChannels, kH, kW) = (array.dim(0), array.dim(1), array.dim(2))
                return outChannels >= kH && outChannels >= kW && kH == kW
            } else if array.ndim == 5 {
                let (outChannels, kH, kW) = (array.dim(0), array.dim(2), array.dim(3))
                return outChannels >= kH && outChannels >= kW && kH == kW
            }
            return false
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()

            for (k, v) in weights {
                if k.contains("position_id") {
                    continue
                } else if k.contains("patch_embed.proj.weight")
                    || k.contains("downsample.weight")
                {
                    if isMLXWeight(v) {
                        sanitizedWeights[k] = v
                    } else {
                        if v.ndim == 5 {
                            sanitizedWeights[k] = v.transposed(0, 2, 3, 4, 1)
                        } else if v.ndim == 4 {
                            sanitizedWeights[k] = v.transposed(0, 2, 3, 1)
                        } else {
                            sanitizedWeights[k] = v
                        }
                    }
                } else {
                    sanitizedWeights[k] = v
                }
            }

            return sanitizedWeights
        }
    }
}

// MARK: - Processor
//
// Identical preprocessing to glm4v / glm_ocr. Reuses GlmOcrProcessorConfiguration and the
// Glm4vMessageGenerator (already defined in Glm4v.swift, same module). We register the
// existing Glm46VProcessor / GlmOcrProcessor for glm4v_moe — no new processor needed.

// MARK: - Model

/// Glm4vMoe VLM (GLM-4.5V etc.)
public class Glm4vMoe: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionModel: Glm4vMoeVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Glm4vMoeLanguage.LanguageModel

    public let config: Glm4vMoeConfiguration

    public var vocabularySize: Int { config.baseConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: Glm4vMoeConfiguration) {
        self.config = config
        self._visionModel.wrappedValue = Glm4vMoeVision.VisionModel(config.visionConfiguration)
        self._languageModel.wrappedValue = Glm4vMoeLanguage.LanguageModel(
            config.textConfiguration)
    }

    /// Compute 3D M-RoPE position IDs for prefill with images.
    /// Returns position_ids (3, batch, seq) and rope_deltas.
    private func getRopeIndex(
        inputIds: MLXArray, imageGridThw: [THW]?
    ) -> (MLXArray, MLXArray) {
        let batchSize = inputIds.dim(0)
        let seqLength = inputIds.dim(1)
        let spatialMergeSize = config.visionConfiguration.spatialMergeSize
        let imageTokenId = config.baseConfiguration.imageTokenId

        guard let imageGridThw, !imageGridThw.isEmpty else {
            let positions = MLXArray(0 ..< Int32(seqLength)).expandedDimensions(axis: 0)
            let positionIds = tiled(
                broadcast(positions, to: [batchSize, seqLength]).expandedDimensions(axis: 0),
                repetitions: [3, 1, 1])
            let deltas = MLXArray(Int32(0))
            return (positionIds, deltas)
        }

        precondition(batchSize == 1, "Glm4vMoe getRopeIndex only supports batchSize == 1")
        let positionIds = zeros([3, batchSize, seqLength], type: Int32.self)
        var imageIndex = 0
        var mropePositionDelta: Int = 0

        for batchIdx in 0 ..< batchSize {
            let inputTokens: [Int32] = inputIds[batchIdx].asArray(Int32.self)

            var dimT = [Int32]()
            var dimH = [Int32]()
            var dimW = [Int32]()
            dimT.reserveCapacity(seqLength)
            dimH.reserveCapacity(seqLength)
            dimW.reserveCapacity(seqLength)
            var st = 0
            var lastMax: Int32 = -1

            let appendTextPositions = { (count: Int) in
                guard count > 0 else { return }
                let base: Int32 = lastMax + 1
                for j in 0 ..< count {
                    let pos = base + Int32(j)
                    dimT.append(pos)
                    dimH.append(pos)
                    dimW.append(pos)
                }
                lastMax = base + Int32(count) - 1
            }

            while imageIndex < imageGridThw.count {
                guard let ed = inputTokens[st...].firstIndex(of: Int32(imageTokenId)) else {
                    break
                }

                let frame = imageGridThw[imageIndex]
                let llmGridT = frame.t
                let llmGridH = frame.h / spatialMergeSize
                let llmGridW = frame.w / spatialMergeSize
                imageIndex += 1

                appendTextPositions(ed - st)

                let imgOffset: Int32 = lastMax + 1
                for t in 0 ..< llmGridT {
                    for h in 0 ..< llmGridH {
                        for w in 0 ..< llmGridW {
                            dimT.append(Int32(t) + imgOffset)
                            dimH.append(Int32(h) + imgOffset)
                            dimW.append(Int32(w) + imgOffset)
                        }
                    }
                }
                let tMax = Int32(llmGridT - 1) + imgOffset
                let hMax = Int32(llmGridH - 1) + imgOffset
                let wMax = Int32(llmGridW - 1) + imgOffset
                lastMax = max(tMax, max(hMax, wMax))

                st = ed + llmGridT * llmGridH * llmGridW
            }

            appendTextPositions(inputTokens.count - st)

            positionIds[0, batchIdx] = MLXArray(dimT)
            positionIds[1, batchIdx] = MLXArray(dimH)
            positionIds[2, batchIdx] = MLXArray(dimW)

            mropePositionDelta = Int(lastMax) + 1 - inputTokens.count
        }

        let deltas = MLXArray(Int32(mropePositionDelta))
        return (positionIds, deltas)
    }

    private func inputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?, frames: [THW]?)
        -> MLXArray
    {
        guard let pixelValues, let frames else {
            languageModel._positionIds = nil
            languageModel._ropeDeltas = nil
            return languageModel.model.embedTokens(inputIds[.newAxis, .ellipsis])
        }

        let inputEmbeds = languageModel.model.embedTokens(inputIds)

        var hiddenStates = self.visionModel(pixelValues, frames: frames)

        if hiddenStates.ndim == 2 {
            hiddenStates = hiddenStates[.newAxis, 0..., 0...]
        }

        let merged = QwenVL.mergeInputIdsWithImageFeatures(
            inputIds: inputIds, inputEmbeds: inputEmbeds, imageFeatures: hiddenStates,
            imageTokenId: config.baseConfiguration.imageTokenId,
            videoTokenId: config.baseConfiguration.videoTokenId)

        let (positionIds, ropeDeltas) = getRopeIndex(
            inputIds: inputIds, imageGridThw: frames)
        languageModel._positionIds = positionIds
        languageModel._ropeDeltas = ropeDeltas

        return merged
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let dtype = visionModel.patchEmbed.proj.weight.dtype

        var allPixels: MLXArray?
        var allFrames: [THW] = []

        if let imagePixels = input.image?.pixels, let imageFrames = input.image?.frames {
            allPixels = imagePixels.asType(dtype)
            allFrames.append(contentsOf: imageFrames)
        }

        let inputEmbeddings = self.inputEmbeddings(
            inputIds: input.text.tokens, pixelValues: allPixels,
            frames: allFrames.isEmpty ? nil : allFrames)

        let result = languageModel(nil, cache: cache, inputEmbedding: inputEmbeddings)

        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Step 1: HuggingFace -> internal key remap (mirrors Glm4v.sanitize).
        var transformed = [String: MLXArray]()
        for (key, value) in weights {
            var k = key

            // Map visual -> vision_tower
            if k.contains("visual") && !k.contains("vision_tower") {
                k = k.replacingOccurrences(of: "model.", with: "")
                k = k.replacingOccurrences(of: "visual", with: "vision_tower")
            }

            // Map model.language_model -> language_model.model
            if k.contains("model.language_model") {
                k = k.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            }

            // Map lm_head -> language_model.lm_head
            if k.contains("lm_head") && !k.hasPrefix("language_model") {
                k = k.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            }

            transformed[k] = value
        }

        // Step 2: Stack per-expert tensors into switch_mlp (mirrors GLM4MOE.sanitize, but
        // over the post-remap key layout: language_model.model.layers.{l}.mlp.experts.{e}…).
        var sanitized = transformed
        let nLayers = config.textConfiguration.hiddenLayers
        let nExperts = config.textConfiguration.nRoutedExperts
        for l in 0 ..< nLayers {
            let prefix = "language_model.model.layers.\(l)"
            for n in ["gate_proj", "down_proj", "up_proj"] {
                for k in ["weight", "scales", "biases"] {
                    let probe = "\(prefix).mlp.experts.0.\(n).\(k)"
                    if sanitized[probe] != nil {
                        let toJoin = (0 ..< nExperts).map { e in
                            sanitized.removeValue(
                                forKey: "\(prefix).mlp.experts.\(e).\(n).\(k)")!
                        }
                        sanitized["\(prefix).mlp.switch_mlp.\(n).\(k)"] = MLX.stacked(toJoin)
                    }
                }
            }
        }

        // Step 3: Drop the optional MTP / next-token-prediction layer at index == hiddenLayers
        // (one past the last real decoder layer). Harmless if absent.
        let mtpPrefix = "language_model.model.layers.\(nLayers)"
        sanitized = sanitized.filter { !$0.key.hasPrefix(mtpPrefix) }

        // Step 4: Sanitize vision conv weight transposes.
        return visionModel.sanitize(weights: sanitized)
    }
}

// MARK: - Configuration

/// Configuration for ``Glm4vMoe``
public struct Glm4vMoeConfiguration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let moeIntermediateSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        private let _headDim: Int?
        // EXPLICIT head_dim (128). NOTE: attentionHeads * headDim ≠ hiddenSize (DeepSeek-style).
        public var headDim: Int { _headDim ?? (hiddenSize / attentionHeads) }
        public let vocabularySize: Int

        // MoE routing
        public let nRoutedExperts: Int
        public let numExpertsPerTok: Int
        private let _nSharedExperts: Int?
        public var nSharedExperts: Int { _nSharedExperts ?? 0 }
        public let firstKDenseReplace: Int
        public let nGroup: Int
        public let topkGroup: Int
        public let normTopkProb: Bool
        public let routedScalingFactor: Float
        private let _scoringFunc: String?
        public var scoringFunc: String { _scoringFunc ?? "sigmoid" }
        private let _topkMethod: String?
        public var topkMethod: String { _topkMethod ?? "noaux_tc" }

        // RoPE / norms
        private let _ropeTheta: Float?
        public var ropeTheta: Float { _ropeTheta ?? 10_000 }
        private let _partialRotaryFactor: Float?
        public var partialRotaryFactor: Float { _partialRotaryFactor ?? 0.5 }
        public let ropeScaling: RopeScaling
        public var mropeSection: [Int] { ropeScaling.mropeSection }
        private let _rmsNormEps: Float?
        public var rmsNormEps: Float { _rmsNormEps ?? 1e-5 }
        private let _attentionBias: Bool?
        public var attentionBias: Bool { _attentionBias ?? true }
        private let _useQkNorm: Bool?
        public var useQkNorm: Bool { _useQkNorm ?? false }

        // Threaded from the top-level config (see Glm4vMoeConfiguration.init).
        public var tieWordEmbeddings: Bool = false

        public struct RopeScaling: Codable, Sendable {
            public let mropeSection: [Int]
            enum CodingKeys: String, CodingKey {
                case mropeSection = "mrope_section"
            }
        }

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case moeIntermediateSize = "moe_intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case _headDim = "head_dim"
            case vocabularySize = "vocab_size"
            case nRoutedExperts = "n_routed_experts"
            case numExpertsPerTok = "num_experts_per_tok"
            case _nSharedExperts = "n_shared_experts"
            case firstKDenseReplace = "first_k_dense_replace"
            case nGroup = "n_group"
            case topkGroup = "topk_group"
            case normTopkProb = "norm_topk_prob"
            case routedScalingFactor = "routed_scaling_factor"
            case _scoringFunc = "scoring_func"
            case _topkMethod = "topk_method"
            case _ropeTheta = "rope_theta"
            case _partialRotaryFactor = "partial_rotary_factor"
            case ropeScaling = "rope_scaling"
            case _rmsNormEps = "rms_norm_eps"
            case _attentionBias = "attention_bias"
            case _useQkNorm = "use_qk_norm"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let depth: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHeads: Int
        public let patchSize: Int
        public let outHiddenSize: Int
        public let spatialMergeSize: Int
        public let temporalPatchSize: Int
        private let _imageSize: Int?
        public var imageSize: Int { _imageSize ?? 336 }
        private let _inChannels: Int?
        public var inChannels: Int { _inChannels ?? 3 }
        private let _rmsNormEps: Float?
        public var rmsNormEps: Float { _rmsNormEps ?? 1e-5 }
        private let _attentionBias: Bool?
        public var attentionBias: Bool { _attentionBias ?? false }

        enum CodingKeys: String, CodingKey {
            case depth
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHeads = "num_heads"
            case patchSize = "patch_size"
            case outHiddenSize = "out_hidden_size"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
            case _imageSize = "image_size"
            case _inChannels = "in_channels"
            case _rmsNormEps = "rms_norm_eps"
            case _attentionBias = "attention_bias"
        }
    }

    public struct BaseConfiguration: Codable, Sendable {
        public let modelType: String
        private let _vocabularySize: Int?
        private let _imageTokenId: Int?
        private let _videoTokenId: Int?
        private let _imageStartTokenId: Int?
        private let _imageEndTokenId: Int?
        private let _hiddenSize: Int?
        private let _tieWordEmbeddings: Bool?

        public var vocabularySize: Int { _vocabularySize ?? 151552 }
        public var imageTokenId: Int { _imageTokenId ?? 151363 }
        public var videoTokenId: Int { _videoTokenId ?? 151364 }
        public var imageStartTokenId: Int { _imageStartTokenId ?? 151339 }
        public var imageEndTokenId: Int { _imageEndTokenId ?? 151340 }
        public var hiddenSize: Int { _hiddenSize ?? 4096 }
        public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? false }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case _vocabularySize = "vocab_size"
            case _imageTokenId = "image_token_id"
            case _videoTokenId = "video_token_id"
            // vision_start_token_id / vision_end_token_id in config.json
            case _imageStartTokenId = "vision_start_token_id"
            case _imageEndTokenId = "vision_end_token_id"
            case _hiddenSize = "hidden_size"
            case _tieWordEmbeddings = "tie_word_embeddings"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let baseConfiguration: BaseConfiguration

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        var text = try container.decode(
            TextConfiguration.self, forKey: .textConfiguration)
        self.visionConfiguration = try container.decode(
            VisionConfiguration.self, forKey: .visionConfiguration)

        // BaseConfiguration overlaid at top level (fields may be absent).
        let base = try BaseConfiguration(from: decoder)
        self.baseConfiguration = base

        // tie_word_embeddings lives at the TOP level (not in text_config); thread it
        // into the text config so LanguageModel can decide whether to build lm_head.
        text.tieWordEmbeddings = base.tieWordEmbeddings
        self.textConfiguration = text
    }
}
