//
//  Glm4v.swift
//  mlx-swift-lm
//
//  port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/glm4v
//
//  GLM-4V family (e.g. GLM-4.6V-Flash). model_type "glm4v",
//  architecture "Glm4vForConditionalGeneration".
//
//  Adapted from GlmOcr.swift. The DENSE language decoder + M-RoPE machinery is
//  byte-for-byte equivalent to glm_ocr (glm4v language.py uses the same
//  "sectioned_even_odd" mrope as glm_ocr). The VISION tower differs:
//    * NO qk_norm in vision attention (glm_ocr has q_norm/k_norm)
//    * learned absolute position embedding bilinearly resampled onto the patch
//      grid (Glm4vVisionEmbeddings + grid_sample) — absent in glm_ocr
//    * post_conv_layernorm after patch_embed — absent in glm_ocr
//    * vision block RMSNorms use eps 1e-6 (hardcoded upstream), not config eps
//    * merger contextDim = intermediate_size (glm_ocr used out_hidden*in_channels)
//    * vision MLP hidden dim = out_hidden_size (glm_ocr used intermediate_size)
//    * all vision Linear layers are bias-free (attention_bias=false)
//

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Language

private enum Glm4vLanguage {

    // MARK: M-RoPE helpers

    /// Interleaved rotate_half: takes even/odd indices instead of splitting in half.
    static func rotateHalfInterleaved(_ x: MLXArray) -> MLXArray {
        let lastDim = x.dim(-1)
        let x1 = x[.ellipsis, stride(from: 0, to: lastDim, by: 2)]
        let x2 = x[.ellipsis, stride(from: 1, to: lastDim, by: 2)]
        let neg = -x2
        let stacked = MLX.stacked([neg, x1], axis: -1)
        return stacked.reshaped(x.shape)
    }

    /// repeat_interleave: [a,b,c] repeats=2 -> [a,a,b,b,c,c] along axis
    static func repeatInterleave(_ x: MLXArray, repeats: Int, axis: Int) -> MLXArray {
        let resolvedAxis = axis >= 0 ? axis : x.ndim + axis
        let expanded = expandedDimensions(x, axis: resolvedAxis + 1)
        var tileShape = [Int](repeating: 1, count: expanded.ndim)
        tileShape[resolvedAxis + 1] = repeats
        let t = tiled(expanded, repetitions: tileShape)
        var newShape = x.shape
        newShape[resolvedAxis] *= repeats
        return t.reshaped(newShape)
    }

    /// Apply rotary position embedding (language model style - interleaved).
    static func applyRotaryPosEmb(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        var cos = cos[0..., .newAxis, 0..., 0...]
        var sin = sin[0..., .newAxis, 0..., 0...]

        let halfDim = cos.dim(-1) / 2
        cos = repeatInterleave(cos[.ellipsis, ..<halfDim], repeats: 2, axis: -1)
        sin = repeatInterleave(sin[.ellipsis, ..<halfDim], repeats: 2, axis: -1)

        let rotaryDim = cos.dim(-1)
        let qRot = q[.ellipsis, ..<rotaryDim]
        let qPass = q[.ellipsis, rotaryDim...]
        let kRot = k[.ellipsis, ..<rotaryDim]
        let kPass = k[.ellipsis, rotaryDim...]

        let qEmbed = (qRot * cos) + (rotateHalfInterleaved(qRot) * sin)
        let kEmbed = (kRot * cos) + (rotateHalfInterleaved(kRot) * sin)

        return (
            concatenated([qEmbed, qPass], axis: -1),
            concatenated([kEmbed, kPass], axis: -1)
        )
    }

    // MARK: M-RoPE Rotary Embedding

    fileprivate class Glm4vRotaryEmbedding {
        let mropeSplitIndices: [Int]
        let invFreq: MLXArray
        let attentionScaling: Float

        init(_ config: Glm4vConfiguration.TextConfiguration) {
            var indices = [Int]()
            var cumsum = 0
            for s in config.ropeParameters.mropeSection.dropLast() {
                cumsum += s
                indices.append(cumsum)
            }
            self.mropeSplitIndices = indices

            let dim = Int(Float(config.headDim) * config.ropeParameters.partialRotaryFactor)
            let base = config.ropeParameters.ropeTheta
            self.attentionScaling = 1.0

            let p =
                MLXArray(stride(from: 0, to: dim, by: 2)).asType(.int64).asType(.float32)
                / Float(dim)
            self.invFreq = 1.0 / pow(base, p)
        }

        /// Apply M-RoPE: select different frequency dimensions for T, H, W.
        func applyMrope(_ freqs: MLXArray) -> MLXArray {
            let chunks = split(freqs, indices: mropeSplitIndices, axis: -1)
            let selected = chunks.enumerated().map { i, chunk in
                chunk[i % 3]
            }
            return concatenated(selected, axis: -1)
        }

        func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
            // positionIds: (3, batch, seq)
            let batchSize = positionIds.dim(1)

            var invFreqExpanded = invFreq[.newAxis, .newAxis, 0..., .newAxis].asType(.float32)
            invFreqExpanded = broadcast(
                invFreqExpanded,
                to: [3, batchSize, invFreq.dim(0), 1])

            let positionIdsExpanded = positionIds[0..., 0..., .newAxis, 0...].asType(.float32)

            let freqs = matmul(invFreqExpanded, positionIdsExpanded).transposed(0, 1, 3, 2)

            let mropeFreqs = applyMrope(freqs)

            let emb = concatenated([mropeFreqs, mropeFreqs], axis: -1)
            let cos = MLX.cos(emb) * attentionScaling
            let sin = MLX.sin(emb) * attentionScaling

            return (cos.asType(x.dtype), sin.asType(x.dtype))
        }
    }

    // MARK: Attention

    fileprivate class Attention: Module {

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        public init(_ args: Glm4vConfiguration.TextConfiguration) {
            let dim = args.hiddenSize
            self.heads = args.attentionHeads
            self.kvHeads = args.kvHeads
            self.headDim = args.headDim
            self.scale = pow(Float(headDim), -0.5)

            // GLM-4V uses attention_bias for q/k/v; o_proj is always bias-free.
            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)
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
            (queries, keys) = applyRotaryPosEmb(q: queries, k: keys, cos: cos, sin: sin)

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

    // MARK: MLP

    fileprivate class MLP: Module, UnaryLayer {

        @ModuleInfo(key: "gate_up_proj") var gateUpProj: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gateUpProj.wrappedValue = Linear(dimensions, hiddenDimensions * 2, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            let x = gateUpProj(x)
            let parts = split(x, parts: 2, axis: -1)
            return down(silu(parts[0]) * parts[1])
        }
    }

    // MARK: Decoder Layer

    fileprivate class Glm4vDecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_self_attn_layernorm") var postSelfAttnLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
        @ModuleInfo(key: "post_mlp_layernorm") var postMlpLayerNorm: RMSNorm

        public init(_ args: Glm4vConfiguration.TextConfiguration) {
            self._attention.wrappedValue = Attention(args)
            self.mlp = MLP(
                dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postSelfAttnLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postMlpLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
            positionEmbeddings: (MLXArray, MLXArray)
        ) -> MLXArray {
            var r = x
            var h = attention(
                inputLayerNorm(x), mask: mask, cache: cache,
                positionEmbeddings: positionEmbeddings)
            h = postSelfAttnLayerNorm(h)
            h = r + h
            r = h
            h = postAttentionLayerNorm(h)
            h = mlp(h)
            h = postMlpLayerNorm(h)
            h = r + h
            return h
        }
    }

    // MARK: Text Model

    fileprivate class Glm4vTextModel: Module {

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        fileprivate let layers: [Glm4vDecoderLayer]
        fileprivate let norm: RMSNorm
        let rotaryEmb: Glm4vRotaryEmbedding

        public init(_ args: Glm4vConfiguration.TextConfiguration) {
            precondition(args.vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

            self.layers = (0 ..< args.hiddenLayers)
                .map { _ in Glm4vDecoderLayer(args) }
            self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self.rotaryEmb = Glm4vRotaryEmbedding(args)
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
        @ModuleInfo var model: Glm4vTextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int]
        var _positionIds: MLXArray?
        var _ropeDeltas: MLXArray?

        public init(_ args: Glm4vConfiguration.TextConfiguration) {
            self.model = Glm4vTextModel(args)

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

private enum Glm4vVision {

    /// Pure-MLX bilinear grid_sample, matching mlx_vlm kernels._grid_sample_mlx.
    /// x:    (B, H, W, C)   — NHWC layout (already permuted by caller)
    /// grid: (B, gN, gM, 2) — last dim is (x, y) in normalized [-1, 1] coords
    /// returns (B, gN, gM, C)

    static fileprivate func applyRotaryPosEmbVision(
        _ tensor: MLXArray, freqs: MLXArray
    ) -> MLXArray {
        var cosVal = MLX.cos(freqs)
        var sinVal = MLX.sin(freqs)

        // freqs: (seq, dim/2) -> expand to (seq, 1, dim) for broadcasting with (seq, heads, dim)
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

        init(_ config: Glm4vConfiguration.VisionConfiguration) {
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

    /// Learned absolute position embedding, bilinearly resampled to the patch grid.
    /// Mirrors Glm4vVisionEmbeddings in mlx_vlm/models/glm4v/vision.py.
    fileprivate class VisionEmbeddings: Module {
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        let embedDim: Int
        let imageSize: Int
        let patchSize: Int
        let numPositions: Int

        init(_ config: Glm4vConfiguration.VisionConfiguration) {
            self.embedDim = config.hiddenSize
            self.imageSize = config.imageSize
            self.patchSize = config.patchSize
            let numPatches = (config.imageSize / config.patchSize)
                * (config.imageSize / config.patchSize)
            self.numPositions = numPatches
            self._positionEmbedding.wrappedValue = Embedding(
                embeddingCount: numPatches, dimensions: config.hiddenSize)
        }

        /// embeddings: (totalSeq, embedDim)
        /// frames:     image grid dims, used to derive target H/W per patch
        /// hCoords/wCoords: (totalSeq,) patch row/col indices within each image grid
        func callAsFunction(
            _ embeddings: MLXArray, frames: [THW], hCoords: MLXArray, wCoords: MLXArray
        ) -> MLXArray {
            let totalSeq = hCoords.dim(0)
            if totalSeq == 0 {
                return embeddings
            }

            let posWeight = positionEmbedding.weight  // (numPositions, embedDim)
            let hiddenSize = posWeight.dim(1)
            let origSizeSq = posWeight.dim(0)
            let origSize = Int(Double(origSizeSq).squareRoot().rounded())

            // (1, embedDim, origSize, origSize) then to NHWC for grid_sample.
            let pos2d = posWeight.reshaped(origSize, origSize, hiddenSize)
                .transposed(2, 0, 1)[.newAxis, 0..., 0..., 0...]
                .asType(.float32)
            let pos2dNHWC = pos2d.transposed(0, 2, 3, 1)  // (1, origSize, origSize, embedDim)

            // Per-patch target image dimensions (in patch units): grid h, grid w.
            var targetH = [Float]()
            var targetW = [Float]()
            targetH.reserveCapacity(totalSeq)
            targetW.reserveCapacity(totalSeq)
            for frame in frames {
                // one image contributes t*h*w patch rows; h_coords length == sum.
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

            // grid: (1, totalSeq, 1, 2) with last dim (x=w, y=h)
            let grid = MLX.stacked([normW, normH], axis: -1)[.newAxis, 0..., .newAxis, 0...]

            let interp = Glm4SharedVision.gridSampleBicubic(pos2dNHWC, grid: grid)
            // (1, totalSeq, 1, embedDim) -> (totalSeq, embedDim)
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

        public init(_ config: Glm4vConfiguration.VisionConfiguration) {
            self.numHeads = config.numHeads
            self.headDim = config.hiddenSize / config.numHeads
            self.scale = pow(Float(headDim), -0.5)

            // glm4v vision: attention_bias = false, NO q_norm/k_norm.
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

            // NOTE: glm4v vision has no qk_norm (unlike glm_ocr).
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

        // glm4v vision MLP hidden dim == out_hidden_size (NOT intermediate_size).
        public init(dim: Int, hiddenDim: Int) {
            self._gate.wrappedValue = Linear(dim, hiddenDim, bias: false)
            self._up.wrappedValue = Linear(dim, hiddenDim, bias: false)
            self._down.wrappedValue = Linear(hiddenDim, dim, bias: false)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate class Glm4vVisionBlock: Module {

        @ModuleInfo var norm1: RMSNorm
        @ModuleInfo var norm2: RMSNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo var mlp: MLP

        public init(_ config: Glm4vConfiguration.VisionConfiguration) {
            // Upstream hardcodes eps=1e-6 for the block norms (not config eps).
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
        @ModuleInfo(key: "blocks") var blocks: [Glm4vVisionBlock]
        @ModuleInfo(key: "post_conv_layernorm") var postConvLayernorm: RMSNorm
        @ModuleInfo var downsample: Conv2d
        @ModuleInfo var merger: PatchMerger
        @ModuleInfo(key: "post_layernorm") var postLayernorm: RMSNorm

        let spatialMergeSize: Int

        public init(_ config: Glm4vConfiguration.VisionConfiguration) {
            self.spatialMergeSize = config.spatialMergeSize

            self._embeddings.wrappedValue = VisionEmbeddings(config)
            self._patchEmbed.wrappedValue = PatchEmbed(config)

            let headDim = config.hiddenSize / config.numHeads
            self.rotaryPosEmb = QwenVL.VisionRotaryEmbedding(
                dimensions: headDim / 2, theta: 10_000)

            self._blocks.wrappedValue = (0 ..< config.depth).map { _ in
                Glm4vVisionBlock(config)
            }

            self._postConvLayernorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)

            self._downsample.wrappedValue = Conv2d(
                inputChannels: config.hiddenSize,
                outputChannels: config.outHiddenSize,
                kernelSize: IntOrPair(config.spatialMergeSize),
                stride: IntOrPair(config.spatialMergeSize),
                bias: true)

            // glm4v: merger contextDim == intermediate_size.
            self._merger.wrappedValue = PatchMerger(
                dim: config.outHiddenSize,
                contextDim: config.intermediateSize)

            self._postLayernorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        /// Returns (rotaryPositionEmbedding, posIds) where posIds is (seq, 2)
        /// holding the (h, w) patch coordinates used by the learned embeddings.
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

            let indices = concatenated(positionIds, axis: 0)  // (seq, 2)
            let maxFrameSize = frames.lazy.map { max($0.h, $0.w) }.max() ?? 0
            let rotaryPosEmbFull = rotaryPosEmb(sequenceLength: maxFrameSize)[indices]

            return (rotaryPosEmbFull.reshaped(indices.dim(0), -1), indices)
        }

        public func callAsFunction(_ hiddenStates: MLXArray, frames: [THW]) -> MLXArray {
            var hiddenStates = patchEmbed(hiddenStates)
            hiddenStates = postConvLayernorm(hiddenStates)

            let (rotaryPosEmbedding, posIds) = rotaryPositionEmbedding(frames)

            // Compute cu_seqlens from frames (one entry per temporal slice).
            var cuSeqlens = [0]
            var cumsum = 0
            for frame in frames {
                for _ in 0 ..< frame.t {
                    cumsum += frame.h * frame.w
                    cuSeqlens.append(cumsum)
                }
            }

            // Add learned absolute position embedding (bilinear-resampled).
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

            // Spatial merge via Conv2d downsample.
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

/// Glm4v VLM `UserInputProcessor`. Identical preprocessing to glm_ocr; reuses the
/// `GlmOcrProcessorConfiguration` (preprocessor_config.json fields are the same).
// MARK: - Model

/// Glm4v VLM (GLM-4.6V-Flash etc.)
public class Glm4v: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionModel: Glm4vVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Glm4vLanguage.LanguageModel

    public let config: Glm4vConfiguration

    public var vocabularySize: Int { config.baseConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: Glm4vConfiguration) {
        self.config = config
        self._visionModel.wrappedValue = Glm4vVision.VisionModel(config.visionConfiguration)
        self._languageModel.wrappedValue = Glm4vLanguage.LanguageModel(config.textConfiguration)
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

        precondition(batchSize == 1, "Glm4v getRopeIndex only supports batchSize == 1")
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
        // Step 1: Transform keys from HuggingFace format to internal format.
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

            // Skip any "next-n"/MTP prediction head at the layer index == hiddenLayers
            // (one past the last real decoder layer). Harmless if absent.
            if k.contains("layers.\(config.textConfiguration.hiddenLayers).") {
                continue
            }

            transformed[k] = value
        }

        // Step 2: Sanitize vision weights (conv weight transposes).
        return visionModel.sanitize(weights: transformed)
    }
}

// MARK: - Configuration

/// Configuration for ``Glm4v``
public struct Glm4vConfiguration: Codable, Sendable {

    public struct RopeParameters: Codable, Sendable {
        public let mropeSection: [Int]
        private let _partialRotaryFactor: Float?
        public var partialRotaryFactor: Float { _partialRotaryFactor ?? 0.5 }
        private let _ropeTheta: Float?
        public var ropeTheta: Float { _ropeTheta ?? 10_000 }

        enum CodingKeys: String, CodingKey {
            case mropeSection = "mrope_section"
            case _partialRotaryFactor = "partial_rotary_factor"
            case _ropeTheta = "rope_theta"
        }
    }

    public struct TextConfiguration: Codable, Sendable {
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        private let _headDim: Int?
        public var headDim: Int { _headDim ?? (hiddenSize / attentionHeads) }
        public let vocabularySize: Int
        public let ropeParameters: RopeParameters
        private let _rmsNormEps: Float?
        public var rmsNormEps: Float { _rmsNormEps ?? 1e-5 }
        public var ropeTheta: Float { ropeParameters.ropeTheta }
        private let _attentionBias: Bool?
        public var attentionBias: Bool { _attentionBias ?? true }
        // Decoded from the top-level config (see Glm4vConfiguration.init).
        public var tieWordEmbeddings: Bool = false

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case _headDim = "head_dim"
            case vocabularySize = "vocab_size"
            case ropeParameters = "rope_parameters"
            case _rmsNormEps = "rms_norm_eps"
            case _attentionBias = "attention_bias"
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
            case _imageStartTokenId = "image_start_token_id"
            case _imageEndTokenId = "image_end_token_id"
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

// MARK: - Message Generator

/// Message Generator for Glm4v
public struct Glm4vMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        [
            "role": message.role.rawValue,
            "content": [
                ["type": "text", "text": message.content]
            ]
                + message.images.map { _ in
                    ["type": "image"]
                },
        ]
    }
}
