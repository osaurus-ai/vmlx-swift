// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen2_5_vl

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Language

private enum Language {

    /// Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors
    static private func applyMultimodalRotaryPositionEmbedding(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray,
        positionIds: MLXArray, mropeSection: [Int]
    ) -> (MLXArray, MLXArray) {
        var cos = cos[positionIds]
        var sin = sin[positionIds]

        cos =
            concatenated(
                // [m[i % 3] for i, m in enumerate(mx.split(cos, mrope_section, axis=-1))]
                split(cos, indices: mropeSection, axis: -1).enumerated().map { i, m in m[i % 3] },
                axis: -1
            )[0..., .newAxis, 0..., 0...]

        sin =
            concatenated(
                split(sin, indices: mropeSection, axis: -1).enumerated().map { i, m in m[i % 3] },
                axis: -1
            )[0..., .newAxis, 0..., 0...]

        // Apply rotary embedding
        let qEmbed = (q * cos) + (QwenVL.rotateHalf(q) * sin)
        let kEmbed = (k * cos) + (QwenVL.rotateHalf(k) * sin)
        return (qEmbed, kEmbed)
    }

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

        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        public init(_ args: Qwen25VLConfiguration.TextConfiguration) {
            let dim = args.hiddenSize
            self.heads = args.attentionHeads
            self.kvHeads = args.kvHeads
            self.headDim = dim / heads
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            self.mropeSection = args.mropeSection

            self._rotaryEmbedding.wrappedValue = RoPE(
                dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            // prepare the queries, keys and values for the attention computation
            queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            queries = applyRotaryPosition(rotaryEmbedding, to: queries, cache: cache)
            keys = applyRotaryPosition(rotaryEmbedding, to: keys, cache: cache)

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

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate class Qwen25VLDecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        public init(_ args: Qwen25VLConfiguration.TextConfiguration) {
            self._attention.wrappedValue = Attention(args)
            self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            r = mlp(postAttentionLayerNorm(h))
            let out = h + r
            return out
        }
    }

    fileprivate class Qwen25Model: Module {

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        fileprivate let layers: [Qwen25VLDecoderLayer]
        fileprivate let norm: RMSNorm

        public init(_ args: Qwen25VLConfiguration.TextConfiguration) {
            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

            self.layers = (0 ..< args.hiddenLayers)
                .map { _ in
                    Qwen25VLDecoderLayer(args)
                }
            self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
            forward(hiddenStates: embedTokens(inputs), cache: cache)
        }

        func embeddingsForward(_ inputEmbedding: MLXArray?, cache: [KVCache]? = nil) throws
            -> MLXArray
        {
            guard let inputEmbedding else {
                throw VLMError.processing(
                    "Qwen2.5-VL prepare requires input ids or input embeddings.")
            }
            return forward(hiddenStates: inputEmbedding, cache: cache)
        }

        private func forward(hiddenStates: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
            var h: MLXArray
            h = hiddenStates

            let mask = createAttentionMask(h: h, cache: cache?.first)

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }

            return norm(h)
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: Qwen25Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int]

        public init(_ args: Qwen25VLConfiguration.TextConfiguration) {
            self.model = Qwen25Model(args)

            if !args.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
            }

            self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        }

        public func callAsFunction(
            _ inputs: MLXArray, cache: [KVCache]? = nil
        ) -> LMOutput {
            var out = model(inputs, cache: cache)
            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }
            return LMOutput(logits: out)
        }

        func embeddingsForward(_ inputEmbedding: MLXArray?, cache: [KVCache]? = nil) throws
            -> LMOutput
        {
            var out = try model.embeddingsForward(inputEmbedding, cache: cache)
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

enum Qwen25Vision {

    static fileprivate func applyMultimodalRotaryPositionEmbedding(
        _ tensor: MLXArray, freqs: MLXArray
    ) -> MLXArray {
        var cos = cos(freqs)
        var sin = sin(freqs)

        cos = expandedDimensions(cos, axis: 1)
        cos = tiled(cos, repetitions: [1, 1, 2])
        cos = expandedDimensions(cos, axis: 0)

        sin = expandedDimensions(sin, axis: 1)
        sin = tiled(sin, repetitions: [1, 1, 2])
        sin = expandedDimensions(sin, axis: 0)

        let output = (tensor * cos) + (QwenVL.rotateHalf(tensor) * sin)
        return output.asType(tensor.dtype)
    }

    final class PatchMerger: Module, UnaryLayer {
        let hiddenSize: Int
        @ModuleInfo(key: "ln_q") var layerNormQ: RMSNorm
        @ModuleInfo var mlp: (Linear, GELU, Linear)

        init(dimensions: Int, contextDimensions: Int, spatialMergeSize: Int) {
            self.hiddenSize = contextDimensions * (spatialMergeSize * spatialMergeSize)
            self._layerNormQ.wrappedValue = RMSNorm(dimensions: contextDimensions, eps: 1e-6)
            self.mlp = (
                Linear(hiddenSize, hiddenSize),
                GELU(),
                Linear(hiddenSize, dimensions)
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var x = layerNormQ(x).reshaped(-1, hiddenSize)
            x = mlp.0(x)
            x = mlp.1(x)
            x = mlp.2(x)
            return x
        }
    }

    final class Attention: Module {

        let numHeads: Int
        let scale: Float

        @ModuleInfo(key: "qkv") var qkv: Linear
        @ModuleInfo(key: "proj") var proj: Linear

        public init(dims: Int, numHeads: Int) {
            self.numHeads = numHeads
            let headDim = dims / numHeads
            self.scale = pow(Float(headDim), -0.5)

            self._qkv.wrappedValue = Linear(dims, 3 * dims, bias: true)
            self._proj.wrappedValue = Linear(dims, dims)
        }

        public func callAsFunction(
            _ x: MLXArray, attentionMask: MLXArray, rotaryPositionEmbedding: MLXArray
        ) -> MLXArray {
            let sequenceLength = x.dim(0)

            let qkv = qkv(x)
            let s = split(qkv, parts: 3, axis: -1)
            var (q, k, v) = (s[0], s[1], s[2])

            q = q.reshaped(sequenceLength, numHeads, -1)
            k = k.reshaped(sequenceLength, numHeads, -1)
            v = v.reshaped(sequenceLength, numHeads, -1)

            q = applyMultimodalRotaryPositionEmbedding(q, freqs: rotaryPositionEmbedding)
            k = applyMultimodalRotaryPositionEmbedding(k, freqs: rotaryPositionEmbedding)

            q = q.reshaped(1, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)
            k = k.reshaped(1, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)
            v = v.reshaped(1, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)

            let output = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: .none
            )
            .transposed(0, 2, 1, 3)
            .reshaped(sequenceLength, -1)

            return proj(output)
        }
    }

    final class MLP: Module, UnaryLayer {

        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    final class Qwen25VLVisionBlock: Module {

        @ModuleInfo var norm1: RMSNorm
        @ModuleInfo var norm2: RMSNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo var mlp: MLP

        public init(_ config: Qwen25VLConfiguration.VisionConfiguration) {
            self.norm1 = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
            self.norm2 = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)

            self._attention.wrappedValue = Attention(
                dims: config.hiddenSize, numHeads: config.numHeads)

            self.mlp = MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray, attentionMask: MLXArray, rotaryPositionEmbedding: MLXArray
        ) -> MLXArray {
            var hiddenStates =
                hiddenStates
                + attention(
                    norm1(hiddenStates),
                    attentionMask: attentionMask,
                    rotaryPositionEmbedding: rotaryPositionEmbedding
                )
            hiddenStates = hiddenStates + mlp(norm2(hiddenStates))
            return hiddenStates
        }
    }

    final class VisionModel: Module {

        @ModuleInfo(key: "patch_embed") var patchEmbed: QwenVL.PatchEmbed
        @ModuleInfo(key: "rotary_pos_emb") var rotaryPositionEmbedding: QwenVL.VisionRotaryEmbedding
        @ModuleInfo(key: "blocks") var blocks: [Qwen25VLVisionBlock]
        @ModuleInfo(key: "merger") var patchMerger: PatchMerger

        let spatialMergeSize: Int
        let windowSize: Int
        let patchSize: Int
        let spatialMergeUnit: Int
        let fullattBlockIndexes: [Int]

        public init(_ config: Qwen25VLConfiguration.VisionConfiguration) {
            self.spatialMergeSize = config.spatialMergeSize
            self.windowSize = config.windowSize
            self.patchSize = config.patchSize
            self.spatialMergeUnit = config.spatialMergeSize * config.spatialMergeSize
            self.fullattBlockIndexes = config.fullattBlockIndexes

            self._patchEmbed.wrappedValue = QwenVL.PatchEmbed(
                patchSize: config.patchSize,
                temporalPatchSize: config.temporalPatchSize,
                inChannels: config.inChannels,
                hiddenSize: config.hiddenSize)

            let headDimensions = config.hiddenSize / config.numHeads
            self._rotaryPositionEmbedding.wrappedValue = QwenVL.VisionRotaryEmbedding(
                dimensions: headDimensions / 2, theta: 10_000)

            self._blocks.wrappedValue = (0 ..< config.depth).map { _ in
                Qwen25VLVisionBlock(config)
            }
            self._patchMerger.wrappedValue = PatchMerger(
                dimensions: config.outHiddenSize, contextDimensions: config.hiddenSize,
                spatialMergeSize: config.spatialMergeSize)
        }

        func rotaryPositionEmbedding(_ frames: [THW]) -> MLXArray {
            var positionIds = [MLXArray]()

            for row in frames {
                let (t, h, w) = row.values

                var hposIds = expandedDimensions(MLXArray(0 ..< h), axis: 1)
                hposIds = repeated(hposIds, count: w, axis: 1)
                hposIds =
                    hposIds
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                var wposIds = expandedDimensions(MLXArray(0 ..< w), axis: 0)
                wposIds = repeated(wposIds, count: h, axis: 0)
                wposIds =
                    wposIds
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                let stackedPosIds = stacked([hposIds, wposIds], axis: -1)
                positionIds.append(tiled(stackedPosIds, repetitions: [t, 1]))
            }

            let indices = concatenated(positionIds, axis: 0)
            let maxFrameSize = frames.lazy.map { max($0.h, $0.w) }.max() ?? 0
            let rotaryPositionEmbedFull = rotaryPositionEmbedding(sequenceLength: maxFrameSize)[
                indices]

            return rotaryPositionEmbedFull.reshaped(indices.dim(0), -1)
        }

        func getWindowIndex(_ frames: [THW]) -> (MLXArray, MLXArray) {
            var windowIndex = [MLXArray]()
            var cuWindowSeqlens = [0]
            var windowIndexId = 0
            let vitMergerWindowSize = windowSize / spatialMergeSize / patchSize

            for frame in frames {
                let (gridT, gridH, gridW) = frame.values
                let llmGridH = gridH / spatialMergeSize
                let llmGridW = gridW / spatialMergeSize

                let index = MLXArray(0 ..< (gridT * llmGridH * llmGridW)).reshaped(
                    gridT, llmGridH, llmGridW)

                let padH = vitMergerWindowSize - llmGridH % vitMergerWindowSize
                let padW = vitMergerWindowSize - llmGridW % vitMergerWindowSize
                let numWindowsH = (llmGridH + padH) / vitMergerWindowSize
                let numWindowsW = (llmGridW + padW) / vitMergerWindowSize

                // Pad the index
                let indexPadded = padded(
                    index,
                    widths: [[0, 0], [0, padH], [0, padW]],
                    mode: .constant,
                    value: MLXArray(-100)
                )

                // Reshape and transpose
                let indexReshaped = indexPadded.reshaped(
                    gridT,
                    numWindowsH,
                    vitMergerWindowSize,
                    numWindowsW,
                    vitMergerWindowSize
                )

                let indexTransposed = indexReshaped.transposed(0, 1, 3, 2, 4).reshaped(
                    gridT,
                    numWindowsH * numWindowsW,
                    vitMergerWindowSize,
                    vitMergerWindowSize
                )

                // Calculate sequence lengths
                let seqlens = sum(indexTransposed .!= -100, axes: [2, 3]).reshaped(-1)

                // Get valid indices
                let indexFlattened = indexTransposed.flattened()
                let validIndices = indexFlattened.asArray(Int.self).enumerated()
                    .filter { $0.element != -100 }
                    .map { $0.offset }

                let validValues = indexFlattened[MLXArray(validIndices)]

                // Add to window index
                windowIndex.append(validValues + windowIndexId)

                // Update cumulative sequence lengths
                let cuSeqlensTmp =
                    cumsum(seqlens, axis: 0) * spatialMergeUnit + cuWindowSeqlens.last!
                cuWindowSeqlens.append(contentsOf: cuSeqlensTmp.asArray(Int.self))

                windowIndexId += gridT * llmGridH * llmGridW
            }

            // Concatenate all window indices
            let combinedWindowIndex = concatenated(windowIndex, axis: 0)
            let cuWindowSeqlensArray = MLXArray(cuWindowSeqlens)

            // Get unique values in cuWindowSeqlens
            var seen = Set<Int>()
            var uniqueIndices = [Int]()

            for (i, value) in cuWindowSeqlens.enumerated() {
                if !seen.contains(value) {
                    seen.insert(value)
                    uniqueIndices.append(i)
                }
            }

            let uniqueCuWindowSeqlens = cuWindowSeqlensArray[MLXArray(uniqueIndices)]

            return (combinedWindowIndex, uniqueCuWindowSeqlens)
        }

        func attentionMask(sequenceLength: Int, cuSeqlens: MLXArray) -> MLXArray {
            // Create attention mask
            let attentionMask = full(
                [1, sequenceLength, sequenceLength],
                values: false)

            // Update mask for each sequence
            let cuSeqlens = cuSeqlens.asArray(Int.self)
            for i in 1 ..< cuSeqlens.count {
                let start = cuSeqlens[i - 1]
                let end = cuSeqlens[i]
                attentionMask[0..., start ..< end, start ..< end] = MLXArray(true)
            }

            return attentionMask
        }

        public func callAsFunction(_ hiddenStates: MLXArray, frames: [THW]) -> MLXArray {
            var hiddenStates = patchEmbed(hiddenStates)
            let rotaryPosEmb = rotaryPositionEmbedding(frames)

            // Get window indices and sequence lengths
            let (windowIndex, cuWindowSeqlens) = getWindowIndex(frames)

            // prepare attention masks
            let seqLen = hiddenStates.dim(0)
            var cuSeqlens = [0]
            for frame in frames {
                let seqLen = frame.h * frame.w
                cuSeqlens.append(
                    contentsOf: Array(repeating: seqLen, count: frame.t).map {
                        cuSeqlens.last! + $0
                    })
            }
            let cuSeqlensArray = MLXArray(cuSeqlens)

            let fullAttentionMask = attentionMask(sequenceLength: seqLen, cuSeqlens: cuSeqlensArray)
            let windowAttentionMask = attentionMask(
                sequenceLength: seqLen, cuSeqlens: cuWindowSeqlens)

            // Reshape and reindex hidden states
            hiddenStates = hiddenStates.reshaped(seqLen / spatialMergeUnit, spatialMergeUnit, -1)
            hiddenStates = hiddenStates[windowIndex, 0..., 0...]
            hiddenStates = hiddenStates.reshaped(seqLen, -1)

            // Reshape and reindex rotary position embeddings
            var rotaryPosEmbReshaped = rotaryPosEmb.reshaped(
                seqLen / spatialMergeUnit, spatialMergeUnit, -1)
            rotaryPosEmbReshaped = rotaryPosEmbReshaped[windowIndex, 0..., 0...]
            rotaryPosEmbReshaped = rotaryPosEmbReshaped.reshaped(seqLen, -1)

            // Process through blocks
            for (i, block) in blocks.enumerated() {
                // Use full attention for specific blocks, window attention for others
                let attentionMask =
                    fullattBlockIndexes.contains(i) ? fullAttentionMask : windowAttentionMask

                hiddenStates = block(
                    hiddenStates,
                    attentionMask: attentionMask,
                    rotaryPositionEmbedding: rotaryPosEmbReshaped
                )
            }

            // Apply patch merger
            hiddenStates = patchMerger(hiddenStates)

            // Reorder back to original sequence
            let reverseIndices = argSort(windowIndex, axis: 0)
            hiddenStates = hiddenStates[reverseIndices, 0...]

            return hiddenStates
        }

        private func isMLXWeight(_ array: MLXArray) -> Bool {
            if array.ndim != 4, array.ndim != 5 {
                return false
            }

            if array.dim(-1) == 3 {
                return true
            }

            let (outChannels, kH, kW) = (array.dim(1), array.dim(2), array.dim(3))
            return outChannels >= kH && outChannels >= kW && kH == kW
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()

            for (k, v) in weights {
                if k.contains("position_id") {
                    // Remove unused position_ids
                    continue
                } else if k.contains("patch_embed.proj.weight") {
                    // PyTorch conv2d weight tensors have shape:
                    //   [B, out_channels, in_channels, kH, KW]
                    // MLX conv2d expects the weight be of shape:
                    //   [B, out_channels, kH, KW, in_channels]
                    if isMLXWeight(v) {
                        sanitizedWeights[k] = v
                    } else {
                        sanitizedWeights[k] = v.transposed(0, 2, 3, 4, 1)
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

/// Qwen2.5VL VLM `UserInputProcessor`.
///
/// This is meant to be used with ``Qwen25VL`` and is typically created by ``VLMModelFactory``.
public struct Qwen25VLProcessor: UserInputProcessor {
    private let config: Qwen25VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Qwen25VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    func preprocess(image: CIImage, resizedSize: CGSize) -> CIImage {
        image
            .toSRGB()
            .resampled(to: resizedSize, method: .bicubic)
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        // First apply the user requested resizing, etc. if any
        let images = images.map { MediaProcessing.apply($0, processing: processing) }

        // image_processing_qwen2_vl._preprocess
        let (extentH, extentW) = try QwenVL.intExtent(images[0].extent.size)
        let (resizedHeight, resizedWidth) = try QwenVL.targetSize(
            height: extentH, width: extentW,
            factor: config.patchSize * config.mergeSize,
            minPixels: config.size.minPixels, maxPixels: config.size.maxPixels)
        let resizedSize = CGSize(width: resizedWidth, height: resizedHeight)

        // Process images
        let processedImages =
            images
            .map {
                MediaProcessing.inSRGBToneCurveSpace($0)
            }
            .map {
                return MediaProcessing.resampleBicubic($0, to: resizedSize)
            }
            .map {
                MediaProcessing.normalize(
                    $0, mean: config.imageMeanTuple, std: config.imageStdTuple)
            }
            .map {
                MediaProcessing.asMLXArray($0)
            }

        return try QwenVL.patchify(
            images: processedImages, mergeSize: config.mergeSize, patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Qwen2VLMessageGenerator().generate(from: input)

        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)

        // Text-only input
        if input.images.isEmpty, input.videos.isEmpty {
            return LMInput(
                tokens: MLXArray(promptTokens),
                cacheScopeSalt: cacheScopeSalt(from: input.additionalContext))
        }

        // Process images if any
        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            let imagePixelsConcatenated = concatenated(imagePixelsAndFrames.map { $0.0 })
            processedImage = LMInput.ProcessedImage(
                pixels: imagePixelsConcatenated, frames: imagePixelsAndFrames.map { $0.1 })

            if let imageFrames = processedImage?.frames {
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens, frames: imageFrames, paddingToken: "<|image_pad|>",
                    mergeSize: config.mergeSize, tokenizer: tokenizer)
            }
        }

        // Process videos if any
        var processedVideo: LMInput.ProcessedVideo?
        if !input.videos.isEmpty {
            var videosAsImageSequences = [[MLXArray]]()
            for video in input.videos {

                var resizedSize: CGSize = .zero

                let imageSequence = try await MediaProcessing.asProcessedSequence(
                    video, targetFPS: { _ in Double(2) }
                ) { frame in
                    // first apply the user requested resizing, etc. if any
                    let resizedImage = MediaProcessing.apply(
                        frame.frame, processing: input.processing)
                    if resizedSize == .zero {
                        let (extentH, extentW) = try QwenVL.intExtent(resizedImage.extent.size)
                        let (resizedHeight, resizedWidth) = try QwenVL.targetSize(
                            height: extentH, width: extentW,
                            factor: config.patchSize * config.mergeSize,
                            minPixels: config.minPixels, maxPixels: config.maxPixels)
                        resizedSize = CGSize(width: resizedWidth, height: resizedHeight)
                    }
                    let processedImage = preprocess(image: resizedImage, resizedSize: resizedSize)
                    return VideoFrame(frame: processedImage, timeStamp: frame.timeStamp)
                }

                videosAsImageSequences.append(imageSequence.frames)
            }
            let videoPixelsAndFrames = try videosAsImageSequences.map {
                try QwenVL.patchify(
                    images: $0, mergeSize: config.mergeSize, patchSize: config.patchSize,
                    temporalPatchSize: config.temporalPatchSize)
            }
            let videoPixelsConcatenated = concatenated(videoPixelsAndFrames.map { $0.0 })
            processedVideo = LMInput.ProcessedVideo(
                pixels: videoPixelsConcatenated, frames: videoPixelsAndFrames.map { $0.1 })
            if let videoFrames = processedVideo?.frames {
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens, frames: videoFrames, paddingToken: "<|video_pad|>",
                    mergeSize: config.mergeSize, tokenizer: tokenizer)
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage,
            video: processedVideo,
            cacheScopeSalt: cacheScopeSalt(from: input.additionalContext))
    }
}

// MARK: - Model

/// Qwen2.5VL VLM
///
/// This is typically created by ``VLMModelFactory``.
public class Qwen25VL: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionModel: Qwen25Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel

    public let config: Qwen25VLConfiguration

    public var vocabularySize: Int { config.baseConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: Qwen25VLConfiguration) {
        self.config = config
        self._visionModel.wrappedValue = Qwen25Vision.VisionModel(config.visionConfiguration)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfiguration)
    }

    private func inputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?, frames: [THW]?)
        -> MLXArray
    {
        guard let pixelValues, let frames else {
            return languageModel.model.embedTokens(inputIds[.newAxis, .ellipsis])
        }

        // Get the input embeddings from the language model
        let inputEmbeds = languageModel.model.embedTokens(inputIds)

        // Get the ouptut hidden states from the vision model
        var hiddenStates = self.visionModel(pixelValues, frames: frames)

        if hiddenStates.ndim == 2 {
            hiddenStates = hiddenStates[.newAxis, 0..., 0...]
        }

        // Insert special image tokens in the input_ids
        return QwenVL.mergeInputIdsWithImageFeatures(
            inputIds: inputIds, inputEmbeds: inputEmbeds, imageFeatures: hiddenStates,
            imageTokenId: config.baseConfiguration.imageTokenId,
            videoTokenId: config.baseConfiguration.videoTokenId)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let dtype = visionModel.patchEmbed.proj.weight.dtype

        // Process both images and videos together
        var allPixels: MLXArray?
        var allFrames: [THW] = []

        if let imagePixels = input.image?.pixels, let imageFrames = input.image?.frames {
            allPixels = imagePixels.asType(dtype)
            allFrames.append(contentsOf: imageFrames)
        }

        if let videoPixels = input.video?.pixels, let videoFrames = input.video?.frames {
            if allPixels == nil {
                allPixels = videoPixels.asType(dtype)
            } else {
                allPixels = concatenated([allPixels!, videoPixels.asType(dtype)])
            }
            allFrames.append(contentsOf: videoFrames)
        }

        let inputEmbeddings = self.inputEmbeddings(
            inputIds: input.text.tokens, pixelValues: allPixels,
            frames: allFrames.isEmpty ? nil : allFrames)

        let result = try languageModel.embeddingsForward(inputEmbeddings, cache: cache)

        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        visionModel.sanitize(
            weights:
                Dictionary(
                    uniqueKeysWithValues: weights.map { key, value in
                        var key = key
                        if !key.contains("vision_tower") {
                            key = key.replacingOccurrences(of: "visual", with: "vision_tower")
                        }
                        if !key.contains("language_model") {
                            key = key.replacingOccurrences(
                                of: "model", with: "language_model.model")
                            key = key.replacingOccurrences(
                                of: "lm_head", with: "language_model.lm_head")
                        }

                        return (key, value)
                    })
        )
    }
}

// MARK: - Configuration

/// Configuration for ``Qwen25VL``
public struct Qwen25VLConfiguration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        private let _rmsNormEps: Float?
        public var rmsNormEps: Float { _rmsNormEps ?? 1e-6 }
        public let vocabularySize: Int
        public let kvHeads: Int
        private let _maxPositionEmbeddings: Int?
        public var maxPositionEmbeddings: Int { _maxPositionEmbeddings ?? 128000 }
        private let _ropeTheta: Float?
        public var ropeTheta: Float { _ropeTheta ?? 1_000_000 }
        private let _ropeTraditional: Bool?
        public var ropeTraditional: Bool { _ropeTraditional ?? false }
        public let ropeScaling: [String: StringOrNumber]?
        private let _tieWordEmbeddings: Bool?
        public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? true }
        private let _slidingWindow: Int?
        public var slidingWindow: Int { _slidingWindow ?? 32768 }
        private let _useSlidingWindow: Bool?
        public var useSlidingWindow: Bool { _useSlidingWindow ?? false }
        public var mropeSection: [Int] {
            guard let array = ropeScaling?["mrope_section"]?.asInts() else {
                return []
            }
            // mrope_section = np.cumsum(mrope_section * 2)[:-1].tolist()
            return sequence(state: (0, array.makeIterator())) { state in
                if let v = state.1.next() {
                    state.0 += v * 2
                    return state.0
                } else {
                    return nil
                }
            }.dropLast()
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case _rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case kvHeads = "num_key_value_heads"
            case _maxPositionEmbeddings = "max_position_embeddings"
            case _ropeTheta = "rope_theta"
            case _ropeTraditional = "rope_traditional"
            case ropeScaling = "rope_scaling"
            case _tieWordEmbeddings = "tie_word_embeddings"
            case _slidingWindow = "sliding_window"
            case _useSlidingWindow = "use_sliding_window"
        }

        public init(from decoder: any Swift.Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.modelType = try container.decode(String.self, forKey: .modelType)
            self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
            self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
            self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
            self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
            self._rmsNormEps = try container.decodeIfPresent(Float.self, forKey: ._rmsNormEps)
            self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
            self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
            self._maxPositionEmbeddings = try container.decodeIfPresent(
                Int.self, forKey: ._maxPositionEmbeddings)
            self._ropeTheta = try container.decodeIfPresent(Float.self, forKey: ._ropeTheta)
            self._ropeTraditional = try container.decodeIfPresent(
                Bool.self, forKey: ._ropeTraditional)
            self.ropeScaling = try container.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeScaling)
            self._tieWordEmbeddings = try container.decodeIfPresent(
                Bool.self, forKey: ._tieWordEmbeddings)
            self._slidingWindow = try container.decodeIfPresent(Int.self, forKey: ._slidingWindow)
            self._useSlidingWindow = try container.decodeIfPresent(
                Bool.self, forKey: ._useSlidingWindow)

            try Self.validatePositive(hiddenSize, key: .hiddenSize, in: container)
            try Self.validatePositive(hiddenLayers, key: .hiddenLayers, in: container)
            try Self.validatePositive(intermediateSize, key: .intermediateSize, in: container)
            try Self.validatePositive(attentionHeads, key: .attentionHeads, in: container)
            try Self.validatePositive(vocabularySize, key: .vocabularySize, in: container)
            try Self.validatePositive(kvHeads, key: .kvHeads, in: container)
            try Self.validatePositive(slidingWindow, key: ._slidingWindow, in: container)
            guard hiddenSize % attentionHeads == 0 else {
                throw DecodingError.dataCorruptedError(
                    forKey: .hiddenSize,
                    in: container,
                    debugDescription:
                        "Qwen2.5-VL hidden_size must be divisible by num_attention_heads.")
            }
            guard attentionHeads % kvHeads == 0 else {
                throw DecodingError.dataCorruptedError(
                    forKey: .kvHeads,
                    in: container,
                    debugDescription:
                        "Qwen2.5-VL num_attention_heads must be divisible by num_key_value_heads.")
            }
            guard let mropeSection = ropeScaling?["mrope_section"]?.asInts(),
                mropeSection.count >= 2,
                mropeSection.allSatisfy({ $0 > 0 })
            else {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeScaling,
                    in: container,
                    debugDescription:
                        "Qwen2.5-VL rope_scaling.mrope_section must be an array of positive integers.")
            }
        }

        private static func validatePositive<K: CodingKey>(
            _ value: Int, key: K, in container: KeyedDecodingContainer<K>
        ) throws {
            guard value > 0 else {
                throw DecodingError.dataCorruptedError(
                    forKey: key,
                    in: container,
                    debugDescription: "\(key.stringValue) must be greater than zero.")
            }
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let depth: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let outHiddenSize: Int
        public let numHeads: Int
        public let patchSize: Int
        private let _inChans: Int?
        public var inChannels: Int { _inChans ?? 3 }
        private let _layerNormEps: Float?
        public var layerNormEps: Float { _layerNormEps ?? 1e-6 }
        public let spatialPatchSize: Int
        public let spatialMergeSize: Int
        public let temporalPatchSize: Int
        public let windowSize: Int
        public let fullattBlockIndexes: [Int]
        public let tokensPerSecond: Int
        private let _skipVision: Bool?
        public var skipVision: Bool { _skipVision ?? false }
        private let _hiddenAct: String?
        public var hiddenAct: String { _hiddenAct ?? "silu" }

        enum CodingKeys: String, CodingKey {
            case depth
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case outHiddenSize = "out_hidden_size"
            case numHeads = "num_heads"
            case patchSize = "patch_size"
            case _inChans = "in_chans"
            case _layerNormEps = "layer_norm_eps"  // Added this line
            case spatialPatchSize = "spatial_patch_size"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
            case windowSize = "window_size"
            case fullattBlockIndexes = "fullatt_block_indexes"
            case tokensPerSecond = "tokens_per_second"
            case _skipVision = "skip_vision"
            case _hiddenAct = "hidden_act"
        }

        public init(from decoder: any Swift.Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.depth = try container.decode(Int.self, forKey: .depth)
            self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
            self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
            self.outHiddenSize = try container.decode(Int.self, forKey: .outHiddenSize)
            self.numHeads = try container.decode(Int.self, forKey: .numHeads)
            self.patchSize = try container.decode(Int.self, forKey: .patchSize)
            self._inChans = try container.decodeIfPresent(Int.self, forKey: ._inChans)
            self._layerNormEps = try container.decodeIfPresent(Float.self, forKey: ._layerNormEps)
            self.spatialPatchSize = try container.decode(Int.self, forKey: .spatialPatchSize)
            self.spatialMergeSize = try container.decode(Int.self, forKey: .spatialMergeSize)
            self.temporalPatchSize = try container.decode(Int.self, forKey: .temporalPatchSize)
            self.windowSize = try container.decode(Int.self, forKey: .windowSize)
            self.fullattBlockIndexes = try container.decode(
                [Int].self, forKey: .fullattBlockIndexes)
            self.tokensPerSecond = try container.decode(Int.self, forKey: .tokensPerSecond)
            self._skipVision = try container.decodeIfPresent(Bool.self, forKey: ._skipVision)
            self._hiddenAct = try container.decodeIfPresent(String.self, forKey: ._hiddenAct)

            try Self.validatePositive(depth, key: .depth, in: container)
            try Self.validatePositive(hiddenSize, key: .hiddenSize, in: container)
            try Self.validatePositive(intermediateSize, key: .intermediateSize, in: container)
            try Self.validatePositive(outHiddenSize, key: .outHiddenSize, in: container)
            try Self.validatePositive(numHeads, key: .numHeads, in: container)
            try Self.validatePositive(patchSize, key: .patchSize, in: container)
            try Self.validatePositive(inChannels, key: ._inChans, in: container)
            try Self.validatePositive(spatialPatchSize, key: .spatialPatchSize, in: container)
            try Self.validatePositive(spatialMergeSize, key: .spatialMergeSize, in: container)
            try Self.validatePositive(temporalPatchSize, key: .temporalPatchSize, in: container)
            try Self.validatePositive(windowSize, key: .windowSize, in: container)
            try Self.validatePositive(tokensPerSecond, key: .tokensPerSecond, in: container)
            guard hiddenSize % numHeads == 0 else {
                throw DecodingError.dataCorruptedError(
                    forKey: .hiddenSize,
                    in: container,
                    debugDescription:
                        "Qwen2.5-VL vision hidden_size must be divisible by num_heads.")
            }
            guard patchSize % spatialMergeSize == 0 else {
                throw DecodingError.dataCorruptedError(
                    forKey: .spatialMergeSize,
                    in: container,
                    debugDescription:
                        "Qwen2.5-VL patch_size must be divisible by spatial_merge_size.")
            }
            guard fullattBlockIndexes.allSatisfy({ $0 >= 0 && $0 < depth }) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .fullattBlockIndexes,
                    in: container,
                    debugDescription:
                        "Qwen2.5-VL fullatt_block_indexes must be within the vision layer range.")
            }
        }

        private static func validatePositive<K: CodingKey>(
            _ value: Int, key: K, in container: KeyedDecodingContainer<K>
        ) throws {
            guard value > 0 else {
                throw DecodingError.dataCorruptedError(
                    forKey: key,
                    in: container,
                    debugDescription: "\(key.stringValue) must be greater than zero.")
            }
        }
    }

    public struct BaseConfiguration: Codable, Sendable {
        public let modelType: String
        public let vocabularySize: Int
        public let imageTokenId: Int
        public let videoTokenId: Int
        public let visionStartTokenId: Int
        public let visionEndTokenId: Int
        public let visionTokenId: Int
        public let hiddenSize: Int
        public let numAttentionHeads: Int
        public let numHiddenLayers: Int
        public let intermediateSize: Int
        public let numKeyValueHeads: Int
        public let slidingWindow: Int
        public let useSlidingWindow: Bool
        public let maxWindowLayers: Int

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case imageTokenId = "image_token_id"
            case videoTokenId = "video_token_id"
            case visionStartTokenId = "vision_start_token_id"
            case visionEndTokenId = "vision_end_token_id"
            case visionTokenId = "vision_token_id"
            case hiddenSize = "hidden_size"
            case numAttentionHeads = "num_attention_heads"
            case numHiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case numKeyValueHeads = "num_key_value_heads"
            case slidingWindow = "sliding_window"
            case useSlidingWindow = "use_sliding_window"
            case maxWindowLayers = "max_window_layers"
        }

        public init(from decoder: any Swift.Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.modelType = try container.decode(String.self, forKey: .modelType)
            self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
            self.imageTokenId = try container.decode(Int.self, forKey: .imageTokenId)
            self.videoTokenId = try container.decode(Int.self, forKey: .videoTokenId)
            self.visionStartTokenId = try container.decode(Int.self, forKey: .visionStartTokenId)
            self.visionEndTokenId = try container.decode(Int.self, forKey: .visionEndTokenId)
            self.visionTokenId = try container.decode(Int.self, forKey: .visionTokenId)
            self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
            self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
            self.numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
            self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
            self.numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
            self.slidingWindow = try container.decode(Int.self, forKey: .slidingWindow)
            self.useSlidingWindow = try container.decode(Bool.self, forKey: .useSlidingWindow)
            self.maxWindowLayers = try container.decode(Int.self, forKey: .maxWindowLayers)

            try Self.validatePositive(vocabularySize, key: .vocabularySize, in: container)
            try Self.validatePositive(hiddenSize, key: .hiddenSize, in: container)
            try Self.validatePositive(numAttentionHeads, key: .numAttentionHeads, in: container)
            try Self.validatePositive(numHiddenLayers, key: .numHiddenLayers, in: container)
            try Self.validatePositive(intermediateSize, key: .intermediateSize, in: container)
            try Self.validatePositive(numKeyValueHeads, key: .numKeyValueHeads, in: container)
            try Self.validatePositive(slidingWindow, key: .slidingWindow, in: container)
            try Self.validatePositive(maxWindowLayers, key: .maxWindowLayers, in: container)
            guard hiddenSize % numAttentionHeads == 0 else {
                throw DecodingError.dataCorruptedError(
                    forKey: .hiddenSize,
                    in: container,
                    debugDescription:
                        "Qwen2.5-VL base hidden_size must be divisible by num_attention_heads.")
            }
            guard numAttentionHeads % numKeyValueHeads == 0 else {
                throw DecodingError.dataCorruptedError(
                    forKey: .numKeyValueHeads,
                    in: container,
                    debugDescription:
                        "Qwen2.5-VL base num_attention_heads must be divisible by num_key_value_heads.")
            }
            guard maxWindowLayers <= numHiddenLayers else {
                throw DecodingError.dataCorruptedError(
                    forKey: .maxWindowLayers,
                    in: container,
                    debugDescription:
                        "Qwen2.5-VL max_window_layers must not exceed num_hidden_layers.")
            }
        }

        private static func validatePositive<K: CodingKey>(
            _ value: Int, key: K, in container: KeyedDecodingContainer<K>
        ) throws {
            guard value > 0 else {
                throw DecodingError.dataCorruptedError(
                    forKey: key,
                    in: container,
                    debugDescription: "\(key.stringValue) must be greater than zero.")
            }
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let baseConfiguration: BaseConfiguration

    enum CodingKeys: String, CodingKey {
        case visionConfiguration = "vision_config"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // this is a sub-dictionary
        self.visionConfiguration = try container.decode(
            VisionConfiguration.self, forKey: .visionConfiguration)

        // these are overlaid in the top level
        self.textConfiguration = try TextConfiguration(from: decoder)
        self.baseConfiguration = try BaseConfiguration(from: decoder)
    }
}

/// Configuration for ``Qwen25VLProcessor``
public struct Qwen25VLProcessorConfiguration: Codable, Sendable {
    public struct Size: Codable, Sendable {
        public let maxPixels: Int
        public let minPixels: Int

        enum CodingKeys: String, CodingKey {
            case maxPixels = "max_pixels"
            case minPixels = "min_pixels"
        }

        public init(maxPixels: Int, minPixels: Int) {
            self.maxPixels = maxPixels
            self.minPixels = minPixels
        }

        public init(from decoder: any Swift.Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.maxPixels = try container.decode(Int.self, forKey: .maxPixels)
            self.minPixels = try container.decode(Int.self, forKey: .minPixels)

            try Self.validatePositive(maxPixels, key: .maxPixels, in: container)
            try Self.validatePositive(minPixels, key: .minPixels, in: container)
            guard maxPixels >= minPixels else {
                throw DecodingError.dataCorruptedError(
                    forKey: .maxPixels,
                    in: container,
                    debugDescription:
                        "Qwen2.5-VL processor size.max_pixels must be greater than or equal to min_pixels.")
            }
        }

        private static func validatePositive<K: CodingKey>(
            _ value: Int, key: K, in container: KeyedDecodingContainer<K>
        ) throws {
            guard value > 0 else {
                throw DecodingError.dataCorruptedError(
                    forKey: key,
                    in: container,
                    debugDescription: "\(key.stringValue) must be greater than zero.")
            }
        }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let minPixels: Int
    public let maxPixels: Int
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let imageProcessorType: String

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    public var size: Size {
        Size(maxPixels: maxPixels, minPixels: minPixels)
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case minPixels = "min_pixels"
        case maxPixels = "max_pixels"
        case mergeSize = "merge_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case imageProcessorType = "image_processor_type"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.imageMean = try container.decode([CGFloat].self, forKey: .imageMean)
        self.imageStd = try container.decode([CGFloat].self, forKey: .imageStd)
        self.minPixels = try container.decode(Int.self, forKey: .minPixels)
        self.maxPixels = try container.decode(Int.self, forKey: .maxPixels)
        self.mergeSize = try container.decode(Int.self, forKey: .mergeSize)
        self.patchSize = try container.decode(Int.self, forKey: .patchSize)
        self.temporalPatchSize = try container.decode(Int.self, forKey: .temporalPatchSize)
        self.imageProcessorType = try container.decode(String.self, forKey: .imageProcessorType)

        try Self.validateRGBTriplet(imageMean, key: .imageMean, in: container)
        try Self.validateRGBTriplet(imageStd, key: .imageStd, in: container)
        try Self.validatePositive(minPixels, key: .minPixels, in: container)
        try Self.validatePositive(maxPixels, key: .maxPixels, in: container)
        try Self.validatePositive(mergeSize, key: .mergeSize, in: container)
        try Self.validatePositive(patchSize, key: .patchSize, in: container)
        try Self.validatePositive(temporalPatchSize, key: .temporalPatchSize, in: container)
        guard maxPixels >= minPixels else {
            throw DecodingError.dataCorruptedError(
                forKey: .maxPixels,
                in: container,
                debugDescription:
                    "Qwen2.5-VL processor max_pixels must be greater than or equal to min_pixels.")
        }
        guard patchSize % mergeSize == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .mergeSize,
                in: container,
                debugDescription:
                    "Qwen2.5-VL processor patch_size must be divisible by merge_size.")
        }
        guard !imageProcessorType.isEmpty else {
            throw DecodingError.dataCorruptedError(
                forKey: .imageProcessorType,
                in: container,
                debugDescription: "Qwen2.5-VL image_processor_type must not be empty.")
        }
    }

    private static func validateRGBTriplet<K: CodingKey>(
        _ values: [CGFloat], key: K, in container: KeyedDecodingContainer<K>
    ) throws {
        guard values.count == 3 else {
            throw DecodingError.dataCorruptedError(
                forKey: key,
                in: container,
                debugDescription: "\(key.stringValue) must contain exactly three RGB values.")
        }
    }

    private static func validatePositive<K: CodingKey>(
        _ value: Int, key: K, in container: KeyedDecodingContainer<K>
    ) throws {
        guard value > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: key,
                in: container,
                debugDescription: "\(key.stringValue) must be greater than zero.")
        }
    }
}
