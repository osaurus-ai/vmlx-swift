//
//  LFM3VL.swift
//  mlx-swift-lm
//
//  LiquidAI LFM3-VL: a hybrid conv / attention vision tower (2-D axial RoPE,
//  Qwen-VL patching), a 2x2-merge GELU projector, and the LFM3 decoder.
//
//  References: the bundles' own `modeling_lfm3_vl.py` / `processing_lfm3_vl.py`,
//  and the MLX port in mlx-vlm `mlx_vlm/models/lfm3_vl/`.
//

import CoreImage
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct LFM3VLVisionConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let inChannels: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let spatialMergeSize: Int
    public let convKernelSize: Int
    public let normEps: Float
    /// One entry per layer: `conv` or `full_attention`.
    public let layerTypes: [String]
    public let ropeTheta: Float

    public var headDim: Int { hiddenSize / attentionHeads }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case inChannels = "in_channels"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case convKernelSize = "conv_kernel_size"
        case normEps = "norm_eps"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2560
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 14
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        inChannels = try c.decodeIfPresent(Int.self, forKey: .inChannels) ?? 3
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        temporalPatchSize = try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
        spatialMergeSize = try c.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
        convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 3
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
        guard layerTypes.count == hiddenLayers,
            layerTypes.allSatisfy({ $0 == "conv" || $0 == "full_attention" })
        else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: c,
                debugDescription: "vision layer_types must be \(hiddenLayers) conv/full_attention entries")
        }
        let rope = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
        if case .string(let type) = rope?["rope_type"], type != "axial" {
            throw DecodingError.dataCorruptedError(
                forKey: .ropeParameters, in: c,
                debugDescription: "LFM3 vision rope_type '\(type)' is not supported")
        }
        ropeTheta = rope?["rope_theta"]?.asFloat() ?? 1e6
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(inChannels, forKey: .inChannels)
        try c.encode(patchSize, forKey: .patchSize)
        try c.encode(temporalPatchSize, forKey: .temporalPatchSize)
        try c.encode(spatialMergeSize, forKey: .spatialMergeSize)
        try c.encode(convKernelSize, forKey: .convKernelSize)
        try c.encode(normEps, forKey: .normEps)
        try c.encode(layerTypes, forKey: .layerTypes)
        try c.encode(
            ["rope_type": StringOrNumber.string("axial"), "rope_theta": .float(ropeTheta)],
            forKey: .ropeParameters)
    }
}

public struct LFM3VLConfiguration: Codable, Sendable {
    public let textConfiguration: LFM3Configuration
    public let visionConfiguration: LFM3VLVisionConfiguration
    public let downsampleFactor: Int
    public let projectorHiddenSize: Int
    public let projectorBias: Bool
    public let projectorLayerNormEps: Float
    public let imageTokenId: Int
    public let videoTokenId: Int

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case downsampleFactor = "downsample_factor"
        case projectorHiddenSize = "projector_hidden_size"
        case projectorBias = "projector_bias"
        case projectorLayerNormEps = "projector_layer_norm_eps"
        case projectorHiddenAct = "projector_hidden_act"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration = try c.decode(LFM3Configuration.self, forKey: .textConfiguration)
        visionConfiguration = try c.decode(
            LFM3VLVisionConfiguration.self, forKey: .visionConfiguration)
        downsampleFactor = try c.decodeIfPresent(Int.self, forKey: .downsampleFactor) ?? 2
        projectorHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .projectorHiddenSize) ?? 4096
        projectorBias = try c.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? true
        projectorLayerNormEps =
            try c.decodeIfPresent(Float.self, forKey: .projectorLayerNormEps) ?? 1e-6
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 125030
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 125029
        // The projector concatenates one merge block (downsample^2 tokens) per output row,
        // which only lines up with the tower's merge-ordered tokens when the two agree.
        guard downsampleFactor == visionConfiguration.spatialMergeSize else {
            throw DecodingError.dataCorruptedError(
                forKey: .downsampleFactor, in: c,
                debugDescription:
                    "downsample_factor \(downsampleFactor) != spatial_merge_size \(visionConfiguration.spatialMergeSize)")
        }
        if let act = try c.decodeIfPresent(String.self, forKey: .projectorHiddenAct), act != "gelu" {
            throw DecodingError.dataCorruptedError(
                forKey: .projectorHiddenAct, in: c,
                debugDescription: "LFM3-VL projector activation '\(act)' is not supported")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(textConfiguration, forKey: .textConfiguration)
        try c.encode(visionConfiguration, forKey: .visionConfiguration)
        try c.encode(downsampleFactor, forKey: .downsampleFactor)
        try c.encode(projectorHiddenSize, forKey: .projectorHiddenSize)
        try c.encode(projectorBias, forKey: .projectorBias)
        try c.encode(projectorLayerNormEps, forKey: .projectorLayerNormEps)
        try c.encode("gelu", forKey: .projectorHiddenAct)
        try c.encode(imageTokenId, forKey: .imageTokenId)
        try c.encode(videoTokenId, forKey: .videoTokenId)
    }
}

// MARK: - Vision tower

enum LFM3VLVision {

    /// Patch coordinates in the order the tower sees tokens: 2x2 merge blocks row-major,
    /// then the patches inside each block row-major.
    static func mergeOrderedCoordinates(height: Int, width: Int, merge: Int) -> (
        rows: [Float], cols: [Float]
    ) {
        var rows: [Float] = [], cols: [Float] = []
        rows.reserveCapacity(height * width)
        cols.reserveCapacity(height * width)
        for bi in 0 ..< height / merge {
            for bj in 0 ..< width / merge {
                for i in 0 ..< merge {
                    for j in 0 ..< merge {
                        rows.append(Float(bi * merge + i))
                        cols.append(Float(bj * merge + j))
                    }
                }
            }
        }
        return (rows, cols)
    }

    /// Axial rotary angles: the first half of each head rotates by row, the second by
    /// column, each with `headDim / 4` frequencies; `[N, headDim]` after the rotate-half
    /// duplication.
    static func axialAngles(
        height: Int, width: Int, merge: Int, headDim: Int, theta: Float
    ) -> (cos: MLXArray, sin: MLXArray) {
        let spatialDim = headDim / 2
        let inv = stride(from: 0, to: spatialDim, by: 2).map {
            Float(pow(Double(theta), -Double($0) / Double(spatialDim)))
        }
        let (rows, cols) = mergeOrderedCoordinates(height: height, width: width, merge: merge)
        let invFreq = MLXArray(inv)
        let rowAngles = MLXArray(rows).expandedDimensions(axis: -1) * invFreq
        let colAngles = MLXArray(cols).expandedDimensions(axis: -1) * invFreq
        let half = concatenated([rowAngles, colAngles], axis: -1)  // [N, headDim / 2]
        let angles = concatenated([half, half], axis: -1)  // [N, headDim]
        return (cos(angles), sin(angles))
    }

    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear
        @ModuleInfo(key: "q_layernorm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_layernorm") var kNorm: RMSNorm

        init(_ config: LFM3VLVisionConfiguration) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)
            let d = config.hiddenSize
            _qProj.wrappedValue = Linear(d, heads * headDim, bias: false)
            _kProj.wrappedValue = Linear(d, kvHeads * headDim, bias: false)
            _vProj.wrappedValue = Linear(d, kvHeads * headDim, bias: false)
            _outProj.wrappedValue = Linear(heads * headDim, d, bias: false)
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
        }

        func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
            let n = x.dim(0)
            func rotate(_ t: MLXArray) -> MLXArray {
                // Rotation in float32, as the reference does, then back to the compute dtype.
                let f = t.asType(.float32)
                let halves = split(f, parts: 2, axis: -1)
                let rotated = concatenated([-halves[1], halves[0]], axis: -1)
                let c = cos.expandedDimensions(axis: 1)
                let s = sin.expandedDimensions(axis: 1)
                return (f * c + rotated * s).asType(t.dtype)
            }
            let q = rotate(qNorm(qProj(x).reshaped(n, heads, headDim)))
            let k = rotate(kNorm(kProj(x).reshaped(n, kvHeads, headDim)))
            let v = vProj(x).reshaped(n, kvHeads, headDim)
            // Every patch of a frame attends to every other: no mask.
            let out = MLXFast.scaledDotProductAttention(
                queries: q.transposed(1, 0, 2).expandedDimensions(axis: 0),
                keys: k.transposed(1, 0, 2).expandedDimensions(axis: 0),
                values: v.transposed(1, 0, 2).expandedDimensions(axis: 0),
                scale: scale, mask: .none)
            return outProj(out[0].transposed(1, 0, 2).reshaped(n, -1))
        }
    }

    /// Gated depthwise 3x3 convolution over the patch grid.
    final class Conv: Module {
        let merge: Int

        @ModuleInfo(key: "in_proj") var inProj: Linear
        @ModuleInfo(key: "spatial_conv") var spatialConv: Conv2d
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: LFM3VLVisionConfiguration) {
            merge = config.spatialMergeSize
            let d = config.hiddenSize
            _inProj.wrappedValue = Linear(d, 3 * d, bias: false)
            _spatialConv.wrappedValue = Conv2d(
                inputChannels: d, outputChannels: d,
                kernelSize: IntOrPair(config.convKernelSize),
                padding: IntOrPair(config.convKernelSize / 2), groups: d, bias: false)
            _outProj.wrappedValue = Linear(d, d, bias: false)
        }

        func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
            let parts = inProj(x).split(parts: 3, axis: -1)
            let (b, c, v) = (parts[0], parts[1], parts[2])
            let m = merge
            // Merge order -> spatial grid (NHWC) for the convolution, and back.
            let grid = (b * v)
                .reshaped(height / m, width / m, m, m, -1)
                .transposed(0, 2, 1, 3, 4)
                .reshaped(1, height, width, -1)
            let convolved = spatialConv(grid)
                .reshaped(height / m, m, width / m, m, -1)
                .transposed(0, 2, 1, 3, 4)
                .reshaped(height * width, -1)
            return outProj(c * convolved)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "w1") var w1: Linear
        @ModuleInfo(key: "w2") var w2: Linear
        @ModuleInfo(key: "w3") var w3: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _w1.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _w2.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            _w3.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            w2(silu(w1(x)) * w3(x))
        }
    }

    final class Layer: Module {
        let isAttention: Bool

        @ModuleInfo(key: "self_attn") var attention: Attention?
        @ModuleInfo(key: "conv") var conv: Conv?
        @ModuleInfo(key: "feed_forward") var feedForward: MLP
        @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

        init(_ config: LFM3VLVisionConfiguration, index: Int) {
            isAttention = config.layerTypes[index] == "full_attention"
            if isAttention {
                _attention.wrappedValue = Attention(config)
            } else {
                _conv.wrappedValue = Conv(config)
            }
            _feedForward.wrappedValue = MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
            _operatorNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
            _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(
            _ x: MLXArray, height: Int, width: Int, cos: MLXArray, sin: MLXArray
        ) -> MLXArray {
            let h = operatorNorm(x)
            let r =
                isAttention
                ? attention!(h, cos: cos, sin: sin) : conv!(h, height: height, width: width)
            let x = x + r
            return x + feedForward(ffnNorm(x))
        }
    }

    final class VisionModel: Module {
        let config: LFM3VLVisionConfiguration

        @ModuleInfo(key: "patchifier") var patchifier: Linear
        @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm
        let layers: [Layer]

        init(_ config: LFM3VLVisionConfiguration) {
            self.config = config
            _patchifier.wrappedValue = Linear(
                config.inChannels * config.temporalPatchSize * config.patchSize * config.patchSize,
                config.hiddenSize, bias: false)
            _embeddingNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
            layers = (0 ..< config.hiddenLayers).map { Layer(config, index: $0) }
        }

        /// `pixels`: `[sum(t * h * w), C * T * P * P]` merge-ordered patches; one forward per
        /// temporal group (frames do not attend across groups).
        func callAsFunction(_ pixels: MLXArray, grids: [THW]) throws -> MLXArray {
            var outputs: [MLXArray] = []
            var offset = 0
            for grid in grids {
                let (cosA, sinA) = LFM3VLVision.axialAngles(
                    height: grid.h, width: grid.w, merge: config.spatialMergeSize,
                    headDim: config.headDim, theta: config.ropeTheta)
                for _ in 0 ..< grid.t {
                    let count = grid.h * grid.w
                    guard offset + count <= pixels.dim(0) else {
                        throw VLMError.processing(
                            "LFM3-VL: \(pixels.dim(0)) patches for grids \(grids.map { [$0.t, $0.h, $0.w] })")
                    }
                    var x = embeddingNorm(
                        patchifier(pixels[offset ..< offset + count].asType(patchifier.weight.dtype)))
                    for layer in layers {
                        x = layer(x, height: grid.h, width: grid.w, cos: cosA, sin: sinA)
                    }
                    outputs.append(x)
                    offset += count
                }
            }
            guard offset == pixels.dim(0) else {
                throw VLMError.processing(
                    "LFM3-VL: \(pixels.dim(0)) patches but grids cover \(offset)")
            }
            return concatenated(outputs, axis: 0)
        }
    }
}

final class LFM3VLProjector: Module {
    let inputDim: Int

    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    convenience init(_ config: LFM3VLConfiguration) {
        self.init(
            visionHidden: config.visionConfiguration.hiddenSize,
            downsampleFactor: config.downsampleFactor, hidden: config.projectorHiddenSize,
            textHidden: config.textConfiguration.hiddenSize, bias: config.projectorBias,
            eps: config.projectorLayerNormEps)
    }

    init(visionHidden: Int, downsampleFactor: Int, hidden: Int, textHidden: Int, bias: Bool, eps: Float) {
        inputDim = visionHidden * downsampleFactor * downsampleFactor
        _layerNorm.wrappedValue = LayerNorm(dimensions: visionHidden, eps: eps)
        _linear1.wrappedValue = Linear(inputDim, hidden, bias: bias)
        _linear2.wrappedValue = Linear(hidden, textHidden, bias: bias)
    }

    /// Normalizes each patch token, then concatenates every merge block (consecutive in the
    /// tower's output order) into one row.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(layerNorm(x).reshaped(-1, inputDim))))
    }
}

// MARK: - Model

public final class LFM3VL: Module, VLMModel, KVCacheDimensionProvider, ModalityBearing {
    @ModuleInfo(key: "vision_tower") var visionTower: LFM3VLVision.VisionModel
    @ModuleInfo(key: "multi_modal_projector") var projector: LFM3VLProjector
    @ModuleInfo(key: "language_model") var languageModel: LFM3Model

    public let config: LFM3VLConfiguration

    public init(_ config: LFM3VLConfiguration) {
        self.config = config
        _visionTower.wrappedValue = LFM3VLVision.VisionModel(config.visionConfiguration)
        _projector.wrappedValue = LFM3VLProjector(config)
        _languageModel.wrappedValue = LFM3Model(config.textConfiguration)
    }

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.loraLayers }
    public var modalities: Set<ModelRuntimeRequestModality> { [.text, .vision, .video] }

    /// Text-model rows for every visual token, in prompt order per medium.
    func mediaFeatures(pixels: MLXArray, grids: [THW]) throws -> MLXArray {
        projector(try visionTower(pixels, grids: grids))
    }

    /// Replaces the embeddings at `tokenId` positions with `features`, row for row.
    static func scatter(
        _ features: MLXArray, into embeddings: MLXArray, tokenIds: [Int], tokenId: Int
    ) throws -> MLXArray {
        let positions = tokenIds.indices.filter { tokenIds[$0] == tokenId }
        guard positions.count == features.dim(0) else {
            throw VLMError.processing(
                "LFM3-VL: \(positions.count) placeholder tokens for \(features.dim(0)) feature rows")
        }
        guard !positions.isEmpty else { return embeddings }
        let flat = embeddings.reshaped(-1, embeddings.dim(-1))
        flat[MLXArray(positions.map(Int32.init))] = features.asType(flat.dtype)
        return flat.reshaped(embeddings.shape)
    }

    func inputEmbeddings(for input: LMInput) throws -> MLXArray {
        let ids = input.text.tokens.reshaped(1, -1)
        var embeddings = languageModel.model.embedTokens(ids)
        let tokenIds = ids.reshaped(-1).asArray(Int.self)
        if let image = input.image, let grids = image.frames, !grids.isEmpty {
            embeddings = try Self.scatter(
                mediaFeatures(pixels: image.pixels, grids: grids), into: embeddings,
                tokenIds: tokenIds, tokenId: config.imageTokenId)
        }
        if let video = input.video, let grids = video.frames, !grids.isEmpty {
            embeddings = try Self.scatter(
                mediaFeatures(pixels: video.pixels, grids: grids), into: embeddings,
                tokenIds: tokenIds, tokenId: config.videoTokenId)
        }
        return embeddings
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let embeddings = try inputEmbeddings(for: input)
        let logits = try chunkedPrefillEmbedding(
            inputEmbedding: embeddings, cache: cache, prefillStepSize: windowSize ?? 512
        ) { chunk in
            languageModel.logits(nil, cache: cache, inputEmbeddings: chunk, lastPositionOnly: true)
        }
        return .logits(LMOutput(logits: logits))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    /// Accepts the MLX layout the released bundles use (`vision_tower.*`,
    /// `multi_modal_projector.*`, `language_model.model.*`) and the transformers layout
    /// (`model.vision_tower.*`, `model.language_model.*`, `lm_head.*`).
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            var name = key
            if name.hasPrefix("model.language_model.") {
                name = "language_model.model." + name.dropFirst("model.language_model.".count)
            } else if name.hasPrefix("model.") {
                name = String(name.dropFirst("model.".count))
            } else if name.hasPrefix("lm_head.") {
                name = "language_model." + name
            }
            if config.textConfiguration.tieWordEmbeddings, name.hasPrefix("language_model.lm_head.") {
                continue
            }
            var value = value
            if name.hasSuffix("conv.conv.weight"), value.ndim == 3, value.dim(1) == 1,
                value.dim(2) > 1
            {
                value = value.transposed(0, 2, 1)  // [C, 1, K] -> [C, K, 1]
            } else if name.hasSuffix("spatial_conv.weight"), value.ndim == 4, value.dim(1) == 1,
                value.dim(3) > 1
            {
                value = value.transposed(0, 2, 3, 1)  // [C, 1, kh, kw] -> [C, kh, kw, 1]
            }
            result[name] = value
        }
        return result
    }
}

// MARK: - Processor

/// Images and pre-sampled video frames into Qwen-VL patches, with LFM3's placeholder
/// expansion.
///
/// INVARIANT: the chat template renders each medium as `<|vision_start|><|image|>
/// <|vision_end|>` (or `<|video|>`), and only the inner `<|image|>` / `<|video|>` token is
/// expanded. Images become `t*h*w/4` image tokens; each video frame group becomes
/// `<{s:.1f} seconds><|vision_start|>` + `h*w/4` video tokens + `<|vision_end|>`, nested
/// inside the template's own pair, exactly as the reference processor builds it.
public struct LFM3VLProcessor: UserInputProcessor {
    private let config: Qwen3VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Qwen3VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    static let imageToken = "<|image|>"
    static let videoToken = "<|video|>"

    private var factor: Int { config.patchSize * config.mergeSize }

    /// The reference upscales anything thinner than one merge block instead of refusing it.
    private func minimumExtent(_ h: Int, _ w: Int) -> (Int, Int) {
        let shortest = min(h, w)
        guard shortest < factor, shortest > 0 else { return (h, w) }
        let scale = Double(factor) / Double(shortest)
        return (Int(Double(h) * scale), Int(Double(w) * scale))
    }

    func patches(for image: CIImage, processing: UserInput.Processing?) throws -> (MLXArray, THW) {
        let source = MediaProcessing.apply(image, processing: processing)
        let (extentH, extentW) = try QwenVL.intExtent(source.extent.size)
        let (h, w) = minimumExtent(extentH, extentW)
        let (resizedH, resizedW) = try QwenVL.targetSize(
            height: h, width: w, factor: factor,
            minPixels: config.size.minPixels, maxPixels: config.size.maxPixels)
        let size = CGSize(width: resizedW, height: resizedH)
        let pixels = MediaProcessing.asMLXArrayUnmanaged(
            MediaProcessing.normalize(
                MediaProcessing.resampleBicubic(source, to: size),
                mean: config.imageMeanTuple, std: config.imageStdTuple))
        return try QwenVL.patchify(
            images: [pixels], mergeSize: config.mergeSize, patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize)
    }

    private func tokenId(_ token: String) throws -> Int {
        guard let id = tokenizer.convertTokenToId(token) else {
            throw VLMError.processing("LFM3-VL tokenizer has no \(token)")
        }
        return id
    }

    /// Replaces each occurrence of `placeholder`, in order, with the next replacement, and
    /// carries prefix boundaries (token counts) across the length change.
    static func expand(
        _ tokens: [Int], placeholder: Int, replacements: [[Int]], boundaries: [Int] = []
    ) throws -> (tokens: [Int], boundaries: [Int]) {
        var result: [Int] = []
        result.reserveCapacity(tokens.count + replacements.reduce(0) { $0 + $1.count })
        var shiftAt: [(position: Int, delta: Int)] = []
        var next = 0
        for (position, token) in tokens.enumerated() {
            if token == placeholder {
                guard next < replacements.count else {
                    throw VLMError.processing("LFM3-VL: more placeholders than supplied media")
                }
                result.append(contentsOf: replacements[next])
                shiftAt.append((position, replacements[next].count - 1))
                next += 1
            } else {
                result.append(token)
            }
        }
        guard next == replacements.count else {
            throw VLMError.processing(
                "LFM3-VL: \(replacements.count) media items but \(next) placeholders")
        }
        // A boundary b counts tokens [0, b); every placeholder before it grew by `delta`.
        let mapped = boundaries.map { b in
            b + shiftAt.filter { $0.position < b }.reduce(0) { $0 + $1.delta }
        }
        return (result, mapped)
    }

    /// Expanded prompt tokens, carried boundaries, and the processed images / videos.
    struct VisualMedia {
        var tokens: [Int]
        var boundaries: CanonicalChatCacheBoundaries
        var image: LMInput.ProcessedImage?
        var video: LMInput.ProcessedVideo?
    }

    /// Patchifies `input`'s images and videos and expands their placeholders in `tokens`.
    func expandVisualMedia(
        tokens: [Int], boundaries: CanonicalChatCacheBoundaries, input: UserInput
    ) async throws -> VisualMedia {
        var tokens = tokens
        var boundaries = boundaries
        let mergeArea = config.mergeSize * config.mergeSize

        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let items = try input.images.map {
                try patches(for: try $0.asCIImage(), processing: input.processing)
            }
            processedImage = .init(
                pixels: concatenated(items.map(\.0)), frames: items.map(\.1))
            let id = try tokenId(Self.imageToken)
            let replacements = items.map { Array(repeating: id, count: $0.1.product / mergeArea) }
            let all = try Self.expand(
                tokens, placeholder: id, replacements: replacements, boundaries: boundaries.all)
            let stable = try Self.expand(
                tokens, placeholder: id, replacements: replacements, boundaries: boundaries.stable)
            tokens = all.tokens
            boundaries = CanonicalChatCacheBoundaries(all: all.boundaries, stable: stable.boundaries)
        }

        var processedVideo: LMInput.ProcessedVideo?
        if !input.videos.isEmpty {
            let id = try tokenId(Self.videoToken)
            let start = try tokenId("<|vision_start|>")
            let end = try tokenId("<|vision_end|>")
            var pixelParts: [MLXArray] = []
            var grids: [THW] = []
            var replacements: [[Int]] = []
            for video in input.videos {
                let sequence = try await MediaProcessing.asProcessedVideoFrames(
                    video, targetFPS: { _ in config.videoFPS },
                    minFrames: config.videoMinFrames, maxFrames: config.videoMaxFrames
                ) { frame in
                    VideoFrame(
                        frame: MediaProcessing.apply(frame.frame, processing: input.processing),
                        timeStamp: frame.timeStamp)
                }
                guard let first = sequence.frames.first else {
                    throw VLMError.imageProcessingFailure("No video frames were decoded")
                }
                let (extentH, extentW) = try QwenVL.intExtent(first.frame.extent.size)
                let (h, w) = minimumExtent(extentH, extentW)
                let (resizedH, resizedW) = try Qwen3VLProcessor.videoTargetSize(
                    frameCount: sequence.frames.count, height: h, width: w,
                    temporalFactor: config.temporalPatchSize, factor: factor,
                    minPixels: config.videoMinPixels, maxPixels: config.videoMaxPixels)
                let size = CGSize(width: resizedW, height: resizedH)
                let frames = sequence.frames.map {
                    MediaProcessing.asMLXArrayUnmanaged(
                        MediaProcessing.normalize(
                            MediaProcessing.resampleBicubic($0.frame, to: size),
                            mean: config.imageMeanTuple, std: config.imageStdTuple))
                }
                let (pixels, grid) = try QwenVL.patchify(
                    images: frames, mergeSize: config.mergeSize, patchSize: config.patchSize,
                    temporalPatchSize: config.temporalPatchSize)
                pixelParts.append(pixels)
                grids.append(grid)

                // Group timestamp: midpoint of the group's first and last frame (padding
                // repeats the last frame), printed with one decimal.
                let times = sequence.frames.map { $0.timeStamp.seconds }
                var replacement: [Int] = []
                for group in 0 ..< grid.t {
                    let lo = min(group * config.temporalPatchSize, times.count - 1)
                    let hi = min(lo + config.temporalPatchSize - 1, times.count - 1)
                    let seconds = (times[lo] + times[hi]) / 2
                    let label = String(
                        format: "<%.1f seconds>", locale: Locale(identifier: "en_US_POSIX"),
                        seconds)
                    replacement += tokenizer.encode(text: label, addSpecialTokens: false)
                    replacement.append(start)
                    replacement += Array(repeating: id, count: grid.h * grid.w / mergeArea)
                    replacement.append(end)
                }
                replacements.append(replacement)
            }
            processedVideo = .init(pixels: concatenated(pixelParts), frames: grids)
            let all = try Self.expand(
                tokens, placeholder: id, replacements: replacements, boundaries: boundaries.all)
            let stable = try Self.expand(
                tokens, placeholder: id, replacements: replacements, boundaries: boundaries.stable)
            tokens = all.tokens
            boundaries = CanonicalChatCacheBoundaries(all: all.boundaries, stable: stable.boundaries)
        }

        return VisualMedia(
            tokens: tokens, boundaries: boundaries, image: processedImage, video: processedVideo)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Qwen3VLMessageGenerator().generate(from: input)
        let rendered = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        let media = try await expandVisualMedia(
            tokens: rendered,
            boundaries: canonicalChatCacheBoundaries(
                tokenizer: tokenizer, messages: messages, tools: input.tools,
                additionalContext: input.additionalContext, promptTokens: rendered,
                staticSystemPrefix: input.cacheStableSystemPrefix),
            input: input)
        let (tokens, boundaries) = (media.tokens, media.boundaries)
        let (processedImage, processedVideo) = (media.image, media.video)

        let promptArray = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: promptArray, mask: ones(like: promptArray).asType(.int8), tokenIds: tokens),
            image: processedImage,
            video: processedVideo,
            mediaTokenIds: QwenVL.mediaTokenIds(
                tokenizer: tokenizer, tokens: [Self.imageToken, Self.videoToken]),
            cacheScopeSalt: cacheScopeSalt(from: input.additionalContext),
            cachePrefixTokenCounts: boundaries.all,
            cacheStablePrefixTokenCounts: boundaries.stable,
            toolSchemas: input.tools)
    }
}
