//
//  MuseGlimmer.swift
//  vmlx-swift
//
//  Top-level Muse Glimmer 30B VLM: the text tower from MLXLLM plus the vision
//  tower, adapter and projection. See `MuseGlimmer_ARCHITECTURE.md`.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct MuseGlimmerConfiguration: Codable, Sendable {
    public let textConfiguration: MuseGlimmerTextConfiguration
    public let visionConfiguration: MuseGlimmerVisionConfiguration
    public let imageTokenId: Int
    public let videoTokenId: Int
    /// Width of a merged vision token before the adapter — `hidden * merge²`.
    /// The config's `out_hidden_size` says the same thing, but it reads like a
    /// tower width and is not one, so it is only a cross-check here.
    public let outHiddenSize: Int
    public let projectorHiddenSize: Int

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case outHiddenSize = "out_hidden_size"
        case projectorHiddenSize = "projector_hidden_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // The text config decoder accepts either the nested or flat shape, so
        // hand it the whole document.
        textConfiguration = try MuseGlimmerTextConfiguration(from: decoder)
        visionConfiguration = try c.decode(
            MuseGlimmerVisionConfiguration.self, forKey: .visionConfig)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 200_092
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 200_091
        projectorHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .projectorHiddenSize) ?? 4096
        let merged =
            visionConfiguration.hiddenSize * visionConfiguration.mergeSize
            * visionConfiguration.mergeSize
        outHiddenSize = try c.decodeIfPresent(Int.self, forKey: .outHiddenSize) ?? merged
    }

    /// Written by hand because the `CodingKeys` case names deliberately differ
    /// from the property names (`textConfig` vs `textConfiguration`), which
    /// blocks synthesis.
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(textConfiguration, forKey: .textConfig)
        try c.encode(visionConfiguration, forKey: .visionConfig)
        try c.encode(imageTokenId, forKey: .imageTokenId)
        try c.encode(videoTokenId, forKey: .videoTokenId)
        try c.encode(outHiddenSize, forKey: .outHiddenSize)
        try c.encode(projectorHiddenSize, forKey: .projectorHiddenSize)
    }
}

// MARK: - Model

public class MuseGlimmer: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionTower: MuseGlimmerVisionModel
    @ModuleInfo(key: "vision_adapter") private var visionAdapter: MuseGlimmerVisionProjector
    @ModuleInfo(key: "vision_projection") private var visionProjection: Linear
    @ModuleInfo(key: "language_model") private var languageModel: MuseGlimmerTextModel

    public let config: MuseGlimmerConfiguration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.loraLayers }

    public init(_ config: MuseGlimmerConfiguration) {
        self.config = config
        self._visionTower.wrappedValue = MuseGlimmerVisionModel(config.visionConfiguration)
        self._visionAdapter.wrappedValue = MuseGlimmerVisionProjector(
            inputDimensions: config.outHiddenSize,
            hiddenDimensions: config.projectorHiddenSize)
        self._visionProjection.wrappedValue = Linear(
            config.projectorHiddenSize, config.textConfiguration.hiddenSize, bias: false)
        self._languageModel.wrappedValue = MuseGlimmerTextModel(config.textConfiguration)
        super.init()
    }

    /// Vision tower → 2×2-merged tokens → adapter → projection into the text
    /// width, then scattered over the `<|patch|>` / `<|video|>` placeholders.
    private func visionFeatures(_ pixels: MLXArray, frames: [THW]) -> MLXArray {
        visionProjection(visionAdapter(visionTower(pixels, frames: frames)))
    }

    private func inputEmbeddings(
        inputIds: MLXArray, pixelValues: MLXArray?, frames: [THW]?
    ) -> MLXArray {
        guard let pixelValues, let frames, !frames.isEmpty else {
            return languageModel.model.embedTokens(inputIds[.newAxis, .ellipsis])
        }

        let inputEmbeds = languageModel.model.embedTokens(inputIds)
        var features = visionFeatures(pixelValues, frames: frames)
        if features.ndim == 2 {
            features = features[.newAxis, 0..., 0...]
        }

        return QwenVL.mergeInputIdsWithImageFeatures(
            inputIds: inputIds, inputEmbeds: inputEmbeds, imageFeatures: features,
            imageTokenId: config.imageTokenId,
            videoTokenId: config.videoTokenId)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        // The vision tower is unquantized F16 even in the JANG bundles, so the
        // dtype comes from the tower rather than being assumed to match the
        // quantized text tower.
        let dtype = visionTower.lnPre.weight?.dtype ?? .float16

        var allPixels: MLXArray?
        var allFrames: [THW] = []

        if let pixels = input.image?.pixels, let frames = input.image?.frames {
            allPixels = pixels.asType(dtype)
            allFrames.append(contentsOf: frames)
        }
        if let pixels = input.video?.pixels, let frames = input.video?.frames {
            let converted = pixels.asType(dtype)
            allPixels = allPixels.map { concatenated([$0, converted]) } ?? converted
            allFrames.append(contentsOf: frames)
        }

        let embeddings = inputEmbeddings(
            inputIds: input.text.tokens, pixelValues: allPixels,
            frames: allFrames.isEmpty ? nil : allFrames)

        let hidden = languageModel.model(nil, inputEmbedding: embeddings, cache: cache)
        let logits = MuseGlimmerTextModel.applyLogitTail(
            languageModel.lmHead(hidden), config: config.textConfiguration)
        return .logits(LMOutput(logits: logits))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    /// The checkpoint stores the text tower under `language_model.` and the
    /// vision stack under `model.vision_*`; both need rehoming onto this
    /// module's own key layout.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            var rehomed = key
            if rehomed.hasPrefix("model.vision_") {
                rehomed = String(rehomed.dropFirst("model.".count))
            }
            out[rehomed] = value
        }
        return out
    }
}
