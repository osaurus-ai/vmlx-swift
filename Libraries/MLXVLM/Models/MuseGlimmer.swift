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

    /// `RotatingKVCache`'s stock allocation step. Prefill chunks must not
    /// exceed it — see the note in `prepare`.
    static let rotatingCacheGrowthStep = 256

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
            // `input.text.tokens` already arrives as `(1, T)` on this path.
            // Adding `.newAxis` unconditionally (as the Qwen VLMs do, where the
            // ids are 1-D) yields `(1, 1, T, hidden)`, and then the prefill
            // chunk slice cuts axis 1 — a size-1 axis — instead of the token
            // axis. The forward pass still runs, so the only symptom is a
            // non-3-D logits tensor blowing up later in `convertToToken`.
            let ids = inputIds.ndim == 1 ? inputIds[.newAxis, .ellipsis] : inputIds
            // `embed`, not `embedTokens`: the checkpoint's embedding is normed.
            return languageModel.model.embed(ids)
        }

        let inputEmbeds = languageModel.model.embed(inputIds)
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

        // Prefill has to be chunked, not fed in one pass.
        //
        // 39 of the 52 layers hold a `RotatingKVCache` bounded by the 2048
        // sliding window. A single forward pass carrying a prompt longer than
        // that window overruns the cache's in-place write and trips a
        // precondition inside the MLX scatter — the crash is in the indexing
        // layer, which makes it look like a shape bug rather than a prefill
        // policy bug. `LLMModel.prepare()` chunks for exactly this reason;
        // VLMs that skip the chunking (Qwen2.5-VL) only get away with it
        // because their caches are non-rotating.
        // Capped to `rotatingCacheGrowthStep`, NOT to the sliding window.
        //
        // `RotatingKVCache.updateInPlace` allocates `step` rows at a time and
        // writes the whole incoming chunk into that block, so the chunk must
        // fit the cache's growth step. Sizing the step in `newCache` is not
        // enough: the host's cache coordinator builds these caches itself from
        // the model topology and uses the stock 256 default, so the only value
        // this path can rely on is the default.
        let step = max(
            1,
            min(
                windowSize ?? MuseGlimmer.rotatingCacheGrowthStep,
                MuseGlimmer.rotatingCacheGrowthStep,
                config.textConfiguration.slidingWindow))
        let total = embeddings.dim(1)
        var offset = 0
        var hidden: MLXArray?

        while offset < total {
            var end = min(offset + step, total)
            // Never leave a trailing 1-token chunk.
            //
            // `RotatingKVCache.update` routes `S == 1` to `updateInPlace` and
            // everything else to `updateConcat`. After a concat, `idx` equals
            // the full buffer width; a following single-token in-place write
            // then targets `idx ..< idx+1`, which is one row past the end
            // whenever the buffer has not also grown. A disk-restored cache
            // reaches exactly that state, so a prompt of length `k*step + 1`
            // crashes while `k*step + 2` is fine. Absorbing the stray token
            // into the previous chunk keeps every write on the concat path.
            if total - end == 1 { end = total }
            let chunk = embeddings[0..., offset ..< end, 0...]
            hidden = languageModel.model(nil, inputEmbedding: chunk, cache: cache)
            // Keep the graph from growing across the whole prompt.
            eval(cache)
            offset = end
        }

        guard let hidden else {
            throw VLMError.processing("MuseGlimmer: empty prompt")
        }

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
        // The VLM does not route through `MuseGlimmerTextModel.sanitize`, so the
        // centered-norm fold has to happen here too. Without it the text tower
        // loads unfolded gains into a forward that no longer adds the `+1`,
        // which leaves every one of its norms at zero gain — the model still
        // loads and still emits text, and only a behavioural probe catches it.
        for (key, value) in out where MuseGlimmerTextModel.isCenteredNormWeight(key) {
            out[key] = value + 1
        }
        return out
    }
}
