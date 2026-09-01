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
    /// Optional, following `Gemma4Configuration.audioConfig`. A text-only Muse Glimmer bundle has
    /// no `vision_config`, and until this was optional such a bundle could not DECODE this
    /// configuration at all — the first of the reasons a second `muse_glimmer` registration
    /// (`MuseGlimmerTextConfiguration` / `MuseGlimmerTextModel`) exists in LLMModelFactory.
    ///
    /// It is NOT the only reason, and this change does not retire that entry — because the entry
    /// is not in fact a duplicate. `muse_glimmer` names two different BUNDLE SHAPES: one with a
    /// vision section, one without. The two registrations serve one each, and `ModelFactory`'s
    /// `load(loader:)` already routes between them by content: `ModelFactoryRegistry` seeds
    /// itself with an `MLXVLM` trampoline ahead of an `MLXLLM` one, and the loop falls through on
    /// ANY error, not only `unsupportedModelType`. So a text-only bundle is tried against the VLM
    /// factory, fails there when `loadProcessorConfig` finds no `preprocessor_config.json`
    /// (surfacing as `ModelFactoryError.configurationFileError`), and is then served by the LLM
    /// factory. That is content-based dispatch, not redundancy.
    ///
    /// What THIS change buys is orthogonal to that routing, and is the actual point: a bundle
    /// that does carry a vision section can now be constructed text-only, leaving the tower
    /// unallocated. On a 30B Muse Glimmer that spares 3.84 GB. Nothing about the two-entry
    /// routing needs to change for it, and retiring either entry would remove a bundle shape's
    /// only route.
    public let visionConfiguration: MuseGlimmerVisionConfiguration?
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
        visionConfiguration = try c.decodeIfPresent(
            MuseGlimmerVisionConfiguration.self, forKey: .visionConfig)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 200_092
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 200_091
        projectorHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .projectorHiddenSize) ?? 4096
        // Derived from the tower when there is one; a declared value still wins. With no tower the
        // adapter is never built, so the value is unused rather than wrong — 0 is honest here.
        let merged = visionConfiguration.map {
            $0.hiddenSize * $0.mergeSize * $0.mergeSize
        }
        outHiddenSize = try c.decodeIfPresent(Int.self, forKey: .outHiddenSize) ?? merged ?? 0
    }

    /// Written by hand because the `CodingKeys` case names deliberately differ
    /// from the property names (`textConfig` vs `textConfiguration`), which
    /// blocks synthesis.
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(textConfiguration, forKey: .textConfig)
        try c.encodeIfPresent(visionConfiguration, forKey: .visionConfig)
        try c.encode(imageTokenId, forKey: .imageTokenId)
        try c.encode(videoTokenId, forKey: .videoTokenId)
        try c.encode(outHiddenSize, forKey: .outHiddenSize)
        try c.encode(projectorHiddenSize, forKey: .projectorHiddenSize)
    }
}

// MARK: - Model

public class MuseGlimmer: Module, VLMModel, KVCacheDimensionProvider, ModalityBearing,
    ModelComponentMapping
{

    @ModuleInfo(key: "vision_tower") private var visionTower: MuseGlimmerVisionModel?
    @ModuleInfo(key: "vision_adapter") private var visionAdapter: MuseGlimmerVisionProjector?
    @ModuleInfo(key: "vision_projection") private var visionProjection: Linear?
    @ModuleInfo(key: "language_model") private var languageModel: MuseGlimmerTextModel

    /// `RotatingKVCache`'s stock allocation step. Prefill chunks must not
    /// exceed it — see the note in `prepare`.
    static let rotatingCacheGrowthStep = 256

    public let config: MuseGlimmerConfiguration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.loraLayers }

    /// What this instance actually carries — not what the bundle advertises. The bundle-level
    /// claim lives in `ModelRuntimeCapabilitySnapshot.supportsVision` and answers a different
    /// question; see `ModelConstructionModalities.swift` for why the two are kept apart.
    public let modalities: Set<ModelRuntimeRequestModality>

    public convenience init(_ config: MuseGlimmerConfiguration) {
        // Unchanged behaviour for every existing caller: build everything the config permits.
        //
        // Non-throwing by CONSTRUCTION, not by argument. `resolveForConstruction` can only throw
        // when a request exceeds what is constructible, and this path makes no request — but
        // "provably unreachable" is a property of the function above it, and that function can be
        // edited. Passing the constructible set directly removes the possibility instead of
        // reasoning about it.
        // nil = everything the configuration offers, which cannot fail validation.
        self.init(config, plan: try! Self.resolveConstruction(config, requesting: nil))
    }

    /// Which towers this configuration can actually instantiate.
    ///
    /// Structural, and deliberately not read from the bundle's `supports_vision` stamp: a config
    /// with no vision section cannot build a vision tower no matter what the stamp claims, and a
    /// mismatch between the two is a broken conversion worth surfacing rather than smoothing over.
    public static func constructibleModalities(
        of config: MuseGlimmerConfiguration
    ) -> Set<ModelRuntimeRequestModality> {
        config.visionConfiguration != nil ? [.text, .vision, .video] : [.text]
    }

    /// - Parameter requesting: the caller's subset, or nil for "everything this config offers".
    ///   Asking for a modality the config cannot build throws rather than yielding a model that
    ///   fails later at `prepare()`.
    public convenience init(
        _ config: MuseGlimmerConfiguration,
        requesting: Set<ModelRuntimeRequestModality>?
    ) throws {
        self.init(config, plan: try Self.resolveConstruction(config, requesting: requesting))
    }


    /// What the built modules serve. `.text` is this family's core being a text LM, not a default.
    public static func servedModalities(
        by components: Set<ModelComponent>, of config: MuseGlimmerConfiguration
    ) -> Set<ModelRuntimeRequestModality> {
        var m: Set<ModelRuntimeRequestModality> = []
        if components.contains(.languageCore) { m.insert(.text) }
        if components.contains(.visionTower) { m.formUnion([.vision, .video]) }
        return m
    }

    /// Which MODULES a request needs. `.vision` and `.video` are different lanes served by the SAME
    /// tower, so gating on `.vision` alone accepted `requesting: [.text, .video]` and then built no
    /// tower at all — the late `prepare()` failure this whole mechanism exists to prevent.
    public static func components(
        for requested: Set<ModelRuntimeRequestModality>, of config: MuseGlimmerConfiguration
    ) -> Set<ModelComponent> {
        var out: Set<ModelComponent> = []
        if requested.contains(.vision) || requested.contains(.video) { out.insert(.visionTower) }
        return out
    }

    /// What was actually built.
    public let plan: ResolvedConstructionPlan

    /// The one real initialiser. Private: a plan can only come from `resolveConstruction`, so no
    /// caller can construct an empty or unsupported instance.
    private init(
        _ config: MuseGlimmerConfiguration,
        plan: ResolvedConstructionPlan
    ) {
        self.modalities = plan.served
        self.plan = plan
        self.config = config
        // Built only when the bundle has a tower AND the caller asked for it. Skipping it is not a
        // degraded mode: it is the text-only load that previously required a separate registration,
        // and on a 30B bundle it leaves 3.84 GB of vision weights unallocated.
        if config.visionConfiguration != nil, plan.builds(.visionTower) {
            self._visionTower.wrappedValue = MuseGlimmerVisionModel(config.visionConfiguration!)
            self._visionAdapter.wrappedValue = MuseGlimmerVisionProjector(
                inputDimensions: config.outHiddenSize,
                hiddenDimensions: config.projectorHiddenSize)
            self._visionProjection.wrappedValue = Linear(
                config.projectorHiddenSize, config.textConfiguration.hiddenSize, bias: false)
        }
        self._languageModel.wrappedValue = MuseGlimmerTextModel(config.textConfiguration)
        super.init()
    }

    /// Vision tower → 2×2-merged tokens → adapter → projection into the text
    /// width, then scattered over the `<|patch|>` / `<|video|>` placeholders.
    /// Nil when this instance carries no vision tower — either the bundle has none, or the caller
    /// asked for text only. Callers must already handle "no image embeddings"; that path exists for
    /// text-only prompts to a multimodal model.
    private func visionFeatures(_ pixels: MLXArray, frames: [THW]) -> MLXArray? {
        guard let visionTower, let visionAdapter, let visionProjection else { return nil }
        return visionProjection(visionAdapter(visionTower(pixels, frames: frames)))
    }

    private func inputEmbeddings(
        inputIds: MLXArray, pixelValues: MLXArray?, frames: [THW]?
    ) throws -> MLXArray {
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
        // Images arrived at a model that carries no vision tower. Refuse rather than embed the
        // prompt without them: a silently image-free answer looks like a bad model, not a bad call.
        // Same stance as Gemma4's video guard — "do not silently generate over missing embeddings".
        guard var features = visionFeatures(pixelValues, frames: frames) else {
            throw VLMError.processing(
                "image input requires the vision tower, and this instance carries none "
                + "(modalities: \(modalities.modalityDescription)). "
                + "Load with .vision requested, or send text only.")
        }
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
        // With no tower there is nothing to convert — the fallback was already the default here.
        let dtype = visionTower?.lnPre.weight?.dtype ?? .float16

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

        let embeddings = try inputEmbeddings(
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
                // No tower to receive them. A multimodal bundle loaded text-only still SHIPS its
                // vision weights; rehoming them onto a module that was never built leaves keys with
                // no destination. Dropping them is what MuseGlimmerTextModel.sanitize already does,
                // and what Gemma4 does with `audio_tower.*` on audio-less bundles.
                // The MODULE, not the lane: a video-only instance HAS the tower, so its
                // weights must survive. Asking about `.vision` here would drop the weights
                // of a tower that was just built.
                guard plan.builds(.visionTower) else { continue }
                rehomed = String(rehomed.dropFirst("model.".count))
            }
            out[rehomed] = value
        }
        // The VLM does not route through `MuseGlimmerTextModel.sanitize` — the key mapping
        // genuinely differs, since here the text tower is a SUBMODULE and keeps its
        // `language_model.` prefix. But the fold itself must be identical on both paths, so it
        // is called, not copied: `foldCenteredNorms` is its single owner. Copying it is what
        // this file used to do, and the drift failure is silent — unfolded gains against a
        // forward that no longer adds the `+1` leave every norm at zero gain, and the model
        // still loads and still emits text.
        return MuseGlimmerTextModel.foldCenteredNorms(out)
    }
}
