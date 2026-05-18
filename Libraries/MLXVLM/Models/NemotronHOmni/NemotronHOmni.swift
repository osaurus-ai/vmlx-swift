// NemotronHOmni.swift
// Native Swift multimodal wrapper for Nemotron-3-Nano-Omni-30B-A3B-Reasoning.
//
// Combines:
//   • LLM (NemotronHModel from MLXLLM)
//   • RADIO ViT vision tower
//   • Parakeet Conformer audio encoder
//   • mlp1 vision projector + sound_projection audio projector
//
// Mirrors jang_tools/nemotron_omni/model.py NemotronHOmni.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import CoreImage

// MARK: - Configuration

/// Top-level config for NemotronHOmni.
/// Decoded from the omni bundle's `config.json` (which is the LLM config —
/// the wrapper hardcodes the multimodal dims since they are fixed in V3).
public struct NemotronHOmniConfiguration: Codable, Sendable {
    public let llmConfig: NemotronHConfiguration

    // Multimodal dims — fixed for Nemotron-3-Nano-Omni V3 (matches config_omni.json).
    public let imageSize: Int
    public let downsampleRatio: Float
    public let vitHiddenSize: Int
    public let visionPatchSize: Int
    public let visionNumBlocks: Int
    public let visionNumHeads: Int
    public let visionNumClsTokens: Int
    public let visionMaxGrid: Int
    public let projectorHiddenSize: Int

    public let soundHiddenSize: Int
    public let soundNumLayers: Int
    public let soundNumHeads: Int
    public let soundFFHidden: Int
    public let soundConvKernel: Int
    public let soundProjectionHidden: Int
    public let soundNumMelBins: Int
    public let soundSampleRate: Int

    public let imageContextTokenId: Int
    public let videoContextTokenId: Int
    public let soundContextTokenId: Int

    /// Non-nil when the bundle is JANGTQ-quantized — opts the LLM
    /// backbone's routed-expert switch_mlp into TurboQuantSwitchLinear
    /// instead of the affine SwitchLinear. Resolved at decode time
    /// from `weight_format` + `mxtq_bits` injected into config.json by
    /// the factory layer (see `VLMModelFactory._load` jang merge).
    public let jangtqContext: NemotronHJANGTQContext?

    enum JANGTQCodingKeys: String, CodingKey {
        case weightFormat = "weight_format"
        case mxtqBits = "mxtq_bits"
        case mxtqSeed = "mxtq_seed"
    }

    public init(from decoder: Decoder) throws {
        // The bundle's config.json is the LLM config directly. Decode it as
        // NemotronHConfiguration; multimodal dims are fixed defaults.
        self.llmConfig = try NemotronHConfiguration(from: decoder)

        // Hardcoded V3 multimodal dims (match config_omni.json from
        // OsaurusAI/Nemotron-3-Nano-Omni-30B-A3B-{MXFP4,JANGTQ4,JANGTQ2}).
        self.imageSize = 512
        self.downsampleRatio = 0.5
        self.vitHiddenSize = 1280
        self.visionPatchSize = 16
        self.visionNumBlocks = 32
        self.visionNumHeads = 16
        self.visionNumClsTokens = 10
        self.visionMaxGrid = 128
        self.projectorHiddenSize = 20480

        self.soundHiddenSize = 1024
        self.soundNumLayers = 24
        self.soundNumHeads = 8
        self.soundFFHidden = 4096
        self.soundConvKernel = 9
        self.soundProjectionHidden = 4096
        self.soundNumMelBins = 128
        self.soundSampleRate = 16000

        self.imageContextTokenId = 18
        self.videoContextTokenId = 131_081
        self.soundContextTokenId = 27

        // Detect JANGTQ from the merged config.json (the factory layer
        // injects `weight_format` + `mxtq_bits` from `jang_config.json`
        // into the config.json data before decode — see
        // `VLMModelFactory._load`'s JANG merge).
        let c = try? decoder.container(keyedBy: JANGTQCodingKeys.self)
        let weightFormat = (try? c?.decodeIfPresent(String.self, forKey: .weightFormat)) ?? nil
        if weightFormat == "mxtq" {
            let bits = (try? c?.decodeIfPresent(Int.self, forKey: .mxtqBits)) ?? 2
            let seed = (try? c?.decodeIfPresent(Int.self, forKey: .mxtqSeed)) ?? 42
            self.jangtqContext = NemotronHJANGTQContext(bits: bits, mxtqSeed: seed)
        } else {
            self.jangtqContext = nil
        }
    }

    public func encode(to encoder: Encoder) throws {
        try llmConfig.encode(to: encoder)
    }
}

// MARK: - Multimodal model

public class NemotronHOmni: Module, VLMModel, KVCacheDimensionProvider, LoRAModel {

    @ModuleInfo(key: "language_model") private var languageModel: NemotronHModel

    // Tower modules. The on-disk weights for these are fp16/bf16 (NOT
    // quantized); sanitize() routes them through the remap helpers.
    //
    // NOTE: @ModuleInfo keys must be single-segment (no dots). Multi-level
    // namespaces from the bundle's safetensors keys are flattened by
    // sanitize() into one-segment paths that match these keys directly.
    @ModuleInfo(key: "vision_model") private var radioModel: NemotronHRADIOVisionModel
    @ModuleInfo(key: "mlp1") private var visionMLP: NemotronHVisionMLPProjector
    @ModuleInfo(key: "sound_encoder") private var soundEncoder: NemotronHParakeetEncoder
    @ModuleInfo(key: "sound_projection") private var soundProjection: NemotronHSoundProjector

    public let config: NemotronHOmniConfiguration

    public var vocabularySize: Int { languageModel.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.loraLayers }

    public init(_ config: NemotronHOmniConfiguration) {
        self.config = config

        if let jangtq = config.jangtqContext {
            self._languageModel.wrappedValue = NemotronHModel(
                jangtqContext: jangtq, configuration: config.llmConfig)
        } else {
            self._languageModel.wrappedValue = NemotronHModel(config.llmConfig)
        }
        self._radioModel.wrappedValue = NemotronHRADIOVisionModel(
            embedDim: config.vitHiddenSize,
            numBlocks: config.visionNumBlocks,
            numHeads: config.visionNumHeads,
            patchSize: config.visionPatchSize,
            numClsTokens: config.visionNumClsTokens,
            maxGrid: config.visionMaxGrid)
        // Post-pixel-shuffle dim = vit_hidden * (1/downsample_ratio)^2 = 1280 * 4 = 5120
        let postShuffleDim = config.vitHiddenSize
            * Int(round(1.0 / config.downsampleRatio))
            * Int(round(1.0 / config.downsampleRatio))
        self._visionMLP.wrappedValue = NemotronHVisionMLPProjector(
            inDim: postShuffleDim,
            projectorDim: config.projectorHiddenSize,
            llmDim: config.llmConfig.hiddenSize)

        self._soundEncoder.wrappedValue = NemotronHParakeetEncoder(
            hiddenSize: config.soundHiddenSize,
            numLayers: config.soundNumLayers,
            numHeads: config.soundNumHeads,
            ffHidden: config.soundFFHidden,
            convKernel: config.soundConvKernel)
        self._soundProjection.wrappedValue = NemotronHSoundProjector(
            soundHidden: config.soundHiddenSize,
            projectionHidden: config.soundProjectionHidden,
            llmHidden: config.llmConfig.hiddenSize)
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    /// LM hot path — takes raw token IDs and produces logits (text-only).
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel.callAsFunction(inputs, cache: cache)
    }

    /// VLM prepare — accepts LMInput with text + optional image / video /
    /// audio. Each non-text modality gets encoded by its tower and
    /// spliced into the token-embedding sequence at its placeholder
    /// positions before the LLM forward pass.
    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let convertedCache = cache.compactMap { $0 as KVCache }

        if input.image == nil && input.video == nil && input.audio == nil {
            // Text-only path. We deliberately return `.logits` (run the
            // prefill ourselves) rather than `.tokens(input.text)` because
            // `BatchEngine.stepPrefill` calls
            //     context.model(remainingText[text: .newAxis], ...)
            // for the `.tokens` branch, adding an extra axis on top of
            // the already-2D `[1, T]` token tensor that processors emit.
            // For omni's hybrid Mamba layers a 3D token input cascades
            // into a 4-vs-3-dim concat trap inside `applyConv` —
            // observed crash on the BatchEngine omni text-only path.
            //
            // 2026-04-30 (Bug 2 fix): the previous implementation ran
            // the ENTIRE prompt unchunked through the model. For prompts
            // > ~8k tokens the SSM-attention path in `ssmAttn` (SSM.swift)
            // materializes a `[B, n_heads, L, L]` segsum tensor that grows
            // O(L²): 34 GB per Mamba layer at L=16k bf16, multiplied by
            // 23 sequential Mamba layers = peaks of 100s of GB on long
            // prompts. Repro under `OSAURUS_MLX_MALLOC_TRACE=1` showed
            // single 298 GiB ternary_op allocations during the segsum
            // `which` mask + the `surrogateAttentionMatrix.matmul(dtx)`.
            //
            // Fix: chunked prefill mirroring `LLMModel.prepare`. Mamba
            // layers carry running state across chunks via `MambaCache`
            // (that's what the cache is for); attention layers update
            // KV in place. Each chunk materializes lazily-built
            // intermediates and clears Metal cache before the next chunk
            // runs, bounding peak allocation to O(chunk_size²) per layer
            // instead of O(prompt_length²). We always return `.logits` so
            // the BatchEngine never re-axises this output and the .newAxis
            // trap stays dodged.
            let prefillStepSize = windowSize ?? 512
            let tokensShape = input.text.tokens.shape
            if tokensShape.count >= 2 && tokensShape[0] != 1 {
                fatalError(
                    "NemotronHOmni.prepare expects single-sequence input (batch=1), "
                    + "got shape \(tokensShape).")
            }
            var flatTokens = input.text.tokens.reshaped([-1])
            while flatTokens.size > prefillStepSize {
                let chunkTokens = flatTokens[..<prefillStepSize][.newAxis, 0...]
                _ = languageModel.callAsFunction(
                    chunkTokens, cache: convertedCache)
                MLX.eval(convertedCache)
                flatTokens = flatTokens[prefillStepSize...]
                Memory.clearCache()
            }
            let lastChunk = flatTokens[.newAxis, 0...]
            let logits = languageModel.callAsFunction(
                lastChunk, cache: convertedCache)
            return .logits(LMOutput(logits: logits))
        }

        // Build embeddings for tokens + splice multimodal at placeholder tokens.
        let textEmbeds = languageModel.embedTokens(input.text.tokens)
        var spliced = textEmbeds
        // Image and video share the same `<image>` placeholder per Python
        // model.py (img_context_token_id is reused for both — the
        // distinguishing factor is which tower produced the embedding).
        // The processor emits placeholders in image-first-then-video order
        // and `mask == imageContextTokenId` matches BOTH groups in one
        // sweep — so splicing image and video separately would either
        // (a) trip the placeholder-count precondition (mask matches
        // image+video tokens but replacement only has image rows), or
        // (b) silently overwrite image embeddings with video embeddings.
        // Concatenate image and video embeds (in the same order the
        // processor wrote their placeholders) and splice in one pass.
        var visualEmbeds: MLXArray? = nil
        if let pixelValues = input.image?.pixels {
            visualEmbeds = extractImageEmbeds(pixelValues: pixelValues)
        }
        if let video = input.video {
            let videoEmbeds = extractVideoEmbeds(
                pixelValues: video.pixels,
                targetTokenCount: video.embeddingTokenCount)
            visualEmbeds = visualEmbeds.map {
                MLX.concatenated([$0, videoEmbeds], axis: 0)
            } ?? videoEmbeds
        }
        if let visualEmbeds {
            spliced = spliceAtToken(
                tokens: input.text.tokens,
                inputsEmbeds: spliced,
                replacement: visualEmbeds,
                tokenId: config.imageContextTokenId)
        }
        if let audio = input.audio {
            // Use the pre-encoded embedding when the processor already
            // ran Parakeet (avoids re-encoding the same audio across
            // turns); otherwise encode the raw waveform now.
            let audioEmbeds: MLXArray = audio.preEncodedEmbedding
                ?? extractAudioEmbeds(waveformArray: audio.waveform,
                                      sampleRate: audio.sampleRate)
            spliced = spliceAtToken(
                tokens: input.text.tokens,
                inputsEmbeds: spliced,
                replacement: audioEmbeds,
                tokenId: config.soundContextTokenId)
        }

        let logits = languageModel.callAsFunction(
            inputsEmbeds: spliced, cache: convertedCache)
        return .logits(LMOutput(logits: logits))
    }

    // MARK: - Multimodal embedding extraction

    /// Run RADIO + mlp1 on a (B, 3, H, W) pixel tensor (already CLIP-normalized).
    /// Returns flat (totalTokens, llmHidden) embeddings in tile-row-major order.
    public func extractImageEmbeds(pixelValues: MLXArray, video: Bool = false) -> MLXArray {
        var feats = radioModel(pixelValues, video: video)
        // Strip cls/register tokens (first numClsTokens)
        feats = feats[0..., config.visionNumClsTokens..., 0...]
        // Reshape (N, P, D) → (N, h, w, D) using the dynamic-resolution
        // patch grid from the actual pixel tensor. The source processor can
        // emit non-square images (for example 736x384), so sqrt(P) is not a
        // valid production contract.
        let N = feats.dim(0)
        let P = feats.dim(1)
        let D = feats.dim(2)
        let gridH = pixelValues.dim(2) / config.visionPatchSize
        let gridW = pixelValues.dim(3) / config.visionPatchSize
        precondition(
            gridH * gridW == P,
            "RADIO patch count \(P) does not match pixel grid \(gridH)x\(gridW)")
        feats = feats.reshaped([N, gridH, gridW, D])
        // Pixel shuffle (scale = 0.5)
        feats = nemotronOmniPixelShuffle(feats, scaleFactor: config.downsampleRatio)
        // Flatten spatial dims → (N, tokens, post_shuffle_dim)
        let tokens = feats.dim(1) * feats.dim(2)
        let cIn = feats.dim(3)
        feats = feats.reshaped([N, tokens, cIn])
        // mlp1 projector → (N, tokens, llm_hidden)
        feats = visionMLP(feats)
        // Flatten to (N*tokens, llm_hidden)
        return feats.reshaped([N * tokens, feats.dim(-1)])
    }

    /// Run RADIO's video embedder, then apply Efficient Video Sampling
    /// before the placeholders are spliced into the language-model prompt.
    /// The source runtime builds its prompt from the post-EVS token count, so
    /// video generation must not feed the full redundant 32-frame token stream.
    public func extractVideoEmbeds(
        pixelValues: MLXArray,
        targetTokenCount: Int? = nil,
        applyEVS: Bool = true,
        pruningRate: Float = 0.7
    ) -> MLXArray {
        var feats = radioModel(pixelValues, video: true)
        feats = feats[0..., config.visionNumClsTokens..., 0...]
        let nGroups = feats.dim(0)
        let patches = feats.dim(1)
        let hidden = feats.dim(2)
        let side = Int(Double(patches).squareRoot())
        precondition(side * side == patches,
                     "RADIO patch count must be a perfect square; got P=\(patches)")
        feats = feats.reshaped([nGroups, side, side, hidden])
        feats = nemotronOmniPixelShuffle(feats, scaleFactor: config.downsampleRatio)
        let tokensPerGroup = feats.dim(1) * feats.dim(2)
        let cIn = feats.dim(3)
        feats = feats.reshaped([nGroups, tokensPerGroup, cIn])
        feats = visionMLP(feats)
        if applyEVS {
            if let targetTokenCount {
                feats = nemotronOmniApplyEVS(feats, targetTokenCount: targetTokenCount)
            } else {
                feats = nemotronOmniApplyEVS(feats, pruningRate: pruningRate)
            }
        }
        return feats.reshaped([feats.dim(0) * feats.dim(1), feats.dim(-1)])
    }

    /// Run STFT + Parakeet + sound_projection on a mono waveform stored
    /// as an `MLXArray` (any rate; resampled to 16 kHz internally if
    /// necessary). Convenience wrapper around the [Float] form for
    /// `LMInput.ProcessedAudio` consumers.
    public func extractAudioEmbeds(waveformArray: MLXArray, sampleRate: Int = 16_000) -> MLXArray {
        // Flatten to mono Float32 array. ProcessedAudio.waveform is
        // typically shape `[1, samples]` or `[samples]`; both flatten
        // to `[samples]`.
        let flat = waveformArray.reshaped([-1]).asType(.float32)
        let pcm = flat.asArray(Float.self)
        // If sample rate differs from the model's required rate the
        // raw mel STFT will be off — but ProcessedAudio is documented
        // as "model handles resampling". Linear resample to 16 kHz
        // when needed (cheap; AVAudioConverter is the file path that
        // already gets us 16 kHz, but in-memory PCM may arrive at any
        // rate).
        let pcm16k: [Float] =
            sampleRate == config.soundSampleRate
            ? pcm : linearResamplePCM(pcm, fromRate: sampleRate, toRate: config.soundSampleRate)
        return extractAudioEmbeds(waveform: pcm16k)
    }

    /// Run STFT + Parakeet + sound_projection on a 16 kHz mono waveform.
    /// Returns flat (frames, llmHidden) embeddings.
    public func extractAudioEmbeds(waveform: [Float]) -> MLXArray {
        let mel = nemotronOmniExtractMelFeatures(
            waveform,
            sampleRate: config.soundSampleRate,
            nMels: config.soundNumMelBins)
        var feats = soundEncoder(mel) // (1, F_sub, 1024)
        feats = soundProjection(feats) // (1, F_sub, llm_hidden)
        let f = feats.dim(1)
        let h = feats.dim(2)
        return feats.reshaped([f, h])
    }

    /// Splice `replacement` embeddings at every position where `tokens == tokenId`.
    /// Lengths must match. Returns embedding tensor of same shape as inputsEmbeds.
    private func spliceAtToken(
        tokens: MLXArray,
        inputsEmbeds: MLXArray,
        replacement: MLXArray,
        tokenId: Int
    ) -> MLXArray {
        // tokens: (B, T) or (T,); inputsEmbeds: (B, T, D); replacement: (N, D)
        let mask = MLX.equal(tokens, MLXArray(tokenId))
        // Squeeze batch dim to (T,), find positions
        let flatMask = mask.reshaped([-1])
        let positions = flatMask.asArray(Int.self)
        // Build a boolean mask broadcastable over D
        let D = inputsEmbeds.dim(-1)
        var maskExpanded = mask.expandedDimensions(axis: -1)
        maskExpanded = MLX.broadcast(maskExpanded, to: inputsEmbeds.shape)

        // Count placeholder positions; assemble a scattered tensor by iterating.
        let nReplace = positions.reduce(0, +)
        if nReplace == 0 { return inputsEmbeds }
        precondition(nReplace == replacement.dim(0),
                     "Multimodal placeholder count (\(nReplace)) does not match replacement embeds (\(replacement.dim(0)))")

        // Build replacement-broadcast tensor: same shape as inputsEmbeds with
        // replacement[i] at the i-th placeholder slot, zeros elsewhere.
        let replaceBuffer = MLXArray.zeros(inputsEmbeds.shape, dtype: inputsEmbeds.dtype)
        var replIdx = 0
        let totalSlots = positions.count
        let B = inputsEmbeds.dim(0)
        precondition(B == 1, "spliceAtToken currently supports batch=1 only")
        for slot in 0 ..< totalSlots {
            if positions[slot] != 0 {
                let row = replacement[replIdx ..< (replIdx + 1)] // (1, D)
                replaceBuffer[0, slot, 0..<D] = row.reshaped([D])
                replIdx += 1
            }
        }
        return MLX.where(maskExpanded, replaceBuffer.asType(inputsEmbeds.dtype), inputsEmbeds)
    }

    // MARK: - Sanitize

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // 1. Route all keys: LLM keys go through NemotronHModel.sanitize via
        //    "language_model." prefix; vision/audio/projector go through their
        //    own remap helpers.
        var llmKeys = [String: MLXArray]()
        var visionKeys = [String: MLXArray]()
        var soundKeys = [String: MLXArray]()
        var mlp1Keys = [String: MLXArray]()
        var soundProjKeys = [String: MLXArray]()

        for (k, v) in weights {
            if k.hasPrefix("vision_model.radio_model.") {
                visionKeys[k] = v
            } else if k.hasPrefix("sound_encoder.") {
                soundKeys[k] = v
            } else if k.hasPrefix("mlp1.") {
                mlp1Keys[k] = v
            } else if k.hasPrefix("sound_projection.") {
                soundProjKeys[k] = v
            } else if k.hasPrefix("vision_model.input_conditioner.") {
                // Skip — preprocess applies CLIP norm.
                continue
            } else {
                // Treat as LLM weight — strip any leading "language_model." (rare)
                // and forward to NemotronHModel.sanitize via a fresh dict.
                llmKeys[k] = v
            }
        }

        // LLM sanitize (handles conv1d transpose, JANG expert remap, expert stacking).
        let llmSanitized = languageModel.sanitize(weights: llmKeys)
        // Multimodal remap.
        let visionRemapped = remapRadioWeights(visionKeys)
        let soundRemapped = remapParakeetWeights(soundKeys)
        let mlp1Remapped = remapMlp1Weights(mlp1Keys)
        let soundProjRemapped = remapSoundProjectionWeights(soundProjKeys)

        // Combine under @ModuleInfo single-segment prefixes:
        //   "language_model.*"   → NemotronHModel root
        //   "vision_model.*"     → NemotronHRADIOVisionModel root (RADIO ViT body)
        //   "mlp1.*"             → NemotronHVisionMLPProjector root
        //   "sound_encoder.*"    → NemotronHParakeetEncoder root
        //   "sound_projection.*" → NemotronHSoundProjector root
        // The remap helpers return unprefixed paths; we add the single
        // top-level segment here.
        var out = [String: MLXArray]()
        for (k, v) in llmSanitized { out["language_model.\(k)"] = v }
        for (k, v) in visionRemapped { out["vision_model.\(k)"] = v }
        for (k, v) in soundRemapped { out["sound_encoder.\(k)"] = v }
        for (k, v) in mlp1Remapped { out["mlp1.\(k)"] = v }
        for (k, v) in soundProjRemapped { out["sound_projection.\(k)"] = v }

        return out
    }
}

// MARK: - User input processor (UserInputProcessor)

public struct NemotronHOmniProcessorConfiguration: Codable, Sendable {
    public let processorClass: String?
    public let imageSize: Int
    public let minNumPatches: Int
    public let maxNumPatches: Int
    public let maxModelLen: Int
    public let patchSize: Int
    public let downsampleRatio: Float
    public let useThumbnail: Bool
    public let videoPruningRate: Float

    public init() {
        self.processorClass = nil
        self.imageSize = 512
        self.minNumPatches = 1024
        self.maxNumPatches = 13312
        self.maxModelLen = 16384
        self.patchSize = 16
        self.downsampleRatio = 0.5
        self.useThumbnail = true
        self.videoPruningRate = 0.7
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.processorClass = try c.decodeIfPresent(String.self, forKey: .processorClass)
        self.imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 512
        self.minNumPatches = try c.decodeIfPresent(Int.self, forKey: .minNumPatches) ?? 1024
        self.maxNumPatches = try c.decodeIfPresent(Int.self, forKey: .maxNumPatches) ?? 13312
        self.maxModelLen = try c.decodeIfPresent(Int.self, forKey: .maxModelLen) ?? 16384
        self.patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        self.downsampleRatio = try c.decodeIfPresent(Float.self, forKey: .downsampleRatio) ?? 0.5
        self.useThumbnail = try c.decodeIfPresent(Bool.self, forKey: .useThumbnail) ?? true
        self.videoPruningRate = try c.decodeIfPresent(Float.self, forKey: .videoPruningRate) ?? 0.7
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageSize = "image_size"
        case minNumPatches = "min_num_patches"
        case maxNumPatches = "max_num_patches"
        case maxModelLen = "max_model_len"
        case patchSize = "patch_size"
        case downsampleRatio = "downsample_ratio"
        case useThumbnail = "use_thumbnail"
        case videoPruningRate = "video_pruning_rate"
    }
}

public struct NemotronHOmniProcessor: UserInputProcessor {
    private let config: NemotronHOmniProcessorConfiguration
    private let tokenizer: any Tokenizer

    private static let imageContextTokenId = 18
    private static let soundContextTokenId = 27

    private struct PreparedAudioClip {
        let waveform: [Float]
        let preEncodedEmbedding: MLXArray?
    }

    public init(_ config: NemotronHOmniProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    /// Tile-preprocess images into (totalTiles, 3, H, W) MLX pixel values.
    public func preprocess(images: [CIImage]) throws -> (MLXArray, [Int]) {
        let (pixels, tokenCounts) = try nemotronOmniPreprocessImages(
            images,
            imageSize: config.imageSize,
            minNum: config.minNumPatches,
            maxNum: config.maxNumPatches,
            useThumbnail: config.useThumbnail,
            patchSize: config.patchSize,
            downsampleRatio: config.downsampleRatio,
            maxModelLen: config.maxModelLen)
        return (pixels, tokenCounts)
    }

    /// Decode + resample audio resources into 16 kHz mono Float32 PCM
    /// (Parakeet's required input rate per `config_omni.json`).
    public func preprocess(audios: [UserInput.Audio]) throws -> [[Float]] {
        try preprocessAudioClips(audios: audios).map(\.waveform)
    }

    /// Decode + resample audio resources while preserving caller-supplied
    /// Parakeet/sound-projection embeddings for low-latency live voice turns.
    private func preprocessAudioClips(audios: [UserInput.Audio]) throws -> [PreparedAudioClip] {
        var clips: [PreparedAudioClip] = []
        for a in audios {
            switch a {
            case .url(let url):
                clips.append(PreparedAudioClip(
                    waveform: try nemotronOmniLoadAudioFile(
                        url, targetSampleRate: 16_000),
                    preEncodedEmbedding: nil))
            case .samples(let pcm, let sr):
                if sr == 16_000 {
                    clips.append(PreparedAudioClip(waveform: pcm, preEncodedEmbedding: nil))
                } else {
                    clips.append(PreparedAudioClip(
                        waveform: linearResamplePCM(pcm, fromRate: sr, toRate: 16_000),
                        preEncodedEmbedding: nil))
                }
            case .array(let arr, let sr):
                let pcm = arr.reshaped([-1]).asType(.float32).asArray(Float.self)
                clips.append(PreparedAudioClip(
                    waveform: sr == 16_000
                        ? pcm
                        : linearResamplePCM(pcm, fromRate: sr, toRate: 16_000),
                    preEncodedEmbedding: nil))
            case .preEncoded(let pcm, let sr, let embedding):
                clips.append(PreparedAudioClip(
                    waveform: sr == 16_000
                        ? pcm
                        : linearResamplePCM(pcm, fromRate: sr, toRate: 16_000),
                    preEncodedEmbedding: embedding))
            }
        }
        return clips
    }

    /// Decode video resources to the (groups, T*3, 512, 512) channel-stack
    /// tensor that NemotronH RADIO's `video_embedder` consumes.
    public func preprocess(videos: [UserInput.Video]) async throws -> (MLXArray, Int) {
        let (pixelValues, groupsPerVideo) = try await preprocessVideoClips(videos: videos)
        return (pixelValues, groupsPerVideo.reduce(0, +))
    }

    private func preprocessVideoClips(videos: [UserInput.Video]) async throws -> (MLXArray, [Int]) {
        // Concatenate all video pixel-tensors into a single (totalGroups,
        // T*3, H, W) tensor and return the total post-pixel-shuffle token
        // count for placeholder budgeting (256 tokens per group × N groups).
        var groupTensors: [MLXArray] = []
        var groupsPerVideo: [Int] = []
        for v in videos {
            let url: URL
            switch v {
            case .url(let u): url = u
            case .avAsset, .frames:
                throw NSError(
                    domain: "NemotronHOmniProcessor", code: -20,
                    userInfo: [NSLocalizedDescriptionKey:
                        "video must be .url(URL); .avAsset / .frames not yet supported"])
            }
            let pixels = try await nemotronOmniPreprocessVideo(
                url: url,
                imageSize: config.imageSize,
                targetFrames: 32,
                videoTemporalPatchDim: 2)
            // pixels shape: (groups, T*3, H, W). Flatten group axis so we
            // can stack across multiple videos.
            groupTensors.append(pixels)
            groupsPerVideo.append(pixels.dim(0))
        }
        let pixelValues = groupTensors.count == 1
            ? groupTensors[0]
            : MLX.concatenated(groupTensors, axis: 0)
        return (pixelValues, groupsPerVideo)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // Build prompt with NVLM 1-D placeholders. After tile selection we
        // know N total tiles → expand 256 image tokens per tile (post pixel
        // shuffle 32×32 → 16×16). Audio takes a parallel placeholder
        // path with `<so_embedding>` tokens — one per Parakeet output
        // frame. Video uses the SAME `<image>` placeholder (per Python
        // model.py: `img_context_token_id` is reused for video frames;
        // the model distinguishes them only by which embedding tower
        // produced the values).
        var processedImage: LMInput.ProcessedImage?
        var processedVideo: LMInput.ProcessedVideo?
        var processedAudio: LMInput.ProcessedAudio?
        var totalImageTokens = 0
        var totalVideoTokens = 0
        var totalAudioTokens = 0
        let tokensPerTile = 256
        let chatMessages: [Chat.Message]?
        if case .chat(let messages) = input.prompt {
            chatMessages = messages
        } else {
            chatMessages = nil
        }
        var mediaByChatMessage: [Int: String] = [:]

        if !input.images.isEmpty {
            let ciImages = try input.images.map { try $0.asCIImage() }
            let (pixels, tokenCounts) = try preprocess(images: ciImages)
            processedImage = LMInput.ProcessedImage(
                pixels: pixels,
                frames: tokenCounts.map { _ in THW(1, pixels.dim(2), pixels.dim(3)) })
            totalImageTokens = tokenCounts.reduce(0, +)
            if let chatMessages {
                var imageIndex = 0
                for (messageIndex, message) in chatMessages.enumerated() where !message.images.isEmpty {
                    var messageTokens = 0
                    for _ in message.images where imageIndex < tokenCounts.count {
                        messageTokens += tokenCounts[imageIndex]
                        imageIndex += 1
                    }
                    if messageTokens > 0 {
                        mediaByChatMessage[messageIndex, default: ""] +=
                            "<img>" + String(repeating: "<image>", count: messageTokens) + "</img>\n"
                    }
                }
            }
        }

        if !input.videos.isEmpty {
            let (pixels, groupsPerVideo) = try await preprocessVideoClips(videos: input.videos)
            let groups = groupsPerVideo.reduce(0, +)
            // Each group emits 256 post-pixel-shuffle tokens before EVS.
            // The source runtime applies EVS, then builds the prompt from
            // the pruned embedding count. Match that contract here so the
            // placeholder positions and video embeddings stay one-to-one.
            let videoTokens = Self.videoTokenCountAfterEVS(
                groups: groups,
                tokensPerGroup: tokensPerTile,
                pruningRate: config.videoPruningRate)
            processedVideo = LMInput.ProcessedVideo(
                pixels: pixels,
                frames: [THW(groups, config.imageSize, config.imageSize)],
                embeddingTokenCount: videoTokens)
            totalVideoTokens = videoTokens
            if let chatMessages {
                var videoIndex = 0
                for (messageIndex, message) in chatMessages.enumerated() where !message.videos.isEmpty {
                    var messageTokens = 0
                    for _ in message.videos where videoIndex < groupsPerVideo.count {
                        messageTokens += Self.videoTokenCountAfterEVS(
                            groups: groupsPerVideo[videoIndex],
                            tokensPerGroup: tokensPerTile,
                            pruningRate: config.videoPruningRate)
                        videoIndex += 1
                    }
                    if messageTokens > 0 {
                        mediaByChatMessage[messageIndex, default: ""] +=
                            "<img>" + String(repeating: "<image>", count: messageTokens) + "</img>\n"
                    }
                }
            }
        }

        if !input.audios.isEmpty {
            // Concat all audio waveforms into one stream — multiple
            // audio inputs serialize into the prompt in order, with a
            // single contiguous run of `<so_embedding>` placeholders.
            // Mirrors Python jang_tools.nemotron_omni: audio embeds
            // are flat (frames, hidden) per turn; the model doesn't
            // care about per-clip boundaries beyond positional order.
            let clips = try preprocessAudioClips(audios: input.audios)
            let combined = clips.flatMap(\.waveform)
            let encodedEmbeddings = clips.compactMap(\.preEncodedEmbedding)
            let combinedPreEncodedEmbedding: MLXArray? =
                encodedEmbeddings.count == clips.count && !encodedEmbeddings.isEmpty
                ? (encodedEmbeddings.count == 1
                    ? encodedEmbeddings[0]
                    : MLX.concatenated(encodedEmbeddings, axis: 0))
                : nil
            let waveArray = MLXArray(combined).reshaped([1, combined.count])
            processedAudio = LMInput.ProcessedAudio(
                waveform: waveArray,
                sampleRate: 16_000,
                preEncodedEmbedding: combinedPreEncodedEmbedding)
            if let combinedPreEncodedEmbedding {
                totalAudioTokens = max(1, combinedPreEncodedEmbedding.dim(0))
            } else {
                // Audio token count = expected Parakeet output frames.
                // Mel STFT: nFrames ≈ 1 + (samples + 2*pad - nFFT)/hop
                // with pad=nFFT/2=256, nFFT=512, hop=160. Parakeet
                // subsamples by 8 -> audio_tokens ≈ nFrames / 8.
                let nFFT = 512, hop = 160, pad = nFFT / 2
                let melFrames = max(0, 1 + (combined.count + 2 * pad - nFFT) / hop)
                // Subsampling factor 8 with stride-2 conv stack (3 levels).
                // Each level: ceil(T_in / 2). For melFrames=101 -> 51 -> 26
                // -> 13. Compute exactly the same way to avoid placeholder
                // count drift between processor and encoder.
                var t = melFrames
                for _ in 0 ..< 3 { t = (t + 1) / 2 }
                totalAudioTokens = t
            }
            if let chatMessages,
               let messageIndex = chatMessages.firstIndex(where: { !$0.audios.isEmpty }),
               totalAudioTokens > 0
            {
                mediaByChatMessage[messageIndex, default: ""] +=
                    "<so_start>" + String(repeating: "<so_embedding>", count: totalAudioTokens) + "<so_end>\n"
            }
        }

        // Insert media placeholders into the user message before tokenization.
        // Source convention (bundled `processing.py`):
        //   "<img>" + N×"<image>" + "</img>\n"
        //   "<so_start>" + N×"<so_embedding>" + "<so_end>\n"
        let media = {
            var media = ""
            if totalImageTokens > 0 {
                media += "<img>"
                media += String(repeating: "<image>", count: totalImageTokens)
                media += "</img>\n"
            }
            if totalVideoTokens > 0 {
                // Video reuses `<image>` placeholders per Python convention
                // (model.py § comment on img_context_token_id reuse). The
                // model's prepare() runs the video tower and splices into
                // the same token positions; image+video in one prompt
                // serialize image-first then video-second by construction.
                media += "<img>"
                media += String(repeating: "<image>", count: totalVideoTokens)
                media += "</img>\n"
            }
            if totalAudioTokens > 0 {
                media += "<so_start>"
                media += String(repeating: "<so_embedding>", count: totalAudioTokens)
                media += "<so_end>\n"
            }
            return media
        }()

        // Build text-only message dictionaries, then inject the expanded
        // NVLM placeholder run once. Using Qwen2VLMessageGenerator here
        // would leave one-token image/video marker parts in earlier chat
        // messages and add the expanded run again, desynchronizing
        // placeholder count from the encoded media embeddings.
        var messages = Self.textOnlyMessages(from: input)
        if !mediaByChatMessage.isEmpty {
            for (index, messageMedia) in mediaByChatMessage where messages.indices.contains(index) {
                Self.prependMedia(messageMedia, toMessageAt: index, in: &messages)
            }
        } else if !media.isEmpty {
            Self.prependMedia(media, toLastUserIn: &messages)
        }

        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)
        if !media.isEmpty {
            // The shipped template's compact no-thinking tail is text-safe, but
            // the omni VLM path grounds images only with an explicitly closed,
            // newline-delimited thought block. Keep reasoning disabled; only
            // canonicalize the closed media prompt tail.
            promptTokens = Self.canonicalizeNoThinkingMediaTail(
                promptTokens,
                tokenizer: tokenizer)
        }
        let cachePrefixTokenCounts = Self.canonicalHistoryCacheBoundaries(
            tokenizer: tokenizer,
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext,
            promptTokens: promptTokens)
        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)

        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage,
            video: processedVideo,
            audio: processedAudio,
            mediaTokenIds: media.isEmpty
                ? nil
                : [Self.imageContextTokenId, Self.soundContextTokenId],
            cacheScopeSalt: cacheScopeSalt(from: input.additionalContext),
            cachePrefixTokenCounts: cachePrefixTokenCounts)
    }

    private static func canonicalHistoryCacheBoundaries(
        tokenizer: any MLXLMCommon.Tokenizer,
        messages: [Message],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?,
        promptTokens: [Int]
    ) -> [Int] {
        guard let controllable = tokenizer as? any GenerationPromptControllableTokenizer else {
            return []
        }
        guard let historyTokens = try? controllable.applyChatTemplate(
            messages: messages,
            tools: tools,
            additionalContext: additionalContext,
            addGenerationPrompt: false)
        else {
            return []
        }
        guard !historyTokens.isEmpty,
              historyTokens.count < promptTokens.count,
              promptTokens.prefix(historyTokens.count).elementsEqual(historyTokens)
        else {
            return []
        }
        return [historyTokens.count]
    }

    private static func textOnlyMessages(from input: UserInput) -> [Message] {
        switch input.prompt {
        case .text(let text):
            return [["role": "user", "content": text]]
        case .chat(let chat):
            return chat.map { defaultMessageDict(for: $0) }
        case .messages(let rawMessages):
            return rawMessages.map { raw in
                var textOnly = raw
                textOnly["content"] = contentText(from: raw["content"])
                return textOnly
            }
        }
    }

    private static func prependMedia(_ media: String, toLastUserIn messages: inout [Message]) {
        guard !messages.isEmpty else {
            messages = [["role": "user", "content": media]]
            return
        }
        for i in messages.indices.reversed() where (messages[i]["role"] as? String) == "user" {
            let text = contentText(from: messages[i]["content"])
            messages[i]["content"] = media + text
            return
        }
        let text = contentText(from: messages[0]["content"])
        messages[0]["content"] = media + text
    }

    private static func prependMedia(
        _ media: String,
        toMessageAt index: Int,
        in messages: inout [Message]
    ) {
        guard messages.indices.contains(index) else { return }
        let text = contentText(from: messages[index]["content"])
        messages[index]["content"] = media + text
    }

    private static func canonicalizeNoThinkingMediaTail(
        _ promptTokens: [Int],
        tokenizer: any MLXLMCommon.Tokenizer
    ) -> [Int] {
        let compactTail = "<|im_start|>assistant\n<think></think>"
        let canonicalTail = "<|im_start|>assistant\n<think>\n</think>\n\n"
        let compactIds = tokenizer.encode(
            text: compactTail,
            addSpecialTokens: false)
        guard promptTokens.count >= compactIds.count,
              Array(promptTokens.suffix(compactIds.count)) == compactIds
        else {
            return promptTokens
        }
        let canonicalIds = tokenizer.encode(
            text: canonicalTail,
            addSpecialTokens: false)
        return Array(promptTokens.dropLast(compactIds.count)) + canonicalIds
    }

    public static func videoTokenCountAfterEVS(
        groups: Int,
        tokensPerGroup: Int = 256,
        pruningRate: Float = 0.7
    ) -> Int {
        guard groups > 0, tokensPerGroup > 0 else { return 0 }
        if groups < 2 { return groups * tokensPerGroup }
        let total = groups * tokensPerGroup
        let q = min(max(Double(pruningRate), 0.0), 1.0)
        let evsTokens = Int(Double(total) * (1.0 - q))
        return max(tokensPerGroup, evsTokens)
    }

    private static func contentText(from value: (any Sendable)?) -> String {
        if let text = value as? String {
            return text
        }
        if let parts = value as? [[String: any Sendable]] {
            return parts.compactMap { part in
                guard (part["type"] as? String) == "text" else { return nil }
                return part["text"] as? String
            }.joined(separator: "\n")
        }
        if let parts = value as? [[String: String]] {
            return parts.compactMap { part in
                guard part["type"] == "text" else { return nil }
                return part["text"]
            }.joined(separator: "\n")
        }
        if let parts = value as? [any Sendable] {
            return parts.compactMap { part in
                guard let dict = part as? [String: any Sendable],
                      (dict["type"] as? String) == "text"
                else { return nil }
                return dict["text"] as? String
            }.joined(separator: "\n")
        }
        return value.map { String(describing: $0) } ?? ""
    }
}
