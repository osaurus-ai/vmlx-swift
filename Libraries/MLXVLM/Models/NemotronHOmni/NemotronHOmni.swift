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

    enum WrapperKeys: String, CodingKey { case llmConfig = "llm_config" }

    public init(from decoder: Decoder) throws {
        // Two bundle layouts carry the nemotron_h LLM config:
        //  - OsaurusAI/JANGTQ bundles: config.json IS the LLM config at top level (multimodal dims in
        //    a sibling config_omni.json), so decode the whole decoder as NemotronHConfiguration.
        //  - mlx-community conversion: the full omni config.json NESTS the LLM under `llm_config`
        //    (alongside vision_config/sound_config), so `vocab_size` isn't at top level — decode from
        //    the `llm_config` sub-container instead. Support both.
        if let wrapper = try? decoder.container(keyedBy: WrapperKeys.self), wrapper.contains(.llmConfig) {
            self.llmConfig = try wrapper.decode(NemotronHConfiguration.self, forKey: .llmConfig)
        } else {
            self.llmConfig = try NemotronHConfiguration(from: decoder)
        }

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
        var imageEmbedCount = 0
        var videoRetention: (keepIndices: [Int], totalTokens: Int)? = nil
        if let pixelValues = input.image?.pixels {
            visualEmbeds = extractImageEmbeds(pixelValues: pixelValues)
            imageEmbedCount = visualEmbeds?.dim(0) ?? 0
        }
        if let video = input.video {
            let videoEmbedsWithRetention = extractVideoEmbedsWithRetention(
                pixelValues: video.pixels,
                targetTokenCount: video.embeddingTokenCount)
            let videoEmbeds = videoEmbedsWithRetention.fullEmbeds
            videoRetention = (
                keepIndices: videoEmbedsWithRetention.keepIndices,
                totalTokens: videoEmbedsWithRetention.totalTokens)
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
        var effectivePromptTokens: [Int]? = nil
        if let videoRetention {
            let pruned = try pruneVideoPlaceholdersAfterEVS(
                tokens: input.text.tokens,
                inputsEmbeds: spliced,
                imageTokenCount: imageEmbedCount,
                videoTotalTokenCount: videoRetention.totalTokens,
                videoKeepIndices: videoRetention.keepIndices)
            spliced = pruned.inputsEmbeds
            effectivePromptTokens = pruned.tokenIds
        }

        // The text-only path above chunks Nemotron-H prefill because every
        // Mamba layer builds sequence-quadratic intermediate state. The
        // multimodal path previously bypassed that protection and forwarded
        // the entire 4K+ image/video/audio embedding sequence in one call.
        // On the real JANGTQ4 Omni bundle a 512px image then reached a 76 GiB
        // physical-footprint high-water mark despite an ~19 GiB bundle.
        // Materialize the media tower once, then feed the language stack in
        // the same bounded chunks used for text prompts.
        MLX.eval(spliced)
        Memory.clearCache()
        let prefillStepSize = max(1, windowSize ?? 512)
        let sequenceLength = spliced.dim(1)
        var offset = 0
        while sequenceLength - offset > prefillStepSize {
            let end = offset + prefillStepSize
            let chunk = spliced[0..., offset..<end, 0...]
            _ = languageModel.callAsFunction(
                inputsEmbeds: chunk, cache: convertedCache)
            MLX.eval(convertedCache)
            offset = end
            Memory.clearCache()
        }
        let logits = languageModel.callAsFunction(
            inputsEmbeds: spliced[0..., offset..<sequenceLength, 0...],
            cache: convertedCache)
        return .logits(LMOutput(
            logits: logits,
            effectivePromptTokens: effectivePromptTokens))
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

    /// Run RADIO's video embedder and optionally apply Efficient Video Sampling.
    ///
    /// The full source-equivalent generation path uses
    /// `extractVideoEmbedsWithRetention`: it splices the full pre-EVS video
    /// placeholder run, then prunes embeddings and token IDs together.
    /// This helper remains useful for direct embedding probes.
    public func extractVideoEmbeds(
        pixelValues: MLXArray,
        targetTokenCount: Int? = nil,
        applyEVS: Bool = true,
        pruningRate: Float = 0.7
    ) -> MLXArray {
        var feats = projectedVideoEmbedsByGroup(pixelValues: pixelValues)
        if applyEVS {
            if let targetTokenCount {
                feats = nemotronOmniApplyEVS(feats, targetTokenCount: targetTokenCount)
            } else {
                feats = nemotronOmniApplyEVS(feats, pruningRate: pruningRate)
            }
        }
        return feats.reshaped([feats.dim(0) * feats.dim(1), feats.dim(-1)])
    }

    private func projectedVideoEmbedsByGroup(pixelValues: MLXArray) -> MLXArray {
        var feats = radioModel(pixelValues, video: true)
        feats = feats[0..., config.visionNumClsTokens..., 0...]
        let nGroups = feats.dim(0)
        let patches = feats.dim(1)
        let hidden = feats.dim(2)
        let gridH = pixelValues.dim(2) / config.visionPatchSize
        let gridW = pixelValues.dim(3) / config.visionPatchSize
        precondition(gridH * gridW == patches,
                     "RADIO video patch grid mismatch; got P=\(patches), H=\(gridH), W=\(gridW)")
        feats = feats.reshaped([nGroups, gridH, gridW, hidden])
        feats = nemotronOmniPixelShuffle(feats, scaleFactor: config.downsampleRatio)
        let tokensPerGroup = feats.dim(1) * feats.dim(2)
        let cIn = feats.dim(3)
        feats = feats.reshaped([nGroups, tokensPerGroup, cIn])
        return visionMLP(feats)
    }

    private func extractVideoEmbedsWithRetention(
        pixelValues: MLXArray,
        targetTokenCount: Int? = nil,
        pruningRate: Float = 0.7
    ) -> (fullEmbeds: MLXArray, keepIndices: [Int], totalTokens: Int) {
        let feats = projectedVideoEmbedsByGroup(pixelValues: pixelValues)
        let nGroups = feats.dim(0)
        let tokensPerGroup = feats.dim(1)
        let hidden = feats.dim(2)
        let totalTokens = nGroups * tokensPerGroup
        let keepIndices: [Int]
        if let targetTokenCount {
            keepIndices = nemotronOmniEVSKeepIndices(
                feats,
                targetTokenCount: targetTokenCount)
        } else {
            let q = min(max(Double(pruningRate), 0.0), 1.0)
            let nKeep = max(tokensPerGroup, Int(Double(totalTokens) * (1.0 - q)))
            keepIndices = nemotronOmniEVSKeepIndices(
                feats,
                targetTokenCount: nKeep)
        }
        return (
            fullEmbeds: feats.reshaped([totalTokens, hidden]),
            keepIndices: keepIndices,
            totalTokens: totalTokens)
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

    /// Apply source-style EVS after full video placeholders have been spliced.
    ///
    /// Nemotron's Python path renders the full video prompt, replaces every
    /// video `<image>` placeholder with a video embedding, then prunes only the
    /// selected video placeholder positions from `inputs_embeds` and `input_ids`
    /// together. Frame labels, wrapper tokens, image placeholders, and audio
    /// placeholders remain in the effective prompt.
    private func pruneVideoPlaceholdersAfterEVS(
        tokens: MLXArray,
        inputsEmbeds: MLXArray,
        imageTokenCount: Int,
        videoTotalTokenCount: Int,
        videoKeepIndices: [Int]
    ) throws -> (inputsEmbeds: MLXArray, tokenIds: [Int]) {
        guard videoTotalTokenCount > 0 else {
            return (inputsEmbeds, tokens.reshaped([-1]).asArray(Int.self))
        }
        let tokenIds = tokens.reshaped([-1]).asArray(Int.self)
        let keepVideo = Set(videoKeepIndices)
        var retainedPositions: [Int32] = []
        var retainedTokenIds: [Int] = []
        var visualOrdinal = 0
        var videoOrdinal = 0

        for (position, tokenId) in tokenIds.enumerated() {
            var keep = true
            if tokenId == config.imageContextTokenId {
                if visualOrdinal < imageTokenCount {
                    keep = true
                } else if videoOrdinal < videoTotalTokenCount {
                    keep = keepVideo.contains(videoOrdinal)
                    videoOrdinal += 1
                }
                visualOrdinal += 1
            }
            if keep {
                retainedPositions.append(Int32(position))
                retainedTokenIds.append(tokenId)
            }
        }

        guard videoOrdinal == videoTotalTokenCount else {
            throw NSError(
                domain: "NemotronHOmni", code: -30,
                userInfo: [NSLocalizedDescriptionKey:
                    "video placeholder count \(videoOrdinal) does not match video embeddings \(videoTotalTokenCount)"])
        }

        let gather = MLXArray(retainedPositions)
        let pruned = inputsEmbeds[0].take(gather, axis: 0)
            .expandedDimensions(axis: 0)
        return (pruned, retainedTokenIds)
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
                // Treat as LLM weight — strip any leading "language_model." so NemotronHModel.sanitize
                // sees ROOT-level keys; the single "language_model." segment is re-added below. The
                // OsaurusAI bundles ship the LLM weights unprefixed (so this is a no-op there), but the
                // mlx-community conversion prefixes them "language_model.*" — without stripping here that
                // re-add produced a DOUBLE "language_model.language_model." prefix → unhandledKeys.
                let stripped = k.hasPrefix("language_model.")
                    ? String(k.dropFirst("language_model.".count)) : k
                llmKeys[stripped] = v
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

    /// Decode video resources to the (groups, T*3, H, W) channel-stack tensor
    /// that NemotronH RADIO's `video_embedder` consumes.
    public func preprocess(videos: [UserInput.Video]) async throws -> (MLXArray, Int, Int) {
        // Concatenate all video pixel-tensors into a single (totalGroups,
        // T*3, H, W) tensor and return the total post-pixel-shuffle token
        // count for placeholder budgeting.
        var groupTensors: [MLXArray] = []
        var totalGroups = 0
        var tokensPerGroup: Int?
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
            let perGroup = Self.videoTokensPerGroup(pixelValues: pixels)
            if let tokensPerGroup, tokensPerGroup != perGroup {
                throw NSError(
                    domain: "NemotronHOmniProcessor", code: -21,
                    userInfo: [NSLocalizedDescriptionKey:
                        "multiple video inputs with different post-shuffle token counts are not supported"])
            }
            tokensPerGroup = perGroup
            groupTensors.append(pixels)
            totalGroups += pixels.dim(0)
        }
        let pixelValues = groupTensors.count == 1
            ? groupTensors[0]
            : MLX.concatenated(groupTensors, axis: 0)
        return (pixelValues, totalGroups, tokensPerGroup ?? 0)
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
        let tokensPerTile = 256
        var totalImageTokens = 0
        var totalVideoTokens = 0
        var totalVideoGroups = 0
        var totalVideoTokensPerGroup = tokensPerTile
        var totalAudioTokens = 0

        if !input.images.isEmpty {
            let ciImages = try input.images.map { try $0.asCIImage() }
            let (pixels, tokenCounts) = try preprocess(images: ciImages)
            processedImage = LMInput.ProcessedImage(
                pixels: pixels,
                frames: tokenCounts.map { _ in THW(1, pixels.dim(2), pixels.dim(3)) })
            totalImageTokens = tokenCounts.reduce(0, +)
        }

        if !input.videos.isEmpty {
            let (pixels, groups, videoTokensPerGroup) = try await preprocess(videos: input.videos)
            // Source processor renders the full pre-EVS placeholder run; the
            // model applies EVS after splicing and prunes `inputs_embeds` and
            // `input_ids` together. Keep both counts: full tokens for prompt
            // rendering, retained tokens for the model-side EVS target.
            let fullVideoTokens = groups * videoTokensPerGroup
            let retainedVideoTokens = Self.videoTokenCountAfterEVS(
                groups: groups,
                tokensPerGroup: videoTokensPerGroup,
                pruningRate: config.videoPruningRate)
            processedVideo = LMInput.ProcessedVideo(
                pixels: pixels,
                frames: [THW(groups, pixels.dim(2), pixels.dim(3))],
                embeddingTokenCount: retainedVideoTokens)
            totalVideoTokens = fullVideoTokens
            totalVideoGroups = groups
            totalVideoTokensPerGroup = videoTokensPerGroup
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
        }

        // Insert media placeholders into the user message before tokenization.
        // Source convention (bundled `processing.py`):
        //   "<img>" + N×"<image>" + "</img>\n"
        //   "<so_start>" + N×"<so_embedding>" + "<so_end>\n"
        var media = ""
        if totalImageTokens > 0 {
            media += "<img>"
            media += String(repeating: "<image>", count: totalImageTokens)
            media += "</img>\n"
        }
        if totalVideoTokens > 0 {
            media += Self.videoPromptMedia(
                totalTokens: totalVideoTokens,
                groups: totalVideoGroups,
                tokensPerGroup: totalVideoTokensPerGroup,
                temporalPatchDim: 2)
        }
        if totalAudioTokens > 0 {
            media += "<so_start>"
            media += String(repeating: "<so_embedding>", count: totalAudioTokens)
            media += "<so_end>\n"
        }

        // Build text-only message dictionaries, then inject the expanded
        // NVLM placeholder run once. Using Qwen2VLMessageGenerator here
        // would leave one-token image/video marker parts in earlier chat
        // messages and add the expanded run again, desynchronizing
        // placeholder count from the encoded media embeddings.
        var messages = Self.textOnlyMessages(from: input)
        if !media.isEmpty {
            if let index = Self.mediaTargetMessageIndex(in: input) {
                Self.prependMedia(media, toMessageAt: index, in: &messages)
            } else {
                Self.prependMedia(media, toLastUserIn: &messages)
            }
        }

        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)
        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)

        return LMInput(
            text: .init(tokens: promptArray, mask: mask, tokenIds: promptTokens),
            image: processedImage,
            video: processedVideo,
            audio: processedAudio,
            mediaTokenIds: media.isEmpty
                ? nil
                : [Self.imageContextTokenId, Self.soundContextTokenId],
            cacheScopeSalt: cacheScopeSalt(from: input.additionalContext))
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

    static func addRequiredToolChoiceInstruction(
        to messages: inout [Message],
        tools _: [ToolSpec]?,
        additionalContext _: [String: any Sendable]?
    ) {
        // `tool_choice=required` is forwarded through `additionalContext` and
        // the tool schema. Do not synthesize an extra system/tail directive for
        // Nemotron Omni: if the model/template cannot satisfy the required
        // tool contract natively, the row must fail or remain partial instead
        // of being made coherent with prompt coercion.
        _ = messages
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
        guard messages.indices.contains(index) else {
            prependMedia(media, toLastUserIn: &messages)
            return
        }
        let text = contentText(from: messages[index]["content"])
        messages[index]["content"] = media + text
    }

    private static func mediaTargetMessageIndex(in input: UserInput) -> Int? {
        guard case .chat(let chat) = input.prompt else {
            return nil
        }
        return chat.indices.reversed().first { index in
            let message = chat[index]
            return !message.images.isEmpty
                || !message.videos.isEmpty
                || !message.audios.isEmpty
        }
    }

    private static func videoTokensPerGroup(pixelValues: MLXArray) -> Int {
        let patchSize = 16
        let downsampleFactor = 2
        let gridH = pixelValues.dim(2) / patchSize
        let gridW = pixelValues.dim(3) / patchSize
        return max(1, (gridH * gridW) / (downsampleFactor * downsampleFactor))
    }

    static func videoPromptMedia(
        totalTokens: Int,
        groups: Int,
        tokensPerGroup: Int = 256,
        temporalPatchDim: Int = 2
    ) -> String {
        guard totalTokens > 0 else { return "" }
        let groupCount = max(1, groups)
        let tokenCounts = videoPromptTokenCounts(
            totalTokens: totalTokens,
            groups: groupCount,
            tokensPerGroup: tokensPerGroup)
        var chunks: [String] = []
        for group in 0 ..< groupCount {
            let count = tokenCounts[group]
            guard count > 0 else { continue }
            chunks.append(
                videoFrameLabel(group: group, temporalPatchDim: temporalPatchDim)
                    + "<img>"
                    + String(repeating: "<image>", count: count)
                    + "</img>")
        }
        return chunks.joined(separator: "\n") + "\n"
    }

    private static func videoPromptTokenCounts(
        totalTokens: Int,
        groups: Int,
        tokensPerGroup: Int
    ) -> [Int] {
        var counts = [Int](repeating: 0, count: groups)
        guard totalTokens > 0, groups > 0 else { return counts }
        var remaining = totalTokens
        counts[0] = min(max(1, tokensPerGroup), remaining)
        remaining -= counts[0]
        guard groups > 1, remaining > 0 else { return counts }
        for group in 1 ..< groups {
            let groupsLeft = groups - group
            let count = (remaining + groupsLeft - 1) / groupsLeft
            counts[group] = count
            remaining -= count
        }
        return counts
    }

    private static func videoFrameLabel(group: Int, temporalPatchDim: Int) -> String {
        let framesPerGroup = max(1, temporalPatchDim)
        let labels = (0 ..< framesPerGroup).map { frameOffset in
            let prefix = frameOffset == 0 ? "Frame" : "frame"
            return "\(prefix) \(group * framesPerGroup + frameOffset + 1)"
        }
        return labels.joined(separator: " and ") + ": "
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
