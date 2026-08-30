// Apertus 1.5 — discrete-token multimodal (VQ codebook), text + vision.
//
// See `Apertus1p5VisionTokenizer.swift` for why this model needs no projector: the image is
// quantised into ids from a 131,072-entry codebook that lives INSIDE the token embedding table, so
// image tokens are ordinary tokens and the language model is unmodified `ApertusModel`.
//
// That shapes the wiring in one non-obvious way. The framework builds a `UserInputProcessor` from
// `(processorConfig, tokenizer)` alone — it never sees the model — but tokenizing an image REQUIRES
// the model's VQ encoder weights. So the split is:
//
//   * the PROCESSOR renders the prompt with a single `<image>` placeholder and hands over the
//     preprocessed pixels, exactly like any other VLM here;
//   * `prepare(_:cache:windowSize:)` on the MODEL runs the encoder, offsets the ids, and splices
//     them in place of the placeholder — then defers to the ordinary text path.
//
// The alternative (giving the processor a reference to the model) would invert the ownership the
// factory is built around, for no gain.

import CoreImage
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Apertus1p5Configuration: Codable, Sendable {
    public let textConfiguration: ApertusConfiguration
    public let visionTokenizerConfiguration: Apertus1p5VisionTokenizerConfiguration
    public let audioTokenizerConfiguration: Apertus1p5AudioTokenizerConfiguration?
    public let imageTokenId: Int
    public let imageTokenOffset: Int
    /// Present on Omni bundles. Absent on a vision-only conversion, in which case audio input is
    /// refused rather than silently ignored.
    public let audioTokenId: Int?
    public let audioTokenOffset: Int?
    /// `<|audio_start|>` / `<|audio_end|>`. Fixed in the Apertus vocabulary; the reference carries
    /// them as constants too, so defaulting is safe and keeps older configs loadable.
    public var audioStartTokenId: Int { 131080 }
    public var audioEndTokenId: Int { 131081 }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionTokenizerConfiguration = "vision_tokenizer_config"
        case audioTokenizerConfiguration = "audio_tokenizer_config"
        case imageTokenId = "image_token_id"
        case imageTokenOffset = "image_token_offset"
        case audioTokenId = "audio_token_id"
        case audioTokenOffset = "audio_token_offset"
    }
}

// MARK: - Model

public class Apertus1p5: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "language_model") public var languageModel: ApertusModel
    @ModuleInfo(key: "vision_tokenizer") public var visionTokenizer: Apertus1p5VisionTokenizer
    @ModuleInfo(key: "audio_tokenizer") public var audioTokenizer: Apertus1p5AudioTokenizer?

    public let config: Apertus1p5Configuration

    public var vocabularySize: Int { config.textConfiguration.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: Apertus1p5Configuration) {
        self.config = config
        self._languageModel.wrappedValue = ApertusModel(config.textConfiguration)
        self._visionTokenizer.wrappedValue = Apertus1p5VisionTokenizer(
            config.visionTokenizerConfiguration)
        self._audioTokenizer.wrappedValue = config.audioTokenizerConfiguration.map {
            Apertus1p5AudioTokenizer($0)
        }
    }

    /// Replace each `<image>` placeholder with the 256 codebook ids for the corresponding image,
    /// offset into the shared vocabulary, then hand the result to the ordinary text path.
    ///
    /// The placeholder is expanded 1 -> 256 here rather than in the processor because the encoder
    /// weights live on this object. A prompt with no image falls straight through.
    /// Replace each media placeholder with the codebook ids for the corresponding item, offset into
    /// the shared vocabulary, then hand the result to the ordinary text path.
    ///
    /// Vision and audio are the SAME mechanism with different offsets — that is the whole point of
    /// this architecture — so they share one splice.
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        var imageIds: [Int] = []
        if let image = input.image {
            imageIds = visionTokenizer.encode(image.pixels).asArray(Int32.self)
                .map { Int($0) + config.imageTokenOffset }
        }
        var audioIds: [Int] = []
        if let audio = input.audio {
            guard let audioTokenizer, let audioOffset = config.audioTokenOffset else {
                throw VLMError.processing(
                    "Apertus1p5: this bundle has no audio tokenizer (vision-only conversion), so "
                        + "audio input cannot be encoded")
            }
            // The encoder's stride ladder is defined in SAMPLES, so a rate mismatch silently
            // rescales time — 16 kHz audio fed to a 24 kHz encoder is heard 1.5x fast, and the
            // tokens are wrong without anything failing.
            guard audio.sampleRate == audioTokenizer.config.sampleRate else {
                throw VLMError.processing(
                    "Apertus1p5: audio is \(audio.sampleRate) Hz but the tokenizer expects "
                        + "\(audioTokenizer.config.sampleRate) Hz; resample before submitting")
            }
            audioIds = audioTokenizer.encode(audio.waveform).asArray(Int32.self)
                .map { Int($0) + audioOffset }
        }
        guard !imageIds.isEmpty || !audioIds.isEmpty else {
            return try languageModel.prepare(input, cache: cache, windowSize: windowSize)
        }

        // ONE placeholder per code, consumed in raster order. The processor already expanded the
        // template's single `<|image|>` into a framed grid — <|img_start|> "H*W" <|img_token_start|>
        // rows separated by <|img_end_of_row|> <|img_end|> — so the framing tokens flow through
        // untouched and only the placeholders are substituted. Splicing every code at the FIRST
        // placeholder (the earlier approach) destroys the row structure the model reads geometry from.
        var out: [Int] = []
        var imageCursor = 0, audioCursor = 0
        for token in input.text.tokens.asArray(Int32.self).map(Int.init) {
            if token == config.imageTokenId, imageCursor < imageIds.count {
                out.append(imageIds[imageCursor]); imageCursor += 1
            } else if token == config.audioTokenId, !audioIds.isEmpty, audioCursor == 0 {
                // The reference wraps audio as `<|audio_start|>` + one `<|audio|>` per code +
                // `<|audio_end|>`, with NO row structure and no text — so unlike the image path
                // this can be built entirely from ids, and the chat template's single placeholder
                // is replaced by the whole run. Doing it here also avoids having to predict the
                // code count in the processor before the encoder has run.
                out.append(config.audioStartTokenId)
                out.append(contentsOf: audioIds)
                out.append(config.audioEndTokenId)
                audioCursor = audioIds.count
            } else {
                out.append(token)
            }
        }
        let replacedImage = imageCursor, replacedAudio = audioCursor
        // A count mismatch means the encoder's grid and the prompt's grid disagree — the model
        // would read a shifted image. Fail rather than emit a subtly wrong prompt.
        if imageCursor != imageIds.count {
            throw VLMError.processing(
                "Apertus1p5: \(imageIds.count) image codes but only \(imageCursor) placeholders; "
                    + "the processor grid and the encoder output disagree")
        }
        if !audioIds.isEmpty, audioCursor != audioIds.count {
            throw VLMError.processing(
                "Apertus1p5: audio supplied but the prompt has no \(config.audioTokenId.map(String.init) ?? "audio") placeholder")
        }
        // Media arrived but the rendered prompt carries no placeholder for it — the template and
        // the config disagree. Failing loudly beats answering about something never inserted.
        if !imageIds.isEmpty, replacedImage == 0 {
            throw VLMError.processing(
                "Apertus1p5: image supplied but the prompt contains no image token "
                    + "(\(config.imageTokenId)); check the chat template")
        }
        if !audioIds.isEmpty, replacedAudio == 0 {
            throw VLMError.processing(
                "Apertus1p5: audio supplied but the prompt contains no audio token "
                    + "(\(config.audioTokenId.map(String.init) ?? "unset")); check the chat template")
        }
        let spliced = LMInput.Text(
            tokens: MLXArray(out.map { Int32($0) }).expandedDimensions(axis: 0))
        return try languageModel.prepare(
            LMInput(text: spliced, cacheScopeSalt: input.cacheScopeSalt),
            cache: cache, windowSize: windowSize)
    }

    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        languageModel(input, cache: cache, state: state)
    }

    /// LoRA targets the TEXT transformer only. Adapting the VQ encoder would change the codebook
    /// ids an image maps to, i.e. change the input alphabet rather than the model's use of it.
    public var loraLayers: [Module] { languageModel.loraLayers }

    /// Route the checkpoint into the text and vision halves, normalising TWO different layouts.
    ///
    /// The bf16 release and the MLX quantisations disagree about where the `language_model` segment
    /// sits, and both are in circulation:
    ///
    ///     bf16 (HF layout):  model.language_model.layers.0...   lm_head.weight
    ///     MLX quantisations: language_model.model.layers.0...   language_model.lm_head.weight
    ///
    /// Handling only one of them fails with `unhandledKeys(["language_model"])` on the other, which
    /// is how this was found. Both normalise to the inner model's own names before being re-prefixed.
    ///
    /// The normalisation itself is `Weights.stripLanguageModelPrefix`, not a local loop. The rule is
    /// identical, but the shared helper additionally resolves the case where a checkpoint carries
    /// BOTH spellings of a key: it binds a fixed winner and logs the rest, where writing as we go
    /// would let `Dictionary` iteration order decide — and that order is seeded per process, so the
    /// same bundle could bind a different tensor from run to run. A 70B Apertus 1.5 quantisation
    /// carries 2,084 body tensors through this path; one silently varying is not a failure anything
    /// downstream would report.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var lm: [String: MLXArray] = [:]
        var vision: [String: MLXArray] = [:]
        var audio: [String: MLXArray] = [:]
        for (k, v) in weights {
            if let r = k.range(of: "vision_tokenizer.") {
                vision[String(k[r.upperBound...])] = v
            } else if let r = k.range(of: "audio_tokenizer.") {
                audio[String(k[r.upperBound...])] = v
            } else {
                lm[k] = v
            }
        }
        // Normalise to what `ApertusModel` expects: `model.<...>` and `lm_head.<...>`.
        lm = Weights.stripLanguageModelPrefix(lm)
        var out: [String: MLXArray] = [:]
        for (k, v) in languageModel.sanitize(weights: lm) { out["language_model.\(k)"] = v }
        for (k, v) in visionTokenizer.sanitize(weights: vision) { out["vision_tokenizer.\(k)"] = v }
        if let audioTokenizer {
            for (k, v) in audioTokenizer.sanitize(weights: audio) { out["audio_tokenizer.\(k)"] = v }
        }
        return out
    }
}

// MARK: - Processor

public struct Apertus1p5ProcessorConfiguration: Codable, Sendable {
    private let _minPixels: Int?
    private let _maxPixels: Int?
    private let _spatialFactor: Int?
    /// Resolution is DYNAMIC, not a fixed 256x256: the area is clamped to this range and both sides
    /// rounded to a multiple of `spatialFactor`, so a 640x480 photo becomes a 30x40 grid (1200
    /// tokens) rather than 16x16 (256). Fixing it at 256 throws away most of the image.
    public var minPixels: Int { _minPixels ?? (256 * 256) }
    public var maxPixels: Int { _maxPixels ?? (1400 * 1400) }
    /// One `spatialFactor x spatialFactor` patch becomes one discrete code — the encoder's total
    /// downsampling, 16.
    public var spatialFactor: Int { _spatialFactor ?? 16 }

    enum CodingKeys: String, CodingKey {
        case _minPixels = "min_pixels"
        case _maxPixels = "max_pixels"
        case _spatialFactor = "spatial_factor"
    }
}

public struct Apertus1p5Processor: UserInputProcessor {
    private let config: Apertus1p5ProcessorConfiguration
    private let tokenizer: any Tokenizer
    /// WavTokenizer's rate. Fixed rather than read from the processor config, which does not
    /// carry it; the model-side guard catches any disagreement.
    private let audioSampleRate = 24000
    static let imageToken = "<|image|>"
    /// The audio placeholder the chat template emits, replaced in `Apertus1p5.prepare` by
    /// `<|audio_start|>` + one code per frame + `<|audio_end|>`. Named here so it can be DECLARED as
    /// a media placeholder — see the `mediaTokenIds` note on the returned `LMInput`.
    static let audioToken = "<|audio|>"
    static let boiToken = "<|img_start|>"
    static let eoiToken = "<|img_end|>"
    static let wrapperToken = "<|img_token_start|>"
    static let eolToken = "<|img_end_of_row|>"

    /// The placeholder ids this processor leaves in the prompt, declared so a cache hit can resume.
    ///
    /// Undeclared, `LMInput.cacheHitSuffixContainsMediaPlaceholder` takes its
    /// `guard let mediaTokenIds else { return true }` branch and rolls back on ANY non-empty suffix,
    /// and `canCaptureHybridStripBoundary` returns false for want of the same value — so every turn
    /// carrying an image or audio re-prefills the whole prompt, vision tower included. Apertus is the
    /// worst case for that: one image expands to a `<|image|>` per grid cell, so the span that cannot
    /// be resumed past is long.
    ///
    /// These are the PRE-SPLICE ids, and that is the load-bearing part. `Apertus1p5.prepare`
    /// substitutes each `<|image|>` for a discrete visual code (`imageTokenOffset + code`, in
    /// 131272..<262344 for the 8B bundle) and expands `<|audio|>` into a code run, so the sequence the
    /// MODEL sees contains no placeholders at all. It would be natural to declare those code ranges
    /// instead, and it would be wrong: the cache compares against `promptTokenIds`, which is the
    /// PROCESSOR's sequence, because `requiresPostPrepareCacheKey` is `video?.embeddingTokenCount
    /// != nil` and this family has no video path. Declaring the codes would name ids that never
    /// appear in the compared sequence — no rollback would ever fire, which is the unsafe direction.
    static func mediaTokenIds(_ tokenizer: any Tokenizer) -> [Int]? {
        MediaTokenIds.resolve(tokenizer: tokenizer, tokens: [imageToken, audioToken])
    }

    public init(_ config: Apertus1p5ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    /// Aspect-preserving resize with the area clamped to [minPixels, maxPixels] and both sides
    /// rounded HALF-UP to a multiple of `factor`. Ported verbatim from the reference
    /// `smart_resize`, including its `int()` truncations — the grid it produces determines how many
    /// image tokens the model receives, so any deviation changes the prompt the model was trained on.
    static func smartResize(height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int)
        -> (height: Int, width: Int)
    {
        let targetArea = Double(max(min(maxPixels, height * width), minPixels))
        let aspect = Double(width) / Double(height)
        var h = Int((targetArea / aspect).squareRoot())
        var w = Int(Double(h) * aspect)
        h = ((h + factor / 2) / factor) * factor
        w = ((w + factor / 2) / factor) * factor
        return (max(h, factor), max(w, factor))
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // Build the user message as STRUCTURED PARTS rather than via a MessageGenerator. The
        // template renders `<|image|>` / `<|audio|>` from `part.type`, and the stock generators
        // (Qwen2VL is vision-only; Default passes content through as a plain string) never emit an
        // audio part — so audio silently never reached the prompt.
        var parts: [[String: any Sendable]] = []
        parts += Array(repeating: ["type": "image"], count: input.images.count)
        parts += Array(repeating: ["type": "audio"], count: input.audios.count)
        parts.append(["type": "text", "text": Self.promptText(from: input)])
        let messages: [[String: any Sendable]] = [["role": "user", "content": parts]]
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)

        var processedAudio: LMInput.ProcessedAudio? = nil
        if !input.audios.isEmpty {
            var pcm: [Float] = []
            for audio in input.audios { pcm += try Self.waveform(audio, targetRate: audioSampleRate) }
            // Peak-normalise to -3 dBFS, as the reference pipeline does before the codec. The
            // feature extractor deliberately does NOT normalise, so skipping this feeds the encoder
            // a different loudness than it was trained on.
            let peak = pcm.map { Swift.abs($0) }.max() ?? 0
            if peak > 1e-10 {
                let target = Float(pow(10.0, -3.0 / 20.0))
                pcm = pcm.map { $0 * (target / peak) }
            }
            processedAudio = .init(waveform: MLXArray(pcm), sampleRate: audioSampleRate)
        }

        guard !input.images.isEmpty else {
            return LMInput(
                text: .init(tokens: MLXArray(promptTokens).expandedDimensions(axis: 0)),
                audio: processedAudio,
                mediaTokenIds: Self.mediaTokenIds(tokenizer),
                cacheScopeSalt: cacheScopeSalt(from: input.additionalContext))
        }

        var frames: [MLXArray] = []
        var expansion: [Int] = []
        for image in input.images {
            let ci = try image.asCIImage().toSRGB()
            let (h, w) = Self.smartResize(
                height: Int(ci.extent.height), width: Int(ci.extent.width),
                factor: config.spatialFactor, minPixels: config.minPixels, maxPixels: config.maxPixels)
            let resized = ci.resampled(to: CGSize(width: w, height: h), method: .bicubic)
            // `asMLXArray` renders CIFormat.RGBAf — float32 ALREADY in [0,1] — and returns NCHW.
            // The bundle asks for rescale 1/255 then mean/std 0.5, which on [0,1] data is x*2-1.
            let chw = resized.asMLXArray().asType(.float32)      // [1, 3, h, w]
            frames.append(chw.transposed(0, 2, 3, 1) * 2 - 1)     // [1, h, w, 3] in [-1, 1]

            // THE FRAMED SEQUENCE, ported from the reference `replace_image_token`:
            //   <|img_start|> "{H}*{W}" <|img_token_start|> row ⏎ row ⏎ … <|img_end|>
            // where each row is `<|image|>` repeated W times and rows are JOINED by
            // <|img_end_of_row|> (H-1 separators, not H). The grid is written as LITERAL TEXT and
            // tokenised normally — that is how the model is told the geometry, and it is the piece
            // that cannot be guessed from the token names alone.
            let (gh, gw) = (h / config.spatialFactor, w / config.spatialFactor)
            let row = Array(repeating: Self.imageToken, count: gw).joined()
            let rows = Array(repeating: row, count: gh).joined(separator: Self.eolToken)
            let framed = "\(Self.boiToken)\(gh)*\(gw)\(Self.wrapperToken)\(rows)\(Self.eoiToken)"
            expansion += tokenizer.encode(text: framed, addSpecialTokens: false)
        }

        // Replace the single `<|image|>` the chat template emitted with the full framed run.
        guard let placeholder = tokenizer.convertTokenToId(Self.imageToken),
            let at = promptTokens.firstIndex(of: placeholder)
        else {
            throw VLMError.processing(
                "Apertus1p5: image supplied but the rendered prompt has no \(Self.imageToken)")
        }
        promptTokens.replaceSubrange(at ... at, with: expansion)

        let tokensArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: tokensArray, mask: ones(like: tokensArray)),
            image: .init(pixels: concatenated(frames, axis: 0)), audio: processedAudio,
            mediaTokenIds: Self.mediaTokenIds(tokenizer),
            cacheScopeSalt: cacheScopeSalt(from: input.additionalContext))
    }

    /// The user's text, however the caller supplied it.
    private static func promptText(from input: UserInput) -> String {
        switch input.prompt {
        case .text(let text): return text
        case .messages(let messages): return messages.last?["content"] as? String ?? ""
        case .chat(let messages): return messages.last?.content ?? ""
        }
    }

    /// Decode any `UserInput.Audio` form to mono Float32 at `targetRate`, mirroring what the other
    /// audio-capable models in this library do.
    private static func waveform(_ audio: UserInput.Audio, targetRate: Int) throws -> [Float] {
        switch audio {
        case .url(let url):
            return try nemotronOmniLoadAudioFile(url, targetSampleRate: Double(targetRate))
        case .samples(let pcm, let rate):
            return rate == targetRate
                ? pcm : linearResamplePCM(pcm, fromRate: rate, toRate: targetRate)
        case .array(let array, let rate):
            let pcm = array.asType(.float32).asArray(Float.self)
            return rate == targetRate
                ? pcm : linearResamplePCM(pcm, fromRate: rate, toRate: targetRate)
        case .preEncoded(let pcm, let rate, _):
            // The pre-computed embedding is for encoder+projector models; this one needs the
            // waveform, because its "embedding" is a codebook id sequence.
            return rate == targetRate
                ? pcm : linearResamplePCM(pcm, fromRate: rate, toRate: targetRate)
        }
    }
}
