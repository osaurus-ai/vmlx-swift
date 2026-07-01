// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Time/Height/Width struct to represent information about input images.
public struct THW: Sendable {

    public let t: Int
    public let h: Int
    public let w: Int

    public init(_ t: Int, _ h: Int, _ w: Int) {
        self.t = t
        self.h = h
        self.w = w
    }

    public var values: (Int, Int, Int) {
        (t, h, w)
    }

    public var product: Int { t * h * w }
}

/// Representation of ``LanguageModel`` input.
///
/// This can contain text (tokens), prepared images (`MLXArray`), or other media as
/// needed. ``LMInput`` is produced by ``UserInputProcessor`` in response
/// to ``UserInput``.
///
/// The ``ModelContext`` holds the ``UserInputProcessor`` associated with a
/// ``LanguageModel``.
public struct LMInput {
    public let text: Text
    public let image: ProcessedImage?
    public let video: ProcessedVideo?
    public let audio: ProcessedAudio?
    public let mediaTokenIds: [Int]?

    /// Optional request-scope cache-key salt independent of media bytes.
    ///
    /// Set by VLM/LM processors when the request carries a runtime flag
    /// that changes prompt rendering or model behavior in a way the
    /// token list alone cannot distinguish — most importantly the
    /// reasoning on/off split for thinking-capable bundles. The
    /// canonical encoding today is `"reasoning=on"` / `"reasoning=off"`,
    /// but the field is intentionally `String?` so future flags can be
    /// composed without changing the type.
    ///
    /// Combined with the media-bytes fingerprint by
    /// `computeCacheSalt(for:)` to form the final per-request cache
    /// salt that the coordinator mixes into its block hash.
    public let cacheScopeSalt: String?
    /// Request-scope tool schemas used by streaming tool-call parsers.
    ///
    /// Chat templates consume ``UserInput/tools`` during prompt rendering, but
    /// the decode loop runs later from prepared ``LMInput``. Keeping the same
    /// schemas here lets parsers validate inline fallback tool attempts against
    /// the actual request contract instead of accepting a tool-shaped object by
    /// name alone.
    public let toolSchemas: [ToolSpec]?

    /// Additional prompt-prefix lengths that are safe to store in the cache.
    ///
    /// Some chat templates append assistant generation-control tokens to the
    /// active turn, but omit those tokens when the same assistant message is
    /// rendered as history on the next request. The full prompt KV state is
    /// then correct for decoding turn 1, but unsafe to key under turn 2's
    /// shorter historical prefix. Processors can record the canonical history
    /// boundary here so cache stores persist a real prefix the next request
    /// actually contains.
    public let cachePrefixTokenCounts: [Int]

    /// Representation of tokenized input text.
    public struct Text {

        /// input token array
        public let tokens: MLXArray

        /// optional mask array
        public let mask: MLXArray?

        /// CPU-side token IDs used by parser/template helpers that need to
        /// inspect prompt text without forcing a GPU readback.
        public let tokenIds: [Int]?

        public init(tokens: MLXArray, mask: MLXArray? = nil, tokenIds: [Int]? = nil) {
            self.tokens = tokens
            self.mask = mask
            self.tokenIds = tokenIds
        }

        public subscript(
            indices: MLXArrayIndex..., stream stream: StreamOrDevice = .default
        ) -> Text {
            Text(tokens: tokens[indices, stream: stream], mask: mask?[indices, stream: stream])
        }

        public subscript(
            text indices: MLXArrayIndex..., stream stream: StreamOrDevice = .default
        ) -> Text {
            Text(tokens: tokens[indices, stream: stream], mask: mask)
        }
    }

    /// Representation of prepared input image(s).
    public struct ProcessedImage {

        /// Concatenated pixels from one or more images
        public let pixels: MLXArray
        /// Time, height, and width of the images
        public let frames: [THW]?

        public init(
            pixels: MLXArray, frames: [THW]? = nil
        ) {
            self.pixels = pixels
            self.frames = frames
        }
    }

    /// Representation of prepared input video(s).
    /// For now, this is virtually identical to ProcessedImage.
    public struct ProcessedVideo {

        public let pixels: MLXArray
        public let frames: [THW]?
        public let embeddingTokenCount: Int?

        public init(
            pixels: MLXArray, frames: [THW]? = nil, embeddingTokenCount: Int? = nil
        ) {
            self.pixels = pixels
            self.frames = frames
            self.embeddingTokenCount = embeddingTokenCount
        }
    }

    /// Representation of prepared input audio for speech-aware models
    /// (Nemotron-3-Nano-Omni Parakeet path, future ASR-conditioned VLMs).
    ///
    /// ``waveform`` is a Float32 mono PCM tensor at ``sampleRate`` Hz —
    /// the model encodes it via its own mel STFT + audio encoder during
    /// `prepare(_:cache:windowSize:)`. The tensor format is intentionally
    /// minimal so different audio-encoder front-ends (Parakeet,
    /// Whisper-style log-mel, raw waveform encoders) can each consume
    /// the same `LMInput.audio` field.
    public struct ProcessedAudio {

        /// Mono Float32 PCM samples. Shape `[1, samples]` or `[samples]`.
        public let waveform: MLXArray

        /// Sampling rate of `waveform` in Hz. Models may resample
        /// internally if their encoder requires a different rate.
        public let sampleRate: Int

        /// Optional pre-encoded embedding `[frames, hidden]`. When
        /// supplied the model SHOULD skip its mel/encoder pipeline and
        /// splice these embeds directly. Used for the
        /// `extractAudioEmbeds` → manual splice workflow that
        /// non-omni-aware code can fall back to without paying the
        /// re-encode cost on every turn.
        public let preEncodedEmbedding: MLXArray?

        public init(
            waveform: MLXArray, sampleRate: Int = 16_000,
            preEncodedEmbedding: MLXArray? = nil
        ) {
            self.waveform = waveform
            self.sampleRate = sampleRate
            self.preEncodedEmbedding = preEncodedEmbedding
        }
    }

    public init(
        tokens: MLXArray,
        mask: MLXArray? = nil,
        tokenIds: [Int]? = nil,
        cacheScopeSalt: String? = nil,
        cachePrefixTokenCounts: [Int] = [],
        toolSchemas: [ToolSpec]? = nil
    ) {
        self.init(
            text: .init(tokens: tokens, mask: mask, tokenIds: tokenIds),
            cacheScopeSalt: cacheScopeSalt,
            cachePrefixTokenCounts: cachePrefixTokenCounts,
            toolSchemas: toolSchemas)
    }

    public init(
        text: LMInput.Text, image: LMInput.ProcessedImage? = nil,
        video: LMInput.ProcessedVideo? = nil,
        audio: LMInput.ProcessedAudio? = nil,
        mediaTokenIds: [Int]? = nil,
        cacheScopeSalt: String? = nil,
        cachePrefixTokenCounts: [Int] = [],
        toolSchemas: [ToolSpec]? = nil
    ) {
        self.text = text
        self.image = image
        self.video = video
        self.audio = audio
        self.mediaTokenIds = mediaTokenIds
        self.cacheScopeSalt = cacheScopeSalt
        self.cachePrefixTokenCounts = cachePrefixTokenCounts
        self.toolSchemas = toolSchemas
    }

    public func withToolSchemas(_ schemas: [ToolSpec]?) -> LMInput {
        LMInput(
            text: text,
            image: image,
            video: video,
            audio: audio,
            mediaTokenIds: mediaTokenIds,
            cacheScopeSalt: cacheScopeSalt,
            cachePrefixTokenCounts: cachePrefixTokenCounts,
            toolSchemas: schemas)
    }
}

public extension LMInput {
    /// True when this prompt carries model-side media embeddings.
    ///
    /// Cache restore paths use this to distinguish ordinary text prefixes
    /// from prompts whose placeholder-token span is backed by image, video,
    /// or audio tensors. Partial cache hits that split that span must fall
    /// back to full prefill.
    var hasMediaContent: Bool {
        image != nil || video != nil || audio != nil
    }

    /// True when a cache hit boundary would leave model-side media
    /// placeholder tokens in the suffix still being prefetched.
    ///
    /// `nil` ``mediaTokenIds`` means the processor did not declare its
    /// placeholder IDs, so media prompts stay on the conservative rollback
    /// path. Models that do declare IDs can safely resume after the
    /// placeholder span, while still rolling back if the remaining suffix
    /// includes any media token.
    func cacheHitSuffixContainsMediaPlaceholder(_ remainingTokenIds: [Int]) -> Bool {
        guard hasMediaContent, !remainingTokenIds.isEmpty else { return false }
        guard let mediaTokenIds else { return true }
        guard !mediaTokenIds.isEmpty else { return false }
        let mediaTokenSet = Set(mediaTokenIds)
        return remainingTokenIds.contains { mediaTokenSet.contains($0) }
    }

    /// True when the model may only know the cache key after
    /// `prepare(_:cache:windowSize:)` runs.
    ///
    /// Nemotron Omni video EVS is the current concrete case: the processor
    /// renders the full pre-EVS video placeholder run, the model computes
    /// video embeddings, then prunes placeholder positions together with
    /// `inputs_embeds` and returns `LMOutput.effectivePromptTokens`. A cache
    /// entry restored under the pre-pruned token stream would be unsafe
    /// because the live KV offset and token sequence describe the post-EVS
    /// prompt.
    var requiresPostPrepareCacheKey: Bool {
        video?.embeddingTokenCount != nil
    }
}

/// ``LanguageModel`` step output. This is consumed internally
/// by the ``TokenIterator``.
public struct LMOutput {

    /// logits (one hot vector of probabilities for tokens)
    public let logits: MLXArray

    /// optional ``State`` to carry forward into the next step
    public let state: State?

    /// Optional effective prompt token IDs after model-side prompt pruning.
    ///
    /// Some multimodal models splice media embeddings into a full prompt and
    /// then prune placeholder positions together with `inputs_embeds` before
    /// language-model prefill. When present, cache storage should key the
    /// resulting KV state by these effective tokens instead of the pre-pruned
    /// template token stream.
    public let effectivePromptTokens: [Int]?

    public struct State {
        public let crossAttentionStates: MLXArray?

        public init(crossAttentionStates: MLXArray? = nil) {
            self.crossAttentionStates = crossAttentionStates
        }
    }

    public init(
        logits: MLXArray,
        state: LMOutput.State? = nil,
        effectivePromptTokens: [Int]? = nil
    ) {
        self.logits = logits
        self.state = state
        self.effectivePromptTokens = effectivePromptTokens
    }
}

/// The result of the call to ``LanguageModel/prepare(_:cache:windowSize:)``
public enum PrepareResult {
    /// tokens to process by the ``TokenIterator``
    case tokens(LMInput.Text)

    /// logits representing the next token
    case logits(LMOutput)
}

/// Marker protocol for models that support vision/image input.
///
/// Conforming to this protocol indicates the model can process images alongside text.
/// Use ``ModelContext/isVLM`` or ``ModelContainer/isVLM`` to check at runtime.
public protocol VisionLanguageModelProtocol: LanguageModel {}

/// Interface for all Language Models (e.g. LLM, VLM).
///
/// The language model is typically called by the ``TokenIterator`` and it:
///
/// - consumes the ``LMInput``
/// - calls ``prepare(_:cache:windowSize:)`` to initialize the KVCache and consume the prompt
/// - calls ``callAsFunction(_:cache:state:)-9kuvf`` for each token, producing an ``LMOutput``
/// - the ``TokenIterator`` accumulates this information into a ``GenerateResult``
public protocol LanguageModel: Module {

    /// Prepare the cache state and consume the ``LMInput``.
    ///
    /// This can return:
    /// - ``PrepareResult/tokens(_:)`` if the caller should evaluate the (remaining) tokens normally
    /// - ``PrepareResult/logits(_:)`` to produce the next token from the prompt
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult

    /// Primary entry point to produce a step (single token) from the model
    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput

    /// Models may implement this simplified interface if they do not produce any ``LMOutput/State``
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray

    /// create a new array of ``KVCache`` -- automatic implementation if self
    /// implements ``KVCacheDimensionProvider``
    func newCache(parameters: GenerateParameters?) -> [KVCache]

    /// Optionally preprocess the weights and modify / remove values as needed.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]

    /// Optionally preprocess the weights with access to safetensor metadata.
    ///
    /// The default implementation forwards to ``sanitize(weights:)``.
    /// Models can override this to inspect metadata (e.g. check `metadata["format"] == "mlx"`)
    /// and skip or customize sanitization accordingly.
    func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray]

    /// The architecture's authoritative RMSNorm "(1 + weight)" convention marker (e.g.
    /// `"mlx_plus_one"`), consulted when neither bundle metadata nor `config.json` declares one.
    /// `nil` (the default) means the architecture declares no such convention. Architectures that
    /// store RMSNorm weights as the deviation-from-1 should override this; see
    /// ``NormConventionResolver`` and `Qwen35` for the template.
    var declaredNormConvention: String? { get }
}

extension LanguageModel {
    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let logits = callAsFunction(input.tokens, cache: cache)
        return .init(logits: logits)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError("callAsFunction(inputs:cache:) not implemented for \(Self.self)")
    }
}

extension LanguageModel {
    /// Default: the architecture declares no `(1 + weight)` convention. Override per architecture.
    public var declaredNormConvention: String? { nil }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        sanitize(weights: weights)
    }
}

/// Optional protocol that can be implemented by ``LanguageModel`` and will
/// provide an automatic implementation of ``LanguageModel/newCache(parameters:)``
public protocol KVCacheDimensionProvider {
    var kvHeads: [Int] { get }
}

extension LanguageModel where Self: KVCacheDimensionProvider {
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        // Create one cache per layer (kvHeads.count = number of layers)
        // The number of heads per layer (kvHeads[i]) is not used for cache creation
        let numLayers = kvHeads.count

        // Follow Python logic: use RotatingKVCache if maxKVSize is provided
        if let maxKVSize = parameters?.maxKVSize {
            return (0 ..< numLayers).map { _ in
                RotatingKVCache(maxSize: maxKVSize, keep: 4)
            }
        } else {
            return (0 ..< numLayers).map { _ in KVCacheSimple() }
        }
    }
}
