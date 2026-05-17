// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import os

/// A `LogitSampler` is responsible for sampling `logits` produced by
/// a ``LanguageModel`` to produce a token.
///
/// See also: ``LogitProcessor``
public protocol LogitSampler {

    /// Given `logits` produce a new `MLXArray` with the token.
    func sample(logits: MLXArray) -> MLXArray
}

/// A `LogitProcessor` is an optional visitor of `logits`.
///
/// The ``LogitProcessor`` is called with the input (prompt) before generating tokens:
///
/// ```swift
/// processor?.prompt(input.text.tokens)
/// ```
///
/// Then for each token generated it has a chance to adjust the logits:
///
/// ```swift
/// logits = processor?.process(logits: logits) ?? logits
/// let y = sampler.sample(logits: logits)
/// processor?.didSample(token: y)
/// ```
///
/// See also: ``LogitSampler``
public protocol LogitProcessor {

    /// Called before token generation starts with the text tokens of the prompt
    mutating func prompt(_ prompt: MLXArray)

    /// Called to visit and possibly modify the logits
    func process(logits: MLXArray) -> MLXArray

    /// Called to provide the sampled token
    mutating func didSample(token: MLXArray)
}

/// Parameters for text generation, see ``TokenIterator``.
///
/// This produces:
///
/// - ``LogitSampler``
/// - ``LogitProcessor``
///
/// for the `TokenIterator`.

/// KV cache quantization/compression mode.
///
/// Controls how the KV cache is compressed during inference:
///
/// ```swift
/// // No compression (default, same as today)
/// var params = GenerateParameters()
///
/// // Affine quantization (existing path, unchanged)
/// var params = GenerateParameters(kvBits: 4, kvGroupSize: 64)
///
/// // TurboQuant compression (Hadamard + Lloyd-Max + QJL)
/// var params = GenerateParameters()
/// params.kvMode = .turboQuant(keyBits: 3, valueBits: 3)
/// ```
public enum KVQuantizationMode: Sendable, Equatable {
    /// No cache compression (float16, default)
    case none

    /// Affine quantization (existing QuantizedKVCache path)
    case affine(bits: Int, groupSize: Int = 64)

    /// TurboQuant compression: randomized Hadamard rotation + Lloyd-Max optimal
    /// codebook quantization + QJL residual correction for keys.
    /// Achieves 4.7-5.0x compression with zero generation speed overhead.
    ///
    /// - Parameters:
    ///   - keyBits: Total bits per key element (default 3). Split as (b-1) codebook + 1 QJL.
    ///   - valueBits: Total bits per value element (default 3). All bits go to codebook.
    case turboQuant(keyBits: Int = 3, valueBits: Int = 3)
}

public struct GenerateParameters: Sendable {

    /// Step size for processing the prompt
    public var prefillStepSize: Int

    /// Maximum tokens to generate
    public var maxTokens: Int?

    /// Maximum size of the key-value cache. Old entries (except the first 4 tokens) will be overwritten.
    /// When set, uses ``RotatingKVCache`` instead of ``KVCacheSimple``
    public var maxKVSize: Int?

    /// Number of bits to use for KV cache quantization. nil implies no cache quantization.
    public var kvBits: Int?

    /// Group size for KV cache quantization (default: 64)
    public var kvGroupSize: Int

    /// Step to begin using a quantized KV cache when kvBits is non-nil (default: 0)
    public var quantizedKVStart: Int

    /// KV cache quantization/compression mode.
    ///
    /// When set to a value other than `.none`, this takes precedence over `kvBits`/`kvGroupSize`.
    /// The legacy `kvBits`/`kvGroupSize` fields continue to work for backward compatibility.
    public var kvMode: KVQuantizationMode = .none

    public var enableCompiledDecode: Bool = false
    public var compiledMaxCacheLength: Int? = nil

    /// Runtime accelerator selection for generation.
    ///
    /// Defaults to `VMLINUX_ACCELERATOR` when present, otherwise `.metal`.
    /// `ane-coreml` is a fail-closed request: it is accepted only when the
    /// selected runtime surface has a validated Core ML island. Text decode
    /// currently has no such island, so it stays on MLX/Metal.
    public var accelerationMode: AccelerationMode = .metal

    /// Enable `compile()` tracing for BATCHED decode. Opt-in; default false.
    ///
    /// When true, the `BatchEngine` routes decode steps through `BatchCompile`
    /// which caches one compiled forward per batch-size bucket. Requests that
    /// carry an incompatible cache type (RotatingKVCache, MambaCache,
    /// CacheList, or — until Stage 2 ships — TurboQuantKVCache) transparently
    /// fall back to the existing uncompiled batched path.
    ///
    /// This is independent from ``enableCompiledDecode`` which gates compile
    /// on the single-sequence `TokenIterator` path. You can enable either,
    /// both, or neither.
    ///
    /// See the "Batch Engine Blockers" spec at
    /// `docs/superpowers/specs/2026-04-18-batch-engine-blockers-design.md`.
    public var enableCompiledBatchDecode: Bool = false

    /// Batch-size buckets for compiled batch decode. Each bucket owns one
    /// compiled trace and one set of `[B, L, maxLen, H_kv, D]` KV buffers.
    /// At decode time, requests pad up to the next bucket >= active-slot
    /// count; dead rows are suppressed via a liveness mask.
    ///
    /// Memory cost: the KV buffers for all active buckets are resident
    /// simultaneously. Per bucket of size `B` on a typical 32-layer /
    /// H_kv=8 / D=128 / maxLen=4096 model: ~536 MB × B. For the default
    /// `[1, 2, 4]` buckets that's ~3.75 GB of compile-side KV buffers.
    ///
    /// Raise to `[1, 2, 4, 8]` only after verifying memory headroom on the
    /// target hardware. Every extra bucket adds compile time on first-hit
    /// and keeps its buffer allocated until `BatchCompile.invalidate()`
    /// (e.g., on `container.unload()`).
    ///
    /// Only consulted when ``enableCompiledBatchDecode`` is `true`. Must be
    /// sorted ascending and non-empty; `BatchCompile` validates at use.
    public var compiledBatchBuckets: [Int] = [1, 2, 4]

    /// Sampling temperature
    public var temperature: Float

    /// Top-p sampling
    public var topP: Float

    /// Top-k sampling (0 disables)
    public var topK: Int

    /// Min-p sampling threshold relative to the highest probability token (0 disables)
    public var minP: Float

    /// Optional random seed for stochastic samplers. nil preserves the
    /// existing time-seeded behavior.
    public var randomSeed: UInt64?

    /// Penalty factor for repeating tokens
    public var repetitionPenalty: Float?

    /// Number of tokens to consider for repetition penalty
    public var repetitionContextSize: Int

    /// additive penalty for tokens that appear in recent context
    public var presencePenalty: Float?

    /// number of tokens to consider for presence penalty
    public var presenceContextSize: Int

    /// additive penalty that scales with token frequency in recent context
    public var frequencyPenalty: Float?

    /// number of tokens to consider for frequency penalty
    public var frequencyContextSize: Int

    /// Speculative-decoding strategy (opt-in). `nil` preserves the existing
    /// autoregressive decode path byte-for-byte — callers who don't set this
    /// see no behaviour change.
    ///
    /// The legacy autoregressive draft-model path in
    /// `SpeculativeTokenIterator` is reached via ``DraftStrategy/autoregressive(draftModel:numDraftTokens:)``.
    ///
    /// Block-diffusion strategies (``DraftStrategy/dflash(drafterPath:blockSize:)``
    /// and ``DraftStrategy/ddtree(drafterPath:branchingBudget:blockSize:)``)
    /// activate the native Swift/MLX SpecDec runtime in
    /// `Libraries/MLXLMCommon/SpecDec/`. See that directory's
    /// `DDTREE-DESIGN.md` for the full spec.
    public var draftStrategy: DraftStrategy? = nil

    /// Additional text-level stop sequences. When any of these strings
    /// appears in the user-visible assistant output, the library halts
    /// generation, truncates the match and everything after it, and
    /// emits `.info(stopReason: .stop)`.
    ///
    /// Matching happens against the `.chunk(String)` stream — i.e.,
    /// reasoning and tool-call bytes are NOT candidates for a
    /// stop-sequence match, matching the semantics an OpenAI-compatible
    /// server expects.
    ///
    /// Empty, orthogonal to `ModelConfiguration.extraEOSTokens` (which
    /// is token-level). Callers can combine both: EOS tokens halt on
    /// token-id match before detokenization; stop strings halt on
    /// decoded-text match after the reasoning + tool-call pipeline.
    ///
    /// See `Libraries/MLXLMCommon/BatchEngine/STOP-SEQUENCES-CONTRACT.md`.
    public var extraStopStrings: [String] = []

    public init(
        maxTokens: Int? = nil,
        maxKVSize: Int? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        kvMode: KVQuantizationMode = .none,
        enableCompiledDecode: Bool = false,
        compiledMaxCacheLength: Int? = nil,
        accelerationMode: AccelerationMode? = nil,
        enableCompiledBatchDecode: Bool = false,
        compiledBatchBuckets: [Int] = [1, 2, 4],
        temperature: Float = 0.6,
        topP: Float = 1.0,
        topK: Int = 0,
        minP: Float = 0.0,
        randomSeed: UInt64? = nil,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20,
        presencePenalty: Float? = nil,
        presenceContextSize: Int = 20,
        frequencyPenalty: Float? = nil,
        frequencyContextSize: Int = 20,
        prefillStepSize: Int = 512,
        extraStopStrings: [String] = []
    ) {
        self.maxTokens = maxTokens
        self.maxKVSize = maxKVSize
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.kvMode = kvMode
        self.enableCompiledDecode = enableCompiledDecode
        self.compiledMaxCacheLength = compiledMaxCacheLength
        self.accelerationMode =
            accelerationMode ?? AccelerationRuntime.requestedMode()
        self.enableCompiledBatchDecode = enableCompiledBatchDecode
        self.compiledBatchBuckets = compiledBatchBuckets
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.randomSeed = randomSeed
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.presencePenalty = presencePenalty
        self.presenceContextSize = presenceContextSize
        self.frequencyPenalty = frequencyPenalty
        self.frequencyContextSize = frequencyContextSize
        self.prefillStepSize = prefillStepSize
        self.extraStopStrings = extraStopStrings
    }

    public init(
        generationConfig: GenerationConfigFile?,
        fallback: GenerateParameters = GenerateParameters()
    ) {
        self = fallback
        guard let generationConfig else { return }

        if let maxNewTokens = generationConfig.maxNewTokens {
            self.maxTokens = maxNewTokens
        }
        if let temperature = generationConfig.temperature {
            self.temperature = temperature
        }
        if let topP = generationConfig.topP {
            self.topP = topP
        }
        if let topK = generationConfig.topK {
            self.topK = topK
        }
        if let minP = generationConfig.minP {
            self.minP = minP
        }
        if let repetitionPenalty = generationConfig.repetitionPenalty {
            self.repetitionPenalty = repetitionPenalty
        }
        if generationConfig.doSample == false {
            self.temperature = 0
        }
    }

    public func sampler() -> LogitSampler {
        let usesTopP = topP > 0 && topP < 1
        let usesTopK = topK > 0
        let usesMinP = minP > 0

        if temperature == 0 {
            return ArgMaxSampler()
        } else if usesTopP || usesTopK || usesMinP {
            return TopPSampler(
                temperature: temperature, topP: topP, topK: topK,
                minP: minP, randomSeed: randomSeed)
        } else {
            return CategoricalSampler(temperature: temperature, randomSeed: randomSeed)
        }
    }

    public func processor() -> LogitProcessor? {
        let repetitionContext: RepetitionContext?
        // 2026-04-30 fix (Bug 3a): also skip the no-op case where
        // `repetitionPenalty == 1.0` — that's the HuggingFace idiom for
        // "no penalty," shipped in many `generation_config.json` files
        // (notably Nemotron-3-Nano-Omni). Multiplying / dividing logits
        // by 1.0 is a mathematical no-op, so building a RepetitionContext
        // for it is wasted work AND, when it happens, exposes a latent
        // bounds-check panic in mlx-swift's `MLXArray[range].subscript`
        // that kills the process on first decode (osaurus crash report
        // 2026-04-30-141326.ips). Treating 1.0 as nil here is the
        // correct semantic AND the safe runtime choice.
        if let repetitionPenalty,
           repetitionPenalty != 0,
           repetitionPenalty != 1.0,
           repetitionContextSize > 0 {
            repetitionContext = RepetitionContext(
                repetitionPenalty: repetitionPenalty,
                repetitionContextSize: repetitionContextSize
            )
        } else {
            repetitionContext = nil
        }

        let presenceContext: PresencePenaltyContext?
        if let presencePenalty, presencePenalty != 0, presenceContextSize > 0 {
            presenceContext = PresencePenaltyContext(
                presencePenalty: presencePenalty,
                presenceContextSize: presenceContextSize
            )
        } else {
            presenceContext = nil
        }

        let frequencyContext: FrequencyPenaltyContext?
        if let frequencyPenalty, frequencyPenalty != 0, frequencyContextSize > 0 {
            frequencyContext = FrequencyPenaltyContext(
                frequencyPenalty: frequencyPenalty,
                frequencyContextSize: frequencyContextSize
            )
        } else {
            frequencyContext = nil
        }

        if repetitionContext == nil && presenceContext == nil && frequencyContext == nil {
            return nil
        }

        return PenaltyProcessor(
            repetitionContext: repetitionContext,
            presenceContext: presenceContext,
            frequencyContext: frequencyContext
        )
    }
}

/// Sampler that uses `argMax` (most likely) to sample the logits.
public struct ArgMaxSampler: LogitSampler {
    public init() {}

    public func sample(logits: MLXArray) -> MLXArray {
        argMax(logits, axis: -1)
    }
}

/// Sampler that uses probability filters (`topP`, `topK`, `minP`) and `temperature`
/// to sample the logits.
///
/// Filters are applied in the same order as Python mlx-lm: top_p → min_p → top_k.
/// Each filter operates on the full vocabulary in original token order, masking
/// rejected tokens with `-inf`. This matches the composable filter chain in
/// `mlx_lm.sample_utils.make_sampler`.
public struct TopPSampler: LogitSampler {
    let temp: MLXArray
    let topP: MLXArray?
    let topK: Int?
    let minP: MLXArray?
    let negInf: MLXArray
    let randomState: MLXRandom.RandomState

    public init(
        temperature: Float, topP: Float = 1.0, topK: Int = 0,
        minP: Float = 0.0, randomSeed: UInt64? = nil
    ) {
        self.temp = MLXArray(temperature)
        if topP > 0 && topP < 1 {
            self.topP = MLXArray(topP)
        } else {
            self.topP = nil
        }
        self.topK = topK > 0 ? topK : nil
        self.minP = minP > 0 ? MLXArray(minP) : nil
        self.negInf = MLXArray(-Float.infinity)
        self.randomState = randomSeed.map { MLXRandom.RandomState(seed: $0) }
            ?? MLXRandom.RandomState()
    }

    public func sample(logits: MLXArray) -> MLXArray {
        var logits = logits
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }

        return withRandomState(randomState) {
            var logprobs = logSoftmax(logits)

            // Apply filters in Python mlx-lm order: top_p → min_p → top_k.
            if let topP {
                logprobs = applyTopP(logprobs, topP: topP)
            }
            if let minP {
                logprobs = applyMinP(logprobs, minP: minP)
            }
            if let topK {
                logprobs = applyTopK(logprobs, topK: topK)
            }

            return categorical(logprobs * (1 / temp))
        }
    }

    /// Keep tokens whose cumulative probability exceeds `1 - topP` (nucleus sampling).
    /// Matches `apply_top_p` from `mlx_lm/sample_utils.py`.
    private func applyTopP(_ logprobs: MLXArray, topP: MLXArray) -> MLXArray {
        let sortedIndices = argSort(logprobs, axis: -1)
        let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
        let sortedProbs = exp(sortedLogprobs)
        let cumulativeProbs = cumsum(sortedProbs, axis: -1)

        // Mask low-probability tail in sorted order, scatter back to original vocab order.
        let filtered = MLX.where(cumulativeProbs .> (1 - topP), sortedLogprobs, negInf)
        return putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
    }

    /// Keep tokens with probability >= maxProb * minP.
    /// Matches `apply_min_p` from `mlx_lm/sample_utils.py`.
    private func applyMinP(_ logprobs: MLXArray, minP: MLXArray) -> MLXArray {
        // threshold in log-space: log(maxProb * minP) = maxLogprob + log(minP)
        let maxLogprob = logprobs.max(axis: -1, keepDims: true)
        let threshold = maxLogprob + log(minP)
        return MLX.where(logprobs .>= threshold, logprobs, negInf)
    }

    /// Keep only the top-k highest-probability tokens.
    /// Mirrors `apply_top_k` from `mlx_lm/sample_utils.py`.
    private func applyTopK(_ logprobs: MLXArray, topK: Int) -> MLXArray {
        let vocabularySize = logprobs.dim(-1)
        guard topK < vocabularySize else { return logprobs }
        // O(V) partition on negated logprobs so top-k land at [0, topK).
        // Indices at [topK, V) are the tokens to mask out.
        let maskIndices = argPartition(-logprobs, kth: topK - 1, axis: -1)[0..., topK...]
        return putAlong(logprobs, maskIndices, values: negInf, axis: -1)
    }
}

/// Sampler that uses `temperature` to sample the logits.
public struct CategoricalSampler: LogitSampler {
    let temp: MLXArray
    let randomState: MLXRandom.RandomState

    public init(temperature: Float, randomSeed: UInt64? = nil) {
        self.temp = MLXArray(temperature)
        self.randomState = randomSeed.map { MLXRandom.RandomState(seed: $0) }
            ?? MLXRandom.RandomState()
    }

    public func sample(logits: MLXArray) -> MLXArray {
        return withRandomState(randomState) {
            categorical(logits * (1 / temp))
        }
    }
}

/// Sampling helper used by exact speculative decoding paths.
///
/// The normal ``LogitSampler`` protocol intentionally returns only a sampled
/// token. Speculative accept/reject also needs the probability assigned to the
/// draft token by both the verifier and draft distributions. This helper keeps
/// the same filter order as ``TopPSampler`` and adds probability-ratio
/// acceptance plus residual correction sampling.
public struct SpeculativeSamplingController {
    public struct Sample {
        public let token: MLXArray
        public let probabilities: MLXArray
    }

    public struct AcceptanceDecision {
        public let accepted: Bool
        public let acceptanceProbability: Float
        public let correction: MLXArray?
    }

    private let temperature: Float
    private let topP: Float
    private let topK: Int
    private let minP: Float
    private let sampleState: MLXRandom.RandomState
    private let acceptanceState: MLXRandom.RandomState
    private let residualState: MLXRandom.RandomState
    private let negInf = MLXArray(-Float.infinity)

    public init(parameters: GenerateParameters) {
        self.temperature = parameters.temperature
        self.topP = parameters.topP
        self.topK = parameters.topK
        self.minP = parameters.minP

        if let seed = parameters.randomSeed {
            self.sampleState = MLXRandom.RandomState(seed: seed)
            self.acceptanceState = MLXRandom.RandomState(seed: seed &+ 0x9E37_79B9_7F4A_7C15)
            self.residualState = MLXRandom.RandomState(seed: seed &+ 0xD1B5_4A32_D192_ED03)
        } else {
            self.sampleState = MLXRandom.RandomState()
            self.acceptanceState = MLXRandom.RandomState()
            self.residualState = MLXRandom.RandomState()
        }
    }

    public var isGreedy: Bool {
        temperature == 0
    }

    public func probabilities(logits: MLXArray) -> MLXArray {
        precondition(!isGreedy, "greedy speculative decoding does not need distributions")
        var logits = normalizedRow(logits)
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }

        var logprobs = logSoftmax(logits)
        if topP > 0 && topP < 1 {
            logprobs = applyTopP(logprobs, topP: MLXArray(topP))
        }
        if minP > 0 {
            logprobs = applyMinP(logprobs, minP: MLXArray(minP))
        }
        if topK > 0 {
            logprobs = applyTopK(logprobs, topK: topK)
        }

        return softmax(logprobs * (1 / MLXArray(temperature)), axis: -1, precise: true)
    }

    public func sample(logits: MLXArray) -> Sample {
        let probabilities = probabilities(logits: logits)
        return Sample(
            token: sample(probabilities: probabilities, state: sampleState),
            probabilities: probabilities)
    }

    public func sampleFromTarget(probabilities: MLXArray) -> MLXArray {
        sample(probabilities: probabilities, state: sampleState)
    }

    public func acceptOrCorrect(
        draftToken: MLXArray,
        targetProbabilities: MLXArray,
        draftProbabilities: MLXArray
    ) -> AcceptanceDecision {
        let p = probability(targetProbabilities, token: draftToken)
        let q = probability(draftProbabilities, token: draftToken)

        let acceptanceProbability: Float
        if q <= 0 {
            acceptanceProbability = p > 0 ? 1 : 0
        } else {
            acceptanceProbability = min(1, p / q)
        }

        if acceptanceProbability >= 1 {
            return AcceptanceDecision(
                accepted: true,
                acceptanceProbability: acceptanceProbability,
                correction: nil)
        }

        let roll = withRandomState(acceptanceState) {
            MLXRandom.uniform(0.0 ..< 1.0).item(Float.self)
        }
        if roll <= acceptanceProbability {
            return AcceptanceDecision(
                accepted: true,
                acceptanceProbability: acceptanceProbability,
                correction: nil)
        }

        let correction = sampleResidual(
            targetProbabilities: targetProbabilities,
            draftProbabilities: draftProbabilities)
        return AcceptanceDecision(
            accepted: false,
            acceptanceProbability: acceptanceProbability,
            correction: correction)
    }

    private func sampleResidual(
        targetProbabilities: MLXArray,
        draftProbabilities: MLXArray
    ) -> MLXArray {
        let target = normalizedRow(targetProbabilities)
        let draft = normalizedRow(draftProbabilities)
        let delta = target - draft
        let residual = MLX.where(delta .> 0, delta, MLXArray(0.0, dtype: delta.dtype))
        let mass = residual.sum().item(Float.self)
        let probabilities = mass > 0 ? residual / MLXArray(mass) : target
        return sample(probabilities: probabilities, state: residualState)
    }

    private func sample(
        probabilities: MLXArray,
        state: MLXRandom.RandomState
    ) -> MLXArray {
        withRandomState(state) {
            categorical(log(normalizedRow(probabilities)))
        }
    }

    private func probability(_ probabilities: MLXArray, token: MLXArray) -> Float {
        let row = normalizedRow(probabilities)
        let tokenID = token.item(Int.self)
        guard tokenID >= 0, tokenID < row.dim(-1) else { return 0 }
        return row[0..., tokenID ..< (tokenID + 1)].item(Float.self)
    }

    private func normalizedRow(_ array: MLXArray) -> MLXArray {
        array.ndim == 1 ? array.reshaped(1, array.dim(0)) : array
    }

    /// Keep tokens whose cumulative probability exceeds `1 - topP`.
    private func applyTopP(_ logprobs: MLXArray, topP: MLXArray) -> MLXArray {
        let sortedIndices = argSort(logprobs, axis: -1)
        let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
        let sortedProbs = exp(sortedLogprobs)
        let cumulativeProbs = cumsum(sortedProbs, axis: -1)
        let filtered = MLX.where(cumulativeProbs .> (1 - topP), sortedLogprobs, negInf)
        return putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
    }

    /// Keep tokens with probability >= maxProb * minP.
    private func applyMinP(_ logprobs: MLXArray, minP: MLXArray) -> MLXArray {
        let maxLogprob = logprobs.max(axis: -1, keepDims: true)
        let threshold = maxLogprob + log(minP)
        return MLX.where(logprobs .>= threshold, logprobs, negInf)
    }

    /// Keep only the top-k highest-probability tokens.
    private func applyTopK(_ logprobs: MLXArray, topK: Int) -> MLXArray {
        let vocabularySize = logprobs.dim(-1)
        guard topK < vocabularySize else { return logprobs }
        let maskIndices = argPartition(-logprobs, kth: topK - 1, axis: -1)[0..., topK...]
        return putAlong(logprobs, maskIndices, values: negInf, axis: -1)
    }
}

/// GPU-resident ring buffer of recent token IDs.
///
/// Shared by penalty processors to avoid duplicating ring buffer logic.
/// Uses `MLX.where` mask operations for GPU-only updates (no CPU←GPU sync),
/// preserving `asyncEval()` pipelining in `TokenIterator`.
struct TokenRing {
    private(set) var buffer: MLXArray
    private(set) var count = 0
    private var writeIndex = 0
    let capacity: Int
    private let positions: MLXArray

    init(capacity: Int) {
        precondition(capacity > 0)
        self.capacity = capacity
        self.buffer = MLXArray.zeros([capacity], type: Int32.self)
        self.positions = MLXArray.arange(capacity)
    }

    /// The valid portion of the ring (all of it once full), or `nil` if empty.
    var validTokens: MLXArray? {
        guard count > 0 else { return nil }
        return count < capacity ? buffer[..<count] : buffer
    }

    /// Bulk-load from a prompt. Keeps the last `capacity` tokens.
    mutating func loadPrompt(_ prompt: MLXArray) {
        let n = prompt.dim(0)
        let promptTokens = prompt.asType(.int32)
        if n <= capacity {
            if n < capacity {
                let padding = MLXArray.zeros([capacity - n], type: Int32.self)
                buffer = concatenated([promptTokens.reshaped(-1), padding])
            } else {
                buffer = promptTokens.reshaped(-1)
            }
            count = n
            writeIndex = n % capacity
        } else {
            buffer = promptTokens[(-capacity)...].reshaped(-1)
            count = capacity
            writeIndex = 0
        }
    }

    /// Append a single token using GPU-only mask write (no CPU←GPU sync).
    mutating func append(_ token: MLXArray) {
        let mask = positions .== Int32(writeIndex)
        buffer = MLX.where(mask, token.asType(.int32), buffer)
        writeIndex = (writeIndex + 1) % capacity
        count = min(count + 1, capacity)
    }
}

/// Processor that implements a `repetitionPenalty`.
public struct RepetitionContext: LogitProcessor {
    private var ring: TokenRing
    let repetitionPenalty: Float

    public init(repetitionPenalty: Float, repetitionContextSize: Int) {
        self.repetitionPenalty = repetitionPenalty
        self.ring = TokenRing(capacity: repetitionContextSize)
    }

    mutating public func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard let indices = ring.validTokens?.asType(.uint32) else { return logits }
        var selectedLogits = logits[0..., indices]

        selectedLogits = MLX.where(
            selectedLogits .< 0, selectedLogits * repetitionPenalty,
            selectedLogits / repetitionPenalty)

        logits[0..., indices] = selectedLogits
        return logits
    }

    mutating public func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Processor that applies an additive presence penalty to tokens in a recent context window.
///
/// The penalty is applied once per unique token via scatter-write (writing the
/// same value to the same index multiple times is idempotent).
public struct PresencePenaltyContext: LogitProcessor {
    private var ring: TokenRing
    let presencePenalty: Float

    public init(presencePenalty: Float, presenceContextSize: Int) {
        self.presencePenalty = presencePenalty
        self.ring = TokenRing(capacity: presenceContextSize)
    }

    mutating public func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard let indices = ring.validTokens?.asType(.uint32) else { return logits }
        logits[0..., indices] = logits[0..., indices] - presencePenalty
        return logits
    }

    mutating public func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Processor that applies an additive frequency penalty to tokens in a recent context window.
///
/// Frequency counting is performed on GPU via `scatter_add` to build a histogram
/// of token occurrences, avoiding CPU←GPU synchronization.
public struct FrequencyPenaltyContext: LogitProcessor {
    private var ring: TokenRing
    let frequencyPenalty: Float

    public init(frequencyPenalty: Float, frequencyContextSize: Int) {
        self.frequencyPenalty = frequencyPenalty
        self.ring = TokenRing(capacity: frequencyContextSize)
    }

    mutating public func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard let validTokens = ring.validTokens else { return logits }

        let vocabSize = logits.dim(-1)
        let ones = MLXArray.ones([validTokens.dim(0)], type: Float32.self)
        let histogram = MLXArray.zeros([vocabSize], type: Float32.self)
            .at[validTokens.asType(.int32)].add(ones)

        return logits - (histogram * frequencyPenalty).reshaped(1, -1)
    }

    mutating public func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Processor that composes penalty processors in Python mlx-lm order.
public struct PenaltyProcessor: LogitProcessor {
    var repetitionContext: RepetitionContext?
    var presenceContext: PresencePenaltyContext?
    var frequencyContext: FrequencyPenaltyContext?

    public init(
        repetitionContext: RepetitionContext?,
        presenceContext: PresencePenaltyContext?,
        frequencyContext: FrequencyPenaltyContext?
    ) {
        self.repetitionContext = repetitionContext
        self.presenceContext = presenceContext
        self.frequencyContext = frequencyContext
    }

    mutating public func prompt(_ prompt: MLXArray) {
        repetitionContext?.prompt(prompt)
        presenceContext?.prompt(prompt)
        frequencyContext?.prompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        var logits = logits
        logits = repetitionContext?.process(logits: logits) ?? logits
        logits = presenceContext?.process(logits: logits) ?? logits
        logits = frequencyContext?.process(logits: logits) ?? logits
        return logits
    }

    mutating public func didSample(token: MLXArray) {
        repetitionContext?.didSample(token: token)
        presenceContext?.didSample(token: token)
        frequencyContext?.didSample(token: token)
    }
}

/// Common properties shared by token-generating iterators.
public protocol TokenIteratorProtocol: Sequence, IteratorProtocol where Element == Int {
    var maxTokens: Int? { get }
    var tokenCount: Int { get }
    var promptPrefillTime: TimeInterval { get }
    var promptTokenIds: [Int] { get }
    mutating func storeCacheAfterGeneration(
        generatedTokenIds: [Int],
        includeGeneratedBoundary: Bool)
}

extension TokenIteratorProtocol {
    public var promptTokenIds: [Int] { [] }

    public mutating func storeCacheAfterGeneration(
        generatedTokenIds: [Int],
        includeGeneratedBoundary: Bool
    ) {}
}

private struct MLXPressGenerationProfileRow {
    var count = 0
    var seconds: Double = 0
}

private final class MLXPressGenerationProfileState: @unchecked Sendable {
    static let shared = MLXPressGenerationProfileState()

    let isEnabled: Bool
    private let lock = NSLock()
    private var rows: [String: MLXPressGenerationProfileRow] = [:]

    private init() {
        let env = ProcessInfo.processInfo.environment
        let raw =
            env["MLXPRESS_GENERATION_PROFILE"]?
            .lowercased()
            ?? env["JANGPRESS_GENERATION_PROFILE"]?.lowercased()
            ?? "0"
        self.isEnabled = raw == "1" || raw == "true" || raw == "yes" || raw == "on"
    }

    func time<T>(_ name: String, _ body: () throws -> T) rethrows -> T {
        guard isEnabled else { return try body() }
        let start = Date.timeIntervalSinceReferenceDate
        do {
            let value = try body()
            record(name, seconds: Date.timeIntervalSinceReferenceDate - start)
            return value
        } catch {
            record(name, seconds: Date.timeIntervalSinceReferenceDate - start)
            throw error
        }
    }

    func dumpAndReset(reason: String) {
        guard isEnabled else { return }
        lock.lock()
        let snapshot = rows
        rows.removeAll(keepingCapacity: true)
        lock.unlock()

        let totalSeconds = snapshot.values.reduce(0) { $0 + $1.seconds }
        let detail = snapshot
            .sorted {
                if $0.value.seconds == $1.value.seconds {
                    return $0.key < $1.key
                }
                return $0.value.seconds > $1.value.seconds
            }
            .map { name, row -> String in
                let totalMS = row.seconds * 1000
                let avgMS = totalMS / Double(max(1, row.count))
                return String(
                    format: "%@ count=%d total=%.1fms avg=%.3fms",
                    name, row.count, totalMS, avgMS)
            }
            .joined(separator: " | ")
        FileHandle.standardError.write(
            Data(
                String(
                    format: "[MLXPressGenerationProfile] %@ total=%.1fms %@\n",
                    reason, totalSeconds * 1000, detail
                ).utf8))
    }

    private func record(_ name: String, seconds: Double) {
        lock.lock()
        var row = rows[name] ?? MLXPressGenerationProfileRow()
        row.count += 1
        row.seconds += seconds
        rows[name] = row
        lock.unlock()
    }
}

private enum MLXPressGenerationProfile {
    static func time<T>(_ name: String, _ body: () throws -> T) rethrows -> T {
        try MLXPressGenerationProfileState.shared.time(name, body)
    }

    static func dumpAndReset(reason: String) {
        MLXPressGenerationProfileState.shared.dumpAndReset(reason: reason)
    }
}

/// Generator of tokens.
///
/// This is typically used via a call to ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>`.
///
/// To use it directly:
///
/// ```swift
/// let generateParameters: GenerateParameters
/// let input: LMInput
/// let model: LanguageModel
///
/// let iterator = try TokenIterator(input: input, model: model, parameters: generateParameters)
///
/// for token in iterator {
///     ...
/// }
/// ```
///
/// Tokens are integers that can be passed through a `Tokenizer` or ``StreamingDetokenizer`` to produce Strings.
///
/// Port of `generate_step()` from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py
///
/// Note: this uses `asyncEval()` and there may be an async evaluation running after a call to `next()`.
public struct TokenIterator: TokenIteratorProtocol {

    private static let logger = Logger(subsystem: "vmlx", category: "TokenIterator")

    private static func compiledDecodeDenied(for model: any LanguageModel) -> Bool {
        let typeName = String(describing: type(of: model)).lowercased()
        if typeName.contains("hy3") || typeName.contains("hunyuan") {
            return true
        }
        if typeName.contains("minimax") {
            return !compiledDecodeAllowsMiniMax()
        }
        return false
    }

    private static func compiledDecodeAllowsMiniMax() -> Bool {
        ["MLXPRESS_COMPILED_DECODE_ALLOW_MINIMAX", "JANGPRESS_COMPILED_DECODE_ALLOW_MINIMAX"]
            .contains { key in
                guard let raw = getenv(key) else { return false }
                switch String(cString: raw).lowercased() {
                case "1", "true", "yes", "on":
                    return true
                default:
                    return false
                }
            }
    }

    let model: any LanguageModel
    var state: LMOutput.State?

    var y: LMInput.Text
    var cache: [KVCache]
    var processor: LogitProcessor?
    let sampler: LogitSampler

    public var tokenCount = 0
    public let maxTokens: Int?

    // Cache quantization parameters
    let kvBits: Int?
    let kvGroupSize: Int
    let quantizedKVStart: Int
    let kvMode: KVQuantizationMode

    private var compiledForward: (@Sendable ([MLXArray]) -> [MLXArray])?

    // Multi-tier cache coordinator (skeleton integration)
    let cacheCoordinator: CacheCoordinator?

    /// Prompt token IDs captured at init for cache store after generation.
    public let promptTokenIds: [Int]

    /// Canonical prompt-prefix boundaries safe to store in addition to the
    /// full generation prompt.
    let cachePrefixTokenCounts: [Int]

    /// Original prepared input, retained for correctness-first re-derive of
    /// cache-prefix boundaries that cannot be produced by trimming.
    let originalInput: LMInput

    /// Parameters used to allocate compatible cache layers for boundary
    /// re-derive. This preserves rotating/sliding cache choices while the
    /// store path still writes raw prompt-boundary KV to disk.
    let cacheInitParameters: GenerateParameters?

    /// Clean cache state captured immediately after prefill and before any
    /// generated token is fed back into the model.
    var promptCacheSnapshot: [KVCache]?

    /// Stable fingerprint of any request-scope or media content in the input.
    /// `nil` for ordinary text-only inputs. Mixed into cache-coordinator keys
    /// so reasoning-mode and VLM multi-turn conversations can cache-hit without
    /// colliding with other modes/media.
    let mediaSalt: String?

    // Internal metrics
    public var promptPrefillTime: TimeInterval = 0.0

    /// DSV4's HybridPoolCache carries compressor/indexer pool state in
    /// addition to the local sliding-window KV. Chunked prefill mutates that
    /// pool across multiple forwards and has diverged from the Python
    /// production path, so force a single prepare-forward for this cache
    /// family unless the caller is only seeding a one-token cache hit.
    private func effectivePrefillWindow(
        requested: Int,
        input: LMInput
    ) -> Int {
        guard cache.contains(where: { $0 is HybridPoolCache }) else {
            return requested
        }
        return Swift.max(requested, input.text.tokens.size)
    }

    /// Initialize a `TokenIterator` with the given tokens. Note: this has been
    /// replaced with ``init(input:model:cache:parameters:)``.
    ///
    /// - Parameters:
    ///   - prompt: the prompt tokens
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - parameters: the generation parameters
    @available(*, deprecated, message: "please use init(input:model:cache:parameters:)")
    public init(
        prompt: MLXArray, model: any LanguageModel, cache: [KVCache]? = nil,
        parameters: GenerateParameters
    ) throws {
        _ = try AccelerationRuntime.resolveTextDecode(parameters.accelerationMode)

        self.model = model
        self.y = .init(tokens: prompt)
        self.cache = cache ?? model.newCache(parameters: parameters)

        self.processor = parameters.processor()
        self.sampler = parameters.sampler()
        self.maxTokens = parameters.maxTokens

        self.kvBits = parameters.kvBits
        self.kvGroupSize = parameters.kvGroupSize
        self.quantizedKVStart = parameters.quantizedKVStart
        self.kvMode = parameters.kvMode

        self.cacheCoordinator = nil
        self.promptTokenIds = []
        self.cachePrefixTokenCounts = []
        self.originalInput = LMInput(text: y)
        self.cacheInitParameters = parameters
        self.promptCacheSnapshot = nil
        self.mediaSalt = nil

        self.promptPrefillTime = try measure {
            let promptInput = LMInput(text: y)
            try prepare(
                input: promptInput,
                windowSize: effectivePrefillWindow(
                    requested: parameters.prefillStepSize,
                    input: promptInput))
        }
        self.promptCacheSnapshot = makePromptBoundaryCacheSnapshot(from: self.cache)
    }

    /// Initialize a `TokenIterator` with the given input.
    ///
    /// If more control is needed over the generation,
    /// ``init(input:model:cache:processor:sampler:prefillStepSize:)``
    /// allows a caller to specify ``LogitProcessor`` and ``LogitSampler``
    /// directly.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - parameters: the generation parameters
    ///   - cacheCoordinator: optional multi-tier cache coordinator for prefix reuse
    public init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        parameters: GenerateParameters,
        cacheCoordinator: CacheCoordinator? = nil
    ) throws {
        _ = try AccelerationRuntime.resolveTextDecode(parameters.accelerationMode)

        self.model = model
        self.y = input.text
        self.cacheCoordinator = cacheCoordinator
        let promptTokenCount = input.text.tokens.size
        var effectiveParameters = parameters
        if let coordinator = cacheCoordinator {
            let resolvedPolicy = coordinator.config.resolveKVPolicy(
                kvMode: parameters.kvMode,
                maxKVSize: parameters.maxKVSize,
                promptTokenCount: promptTokenCount)
            effectiveParameters.kvMode = resolvedPolicy.kvMode
            effectiveParameters.maxKVSize = resolvedPolicy.maxKVSize
        }
        self.cache = cache ?? model.newCache(parameters: effectiveParameters)
        if let coordinator = cacheCoordinator,
           effectiveParameters.kvBits != nil || effectiveParameters.kvMode != .none
        {
            coordinator.setPagedIncompatible(true)
        }

        self.processor = effectiveParameters.processor()
        self.sampler = effectiveParameters.sampler()
        self.maxTokens = effectiveParameters.maxTokens

        self.kvBits = effectiveParameters.kvBits
        self.kvGroupSize = effectiveParameters.kvGroupSize
        self.quantizedKVStart = effectiveParameters.quantizedKVStart
        self.kvMode = effectiveParameters.kvMode

        // Capture prompt token IDs for cache store after generation.
        if promptTokenCount > 0 {
            self.promptTokenIds = input.text.tokens.reshaped(-1).asArray(Int.self)
        } else {
            self.promptTokenIds = []
        }
        self.cachePrefixTokenCounts = input.cachePrefixTokenCounts
        self.originalInput = input
        self.cacheInitParameters = effectiveParameters
        self.promptCacheSnapshot = nil

        // Compute a stable fingerprint of request-scope/media content plus
        // effective cache policy once at init, so both the pre-prepare fetch
        // below and the post-generation store see the same salt.
        self.mediaSalt = computeCacheSalt(for: input, parameters: effectiveParameters)

        // Multi-tier cache: attempt prefix fetch before prepare.
        // On cache hit, restore KV state and only prefill remaining tokens.
        //
        // VLM inputs (image/video/audio) are now supported: the mediaSalt computed
        // above is mixed into the cache keys by the coordinator, so "same
        // text prefix + same media" hits while "same text + different media"
        // misses. Previously image/video bypassed the cache entirely,
        // wasting a full media encoder pass and prefill on every turn.
        var inputForPrepare = input
        // SLIDING-1 (2026-04-15): the legacy guard `!hasRotatingCache` was
        // removed once `TQDiskSerializer` v2 + `restoreRotatingLayer` /
        // `restoreFromV2Arrays` learned to round-trip the ring buffer +
        // 5-tuple `metaState` cleanly. Sliding-window models (Gemma3,
        // Gemma3n, Gemma4 SWA layers, Mistral4 with maxKVSize, MiMoV2Flash,
        // BaichuanM1, Qwen3.5-VL inherited) now get full L2 disk
        // persistence + paged restore on cache hit.
        if let coordinator = cacheCoordinator, !promptTokenIds.isEmpty {
            if !coordinator.isHybrid {
                if cacheContainsPathDependentState(self.cache) {
                    coordinator.setHybrid(true)
                    Self.logger.info(
                        "TokenIterator: coordinator flipped to isHybrid=true"
                    )
                }
            }
            // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass) and
            // 2026-05-06 (Gemma4 SWA cache-hit fix):
            // Mirror BatchEngine.admit's detection. Hybrid-pool and
            // rotating/sliding-window caches must restore through the disk
            // serializer because the paged tier stores only full-history KV
            // blocks and cannot round-trip rotating ring metadata.
            if !coordinator.isPagedIncompatible {
                if cacheRequiresDiskBackedCoordinatorRestore(self.cache) {
                    coordinator.setPagedIncompatible(true)
                    Self.logger.info(
                        "TokenIterator: coordinator flipped to isPagedIncompatible=true"
                    )
                }
            }
            let result = coordinator.fetch(
                tokens: promptTokenIds, mediaSalt: mediaSalt)
            switch result {
            case .hit(_, let remainingTokens, let detail, let blocks, let ssmStates, let diskArrays):
                var restored = false
                if !blocks.isEmpty {
                    let restoredTokens = restoreLayerData(from: blocks, into: self.cache)
                    coordinator.release(blocks: blocks)
                    if restoredTokens > 0 {
                        if let ssm = ssmStates {
                            restoreSSMStates(ssm, into: self.cache)
                        }
                        restored = true
                        Self.logger.info(
                            "Cache \(detail.rawValue) hit: restored \(restoredTokens) tokens, prefilling \(remainingTokens.count) remaining"
                        )
                    }
                }

                // Disk cache restore (blocks are empty, arrays are present)
                if let diskArrays, !restored {
                    let diskRestored = restoreFromDiskArrays(diskArrays, into: &self.cache)
                    if diskRestored > 0 {
                        if let ssm = ssmStates,
                           TQDiskSerializer.formatVersion(of: diskArrays) < 2
                        {
                            restoreSSMStates(ssm, into: self.cache)
                        }
                        // Mirror BatchEngine's disk-hit path: materialize
                        // restored arrays before prefill builds the next
                        // forward graph, instead of fusing restore + model
                        // compute into one high-pressure command buffer.
                        MLX.eval(self.cache)
                        restored = true
                        Self.logger.info(
                            "Cache \(detail.rawValue) hit: restored \(diskRestored) tokens from disk, prefilling \(remainingTokens.count) remaining"
                        )
                    }
                }

                if restored {
                    let hasPathDependentLayer = self.cache.contains { layer in
                        layer is MambaCache || layer is ArraysCache || layer is ZayaCCACache
                    }
                    let unsafePartial =
                        input.cacheHitSuffixContainsMediaPlaceholder(remainingTokens)
                    let unsafeFullHit = remainingTokens.isEmpty && hasPathDependentLayer
                    if unsafePartial || unsafeFullHit {
                        let why: String
                        if unsafePartial {
                            why = "media placeholder tokens remain in cache-hit suffix"
                        } else if unsafeFullHit {
                            why = "path-dependent full cache hit: re-feeding last token would double-count recurrent state"
                        } else {
                            why = "cache hit can't be extended safely"
                        }
                        Self.logger.info(
                            "TokenIterator: cache hit rolling back to full prefill (\(why))"
                        )
                        self.cache = self.model.newCache(parameters: effectiveParameters)
                        inputForPrepare = input
                    } else {
                        // Rebuild inputForPrepare with tokens shaped as `[1, T]`
                        // (2D batch-first). Some model forward paths — notably
                        // the Qwen3.5 VLM `Qwen35Language.LanguageModel` which
                        // reads `inputs.dim(1)` to compute position-ids — crash
                        // with MLX's `SmallVector out of range` (array.cpp:335)
                        // when fed a 1D tensor. Emitting 2D works uniformly
                        // because all `callAsFunction` paths either broadcast
                        // 2D already or tolerate the extra leading axis.
                        if remainingTokens.isEmpty, let last = promptTokenIds.last {
                            // Full cache hit — feed just the last token to seed decode.
                            // Match BatchEngine.stepPrefill: the restored cache already
                            // contains the full prompt, so trim it back to promptLen - 1
                            // before re-feeding the final prompt token. Without this,
                            // RoPE-positioned KV models re-feed the last token one
                            // position too far to the right after a full disk/paged hit,
                            // which can produce blank/newline-only first-token behavior
                            // on the B=1 solo path used by osaurus.
                            let promptLen = promptTokenIds.count
                            let cacheOffset = self.cache.first?.offset ?? promptLen
                            let trimNeeded = cacheOffset - (promptLen - 1)
                            if trimNeeded > 0 {
                                for layer in self.cache where layer.isTrimmable {
                                    _ = layer.trim(trimNeeded)
                                }
                            }
                            let lastToken = MLXArray([Int32(last)])
                                .expandedDimensions(axis: 0)
                            inputForPrepare = LMInput(
                                text: LMInput.Text(tokens: lastToken),
                                image: nil, video: nil)
                        } else {
                            let remainingArray = MLXArray(remainingTokens.map { Int32($0) })
                                .expandedDimensions(axis: 0)
                            inputForPrepare = LMInput(
                                text: LMInput.Text(tokens: remainingArray),
                                image: nil, video: nil)
                        }
                    }
                }
            case .miss:
                let count = promptTokenIds.count
                Self.logger.debug("Cache miss for \(count) prompt tokens")

                // 2026-05-05 (Ling-2.6-flash multi-turn fix): coordinator
                // missed but the cache may already hold a previous turn's
                // state (e.g. ChatSession reuse path with a hybrid
                // KVCache + ArraysCache that the multi-tier disk/paged
                // coordinator can't yet round-trip). Without correcting,
                // the model double-feeds previously-prefilled tokens onto
                // the populated cache → wrong RoPE positions on KVCache
                // layers AND duplicated GLA recurrence on ArraysCache
                // (Linear-Attn) layers → NaN logits → fatalError SIGKILL.
                //
                // We can only safely trim if the new prompt's prefix
                // matches the cached tokens. Some chat templates (Bailing,
                // DeepSeek-R1, Qwen3 reasoning) STRIP `<think>...</think>`
                // content from past assistant turns when re-rendering for
                // the next turn — so the input on Turn 2 is SHORTER than
                // what's actually cached (cache still holds the reasoning
                // tokens that were generated and decoded on Turn 1).
                //
                // Detection: if cacheOffset > promptTokenIds.count, the
                // cache holds content the new prompt doesn't include
                // (chat-template stripping). We can't safely trim — the
                // recurrent state encodes context the model would need
                // to "forget". Reset the cache and prefill the full new
                // prompt from scratch.
                if let cacheOffset = self.cache.first?.offset, cacheOffset > 0 {
                    if cacheOffset > promptTokenIds.count {
                        // Reasoning-strip mismatch — replace cache entirely.
                        // Some chat templates (Bailing, DeepSeek-R1, Qwen3
                        // reasoning) drop `<think>...</think>` blocks from
                        // past assistant turns when re-rendering for the
                        // next turn, so the new prompt is SHORTER than what
                        // was cached. We can't safely trim because we don't
                        // know exactly which positions to drop. Replace the
                        // cache with a fresh one — full re-prefill is
                        // O(prompt) but correct, vs producing garbage.
                        let resetMsg = "Populated-cache miss: cache offset (\(cacheOffset)) > prompt length (\(promptTokenIds.count)) — likely reasoning-strip in chat template; reset cache for full prefill"
                        self.cache = self.model.newCache(parameters: effectiveParameters)
                        Self.logger.info("\(resetMsg)")
                    } else if cacheOffset == promptTokenIds.count,
                              let last = promptTokenIds.last
                    {
                        let lastToken = MLXArray([Int32(last)])
                            .expandedDimensions(axis: 0)
                        inputForPrepare = LMInput(
                            text: LMInput.Text(tokens: lastToken),
                            image: nil, video: nil)
                        Self.logger.info(
                            "Populated-cache miss: full prefix matches cache (offset=\(cacheOffset)), seeding with last token only"
                        )
                    } else {
                        // cacheOffset < promptTokenIds.count — assume
                        // prefix matches (safe for templates that don't
                        // strip past content). Trim and prefill remainder.
                        let remaining = Array(promptTokenIds[cacheOffset...])
                        let remainingArray = MLXArray(remaining.map { Int32($0) })
                            .expandedDimensions(axis: 0)
                        inputForPrepare = LMInput(
                            text: LMInput.Text(tokens: remainingArray),
                            image: nil, video: nil)
                        Self.logger.info(
                            "Populated-cache miss: trimmed \(cacheOffset) cached tokens, prefilling \(remaining.count) remaining"
                        )
                    }
                }
            }
        }

        // Prefill: either full input (cache miss) or remaining tokens (cache hit).
        self.promptPrefillTime = try measure {
            try MLXPressGenerationProfile.time("prompt.prepare_total") {
                try prepare(
                    input: inputForPrepare,
                    windowSize: effectivePrefillWindow(
                        requested: effectiveParameters.prefillStepSize,
                        input: inputForPrepare))
            }
        }
        self.promptCacheSnapshot = makePromptBoundaryCacheSnapshot(from: self.cache)

        if effectiveParameters.enableCompiledDecode && !Self.compiledDecodeDenied(for: model) {
            try setupCompiledDecode(
                maxCacheLength: effectiveParameters.compiledMaxCacheLength ?? 4096)
        }
    }

    /// Initialize a `TokenIterator` with the given input and logit handling.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - processor: the logit processor
    ///   - sampler: the logit sampler
    ///   - prefillStepSize: optional prefill step size
    ///   - maxTokens: maximum number of tokens to generate
    public init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        processor: LogitProcessor?, sampler: LogitSampler, prefillStepSize: Int = 512,
        maxTokens: Int? = nil
    ) throws {
        self.model = model
        self.y = input.text
        self.cache = cache ?? model.newCache(parameters: nil)

        self.processor = processor
        self.sampler = sampler
        self.maxTokens = maxTokens

        // No cache quantization for this direct initialization
        self.kvBits = nil
        self.kvGroupSize = 64
        self.quantizedKVStart = 0
        self.kvMode = .none

        self.cacheCoordinator = nil
        self.promptTokenIds = []
        self.cachePrefixTokenCounts = input.cachePrefixTokenCounts
        self.originalInput = input
        self.cacheInitParameters = nil
        self.promptCacheSnapshot = nil
        self.mediaSalt = nil

        self.promptPrefillTime = try measure {
            try MLXPressGenerationProfile.time("prompt.prepare_total") {
                try prepare(
                    input: input,
                    windowSize: effectivePrefillWindow(
                        requested: prefillStepSize,
                        input: input))
            }
        }
        self.promptCacheSnapshot = makePromptBoundaryCacheSnapshot(from: self.cache)
    }

    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        processor?.prompt(input.text.tokens)

        let prepared = try MLXPressGenerationProfile.time("prompt.model_prepare") {
            try model.prepare(input, cache: cache, windowSize: windowSize)
        }
        switch prepared {
        case .tokens(let tokens):
            y = tokens

            // evaluate the remainder of the prompt -- this primes the pump
            let token = step(previous: y)
            y = .init(tokens: token)
            MLXPressGenerationProfile.time("prompt.async_eval_submit") {
                asyncEval(y.tokens)
            }

        case .logits(let result):
            y = .init(tokens: MLXPressGenerationProfile.time("prompt.sample") {
                convertToToken(logits: result.logits)
            })
            MLXPressGenerationProfile.time("prompt.async_eval_submit") {
                asyncEval(y.tokens)
            }
        }


    }

    mutating func convertToToken(logits: MLXArray) -> MLXArray {
        var logits = logits[0..., -1, 0...]

        if var processor {
            logits = processor.process(logits: logits)
            let y = sampler.sample(logits: logits)
            processor.didSample(token: y)
            self.processor = processor
            return y
        }

        return sampler.sample(logits: logits)
    }

    // Whether cache quantization is needed (skip the function call entirely when not)
    var needsCacheQuantization: Bool { kvBits != nil || kvMode != .none }

    /// Keep TurboQuant's encode/decode phase off the first-token critical path.
    ///
    /// `next()` returns the previous sampled token after it primes the next
    /// decode step. If we compress during that first priming step, TTFT pays
    /// the full TQ encode/decode cost before the caller can see token 1.
    /// Delaying TQ by one surfaced token preserves the sustained decode
    /// memory/throughput benefit while avoiding the misleading TTFT penalty.
    var shouldQuantizeAfterStep: Bool {
        guard needsCacheQuantization else { return false }
        if case .turboQuant = kvMode {
            return tokenCount > 0
        }
        return true
    }

    mutating func setupCompiledDecode(maxCacheLength: Int) throws {
        guard HardwareInfo.isCompiledDecodeSupported else { return }
        // Compiled decode requires no auxiliary state — models with state (e.g. vision
        // encoder cross-attention) use the uncompiled path.
        guard state == nil else { return }

        // Materialize all pending cache operations before conversion.
        eval(cache)

        let promoted: [KVCache]
        switch kvMode {
        case .turboQuant:
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: kvBits,
                kvGroupSize: kvGroupSize,
                quantizedKVStart: quantizedKVStart,
                kvMode: kvMode)
            guard cache.allSatisfy({
                ($0 as? TurboQuantKVCache)?.phase == .compressed
            }) else { return }
            promoted = cache.map { layer in
                CompilableTurboQuantKVCache(from: layer as! TurboQuantKVCache) as KVCache
            }
        case .affine:
            return
        case .none where kvBits != nil:
            return
        case .none:
            // KVCacheSimple -> CompilableKVCache (static buffer, graph-visible
            // offset). Plain KVCacheSimple reads `offset` as an Int, which compile
            // captures at trace-build time and then reuses for later tokens.
            guard cache.allSatisfy({ $0 is KVCacheSimple }) else { return }
            promoted = cache.map { layer in
                CompilableKVCache(from: layer, maxLength: maxCacheLength) as KVCache
            }
        }
        MLX.eval(promoted)
        self.cache = promoted

        let capturedModel = model
        let cacheRef = promoted

        self.compiledForward = compile(
            inputs: cacheRef, outputs: cacheRef
        ) { (args: [MLXArray]) -> [MLXArray] in
            let result = capturedModel(
                LMInput.Text(tokens: args[0])[text: .newAxis],
                cache: cacheRef.isEmpty ? nil : cacheRef,
                state: nil)
            return [result.logits]
        }
    }

    /// Evaluate the next token and return the new token (y), updating cache state
    mutating func step(previous: LMInput.Text) -> MLXArray {
        if self.compiledForward != nil {
            let input = previous.tokens
            let result = MLXPressGenerationProfile.time("decode.compiled_forward") {
                self.compiledForward!([input])
            }

            if result.count > 0 {
                self.state = nil
                if shouldQuantizeAfterStep {
                    MLXPressGenerationProfile.time("decode.kv_quantize") {
                        maybeQuantizeKVCache(
                            cache: &cache, kvBits: kvBits,
                            kvGroupSize: kvGroupSize, quantizedKVStart: quantizedKVStart,
                            kvMode: kvMode)
                    }
                }
                return MLXPressGenerationProfile.time("decode.sample") {
                    convertToToken(logits: result[0])
                }
            }
            self.compiledForward = nil
        }

        // Models expect [B, L] input. If the caller passed 1D tokens [L], add a batch
        // axis. If they passed 2D [B, L] already (some VLM bench/test paths), use as-is —
        // adding another newAxis would produce 3D and break QuantizedLinear matmul on
        // pure-LLM model paths (Llama, Mistral, Phi, etc).
        let stepInput: LMInput.Text =
            previous.tokens.ndim == 1 ? previous[text: .newAxis] : previous
        let result = MLXPressGenerationProfile.time("decode.model_forward") {
            model(stepInput, cache: cache.isEmpty ? nil : cache, state: state)
        }
        self.state = result.state

        if shouldQuantizeAfterStep {
            MLXPressGenerationProfile.time("decode.kv_quantize") {
                maybeQuantizeKVCache(
                    cache: &cache,
                    kvBits: kvBits,
                    kvGroupSize: kvGroupSize,
                    quantizedKVStart: quantizedKVStart,
                    kvMode: kvMode
                )
            }
        }

        return MLXPressGenerationProfile.time("decode.sample") {
            convertToToken(logits: result.logits)
        }
    }

    mutating public func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        let previousY = y

        let token = MLXPressGenerationProfile.time("decode.step_build") {
            step(previous: previousY)
        }
        y = .init(tokens: token)

        MLXPressGenerationProfile.time("decode.async_eval_submit") {
            asyncEval(token)
        }

        tokenCount += 1

        if tokenCount % 256 == 0 {
            Memory.clearCache()
        }

        return MLXPressGenerationProfile.time("decode.token_item_sync") {
            previousY.tokens.item(Int.self)
        }
    }

    public mutating func storeCacheAfterGeneration(
        generatedTokenIds: [Int],
        includeGeneratedBoundary: Bool
    ) {
        guard let coordinator = cacheCoordinator, !promptTokenIds.isEmpty else {
            return
        }

        func store(
            tokens: [Int],
            cache cacheToStore: [KVCache],
            kvBits diskKVBits: Int?,
            kvMode diskKVMode: KVQuantizationMode
        ) {
            guard !tokens.isEmpty else { return }
            let snapshot = cacheToStore.map { $0.copy() }
            MLX.eval(snapshot)
            let requiresDiskBackedRestore =
                cacheRequiresDiskBackedCoordinatorRestore(snapshot)
            let perLayerData = requiresDiskBackedRestore
                ? []
                : extractLayerData(from: snapshot)
            let ssmCapture: [MLXArray]? = coordinator.isHybrid &&
                coordinator.config.enableSSMReDerive &&
                !requiresDiskBackedRestore
                ? reDeriveAndStoreSSMStatesForPromptBoundaries(
                    coordinator: coordinator,
                    model: model,
                    promptTokenIds: tokens,
                    mediaSalt: mediaSalt)
                : (coordinator.isHybrid ? extractSSMStates(from: snapshot) : nil)
            let diskStoreCache = makeDiskStoreCache(
                fromPromptBoundary: snapshot,
                kvBits: diskKVBits,
                kvGroupSize: kvGroupSize,
                quantizedKVStart: quantizedKVStart,
                kvMode: diskKVMode)
            coordinator.storeAfterGeneration(
                promptTokens: tokens,
                perLayerData: perLayerData,
                ssmStates: ssmCapture,
                cache: diskStoreCache,
                mediaSalt: mediaSalt
            )
        }

        if let promptCacheSnapshot {
            // Prompt-boundary disk entries must remain raw KV even when the
            // live decode path uses TurboQuant/affine KV. Cold decode delays
            // lossy KV compression until after the first surfaced token; a
            // warm full-prefix hit must therefore seed first-token sampling
            // from the same exact prompt KV, not from a compressed prompt.
            store(
                tokens: promptTokenIds,
                cache: promptCacheSnapshot,
                kvBits: nil,
                kvMode: .none)

            for boundary in Set(cachePrefixTokenCounts).sorted()
            where boundary > 0 && boundary < promptTokenIds.count {
                let boundaryTokens = Array(promptTokenIds.prefix(boundary))
                if let boundarySnapshot = cacheSnapshotForBoundary(
                    tokens: boundaryTokens,
                    promptSnapshot: promptCacheSnapshot)
                {
                    store(
                        tokens: boundaryTokens,
                        cache: boundarySnapshot,
                        kvBits: nil,
                        kvMode: .none)
                }
            }
        }

        guard includeGeneratedBoundary, !generatedTokenIds.isEmpty else { return }
        guard !needsCacheQuantization else { return }
        let generatedBoundaryTokens = promptTokenIds + generatedTokenIds
        guard (cache.map(\.offset).max() ?? 0) >= generatedBoundaryTokens.count
        else { return }
        store(tokens: generatedBoundaryTokens, cache: cache, kvBits: kvBits, kvMode: kvMode)
    }

    private func cacheSnapshotForBoundary(
        tokens: [Int],
        promptSnapshot: [KVCache]
    ) -> [KVCache]? {
        guard !tokens.isEmpty, tokens.count < promptTokenIds.count else {
            return nil
        }
        let trimCount = promptTokenIds.count - tokens.count
        let trimmed = promptSnapshot.map { $0.copy() }
        if canTrimPromptCache(trimmed),
           trimPromptCache(trimmed, numTokens: trimCount) == trimCount
        {
            MLX.eval(trimmed)
            return trimmed
        }

        do {
            let boundaryTokens = MLXArray(tokens.map { Int32($0) })
                .reshaped(1, tokens.count)
            let boundaryInput = LMInput(
                text: LMInput.Text(tokens: boundaryTokens),
                image: originalInput.image,
                video: originalInput.video,
                audio: originalInput.audio,
                mediaTokenIds: originalInput.mediaTokenIds,
                cacheScopeSalt: originalInput.cacheScopeSalt)
            let cache = model.newCache(parameters: cacheInitParameters)
            switch try model.prepare(
                boundaryInput,
                cache: cache,
                windowSize: effectivePrefillWindow(
                    requested: promptTokenIds.count,
                    input: boundaryInput))
            {
            case .tokens(let remaining):
                // Keep the solo TokenIterator rederive path aligned with
                // normal prefill/decode: models expect batch-first tokens.
                // ZAYA CCA reaches a 2D activation and traps if this helper
                // feeds the 1D `remaining` tensor directly.
                _ = model(
                    remaining[text: .newAxis],
                    cache: cache,
                    state: nil)
            case .logits:
                break
            }
            MLX.eval(cache)
            return cache
        } catch {
            Self.logger.debug(
                "TokenIterator: skipped history-boundary cache rederive: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
    }
}

/// Generator of tokens using speculative decoding.
///
/// This is typically used via a call to ``generate(input:parameters:context:draftModel:draftCache:numDraftTokens:wiredMemoryTicket:)``
/// returning `AsyncStream<Generation>`.
///
/// To use it directly:
///
/// ```swift
/// let generateParameters: GenerateParameters
/// let input: LMInput
/// let mainModel: LanguageModel
/// let draftModel: LanguageModel
///
/// let iterator = try SpeculativeTokenIterator(
///     input: input, mainModel: mainModel, draftModel: draftModel,
///     parameters: generateParameters, numDraftTokens: 2)
///
/// for token in iterator {
///     ...
/// }
/// ```
///
/// Tokens are integers that can be passed through a `Tokenizer` or ``StreamingDetokenizer`` to produce Strings.
///
/// Port of `speculative_generate_step()` from https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py
public struct SpeculativeTokenIterator: TokenIteratorProtocol {

    var y: LMInput.Text
    var draftY: LMInput.Text

    let mainModel: any LanguageModel
    let draftModel: any LanguageModel

    var mainState: LMOutput.State?
    var mainCache: [KVCache]
    var draftCache: [KVCache]
    let quantizeKVCache: (inout [KVCache]) -> Void

    var processor: LogitProcessor?
    let sampler: LogitSampler

    public var tokenCount = 0
    public let maxTokens: Int?
    let numDraftTokens: Int

    // Buffer of accepted tokens from the current speculation round
    private var pendingTokens = [Int]()
    private var pendingIndex = 0

    // Internal metrics
    public var promptPrefillTime: TimeInterval = 0.0

    /// Initialize a `SpeculativeTokenIterator` with the given input.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - mainModel: the main (verifier) ``LanguageModel``
    ///   - draftModel: the draft ``LanguageModel`` (must share the same tokenizer)
    ///   - mainCache: optional ``KVCache`` for the main model
    ///   - draftCache: optional ``KVCache`` for the draft model
    ///   - parameters: the generation parameters
    ///   - numDraftTokens: number of tokens the draft model proposes per round
    public init(
        input: LMInput,
        mainModel: any LanguageModel,
        draftModel: any LanguageModel,
        mainCache: [KVCache]? = nil,
        draftCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        numDraftTokens: Int
    ) throws {
        _ = try AccelerationRuntime.resolveTextDecode(parameters.accelerationMode)

        self.y = input.text
        self.draftY = input.text
        self.mainModel = mainModel
        self.draftModel = draftModel

        self.mainCache = mainCache ?? mainModel.newCache(parameters: parameters)
        self.draftCache = draftCache ?? draftModel.newCache(parameters: parameters)
        guard canTrimPromptCache(self.mainCache), canTrimPromptCache(self.draftCache) else {
            throw KVCacheError(message: "Speculative decoding requires trimmable KV caches.")
        }

        self.sampler = parameters.sampler()
        self.processor = parameters.processor()

        self.maxTokens = parameters.maxTokens
        self.numDraftTokens = numDraftTokens

        self.quantizeKVCache = { cache in
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart,
                kvMode: parameters.kvMode
            )
        }

        self.promptPrefillTime = try measure {
            try prepare(input: input, windowSize: parameters.prefillStepSize)
        }
    }

    /// Prefill both main and draft models with the prompt, priming caches for generation
    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        processor?.prompt(input.text.tokens)

        // Prefill main model
        switch try mainModel.prepare(input, cache: mainCache, windowSize: windowSize) {
        case .tokens(let tokens):
            y = tokens
        case .logits(let result):
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            y = .init(tokens: token)
            mainState = result.state
        }

        // Prefill draft model, don't call didSample here -- processor tracks main model's accepted sequence only
        switch try draftModel.prepare(input, cache: draftCache, windowSize: windowSize) {
        case .tokens(let tokens):
            draftY = tokens
        case .logits(let result):
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            draftY = .init(tokens: token)
            asyncEval(draftY.tokens)
        }
    }

    /// Run one round of speculative decoding: draft, verify, accept/reject
    mutating func speculateRound() {
        let remaining = maxTokens.map { $0 - tokenCount } ?? numDraftTokens
        let numDraft = Swift.min(remaining, numDraftTokens)
        guard numDraft > 0 else {
            return
        }

        // Draft generation: autoregressive loop with draft model
        var draftProcessor = processor  // Copy to discard later
        var draftTokens = [MLXArray]()
        for _ in 0 ..< numDraft {
            let draftResult = draftModel(draftY[text: .newAxis], cache: draftCache, state: nil)
            var draftLogits = draftResult.logits[0..., -1, 0...]
            draftLogits = draftProcessor?.process(logits: draftLogits) ?? draftLogits
            let draftToken = sampler.sample(logits: draftLogits)
            draftProcessor?.didSample(token: draftToken)
            asyncEval(draftToken)
            draftTokens.append(draftToken)
            draftY = .init(tokens: draftToken)
        }

        // Verification: main model processes proposals in one pass
        let verifyTokens = [y.tokens] + draftTokens
        let verifyInput = LMInput.Text(tokens: concatenated(verifyTokens))
        let verifyStart = verifyInput.tokens.dim(0) - (numDraft + 1)
        let mainResult = mainModel(verifyInput[text: .newAxis], cache: mainCache, state: mainState)
        let mainLogits = mainResult.logits
        mainState = mainResult.state

        let mainTokens: MLXArray
        if var verifyProcessor = processor {
            // Process each position sequentially so that the processor sees tokens sampled at earlier positions
            var sampled = [MLXArray]()
            for i in 0 ..< (numDraft + 1) {
                var logits = mainLogits[0..., verifyStart + i, 0...]
                logits = verifyProcessor.process(logits: logits)
                let token = sampler.sample(logits: logits)
                verifyProcessor.didSample(token: token)
                sampled.append(token)
            }
            mainTokens = concatenated(sampled)
        } else {
            // Batch-sample all verify tokens from main model in one operation
            let verifyLogits = mainLogits[0..., verifyStart..., 0...].squeezed(axis: 0)
            mainTokens = sampler.sample(logits: verifyLogits)
        }

        // Compare and accept proposed tokens
        eval(mainTokens, draftTokens)
        let mainTokensList = mainTokens.asArray(Int.self)
        let draftTokensList = concatenated(draftTokens).asArray(Int.self)
        var accepted = 0
        for i in 0 ..< numDraft {
            guard mainTokensList[i] == draftTokensList[i] else {
                break
            }

            processor?.didSample(token: draftTokens[i])
            pendingTokens.append(mainTokensList[i])
            accepted += 1
        }

        // Always emit the main model's token at position `accepted`
        // (either the correction token or the bonus token if all drafts matched)
        let finalToken = mainTokens[accepted ... accepted]
        processor?.didSample(token: finalToken)
        pendingTokens.append(mainTokensList[accepted])

        // Rewind caches for rejected tokens
        trimPromptCache(mainCache, numTokens: numDraft - accepted)
        trimPromptCache(draftCache, numTokens: Swift.max(numDraft - accepted - 1, 0))

        // Apply dynamic cache quantization after rewind
        quantizeKVCache(&mainCache)
        quantizeKVCache(&draftCache)

        // Set y/draftY for the next round
        y = .init(tokens: finalToken)
        draftY = .init(tokens: finalToken)

        // If all draft tokens were accepted, the draft model hasn't processed
        // the last accepted draft token yet. Feed it through to keep caches in sync.
        if accepted == numDraft {
            draftY = .init(
                tokens: concatenated([
                    draftTokens[numDraft - 1].reshaped([1]),
                    finalToken,
                ])
            )
        }
    }

    mutating public func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        // Drain the pending buffer first
        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            pendingIndex += 1
            tokenCount += 1
            return token
        }

        // Run a new speculation round
        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        speculateRound()

        if pendingTokens.isEmpty {
            return nil
        }

        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }
}

/// Result of a call to a deprecated callback-based generate function.
public struct GenerateResult {

    /// Initializes a new `GenerateResult` instance.
    ///
    /// - Parameters:
    ///   - inputText: The input text used for generation.
    ///   - tokenIds: The array of generated token IDs.
    ///   - output: The generated output string.
    ///   - promptTime: The time taken to prompt the input.
    ///   - generateTime: The time taken to generate the output.
    public init(
        inputText: LMInput.Text, tokenIds: [Int], output: String, promptTime: TimeInterval,
        generateTime: TimeInterval
    ) {
        self.inputText = inputText
        self.tokenIds = tokenIds
        self.output = output
        self.promptTime = promptTime
        self.generateTime = generateTime
    }

    @available(*, deprecated, renamed: "init(inputText:tokenIds:output:promptTime:generateTime:)")
    public init(
        inputText: LMInput.Text, tokens: [Int], output: String, promptTime: TimeInterval,
        generateTime: TimeInterval
    ) {
        self.init(
            inputText: inputText, tokenIds: tokens, output: output, promptTime: promptTime,
            generateTime: generateTime)
    }

    /// input (prompt, images, etc.)
    public let inputText: LMInput.Text

    /// The token IDs of the input prompt.
    public var promptTokenIds: [Int] {
        inputText.tokens.asArray(Int.self)
    }

    @available(*, deprecated, renamed: "promptTokenIds")
    public var promptTokens: [Int] { promptTokenIds }

    /// Generated token IDs
    public let tokenIds: [Int]

    @available(*, deprecated, renamed: "tokenIds")
    public var tokens: [Int] { tokenIds }

    /// Output text
    public let output: String

    /// The number of tokens included in the input prompt.
    public var promptTokenCount: Int { inputText.tokens.size }

    /// The number of tokens generated by the language model.
    public var generationTokenCount: Int { tokenIds.count }

    /// Time to process the prompt (generate the first token)
    public let promptTime: TimeInterval

    /// Time to generate the remaining tokens
    public let generateTime: TimeInterval

    /// The number of tokens processed per second during the prompt phase.
    public var promptTokensPerSecond: Double {
        Double(inputText.tokens.size) / promptTime
    }

    /// The number of tokens generated per second during the generation phase.
    public var tokensPerSecond: Double {
        Double(tokenIds.count) / generateTime
    }

    public func summary() -> String {
        """
        Prompt:     \(promptTokenCount) tokens, \(promptTokensPerSecond.formatted()) tokens/s, \(promptTime.formatted())s
        Generation: \(generationTokenCount) tokens, \(tokensPerSecond.formatted()) tokens/s, \(generateTime.formatted())s
        """
    }
}

/// Action from token visitor callback in deprecated callback-based generate functions.
public enum GenerateDisposition: Sendable {
    /// Keep producing tokens until an EOS token is produced
    case more

    /// Stop producing tokens, e.g. a token limit has been hit
    case stop
}

private struct SynchronousGenerationLoopResult {
    let generatedTokenIds: [Int]
    let promptTime: TimeInterval
    let generateTime: TimeInterval
    let promptPrefillTime: TimeInterval
    let stopReason: GenerateStopReason
}

private func buildStopTokenIds(
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer
) -> Set<Int> {
    // Build complete EOS token set from all sources.
    var stopTokenIds = modelConfiguration.eosTokenIds
    if let tokenizerEOS = tokenizer.eosTokenId {
        stopTokenIds.insert(tokenizerEOS)
    }
    for token in modelConfiguration.extraEOSTokens {
        if let id = tokenizer.convertTokenToId(token) {
            stopTokenIds.insert(id)
        }
    }
    // Match BatchEngine's defensive end-of-turn widening so the
    // single-request generate() path cannot leak a chat-template turn
    // terminator when a bundle's config only lists a partial EOS set.
    let commonEndTokens = [
        "<|im_end|>",
        "<|endoftext|>",
        "<|eot_id|>",
        "<|end_of_text|>",
        "<|end|>",
        "<|end_of_turn|>",
        "<end_of_turn>",
        "〈|EOS|〉",
        "<｜end▁of▁sentence｜>",
        "<|EOT|>",
    ]
    for token in commonEndTokens {
        if let id = tokenizer.convertTokenToId(token) {
            stopTokenIds.insert(id)
        }
    }
    return stopTokenIds
}

private func runSynchronousGenerationLoop(
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: TokenIterator,
    didGenerate: (_ token: Int, _ generatedTokenIds: [Int]) -> GenerateDisposition
) -> SynchronousGenerationLoopResult {
    var start = Date.timeIntervalSinceReferenceDate
    var promptTime: TimeInterval = 0

    let stopTokenIds = buildStopTokenIds(
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer
    )

    var generatedTokenIds = [Int]()
    var iterator = iterator
    var stopReason: GenerateStopReason?

    while let token = iterator.next() {
        // Compute the timing for the prompt.
        if promptTime == 0 {
            let now = Date.timeIntervalSinceReferenceDate
            promptTime = now - start
            start = now
        }

        // Check for end-of-sequence tokens.
        if token == tokenizer.unknownTokenId || stopTokenIds.contains(token) {
            stopReason = .stop
            break
        }

        generatedTokenIds.append(token)

        if didGenerate(token, generatedTokenIds) == .stop {
            stopReason = .cancelled
            break
        }
    }

    // If the iterator ends naturally, the max-token limit was reached.
    if stopReason == nil {
        if let maxTokens = iterator.maxTokens, iterator.tokenCount >= maxTokens {
            stopReason = .length
        } else {
            stopReason = .cancelled
        }
    }

    let now = Date.timeIntervalSinceReferenceDate
    let generateTime = now - start

    Stream().synchronize()

    return SynchronousGenerationLoopResult(
        generatedTokenIds: generatedTokenIds,
        promptTime: promptTime,
        generateTime: generateTime,
        promptPrefillTime: iterator.promptPrefillTime,
        stopReason: stopReason ?? .cancelled
    )
}

/// Given prompt tokens generate text using the given model and parameters.
///
/// ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` is the preferred call.
///
/// - Parameters:
///   - promptTokens: tokenized prompt
///   - parameters: generation parameters
///   - model: model to evaluate
///   - tokenizer: tokenizer to convert tokens back into strings and recognize special tokens
///   - extraEOSTokens: any additional stop tokens
///   - didGenerate: visitor for the tokens as they are generated
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    promptTokens: [Int], parameters: GenerateParameters, model: any LanguageModel,
    tokenizer: Tokenizer,
    extraEOSTokens: Set<String>? = nil,
    didGenerate: ([Int]) -> GenerateDisposition
) throws -> GenerateResult {
    let tokens = MLXArray(promptTokens)
    let iterator = try TokenIterator(
        prompt: tokens, model: model, parameters: parameters)

    // this is a compatibility cover -- create the required values
    // for the iteration
    let input = LMInput(tokens: tokens)
    let configuration = ModelConfiguration(id: "stand-in", extraEOSTokens: extraEOSTokens ?? [])
    let context = ModelContext(
        configuration: configuration, model: model, processor: StandInUserInputProcessor(),
        tokenizer: tokenizer)

    return generate(
        input: input, context: context, iterator: iterator,
        didGenerate: didGenerate)
}

/// Generate tokens from an ``LMInput`` and a ``ModelContext``.
///
/// Prefer using ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` instead.
///
/// - Parameters:
///   - input: prepared language model input
///   - parameters: parameters controlling the token generation
///   - context: model context (model and tokenizer)
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: the generated output
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, parameters: GenerateParameters, context: ModelContext,
    didGenerate: ([Int]) -> GenerateDisposition
) throws -> GenerateResult {
    let iterator = try TokenIterator(
        input: input, model: context.model, parameters: parameters)
    return generate(
        input: input, context: context, iterator: iterator,
        didGenerate: didGenerate)
}

/// Low-level token generation using a ``TokenIterator``.
///
/// ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` is the preferred call.
///
/// - Parameters:
///   - input: prepared language model input
///   - context: model context (model and tokenizer)
///   - iterator: token iterator
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: the generated output
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    didGenerate: ([Int]) -> GenerateDisposition
) -> GenerateResult {
    let result = runSynchronousGenerationLoop(
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator
    ) { _, generatedTokens in
        didGenerate(generatedTokens)
    }

    return GenerateResult(
        inputText: input.text, tokenIds: result.generatedTokenIds,
        output: context.tokenizer.decode(tokenIds: result.generatedTokenIds),
        promptTime: result.promptTime + result.promptPrefillTime,
        generateTime: result.generateTime
    )
}

/// Generate tokens from an ``LMInput`` and a ``ModelContext``.
///
/// Prefer using ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` instead.
///
/// - Parameters:
///   - input: prepared language model input
///   - parameters: parameters controlling the token generation
///   - context: model context (model and tokenizer)
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: Information about the generation
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, parameters: GenerateParameters, context: ModelContext,
    didGenerate: (Int) -> GenerateDisposition
) throws -> GenerateCompletionInfo {
    let iterator = try TokenIterator(
        input: input, model: context.model, parameters: parameters)
    return generate(
        input: input, context: context, iterator: iterator,
        didGenerate: didGenerate)
}

/// Low-level token generation using a ``TokenIterator``.
///
/// ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` is the preferred call.
///
/// - Parameters:
///   - input: prepared language model input
///   - context: model context (model and tokenizer)
///   - iterator: token iterator
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: Information about the generation
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    didGenerate: (Int) -> GenerateDisposition
) -> GenerateCompletionInfo {
    let result = runSynchronousGenerationLoop(
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator
    ) { token, _ in
        didGenerate(token)
    }

    return GenerateCompletionInfo(
        promptTokenCount: input.text.tokens.size,
        generationTokenCount: result.generatedTokenIds.count,
        promptTime: result.promptTime + result.promptPrefillTime,
        generationTime: result.generateTime,
        stopReason: result.stopReason
    )
}

/// Generates tokens asynchronously using the provided language model input, parameters, and context.
///
/// This function initializes a `TokenIterator` with the given input, model, and generation parameters,
/// and then streams the token generation process via an `AsyncStream`. The resulting stream yields
/// instances of the `Generation` enum, which can represent text chunks, tool calls, or summary
/// completion information.
///
/// * Important: if the stream is terminated early (e.g. break from the loop) computation will continue
/// using the model, parameters, KVCache, etc. for some time (typically a few ms).  This is typically OK for
/// one-shot calls, but for "chat session" type calls consider using
/// ``generateTask(promptTokenCount:modelConfiguration:tokenizer:iterator:)``
/// so that the end of the generation task can be observed.
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache``
///   - parameters: The configuration options for token generation.
///   - context: The model context, including the model itself and associated tokenizer.
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination across
///     concurrent tasks. This is opt-in and only applied on GPU devices that support wired
///     memory control (macOS 15 / iOS 18 / tvOS 18 or newer).
/// - Returns: An `AsyncStream` that emits `Generation` values, including generated text chunks (`.chunk`),
///   tool calls (`.toolCall`), and completion information (`.info`).
/// - Throws: An error if the `TokenIterator` initialization fails due to invalid input or model configuration.
///
/// ### Example Usage:
/// ```swift
/// // Define the input, parameters, and context for token generation.
/// let generateParameters: GenerateParameters
/// let input: UserInput
/// let context: ModelContext
///
/// let lmInput = try context.processor.prepare(input: input)
///
/// // Call the generate function to get an AsyncStream.
/// let stream = try generate(input: lmInput, parameters: generateParameters, context: context)
///
/// // Process the stream asynchronously to handle text chunks and completion info.
/// for await generation in stream {
///     switch generation {
///     case .chunk(let text):
///         print("Generated text: \(text)")
///     case .info(let info):
///         print("Finished: \(info.tokensPerSecond) tokens/s.")
///     case .toolCall(let call):
///         print("Tool call: \(call.function.name)")
///     }
/// }
/// ```
public func generate(
    input: LMInput, cache: [KVCache]? = nil, parameters: GenerateParameters, context: ModelContext,
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    cacheCoordinator: CacheCoordinator? = nil
) throws -> AsyncStream<Generation> {
    _ = try AccelerationRuntime.resolveTextDecode(parameters.accelerationMode)

    context.jangPressRuntime.recordPromptTokenActivity(
        input.text.tokens.reshaped(-1).asArray(Int.self))

    let promptTail = _decodePromptTail(
        input: input, tokenizer: context.tokenizer, tokens: 64)
    if let strategy = parameters.draftStrategy,
        case .nativeMTP(let depth) = strategy
    {
        guard let nativeModel = context.model as? any NativeMTPModel else {
            throw NativeMTPRuntimeError.modelDoesNotExposeNativeMTP
        }
        let iterator = try NativeMTPTokenIterator(
            input: input,
            model: nativeModel,
            cache: cache,
            parameters: parameters,
            depth: depth,
            cacheCoordinator: cacheCoordinator)
        let (stream, _) = generateTask(
            promptTokenCount: input.text.tokens.size,
            modelConfiguration: context.configuration,
            tokenizer: context.tokenizer,
            iterator: iterator,
            wiredMemoryTicket: wiredMemoryTicket,
            extraStopStrings: parameters.extraStopStrings,
            promptTail: promptTail)
        return stream
    }
    // Block-diffusion speculative decoding dispatch. When
    // parameters.draftStrategy is .dflash or .ddtree AND the target
    // model conforms to HiddenStateCaptureModel + TokenEmbedderModel,
    // route through SpecDecStream. Zero API churn for callers using
    // .none / nil / .autoregressive — those fall through to the
    // existing TokenIterator path below.
    if let strategy = parameters.draftStrategy,
        strategy.usesBlockDiffusion,
        let stream = SpecDecStream.streamViaStrategy(
            strategy: strategy,
            inputIds: input.text.tokens,
            context: context,
            maxNewTokens: parameters.maxTokens ?? 256,
            stopTokenIDs: [],
            temperature: parameters.temperature)
    {
        return stream
    }
    let iterator = try TokenIterator(
        input: input, model: context.model, cache: cache, parameters: parameters,
        cacheCoordinator: cacheCoordinator)
    let (stream, _) = generateTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        extraStopStrings: parameters.extraStopStrings,
        promptTail: promptTail)
    return stream
}

/// Generates text and tool calls asynchronously using speculative decoding with a draft model.
///
/// This function uses a smaller draft model to propose tokens that are verified in batch
/// by the main model, potentially accelerating generation. The resulting stream yields
/// decoded text chunks, tool calls, and completion information. It has the same output as the
/// non-speculative ``generate(input:cache:parameters:context:wiredMemoryTicket:)``.
///
/// Both models must share the same tokenizer.
///
/// ### Example Usage:
/// ```swift
/// let generateParameters: GenerateParameters
/// let input: UserInput
/// let mainContext: ModelContext
/// let draftModel: LanguageModel
///
/// let lmInput = try mainContext.processor.prepare(input: input)
///
/// let stream = try generate(
///     input: lmInput, parameters: generateParameters,
///     context: mainContext, draftModel: draftModel)
///
/// for await generation in stream {
///     switch generation {
///     case .chunk(let text):
///         print("Generated text: \(text)")
///     case .info(let info):
///         print("Finished: \(info.tokensPerSecond) tokens/s.")
///     case .toolCall(let call):
///         print("Tool call: \(call.function.name)")
///     }
/// }
/// ```
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache`` for the main model.
///   - parameters: The configuration options for token generation.
///   - context: The model context for the main (verifier) model.
///   - draftModel: The draft ``LanguageModel`` for speculative token proposals.
///   - draftCache: optional ``KVCache`` for the draft model.
///   - numDraftTokens: Number of tokens the draft model proposes per round (default: 2).
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination.
/// - Returns: An `AsyncStream` that emits `Generation` values.
/// - Throws: An error if the iterator initialization fails.
public func generate(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    draftModel: any LanguageModel,
    draftCache: [KVCache]? = nil,
    numDraftTokens: Int = 2,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<Generation> {
    context.jangPressRuntime.recordPromptTokenActivity(
        input.text.tokens.reshaped(-1).asArray(Int.self))

    let iterator = try SpeculativeTokenIterator(
        input: input,
        mainModel: context.model,
        draftModel: draftModel,
        mainCache: cache,
        draftCache: draftCache,
        parameters: parameters,
        numDraftTokens: numDraftTokens
    )
    let (stream, _) = generateLoopTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        handler: TextToolTokenLoopHandler(
            tokenizer: context.tokenizer,
            format: context.configuration.toolCallFormat ?? .json,
            reasoningParser: ReasoningParser.forPrompt(
                stampName: context.configuration.reasoningParserName,
                promptTail: _decodePromptTail(
                    input: input, tokenizer: context.tokenizer, tokens: 64)),
            stopStringMatcher: StopStringMatcher(
                stopStrings: parameters.extraStopStrings)
        )
    )
    return stream
}

@available(
    *, deprecated,
    message: "use a higher level generate() call or use generateTask() for fine grained control"
)
public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) -> AsyncStream<Generation> {
    let (stream, _) = generateTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket)
    return stream
}

/// Low-level token generation using a ``TokenIterator``, returning an
/// `AsyncStream<Generation>` and a `Task`.
///
/// * Important: if the stream is terminated early (e.g. break from the loop) computation will continue
/// using the model, parameters, KVCache, etc. for some time (typically a few ms).  Callers can await
/// the `task` to observe when the use of the parameters is complete.
///
/// - Parameters:
///   - promptTokenCount: number of tokens in the prompt
///   - modelConfiguration: model configuration (for EOS/extra EOS tokens and tool-call format)
///   - tokenizer: tokenizer (for EOS id, unknown token id, and detokenization)
///   - iterator: token iterator
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination.
/// - Returns: An `AsyncStream` that emits `Generation` values and a `Task`
public func generateTask(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming any TokenIteratorProtocol,
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    extraStopStrings: [String] = [],
    promptTail: String? = nil
) -> (AsyncStream<Generation>, Task<Void, Never>) {
    let effectivePromptTail =
        promptTail
        ?? _decodePromptTail(
            tokenIds: iterator.promptTokenIds, tokenizer: tokenizer, tokens: 64)

    return generateLoopTask(
        promptTokenCount: promptTokenCount,
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        handler: TextToolTokenLoopHandler(
            tokenizer: tokenizer,
            format: modelConfiguration.toolCallFormat ?? .json,
            reasoningParser: ReasoningParser.forPrompt(
                stampName: modelConfiguration.reasoningParserName,
                promptTail: effectivePromptTail),
            stopStringMatcher: StopStringMatcher(stopStrings: extraStopStrings)
        )
    )
}

/// Generates raw token IDs asynchronously using the provided language model input, parameters, and context.
///
/// This is similar to `generate(input:cache:parameters:context:)`, but yields raw token IDs instead of decoded text/tool calls.
/// This is useful for downstream parsers that need access to token IDs directly (e.g. Harmony parsing).
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache``
///   - parameters: The configuration options for token generation.
///   - context: The model context, including the model itself and associated tokenizer.
///   - includeStopToken: when true, the terminating EOS/unknown token is yielded before finishing
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination across
///     concurrent tasks. This is opt-in and only applied on GPU devices that support wired
///     memory control (macOS 15 / iOS 18 / tvOS 18 or newer).
///   - cacheCoordinator: Optional multi-tier cache coordinator for prefix reuse.
/// - Returns: An `AsyncStream` that emits `TokenGeneration` values.
public func generateTokens(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    includeStopToken: Bool = false,
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    cacheCoordinator: CacheCoordinator? = nil
) throws -> AsyncStream<TokenGeneration> {
    context.jangPressRuntime.recordPromptTokenActivity(
        input.text.tokens.reshaped(-1).asArray(Int.self))

    let iterator = try TokenIterator(
        input: input, model: context.model, cache: cache, parameters: parameters,
        cacheCoordinator: cacheCoordinator)
    let (stream, _) = generateTokenTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        includeStopToken: includeStopToken,
        wiredMemoryTicket: wiredMemoryTicket
    )
    return stream
}

/// Generates raw token IDs asynchronously using speculative decoding with a draft model.
///
/// This is similar to `generate(input:parameters:context:draftModel:draftCache:numDraftTokens:wiredMemoryTicket:)`,
/// but yields raw token IDs instead of decoded text/tool calls.
///
/// Both models must share the same tokenizer.
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache`` for the main model.
///   - parameters: The configuration options for token generation.
///   - context: The model context for the main (verifier) model.
///   - draftModel: The draft ``LanguageModel`` for speculative token proposals.
///   - draftCache: optional ``KVCache`` for the draft model.
///   - numDraftTokens: Number of tokens the draft model proposes per round (default: 2).
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination.
/// - Returns: An `AsyncStream` that emits `TokenGeneration` values.
/// - Throws: An error if the iterator initialization fails.
public func generateTokens(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    draftModel: any LanguageModel,
    draftCache: [KVCache]? = nil,
    numDraftTokens: Int = 2,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<TokenGeneration> {
    context.jangPressRuntime.recordPromptTokenActivity(
        input.text.tokens.reshaped(-1).asArray(Int.self))

    let iterator = try SpeculativeTokenIterator(
        input: input,
        mainModel: context.model,
        draftModel: draftModel,
        mainCache: cache,
        draftCache: draftCache,
        parameters: parameters,
        numDraftTokens: numDraftTokens
    )
    let (stream, _) = generateLoopTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        handler: RawTokenLoopHandler()
    )
    return stream
}

/// Generates raw token IDs asynchronously and returns the stream plus a `Task`.
///
/// Prefer this overload if you want to be able to observe when the underlying generation work is finished
/// (especially if the consumer terminates the stream early).
///
/// - Returns: An `AsyncStream` that emits `TokenGeneration` values and a `Task`.
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache``
///   - parameters: The configuration options for token generation.
///   - context: The model context, including the model itself and associated tokenizer.
///   - includeStopToken: when true, the terminating EOS/unknown token is yielded before finishing
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination across
///     concurrent tasks. This is opt-in and only applied on GPU devices that support wired
///     memory control (macOS 15 / iOS 18 / tvOS 18 or newer).
///   - cacheCoordinator: Optional multi-tier cache coordinator for prefix reuse.
public func generateTokensTask(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    includeStopToken: Bool = false,
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    cacheCoordinator: CacheCoordinator? = nil
) throws -> (AsyncStream<TokenGeneration>, Task<Void, Never>) {
    context.jangPressRuntime.recordPromptTokenActivity(
        input.text.tokens.reshaped(-1).asArray(Int.self))

    if let strategy = parameters.draftStrategy,
        case .nativeMTP(let depth) = strategy
    {
        guard let nativeModel = context.model as? any NativeMTPModel else {
            throw NativeMTPRuntimeError.modelDoesNotExposeNativeMTP
        }
        let iterator = try NativeMTPTokenIterator(
            input: input,
            model: nativeModel,
            cache: cache,
            parameters: parameters,
            depth: depth,
            cacheCoordinator: cacheCoordinator)
        return generateTokenTask(
            promptTokenCount: input.text.tokens.size,
            modelConfiguration: context.configuration,
            tokenizer: context.tokenizer,
            iterator: iterator,
            includeStopToken: includeStopToken,
            wiredMemoryTicket: wiredMemoryTicket)
    }

    let iterator = try TokenIterator(
        input: input, model: context.model, cache: cache, parameters: parameters,
        cacheCoordinator: cacheCoordinator)
    return generateTokenTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        includeStopToken: includeStopToken,
        wiredMemoryTicket: wiredMemoryTicket
    )
}

/// Low-level raw token generation using a `TokenIterator`, returning an
/// `AsyncStream<TokenGeneration>` and a `Task`.
///
/// This is useful for parsers that need access to the token IDs directly (e.g. Harmony parsing)
/// without detokenization or tool-call parsing.
///
/// - Parameters:
///   - promptTokenCount: number of tokens in the prompt
///   - modelConfiguration: model configuration (for EOS/extra EOS tokens)
///   - tokenizer: tokenizer (for EOS id and unknown token id)
///   - iterator: token iterator
///   - includeStopToken: when true, the terminating EOS/unknown token is yielded before finishing
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination across
///     concurrent tasks. This is opt-in and only applied on GPU devices that support wired
///     memory control (macOS 15 / iOS 18 / tvOS 18 or newer).
/// - Returns: An `AsyncStream` that emits token IDs and a final `.info`, plus a `Task`.
public func generateTokenTask(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming any TokenIteratorProtocol,
    includeStopToken: Bool = false,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) -> (AsyncStream<TokenGeneration>, Task<Void, Never>) {
    return generateLoopTask(
        promptTokenCount: promptTokenCount,
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        includeStopToken: includeStopToken,
        handler: RawTokenLoopHandler()
    )
}

private func generateLoopTask<Handler: TokenLoopHandler>(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming any TokenIteratorProtocol,
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    includeStopToken: Bool = false,
    handler: consuming Handler
) -> (AsyncStream<Handler.Output>, Task<Void, Never>) {

    let (stream, continuation) = AsyncStream<Handler.Output>.makeStream()

    let iterator = SendableBox(iterator)
    let handler = SendableBox(handler)

    // Launch a Task to perform iteration asynchronously.
    let task = Task {
        let performIteration = {
            var iterator = iterator.consume()
            var handler = handler.consume()

            var start = Date.timeIntervalSinceReferenceDate
            var promptTime: TimeInterval = 0
            var tokenCount = 0
            var generatedTokenIds: [Int] = []
            var stopReason: GenerateStopReason?

            let stopTokenIds = buildStopTokenIds(
                modelConfiguration: modelConfiguration,
                tokenizer: tokenizer
            )

            while let token = iterator.next() {
                // Check for cancellation on every loop iteration.
                if Task.isCancelled {
                    stopReason = .cancelled
                    break
                }

                if promptTime == 0 {
                    let now = Date.timeIntervalSinceReferenceDate
                    promptTime = now - start
                    start = now
                }

                // Check for end-of-sequence tokens
                if token == tokenizer.unknownTokenId || stopTokenIds.contains(token) {
                    if includeStopToken {
                        tokenCount += 1
                        if !handler.onStopToken(token, emit: continuation.yield) {
                            stopReason = .cancelled
                            break
                        }
                    }
                    stopReason = .stop
                    break
                }

                tokenCount += 1
                if !handler.onToken(token, emit: continuation.yield) {
                    // Distinguish "downstream consumer terminated the
                    // stream" from "library-internal stop-sequence
                    // match" — the latter should report `stopReason =
                    // .stop`, not `.cancelled`.
                    stopReason = handler.stopSequenceHit ? .stop : .cancelled
                    break
                }
                generatedTokenIds.append(token)
            }

            if stopReason == nil {
                if Task.isCancelled {
                    stopReason = .cancelled
                } else if let maxTokens = iterator.maxTokens, tokenCount >= maxTokens {
                    stopReason = .length
                } else {
                    stopReason = .cancelled
                }
            }

            let unclosedReasoning = handler.unclosedReasoning
            handler.onGenerationEnd(emit: continuation.yield)

            let now = Date.timeIntervalSinceReferenceDate
            let generateTime = now - start
            MLXPressGenerationProfile.dumpAndReset(
                reason: "generation-end tokens=\(tokenCount)")

            let info = GenerateCompletionInfo(
                promptTokenCount: promptTokenCount,
                generationTokenCount: tokenCount,
                promptTime: promptTime + iterator.promptPrefillTime,
                generationTime: generateTime,
                stopReason: stopReason ?? .cancelled,
                unclosedReasoning: unclosedReasoning
            )
            _ = continuation.yield(handler.infoEvent(info))

            // Multi-tier cache: store cache state after completion info is
            // visible to stream consumers, but keep the store serialized on
            // this generation task. Detached SSM re-derive previously raced
            // Metal command encoders; yielding `.info` first removes the UI
            // end-of-stream wait without reintroducing concurrent model use.
            iterator.storeCacheAfterGeneration(
                generatedTokenIds: generatedTokenIds,
                includeGeneratedBoundary: stopReason == .stop && !handler.stopSequenceHit)

            // Synchronize with the stream to ensure tasks are completed
            Stream().synchronize()

            // Router-advice readback runs on its own Dispatch queue. Drain it
            // after MLX synchronization so short-lived CLI runs and app unload
            // paths do not tear down runtime state while the advisor is still
            // applying mmap page advice.
            MLXPressCanonicalExpertAdvisor.shared.waitUntilIdle()

            // Finalize the stream
            continuation.finish()
        }

        if let ticket = wiredMemoryTicket {
            await WiredMemoryTicket.withWiredLimit(ticket) {
                performIteration()
            }
        } else {
            performIteration()
        }
    }

    // When the consumer cancels (or ends) the stream, cancel our underlying task.
    continuation.onTermination = { termination in
        if case .cancelled = termination {
            task.cancel()
        }
    }

    return (stream, task)
}

/// Measures the execution time of a closure.
private func measure(_ closure: () throws -> Void) rethrows -> TimeInterval {
    let start = Date.timeIntervalSinceReferenceDate
    try closure()
    return Date.timeIntervalSinceReferenceDate - start
}

// MARK: - Generation structs

/// Reason why token generation stopped.
public enum GenerateStopReason: Sendable {
    /// Generation stopped because an EOS/unknown stop token was encountered.
    case stop

    /// Generation stopped because the configured max token limit was reached.
    case length

    /// Generation stopped due to explicit task cancellation or early stream termination.
    case cancelled
}

/// Represents metadata and statistics related to token generation.
///
/// Provides information about the number of tokens processed during both the prompt and generation phases, as well as the time taken for each phase.
public struct GenerateCompletionInfo: Sendable {
    /// The number of tokens included in the input prompt.
    public let promptTokenCount: Int

    /// The number of tokens generated by the language model.
    public let generationTokenCount: Int

    /// The time interval (in seconds) taken to process the input prompt.
    public let promptTime: TimeInterval

    /// The time interval (in seconds) taken to generate the output tokens.
    public let generateTime: TimeInterval

    /// Reason generation stopped.
    public let stopReason: GenerateStopReason

    /// True when the stream ended with the reasoning parser still in
    /// REASONING state — i.e. the model never emitted `</think>` (or
    /// the family-specific close tag) before EOS or `max_tokens`.
    ///
    /// Indicates the model got "trapped" in chain-of-thought without
    /// producing a final answer in the visible content stream.
    /// `Generation.chunk` events for this turn are typically empty
    /// while `Generation.reasoning` carries the entire output.
    ///
    /// Reasoning-trained models (Qwen3.6-A3B fine-tunes, some DeepSeek-V4
    /// variants) exhibit this on validation-style prompts ("give me a
    /// 20-digit number") because their training data extends thought
    /// through arbitrary self-verification. The fix is at the prompt
    /// layer (use `enable_thinking: false` for chat workloads, or
    /// implement a UI-level "answer trapped in thinking" fallback that
    /// surfaces the last sentence of `Generation.reasoning`).
    ///
    /// `false` for any caller that didn't wire a reasoning parser
    /// (no behavior change on non-reasoning workloads).
    public let unclosedReasoning: Bool

    /// The number of tokens processed per second during the prompt phase.
    public var promptTokensPerSecond: Double {
        Double(promptTokenCount) / promptTime
    }

    /// The number of tokens generated per second during the generation phase.
    public var tokensPerSecond: Double {
        Double(generationTokenCount) / generateTime
    }

    public init(
        promptTokenCount: Int,
        generationTokenCount: Int,
        promptTime: TimeInterval,
        generationTime: TimeInterval,
        stopReason: GenerateStopReason = .stop,
        unclosedReasoning: Bool = false
    ) {
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.promptTime = promptTime
        self.generateTime = generationTime
        self.stopReason = stopReason
        self.unclosedReasoning = unclosedReasoning
    }

    public func summary() -> String {
        """
        Prompt:     \(promptTokenCount) tokens, \(promptTokensPerSecond.formatted()) tokens/s, \(promptTime.formatted())s
        Generation: \(generationTokenCount) tokens, \(tokensPerSecond.formatted()) tokens/s, \(generateTime.formatted())s
        """
    }
}

/// Represents the different stages or outputs of the token generation process.
///
/// This enum distinguishes between the following:
/// - `.chunk`: A decoded string from one or more tokens generated by the language model.
/// - `.reasoning`: A streaming chain-of-thought chunk (content between `<think>` /
///   `</think>` tags, or the family-specific equivalent). Emitted only when the
///   runtime has an active `ReasoningParser` stamped on the model configuration.
/// - `.toolCall`: A tool call parsed from the generated output.
/// - `.info`: Metadata and performance statistics about the generation process.
public enum Generation: Sendable {
    /// A generated text chunk as a String.
    ///
    /// This is pure user-visible assistant text — reasoning has been peeled
    /// off (emitted as `.reasoning` instead) and tool-call envelopes have
    /// been extracted (emitted as `.toolCall`).
    case chunk(String)

    /// A streaming reasoning (chain-of-thought) text chunk.
    ///
    /// Emitted when the runtime has a `ReasoningParser` for this model and
    /// the model emits tokens inside a `<think>…</think>` block (or the
    /// family-specific analogue). Callers that render a "thinking" UI pane
    /// should route these separately from `.chunk`. Callers that do not
    /// need reasoning can safely ignore this case — `.chunk` remains the
    /// final user-visible answer.
    ///
    /// The library emits one `.reasoning` event per parser segment; a
    /// long reasoning block typically produces many small deltas. No
    /// `.chunk` event is ever emitted for the same bytes.
    case reasoning(String)

    /// Completion information summarizing token counts and performance metrics.
    case info(GenerateCompletionInfo)

    /// A tool call from the language model.
    case toolCall(ToolCall)

    /// Generated text or nil
    public var chunk: String? {
        switch self {
        case .chunk(let string): string
        case .reasoning: nil
        case .info: nil
        case .toolCall: nil
        }
    }

    /// Reasoning text or nil
    public var reasoning: String? {
        switch self {
        case .chunk: nil
        case .reasoning(let string): string
        case .info: nil
        case .toolCall: nil
        }
    }

    /// Completion info or nil
    public var info: GenerateCompletionInfo? {
        switch self {
        case .chunk: nil
        case .reasoning: nil
        case .info(let info): info
        case .toolCall: nil
        }
    }

    /// Tool call or nil
    public var toolCall: ToolCall? {
        switch self {
        case .chunk: nil
        case .reasoning: nil
        case .info: nil
        case .toolCall(let toolCall): toolCall
        }
    }

    /// Reducer that can be used with `throttle()` to gather elements into a batch
    @Sendable
    public static func collect(_ batch: [Generation]?, _ element: Generation) -> [Generation] {
        (batch ?? []) + [element]
    }
}

/// Represents the different stages or outputs of raw-token generation.
///
/// This mirrors `Generation`, but yields raw token IDs instead of decoded text/tool calls.
public enum TokenGeneration: Sendable {
    /// A generated token ID.
    case token(Int)

    /// Completion information summarizing token counts and performance metrics.
    case info(GenerateCompletionInfo)

    /// Token ID or nil
    public var token: Int? {
        switch self {
        case .token(let token): token
        case .info: nil
        }
    }

    /// Completion info or nil
    public var info: GenerateCompletionInfo? {
        switch self {
        case .token: nil
        case .info(let info): info
        }
    }

    /// Reducer that can be used with `throttle()` to gather elements into a batch
    @Sendable
    public static func collect(_ batch: [TokenGeneration]?, _ element: TokenGeneration)
        -> [TokenGeneration]
    {
        (batch ?? []) + [element]
    }
}

// MARK: - TokenLoopHandlers

private protocol TokenLoopHandler: Sendable {
    associatedtype Output

    /// Return false to stop the loop early.
    mutating func onToken(
        _ token: Int,
        emit: (sending Output) -> AsyncStream<Output>.Continuation.YieldResult
    ) -> Bool

    /// Called only when includeStopToken == true and a stop token was hit.
    mutating func onStopToken(
        _ token: Int,
        emit: (sending Output) -> AsyncStream<Output>.Continuation.YieldResult
    ) -> Bool

    /// Called after the token loop finishes, before the info event.
    mutating func onGenerationEnd(
        emit: (sending Output) -> AsyncStream<Output>.Continuation.YieldResult
    )

    func infoEvent(_ info: GenerateCompletionInfo) -> Output

    /// True when the last `onToken` returned false because a text-level
    /// stop sequence matched — the generation loop uses this to set
    /// `stopReason = .stop` rather than `.cancelled` on the terminal
    /// `.info` event. Default `false` for handlers that don't consume
    /// text (e.g., the raw-token handler).
    var stopSequenceHit: Bool { get }

    /// True when the handler is still inside a reasoning envelope before
    /// terminal flush. Must be snapshotted before `onGenerationEnd`, because
    /// flushing drains and closes parser state.
    var unclosedReasoning: Bool { get }
}

extension TokenLoopHandler {
    var stopSequenceHit: Bool { false }
    var unclosedReasoning: Bool { false }
}

private struct TextToolTokenLoopHandler: TokenLoopHandler, @unchecked Sendable {
    typealias Output = Generation

    var detokenizer: NaiveStreamingDetokenizer
    let toolCallProcessor: ToolCallProcessor
    /// Optional `<think>...</think>` stripper pipelined BEFORE the tool-call
    /// processor. When `nil` every decoded chunk goes straight to the
    /// tool-call processor (matches upstream ml-explore/mlx-swift-lm
    /// behaviour byte-for-byte).
    var reasoningParser: ReasoningParser?
    /// Text-level stop-sequence matcher. Runs at the tail of the
    /// pipeline against `.chunk` text only (reasoning + tool-call bytes
    /// are scoped out by construction). When a stop string matches,
    /// `onToken` returns false to halt the loop; the `.info` event
    /// reports `stopReason = .stop`.
    var stopStringMatcher: StopStringMatcher
    /// Flipped by `dispatch` when the stop matcher fires, so the loop
    /// task can signal `.stop` in its terminal `.info` event.
    private(set) var stopSequenceHit: Bool = false

    init(
        tokenizer: Tokenizer,
        format: ToolCallFormat,
        reasoningParser: ReasoningParser? = nil,
        stopStringMatcher: StopStringMatcher = StopStringMatcher(stopStrings: [])
    ) {
        detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        toolCallProcessor = ToolCallProcessor(format: format)
        self.reasoningParser = reasoningParser
        self.stopStringMatcher = stopStringMatcher
    }

    /// Feed a raw decoded chunk through the reasoning parser (if any) and
    /// the tool-call processor, yielding the user-visible text plus any
    /// complete tool-call events.
    ///
    /// Returns `false` to stop the loop when the consumer terminates.
    private mutating func dispatch(
        _ chunk: String,
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) -> Bool {
        // 1. Reasoning pass (if configured). Reasoning segments are
        //    surfaced as `.reasoning(String)` so callers can render a
        //    think-pane UI without re-parsing; content segments flow on
        //    to the tool-call processor.
        let contentChunks: [String]
        if var parser = reasoningParser {
            var pieces: [String] = []
            for segment in parser.feed(chunk) {
                switch segment {
                case .content(let c):
                    pieces.append(c)
                case .reasoning(let r):
                    for event in routeGenerationText(
                        r,
                        channel: .reasoning,
                        through: toolCallProcessor
                    ) {
                        if !emitRouted(event, emit: emit) {
                            reasoningParser = parser
                            return false
                        }
                    }
                }
            }
            reasoningParser = parser
            contentChunks = pieces
        } else {
            contentChunks = [chunk]
        }

        // 2. Tool-call pass. Each content piece is processed in order so
        //    the state machine inside `ToolCallProcessor` sees the same
        //    byte stream it would have seen without a reasoning parser.
        //
        // 3. Stop-string pass (if configured). Runs at the TAIL — only
        //    user-visible `.chunk` text is a candidate for a stop match,
        //    matching OpenAI semantics where stop sequences match the
        //    assistant answer, not the reasoning or tool envelope.
        for contentChunk in contentChunks {
            for event in routeGenerationText(
                contentChunk,
                channel: .content,
                through: toolCallProcessor
            ) {
                if !emitRouted(event, emit: emit) {
                    return false
                }
            }
        }
        return true
    }

    mutating func onToken(
        _ token: Int,
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) -> Bool {
        detokenizer.append(token: token)
        if let chunk = detokenizer.next() {
            return dispatch(chunk, emit: emit)
        }
        return true
    }

    mutating func onStopToken(
        _ token: Int,
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) -> Bool {
        true
    }

    mutating func onGenerationEnd(
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) {
        // Flush the reasoning parser — any buffered tail becomes content
        // (or a trailing `.reasoning` segment if the model stopped mid-
        // think block) per ReasoningParser.flush contract. The tool-call
        // processor then sees the final content piece, then goes through
        // the stop matcher tail before processEOS.
        if var parser = reasoningParser {
            for segment in parser.flush() {
                switch segment {
                case .content(let c):
                    for event in routeGenerationText(
                        c,
                        channel: .content,
                        through: toolCallProcessor
                    ) {
                        if !emitRouted(event, emit: emit) {
                            reasoningParser = parser
                            return
                        }
                    }
                case .reasoning(let r):
                    for event in routeGenerationText(
                        r,
                        channel: .reasoning,
                        through: toolCallProcessor
                    ) {
                        if !emitRouted(event, emit: emit) {
                            reasoningParser = parser
                            return
                        }
                    }
                }
            }
            reasoningParser = parser
        }

        for event in flushGenerationText(
            channel: reasoningParser?.isInsideReasoning == true ? .reasoning : .content,
            through: toolCallProcessor
        ) {
            if case .terminated = emit(event) {
                return
            }
        }

        // Drain the stop-string matcher's tail (anything held back while
        // waiting for disambiguation is now safe — no more tokens).
        if stopStringMatcher.isEnabled {
            let tail = stopStringMatcher.flush()
            if !tail.isEmpty {
                if case .terminated = emit(.chunk(tail)) {
                    return
                }
            }
        }

    }

    private mutating func emitRouted(
        _ event: Generation,
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) -> Bool {
        switch event {
        case .chunk(let text):
            if case .terminated = emitChunkThroughStopMatcher(text, emit: emit) {
                return false
            }
            return !stopSequenceHit
        case .reasoning, .toolCall, .info:
            if case .terminated = emit(event) {
                return false
            }
            return true
        }
    }

    /// Emit a `.chunk` through the stop-string matcher. Returns
    /// `.terminated` when the downstream consumer stops OR when the
    /// stop matcher fires (so the caller halts the loop).
    private mutating func emitChunkThroughStopMatcher(
        _ text: String,
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) -> AsyncStream<Generation>.Continuation.YieldResult {
        guard stopStringMatcher.isEnabled else {
            return emit(.chunk(text))
        }
        switch stopStringMatcher.feed(text) {
        case .streaming(let out):
            if out.isEmpty { return .enqueued(remaining: 0) }
            return emit(.chunk(out))
        case .stopped(let out):
            stopSequenceHit = true
            if out.isEmpty { return .terminated }
            _ = emit(.chunk(out))
            return .terminated
        }
    }

    func infoEvent(_ info: GenerateCompletionInfo) -> Generation {
        .info(info)
    }

    var unclosedReasoning: Bool {
        reasoningParser?.isInsideReasoning ?? false
    }
}

private struct RawTokenLoopHandler: TokenLoopHandler {
    typealias Output = TokenGeneration

    mutating func onToken(
        _ token: Int,
        emit: (sending TokenGeneration) -> AsyncStream<TokenGeneration>.Continuation.YieldResult
    ) -> Bool {
        if case .terminated = emit(.token(token)) {
            return false
        }
        return true
    }

    mutating func onStopToken(
        _ token: Int,
        emit: (sending TokenGeneration) -> AsyncStream<TokenGeneration>.Continuation.YieldResult
    ) -> Bool {
        if case .terminated = emit(.token(token)) {
            return false
        }
        return true
    }

    mutating func onGenerationEnd(
        emit: (sending TokenGeneration) -> AsyncStream<TokenGeneration>.Continuation.YieldResult
    ) {}

    func infoEvent(_ info: GenerateCompletionInfo) -> TokenGeneration {
        .info(info)
    }
}

// MARK: - Prompt-tail decoding helper (file-private, used by generate paths)

/// Decode the last `tokens` token ids of a prompt into text for use
/// with `ReasoningParser.forPrompt(stampName:promptTail:)`. Tells the
/// parser whether the prompt ends inside a think/harmony block (so
/// the model's first output byte is reasoning) or after a closed
/// block (content).
///
/// Returns `nil` on empty input or decode failure — the caller then
/// falls back to the stamp-inferred default in `forPrompt`.
internal func _decodePromptTail(
    input: LMInput,
    tokenizer: any Tokenizer,
    tokens: Int
) -> String? {
    let promptTokens = input.text.tokens
    guard promptTokens.ndim >= 1 else { return nil }
    let total = promptTokens.ndim == 1
        ? promptTokens.dim(0)
        : promptTokens.dim(promptTokens.ndim - 1)
    guard total > 0 else { return nil }
    let tailLen = min(tokens, total)
    let startIdx = total - tailLen
    let tailArray: MLXArray
    if promptTokens.ndim == 1 {
        tailArray = promptTokens[startIdx ..< total]
    } else {
        tailArray = promptTokens[.ellipsis, startIdx ..< total]
    }
    let tailInts = tailArray.asArray(Int32.self).map { Int($0) }
    return tokenizer.decode(tokenIds: tailInts, skipSpecialTokens: false)
}

internal func _decodePromptTail(
    tokenIds: [Int],
    tokenizer: any Tokenizer,
    tokens: Int
) -> String? {
    guard !tokenIds.isEmpty else { return nil }
    let tail = Array(tokenIds.suffix(max(1, tokens)))
    return tokenizer.decode(tokenIds: tail, skipSpecialTokens: false)
}
