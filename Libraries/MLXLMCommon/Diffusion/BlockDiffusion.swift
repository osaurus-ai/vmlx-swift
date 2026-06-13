// Copyright © 2026 Jinho Jang (eric@jangq.ai)
//
// Block-diffusion generation primitives.
//
// DiffusionGemma generates text by autoregressively producing fixed-size
// canvases (token blocks). Each canvas is denoised over several decoder
// forwards: sample candidate tokens, accept the approximately-independent
// low-entropy subset, renoise the rest, and stop early once the canvas is
// stable and confident.
//
// Python reference: transformers
// src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py
// (EntropyBoundSampler, LinearTemperatureScheduleLogitsProcessor,
//  StableAndConfidentStoppingCriteria).

import Foundation
import MLX

/// Diffusion-loop parameters for a ``BlockDiffusionModel``.
///
/// Values originate from the bundle (`config.json` for `canvasLength`,
/// `generation_config.json` for the rest) — per repo policy they are not
/// user-tunable sampling knobs.
public struct BlockDiffusionParameters: Sendable {
    public var canvasLength: Int
    public var maxNewTokens: Int
    public var maxDenoisingSteps: Int
    public var entropyBound: Float
    public var tMin: Float
    public var tMax: Float
    public var stabilityThreshold: Int
    public var confidenceThreshold: Float
    public var eosTokenIds: Set<Int>
    public var padTokenId: Int

    public init(
        canvasLength: Int,
        maxNewTokens: Int = 256,
        maxDenoisingSteps: Int = 48,
        entropyBound: Float = 0.1,
        tMin: Float = 0.4,
        tMax: Float = 0.8,
        stabilityThreshold: Int = 1,
        confidenceThreshold: Float = 0.005,
        eosTokenIds: Set<Int> = [],
        padTokenId: Int = 0
    ) {
        self.canvasLength = canvasLength
        self.maxNewTokens = maxNewTokens
        self.maxDenoisingSteps = maxDenoisingSteps
        self.entropyBound = entropyBound
        self.tMin = tMin
        self.tMax = tMax
        self.stabilityThreshold = stabilityThreshold
        self.confidenceThreshold = confidenceThreshold
        self.eosTokenIds = eosTokenIds
        self.padTokenId = padTokenId
    }

    /// Overlay `generation_config.json` values (when present) onto the
    /// model's defaults.
    public func resolving(generationConfig: GenerationConfigFile?) -> BlockDiffusionParameters {
        guard let generationConfig else { return self }
        var resolved = self
        if let v = generationConfig.maxNewTokens { resolved.maxNewTokens = v }
        if let v = generationConfig.maxDenoisingSteps { resolved.maxDenoisingSteps = v }
        if let v = generationConfig.samplerConfig?.entropyBound { resolved.entropyBound = v }
        if let v = generationConfig.tMin { resolved.tMin = v }
        if let v = generationConfig.tMax { resolved.tMax = v }
        if let v = generationConfig.stabilityThreshold { resolved.stabilityThreshold = v }
        if let v = generationConfig.confidenceThreshold { resolved.confidenceThreshold = v }
        if let v = generationConfig.eosTokenIds?.values, !v.isEmpty {
            resolved.eosTokenIds = Set(v)
        }
        if let v = generationConfig.padTokenId { resolved.padTokenId = v }
        return resolved
    }

    /// Apply explicit per-request overrides (e.g. an app-level speed/quality
    /// slider). Clamped to at least 1 step.
    public func overriding(parameters: GenerateParameters) -> BlockDiffusionParameters {
        var resolved = self
        if let steps = parameters.diffusionMaxDenoisingSteps {
            resolved.maxDenoisingSteps = Swift.max(1, steps)
        }
        return resolved
    }
}

/// A language model that generates via block diffusion instead of
/// autoregressive next-token decoding.
///
/// The model exposes two explicit forwards:
/// - the **encoder** runs causally over committed tokens (prompt, then each
///   finalized canvas) and writes the KV cache;
/// - the **decoder** runs bidirectionally over a noisy canvas, reading the
///   encoder cache without mutating it, and returns per-position logits.
///
/// `generate()` routes conforming models to ``BlockDiffusionTokenIterator``.
/// The model's `prepare(_:cache:windowSize:)` must `throw` so the
/// autoregressive `TokenIterator` path fails loudly instead of silently
/// producing garbage.
public protocol BlockDiffusionModel: LanguageModel {
    /// Diffusion defaults from the bundle's `config.json`
    /// (`generation_config.json` overrides are applied at dispatch via
    /// ``BlockDiffusionParameters/resolving(generationConfig:)``).
    var blockDiffusionDefaults: BlockDiffusionParameters { get }

    /// Vocabulary size used for random canvas initialization.
    var diffusionVocabularySize: Int { get }

    /// Causal forward over `tokens` [1, T]; writes KV into `cache`.
    func encoderForward(_ tokens: MLXArray, cache: [KVCache])

    /// Bidirectional forward over `canvas` [1, C] reading (not writing)
    /// `cache`. Returns logits [1, C, vocab] with any final softcapping
    /// already applied.
    func decoderForward(
        canvas: MLXArray, cache: [KVCache], selfConditioningLogits: MLXArray?
    ) -> MLXArray

    /// Multimodal prefill hook: the full spliced prompt embeddings (text
    /// embeds + image features over the placeholder positions) plus
    /// per-position contiguous image-block ids (−1 for text). Returns `nil`
    /// for text-only models/inputs; the iterator then uses the token path.
    func encoderPromptEmbeddings(
        for input: LMInput
    ) throws -> (embeddings: MLXArray, visionBlockIds: MLXArray)?

    /// Causal forward over precomputed `embeddings` [1, T]; writes KV into
    /// `cache`. `visionBlockIds` enables bidirectional attention inside
    /// each image block.
    func encoderForward(
        embeddings: MLXArray, cache: [KVCache], visionBlockIds: MLXArray?)
}

extension BlockDiffusionModel {
    public func encoderPromptEmbeddings(
        for input: LMInput
    ) throws -> (embeddings: MLXArray, visionBlockIds: MLXArray)? { nil }

    public func encoderForward(
        embeddings: MLXArray, cache: [KVCache], visionBlockIds: MLXArray?
    ) {
        fatalError(
            "\(Self.self) does not implement embeddings-based encoder prefill")
    }
}

public enum BlockDiffusionModelError: Error, CustomStringConvertible {
    /// The model only supports block-diffusion generation; it was routed
    /// through an autoregressive prepare/decode path.
    case requiresBlockDiffusionEngine(String)
    case emptyPrompt

    public var description: String {
        switch self {
        case .requiresBlockDiffusionEngine(let modelType):
            "\(modelType) generates via block diffusion and cannot be driven "
                + "by the autoregressive TokenIterator; use generate() which "
                + "dispatches to BlockDiffusionTokenIterator"
        case .emptyPrompt:
            "block diffusion requires a non-empty prompt"
        }
    }
}

// MARK: - Pure sampling primitives

/// Linear temperature schedule. `curStep` counts DOWN from `maxSteps` to 1
/// (reverse diffusion), so the first denoising step runs at `tMax` and the
/// temperature anneals toward `tMin`.
func blockDiffusionTemperature(curStep: Int, maxSteps: Int, tMin: Float, tMax: Float) -> Float {
    tMin + ((tMax - tMin) * (Float(curStep) / Float(maxSteps)))
}

/// Per-position token entropy of `processedLogits` [B, C, V] → [B, C], fp32.
///
/// Numerically stable form: H = logSumExp(z) − Σ softmax(z)·z.
func canvasTokenEntropy(processedLogits: MLXArray) -> MLXArray {
    let z = processedLogits.asType(.float32)
    let lse = logSumExp(z, axis: -1)
    let p = softmax(z, axis: -1, precise: true)
    return lse - (p * z).sum(axis: -1)
}

/// Entropy-bound acceptance mask (https://arxiv.org/pdf/2505.24857).
///
/// Sort entropies ascending and accept the largest k positions such that
/// `cumsum(H)_k − H_k <= entropyBound` — an upper bound on the joint mutual
/// information between the accepted tokens, so they are approximately
/// independent. Returns a bool mask [B, C] in canvas order.
func entropyBoundAcceptMask(tokenEntropy: MLXArray, entropyBound: Float) -> MLXArray {
    let sortedIndices = argSort(tokenEntropy, axis: -1)
    let sortedEntropy = takeAlong(tokenEntropy, sortedIndices, axis: -1)
    let cumulative = cumsum(sortedEntropy, axis: -1)
    // sortedEntropy is the running max because the sort is ascending.
    let sortedAccept = (cumulative - sortedEntropy) .<= entropyBound
    // Scatter back to canvas order.
    return putAlong(
        MLXArray.zeros(tokenEntropy.shape, dtype: .bool),
        sortedIndices,
        values: sortedAccept,
        axis: -1)
}

/// Stops a denoising loop when the canvas is stable (identical argmax canvas
/// across `stabilityThreshold` consecutive prior steps) and confident (mean
/// token entropy below `confidenceThreshold`).
struct StableConfidentStopper {
    let stabilityThreshold: Int
    let confidenceThreshold: Float
    private var history: [[Int32]] = []

    init(stabilityThreshold: Int, confidenceThreshold: Float) {
        self.stabilityThreshold = stabilityThreshold
        self.confidenceThreshold = confidenceThreshold
    }

    mutating func reset() {
        history.removeAll(keepingCapacity: true)
    }

    mutating func shouldStop(argmaxCanvas: [Int32], meanEntropy: Float) -> Bool {
        let stable: Bool
        if stabilityThreshold == 0 {
            stable = true
        } else {
            stable =
                history.count >= stabilityThreshold
                && history.suffix(stabilityThreshold).allSatisfy { $0 == argmaxCanvas }
            history.append(argmaxCanvas)
            if history.count > stabilityThreshold {
                history.removeFirst(history.count - stabilityThreshold)
            }
        }
        return stable && meanEntropy < confidenceThreshold
    }
}
