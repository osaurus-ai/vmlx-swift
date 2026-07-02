// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// Marker protocol for LLMModels
public protocol LLMModel: LanguageModel, LoRAModel {

    /// Models can implement this is they need a custom `MessageGenerator`.
    ///
    /// The default implementation returns `DefaultMessageGenerator`.
    func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator
}

extension LLMModel {

    /// Default prepare step for ``LLMModel``.
    ///
    /// This will evaluate the prompt in chunks until there is a small number of
    /// tokens left to feed into the `TokenIterator`.
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let prefillStepSize = windowSize ?? 512

        // Work on a flat 1D view of the tokens internally so the slicing math
        // below is dimension-independent. Callers may pass tokens either as
        // 1D `[T]` (legacy) or 2D `[B=1, T]` (TokenIterator / Bench / Osaurus
        // path). The old code used `y[.newAxis, ..<step]` and `y[step...]`
        // which silently sliced the WRONG axis when the input was 2D — the
        // `..<step` would apply to the batch dim (size 1) leaving the chunk
        // shape unchanged, and `y[step...]` would produce an empty `[0, T]`
        // tensor which then crashed the next forward pass with
        // `[reshape] Cannot infer the shape of an empty array`.
        // This extension is single-sequence only: generation in vmlx-swift-lm
        // always uses batch=1, and the forward-pass shapes downstream (attention
        // masking, KV cache append, etc.) assume it. If a caller ever feeds a
        // truly batched input (dim 0 > 1), the flatten below would interleave
        // sequences into one; fail fast instead of producing silent garbage.
        let tokensShape = input.text.tokens.shape
        if tokensShape.count >= 2 && tokensShape[0] != 1 {
            fatalError(
                "LLMModel.prepare expects single-sequence input (batch=1), "
                + "got shape \(tokensShape). BatchEngine handles multi-sequence "
                + "batching outside this extension."
            )
        }

        var flatTokens = input.text.tokens.reshaped([-1])
        var flatMask: MLXArray? = nil
        if let m = input.text.mask {
            flatMask = m.ndim >= 2 ? m.reshaped([-1]) : m
        }

        // Prepare the prompt in chunks if larger than the prefill size.
        // Clear Metal cache between chunks to reduce memory pressure,
        // matching Python mlx-lm behavior. Critical for MoE models.
        while flatTokens.size > prefillStepSize {
            // Prefill is the only long-running generation phase with no
            // cancellation points: a client that disconnects mid-prefill
            // cancels the producer task, but without this check the loop
            // keeps encoding GPU work for the dead request until the whole
            // prompt is consumed. A follow-up request racing that orphan
            // producer on the shared command queue aborts the process
            // ("A command encoder is already encoding to this command
            // buffer"). Checking between chunks bounds the orphan window
            // to a single chunk.
            try Task.checkCancellation()
            // Build a [1, prefillStepSize] chunk for the model forward pass.
            let chunkTokens = flatTokens[..<prefillStepSize][.newAxis, 0...]
            let chunkMask = flatMask.map { $0[..<prefillStepSize] }
            let chunkText = LMInput.Text(tokens: chunkTokens, mask: chunkMask)
            _ = self(chunkText, cache: cache.isEmpty ? nil : cache, state: nil)
            MLX.eval(cache)
            flatTokens = flatTokens[prefillStepSize...]
            if let m = flatMask { flatMask = m[prefillStepSize...] }
            PrefillProgressReporter.reportCompletedUnits(input.text.tokens.size - flatTokens.size)
            Memory.clearCache()
        }

        // Return the remainder as a 1D `[T]` tensor regardless of the
        // caller's original rank. Downstream consumers —
        // `TokenIterator.step(previous:)` and
        // `BatchEngine.stepPrefill`'s post-prepare forward — both expect
        // the returned `.tokens` payload to be 1D and add a leading batch
        // axis themselves before invoking the model. Returning 2D here
        // would cause them to produce a 3D `[1, 1, T]` input and crash
        // the forward pass.
        return .tokens(LMInput.Text(tokens: flatTokens, mask: flatMask))
    }

    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        DefaultMessageGenerator()
    }
}
