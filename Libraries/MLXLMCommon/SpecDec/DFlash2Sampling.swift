// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Sampling + acceptance for DFlash 2.
//
// Port target: `_sampling_probs`, `_sample_probs` and `_rejection_sample`
// in z-lab/dflash `dflash/model_mlx.py`.
//
// Why these live here rather than reusing ``TopPSampler``: speculative
// acceptance needs the target's full probability VECTOR, not just the
// sampled token, and it needs the drafter's q evaluated at the same
// filtered distribution. ``LogitSampler`` only returns a token, so the
// probability construction has to be spelled out. The filter order and
// the renormalisation below match the reference exactly — any drift here
// silently changes the output distribution rather than failing loudly.

import Foundation
import MLX
import MLXRandom

public enum DFlash2Sampling {

    /// Filtered probability distribution over the vocabulary.
    ///
    /// Applies temperature, then top-k, then top-p, in the reference's
    /// order. Positions filtered out come back as exact zeros, and the
    /// result is renormalised so it sums to one — the acceptance test
    /// needs a genuine distribution, not a masked score vector.
    ///
    /// - Parameter logits: `(…, vocab)`.
    public static func probabilities(
        _ logits: MLXArray, temperature: Float, topP: Float = 1.0, topK: Int = 0
    ) -> MLXArray {
        var scores = logits.asType(.float32) / temperature
        let vocab = scores.dim(-1)

        var indices: MLXArray?
        if topK > 0, topK < vocab {
            let selected = argPartition(scores, kth: vocab - topK, axis: -1)[
                .ellipsis, (vocab - topK)...]
            scores = takeAlong(scores, selected, axis: -1)
            indices = selected
        }

        var probs = softmax(scores, axis: -1)

        if topP < 1.0 {
            // Descending order via argSort on the negated probabilities.
            let order = argSort(-probs, axis: -1)
            let sorted = takeAlong(probs, order, axis: -1)
            // Keep everything whose EXCLUSIVE cumulative mass is still
            // under topP — so the token that crosses the threshold is
            // itself kept, and at least one token always survives.
            let keep = (cumsum(sorted, axis: -1) - sorted) .< topP
            let masked = MLX.where(keep, sorted, MLXArray(Float(0)))
            probs = putAlong(MLXArray.zeros(like: probs), order, values: masked, axis: -1)
            probs = probs / probs.sum(axis: -1, keepDims: true)
        }

        if let indices {
            // Scatter the top-k probabilities back into vocabulary space
            // so downstream gathers can index by token id.
            var shape = logits.shape
            shape[shape.count - 1] = vocab
            probs = putAlong(
                MLXArray.zeros(shape, dtype: probs.dtype), indices, values: probs, axis: -1)
        }
        return probs
    }

    /// Sample one index per row from a probability array.
    public static func sample(probabilities: MLXArray) -> MLXArray {
        categorical(MLX.log(probabilities))
    }

    /// Outcome of verifying one drafted block against the target.
    public struct Acceptance {
        /// Number of leading draft tokens the target accepted.
        public let accepted: Int
        /// The extra token appended after the accepted prefix — the
        /// target's own next token when everything was accepted, or the
        /// residual-corrected replacement for the first rejection.
        public let bonus: Int
    }

    /// Greedy acceptance: take the longest prefix where the draft agrees
    /// with the target's argmax, then append the target's token at the
    /// first divergence. This is what makes greedy DFlash 2 lossless —
    /// the emitted sequence is exactly the target's own greedy output.
    ///
    /// - Parameters:
    ///   - draftTokens: `gamma` drafted ids.
    ///   - targetTokens: `gamma + 1` argmax ids from the verify forward.
    public static func acceptGreedy(
        draftTokens: [Int], targetTokens: [Int]
    ) -> Acceptance {
        var accepted = 0
        while accepted < draftTokens.count, accepted < targetTokens.count,
            draftTokens[accepted] == targetTokens[accepted]
        {
            accepted += 1
        }
        let bonusIndex = Swift.min(accepted, targetTokens.count - 1)
        return Acceptance(accepted: accepted, bonus: targetTokens[bonusIndex])
    }

    /// Speculative rejection sampling over a candidate-restricted draft.
    ///
    /// Port of `_rejection_sample`. Accepts draft token `i` with
    /// probability `min(1, p_i / q_i)`; on the first rejection, samples
    /// the replacement from the normalised residual `max(0, p − q)`.
    /// That is the standard construction, so the emitted distribution
    /// equals the target's — DFlash 2's "sampling preserves the target
    /// distribution" claim rests on this function alone.
    ///
    /// - Parameters:
    ///   - draftTokens: `(1, gamma)` the traced path.
    ///   - targetProbabilities: `(1, gamma + 1, vocab)` filtered target
    ///     distribution for every verify position.
    ///   - draftProbabilities: `(1, gamma, top_k)` q over the candidate
    ///     sets.
    ///   - draftIndices: `(1, gamma, top_k)` the candidate token ids that
    ///     `draftProbabilities` is indexed by.
    public static func acceptSampled(
        draftTokens: MLXArray,
        targetProbabilities: MLXArray,
        draftProbabilities: MLXArray,
        draftIndices: MLXArray
    ) -> Acceptance {
        let gamma = draftTokens.dim(1)
        let tokenColumn = draftTokens.expandedDimensions(axis: -1)

        // p(x) under the target, at the drafted tokens.
        let p = takeAlong(
            targetProbabilities[0..., ..<gamma, 0...], tokenColumn, axis: -1)[.ellipsis, 0]
        // q(x) under the drafter: the candidate row's probability at
        // whichever slot holds the drafted token.
        let q = (draftProbabilities * (draftIndices .== tokenColumn)).sum(axis: -1)

        let u = MLXRandom.uniform(low: 0, high: 1, q.shape)
        let acceptedArray = cumprod(((u * q) .< p).asType(.int32), axis: -1).sum(axis: -1)
        let accepted = acceptedArray[0].item(Int.self)

        if accepted == gamma {
            let bonus = sample(probabilities: targetProbabilities[0..., -1, 0...])[0].item(Int.self)
            return Acceptance(accepted: accepted, bonus: bonus)
        }

        // Residual correction at the rejection point. The draft's mass
        // is subtracted only at the candidate ids it could have produced.
        var residual = targetProbabilities[0, accepted]
        let indices = draftIndices[0, accepted]
        let values = MLX.take(residual, indices) - draftProbabilities[0, accepted]
        residual = putAlong(
            residual.expandedDimensions(axis: 0),
            indices.expandedDimensions(axis: 0),
            values: values.expandedDimensions(axis: 0),
            axis: -1)[0]
        residual = maximum(residual, MLXArray(Float(0)))
        let total = residual.sum()
        // An all-zero residual means the draft distribution dominated the
        // target everywhere it had mass; fall back to the target itself
        // rather than dividing by zero.
        residual = MLX.where(
            total .> 0, residual / maximum(total, MLXArray(Float(1e-30))),
            targetProbabilities[0, accepted])
        let bonus = sample(probabilities: residual.expandedDimensions(axis: 0))[0].item(Int.self)
        return Acceptance(accepted: accepted, bonus: bonus)
    }
}
