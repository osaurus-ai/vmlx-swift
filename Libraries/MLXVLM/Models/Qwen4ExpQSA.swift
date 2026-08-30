// Copyright © 2026 Apple Inc.

// Qwen 3.8 Next Flash (qwen4_exp) — QSA (Qwen Sparse Attention) block
// selection. Reference: `Qwen4ExpQSAIndexer` in
// `mlx_vlm/models/qwen4_exp/language.py`.
//
// Full-attention layers score mean-pooled key BLOCKS (compress_ratio tokens
// each) with ReLU(q·k̄) summed over indexer heads, keep the top
// `token_budget / compress_ratio` blocks per query position, and attend
// densely only to those blocks plus the incomplete tail. Queries with too
// few complete blocks fall back to plain causal attention.

import Foundation
import MLX

enum Qwen4ExpQSA {
    /// Boolean attention mask [B, 1, T, keyLen] from indexer scores.
    ///
    /// - Parameters:
    ///   - query: roped indexer queries [B, H, T, D]
    ///   - pooledKeys: roped, normed mean-pooled blocks [B, 1, numBlocks, D]
    ///   - pastLen: tokens already in the cache before this segment
    ///   - compressRatio: tokens per block
    ///   - blockTopK: blocks kept per query position
    ///   - keyLen: total raw key length (cache + this segment)
    /// - Returns: nil when every query still falls below the sparse
    ///   threshold (dense attention should run unmasked).
    static func selectedTokenMask(
        query: MLXArray,
        pooledKeys: MLXArray,
        pastLen: Int,
        compressRatio: Int,
        blockTopK: Int,
        keyLen: Int
    ) -> MLXArray? {
        let seqLen = query.dim(2)
        let maxCompleteBlocks = keyLen / compressRatio
        guard maxCompleteBlocks > blockTopK else { return nil }

        // ReLU(q·k̄) summed over indexer heads, scaled by sqrt(D).
        var scores = matmul(query, pooledKeys.transposed(0, 1, 3, 2))
        scores = maximum(scores.asType(.float32), MLXArray(Float(0))).sum(axis: 1)
        scores = scores / sqrt(Float(query.dim(3)))
        // scores: [B, T, numBlocks]

        // Only blocks fully behind each query are candidates.
        let queryEnds = MLXArray(
            (0 ..< seqLen).map { Int32(pastLen + $0 + 1) })  // [T]
        let completeCounts = floorDivide(queryEnds, MLXArray(Int32(compressRatio)))  // [T]
        let blockIdx = MLXArray((0 ..< maxCompleteBlocks).map(Int32.init))  // [numBlocks]
        let validBlocks = MLX.less(
            blockIdx.reshaped(1, 1, maxCompleteBlocks),
            completeCounts.reshaped(1, seqLen, 1))
        scores = MLX.where(
            validBlocks, scores, MLXArray(-Float.infinity))

        let selectedBlocks = argPartition(scores, kth: -blockTopK, axis: -1)[
            .ellipsis, (-blockTopK)...]
        // selectedBlocks: [B, T, blockTopK]

        // Mark winners on the compressed block axis, then widen each block to
        // its raw tokens. Comparing every raw token against every selected
        // block materializes [B,T,blockTopK,keyLen]: at an 18k prompt with the
        // production top-k of 512 that exceeds MLX's maximum buffer size.
        // `putAlong` keeps the largest intermediate at [B,T,keyLen] while
        // producing the exact same membership mask.
        let batch = query.dim(0)
        let blockHits = putAlong(
            MLXArray.zeros([batch, seqLen, maxCompleteBlocks], dtype: .bool),
            selectedBlocks,
            values: MLXArray(true),
            axis: -1)
        var selectedTokens = repeated(blockHits, count: compressRatio, axis: -1)
        let completeKeyLen = maxCompleteBlocks * compressRatio
        if completeKeyLen < keyLen {
            selectedTokens = concatenated(
                [
                    selectedTokens,
                    MLXArray.zeros(
                        [batch, seqLen, keyLen - completeKeyLen], dtype: .bool),
                ],
                axis: -1)
        }
        // selectedTokens: [B, T, keyLen]

        // The incomplete tail block up to each query position always attends.
        let tokenIdx = MLXArray((0 ..< keyLen).map(Int32.init))  // [keyLen]
        let tailStarts = completeCounts * Int32(compressRatio)  // [T]
        let tokenRow = tokenIdx.reshaped(1, 1, keyLen)
        let beforeQueryEnd = MLX.less(tokenRow, queryEnds.reshaped(1, seqLen, 1))
        let tail = MLX.logicalAnd(
            MLX.greaterEqual(tokenRow, tailStarts.reshaped(1, seqLen, 1)),
            beforeQueryEnd)
        let causal = beforeQueryEnd

        // Rows with too few complete blocks stay dense-causal.
        let useSparse = MLX.greater(completeCounts, MLXArray(Int32(blockTopK)))  // [T]
        let mask = MLX.where(
            useSparse.reshaped(1, seqLen, 1),
            MLX.logicalOr(selectedTokens, tail),
            causal)
        return expandedDimensions(mask, axis: 1)  // [B, 1, T, keyLen]
    }
}
