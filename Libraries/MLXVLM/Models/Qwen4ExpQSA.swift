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
    /// Keep the portable gathered path bounded without turning every prompt
    /// into dozens of tiny dispatches.
    static func queryChunkSize(keyLen: Int) -> Int {
        if keyLen <= 4_096 { return 32 }
        if keyLen <= 16_384 { return 64 }
        return 128
    }

    /// Exact QSA attention over only the selected K/V rows.
    ///
    /// This is the bounded fallback for the production batch-one text path.
    /// It never constructs `[B,T,keyLen]` (or the still larger
    /// `[B,T,topK,keyLen]`) masks: indexer scoring and main attention are both
    /// query-chunked, and each row gathers at most the configured token budget
    /// plus the incomplete compressed tail.
    static func gatheredAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        indexQueries: MLXArray,
        pooledIndexKeys: MLXArray,
        pastLen: Int,
        compressRatio: Int,
        blockTopK: Int,
        scale: Float,
        queryChunk: Int? = nil
    ) -> MLXArray {
        let batch = queries.dim(0)
        let queryHeads = queries.dim(1)
        let queryTokens = queries.dim(2)
        let headDim = queries.dim(3)
        let kvHeads = keys.dim(1)
        let keyTokens = keys.dim(2)
        let indexHeads = indexQueries.dim(1)
        let indexHeadDim = indexQueries.dim(3)
        let maxBlocks = keyTokens / compressRatio

        precondition(batch == 1, "gathered QSA requires batch size 1")
        precondition(queryTokens > 0 && pastLen + queryTokens == keyTokens)
        precondition(keys.shape == values.shape)
        precondition(keys.dim(0) == batch && keys.dim(3) == headDim)
        precondition(queryHeads % kvHeads == 0)
        precondition(indexQueries.shape == [batch, indexHeads, queryTokens, indexHeadDim])
        precondition(pooledIndexKeys.shape == [batch, maxBlocks, indexHeadDim])
        precondition(compressRatio > 0 && blockTopK > 0)

        let groups = queryHeads / kvHeads
        let chunkSize = queryChunk ?? queryChunkSize(keyLen: keyTokens)
        precondition(chunkSize > 0)
        let keyRows = keys.transposed(0, 2, 1, 3)
        let valueRows = values.transposed(0, 2, 1, 3)
        let pooled4 = expandedDimensions(pooledIndexKeys, axis: 1)
        var outputs: [MLXArray] = []

        func batchGather(_ source: MLXArray, indices: MLXArray) -> MLXArray {
            let width = indices.dim(2)
            let offsets = MLXArray((0 ..< batch).map { Int32($0 * keyTokens) })
                .reshaped(batch, 1, 1)
            let flatIndices = (indices.asType(.int32) + offsets).reshaped(-1)
            let flat = source.reshaped(batch * keyTokens, kvHeads, headDim)
            return flat.take(flatIndices, axis: 0)
                .reshaped(batch, indices.dim(1), width, kvHeads, headDim)
        }

        for start in stride(from: 0, to: queryTokens, by: chunkSize) {
            let stop = min(start + chunkSize, queryTokens)
            let chunkTokens = stop - start
            let queryEnds = MLXArray(
                (start ..< stop).map { Int32(pastLen + $0 + 1) })
                .reshaped(1, chunkTokens)
            let completeCounts = floorDivide(
                queryEnds, MLXArray(Int32(compressRatio)))

            let chunkIndexQueries = indexQueries[0..., 0..., start ..< stop, 0...]
            var scores = matmul(
                chunkIndexQueries.asType(.float32),
                pooled4.asType(.float32).transposed(0, 1, 3, 2))
            scores = maximum(scores, MLXArray(Float(0))).sum(axis: 1)
                / sqrt(Float(indexHeadDim))

            let blockIndices = MLXArray((0 ..< maxBlocks).map(Int32.init))
                .reshaped(1, 1, maxBlocks)
            let validBlocks = MLX.less(
                blockIndices, completeCounts.reshaped(1, chunkTokens, 1))
            scores = MLX.where(validBlocks, scores, MLXArray(-Float.infinity))

            let selectedWidth = min(maxBlocks, blockTopK)
            let canonical = MLX.broadcast(
                MLXArray((0 ..< selectedWidth).map(Int32.init)).reshaped(1, 1, selectedWidth),
                to: [batch, chunkTokens, selectedWidth])
            var selectedBlocks = canonical
            if maxBlocks > blockTopK {
                let ranked = argPartition(scores, kth: -blockTopK, axis: -1)[
                    .ellipsis, (-blockTopK)...].asType(.int32)
                selectedBlocks = MLX.where(
                    MLX.lessEqual(
                        completeCounts, MLXArray(Int32(blockTopK)))
                        .reshaped(batch, chunkTokens, 1),
                    canonical,
                    ranked)
            }
            selectedBlocks = MLX.sorted(selectedBlocks, axis: -1)

            let selectedCount = MLX.minimum(
                completeCounts, MLXArray(Int32(blockTopK)))
            var selectedIndices = (
                expandedDimensions(selectedBlocks, axis: -1) * Int32(compressRatio)
                    + MLXArray((0 ..< compressRatio).map(Int32.init))
                        .reshaped(1, 1, 1, compressRatio)
            ).reshaped(batch, chunkTokens, selectedWidth * compressRatio)
            var selectedValid = MLX.broadcast(
                MLX.less(
                    MLXArray((0 ..< selectedWidth).map(Int32.init))
                        .reshaped(1, 1, selectedWidth, 1),
                    selectedCount.reshaped(batch, chunkTokens, 1, 1)),
                to: [batch, chunkTokens, selectedWidth, compressRatio])
                .reshaped(batch, chunkTokens, selectedWidth * compressRatio)

            // The zero-to-three visible tokens after the last complete block
            // are part of Qwen's published sparse-attention contract.
            let tailWidth = compressRatio - 1
            if tailWidth > 0 {
                let tail = completeCounts.reshaped(batch, chunkTokens, 1)
                    * Int32(compressRatio)
                    + MLXArray((0 ..< tailWidth).map(Int32.init)).reshaped(1, 1, tailWidth)
                let tailValid = MLX.less(
                    tail, queryEnds.reshaped(batch, chunkTokens, 1))
                selectedIndices = concatenated([selectedIndices, tail], axis: -1)
                selectedValid = concatenated([selectedValid, tailValid], axis: -1)
            }

            let safeIndices = MLX.where(
                selectedValid, selectedIndices, MLXArray(Int32(0)))
            let selectedKeys = batchGather(keyRows, indices: safeIndices)
                .transposed(0, 1, 3, 2, 4)
            let selectedValues = batchGather(valueRows, indices: safeIndices)
                .transposed(0, 1, 3, 2, 4)

            let chunkQueries = queries[0..., 0..., start ..< stop, 0...]
                .transposed(0, 2, 1, 3)
            let groupedQueries = chunkQueries.reshaped(
                batch, chunkTokens, kvHeads, groups, headDim)
            var attentionScores = matmul(
                groupedQueries.asType(.float32),
                selectedKeys.asType(.float32).transposed(0, 1, 2, 4, 3)) * scale
            attentionScores = MLX.where(
                selectedValid.reshaped(batch, chunkTokens, 1, 1, -1),
                attentionScores,
                MLXArray(-Float.infinity))
            let probabilities = MLX.softmax(
                attentionScores, axis: -1, precise: true).asType(queries.dtype)
            let output = matmul(probabilities, selectedValues)
                .reshaped(batch, chunkTokens, queryHeads, headDim)
                .transposed(0, 2, 1, 3)
            outputs.append(output)
        }

        return concatenated(outputs, axis: 2)
    }

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
        var scores = matmul(
            query.asType(.float32),
            pooledKeys.asType(.float32).transposed(0, 1, 3, 2))
        scores = maximum(scores, MLXArray(Float(0))).sum(axis: 1)
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
