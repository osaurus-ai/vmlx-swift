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
import MLXLMCommon

enum Qwen4ExpQSA {
    private static let decodeMaskEnabled =
        RuntimeEnvironment.value("VMLX_QSA_DECODE_MASK") != "0"

    #if canImport(Metal)
    // One SIMD group owns a block. Lanes test disjoint selected IDs, then
    // write all tokens in that block. No score, norm, RoPE or top-k arithmetic
    // is changed; the output is exactly the original integer membership/tail
    // predicate. KEY_LENGTH is an input, not a per-context shader variant.
    private static let decodeMaskKernel = MLXFast.metalKernel(
        name: "vmlx_qsa_decode_block_mask",
        inputNames: ["selected", "limits"], outputNames: ["mask"],
        source: """
            const uint block = thread_position_in_grid.x / 32u;
            const uint lane = thread_index_in_simdgroup;
            const uint key_length = uint(limits[0]);
            bool member = block == key_length / RATIO;
            for (uint i = lane; i < uint(limits[1]); i += 32u) {
                member = member || selected[i] == block;
            }
            member = simd_any(member);
            for (uint i = lane; i < RATIO; i += 32u) {
                const uint token = block * RATIO + i;
                if (token < key_length) mask[token] = member;
            }
            """)

    private static let reportDecodeMask: Void = {
        RuntimeDiagnostics.write(Data(
            "[Qwen4Exp] qsa_decode_mask=active score_math=unchanged selection=unchanged\n".utf8))
    }()
    #endif

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
        keyLen: Int,
        useDecodeMask: Bool? = nil
    ) -> MLXArray? {
        let seqLen = query.dim(2)
        let maxCompleteBlocks = keyLen / compressRatio
        guard maxCompleteBlocks > blockTopK else { return nil }

        // ReLU(q·k̄) summed over indexer heads, scaled by sqrt(D).
        var scores = matmul(query, pooledKeys.transposed(0, 1, 3, 2))
        scores = maximum(scores.asType(.float32), MLXArray(Float(0))).sum(axis: 1)
        scores = scores / sqrt(Float(query.dim(3)))
        // scores: [B, T, numBlocks]

        #if canImport(Metal)
        if useDecodeMask ?? decodeMaskEnabled,
            query.dim(0) == 1, seqLen == 1, pastLen + 1 == keyLen,
            Device.defaultDevice().deviceType == .gpu
        {
            // Every complete block is behind this single query. The generic
            // valid-block where() would retain every score unchanged. Keep the
            // same partition and tie handling, and fuse only boolean mask work.
            let selected = argPartition(scores, kth: -blockTopK, axis: -1)[
                .ellipsis, (-blockTopK)...]
            let blocks = (keyLen + compressRatio - 1) / compressRatio
            let result = decodeMaskKernel(
                [selected, MLXArray([Int32(keyLen), Int32(blockTopK)])],
                template: [("RATIO", compressRatio)],
                grid: (blocks * 32, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: [[1, 1, 1, keyLen]], outputDTypes: [.bool])[0]
            _ = reportDecodeMask
            return result
        }
        #endif

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

        let tokenIdx = MLXArray((0 ..< keyLen).map(Int32.init))  // [keyLen]
        let selectedTokens = tokenMembership(
            selectedBlocks: selectedBlocks, keyLen: keyLen, compressRatio: compressRatio)
        // selectedTokens: [B, T, keyLen]

        // The incomplete tail block up to each query position always attends.
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

    /// Scatter selected block membership, then expand to tokens. Integer max
    /// makes duplicate IDs order-independent without a [B,T,K,blocks] equality
    /// grid. Invalid IDs contribute zero to a safe index, preserving the old
    /// equality implementation's behavior. Causal/tail policy remains above.
    static func tokenMembership(
        selectedBlocks: MLXArray, keyLen: Int, compressRatio: Int
    ) -> MLXArray {
        let blocks = (keyLen + compressRatio - 1) / compressRatio
        let batch = selectedBlocks.dim(0), queries = selectedBlocks.dim(1)
        guard blocks > 0, selectedBlocks.dim(2) > 0 else {
            return MLXArray.zeros([batch, queries, keyLen], dtype: .bool)
        }
        let valid = logicalAnd(
            greaterEqual(selectedBlocks, MLXArray(Int32(0))),
            less(selectedBlocks, MLXArray(Int32(blocks))))
        let safeIDs = MLX.where(valid, selectedBlocks, MLXArray(Int32(0)))
        let batchIDs = MLXArray((0..<batch).map(Int32.init)).reshaped(batch, 1, 1)
        let queryIDs = MLXArray((0..<queries).map(Int32.init)).reshaped(1, queries, 1)
        let membership = MLXArray.zeros([batch, queries, blocks], dtype: .int32)
            .at[batchIDs, queryIDs, safeIDs].maximum(valid.asType(.int32))
            .asType(.bool)
        return broadcast(
            expandedDimensions(membership, axis: -1),
            to: [batch, queries, blocks, compressRatio])
            .reshaped(batch, queries, blocks * compressRatio)[.ellipsis, ..<keyLen]
    }
}
