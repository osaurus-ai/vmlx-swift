// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Port target: humanrouter/ddtree-mlx/ddtree_mlx/tree.py (234 lines).
// Phase 2 iter 6 — full algorithmic port. Byte-identical with the Python
// reference on synthetic logits; pinned by DDTreeBuilderTests.

import Foundation
import MLX

// MARK: - Minimal priority queue (Swift has no built-in heap)

/// Very small binary min-heap. Pops the element where `priority` is
/// smallest; ties broken by `rankTuple` (lexicographic). Good enough
/// for DDTree's budget (typically <= 256 nodes).
fileprivate struct _MinHeap<Payload> {
    /// (priority, rankTuple, payload). Heap ordered by (priority, rankTuple).
    fileprivate struct Entry {
        var priority: Double
        var rankTuple: [Int]
        var payload: Payload
    }

    private var storage: [Entry] = []

    var isEmpty: Bool { storage.isEmpty }
    var count: Int { storage.count }

    mutating func push(_ e: Entry) {
        storage.append(e)
        siftUp(storage.count - 1)
    }

    mutating func pop() -> Entry? {
        guard !storage.isEmpty else { return nil }
        storage.swapAt(0, storage.count - 1)
        let last = storage.removeLast()
        if !storage.isEmpty { siftDown(0) }
        return last
    }

    @inline(__always)
    private static func lt(_ a: Entry, _ b: Entry) -> Bool {
        if a.priority != b.priority { return a.priority < b.priority }
        // Lexicographic on rankTuple.
        for i in 0..<min(a.rankTuple.count, b.rankTuple.count) {
            if a.rankTuple[i] != b.rankTuple[i] {
                return a.rankTuple[i] < b.rankTuple[i]
            }
        }
        return a.rankTuple.count < b.rankTuple.count
    }

    private mutating func siftUp(_ start: Int) {
        var i = start
        while i > 0 {
            let parent = (i - 1) / 2
            if Self.lt(storage[i], storage[parent]) {
                storage.swapAt(i, parent)
                i = parent
            } else {
                break
            }
        }
    }

    private mutating func siftDown(_ start: Int) {
        var i = start
        let n = storage.count
        while true {
            let l = 2 * i + 1
            let r = 2 * i + 2
            var smallest = i
            if l < n && Self.lt(storage[l], storage[smallest]) { smallest = l }
            if r < n && Self.lt(storage[r], storage[smallest]) { smallest = r }
            if smallest == i { break }
            storage.swapAt(i, smallest)
            i = smallest
        }
    }
}

// MARK: - Heap entry payload

fileprivate struct _HeapPayload {
    let parentIndex: Int
    let depth: Int        // 1-based: root's children are depth 1
    let rank: Int         // position in top-K at this depth
    let logw: Double
}

// MARK: - TreeBuilder

public enum TreeBuilder {

    /// Build a DDTree from `(L, vocab)` draft logits.
    ///
    /// Steps:
    /// 1. Compute per-position top-K (`K = min(budget, vocab)`).
    /// 2. Convert top-K logits to log-probabilities via log-softmax.
    /// 3. Delegate to ``buildFromTopK(topTokenIds:topLogProbs:budget:)``.
    public static func build(draftLogits: MLXArray, budget: Int) throws -> DDTree {
        if budget <= 0 || draftLogits.dim(0) == 0 {
            return DDTree.empty()
        }

        let topK = min(budget, draftLogits.dim(-1))

        // MLX's sort / argSort are descending via negation; compute
        // top-K indices by sorting then slicing. For small K this is
        // fine; a partial-sort optimisation is iter 7 work if needed.
        let logitsF32 = draftLogits.asType(.float32)
        // Sort descending: negate, argSort ascending, take first K.
        let sortedIndices = argSort(-logitsF32, axis: -1)
        let topIndices = sortedIndices[.ellipsis, 0..<topK]
        // Gather logits at top indices.
        let topLogits = MLX.takeAlong(logitsF32, topIndices, axis: -1)
        // Log-softmax on the full row for normalizing constant.
        let maxPer = max(logitsF32, axis: -1, keepDims: true)
        let logZ = log(sum(exp(logitsF32 - maxPer), axis: -1, keepDims: true)) + maxPer
        let topLogProbs = topLogits - logZ

        return try buildFromTopK(
            topTokenIds: topIndices,
            topLogProbs: topLogProbs,
            budget: budget)
    }

    /// Build a DDTree from precomputed per-position top-K log-probabilities.
    ///
    /// Ports humanrouter/ddtree-mlx `build_ddtree_tree_from_topk` line
    /// for line. Algorithm 1 in the DDTree paper: best-first heap search
    /// over prefix log-weights under a fixed node budget.
    public static func buildFromTopK(
        topTokenIds: MLXArray,
        topLogProbs: MLXArray,
        budget: Int
    ) throws -> DDTree {
        if budget <= 0 || topTokenIds.dim(0) == 0 || topTokenIds.dim(-1) == 0 {
            return DDTree.empty()
        }

        let depthLimit = topTokenIds.dim(0)
        let topK = min(budget, topTokenIds.dim(-1))

        let idsBuffer = topTokenIds.asType(.int64).asArray(Int64.self)
        let lpBuffer = topLogProbs.asType(.float32).asArray(Float.self)
        let fullK = topTokenIds.dim(-1)

        @inline(__always)
        func tokenID(depth: Int, rank: Int) -> Int32 {
            // depth is 1-based — idsBuffer row = depth - 1
            let flatIdx = (depth - 1) * fullK + rank
            return Int32(idsBuffer[flatIdx])
        }

        @inline(__always)
        func logProb(depth: Int, rank: Int) -> Double {
            let flatIdx = (depth - 1) * fullK + rank
            return Double(lpBuffer[flatIdx])
        }

        // Output buffers.
        var nodeTokenIds: [Int32] = []
        nodeTokenIds.reserveCapacity(budget)
        var nodeDepths: [Int32] = []
        nodeDepths.reserveCapacity(budget)
        // parents[0] = -1 (root)
        var parents: [Int32] = [-1]
        var childMaps: [[Int32: Int32]] = [[:]]

        // Initial heap: push the first-rank child of the root (depth 1, rank 0).
        var heap = _MinHeap<_HeapPayload>()
        let firstLogw = logProb(depth: 1, rank: 0)
        heap.push(.init(
            priority: -firstLogw,
            rankTuple: [0],
            payload: _HeapPayload(
                parentIndex: 0, depth: 1, rank: 0, logw: firstLogw)))

        var nodeCount = 0
        while nodeCount < budget, let entry = heap.pop() {
            let p = entry.payload
            let token = tokenID(depth: p.depth, rank: p.rank)
            let currentIndex = nodeCount + 1
            nodeTokenIds.append(token)
            nodeDepths.append(Int32(p.depth))
            parents.append(Int32(p.parentIndex))
            childMaps.append([:])
            childMaps[p.parentIndex][token] = Int32(currentIndex)
            nodeCount += 1

            // Push sibling (same depth, next rank).
            if p.rank + 1 < topK {
                let lpCur = logProb(depth: p.depth, rank: p.rank)
                let lpNext = logProb(depth: p.depth, rank: p.rank + 1)
                let siblingLogw = p.logw - lpCur + lpNext
                var siblingRanks = entry.rankTuple
                siblingRanks[siblingRanks.count - 1] = p.rank + 1
                heap.push(.init(
                    priority: -siblingLogw,
                    rankTuple: siblingRanks,
                    payload: _HeapPayload(
                        parentIndex: p.parentIndex,
                        depth: p.depth,
                        rank: p.rank + 1,
                        logw: siblingLogw)))
            }

            // Push first child (next depth, rank 0).
            if p.depth < depthLimit {
                let childLogw = p.logw + logProb(depth: p.depth + 1, rank: 0)
                var childRanks = entry.rankTuple
                childRanks.append(0)
                heap.push(.init(
                    priority: -childLogw,
                    rankTuple: childRanks,
                    payload: _HeapPayload(
                        parentIndex: currentIndex,
                        depth: p.depth + 1,
                        rank: 0,
                        logw: childLogw)))
            }
        }

        // Build visibility matrix — ancestor-only attention mask.
        let currentLength = 1 + nodeCount
        let visibility = buildVisibilityMatrix(
            parents: parents, currentLength: currentLength)

        // Convert to MLX arrays.
        let tokenArr = MLXArray(nodeTokenIds).reshaped(nodeTokenIds.count)
        let depthArr = MLXArray(nodeDepths).reshaped(nodeDepths.count)

        return DDTree(
            nodeTokenIds: tokenArr,
            nodeDepths: depthArr,
            parents: parents,
            childMaps: childMaps,
            visibility: visibility,
            nodeCount: nodeCount)
    }

    /// Walk the verified tree greedily against the target's argmax tokens.
    /// Ports `follow_verified_tree` from tree.py.
    ///
    /// - Returns: `(acceptedIndices, bonusToken)`. `acceptedIndices[0]`
    ///   is always `0` (the root). `bonusToken` is the first target
    ///   token that didn't match any child at the walk's terminal node.
    public static func followVerifiedTree(
        childMaps: [[Int32: Int32]],
        posteriorTokens: [Int32]
    ) throws -> (acceptedIndices: [Int32], bonusToken: Int32) {
        guard !posteriorTokens.isEmpty else {
            throw SpecDecError.invalidRequest(
                "DDTree posteriorTokens must have at least one entry")
        }
        guard !childMaps.isEmpty else {
            throw SpecDecError.invalidRequest(
                "DDTree child map index out of range at node 0")
        }

        var acceptedIndices: [Int32] = [0]
        var currentIndex = 0
        var nextToken = posteriorTokens[0]

        while let childIdx = childMaps[currentIndex][nextToken] {
            currentIndex = Int(childIdx)
            guard currentIndex >= 0 && currentIndex < childMaps.count else {
                throw SpecDecError.invalidRequest(
                    "DDTree child map index out of range at node \(currentIndex)")
            }
            acceptedIndices.append(childIdx)
            if currentIndex >= posteriorTokens.count { break }
            nextToken = posteriorTokens[currentIndex]
        }
        return (acceptedIndices, nextToken)
    }

    /// Depth-first traversal order, highest-probability child first.
    /// Ports `compute_dfs_order` from tree.py.
    public static func computeDfsOrder(
        tree: DDTree
    ) throws -> (dfsOrder: [Int32], invDfsOrder: [Int32]) {
        if tree.nodeCount == 0 {
            return ([0], [0])
        }

        let n = 1 + tree.nodeCount
        var children: [[Int]] = Array(repeating: [], count: n)
        for idx in 1..<n {
            let parent = Int(tree.parents[idx])
            children[parent].append(idx)
        }

        var dfsOrder: [Int32] = []
        dfsOrder.reserveCapacity(n)
        var stack: [Int] = [0]
        while let node = stack.popLast() {
            dfsOrder.append(Int32(node))
            // Push children in reverse so the first child is popped first.
            for child in children[node].reversed() {
                stack.append(child)
            }
        }

        var invDfsOrder: [Int32] = Array(repeating: 0, count: n)
        for (pos, idx) in dfsOrder.enumerated() {
            invDfsOrder[Int(idx)] = Int32(pos)
        }
        return (dfsOrder, invDfsOrder)
    }

    // MARK: - Internals

    /// Build the (N+1, N+1) Bool ancestor-only visibility matrix.
    /// Ports the post-heap-build loop in `build_ddtree_tree_from_topk`.
    static func buildVisibilityMatrix(
        parents: [Int32], currentLength: Int
    ) -> MLXArray {
        // Use Bool storage for exact parity with Python's np.bool_.
        var mask: [Bool] = Array(repeating: false, count: currentLength * currentLength)
        mask[0] = true  // root sees itself
        for index in 1..<currentLength {
            let parentIndex = Int(parents[index])
            // Copy parent's row up to column `index`
            for col in 0..<index {
                mask[index * currentLength + col] =
                    mask[parentIndex * currentLength + col]
            }
            // Self-visible
            mask[index * currentLength + index] = true
        }
        return MLXArray(mask).reshaped(currentLength, currentLength)
    }
}
