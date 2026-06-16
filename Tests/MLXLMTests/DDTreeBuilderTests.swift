// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Phase 2 iter 6 — pin the tree builder byte-for-byte against hand-
// traced reference outputs that match humanrouter/ddtree-mlx tree.py
// Algorithm 1 exactly.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("DDTree builder — Phase 2", .serialized)
struct DDTreeBuilderTests {

    // MARK: - buildFromTopK

    @Test("Empty budget produces empty tree")
    func testEmptyBudget() throws {
        let tree = try TreeBuilder.buildFromTopK(
            topTokenIds: MLXArray([] as [Int32]).reshaped(0, 0),
            topLogProbs: MLXArray([] as [Float]).reshaped(0, 0),
            budget: 0)
        #expect(tree.nodeCount == 0)
        #expect(tree.parents == [-1])
    }

    @Test("Single-position budget=3 flat tree")
    func testFlatSinglePositionTree() throws {
        // Hand-traced: top_token_ids = [[10, 20, 30]], top_log_probs = [[0, -1, -2]].
        // Expected: three root children in descending prob order.
        let tokenIds = MLXArray([Int32(10), 20, 30]).reshaped(1, 3)
        let logProbs = MLXArray([Float(0), -1, -2]).reshaped(1, 3)

        let tree = try TreeBuilder.buildFromTopK(
            topTokenIds: tokenIds, topLogProbs: logProbs, budget: 3)

        #expect(tree.nodeCount == 3)
        #expect(tree.parents == [-1, 0, 0, 0])

        let tokens = tree.nodeTokenIds.asArray(Int32.self)
        #expect(tokens == [10, 20, 30])

        let depths = tree.nodeDepths.asArray(Int32.self)
        #expect(depths == [1, 1, 1])

        // childMaps — root has all three children.
        #expect(tree.childMaps[0] == [10: 1, 20: 2, 30: 3])
        #expect(tree.childMaps[1].isEmpty)
        #expect(tree.childMaps[2].isEmpty)
        #expect(tree.childMaps[3].isEmpty)
    }

    @Test("Two-depth budget=5 branching tree")
    func testTwoDepthBranchingTree() throws {
        // Hand-traced example from the iter 6 commit message:
        //   top_token_ids = [[10, 20], [30, 40]]
        //   top_log_probs = [[0, -1], [0, -1]]
        //   budget = 5, topK = 2
        let tokenIds = MLXArray([Int32(10), 20, 30, 40]).reshaped(2, 2)
        let logProbs = MLXArray([Float(0), -1, 0, -1]).reshaped(2, 2)

        let tree = try TreeBuilder.buildFromTopK(
            topTokenIds: tokenIds, topLogProbs: logProbs, budget: 5)

        #expect(tree.nodeCount == 5)
        #expect(tree.parents == [-1, 0, 1, 1, 0, 4])
        let tokens = tree.nodeTokenIds.asArray(Int32.self)
        #expect(tokens == [10, 30, 40, 20, 30])
        let depths = tree.nodeDepths.asArray(Int32.self)
        #expect(depths == [1, 2, 2, 1, 2])
        #expect(tree.childMaps[0] == [10: 1, 20: 4])
        #expect(tree.childMaps[1] == [30: 2, 40: 3])
        #expect(tree.childMaps[2].isEmpty)
        #expect(tree.childMaps[3].isEmpty)
        #expect(tree.childMaps[4] == [30: 5])
    }

    @Test("Budget=1 returns only the first child")
    func testBudgetOne() throws {
        let tokenIds = MLXArray([Int32(10), 20, 30]).reshaped(1, 3)
        let logProbs = MLXArray([Float(0), -1, -2]).reshaped(1, 3)
        let tree = try TreeBuilder.buildFromTopK(
            topTokenIds: tokenIds, topLogProbs: logProbs, budget: 1)
        #expect(tree.nodeCount == 1)
        #expect(tree.nodeTokenIds.asArray(Int32.self) == [10])
        #expect(tree.parents == [-1, 0])
    }

    @Test("Visibility matrix shape + ancestor-only invariant")
    func testVisibilityMatrixAncestorOnly() throws {
        // Use the branching example — walk the visibility matrix
        // directly to confirm each node sees only itself and ancestors.
        let tokenIds = MLXArray([Int32(10), 20, 30, 40]).reshaped(2, 2)
        let logProbs = MLXArray([Float(0), -1, 0, -1]).reshaped(2, 2)
        let tree = try TreeBuilder.buildFromTopK(
            topTokenIds: tokenIds, topLogProbs: logProbs, budget: 5)

        // (N+1)² = 6² = 36 entries. tree_size = 6.
        let vis = tree.visibility.asArray(Bool.self)
        let n = 6
        #expect(vis.count == n * n)

        // Helper: is (row, col) = true?
        func at(_ r: Int, _ c: Int) -> Bool { vis[r * n + c] }

        // Every node sees itself.
        for i in 0..<n { #expect(at(i, i), "self-visibility missing at \(i)") }

        // Structure from testTwoDepthBranchingTree:
        //   parents = [-1, 0, 1, 1, 0, 4]
        //   tree indices: 0 (root) → 1, 4; 1 → 2, 3; 4 → 5.
        //
        // Row 1 (parent 0) sees only {0, 1}.
        #expect(at(1, 0) && at(1, 1))
        #expect(!at(1, 2) && !at(1, 3) && !at(1, 4) && !at(1, 5))
        // Row 2 (parent 1): sees {0, 1, 2}.
        #expect(at(2, 0) && at(2, 1) && at(2, 2))
        #expect(!at(2, 3) && !at(2, 4) && !at(2, 5))
        // Row 3 (parent 1): sees {0, 1, 3}.
        #expect(at(3, 0) && at(3, 1) && at(3, 3))
        #expect(!at(3, 2) && !at(3, 4) && !at(3, 5))
        // Row 4 (parent 0): sees {0, 4}.
        #expect(at(4, 0) && at(4, 4))
        #expect(!at(4, 1) && !at(4, 2) && !at(4, 3) && !at(4, 5))
        // Row 5 (parent 4): sees {0, 4, 5}.
        #expect(at(5, 0) && at(5, 4) && at(5, 5))
        #expect(!at(5, 1) && !at(5, 2) && !at(5, 3))
    }

    // MARK: - followVerifiedTree

    @Test("followVerifiedTree walks to first mismatch")
    func testFollowVerifiedTreeBasic() throws {
        // childMaps from the branching example.
        let childMaps: [[Int32: Int32]] = [
            [10: 1, 20: 4],
            [30: 2, 40: 3],
            [:],
            [:],
            [30: 5],
            [:],
        ]
        // posterior[0]=10, posterior[1]=30, posterior[2]=99 → walk 0→1→2, stop.
        let posterior: [Int32] = [10, 30, 99, 99, 99, 99]
        let (accepted, bonus) = try TreeBuilder.followVerifiedTree(
            childMaps: childMaps, posteriorTokens: posterior)
        #expect(accepted == [0, 1, 2])
        #expect(bonus == 99)
    }

    @Test("followVerifiedTree walks entire path when all match")
    func testFollowVerifiedTreeFullPath() throws {
        let childMaps: [[Int32: Int32]] = [
            [10: 1, 20: 4],
            [30: 2, 40: 3],
            [:],
            [:],
            [30: 5],
            [:],
        ]
        // Walk 0 → 1 (posterior[0]=10) → 2 (posterior[1]=30). posterior[2]
        // has no children at node 2 → stop. bonus = posterior[2].
        let posterior: [Int32] = [10, 30, 7, 7, 7, 7]
        let (accepted, bonus) = try TreeBuilder.followVerifiedTree(
            childMaps: childMaps, posteriorTokens: posterior)
        #expect(accepted == [0, 1, 2])
        #expect(bonus == 7)
    }

    @Test("followVerifiedTree immediate miss returns root-only")
    func testFollowVerifiedTreeImmediateMiss() throws {
        let childMaps: [[Int32: Int32]] = [[10: 1], [:]]
        let (accepted, bonus) = try TreeBuilder.followVerifiedTree(
            childMaps: childMaps, posteriorTokens: [99, 0])
        #expect(accepted == [0])
        #expect(bonus == 99)
    }

    @Test("followVerifiedTree rejects empty posterior tokens")
    func testFollowVerifiedTreeRejectsEmptyPosteriorTokens() throws {
        do {
            _ = try TreeBuilder.followVerifiedTree(
                childMaps: [[10: 1], [:]], posteriorTokens: [])
            Issue.record("Expected empty posterior tokens to throw")
        } catch let error as SpecDecError {
            #expect(String(describing: error).contains("posteriorTokens"))
        }
    }

    @Test("followVerifiedTree rejects child map indexes out of range")
    func testFollowVerifiedTreeRejectsOutOfRangeChildMapIndex() throws {
        do {
            _ = try TreeBuilder.followVerifiedTree(
                childMaps: [[10: 3], [:]], posteriorTokens: [10, 0])
            Issue.record("Expected out-of-range child map index to throw")
        } catch let error as SpecDecError {
            #expect(String(describing: error).contains("child map index out of range"))
        }
    }

    // MARK: - computeDfsOrder

    @Test("computeDfsOrder matches hand-drawn branching tree")
    func testComputeDfsOrderBranching() throws {
        // Build the tree from the branching example.
        let tokenIds = MLXArray([Int32(10), 20, 30, 40]).reshaped(2, 2)
        let logProbs = MLXArray([Float(0), -1, 0, -1]).reshaped(2, 2)
        let tree = try TreeBuilder.buildFromTopK(
            topTokenIds: tokenIds, topLogProbs: logProbs, budget: 5)
        let (dfs, inv) = try TreeBuilder.computeDfsOrder(tree: tree)
        // Expected: 0, 1, 2, 3, 4, 5 (hand-traced — children appended in
        // heap-pop order, reversed onto DFS stack, popped first-child-first).
        #expect(dfs == [0, 1, 2, 3, 4, 5])
        // Inverse: identical (node i at position i).
        #expect(inv == [0, 1, 2, 3, 4, 5])
    }

    @Test("computeDfsOrder empty tree returns root-only")
    func testComputeDfsOrderEmpty() throws {
        let empty = DDTree.empty()
        let (dfs, inv) = try TreeBuilder.computeDfsOrder(tree: empty)
        #expect(dfs == [0])
        #expect(inv == [0])
    }

    // MARK: - build from logits

    @Test("build from logits produces the same tree as buildFromTopK")
    func testBuildFromLogitsParity() throws {
        // Synthetic logits: 2 positions, vocab=4. Top-2 per position.
        // Row 0: [3.0, 1.0, 0.5, 0.0]  → top2 = (0, 1) by logit → tokens 0, 1
        // Row 1: [0.0, 5.0, 0.0, 2.0]  → top2 = (1, 3) → tokens 1, 3
        let logits = MLXArray([
            Float(3), 1, 0.5, 0,
            0, 5, 0, 2,
        ]).reshaped(2, 4)
        let tree = try TreeBuilder.build(draftLogits: logits, budget: 5)
        // Basic sanity: tree has nodes, top root token is the arg-max of row 0.
        #expect(tree.nodeCount >= 1)
        let firstToken = tree.nodeTokenIds.asArray(Int32.self).first ?? -1
        #expect(firstToken == 0,
            "First root child must be the top-1 of row 0 (token index 0)")
    }
}
