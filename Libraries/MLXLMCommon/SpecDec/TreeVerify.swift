// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Port target: humanrouter/ddtree-mlx/ddtree_mlx/verify.py (810 lines)
//
// Iter 8 ships v1 — correct but slow:
// - Attention-only path.
// - For each tree node, run target ONCE on (prefix_tokens + path_to_node).
//   O(N) forwards per verify round. Same wall time as plain AR with the
//   same number of positions, so speedup is 0× for now.
// - Correctness is the goal: `tree_verify_forward` posterior at node i
//   must equal greedy AR's argmax at position (prefix_len + depth[i]).
//
// Phase 2 optimization (iter 10+) adds the single-forward path with
// combined (1, 1, T, prefix_len+T) attention mask + per-token RoPE.
// That's where the actual DDTree speedup comes from. Phase 3 adds
// hybrid-SSM per-node recurrent-state forking.

import Foundation
import MLX

/// Result of one tree-verify forward pass.
public struct TreeVerifyResult: @unchecked Sendable {

    /// Posterior (greedy argmax) token for each tree position. Length
    /// = `compiled.treeSize`. Consumed by
    /// ``TreeBuilder/followVerifiedTree(childMaps:posteriorTokens:)``.
    public let posteriorTokens: [Int32]

    /// Target model logits for every tree position. Shape
    /// `(1, treeSize, vocab)`. Retained so sampling strategies other
    /// than greedy argmax can swap in later.
    public let logits: MLXArray

    /// Final-accepted-node per-layer recurrent state snapshots, keyed
    /// by layer index. `nil` for pure-attention target models;
    /// populated in Phase 3 when hybrid SSM support lands.
    public let recurrentSnapshots: [Int: RecurrentSnapshot]?

    public init(
        posteriorTokens: [Int32],
        logits: MLXArray,
        recurrentSnapshots: [Int: RecurrentSnapshot]? = nil
    ) {
        self.posteriorTokens = posteriorTokens
        self.logits = logits
        self.recurrentSnapshots = recurrentSnapshots
    }
}

/// Per-layer recurrent state captured during a tree-verify forward.
/// Populated in Phase 3 when hybrid-SSM per-node forking lands.
public struct RecurrentSnapshot: @unchecked Sendable {
    public let convStates: MLXArray
    public let states: MLXArray
    public init(convStates: MLXArray, states: MLXArray) {
        self.convStates = convStates
        self.states = states
    }
}

public enum TreeVerify {

    /// Correct-but-slow tree verify (v1). Runs the target model once per
    /// tree node with a freshly-built `(prefix_tokens + path_tokens)`
    /// input. Produces the same posterior set as a single-forward
    /// tree-verify would, byte-for-byte.
    ///
    /// - Parameters:
    ///   - target: any model conforming to ``HiddenStateCaptureModel``.
    ///   - compiled: from ``TreeCompile/compile(tree:rootTokenID:prefixLen:)``.
    ///   - prefixTokens: `(prefix_len,)` Int32 tokens already consumed
    ///     by the target (prompt + accepted tokens so far).
    ///   - captureLayerIDs: which target blocks to capture hidden states
    ///     from. Set to the DFlash drafter's `target_layer_ids` to feed
    ///     back into the next round.
    public static func verifyForward(
        target: any HiddenStateCaptureModel,
        compiled: CompiledTree,
        prefixTokens: [Int32],
        captureLayerIDs: Set<Int> = []
    ) throws -> TreeVerifyResult {
        let treeSize = compiled.treeSize
        guard treeSize >= 1 else {
            throw SpecDecError.invalidRequest("DDTree verify tree must have a root")
        }

        let inputIds = compiled.inputIds.asType(.int32).asArray(Int32.self)
        let parents = compiled.parents

        // Per-tree-node walk: index 0 is root, then each node's path
        // goes root → parent(parent(...i)) → i (reversed).
        var posteriorTokens: [Int32] = Array(repeating: 0, count: treeSize)
        // Gather captured hidden state per tree index.
        // For now we don't stitch per-node captures into a single
        // `(1, tree_size, hidden)` tensor — the linear-verify
        // runtime path only needs the tail slice anyway. Phase 3 will
        // revisit.
        var perNodeCaptured: [Int: [Int: MLXArray]] = [:]

        // For each tree node, build the input `prefix + path_to_node`.
        // Walk parents[] from the node up to root, then reverse.
        for i in 0..<treeSize {
            var path: [Int32] = []
            path.reserveCapacity(treeSize)
            var cursor = i
            while cursor != 0 {
                path.append(inputIds[cursor])
                cursor = Int(parents[cursor])
                guard cursor >= 0 else {
                    throw SpecDecError.invalidRequest(
                        "DDTree verify parent chain broken at node \(i)")
                }
            }
            path.append(inputIds[0])   // root
            path.reverse()
            // Full model input = prefix + path.
            var full = prefixTokens
            full.append(contentsOf: path)
            let mlxInput = MLXArray(full).reshaped(1, full.count)
            let (logits, captured) = target(
                mlxInput, cache: nil, captureLayerIDs: captureLayerIDs)
            let lastPos = logits.dim(1) - 1
            let lastLogits = logits[0, lastPos, 0...]
            let argmaxIdx = argMax(lastLogits, axis: -1).asType(.int32)
            posteriorTokens[i] = argmaxIdx.item(Int32.self)
            if !captureLayerIDs.isEmpty {
                perNodeCaptured[i] = captured
            }
        }

        // Build (1, tree_size, vocab) logits via a final one-shot target
        // forward on the flattened DFS-ordered sequence. We don't need
        // this for the walk (posterior is already computed), but consumers
        // of `TreeVerifyResult` may want it for temperature / top-K
        // sampling in Phase 3+. For the v1 attention-only build we
        // populate a (1, tree_size, vocab) logits tensor whose row `i`
        // holds the last-position logits from the path-to-node-i forward.
        //
        // Cheaper: we already ran those forwards above. Re-capture them.
        // For simplicity and v1 correctness, we re-run the argmax path
        // once more and stack last-position logits.
        var lastRowLogits: [MLXArray] = []
        for i in 0..<treeSize {
            var path: [Int32] = []
            var cursor = i
            while cursor != 0 {
                path.append(inputIds[cursor])
                cursor = Int(parents[cursor])
                guard cursor >= 0 else {
                    throw SpecDecError.invalidRequest(
                        "DDTree verify parent chain broken at node \(i)")
                }
            }
            path.append(inputIds[0])
            path.reverse()
            var full = prefixTokens
            full.append(contentsOf: path)
            let mlxInput = MLXArray(full).reshaped(1, full.count)
            let (logits, _) = target(
                mlxInput, cache: nil, captureLayerIDs: [])
            let lastPos = logits.dim(1) - 1
            // Shape: (vocab,) — unsqueeze to (1, vocab) for stacking.
            lastRowLogits.append(
                logits[0, lastPos, 0...].expandedDimensions(axis: 0))
        }
        let stackedLogits = concatenated(lastRowLogits, axis: 0)
            .expandedDimensions(axis: 0)  // (1, tree_size, vocab)

        return TreeVerifyResult(
            posteriorTokens: posteriorTokens,
            logits: stackedLogits,
            recurrentSnapshots: nil)
    }
}
