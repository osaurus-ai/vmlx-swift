// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MiniMax-M3 / M3.1 DSpark verify: a T-row forward whose row r produces exactly the logits (and the K / V, index keys
// and drafter taps) of the r-th of T sequential one-token decode steps. Port of the vMLX Python runtime's
// `vmlx_engine/dspark/minimax_target.py` (`MiniMaxM3DSparkTarget.verify`, 630929ed).
//
// Per layer:
// - projections (q / k / v / o, index q / k, dense MLP, shared expert, router, lm_head): every row on decode
//   arithmetic. MLX's multi-row kernels reduce in another order than its one-row kernels, so affine projections run
//   `RowExactAffineQMV` and mxfp8 projections `DeepseekV41FP8Rows` (MLX's one-row kernel for a few rows sharing one
//   weight walk), each admitted per shape only after a bit-equality check against stock one-row calls; every other
//   projection (float, unadmitted shapes, other module types) runs as one-row calls.
// - attention, per row: each row attends exactly the keys a decode step at its position sees (the cache prefix up to
//   and including itself), with the MSA selection a decode step makes there: the indexer scores that row's index query
//   against the index keys up to its position (same kernels, same block max, local block, per-group top-k), through
//   the layer's own `attend` (the same path a one-row step routes to).
// - routed experts: one call for all rows while every (row, expert) pair stays on the unsorted gather kernels a
//   decode step uses (MiniMaxM3SwitchGLU below 64 routed pairs, JANGT below its sort threshold); otherwise one call
//   per row.
// - norms, RoPE, router top-k, SwiGLU-OAI, embeddings: row-independent ops.
// K / V and index keys of all T rows are appended to the cache in the forward. The verify runs when a DSpark session
// armed the cache list's tap holder (`MiniMaxM3SparseCache.dsparkVerifyArmed`) and the forward has several rows; every
// other forward is the model's ordinary one.

/// A projection call: `module(x)` or a replacement with the same result per row.
typealias MiniMaxM3LinearCall = (Linear, MLXArray) -> MLXArray

enum MiniMaxM3RowExact {
    /// The module's own call.
    static func plain(_ module: Linear, _ x: MLXArray) -> MLXArray { module(x) }

    /// Rows per multi-row kernel launch (every row of a launch stays in registers up to four rows at every bit width
    /// on the affine kernel; the Python runtime's `_ROWS_PER_LAUNCH`). Wider verifies run as several launches.
    static let rowsPerLaunch = 4

    /// `VMLX_ROW_EXACT_VERIFY_QMV=0|false|no|off`: one-row calls for every projection (A/B and diagnosis).
    static let multiRowKernels: Bool = {
        let raw = RuntimeEnvironment.value("VMLX_ROW_EXACT_VERIFY_QMV")?.lowercased() ?? ""
        return !["0", "false", "no", "off"].contains(raw)
    }()

    /// Verify cost diagnostics: projection calls that ran batched on an exact multi-row kernel vs as one-row calls,
    /// routed-expert layers batched vs per row (the Python runtime's `verify_paths`).
    final class Counters: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [String: Int] = [:]
        func add(_ key: String) { lock.withLock { values[key, default: 0] += 1 } }
        func reset() { lock.withLock { values = [:] } }
        var snapshot: [String: Int] { lock.withLock { values } }
    }

    static let counters = Counters()

    /// Balanced consecutive row blocks of at most `width` rows: (start, size).
    static func blocks(_ rows: Int, width: Int) -> [(Int, Int)] {
        let n = (rows + width - 1) / width
        let (base, extra) = (rows / n, rows % n)
        var start = 0
        return (0 ..< n).map { b in
            let size = base + (b < extra ? 1 : 0)
            defer { start += size }
            return (start, size)
        }
    }

    /// `f` applied to each row `[1, 1, K]` of `x` `[1, T, K]` alone, concatenated.
    static func rows(_ x: MLXArray, _ f: (MLXArray) -> MLXArray) -> MLXArray {
        let n = x.dim(1)
        if n == 1 { return f(x) }
        return concatenated((0 ..< n).map { f(contiguous(x[0..., $0 ..< ($0 + 1)])) }, axis: 1)
    }

    /// `module(x)` for `[1, T, K]` rows, every row equal to its one-row call.
    static func linear(_ module: Linear, _ x: MLXArray) -> MLXArray {
        let n = x.dim(1)
        guard n > 1 else { return module(x) }
        guard x.ndim == 3, x.dim(0) == 1, type(of: module) == QuantizedLinear.self, let q = module as? QuantizedLinear,
            multiRowKernels
        else {
            counters.add("linear_rows")
            return rows(x) { module($0) }
        }
        var batched = 0
        let outs = blocks(n, width: rowsPerLaunch).map { start, size -> MLXArray in
            let xb = x[0..., start ..< (start + size)]
            if size > 1, let y = multiRow(q, xb) {
                batched += 1
                return y
            }
            return rows(xb) { module($0) }
        }
        counters.add(batched > 0 ? "linear_batched" : "linear_rows")
        return outs.count == 1 ? outs[0] : concatenated(outs, axis: 1)
    }

    /// Rows `[1, n, K]` through a bias-free quantized linear on an exact multi-row kernel, or nil when the shape is not
    /// served (or not admitted: each kernel checks every shape against stock one-row calls before using it).
    static func multiRow(_ q: QuantizedLinear, _ x: MLXArray) -> MLXArray? {
        guard q.bias == nil else { return nil }
        let (n, k) = (x.dim(1), x.dim(2))
        switch q.mode {
        case .affine:
            guard let biases = q.biases else { return nil }
            let y = RowExactAffineQMV.project(
                x.reshaped(n, k), weight: q.weight, scales: q.scales, biases: biases, groupSize: q.groupSize,
                bits: q.bits
            ) { probe in
                concatenated((0 ..< probe.dim(0)).map { q(probe[$0 ..< ($0 + 1)].reshaped(1, 1, k)) }, axis: 1)
                    .reshaped(probe.dim(0), -1)
            }
            return y?.reshaped(1, n, -1)
        case .mxfp8:
            return DeepseekV41FP8Rows.rows(x, weight: q.weight, scales: q.scales)
        default:
            return nil
        }
    }
}

extension MiniMaxM3Attention {
    /// Verify rows `x` `[1, T, D]` at positions `offset ..< offset + T` (see the file comment): K / V (and index keys)
    /// of every row appended to `cache`; row r attends the keys `0 ... offset + r` with the selection a decode step at
    /// `offset + r` makes. Returns the o_proj output `[1, T, D]`.
    func verifyRows(_ x: MLXArray, cache: KVCache, offset: Int) -> MLXArray {
        let (B, T) = (x.dim(0), x.dim(1))
        let (queries, newKeys, newValues) = project(x, offset: offset, linear: MiniMaxM3RowExact.linear)
        let keys: MLXArray
        let values: MLXArray
        var indexQueries: MLXArray?
        var indexKeys: MLXArray?
        if let indexer {
            guard let sparse = cache as? MiniMaxM3SparseCache else {
                preconditionFailure("MiniMax-M3 sparse attention layer needs a MiniMaxM3SparseCache")
            }
            (keys, values) = sparse.update(keys: newKeys, values: newValues)
            let (iq, ik) = indexer.project(x, offset: offset, linear: MiniMaxM3RowExact.linear)
            indexQueries = iq
            indexKeys = sparse.updateIndexKeys(ik)
        } else {
            (keys, values) = cache.update(keys: newKeys, values: newValues)
        }
        let outputs = (0 ..< T).map { r -> MLXArray in
            let end = offset + r + 1
            var selection: MiniMaxM3MSASelection?
            if let indexer, let indexQueries, let indexKeys {
                selection = indexer.select(
                    queries: indexQueries[0..., 0..., r ..< (r + 1)], keys: indexKeys[0..., 0..., ..<end],
                    offset: offset + r)
            }
            return attend(
                contiguous(queries[0..., 0..., r ..< (r + 1)]), keys[0..., 0..., ..<end],
                values[0..., 0..., ..<end], selection: selection, mask: .none)
        }
        let o = concatenated(outputs, axis: 2)
        return MiniMaxM3RowExact.linear(wo, o.transposed(0, 2, 1, 3).reshaped(B, T, -1))
    }
}

extension MiniMaxM3MLP {
    /// `self(x)` with every projection on decode arithmetic per row.
    func verifyRows(_ x: MLXArray) -> MLXArray {
        let linear = MiniMaxM3RowExact.linear
        return linear(
            downProj, miniMaxM3SwiGLU(up: linear(upProj, x), gate: linear(gateProj, x), alpha: alpha, limit: limit))
    }
}

extension MiniMaxM3SparseMoeBlock {
    /// Pairs from which `MiniMaxM3SwitchGLU` sorts (row, expert) pairs onto the multi-row gather kernels.
    static let switchSortPairs = 64

    /// True when one routed call for all rows keeps every (row, expert) pair on the kernels a one-row call uses.
    func routedBatchIsExact(_ x: MLXArray, indices: MLXArray) -> Bool {
        let pairs = indices.size
        if switchMLP != nil { return pairs < Self.switchSortPairs }
        if experts is JANGTRoutedExpertLayer {
            return JANGTRoutedExpertLayer.usesDecodeKernels(pairs: pairs, dtype: x.dtype)
        }
        return false  // an unknown routed implementation: one call per row
    }

    func routed(_ x: MLXArray, indices: MLXArray, scores: MLXArray) -> MLXArray {
        if let switchMLP {
            return (switchMLP(x, indices) * scores[.ellipsis, .newAxis]).sum(axis: -2)
        }
        return (experts as! any WeightedRoutedExpertLayer).callRouted(x, indices: indices, scores: scores)
    }

    /// `self(x)` for verify rows: router and shared expert on decode arithmetic per row, routed experts batched only
    /// while that is exact.
    func verifyRows(_ x: MLXArray) -> MLXArray {
        let routerScores = sigmoid(MiniMaxM3RowExact.linear(gate, x).asType(.float32))
        let choice = routerScores + eScoreCorrectionBias
        let indices = argPartition(-choice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var scores = takeAlong(routerScores, indices, axis: -1)
        if topK > 1 && normTopKProb {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }
        scores = scores * routedScalingFactor
        let out: MLXArray
        if routedBatchIsExact(x, indices: indices) {
            MiniMaxM3RowExact.counters.add("routed_batched")
            out = routed(x, indices: indices, scores: scores)
        } else {
            MiniMaxM3RowExact.counters.add("routed_rows")
            out = concatenated(
                (0 ..< x.dim(1)).map { r in
                    routed(
                        contiguous(x[0..., r ..< (r + 1)]), indices: contiguous(indices[0..., r ..< (r + 1)]),
                        scores: contiguous(scores[0..., r ..< (r + 1)]))
                }, axis: 1)
        }
        return (out + sharedExperts.verifyRows(x)).asType(x.dtype)
    }
}

extension MiniMaxM3ModelInner {
    /// The verify forward (see the file comment): rows `inputs` `[1, T]` at the cache's offset, the final-norm output
    /// `[1, T, D]`; the drafter taps of every row are recorded like an ordinary forward's.
    func verifyForward(_ inputs: MLXArray, cache: [KVCache], holder: MiniMaxM3SparseCache) -> MLXArray {
        let offset = cache[0].offset
        var taps: [Int: MLXArray] = [:]
        var h = embedTokens(inputs)
        for (i, layer) in layers.enumerated() {
            h = h + layer.selfAttn.verifyRows(layer.inputLayerNorm(h), cache: cache[i], offset: offset)
            let y = layer.postAttentionLayerNorm(h)
            switch layer.mlp {
            case let moe as MiniMaxM3SparseMoeBlock: h = h + moe.verifyRows(y)
            case let dense as MiniMaxM3MLP: h = h + dense.verifyRows(y)
            default: h = h + MiniMaxM3RowExact.rows(y) { layer.mlp($0) }
            }
            if dsparkTapLayers.contains(i) { taps[i] = h }
        }
        if !dsparkTapLayers.isEmpty { recordTaps(taps, start: offset, holder: holder) }
        return norm(h)
    }

    /// Record the drafter's tap rows of this forward (positions `start ..< start + L`) on `holder`: the last
    /// `dsparkTailCapacity` rows of each tap layer, concatenated in the drafter's order.
    func recordTaps(_ taps: [Int: MLXArray], start: Int, holder: MiniMaxM3SparseCache) {
        guard dsparkTailCapacity > 0, let first = taps.values.first else { return }
        let L = first.dim(1)
        let n = Swift.min(L, dsparkTailCapacity)
        var rows: [MLXArray] = []
        for layer in dsparkTapLayers {
            guard let t = taps[layer] else { return }
            rows.append(t[0, (L - n)..., 0...])
        }
        let joined = rows.count == 1 ? rows[0] : concatenated(rows, axis: -1)
        var tail = holder.dsparkTail ?? HybridPoolHiddenTail(capacity: dsparkTailCapacity)
        if tail.capacity != dsparkTailCapacity { tail = HybridPoolHiddenTail(capacity: dsparkTailCapacity) }
        tail.record(rows: joined, startPosition: start + L - n)
        holder.dsparkTail = tail
    }
}
