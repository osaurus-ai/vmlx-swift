// Copyright © 2026 Osaurus AI. All rights reserved.
//
// DFlash 2 draft TREES: one drafter forward → a candidate lattice → a best-first tree of up to
// 15 nodes → ONE ≤16-row target verify with ancestor-path attention and a per-branch GatedDelta
// recurrence → the longest accepted root path is committed (KV rows compacted, recurrent state
// replayed along the kept path only).
//
// Mechanism ported from TensorFold 0.6.6 (Apache-2.0): `drafters/dflash_tree.py` (best_first_tree),
// `drafters/dflash_proposer.py` (lattice, constants), `kernels/qwen/dense/v1/lane_tree.py` (tree
// GatedDelta kernel, kept-path replay, accept_path) and `lane_multi.py` (tree forward / commit).
// Why: a chain verifies ONE candidate per position, so a single early miss wastes the whole
// window. With the lane matmul a 16-row verify costs about what 8 rows do, so spending the
// rows on the most likely BRANCHES instead of one deep chain raises tokens per cycle on prose.

import Foundation
import MLX

/// The tree a verify forward runs over. Row 0 is the pending (anchor) token; every other row is
/// a drafted node. `parents[r]` is the row index of r's parent (-1 for row 0), always < r.
public struct DFlash2TreePlan {
    public let tokens: [Int]
    public let parents: [Int]
    public let depths: [Int]
    /// Rows of each row's root path, root first.
    public let paths: [[Int]]

    public init(tokens: [Int], parents: [Int]) {
        precondition(tokens.count == parents.count && parents.first == -1)
        self.tokens = tokens
        self.parents = parents
        var depths: [Int] = []
        var paths: [[Int]] = []
        for (row, parent) in parents.enumerated() {
            if parent < 0 {
                paths.append([row])
            } else {
                precondition(parent < row, "parents must precede children")
                paths.append(paths[parent] + [row])
            }
            depths.append(paths[row].count - 1)
        }
        self.depths = depths
        self.paths = paths
    }

    public var rows: Int { tokens.count }

    /// A plain chain (each row's parent is the previous row).
    public var isChain: Bool { parents == Array(-1 ..< (rows - 1)) }

    /// `[3, 1, W]` M-RoPE position ids for rows at absolute `start` (text: all three equal).
    public func positionIds(start: Int) -> MLXArray {
        let base = MLXArray(depths.map { Int32(start + $0) }).reshaped(1, 1, rows)
        return tiled(base, repetitions: [3, 1, 1])
    }

    /// Additive attention mask `[1, 1, W, prefix + W]`: every row sees the whole committed prefix
    /// and, inside the window, exactly its own root path.
    /// Row r sees window row a (its own root path) — the window part of `attentionMask`.
    public var windowVisibility: [[Bool]] {
        (0 ..< rows).map { row in
            var seen = [Bool](repeating: false, count: rows)
            for a in paths[row] { seen[a] = true }
            return seen
        }
    }

    public func attentionMask(prefix: Int, dtype: DType) -> MLXArray {
        var allowed = [Bool](repeating: false, count: rows * rows)
        for row in 0 ..< rows {
            for a in paths[row] { allowed[row * rows + a] = true }
        }
        let window = MLXArray(allowed).reshaped(rows, rows)
        let full = prefix > 0
            ? concatenated([MLXArray.ones([rows, prefix], type: Bool.self), window], axis: 1)
            : window
        let zero = MLXArray(Float(0)).asType(dtype)
        let negInf = MLXArray(-Float.infinity).asType(dtype)
        return MLX.where(full, zero, negInf).reshaped(1, 1, rows, prefix + rows)
    }

    /// `[W, nKeep + 1]` row indices into `[convState (nKeep rows); window rows]`: the depthwise
    /// conv window of each row along its own root path.
    public func convWindows(nKeep: Int) -> MLXArray {
        var flat: [Int32] = []
        flat.reserveCapacity(rows * (nKeep + 1))
        for row in 0 ..< rows {
            let seq = Array(0 ..< nKeep) + paths[row].map { nKeep + $0 }
            flat.append(contentsOf: seq.suffix(nKeep + 1).map(Int32.init))
        }
        return MLXArray(flat).reshaped(rows, nKeep + 1)
    }

    /// Children of each row, in row order (= draft score order: best-first pop order).
    public var children: [[Int]] {
        var out = [[Int]](repeating: [], count: rows)
        for (row, parent) in parents.enumerated() where parent >= 0 { out[parent].append(row) }
        return out
    }
}

/// Per-request scope a tree verify forward runs inside: the plan, plus what each recurrent layer
/// left for the commit. Bound TASK-LOCALLY for the duration of one forward only, so a concurrent
/// forward of another model (or another request) on another task never sees it — a process-wide
/// static here would send an unrelated Qwen3.5-family forward down the tree branch.
public final class DFlash2TreeScope: @unchecked Sendable {
    @TaskLocal public static var current: DFlash2TreeScope?

    public let plan: DFlash2TreePlan
    /// Added to the cache offset for row positions (M-RoPE delta after a media prefill).
    public var positionDelta = 0
    /// Per recurrent layer cache identity: whatever its commit needs (q, k, v, a, b, conv source…).
    public var records: [ObjectIdentifier: [MLXArray]] = [:]
    private var maskCache: (prefix: Int, mask: MLXArray)?
    private var convCache: [Int: MLXArray] = [:]

    public init(plan: DFlash2TreePlan) { self.plan = plan }

    private var windowBitsCache: MLXArray?
    /// The tree's window visibility as MultiRowDecodeAttention window bits (built once per plan).
    public var multiRowWindowBits: MLXArray {
        if let windowBitsCache { return windowBitsCache }
        let bits = MultiRowDecodeAttention.windowBits(plan.windowVisibility)
        windowBitsCache = bits
        return bits
    }

    public func attentionMask(prefix: Int, dtype: DType) -> MLXArray {
        if let maskCache, maskCache.prefix == prefix, maskCache.mask.dtype == dtype {
            return maskCache.mask
        }
        let mask = plan.attentionMask(prefix: prefix, dtype: dtype)
        maskCache = (prefix, mask)
        return mask
    }

    public func convWindows(nKeep: Int) -> MLXArray {
        if let cached = convCache[nKeep] { return cached }
        let windows = plan.convWindows(nKeep: nKeep)
        convCache[nKeep] = windows
        return windows
    }

    public static func with<T>(_ scope: DFlash2TreeScope, _ body: () -> T) -> T {
        $current.withValue(scope) { body() }
    }
}

/// A target that can verify a draft tree and commit one root path of it.
public protocol DFlash2TreeVerifyModel {
    /// Forward the plan's rows through the target inside `scope`. Logits `[1, W, V]`; captured
    /// hiddens per requested layer `[1, W, H]`. Attention caches receive all W rows; recurrent
    /// layers leave their committed state untouched and record what the commit needs in `scope`.
    func dflash2TreeForward(
        _ inputs: MLXArray, cache: [KVCache], captureLayerIDs: Set<Int>, scope: DFlash2TreeScope
    ) -> (MLXArray, [Int: MLXArray])

    /// Keep exactly `keptRows` (a root path, root first) of the last tree window.
    func dflash2CommitTree(cache: [KVCache], scope: DFlash2TreeScope, keptRows: [Int]) -> Bool

    /// Whether every cache layer supports the tree commit (else the iterator keeps chains).
    func dflash2SupportsTree(cache: [KVCache]) -> Bool
}

/// The drafter's lattice for one round, on the host.
public struct DFlash2Lattice {
    /// `[D][K]` candidate token ids per depth.
    public let candidates: [[Int]]
    /// `[D][K]` candidate logits (unscaled).
    public let unary: [[Float]]
    /// `[K]` selector edge from the anchor to each depth-0 candidate.
    public let rootEdges: [Float]
    /// `[D-1][K][K]` edge from depth-d candidate i to depth-(d+1) candidate j.
    public let edges: [[[Float]]]

    public var depth: Int { candidates.count }
}

public enum DFlash2TreeSearch {
    /// TensorFold's fitted constants (`DFlashProposer.tree_*`).
    public static let tau = 1.5
    public static let edgeWeight = 0.6
    public static let children = 4

    /// Best-first tree over the lattice: node score = Σ log-softmax((unary/T + edge·E/T) / τ)
    /// along its path; pop the best frontier node until `maxNodes`. Returns (tokens, parents)
    /// in pop order with parent -1 = the anchor. Port of `best_first_tree` (no noise, no prior).
    public static func bestFirst(
        lattice: DFlash2Lattice, maxNodes: Int, temperature: Float
    ) -> (tokens: [Int], parents: [Int], scores: [Double]) {
        let t = temperature > 0 ? Double(temperature) : 1
        func normalized(depth: Int, edgeRow: [Float]) -> [Double] {
            let u = lattice.unary[depth]
            var s = (0 ..< u.count).map { j in
                (Double(u[j]) / t + edgeWeight * Double(edgeRow[j]) / t) / tau
            }
            let m = s.max() ?? 0
            let lse = m + log(s.reduce(0) { $0 + exp($1 - m) })
            for j in s.indices { s[j] -= lse }
            return s
        }
        struct Entry { let score: Double; let parent: Int; let depth: Int; let index: Int }
        var heap: [Entry] = []
        func push(_ e: Entry) {
            heap.append(e)
            var i = heap.count - 1
            while i > 0 {
                let p = (i - 1) / 2
                if heap[p].score >= heap[i].score { break }
                heap.swapAt(p, i)
                i = p
            }
        }
        func pop() -> Entry {
            let top = heap[0]
            let last = heap.removeLast()
            if !heap.isEmpty {
                heap[0] = last
                var i = 0
                while true {
                    let l = 2 * i + 1, r = l + 1
                    var m = i
                    if l < heap.count, heap[l].score > heap[m].score { m = l }
                    if r < heap.count, heap[r].score > heap[m].score { m = r }
                    if m == i { break }
                    heap.swapAt(m, i)
                    i = m
                }
            }
            return top
        }
        func expand(score: Double, parent: Int, depth: Int, edgeRow: [Float]) {
            let s = normalized(depth: depth, edgeRow: edgeRow)
            let order = s.indices.sorted { s[$0] > s[$1] }.prefix(children)
            for j in order {
                push(Entry(score: score + s[j], parent: parent, depth: depth, index: j))
            }
        }
        var tokens: [Int] = []
        var parents: [Int] = []
        var scores: [Double] = []
        var nodeDepthIndex: [(depth: Int, index: Int)] = []
        guard lattice.depth > 0, maxNodes > 0 else { return ([], [], []) }
        expand(score: 0, parent: -1, depth: 0, edgeRow: lattice.rootEdges)
        while !heap.isEmpty, tokens.count < maxNodes {
            let e = pop()
            tokens.append(lattice.candidates[e.depth][e.index])
            parents.append(e.parent)
            scores.append(e.score)
            nodeDepthIndex.append((e.depth, e.index))
            let me = tokens.count - 1
            if e.depth + 1 < lattice.depth {
                expand(
                    score: e.score, parent: me, depth: e.depth + 1,
                    edgeRow: lattice.edges[e.depth][e.index])
            }
        }
        return (tokens, parents, scores)
    }
}

public enum DFlash2TreeAcceptance {
    /// Greedy: follow the child whose token is the target's argmax at each row.
    public static func greedyPath(plan: DFlash2TreePlan, argmax: [Int]) -> (rows: [Int], bonus: Int) {
        let children = plan.children
        var path = [0]
        while true {
            let want = argmax[path.last!]
            guard let next = children[path.last!].first(where: { plan.tokens[$0] == want }) else {
                return (path, want)
            }
            path.append(next)
        }
    }

    /// Sampled, lossless: deterministic multi-candidate rejection. At each row the children are
    /// tried in draft order; child c is accepted with probability p'(c) where p' is the target
    /// distribution with every previously rejected sibling removed and renormalized; when all
    /// children are rejected (or the row is a leaf) the next token is sampled from p'. The
    /// emitted token at every position is an exact sample of the target distribution.
    public static func sampledPath(
        plan: DFlash2TreePlan, distributions: [[(id: Int, p: Double)]],
        rng: inout SpeculativeHostRNG
    ) -> (rows: [Int], bonus: Int, tried: Int, acceptedProbability: Double) {
        let children = plan.children
        var path = [0]
        var tried = 0
        var probabilitySum = 0.0
        while true {
            let row = path.last!
            var residual = distributions[row]
            var moved = false
            for child in children[row] {
                let token = plan.tokens[child]
                let total = residual.reduce(0) { $0 + $1.p }
                guard total > 0 else { break }
                let pc = (residual.first(where: { $0.id == token })?.p ?? 0) / total
                tried += 1
                probabilitySum += pc
                if rng.uniform() < pc {
                    path.append(child)
                    moved = true
                    break
                }
                residual.removeAll { $0.id == token }
            }
            if moved { continue }
            let bonus = residual.isEmpty ? rng.sample(distributions[row]) : rng.sample(residual)
            return (path, bonus, tried, probabilitySum)
        }
    }
}

/// A DFlash 2 target that can prefill a prompt carrying images/video through its vision path
/// while capturing the drafter's hidden-state taps, then continue with text-only drafting and
/// verify at the media-shifted (M-RoPE) positions. Without it a media request decodes plain AR.
public protocol DFlash2MediaPrefillModel {
    /// Full-prompt prefill of `input` (vision tower → merged embeddings → full-prompt M-RoPE
    /// positions) in chunks of `stepSize`; `onChunk` receives each chunk's captured hiddens.
    /// Returns the last row's logits `[1, 1, V]`.
    /// `boundary`: absolute prompt index where a chunk must end so the caller can snapshot the
    /// cache (`onBoundary`, after that chunk is evaluated) — the cross-turn checkpoint.
    /// `restoredPrefix` > 0: the cache already holds that many prompt tokens (a restored entry)
    /// and the remaining suffix carries no media placeholder; only the suffix is prefilled, at
    /// the full prompt's M-RoPE positions (recomputed from token ids + media grids, no vision).
    func dflash2MediaPrefill(
        _ input: LMInput, cache: [KVCache], captureLayerIDs: Set<Int>, stepSize: Int,
        restoredPrefix: Int, boundary: Int?, onBoundary: () -> Void,
        onChunk: ([Int: MLXArray]) -> Void
    ) throws -> MLXArray

    /// Text rows after a media prefill sit `delta` positions away from their cache offset
    /// (M-RoPE compresses a media span). 0 when the current request carries no media.
    var dflash2PositionDelta: Int { get }

    /// Clear media position state left by an EARLIER request (text-only requests must not
    /// inherit another request's M-RoPE delta).
    func dflash2ResetPositionState()
}
