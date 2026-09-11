// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// Host side of one ANE head program: owns the I/O planes and the head's
/// K/V window, packs a step's inputs, evals, and unpacks the results.
///
/// Positions are the head's own sequence positions (what the GPU head's
/// `KVCacheSimple.offset` counts). Slot `position % window` holds a position's
/// K/V; `length` is how many positions are live (committed + speculative).
/// `truncate(to:)` is the `trimHeadChain` equivalent: it just moves `length`.
public final class ANEHeadRunner {
    public let geometry: ANEHeadGeometry
    public let program: ANEProgram
    private let source: ANEHeadWeightSource
    public var embeddingTableBytes: Int { embeddingTable?.byteCount ?? 0 }

    private let aHidden: ANEPlane, bEmbed: ANEPlane, cK: ANEPlane, dV: ANEPlane
    private let eMask: ANEPlane, fCos: ANEPlane, gSin: ANEPlane
    private let oHidden: ANEPlane, oKNew: ANEPlane, oVNew: ANEPlane, oMax: ANEPlane, oIdx: ANEPlane

    /// Live head positions.
    public private(set) var length = 0
    /// Absolute position stored in each slot (−1 = empty).
    private var slotPosition: [Int]
    private var ropeCache: [Int: (cos: [Float16], sin: [Float16])] = [:]
    /// CPU embedding table (`VMLX_ANE_MTP_EMBED_TABLE=0` falls back to MLX lookups).
    private let embeddingTable: ANEEmbeddingTable?
    private var embedScratch: [Float16]

    /// A large negative that survives fp16 and still zeroes a softmax term.
    private static let maskedOut = Float16(-30000)

    public struct StepResult {
        public let token: Int
        public let hidden: [Float16]
    }

    public init(source: ANEHeadWeightSource, name: String = "ane-mtp-head",
                cacheDirectory: URL? = ANEProgram.cacheDirectory) throws {
        self.source = source
        let emission = ANEHeadEmitter.emit(from: source)
        self.geometry = emission.geometry
        let ins = emission.inputByteCounts.map { ANEPlane(byteCount: $0)! }
        let outs = emission.outputByteCounts.map { ANEPlane(byteCount: $0)! }
        aHidden = ins[0]; bEmbed = ins[1]; cK = ins[2]; dV = ins[3]; eMask = ins[4]; fCos = ins[5]; gSin = ins[6]
        oHidden = outs[0]; oKNew = outs[1]; oVNew = outs[2]; oMax = outs[3]; oIdx = outs[4]
        slotPosition = [Int](repeating: -1, count: geometry.window)
        embedScratch = [Float16](repeating: 0, count: geometry.hidden)
        embeddingTable = ProcessInfo.processInfo.environment["VMLX_ANE_MTP_EMBED_TABLE"] == "0"
            ? nil : ANEEmbeddingTable(source: source)
        program = try ANEProgram(name: name, mil: emission.mil, weights: emission.weights,
                                 inputs: ins, outputs: outs, cacheDirectory: cacheDirectory)
    }

    public func reset() {
        length = 0
        for i in slotPosition.indices { slotPosition[i] = -1 }
    }

    public func truncate(to newLength: Int) {
        precondition(newLength <= length)
        for s in slotPosition.indices where slotPosition[s] >= newLength { slotPosition[s] = -1 }
        length = newLength
    }

    // MARK: - packing

    private func embedding(_ token: Int) -> [Float16] {
        if let embeddingTable {
            embeddingTable.row(token, into: &embedScratch)
            return embedScratch
        }
        return source.embedding(token: token).asType(.float32).asArray(Float.self).map { Float16($0) }
    }

    private func rope(_ position: Int) -> (cos: [Float16], sin: [Float16]) {
        if let hit = ropeCache[position] { return hit }
        let (c, s) = source.ropeTables(positions: [position])
        let entry = (c.asType(.float32).asArray(Float.self).map { Float16($0) },
                     s.asType(.float32).asArray(Float.self).map { Float16($0) })
        ropeCache[position] = entry
        return entry
    }

    /// Writes row `r` of the tile: hidden, embedding, rope, and its mask row.
    private func packRow(_ r: Int, hidden: [Float16], embed: [Float16], position: Int, tileRows: Int) {
        let g = geometry, R = g.rows, W = g.window
        precondition(hidden.count == g.hidden && embed.count == g.hidden)
        for c in 0 ..< g.hidden {
            aHidden.fp16[c * R + r] = hidden[c]
            bEmbed.fp16[c * R + r] = embed[c]
        }
        let (cos, sin) = rope(position)
        let half = g.rotaryDims / 2
        for d in 0 ..< half {
            fCos.fp16[d * R + r] = cos[d]
            gSin.fp16[d * R + r] = sin[d]
        }
        // Mask row: window slots holding positions < this row's position, plus
        // in-tile rows 0...r (causal within the tile; self always visible).
        let base = r * (W + R)
        for s in 0 ..< W {
            let p = slotPosition[s]
            eMask.fp16[base + s] = (p >= 0 && p < position) ? 0 : Self.maskedOut
        }
        for t in 0 ..< R {
            eMask.fp16[base + W + t] = t <= r && t < tileRows ? 0 : Self.maskedOut
        }
    }

    /// Fills unused tile rows with something finite whose mask keeps softmax defined.
    private func packIdleRow(_ r: Int) {
        let g = geometry, R = g.rows, W = g.window
        for c in 0 ..< g.hidden {
            aHidden.fp16[c * R + r] = 0
            bEmbed.fp16[c * R + r] = 0
        }
        let base = r * (W + R)
        for s in 0 ..< W + R { eMask.fp16[base + s] = s == W + r ? 0 : Self.maskedOut }
    }

    /// Stores tile row `r`'s new K/V into the slot for `position`.
    private func storeRow(_ r: Int, position: Int) {
        let g = geometry, R = g.rows, W = g.window, KVH = g.kvHeads, HD = g.headDim
        let slot = position % W
        slotPosition[slot] = position
        for kvh in 0 ..< KVH {
            for d in 0 ..< HD {
                let ch = kvh * HD + d
                cK.fp16[(kvh * HD + d) * W + slot] = oKNew.fp16[ch * R + r]
                dV.fp16[(kvh * W + slot) * HD + d] = oVNew.fp16[ch * R + r]
            }
        }
    }

    private func readHidden(_ r: Int) -> [Float16] {
        let g = geometry, R = g.rows
        var h = [Float16](repeating: 0, count: g.hidden)
        for c in 0 ..< g.hidden { h[c] = oHidden.fp16[c * R + r] }
        return h
    }

    private func readArgmax(_ r: Int) -> Int {
        let R = geometry.rows, chunks = geometry.lmHeadChunks
        var best = -Float.infinity, bestIdx = 0
        for c in 0 ..< chunks {
            let m = Float(oMax.fp16[c * R + r])
            if m > best {
                best = m
                bestIdx = c * ANEHeadGeometry.lmHeadChunk + Int(Float(oIdx.fp16[c * R + r]))
            }
        }
        return bestIdx
    }

    // MARK: - steps

    /// One draft step at head position `length`: predicts the token after
    /// `token`, appends this position's K/V, returns the head hidden for the
    /// next chained step.
    public func draftStep(hidden: [Float16], token: Int) throws -> StepResult {
        let position = length
        packRow(0, hidden: hidden, embed: embedding(token), position: position, tileRows: 1)
        for r in 1 ..< geometry.rows { packIdleRow(r) }
        try program.eval()
        storeRow(0, position: position)
        length = position + 1
        return StepResult(token: readArgmax(0), hidden: readHidden(0))
    }

    /// The aligned commit folded into the first draft: pairs `0 ..< n-1` are
    /// confirmed (trunk hidden, token) rows, pair `n-1` seeds the chain. One
    /// tile eval stores every row's K/V and returns the last row's draft.
    public func draftChainStart(pairs: [(hidden: [Float16], token: Int)]) throws -> StepResult {
        precondition(!pairs.isEmpty && pairs.count <= geometry.rows)
        let start = length
        for (r, pair) in pairs.enumerated() {
            packRow(r, hidden: pair.hidden, embed: embedding(pair.token), position: start + r, tileRows: pairs.count)
        }
        for r in pairs.count ..< geometry.rows { packIdleRow(r) }
        try program.eval()
        for r in 0 ..< pairs.count { storeRow(r, position: start + r) }
        length = start + pairs.count
        let last = pairs.count - 1
        return StepResult(token: readArgmax(last), hidden: readHidden(last))
    }

    /// Commits up to `rows` (hidden, token) pairs at positions `length...`
    /// in ONE tile eval (the aligned head-cache commit).
    public func commit(pairs: [(hidden: [Float16], token: Int)]) throws {
        precondition(pairs.count <= geometry.rows)
        guard !pairs.isEmpty else { return }
        let start = length
        for (r, pair) in pairs.enumerated() {
            packRow(r, hidden: pair.hidden, embed: embedding(pair.token), position: start + r, tileRows: pairs.count)
        }
        for r in pairs.count ..< geometry.rows { packIdleRow(r) }
        try program.eval()
        for r in 0 ..< pairs.count { storeRow(r, position: start + r) }
        length = start + pairs.count
    }
}
