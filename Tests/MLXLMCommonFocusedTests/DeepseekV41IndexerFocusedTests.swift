// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 shared lightning indexer: the selection math against a straightforward loop
// re-implementation of jang_tools/dsv41/model.py (`Indexer.__call__`, `select_candidate_blocks`,
// `topk_indices`) on small synthetic tensors. MLX CPU device only.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("DeepSeek-V4.1 indexer selection")
struct DeepseekV41IndexerFocusedTests {

    static func values(_ n: Int, seed: UInt64, scale: Float = 1) -> [Float] {
        var rng = SplitMix64(seed: seed)
        // Multiples of 1/16 in [-2, 2]: exact in bf16, so bf16 rounding cannot flip a comparison.
        return (0..<n).map { _ in Float(Int(rng.next() % 65) - 32) / 16 * scale }
    }

    /// Loop reference of the reference `Indexer` scoring + causal mask: [C][T] (nil = -inf).
    static func referenceScores(
        q: [Float], k: [Float], w: [Float], C: Int, h: Int, d: Int, T: Int,
        startPos: Int, ratio: Int
    ) -> [[Float?]] {
        (0..<C).map { c in
            let lens = (startPos + c + 1) / ratio
            return (0..<T).map { t -> Float? in
                guard t < lens else { return nil }
                var total: Float = 0
                for hh in 0..<h {
                    var dot: Float = 0
                    for dd in 0..<d { dot += q[(c * h + hh) * d + dd] * k[t * d + dd] }
                    total += max(dot, 0) * w[c * h + hh]
                }
                return total
            }
        }
    }

    @Test("head-weighted ReLU scores and causal lengths match the loop reference")
    func scoresMatchReference() {
        Device.withDefaultDevice(.cpu) {
            let C = 5, h = 3, d = 8, T = 11, startPos = 6, ratio = 2
            let q = Self.values(C * h * d, seed: 1, scale: 0.5)
            let k = Self.values(T * d, seed: 2, scale: 0.5)
            let w = Self.values(C * h, seed: 3, scale: 0.25)
            var sc = DeepseekV41IndexerMath.scores(
                q: MLXArray(q, [C, h, d]), k: MLXArray(k, [T, d]), w: MLXArray(w, [C, h]))
            let lens = DeepseekV41IndexerMath.visibleLengths(
                startPos: startPos, chunkStart: 0, count: C, ratio: ratio)
            #expect(lens.reshaped(-1).asArray(Int32.self) == (0..<C).map { Int32((startPos + $0 + 1) / ratio) })
            sc = DeepseekV41IndexerMath.maskInvisible(sc, lens: lens)
            let got = sc.asArray(Float.self)
            let ref = Self.referenceScores(
                q: q, k: k, w: w, C: C, h: h, d: d, T: T, startPos: startPos, ratio: ratio)
            for c in 0..<C {
                for t in 0..<T {
                    let g = got[c * T + t]
                    if let r = ref[c][t] {
                        #expect(abs(g - r) <= max(0.02, abs(r) * 0.01), "c\(c) t\(t): \(g) vs \(r)")
                    } else {
                        #expect(g == -Float.infinity, "c\(c) t\(t) must be masked")
                    }
                }
            }
        }
    }

    /// Loop reference of `select_candidate_blocks`.
    static func referenceCandidates(
        _ logits: [[Float]], lens: [Int], topkBlocks: Int, blockSize: Int
    ) -> [[Bool]] {
        let T = logits[0].count
        let nb = (T + blockSize - 1) / blockSize
        return logits.enumerated().map { c, row in
            var blocks = (0..<nb).map { b -> Float in
                (b * blockSize ..< min(T, (b + 1) * blockSize)).map { row[$0] }.max() ?? -.infinity
            }
            blocks[(lens[c] - 1) / blockSize] = .infinity
            let keepBlock: [Bool]
            if topkBlocks >= nb {
                keepBlock = blocks.map { $0 > -.infinity }
            } else {
                let order = blocks.enumerated().sorted { $0.element > $1.element }.prefix(topkBlocks)
                var k = [Bool](repeating: false, count: nb)
                for (b, v) in order where v > -.infinity { k[b] = true }
                keepBlock = k
            }
            return (0..<T).map { keepBlock[$0 / blockSize] }
        }
    }

    @Test("candidate blocks match the loop reference (top-k blocks and all-visible cases)")
    func candidatesMatchReference() {
        Device.withDefaultDevice(.cpu) {
            let C = 4, T = 37, block = 4
            let lensList = [9, 20, 33, 37]
            var rng = SplitMix64(seed: 5)
            // Distinct values so block ordering has no ties.
            var logits: [[Float]] = (0..<C).map { _ in
                (0..<T).map { _ in Float(rng.next() % 1_000_000) / 1000 }
            }
            for c in 0..<C { for t in lensList[c]..<T { logits[c][t] = -.infinity } }
            let lens = MLXArray(lensList.map(Int32.init), [C, 1])
            for topkBlocks in [3, 50] {
                let got = DeepseekV41IndexerMath.candidateBlocks(
                    MLXArray(logits.flatMap { $0 }, [C, T]), lens: lens,
                    topkBlocks: topkBlocks, blockSize: block)
                #expect(got.shape == [C, T])
                let ref = Self.referenceCandidates(
                    logits, lens: lensList, topkBlocks: topkBlocks, blockSize: block)
                #expect(got.asArray(Bool.self) == ref.flatMap { $0 }, "topkBlocks \(topkBlocks)")
            }
        }
    }

    @Test("top-k returns the unique k best ids, or arange when k >= T")
    func topKMatchesReference() {
        Device.withDefaultDevice(.cpu) {
            let T = 30, k = 7
            var rng = SplitMix64(seed: 8)
            let row = (0..<T).map { _ in Float(rng.next() % 1_000_000) }
            let got = Set(DeepseekV41IndexerMath.topK(MLXArray(row, [1, T]), k: k).asArray(Int32.self))
            let ref = Set(row.enumerated().sorted { $0.element > $1.element }.prefix(k).map { Int32($0.offset) })
            #expect(got == ref)
            let all = DeepseekV41IndexerMath.topK(MLXArray(row, [1, T]), k: 64)
            #expect(all.asArray(Int32.self) == (0..<T).map(Int32.init))
        }
    }

    static func tinyConfig() -> DeepseekV4Configuration {
        var c = DeepseekV4Configuration()
        c.hiddenSize = 20
        c.qLoraRank = 12
        c.headDim = 32
        c.indexNHeads = 4
        c.indexHeadDim = 16
        c.indexTopk = 6
        c.kvSourceLayerIds = [2]
        c.indexSourceLayerIds = [2, 4]
        c.candidateSourceLayerId = 2
        c.candidateTopkBlocks = 3
        c.candidateBlockSize = 4
        return c
    }

    @Test("block top-k on long rows selects exactly the values the direct partition selects")
    func blockTopKMatchesDirect() {
        Device.withDefaultDevice(.cpu) {
            for (rows, length) in [(5, 70000), (3, 140001), (2, 65536)] {
                let noise = MLXRandom.normal([rows, length], key: MLXRandom.key(UInt64(length)))
                let limit = MLXArray(Int32(0)..<Int32(rows)) + Int32(length - 5)
                let masked = MLX.where(
                    MLXArray(Int32(0)..<Int32(length)).reshaped(1, length) .>= limit.reshaped(rows, 1),
                    MLXArray(-Float.infinity), noise)
                let block = DeepseekV41IndexerMath.topK(masked, k: 512)
                let direct = argPartition(-masked, kth: 511, axis: -1)[0..., 0..<512]
                #expect(block.shape == [rows, 512])
                #expect(block.dtype == .int32)
                let vb = sorted(takeAlong(masked, block, axis: -1), axis: -1)
                let vd = sorted(takeAlong(masked, direct, axis: -1), axis: -1)
                #expect(arrayEqual(vb, vd).item(Bool.self), "rows \(rows) length \(length)")
            }
        }
    }

    @Test("end-to-end select(): owner keys + candidate source + candidate user match the reference")
    func selectMatchesReference() throws {
        try Device.withDefaultDevice(.cpu) {
            let cfg = Self.tinyConfig()
            let rope = DeepseekV4RoPE(
                dim: 8, base: 160_000, factor: 1, origMaxPos: 65536, betaFast: 32, betaSlow: 1)
            let owner = DeepseekV41Indexer(config: cfg, layerIdx: 2, compressRatio: 1)
            let user = DeepseekV41Indexer(config: cfg, layerIdx: 4, compressRatio: 1)
            #expect(owner.ownsK && owner.isCandidateSource && !owner.usesCandidates)
            #expect(!user.ownsK && user.usesCandidates)
            for m in [owner, user] {
                var params: [String: MLXArray] = [:]
                for (i, (key, value)) in m.parameters().flattened().enumerated() {
                    params[key] = MLXArray(Self.values(value.size, seed: UInt64(10 + i), scale: 0.25), value.shape)
                }
                try m.update(parameters: ModuleParameters.unflattened(params), verify: .noUnusedKeys)
            }

            let L = 23, startPos = 9, T = startPos + L  // ratio 1: one group per token
            let latent = MLXArray(Self.values(T * cfg.headDim, seed: 40, scale: 0.25), [1, T, cfg.headDim])
            let indexK = owner.indexKeys(latent: latent, base: 0, rope: rope)  // [1, T, 16]
            #expect(indexK.shape == [1, T, cfg.indexHeadDim])
            let x = MLXArray(Self.values(L * cfg.hiddenSize, seed: 41), [1, L, cfg.hiddenSize])
            let qr = MLXArray(Self.values(L * cfg.qLoraRank, seed: 42), [1, L, cfg.qLoraRank])
            let view = DeepseekV4PoolView(dense: indexK)

            let ownerSel = owner.select(
                x: x, qResidual: qr, indexK: view, startPos: startPos, rope: rope, candidates: nil)
            let cand = try #require(ownerSel.candidates)
            let ownerIdx = try #require(ownerSel.topK)
            #expect(ownerIdx.shape == [1, L, cfg.indexTopk])
            #expect(cand.shape == [L, T])
            let userSel = user.select(
                x: x, qResidual: qr, indexK: view, startPos: startPos, rope: rope, candidates: cand)
            let userIdx = try #require(userSel.topK)
            #expect(userSel.candidates == nil)

            // Reference: rebuild q/w exactly as the reference does and run the loop math.
            func reference(_ m: DeepseekV41Indexer, candidates: [[Bool]]?) -> (scores: [[Float?]], cand: [[Bool]]) {
                var q = m.wqB(qr).reshaped(1, L, cfg.indexNHeads, cfg.indexHeadDim).transposed(0, 2, 1, 3)
                let (cosT, sinT) = rope.cosSin(offset: startPos, length: L)
                q = DeepseekV4Math.applyPartialRoPE(
                    q, cos: cosT.expandedDimensions(axes: [0, 1]), sin: sinT.expandedDimensions(axes: [0, 1]),
                    ropeDim: 8)
                let qv = q.transposed(0, 2, 1, 3)[0].asType(.bfloat16).asType(.float32).asArray(Float.self)
                let wv = (m.weightsProj(x) * MLXArray(pow(Float(16), -0.5) * pow(Float(4), -0.5)))[0]
                    .asType(.bfloat16).asType(.float32).asArray(Float.self)
                let kv = indexK[0].asType(.bfloat16).asType(.float32).asArray(Float.self)
                var sc = Self.referenceScores(
                    q: qv, k: kv, w: wv, C: L, h: 4, d: 16, T: T, startPos: startPos, ratio: 1)
                let lens = (0..<L).map { startPos + $0 + 1 }
                var candOut: [[Bool]] = []
                if m.isCandidateSource {
                    candOut = Self.referenceCandidates(
                        sc.map { $0.map { $0 ?? -.infinity } }, lens: lens,
                        topkBlocks: cfg.candidateTopkBlocks, blockSize: cfg.candidateBlockSize)
                } else if let candidates {
                    for c in 0..<L { for t in 0..<T where !candidates[c][t] { sc[c][t] = nil } }
                }
                return (sc, candOut)
            }

            /// The selection is a valid top-k of the reference scores: every finite-score row it
            /// keeps clears the k-th best reference score, and it keeps min(k, #finite) of them.
            /// (Rows the reference scores -inf — invisible, or outside the candidates — may fill the
            /// remainder in either implementation; their tie order is unspecified.)
            func expectValidTopK(_ idx: MLXArray, _ ref: [[Float?]], label: String) {
                let flat = idx.asArray(Int32.self)
                let k = idx.dim(-1)
                for c in 0..<L {
                    let ids = Array(flat[(c * k)..<((c + 1) * k)])
                    #expect(Set(ids).count == k, "\(label) query \(c): ids must be unique")
                    let finite = ref[c].compactMap { $0 }.sorted(by: >)
                    let keptFinite = ids.compactMap { ref[c][Int($0)] }
                    let want = min(k, finite.count)
                    #expect(keptFinite.count == want, "\(label) query \(c): kept \(keptFinite.count) of \(want)")
                    if want > 0 {
                        let threshold = finite[want - 1]
                        #expect(keptFinite.allSatisfy { $0 >= threshold - 0.02 },
                            "\(label) query \(c): \(keptFinite) vs threshold \(threshold)")
                    }
                }
            }

            let refOwner = reference(owner, candidates: nil)
            #expect(cand.asArray(Bool.self) == refOwner.cand.flatMap { $0 })
            expectValidTopK(ownerIdx, refOwner.scores, label: "owner")
            let refUser = reference(user, candidates: refOwner.cand)
            expectValidTopK(userIdx, refUser.scores, label: "user")

            // Budget not exceeded -> nil (attend every visible row), like the reference arange top-k.
            let small = owner.select(
                x: x[0..., 0..<2], qResidual: qr[0..., 0..<2], indexK: DeepseekV4PoolView(dense: indexK[0..., 0..<4]),
                startPos: 2, rope: rope, candidates: nil)
            #expect(small.topK == nil)
        }
    }
}
