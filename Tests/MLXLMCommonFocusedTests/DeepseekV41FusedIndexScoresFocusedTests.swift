// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 fused indexer scores (`DeepseekV41FusedIndexScores`, port of vMLX
// `vmlx_engine/metal/dsa_index_scores.py` `dsa_pool_index_scores`). Small synthetic tensors on the
// GPU (the kernel is Metal); no model loads and no shared MLX test lock, so run this suite on its own.
//
// Python parity (tests/test_dsa_fused_index_scores.py): fp32 accuracy against an fp64 reference
// (2e-6 of the largest score), rows bit-identical however many queries share the call and whatever
// the tile height, and nil (stock path) for decode / unsupported inputs.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("DeepSeek-V4.1 fused indexer scores", .serialized)
struct DeepseekV41FusedIndexScoresFocusedTests {

    static func normals(_ n: Int, seed: UInt64, scale: Float = 1) -> [Float] {
        var rng = SplitMix64(seed: seed)
        func uniform() -> Double { (Double(rng.next() >> 11) + 0.5) / Double(1 << 53) }
        return (0..<n).map { _ in
            Float((-2 * log(uniform())).squareRoot() * cos(2 * Double.pi * uniform())) * scale
        }
    }

    /// fp64 `out[s, p] = sum_h w[s, h] * relu(scale * q[s, h, :] . k[p, :])`; q [S, H, D], k [P, D], w [S, H].
    static func reference(
        q: [Float], k: [Float], w: [Float], S: Int, P: Int, H: Int, D: Int, scale: Double = 1
    ) -> [Double] {
        var out = [Double](repeating: 0, count: S * P)
        q.withUnsafeBufferPointer { qp in
            k.withUnsafeBufferPointer { kp in
                for s in 0..<S {
                    for p in 0..<P {
                        var total = 0.0
                        for h in 0..<H {
                            var dot = 0.0
                            let qo = (s * H + h) * D
                            let ko = p * D
                            for d in 0..<D { dot += Double(qp[qo + d]) * Double(kp[ko + d]) }
                            total += Double(w[s * H + h]) * max(dot * scale, 0)
                        }
                        out[s * P + p] = total
                    }
                }
            }
        }
        return out
    }

    static func inputs(S: Int, P: Int, H: Int, D: Int = 128, seed: UInt64)
        -> (q: [Float], k: [Float], w: [Float])
    {
        (normals(S * H * D, seed: seed), normals(P * D, seed: seed &+ 1),
         normals(S * H, seed: seed &+ 2, scale: 0.25))
    }

    static func fused(
        _ x: (q: [Float], k: [Float], w: [Float]), S: Int, P: Int, H: Int, D: Int = 128,
        scale: Float = 1, rowsFrags: Int? = nil
    ) -> MLXArray? {
        DeepseekV41FusedIndexScores.poolIndexScores(
            q: MLXArray(x.q, [1, S, H, D]), poolKeys: MLXArray(x.k, [1, P, D]),
            headW: MLXArray(x.w, [1, S, H]), scale: scale, rowsFrags: rowsFrags)
    }

    @Test("fused scores equal sum_h w * relu(q . k) at fp32 accuracy (tails, short calls, scale)")
    func matchesFP64Reference() throws {
        try Device.withDefaultDevice(.gpu) {
            // (queries, keys, heads, scale): partial row/key tiles, 16-row tiles (< 32 rows), padded
            // 64-row tiles (32..63 rows), one exact key tile, and the HAS_SCALE path.
            let cases: [(Int, Int, Int, Float)] = [
                (100, 333, 32, 1), (64, 32, 32, 1), (65, 97, 32, 1), (2, 40, 32, 1), (40, 77, 32, 1),
                (31, 1029, 8, 1), (513, 200, 4, 1), (100, 300, 32, Float(pow(128.0, -0.5))),
            ]
            for (S, P, H, scale) in cases {
                let x = Self.inputs(S: S, P: P, H: H, seed: UInt64(S * 7919 + P))
                let out = try #require(Self.fused(x, S: S, P: P, H: H, scale: scale), "S\(S) P\(P)")
                #expect(out.shape == [1, S, P] && out.dtype == .float32)
                let got = out.asArray(Float.self)
                let ref = Self.reference(
                    q: x.q, k: x.k, w: x.w, S: S, P: P, H: H, D: 128, scale: Double(scale))
                let peak = ref.map(abs).max() ?? 0
                let err = zip(got, ref).map { abs(Double($0) - $1) }.max() ?? 0
                #expect(peak > 0 && err <= 2e-6 * peak, "S\(S) P\(P) H\(H): max err \(err) vs peak \(peak)")
            }
        }
    }

    @Test("rows do not depend on how many queries share the call or on the tile height")
    func rowsIndependentOfCallAndTile() throws {
        try Device.withDefaultDevice(.gpu) {
            let S = 256, P = 700, H = 16, D = 128
            let x = Self.inputs(S: S, P: P, H: H, seed: 7)
            let q = MLXArray(x.q, [1, S, H, D]), k = MLXArray(x.k, [1, P, D]), w = MLXArray(x.w, [1, S, H])
            let full = try #require(DeepseekV41FusedIndexScores.poolIndexScores(
                q: q, poolKeys: k, headW: w, scale: 1))
            for (a, b) in [(250, 256), (100, 133), (0, 64), (5, 9), (31, 62)] {
                let part = try #require(DeepseekV41FusedIndexScores.poolIndexScores(
                    q: q[0..., a ..< b], poolKeys: k, headW: w[0..., a ..< b], scale: 1))
                #expect(arrayEqual(part, full[0..., a ..< b]).item(Bool.self), "rows \(a)..<\(b)")
            }
            // Under 32 rows the call runs 16-row tiles; every row equals the 64-row tile's.
            #expect([2, 16, 31, 32, 63, 64, 2048].map(DeepseekV41FusedIndexScores.rowsFrags)
                == [1, 1, 1, 4, 4, 4, 4])
            let y = Self.inputs(S: 31, P: 4099, H: 8, seed: 3)
            let sized = try #require(Self.fused(y, S: 31, P: 4099, H: 8))
            let tall = try #require(Self.fused(y, S: 31, P: 4099, H: 8, rowsFrags: 4))
            #expect(arrayEqual(sized, tall).item(Bool.self))
        }
    }

    @Test("decode, unsupported inputs and non-GPU devices fall back (nil)")
    func fallsBack() {
        let x = Self.inputs(S: 64, P: 500, H: 8, seed: 11)
        Device.withDefaultDevice(.gpu) {
            let one = Self.inputs(S: 1, P: 500, H: 8, seed: 12)
            #expect(Self.fused(one, S: 1, P: 500, H: 8) == nil)  // decode: one query row
            #expect(Self.fused(x, S: 64, P: 500, H: 8) != nil)
            let q = MLXArray(x.q, [1, 64, 8, 128]), k = MLXArray(x.k, [1, 500, 128])
            let w = MLXArray(x.w, [1, 64, 8])
            #expect(DeepseekV41FusedIndexScores.poolIndexScores(
                q: q.asType(.bfloat16), poolKeys: k, headW: w, scale: 1) == nil)
            #expect(DeepseekV41FusedIndexScores.poolIndexScores(
                q: q, poolKeys: k[0..., 0 ..< 31], headW: w, scale: 1) == nil)  // < one key tile
            #expect(DeepseekV41FusedIndexScores.poolIndexScores(
                q: q[0..., 0..., 0..., 0 ..< 12], poolKeys: k[0..., 0..., 0 ..< 12], headW: w,
                scale: 1) == nil)  // head dim not a multiple of 8
        }
        Device.withDefaultDevice(.cpu) {
            #expect(Self.fused(x, S: 64, P: 500, H: 8) == nil)
            #expect(!DeepseekV41FusedIndexScores.eligible(queries: 2048, keys: 100_000))
        }
    }

    @Test("chunk rule max(16, min(S, 2^28 / T)) and the prefill-only gate")
    func chunkRule() {
        typealias F = DeepseekV41FusedIndexScores
        #expect(F.queryChunk(queries: 2048, keys: 100_000) == 2048)
        #expect(F.queryChunk(queries: 512, keys: 100_000) == 512)
        #expect(F.queryChunk(queries: 2048, keys: 200_000) == 1342)
        #expect(F.queryChunk(queries: 2048, keys: 1_000_000) == 268)
        #expect(F.queryChunk(queries: 2048, keys: 1 << 25) == 16)  // 2^28 / T = 8 -> floor of 16
        #expect(F.queryChunk(queries: 9, keys: 600) == 16)
        // The stock rule (decode, dSpark verify, fallback) is unchanged: 20 rows at 100k keys.
        #expect(DeepseekV41IndexerMath.queryChunk(heads: 32, rows: 100_000) == 20)
        #expect(DeepseekV41IndexerMath.queryChunk(heads: 32, rows: 600) == 512)
        Device.withDefaultDevice(.gpu) {
            #expect(F.enabled)  // VMLX_DSV41_FUSED_INDEX unset in the test environment
            #expect(!F.eligible(queries: 1, keys: 100_000))  // decode
            #expect(!F.eligible(queries: 8, keys: 100_000))  // dSpark verify (<= 8 rows)
            #expect(F.eligible(queries: 9, keys: 100_000))
            #expect(!F.eligible(queries: 2048, keys: 31))
            #expect(F.eligible(queries: 2048, keys: 32))
        }
    }

    @Test("top-k on random data matches the stock bf16 path except at exact ties")
    func topKMatchesStockExceptTies() throws {
        try Device.withDefaultDevice(.gpu) {
            // Exact regime: q, k in {-1, 0, 1} and w in {0, +-1, +-2^-1, +-2^-5, +-2^-6}, so every dot
            // product, relu and weight product is exact in fp32 AND in the stock path's bf16
            // intermediates, and the fp32 head sum is exact. The stock score is then bf16(exact score):
            // a monotone rounding that merges nearby scores (11+ significant bits -> 8), so its top-k
            // can differ from the exact (fused) top-k only among rows whose stock scores tie exactly.
            let C = 100, T = 3000, H = 32, D = 128, k = 512, startPos = T - C
            var rng = SplitMix64(seed: 21)
            let tern: () -> Float = { [-1, 0, 0, 1][Int(rng.next() % 4)] }
            let q = (0..<(C * H * D)).map { _ in tern() }
            let kk = (0..<(T * D)).map { _ in tern() }
            let wSet: [Float] = [0, 1, -1, 1 / 2, -1 / 2, 1 / 32, -1 / 32, 1 / 64, -1 / 64]
            let w = (0..<(C * H)).map { _ in wSet[Int(rng.next() % UInt64(wSet.count))] }
            let lens = DeepseekV41IndexerMath.visibleLengths(
                startPos: startPos, chunkStart: 0, count: C, ratio: 1)

            let fusedRaw = try #require(DeepseekV41FusedIndexScores.poolIndexScores(
                q: MLXArray(q, [1, C, H, D]), poolKeys: MLXArray(kk, [1, T, D]),
                headW: MLXArray(w, [1, C, H]), scale: 1))[0]
            // Every intermediate is exact in fp32, so a plain fp32 CPU evaluation is the exact score.
            let exact = Device.withDefaultDevice(.cpu) {
                (maximum(matmul(MLXArray(q, [C, H, D]), MLXArray(kk, [T, D]).transposed()), Float(0))
                    * MLXArray(w, [C, H]).expandedDimensions(axis: -1)).sum(axis: 1).asArray(Float.self)
            }
            #expect(fusedRaw.asArray(Float.self) == exact, "fused must be exact here")

            let stockRaw = DeepseekV41IndexerMath.scores(
                q: MLXArray(q, [C, H, D]), k: MLXArray(kk, [T, D]), w: MLXArray(w, [C, H]))
            let fusedSc = DeepseekV41IndexerMath.maskInvisible(fusedRaw, lens: lens)
            let stockSc = DeepseekV41IndexerMath.maskInvisible(stockRaw, lens: lens)
            let fusedIdx = DeepseekV41IndexerMath.topK(fusedSc, k: k).asArray(Int32.self)
            let stockIdx = DeepseekV41IndexerMath.topK(stockSc, k: k).asArray(Int32.self)
            let stock = stockSc.asArray(Float.self)
            let fusedV = fusedSc.asArray(Float.self)

            var differingRows = 0
            for c in 0..<C {
                let a = Set(fusedIdx[(c * k) ..< ((c + 1) * k)])
                let b = Set(stockIdx[(c * k) ..< ((c + 1) * k)])
                #expect(a.count == k && b.count == k)
                let row = Array(stock[(c * T) ..< ((c + 1) * T)])
                let threshold = row.sorted(by: >)[k - 1]
                let diff = a.symmetricDifference(b)
                if !diff.isEmpty { differingRows += 1 }
                #expect(diff.allSatisfy { row[Int($0)] == threshold },
                    "query \(c): ids \(diff.sorted()) are not exact stock ties at \(threshold)")
                // The fused selection is an exact top-k of its own scores.
                let fRow = Array(fusedV[(c * T) ..< ((c + 1) * T)])
                let keptMin = a.map { fRow[Int($0)] }.min() ?? .infinity
                let droppedMax = (0..<T).filter { !a.contains(Int32($0)) }.map { fRow[$0] }.max()
                    ?? -.infinity
                #expect(keptMin >= droppedMax, "query \(c)")
            }
            // Not vacuous: bf16 rounding merges near-threshold scores, so some rows do differ.
            print("[fused-index] rows whose top-k differs only at stock ties: \(differingRows)/\(C)")
            #expect(differingRows > 0)
        }
    }

    @Test("end-to-end select() on the GPU (fused path) matches the loop reference")
    func selectFusedMatchesReference() throws {
        try Device.withDefaultDevice(.gpu) {
            let cfg = DeepseekV41IndexerFocusedTests.tinyConfig()
            let rope = DeepseekV4RoPE(
                dim: 8, base: 160_000, factor: 1, origMaxPos: 65536, betaFast: 32, betaSlow: 1)
            let owner = DeepseekV41Indexer(config: cfg, layerIdx: 2, compressRatio: 1)
            let user = DeepseekV41Indexer(config: cfg, layerIdx: 4, compressRatio: 1)
            for m in [owner, user] {
                var params: [String: MLXArray] = [:]
                for (i, (key, value)) in m.parameters().flattened().enumerated() {
                    params[key] = MLXArray(
                        DeepseekV41IndexerFocusedTests.values(value.size, seed: UInt64(10 + i), scale: 0.25),
                        value.shape)
                }
                try m.update(parameters: ModuleParameters.unflattened(params), verify: .noUnusedKeys)
            }
            let L = 40, startPos = 9, T = startPos + L  // ratio 1: one group per token
            #expect(DeepseekV41FusedIndexScores.eligible(queries: L, keys: T))
            let latent = MLXArray(
                DeepseekV41IndexerFocusedTests.values(T * cfg.headDim, seed: 40, scale: 0.25),
                [1, T, cfg.headDim])
            let indexK = owner.indexKeys(latent: latent, base: 0, rope: rope)
            let x = MLXArray(
                DeepseekV41IndexerFocusedTests.values(L * cfg.hiddenSize, seed: 41), [1, L, cfg.hiddenSize])
            let qr = MLXArray(
                DeepseekV41IndexerFocusedTests.values(L * cfg.qLoraRank, seed: 42), [1, L, cfg.qLoraRank])
            let view = DeepseekV4PoolView(dense: indexK)

            let ownerSel = owner.select(
                x: x, qResidual: qr, indexK: view, startPos: startPos, rope: rope, candidates: nil)
            let cand = try #require(ownerSel.candidates)
            let ownerIdx = try #require(ownerSel.topK)
            #expect(ownerIdx.shape == [1, L, cfg.indexTopk] && cand.shape == [L, T])
            let userIdx = try #require(user.select(
                x: x, qResidual: qr, indexK: view, startPos: startPos, rope: rope, candidates: cand).topK)

            func referenceScores(_ m: DeepseekV41Indexer, candidates: [[Bool]]?) -> [[Float?]] {
                var q = m.wqB(qr).reshaped(1, L, cfg.indexNHeads, cfg.indexHeadDim).transposed(0, 2, 1, 3)
                let (cosT, sinT) = rope.cosSin(offset: startPos, length: L)
                q = DeepseekV4Math.applyPartialRoPE(
                    q, cos: cosT.expandedDimensions(axes: [0, 1]), sin: sinT.expandedDimensions(axes: [0, 1]),
                    ropeDim: 8)
                let qv = q.transposed(0, 2, 1, 3)[0].asType(.bfloat16).asType(.float32).asArray(Float.self)
                let wv = (m.weightsProj(x) * MLXArray(pow(Float(16), -0.5) * pow(Float(4), -0.5)))[0]
                    .asType(.bfloat16).asType(.float32).asArray(Float.self)
                let kv = indexK[0].asType(.bfloat16).asType(.float32).asArray(Float.self)
                var sc = DeepseekV41IndexerFocusedTests.referenceScores(
                    q: qv, k: kv, w: wv, C: L, h: 4, d: 16, T: T, startPos: startPos, ratio: 1)
                if let candidates {
                    for c in 0..<L { for t in 0..<T where !candidates[c][t] { sc[c][t] = nil } }
                }
                return sc
            }
            func expectValidTopK(_ idx: MLXArray, _ ref: [[Float?]], label: String) {
                let flat = idx.asArray(Int32.self)
                let k = idx.dim(-1)
                for c in 0..<L {
                    let ids = Array(flat[(c * k) ..< ((c + 1) * k)])
                    #expect(Set(ids).count == k, "\(label) query \(c): ids must be unique")
                    let finite = ref[c].compactMap { $0 }.sorted(by: >)
                    let kept = ids.compactMap { ref[c][Int($0)] }
                    let want = min(k, finite.count)
                    #expect(kept.count == want, "\(label) query \(c): kept \(kept.count) of \(want)")
                    if want > 0 {
                        // fp32 scores vs the fp32 loop: summation order only.
                        let threshold = finite[want - 1]
                        #expect(kept.allSatisfy { $0 >= threshold - 1e-4 * max(1, abs(threshold)) },
                            "\(label) query \(c): \(kept) vs threshold \(threshold)")
                    }
                }
            }
            let refOwner = referenceScores(owner, candidates: nil)
            let refCand = DeepseekV41IndexerFocusedTests.referenceCandidates(
                refOwner.map { $0.map { $0 ?? -.infinity } }, lens: (0..<L).map { startPos + $0 + 1 },
                topkBlocks: cfg.candidateTopkBlocks, blockSize: cfg.candidateBlockSize)
            #expect(cand.asArray(Bool.self) == refCand.flatMap { $0 })
            expectValidTopK(ownerIdx, refOwner, label: "owner")
            expectValidTopK(userIdx, referenceScores(user, candidates: refCand), label: "user")
        }
    }
}
