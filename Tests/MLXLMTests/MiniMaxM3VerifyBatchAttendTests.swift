// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The DSpark verify's batched sparse attention (`MiniMaxM3Attention.verifyRows`, `VMLX_MINIMAX_VERIFY_BATCH_ATTEND`):
/// each row keeps ITS OWN selection (a decode step's selection over its own index-key prefix), and the rows' selections
/// go into ONE block-sparse decode-kernel launch. Every row must equal its one-row call (the decode step) bit for bit.
///
/// M3.1 geometry (64 heads, 4 KV groups, head dim 128, index dim 128, blocks of 128, top-16, local 1), contexts that
/// are not multiples of the block (128), the K / V step (256) or the chunk (2048), rows 2...16 including a row set that
/// crosses a block boundary. Opt-in timing (`VMLX_M3_BATCH_ATTEND_BENCH=1`): one layer's attention of T verify rows,
/// per-row calls vs one launch, selection included and not.
@Suite("MiniMax-M3.1 verify: batched sparse attention", .serialized)
struct MiniMaxM3VerifyBatchAttendTests {
    typealias Base = MiniMaxM3MSABlockSparseTests

    /// Per-row selections (row r over index keys `..< offset + r + 1`) as the verify computes them.
    static func rowSelections(_ iq: MLXArray, _ ik: MLXArray, offset: Int) -> [MiniMaxM3MSASelection?] {
        (0 ..< iq.dim(2)).map { r in
            MiniMaxM3MSA.select(
                queries: iq[0..., 0..., r ..< (r + 1)], keys: ik[0..., 0..., ..<(offset + r + 1)], offset: offset + r,
                blockSize: Base.block, topK: Base.topK, localBlocks: 1)
        }
    }

    static func perRow(
        _ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ selections: [MiniMaxM3MSASelection?], offset: Int
    ) -> MLXArray {
        concatenated(
            selections.enumerated().map { r, s -> MLXArray in
                let end = offset + r + 1
                return MiniMaxM3MSA.decodeKernelAttention(
                    queries: contiguous(q[0..., 0..., r ..< (r + 1)]), keys: k[0..., 0..., ..<end],
                    values: v[0..., 0..., ..<end], selection: s!, scale: Base.scale)
            }, axis: 2)
    }

    static func batched(
        _ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ selections: [MiniMaxM3MSASelection?], offset: Int
    ) -> MLXArray {
        let rows = selections.count
        let all = MiniMaxM3MSASelection(
            topIndices: concatenated(selections.map { $0!.topIndices }, axis: 2), offset: offset,
            keyLength: offset + rows, blockSize: Base.block)
        return MiniMaxM3MSA.decodeKernelAttention(
            queries: q, keys: k[0..., 0..., ..<(offset + rows)], values: v[0..., 0..., ..<(offset + rows)],
            selection: all, scale: Base.scale)
    }

    @Test("Rows' own selections in one decode-kernel launch == one-row calls, bit for bit (M3.1 geometry)")
    func batchedEqualsPerRow() throws {
        try Device.withDefaultDevice(.gpu) {
            var checked = 0
            // (context after the rows, rows): 2178 + 15 crosses the 2176 / 2304 block boundary inside the row set.
            for (sk, rows) in [(3011, 2), (2193, 15), (8269, 5), (8268, 8), (5000, 16), (32845, 11)] {
                let offset = sk - rows
                let seed = 4_100 + sk + rows
                let (k, v, ik) = Base.context(seed: seed, sk: sk, groups: 4, dtype: .bfloat16)
                let (q, iq) = Base.queries(seed: seed, stream: 0, rows: rows, groups: 4, dtype: .bfloat16)
                let selections = Self.rowSelections(iq, ik, offset: offset)
                #expect(selections.allSatisfy { $0 != nil }, "sk \(sk): every row should select (> top-k blocks)")
                guard selections.allSatisfy({ $0 != nil }) else { continue }
                let reference = Self.perRow(q, k, v, selections, offset: offset)
                let got = Self.batched(q, k, v, selections, offset: offset)
                eval(reference, got)
                #expect(abs(reference.asType(.float32)).max().item(Float.self) > 0, "degenerate reference")
                #expect(got.shape == reference.shape && arrayEqual(got, reference).item(Bool.self), "sk \(sk) rows \(rows)")
                checked += 1
            }
            #expect(checked == 6)
        }
    }

    @Test("Control: a row given its neighbour's selection differs (so the check can fail)")
    func controlSwappedSelectionDiffers() throws {
        try Device.withDefaultDevice(.gpu) {
            let (sk, rows) = (8269, 4)
            let offset = sk - rows
            let (k, v, ik) = Base.context(seed: 77, sk: sk, groups: 4, dtype: .bfloat16)
            let (q, iq) = Base.queries(seed: 77, stream: 0, rows: rows, groups: 4, dtype: .bfloat16)
            var selections = Self.rowSelections(iq, ik, offset: offset)
            let reference = Self.perRow(q, k, v, selections, offset: offset)
            // Row 0 with row 3's blocks: the same launch shape, a different row selection.
            selections[0] = MiniMaxM3MSASelection(
                topIndices: selections[3]!.topIndices, offset: offset, keyLength: offset + 1, blockSize: Base.block)
            let got = Self.batched(q, k, v, selections, offset: offset)
            #expect(!arrayEqual(got, reference).item(Bool.self))
        }
    }

    static func time(_ probes: Int = 7, reps: Int = 10, _ body: () -> MLXArray) -> Double {
        var t: [Double] = []
        for p in 0 ... probes {
            let t0 = Date()
            var outs: [MLXArray] = []
            for _ in 0 ..< reps { outs.append(body()) }
            eval(outs)
            if p > 0 { t.append(Date().timeIntervalSince(t0) * 1000 / Double(reps)) }
        }
        t.sort()
        return t[t.count / 2]
    }

    @Test("Benchmark (opt-in VMLX_M3_BATCH_ATTEND_BENCH=1): one layer, T verify rows, per-row vs one launch")
    func benchmark() throws {
        guard ProcessInfo.processInfo.environment["VMLX_M3_BATCH_ATTEND_BENCH"] == "1" else { return }
        try Device.withDefaultDevice(.gpu) {
            for sk in [3011, 8269, 32845] {
                for rows in [2, 4, 8, 15] {
                    let offset = sk - rows
                    let (k, v, ik) = Base.context(seed: 9, sk: sk, groups: 4, dtype: .bfloat16)
                    let (q, iq) = Base.queries(seed: 9, stream: 0, rows: rows, groups: 4, dtype: .bfloat16)
                    eval(k, v, ik, q, iq)
                    let sel = Self.rowSelections(iq, ik, offset: offset)
                    let attendRows = Self.time { Self.perRow(q, k, v, sel, offset: offset) }
                    let attendOne = Self.time { Self.batched(q, k, v, sel, offset: offset) }
                    let fullRows = Self.time {
                        Self.perRow(q, k, v, Self.rowSelections(iq, ik, offset: offset), offset: offset)
                    }
                    let fullOne = Self.time {
                        Self.batched(q, k, v, Self.rowSelections(iq, ik, offset: offset), offset: offset)
                    }
                    print(
                        "[batch attend bench] ctx \(sk) rows \(rows): attend per-row "
                            + String(format: "%.3f", attendRows) + " one launch " + String(format: "%.3f", attendOne)
                            + " | select + attend per-row " + String(format: "%.3f", fullRows) + " batched "
                            + String(format: "%.3f", fullOne) + " ms / layer")
                }
            }
        }
    }
}
