// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// DeepSeek-V4.1 prefill kernels at the model's geometry (384 experts, top-6; 64 heads x 512, one KV head, window 128,
/// 512 picks). The routed-expert cases compare bits, not tolerances: the expert-aligned tiles and the sort glue must
/// return exactly what they replace. The opt-in NAX attention kernel is checked against the fp32 reference.
final class DeepseekV41PrefillKernelTests: XCTestCase {
    private func bits16(_ a: MLXArray) -> [UInt16] { a.view(dtype: .uint16).asArray(UInt16.self) }

    private func withEnv<R>(_ name: String, _ value: String?, _ body: () throws -> R) rethrows -> R {
        let old = getenv(name).map { String(cString: $0) }
        if let value { setenv(name, value, 1) } else { unsetenv(name) }
        defer { if let old { setenv(name, old, 1) } else { unsetenv(name) } }
        return try body()
    }

    // MARK: attention

    /// One heads16 case: q (1, 64, S, 512), window rows R, pool P, picks K per query (some -1, some past P, some
    /// causally invisible), `offset` absolute position of the first query.
    private func heads16Case(seed: UInt64, s: Int, r: Int, p: Int, k: Int, offset: Int, ratio: Int, dtype: DType)
        -> (q: MLXArray, kv: MLXArray, pool: MLXArray, topk: MLXArray, sinks: MLXArray)
    {
        let key = MLXRandom.key(seed)
        let ks = MLXRandom.split(key: key, into: 5)
        let q = (MLXRandom.normal([1, 64, s, 512], key: ks[0]) * 0.6).asType(dtype)
        let kv = (MLXRandom.normal([r, 512], key: ks[1]) * 0.6).asType(dtype)
        let pool = (MLXRandom.normal([p, 512], key: ks[2]) * 0.6).asType(dtype)
        // Picks in [-3, P + 3): negatives and past-the-end entries exercise the skip rule; others may be invisible.
        let topk = MLXRandom.randInt(low: -3, high: Int32(p + 3), [s, k], key: ks[3]).asType(.int32)
        let sinks = MLXRandom.normal([64], key: ks[4]).asType(.float32)
        eval(q, kv, pool, topk, sinks)
        return (q, kv, pool, topk, sinks)
    }

    /// The opt-in NAX kernel computes the heads-16 kernel's attention (not its bits): against the fp32 reference of the
    /// stock dense-mask branch, both stay within bf16 output rounding at cold-prefill, mid-prompt and ratio-1 shapes.
    func testIndexedNAXMatchesReference() throws {
        try XCTSkipUnless(DSV41IndexedAttentionNAX.available, "NAX needs an M5-class GPU")
        let scale = Float(pow(Double(512), -0.5))
        for (i, (s, r, p, k, offset, ratio)) in [(64, 64, 40, 32, 0, 2), (37, 164, 300, 64, 900, 2),
                                                 (20, 147, 400, 48, 380, 1)].enumerated() {
            let c = heads16Case(seed: UInt64(77 + i), s: s, r: r, p: p, k: k, offset: offset, ratio: ratio,
                                dtype: .bfloat16)
            // The reference counts a duplicated pick once; use distinct picks (a permutation prefix per query).
            let perm = argSort(MLXRandom.uniform(0 ..< 1, [s, p], key: MLXRandom.key(UInt64(5 + i))), axis: -1)
            let topk = perm[0..., ..<k].asType(.int32)
            let ref = DeepseekV4Math.heads16Reference(
                q: c.q, kv2d: c.kv, pool2d: c.pool, topk2d: topk, sinks32: c.sinks,
                offset: offset, window: 128, ratio: ratio, scale: scale).asType(.float32)
            let h16 = DeepseekV4Math.heads16RunKernel(
                q: c.q, kv2d: c.kv, pool2d: c.pool, topk2d: topk, sinks32: c.sinks,
                offset: offset, window: 128, ratio: ratio, scale: scale).asType(.float32)
            let nax = try XCTUnwrap(DSV41IndexedAttentionNAX.attend(
                q: c.q, kv2d: c.kv, pool2d: c.pool, topk2d: topk, sinks: c.sinks, offset: offset, window: 128,
                ratio: ratio, scale: scale)).asType(.float32)
            let denom = MLX.abs(ref).max().item(Float.self)
            let relN = MLX.abs(nax - ref).max().item(Float.self) / denom
            let relH = MLX.abs(h16 - ref).max().item(Float.self) / denom
            XCTAssertLessThan(relN, 2e-2, "shape \(i): NAX rel \(relN)")
            XCTAssertLessThan(relH, 2e-2, "shape \(i): heads16 rel \(relH)")
            XCTAssertFalse(isNaN(nax).any().item(Bool.self))
            print("DSV41_NAX shape \(i) rel vs fp32 reference: nax \(relN) heads16 \(relH)")
        }
        // Control: a changed visible pool row changes the NAX output.
        let (s, r, p, k, offset, ratio) = (37, 164, 300, 64, 900, 2)
        let c = heads16Case(seed: 78, s: s, r: r, p: p, k: k, offset: offset, ratio: ratio, dtype: .bfloat16)
        let perm = argSort(MLXRandom.uniform(0 ..< 1, [s, p], key: MLXRandom.key(6)), axis: -1)
        let topk = perm[0..., ..<k].asType(.int32)
        let picks = topk[s - 1].asArray(Int32.self)
        let row = try XCTUnwrap(picks.first(where: { (Int($0) + 1) * ratio <= offset + s }))
        let mask = (MLXArray(0 ..< Int32(p)) .== MLXArray(row)).reshaped(p, 1)
        let bumped = MLX.where(mask, MLXArray(Float(0.25)).asType(.bfloat16), c.pool)
        let a = DSV41IndexedAttentionNAX.attend(q: c.q, kv2d: c.kv, pool2d: c.pool, topk2d: topk, sinks: c.sinks,
                                                offset: offset, window: 128, ratio: ratio, scale: scale)!
        let b = DSV41IndexedAttentionNAX.attend(q: c.q, kv2d: c.kv, pool2d: bumped, topk2d: topk, sinks: c.sinks,
                                                offset: offset, window: 128, ratio: ratio, scale: scale)!
        XCTAssertNotEqual(bits16(a), bits16(b), "control: a changed visible pool row must change the output")
    }

    /// The long-context variant reads the pool from its 4-bit slab and dequantizes the picked rows in place exactly as
    /// the dense view does: its output equals the dense kernel's on the dequantized pool, bit for bit.
    func testIndexedNAXSlabEqualsDenseOnDequantizedPool() throws {
        try XCTSkipUnless(DSV41IndexedAttentionNAX.available, "NAX needs an M5-class GPU")
        let scale = Float(pow(Double(512), -0.5))
        for (i, (s, r, p, k, offset, ratio)) in [(37, 164, 300, 64, 900, 2), (129, 256, 4_000, 512, 3_900, 1)]
            .enumerated()
        {
            let c = heads16Case(seed: UInt64(91 + i), s: s, r: r, p: p, k: k, offset: offset, ratio: ratio,
                                dtype: .bfloat16)
            let seg = DeepseekV4PoolStorage.quantize(c.pool, bits: 4, groupSize: 32)
            XCTAssertEqual(seg.bits, 4)
            let dense = DeepseekV4PoolStorage.dequantize(seg).reshaped(p, 512)
            let a = try XCTUnwrap(DSV41IndexedAttentionNAX.attend(
                q: c.q, kv2d: c.kv, pool2d: dense, topk2d: c.topk, sinks: c.sinks, offset: offset, window: 128,
                ratio: ratio, scale: scale))
            let b = try XCTUnwrap(DSV41IndexedAttentionNAX.attendSlab(
                q: c.q, kv2d: c.kv, codes: seg.codes, scales: seg.scales, biases: seg.biases,
                groupSize: seg.groupSize, poolRows: p, topk2d: c.topk, sinks: c.sinks, offset: offset, window: 128,
                ratio: ratio, scale: scale))
            XCTAssertFalse(isNaN(a.asType(.float32)).any().item(Bool.self))
            XCTAssertEqual(bits16(a), bits16(b), "shape \(i): slab != dense")
        }
    }

    // MARK: routed experts

    private func projection(_ key: MLXArray, k: Double, experts: Int, outputs: Int, inputs: Int,
                            outlierColumns: [[Int32]], columnScales: Bool) throws -> JANGTProjection
    {
        let ks = MLXRandom.split(key: key, into: 5)
        let words = JANGTCode.wordsPerRow(inputs: inputs, k: k)
        let width = outlierColumns.map(\.count).max() ?? 0
        let cols = outlierColumns.flatMap { $0 + Array(repeating: Int32(-1), count: width - $0.count) }
        return try JANGTProjection(
            k: k,
            packed: MLXRandom.randInt(low: 0, high: Int32.max, [experts, outputs, words], key: ks[0]).asType(.uint32),
            scale: MLXRandom.uniform(low: 0.002, high: 0.02, [experts, outputs], key: ks[1]),
            signs: MLX.where(MLXRandom.uniform(0 ..< 1, [inputs], key: ks[2]) .< 0.5, MLXArray(Float(-1)),
                             MLXArray(Float(1))),
            outlierColumnIndices: MLXArray(cols, [experts, width]),
            outlierWeights: (MLXRandom.normal([experts, outputs, width], key: ks[3]) * 0.05).asType(.float16),
            columnScales: columnScales
                ? MLXRandom.uniform(low: 0.5, high: 1.5, [experts, inputs], key: ks[4]) : nil)
    }

    /// Router-like top-k ids: k distinct experts per token, skewed (a few hot experts, many with 0-3 rows).
    private func routes(tokens: Int, experts: Int, topK: Int, seed: UInt64) -> (MLXArray, MLXArray) {
        let key = MLXRandom.key(seed)
        let ks = MLXRandom.split(key: key, into: 2)
        let logits = MLXRandom.gumbel([tokens, experts], key: ks[0])
            + MLXArray((0 ..< experts).map { Float(-0.02 * Double($0 % 97)) })
        let ids = argPartition(-logits, kth: topK - 1, axis: -1)[0..., ..<topK].asType(.int32)
        let w = MLXRandom.uniform(low: 0.05, high: 1, [tokens, topK], key: ks[1])
        eval(ids, w)
        return (ids, w)
    }

    /// The routed layer at DeepSeek-V4.1's geometry (384 experts, top-6, ~16 rows per expert at 1024 tokens; narrow
    /// hidden / intermediate widths to keep the test small): the expert-aligned schedule returns the row-tiled
    /// schedule's output bit for bit, gate/up with layer-wide outlier columns (the fused epilogue) and without, down
    /// with per-expert outlier columns and column scales, every K the bundle uses.
    func testExpertAlignedRoutedLayerEqualsRowTiledAtDSV41Geometry() throws {
        try XCTSkipUnless(JANGTPrefillKernel.available, "NAX prefill tiles need an M5-class GPU")
        let experts = 384, topK = 6, hidden = 512, inter = 256
        let pairs: [(Double, Double)] = [(1.0, 1.5), (1.0, 2.0), (1.5, 2.0), (1.5, 1.5), (1.0, 1.0), (2.0, 2.5)]
        var compared = 0
        for (pi, (kg, kd)) in pairs.enumerated() {
            for gateOutliers in [false, true] {
                let key = MLXRandom.key(UInt64(1_000 + pi * 2 + (gateOutliers ? 1 : 0)))
                let ks = MLXRandom.split(key: key, into: 3)
                let gu: [[Int32]] = Array(repeating: gateOutliers ? [3, 400] : [], count: experts)
                let dn: [[Int32]] = (0 ..< experts).map { e in [Int32((e * 7) % inter), Int32((e * 13 + 5) % inter)] }
                let gate = try projection(ks[0], k: kg, experts: experts, outputs: inter, inputs: hidden,
                                          outlierColumns: gu, columnScales: false)
                let up = try projection(ks[1], k: kg, experts: experts, outputs: inter, inputs: hidden,
                                        outlierColumns: gu, columnScales: false)
                let down = try projection(ks[2], k: kd, experts: experts, outputs: hidden, inputs: inter,
                                          outlierColumns: dn, columnScales: true)
                let layer = try JANGTRoutedExpertLayer(
                    gate: gate, up: up, down: down, hiddenSize: hidden, intermediateSize: inter,
                    activationLimit: 10)
                // 1024 tokens (the served chunk) and 1,382 (a non-aligned remainder-sized call).
                for tokens in [1_024, 1_382] {
                    let (ids, w) = routes(tokens: tokens, experts: experts, topK: topK, seed: UInt64(tokens + pi))
                    let x = (MLXRandom.normal([1, tokens, hidden], key: MLXRandom.key(UInt64(tokens * 3 + pi)))
                        * 0.5).asType(.bfloat16)
                    func run(_ tiles: String) -> MLXArray {
                        withEnv("VMLX_JANGT_EXPERT_TILES", tiles) {
                            let y = layer.callRouted(x, indices: ids.reshaped(1, tokens, topK),
                                                     scores: w.reshaped(1, tokens, topK))
                            eval(y)
                            return y
                        }
                    }
                    let rowTiled = run("0"), aligned = run("1")
                    XCTAssertEqual(rowTiled.dtype, .bfloat16)
                    XCTAssertFalse(isNaN(rowTiled.asType(.float32)).any().item(Bool.self))
                    XCTAssertEqual(bits16(rowTiled), bits16(aligned),
                                   "K \(kg)/\(kd) outliers \(gateOutliers) tokens \(tokens): aligned != row-tiled")
                    compared += 1
                }
            }
        }
        XCTAssertEqual(compared, pairs.count * 2 * 2)
    }

    /// `sourceRows` (the token of each sorted pair) equals MLX's integer floorDivide for top-6 (the gather table) and
    /// top-8 (the shift), including a non-aligned row count.
    func testSourceRowsEqualFloorDivide() {
        for (tokens, routes) in [(1_024, 6), (1_382, 6), (17, 6), (300, 8), (5, 3)] {
            let ids = MLXRandom.randInt(low: 0, high: 384, [tokens * routes], key: MLXRandom.key(UInt64(tokens)))
            let order = argSort(ids)
            let got = JANGTRoutedExpertLayer.sourceRows(order, routes: routes)
            let want = order.floorDivide(routes)
            XCTAssertEqual(got.dtype, order.dtype)
            XCTAssertEqual(got.asArray(UInt32.self), want.asArray(UInt32.self), "tokens \(tokens) routes \(routes)")
        }
    }

    /// `gatherSortWithOrder` (one argSort, no floorDivide) returns what the floorDivide formulation returned, and its
    /// order aligns the scores exactly as a second argSort of the same ids did (SwitchGLU's affine routed path).
    func testGatherSortWithOrderMatchesFloorDivideFormulation() {
        for (tokens, routes) in [(1_024, 6), (1_382, 6), (11, 6), (300, 8), (7, 4)] {
            let ks = MLXRandom.split(key: MLXRandom.key(UInt64(tokens + routes)), into: 3)
            let ids = MLXRandom.randInt(low: 0, high: 384, [1, tokens, routes], key: ks[0]).asType(.int32)
            let x = MLXRandom.normal([1, tokens, 1, 1, 16], key: ks[1]).asType(.bfloat16)
            let scores = MLXRandom.uniform(0 ..< 1, [1, tokens, routes], key: ks[2])
            let got = gatherSortWithOrder(x: x, indices: ids)
            let flat = ids.flattened(), order = argSort(flat)
            let wantX = x.flattened(start: 0, end: -3)[order.floorDivide(routes)]
            XCTAssertEqual(bits16(got.x), bits16(wantX), "x rows tokens \(tokens) routes \(routes)")
            XCTAssertEqual(got.indices.asArray(Int32.self), flat[order].asArray(Int32.self))
            XCTAssertEqual(got.inverseOrder.asArray(UInt32.self), argSort(order).asArray(UInt32.self))
            XCTAssertEqual(scores.flattened()[got.order].asArray(Float.self),
                           scores.flattened()[argSort(ids.flattened())].asArray(Float.self))
        }
    }

    /// The expert tile plan of a DSV4.1-like sort: one 64-row tile per non-empty expert at <= 64 rows, segment starts
    /// at the first row of each expert.
    func testExpertTilePlanAtDSV41Geometry() {
        let experts = 384
        var counts = Array(repeating: 0, count: experts)
        for e in 0 ..< experts { counts[e] = (e * 37) % 41 }   // 0 ... 40 rows, some empty
        counts[5] = 64; counts[6] = 65; counts[7] = 130
        var ids: [UInt32] = []
        for (e, c) in counts.enumerated() { ids += Array(repeating: UInt32(e), count: c) }
        let plan = JANGTPrefillKernel.expertTilePlan(sortedIndices: MLXArray(ids), experts: experts)
        let starts = plan.starts.asArray(Int32.self), tiles = plan.tiles.asArray(Int32.self)
        var s = 0, t = 0
        for e in 0 ..< experts {
            XCTAssertEqual(Int(starts[e]), s, "start of expert \(e)")
            XCTAssertEqual(Int(tiles[e]), t, "tiles before expert \(e)")
            s += counts[e]; t += (counts[e] + 63) / 64
        }
        XCTAssertEqual(Int(starts[experts]), ids.count)   // the invalid-id segment is empty
        XCTAssertEqual(Int(starts[experts + 1]), ids.count)
        XCTAssertEqual(Int(tiles[experts + 1]), t)
        XCTAssertEqual(plan.segments, experts + 1)
    }
}

/// The routed layer against a re-implementation of the 2c35065b4 prefill glue (integer floorDivide, fp32 gather then
/// narrow, k copies of every raw row for the outlier epilogue, widen-then-gather down outliers) at the real
/// DeepSeek-V4.1 dimensions (`DSV41_PERF=1`: ~3 GB of synthetic banks).
final class DeepseekV41RoutedGlueRealDimsTests: XCTestCase {
    func testNewGlueEqualsOldGlueAtRealDims() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["DSV41_PERF"] == "1" && JANGTPrefillKernel.available)
        let experts = 384, topK = 6, hidden = 5_120, inter = 2_304, tokens = 1_024
        func proj(_ k: Double, out: Int, inp: Int, seed: UInt64, cols: MLXArray, colScales: Bool) throws
            -> JANGTProjection
        {
            let ks = MLXRandom.split(key: MLXRandom.key(seed), into: 4)
            let w = JANGTCode.wordsPerRow(inputs: inp, k: k)
            return try JANGTProjection(
                k: k, packed: MLXRandom.randInt(low: 0, high: Int32.max, [experts, out, w], key: ks[0]).asType(.uint32),
                scale: MLXRandom.uniform(low: 0.002, high: 0.02, [experts, out], key: ks[1]),
                signs: MLX.where(MLXRandom.uniform(0 ..< 1, [inp], key: ks[2]) .< 0.5, MLXArray(Float(-1)),
                                 MLXArray(Float(1))),
                outlierColumnIndices: cols,
                outlierWeights: (MLXRandom.normal([experts, out, cols.dim(1)], key: ks[3]) * 0.05).asType(.float16),
                columnScales: colScales ? MLXRandom.uniform(low: 0.5, high: 1.5, [experts, inp]) : nil)
        }
        let guCols = MLXArray((0 ..< experts).flatMap { _ in [Int32(3), Int32(400), Int32(4_000)] }, [experts, 3])
        let dnCols = MLXArray((0 ..< experts).flatMap { e in (0 ..< 8).map { Int32((e * 31 + $0 * 277) % inter) } },
                              [experts, 8])
        for (kg, kd) in [(1.0, 1.5), (2.0, 2.5)] {
            let gate = try proj(kg, out: inter, inp: hidden, seed: 11, cols: guCols, colScales: false)
            let up = try proj(kg, out: inter, inp: hidden, seed: 12, cols: guCols, colScales: false)
            let down = try proj(kd, out: hidden, inp: inter, seed: 13, cols: dnCols, colScales: true)
            eval(gate.packed, up.packed, down.packed)
            let layer = try JANGTRoutedExpertLayer(
                gate: gate, up: up, down: down, hiddenSize: hidden, intermediateSize: inter, activationLimit: 10)
            // Skewed routes: a few hot experts get > 64 rows (multi-tile segments), many get 0-3.
            let bias = MLXArray((0 ..< experts).map { Float($0 < 8 ? 3.0 : -0.01 * Double($0 % 97)) })
            let logits = MLXRandom.gumbel([tokens, experts], key: MLXRandom.key(21)) + bias
            let ids = argPartition(-logits, kth: topK - 1, axis: -1)[0..., ..<topK].asType(.int32)
            let w = MLXRandom.uniform(low: 0.05, high: 1, [tokens, topK], key: MLXRandom.key(22))
            let x = (MLXRandom.normal([tokens, hidden], key: MLXRandom.key(23)) * 0.5).asType(.bfloat16)
            eval(ids, w, x)

            // The 2c35065b4 glue.
            let flat = ids.flattened()
            let order = argSort(flat)
            let sortedIDs = take(flat, order).asType(.uint32)
            let tok = order.floorDivide(topK)
            let guSigns = gate.signs * JANGTRoutedExpertLayer.keepMask(
                width: hidden, columns: gate.outlierColumnIndices[0 ..< 1])[0]
            let xr = take(JANGTKernels.rotateRows(x, signs: guSigns), tok, axis: 0).asType(.bfloat16)
            let h = JANGTPrefillKernel.projectGateUpWithOutliers(
                xr, rawRows: take(x, tok, axis: 0), gate: gate, up: up,
                columns: guCols[0].asType(.int32), indices: sortedIDs, limit: 10)
            let dSigns = down.columnScales! * down.signs.expandedDimensions(axis: 0)
                * JANGTRoutedExpertLayer.keepMask(width: inter, columns: down.outlierColumnIndices)
            let hr = JANGTKernels.rotateRows(h, signs: dSigns, rowExperts: sortedIDs).asType(.bfloat16)
            var y = JANGTPrefillKernel.project(hr, down, indices: sortedIDs).asType(.float32)
            let cols = take(down.outlierColumnIndices, sortedIDs.asType(.int32), axis: 0)
            let valid = (cols .>= MLXArray(Int32(0))).asType(.float32)
            let xo = takeAlong(h.asType(.float32), maximum(cols, MLXArray(Int32(0))), axis: 1) * valid
            let wT = down.outlierWeights.swappedAxes(-1, -2).asType(.float32)
            y = y + gatherMM(xo.expandedDimensions(axis: 1), wT, rhsIndices: sortedIDs, sortedIndices: true)
                .squeezed(axis: 1)
            let restored = take(y, argSort(order), axis: 0).reshaped(tokens, topK, hidden).asType(.float32)
            let old = matmul(w.reshaped(tokens, 1, topK), restored).reshaped(tokens, hidden).asType(.bfloat16)
            eval(old)
            for tiles in ["0", "1"] {
                setenv("VMLX_JANGT_EXPERT_TILES", tiles, 1)
                let new = layer.callRouted(x.reshaped(1, tokens, hidden), indices: ids.reshaped(1, tokens, topK),
                                           scores: w.reshaped(1, tokens, topK)).reshaped(tokens, hidden)
                eval(new)
                let same = (old.view(dtype: .uint16) .== new.view(dtype: .uint16)).all().item(Bool.self)
                let diff = MLX.abs(old.asType(.float32) - new.asType(.float32)).max().item(Float.self)
                print("DSV41_GLUE K \(kg)/\(kd) tiles=\(tiles): new == old bitwise \(same), max |diff| \(diff)")
                XCTAssertTrue(same, "K \(kg)/\(kd) tiles=\(tiles): new glue != 2c35065b4 glue")
            }
            unsetenv("VMLX_JANGT_EXPERT_TILES")
        }
    }
}

/// Timings at the real DeepSeek-V4.1 shapes (diagnostic, `DSV41_PERF=1`). Synthetic weights; the model-level speed is
/// measured with RunBench `BENCH_DSV41_PREFILL`. Arms alternate call by call (A B C A B C ...), medians reported.
final class DeepseekV41PrefillKernelPerfTests: XCTestCase {
    private func interleaved(_ arms: [(String, () -> MLXArray)], rounds: Int = 15) {
        for (_, f) in arms { for _ in 0 ..< 2 { eval(f()) } }
        var ms = Array(repeating: [Double](), count: arms.count)
        for _ in 0 ..< rounds {
            for (i, (_, f)) in arms.enumerated() {
                let t0 = Date()
                eval(f())
                ms[i].append(Date().timeIntervalSince(t0) * 1e3)
            }
        }
        for (i, (name, _)) in arms.enumerated() {
            let v = ms[i].sorted()
            print(String(format: "DSV41_KERNEL_PERF %@ median %.2f ms (min %.2f max %.2f, n=%d)", name,
                         v[v.count / 2], v.first!, v.last!, rounds))
        }
    }

    func testAttentionCoreTiming() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["DSV41_PERF"] == "1")
        // A 512-row prefill sub-chunk at ~16k context of a ratio-2 layer: window rows = 127 + 512, pool 8,192, 512 picks.
        let s = 512, r = 639, p = 8_192, k = 512, offset = 16_000
        let ks = MLXRandom.split(key: MLXRandom.key(7), into: 5)
        let q = (MLXRandom.normal([1, 64, s, 512], key: ks[0]) * 0.6).asType(.bfloat16)
        let kv = (MLXRandom.normal([1, 1, r, 512], key: ks[1]) * 0.6).asType(.bfloat16)
        let pool = (MLXRandom.normal([1, p, 512], key: ks[2]) * 0.6).asType(.bfloat16)
        // Distinct visible picks per query (a random permutation prefix), as the indexer's top-k gives.
        let topk = argSort(MLXRandom.uniform(0 ..< 1, [s, p], key: ks[3]), axis: -1)[0..., ..<k].asType(.int32)
            .reshaped(1, s, k)
        let sinks = MLXRandom.normal([64], key: ks[4])
        eval(q, kv, pool, topk, sinks)
        let scale = Float(pow(Double(512), -0.5))
        func h16() -> MLXArray {
            DeepseekV4Math.heads16RunKernel(
                q: q, kv2d: kv.reshaped(r, 512), pool2d: pool.reshaped(p, 512), topk2d: topk.reshaped(s, k),
                sinks32: sinks, offset: offset, window: 128, ratio: 2, scale: scale)
        }
        let a = h16().asType(.float32)
        let nax = {
            DSV41IndexedAttentionNAX.attend(
                q: q, kv2d: kv.reshaped(r, 512), pool2d: pool.reshaped(p, 512), topk2d: topk.reshaped(s, k),
                sinks: sinks, offset: offset, window: 128, ratio: 2, scale: scale)!
        }
        let c = nax().asType(.float32)
        let relN = (MLX.abs(a - c).max() / MLX.abs(a).max()).item(Float.self)
        print(String(format: "DSV41_KERNEL_PERF nax vs heads16 max rel diff %.3e", relN))
        let seg = DeepseekV4PoolStorage.quantize(pool.reshaped(p, 512), bits: 4, groupSize: 32)
        eval(seg.codes, seg.scales, seg.biases)
        let slab = {
            DSV41IndexedAttentionNAX.attendSlab(
                q: q, kv2d: kv.reshaped(r, 512), codes: seg.codes, scales: seg.scales, biases: seg.biases,
                groupSize: seg.groupSize, poolRows: p, topk2d: topk.reshaped(s, k), sinks: sinks, offset: offset,
                window: 128, ratio: 2, scale: scale)!
        }
        interleaved([("heads16 kernel", h16), ("nax indexed", nax), ("nax indexed, 4-bit slab", slab)])
    }

    func testRoutedPrefillTimingAtDSV41Shapes() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["DSV41_PERF"] == "1" && JANGTPrefillKernel.available)
        let experts = 384, topK = 6, hidden = 5_120, inter = 2_304, tokens = 1_024
        func proj(_ k: Double, out: Int, inp: Int, seed: UInt64, outliers: Bool) throws -> JANGTProjection {
            let ks = MLXRandom.split(key: MLXRandom.key(seed), into: 3)
            let w = JANGTCode.wordsPerRow(inputs: inp, k: k)
            let cols = outliers
                ? MLXArray((0 ..< experts).flatMap { _ in [Int32(3), Int32(400)] }, [experts, 2])
                : MLXArray.zeros([experts, 1], dtype: .int32) - MLXArray(Int32(1))
            return try JANGTProjection(
                k: k, packed: MLXRandom.randInt(low: 0, high: Int32.max, [experts, out, w], key: ks[0]).asType(.uint32),
                scale: MLXArray.ones([experts, out]) * 0.01, signs: MLXArray.ones([inp]),
                outlierColumnIndices: cols,
                outlierWeights: (MLXRandom.normal([experts, out, cols.dim(1)], key: ks[1]) * 0.01).asType(.float16),
                columnScales: nil)
        }
        let gate = try proj(1.5, out: inter, inp: hidden, seed: 1, outliers: true)
        let up = try proj(1.5, out: inter, inp: hidden, seed: 2, outliers: true)
        let down = try proj(2.0, out: hidden, inp: inter, seed: 3, outliers: false)
        eval(gate.packed, up.packed, down.packed)
        let layer = try JANGTRoutedExpertLayer(
            gate: gate, up: up, down: down, hiddenSize: hidden, intermediateSize: inter, activationLimit: 10)
        let logits = MLXRandom.gumbel([tokens, experts], key: MLXRandom.key(9))
        let ids = argPartition(-logits, kth: topK - 1, axis: -1)[0..., ..<topK].reshaped(1, tokens, topK)
        let w = MLXRandom.uniform(low: 0.05, high: 1, [1, tokens, topK])
        let x = (MLXRandom.normal([1, tokens, hidden]) * 0.5).asType(.bfloat16)
        eval(ids, w, x)
        func arm(_ tiles: String) -> () -> MLXArray {
            {
                setenv("VMLX_JANGT_EXPERT_TILES", tiles, 1)
                return layer.callRouted(x, indices: ids, scores: w)
            }
        }
        let rowTiled = arm("0")(), aligned = arm("1")()
        eval(rowTiled, aligned)
        let same = (rowTiled.view(dtype: .uint16) .== aligned.view(dtype: .uint16)).all().item(Bool.self)
        let diff = MLX.abs(rowTiled.asType(.float32) - aligned.asType(.float32)).max().item(Float.self)
        print("DSV41_KERNEL_PERF routed layer real dims: aligned == row-tiled bitwise \(same), max |diff| \(diff)")
        interleaved([("routed layer 1024x6 row-tiled", arm("0")), ("routed layer 1024x6 expert-aligned", arm("1"))])
        unsetenv("VMLX_JANGT_EXPERT_TILES")
    }
}

private extension MLXArray {
    /// A copy of a (rows, cols) zero array with `value` at every column of `row` (test helper).
    func scattered(row: Int, value: Float) -> MLXArray {
        let rows = dim(0)
        let mask = (MLXArray(0 ..< Int32(rows)) .== MLXArray(Int32(row))).reshaped(rows, 1)
        return MLX.where(mask, MLXArray(value).asType(dtype), self)
    }
}
