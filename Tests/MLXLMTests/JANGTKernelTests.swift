import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// JANGT trellis decode kernels vs an independent CPU decoder. The reference reads each row's bitstream at the GLOBAL
/// step offsets (state t at stepOffsets[t]) — no lane/block geometry — so a wrong lane window, block advance, half
/// order, normalization or outlier epilogue in the generated Metal shows up as a mismatch.
final class JANGTKernelTests: XCTestCase {
    private struct Rng {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state
        }
        mutating func word() -> UInt32 { UInt32(truncatingIfNeeded: next() >> 17) }
        mutating func uniform() -> Float { Float(next() >> 40) / Float(1 << 24) }
        mutating func normal() -> Float {
            let u = max(uniform(), 1e-7), v = uniform()
            return sqrt(-2 * log(u)) * cos(2 * .pi * v)
        }
    }

    private struct CPUProjection {
        let k: Double, experts: Int, outputs: Int, inputs: Int, words: Int
        let packed: [UInt32], scale: [Float], signs: [Float], ocol: [Int32], wout: [Float], cs: [Float]?
        let outliers: Int

        func row(_ e: Int, _ n: Int) -> [Float] {
            let so = JANGTCode.stepOffsets(k, inputs / 2)
            let base = (e * outputs + n) * words
            var out = [Float](repeating: 0, count: inputs)
            for t in 0 ..< inputs / 2 {
                var s: UInt32 = 0
                for b in 0 ..< JANGTCode.stateBits {
                    let bit = so[t] + b
                    s |= ((packed[base + bit >> 5] >> UInt32(bit & 31)) & 1) << UInt32(b)
                }
                let (lo, hi) = JANGTCode.values(state: s, k: k)
                out[2 * t] = lo
                out[2 * t + 1] = hi
            }
            return out
        }

        func columns(_ e: Int) -> [Int] {
            (0 ..< outliers).map { Int(ocol[e * outliers + $0]) }.filter { $0 >= 0 }
        }

        func mlx() throws -> JANGTProjection {
            try JANGTProjection(
                k: k, packed: MLXArray(packed, [experts, outputs, words]),
                scale: MLXArray(scale, [experts, outputs]), signs: MLXArray(signs, [inputs]),
                outlierColumnIndices: MLXArray(ocol, [experts, outliers]),
                outlierWeights: MLXArray(wout, [experts, outputs, outliers]).asType(.float16),
                columnScales: cs.map { MLXArray($0, [experts, inputs]) })
        }
    }

    private func projection(_ rng: inout Rng, k: Double, experts: Int, outputs: Int, inputs: Int,
                            columns: [[Int32]], columnScales: Bool) -> CPUProjection {
        let words = JANGTCode.wordsPerRow(inputs: inputs, k: k)
        let outliers = columns[0].count
        // fp16 round trip so the CPU reference sees the kernel's outlier weights exactly.
        let wout = (0 ..< experts * outputs * outliers).map { _ in Float(Float16(rng.normal() * 0.05)) }
        return CPUProjection(
            k: k, experts: experts, outputs: outputs, inputs: inputs, words: words,
            packed: (0 ..< experts * outputs * words).map { _ in rng.word() },
            scale: (0 ..< experts * outputs).map { _ in 0.01 + 0.01 * rng.uniform() },
            signs: (0 ..< inputs).map { _ in rng.uniform() < 0.5 ? -1 : 1 },
            ocol: columns.flatMap { $0 }, wout: wout,
            cs: columnScales ? (0 ..< experts * inputs).map { _ in 0.5 + rng.uniform() } : nil,
            outliers: outliers)
    }

    /// Orthonormal blockwise Walsh-Hadamard (natural order), 128-wide blocks.
    private func h128(_ v: [Float]) -> [Float] {
        var x = v
        for block in stride(from: 0, to: x.count, by: 128) {
            var h = 1
            while h < 128 {
                for i in stride(from: block, to: block + 128, by: 2 * h) {
                    for j in i ..< i + h {
                        let a = x[j], b = x[j + h]
                        x[j] = a + b
                        x[j + h] = a - b
                    }
                }
                h *= 2
            }
            for j in block ..< block + 128 { x[j] *= 0.08838834764831845 }
        }
        return x
    }

    private func dot(_ a: [Float], _ b: [Float]) -> Float {
        var s: Double = 0
        for i in a.indices { s += Double(a[i]) * Double(b[i]) }
        return Float(s)
    }

    private func relL2(_ got: [Float], _ want: [Float]) -> Double {
        var num = 0.0, den = 0.0
        for i in want.indices {
            num += Double(got[i] - want[i]) * Double(got[i] - want[i])
            den += Double(want[i]) * Double(want[i])
        }
        return (num / max(den, 1e-30)).squareRoot()
    }

    /// Untrusted tensors (e.g. a dSpark draft file) are checked before any kernel reads them: an outlier column index
    /// outside [-1, inputs) or an outlier-weight / column-scale shape that disagrees with the bank is refused.
    func testProjectionRejectsMalformedOutliers() throws {
        let e = 2, n = 8, kin = 128
        let words = JANGTCode.wordsPerRow(inputs: kin, k: 2)
        func make(ocol: [Int32], woutShape: [Int]? = nil, csShape: [Int]? = nil) throws -> JANGTProjection {
            let outs = ocol.count / e
            let ws = woutShape ?? [e, n, outs]
            return try JANGTProjection(
                k: 2, packed: MLXArray.zeros([e, n, words], dtype: .uint32),
                scale: MLXArray.ones([e, n], dtype: .float32), signs: MLXArray.ones([kin], dtype: .float32),
                outlierColumnIndices: MLXArray(ocol, [e, outs]),
                outlierWeights: MLXArray.zeros(ws, dtype: .float16),
                columnScales: csShape.map { MLXArray.ones($0, dtype: .float32) })
        }
        XCTAssertEqual(try make(ocol: [0, 127, -1, 5]).outlierColumns, 2)
        XCTAssertEqual(try make(ocol: [-1, -1, -1, -1]).outlierColumns, 0)
        XCTAssertThrowsError(try make(ocol: [0, 128, 1, 2]))
        XCTAssertThrowsError(try make(ocol: [0, -2, 1, 2]))
        XCTAssertThrowsError(try make(ocol: [0, 1, 2, 3], woutShape: [e, n, 1]))
        XCTAssertThrowsError(try make(ocol: [0, 1, 2, 3], csShape: [e, kin / 2]))
        XCTAssertNoThrow(try make(ocol: [0, 1, 2, 3], csShape: [e, kin]))
    }

    func testCodeStatisticsMatchTheStateSpace() {
        for k in [2.0, 2.5, 3.0] {
            var sum = [0.0, 0.0], sq = [0.0, 0.0]
            let st = JANGTCode.stats[k]!
            let mu = [Float(bitPattern: st.0), Float(bitPattern: st.1)]
            let isd = [Float(bitPattern: st.2), Float(bitPattern: st.3)]
            for s in 0 ..< UInt32(1 << JANGTCode.stateBits) {
                let (a, b) = JANGTCode.values(state: s, k: k)
                // Undo the normalization to recover the raw halves.
                let raw = [Double(a / isd[0] + mu[0]), Double(b / isd[1] + mu[1])]
                for i in 0 ..< 2 { sum[i] += raw[i]; sq[i] += raw[i] * raw[i] }
            }
            let n = Double(1 << JANGTCode.stateBits)
            for i in 0 ..< 2 {
                let mean = sum[i] / n, sd = (sq[i] / n - mean * mean).squareRoot()
                XCTAssertEqual(mean, Double(mu[i]), accuracy: 1e-6, "mu K=\(k)")
                XCTAssertEqual(1 / sd, Double(isd[i]), accuracy: 1e-5, "isd K=\(k)")
            }
        }
    }

    func testWordsPerRowMatchBundleShapes() {
        // GLM-5.3-Flash JANGHT2.4 headers: gate K=2 (4096 in) -> 258 words, down K=2.5 (2048 in) -> 162 words.
        XCTAssertEqual(JANGTCode.wordsPerRow(inputs: 4096, k: 2), 258)
        XCTAssertEqual(JANGTCode.wordsPerRow(inputs: 2048, k: 2.5), 162)
        XCTAssertEqual(JANGTCode.laneBits(2), 32)
        XCTAssertEqual(JANGTCode.laneBits(2.5), 40)
        XCTAssertEqual(JANGTCode.laneBits(3), 48)
    }

    private func runGateUpAndDown(k: Double, limit: Float, largeHidden: Bool) throws {
        var rng = Rng(state: UInt64(k * 1000) + (largeHidden ? 7 : 3))
        let experts = 4, d = 1024, i = 512, tokens = 3, slots = 2
        // Gate/up outliers are layer-wide (same columns for every expert); down columns differ per expert.
        let guCols = (0 ..< experts).map { _ in [Int32(5), Int32(700)] }
        let gate = projection(&rng, k: k, experts: experts, outputs: i, inputs: d, columns: guCols, columnScales: false)
        let up = projection(&rng, k: k, experts: experts, outputs: i, inputs: d, columns: guCols, columnScales: false)
        let dnCols: [[Int32]] = [[3, 400, -1], [9, -1, -1], [100, 200, 300], [-1, -1, -1]]
        let dn = projection(&rng, k: k, experts: experts, outputs: d, inputs: i, columns: dnCols, columnScales: true)
        let x = (0 ..< tokens * d).map { _ in rng.normal() }
        let idx: [UInt32] = [0, 2, 3, 1, 2, 2]
        let w = (0 ..< tokens * slots).map { _ in rng.uniform() }

        // --- CPU reference
        var keep = [Float](repeating: 1, count: d)
        for c in gate.columns(0) { keep[c] = 0 }
        var hRef = [Float](repeating: 0, count: tokens * slots * i)
        for t in 0 ..< tokens {
            let xt = Array(x[t * d ..< (t + 1) * d])
            let xr = h128((0 ..< d).map { xt[$0] * gate.signs[$0] * keep[$0] })
            for s in 0 ..< slots {
                let e = Int(idx[t * slots + s])
                for n in 0 ..< i {
                    var g = dot(gate.row(e, n), xr) * gate.scale[e * i + n]
                    var u = dot(up.row(e, n), xr) * up.scale[e * i + n]
                    for (j, c) in gate.columns(e).enumerated() {
                        g += xt[c] * gate.wout[(e * i + n) * gate.outliers + j]
                        u += xt[c] * up.wout[(e * i + n) * up.outliers + j]
                    }
                    if limit > 0 { g = min(g, limit); u = min(max(u, -limit), limit) }
                    hRef[(t * slots + s) * i + n] = g / (1 + exp(-g)) * u
                }
            }
        }
        // Naive-N0.5 expert hidden units reach 1e5+: exercise the fp16 range guard.
        if largeHidden {
            for r in [1, 4] { for c in 0 ..< i { hRef[r * i + c] *= 100_000 } }
        }
        var yRef = [Float](repeating: 0, count: tokens * d)
        for t in 0 ..< tokens {
            for s in 0 ..< slots {
                let ts = t * slots + s, e = Int(idx[ts])
                let h = Array(hRef[ts * i ..< (ts + 1) * i])
                var keepE = [Float](repeating: 1, count: i)
                for c in dn.columns(e) { keepE[c] = 0 }
                let hr = h128((0 ..< i).map { h[$0] * dn.cs![e * i + $0] * dn.signs[$0] * keepE[$0] })
                for n in 0 ..< d {
                    var y = dot(dn.row(e, n), hr) * dn.scale[e * d + n]
                    for (j, c) in dn.columns(e).enumerated() { y += h[c] * dn.wout[(e * d + n) * dn.outliers + j] }
                    yRef[t * d + n] += w[ts] * y
                }
            }
        }

        // --- GPU kernels (the same composition as JANGTRoutedExpertLayer)
        let g = try gate.mlx(), u = try up.mlx(), down = try dn.mlx()
        XCTAssertEqual(g.outlierColumns, 2)
        let xm = MLXArray(x, [tokens, d]), ids = MLXArray(idx, [tokens, slots])
        let suGU = g.signs * JANGTRoutedExpertLayer.keepMask(width: d, columns: g.outlierColumnIndices[0 ..< 1])[0]
        let h = JANGTKernels.gateUp(xr: JANGTKernels.rotateRows(xm, signs: suGU), x: xm, indices: ids, gate: g, up: u,
                                    limit: limit, outputDType: .float32)
        let hErr = relL2(h.asArray(Float.self), largeHidden ? hRef.enumerated().map { [1, 4].contains($0.offset / i) ? $0.element / 100_000 : $0.element } : hRef)
        XCTAssertLessThan(hErr, 1e-2, "gate/up K=\(k) limit=\(limit) rel-L2")

        let svDN = down.columnScales! * down.signs.expandedDimensions(axis: 0)
            * JANGTRoutedExpertLayer.keepMask(width: i, columns: down.outlierColumnIndices)
        let hm = MLXArray(hRef, [tokens * slots, i])
        let y = JANGTKernels.down(hr: JANGTKernels.rotateRows(hm, signs: svDN, rowExperts: ids.flattened()), h: hm,
                                  indices: ids, weights: MLXArray(w, [tokens, slots]), projection: down)
        let yErr = relL2(y.asArray(Float.self), yRef)
        XCTAssertLessThan(yErr, 1e-2, "down K=\(k) large=\(largeHidden) rel-L2")
        print("JANGT K=\(k) limit=\(limit) large=\(largeHidden): gate/up rel-L2 \(hErr), down rel-L2 \(yErr)")
    }

    /// Expert-sorted NAX prefill tiles (JANGTPrefillKernel) against the CPU decoder: single projection, fused
    /// gate+up with a clamp that bites, and the outlier term, for 96 sorted rows over a ragged expert split.
    private func runPrefill(k: Double) throws {
        try XCTSkipUnless(JANGTPrefillKernel.available, "NAX prefill tiles need an M5-class GPU")
        var rng = Rng(state: UInt64(k * 100) + 11)
        let experts = 4, n = 256, kin = 1024, rows = 96
        let cols = (0 ..< experts).map { _ in [Int32(3), Int32(900)] }
        let gate = projection(&rng, k: k, experts: experts, outputs: n, inputs: kin, columns: cols, columnScales: false)
        let up = projection(&rng, k: k, experts: experts, outputs: n, inputs: kin, columns: cols, columnScales: false)
        // Sorted expert ids with a ragged split (segments of 7, 40, 0, 49 rows).
        let ids: [UInt32] = Array(repeating: 0, count: 7) + Array(repeating: 1, count: 40) + Array(repeating: 3, count: 49)
        let x = (0 ..< rows * kin).map { _ in Float(Float16(rng.normal())) }  // already "rotated" rows
        let raw = (0 ..< rows * kin).map { _ in rng.normal() }
        let limit: Float = 0.3
        var single = [Float](), fused = [Float](), outl = [Float](), fusedOut = [Float]()
        for r in 0 ..< rows {
            let e = Int(ids[r]), xr = Array(x[r * kin ..< (r + 1) * kin])
            for c in 0 ..< n {
                let g = dot(gate.row(e, c), xr) * gate.scale[e * n + c]
                let u = dot(up.row(e, c), xr) * up.scale[e * n + c]
                single.append(g)
                let gc = min(g, limit), uc = min(max(u, -limit), limit)
                fused.append(gc / (1 + exp(-gc)) * uc)
                var o: Float = 0, ou: Float = 0
                for (j, col) in gate.columns(e).enumerated() {
                    o += raw[r * kin + col] * gate.wout[(e * n + c) * gate.outliers + j]
                    ou += raw[r * kin + col] * up.wout[(e * n + c) * up.outliers + j]
                }
                outl.append(o)
                let go = min(g + o, limit), uo = min(max(u + ou, -limit), limit)
                fusedOut.append(go / (1 + exp(-go)) * uo)
            }
        }
        let g = try gate.mlx(), u = try up.mlx()
        let xm = MLXArray(x, [rows, kin]).asType(.bfloat16), idx = MLXArray(ids)
        let ySingle = JANGTPrefillKernel.project(xm, g, indices: idx).asType(.float32).asArray(Float.self)
        let yFused = JANGTPrefillKernel.project(xm, g, up: u, indices: idx, limit: limit).asType(.float32).asArray(Float.self)
        let yOut = try XCTUnwrap(JANGTPrefillKernel.outliers(MLXArray(raw, [rows, kin]), g, indices: idx)).asArray(Float.self)
        let yFusedOut = JANGTPrefillKernel.projectGateUpWithOutliers(
            xm, rawRows: MLXArray(raw, [rows, kin]), gate: g, up: u, columns: MLXArray([Int32(3), Int32(900)]),
            indices: idx, limit: limit).asType(.float32).asArray(Float.self)
        let e4 = relL2(yFusedOut, fusedOut)
        XCTAssertLessThan(e4, 2e-2, "fused gate/up with outlier epilogue")
        let e1 = relL2(ySingle, single), e2 = relL2(yFused, fused), e3 = relL2(yOut, outl)
        print("JANGT prefill K=\(k): single rel-L2 \(e1), fused \(e2), outliers \(e3), fused+outliers \(e4)")
        XCTAssertLessThan(e1, 1e-2)
        XCTAssertLessThan(e2, 2e-2)
        XCTAssertLessThan(e3, 1e-3)  // fp32 gather_mm runs in TF32 under MLX defaults (as in the reference)
    }

    /// The expert-aligned schedule (`JANGTPrefillKernel.expertTilePlan`) returns the row-tiled schedule's output BIT
    /// FOR BIT: every variant (single, fused SwiGLU clamped, fused + outlier epilogue, H32 output rotation), K 1.5 / 2 /
    /// 2.5 / 3, ragged segments (empty experts, segments of 1, of exactly 64, of more than two tiles, row counts that
    /// are not multiples of 64), a call of < 8 rows, and rows whose id is not a valid expert (NaN in both). Control: a
    /// deliberately wrong plan (a segment shifted by one row) changes the output.
    func testExpertAlignedTilesEqualRowTiledBitForBit() throws {
        try XCTSkipUnless(JANGTPrefillKernel.available, "NAX prefill tiles need an M5-class GPU")
        func bits(_ a: MLXArray) -> [UInt16] { a.asType(.bfloat16).view(dtype: .uint16).asArray(UInt16.self) }
        let layouts: [[Int]] = [
            [7, 40, 0, 49],                // the runPrefill split (96 rows)
            [64, 1, 0, 0, 130, 64, 3, 65], // exact tile, singletons, empty experts, > 2 tiles (327 rows)
            [0, 5, 0, 0],                  // 5 rows: < 8 (padded ids) and one tile
            [200, 0, 17, 39, 0, 1, 0, 2],  // 259 rows
        ]
        var controls = 0
        for (li, counts) in layouts.enumerated() {
            for k in [1.5, 2, 2.5, 3] {
                var rng = Rng(state: UInt64(k * 1000) + UInt64(li) * 7919 + 5)
                let experts = counts.count, n = 128, kin = 512
                let cols = (0 ..< experts).map { _ in [Int32(3), Int32(400)] }
                let gate = try projection(&rng, k: k, experts: experts, outputs: n, inputs: kin, columns: cols,
                                          columnScales: false).mlx()
                let up = try projection(&rng, k: k, experts: experts, outputs: n, inputs: kin, columns: cols,
                                        columnScales: false).mlx()
                var ids: [UInt32] = []
                for (e, c) in counts.enumerated() { ids += Array(repeating: UInt32(e), count: c) }
                let rows = ids.count
                let x = MLXArray((0 ..< rows * kin).map { _ in rng.normal() }, [rows, kin]).asType(.bfloat16)
                let xo = MLXArray((0 ..< rows * 2).map { _ in rng.normal() }, [rows, 2])
                let idx = MLXArray(ids)
                let plan = JANGTPrefillKernel.expertTilePlan(sortedIndices: idx, experts: experts)
                let cases: [(String, (JANGTPrefillKernel.ExpertTilePlan?) -> MLXArray)] = [
                    ("single", { JANGTPrefillKernel.project(x, gate, indices: idx, plan: $0) }),
                    ("fused", { JANGTPrefillKernel.project(x, gate, up: up, indices: idx, limit: 0.3, plan: $0) }),
                    ("fused_h32", { JANGTPrefillKernel.project(x, gate, up: up, indices: idx, rotateOutput: true,
                                                               plan: $0) }),
                    ("fused_outliers", { JANGTPrefillKernel.projectGateUpWithOutliers(
                        x, outlierRows: xo, gate: gate, up: up, columns: 2, indices: idx, limit: 0.3, plan: $0) }),
                ]
                for (name, f) in cases {
                    let a = bits(f(nil)), b = bits(f(plan))
                    XCTAssertEqual(a, b, "layout \(li) K=\(k) \(name): expert tiles != row tiles")
                }
                if li == 1 && k == 2 {
                    // Control: a plan whose second segment starts one row late misses that row (and computes the next
                    // expert's first row with the wrong weights): the comparison above can fail.
                    var s = plan.starts.asArray(Int32.self)  // 8 experts: 10 entries, no padding
                    s[1] += 1
                    let bad = JANGTPrefillKernel.ExpertTilePlan(
                        starts: MLXArray(s), tiles: plan.tiles, segments: plan.segments, rows: plan.rows)
                    let a = bits(JANGTPrefillKernel.project(x, gate, indices: idx))
                    let b = bits(JANGTPrefillKernel.project(x, gate, indices: idx, plan: bad))
                    XCTAssertNotEqual(a, b, "control: a wrong plan must change the output")
                    controls += 1
                }
            }
        }
        XCTAssertEqual(controls, 1)
        // Rows whose id is not a valid expert: NaN rows on both schedules (the last plan segment), others equal.
        var rng = Rng(state: 99)
        let gate = try projection(&rng, k: 2, experts: 3, outputs: 64, inputs: 512, columns: [[3], [4], [5]],
                                  columnScales: false).mlx()
        let ids: [UInt32] = Array(repeating: 0, count: 10) + Array(repeating: 2, count: 70) + [7, 9, 9]
        let x = MLXArray((0 ..< ids.count * 512).map { _ in rng.normal() }, [ids.count, 512]).asType(.bfloat16)
        let idx = MLXArray(ids)
        let plan = JANGTPrefillKernel.expertTilePlan(sortedIndices: idx, experts: 3)
        let a = JANGTPrefillKernel.project(x, gate, indices: idx).asType(.float32)
        let b = JANGTPrefillKernel.project(x, gate, indices: idx, plan: plan).asType(.float32)
        XCTAssertEqual(isNaN(a[80...]).all().item(Bool.self), true, "row tiles: invalid ids -> NaN")
        XCTAssertEqual(isNaN(b[80...]).all().item(Bool.self), true, "expert tiles: invalid ids -> NaN")
        XCTAssertEqual(bits(a[..<80]), bits(b[..<80]))
        XCTAssertEqual(plan.starts.asArray(Int32.self), [0, 10, 10, 80, 83, 83, 83, 83])  // padded to 8 (device ptr)
        XCTAssertEqual(plan.tiles.asArray(Int32.self), [0, 1, 1, 3, 4, 4, 4, 4])
    }

    func testPrefillTilesMatchCPUReferenceK2() throws { try runPrefill(k: 2) }
    func testPrefillTilesMatchCPUReferenceK25() throws { try runPrefill(k: 2.5) }
    func testPrefillTilesMatchCPUReferenceK3() throws { try runPrefill(k: 3) }

    func testKernelsMatchCPUReferenceK2() throws { try runGateUpAndDown(k: 2, limit: 0, largeHidden: false) }
    func testKernelsMatchCPUReferenceK25Clamped() throws { try runGateUpAndDown(k: 2.5, limit: 0.25, largeHidden: false) }
    func testKernelsMatchCPUReferenceK3LargeHidden() throws { try runGateUpAndDown(k: 3, limit: 0, largeHidden: true) }
}

/// Decode-kernel throughput at GLM-5.3-Flash JANGHT shapes (288 experts, top-8, one token). Diagnostic: runs only
/// with JANGT_PERF=1 and prints effective weight GB/s per launch.
final class JANGTKernelPerfTests: XCTestCase {
    func testDecodeKernelThroughputAtGLMShapes() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["JANGT_PERF"] == "1")
        let e = 288, d = 4096, i = 2048, slots = 8
        func proj(_ k: Double, out: Int, inp: Int, cols: Int, cs: Bool) throws -> JANGTProjection {
            let w = JANGTCode.wordsPerRow(inputs: inp, k: k)
            return try JANGTProjection(
                k: k, packed: MLXRandom.randInt(low: 0, high: Int32.max, [e, out, w]).asType(.uint32),
                scale: MLXArray.ones([e, out]) * 0.01, signs: MLXArray.ones([inp]),
                outlierColumnIndices: MLXArray.zeros([e, cols], dtype: .int32) + MLXArray(Int32(7)),
                outlierWeights: MLXArray.zeros([e, out, cols], dtype: .float16),
                columnScales: cs ? MLXArray.ones([e, inp]) : nil)
        }
        let gate = try proj(2, out: i, inp: d, cols: 1, cs: false), up = try proj(2, out: i, inp: d, cols: 1, cs: false)
        let down = try proj(2.5, out: d, inp: i, cols: 8, cs: true)
        eval(gate.packed, up.packed, down.packed)
        let x = MLXRandom.normal([1, d]).asType(.bfloat16)
        let ids = MLXArray((0 ..< slots).map { UInt32($0 * 31) }, [1, slots])
        let w = MLXArray.ones([1, slots]) / Float(slots)
        let su = gate.signs, sv = down.columnScales! * down.signs.expandedDimensions(axis: 0)
        func time(_ name: String, bytes: Int, _ body: () -> MLXArray) {
            for _ in 0 ..< 20 { eval(body()) }
            let n = 300
            let t0 = Date()
            for _ in 0 ..< n { eval(body()) }
            let dt = Date().timeIntervalSince(t0) / Double(n)
            print(String(format: "JANGT_PERF %@ %.1f us  %.0f GB/s", name, dt * 1e6, Double(bytes) / dt / 1e9))
        }
        let xr = JANGTKernels.rotateRows(x, signs: su)
        let h = JANGTKernels.gateUp(xr: xr, x: x, indices: ids, gate: gate, up: up, limit: 10, outputDType: .float32)
        let hr = JANGTKernels.rotateRows(h, signs: sv, rowExperts: ids.flattened())
        eval(xr, h, hr)
        time("rotate_x", bytes: d * 4) { JANGTKernels.rotateRows(x, signs: su) }
        time("gate_up", bytes: 2 * slots * i * gate.words * 4) {
            JANGTKernels.gateUp(xr: xr, x: x, indices: ids, gate: gate, up: up, limit: 10, outputDType: .float32)
        }
        time("rotate_h", bytes: slots * i * 8) { JANGTKernels.rotateRows(h, signs: sv, rowExperts: ids.flattened()) }
        time("down", bytes: slots * d * down.words * 4) {
            JANGTKernels.down(hr: hr, h: h, indices: ids, weights: w, projection: down)
        }
        // 42 chained layers per eval (one GLM decode token's routed experts): no per-launch sync in the number.
        time("x42_layers", bytes: 42 * (2 * slots * i * gate.words * 4 + slots * d * down.words * 4)) {
            var y = x
            for _ in 0 ..< 42 {
                let a = JANGTKernels.rotateRows(y, signs: su)
                let b = JANGTKernels.gateUp(xr: a, x: y, indices: ids, gate: gate, up: up, limit: 10, outputDType: .float32)
                let c = JANGTKernels.rotateRows(b, signs: sv, rowExperts: ids.flattened())
                y = JANGTKernels.down(hr: c, h: b, indices: ids, weights: w, projection: down).asType(.bfloat16)
            }
            return y
        }
        time("x42_gate_up_only", bytes: 42 * 2 * slots * i * gate.words * 4) {
            var y = x
            for _ in 0 ..< 42 {
                let b = JANGTKernels.gateUp(xr: xr, x: y, indices: ids, gate: gate, up: up, limit: 10, outputDType: .float32)
                y = y + b[0 ..< 1, 0 ..< 1].asType(.bfloat16)
            }
            return y
        }
        time("whole_layer", bytes: 2 * slots * i * gate.words * 4 + slots * d * down.words * 4) {
            let a = JANGTKernels.rotateRows(x, signs: su)
            let b = JANGTKernels.gateUp(xr: a, x: x, indices: ids, gate: gate, up: up, limit: 10, outputDType: .float32)
            let c = JANGTKernels.rotateRows(b, signs: sv, rowExperts: ids.flattened())
            return JANGTKernels.down(hr: c, h: b, indices: ids, weights: w, projection: down)
        }
    }
}

/// Prefill kernel throughput at GLM-5.3 shapes: 2,048 tokens x top-8 = 16,384 expert-sorted rows over 288 experts.
/// Diagnostic (JANGT_PERF=1). Reference (vMLX Python, M5 Max): fused gate/up 19.3 ms per layer-chunk.
final class JANGTPrefillPerfTests: XCTestCase {
    func testPrefillKernelTimingAtGLMShapes() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["JANGT_PERF"] == "1" && JANGTPrefillKernel.available)
        let e = 288, d = 4096, i = 2048, rows = 16384
        func proj(_ k: Double, out: Int, inp: Int) throws -> JANGTProjection {
            let w = JANGTCode.wordsPerRow(inputs: inp, k: k)
            return try JANGTProjection(
                k: k, packed: MLXRandom.randInt(low: 0, high: Int32.max, [e, out, w]).asType(.uint32),
                scale: MLXArray.ones([e, out]) * 0.01, signs: MLXArray.ones([inp]),
                outlierColumnIndices: MLXArray.zeros([e, 1], dtype: .int32) - MLXArray(Int32(1)),
                outlierWeights: MLXArray.zeros([e, out, 1], dtype: .float16), columnScales: nil)
        }
        let gate = try proj(2, out: i, inp: d), up = try proj(2, out: i, inp: d), down = try proj(2.5, out: d, inp: i)
        eval(gate.packed, up.packed, down.packed)
        // Router-like expert loads: every expert gets rows, uneven.
        let ids = MLXArray((0 ..< rows).map { UInt32(($0 * 7919 % 9973) % e) })
        let sorted = take(ids, argSort(ids))
        let x = MLXRandom.normal([rows, d]).asType(.bfloat16), h = MLXRandom.normal([rows, i]).asType(.bfloat16)
        eval(sorted, x, h)
        func time(_ name: String, _ body: () -> MLXArray) {
            for _ in 0 ..< 3 { eval(body()) }
            let n = 10, t0 = Date()
            for _ in 0 ..< n { eval(body()) }
            print(String(format: "JANGT_PREFILL_PERF %@ %.2f ms", name, Date().timeIntervalSince(t0) / Double(n) * 1e3))
        }
        time("gate_up_fused") { JANGTPrefillKernel.project(x, gate, up: up, indices: sorted, limit: 10) }
        time("gate_up_fused_rot") { JANGTPrefillKernel.project(x, gate, up: up, indices: sorted, limit: 10, rotateOutput: true) }
        time("down_K2.5") { JANGTPrefillKernel.project(h, down, indices: sorted) }
        let downO = try JANGTProjection(
            k: 2.5, packed: down.packed, scale: down.scale, signs: down.signs,
            outlierColumnIndices: MLXRandom.randInt(low: 0, high: Int32(i), [e, 8]).asType(.int32),
            outlierWeights: (MLXRandom.normal([e, d, 8]) * 0.01).asType(.float16), columnScales: MLXArray.ones([e, i]))
        let sv = downO.columnScales! * downO.signs.expandedDimensions(axis: 0)
        eval(downO.outlierColumnIndices, downO.outlierWeights, sv)
        time("rotate_rows_per_expert") { JANGTKernels.rotateRows(h, signs: sv, rowExperts: sorted) }
        time("outliers_gather_mm") { JANGTPrefillKernel.outliers(h, downO, indices: sorted)! }
        let y = MLXRandom.normal([rows, d]).asType(.bfloat16), w = MLXRandom.uniform(0 ..< 1, [2048, 8])
        let order = argSort(ids)
        eval(y, w, order)
        time("unsort_combine") {
            let restored = take(y, argSort(order), axis: 0).reshaped(2048, 8, d).asType(.float32)
            return matmul(w.reshaped(2048, 1, 8), restored).reshaped(2048, d)
        }
        time("sort_glue") {
            let order = argSort(ids); let s = take(ids, order); let xs = take(x, order.floorDivide(8), axis: 0)
            return xs.sum() + s.asType(.float32).sum()
        }
    }
}
