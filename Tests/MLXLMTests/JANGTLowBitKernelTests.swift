import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// Qwen3.8-Flash-Next JANGHT (2026-10-16): sub-2-bit trellis widths K = 1 and 1.5 next to K = 2 and 2.5, at the
/// routed-expert geometry D = 2560, I = 640. Down has kin = 640, which is NOT a multiple of the 512-value decode block,
/// so every down test here runs the decode kernels' masked tail block (lanes 0..7 of the second block).
///
/// Oracle chain:
///  1. `JANGTEncoderGolden` (generated from the encoder's own `jang_tools/jangt/v2l.py` + the campaign `jt_patch.py`)
///     pins the code: multipliers, step patterns, words per row, every state's value pair, and a decoded 640-wide row.
///  2. The CPU decoder below reads each row's bitstream at the GLOBAL step offsets (no lane/block geometry) and must
///     reproduce the encoder golden bit for bit.
///  3. The GPU decode and NAX prefill kernels are compared against that decoder: every single weight (one-hot inputs,
///     fp32/bf16-exact) and the composed SwiGLU / weighted-down paths (rel-L2 against a float64 dense reference).
final class JANGTLowBitKernelTests: XCTestCase {
    static let widths: [Double] = [1, 1.5, 2, 2.5]

    struct Rng {
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

    /// Value pair of every state (the kernels' exact float math), state-major.
    static func stateTable(_ k: Double) -> [Float] {
        var t = [Float](repeating: 0, count: 2 << JANGTCode.stateBits)
        for s in 0 ..< (1 << JANGTCode.stateBits) {
            let (a, b) = JANGTCode.values(state: UInt32(s), k: k)
            t[2 * s] = a
            t[2 * s + 1] = b
        }
        return t
    }

    /// CPU decoder at global step offsets: rows (0-based over E*N rows) of `packed` -> kin normalized values each.
    /// `codeK` swaps only the value code (multiplier and statistics) for the negative control.
    static func decodeRows(_ packed: [UInt32], words: Int, k: Double, inputs: Int, rows: Range<Int>,
                           codeK: Double? = nil) -> [Float] {
        let so = JANGTCode.stepOffsets(k, inputs / 2)
        let table = stateTable(codeK ?? k)
        let mask = UInt64((1 << JANGTCode.stateBits) - 1)
        var out = [Float](repeating: 0, count: rows.count * inputs)
        packed.withUnsafeBufferPointer { p in
            out.withUnsafeMutableBufferPointer { o in
                for (ri, r) in rows.enumerated() {
                    let base = r * words
                    for t in 0 ..< inputs / 2 {
                        let bit = so[t], wi = base + (bit >> 5), sh = UInt64(bit & 31)
                        let w = UInt64(p[wi]) | (UInt64(p[wi + 1]) << 32)
                        let s = Int((w >> sh) & mask)
                        o[ri * inputs + 2 * t] = table[2 * s]
                        o[ri * inputs + 2 * t + 1] = table[2 * s + 1]
                    }
                }
            }
        }
        return out
    }

    static func fnv(_ bits: [UInt32]) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for v in bits {
            var x = v.littleEndian
            for _ in 0 ..< 4 {
                h ^= UInt64(x & 0xFF)
                h = h &* 0x100_0000_01b3
                x >>= 8
            }
        }
        return h
    }

    /// One synthetic JANGT projection plus its CPU-decoded rows.
    struct Projection {
        let k: Double, experts: Int, outputs: Int, inputs: Int, words: Int
        let packed: [UInt32], scale: [Float], signs: [Float], ocol: [Int32], outliers: Int, wout: [Float]
        let cs: [Float]?
        /// (E * N * kin) normalized (unscaled) trellis values.
        let values: [Float]

        init(_ rng: inout Rng, k: Double, experts: Int, outputs: Int, inputs: Int, columns: [[Int32]],
             columnScales: Bool) {
            self.k = k
            self.experts = experts
            self.outputs = outputs
            self.inputs = inputs
            words = JANGTCode.wordsPerRow(inputs: inputs, k: k)
            outliers = columns[0].count
            // Any bitstream is a valid trellis path.
            packed = (0 ..< experts * outputs * words).map { _ in rng.word() }
            scale = (0 ..< experts * outputs).map { _ in 0.01 + 0.01 * rng.uniform() }
            signs = (0 ..< inputs).map { _ in rng.uniform() < 0.5 ? -1 : 1 }
            ocol = columns.flatMap { $0 }
            // fp16 round trip so the reference sees the kernel's outlier weights exactly.
            wout = (0 ..< experts * outputs * outliers).map { _ in Float(Float16(rng.normal() * 0.05)) }
            cs = columnScales ? (0 ..< experts * inputs).map { _ in 0.5 + rng.uniform() } : nil
            values = JANGTLowBitKernelTests.decodeRows(packed, words: words, k: k, inputs: inputs,
                                                       rows: 0 ..< experts * outputs)
        }

        func columns(_ e: Int) -> [(slot: Int, column: Int)] {
            (0 ..< outliers).compactMap { j in
                let c = Int(ocol[e * outliers + j])
                return c >= 0 ? (j, c) : nil
            }
        }

        /// Effective dense weight W[e][n][c] as the kernels apply it when the SAME vector feeds the rotated and the
        /// raw input (one-hot tests): trellis value * row scale, plus the outlier weight on outlier columns.
        func oneHotWeight(_ e: Int, _ n: Int, _ c: Int) -> Float {
            var w = values[(e * outputs + n) * inputs + c] * scale[e * outputs + n]
            for (j, col) in columns(e) where col == c { w += wout[(e * outputs + n) * outliers + j] }
            return w
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

    static func h128(_ v: [Float]) -> [Float] {
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

    static func dot(_ a: UnsafeBufferPointer<Float>, _ offset: Int, _ b: [Float]) -> Double {
        var s = 0.0
        for i in b.indices { s += Double(a[offset + i]) * Double(b[i]) }
        return s
    }

    /// Round to bf16 (round-to-nearest-even, MLX's own conversion), back to Float.
    static func bf16(_ v: [Float]) -> [Float] {
        MLXArray(v).asType(.bfloat16).asType(.float32).asArray(Float.self)
    }

    static func relL2(_ got: [Float], _ want: [Float]) -> Double {
        precondition(got.count == want.count)
        var num = 0.0, den = 0.0
        for i in want.indices {
            num += Double(got[i] - want[i]) * Double(got[i] - want[i])
            den += Double(want[i]) * Double(want[i])
        }
        return (num / max(den, 1e-30)).squareRoot()
    }

    /// Fail closed: a reference that is empty or all zeros would make every comparison vacuous.
    static func assertLive(_ v: [Float], _ what: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertFalse(v.isEmpty, "\(what): empty baseline", file: file, line: line)
        let nonzero = v.lazy.filter { $0 != 0 && $0.isFinite }.count
        XCTAssertGreaterThan(nonzero, v.count / 2, "\(what): baseline is mostly zero or non-finite", file: file,
                             line: line)
    }

    // MARK: - code definition vs the encoder

    func testCodeMatchesEncoderGolden() {
        XCTAssertEqual(JANGTEncoderGolden.widths.count, 5)
        for g in JANGTEncoderGolden.widths {
            XCTAssertTrue(JANGTCode.isSupported(g.k), "K=\(g.k)")
            XCTAssertEqual(JANGTCode.mult[g.k], g.mult, "multiplier K=\(g.k)")
            XCTAssertTrue(JANGTCode.kpat(g.k) == g.kpat, "kpat K=\(g.k)")
            XCTAssertEqual(JANGTCode.wordsPerRow(inputs: 640, k: g.k), g.wordsPerRow640, "W(640) K=\(g.k)")
            XCTAssertEqual(JANGTCode.wordsPerRow(inputs: 2560, k: g.k), g.wordsPerRow2560, "W(2560) K=\(g.k)")
            XCTAssertEqual(Self.fnv(Self.stateTable(g.k).map(\.bitPattern)), g.stateTableFNV,
                           "every state's value pair K=\(g.k)")
            let words = JANGTCode.wordsPerRow(inputs: 640, k: g.k)
            XCTAssertEqual(g.packed.count, g.rows * words)
            let decoded = Self.decodeRows(g.packed, words: words, k: g.k, inputs: 640, rows: 0 ..< g.rows)
            XCTAssertEqual(Self.fnv(decoded.map(\.bitPattern)), g.valuesFNV, "decoded 640-wide rows K=\(g.k)")
            for (i, bits) in g.probes { XCTAssertEqual(decoded[i].bitPattern, bits, "K=\(g.k) value \(i)") }
        }
    }

    func testLaneGeometryAtQwenWidths() {
        // Bits per 16-weight lane and the window rule (Python `_window`): K=1/1.5 load 2 words realigned into one.
        let expected: [Double: (Int, Int, Int)] = [1: (16, 16, 2), 1.5: (24, 24, 2), 2: (32, 0, 2), 2.5: (40, 24, 3),
                                                   3: (48, 16, 3)]
        for (k, e) in expected {
            let w = JANGTCode.window(k)
            XCTAssertEqual(w.pl, e.0, "PL K=\(k)")
            XCTAssertEqual(w.maxShift, e.1, "max shift K=\(k)")
            XCTAssertEqual(w.words, e.2, "window words K=\(k)")
        }
        // K=1.5: so[7] + L = 20 + 12 = 32 (all 8 states in one realigned word).
        XCTAssertEqual(JANGTCode.stepOffsets(1.5, 9)[7] + JANGTCode.stateBits, 32)
        // The last tail lane's window stays inside words_per_row at kin = 640 for every width.
        for k in Self.widths + [3] {
            let w = JANGTCode.window(k), words = JANGTCode.wordsPerRow(inputs: 640, k: k)
            let lastLaneBit = 1 * 32 * w.pl + 7 * w.pl
            XCTAssertLessThanOrEqual((lastLaneBit >> 5) + w.words, words, "K=\(k)")
        }
    }

    func testDeclaredCodeTablesAreChecked() {
        let good: [String: Any] = [
            "code_multipliers": ["1": 590_428_815, "1.5": 2_124_823_417, "2": 636_231_159, "2.5": 636_231_159],
            "step_bits": ["1": [2, 2], "1.5": [4, 2], "2": [4, 4], "2.5": [6, 4]],
        ]
        XCTAssertNil(JANGTCode.validateDeclaredCode(good))
        XCTAssertNil(JANGTCode.validateDeclaredCode([:]))
        var badMult = good
        badMult["code_multipliers"] = ["1.5": 636_231_159]
        XCTAssertNotNil(JANGTCode.validateDeclaredCode(badMult))
        var badSteps = good
        badSteps["step_bits"] = ["1.5": [2, 4]]
        XCTAssertNotNil(JANGTCode.validateDeclaredCode(badSteps))
        var unknown = good
        unknown["code_multipliers"] = ["0.5": 1]
        XCTAssertNotNil(JANGTCode.validateDeclaredCode(unknown))
    }

    // MARK: - decode kernels, every weight (one-hot inputs)

    /// Weighted down with one-hot ROTATED and RAW inputs: output row t is column t of the effective weight matrix of
    /// one expert. Every weight, including the 512..639 tail and the outlier columns, at fp32-exact tolerance.
    private func runDownEveryWeight(k: Double, kin: Int, outputs: Int) throws {
        var rng = Rng(state: UInt64(k * 1000) + UInt64(kin))
        let experts = 2, e = 1
        let cols: [[Int32]] = [[3, 600 % Int32(kin), -1, -1], [7, Int32(kin - 1), 513 % Int32(kin), -1]]
        let p = Projection(&rng, k: k, experts: experts, outputs: outputs, inputs: kin, columns: cols,
                           columnScales: true)
        let eye = MLXArray.eye(kin, dtype: .float32)
        let ids = MLXArray([UInt32](repeating: UInt32(e), count: kin), [kin, 1])
        let y = JANGTKernels.down(hr: eye, h: eye, indices: ids, weights: MLXArray.ones([kin, 1]),
                                  projection: try p.mlx()).asArray(Float.self)
        var want = [Float](repeating: 0, count: kin * outputs)
        for t in 0 ..< kin { for n in 0 ..< outputs { want[t * outputs + n] = p.oneHotWeight(e, n, t) } }
        Self.assertLive(want, "down one-hot K=\(k) kin=\(kin)")
        var bad = 0, tailBad = 0, worst: Float = 0
        for i in want.indices {
            let d = abs(y[i] - want[i])
            if d > 1e-6 * abs(want[i]) + 1e-8 {
                bad += 1
                if i / outputs >= kin / 512 * 512 { tailBad += 1 }
            }
            worst = max(worst, d / max(abs(want[i]), 1e-12))
        }
        print("JANGT_LOWBIT down every-weight K=\(k) kin=\(kin): \(bad)/\(want.count) wrong (tail \(tailBad)), "
              + "worst rel \(worst)")
        XCTAssertEqual(bad, 0, "down K=\(k) kin=\(kin): \(bad) of \(want.count) weights wrong (\(tailBad) in the tail)")
    }

    /// Gate/up + SwiGLU with one-hot inputs: out[t][n] = silu(Wg[n][t]) * Wu[n][t] for every (n, t).
    private func runGateUpEveryWeight(k: Double, kin: Int, outputs: Int) throws {
        var rng = Rng(state: UInt64(k * 100) + UInt64(kin) + 5)
        let experts = 2, e = 1
        let cols = (0 ..< experts).map { _ in [Int32(2), Int32(kin - 3), Int32(kin / 2 + 1)] }   // layer-wide
        let g = Projection(&rng, k: k, experts: experts, outputs: outputs, inputs: kin, columns: cols,
                           columnScales: false)
        let u = Projection(&rng, k: k, experts: experts, outputs: outputs, inputs: kin, columns: cols,
                           columnScales: false)
        let eye = MLXArray.eye(kin, dtype: .float32)
        let ids = MLXArray([UInt32](repeating: UInt32(e), count: kin), [kin, 1])
        let h = JANGTKernels.gateUp(xr: eye, x: eye, indices: ids, gate: try g.mlx(), up: try u.mlx(), limit: 0,
                                    outputDType: .float32).asArray(Float.self)
        var want = [Float](repeating: 0, count: kin * outputs)
        for t in 0 ..< kin {
            for n in 0 ..< outputs {
                let gv = g.oneHotWeight(e, n, t), uv = u.oneHotWeight(e, n, t)
                want[t * outputs + n] = gv / (1 + exp(-gv)) * uv
            }
        }
        Self.assertLive(want, "gate/up one-hot K=\(k) kin=\(kin)")
        var bad = 0, tailBad = 0
        for i in want.indices where abs(h[i] - want[i]) > 2e-5 * abs(want[i]) + 1e-9 {
            bad += 1
            if i / outputs >= kin / 512 * 512 { tailBad += 1 }
        }
        print("JANGT_LOWBIT gate/up every-weight K=\(k) kin=\(kin): \(bad)/\(want.count) wrong (tail \(tailBad))")
        XCTAssertEqual(bad, 0, "gate/up K=\(k) kin=\(kin): \(bad) of \(want.count) wrong (\(tailBad) in the tail)")
    }

    func testDownEveryWeightK1Kin640() throws { try runDownEveryWeight(k: 1, kin: 640, outputs: 2560) }
    func testDownEveryWeightK15Kin640() throws { try runDownEveryWeight(k: 1.5, kin: 640, outputs: 2560) }
    func testDownEveryWeightK2Kin640() throws { try runDownEveryWeight(k: 2, kin: 640, outputs: 2560) }
    func testDownEveryWeightK25Kin640() throws { try runDownEveryWeight(k: 2.5, kin: 640, outputs: 2560) }
    func testDownEveryWeightK15Kin2560() throws { try runDownEveryWeight(k: 1.5, kin: 2560, outputs: 64) }
    func testDownEveryWeightK1Kin2560() throws { try runDownEveryWeight(k: 1, kin: 2560, outputs: 64) }

    func testGateUpEveryWeightK1Kin2560() throws { try runGateUpEveryWeight(k: 1, kin: 2560, outputs: 640) }
    func testGateUpEveryWeightK15Kin2560() throws { try runGateUpEveryWeight(k: 1.5, kin: 2560, outputs: 640) }
    func testGateUpEveryWeightK2Kin2560() throws { try runGateUpEveryWeight(k: 2, kin: 2560, outputs: 640) }
    func testGateUpEveryWeightK25Kin2560() throws { try runGateUpEveryWeight(k: 2.5, kin: 2560, outputs: 640) }
    func testGateUpEveryWeightK1Kin640() throws { try runGateUpEveryWeight(k: 1, kin: 640, outputs: 256) }
    func testGateUpEveryWeightK15Kin640() throws { try runGateUpEveryWeight(k: 1.5, kin: 640, outputs: 256) }
    func testGateUpEveryWeightK2Kin640() throws { try runGateUpEveryWeight(k: 2, kin: 640, outputs: 256) }
    func testGateUpEveryWeightK25Kin640() throws { try runGateUpEveryWeight(k: 2.5, kin: 640, outputs: 256) }

    // MARK: - decode kernels, composed routed path vs dense reference

    /// The `JANGTRoutedExpertLayer` decode composition at the Qwen geometry (D=2560, I=640, top-k slots over E
    /// experts): rotation with layer-wide gate/up outlier columns masked, gate/up SwiGLU with outliers from the raw x,
    /// per-expert down signs folded with column scales, weighted down with per-expert -1-padded outlier columns.
    private func runComposed(kGateUp: Double, kDown: Double) throws -> (Double, Double) {
        var rng = Rng(state: UInt64(kGateUp * 1000 + kDown * 10) + 17)
        let experts = 6, d = 2560, i = 640, tokens = 2, slots = 4
        let guCols = (0 ..< experts).map { _ in [Int32(5), Int32(700), Int32(2559)] }
        let gate = Projection(&rng, k: kGateUp, experts: experts, outputs: i, inputs: d, columns: guCols,
                              columnScales: false)
        let up = Projection(&rng, k: kGateUp, experts: experts, outputs: i, inputs: d, columns: guCols,
                            columnScales: false)
        let dnCols: [[Int32]] = [[3, 600, -1, -1], [9, -1, -1, -1], [100, 520, 639, 300], [-1, -1, -1, -1],
                                 [512, 513, -1, -1], [0, -1, -1, -1]]
        let dn = Projection(&rng, k: kDown, experts: experts, outputs: d, inputs: i, columns: dnCols,
                            columnScales: true)
        let x = (0 ..< tokens * d).map { _ in rng.normal() }
        let idx: [UInt32] = [0, 2, 4, 5, 1, 3, 2, 4]
        let w = (0 ..< tokens * slots).map { _ in rng.uniform() }

        var keep = [Float](repeating: 1, count: d)
        for (_, c) in gate.columns(0) { keep[c] = 0 }
        var hRef = [Float](repeating: 0, count: tokens * slots * i)
        gate.values.withUnsafeBufferPointer { gv in
            up.values.withUnsafeBufferPointer { uv in
                for t in 0 ..< tokens {
                    let xt = Array(x[t * d ..< (t + 1) * d])
                    let xr = Self.h128((0 ..< d).map { xt[$0] * gate.signs[$0] * keep[$0] })
                    for s in 0 ..< slots {
                        let e = Int(idx[t * slots + s])
                        for n in 0 ..< i {
                            var g = Self.dot(gv, (e * i + n) * d, xr) * Double(gate.scale[e * i + n])
                            var u = Self.dot(uv, (e * i + n) * d, xr) * Double(up.scale[e * i + n])
                            for (j, c) in gate.columns(e) {
                                g += Double(xt[c]) * Double(gate.wout[(e * i + n) * gate.outliers + j])
                                u += Double(xt[c]) * Double(up.wout[(e * i + n) * up.outliers + j])
                            }
                            hRef[(t * slots + s) * i + n] = Float(g / (1 + Foundation.exp(-g)) * u)
                        }
                    }
                }
            }
        }
        var yRef = [Float](repeating: 0, count: tokens * d)
        dn.values.withUnsafeBufferPointer { dv in
            for t in 0 ..< tokens {
                var acc = [Double](repeating: 0, count: d)
                for s in 0 ..< slots {
                    let ts = t * slots + s, e = Int(idx[ts])
                    let h = Array(hRef[ts * i ..< (ts + 1) * i])
                    var keepE = [Float](repeating: 1, count: i)
                    for (_, c) in dn.columns(e) { keepE[c] = 0 }
                    let hr = Self.h128((0 ..< i).map { h[$0] * dn.cs![e * i + $0] * dn.signs[$0] * keepE[$0] })
                    for n in 0 ..< d {
                        var y = Self.dot(dv, (e * d + n) * i, hr) * Double(dn.scale[e * d + n])
                        for (j, c) in dn.columns(e) { y += Double(h[c]) * Double(dn.wout[(e * d + n) * dn.outliers + j]) }
                        acc[n] += Double(w[ts]) * y
                    }
                }
                for n in 0 ..< d { yRef[t * d + n] = Float(acc[n]) }
            }
        }
        Self.assertLive(hRef, "composed hidden reference")
        Self.assertLive(yRef, "composed output reference")

        let g = try gate.mlx(), u = try up.mlx(), down = try dn.mlx()
        XCTAssertEqual(g.outlierColumns, 3)
        XCTAssertEqual(down.outlierColumns, 4)
        let xm = MLXArray(x, [tokens, d]), ids = MLXArray(idx, [tokens, slots])
        let suGU = g.signs * JANGTRoutedExpertLayer.keepMask(width: d, columns: g.outlierColumnIndices[0 ..< 1])[0]
        let h = JANGTKernels.gateUp(xr: JANGTKernels.rotateRows(xm, signs: suGU), x: xm, indices: ids, gate: g, up: u,
                                    limit: 0, outputDType: .float32)
        let hErr = Self.relL2(h.asArray(Float.self), hRef)
        let svDN = down.columnScales! * down.signs.expandedDimensions(axis: 0)
            * JANGTRoutedExpertLayer.keepMask(width: i, columns: down.outlierColumnIndices)
        let hm = MLXArray(hRef, [tokens * slots, i])
        let y = JANGTKernels.down(hr: JANGTKernels.rotateRows(hm, signs: svDN, rowExperts: ids.flattened()), h: hm,
                                  indices: ids, weights: MLXArray(w, [tokens, slots]), projection: down)
        let yErr = Self.relL2(y.asArray(Float.self), yRef)
        print("JANGT_LOWBIT composed gate/up K=\(kGateUp) rel-L2 \(hErr), down K=\(kDown) (kin 640) rel-L2 \(yErr)")
        return (hErr, yErr)
    }

    func testComposedDecodeMatchesDenseReferenceAllWidths() throws {
        var errors: [Double: (Double, Double)] = [:]
        for k in Self.widths { errors[k] = try runComposed(kGateUp: k, kDown: k) }
        let base = errors[2]!
        for k in Self.widths {
            let (h, y) = errors[k]!
            // The error is the kernels' half2 inner products: the same order at every width.
            XCTAssertLessThan(h, 5e-3, "gate/up K=\(k)")
            XCTAssertLessThan(y, 5e-3, "down K=\(k)")
            XCTAssertLessThan(h, 3 * base.0 + 1e-4, "gate/up K=\(k) vs K=2")
            XCTAssertLessThan(y, 3 * base.1 + 1e-4, "down K=\(k) vs K=2")
        }
        // Mixed widths per (layer, gate+up | down), as the Qwen plan allocates them.
        let mixed = try runComposed(kGateUp: 1.5, kDown: 2)
        XCTAssertLessThan(max(mixed.0, mixed.1), 5e-3)
        let mixed2 = try runComposed(kGateUp: 1, kDown: 2.5)
        XCTAssertLessThan(max(mixed2.0, mixed2.1), 5e-3)
    }

    /// Negative control: the same rows decoded with the wrong code must NOT match (the comparison can fail).
    func testWrongCodeIsDetected() throws {
        var rng = Rng(state: 99)
        let p = Projection(&rng, k: 1.5, experts: 1, outputs: 64, inputs: 640, columns: [[-1]], columnScales: false)
        // The K=1.5 bit positions with the K=2 multiplier and statistics: the code alone must break the match.
        let swappedMultiplier = Self.decodeRows(p.packed, words: p.words, k: 1.5, inputs: 640, rows: 0 ..< 64,
                                                codeK: 2)
        XCTAssertGreaterThan(Self.relL2(swappedMultiplier, p.values), 0.3)
        let eye = MLXArray.eye(640, dtype: .float32)
        let y = JANGTKernels.down(hr: eye, h: MLXArray.zeros([640, 640]), indices: MLXArray.zeros([640, 1],
                                  dtype: .uint32), weights: MLXArray.ones([640, 1]), projection: try p.mlx())
        let got = y.asArray(Float.self)
        var right = [Float](repeating: 0, count: 640 * 64), wrong = right
        for t in 0 ..< 640 {
            for n in 0 ..< 64 {
                right[t * 64 + n] = p.values[n * 640 + t] * p.scale[n]
                wrong[t * 64 + n] = swappedMultiplier[n * 640 + t] * p.scale[n]
            }
        }
        XCTAssertLessThan(Self.relL2(got, right), 1e-6)
        XCTAssertGreaterThan(Self.relL2(got, wrong), 0.3)
    }

    // MARK: - NAX prefill tiles

    /// Expert-sorted prefill tiles with one-hot bf16 rows: every weight of two experts, bit-exact in bf16.
    private func runPrefillEveryWeight(
        k: Double, kin: Int, outputs: Int, backend: JANGHPrefillKernel.Backend? = nil, dtype: DType = .bfloat16
    ) throws {
        try XCTSkipUnless(backend == .steel || JANGTPrefillKernel.available, "NAX prefill tiles need an M5-class GPU")
        var rng = Rng(state: UInt64(k * 10) + UInt64(kin) + 31)
        let experts = 2
        let p = Projection(&rng, k: k, experts: experts, outputs: outputs, inputs: kin, columns: [[-1], [-1]],
                           columnScales: false)
        let eye = MLXArray.eye(kin, dtype: .float32)
        let x = concatenated([eye, eye], axis: 0).asType(dtype)
        let ids = MLXArray([UInt32](repeating: 0, count: kin) + [UInt32](repeating: 1, count: kin))
        let y = JANGTPrefillKernel.project(x, try p.mlx(), indices: ids, backend: backend).asType(.float32).asArray(Float.self)
        var exact = [Float](repeating: 0, count: experts * kin * outputs)
        for e in 0 ..< experts {
            for t in 0 ..< kin {
                for n in 0 ..< outputs {
                    exact[(e * kin + t) * outputs + n] = p.values[(e * outputs + n) * kin + t] * p.scale[e * outputs + n]
                }
            }
        }
        let want = MLXArray(exact).asType(dtype).asType(.float32).asArray(Float.self)
        Self.assertLive(want, "prefill one-hot K=\(k) kin=\(kin)")
        var bad = 0
        for i in want.indices where y[i].bitPattern != want[i].bitPattern { bad += 1 }
        print("JANGT_LOWBIT prefill every-weight backend=\(backend?.rawValue ?? "native") dtype=\(dtype) K=\(k) kin=\(kin): \(bad)/\(want.count) not dtype-exact")
        XCTAssertEqual(bad, 0, "prefill K=\(k) kin=\(kin)")
    }

    func testPrefillEveryWeightK1() throws {
        try runPrefillEveryWeight(k: 1, kin: 2560, outputs: 640)
        try runPrefillEveryWeight(k: 1, kin: 640, outputs: 2560)
    }
    func testPrefillEveryWeightK15() throws {
        try runPrefillEveryWeight(k: 1.5, kin: 2560, outputs: 640)
        try runPrefillEveryWeight(k: 1.5, kin: 640, outputs: 2560)
    }
    func testPrefillEveryWeightK2() throws {
        try runPrefillEveryWeight(k: 2, kin: 2560, outputs: 640)
        try runPrefillEveryWeight(k: 2, kin: 640, outputs: 2560)
    }
    func testPrefillEveryWeightK25() throws {
        try runPrefillEveryWeight(k: 2.5, kin: 2560, outputs: 640)
        try runPrefillEveryWeight(k: 2.5, kin: 640, outputs: 2560)
    }

    func testSteelPrefillEveryWeightAllWidthsAndDTypes() throws {
        for dtype in [DType.float16, .bfloat16] {
            for k in Self.widths + [3] {
                // Both Allosaurus projection widths, plus a ragged output tile.
                try runPrefillEveryWeight(k: k, kin: 2560, outputs: 40, backend: .steel, dtype: dtype)
                try runPrefillEveryWeight(k: k, kin: 640, outputs: 64, backend: .steel, dtype: dtype)
            }
        }
    }

    /// Fused gate/up prefill with the layer-wide outlier epilogue and the outlier term, sorted rows over a ragged
    /// expert split, at kin = 2560 (gate/up) and the 640-wide down.
    private func runPrefillComposed(k: Double, backend: JANGHPrefillKernel.Backend? = nil) throws -> (Double, Double) {
        var rng = Rng(state: UInt64(k * 100) + 77)
        let experts = 4, d = 2560, i = 640, rows = 72
        let cols = (0 ..< experts).map { _ in [Int32(3), Int32(1900)] }
        let gate = Projection(&rng, k: k, experts: experts, outputs: i, inputs: d, columns: cols, columnScales: false)
        let up = Projection(&rng, k: k, experts: experts, outputs: i, inputs: d, columns: cols, columnScales: false)
        let dnCols: [[Int32]] = [[5, 600, -1], [-1, -1, -1], [513, -1, -1], [0, 639, 2]]
        let dn = Projection(&rng, k: k, experts: experts, outputs: d, inputs: i, columns: dnCols, columnScales: true)
        let ids: [UInt32] = Array(repeating: 0, count: 5) + Array(repeating: 1, count: 30) + Array(repeating: 3, count: 37)
        let x = Self.bf16((0 ..< rows * d).map { _ in rng.normal() })   // already "rotated"
        let raw = (0 ..< rows * d).map { _ in rng.normal() }
        let hIn = Self.bf16((0 ..< rows * i).map { _ in rng.normal() })
        let hRaw = (0 ..< rows * i).map { _ in rng.normal() }
        var fusedRef = [Float](), downRef = [Float]()
        gate.values.withUnsafeBufferPointer { gv in
            up.values.withUnsafeBufferPointer { uv in
                dn.values.withUnsafeBufferPointer { dv in
                    for r in 0 ..< rows {
                        let e = Int(ids[r]), xr = Array(x[r * d ..< (r + 1) * d]), hr = Array(hIn[r * i ..< (r + 1) * i])
                        for c in 0 ..< i {
                            var g = Self.dot(gv, (e * i + c) * d, xr) * Double(gate.scale[e * i + c])
                            var u = Self.dot(uv, (e * i + c) * d, xr) * Double(up.scale[e * i + c])
                            for (j, col) in gate.columns(e) {
                                g += Double(raw[r * d + col]) * Double(gate.wout[(e * i + c) * gate.outliers + j])
                                u += Double(raw[r * d + col]) * Double(up.wout[(e * i + c) * up.outliers + j])
                            }
                            fusedRef.append(Float(g / (1 + Foundation.exp(-g)) * u))
                        }
                        for n in 0 ..< d {
                            var y = Self.dot(dv, (e * d + n) * i, hr) * Double(dn.scale[e * d + n])
                            for (j, col) in dn.columns(e) {
                                y += Double(hRaw[r * i + col]) * Double(dn.wout[(e * d + n) * dn.outliers + j])
                            }
                            downRef.append(Float(y))
                        }
                    }
                }
            }
        }
        Self.assertLive(fusedRef, "prefill fused reference K=\(k)")
        Self.assertLive(downRef, "prefill down reference K=\(k)")
        let g = try gate.mlx(), u = try up.mlx(), down = try dn.mlx()
        let idx = MLXArray(ids)
        let fused = JANGTPrefillKernel.projectGateUpWithOutliers(
            MLXArray(x, [rows, d]).asType(.bfloat16), rawRows: MLXArray(raw, [rows, d]), gate: g, up: u,
            columns: MLXArray([Int32(3), Int32(1900)]), indices: idx, backend: backend).asType(.float32).asArray(Float.self)
        var y = JANGTPrefillKernel.project(MLXArray(hIn, [rows, i]).asType(.bfloat16), down, indices: idx, backend: backend)
            .asType(.float32)
        if let o = JANGTPrefillKernel.outliers(MLXArray(hRaw, [rows, i]), down, indices: idx) { y = y + o }
        let e1 = Self.relL2(fused, fusedRef), e2 = Self.relL2(y.asArray(Float.self), downRef)
        print("JANGT_LOWBIT prefill composed K=\(k): fused gate/up+outliers rel-L2 \(e1), down kin 640 rel-L2 \(e2)")
        return (e1, e2)
    }

    func testPrefillComposedAllWidths() throws {
        try XCTSkipUnless(JANGTPrefillKernel.available, "NAX prefill tiles need an M5-class GPU")
        for k in Self.widths {
            let (a, b) = try runPrefillComposed(k: k)
            // bf16 activations and tiles.
            XCTAssertLessThan(a, 2e-2, "fused gate/up K=\(k)")
            XCTAssertLessThan(b, 2e-2, "down K=\(k)")
        }
    }

    func testSteelPrefillComposedAllWidths() throws {
        for k in Self.widths + [3] {
            let (a, b) = try runPrefillComposed(k: k, backend: .steel)
            XCTAssertLessThan(a, 2e-2, "Steel gate/up K=\(k)")
            XCTAssertLessThan(b, 2e-2, "Steel down K=\(k)")
        }
    }

    func testSteelPrefillTinyAndRaggedSegmentsRejectInvalidRoutesWithoutBankRead() throws {
        var rng = Rng(state: 19)
        let p = Projection(&rng, k: 1.5, experts: 2, outputs: 40, inputs: 640,
                           columns: [[-1], [-1]], columnScales: false)
        let bank = try p.mlx()
        for rows in [1, 7, 8, 15, 16, 17, 31, 32, 33, 72] {
            // Three expert segments cross tile boundaries. The final invalid
            // segment must publish NaNs without indexing its unmapped bank.
            let ids = (0..<rows).map { UInt32($0 == rows - 1 ? 2 : $0 / max(1, rows / 2)) }
            let x = MLXArray.ones([rows, 640], dtype: .float16)
            let y = JANGTPrefillKernel.project(x, bank, indices: MLXArray(ids), backend: .steel)
                .asType(.float32).asArray(Float.self)
            for row in 0..<rows {
                let values = y[(row * 40)..<((row + 1) * 40)]
                if ids[row] >= 2 { XCTAssertTrue(values.allSatisfy(\.isNaN)) }
                else { XCTAssertTrue(values.allSatisfy(\.isFinite)) }
            }
        }
    }

    func testSteelPrefillClampedActivationAndH32AgainstDenseCPU() throws {
        var rng = Rng(state: 918)
        let experts = 3, n = 64, kin = 128
        let gate = Projection(&rng, k: 1.5, experts: experts, outputs: n, inputs: kin,
                              columns: [[-1], [-1], [-1]], columnScales: false)
        let up = Projection(&rng, k: 1.5, experts: experts, outputs: n, inputs: kin,
                            columns: [[-1], [-1], [-1]], columnScales: false)
        let g = try gate.mlx(), u = try up.mlx()
        let limit: Float = 0.3
        for dtype in [DType.float16, .bfloat16] {
            for rows in [1, 7, 17, 33] {
                let ids = (0..<rows).map { UInt32(min(2, $0 * 3 / rows)) }
                let x = MLXArray((0..<(rows * kin)).map { _ in rng.normal() }, [rows, kin]).asType(dtype)
                let inputs = x.asType(.float32).asArray(Float.self)
                var expected = [Float]()
                for row in 0..<rows {
                    let e = Int(ids[row])
                    var values = [Float](repeating: 0, count: n)
                    for col in 0..<n {
                        var gv: Double = 0, uv: Double = 0
                        for k in 0..<kin {
                            let offset = (e * n + col) * kin + k
                            gv += Double(inputs[row * kin + k]) * Double(gate.values[offset]) * Double(gate.scale[e * n + col])
                            uv += Double(inputs[row * kin + k]) * Double(up.values[offset]) * Double(up.scale[e * n + col])
                        }
                        gv = min(gv, Double(limit)); uv = min(max(uv, -Double(limit)), Double(limit))
                        values[col] = Float(gv / (1 + Foundation.exp(-gv)) * uv)
                    }
                    for block in stride(from: 0, to: n, by: 32) {
                        var step = 1
                        while step < 32 {
                            for i in stride(from: block, to: block + 32, by: 2 * step) {
                                for j in i..<(i + step) {
                                    let a = values[j], b = values[j + step]
                                    values[j] = a + b; values[j + step] = a - b
                                }
                            }
                            step *= 2
                        }
                        for col in block..<(block + 32) { values[col] /= sqrt(Float(32)) }
                    }
                    expected += values
                }
                let actual = JANGTPrefillKernel.project(x, g, up: u, indices: MLXArray(ids),
                    limit: limit, rotateOutput: true, backend: .steel).asType(.float32).asArray(Float.self)
                XCTAssertLessThan(Self.relL2(actual, expected), 2e-2, "Steel clamped/H32 rows=\(rows) dtype=\(dtype)")
            }
        }
    }

    // MARK: - format admission

    func testProjectionAdmitsQwenWidthsAndRefusesMalformedGeometry() throws {
        func make(_ k: Double, inputs: Int, outliers: Int = 1, words: Int? = nil) throws -> JANGTProjection {
            let w = words ?? JANGTCode.wordsPerRow(inputs: inputs, k: k)
            return try JANGTProjection(
                k: k, packed: MLXArray.zeros([1, 8, w], dtype: .uint32), scale: MLXArray.ones([1, 8]),
                signs: MLXArray.ones([inputs]),
                outlierColumnIndices: MLXArray.zeros([1, outliers], dtype: .int32) - MLXArray(Int32(1)),
                outlierWeights: MLXArray.zeros([1, 8, outliers], dtype: .float16), columnScales: nil)
        }
        for k in Self.widths {
            XCTAssertEqual(try make(k, inputs: 640).inputs, 640)
            XCTAssertEqual(try make(k, inputs: 2560).inputs, 2560)
        }
        XCTAssertThrowsError(try make(1.5, inputs: 640, words: JANGTCode.wordsPerRow(inputs: 640, k: 2)))
        XCTAssertThrowsError(try make(1.5, inputs: 600))          // not Hadamard-128 blocked
        XCTAssertThrowsError(try make(1.5, inputs: 640, outliers: 33))   // more columns than decode lanes
        XCTAssertNoThrow(try make(1.5, inputs: 640, outliers: 32))
        XCTAssertThrowsError(try make(0.5, inputs: 640, words: 10))
    }
}
