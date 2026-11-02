// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 verify-row kernels (2-8 rows): hc stage + split, hc mix + boundary, the expand, router logits and
// routing. Each row of a batched call is compared, in raw bits, against the same row alone: through the 1-row kernels
// and through the stock graph (the control). Row counts include 3, 5 and 7, and every row has its own values, so a
// kernel that loses the row offset cannot pass. The first suite has no model and no config: synthetic tensors at
// the real V4.1 shapes.
//
// VMLX_DSV41_DEC_BENCH=1 prints interleaved microbenchmarks; VMLX_DSV41_GRAPH_DIR=<dir> writes dispatch censuses.

import Cmlx
import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

private enum VR {
    static let rowCounts = [2, 3, 5, 6, 7, 8]
    static let eps: Float = 1e-6
    static let normEps: Float = 1e-20
    static let iters = 20

    static func hcModules(D: Int, seed: UInt64) -> (DeepseekV4HyperConnection, RMSNorm) {
        let hc = DeepseekV4HyperConnection(
            hcMult: 4, sinkhornIterations: iters, eps: eps, normEps: normEps, hiddenSize: D)
        hc.update(parameters: ModuleParameters.unflattened([
            "fn": MLXRandom.normal([24, 4 * D], key: MLXRandom.key(seed)) * 0.02,
            "scale": MLXRandom.uniform(low: 0.5, high: 2.0, [3], key: MLXRandom.key(seed &+ 1)),
            "base": MLXRandom.normal([24], key: MLXRandom.key(seed &+ 2)) * 0.5,
        ]))
        let norm = RMSNorm(dimensions: D, eps: 1e-20)
        norm.update(parameters: ModuleParameters.unflattened([
            "weight": MLXRandom.uniform(low: 0.5, high: 1.5, [D], key: MLXRandom.key(seed &+ 3)).asType(.bfloat16)
        ]))
        eval(hc, norm)
        return (hc, norm)
    }

    struct Inputs { let h: MLXArray, y: MLXArray, cpre: MLXArray, post: MLXArray, comb: MLXArray }

    static func inputs(rows R: Int, D: Int, seed: UInt64) -> Inputs {
        let h = (MLXRandom.normal([1, R, 4, D], key: MLXRandom.key(seed)) * 4).asType(.bfloat16)
        let y = (MLXRandom.normal([1, R, D], key: MLXRandom.key(seed &+ 1)) * 2).asType(.bfloat16)
        let cpre = MLXRandom.uniform(low: 0.0, high: 1.0, [1, R, 4], key: MLXRandom.key(seed &+ 2))
        let post = MLXRandom.uniform(low: 0.0, high: 2.0, [1, R, 4], key: MLXRandom.key(seed &+ 3))
        let comb = softmax(MLXRandom.normal([1, R, 4, 4], key: MLXRandom.key(seed &+ 4)), axis: -1)
        eval(h, y, cpre, post, comb)
        return Inputs(h: h, y: y, cpre: cpre, post: post, comb: comb)
    }

    static func row(_ a: MLXArray, _ r: Int) -> MLXArray { a[0..., r ..< (r + 1)] }

    /// Raw bits: arrayEqual would call -0 and 0 equal, and NaN unequal to itself.
    static func bitEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
        guard a.shape == b.shape, a.dtype == b.dtype else { return false }
        let t: DType = a.dtype == .float32 || a.dtype == .int32 || a.dtype == .uint32 ? .uint32 : .uint16
        return arrayEqual(a.view(dtype: t), b.view(dtype: t)).item(Bool.self)
    }

    static func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// Fail closed: a non-empty, finite, non-zero baseline whose rows differ from one another.
    static func expectBaseline(
        _ a: MLXArray, rows: Int, _ what: String, sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(a.size > 0, "\(what): empty baseline", sourceLocation: sourceLocation)
        let f = a.asType(.float32)
        let peak = MLX.abs(f).max().item(Float.self)
        #expect(peak.isFinite && peak > 0, "\(what): degenerate baseline (peak \(peak))", sourceLocation: sourceLocation)
        for r in 1 ..< rows {
            #expect(!bitEqual(row(a, 0), row(a, r)), "\(what): rows 0 and \(r) identical", sourceLocation: sourceLocation)
        }
    }

    /// Each row of `got` against `refRow(r)`, output by output, with the baseline checked on the batched result.
    static func expectRowsEqual(
        _ got: DeepseekV41HCStage.Output, rows R: Int, _ what: String,
        ref: (Int) -> DeepseekV41HCStage.Output, sourceLocation: SourceLocation = #_sourceLocation
    ) {
        var pairs: [(String, MLXArray)] = [
            ("normed", got.normed), ("pre", got.pre), ("post", got.post), ("comb", got.comb),
        ]
        if let s = got.stream { pairs.append(("stream", s)) }
        for (name, a) in pairs { expectBaseline(a, rows: R, "\(what) \(name)", sourceLocation: sourceLocation) }
        for r in 0 ..< R {
            let o = ref(r)
            var refs = [o.normed, o.pre, o.post, o.comb]
            if let s = o.stream { refs.append(s) }
            #expect(refs.count == pairs.count, "\(what) row \(r): output count", sourceLocation: sourceLocation)
            for ((name, a), b) in zip(pairs, refs) {
                let ar = row(a, r)
                eval(ar, b)
                #expect(
                    bitEqual(ar, b),
                    "\(what) row \(r) \(name): max |d| \(ar.shape == b.shape ? maxAbsDiff(ar, b) : -1)",
                    sourceLocation: sourceLocation)
            }
        }
    }

    /// Runs `body` with every verify-row flag at the state a default process has on an M5, restoring afterwards.
    static func withRowKernels<R>(tf32: Bool? = true, _ body: () throws -> R) rethrows -> R {
        let saved = (
            DeepseekV41HCStage.enabled, DeepseekV41HCRows.enabled, DeepseekV41HCRows.boundaryEnabled,
            DeepseekV41HCRows.tf32Override, DeepseekV41RouterGEMV.enabled, DeepseekV41RouterGEMV.rowsEnabled,
            DeepseekV41RouteRows.enabled)
        defer {
            DeepseekV41HCStage.enabled = saved.0; DeepseekV41HCRows.enabled = saved.1
            DeepseekV41HCRows.boundaryEnabled = saved.2; DeepseekV41HCRows.tf32Override = saved.3
            DeepseekV41RouterGEMV.enabled = saved.4; DeepseekV41RouterGEMV.rowsEnabled = saved.5
            DeepseekV41RouteRows.enabled = saved.6
            DeepseekV41HCRows.resetVerdicts(); DeepseekV41RouterGEMV.resetVerdicts()
            DeepseekV41RouteRows.resetVerdicts()
        }
        DeepseekV41HCStage.enabled = true; DeepseekV41HCRows.enabled = true; DeepseekV41HCRows.boundaryEnabled = true
        DeepseekV41HCRows.tf32Override = tf32
        DeepseekV41RouterGEMV.enabled = true; DeepseekV41RouterGEMV.rowsEnabled = true
        DeepseekV41RouteRows.enabled = true
        DeepseekV41HCRows.resetVerdicts(); DeepseekV41RouterGEMV.resetVerdicts()
        DeepseekV41RouteRows.resetVerdicts()
        return try body()
    }

    static var benchEnabled: Bool { ProcessInfo.processInfo.environment["VMLX_DSV41_DEC_BENCH"] == "1" }

    /// Interleaved A/B: the bodies alternate rep by rep (first reps discarded); median and min ms per body.
    static func abMedian(reps: Int = 31, warm: Int = 4, _ bodies: [() -> [MLXArray]]) -> [(median: Double, min: Double)] {
        for _ in 0 ..< warm { for b in bodies { eval(b()) } }
        var t = [[Double]](repeating: [], count: bodies.count)
        for _ in 0 ..< reps {
            for (k, b) in bodies.enumerated() {
                let t0 = CFAbsoluteTimeGetCurrent()
                eval(b())
                t[k].append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
        }
        return t.map { v in
            let s = v.sorted()
            return (s[s.count / 2], s[0])
        }
    }

    /// Lazy graph of `arrays` as `mlx_print_graph` writes it (one line per unevaluated primitive output).
    static func writeGraph(_ path: String, _ arrays: [MLXArray]) {
        guard let f = fopen(path, "w") else { return }
        let v = mlx_vector_array_new()
        for a in arrays { mlx_vector_array_append_value(v, a.ctx) }
        let namer = mlx_node_namer_new()
        mlx_print_graph(f, namer, v)
        mlx_node_namer_free(namer)
        mlx_vector_array_free(v)
        fclose(f)
    }
}

@Suite("DeepSeek-V4.1 verify-row kernels", .serialized)
struct DeepseekV41VerifyRowKernelsTests {

    // MARK: - hc stage + split

    @Test("hc stage + split: each row is bit-equal to its 1-row kernels and to the stock graph on that row alone")
    func hcStageRowsBitExact() {
        Device.withDefaultDevice(.gpu) {
            for D in [5120, 6144] {
                for R in VR.rowCounts {
                    for seed in UInt64(1) ... 2 {
                        let (hc, norm) = VR.hcModules(D: D, seed: seed &* 13 &+ UInt64(D))
                        let i = VR.inputs(rows: R, D: D, seed: seed &* 7 &+ UInt64(R))
                        #expect(DeepseekV41HCRows.admits(i.h, hc: hc, norm: norm), "D \(D) rows \(R)")
                        for expand in [false, true] {
                            let a: (MLXArray?, MLXArray?, MLXArray?) = expand ? (i.y, i.post, i.comb) : (nil, nil, nil)
                            func at(_ v: MLXArray?, _ r: Int) -> MLXArray? { v.map { VR.row($0, r) } }
                            let got = DeepseekV41HCRows.runStage(
                                h: i.h, y: a.0, post: a.1, comb: a.2, cpre: i.cpre, hc: hc, norm: norm)
                            let tag = "stage D \(D) rows \(R) seed \(seed) expand \(expand)"
                            VR.expectRowsEqual(got, rows: R, tag + " vs 1-row kernels") { r in
                                DeepseekV41HCStage.run(
                                    h: VR.row(i.h, r), y: at(a.0, r), post: at(a.1, r), comb: at(a.2, r),
                                    cpre: VR.row(i.cpre, r), hc: hc, norm: norm)
                            }
                            VR.expectRowsEqual(got, rows: R, tag + " vs stock") { r in
                                DeepseekV41HCStage.stock(
                                    h: VR.row(i.h, r), y: at(a.0, r), post: at(a.1, r), comb: at(a.2, r),
                                    cpre: VR.row(i.cpre, r), hc: hc, norm: norm)
                            }
                        }
                    }
                }
            }
        }
    }

    // MARK: - hc mix + boundary

    @Test("hc mix + boundary: each row equals itself run alone; normed and stream bit-equal the stage; split within rounding")
    func hcBoundaryRows() {
        Device.withDefaultDevice(.gpu) {
            var worst: Float = 0
            for D in [5120, 6144] {
                for R in VR.rowCounts {
                    let (hc, norm) = VR.hcModules(D: D, seed: 91 &+ UInt64(D))
                    let i = VR.inputs(rows: R, D: D, seed: 300 &+ UInt64(R))
                    for expand in [false, true] {
                        let a: (MLXArray?, MLXArray?, MLXArray?) = expand ? (i.y, i.post, i.comb) : (nil, nil, nil)
                        func at(_ v: MLXArray?, _ r: Int) -> MLXArray? { v.map { VR.row($0, r) } }
                        let got = DeepseekV41HCRows.runBoundary(
                            h: i.h, y: a.0, post: a.1, comb: a.2, cpre: i.cpre, hc: hc, norm: norm)
                        let tag = "boundary D \(D) rows \(R) expand \(expand)"
                        VR.expectRowsEqual(got, rows: R, tag + " vs its own 1-row") { r in
                            DeepseekV41HCRows.runBoundary(
                                h: VR.row(i.h, r), y: at(a.0, r), post: at(a.1, r), comb: at(a.2, r),
                                cpre: VR.row(i.cpre, r), hc: hc, norm: norm)
                        }
                        let stage = DeepseekV41HCRows.runStage(
                            h: i.h, y: a.0, post: a.1, comb: a.2, cpre: i.cpre, hc: hc, norm: norm)
                        eval(stage.normed, stage.pre, stage.post, stage.comb)
                        #expect(VR.bitEqual(got.normed, stage.normed), "\(tag) normed")
                        if expand {
                            #expect(VR.bitEqual(got.stream!, stage.stream!), "\(tag) stream")
                        }
                        #expect(DeepseekV41HCRows.withinRounding(got, stage), "\(tag) within the self-check")
                        for (x, y) in [(got.pre, stage.pre), (got.post, stage.post), (got.comb, stage.comb)] {
                            worst = max(worst, VR.maxAbsDiff(x, y))
                        }
                    }
                }
            }
            print("[dsv41-rows] boundary vs stage split: worst max |d| \(worst)")
            // Not vacuous: the partial sums really do run in another order.
            #expect(worst > 0, "boundary split bit-identical to the stage everywhere: its check proved nothing")
        }
    }

    // MARK: - expand

    @Test("the rows expand: each row is bit-equal to the 1-row expand; the batched stock GEMM is the control")
    func hcExpandRows() {
        Device.withDefaultDevice(.gpu) {
            var gemmDiffers = 0
            for R in VR.rowCounts {
                let i = VR.inputs(rows: R, D: 5120, seed: 500 &+ UInt64(R))
                let got = DeepseekV4Math.hcPostRows(blockOut: i.y, residual: i.h, post: i.post, comb: i.comb)
                let gemm = DeepseekV4Math.hcPost(blockOut: i.y, residual: i.h, post: i.post, comb: i.comb)
                eval(got, gemm)
                VR.expectBaseline(got, rows: R, "expand rows \(R)")
                for r in 0 ..< R {
                    let one = DeepseekV4Math.hcPost(
                        blockOut: VR.row(i.y, r), residual: VR.row(i.h, r), post: VR.row(i.post, r),
                        comb: VR.row(i.comb, r))
                    eval(one)
                    #expect(VR.bitEqual(VR.row(got, r), one), "expand rows \(R) row \(r)")
                }
                if !VR.bitEqual(got, gemm) { gemmDiffers += 1 }
            }
            print("[dsv41-rows] batched stock expand differs from the row kernel at \(gemmDiffers)/\(VR.rowCounts.count) row counts")
        }
    }

    // MARK: - chain

    @Test("five hc layers chained: every row through the rows kernels is bit-equal to that row through the 1-row kernels")
    func hcChain() {
        Device.withDefaultDevice(.gpu) {
            let D = 5120
            let mods = (0 ..< 5).map { k in
                (VR.hcModules(D: D, seed: 700 &+ UInt64(k)), VR.hcModules(D: D, seed: 800 &+ UInt64(k)))
            }
            // Stand-ins for attention and MoE: deterministic per row, so only the hc path can differ.
            func branch(_ normed: MLXArray) -> MLXArray { (normed.asType(.float32) * 0.5).asType(.bfloat16) }
            typealias Run = (MLXArray, MLXArray?, MLXArray?, MLXArray?, MLXArray, DeepseekV4HyperConnection, RMSNorm)
                -> DeepseekV41HCStage.Output
            func chain(_ h0: MLXArray, _ pre0: MLXArray, _ run: Run) -> (MLXArray, MLXArray) {
                var h = h0, pre = pre0
                for ((ah, an), (fh, fn)) in mods {
                    let a = run(h, nil, nil, nil, pre, ah, an)
                    let f = run(h, branch(a.normed), a.post, a.comb, a.pre, fh, fn)
                    h = DeepseekV4Math.hcPostRows(blockOut: branch(f.normed), residual: f.stream!, post: f.post, comb: f.comb)
                    pre = f.pre
                }
                return (h, pre)
            }
            let stage: Run = { h, y, p, c, cpre, hc, n in
                h.dim(1) > 1
                    ? DeepseekV41HCRows.runStage(h: h, y: y, post: p, comb: c, cpre: cpre, hc: hc, norm: n)
                    : DeepseekV41HCStage.run(h: h, y: y, post: p, comb: c, cpre: cpre, hc: hc, norm: n)
            }
            let boundary: Run = { h, y, p, c, cpre, hc, n in
                DeepseekV41HCRows.runBoundary(h: h, y: y, post: p, comb: c, cpre: cpre, hc: hc, norm: n)
            }
            for R in VR.rowCounts {
                let i = VR.inputs(rows: R, D: D, seed: 900 &+ UInt64(R))
                for (name, run) in [("stage", stage), ("boundary", boundary)] {
                    let all = chain(i.h, i.cpre, run)
                    eval(all.0, all.1)
                    VR.expectBaseline(all.0, rows: R, "chain \(name) rows \(R)")
                    for r in 0 ..< R {
                        let one = chain(VR.row(i.h, r), VR.row(i.cpre, r), run)
                        eval(one.0, one.1)
                        #expect(VR.bitEqual(VR.row(all.0, r), one.0), "chain \(name) rows \(R) row \(r) h")
                        #expect(VR.bitEqual(VR.row(all.1, r), one.1), "chain \(name) rows \(R) row \(r) pre")
                    }
                }
            }
        }
    }

    // MARK: - hc admission and fallback

    @Test("hc apply admits against the live comparison and actually serves from the fused path")
    func hcApplyRuns() throws {
        try Device.withDefaultDevice(.gpu) {
            try VR.withRowKernels {
                let D = 5120
                let (hc, norm) = VR.hcModules(D: D, seed: 11)
                for R in VR.rowCounts {
                    let i = VR.inputs(rows: R, D: D, seed: 1000 &+ UInt64(R))
                    for expand in [false, true] {
                        let a: (MLXArray?, MLXArray?, MLXArray?) = expand ? (i.y, i.post, i.comb) : (nil, nil, nil)
                        let stageKey = DeepseekV41HCRows.Key(path: .stage, rows: R, hc: 4, hidden: D, expand: expand)
                        let boundaryKey = DeepseekV41HCRows.Key(path: .boundary, rows: R, hc: 4, hidden: D, expand: expand)
                        #expect(DeepseekV41HCRows.verdict(stageKey) == nil)
                        // The admitting call, then one served without a check.
                        let before = DeepseekV41HCRows.runCount(.boundary)
                        let first = try #require(DeepseekV41HCRows.apply(
                            h: i.h, y: a.0, post: a.1, comb: a.2, cpre: i.cpre, hc: hc, norm: norm))
                        let second = try #require(DeepseekV41HCRows.apply(
                            h: i.h, y: a.0, post: a.1, comb: a.2, cpre: i.cpre, hc: hc, norm: norm))
                        #expect(DeepseekV41HCRows.verdict(stageKey) == true, "stage rows \(R) expand \(expand)")
                        #expect(DeepseekV41HCRows.verdict(boundaryKey) == true, "boundary rows \(R) expand \(expand)")
                        #expect(DeepseekV41HCRows.runCount(.boundary) == before + 2, "served from the boundary path")
                        let direct = DeepseekV41HCRows.runBoundary(
                            h: i.h, y: a.0, post: a.1, comb: a.2, cpre: i.cpre, hc: hc, norm: norm)
                        for (x, y) in [(first.comb, direct.comb), (second.comb, direct.comb), (second.normed, direct.normed)] {
                            eval(x, y)
                            #expect(VR.bitEqual(x, y))
                        }
                    }
                }
            }
        }
    }

    @Test("hc apply falls back: kill switches, TF32 off, a recorded mismatch, a boundary mismatch keeps the stage")
    func hcApplyFallback() throws {
        try Device.withDefaultDevice(.gpu) {
            let D = 5120, R = 6
            let (hc, norm) = VR.hcModules(D: D, seed: 12)
            let i = VR.inputs(rows: R, D: D, seed: 1100)
            func call() -> DeepseekV41HCStage.Output? {
                DeepseekV41HCRows.apply(h: i.h, cpre: i.cpre, hc: hc, norm: norm)
            }
            let stageKey = DeepseekV41HCRows.Key(path: .stage, rows: R, hc: 4, hidden: D, expand: false)
            let boundaryKey = DeepseekV41HCRows.Key(path: .boundary, rows: R, hc: 4, hidden: D, expand: false)
            try VR.withRowKernels(tf32: false) {
                #expect(call() == nil, "TF32 not trusted: stock")
                #expect(DeepseekV41HCRows.verdict(stageKey) == nil, "no admission run without TF32")
            }
            try VR.withRowKernels {
                DeepseekV41HCStage.enabled = false
                #expect(call() == nil, "VMLX_DSV41_HC_STAGE=0 turns rows off too")
                DeepseekV41HCStage.enabled = true
                DeepseekV41HCRows.enabled = false
                #expect(call() == nil)
                DeepseekV41HCRows.enabled = true
                DeepseekV41HCRows.record(stageKey, false)
                #expect(call() == nil, "stage mismatch keeps the stock graph for this shape")
                DeepseekV41HCRows.record(stageKey, true)
                DeepseekV41HCRows.record(boundaryKey, false)
                var before = DeepseekV41HCRows.runCount(.stage)
                _ = try #require(call(), "boundary mismatch keeps the stage")
                #expect(DeepseekV41HCRows.runCount(.stage) == before + 1)
                DeepseekV41HCRows.resetVerdicts()
                DeepseekV41HCRows.boundaryEnabled = false
                before = DeepseekV41HCRows.runCount(.stage)
                let viaStage = try #require(call(), "VMLX_DSV41_HC_BOUNDARY=0 keeps the stage")
                #expect(DeepseekV41HCRows.runCount(.stage) == before + 1)
                #expect(DeepseekV41HCRows.verdict(boundaryKey) == nil)
                eval(viaStage.comb)
                // Declined shapes.
                let h9 = MLXArray.zeros([1, 9, 4, D], dtype: .bfloat16)
                #expect(!DeepseekV41HCRows.admits(h9, hc: hc, norm: norm), "9 rows are not a verify row")
                #expect(!DeepseekV41HCRows.admits(MLXArray.zeros([1, 1, 4, D], dtype: .bfloat16), hc: hc, norm: norm))
                #expect(!DeepseekV41HCRows.admits(i.h.asType(.float32), hc: hc, norm: norm))
            }
        }
        Device.withDefaultDevice(.cpu) {
            let (hc, norm) = VR.hcModules(D: 5120, seed: 12)
            #expect(!DeepseekV41HCRows.admits(MLXArray.zeros([1, 6, 4, 5120], dtype: .bfloat16), hc: hc, norm: norm))
        }
    }

    @Test("the admission comparisons can fail: one flipped ulp fails bitEqual, a real difference fails withinRounding")
    func comparatorsDetect() {
        Device.withDefaultDevice(.gpu) {
            let (hc, norm) = VR.hcModules(D: 5120, seed: 13)
            let i = VR.inputs(rows: 3, D: 5120, seed: 1200)
            let o = DeepseekV41HCRows.runStage(h: i.h, y: i.y, post: i.post, comb: i.comb, cpre: i.cpre, hc: hc, norm: norm)
            eval(o.normed, o.pre, o.post, o.comb, o.stream!)
            #expect(DeepseekV41HCRows.bitEqual(o, o))
            #expect(DeepseekV41HCRows.withinRounding(o, o))
            let ulp = (o.pre.view(dtype: .uint32) + MLXArray(UInt32(1))).view(dtype: .float32)
            let off = DeepseekV41HCStage.Output(normed: o.normed, pre: ulp, post: o.post, comb: o.comb, stream: o.stream)
            eval(off.pre)
            #expect(!DeepseekV41HCRows.bitEqual(off, o), "one ulp in pre")
            #expect(DeepseekV41HCRows.withinRounding(off, o), "one ulp in pre is rounding")
            let wrong = DeepseekV41HCStage.Output(normed: o.normed, pre: o.pre * 1.01, post: o.post, comb: o.comb, stream: o.stream)
            #expect(!DeepseekV41HCRows.withinRounding(wrong, o), "1% in pre is not rounding")
            let normedOff = DeepseekV41HCStage.Output(
                normed: (o.normed.view(dtype: .uint16) + MLXArray(UInt16(1))).view(dtype: .bfloat16),
                pre: o.pre, post: o.post, comb: o.comb, stream: o.stream)
            #expect(!DeepseekV41HCRows.withinRounding(normedOff, o), "normed must stay bitwise")
        }
    }

    @Test("a verify-row kernel that fails to build reports false instead of taking the process down")
    func launchFailureIsCaught() {
        Device.withDefaultDevice(.gpu) {
            let bad = MLXFast.metalKernel(
                name: "vmlx_dsv41_test_does_not_compile", inputNames: ["x"], outputNames: ["y"],
                source: "this is not metal;")
            let x = MLXArray.ones([8], dtype: .float32)
            var compared = false
            let ok = DeepseekV41RowKernels.check([]) {
                compared = true
                return true
            }
            #expect(ok && compared, "control: a passing check passes")
            let failed = DeepseekV41RowKernels.check(
                bad([x], grid: (8, 1, 1), threadGroup: (8, 1, 1), outputShapes: [[8]], outputDTypes: [.float32])
            ) { true }
            #expect(!failed)
        }
    }

    @Test("TF32 probe follows MLX_ENABLE_TF32")
    func tf32Probe() {
        Device.withDefaultDevice(.gpu) {
            let raw = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"]
            print("[dsv41-rows] MLX_ENABLE_TF32=\(raw ?? "unset") gemmRoundsToTF32=\(DeepseekV41HCRows.gemmRoundsToTF32)")
            if raw == "0" { #expect(!DeepseekV41HCRows.gemmRoundsToTF32) }
        }
    }

    // MARK: - router logits

    @Test("router rows: each row is bit-equal to its 1-row kernel and to the stock matmul on that row alone")
    func routerRowsBitExact() throws {
        try Device.withDefaultDevice(.gpu) {
            for (E, D) in [(384, 5120), (288, 4096), (192, 2048)] {
                for R in VR.rowCounts {
                    for dtype in [DType.bfloat16, .float32] {
                        let w = (MLXRandom.normal([E, D], key: MLXRandom.key(UInt64(E + R))) * 0.03).asType(.bfloat16)
                        let x = (MLXRandom.normal([1, R, D], key: MLXRandom.key(UInt64(D + R))) * 3).asType(dtype)
                        let got = DeepseekV41RouterGEMV.runRows(x, weight: w)
                        let stock = DeepseekV41RouterGEMV.stockRows(x, weight: w)
                        eval(got, stock)
                        #expect(got.shape == [1, R, E] && got.dtype == .float32)
                        VR.expectBaseline(got, rows: R, "router E \(E) rows \(R)")
                        #expect(VR.bitEqual(got, stock), "E \(E) D \(D) rows \(R) \(dtype) vs stock: \(VR.maxAbsDiff(got, stock))")
                        for r in 0 ..< R {
                            let one = try #require(DeepseekV41RouterGEMV.logits(VR.row(x, r), weight: w))
                            eval(one)
                            #expect(VR.bitEqual(VR.row(got, r), one), "E \(E) D \(D) rows \(R) row \(r) \(dtype)")
                        }
                    }
                }
            }
        }
    }

    @Test("router rows admit against the live comparison, serve from the kernel, and honor the kill switch")
    func routerRowsApply() throws {
        try Device.withDefaultDevice(.gpu) {
            try VR.withRowKernels {
                let E = 384, D = 5120
                let w = (MLXRandom.normal([E, D], key: MLXRandom.key(5)) * 0.03).asType(.bfloat16)
                eval(w)
                for R in VR.rowCounts {
                    let x = (MLXRandom.normal([1, R, D], key: MLXRandom.key(UInt64(40 + R))) * 3).asType(.bfloat16)
                    let key = DeepseekV41RouterGEMV.Key(rows: R, experts: E, hidden: D, dtype: .bfloat16)
                    #expect(DeepseekV41RouterGEMV.verdict(key) == nil)
                    let before = DeepseekV41RouterGEMV.rowRuns
                    let a = try #require(DeepseekV41RouterGEMV.rowLogits(x, weight: w))
                    let b = try #require(DeepseekV41RouterGEMV.rowLogits(x, weight: w))
                    eval(a, b)
                    #expect(DeepseekV41RouterGEMV.verdict(key) == true)
                    #expect(DeepseekV41RouterGEMV.rowRuns == before + 2)
                    #expect(VR.bitEqual(a, b))
                }
                let x = (MLXRandom.normal([1, 6, D], key: MLXRandom.key(9)) * 3).asType(.bfloat16)
                #expect(DeepseekV41RouterGEMV.rowLogits(x[0..., 0 ..< 1], weight: w) == nil, "one row is the 1-row kernel's")
                #expect(DeepseekV41RouterGEMV.rowLogits(
                    MLXArray.zeros([1, 9, D], dtype: .bfloat16), weight: w) == nil, "9 rows are not a verify row")
                DeepseekV41RouterGEMV.rowsEnabled = false
                #expect(DeepseekV41RouterGEMV.rowLogits(x, weight: w) == nil)
                DeepseekV41RouterGEMV.rowsEnabled = true
                DeepseekV41RouterGEMV.enabled = false
                #expect(DeepseekV41RouterGEMV.rowLogits(x, weight: w) == nil, "VMLX_DSV41_ROUTER_KERNEL=0 turns rows off too")
                DeepseekV41RouterGEMV.enabled = true
                DeepseekV41RouterGEMV.record(
                    .init(rows: 6, experts: E, hidden: D, dtype: .bfloat16), false)
                #expect(DeepseekV41RouterGEMV.rowLogits(x, weight: w) == nil, "a mismatch keeps the stock matmul")
            }
        }
    }

    // MARK: - routing

    /// The selector's own statements on one row (`DeepseekV4CompiledSelectorCache` is private to the model file).
    static func stockRoute(_ logits: MLXArray, bias: MLXArray, topK: Int, normalize: Bool, scale: MLXArray) -> [MLXArray] {
        let scores = sqrt(log1p(exp(logits)))
        let idx = argPartition(-(scores + bias), kth: topK - 1, axis: -1)[.ellipsis, 0 ..< topK].asType(.int32)
        var w = takeAlong(scores, idx, axis: -1)
        if topK > 1 && normalize { w = w / w.sum(axis: -1, keepDims: true) }
        return [idx, w * scale]
    }

    @Test("route rows: each row is bit-equal to the selector run on that row: experts, their order, and weights")
    func routeRowsBitExact() {
        Device.withDefaultDevice(.gpu) {
            let scale = MLXArray([Float(1.5)])
            for (E, K, norm) in [(384, 6, true), (384, 6, false), (128, 3, true), (64, 1, true), (32, 16, true)] {
                for R in VR.rowCounts {
                    let logits = MLXRandom.normal([1, R, E], key: MLXRandom.key(UInt64(E + R))) * 3
                    let bias = MLXRandom.normal([E], key: MLXRandom.key(UInt64(40 + R))) * 0.2
                    let got = DeepseekV41RouteRows.run(logits, bias: bias, topK: K, normalize: norm, scale: scale)
                    eval(got.indices, got.weights)
                    #expect(got.indices.shape == [1, R, K] && got.weights.shape == [1, R, K])
                    VR.expectBaseline(got.indices, rows: R, "route E \(E) K \(K) rows \(R) idx")
                    VR.expectBaseline(got.weights, rows: R, "route E \(E) K \(K) rows \(R) w")
                    for r in 0 ..< R {
                        let one = Self.stockRoute(VR.row(logits, r), bias: bias, topK: K, normalize: norm, scale: scale)
                        eval(one)
                        #expect(VR.bitEqual(VR.row(got.indices, r), one[0]), "E \(E) K \(K) norm \(norm) rows \(R) row \(r) idx")
                        #expect(
                            VR.bitEqual(VR.row(got.weights, r), one[1]),
                            "E \(E) K \(K) norm \(norm) rows \(R) row \(r) w: \(VR.maxAbsDiff(VR.row(got.weights, r), one[1]))")
                    }
                }
            }
        }
    }

    @Test("route rows match the selector on the hard cases: every key negative, ties, large logits")
    func routeRowsHardCases() {
        Device.withDefaultDevice(.gpu) {
            let scale = MLXArray([Float(1.5)])
            let E = 384, K = 6, R = 5
            let logits = MLXRandom.normal([1, R, E], key: MLXRandom.key(71)) * 3
            // Every biased score negative: picks must not be invented by lanes past the simdgroup's end.
            let negative = MLXRandom.normal([E], key: MLXRandom.key(72)) * 0.2 - 100
            // Ties: runs of equal logits with no bias, so the stable sort's id order decides.
            let tied = MLX.floor(MLXRandom.uniform(low: 0.0, high: 4.0, [1, R, E], key: MLXRandom.key(73)))
            // Scores far from zero, up to the top of exp's fp32 range.
            let large = MLXRandom.uniform(low: -80.0, high: 80.0, [1, R, E], key: MLXRandom.key(74))
            let zeroBias = MLXArray.zeros([E], dtype: .float32)
            for (name, l, b) in [("negative", logits, negative), ("ties", tied, zeroBias), ("large", large, zeroBias)] {
                let got = DeepseekV41RouteRows.run(l, bias: b, topK: K, normalize: true, scale: scale)
                eval(got.indices, got.weights)
                for r in 0 ..< R {
                    let one = Self.stockRoute(VR.row(l, r), bias: b, topK: K, normalize: true, scale: scale)
                    eval(one)
                    #expect(VR.bitEqual(VR.row(got.indices, r), one[0]), "\(name) row \(r) idx")
                    #expect(VR.bitEqual(VR.row(got.weights, r), one[1]), "\(name) row \(r) w")
                }
            }
        }
    }

    @Test("route rows admit against the live comparison, serve from the kernel, and honor the kill switch")
    func routeRowsApply() throws {
        try Device.withDefaultDevice(.gpu) {
            try VR.withRowKernels {
                let scale = MLXArray([Float(1.5)])
                let E = 384, K = 6
                let bias = MLXRandom.normal([E], key: MLXRandom.key(81)) * 0.2
                func stock(_ l: MLXArray) -> [MLXArray] {
                    Self.stockRoute(l, bias: bias, topK: K, normalize: true, scale: scale)
                }
                for R in VR.rowCounts {
                    let l = MLXRandom.normal([1, R, E], key: MLXRandom.key(UInt64(90 + R))) * 3
                    let key = DeepseekV41RouteRows.Key(rows: R, experts: E, topK: K, normalize: true)
                    #expect(DeepseekV41RouteRows.verdict(key) == nil)
                    let before = DeepseekV41RouteRows.runCount
                    let a = try #require(DeepseekV41RouteRows.apply(l, bias: bias, topK: K, normalize: true, scale: scale, stock: stock))
                    let b = try #require(DeepseekV41RouteRows.apply(l, bias: bias, topK: K, normalize: true, scale: scale, stock: stock))
                    eval(a.indices, a.weights, b.indices, b.weights)
                    #expect(DeepseekV41RouteRows.verdict(key) == true, "rows \(R)")
                    #expect(DeepseekV41RouteRows.runCount == before + 2)
                    #expect(a.indices.dtype == .uint32 && VR.bitEqual(a.indices, b.indices) && VR.bitEqual(a.weights, b.weights))
                }
                let l = MLXRandom.normal([1, 6, E], key: MLXRandom.key(99)) * 3
                #expect(DeepseekV41RouteRows.apply(
                    l[0..., 0 ..< 1], bias: bias, topK: K, normalize: true, scale: scale, stock: stock) == nil, "one row keeps the selector")
                DeepseekV41RouteRows.enabled = false
                #expect(DeepseekV41RouteRows.apply(l, bias: bias, topK: K, normalize: true, scale: scale, stock: stock) == nil)
                DeepseekV41RouteRows.enabled = true
                // A live mismatch: the reference disagrees, so the shape stays on the selector.
                DeepseekV41RouteRows.resetVerdicts()
                let wrong = DeepseekV41RouteRows.apply(l, bias: bias, topK: K, normalize: true, scale: scale) {
                    let s = stock($0)
                    return [s[0], s[1] * 1.01]
                }
                #expect(wrong == nil)
                #expect(DeepseekV41RouteRows.verdict(.init(rows: 6, experts: E, topK: K, normalize: true)) == false)
                #expect(DeepseekV41RouteRows.apply(l, bias: bias, topK: K, normalize: true, scale: scale, stock: stock) == nil)
            }
        }
    }

    // MARK: - census and benchmarks (opt in)

    /// Per-sublayer (hc) and per-layer (router, routing) graphs at `rows`, stock vs kernels.
    @Test("hc / router / routing dispatch census (VMLX_DSV41_GRAPH_DIR)")
    func census() {
        guard let dir = ProcessInfo.processInfo.environment["VMLX_DSV41_GRAPH_DIR"] else { return }
        Device.withDefaultDevice(.gpu) {
            let D = 5120, E = 384, K = 6
            let (hc, norm) = VR.hcModules(D: D, seed: 1)
            let w = (MLXRandom.normal([E, D], key: MLXRandom.key(5)) * 0.03).asType(.bfloat16)
            let bias = MLXRandom.normal([E], key: MLXRandom.key(6)) * 0.2
            let scale = MLXArray([Float(1.5)])
            eval(w, bias, scale)
            for R in [1, 6] {
                let i = VR.inputs(rows: R, D: D, seed: 2)
                let stockFFN = DeepseekV41HCStage.stock(h: i.h, y: i.y, post: i.post, comb: i.comb, cpre: i.cpre, hc: hc, norm: norm)
                let arrays = { (o: DeepseekV41HCStage.Output) in [o.normed, o.pre, o.post, o.comb, o.stream!] }
                VR.writeGraph("\(dir)/hc_ffn_stock_r\(R).txt", arrays(stockFFN))
                if R > 1 {
                    VR.writeGraph("\(dir)/hc_ffn_stage_r\(R).txt", arrays(DeepseekV41HCRows.runStage(
                        h: i.h, y: i.y, post: i.post, comb: i.comb, cpre: i.cpre, hc: hc, norm: norm)))
                    VR.writeGraph("\(dir)/hc_ffn_boundary_r\(R).txt", arrays(DeepseekV41HCRows.runBoundary(
                        h: i.h, y: i.y, post: i.post, comb: i.comb, cpre: i.cpre, hc: hc, norm: norm)))
                }
                let x = (MLXRandom.normal([1, R, D], key: MLXRandom.key(7)) * 3).asType(.bfloat16)
                eval(x)
                let stockLogits = x.asType(.float32).matmul(w.asType(.float32).transposed())
                VR.writeGraph("\(dir)/router_stock_r\(R).txt", [stockLogits])
                eval(stockLogits)
                let route = Self.stockRoute(stockLogits, bias: bias, topK: K, normalize: true, scale: scale)
                VR.writeGraph("\(dir)/route_stock_r\(R).txt", route)
                if R > 1 {
                    VR.writeGraph("\(dir)/router_rows_r\(R).txt", [DeepseekV41RouterGEMV.runRows(x, weight: w)])
                    let r = DeepseekV41RouteRows.run(stockLogits, bias: bias, topK: K, normalize: true, scale: scale)
                    VR.writeGraph("\(dir)/route_rows_r\(R).txt", [r.indices, r.weights])
                }
            }
        }
    }

    @Test("verify-row kernels microbenchmark, interleaved (VMLX_DSV41_DEC_BENCH=1)")
    func bench() {
        guard VR.benchEnabled else { return }
        Device.withDefaultDevice(.gpu) {
            let D = 5120, E = 384, K = 6, layers = 40
            let mods = (0 ..< 8).map { VR.hcModules(D: D, seed: 2000 &+ UInt64($0)) }
            let w = (0 ..< 8).map { (MLXRandom.normal([E, D], key: MLXRandom.key(UInt64($0))) * 0.03).asType(.bfloat16) }
            let bias = MLXRandom.normal([E], key: MLXRandom.key(6)) * 0.2
            let scale = MLXArray([Float(1.5)])
            eval(w + [bias, scale])
            for R in [2, 4, 6, 8] {
                let ins = (0 ..< layers).map { VR.inputs(rows: R, D: D, seed: 3000 &+ UInt64($0 * 11 + R)) }
                // One whole layer's hc: the attention boundary, the FFN boundary with the branch, the final expand.
                func layerHC(_ k: Int, _ run: (MLXArray, MLXArray?, MLXArray?, MLXArray?, MLXArray, DeepseekV4HyperConnection, RMSNorm) -> DeepseekV41HCStage.Output, expand: (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray) -> [MLXArray] {
                    let i = ins[k]
                    let (hc, norm) = mods[k % mods.count]
                    let a = run(i.h, nil, nil, nil, i.cpre, hc, norm)
                    let f = run(i.h, i.y, a.post, a.comb, a.pre, hc, norm)
                    return [a.normed, f.normed, f.pre, expand(i.y, f.stream!, f.post, f.comb)]
                }
                let rowsExpand: (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray = {
                    DeepseekV4Math.hcPostRows(blockOut: $0, residual: $1, post: $2, comb: $3)
                }
                let stockExpand: (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray = {
                    DeepseekV4Math.hcPost(blockOut: $0, residual: $1, post: $2, comb: $3)
                }
                let t = VR.abMedian([
                    { (0 ..< layers).flatMap { layerHC($0, { h, y, p, c, cp, hc, n in
                        DeepseekV41HCStage.stock(h: h, y: y, post: p, comb: c, cpre: cp, hc: hc, norm: n) }, expand: stockExpand) } },
                    { (0 ..< layers).flatMap { layerHC($0, { h, y, p, c, cp, hc, n in
                        DeepseekV41HCRows.runStage(h: h, y: y, post: p, comb: c, cpre: cp, hc: hc, norm: n) }, expand: rowsExpand) } },
                    { (0 ..< layers).flatMap { layerHC($0, { h, y, p, c, cp, hc, n in
                        DeepseekV41HCRows.runBoundary(h: h, y: y, post: p, comb: c, cpre: cp, hc: hc, norm: n) }, expand: rowsExpand) } },
                ])
                print(String(format: "[dsv41-rows-bench] hc per token (40 layers) rows=%d: stock %.2f (min %.2f), stage %.2f (min %.2f), boundary %.2f (min %.2f) ms",
                    R, t[0].median, t[0].min, t[1].median, t[1].min, t[2].median, t[2].min))
                let xs = (0 ..< layers).map { k -> MLXArray in
                    let x = (MLXRandom.normal([1, R, D], key: MLXRandom.key(UInt64(4000 + k))) * 3).asType(.bfloat16)
                    eval(x)
                    return x
                }
                let logits = xs.indices.map { xs[$0].asType(.float32).matmul(w[$0 % w.count].asType(.float32).transposed()) }
                eval(logits)
                let r = VR.abMedian([
                    { xs.indices.map { xs[$0].asType(.float32).matmul(w[$0 % w.count].asType(.float32).transposed()) } },
                    { xs.indices.map { DeepseekV41RouterGEMV.runRows(xs[$0], weight: w[$0 % w.count]) } },
                    { logits.flatMap { Self.stockRoute($0, bias: bias, topK: K, normalize: true, scale: scale) } },
                    { logits.flatMap { l -> [MLXArray] in
                        let o = DeepseekV41RouteRows.run(l, bias: bias, topK: K, normalize: true, scale: scale)
                        return [o.indices, o.weights]
                    } },
                ])
                print(String(format: "[dsv41-rows-bench] router per token rows=%d: stock %.2f (min %.2f), rows %.2f (min %.2f) ms; routing: stock %.2f (min %.2f), rows %.2f (min %.2f) ms",
                    R, r[0].median, r[0].min, r[1].median, r[1].min, r[2].median, r[2].min, r[3].median, r[3].min))
            }
        }
    }
}

// MARK: - Through the model's layer path (needs a V4.1 config; about 5 GB of synthetic banks: run under the heavy lock)

@Suite(
    "DeepSeek-V4.1 verify-row kernels through the decoder layers", .serialized,
    .enabled(if: DSV41Synth.config != nil, "needs VMLX_DSV41_CONFIG or VMLX_DSV41_BUNDLE"))
struct DeepseekV41VerifyRowLayerTests {
    private func setKernels(_ on: Bool) {
        DeepseekV41HCRows.enabled = on; DeepseekV41RouterGEMV.rowsEnabled = on; DeepseekV41RouteRows.enabled = on
    }

    private func rowCosines(_ a: MLXArray, _ b: MLXArray, rows: Int) -> [Float] {
        let af = a.asType(.float32).reshaped(rows, -1), bf = b.asType(.float32).reshaped(rows, -1)
        let cos = (af * bf).sum(axis: -1) / (sqrt((af * af).sum(axis: -1)) * sqrt((bf * bf).sum(axis: -1)))
        return cos.asArray(Float.self)
    }

    /// Random weights blow most rows up to norm ~2e6 and leave a few near 1e2; on those few, rounding decides the
    /// whole direction. The control shows the same rows moving: the stock graph against itself with one element of
    /// every row nudged by one bf16 ulp. A wiring fault (a wrong `pre`, row or sublayer) would decorrelate the
    /// large rows too, so those are what must agree.
    @Test("verify rows through synthetic layers: the kernels ran, and every large row agrees with the stock graph")
    func layerForward() throws {
        guard let c = DSV41Synth.config else { return }
        try Device.withDefaultDevice(.gpu) {
            try VR.withRowKernels {
                let idx = [8, 9, 20, 21, 24]
                setKernels(true)
                let on = try DeepseekV41DecodeFusionFocusedTests.LayerStack(c, indices: idx, context: 700)
                setKernels(false)
                let off = try DeepseekV41DecodeFusionFocusedTests.LayerStack(c, indices: idx, context: 700)
                let ctl = try DeepseekV41DecodeFusionFocusedTests.LayerStack(c, indices: idx, context: 700)
                var kernelCos: [Float] = [], controlCos: [Float] = []
                for (step, rows) in Array(repeating: VR.rowCounts, count: 10).flatMap({ $0 }).enumerated() {
                    let h = MLXRandom.normal([1, rows, c.hcMult, c.hiddenSize], key: MLXRandom.key(UInt64(9000 + step)))
                        .asType(.bfloat16)
                    eval(h)
                    let before = (
                        DeepseekV41HCRows.runCount(.boundary) + DeepseekV41HCRows.runCount(.stage),
                        DeepseekV41RouterGEMV.rowRuns, DeepseekV41RouteRows.runCount
                    )
                    setKernels(true); let a = on.forward(h); eval(a)
                    let ranHC = DeepseekV41HCRows.runCount(.boundary) + DeepseekV41HCRows.runCount(.stage) - before.0
                    let ranRouter = DeepseekV41RouterGEMV.rowRuns - before.1
                    let ranRoute = DeepseekV41RouteRows.runCount - before.2
                    #expect(ranHC == 2 * idx.count, "rows \(rows): hc kernels ran \(ranHC) times")
                    #expect(ranRouter == idx.count, "rows \(rows): router kernel ran \(ranRouter) times")
                    #expect(ranRoute == idx.count, "rows \(rows): routing kernel ran \(ranRoute) times")
                    setKernels(false)
                    let b = off.forward(h); eval(b)
                    #expect(
                        DeepseekV41HCRows.runCount(.boundary) + DeepseekV41HCRows.runCount(.stage) == before.0 + ranHC
                            && DeepseekV41RouterGEMV.rowRuns == before.1 + ranRouter
                            && DeepseekV41RouteRows.runCount == before.2 + ranRoute,
                        "the control path ran no kernels")
                    // One element of every row nudged, as the kernels' rounding touches every row.
                    var flat = h.view(dtype: .uint16).reshaped(-1).asArray(UInt16.self)
                    for r in 0 ..< rows { flat[r * c.hcMult * c.hiddenSize + 3 + r] &+= 1 }
                    let nudged = MLXArray(flat, h.shape).view(dtype: .bfloat16)
                    let z = ctl.forward(nudged); eval(z)
                    let peak = MLX.abs(b.asType(.float32)).max().item(Float.self)
                    #expect(peak.isFinite && peak > 1e-3, "non-degenerate reference")
                    let cosK = rowCosines(a, b, rows: rows), cosC = rowCosines(z, b, rows: rows)
                    kernelCos += cosK; controlCos += cosC
                    let bf = b.asType(.float32).reshaped(rows, -1)
                    let norms = sqrt((bf * bf).sum(axis: -1)).asArray(Float.self)
                    for r in 0 ..< rows where norms[r] > 1e4 {
                        #expect(cosK[r] > 0.99, "step \(step) rows \(rows) row \(r): cosine \(cosK[r]), norm \(norms[r])")
                    }
                    if (cosK + cosC).contains(where: { $0 < 0.99 }) {
                        print(String(format: "[dsv41-rows] layers step %d rows=%d: on vs off max |d| %.3e, peak %.1f, row cos %@; 1-ulp control row cos %@; row norms %@",
                            step, rows, VR.maxAbsDiff(a, b), peak,
                            cosK.map { String(format: "%.5f", $0) }.joined(separator: " "),
                            cosC.map { String(format: "%.5f", $0) }.joined(separator: " "),
                            norms.map { String(format: "%.0f", $0) }.joined(separator: " ")))
                    }
                }
                kernelCos.sort(); controlCos.sort()
                let below = kernelCos.filter { $0 < 0.99 }.count, belowC = controlCos.filter { $0 < 0.99 }.count
                print("[dsv41-rows] layers: rows below cosine 0.99: kernels \(below)/\(kernelCos.count), 1-ulp control \(belowC)/\(controlCos.count)")
                #expect(kernelCos[kernelCos.count / 2] > 0.99, "median row cosine \(kernelCos[kernelCos.count / 2])")
            }
        }
    }

    /// One 6-row verify step through the layers, on and off, per layer: $VMLX_DSV41_GRAPH_DIR/layer_<on|off>_<k>.txt.
    @Test("6-row layer dispatch census (VMLX_DSV41_GRAPH_DIR)")
    func layerCensus() throws {
        guard let dir = ProcessInfo.processInfo.environment["VMLX_DSV41_GRAPH_DIR"], let c = DSV41Synth.config
        else { return }
        try Device.withDefaultDevice(.gpu) {
            try VR.withRowKernels {
                let stack = try DeepseekV41DecodeFusionFocusedTests.LayerStack(c, context: 2048)
                let R = 6
                DeepseekV41HCRows.boundaryEnabled = false
                for on in [false, true] {
                    setKernels(on)
                    // Admit first, so the graph written below is the served one and not the admission's.
                    _ = stack.forward(MLXRandom.normal([1, R, c.hcMult, c.hiddenSize], key: MLXRandom.key(4)).asType(.bfloat16))
                    var h = MLXRandom.normal([1, R, c.hcMult, c.hiddenSize], key: MLXRandom.key(5)).asType(.bfloat16)
                    var preMix = concatenated([MLXArray.ones([1, R, 1]), MLXArray.zeros([1, R, c.hcMult - 1])], axis: -1)
                        .asType(.float32)
                    eval(h, preMix)
                    let share = DSV41CompressShare()
                    let mask = createAttentionMask(h: h.reshaped(1, R, -1), cache: stack.caches.first)
                    for (k, (layer, cache)) in zip(stack.layers, stack.caches).enumerated() {
                        let r = layer.callStaggered(h, mask: mask, cache: cache, inputIds: nil, preMix: preMix, compressShare: share)
                        VR.writeGraph("\(dir)/layer_\(on ? "on" : "off")_\(stack.indices[k]).txt", [r.h, r.ffnPre])
                        eval(r.h, r.ffnPre)
                        h = r.h; preMix = r.ffnPre
                    }
                }
            }
        }
    }

    /// A 6-row step through the stack, kernels on vs off, in one process on one stack (alternating, first steps dropped).
    @Test("6-row layer microbenchmark, interleaved (VMLX_DSV41_DEC_BENCH=1)")
    func layerBench() throws {
        guard VR.benchEnabled, let c = DSV41Synth.config else { return }
        try Device.withDefaultDevice(.gpu) {
            try VR.withRowKernels {
                let stack = try DeepseekV41DecodeFusionFocusedTests.LayerStack(c, context: 2048)
                for R in [2, 6] {
                    var t: [String: [Double]] = [:]
                    let configs: [(String, Bool, Bool)] = [("stock", false, false), ("stage", true, false), ("boundary", true, true)]
                    for s in 0 ..< 45 {
                        for (name, on, boundary) in configs {
                            setKernels(on)
                            DeepseekV41HCRows.boundaryEnabled = boundary
                            let h = MLXRandom.normal([1, R, c.hcMult, c.hiddenSize], key: MLXRandom.key(UInt64(60_000 + s)))
                                .asType(.bfloat16)
                            eval(h)
                            let t0 = CFAbsoluteTimeGetCurrent()
                            eval(stack.forward(h))
                            if s >= 5 { t[name, default: []].append((CFAbsoluteTimeGetCurrent() - t0) * 1000) }
                        }
                    }
                    for (name, _, _) in configs {
                        let v = t[name]!.sorted()
                        print(String(format: "[dsv41-rows-bench] layer stack rows=%d %@: median %.3f ms/layer (min %.3f)",
                            R, name, v[v.count / 2] / Double(stack.layers.count), v[0] / Double(stack.layers.count)))
                    }
                }
            }
        }
    }
}

// MARK: - On a real bundle (VMLX_DSV41_BUNDLE; loads the model: run under the heavy lock)

private struct NullTokenizerLoader: TokenizerLoader {
    struct NullTokenizer: Tokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }
    func load(from directory: URL) async throws -> any Tokenizer { NullTokenizer() }
}

@Suite(
    "DeepSeek-V4.1 verify-row kernels on a real bundle", .serialized,
    .enabled(if: DSV41TestBundle.directory != nil, "needs VMLX_DSV41_BUNDLE"))
struct DeepseekV41VerifyRowBundleTests {
    /// Teacher-forced from one prefilled cache, restored in place between arms: the same rows as sequential 1-row steps (the
    /// reference), as a batched stock verify (the control), and through the kernels. Attention runs its 1-row split path
    /// in the reference and its dense path at 2+ rows in every other arm, so none of them is bit-equal to the reference;
    /// the question is whether the kernels sit at least as close to it as the stock graph does.
    @Test("real weights: every kernel admitted and served; verify rows at least as close to AR as the stock graph")
    func realBundle() async throws {
        let dir = try #require(DSV41TestBundle.directory)
        let context = try await LLMModelFactory.shared.load(from: dir, using: NullTokenizerLoader())
        let model = context.model
        let layers = model is DeepseekV4Model ? (model as! DeepseekV4Model).model.layers.count : 0
        #expect(layers == 40, "layers \(layers)")
        try Device.withDefaultDevice(.gpu) {
            try VR.withRowKernels(tf32: nil) {
                #expect(DeepseekV41HCRows.gemmRoundsToTF32, "run this with MLX_ENABLE_TF32 on, on an M5")
                let vocabulary = 60_000
                func ids(_ n: Int, seed: UInt64) -> MLXArray {
                    MLXRandom.randInt(Int32(1000) ..< Int32(vocabulary), [1, n], key: MLXRandom.key(seed)).asType(.int32)
                }
                let cache = model.newCache(parameters: GenerateParameters())
                let prompt = LMInput(tokens: ids(2700, seed: 1))
                switch try model.prepare(prompt, cache: cache, windowSize: GenerateParameters().prefillStepSize) {
                case .tokens(let rest):
                    eval(model(LMInput.Text(tokens: rest.tokens.reshaped(1, -1)), cache: cache, state: nil).logits)
                case .logits(let out):
                    eval(out.logits)
                }
                let checkpoint = try #require(SpeculativeCacheListCheckpoint(cache))
                func forward(_ x: MLXArray) -> MLXArray {
                    let l = model(LMInput.Text(tokens: x), cache: cache, state: nil).logits
                    eval(l)
                    checkpoint.restore()
                    return l
                }
                func ar(_ x: MLXArray) -> MLXArray {
                    let rows = (0 ..< x.dim(1)).map { r -> MLXArray in
                        let l = model(LMInput.Text(tokens: x[0..., r ..< (r + 1)]), cache: cache, state: nil).logits
                        eval(l)
                        return l
                    }
                    checkpoint.restore()
                    return concatenated(rows, axis: 1)
                }
                /// (top-1 agreement count, mean KL(reference || row), max |logit difference|).
                func score(_ ref: MLXArray, _ got: MLXArray) -> (Int, Float, Float) {
                    let a = ref.asType(.float32), b = got.asType(.float32)
                    let agree = (argMax(a, axis: -1) .== argMax(b, axis: -1)).asType(.int32).sum().item(Int32.self)
                    let la = logSoftmax(a, axis: -1), lb = logSoftmax(b, axis: -1)
                    let kl = (exp(la) * (la - lb)).sum(axis: -1).mean().item(Float.self)
                    return (Int(agree), kl, MLX.abs(a - b).max().item(Float.self))
                }
                func set(_ on: Bool, boundary: Bool) {
                    DeepseekV41HCRows.enabled = on; DeepseekV41RouterGEMV.rowsEnabled = on
                    DeepseekV41RouteRows.enabled = on; DeepseekV41HCRows.boundaryEnabled = boundary
                }
                for R in VR.rowCounts {
                    var totals: [String: (agree: Int, kl: Float, maxD: Float, rows: Int)] = [:]
                    var servedStage = 0, servedBoundary = 0
                    for step in 0 ..< 10 {
                        let x = ids(R, seed: UInt64(100 + R * 1000 + step))
                        eval(x)
                        let reference = ar(x)
                        let peak = MLX.abs(reference.asType(.float32)).max().item(Float.self)
                        #expect(reference.dim(1) == R && peak.isFinite && peak > 0, "non-degenerate reference")
                        for (name, on, boundary) in [("stock", false, false), ("stage", true, false), ("boundary", true, true)] {
                            set(on, boundary: boundary)
                            let before = (
                                DeepseekV41HCRows.runCount(.stage), DeepseekV41HCRows.runCount(.boundary),
                                DeepseekV41RouterGEMV.rowRuns, DeepseekV41RouteRows.runCount
                            )
                            let l = forward(x)
                            let ran = (
                                DeepseekV41HCRows.runCount(.stage) - before.0, DeepseekV41HCRows.runCount(.boundary) - before.1,
                                DeepseekV41RouterGEMV.rowRuns - before.2, DeepseekV41RouteRows.runCount - before.3
                            )
                            if on {
                                #expect(
                                    (boundary ? ran.1 : ran.0) == 2 * layers && ran.2 == layers && ran.3 == layers,
                                    "\(name) rows \(R): served hc \(ran.0)+\(ran.1), router \(ran.2), routing \(ran.3)")
                                servedStage += ran.0; servedBoundary += ran.1
                            } else {
                                #expect(ran == (0, 0, 0, 0), "the stock arm ran no kernels")
                            }
                            let sc = score(reference, l)
                            var t = totals[name] ?? (0, 0, 0, 0)
                            t.agree += sc.0; t.kl += sc.1; t.maxD = max(t.maxD, sc.2); t.rows += R
                            totals[name] = t
                        }
                    }
                    for name in ["stock", "stage", "boundary"] {
                        let t = totals[name]!
                        print(String(format: "[dsv41-rows-real] rows=%d %@ vs AR: top-1 %d/%d, mean KL %.3e, max |dlogit| %.4f",
                            R, name, t.agree, t.rows, t.kl / 10, t.maxD))
                    }
                    print("[dsv41-rows-real] rows=\(R) served: stage \(servedStage), boundary \(servedBoundary)")
                    #expect(totals["stage"]!.agree >= totals["stock"]!.agree - totals["stock"]!.rows / 20, "rows \(R)")
                }
            }
        }
    }
}
