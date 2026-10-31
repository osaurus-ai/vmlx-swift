// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The Swift port of vMLX's fast JANGT decode kernels (`JANGTFastDecode`, `VMLX_JANGT_FAST=1`).
///
/// - Sources: the generator's Metal equals vMLX Python's string for string (SHA-256) for every production configuration
///   at the test geometries and at MiniMax-M3.1's real shapes (`Fixtures/MiniMaxM3/jangt-fast.json`,
///   `scripts/minimax_m3_fixtures/generate_jangt_fast_fixtures.py` on vMLX c1a840a6).
/// - Python parity: the fused-MoE fixture block (same MLX PRNG keys, input digests checked first), rows 1 / 2 / 5 / 16:
///   pack, gate / up, down pack, routed sum, the routed layer's output and the fused M3 MoE block with the fast routed
///   experts (K6 ROT = 0, K7 ROUTED = 0, K8 ROUTED = 0) equal vMLX Python's bit for bit.
/// - Row invariance: R rows in one call == R one-row calls, R = 1...16, distinct / shared / identical routing; one row
///   changed changes only that row (control); the v3 kernels' rows differ from the fast ones (the comparison can fail).
/// - Model level: DSpark verify rows == decode steps with the lever on (JANGT routed layer and fused MoE block).
@Suite("JANGT fast decode kernels (vMLX fast_decode port)", .serialized)
struct JANGTFastDecodeTests {
    typealias FMT = MiniMaxM3FusedMoETests
    typealias FD = JANGTFastDecode

    struct Fixture: Decodable {
        struct Source: Decodable {
            let inputs: [String]
            let outputs: [String]
            let sha256: String
            let length: Int
        }
        struct Case: Decodable {
            let rows: Int
            let x: String
            let resid: String
            let outputs: [String: String]
        }
        struct Geometry: Decodable {
            let name: String
            let kgu: Double
            let kdn: Double
            let nog: Int
            let nod: Int
            let seed: Int
            let arrays: [String: String]
            let cases: [Case]
        }
        let vmlxCommit: String
        let sources: [String: Source]
        let geometries: [Geometry]
    }

    static func fixture() throws -> Fixture {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/MiniMaxM3/jangt-fast.json")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    static func sha(_ s: String) -> String { SHA256.hash(data: Data(s.utf8)).map { String(format: "%02x", $0) }.joined() }

    // MARK: sources

    /// Every kernel the production path builds for one routed layer shape (gate / up, down one row, down 2+ rows, the
    /// combine, both packs).
    static func kernels(D: Int, I: Int, kgu: Double, kdn: Double, nog: Int, nod: Int, slots: Int = 4)
        -> [FD.Kernel]
    {
        let wgu = JANGTCode.wordsPerRow(inputs: D, k: kgu), wdn = JANGTCode.wordsPerRow(inputs: I, k: kdn)
        return [
            FD.gateUpKernel(
                kin: D, n: I, k: kgu, outliers: nog, words: wgu, xdt: "bfloat16_t", odt: "float", limit: 7,
                activation: .oai(alpha: 1.702)),
            FD.downCombinedKernel(kin: I, n: D, k: kdn, outliers: nod, words: wdn, hdt: "float", slots: slots),
            FD.downSlotsKernel(kin: I, n: D, k: kdn, outliers: nod, words: wdn, hdt: "float"),
            FD.combineKernel(n: D, slots: slots),
            FD.pack1Kernel(kin: D, xdt: "bfloat16_t", perRowSigns: false),
            FD.pack1Kernel(kin: I, xdt: "float", perRowSigns: true),
        ]
    }

    @Test("Generated Metal == vMLX Python's, string for string: test geometries and every M3.1 real-shape configuration")
    func sourcesEqualPython() throws {
        let f = try Self.fixture()
        var configs: [(String, [FD.Kernel])] = [
            ("test k1.5/2.5 outliers", Self.kernels(D: 1024, I: 512, kgu: 1.5, kdn: 2.5, nog: 1, nod: 3)),
            ("test k2/3 plain", Self.kernels(D: 1024, I: 512, kgu: 2, kdn: 3, nog: 0, nod: 0)),
        ]
        for k in [1.0, 1.5, 2.0, 2.5] {
            for nOut in [0, 8] {
                configs.append(("M3.1 K\(k) outliers \(nOut)", Self.kernels(D: 6144, I: 3072, kgu: k, kdn: k, nog: nOut, nod: nOut)))
            }
        }
        var compared = 0
        let dump = FileManager.default.temporaryDirectory.appendingPathComponent("jangt-fast-sources")
        for (label, kernels) in configs {
            for k in kernels {
                guard let py = f.sources[k.name] else {
                    Issue.record("\(label): vMLX Python built no kernel named \(k.name)")
                    continue
                }
                #expect(py.inputs == k.inputs && py.outputs == k.outputs, "\(k.name): names")
                if Self.sha(k.source) != py.sha256 {
                    try? FileManager.default.createDirectory(at: dump, withIntermediateDirectories: true)
                    try? k.source.write(to: dump.appendingPathComponent(k.name + ".metal"), atomically: true, encoding: .utf8)
                    let message = "\(label): \(k.name) source differs (Swift \(k.source.utf8.count) B, Python "
                        + "\(py.length) B); Swift copy in \(dump.path)"
                    Issue.record(Comment(rawValue: message))
                }
                compared += 1
            }
        }
        // Every fast kernel Python recorded is one of these.
        let pythonFast = Set(f.sources.keys.filter { $0.hasPrefix("jtf") })
        let swift = Set(configs.flatMap { $0.1.map(\.name) })
        #expect(pythonFast.subtracting(swift).isEmpty, "Python kernels not generated here: \(pythonFast.subtracting(swift).sorted())")
        print("[jangt fast] \(compared) generated sources compared with vMLX Python \(f.vmlxCommit)")
    }

    // MARK: Python parity

    @Test("Every kernel output == vMLX Python's bit for bit, and the fused M3 MoE block with the fast routed experts")
    func pythonParity() throws {
        try Device.withDefaultDevice(.gpu) {
            let f = try Self.fixture()
            let saved = MiniMaxM3FusedMoE.sharedInKernelRows
            defer { MiniMaxM3FusedMoE.sharedInKernelRows = saved }
            MiniMaxM3FusedMoE.sharedInKernelRows = 16
            for g in f.geometries {
                let (routed, routedDigests) = try FMT.routed(kgu: g.kgu, kdn: g.kdn, nog: g.nog, nod: g.nod, seed: g.seed)
                var digests = routedDigests
                let block = try FMT.block(routed, seed: g.seed, digests: &digests)
                for (name, want) in g.arrays {
                    #expect(digests[name] == want, "\(g.name): input array \(name) differs from the generator's")
                }
                let fm = MiniMaxM3FusedMoE.shared
                let L = try #require(fm.layer(block))
                let t = L.trellis
                #expect(FD.supported(t))
                let (xAll, rAll) = FMT.rows(seed: g.seed)
                for c in g.cases {
                    let R = c.rows
                    let x = xAll[0..., ..<R], res = rAll[0..., ..<R]
                    #expect(FMT.digest(x) == c.x && FMT.digest(res) == c.resid, "\(g.name) rows \(R): inputs")
                    let x2 = contiguous(x.reshaped(R, FMT.D)), r2 = contiguous(res.reshaped(R, FMT.D))
                    let (sig, _) = fm.router(L, x2, rot: false)
                    let gu = fm.gateUp(L, x2, sig: sig, xr: nil, shared: true, routed: false)
                    let pgu = FD.packRows1(x2, signs: t.gateUpSigns)
                    let h = FD.gateUp(x2, ids: gu.inds, t)
                    let pdn = FD.packRows1(h, signs: t.downSigns, rowExperts: gu.inds.flattened())
                    let ry = FD.routed(x2, ids: gu.inds, weights: gu.scores, t)
                    let out = fm.down(L, hr: nil, h: gu.h, inds: gu.inds, scores: gu.scores, hs: gu.hs, r2: r2, sdo: nil, ry: ry)
                    let (switchOut, full) = FD.$override.withValue(true) {
                        (routed.callRouted(x2.reshaped(1, R, FMT.D), indices: gu.inds.asType(.int32).reshaped(1, R, -1),
                                           scores: gu.scores.reshaped(1, R, -1)),
                         fm.residual(block, xNormed: x, resid: res))
                    }
                    let got: [String: MLXArray] = [
                        "sig": sig, "hs": gu.hs, "inds": gu.inds, "scores": gu.scores, "pgu_xq": pgu.0, "pgu_xinv": pgu.1,
                        "pgu_st": pgu.2, "h": h, "pdn_xq": pdn.0, "pdn_xinv": pdn.1, "pdn_st": pdn.2, "ry": ry,
                        "out": out, "switch": switchOut.reshaped(R, FMT.D),
                    ]
                    for (name, want) in c.outputs {
                        let a = try #require(got[name], "\(name) not computed")
                        #expect(FMT.digest(a) == want, "\(g.name) rows \(R): \(name) differs from vMLX Python")
                    }
                    #expect(arrayEqual(full.reshaped(R, FMT.D), out).item(Bool.self), "\(g.name) rows \(R): residual != kernels")
                }
            }
        }
    }

    // MARK: row invariance

    /// One routed layer of the test geometry and `R` rows of inputs, routing and weights: `distinct` (random 4 of 16
    /// per row), `shared` (every row the same 4 experts in different orders), `identical` (same experts, same order).
    static func setup(_ routing: String, seed: Int) throws
        -> (JANGTRoutedExpertLayer.TrellisTensors, MLXArray, MLXArray, MLXArray)
    {
        let routed = try FMT.routed(kgu: 1.5, kdn: 2.5, nog: 1, nod: 3, seed: 100).0
        let t = try #require(routed.trellisTensors)
        let x = (FMT.uniform(seed, [16, FMT.D]) * Float(0.5)).asType(.bfloat16)
        var ids: [[Int32]] = []
        var rng = UInt64(seed)
        func next() -> Int { rng = rng &* 6_364_136_223_846_793_005 &+ 1; return Int(rng >> 33) }
        for r in 0 ..< 16 {
            switch routing {
            case "shared": ids.append([3, 7, 11, 14].shuffledDeterministically(seed: r))
            case "identical": ids.append([3, 7, 11, 14])
            default:
                var row: [Int32] = []
                while row.count < 4 { let e = Int32(next() % 16); if !row.contains(e) { row.append(e) } }
                ids.append(row)
            }
        }
        let w = MLXRandom.uniform(Float(0.1) ..< Float(0.6), [16, 4], key: MLXRandom.key(UInt64(seed + 1)), stream: .cpu)
        eval(x, w)
        return (t, x, MLXArray(ids.flatMap { $0 }).reshaped(16, 4), w)
    }

    @Test("Row invariance: R rows in one call == R one-row calls (R = 1...16; distinct / shared / identical routing)",
          arguments: ["distinct", "shared", "identical"])
    func rowInvariance(routing: String) throws {
        try Device.withDefaultDevice(.gpu) {
            let (t, x, ids, w) = try Self.setup(routing, seed: 31)
            let alone = (0 ..< 16).map { FD.routed(x[$0 ..< ($0 + 1)], ids: ids[$0 ..< ($0 + 1)], weights: w[$0 ..< ($0 + 1)], t) }
            eval(alone)
            var bad: [Int] = []
            for R in 1 ... 16 {
                let together = FD.routed(x[..<R], ids: ids[..<R], weights: w[..<R], t)
                for r in 0 ..< R where !arrayEqual(together[r ..< (r + 1)], alone[r]).item(Bool.self) { bad.append(R) }
            }
            #expect(bad.isEmpty, "\(routing): window sizes \(Set(bad).sorted()) changed a row")
            // Control 1: one row's input changed (one bf16 step) changes that row only.
            var xs = x.asType(.float32).asArray(Float.self)
            xs[5 * FMT.D + 17] = xs[5 * FMT.D + 17] * 1.01 + 0.05
            let perturbed = MLXArray(xs).reshaped(16, FMT.D).asType(.bfloat16)
            let base = FD.routed(x, ids: ids, weights: w, t), moved = FD.routed(perturbed, ids: ids, weights: w, t)
            let changed = (0 ..< 16).filter { !arrayEqual(base[$0], moved[$0]).item(Bool.self) }
            #expect(changed == [5], "\(routing): perturbing row 5 changed rows \(changed)")
            // Control 2: the v3 kernels' rows are not the fast kernels' (so equality above is a real check).
            let v3 = JANGTKernels.down(
                hr: JANGTKernels.rotateRows(
                    JANGTKernels.gateUp(xr: JANGTKernels.rotateRows(x, signs: t.gateUpSigns), x: x, indices: ids.asType(.uint32),
                                        gate: t.gate, up: t.up, limit: t.limit, activation: t.activation, outputDType: .float32),
                    signs: t.downSigns, rowExperts: ids.flattened().asType(.uint32)),
                h: JANGTKernels.gateUp(xr: JANGTKernels.rotateRows(x, signs: t.gateUpSigns), x: x, indices: ids.asType(.uint32),
                                       gate: t.gate, up: t.up, limit: t.limit, activation: t.activation, outputDType: .float32),
                indices: ids.asType(.uint32), weights: w, projection: t.down)
            let rel = (abs(v3 - base).max() / abs(v3).max()).item(Float.self)
            #expect(!arrayEqual(v3, base).item(Bool.self) && rel < 2e-2, "\(routing): v3 vs fast rel \(rel)")
            print("[jangt fast] \(routing): rows invariant over windows 1...16; v3 vs fast max rel \(String(format: "%.2e", rel))")
        }
    }

    // MARK: model level

    @Test("Lever on: DSpark verify rows == decode steps (JANGT routed layers, and with the fused MoE block on)",
          arguments: [(773, 2, false), (765, 8, false), (771, 16, true), (300, 17, true)])
    func verifyRowsEqualDecodeStepsFast(promptLength: Int, width: Int, fusedMoE: Bool) throws {
        try Device.withDefaultDevice(.gpu) {
            try FD.$override.withValue(true) {
                MiniMaxM3FusedDecode.forceEnabled = fusedMoE
                MiniMaxM3FusedDecode.path = fusedMoE ? MiniMaxM3FusedKernelPath(attention: false, moe: true) : nil
                defer {
                    MiniMaxM3FusedDecode.forceEnabled = nil
                    MiniMaxM3FusedDecode.path = nil
                }
                let model = try FMT.model()
                model.model.dsparkTapLayers = [1, 3]
                model.model.dsparkTailCapacity = 64
                let prompt = MiniMaxM3DSparkTests.ids(promptLength, seed: UInt64(promptLength))
                let rows = MiniMaxM3DSparkTests.ids(width, seed: UInt64(7 * width))
                let cache = model.newCache(parameters: nil)
                MiniMaxM3DSparkTests.prefill(model, prompt, cache: cache)
                MiniMaxM3FusedMoE.stats.reset()
                let logits = try MiniMaxM3DSparkTests.verify(model, rows, cache: cache)
                let reference = model.newCache(parameters: nil)
                MiniMaxM3DSparkTests.prefill(model, prompt, cache: reference)
                let steps = rows.map { MiniMaxM3DSparkTests.decode(model, $0, cache: reference) }
                if fusedMoE { #expect((MiniMaxM3FusedMoE.stats.snapshot["fast_routed"] ?? 0) > 0, "fused fast path ran") }
                let bad = (0 ..< width).filter { !MiniMaxM3DSparkTests.equal(logits[$0], steps[$0]) }
                #expect(bad.isEmpty, "verify rows \(bad) differ from decode steps (fused MoE \(fusedMoE))")
                let problems = MiniMaxM3DSparkTests.cacheProblems(cache, reference)
                #expect(problems.isEmpty, "\(problems.prefix(4))")
            }
        }
    }

    @Test("Lever on changes the routed output (the switch takes the fast kernels), off keeps v3")
    func leverSwitchesKernels() throws {
        try Device.withDefaultDevice(.gpu) {
            let routed = try FMT.routed(kgu: 2, kdn: 3, nog: 0, nod: 0, seed: 200).0
            let (_, x, ids, w) = try Self.setup("distinct", seed: 41)
            let input = x[..<3].reshaped(1, 3, FMT.D), i = ids[..<3].reshaped(1, 3, 4), s = w[..<3].reshaped(1, 3, 4)
            let off = FD.$override.withValue(false) { routed.callRouted(input, indices: i, scores: s) }
            let on = FD.$override.withValue(true) { routed.callRouted(input, indices: i, scores: s) }
            let t = try #require(routed.trellisTensors)
            let direct = FD.routed(x[..<3], ids: ids[..<3], weights: w[..<3], t).asType(.bfloat16).reshaped(1, 3, FMT.D)
            #expect(arrayEqual(on, direct).item(Bool.self) && !arrayEqual(on, off).item(Bool.self))
        }
    }

    // MARK: speed (opt-in)

    @Test("Benchmark (opt-in VMLX_JANGT_FAST_BENCH=1): routed experts v3 vs fast, M3.1 shapes, ms per 57 layers")
    func benchmark() throws {
        guard ProcessInfo.processInfo.environment["VMLX_JANGT_FAST_BENCH"] == "1" else { return }
        try Device.withDefaultDevice(.gpu) {
            let (D, I, E) = (6144, 3072, 128)
            // M3.1's per-layer plan (gate/up, down K): 18 x 1.5/2.5, 15 x 2/2.5, 12 x 1/1.5, 8 x 1.5/2, 2 x 1/1, 2 x 1/2.
            let plan: [(Double, Double, Int)] = [(1.5, 2.5, 18), (2, 2.5, 15), (1, 1.5, 12), (1.5, 2, 8), (1, 1, 2), (1, 2, 2)]
            for rows in [1, 4, 8, 16] {
                var v3Total = 0.0, fastTotal = 0.0
                var line: [String] = []
                for (kgu, kdn, count) in plan {
                    // Distinct banks so weights stream: 6 at one row, 3 at 4-16 rows (each probe walks them all).
                    let nb = rows == 1 ? 6 : 3
                    var layers: [JANGTRoutedExpertLayer.TrellisTensors] = []
                    for l in 0 ..< nb {
                        let r = try FMT.routed(kgu: kgu, kdn: kdn, nog: 0, nod: 3, seed: 3000 + 100 * l + Int(kgu * 10), D: D, I: I, E: E).0
                        layers.append(try #require(r.trellisTensors))
                    }
                    let x = (FMT.uniform(77 + rows, [rows, D]) * Float(0.5)).asType(.bfloat16)
                    var ids: [Int32] = []
                    for r in 0 ..< rows { for j in 0 ..< 4 { ids.append(Int32((r * 37 + j * 29 + 5) % E)) } }
                    let idx = MLXArray(ids).reshaped(rows, 4)
                    let w = MLXArray([Float](repeating: 0.25, count: rows * 4)).reshaped(rows, 4)
                    eval(x, idx, w)
                    func time(_ f: (JANGTRoutedExpertLayer.TrellisTensors) -> MLXArray) -> Double {
                        var t: [Double] = []
                        for probe in 0 ... 9 {
                            let t0 = Date()
                            var outs: [MLXArray] = []
                            for L in layers { outs.append(f(L)) }
                            eval(outs)
                            if probe > 1 { t.append(Date().timeIntervalSince(t0) * 1000 / Double(nb)) }
                        }
                        t.sort()
                        return t[t.count / 2]
                    }
                    func v3(_ t: JANGTRoutedExpertLayer.TrellisTensors) -> MLXArray {
                        let h = JANGTKernels.gateUp(
                            xr: JANGTKernels.rotateRows(x, signs: t.gateUpSigns), x: x, indices: idx.asType(.uint32),
                            gate: t.gate, up: t.up, limit: t.limit, activation: t.activation, outputDType: .float32)
                        return JANGTKernels.down(
                            hr: JANGTKernels.rotateRows(h, signs: t.downSigns, rowExperts: idx.flattened().asType(.uint32)),
                            h: h, indices: idx.asType(.uint32), weights: w, projection: t.down)
                    }
                    // Interleaved: v3, fast, v3, fast (medians of 8 after 2 discarded each).
                    let a1 = time(v3), b1 = time { FD.routed(x, ids: idx, weights: w, $0) }
                    let a2 = time(v3), b2 = time { FD.routed(x, ids: idx, weights: w, $0) }
                    let a = (a1 + a2) / 2, b = (b1 + b2) / 2
                    v3Total += a * Double(count)
                    fastTotal += b * Double(count)
                    line.append("K\(kgu)/\(kdn) \(String(format: "%.3f/%.3f", a, b))")
                }
                print("[jangt fast bench] rows \(rows): v3 \(String(format: "%.2f", v3Total)) -> fast \(String(format: "%.2f", fastTotal)) ms per 57 layers (\(String(format: "%.2fx", v3Total / fastTotal))); per layer v3/fast ms: \(line.joined(separator: ", "))")
            }
        }
    }
}

extension Array where Element == Int32 {
    /// A fixed permutation per seed (tests only).
    func shuffledDeterministically(seed: Int) -> [Int32] {
        var a = self
        var s = UInt64(seed &+ 7)
        for i in stride(from: a.count - 1, to: 0, by: -1) {
            s = s &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            a.swapAt(i, Int(s >> 33) % (i + 1))
        }
        return a
    }
}
