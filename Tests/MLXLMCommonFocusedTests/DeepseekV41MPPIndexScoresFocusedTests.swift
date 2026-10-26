// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 indexer scores on the M5 matrix units (`DeepseekV41MPPIndexScores`, port of vMLX
// `vmlx_engine/metal/dsa_index_scores_mpp.py`). Small synthetic tensors on the GPU; no model loads and no
// shared MLX test lock, so run this suite on its own.
//
// Python parity (tests/test_dsa_index_scores_mpp.py): bf16 q / k with fp32 accumulation match an fp64
// reference of the same bf16 values to 2e-6 of the largest score, clamped row / column tails and padded
// sub-tile calls included; the top-512 sets match the fp32 fused kernel (overlap >= 0.999); unsupported
// inputs, `enabled: false`, the CPU and unavailable matrix units fall back (nil -> fp32 fused -> stock).
//
// Kernel tests need Metal 4 tensor ops (M5-class GPU) and are skipped elsewhere. The S=2048, T=100k
// microbenchmark runs only with VMLX_DSV41_MPP_BENCH=1.

import Accelerate
import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

private let mppAvailable: Bool = Device.withDefaultDevice(.gpu) { DeepseekV41MPPIndexScores.available() }

@Suite("DeepSeek-V4.1 matrix-unit indexer scores", .serialized)
struct DeepseekV41MPPIndexScoresFocusedTests {
    typealias M = DeepseekV41MPPIndexScores

    /// bf16-valued inputs (python `_inputs`): q (1, S, H, D), k (1, P, D) as `dtype`; w = bf16(0.2 * normal)
    /// held in fp32; plus their exact fp32 values for the reference.
    struct Inputs {
        let q: MLXArray, k: MLXArray, w: MLXArray
        let qv: [Float], kv: [Float], wv: [Float]
        let S: Int, P: Int, H: Int, D: Int
    }

    static func inputs(S: Int, P: Int, H: Int = 32, D: Int = 128, seed: UInt64, dtype: DType = .bfloat16)
        -> Inputs
    {
        typealias F = DeepseekV41FusedIndexScoresFocusedTests
        let q = MLXArray(F.normals(S * H * D, seed: seed), [1, S, H, D]).asType(dtype)
        let k = MLXArray(F.normals(P * D, seed: seed &+ 1), [1, P, D]).asType(dtype)
        let w = MLXArray(F.normals(S * H, seed: seed &+ 2, scale: 0.2), [1, S, H])
            .asType(.bfloat16).asType(.float32)
        eval(q, k, w)
        return Inputs(
            q: q, k: k, w: w, qv: q.asType(.float32).asArray(Float.self),
            kv: k.asType(.float32).asArray(Float.self), wv: w.asArray(Float.self), S: S, P: P, H: H, D: D)
    }

    /// fp64 `out[s, p] = sum_h w[s, h] * relu(scale * q[s, h, :] . k[p, :])` (dot products by vDSP in fp64).
    static func reference64(_ x: Inputs, scale: Double) -> [Double] {
        let (S, P, H, D) = (x.S, x.P, x.H, x.D)
        let qd = x.qv.map(Double.init)
        var kT = [Double](repeating: 0, count: D * P)  // [D][P]
        for p in 0..<P { for d in 0..<D { kT[d * P + p] = Double(x.kv[p * D + d]) } }
        var out = [Double](repeating: 0, count: S * P)
        var dots = [Double](repeating: 0, count: H * P)
        for s in 0..<S {
            qd.withUnsafeBufferPointer { qp in
                vDSP_mmulD(
                    qp.baseAddress! + s * H * D, 1, kT, 1, &dots, 1, vDSP_Length(H), vDSP_Length(P),
                    vDSP_Length(D))
            }
            for h in 0..<H {
                let wv = Double(x.wv[s * H + h])
                for p in 0..<P { out[s * P + p] += wv * max(dots[h * P + p] * scale, 0) }
            }
        }
        return out
    }

    static func maxRel(_ out: MLXArray, _ ref: [Double]) -> Double {
        let got = out.asArray(Float.self)
        let peak = ref.map(abs).max() ?? 0
        return (zip(got, ref).map { abs(Double($0) - $1) }.max() ?? .infinity) / peak
    }

    @Test("availability: M5-class architectures, env switch, self-check")
    func availability() {
        #expect(M.hasMatrixUnits(architecture: "applegpu_g17s"))
        #expect(M.hasMatrixUnits(architecture: "applegpu_g17g"))
        #expect(M.hasMatrixUnits(architecture: "applegpu_g18p"))
        #expect(!M.hasMatrixUnits(architecture: "applegpu_g17p"))
        #expect(!M.hasMatrixUnits(architecture: "applegpu_g16s"))
        #expect(!M.hasMatrixUnits(architecture: "applegpu_g15d"))
        #expect(!M.hasMatrixUnits(architecture: "Unknown"))
        #expect(M.requested([:]))
        #expect(M.requested(["VMLX_DSV41_INDEX_MPP": "1"]))
        for off in ["0", "false", "OFF", " no "] { #expect(!M.requested(["VMLX_DSV41_INDEX_MPP": off])) }
        #expect(M.enabled)  // unset in the test environment
        let arch = GPU.deviceInfo().architecture
        print("[mpp-index] architecture \(arch), hardware \(M.hardwareSupported()), available \(mppAvailable)")
        #expect(mppAvailable == M.hardwareSupported())  // the self-check passes wherever the hardware is there
        #expect(M.available() == mppAvailable)  // cached
    }

    @Test("a kernel that does not compile is an MLX error the self-check catches (unavailable, no crash)")
    func compileFailureIsCaught() {
        let broken = MLXFast.metalKernel(
            name: "vmlx_dsv41_mpp_index_scores_broken_probe", inputNames: ["x"], outputNames: ["out"],
            source: "  out[0] = this_symbol_does_not_exist(x[0]);\n")
        let caught: Bool = Device.withDefaultDevice(.gpu) {
            do {
                _ = try withError { error in
                    let y = broken(
                        [MLXArray([Float(1)])], grid: (1, 1, 1), threadGroup: (1, 1, 1),
                        outputShapes: [[1]], outputDTypes: [.float32])[0]
                    eval(y)
                    try error.check()
                    return y
                }
                return false
            } catch {
                return true
            }
        }
        #expect(caught)
    }

    @Test(
        "matches an fp64 reference of the bf16 values (whole tiles, tails, sub-tile calls, head shapes)",
        .enabled(if: mppAvailable))
    func matchesFP64Reference() throws {
        try Device.withDefaultDevice(.gpu) {
            let cases: [(Int, Int, Int, Int)] = [
                (16, 256, 32, 128),  // exactly one tile
                (64, 1024, 32, 128),  // whole tiles
                (100, 999, 32, 128),  // row and column tails (clamped, overlapping last tiles)
                (17, 257, 32, 128),  // one past a tile in both axes
                (300, 4097, 32, 128),
                (5, 100, 32, 128),  // below one tile: zero-padded rows and keys
                (1, 3000, 32, 128),
                (2048, 2500, 32, 128),  // a full prefill chunk
                (40, 49, 4, 32), (33, 300, 8, 64), (20, 600, 1, 256),  // other head geometries
            ]
            for (S, P, H, D) in cases {
                let x = Self.inputs(S: S, P: P, H: H, D: D, seed: UInt64(S * 7 + P))
                for scale in [1.0, pow(Double(D), -0.5)] {
                    let out = try #require(
                        M.poolIndexScores(q: x.q, poolKeys: x.k, headW: x.w, scale: Float(scale)),
                        "S\(S) P\(P)")
                    #expect(out.shape == [1, S, P] && out.dtype == .float32)
                    let err = Self.maxRel(out, Self.reference64(x, scale: scale))
                    print("[mpp-index] S=\(S) P=\(P) H=\(H) D=\(D) scale=\(scale) max rel err \(err)")
                    #expect(err <= 2e-6, "S\(S) P\(P) H\(H) D\(D) scale \(scale): \(err)")
                }
            }
        }
    }

    @Test("fp16 operands and bf16 head weights", .enabled(if: mppAvailable))
    func fp16OperandsBF16Weights() throws {
        try Device.withDefaultDevice(.gpu) {
            let x = Self.inputs(S: 100, P: 999, seed: 11, dtype: .float16)
            let out = try #require(M.poolIndexScores(
                q: x.q, poolKeys: x.k, headW: x.w.asType(.bfloat16), scale: 1))
            #expect(Self.maxRel(out, Self.reference64(x, scale: 1)) <= 2e-6)
        }
    }

    @Test("every tail cell is written, last row and last column included", .enabled(if: mppAvailable))
    func everyTailCellWritten() throws {
        try Device.withDefaultDevice(.gpu) {
            for (S, P) in [(33, 513), (47, 1023), (16, 257)] {
                let x = Self.inputs(S: S, P: P, seed: 5 &+ UInt64(S))
                let out = try #require(M.poolIndexScores(q: x.q, poolKeys: x.k, headW: x.w, scale: 0.5))
                let got = out.asArray(Float.self)
                let ref = Self.reference64(x, scale: 0.5)
                let tol = 2e-6 * (ref.map(abs).max() ?? 0)
                #expect(got.allSatisfy { $0.isFinite })
                let lastRow = (0..<P).map { abs(Double(got[(S - 1) * P + $0]) - ref[(S - 1) * P + $0]) }
                let lastCol = (0..<S).map { abs(Double(got[$0 * P + P - 1]) - ref[$0 * P + P - 1]) }
                #expect((lastRow.max() ?? .infinity) <= tol, "S\(S) P\(P) last row")
                #expect((lastCol.max() ?? .infinity) <= tol, "S\(S) P\(P) last column")
                // A query's scores do not depend on the rows that share its call.
                let part = try #require(M.poolIndexScores(
                    q: x.q[0..., (S - 3) ..< S], poolKeys: x.k, headW: x.w[0..., (S - 3) ..< S], scale: 0.5))
                let sub = Array(ref[((S - 3) * P) ..< (S * P)])
                #expect(Self.maxRel(part, sub) * (sub.map(abs).max() ?? 0) <= tol)
            }
        }
    }

    @Test("top-512 sets match the fp32 fused kernel", .enabled(if: mppAvailable))
    func top512MatchesFP32Kernel() throws {
        try Device.withDefaultDevice(.gpu) {
            for (S, P) in [(256, 8192), (2048, 6000)] {
                let x = Self.inputs(S: S, P: P, seed: UInt64(S + P))
                let fast = try #require(M.poolIndexScores(q: x.q, poolKeys: x.k, headW: x.w, scale: 1))
                let f32 = try #require(DeepseekV41FusedIndexScores.poolIndexScores(
                    q: x.q.asType(.float32), poolKeys: x.k.asType(.float32), headW: x.w, scale: 1))
                let top: (MLXArray) -> [Int32] = {
                    argPartition(-$0[0], kth: 511, axis: -1)[0..., 0 ..< 512].asArray(Int32.self)
                }
                let a = top(fast), b = top(f32)
                var overlap = 0.0
                for i in 0..<S {
                    overlap += Double(Set(a[(i * 512) ..< ((i + 1) * 512)])
                        .intersection(b[(i * 512) ..< ((i + 1) * 512)]).count) / 512
                }
                overlap /= Double(S)
                let diff = (abs(fast - f32).max() / abs(f32).max()).item(Float.self)
                print("[mpp-index] S=\(S) P=\(P) top-512 overlap \(overlap), max rel diff vs fp32 \(diff)")
                #expect(overlap >= 0.999)
            }
        }
    }

    @Test("admission, off switch, CPU and unavailable matrix units fall back")
    func fallsBack() {
        Device.withDefaultDevice(.gpu) {
            let x = Self.inputs(S: 64, P: 512, seed: 1)
            let q = x.q, k = x.k, w = x.w
            #expect(M.admit(q: q, poolKeys: k, headW: w, scale: 1))
            #expect((M.poolIndexScores(q: q, poolKeys: k, headW: w, scale: 1) != nil) == mppAvailable)
            #expect(!M.admit(q: q.asType(.float32), poolKeys: k.asType(.float32), headW: w, scale: 1))
            #expect(!M.admit(q: q, poolKeys: k.asType(.float16), headW: w, scale: 1))  // mixed dtypes
            #expect(!M.admit(q: q, poolKeys: k, headW: w, scale: 0))  // scale <= 0
            #expect(!M.admit(  // D % 32
                q: q[0..., 0..., 0..., 0 ..< 120], poolKeys: k[0..., 0..., 0 ..< 120], headW: w, scale: 1))
            #expect(!M.admit(q: concatenated([q, q]), poolKeys: k, headW: w, scale: 1))  // batch 2
            #expect(M.poolIndexScores(q: q.asType(.float32), poolKeys: k.asType(.float32), headW: w, scale: 1)
                == nil)
            #expect(M.poolIndexScores(q: q, poolKeys: k, headW: w, scale: 1, enabled: false) == nil)

            // The indexer's gate. Decode and dSpark verify stay stock; prefill prefers the matrix units,
            // else (turned off, unavailable, head geometry not admitted) the fp32 fused kernel.
            typealias Path = DeepseekV41MPPIndexScores.PrefillPath
            let yes = { true }, no = { false }
            #expect(M.prefillPath(queries: 1, keys: 100_000, heads: 32, dim: 128, mppAvailable: yes) == .stock)
            #expect(M.prefillPath(queries: 8, keys: 100_000, heads: 32, dim: 128, mppAvailable: yes) == .stock)
            #expect(M.prefillPath(queries: 2048, keys: 31, heads: 32, dim: 128, mppAvailable: yes) == .stock)
            #expect(M.prefillPath(queries: 9, keys: 100_000, heads: 32, dim: 128, mppAvailable: yes)
                == .matrixUnits)
            #expect(M.prefillPath(queries: 2048, keys: 100_000, heads: 32, dim: 128, mppAvailable: no)
                == .fp32Fused)
            #expect(M.prefillPath(
                queries: 2048, keys: 100_000, heads: 32, dim: 128, mppEnabled: false, mppAvailable: yes)
                == .fp32Fused)
            #expect(M.prefillPath(queries: 2048, keys: 100_000, heads: 4, dim: 16, mppAvailable: yes)
                == .fp32Fused)
            #expect(M.prefillPath(queries: 2048, keys: 100_000, heads: 32, dim: 128)
                == (mppAvailable ? Path.matrixUnits : .fp32Fused))
        }
        Device.withDefaultDevice(.cpu) {
            let x = Self.inputs(S: 64, P: 512, seed: 2)
            #expect(M.poolIndexScores(q: x.q, poolKeys: x.k, headW: x.w, scale: 1) == nil)
            #expect(M.prefillPath(queries: 2048, keys: 100_000, heads: 32, dim: 128, mppAvailable: { true })
                == .stock)
        }
    }

    @Test("end-to-end select() on the matrix-unit path matches the fp64 loop reference",
        .enabled(if: mppAvailable))
    func selectMatrixUnitsMatchesReference() throws {
        try Device.withDefaultDevice(.gpu) {
            typealias I = DeepseekV41IndexerFocusedTests
            var cfg = I.tinyConfig()
            cfg.indexHeadDim = 32  // admitted by the matrix units (D % 32 == 0)
            let h = cfg.indexNHeads, d = cfg.indexHeadDim
            let rope = DeepseekV4RoPE(
                dim: 8, base: 160_000, factor: 1, origMaxPos: 65536, betaFast: 32, betaSlow: 1)
            let owner = DeepseekV41Indexer(config: cfg, layerIdx: 2, compressRatio: 1)
            var params: [String: MLXArray] = [:]
            for (i, (key, value)) in owner.parameters().flattened().enumerated() {
                params[key] = MLXArray(I.values(value.size, seed: UInt64(30 + i), scale: 0.25), value.shape)
            }
            try owner.update(parameters: ModuleParameters.unflattened(params), verify: .noUnusedKeys)
            let L = 40, startPos = 9, T = startPos + L
            #expect(M.prefillPath(queries: L, keys: T, heads: h, dim: d) == .matrixUnits)
            let latent = MLXArray(I.values(T * cfg.headDim, seed: 50, scale: 0.25), [1, T, cfg.headDim])
            let indexK = owner.indexKeys(latent: latent, base: 0, rope: rope)
            let x = MLXArray(I.values(L * cfg.hiddenSize, seed: 51), [1, L, cfg.hiddenSize])
            let qr = MLXArray(I.values(L * cfg.qLoraRank, seed: 52), [1, L, cfg.qLoraRank])
            let sel = owner.select(
                x: x, qResidual: qr, indexK: DeepseekV4PoolView(dense: indexK), startPos: startPos,
                rope: rope, candidates: nil)
            let idx = try #require(sel.topK)
            #expect(idx.shape == [1, L, cfg.indexTopk] && sel.candidates?.shape == [L, T])

            var q = owner.wqB(qr).reshaped(1, L, h, d).transposed(0, 2, 1, 3)
            let (cosT, sinT) = rope.cosSin(offset: startPos, length: L)
            q = DeepseekV4Math.applyPartialRoPE(
                q, cos: cosT.expandedDimensions(axes: [0, 1]), sin: sinT.expandedDimensions(axes: [0, 1]),
                ropeDim: 8)
            let qv = q.transposed(0, 2, 1, 3)[0].asType(.bfloat16).asType(.float32).asArray(Float.self)
            let wv = (owner.weightsProj(x) * MLXArray(pow(Float(d), -0.5) * pow(Float(h), -0.5)))[0]
                .asType(.bfloat16).asType(.float32).asArray(Float.self)
            let kv = indexK[0].asType(.bfloat16).asType(.float32).asArray(Float.self)
            let ref = I.referenceScores(
                q: qv, k: kv, w: wv, C: L, h: h, d: d, T: T, startPos: startPos, ratio: 1)
            let flat = idx.asArray(Int32.self)
            let k = idx.dim(-1)
            for c in 0..<L {
                let ids = Array(flat[(c * k) ..< ((c + 1) * k)])
                #expect(Set(ids).count == k, "query \(c): ids must be unique")
                let finite = ref[c].compactMap { $0 }.sorted(by: >)
                let kept = ids.compactMap { ref[c][Int($0)] }
                let want = min(k, finite.count)
                #expect(kept.count == want, "query \(c): kept \(kept.count) of \(want)")
                if want > 0 {
                    let threshold = finite[want - 1]
                    #expect(kept.allSatisfy { $0 >= threshold - 1e-4 * max(1, abs(threshold)) },
                        "query \(c): \(kept) vs threshold \(threshold)")
                }
            }
        }
    }

    @Test("microbenchmark S=2048 T=100k: matrix units vs fp32 fused",
        .enabled(if: mppAvailable && ProcessInfo.processInfo.environment["VMLX_DSV41_MPP_BENCH"] == "1"))
    func benchmark() throws {
        try Device.withDefaultDevice(.gpu) {
            let H = 32, D = 128
            let rounds = Int(ProcessInfo.processInfo.environment["VMLX_DSV41_MPP_BENCH_ROUNDS"] ?? "") ?? 3
            for (S, T) in [(2048, 25_000), (2048, 100_000)] {
                let q = (MLXRandom.normal([1, S, H, D], key: MLXRandom.key(1))).asType(.bfloat16)
                let k = (MLXRandom.normal([1, T, D], key: MLXRandom.key(2))).asType(.bfloat16)
                let w = MLXRandom.normal([1, S, H], key: MLXRandom.key(3)).asType(.bfloat16).asType(.float32)
                let qf = q.asType(.float32), kf = k.asType(.float32)
                eval(q, k, w, qf, kf)
                func medianMS(_ f: () -> MLXArray?) throws -> Double {
                    eval(try #require(f()))
                    var ts: [Double] = []
                    for _ in 0..<5 {
                        let t0 = DispatchTime.now().uptimeNanoseconds
                        eval(try #require(f()))
                        ts.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
                    }
                    return ts.sorted()[2]
                }
                var mpp: [Double] = [], fp32: [Double] = []
                for _ in 0..<rounds {
                    fp32.append(try medianMS {
                        DeepseekV41FusedIndexScores.poolIndexScores(q: qf, poolKeys: kf, headW: w, scale: 1)
                    })
                    mpp.append(try medianMS { M.poolIndexScores(q: q, poolKeys: k, headW: w, scale: 1) })
                    Memory.clearCache()
                }
                let speed = zip(fp32, mpp).map { $0 / $1 }.sorted()
                let tflops = Double(2 * S * T * H * D) / 1e9
                print(String(
                    format: "[mpp-bench] S=%d T=%d fp32 fused %.1f-%.1f ms, matrix units %.1f-%.1f ms, "
                        + "speedup %.2f-%.2fx, mpp %.0f-%.0f TFLOPS",
                    S, T, fp32.min()!, fp32.max()!, mpp.min()!, mpp.max()!, speed.first!, speed.last!,
                    tflops / mpp.max()!, tflops / mpp.min()!))
            }
        }
    }
}
