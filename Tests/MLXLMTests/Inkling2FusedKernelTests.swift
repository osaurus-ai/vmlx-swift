import MLX
import XCTest

@testable import MLXLLM

/// The fused Inkling-2 loop message-gate kernels (`gate_down` / `gate_mix` / `gate_norm`) against
/// the reference arithmetic written with plain MLX ops:
///   gin = [rms(base), rms(msg)] (fp32) -> down dtype -> down_proj -> fp32
///   y = (1 - tanh(gin_d @ Wf^T)) * base + tanh(gin_d @ Wg^T) * msg            (fp32)
///   out = rms_norm(y, layer weight) in the model dtype
/// Summation order differs (kernel vs GEMV), so a few elements may land one bf16 step apart;
/// anything beyond that is a defect.
final class Inkling2FusedKernelTests: XCTestCase {
    private func values(_ shape: [Int], seed: UInt64, scale: Float) -> MLXArray {
        MLXRandom.normal(shape, key: MLXRandom.key(seed)) * scale
    }

    private func referenceGate(
        base: MLXArray, msg: MLXArray, down: MLXArray, wf: MLXArray, wg: MLXArray,
        norm: MLXArray, eps: Float
    ) -> MLXArray {
        let ones = MLXArray.ones([base.dim(-1)], dtype: .float32)
        let gin = MLX.concatenated(
            [
                MLXFast.rmsNorm(base, weight: ones, eps: eps),
                MLXFast.rmsNorm(msg, weight: ones, eps: eps),
            ], axis: -1
        ).asType(down.dtype)
        let d = MLX.matmul(gin, down.T).asType(.float32)
        let forget = MLX.tanh(MLX.matmul(d, wf.asType(.float32).T))
        let gain = MLX.tanh(MLX.matmul(d, wg.asType(.float32).T))
        let y = (1 - forget) * base + gain * msg
        return MLXFast.rmsNorm(y, weight: norm, eps: eps).asType(down.dtype)
    }

    private func check(gates G: Int, rows T: Int, hidden H: Int, rank R: Int) {
        let eps: Float = 1e-6
        let dtype = DType.bfloat16
        // x: one base row per (t, g) (NX = G); msgs: MS = 4 message rows per t, gates at mrow 0.
        let x = values([T * G, H], seed: 1, scale: 1.3)
        let msgs = values([T * 4, H], seed: 2, scale: 0.8)
        let down = values([G, R, 2 * H], seed: 3, scale: 0.05).asType(dtype)
        let wf = values([G, H, R], seed: 4, scale: 0.2).asType(dtype)
        let wg = values([G, H, R], seed: 5, scale: 0.2).asType(dtype)
        let norm = (1 + values([G, H], seed: 6, scale: 0.1)).asType(.float32)
        let fg = MLX.contiguous(
            MLX.stacked(
                (0 ..< G).map { MLX.stacked([wf[$0].T, wg[$0].T], axis: 0) }, axis: 0))
        let meta = MLXArray([Int32(0), Int32(G), Int32(T), Int32(4)])
        let (out, out32) = Inkling2DecodeKernels.messageGates(
            x: x, msgs: msgs, down: down, fg: fg, norm: norm, normEps: MLXArray([eps]),
            gateEps: MLXArray([eps]), meta: meta, gates: G, rows: T, dtype: dtype)
        eval(out, out32)
        XCTAssertEqual(out.shape, [G * T, H])

        // kernel row r = g * T + t  <->  base x[t * G + g], message msgs[t * 4 + g]
        var maxAbs: Float = 0
        var mismatched = 0
        for g in 0 ..< G {
            for t in 0 ..< T {
                let ref = referenceGate(
                    base: x[(t * G + g) ..< (t * G + g + 1)],
                    msg: msgs[(t * 4 + g) ..< (t * 4 + g + 1)], down: down[g], wf: wf[g],
                    wg: wg[g], norm: norm[g], eps: eps
                ).asType(.float32)
                // fail closed: a zero reference would make any comparison vacuous
                XCTAssertGreaterThan(MLX.abs(ref).mean().item(Float.self), 0.1)
                let got = out[(g * T + t) ..< (g * T + t + 1)].asType(.float32)
                let diff = MLX.abs(got - ref)
                maxAbs = max(maxAbs, diff.max().item(Float.self))
                // one bf16 step at |v| ~ 4 is 2^-5; count anything that differs at all
                mismatched += (diff .> 0).sum().item(Int.self)
                XCTAssertEqual(
                    MLX.abs(out32[(g * T + t) ..< (g * T + t + 1)] - got).max().item(Float.self),
                    0, "out32 must be the fp32 of out")
            }
        }
        let total = G * T * H
        print(
            "[Inkling2FusedKernelTests] G=\(G) T=\(T) H=\(H): max |diff| \(maxAbs), "
                + "differing \(mismatched)/\(total)")
        XCTAssertLessThanOrEqual(maxAbs, 0.07, "gate output off by more than ~1 bf16 step")
        XCTAssertLessThan(Double(mismatched) / Double(total), 0.02, "too many differing elements")
        XCTAssertGreaterThan(total, 0)
    }

    func testDecodeQKVGatesMatchReference() { check(gates: 3, rows: 1, hidden: 512, rank: 64) }
    func testVerifyQKVGatesMatchReference() { check(gates: 3, rows: 5, hidden: 512, rank: 64) }
    func testMLPGateMatchesReference() { check(gates: 1, rows: 3, hidden: 2304, rank: 64) }

    /// The batch-1 NVFP4 expert kernels against MLX's own `gatherQuantizedMM` (mode .nvfp4) plus
    /// the reference activation; T = 2 rows x 8 picks, picks repeat an expert across rows.
    func testNVFP4ExpertKernelsMatchMLXGather() {
        let (E, H, I, K, T) = (16, 256, 128, 8, 2)
        let wgu = values([E, 2 * I, H], seed: 11, scale: 0.05)
        let wd = values([E, H, I], seed: 12, scale: 0.05)
        let qgu = MLX.quantized(wgu, groupSize: 16, bits: 4, mode: .nvfp4)
        let qd = MLX.quantized(wd, groupSize: 16, bits: 4, mode: .nvfp4)
        let x = values([T, H], seed: 13, scale: 1.0)
        let inds = MLXArray([0, 3, 5, 7, 9, 11, 13, 15, 3, 2, 4, 6, 8, 10, 12, 14] as [UInt32])
            .reshaped([T, K])
        let es = (1 + values([E], seed: 14, scale: 0.2)).asType(.float32)
        let caps = MLXArray([Float(4), Float(25)])
        let (act, down) = Inkling2DecodeKernels.nvfp4Experts(
            x32: x, inds: inds, gateUpCodes: qgu.wq, gateUpScales: qgu.scales, downCodes: qd.wq,
            downScales: qd.scales, gateUpExpertScale: es, caps: caps, topK: K)

        let gu = MLX.gatherQuantizedMM(
            x.reshaped([T, 1, 1, H]), qgu.wq, scales: qgu.scales, biases: nil, rhsIndices: inds,
            transpose: true, groupSize: 16, bits: 4, mode: .nvfp4
        ).reshaped([T * K, 2 * I])
        let s = es[inds.reshaped([T * K])].reshaped([T * K, 1])
        let g = gu[0..., .stride(from: 0, to: 2 * I, by: 2)] * s
        let u = gu[0..., .stride(from: 1, to: 2 * I, by: 2)] * s
        let refAct = ((MLX.tanh(g / 4) * 4) * MLX.sigmoid(g) * (MLX.tanh(u / 25) * 25))
            .asType(.bfloat16).asType(.float32)
        let refDown = MLX.gatherQuantizedMM(
            act.reshaped([T, K, 1, I]), qd.wq, scales: qd.scales, biases: nil, rhsIndices: inds,
            transpose: true, groupSize: 16, bits: 4, mode: .nvfp4
        ).reshaped([T * K, H])
        eval(act, down, refAct, refDown)
        XCTAssertGreaterThan(MLX.abs(refAct).mean().item(Float.self), 0.01)
        XCTAssertGreaterThan(MLX.abs(refDown).mean().item(Float.self), 0.01)
        // activation: fp32 accumulation order differs from MLX's, then one bf16 rounding
        let actDiff = MLX.abs(act - refAct)
        let actBad = (actDiff .> (MLX.abs(refAct) * 0.008 + 1e-6)).sum().item(Int.self)
        let downRel = (MLX.abs(down - refDown).max() / MLX.abs(refDown).max()).item(Float.self)
        print(
            "[Inkling2FusedKernelTests] nvfp4 experts: act beyond 1 bf16 step \(actBad)/\(T * K * I), "
                + "down max rel \(downRel)")
        XCTAssertEqual(actBad, 0)
        XCTAssertLessThan(downRel, 1e-5)
    }

    /// Fused zero-mask FlashAttention prefill against reference `attend()` math for both
    /// sliding-window and full-attention (beyond `relExtent`) regimes with per-token sinks.
    func testPrefillAttentionMatchesReference() throws {
        let (B, NQ, NKV, D, DR, E) = (1, 24, 8, 128, 16, 64)
        let scale: Float = 1 / Float(D).squareRoot()
        for (seen, kStart, T, S, window) in [
            (128, 65, 32, 95, Optional(64)),
            (100, 0, 16, 116, Optional<Int>.none),
            (240, 0, 37, 277, Optional<Int>.none),
        ] {
            let q = values([B, NQ, T, D], seed: 21, scale: 0.5).asType(.bfloat16)
            let k = values([B, NKV, S, D], seed: 22, scale: 0.5).asType(.bfloat16)
            let v = values([B, NKV, S, D], seed: 23, scale: 0.5).asType(.bfloat16)
            let r = values([B, T, NQ, DR], seed: 24, scale: 0.5).asType(.bfloat16)
            let proj = values([DR, E], seed: 25, scale: 0.3).asType(.bfloat16)
            let projT = MLX.contiguous(proj.T)
            let sinks = values([B, T, NQ], seed: 26, scale: 0.8).asType(.bfloat16)

            guard let got = Inkling2Kernels.prefillAttention(
                q: q, k: k, v: v, r: r, projT: projT, sinks: sinks,
                seen: seen, kStart: kStart, window: window, relExtent: E, scale: scale)
            else {
                XCTFail("prefillAttention returned nil on GPU")
                return
            }

            // Reference: explicit mask + zero-key sink column (Inkling2Loop.swift).
            let qPos = MLXArray(Int32(seen) ..< Int32(seen + T))
            let kPos = MLXArray(Int32(kStart) ..< Int32(kStart + S))
            let d = qPos.expandedDimensions(axis: 1) - kPos.expandedDimensions(axis: 0)
            let idx = MLX.clip(d, min: 0, max: E - 1)
            let rel = MLX.matmul(r, proj).transposed(0, 2, 1, 3)
            let gathered = MLX.takeAlong(
                rel, MLX.broadcast(idx.reshaped([1, 1, T, S]), to: [B, NQ, T, S]), axis: -1)
            var valid = d .>= 0
            if let window { valid = valid .&& (d .< window) }
            let inBand = (d .>= 0) .&& (d .< E)
            let bias = MLX.where(inBand, gathered, MLXArray(Float(0)).asType(.bfloat16))
            let negInf = MLXArray(-Float.infinity).asType(.bfloat16)
            let additive = MLX.where(valid, bias, negInf)
            let sinkCol = sinks.transposed(0, 2, 1).expandedDimensions(axis: -1)
            let mask = MLX.concatenated([sinkCol, additive], axis: -1)
            let zk = MLXArray.zeros([B, NKV, 1, D], dtype: .bfloat16)
            let ref = MLXFast.scaledDotProductAttention(
                queries: q, keys: MLX.concatenated([zk, k], axis: 2),
                values: MLX.concatenated([zk, v], axis: 2),
                scale: scale, mask: .array(mask))
            eval(got, ref)
            let maxDiff = MLX.abs(got.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
            XCTAssertLessThan(maxDiff, 0.03, "window=\(String(describing: window)) maxDiff=\(maxDiff)")
        }
    }

    /// Zero-mask Split-K FlashDecoding (T=1 AR and T=8/12 DFlash 2 verify) matches reference
    /// attention across both far-past unmasked keys (d >= E) and relative-bias keys (d < E).
    func testDecodeSplitKAttentionMatchesReference() throws {
        let (B, NQ, NKV, D, DR, E) = (1, 24, 8, 128, 16, 64)
        let scale: Float = 1 / Float(D).squareRoot()
        for T in [1, 8, 12] {
            let seen = 1500
            let S = seen + T
            let q = values([B, NQ, T, D], seed: 31 + UInt64(T), scale: 0.4).asType(.bfloat16)
            let k = values([B, NKV, S, D], seed: 32 + UInt64(T), scale: 0.4).asType(.bfloat16)
            let v = values([B, NKV, S, D], seed: 33 + UInt64(T), scale: 0.4).asType(.bfloat16)
            let r = values([B, T, NQ, DR], seed: 34 + UInt64(T), scale: 0.4).asType(.bfloat16)
            let proj = values([DR, E], seed: 35 + UInt64(T), scale: 0.3).asType(.bfloat16)
            let projT = MLX.contiguous(proj.T)
            let sinks = values([B, T, NQ], seed: 36 + UInt64(T), scale: 0.8).asType(.bfloat16)

            guard let got = Inkling2Kernels.decodeSplitKAttention(
                q: q, k: k, v: v, r: r, projT: projT, sinks: sinks,
                seen: seen, kStart: 0, relExtent: E, scale: scale)
            else {
                XCTFail("decodeSplitKAttention returned nil on GPU for T=\(T)")
                return
            }

            let qPos = MLXArray(Int32(seen) ..< Int32(seen + T))
            let kPos = MLXArray(Int32(0) ..< Int32(S))
            let d = qPos.expandedDimensions(axis: 1) - kPos.expandedDimensions(axis: 0)
            let idx = MLX.clip(d, min: 0, max: E - 1)
            let rel = MLX.matmul(r, proj).transposed(0, 2, 1, 3)
            let gathered = MLX.takeAlong(
                rel, MLX.broadcast(idx.reshaped([1, 1, T, S]), to: [B, NQ, T, S]), axis: -1)
            let valid = d .>= 0
            let inBand = (d .>= 0) .&& (d .< E)
            let bias = MLX.where(inBand, gathered, MLXArray(Float(0)).asType(.bfloat16))
            let negInf = MLXArray(-Float.infinity).asType(.bfloat16)
            let additive = MLX.where(valid, bias, negInf)
            let sinkCol = sinks.transposed(0, 2, 1).expandedDimensions(axis: -1)
            let mask = MLX.concatenated([sinkCol, additive], axis: -1)
            let zk = MLXArray.zeros([B, NKV, 1, D], dtype: .bfloat16)
            let ref = MLXFast.scaledDotProductAttention(
                queries: q, keys: MLX.concatenated([zk, k], axis: 2),
                values: MLX.concatenated([zk, v], axis: 2),
                scale: scale, mask: .array(mask))
            eval(got, ref)
            let maxDiff = MLX.abs(got.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
            XCTAssertLessThan(maxDiff, 0.03, "splitK T=\(T) maxDiff=\(maxDiff)")
        }
    }
}
