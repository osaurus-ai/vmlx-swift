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
}
