import Foundation
import MLX
import XCTest

@testable import MLXLLM

/// The two-launch GLM router against an fp32 reference of the graph (the bundle's router dtype is fp32): same
/// chosen experts per row (as sets) and the same normalized, scaled weights, at 288 experts (> 256: the scratch
/// sizing the reference once got wrong).
final class Glm5NextRouterKernelsTests: XCTestCase {
    func testMatchesFP32Reference() throws {
        try check(experts: 288, weightDType: .bfloat16, scaling: 2.5)  // GLM-5.3
    }

    func testMatchesFP32ReferenceNaiveShape() throws {
        try check(experts: 256, weightDType: .float32, scaling: 1)  // Naive-N0.5: fp32 router weight
    }

    private func check(experts: Int, weightDType: DType, scaling: Float) throws {
        MLXRandom.seed(7)
        let d = 4096, k = 8
        let w = (MLXRandom.normal([experts, d]) * 0.02).asType(weightDType)
        let bias = MLXRandom.normal([experts]) * 0.05
        for rows in [1, 8] {
            let x = MLXRandom.normal([1, rows, d]).asType(.bfloat16)
            let routed = try XCTUnwrap(SigmoidRouterKernels.route(
                x, weight: w, correctionBias: bias, topK: k, normalize: true, scaling: scaling))
            let logits = x.asType(.float32).matmul(w.asType(.float32).T)
            let scores = sigmoid(logits)
            let choice = scores + bias
            for r in 0 ..< rows {
                let c = choice[0, r].asArray(Float.self), s = scores[0, r].asArray(Float.self)
                let ref = Array(c.indices.sorted { c[$0] > c[$1] }.prefix(k))
                let got = routed.indices[0, r].asArray(UInt32.self).map(Int.init)
                XCTAssertEqual(Set(got), Set(ref), "row \(r)")
                let total = ref.reduce(Float(0)) { $0 + s[$1] }
                let gotW = routed.weights[0, r].asArray(Float.self)
                // The reference's fp32 matmul runs in TF32 under MLX defaults (10-bit mantissa); the kernel
                // accumulates in true fp32, so weights agree to ~1e-4 relative, not bit for bit.
                for (j, e) in got.enumerated() {
                    let want = s[e] / total * scaling
                    XCTAssertEqual(gotW[j], want, accuracy: abs(want) * 1e-4)
                }
            }
        }
    }
}

/// Diagnostic (JANGT_PERF=1): does a following ordinary kernel slow down after the router kernels ran?
final class Glm5NextRouterAfterEffectTests: XCTestCase {
    func testFollowingKernelTiming() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["JANGT_PERF"] == "1")
        let d = 4096
        let w = (MLXRandom.normal([288, d]) * 0.02).asType(.bfloat16)
        let bias = MLXRandom.normal([288]) * 0.05
        let big = MLXRandom.normal([8192, d]).asType(.bfloat16)
        let (bq, bs, bb) = MLX.quantized(big, groupSize: 64, bits: 4)
        let x = MLXRandom.normal([1, 1, d]).asType(.bfloat16)
        func mm() -> MLXArray { quantizedMatmul(x, bq, scales: bs, biases: bb, transpose: true, groupSize: 64, bits: 4) }
        func time(_ label: String, _ body: () -> Void) {
            for _ in 0 ..< 5 { body() }
            let t0 = Date(); for _ in 0 ..< 50 { body() }
            print(String(format: "AFTER_EFFECT %@ %.1f us", label, Date().timeIntervalSince(t0) / 50 * 1e6))
        }
        time("matmul alone") { eval(mm()) }
        time("router then matmul (separate evals)") {
            let r = SigmoidRouterKernels.route(x, weight: w, correctionBias: bias, topK: 8, normalize: true, scaling: 2.5)!
            eval(r.indices, r.weights); eval(mm())
        }
        time("router alone") {
            let r = SigmoidRouterKernels.route(x, weight: w, correctionBias: bias, topK: 8, normalize: true, scaling: 2.5)!
            eval(r.indices, r.weights)
        }
        time("matmul alone again") { eval(mm()) }
    }
}
