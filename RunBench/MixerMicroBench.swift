// BENCH_MIXER_MICRO=1 — hyper-connection mixer projections at the Allosaurus 0.2 shapes (8-bit, g64, f16 metadata,
// bf16 activations), 48 DISTINCT weights per pass (a decode step's mixers do not share L2): down+inject N=324 x
// K=10240 (fused) vs split N=320 + N=4, and up N=10240 x K=320.

import Foundation
import MLX
import MLXLMCommon
import MLXRandom

func runMixerMicroBench() {
    struct Q { let w: MLXArray; let s: MLXArray; let b: MLXArray }
    func make(_ n: Int, _ k: Int) -> [Q] {
        (0 ..< 48).map { _ in
            let w = MLXRandom.normal([n, k]).asType(.bfloat16) * 0.02
            let (wq, s, b) = quantized(w, groupSize: 64, bits: 8)
            let q = Q(w: wq, s: s.asType(.float16), b: b!.asType(.float16))
            MLX.eval(q.w, q.s, q.b)
            return q
        }
    }
    func time(_ label: String, _ body: (Int) -> MLXArray) {
        for _ in 0 ..< 5 { MLX.eval((0 ..< 48).map(body)) }
        var t: [Double] = []
        for _ in 0 ..< 40 {
            let s = CFAbsoluteTimeGetCurrent()
            MLX.eval((0 ..< 48).map(body))
            t.append(CFAbsoluteTimeGetCurrent() - s)
        }
        t.sort()
        print(String(format: "[MIXERMICRO] %@ x48 median_ms=%.3f per_call_us=%.1f", label, t[t.count / 2] * 1000,
                     t[t.count / 2] * 1e6 / 48))
    }
    func mm(_ x: MLXArray, _ q: Q) -> MLXArray {
        quantizedMM(x, q.w, scales: q.s, biases: q.b, transpose: true, groupSize: 64, bits: 8, mode: .affine)
            .asType(.bfloat16)
    }
    let k = 10240
    let fused = make(324, k)
    let down = fused.map { Q(w: $0.w[0 ..< 320], s: $0.s[0 ..< 320], b: $0.b[0 ..< 320]) }
    let inject = fused.map { Q(w: $0.w[320 ..< 324], s: $0.s[320 ..< 324], b: $0.b[320 ..< 324]) }
    MLX.eval(down.map(\.w), inject.map(\.w))
    let x = MLXRandom.normal([1, 1, k]).asType(.bfloat16)
    MLX.eval(x)
    time("down+inject fused324 bf16affine_dense") {
        Qwen4ExpBF16Affine.dense(x, fused[$0].w, scales: fused[$0].s, biases: fused[$0].b, groupSize: 64, bits: 8,
                                 mode: .affine)
    }
    time("down+inject fused324 quantizedMM") { mm(x, fused[$0]) }
    time("down320 mixed + inject4 quantizedMM") { concatenated([mm(x, down[$0]), mm(x, inject[$0])], axis: -1) }
    let up = make(10240, 320)
    let h = MLXRandom.normal([1, 1, 320]).asType(.bfloat16)
    MLX.eval(h)
    time("up10240 bf16affine_dense") {
        Qwen4ExpBF16Affine.dense(h, up[$0].w, scales: up[$0].s, biases: up[$0].b, groupSize: 64, bits: 8, mode: .affine)
    }
    time("up10240 quantizedMM") { mm(h, up[$0]) }
}
