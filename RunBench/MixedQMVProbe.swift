// BENCH_MIXED_QMV_PROBE=1 — bf16 activations against fp16 scales/biases (JANG bundles): compares the 1-row
// mixed qmv the AR step runs with the multi-row forms a verify window can use (plain M rows, batched M = 1 per
// row, per-row 1-row calls, RowExactAffineQMV). Prints max |diff| vs the per-row 1-row result.

import Foundation
import MLX
import MLXLMCommon
import MLXRandom

func runMixedQMVProbe() {
    for bits in [4, 8] {
        for (k, n) in [(2560, 12288), (2560, 512), (6144, 2560), (2560, 2560)] {
            let w = MLXRandom.normal([n, k]).asType(.bfloat16)
            let (wq, s, b) = quantized(w, groupSize: 64, bits: bits)
            let scales = s.asType(.float16), biases = b!.asType(.float16)
            let rows = 3
            let x = (MLXRandom.normal([rows, k]) * 0.5).asType(.bfloat16)
            func mm(_ input: MLXArray, _ weight: MLXArray, _ sc: MLXArray, _ bi: MLXArray) -> MLXArray {
                quantizedMM(input, weight, scales: sc, biases: bi, transpose: true, groupSize: 64, bits: bits,
                            mode: .affine)
            }
            let perRow = concatenated((0 ..< rows).map {
                mm(x[$0 ..< ($0 + 1)].reshaped(1, 1, k), wq, scales, biases).reshaped(1, n).asType(.bfloat16)
            }, axis: 0)
            let plain = mm(x.reshaped(1, rows, k), wq, scales, biases).reshaped(rows, n).asType(.bfloat16)
            let batched = mm(x.reshaped(rows, 1, k), wq[.newAxis], scales[.newAxis], biases[.newAxis])
                .reshaped(rows, n).asType(.bfloat16)
            let exact = RowExactAffineQMV.project(
                x, weight: wq, scales: scales, biases: biases, groupSize: 64, bits: bits,
                reference: { input in
                    concatenated((0 ..< input.dim(0)).map {
                        mm(input[$0 ..< ($0 + 1)].reshaped(1, 1, k), wq, scales, biases).reshaped(1, n)
                            .asType(.bfloat16)
                    }, axis: 0)
                })
            let ref = matmul(x.asType(.float32), dequantized(wq, scales: scales.asType(.float32),
                biases: biases.asType(.float32), groupSize: 64, bits: bits).asType(.float32).T)
            func e(_ a: MLXArray) -> String {
                String(format: "%.4g", abs(a.asType(.float32) - ref).mean().item(Float.self))
            }
            print("[MIXEDQMV] err-vs-fp32 bits=\(bits) K=\(k) N=\(n) perRow=\(e(perRow)) batched=\(e(batched)) ref_rms=\(String(format: "%.3g", sqrt((ref * ref).mean()).item(Float.self)))")
            func d(_ a: MLXArray?) -> String {
                guard let a else { return "n/a" }
                return String(format: "%.4g", abs(a.asType(.float32) - perRow.asType(.float32)).max().item(Float.self))
            }
            print("[MIXEDQMV] bits=\(bits) K=\(k) N=\(n) rows=\(rows) plain=\(d(plain)) batched=\(d(batched)) rowexact=\(d(exact))")
        }
    }
}
