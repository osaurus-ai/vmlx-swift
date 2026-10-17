// BENCH_KERNEL_MICRO=1 — synthetic-weight timing of the row-exact verify kernels at Nemotron-3.5
// Super-VL shapes (no model load). For each shape: MLX's single-row kernel (R = 1), the batched
// M = 1-per-row reference, and the multi-row kernel at R = 2, 4, 8; median of BENCH_MICRO_REPS
// chained launches. Diagnostic only.

import Foundation
import MLX
import MLXLMCommon

func runKernelMicroBench() {
    let env = ProcessInfo.processInfo.environment
    let reps = Int(env["BENCH_MICRO_REPS"] ?? "40") ?? 40
    func time(_ body: () -> MLXArray) -> Double {
        MLX.eval(body())
        var samples: [Double] = []
        for _ in 0 ..< 7 {
            let t0 = CFAbsoluteTimeGetCurrent()
            var outs: [MLXArray] = []
            for _ in 0 ..< reps { outs.append(body()) }
            MLX.eval(outs)
            samples.append((CFAbsoluteTimeGetCurrent() - t0) / Double(reps))
        }
        samples.sort()
        return samples[3] * 1000
    }
    let dense: [(Int, Int, Int)] = [
        (4096, 18560, 8), (8192, 4096, 8), (4096, 5376, 8), (5376, 4096, 8), (4096, 131072, 8),
        (4096, 18560, 4),
    ]
    for (k, n, bits) in dense {
        let wf = MLXRandom.normal([n, k]).asType(.bfloat16) * 0.02
        let (w, s, b) = MLX.quantized(wf, groupSize: 64, bits: bits)
        let bytes = Double(w.nbytes + s.nbytes + (b?.nbytes ?? 0)) / 1e9
        func single(_ x: MLXArray) -> MLXArray {
            quantizedMM(x, w, scales: s, biases: b, transpose: true, groupSize: 64, bits: bits)
        }
        func batched(_ x: MLXArray) -> MLXArray {
            quantizedMM(
                x.reshaped(x.dim(0), 1, k), w[.newAxis], scales: s[.newAxis], biases: b?[.newAxis],
                transpose: true, groupSize: 64, bits: bits
            ).reshaped(x.dim(0), n)
        }
        let x1 = MLXRandom.normal([1, k]).asType(.bfloat16)
        let t1 = time { single(x1) }
        var line = String(format: "[MICRO] qmv K=%d N=%d b%d %.2f GB  R1 %.3f ms (%.0f GB/s)", k, n, bits, bytes, t1, bytes / t1 * 1000)
        for r in [2, 4, 8] {
            let x = MLXRandom.normal([r, k]).asType(.bfloat16)
            let tb = time { batched(x) }
            let tk = time {
                RowExactAffineQMV.project(
                    x, weight: w, scales: s, biases: b, groupSize: 64, bits: bits, reference: batched)
                    ?? batched(x)
            }
            let tw = time { single(x) }
            line += String(format: " | R%d batched %.2fx rows %.2fx wide %.2fx", r, tb / t1, tk / t1, tw / t1)
        }
        print(line)
    }
    // Routed experts (latent 1024 -> 2688 -> 1024, 4-bit, 512 experts, top-22).
    let e = 512
    let topK = 22
    for (k, n, div) in [(1024, 2688, topK), (2688, 1024, 1)] {
        let wf = MLXRandom.normal([e, n, k]).asType(.bfloat16) * 0.02
        let (w, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4)
        func gatherRef(_ x: MLXArray, _ idx: MLXArray) -> MLXArray {
            let pairs = idx.dim(0)
            let rowOfPair = MLXArray(Array(0 ..< pairs).map { Int32($0 / div) })
            return gatherQuantizedMM(
                x[rowOfPair].reshaped(pairs, 1, k), w, scales: s, biases: b, rhsIndices: idx,
                transpose: true, groupSize: 64, bits: 4
            ).reshaped(pairs, n)
        }
        var line = String(format: "[MICRO] gather K=%d N=%d", k, n)
        var t1 = 0.0
        for r in [1, 2, 4, 8] {
            // Realistic overlap: each row keeps ~60% of the previous row's experts.
            var rowsIdx: [[Int32]] = []
            var prev = Array((0 ..< e).shuffled().prefix(topK)).map(Int32.init)
            for _ in 0 ..< r {
                rowsIdx.append(prev)
                let keep = Array(prev.shuffled().prefix(topK * 6 / 10))
                let fresh = (0 ..< e).map(Int32.init).filter { !keep.contains($0) }.shuffled().prefix(topK - keep.count)
                prev = keep + fresh
            }
            let idx = MLXArray(rowsIdx.flatMap { $0 }).asType(.uint32)
            let xRows = div == 1 ? r * topK : r
            let x = MLXRandom.normal([xRows, k]).asType(.bfloat16)
            let tr = time { gatherRef(x, idx) }
            if r == 1 { t1 = tr; line += String(format: " R1 %.3f ms", tr); continue }
            let td = time {
                RowExactGatherQMV.gather(
                    x, indices: idx, xDivisor: div, maxRun: r, weight: w, scales: s, biases: b!,
                    groupSize: 64, bits: 4) ?? gatherRef(x, idx)
            }
            line += String(format: " | R%d gather %.2fx dedup %.2fx", r, tr / t1, td / t1)
        }
        print(line)
    }
}
