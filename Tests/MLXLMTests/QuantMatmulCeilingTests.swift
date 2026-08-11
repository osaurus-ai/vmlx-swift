import Foundation
import MLX
import Testing

/// Muse Glimmer decode runs at ~245 GB/s against a machine capable of far more,
/// and neither compiled decode nor removing per-token tensor churn moved it.
/// That points at the quantized matmul itself rather than the code around it.
///
/// This times a bare 4-bit matrix-vector product at the model's own MLP shape,
/// with nothing else in the graph. Whatever bandwidth this reaches is the
/// ceiling the whole decode path inherits — if it lands near the model's
/// measured rate, no amount of fusion around it will reach the 25 tok/s gate.
@Suite("Quantized matmul ceiling")
struct QuantMatmulCeilingTests {

    static var enabled: Bool { ProcessInfo.processInfo.environment["MUSE_PERF"] == "1" }

    @Test("bare 4-bit matvec bandwidth at the model's MLP shape", .enabled(if: enabled))
    func matvecCeiling() throws {
        let hidden = 6656
        let inter = 19968
        let bits = 4
        let groupSize = 64

        // One MLP projection's worth of weights.
        let w = MLXRandom.normal([inter, hidden])
        let (wq, scales, biases) = MLX.quantized(w, groupSize: groupSize, bits: bits)
        let x = MLXRandom.normal([1, hidden])
        eval(wq, scales, biases, x)

        func run(_ n: Int) -> Double {
            let start = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< n {
                let y = MLX.quantizedMatmul(
                    x, wq, scales: scales, biases: biases,
                    transpose: true, groupSize: groupSize, bits: bits)
                eval(y)
            }
            return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        }
        _ = run(10)

        let iters = 200
        let elapsed = run(iters)
        let perCall = elapsed / Double(iters)
        // 4-bit weights plus fp16 scales/biases per group.
        let bytes = Double(inter * hidden) * 0.5
            + Double(inter * hidden / groupSize) * 2 * 2
        let gbps = bytes / perCall / 1e9
        print("[matvec] shape \(inter)x\(hidden) 4-bit: \(String(format: "%.3f", perCall * 1000)) ms/call")
        print("[matvec] achieved \(String(format: "%.0f", gbps)) GB/s on a single projection")
        // A single call is partly launch-bound, so also time a chain of 12 —
        // roughly what one decoder layer issues — which is closer to how the
        // real forward pass keeps the GPU fed.
        let chain = 12
        let startChain = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< 40 {
            var acc: MLXArray? = nil
            for _ in 0 ..< chain {
                let y = MLX.quantizedMatmul(
                    x, wq, scales: scales, biases: biases,
                    transpose: true, groupSize: groupSize, bits: bits)
                acc = acc.map { $0 + y[0..., 0 ..< hidden] } ?? y[0..., 0 ..< hidden]
            }
            eval(acc!)
        }
        let chainElapsed = Double(DispatchTime.now().uptimeNanoseconds - startChain) / 1e9
        let perChainCall = chainElapsed / Double(40 * chain)
        print("[matvec] chained x\(chain): \(String(format: "%.3f", perChainCall * 1000)) ms/call -> \(String(format: "%.0f", bytes / perChainCall / 1e9)) GB/s")
        print("[matvec] full model decode measured ~245 GB/s")

        // Same shape, unquantized. If fp16 reaches far more than the 4-bit
        // path, the gap is dequantization cost rather than the machine's
        // bandwidth — which is the difference between "buy a faster Mac" and
        // "write a faster kernel".
        let w16 = w.asType(.float16)
        let x16 = x.asType(.float16)
        eval(w16, x16)
        func runDense(_ n: Int) -> Double {
            let start = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< n {
                let y = x16.matmul(w16.transposed())
                eval(y)
            }
            return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        }
        _ = runDense(5)
        let denseIters = 60
        let denseElapsed = runDense(denseIters)
        let densePer = denseElapsed / Double(denseIters)
        let denseBytes = Double(inter * hidden) * 2
        let denseGbps = denseBytes / densePer / 1e9
        print("[matvec] fp16 same shape: \(String(format: "%.3f", densePer * 1000)) ms/call -> \(String(format: "%.0f", denseGbps)) GB/s")
        print("[matvec] 4-bit reaches \(String(format: "%.0f", 100 * gbps / denseGbps))% of the fp16 bandwidth")

        #expect(gbps > 0)
    }
}
