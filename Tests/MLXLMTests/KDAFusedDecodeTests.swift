import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// The one-dispatch KDA decode step against the stock multi-dispatch path of the same layer, from the same primed
/// cache: output, recurrent state and conv state.
final class KDAFusedDecodeTests: XCTestCase {
    // GLM-5.3 geometry by default (64 heads x 128, hidden 4096); JANGT_PERF_SMALL=1 for a 4-head layer.
    private let hidden = ProcessInfo.processInfo.environment["JANGT_PERF_SMALL"] == "1" ? 512 : 4096
    private var heads: Int { hidden == 512 ? 4 : 64 }

    private func layer() -> KDAAttention {
        MLXRandom.seed(11)
        let kda = KDAAttention(
            hiddenSize: hidden, numHeads: heads, headDim: 128, convKernelSize: 4, safeGate: true, lowerBound: -5,
            useLoRAGates: true, rmsNormEps: 1e-5)
        // Random weights at a realistic scale (the module initialises some parameters to zero).
        var params: [String: MLXArray] = [:]
        for (key, value) in kda.parameters().flattened() {
            params[key] = (MLXRandom.normal(value.shape) * 0.05).asType(.bfloat16)
        }
        params["A_log"] = MLXRandom.normal([heads]) * 0.3
        params["dt_bias"] = MLXRandom.normal([heads * 128]) * 0.3
        params["o_norm.weight"] = MLXArray.ones([128]).asType(.bfloat16)
        kda.update(parameters: ModuleParameters.unflattened(params))
        eval(kda)
        return kda
    }

    private lazy var prompt = MLXRandom.normal([1, 6, hidden], key: MLXRandom.key(5)).asType(.bfloat16)

    private func primed(_ kda: KDAAttention) -> MambaCache {
        let cache = MambaCache()
        eval(kda(prompt, cache: cache))
        eval(cache[0]!, cache[1]!)
        return cache
    }

    func testFusedDecodeMatchesStockStep() throws {
        let kda = layer()
        let x = MLXRandom.normal([1, 1, hidden]).asType(.bfloat16)
        let a = primed(kda), b = primed(kda)
        XCTAssertTrue(KDAFusedDecode.enabled)
        let fused = kda(x, cache: a)
        kda.fusedDecodeForTesting = false
        let stock = kda(x, cache: b)
        kda.fusedDecodeForTesting = true
        eval(fused, stock, a[0]!, a[1]!, b[0]!, b[1]!)
        func rel(_ p: MLXArray, _ q: MLXArray) -> Float {
            let d = (p.asType(.float32) - q.asType(.float32)).square().sum().item(Float.self)
            let n = q.asType(.float32).square().sum().item(Float.self)
            return (d / max(n, 1e-30)).squareRoot()
        }
        let outErr = rel(fused, stock), stateErr = rel(a[1]!, b[1]!), convErr = rel(a[0]!, b[0]!)
        print("KDA fused vs stock: out rel \(outErr) state rel \(stateErr) conv rel \(convErr)")
        XCTAssertEqual(a.offset, b.offset)
        XCTAssertEqual(convErr, 0)
        XCTAssertLessThan(stateErr, 1e-2)
        XCTAssertLessThan(outErr, 2e-2)
    }

    func testFusedDecodeSpeed() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["JANGT_PERF"] == "1")
        let kda = layer()
        let x = MLXRandom.normal([1, 1, hidden]).asType(.bfloat16)
        for fusedOn in [true, false, true, false] {
            kda.fusedDecodeForTesting = fusedOn
            let cache = primed(kda)
            for _ in 0 ..< 5 { eval(kda(x, cache: cache)) }
            let n = 100, t0 = Date()
            var y = x
            for _ in 0 ..< n { y = kda(y, cache: cache) }
            eval(y, cache[1]!)
            print(String(format: "KDA_PERF fused=%d %.1f us/step (100 chained)", fusedOn ? 1 : 0,
                         Date().timeIntervalSince(t0) / Double(n) * 1e6))
        }
    }
}
