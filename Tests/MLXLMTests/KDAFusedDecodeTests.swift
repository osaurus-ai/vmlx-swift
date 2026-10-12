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

    private func rel(_ p: MLXArray, _ q: MLXArray) -> Float {
        let d = (p.asType(.float32) - q.asType(.float32)).square().sum().item(Float.self)
        let n = q.asType(.float32).square().sum().item(Float.self)
        return (d / max(n, 1e-30)).squareRoot()
    }

    /// Multi-row verify (input capture) against the stock multi-row path of the same layer and cache.
    func testFusedRowsMatchStockRows() throws {
        let kda = layer()
        for rows in [2, 5, 8, 16] {
            let x = MLXRandom.normal([1, rows, hidden], key: MLXRandom.key(UInt64(40 + rows))).asType(.bfloat16)
            let a = primed(kda), b = primed(kda)
            let fused = NativeMTPVerifierStatePolicy.withVerifierMode("input_capture") { kda(x, cache: a) }
            XCTAssertEqual(a.verifyInputStash?.arrays.count, 3, "rows \(rows) took the fused path")
            kda.fusedDecodeForTesting = false
            let stock = NativeMTPVerifierStatePolicy.withVerifierMode("input_capture") { kda(x, cache: b) }
            kda.fusedDecodeForTesting = true
            eval(fused, stock, a[0]!, a[1]!, b[0]!, b[1]!)
            let outErr = rel(fused, stock), stateErr = rel(a[1]!, b[1]!), convErr = rel(a[0]!, b[0]!)
            print("KDA fused rows=\(rows) vs stock: out rel \(outErr) state rel \(stateErr) conv rel \(convErr)")
            XCTAssertEqual(a.offset, b.offset)
            XCTAssertEqual(convErr, 0)
            XCTAssertLessThan(stateErr, 1e-2)
            XCTAssertLessThan(outErr, 2e-2)
        }
    }

    /// The multi-row kernel IS the one-token kernel applied row after row: bit-identical outputs and state, so a
    /// verify's rows match decode and the rollback replay of n rows reproduces the state after row n exactly.
    func testFusedRowsEqualChainedDecodeKernel() throws {
        let kda = layer()
        let c = heads * 128, rows = 8
        MLXRandom.seed(3)
        let mixed = (MLXRandom.normal([1, rows, 3 * c]) * 0.5).asType(.bfloat16)
        let fRaw = MLXRandom.normal([1, rows, c]).asType(.bfloat16)
        let gateRaw = MLXRandom.normal([1, rows, c]).asType(.bfloat16)
        let bRaw = MLXRandom.normal([1, rows, heads]).asType(.bfloat16)
        let cache = primed(kda)
        let conv0 = cache[0]!, state0 = cache[1]!
        func many(_ n: Int) -> (gated: MLXArray, state: MLXArray, conv: MLXArray) {
            KDAFusedDecode.runRows(
                mixed: mixed[0..., ..<n], convState: conv0, qWeight: kda.qConv.weight, kWeight: kda.kConv.weight,
                vWeight: kda.vConv.weight, fRaw: fRaw[0..., ..<n], dtBias: kda.dtBias, aLog: kda.aLog,
                bRaw: bRaw[0..., ..<n], gateRaw: gateRaw[0..., ..<n], oNormWeight: kda.oNorm.weight,
                eps: kda.oNorm.eps, state: state0, heads: heads, keyDim: 128, lowerBound: -5)!
        }
        var conv = conv0, state = state0
        var outs: [MLXArray] = []
        var states: [MLXArray] = []
        for r in 0 ..< rows {
            let one = KDAFusedDecode.run(
                mixed: mixed[0..., r ..< (r + 1)], convState: conv, qWeight: kda.qConv.weight,
                kWeight: kda.kConv.weight, vWeight: kda.vConv.weight, fRaw: fRaw[0..., r ..< (r + 1)],
                dtBias: kda.dtBias, aLog: kda.aLog, bRaw: bRaw[0..., r ..< (r + 1)],
                gateRaw: gateRaw[0..., r ..< (r + 1)], oNormWeight: kda.oNorm.weight, eps: kda.oNorm.eps,
                state: state, heads: heads, keyDim: 128, lowerBound: -5)!
            outs.append(one.gated); states.append(one.state)
            conv = one.conv; state = one.state
        }
        let all = many(rows)
        let chained = concatenated(outs, axis: 1)
        eval(all.gated, all.state, all.conv, chained, state, conv)
        XCTAssertEqual(abs(all.gated.asType(.float32) - chained.asType(.float32)).max().item(Float.self), 0)
        XCTAssertEqual(abs(all.state - state).max().item(Float.self), 0)
        XCTAssertEqual(abs(all.conv.asType(.float32) - conv.asType(.float32)).max().item(Float.self), 0)
        // Rollback replay: the first n rows give the state after row n - 1 of the chain.
        for n in [1, 3, 7] {
            let part = many(n)
            eval(part.state)
            XCTAssertEqual(abs(part.state - states[n - 1]).max().item(Float.self), 0, "replay n=\(n)")
        }
    }

    /// Verify 8 rows, accept 3: the committed cache equals a cache that only ever saw those 3 rows (the projections
    /// run at 8 vs 3 rows, which may pick different matmul kernels, so this compares within tolerance; the kernel
    /// replay itself is bit-exact, `testFusedRowsEqualChainedDecodeKernel`).
    func testFusedRowsRollback() throws {
        let kda = layer()
        let x = MLXRandom.normal([1, 8, hidden], key: MLXRandom.key(77)).asType(.bfloat16)
        let a = primed(kda), b = primed(kda)
        let base = a.offset
        _ = NativeMTPVerifierStatePolicy.withVerifierMode("input_capture") { kda(x, cache: a) }
        XCTAssertTrue(kda.commitVerifyStash(cache: a, acceptedInputs: 3))
        _ = NativeMTPVerifierStatePolicy.withVerifierMode("input_capture") { kda(x[0..., ..<3], cache: b) }
        eval(a[0]!, a[1]!, b[0]!, b[1]!)
        XCTAssertEqual(a.offset, base + 3)
        XCTAssertEqual(b.offset, base + 3)
        let stateErr = rel(a[1]!, b[1]!), convErr = rel(a[0]!, b[0]!)
        print("KDA fused rollback 8 -> 3 vs 3 rows: state rel \(stateErr) conv rel \(convErr)")
        XCTAssertLessThan(stateErr, 1e-3)
        XCTAssertLessThan(convErr, 1e-2)
        XCTAssertNil(a.verifyInputStash)
    }

    /// The stock multi-row path's stash (fused path off) rolls back the same way.
    func testStockRowsRollback() throws {
        let kda = layer()
        kda.fusedDecodeForTesting = false
        defer { kda.fusedDecodeForTesting = true }
        let x = MLXRandom.normal([1, 8, hidden], key: MLXRandom.key(78)).asType(.bfloat16)
        let a = primed(kda), b = primed(kda)
        let base = a.offset
        _ = NativeMTPVerifierStatePolicy.withVerifierMode("input_capture") { kda(x, cache: a) }
        XCTAssertEqual(a.verifyInputStash?.arrays.count, 6)
        XCTAssertTrue(kda.commitVerifyStash(cache: a, acceptedInputs: 3))
        _ = kda(x[0..., ..<3], cache: b)
        eval(a[0]!, a[1]!, b[0]!, b[1]!)
        let stateErr = rel(a[1]!, b[1]!), convErr = rel(a[0]!, b[0]!)
        print("KDA stock rollback 8 -> 3 vs 3 rows: state rel \(stateErr) conv rel \(convErr)")
        XCTAssertEqual(a.offset, base + 3)
        XCTAssertLessThan(stateErr, 1e-3)
        XCTAssertLessThan(convErr, 1e-2)
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
