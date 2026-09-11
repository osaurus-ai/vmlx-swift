import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import Testing

@testable import MLXVLM

@Suite("Flash GDN post-projection compile candidate", .serialized)
struct Qwen4ExpGDNPreprocessTests {
    @Test("unsupported dtype, width and nested trace retain native fallback")
    func rejectionBoundaries() throws {
        try MLXMetalTestLock.withLock {
            let input = MLXArray.zeros([1, 2, 128], dtype: .bfloat16)
            let weight = MLXArray.ones([128, 2, 1], dtype: .bfloat16)
            func call(_ x: MLXArray, _ w: MLXArray) -> [MLXArray]? {
                Qwen4ExpCompiledGDNInputs.callPreprocess(
                    convInput: x, convWeight: w,
                    numKHeads: 2, headKDim: 16, numVHeads: 4, headVDim: 16)
            }
            #expect(call(input.asType(.float32), weight) == nil)
            #expect(call(input.asType(.float16), weight.asType(.float16)) == nil)
            #expect(call(input, weight.asType(.float32)) == nil)
            #expect(call(MLXArray.zeros([1, 3, 128], dtype: .bfloat16), weight) == nil)
            #expect(call(MLXArray.zeros([1, 2, 127], dtype: .bfloat16), weight) == nil)
            #expect(call(input, MLXArray.ones([128, 2, 2], dtype: .bfloat16)) == nil)
            #expect(CompiledDecodeTrace.withActive { call(input, weight) } == nil)
        }
    }

    @Test("explicit layer geometry, weights and batch preserve native preprocessing")
    func parityAndTiming() throws {
        try MLXMetalTestLock.withLock {
            // Tiny geometry plus actual Flash head geometry; no projection
            // matrices, model files or model-sized allocations are involved.
            for (hk, dk, hv, dv, taps) in [(2, 16, 4, 16, 2), (16, 128, 48, 128, 4)] {
                let width = 2 * hk * dk + hv * dv
                for batch in [1, 2] {
                    for seed: UInt64 in [71, 83] {
                        let input = MLXRandom.uniform(low: -1, high: 1,
                            [batch, taps, width], key: MLXRandom.key(seed)).asType(.bfloat16)
                        let weight = MLXRandom.uniform(low: -0.25, high: 0.25,
                            [width, taps, 1], key: MLXRandom.key(seed + 1)).asType(.bfloat16)
                        MLX.eval(input, weight)
                        func plain() -> [MLXArray] {
                            let output = silu(conv1d(input, weight, stride: 1,
                                padding: 0, dilation: 1, groups: width))
                            let split = MLX.split(output, indices: [hk * dk, 2 * hk * dk], axis: -1)
                            let q = split[0].reshaped(batch, 1, hk, dk)
                            let k = split[1].reshaped(batch, 1, hk, dk)
                            let scale = pow(Float(dk), -0.5)
                            return [MLXArray(pow(scale, 2), dtype: q.dtype)
                                * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6),
                                MLXArray(scale, dtype: k.dtype)
                                * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6),
                                split[2].reshaped(batch, 1, hv, dv)]
                        }
                        func candidate() -> [MLXArray] {
                            Qwen4ExpCompiledGDNInputs.callPreprocess(
                                convInput: input, convWeight: weight,
                                numKHeads: hk, headKDim: dk, numVHeads: hv, headVDim: dv)!
                        }
                        let reference = plain()
                        let actual = candidate()
                        MLX.eval(reference); MLX.eval(actual)
                        for (slot, arrays) in zip(actual, reference).enumerated() {
                            #expect(arrays.0.shape == arrays.1.shape)
                            #expect(arrays.0.dtype == arrays.1.dtype)
                            let error = MLX.max(abs(arrays.0.asType(.float32)
                                - arrays.1.asType(.float32))).item(Float.self)
                            #expect(error == 0, "slot=\(slot) seed=\(seed) batch=\(batch) maxAbs=\(error)")
                        }
                        for _ in 0 ..< 5 { MLX.eval(plain()); MLX.eval(candidate()) }
                        func measure(_ operation: () -> [MLXArray]) -> Double {
                            let start = DispatchTime.now().uptimeNanoseconds
                            for _ in 0 ..< 50 { MLX.eval(operation()) }
                            return Double(DispatchTime.now().uptimeNanoseconds - start) / 50_000_000
                        }
                        var p: [Double] = [], c: [Double] = []
                        for round in 0 ..< 7 {
                            if round.isMultiple(of: 2) {
                                p.append(measure(plain)); c.append(measure(candidate))
                            } else {
                                c.append(measure(candidate)); p.append(measure(plain))
                            }
                        }
                        print("[GDNPreprocessBench] hk=\(hk) dk=\(dk) hv=\(hv) dv=\(dv)"
                            + " taps=\(taps) batch=\(batch) seed=\(seed)"
                            + " plain_ms=\(p) compiled_ms=\(c)"
                            + " median_ratio=\(p.sorted()[3] / c.sorted()[3])")
                    }
                }
            }
        }
    }
}
