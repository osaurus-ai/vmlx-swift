// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXVLM

@Suite("Flash-Next text-position rotary shortcut", .serialized)
struct Qwen4ExpTextRotaryTests {
    private func rotary(_ fast: Bool, dim: Int = 64, sections: [Int] = [11, 11, 10])
        -> Qwen35Language.RotaryEmbedding
    {
        Qwen35Language.RotaryEmbedding(
            dim: dim, base: 10_000_000, mropeSection: sections,
            textPositionFastPath: fast)
    }

    private func expectExact(_ actual: MLXArray, _ expected: MLXArray) {
        #expect(actual.shape == expected.shape)
        #expect(actual.dtype == expected.dtype)
        let a = actual.asType(.float32).asArray(Float.self)
        let b = expected.asType(.float32).asArray(Float.self)
        let finite = a.allSatisfy { $0.isFinite }
        #expect(finite)
        #expect(a.map(\.bitPattern) == b.map(\.bitPattern))
    }

    @Test("text angles preserve exact bits across short, QSA-boundary and long positions")
    func textAngles() throws {
        try MLXMetalTestLock.withLock {
            for dim in [32, 64, 128] {
                let reference = rotary(false, dim: dim)
                let candidate = rotary(true, dim: dim)
                for dtype: DType in [.float16, .bfloat16, .float32] {
                    let x = MLXArray.zeros([1, 1, 1, dim], dtype: dtype)
                    for (offset, count, stride) in [
                        (0, 1, 1), (2047, 4, 1), (2051, 4, 1),
                        (8191, 128, 1), (32767, 4, 1), (65535, 1, 1),
                        (222_000, 4, 1), (8192, 128, 4),
                    ] {
                        let positions = MLXArray((0 ..< count).map { Int32(offset + $0 * stride) })
                            .reshaped(1, count)
                        let a = candidate(x: x, positionIds: positions)
                        let b = reference(x: x, positionIds: positions)
                        expectExact(a.0, b.0)
                        expectExact(a.1, b.1)
                    }
                }
            }
        }
    }

    @Test("independent batch positions and M-RoPE section layouts retain exact text rotations")
    func batchAndSections() throws {
        try MLXMetalTestLock.withLock {
            for sections in [[11, 11, 10], [1, 1, 1], [32, 0, 0], []] {
                let reference = rotary(false, sections: sections)
                let candidate = rotary(true, sections: sections)
                let positions = MLXArray([0, 2048, 2052, 8192, 32768, 65536, 222_000, -17])
                    .reshaped(2, 4)
                let qValues: [Float] = (0 ..< 3072).map { (index: Int) -> Float in
                    Float(index % 53) / 32.0
                }
                let q = MLXArray(qValues)
                    .reshaped(2, 3, 4, 128).asType(.bfloat16)
                let k = q[0..., 0 ..< 1, 0..., 0...]
                let a = candidate(x: q, positionIds: positions)
                let b = reference(x: q, positionIds: positions)
                let aqk = Qwen35Language.applyMultimodalRotaryPosEmb(
                    q: q, k: k, cos: a.0, sin: a.1)
                let bqk = Qwen35Language.applyMultimodalRotaryPosEmb(
                    q: q, k: k, cos: b.0, sin: b.1)
                expectExact(aqk.0, bqk.0)
                expectExact(aqk.1, bqk.1)
            }
        }
    }

    @Test("explicit three-channel media positions never use the text shortcut")
    func mediaPositions() throws {
        try MLXMetalTestLock.withLock {
            let reference = rotary(false)
            let candidate = rotary(true)
            let x = MLXArray.zeros([2, 1, 16, 128], dtype: .bfloat16)
            let positionValues: [Int32] = (0 ..< 96).map { (index: Int) -> Int32 in
                let channel = (index / 32) * 2048
                let withinChannel = (index % 32) * 7
                return Int32(channel + withinChannel - 17)
            }
            let positions = MLXArray(positionValues).reshaped(3, 2, 16)
            let a = candidate(x: x, positionIds: positions)
            let b = reference(x: x, positionIds: positions)
            expectExact(a.0, b.0)
            expectExact(a.1, b.1)
            let collapsed = candidate(x: x, positionIds: positions[0, 0..., 0...])
            #expect(a.0.asArray(Float.self) != collapsed.0.asArray(Float.self))
        }
    }

    @Test("prefill and resumed decode angles agree without retaining request state")
    func chunkAndResume() throws {
        try MLXMetalTestLock.withLock {
            let candidate = rotary(true)
            let reference = rotary(false)
            let x = MLXArray.zeros([1, 1, 8, 64], dtype: .bfloat16)
            for offset in [2047, 2051, 8191, 32767, 222_000] {
                let full = reference(
                    x: x, positionIds: MLXArray(offset ..< (offset + 8)).reshaped(1, 8))
                var cosRows: [MLXArray] = []
                var sinRows: [MLXArray] = []
                for position in offset ..< (offset + 8) {
                    let row = candidate(x: x, positionIds: MLXArray([position]).reshaped(1, 1))
                    cosRows.append(row.0)
                    sinRows.append(row.1)
                }
                expectExact(concatenated(cosRows, axis: 1), full.0)
                expectExact(concatenated(sinRows, axis: 1), full.1)
                // A later request, followed by an earlier restored prefix,
                // must depend only on supplied positions, not the last call.
                let restored = candidate(x: x, positionIds: MLXArray([offset]).reshaped(1, 1))
                expectExact(restored.0, full.0[0..., 0 ..< 1, 0...])
            }
        }
    }

    @Test("bounded text rotary timing diagnostic, not whole-model performance")
    func timing() throws {
        guard ProcessInfo.processInfo.environment["VMLX_TEXT_ROPE_BENCH"] == "1" else { return }
        try MLXMetalTestLock.withLock {
            let reference = rotary(false)
            let candidate = rotary(true)
            let x = MLXArray.zeros([1, 1, 1, 64], dtype: .bfloat16)
            for length in [1, 4, 128, 8192] {
                let positions = MLXArray(32768 ..< (32768 + length)).reshaped(1, length)
                for round in 0 ..< 5 {
                    for fast in (round % 2 == 0 ? [false, true] : [true, false]) {
                        let model = fast ? candidate : reference
                        let start = DispatchTime.now().uptimeNanoseconds
                        for _ in 0 ..< 24 {
                            let result = model(x: x, positionIds: positions)
                            MLX.eval(result.0, result.1)
                        }
                        let elapsed = DispatchTime.now().uptimeNanoseconds - start
                        print(
                            "TEXT_ROPE_BENCH round=\(round) fast=\(fast) length=\(length) calls=24 ns=\(elapsed)"
                        )
                    }
                }
            }
        }
    }
}
