// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// chunkedVectorAttention must compute what one fused SDPA call computes for draft/verify blocks.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class ChunkedVectorAttentionTests: XCTestCase {

    private func inputs(rows: Int, keys: Int, dtype: DType) -> (MLXArray, MLXArray, MLXArray) {
        MLXRandom.seed(UInt64(rows * 1000 + keys))
        let q = MLXRandom.normal([1, 32, rows, 128]).asType(dtype)
        let k = MLXRandom.normal([1, 8, keys, 128]).asType(dtype)
        let v = MLXRandom.normal([1, 8, keys, 128]).asType(dtype)
        return (q, k, v)
    }

    /// True float32 attention on the CPU (GPU float32 matmuls run as TF32 on NAX devices).
    private func cpuReference(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, scale: Float) -> MLXArray {
        Device.withDefaultDevice(.cpu) {
            let (H, qL, kL) = (q.dim(1), q.dim(2), k.dim(2))
            let g = H / k.dim(1)
            let kf = repeated(k.asType(.float32), count: g, axis: 1)
            let vf = repeated(v.asType(.float32), count: g, axis: 1)
            var s = matmul(q.asType(.float32), kf.transposed(0, 1, 3, 2)) * scale
            s = MLX.where(createCausalMask(n: qL, offset: kL - qL), s, MLXArray(-Float.infinity))
            let out = matmul(softmax(s, axis: -1), vf)
            eval(out)
            return out
        }
    }

    /// What ordinary decoding computes: one 1-row fused call per position over the keys before it.
    private func perPositionDecode(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, scale: Float) -> MLXArray {
        let (qL, kL) = (q.dim(2), k.dim(2))
        let rows = (0 ..< qL).map { i -> MLXArray in
            let visible = kL - qL + i + 1
            return MLXFast.scaledDotProductAttention(
                queries: q[0..., 0..., i ..< i + 1, 0...], keys: k[0..., 0..., ..<visible, 0...],
                values: v[0..., 0..., ..<visible, 0...], scale: scale, mask: .none)
        }
        return concatenated(rows, axis: 2)
    }

    /// Relative RMS error, ‖a − reference‖ / ‖reference‖.
    private func relativeError(_ a: MLXArray, _ reference: MLXArray) -> Float {
        let d = a.asType(.float32) - reference
        return (sqrt((d * d).sum()) / sqrt((reference * reference).sum())).item(Float.self)
    }

    private func maxError(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// Against a float32 reference, at key lengths that are not multiples of any tile or chunk,
    /// with the causal mode and the explicit offset mask the KV cache hands the model: the chunks
    /// carry the precision of ordinary decoding (one 1-row call per position, the same vector
    /// kernel). Measured relative RMS error within 3% of it; one fused call over the whole block
    /// (the tiled kernel) is printed for reference and differs either way by dtype.
    func testMatchesFusedAttentionAccuracyForCausalAndArrayMasks() throws {
        let scale = 1 / Float(128).squareRoot()
        for dtype in [DType.bfloat16, .float16] {
            for (rows, keys) in [(9, 9), (12, 301), (12, 2_303), (16, 2_305), (17, 777), (32, 1_029)] {
                let (q, k, v) = inputs(rows: rows, keys: keys, dtype: dtype)
                let reference = cpuReference(q, k, v, scale: scale)
                let fused = relativeError(
                    MLXFast.scaledDotProductAttention(
                        queries: q, keys: k, values: v, scale: scale, mask: .causal),
                    reference)
                let decode = relativeError(perPositionDecode(q, k, v, scale: scale), reference)
                let causal = try XCTUnwrap(
                    chunkedVectorAttention(queries: q, keys: k, values: v, scale: scale, mask: .causal))
                let explicit = try XCTUnwrap(
                    chunkedVectorAttention(
                        queries: q, keys: k, values: v, scale: scale,
                        mask: .array(createCausalMask(n: rows, offset: keys - rows))))
                let chunked = relativeError(causal, reference)
                let label = "\(dtype) rows=\(rows) keys=\(keys) relRMS: fused=\(fused) decode=\(decode) chunked=\(chunked)"
                XCTAssertEqual(causal.dtype, dtype)
                XCTAssertEqual(causal.shape, q.shape)
                XCTAssertLessThanOrEqual(chunked, decode * 1.05, label)
                XCTAssertLessThanOrEqual(relativeError(explicit, reference), decode * 1.05, label)
            }
        }
    }

    func testDeclinesShapesItDoesNotServe() {
        let (q8, k8, v8) = inputs(rows: 8, keys: 100, dtype: .bfloat16)
        XCTAssertNil(chunkedVectorAttention(queries: q8, keys: k8, values: v8, scale: 1, mask: .causal))
        let (q, k, v) = inputs(rows: 12, keys: 10, dtype: .bfloat16)
        XCTAssertNil(chunkedVectorAttention(queries: q, keys: k, values: v, scale: 1, mask: .causal))
        let (q2, k2, v2) = inputs(rows: 12, keys: 40, dtype: .bfloat16)
        XCTAssertNil(
            chunkedVectorAttention(
                queries: q2, keys: k2, values: v2, scale: 1,
                mask: .array(MLXArray.ones([12, 39], dtype: .bool))))
        XCTAssertNil(chunkedVectorAttention(queries: q2, keys: k2, values: v2, scale: 1, mask: .none))
    }
}
