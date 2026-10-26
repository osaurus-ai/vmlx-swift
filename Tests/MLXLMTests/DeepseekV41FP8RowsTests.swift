import Foundation
import MLX
import XCTest

@testable import MLXLLM

/// The multi-row mxfp8 QMV (DeepSeek-V4.1 dSpark verify / draft rows): every row equals the stock 1-row mxfp8
/// `quantizedMM` bit for bit, for shared weights and for the grouped (batched) wo_a layout; unserved row counts and
/// layouts return nil so the caller keeps the stock kernel.
final class DeepseekV41FP8RowsTests: XCTestCase {
    private func mxfp8(_ shape: [Int], seed: UInt64) -> (w: MLXArray, s: MLXArray) {
        let dense = MLXRandom.normal(shape, key: MLXRandom.key(seed)) * 0.05
        let q = quantized(dense, groupSize: 32, bits: 8, mode: .mxfp8)
        return (q.wq, q.scales)
    }

    private func stock(_ x: MLXArray, _ w: MLXArray, _ s: MLXArray) -> MLXArray {
        quantizedMM(x, w, scales: s, biases: nil, transpose: true, groupSize: 32, bits: 8, mode: .mxfp8)
    }

    func testRowsEqualOneRowCalls() throws {
        let (w, s) = mxfp8([96, 512], seed: 3)
        for rows in 3 ... 7 {
            for dtype in [DType.bfloat16, .float32] {
                let x = (MLXRandom.normal([1, rows, 512], key: MLXRandom.key(UInt64(rows))) * 2).asType(dtype)
                let y = try XCTUnwrap(DeepseekV41FP8Rows.rows(x, weight: w, scales: s))
                XCTAssertEqual(y.shape, [1, rows, 96])
                for r in 0 ..< rows {
                    let ref = stock(x[0..., r ..< r + 1, 0...], w, s)
                    XCTAssertTrue(arrayEqual(y[0..., r ..< r + 1, 0...], ref).item(Bool.self), "rows \(rows) row \(r) \(dtype)")
                }
            }
        }
    }

    func testGroupedWeightsEqualOneRowCalls() throws {
        let (w, s) = mxfp8([4, 64, 256], seed: 9)
        let x = (MLXRandom.normal([4, 6, 256], key: MLXRandom.key(5)) * 2).asType(.bfloat16)
        let y = try XCTUnwrap(DeepseekV41FP8Rows.rows(x, weight: w, scales: s))
        XCTAssertEqual(y.shape, [4, 6, 64])
        for r in 0 ..< 6 {
            XCTAssertTrue(arrayEqual(y[0..., r ..< r + 1, 0...], stock(x[0..., r ..< r + 1, 0...], w, s)).item(Bool.self))
        }
    }

    func testUnservedCallsKeepTheStockKernel() {
        let (w, s) = mxfp8([96, 512], seed: 3)
        for rows in [1, 2, 8, 9] {
            XCTAssertNil(DeepseekV41FP8Rows.rows(MLXArray.zeros([1, rows, 512], dtype: .bfloat16), weight: w, scales: s))
        }
        // K not a multiple of 256, and two shared-weight matrices of rows.
        let (w2, s2) = mxfp8([96, 288], seed: 4)
        XCTAssertNil(DeepseekV41FP8Rows.rows(MLXArray.zeros([1, 4, 288], dtype: .bfloat16), weight: w2, scales: s2))
        XCTAssertNil(DeepseekV41FP8Rows.rows(MLXArray.zeros([2, 4, 512], dtype: .bfloat16), weight: w, scales: s))
    }
}
