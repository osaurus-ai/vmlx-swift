// Copyright © 2026 Osaurus contributors.

import Foundation
import MLX
import MLXFast
import XCTest

final class MLX0323IntegrationTests: XCTestCase {
    override func setUp() { prepareMLXMetallibForTests() }

    func testBatchedLongGQAReadsEachBatchValues() {
        for dtype: DType in [.float32, .float16, .bfloat16] {
            for queryHeads in [48, 64] {
                let shape = [1, 4, 8192, 128]
                let keys = MLXArray.zeros([2, 4, 8192, 128], dtype: dtype)
                let values = concatenated(
                    [
                        MLXArray.full(shape, values: MLXArray(Float(2))).asType(dtype),
                        MLXArray.full(shape, values: MLXArray(Float(-3))).asType(dtype),
                    ], axis: 0)
                let queries = MLXArray.ones([2, queryHeads, 1, 128], dtype: dtype)
                let actual = MLXFast.scaledDotProductAttention(
                    queries: queries, keys: keys, values: values,
                    scale: 1 / sqrt(Float(128)), mask: nil)
                eval(actual)
                XCTAssertEqual(actual.shape, [2, queryHeads, 1, 128])
                XCTAssertTrue(allClose(actual[0], MLXArray(Float(2)), atol: 1e-5).item(Bool.self))
                XCTAssertTrue(allClose(actual[1], MLXArray(Float(-3)), atol: 1e-5).item(Bool.self))
            }
        }
    }

    func testCompiledConstantsPreserveFloatBits() {
        for value: Float in [0.33333334, 1.2345678, 0.000012345678, -1234.5677] {
            let constant = MLXArray(value)
            let compiled = compile { (x: MLXArray) in x * constant }
            let input = MLXArray.ones([4], dtype: .float32)
            let actual = compiled(input)
            let expected = input * constant
            eval(actual, expected)
            XCTAssertEqual(
                actual.asArray(Float.self).map(\.bitPattern),
                expected.asArray(Float.self).map(\.bitPattern))
        }
    }

    func testStoredCacheDtypesAndStridedViewsRoundTrip() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        for dtype: DType in [.float16, .bfloat16, .float32] {
            let original = MLXArray((0 ..< 48).map { Float($0 - 20) / 8 }, [2, 3, 8])
                .asType(dtype).swappedAxes(1, 2)
            let url = directory.appendingPathComponent("cache.safetensors")
            try MLX.save(arrays: ["keys": original], url: url)
            let loaded = try XCTUnwrap(MLX.loadArrays(url: url)["keys"])
            XCTAssertEqual(loaded.dtype, dtype)
            XCTAssertEqual(loaded.shape, original.shape)
            XCTAssertEqual(
                loaded.asType(.float32).asArray(Float.self),
                original.asType(.float32).asArray(Float.self))
        }
    }
}
