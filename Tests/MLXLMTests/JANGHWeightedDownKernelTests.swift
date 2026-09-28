import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class JANGHWeightedDownKernelTests: XCTestCase {
    private let prefix = "model.layers.0.mlp.switch_mlp"

    private func makeKernel(
        _ bits: Int, rotation: String = "none", alpha: Double = 0.03125,
        beta: Double = 0.000001
    ) throws -> JANGHWeightedDownKernel {
        let levels = (0 ..< (1 << bits)).map { code -> Double in
            let u = Double(code) - Double((1 << bits) - 1) / 2
            return Double(Float(u * (alpha + beta * u * u)))
        }
        let config: [String: Any] = [
            "jangtq": [
                "version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
                "codebook_family": "odd-cubic", "rotation": rotation,
                "codebooks": [String(bits): ["alpha": alpha, "beta": beta, "levels": levels]],
            ],
            "quantization": Dictionary(
                uniqueKeysWithValues: ["gate_proj", "up_proj", "down_proj"].map {
                    (
                        prefix + "." + $0,
                        ["mode": "jangtq2", "bits": bits, "rotation": rotation] as [String: Any]
                    )
                }),
        ]
        let contract = try JANGHFormatContract(
            configuration: JSONSerialization.data(withJSONObject: config))
        return try JANGHWeightedDownKernel(contract: contract, module: prefix + ".down_proj")
    }

    // Test pack/unpack operates one bit at a time, independently of the shader's
    // cross-word shifts; the oracle reads the actual packed words, not source codes.
    private func pack(_ codes: [UInt32], bits: Int) -> [UInt32] {
        var words = [UInt32](repeating: 0, count: codes.count * bits / 32)
        for (i, code) in codes.enumerated() {
            for b in 0 ..< bits where (code >> b) & 1 != 0 {
                words[(i * bits + b) / 32] |= 1 << ((i * bits + b) % 32)
            }
        }
        return words
    }

    private func oracle(
        _ h: [Float], words: [UInt32], bits: Int, width: Int, rows: Int,
        scales: [Float16], ids: [UInt32], scores: [Float], tokens: Int,
        alpha: Float = 0.03125, beta: Float = 0.000001
    ) -> [Float] {
        let routes = ids.count / tokens
        return (0 ..< tokens * rows).map { output in
            let token = output / rows
            let row = output % rows
            var total: Double = 0
            for route in 0 ..< routes {
                let dispatch = token * routes + route
                let expert = Int(ids[dispatch])
                var dot: Double = 0
                for column in 0 ..< width {
                    let base = ((expert * rows + row) * width + column) * bits
                    var code: UInt32 = 0
                    for b in 0 ..< bits {
                        code |= ((words[(base + b) / 32] >> ((base + b) % 32)) & 1) << b
                    }
                    let u = Double(code) - Double((1 << bits) - 1) / 2
                    dot +=
                        Double(h[dispatch * width + column]) * u
                        * (Double(alpha) + Double(beta) * u * u)
                }
                total += Double(scores[dispatch]) * dot * Double(scales[expert * rows + row])
            }
            return Float(total)
        }
    }

    func testPackedBitsTailsRoutesScoresAndOutputDTypes() throws {
        try MLXMetalTestLock.withLock {
            for bits in [2, 3, 4, 6, 8] {
                let op = try makeKernel(bits)
                for width in [32, 96, 544] {
                    let rows = 9
                    let experts = 3
                    let tokens = 2
                    let routes = 8
                    let ids: [UInt32] = [2, 0, 2, 1, 1, 2, 0, 1, 1, 0, 2, 2, 0, 1, 2, 0]
                    let scores: [Float] = [
                        0, 2, -0.5, 0.125, -1, 0, 0.25, 3, 0.5, -2, 0, 1, 2, -0.25, 0.125, 0,
                    ]
                    let h = (0 ..< tokens * routes * width).map { Float(($0 * 17) % 41 - 20) / 31 }
                    let codes = (0 ..< experts * rows * width).map {
                        UInt32(($0 * 13 + $0 / width) % (1 << bits))
                    }
                    let words = pack(codes, bits: bits)
                    let scales = (0 ..< experts * rows).map { Float16(Float($0 % 5 + 1) / 8) }
                    let expected = oracle(
                        h, words: words, bits: bits, width: width, rows: rows,
                        scales: scales, ids: ids, scores: scores, tokens: tokens)
                    for dtype in [DType.float16, .bfloat16, .float32] {
                        let result = try op.projectPreparedHidden(
                            MLXArray(h, [tokens * routes, width]), preparedBasis: .none,
                            packed: MLXArray(words, [experts, rows, width * bits / 32]),
                            scales: MLXArray(scales, [experts, rows]),
                            indices: MLXArray(ids, [tokens, routes]),
                            scores: MLXArray(scores, [tokens, routes]), outputDType: dtype)
                        XCTAssertEqual(result.shape, [tokens, rows])
                        XCTAssertEqual(result.dtype, dtype)
                        let actual = result.asType(.float32).asArray(Float.self)
                        let relative: Float =
                            dtype == .bfloat16 ? 0.006 : (dtype == .float16 ? 0.001 : 0.0001)
                        for i in actual.indices {
                            XCTAssertEqual(
                                actual[i], expected[i],
                                accuracy: max(0.0002, abs(expected[i]) * relative),
                                "bits=\(bits) H=\(width) dtype=\(dtype) row=\(i)")
                        }
                    }
                }
            }
        }
    }

    func testF32HiddenAndRouteAccumulationAreNotPrematurelyRoundedOrRotated() throws {
        try MLXMetalTestLock.withLock {
            // Inputs are already H32-prepared. A second rotation changes this answer.
            let op = try makeKernel(2, rotation: "hadamard32", alpha: 1, beta: 0)
            var hidden = [Float](repeating: 0, count: 64)
            hidden[0] = 1.001
            hidden[32] = 1
            let packed = MLXArray(pack([UInt32](repeating: 3, count: 32), bits: 2), [1, 1, 2])
            for dtype in [DType.float16, .bfloat16, .float32] {
                let value = try op.projectPreparedHidden(
                    MLXArray(hidden, [2, 32]), preparedBasis: .hadamard32, packed: packed,
                    scales: MLXArray([Float16(1)], [1, 1]),
                    indices: MLXArray([UInt32(0), 0], [1, 2]),
                    scores: MLXArray([Float(1), -1], [1, 2]), outputDType: dtype
                )
                .asType(.float32).item(Float.self)
                XCTAssertEqual(
                    value, 1.5 * (hidden[0] - 1), accuracy: dtype == .bfloat16 ? 0.00001 : 0.000002)
                XCTAssertGreaterThan(value, 0.001)
            }
        }
    }

    func testInvalidExpertProducesNaNWithoutReadingBankAndZeroScoresRemainZero() throws {
        try MLXMetalTestLock.withLock {
            let op = try makeKernel(3)
            let hidden = MLXArray.ones([4, 32], dtype: .float32)
            let packed = MLXArray([UInt32](repeating: 0, count: 27), [1, 9, 3])
            let scales = MLXArray([Float16](repeating: 1, count: 9), [1, 9])
            let scores = MLXArray.zeros([2, 2], dtype: .float32)
            for dtype in [DType.float16, .bfloat16, .float32] {
                let result = try op.projectPreparedHidden(
                    hidden, preparedBasis: .none, packed: packed, scales: scales,
                    indices: MLXArray([UInt32(0), 0, 0, UInt32.max], [2, 2]), scores: scores,
                    outputDType: dtype
                ).asType(.float32).asArray(Float.self)
                XCTAssertTrue(result.prefix(9).allSatisfy { $0 == 0 })
                XCTAssertTrue(result.suffix(9).allSatisfy { $0.isNaN })
            }
        }
    }

    func testKernelIdentityIncludesBitsCoefficientsAndBasis() throws {
        try MLXMetalTestLock.withLock {
            let base = try makeKernel(3).identity
            XCTAssertEqual(base, try makeKernel(3).identity)
            XCTAssertNotEqual(base, try makeKernel(4).identity)
            XCTAssertNotEqual(base, try makeKernel(3, alpha: 0.0625).identity)
            XCTAssertNotEqual(base, try makeKernel(3, beta: 0.000002).identity)
            XCTAssertNotEqual(base, try makeKernel(3, rotation: "hadamard32").identity)
        }
    }

    func testInvalidContractsRefuseBeforeKernelDispatch() throws {
        try MLXMetalTestLock.withLock {
            let op = try makeKernel(2)
            let h = MLXArray.ones([1, 32], dtype: .float32)
            let packed = MLXArray([UInt32(0), 0], [1, 1, 2])
            let scales = MLXArray([Float16(1)], [1, 1])
            let ids = MLXArray([UInt32(0)], [1, 1])
            let scores = MLXArray([Float(1)], [1, 1])
            func call(
                _ input: MLXArray? = nil, basis: JANGHFormatContract.Rotation = .none,
                p: MLXArray? = nil, s: MLXArray? = nil, i: MLXArray? = nil,
                w: MLXArray? = nil, dtype: DType = .float32
            ) throws {
                _ = try op.projectPreparedHidden(
                    input ?? h, preparedBasis: basis, packed: p ?? packed,
                    scales: s ?? scales, indices: i ?? ids, scores: w ?? scores,
                    outputDType: dtype)
            }
            XCTAssertThrowsError(try call(h.asType(.bfloat16)))
            XCTAssertThrowsError(try call(basis: .hadamard32))
            XCTAssertThrowsError(try call(MLXArray.ones([1, 31])))
            XCTAssertThrowsError(try call(MLXArray.ones([2, 32])))
            XCTAssertThrowsError(try call(p: MLXArray.zeros([1, 1, 3], dtype: .uint32)))
            XCTAssertThrowsError(try call(s: scales.asType(.float32)))
            XCTAssertThrowsError(try call(i: ids.flattened()))
            XCTAssertThrowsError(try call(i: ids.asType(.int32)))
            XCTAssertThrowsError(try call(w: scores.asType(.float16)))
            XCTAssertThrowsError(try call(w: MLXArray.ones([1, 2])))
            XCTAssertThrowsError(try call(dtype: .int32))
        }
    }
}
