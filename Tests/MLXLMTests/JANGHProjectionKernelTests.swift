import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class JANGHProjectionKernelTests: XCTestCase {
    private func coefficients(_ bits: Int) -> (Double, Double) {
        switch bits {
        case 2: return (0.893, 0.05065)
        case 3: return (0.47125, 0.0116)
        case 4: return (0.2405, 0.0021)
        default: return (0.125, 0)
        }
    }

    private func kernel(
        bits: Int, rotation: String, alpha override: Double? = nil, beta betaOverride: Double? = nil
    ) throws
        -> JANGHProjectionKernel
    {
        let module = "model.layers.0.mlp.switch_mlp"
        let (baseAlpha, baseBeta) = coefficients(bits)
        let alpha = override ?? baseAlpha
        let beta = betaOverride ?? baseBeta
        let book: [String: Any] = [
            "alpha": alpha, "beta": beta,
            "levels": (0 ..< (1 << bits)).map { index -> Double in
                let u = Double(index) - Double((1 << bits) - 1) / 2
                return Double(Float(u * (alpha + beta * u * u)))
            },
        ]
        let projection: [String: Any] = ["mode": "jangtq2", "bits": bits, "rotation": rotation]
        let config: [String: Any] = [
            "jangtq": [
                "version": 2, "packing": "lsb-bitstream",
                "scale_dtype": "float16", "codebook_family": "odd-cubic", "rotation": rotation,
                "codebooks": [String(bits): book],
            ],
            "quantization": [
                module + ".gate_proj": projection, module + ".up_proj": projection,
                module + ".down_proj": projection,
            ],
        ]
        let contract = try JANGHFormatContract(
            configuration: JSONSerialization.data(withJSONObject: config))
        return try JANGHProjectionKernel(contract: contract, module: module + ".gate_proj")
    }

    // Deliberately independent scalar bit writer; no Metal unpack expression reused.
    private func pack(_ codes: [UInt32], bits: Int) -> [UInt32] {
        var words = [UInt32](repeating: 0, count: codes.count * bits / 32)
        for (column, code) in codes.enumerated() {
            for bit in 0 ..< bits where code & (1 << bit) != 0 {
                let offset = column * bits + bit
                words[offset / 32] |= 1 << (offset % 32)
            }
        }
        return words
    }

    private func h32(_ row: [Float]) -> [Float] {
        // Dense Sylvester signs instead of the kernel's shuffle butterflies.
        (0 ..< row.count).map { column in
            let base = column / 32 * 32
            var value: Float = 0
            for j in 0 ..< 32 {
                let sign: Float = ((column % 32) & j).nonzeroBitCount.isMultiple(of: 2) ? 1 : -1
                value += row[base + j] * sign
            }
            return value / sqrt(Float(32))
        }
    }

    func testProjectionPackingRotationAndTails() throws {
        try MLXMetalTestLock.withLock {
            for bits in [2, 3, 4, 6, 8] {
                let (alpha, beta) = coefficients(bits)
                for width in [32, 96, 512, 544] {
                    for rotation in ["none", "hadamard32"] {
                        let op = try kernel(bits: bits, rotation: rotation)
                        let experts = 3
                        let outputs = 9
                        let tokens = 2
                        let codes = (0 ..< (experts * outputs * width)).map {
                            UInt32(($0 * 13 + 7) % (1 << bits))
                        }
                        let packed = MLXArray(
                            pack(codes, bits: bits), [experts, outputs, width * bits / 32])
                        let scaleValues = (0 ..< (experts * outputs)).map {
                            Float16(Float($0 + 1) / 64)
                        }
                        let scales = MLXArray(scaleValues, [experts, outputs])
                        let values = (0 ..< (tokens * width)).map { Float(($0 * 7) % 23 - 11) / 31 }
                        let routes: [UInt32] = [2, 0, 1, 2]
                        for dtype in [DType.float16, .bfloat16, .float32] {
                            let inputArray = MLXArray(values, [tokens, width]).asType(dtype)
                            let castValues = inputArray.asType(.float32).asArray(Float.self)
                            let actual = try op.project(
                                inputArray, packed: packed,
                                scales: scales, indices: MLXArray(routes)
                            ).asArray(Float.self)
                            for dispatch in 0 ..< routes.count {
                                let token = dispatch / 2
                                let input = Array(castValues[token * width ..< (token + 1) * width])
                                // H32 uses F32 arithmetic then rounds back to activation dtype.
                                let rotated =
                                    rotation == "hadamard32"
                                    ? MLXArray(h32(input)).asType(dtype).asType(.float32).asArray(
                                        Float.self)
                                    : input
                                for output in 0 ..< outputs {
                                    let row = Int(routes[dispatch]) * outputs + output
                                    var expected: Double = 0
                                    for column in 0 ..< width {
                                        let u =
                                            Double(codes[row * width + column]) - Double(
                                                (1 << bits) - 1) / 2
                                        let level = u * (alpha + beta * u * u)
                                        expected += Double(rotated[column]) * level
                                    }
                                    expected *= Double(scaleValues[row])
                                    XCTAssertEqual(
                                        Double(actual[dispatch * outputs + output]), expected,
                                        accuracy: max(0.0003, abs(expected) * 0.0003),
                                        "bits=\(bits) K=\(width) rotation=\(rotation) dtype=\(dtype) dispatch=\(dispatch) row=\(output)"
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    func testH32DTypesAndIdentity() throws {
        try MLXMetalTestLock.withLock {
            let op = try kernel(bits: 3, rotation: "hadamard32")
            XCTAssertEqual(op.identity, try kernel(bits: 3, rotation: "hadamard32").identity)
            XCTAssertNotEqual(op.identity, try kernel(bits: 3, rotation: "none").identity)
            XCTAssertNotEqual(
                op.identity, try kernel(bits: 3, rotation: "hadamard32", alpha: 0.5).identity)
            let values = (0 ..< 96).map { Float($0 % 7 - 3) / 16 }
            let expected = h32(values)
            for dtype in [DType.float16, .bfloat16, .float32] {
                let input = MLXArray(values, [1, 96]).asType(dtype)
                let output = try op.hadamard32(input)
                XCTAssertEqual(output.dtype, dtype)
                let actual = output.asType(.float32).asArray(Float.self)
                let rounded = MLXArray(expected).asType(dtype).asType(.float32).asArray(Float.self)
                for index in actual.indices {
                    XCTAssertEqual(actual[index], rounded[index], accuracy: 0.00001)
                }
            }
        }
    }

    func testInvalidGeometryRejectedBeforeDispatch() throws {
        try MLXMetalTestLock.withLock {
            let op = try kernel(bits: 3, rotation: "none")
            XCTAssertThrowsError(try op.hadamard32(MLXArray([Float(1)], [1, 1])))
            XCTAssertThrowsError(
                try op.project(
                    MLXArray([Float](repeating: 0, count: 32), [1, 32]),
                    packed: MLXArray([UInt32](repeating: 0, count: 4), [1, 1, 4]),
                    scales: MLXArray([Float16(1)], [1, 1]), indices: MLXArray([UInt32(0)])))
        }
    }
    func testInvalidRoutesAndEmptyInputContracts() throws {
        try MLXMetalTestLock.withLock {
            let op = try kernel(bits: 3, rotation: "none")
            let input = MLXArray([Float](repeating: 1, count: 32), [1, 32])
            let packed = MLXArray(pack([UInt32](repeating: 0, count: 32), bits: 3), [1, 1, 3])
            let scales = MLXArray([Float16(1)], [1, 1])
            let invalid = try op.project(
                input, packed: packed, scales: scales,
                indices: MLXArray([UInt32(1), UInt32.max])
            ).asArray(Float.self)
            XCTAssertEqual(invalid.count, 2)
            XCTAssertTrue(invalid.allSatisfy { $0.isNaN })
            XCTAssertThrowsError(
                try op.project(
                    input, packed: packed, scales: scales,
                    indices: MLXArray([UInt32]())))
            let empty = MLXArray([Float](), [0, 32])
            XCTAssertEqual(try op.hadamard32(empty).shape, [0, 32])
            XCTAssertThrowsError(
                try op.project(
                    empty, packed: packed, scales: scales,
                    indices: MLXArray([UInt32(0)])))
        }
    }
    /// Optional private byte-range fixtures; never stored as model weights in the repository.
    func testPrivatePackedFixturesWhenProvided() throws {
        guard let path = ProcessInfo.processInfo.environment["JANGH_PROJECTION_FIXTURES"] else {
            throw XCTSkip("Set JANGH_PROJECTION_FIXTURES for local packed byte-range evidence")
        }
        try MLXMetalTestLock.withLock {
            let directory = URL(fileURLWithPath: path, isDirectory: true)
            let receipt =
                try JSONSerialization.jsonObject(
                    with: Data(contentsOf: directory.appendingPathComponent("receipt.json")))
                as! [String: Any]
            for fixture in receipt["fixtures"] as! [[String: Any]] {
                let bits = fixture["bits"] as! Int
                let rotation = fixture["rotation"] as! String
                let tensors = fixture["tensors"] as! [String: [String: Any]]
                let packedInfo = tensors["tq2_packed"]!
                let scaleInfo = tensors["tq2_scales"]!
                let shape = packedInfo["shape"] as! [Int]
                let packedData = try Data(
                    contentsOf: directory.appendingPathComponent(packedInfo["file"] as! String))
                let scaleData = try Data(
                    contentsOf: directory.appendingPathComponent(scaleInfo["file"] as! String))
                let words: [UInt32] = packedData.withUnsafeBytes { bytes in
                    stride(from: 0, to: bytes.count, by: 4).map {
                        UInt32(
                            littleEndian: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self))
                    }
                }
                let rowScales: [Float16] = scaleData.withUnsafeBytes { bytes in
                    stride(from: 0, to: bytes.count, by: 2).map {
                        Float16(
                            bitPattern: UInt16(
                                littleEndian: bytes.loadUnaligned(
                                    fromByteOffset: $0, as: UInt16.self)))
                    }
                }
                let width = shape[2] * 32 / bits
                let values = (0 ..< width).map { Float($0 % 23 - 11) / 31 }
                let book = fixture["codebook"] as! [String: Any]
                let alpha = book["alpha"] as! Double
                let beta = book["beta"] as! Double
                let op = try kernel(bits: bits, rotation: rotation, alpha: alpha, beta: beta)
                let actual = try op.project(
                    MLXArray(values, [1, width]),
                    packed: MLXArray(words, shape),
                    scales: MLXArray(rowScales, [shape[0], shape[1]]),
                    indices: MLXArray([UInt32(1), UInt32(0)])
                ).asArray(Float.self)
                let transformed = rotation == "hadamard32" ? h32(values) : values
                for (dispatch, expert) in [1, 0].enumerated() {
                    for output in 0 ..< shape[1] {
                        let row = expert * shape[1] + output
                        var expected: Double = 0
                        for column in 0 ..< width {
                            var code: UInt32 = 0
                            // Independent bit-at-a-time unpacking against the real bytes.
                            for bit in 0 ..< bits {
                                let offset = column * bits + bit
                                let word = words[row * shape[2] + offset / 32]
                                code |= ((word >> (offset % 32)) & 1) << bit
                            }
                            let u = Double(code) - Double((1 << bits) - 1) / 2
                            expected += Double(transformed[column]) * u * (alpha + beta * u * u)
                        }
                        expected *= Double(rowScales[row])
                        XCTAssertEqual(
                            Double(actual[dispatch * shape[1] + output]), expected,
                            accuracy: max(0.0003, abs(expected) * 0.0003))
                    }
                }
            }
        }
    }

}
