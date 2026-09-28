import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class JANGHFusedGateUpKernelTests: XCTestCase {
    private let module = "model.layers.0.mlp.switch_mlp"

    private func coefficients(_ bits: Int) -> (Double, Double) {
        switch bits {
        case 2: return (0.893, 0.05065)
        case 3: return (0.47125, 0.0116)
        case 4: return (0.2405, 0.0021)
        default: return (0.01, 0.0000001)
        }
    }

    private func makeKernel(
        _ g: Int, _ u: Int, inputRotation: String = "hadamard32",
        outputRotation: JANGHFormatContract.Rotation = .none,
        upAlpha: Double? = nil
    ) throws
        -> JANGHFusedGateUpKernel
    {
        var books: [String: Any] = [:]
        for bits in Set([g, u]) {
            let (baseAlpha, b) = coefficients(bits)
            let a = bits == u ? (upAlpha ?? baseAlpha) : baseAlpha
            let levels = (0 ..< (1 << bits)).map { i -> Double in
                let v = Double(i) - Double((1 << bits) - 1) / 2
                return Double(Float(v * (a + b * v * v)))
            }
            books[String(bits)] = ["alpha": a, "beta": b, "levels": levels]
        }
        let config: [String: Any] = [
            "jangtq": [
                "version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
                "codebook_family": "odd-cubic", "rotation": inputRotation, "codebooks": books,
            ],
            "quantization": [
                module + ".gate_proj": ["mode": "jangtq2", "bits": g, "rotation": inputRotation],
                module + ".up_proj": ["mode": "jangtq2", "bits": u, "rotation": inputRotation],
                module + ".down_proj": [
                    "mode": "jangtq2", "bits": g, "rotation": outputRotation.rawValue,
                ],
            ],
        ]
        let contract = try JANGHFormatContract(
            configuration: JSONSerialization.data(withJSONObject: config))
        return try JANGHFusedGateUpKernel(
            contract: contract, gateModule: module + ".gate_proj",
            upModule: module + ".up_proj", outputRotation: outputRotation)
    }

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
        (0 ..< row.count).map { i in
            var sum: Double = 0
            for j in 0 ..< 32 {
                let sign: Double = ((i % 32) & j).nonzeroBitCount.isMultiple(of: 2) ? 1 : -1
                sum += Double(row[i / 32 * 32 + j]) * sign
            }
            return Float(sum / sqrt(32.0))
        }
    }

    private func bank(bits: Int, k: Int, n: Int, seed: Int) -> (MLXArray, [UInt32]) {
        let codes = (0 ..< 2 * n * k).map { UInt32(($0 * 7 + $0 / k + seed) % (1 << bits)) }
        return (MLXArray(pack(codes, bits: bits), [2, n, k * bits / 32]), codes)
    }

    private func dot(_ x: [Float], codes: [UInt32], bits: Int, row: Int, scale: Float) -> Float {
        let (a, b) = coefficients(bits)
        var sum: Double = 0
        for j in x.indices {
            let v = Double(codes[row * x.count + j]) - Double((1 << bits) - 1) / 2
            sum += Double(x[j]) * v * (a + b * v * v)
        }
        return Float(sum) * scale
    }

    func testMixedBitsDTypesAndF32RotationBoundary() throws {
        try MLXMetalTestLock.withLock {
            for gb in [2, 3, 4, 6, 8] {
                for ub in [2, 3, 4, 6, 8] {
                    for dtype in [DType.float16, .bfloat16, .float32] {
                        for rotated in [false, true] {
                            let k = 96
                            let n = rotated ? 32 : 9
                            let op = try makeKernel(
                                gb, ub, outputRotation: rotated ? .hadamard32 : .none)
                            let raw = (0 ..< 2 * k).map { Float(($0 * 17) % 53 - 26) / 37 }
                            let input = MLXArray(raw, [2, k]).asType(dtype)
                            let rounded = input.asType(.float32).asArray(Float.self)
                            let xRows = [
                                h32(Array(rounded[0 ..< k])), h32(Array(rounded[k ..< 2 * k])),
                            ]
                            let prepared = try op.prepareInputForFusedDecode(input)
                            XCTAssertEqual(prepared.dtype, .float32)
                            let preparedValues = prepared.asArray(Float.self)
                            for i in 0 ..< 2 * k {
                                XCTAssertEqual(
                                    preparedValues[i], xRows[i / k][i % k], accuracy: 2e-6)
                            }
                            let (gp, gc) = bank(bits: gb, k: k, n: n, seed: 1)
                            let (up, uc) = bank(bits: ub, k: k, n: n, seed: 3)
                            let gs = (0 ..< 2 * n).map { Float16(Float($0 % 5 + 1) / 8) }
                            let us = (0 ..< 2 * n).map { Float16(Float($0 % 7 + 1) / 4) }
                            let routes: [UInt32] = [1, 0, 0, 1]
                            let result = try op.activatePreparedInput(
                                prepared, gatePacked: gp, gateScales: MLXArray(gs, [2, n]),
                                upPacked: up, upScales: MLXArray(us, [2, n]),
                                indices: MLXArray(routes), limit: 10)
                            XCTAssertEqual(result.dtype, .float32)
                            let actual = result.asArray(Float.self)
                            for dispatch in 0 ..< 4 {
                                var hidden: [Float] = []
                                for r in 0 ..< n {
                                    let row = Int(routes[dispatch]) * n + r
                                    let g = min(
                                        dot(
                                            xRows[dispatch / 2], codes: gc, bits: gb, row: row,
                                            scale: Float(gs[row])), 10)
                                    let u = max(
                                        -10,
                                        min(
                                            dot(
                                                xRows[dispatch / 2], codes: uc, bits: ub, row: row,
                                                scale: Float(us[row])), 10))
                                    hidden.append(g / (1 + exp(-g)) * u)
                                }
                                let expected = rotated ? h32(hidden) : hidden
                                for r in 0 ..< n {
                                    XCTAssertEqual(
                                        actual[dispatch * n + r], expected[r],
                                        accuracy: max(0.002, abs(expected[r]) * 0.001),
                                        "gate=\(gb) up=\(ub) dtype=\(dtype) rotated=\(rotated)")
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    func testRawClampContractAndUnclampedBranch() throws {
        try MLXMetalTestLock.withLock {
            let op = try makeKernel(2, 2, inputRotation: "none")
            let k = 32
            let n = 4
            let input = MLXArray([Float](repeating: 1, count: k), [1, k])
            let gates = [0, 3, 0, 3].flatMap { [UInt32](repeating: UInt32($0), count: k) }
            let ups = [0, 0, 3, 3].flatMap { [UInt32](repeating: UInt32($0), count: k) }
            let gp = MLXArray(pack(gates, bits: 2), [1, n, 2])
            let up = MLXArray(pack(ups, bits: 2), [1, n, 2])
            let gs = MLXArray([Float16](repeating: 0.5, count: n), [1, n])
            let us = MLXArray([Float16](repeating: 8, count: n), [1, n])
            for limit: Float? in [nil, 10] {
                let result = try op.activatePreparedInput(
                    input, gatePacked: gp, gateScales: gs,
                    upPacked: up, upScales: us,
                    indices: MLXArray([UInt32(0)]), limit: limit
                ).asArray(Float.self)
                for r in 0 ..< n {
                    var g = dot(
                        [Float](repeating: 1, count: k), codes: gates, bits: 2, row: r, scale: 0.5)
                    var u = dot(
                        [Float](repeating: 1, count: k), codes: ups, bits: 2, row: r, scale: 8)
                    if let limit {
                        g = min(g, limit)
                        u = max(-limit, min(u, limit))
                    }
                    let expected = g / (1 + exp(-g)) * u
                    XCTAssertEqual(
                        result[r], expected, accuracy: max(1e-10, abs(expected) * 0.0002))
                }
            }
        }
    }

    func testGuardsAndInvalidExpertBarrier() throws {
        try MLXMetalTestLock.withLock {
            for rotated in [false, true] {
                let n = rotated ? 32 : 9
                let k = 544
                let op = try makeKernel(
                    3, 2, inputRotation: "none", outputRotation: rotated ? .hadamard32 : .none)
                let (gp, gc) = bank(bits: 3, k: k, n: n, seed: 1)
                let (up, uc) = bank(bits: 2, k: k, n: n, seed: 3)
                let scales = MLXArray([Float16](repeating: 1, count: 2 * n), [2, n])
                let x = MLXArray([Float](repeating: 0.1, count: k), [1, k])
                func run(_ indices: MLXArray, _ limit: Float? = nil) throws -> MLXArray {
                    try op.activatePreparedInput(
                        x, gatePacked: gp, gateScales: scales,
                        upPacked: up, upScales: scales, indices: indices, limit: limit)
                }
                let invalid = try run(MLXArray([UInt32(2), UInt32.max])).asArray(Float.self)
                XCTAssertTrue(invalid.allSatisfy(\.isNaN))
                XCTAssertThrowsError(try run(MLXArray([UInt32]())))
                XCTAssertThrowsError(try run(MLXArray([UInt32(0)]), .infinity))
                XCTAssertThrowsError(try run(MLXArray([UInt32(0)]), 0))
                XCTAssertThrowsError(
                    try op.prepareInputForFusedDecode(MLXArray([Float](), [0, k])))
                let valid = try run(MLXArray([UInt32(0)])).asArray(Float.self)
                var hidden: [Float] = []
                for row in 0 ..< n {
                    let values = [Float](repeating: 0.1, count: k)
                    let g = dot(values, codes: gc, bits: 3, row: row, scale: 1)
                    let u = dot(values, codes: uc, bits: 2, row: row, scale: 1)
                    hidden.append(g / (1 + exp(-g)) * u)
                }
                let expected = rotated ? h32(hidden) : hidden
                for row in 0 ..< n {
                    XCTAssertEqual(
                        valid[row], expected[row], accuracy: max(0.002, abs(expected[row]) * 0.001))
                }
                XCTAssertThrowsError(
                    try op.activatePreparedInput(
                        x, gatePacked: gp, gateScales: scales.asType(.float32),
                        upPacked: up, upScales: scales, indices: MLXArray([UInt32(0)]), limit: nil))
                if !rotated {
                    let rotatedOp = try makeKernel(
                        3, 2, inputRotation: "none", outputRotation: .hadamard32)
                    XCTAssertThrowsError(
                        try rotatedOp.activatePreparedInput(
                            x, gatePacked: gp, gateScales: scales, upPacked: up, upScales: scales,
                            indices: MLXArray([UInt32(0)]), limit: nil))
                }
            }
            let a = try makeKernel(2, 3)
            let b = try makeKernel(2, 4)
            XCTAssertNotEqual(a.identity, b.identity)
            XCTAssertEqual(a.identity, try makeKernel(2, 3).identity)
            XCTAssertNotEqual(
                a.identity, try makeKernel(2, 3, outputRotation: .hadamard32).identity)
        }
    }

    func testDirectLowPrecisionInputsMultipleH32BlocksAndTopEight() throws {
        try MLXMetalTestLock.withLock {
            let k = 32
            let n = 64
            let (gp, gc) = bank(bits: 3, k: k, n: n, seed: 1)
            let (up, uc) = bank(bits: 2, k: k, n: n, seed: 3)
            let gs = (0 ..< 2 * n).map { Float16(Float($0 % 5 + 1) / 16) }
            let us = (0 ..< 2 * n).map { Float16(Float($0 % 7 + 1) / 8) }
            let routes: [UInt32] = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0]
            for dtype in [DType.float16, .bfloat16, .float32] {
                for rotated in [false, true] {
                    let op = try makeKernel(
                        3, 2, inputRotation: "none", outputRotation: rotated ? .hadamard32 : .none)
                    let raw = (0 ..< 2 * k).map { Float(($0 * 11) % 31 - 15) / 29 }
                    let input = MLXArray(raw, [2, k]).asType(dtype)
                    let prepared = try op.prepareInputForFusedDecode(input)
                    XCTAssertEqual(prepared.dtype, dtype)
                    let rounded = input.asType(.float32).asArray(Float.self)
                    XCTAssertEqual(prepared.asType(.float32).asArray(Float.self), rounded)
                    let actual = try op.activatePreparedInput(
                        prepared, gatePacked: gp, gateScales: MLXArray(gs, [2, n]),
                        upPacked: up, upScales: MLXArray(us, [2, n]),
                        indices: MLXArray(routes, [2, 8]), limit: nil
                    ).asArray(Float.self)
                    for dispatch in routes.indices {
                        let token = dispatch / 8
                        let x = Array(rounded[token * k ..< (token + 1) * k])
                        var hidden: [Float] = []
                        for r in 0 ..< n {
                            let row = Int(routes[dispatch]) * n + r
                            let g = dot(x, codes: gc, bits: 3, row: row, scale: Float(gs[row]))
                            let u = dot(x, codes: uc, bits: 2, row: row, scale: Float(us[row]))
                            hidden.append(g / (1 + exp(-g)) * u)
                        }
                        let expected = rotated ? h32(hidden) : hidden
                        for r in 0 ..< n {
                            XCTAssertEqual(
                                actual[dispatch * n + r], expected[r],
                                accuracy: max(0.002, abs(expected[r]) * 0.001),
                                "direct dtype=\(dtype) rotated=\(rotated) dispatch=\(dispatch) row=\(r)"
                            )
                        }
                    }
                }
            }
        }
    }

    func testNearZeroExtremeFiniteActivationAndCodebookIdentity() throws {
        try MLXMetalTestLock.withLock {
            let op = try makeKernel(3, 2, inputRotation: "none")
            // Different bit widths ensure this override changes only the up codebook.
            let changed = try makeKernel(3, 2, inputRotation: "none", upAlpha: 0.8)
            XCTAssertNotEqual(op.identity, changed.identity)
            XCTAssertEqual(
                changed.identity, try makeKernel(3, 2, inputRotation: "none", upAlpha: 0.8).identity
            )
            let k = 32
            let n = 2
            let gc = [UInt32](repeating: 0, count: k) + [UInt32](repeating: 7, count: k)
            let uc = [UInt32](repeating: 3, count: k) + [UInt32](repeating: 0, count: k)
            let gp = MLXArray(pack(gc, bits: 3), [1, n, 3])
            let up = MLXArray(pack(uc, bits: 2), [1, n, 2])
            let scales = MLXArray([Float16(1), Float16(1)], [1, n])
            for magnitude: Float in [0, 1e-8, -1e-8, 1, -1, 10_000, -10_000] {
                let values = [Float](repeating: magnitude, count: k)
                let actual = try op.activatePreparedInput(
                    MLXArray(values, [1, k]), gatePacked: gp, gateScales: scales,
                    upPacked: up, upScales: scales, indices: MLXArray([UInt32(0)]),
                    limit: nil
                ).asArray(Float.self)
                XCTAssertTrue(actual.allSatisfy(\.isFinite))
                for row in 0 ..< n {
                    let g = Double(dot(values, codes: gc, bits: 3, row: row, scale: 1))
                    let u = Double(dot(values, codes: uc, bits: 2, row: row, scale: 1))
                    // Stable independent scalar sigmoid handles large negative gates.
                    let sigmoid = g >= 0 ? 1 / (1 + exp(-g)) : exp(g) / (1 + exp(g))
                    let expected = Float(g * sigmoid * u)
                    XCTAssertEqual(
                        actual[row], expected, accuracy: max(1e-12, abs(expected) * 0.001))
                }
            }
        }
    }
}
