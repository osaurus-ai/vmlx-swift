import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// Numerical composition only; this does not install either primitive in a model.
final class JANGHDecodeCompositionTests: XCTestCase {
    private let prefix = "model.layers.0.mlp.switch_mlp"
    private let width = 96, hiddenSize = 64, outputs = 9, experts = 3, tokens = 2, routes = 8

    private struct Bank {
        let words: [UInt32]
        let scales: [Float16]
        let bits: Int
        let input: Int
        let output: Int
    }

    private func alpha(_ bits: Int) -> Float { 1 / Float(1 << bits) }
    private let beta: Float = 0.0000001

    private func contract(
        _ bits: (Int, Int, Int), input: JANGHFormatContract.Rotation,
        down: JANGHFormatContract.Rotation
    ) throws -> JANGHFormatContract {
        var books: [String: Any] = [:]
        for b in Set([bits.0, bits.1, bits.2]) {
            let a = Double(alpha(b))
            let beta = Double(self.beta)
            let levels = (0 ..< (1 << b)).map { code -> Double in
                let u = Double(code) - Double((1 << b) - 1) / 2
                return Double(Float(u * (a + beta * u * u)))
            }
            books[String(b)] = ["alpha": a, "beta": beta, "levels": levels]
        }
        let config: [String: Any] = [
            "jangtq": [
                "version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
                "codebook_family": "odd-cubic", "rotation": input.rawValue, "codebooks": books,
            ],
            "quantization": [
                prefix + ".gate_proj": [
                    "mode": "jangtq2", "bits": bits.0, "rotation": input.rawValue,
                ],
                prefix + ".up_proj": [
                    "mode": "jangtq2", "bits": bits.1, "rotation": input.rawValue,
                ],
                prefix + ".down_proj": [
                    "mode": "jangtq2", "bits": bits.2, "rotation": down.rawValue,
                ],
            ],
        ]
        return try JANGHFormatContract(
            configuration: JSONSerialization.data(withJSONObject: config))
    }

    private func bank(bits: Int, input: Int, output: Int, seed: Int) -> Bank {
        var words = [UInt32](repeating: 0, count: experts * output * input * bits / 32)
        for index in 0 ..< experts * output * input {
            let code = UInt32((index * 17 + index / input * 7 + seed) % (1 << bits))
            for bit in 0 ..< bits where (code >> bit) & 1 != 0 {
                let offset = index * bits + bit
                words[offset / 32] |= 1 << (offset % 32)
            }
        }
        let scales = (0 ..< experts * output).map { Float16(Float($0 % 7 + 1) / 8) }
        return Bank(words: words, scales: scales, bits: bits, input: input, output: output)
    }

    private func packed(_ bank: Bank) -> MLXArray {
        MLXArray(bank.words, [experts, bank.output, bank.input * bank.bits / 32])
    }

    private func scales(_ bank: Bank) -> MLXArray { MLXArray(bank.scales, [experts, bank.output]) }

    private func dot(_ values: [Float], bank: Bank, expert: Int, row: Int) -> Double {
        var result: Double = 0
        for column in 0 ..< bank.input {
            let bitOffset = ((expert * bank.output + row) * bank.input + column) * bank.bits
            var code: UInt32 = 0
            // Deliberately independent bit-by-bit unpack, not shader word splicing.
            for bit in 0 ..< bank.bits {
                let offset = bitOffset + bit
                code |= ((bank.words[offset / 32] >> (offset % 32)) & 1) << bit
            }
            let u = Double(code) - Double((1 << bank.bits) - 1) / 2
            let level = u * (Double(alpha(bank.bits)) + Double(beta) * u * u)
            result += Double(values[column]) * level
        }
        return result * Double(bank.scales[expert * bank.output + row])
    }

    private func h32(_ values: [Float]) -> [Float] {
        values.indices.map { index in
            var sum: Double = 0
            for j in 0 ..< 32 {
                let sign: Double = ((index % 32) & j).nonzeroBitCount.isMultiple(of: 2) ? 1 : -1
                sum += sign * Double(values[index / 32 * 32 + j])
            }
            return Float(sum / sqrt(32))
        }
    }

    func testMixedProjectionCompositionAgainstScalarOracle() throws {
        try MLXMetalTestLock.withLock {
            // Six mixed triples, with the two basis choices independent. Three
            // input dtypes and three final dtypes produce 54 bounded cases.
            let cases:
                [((Int, Int, Int), JANGHFormatContract.Rotation, JANGHFormatContract.Rotation)] = [
                    ((2, 3, 4), .none, .none),
                    ((3, 4, 6), .none, .hadamard32),
                    ((4, 6, 8), .hadamard32, .none),
                    ((6, 8, 2), .hadamard32, .hadamard32),
                    ((8, 2, 3), .none, .hadamard32),
                    ((3, 8, 6), .hadamard32, .none),
                ]
            let ids: [UInt32] = [2, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 0, 2, 1, 1, 2]
            let weights: [Float] = [
                0, 2, -0.5, 0.25, 1, -1, 0.125, 0, -1, 0.5, 0, 2, 0.25, -0.5, 1, 0.125,
            ]
            let indices = MLXArray(ids, [tokens, routes])
            let scores = MLXArray(weights, [tokens, routes])
            for (caseIndex, entry) in cases.enumerated() {
                let (bits, inputBasis, downBasis) = entry
                let c = try contract(bits, input: inputBasis, down: downBasis)
                let gu = try JANGHFusedGateUpKernel(
                    contract: c, gateModule: prefix + ".gate_proj",
                    upModule: prefix + ".up_proj", outputRotation: downBasis)
                let down = try JANGHWeightedDownKernel(contract: c, module: prefix + ".down_proj")
                let g = bank(bits: bits.0, input: width, output: hiddenSize, seed: 3)
                let u = bank(bits: bits.1, input: width, output: hiddenSize, seed: 11)
                let d = bank(bits: bits.2, input: hiddenSize, output: outputs, seed: 23)
                let limit: Float? = caseIndex == 5 ? nil : 0.75
                for inputDType in [DType.float16, .bfloat16, .float32] {
                    let values = (0 ..< tokens * width).map { Float(($0 * 19) % 59 - 29) / 31 }
                    let input = MLXArray(values, [tokens, width]).asType(inputDType)
                    let rounded = input.asType(.float32).asArray(Float.self)
                    let rows = (0 ..< tokens).map { token -> [Float] in
                        let row = Array(rounded[token * width ..< (token + 1) * width])
                        return inputBasis == .hadamard32 ? h32(row) : row
                    }
                    let prepared = try gu.prepareInputForFusedDecode(input)
                    XCTAssertEqual(
                        prepared.dtype, inputBasis == .hadamard32 ? .float32 : inputDType)
                    let actualHidden = try gu.activatePreparedInput(
                        prepared, gatePacked: packed(g), gateScales: scales(g), upPacked: packed(u),
                        upScales: scales(u), indices: indices, limit: limit)
                    XCTAssertEqual(actualHidden.dtype, .float32)
                    XCTAssertEqual(actualHidden.shape, [tokens * routes, hiddenSize])
                    var referenceHidden: [[Float]] = []
                    for dispatch in 0 ..< tokens * routes {
                        let expert = Int(ids[dispatch])
                        let x = rows[dispatch / routes]
                        let hidden = (0 ..< hiddenSize).map { row -> Float in
                            var gate = dot(x, bank: g, expert: expert, row: row)
                            var up = dot(x, bank: u, expert: expert, row: row)
                            if let limit {
                                gate = min(gate, Double(limit))
                                up = max(-Double(limit), min(up, Double(limit)))
                            }
                            // Stable double-precision sigmoid; cast once at the hidden boundary.
                            let sigmoid =
                                gate >= 0 ? 1 / (1 + exp(-gate)) : exp(gate) / (1 + exp(gate))
                            return Float(gate * sigmoid * up)
                        }
                        referenceHidden.append(downBasis == .hadamard32 ? h32(hidden) : hidden)
                    }
                    let hiddenValues = actualHidden.asArray(Float.self)
                    for dispatch in referenceHidden.indices {
                        for row in 0 ..< hiddenSize {
                            let expected = referenceHidden[dispatch][row]
                            XCTAssertEqual(
                                hiddenValues[dispatch * hiddenSize + row], expected,
                                accuracy: max(0.00002, abs(expected) * 0.001),
                                "hidden case=\(caseIndex) input=\(inputDType)")
                        }
                    }
                    let expected = (0 ..< tokens * outputs).map { output -> Float in
                        let token = output / outputs
                        let row = output % outputs
                        var total: Double = 0
                        for route in 0 ..< routes {
                            let dispatch = token * routes + route
                            total +=
                                Double(weights[dispatch])
                                * dot(
                                    referenceHidden[dispatch], bank: d,
                                    expert: Int(ids[dispatch]), row: row)
                        }
                        return Float(total)
                    }
                    for outputDType in [DType.float16, .bfloat16, .float32] {
                        let result = try down.projectPreparedHidden(
                            actualHidden, preparedBasis: downBasis, packed: packed(d),
                            scales: scales(d),
                            indices: indices, scores: scores, outputDType: outputDType)
                        XCTAssertEqual(result.dtype, outputDType)
                        XCTAssertEqual(result.shape, [tokens, outputs])
                        let output = result.asType(.float32).asArray(Float.self)
                        let tolerance: Float =
                            outputDType == .bfloat16
                            ? 0.008 : (outputDType == .float16 ? 0.002 : 0.0005)
                        for i in output.indices {
                            XCTAssertEqual(
                                output[i], expected[i],
                                accuracy: max(0.00005, abs(expected[i]) * tolerance),
                                "case=\(caseIndex) in=\(inputDType) out=\(outputDType) row=\(i)")
                        }
                    }
                }
            }
        }
    }
}
