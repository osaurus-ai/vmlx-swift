import Foundation
import MLX
import XCTest
@testable import MLXLMCommon

/// Packed prefill primitives only; no model installation or performance claim.
final class JANGHPrefillKernelTests: XCTestCase {
    private let module = "model.layers.0.mlp.switch_mlp"
    private let beta = Float(0.0000001)
    private struct Bank {
        let bits: Int, experts: Int, n: Int, k: Int
        let words: [UInt32], scales: [Float16]
        var packed: MLXArray { MLXArray(words, [experts, n, k * bits / 32]) }
        var scaleArray: MLXArray { MLXArray(scales, [experts, n]) }
    }
    private func contract(_ gate: Int, _ up: Int) throws -> JANGHFormatContract {
        var books: [String: Any] = [:]
        for bits in Set([gate, up]) {
            let a = Float(1) / Float(1 << bits)
            let levels = (0 ..< (1 << bits)).map { code -> Double in
                let u = Float(code) - Float((1 << bits) - 1) / 2
                return Double(u * (a + beta * u * u))
            }
            books[String(bits)] = ["alpha": Double(a), "beta": Double(beta), "levels": levels]
        }
        let config: [String: Any] = [
            "jangtq": ["version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
                       "codebook_family": "odd-cubic", "rotation": "none", "codebooks": books],
            "quantization": [
                module + ".gate_proj": ["mode": "jangtq2", "bits": gate, "rotation": "none"],
                module + ".up_proj": ["mode": "jangtq2", "bits": up, "rotation": "none"],
                module + ".down_proj": ["mode": "jangtq2", "bits": gate, "rotation": "none"],
            ],
        ]
        return try JANGHFormatContract(configuration: JSONSerialization.data(withJSONObject: config))
    }
    private func bank(bits: Int, experts: Int, n: Int, k: Int, seed: Int) -> Bank {
        var words = [UInt32](repeating: 0, count: experts * n * k * bits / 32)
        for index in 0 ..< experts * n * k {
            let code = (index * 13 + index / k * 7 + seed) % (1 << bits)
            for bit in 0 ..< bits where (code >> bit) & 1 == 1 {
                let offset = index * bits + bit
                words[offset / 32] |= UInt32(1) << (offset % 32)
            }
        }
        let scales = (0 ..< experts * n).map { Float16(Float($0 % 5 + 1) / 8) }
        return Bank(bits: bits, experts: experts, n: n, k: k, words: words, scales: scales)
    }
    // Independent bit extraction, not the shader's packed-word loader.
    private func dequant(_ bank: Bank) -> [Float] {
        (0 ..< bank.experts * bank.n * bank.k).map { index in
            var code = 0
            for bit in 0 ..< bank.bits {
                let offset = index * bank.bits + bit
                code |= Int((bank.words[offset / 32] >> (offset % 32)) & 1) << bit
            }
            let u = Float(code) - Float((1 << bank.bits) - 1) / 2
            let level = u * (Float(1) / Float(1 << bank.bits) + beta * u * u)
            return Float(bank.scales[index / bank.k]) * level
        }
    }
    private func rounded(_ values: [Float], _ dtype: DType) -> [Float] {
        values.map { value in
            if dtype == .float16 { return Float(Float16(value)) }
            if dtype == .bfloat16 {
                if !value.isFinite { return value }
                let bits = value.bitPattern
                let rounded = bits &+ 0x7fff &+ ((bits >> 16) & 1)
                return Float(bitPattern: rounded & 0xffff0000)
            }
            return value
        }
    }
    private func hadamard(_ values: [Float]) -> [Float] {
        values.indices.map { output in
            let base = output / 32 * 32
            var sum: Double = 0
            for input in 0 ..< 32 {
                let sign: Double = ((output % 32 & input).nonzeroBitCount % 2 == 0) ? 1 : -1
                sum += sign * Double(values[base + input])
            }
            return Float(sum / sqrt(32))
        }
    }
    private func reference(
        input: [Float], ids: [UInt32], gate: Bank, up: Bank?, dtype: DType,
        backend: JANGHPrefillKernel.Backend, rotate: Bool, limit: Float?
    ) -> [Float] {
        let gw = rounded(dequant(gate), dtype)
        let uw = up.map { rounded(dequant($0), dtype) }
        var output: [Float] = []
        for token in ids.indices {
            guard Int(ids[token]) < gate.experts else {
                output += [Float](repeating: .nan, count: gate.n); continue
            }
            var row = [Float]()
            for n in 0 ..< gate.n {
                let base = (Int(ids[token]) * gate.n + n) * gate.k
                var g: Double = 0, u: Double = 0
                for k in 0 ..< gate.k {
                    g += Double(input[token * gate.k + k]) * Double(gw[base + k])
                    if let uw { u += Double(input[token * gate.k + k]) * Double(uw[base + k]) }
                }
                var gf = Float(g), uf = Float(u)
                if backend == .steel {
                    gf = rounded([gf], dtype)[0]; uf = rounded([uf], dtype)[0]
                }
                if up != nil {
                    if let limit { gf = min(gf, limit); uf = min(max(uf, -limit), limit) }
                    gf = Float(Double(gf) / (1 + exp(-Double(gf))) * Double(uf))
                }
                row.append(gf)
            }
            if backend == .steel { row = rounded(row, dtype) }
            if rotate { row = hadamard(row) }
            output += rounded(row, dtype)
        }
        return output
    }
    private var backends: [JANGHPrefillKernel.Backend] {
        JANGHPrefillKernel.nativeBackend == .nax ? [.steel, .nax] : [.steel]
    }
    private func check(
        tokens: Int, bits: Int, upBits: Int?, n: Int, k: Int, dtype: DType,
        backend: JANGHPrefillKernel.Backend, rotateInput: Bool = false,
        rotateOutput: Bool = false, experts: Int = 3, invalid: Bool = false,
        routeIDs: [UInt32]? = nil
    ) throws {
        let gate = bank(bits: bits, experts: experts, n: n, k: k, seed: 7)
        let up = upBits.map { bank(bits: $0, experts: experts, n: n, k: k, seed: 19) }
        let kernel = try JANGHPrefillKernel(
            contract: contract(bits, upBits ?? bits), module: module + ".gate_proj",
            upModule: upBits == nil ? nil : module + ".up_proj")
        var values = (0 ..< tokens * k).map { Float(($0 * 11 + $0 / k) % 29 - 14) / 32 }
        if rotateInput {
            values = (0 ..< tokens).flatMap { hadamard(Array(values[$0 * k ..< ($0 + 1) * k])) }
        }
        values = rounded(values, dtype)
        // Sorted segments straddle both16-row steel and64-row NAX tiles.
        var ids = routeIDs ?? (0 ..< tokens).map { UInt32(min(experts - 1, $0 / max(1, tokens / experts))) }
        XCTAssertEqual(ids.count, tokens)
        if invalid { ids[tokens - 1] = UInt32.max }
        let limit: Float? = up == nil ? nil : 0.7
        let actual = try kernel.projectSorted(
            MLXArray(values, [tokens, k]).asType(dtype), packed: gate.packed,
            scales: gate.scaleArray, indices: MLXArray(ids),
            upPacked: up?.packed, upScales: up?.scaleArray,
            limit: limit, rotateOutput: rotateOutput, backend: backend)
        let expected = reference(input: values, ids: ids, gate: gate, up: up,
                                 dtype: dtype, backend: backend, rotate: rotateOutput, limit: limit)
        let got = actual.asType(.float32).asArray(Float.self)
        XCTAssertEqual(actual.shape, [tokens, n]); XCTAssertEqual(actual.dtype, dtype)
        let tolerance: Float = dtype == .bfloat16 ? 0.012 : dtype == .float16 ? 0.004 : 0.0001
        for i in got.indices {
            if expected[i].isNaN { XCTAssertTrue(got[i].isNaN, "invalid route at \(i)") }
            else { XCTAssertEqual(got[i], expected[i], accuracy: tolerance * max(1, abs(expected[i])), "\(backend) T\(tokens) bits\(bits) row\(i)") }
        }
    }
    func testStandaloneRowRotationAllInputOutputDTypes() throws {
        try MLXMetalTestLock.withLock {
            let rotate = JANGHRowRotation()
            for rows in [1, 3] {
                for width in [32, 96] {
                    for inputDType in [DType.float16, .bfloat16, .float32] {
                        let values = rounded((0 ..< rows * width).map {
                            Float(($0 * 17 + $0 / width * 7) % 67 - 33) / 19
                        }, inputDType)
                        let transformed = (0 ..< rows).flatMap {
                            hadamard(Array(values[$0 * width ..< ($0 + 1) * width]))
                        }
                        for outputDType in [DType.float16, .bfloat16, .float32] {
                            let actual = try rotate(MLXArray(values, [rows, width]).asType(inputDType),
                                                    outputDType: outputDType)
                            let expected = rounded(transformed, outputDType)
                            XCTAssertEqual(actual.shape, [rows, width])
                            XCTAssertEqual(actual.dtype, outputDType)
                            let got = actual.asType(.float32).asArray(Float.self)
                            let tolerance: Float = outputDType == .bfloat16 ? 0.008 : outputDType == .float16 ? 0.001 : 0.000002
                            for i in got.indices {
                                XCTAssertEqual(got[i], expected[i], accuracy: tolerance * max(1, abs(expected[i])),
                                    "H32 \(inputDType) -> \(outputDType), \(rows)x\(width) element\(i)")
                            }
                        }
                    }
                }
            }
        }
    }

    func testResolvedBackendKeepsFullFloatPrecision() {
        XCTAssertEqual(JANGHPrefillKernel.resolvedBackend(dtype: .float32, requested: .nax), .steel)
        XCTAssertEqual(JANGHPrefillKernel.resolvedBackend(dtype: .float32, requested: .steel), .steel)
        XCTAssertEqual(JANGHPrefillKernel.resolvedBackend(dtype: .bfloat16, requested: .nax), .nax)
        XCTAssertEqual(JANGHPrefillKernel.resolvedBackend(dtype: .float16, requested: .steel), .steel)
    }

    func testProjectionRaggedRowsAndColumnsBothBackends() throws {
        try MLXMetalTestLock.withLock {
            for backend in backends {
                for (i, tokens) in [1, 3, 7, 8, 16, 33, 65].enumerated() {
                    try check(tokens: tokens, bits: [2, 3, 4][i % 3], upBits: nil,
                              n: i % 2 == 0 ? 7 : 65, k: i % 2 == 0 ? 64 : 128,
                              dtype: [.float32, .float16, .bfloat16][i % 3], backend: backend)
                }
            }
        }
    }
    func testMixedFusedGateUpIndependentInputAndOutputRotations() throws {
        try MLXMetalTestLock.withLock {
            for backend in backends {
                for (i, pair) in [(2, 3), (3, 4), (4, 2)].enumerated() {
                    for inputRotation in [false, true] {
                        for outputRotation in [false, true] {
                            try check(tokens: [3, 33, 65][i], bits: pair.0, upBits: pair.1,
                                      n: 96, k: 64, dtype: [.float32, .float16, .bfloat16][i],
                                      backend: backend, rotateInput: inputRotation, rotateOutput: outputRotation)
                        }
                    }
                }
            }
        }
    }
    func testExpertScheduledTilesEmptyExpertsBoundariesAndInvalidSuffix() throws {
        try MLXMetalTestLock.withLock {
            let patterns: [[UInt32]] = [
                [UInt32](repeating: 64, count: 65), // 64 leading empty groups
                [0, 31, 32, 63, 64], // lane and wave boundaries, interior empties
                [UInt32](repeating: 1, count: 15) + [UInt32](repeating: 33, count: 17),
                [UInt32](repeating: 2, count: 63) + [UInt32](repeating: 64, count: 65),
                [0, 32, 64, 65, UInt32.max], // all invalid IDs share the NaN group
                [UInt32](repeating: UInt32.max, count: 17),
            ]
            for backend in backends {
                for (index, ids) in patterns.enumerated() {
                    try check(tokens: ids.count, bits: [2, 3, 4, 6, 8, 2][index],
                              upBits: index % 2 == 0 ? 4 : nil,
                              n: index % 2 == 0 ? 96 : 65, k: 64,
                              dtype: index % 2 == 0 ? .bfloat16 : .float16,
                              backend: backend, rotateOutput: index % 2 == 0,
                              experts: 65, routeIDs: ids)
                }
            }
        }
    }

    func testTinyDeviceBuffersAndInvalidExpert() throws {
        try MLXMetalTestLock.withLock {
            for backend in backends {
                try check(tokens: 1, bits: 2, upBits: nil, n: 1, k: 64,
                          dtype: .float32, backend: backend, experts: 1)
                try check(tokens: 7, bits: 3, upBits: 2, n: 32, k: 64,
                          dtype: .float16, backend: backend, invalid: true)
            }
        }
    }
}
