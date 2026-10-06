import Foundation
import MLX
import XCTest
@testable import MLXLMCommon

final class JANGHRoutedPrefillTests: XCTestCase {
    private let parent = "model.layers.0.mlp.switch_mlp"
    private let width = 64, experts = 3, routes = 8
    private struct Bank {
        let bits: Int
        let words: [UInt32]
        let scales: [Float16]
    }
    private func bank(_ bits: Int, seed: Int) -> Bank {
        var words = [UInt32](repeating: 0, count: experts * width * width * bits / 32)
        for i in 0 ..< experts * width * width {
            let q = (i * 7 + i / width * 3 + seed) % (1 << bits)
            for b in 0 ..< bits where (q >> b) & 1 != 0 {
                let offset = i * bits + b
                words[offset / 32] |= UInt32(1) << (offset % 32)
            }
        }
        return Bank(bits: bits, words: words,
                    scales: (0 ..< experts * width).map { Float16(Float($0 % 3 + 1) / 16) })
    }
    private func fixture(_ directory: URL, inputRotation: Bool, outputRotation: Bool,
                         bits: [Int] = [2, 3, 4])
        throws -> (JANGHMappedBanks, [Bank]) {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let banks = [bank(bits[0], seed: 5), bank(bits[1], seed: 11), bank(bits[2], seed: 19)]
        var books: [String: Any] = [:], quant: [String: Any] = [:], headers: [String: Any] = [:]
        var map: [String: String] = [:], dimensions: [String: JANGHTensorIndexPlan.Dimensions] = [:]
        var payload = Data()
        for (i, role) in ["gate_proj", "up_proj", "down_proj"].enumerated() {
            let b = banks[i], module = parent + "." + role
            let rotation = (i == 2 ? outputRotation : inputRotation) ? "hadamard32" : "none"
            books[String(b.bits)] = ["alpha": 0.125, "beta": 0,
                "levels": (0 ..< (1 << b.bits)).map { (Double($0) - Double((1 << b.bits) - 1) / 2) * 0.125 }]
            quant[module] = ["mode": "jangtq2", "bits": b.bits, "rotation": rotation]
            dimensions[module] = .init(experts: experts, input: width, output: width)
            let packedBytes = b.words.flatMap { value -> [UInt8] in
                (0 ..< 4).map { UInt8(truncatingIfNeeded: value >> ($0 * 8)) }
            }
            let scaleBytes = b.scales.flatMap { value -> [UInt8] in
                [UInt8(truncatingIfNeeded: value.bitPattern), UInt8(truncatingIfNeeded: value.bitPattern >> 8)]
            }
            for (suffix, dtype, shape, bytes) in [
                ("tq2_packed", "U32", [experts, width, width * b.bits / 32], packedBytes),
                ("tq2_scales", "F16", [experts, width], scaleBytes),
            ] {
                let name = module + "." + suffix
                headers[name] = ["dtype": dtype, "shape": shape, "data_offsets": [payload.count, payload.count + bytes.count]]
                map[name] = "model.safetensors"
                payload.append(contentsOf: bytes)
            }
        }
        var header = try JSONSerialization.data(withJSONObject: headers, options: .sortedKeys)
        let padded = ((header.count + 8 + 4095) / 4096) * 4096 - 8
        header.append(Data(repeating: 0x20, count: padded - header.count))
        var length = UInt64(header.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(header); data.append(payload)
        try data.write(to: directory.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: ["weight_map": map]).write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
        let config: [String: Any] = ["jangtq": ["version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
            "codebook_family": "odd-cubic", "rotation": "none", "codebooks": books], "quantization": quant]
        let contract = try JANGHFormatContract(configuration: JSONSerialization.data(withJSONObject: config))
        let metadata = try JANGHHeaderAdapter.read(directory: directory, indexName: "model.safetensors.index.json")
        return (try JANGHMappedBanks(directory: directory, metadata: metadata, contract: contract, dimensions: dimensions), banks)
    }
    private func round(_ x: Float, dtype: DType) -> Float {
        if dtype == .float16 { return Float(Float16(x)) }
        if dtype == .bfloat16 {
            let b = x.bitPattern
            return Float(bitPattern: (b &+ 0x7fff &+ ((b >> 16) & 1)) & 0xffff0000)
        }
        return x
    }
    private func h32(_ x: [Float]) -> [Float] {
        x.indices.map { i in
            var sum: Double = 0
            for j in 0 ..< 32 {
                sum += ((i % 32 & j).nonzeroBitCount % 2 == 0 ? 1 : -1) * Double(x[i / 32 * 32 + j])
            }
            return Float(sum / sqrt(32))
        }
    }
    private func project(_ x: [Float], bank: Bank, expert: Int, dequantDType: DType?) -> [Float] {
        (0 ..< width).map { row in
            var total: Double = 0
            for column in 0 ..< width {
                let index = (expert * width + row) * width + column
                var code = 0
                for bit in 0 ..< bank.bits {
                    let offset = index * bank.bits + bit
                    code |= Int((bank.words[offset / 32] >> (offset % 32)) & 1) << bit
                }
                var weight = (Float(code) - Float((1 << bank.bits) - 1) / 2) * 0.125
                weight *= Float(bank.scales[expert * width + row])
                if let dtype = dequantDType { weight = round(weight, dtype: dtype) }
                total += Double(x[column]) * Double(weight)
            }
            return Float(total)
        }
    }
    private func reference(_ input: [Float], ids: [UInt32], scores: [Float], banks: [Bank],
                           dtype: DType, inputRotation: Bool, outputRotation: Bool,
                           backend: JANGHPrefillKernel.Backend) -> [Float] {
        let tokens = input.count / width, prefill = ids.count >= 64
        let effective = JANGHPrefillKernel.resolvedBackend(dtype: dtype, requested: backend)
        var result: [Float] = []
        for token in 0 ..< tokens {
            var x = Array(input[token * width ..< (token + 1) * width])
            if inputRotation { x = h32(x) }
            if prefill { x = x.map { round($0, dtype: dtype) } }
            var contributions = [[Float]]()
            for slot in 0 ..< routes {
                let expert = Int(ids[token * routes + slot])
                var g = project(x, bank: banks[0], expert: expert, dequantDType: prefill ? dtype : nil)
                var u = project(x, bank: banks[1], expert: expert, dequantDType: prefill ? dtype : nil)
                if prefill && effective == .steel {
                    g = g.map { round($0, dtype: dtype) }; u = u.map { round($0, dtype: dtype) }
                }
                var hidden = (0 ..< width).map { i -> Float in
                    let gate = min(g[i], 0.7), up = min(max(u[i], -0.7), 0.7)
                    return Float(Double(gate) / (1 + exp(-Double(gate))) * Double(up))
                }
                if prefill && effective == .steel { hidden = hidden.map { round($0, dtype: dtype) } }
                if outputRotation { hidden = h32(hidden) }
                if prefill { hidden = hidden.map { round($0, dtype: dtype) } }
                var down = project(hidden, bank: banks[2], expert: expert, dequantDType: prefill ? dtype : nil)
                if prefill { down = down.map { round($0, dtype: dtype) } }
                let score = prefill ? round(scores[token * routes + slot], dtype: dtype) : scores[token * routes + slot]
                contributions.append(down.map { prefill ? round($0 * score, dtype: dtype) : $0 * score })
            }
            for column in 0 ..< width {
                // MLX low-precision sum uses F32 accumulation then output cast.
                let sum = contributions.reduce(Float(0)) { $0 + $1[column] }
                result.append(round(sum, dtype: dtype))
            }
        }
        return result
    }
    func testMappedThresholdRouteOrderAndBatchShape() throws {
        try MLXMetalTestLock.withLock {
            for (inputRotation, outputRotation) in [(false, false), (true, false), (false, true), (true, true)] {
                let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
                defer { try? FileManager.default.removeItem(at: dir) }
                let (owner, banks) = try fixture(dir, inputRotation: inputRotation, outputRotation: outputRotation)
                let block = try JANGHRoutedDecodeBlock(banks: owner, parentModule: parent, activationLimit: 0.7)
                let backends: [JANGHPrefillKernel.Backend] = JANGHPrefillKernel.nativeBackend == .nax ? [.steel, .nax] : [.steel]
                for backend in backends {
                    for dtype in [DType.float32, .float16, .bfloat16] {
                        for shape in [[1, 7, width], [2, 4, width], [1, 9, width], [1, 1, width]] {
                            let count = shape[0] * shape[1]
                            let values = (0 ..< count * width).map { round(Float(($0 * 11 + $0 / width) % 31 - 15) / 32, dtype: dtype) }
                            let pattern: [UInt32] = [2, 0, 2, 1, 0, 1, 2, 0]
                            let ids = (0 ..< count * routes).map { pattern[($0 + $0 / routes) % routes] }
                            let scores = (0 ..< count * routes).map { Float(($0 * 3) % 13 - 6) / 16 }
                            let actual = try block.routed(MLXArray(values, shape).asType(dtype),
                                indices: MLXArray(ids, [shape[0], shape[1], routes]),
                                scores: MLXArray(scores, [shape[0], shape[1], routes]), outputDType: dtype, backend: backend)
                            let expected = reference(values, ids: ids, scores: scores, banks: banks, dtype: dtype,
                                inputRotation: inputRotation, outputRotation: outputRotation, backend: backend)
                            XCTAssertEqual(actual.shape, shape); XCTAssertEqual(actual.dtype, dtype)
                            let got = actual.asType(.float32).asArray(Float.self)
                            let tolerance: Float = dtype == .float32 ? 0.0001 : dtype == .float16 ? 0.004 : 0.012
                            for i in got.indices {
                                XCTAssertEqual(got[i], expected[i], accuracy: tolerance * max(1, abs(expected[i])),
                                    "\(backend) \(dtype) \(shape) rotations\(inputRotation)/\(outputRotation) element\(i)")
                            }
                        }
                    }
                }
            }
        }
    }

    func testFlashVerificationScopeAdmissionAndRestoration() {
        XCTAssertFalse(FlashVerificationScope.usesMappedDecode(inputShape: [1, 7, 64], routes: 10))
        FlashVerificationScope.withVerification(inputShape: [1, 7]) {
            XCTAssertTrue(FlashVerificationScope.usesMappedDecode(inputShape: [1, 7, 64], routes: 10))
            XCTAssertTrue(FlashVerificationScope.usesMappedDecode(inputShape: [1, 7, 64], routes: 63))
            for shape in [[7, 64], [2, 7, 64], [1, 6, 64], [1, 8, 64]] {
                XCTAssertFalse(FlashVerificationScope.usesMappedDecode(inputShape: shape, routes: 10))
            }
            for routes in [0, 64, 65] {
                XCTAssertFalse(FlashVerificationScope.usesMappedDecode(inputShape: [1, 7, 64], routes: routes))
            }
            for shape in [[1, 1], [1, 9], [2, 7]] {
                FlashVerificationScope.withVerification(inputShape: shape) {
                    XCTAssertFalse(FlashVerificationScope.usesMappedDecode(inputShape: [1, 7, 64], routes: 10))
                }
                XCTAssertTrue(FlashVerificationScope.usesMappedDecode(inputShape: [1, 7, 64], routes: 10))
            }
        }
        XCTAssertFalse(FlashVerificationScope.usesMappedDecode(inputShape: [1, 7, 64], routes: 10))
    }

    func testFlashScopePreservesUnsupportedGeometryAndPrefillRoutes() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let (banks, _) = try fixture(directory, inputRotation: false, outputRotation: false)
        let layer = try JANGHRoutedExpertLayer(banks: banks, parentModule: parent,
            inputDimensions: width, activationLimit: nil)
        for (shape, routeCount) in [([1, 2, width], 64), ([2, 7, width], 10), ([1, 9, width], 10)] {
            let tokens = shape[0] * shape[1]
            let x = MLXArray((0..<tokens * width).map { Float($0 % 17 - 8) / 128 }, shape).asType(.bfloat16)
            let ids = MLXArray((0..<tokens * routeCount).map { UInt32($0 % experts) }, [shape[0], shape[1], routeCount])
            let scores = MLXArray((0..<tokens * routeCount).map { Float($0 % 7) / 64 }, ids.shape)
            let baseline = layer.callRouted(x, indices: ids, scores: scores)
            let scoped = FlashVerificationScope.withVerification(inputShape: Array(shape.prefix(2))) {
                layer.callRouted(x, indices: ids, scores: scores)
            }
            XCTAssertEqual(scoped.asType(.float32).asArray(Float.self).map(\.bitPattern),
                           baseline.asType(.float32).asArray(Float.self).map(\.bitPattern))
        }
    }

    func testFlashMappedVerificationThresholdIsExactlySerialDecode() throws {
        // Pairwise coverage: every admitted bit width occurs in every projection
        // role; all activation dtypes and rotation combinations are represented.
        let cases: [([Int], DType, Bool, Bool)] = [
            ([2, 3, 4], .float16, false, false),
            ([3, 4, 6], .bfloat16, true, false),
            ([4, 6, 8], .float32, false, true),
            ([6, 8, 2], .float16, true, true),
            ([8, 2, 3], .bfloat16, false, false),
        ]
        for (caseIndex, configuration) in cases.enumerated() {
            let (bits, dtype, inputRotation, outputRotation) = configuration
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            let (banks, _) = try fixture(directory, inputRotation: inputRotation,
                                         outputRotation: outputRotation, bits: bits)
            let layer = try JANGHRoutedExpertLayer(banks: banks, parentModule: parent,
                inputDimensions: width, activationLimit: caseIndex.isMultiple(of: 2) ? nil : 0.7)
            for count in [6, 7, 8] {
                for reversed in [false, true] {
                    let routeCount = 10
                    let values = (0..<count * width).map { Float(($0 * 11 + caseIndex) % 31 - 15) / 128 }
                    let pattern: [UInt32] = [2, 0, 2, 1, 0, 1, 2, 0, 1, 2]
                    let ids = (0..<count * routeCount).map { pattern[reversed ? 9 - $0 % 10 : $0 % 10] }
                    let weights = (0..<count * routeCount).map { index -> Float in
                        let slot = reversed ? 9 - index % 10 : index % 10
                        return Float((slot * 3 + index / 10) % 13) / 64
                    }
                    let x = MLXArray(values, [1, count, width]).asType(dtype)
                    let indices = MLXArray(ids, [1, count, routeCount])
                    let scores = MLXArray(weights, [1, count, routeCount])
                    let actual = FlashVerificationScope.withVerification(inputShape: [1, count]) {
                        layer.callRouted(x, indices: indices, scores: scores)
                    }
                    var expected = [Float]()
                    for row in 0..<count {
                        let rowX = MLXArray(Array(values[row * width..<(row + 1) * width]), [1, 1, width]).asType(dtype)
                        let rowIDs = MLXArray(Array(ids[row * routeCount..<(row + 1) * routeCount]), [1, 1, routeCount])
                        let rowScores = MLXArray(Array(weights[row * routeCount..<(row + 1) * routeCount]), [1, 1, routeCount])
                        expected += layer.callRouted(rowX, indices: rowIDs, scores: rowScores).asType(.float32).asArray(Float.self)
                    }
                    XCTAssertEqual(actual.shape, [1, count, width])
                    XCTAssertEqual(actual.dtype, dtype)
                    // Comparing F32 bit patterns preserves signed zero and exact
                    // F16/BF16 values without a tolerance or reduction oracle.
                    XCTAssertEqual(actual.asType(.float32).asArray(Float.self).map(\.bitPattern),
                                   expected.map(\.bitPattern), "case \(caseIndex), S\(count), reordered \(reversed)")
                }
            }
        }
    }
}
