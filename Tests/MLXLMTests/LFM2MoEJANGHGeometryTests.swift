import Foundation
import MLX
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon

/// Real-bank component proof at LFM's 2048/1536 widths; not model generation or a speed benchmark.
final class LFM2MoEJANGHGeometryTests: XCTestCase {
    private struct Expert {
        let words: [UInt32]
        let scales: [Float16]
        let input: Int
        let output: Int
        let bits: Int
        let levels: [Float]
    }

    private func read(_ source: JANGHMappedBanks.SourceLease,
                      location: JANGHTensorIndexPlan.Location, expert: Int) throws -> Data {
        try source.withValidatedFile(for: location) { descriptor, header in
            let bytes = location.byteCount / header.shape[0]
            let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
            try handle.seek(toOffset: UInt64(location.fileOffset + expert * bytes))
            let data = try XCTUnwrap(handle.read(upToCount: bytes))
            XCTAssertEqual(data.count, bytes)
            return data
        }
    }

    private func expert(_ source: JANGHMappedBanks.SourceLease, module: String, index: Int) throws -> Expert {
        let locations = try XCTUnwrap(source.plan.projections[module])
        let declaration = try XCTUnwrap(source.contract.projections[module])
        let header = try XCTUnwrap(source.metadata.shards[locations.packed.shard]?.tensors[locations.packed.tensor])
        let packed = try read(source, location: locations.packed, expert: index)
        let scales = try read(source, location: locations.scales, expert: index)
        let words = packed.withUnsafeBytes { bytes in
            (0 ..< packed.count / 4).map { UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self)) }
        }
        let scaleValues = scales.withUnsafeBytes { bytes in
            (0 ..< scales.count / 2).map { Float16(bitPattern: UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 2, as: UInt16.self))) }
        }
        return Expert(words: words, scales: scaleValues, input: header.shape[2] * 32 / declaration.bits,
                      output: header.shape[1], bits: declaration.bits,
                      levels: try XCTUnwrap(source.contract.codebooks[declaration.bits]).levels.map(Float.init))
    }

    private func h32(_ values: [Float]) -> [Float] {
        values.indices.map { index in
            var sum: Double = 0
            for column in 0 ..< 32 {
                let sign: Double = ((index % 32) & column).nonzeroBitCount.isMultiple(of: 2) ? 1 : -1
                sum += sign * Double(values[index / 32 * 32 + column])
            }
            return Float(sum / sqrt(32))
        }
    }

        private func bf16(_ value: Float) -> Float {
        let bits = value.bitPattern
        return Float(bitPattern: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) & 0xffff0000)
    }

    private func project(_ values: [Float], expert: Expert, roundWeights: Bool = false) -> [Float] {
        (0 ..< expert.output).map { row in
            var sum: Double = 0
            for column in 0 ..< expert.input {
                // Independent bit-by-bit oracle, including 3-bit word crossings.
                let start = (row * expert.input + column) * expert.bits
                var code: UInt32 = 0
                for bit in 0 ..< expert.bits {
                    let offset = start + bit
                    code |= ((expert.words[offset / 32] >> (offset % 32)) & 1) << bit
                }
                                if roundWeights {
                    let weight = bf16(expert.levels[Int(code)] * Float(expert.scales[row]))
                    sum += Double(values[column]) * Double(weight)
                } else {
                    sum += Double(values[column]) * Double(expert.levels[Int(code)])
                }
            }
            return roundWeights ? Float(sum) : Float(sum * Double(expert.scales[row]))
        }
    }

    func testRealLFMHadamardGateUpAndDownAgainstIndependentScalarOracle() throws {
        guard let path = ProcessInfo.processInfo.environment["VMLX_LFM_JANGH_NUMERIC_BUNDLE"] else {
            throw XCTSkip("set VMLX_LFM_JANGH_NUMERIC_BUNDLE for the bounded actual-bank numerical audit")
        }
        try MLXMetalTestLock.withLock {
            let directory = URL(fileURLWithPath: path)
            let preparation = try XCTUnwrap(LFM2MoEJANGHPreparation.loadIfDeclared(directory: directory,
                configurationData: Data(contentsOf: directory.appendingPathComponent("config.json"))))
            let parent = "model.layers.2.feed_forward.switch_mlp"
            let source = preparation.banks.source
            let owner = try JANGHMappedBanks(source: source, modules: Set(["gate_proj", "up_proj", "down_proj"].map { parent + "." + $0 }))
            let block = try JANGHRoutedDecodeBlock(banks: owner, parentModule: parent, activationLimit: nil)
            let x = (0 ..< 2048).map { Float(($0 * 17) % 127 - 63) / 100 }
            let rotated = h32(x)
            let weights: [Float] = [0.125, 0.25, 0.375, 0.25]
            var contributions: [[Float]] = []
            for index in 0 ..< 4 {
                let gate = try expert(source, module: parent + ".gate_proj", index: index)
                let up = try expert(source, module: parent + ".up_proj", index: index)
                let down = try expert(source, module: parent + ".down_proj", index: index)
                XCTAssertEqual(gate.input, 2048); XCTAssertEqual(gate.output, 1536)
                XCTAssertEqual(down.input, 1536); XCTAssertEqual(down.output, 2048)
                XCTAssertEqual(gate.bits, 2); XCTAssertEqual(up.bits, 2); XCTAssertEqual(down.bits, 3)
                let g = project(rotated, expert: gate), u = project(rotated, expert: up)
                let hidden = (0 ..< 1536).map { Float(Double(g[$0]) / (1 + exp(-Double(g[$0]))) * Double(u[$0])) }
                contributions.append(project(h32(hidden), expert: down))
            }
            let expected = (0 ..< 2048).map { column in
                (0 ..< 4).reduce(Float(0)) { $0 + contributions[$1][column] * weights[$1] }
            }
            for rows in [1, 16] {
                let input = MLXArray(Array(repeating: x, count: rows).flatMap { $0 }, [rows, 2048])
                let ids = MLXArray(Array(repeating: [UInt32(0), 1, 2, 3], count: rows).flatMap { $0 }, [rows, 4])
                let scores = MLXArray(Array(repeating: weights, count: rows).flatMap { $0 }, [rows, 4])
                let result = try block.routed(input, indices: ids, scores: scores, outputDType: .float32)
                let values = result.asArray(Float.self)
                var maxError: Float = 0
                for row in 0 ..< rows {
                    for column in 0 ..< 2048 {
                        let actual = values[row * 2048 + column]
                        maxError = max(maxError, abs(actual - expected[column]))
                        XCTAssertEqual(actual, expected[column], accuracy: max(0.0001, abs(expected[column]) * 0.002),
                                       "rows=\(rows) column=\(column)")
                    }
                }
                print("LFM_JANGH_REAL_BANK rows=\(rows) dtype=f32 max_abs=\(maxError)")
            }

            let roundedX = x.map(bf16)
            for rows in [1, 16] {
                let prefill = rows > 1
                let backend = JANGHPrefillKernel.resolvedBackend(dtype: .bfloat16, requested: JANGHPrefillKernel.nativeBackend)
                let basis = prefill ? h32(roundedX).map(bf16) : h32(roundedX)
                var parts: [[Float]] = []
                for index in 0 ..< 4 {
                    let gate = try expert(source, module: parent + ".gate_proj", index: index)
                    let up = try expert(source, module: parent + ".up_proj", index: index)
                    let down = try expert(source, module: parent + ".down_proj", index: index)
                    var g = project(basis, expert: gate, roundWeights: prefill)
                    var u = project(basis, expert: up, roundWeights: prefill)
                    if prefill && backend == .steel { g = g.map(bf16); u = u.map(bf16) }
                    var hidden = (0 ..< 1536).map { Float(Double(g[$0]) / (1 + exp(-Double(g[$0]))) * Double(u[$0])) }
                    if prefill && backend == .steel { hidden = hidden.map(bf16) }
                    hidden = h32(hidden)
                    if prefill { hidden = hidden.map(bf16) }
                    var projected = project(hidden, expert: down, roundWeights: prefill)
                    if prefill { projected = projected.map(bf16) }
                    parts.append(projected)
                }
                let reference = (0 ..< 2048).map { column in
                    let summed = (0 ..< 4).reduce(Float(0)) { sum, slot in
                        let contribution = parts[slot][column] * weights[slot]
                        return sum + (prefill ? bf16(contribution) : contribution)
                    }
                    return bf16(summed)
                }
                let input = MLXArray(Array(repeating: x, count: rows).flatMap { $0 }, [rows, 2048]).asType(.bfloat16)
                let ids = MLXArray(Array(repeating: [UInt32(0), 1, 2, 3], count: rows).flatMap { $0 }, [rows, 4])
                let scores = MLXArray(Array(repeating: weights, count: rows).flatMap { $0 }, [rows, 4])
                let actual = try block.routed(input, indices: ids, scores: scores, outputDType: .bfloat16)
                    .asType(.float32).asArray(Float.self)
                var maxError: Float = 0
                for row in 0 ..< rows {
                    for column in 0 ..< 2048 {
                        let value = actual[row * 2048 + column]
                        maxError = max(maxError, abs(value - reference[column]))
                        XCTAssertEqual(value, reference[column], accuracy: max(0.00005, abs(reference[column]) * 0.016),
                                       "bf16 rows=\(rows) column=\(column)")
                    }
                }
                print("LFM_JANGH_REAL_BANK rows=\(rows) dtype=bf16 backend=\(backend.rawValue) max_abs=\(maxError)")
            }
        }
    }
}
