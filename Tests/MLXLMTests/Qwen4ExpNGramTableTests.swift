// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXVLM

@Suite("qwen4_exp page-granular PLE table")
struct Qwen4ExpNGramTableTests {
    private func pack(_ codes: [UInt32], bits: Int) -> [UInt32] {
        let wordCount = (codes.count * bits + 31) / 32
        var words = [UInt32](repeating: 0, count: wordCount)
        for (index, code) in codes.enumerated() {
            let offset = index * bits
            let word = offset / 32
            let shift = offset % 32
            let value = UInt64(code) << shift
            words[word] |= UInt32(truncatingIfNeeded: value)
            if shift + bits > 32 {
                words[word + 1] |= UInt32(truncatingIfNeeded: value >> 32)
            }
        }
        return words
    }

    @Test("3-bit LSB packing dequantizes across UInt32 boundaries")
    func threeBitBoundary() {
        let codes = (0 ..< 64).map { UInt32(($0 * 5 + 3) & 7) }
        let result = Qwen4ExpNGramTable.dequantizeRow(
            packedWords: pack(codes, bits: 3),
            scales: [0.5, 2.0], biases: [-1.0, 3.0],
            dimensions: 64,
            spec: .init(bits: 3, groupSize: 32))

        let expected = codes.enumerated().map { index, code in
            Float(code) * (index < 32 ? 0.5 : 2.0) + (index < 32 ? -1.0 : 3.0)
        }
        #expect(result == expected)
    }

    @Test("2-bit rows dequantize all four codes")
    func twoBitGroups() {
        let codes: [UInt32] = [0, 1, 2, 3, 3, 2, 1, 0]
        let result = Qwen4ExpNGramTable.dequantizeRow(
            packedWords: pack(codes, bits: 2),
            scales: [2.0, 0.5], biases: [-1.0, 4.0],
            dimensions: 8,
            spec: .init(bits: 2, groupSize: 4))
        #expect(result == [-1, 1, 3, 5, 5.5, 5, 4.5, 4])
    }

    @Test("4-bit rows use independent affine groups")
    func fourBitGroups() {
        let codes: [UInt32] = [0, 1, 2, 3, 12, 13, 14, 15]
        let result = Qwen4ExpNGramTable.dequantizeRow(
            packedWords: pack(codes, bits: 4),
            scales: [1.0, 0.25], biases: [10.0, -2.0],
            dimensions: 8,
            spec: .init(bits: 4, groupSize: 4))
        #expect(result == [10, 11, 12, 13, 1, 1.25, 1.5, 1.75])
    }

    @Test("real gather uses exact F_NOCACHE row reads and records SSD telemetry")
    func diskBackedGatherTelemetry() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4-ple-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let base = "language_model.layers.0.ple.ngram_embedding.shards.0"
        let weightName = base + ".weight"
        let scaleName = base + ".scales"
        let biasName = base + ".biases"
        let shardName = "model.safetensors"
        let header: [String: Any] = [
            weightName: ["dtype": "U32", "shape": [2, 1], "data_offsets": [0, 8]],
            scaleName: ["dtype": "F16", "shape": [2, 2], "data_offsets": [8, 16]],
            biasName: ["dtype": "F16", "shape": [2, 2], "data_offsets": [16, 24]],
        ]
        let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var bytes = Data()
        var headerLength = UInt64(headerData.count).littleEndian
        bytes.append(contentsOf: withUnsafeBytes(of: &headerLength) { Array($0) })
        bytes.append(headerData)
        for word in [pack(Array(0..<8).map(UInt32.init), bits: 4)[0],
                     pack(Array(8..<16).map(UInt32.init), bits: 4)[0]]
        {
            var little = word.littleEndian
            bytes.append(contentsOf: withUnsafeBytes(of: &little) { Array($0) })
        }
        for value in [Float16(1), Float16(1), Float16(0.5), Float16(2),
                      Float16(0), Float16(0), Float16(1), Float16(-1)]
        {
            var little = value.bitPattern.littleEndian
            bytes.append(contentsOf: withUnsafeBytes(of: &little) { Array($0) })
        }
        try bytes.write(to: directory.appendingPathComponent(shardName))

        let index: [String: Any] = [
            "weight_map": [weightName: shardName, scaleName: shardName, biasName: shardName]
        ]
        try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        let config: [String: Any] = [
            "jang_config": [
                "bit_map": [
                    "language_model.layers.*.ple.ngram_embedding.shards.0.weight": [
                        "bits": 4, "group_size": 4,
                    ]
                ]
            ]
        ]
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("config.json"))

        let table = try Qwen4ExpNGramTable(
            modelDirectory: directory, layerIndex: 0, shardCount: 1)
        let values = try table.gather([1], parallelRows: false)
        #expect(values == [5, 5.5, 6, 6.5, 23, 25, 27, 29])

        let parityRows: [Int64] = [1, 0, 1]
        let sequential = try table.gather(parityRows, parallelRows: false)
        let parallel = try table.gather(parityRows, parallelRows: true)
        #expect(parallel == sequential)
        #expect(Array(parallel[0 ..< 8]) == Array(parallel[16 ..< 24]))
        #expect(throws: Qwen4ExpNGramTableError.self) {
            _ = try table.gather([table.rowCount], parallelRows: false)
        }
        #expect(throws: Qwen4ExpNGramTableError.self) {
            _ = try table.gather([table.rowCount], parallelRows: true)
        }
        #expect(table.ioStats() == .init(
            gatherCalls: 3,
            rowsRead: 7,
            payloadBytesRead: 60,
            backingFileCount: 1,
            backingFileBytes: UInt64(bytes.count),
            noCacheFileCount: 1))

        for parallel in [false, true] {
            let before = table.ioStats()
            #expect(try table.gather([], parallelRows: parallel).isEmpty)
            let repeated = try table.gather([1, 1, 1, 1], parallelRows: parallel)
            #expect(repeated == Array(repeating: values, count: 4).flatMap { $0 })
            #expect(table.ioStats().payloadBytesRead - before.payloadBytesRead == 12)
            let uniqueBefore = table.ioStats()
            let unique = try table.gather([0, 1], parallelRows: parallel)
            #expect(Array(unique[8..<16]) == values)
            #expect(table.ioStats().payloadBytesRead - uniqueBefore.payloadBytesRead == 24)
            let invalidBefore = table.ioStats()
            #expect(throws: Qwen4ExpNGramTableError.self) {
                _ = try table.gather([1, -1, 1], parallelRows: parallel)
            }
            #expect(table.ioStats() == invalidBefore)
        }

        // Opt-in host-I/O diagnostic, not a model throughput benchmark. Keep
        // identical fixture, row order and schedules across before/after runs.
        if ProcessInfo.processInfo.environment["VMLX_PLE_IO_BENCH"] == "1" {
            for parallel in [false, true] {
                for (label, rows) in [
                    ("single", [Int64(1)]),
                    ("unique", [Int64(0), 1]),
                    ("duplicate512", (0..<512).map { Int64($0 % 2) }),
                ] {
                    let before = table.ioStats()
                    let start = DispatchTime.now().uptimeNanoseconds
                    var checksum: Float = 0
                    for _ in 0..<32 {
                        let result = try table.gather(rows, parallelRows: parallel)
                        checksum += result.last ?? 0
                    }
                    let elapsed = DispatchTime.now().uptimeNanoseconds - start
                    let readBytes = table.ioStats().payloadBytesRead - before.payloadBytesRead
                    print("PLE_IO_BENCH schedule=\(parallel ? "parallel" : "sequential") "
                        + "case=\(label) iterations=32 ns=\(elapsed) bytes=\(readBytes) checksum=\(checksum)")
                }
            }
        }
    }

    @Test("opt-in real PLE table row parity and bounded host I/O timing")
    func realTableRowTiming() throws {
        guard let path = ProcessInfo.processInfo.environment["VMLX_PLE_TABLE_BENCH_MODEL"] else {
            return
        }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        let config = try JSONDecoder().decode(
            Qwen4ExpConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let layer = try #require(config.extras.pleLayerIds.first)
        let table = try Qwen4ExpNGramTable(
            modelDirectory: directory, layerIndex: layer - 1,
            shardCount: config.extras.splitNgramParts)
        #expect(table.rowCount >= 512)
        guard table.rowCount >= 512 else { return }
        let uniqueRows = (0..<512).map { Int64($0) * (table.rowCount / 512) }
        // An independently gathered row at a time checks the scatter ordering,
        // including shard boundaries, without loading any model weights.
        let reference = try uniqueRows.flatMap { try table.gather([$0], parallelRows: false) }
        let duplicateRows = (0..<512).map { uniqueRows[$0 % 8] }
        let duplicateReference = (0..<512).flatMap { row in
            Array(reference[(row % 8) * table.dimensions..<(row % 8 + 1) * table.dimensions])
        }
        for parallel in [false, true] {
            #expect(try table.gather(uniqueRows, parallelRows: parallel) == reference)
            #expect(try table.gather(duplicateRows, parallelRows: parallel) == duplicateReference)
            for round in 0..<3 {
                for (label, rows) in [("unique512", uniqueRows), ("duplicate512", duplicateRows)] {
                    let before = table.ioStats()
                    let start = DispatchTime.now().uptimeNanoseconds
                    var checksum: Float = 0
                    for _ in 0..<8 {
                        checksum += try table.gather(rows, parallelRows: parallel).last ?? 0
                    }
                    let ns = DispatchTime.now().uptimeNanoseconds - start
                    print("PLE_REAL_BENCH round=\(round) parallel=\(parallel) case=\(label) "
                        + "iterations=8 ns=\(ns) bytes=\(table.ioStats().payloadBytesRead - before.payloadBytesRead) "
                        + "checksum=\(checksum) dimensions=\(table.dimensions)")
                }
            }
        }
    }

    @Test("local Qwen4Exp variants validate every PLE shard and use exact SSD reads")
    func localVariantPLEMatrix() throws {
        guard let rootPath = ProcessInfo.processInfo.environment["QWEN4_VARIANT_ROOT"] else {
            return
        }
        let root = URL(fileURLWithPath: rootPath, isDirectory: true)
        for variant in [
            "JANG_1L", "JANG_2L", "JANG_4M", "JANG_4M-aligned", "JANG_4S", "JANG_6S",
        ] {
            let directory = root.appendingPathComponent(
                "Qwen3.8-Flash-Next-\(variant)", isDirectory: true)
            let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
            let config = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: configData)
            let table = try Qwen4ExpNGramTable(
                modelDirectory: directory,
                layerIndex: config.extras.pleLayerIds[0] - 1,
                shardCount: config.extras.splitNgramParts)
            let values = try table.gather([0, table.rowCount / 2, table.rowCount - 1])
            let stats = table.ioStats()
            #expect(values.count == 3 * table.dimensions)
            #expect(values.allSatisfy { $0.isFinite })
            #expect(stats.gatherCalls == 1)
            #expect(stats.rowsRead == 3)
            #expect(stats.payloadBytesRead > 0)
            #expect(stats.noCacheFileCount == stats.backingFileCount)
        }
    }
}
