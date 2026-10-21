// Copyright © 2026 Jinho Jang. All rights reserved.
//
// MappedSafetensorsShardTests — verify the safetensors header parser, byte
// range lookup, and exact reads work without a real bundle.
//
// We synthesize a minimal safetensors-format file with two tensors:
//   "expert_0": F32, shape=[2, 4], 32 bytes
//   "expert_1": F32, shape=[2, 4], 32 bytes
// laid out as:
//   bytes 0..8:    header_size (LE uint64)
//   bytes 8..N:    JSON header
//   bytes N..N+32: expert_0 data
//   bytes N+32..:  expert_1 data
//
// Then test:
//   1. Both tensors are discovered.
//   2. byteRange returns offsets matching what we wrote.
//   3. bytes(in:) returns the exact bytes we wrote.

import Foundation
import Testing
@testable import MLXLMCommon

@Suite("MappedSafetensorsShard")
struct MappedSafetensorsShardTests {

    // MARK: - Helpers

    static func writeSyntheticSafetensors(at url: URL) throws {
        // Each tensor is 32 bytes (F32, 2×4 = 8 floats × 4 bytes).
        let tensor0 = (0..<32).map { UInt8($0) }
        let tensor1 = (0..<32).map { UInt8($0 + 100) }

        let header: [String: Any] = [
            "expert_0": [
                "dtype": "F32",
                "shape": [2, 4],
                "data_offsets": [0, 32],
            ],
            "expert_1": [
                "dtype": "F32",
                "shape": [2, 4],
                "data_offsets": [32, 64],
            ],
        ]
        let headerJSON = try JSONSerialization.data(
            withJSONObject: header, options: [.sortedKeys])
        let headerSize = UInt64(headerJSON.count)

        var fileBytes = Data()
        fileBytes.append(contentsOf: withUnsafeBytes(of: headerSize.littleEndian) { Array($0) })
        fileBytes.append(headerJSON)
        fileBytes.append(contentsOf: tensor0)
        fileBytes.append(contentsOf: tensor1)

        try fileBytes.write(to: url)
    }

    // MARK: - Tests

    @Test("parses synthetic safetensors header + indexes both tensors")
    func parsesHeader() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mmap-shard-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try Self.writeSyntheticSafetensors(at: tmp)

        let shard = try MappedSafetensorsShard(path: tmp)
        #expect(shard.tensors.count == 2)
        #expect(shard.tensors["expert_0"] != nil)
        #expect(shard.tensors["expert_1"] != nil)
    }

    @Test("byteRange returns offsets matching what we wrote")
    func byteRangeOffsets() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mmap-shard-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try Self.writeSyntheticSafetensors(at: tmp)

        let shard = try MappedSafetensorsShard(path: tmp)
        let r0 = try #require(shard.byteRange(for: "expert_0"))
        let r1 = try #require(shard.byteRange(for: "expert_1"))

        // r0 should come before r1, both length 32
        #expect(r0.upperBound - r0.lowerBound == 32)
        #expect(r1.upperBound - r1.lowerBound == 32)
        #expect(r0.upperBound == r1.lowerBound)

        // Absolute offset = 8 + headerSize + relative_offset.
        let header8 = shard.baseAddress.load(as: UInt64.self).littleEndian
        #expect(r0.lowerBound == 8 + header8)
        #expect(r1.lowerBound == 8 + header8 + 32)
    }

    @Test("bytes(in:) returns exact bytes we wrote")
    func bytesRoundTrip() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mmap-shard-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try Self.writeSyntheticSafetensors(at: tmp)

        let shard = try MappedSafetensorsShard(path: tmp)
        let r0 = try #require(shard.byteRange(for: "expert_0"))
        let buf0 = shard.bytes(in: r0)
        #expect(buf0.count == 32)
        let bytes0 = Array(buf0)
        #expect(bytes0 == (0..<32).map { UInt8($0) })

        let r1 = try #require(shard.byteRange(for: "expert_1"))
        let bytes1 = Array(shard.bytes(in: r1))
        #expect(bytes1 == (0..<32).map { UInt8($0 + 100) })
    }

    @Test("F_NOCACHE pread returns exact tensor bytes without touching mmap payload")
    func noCacheReadRoundTrip() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mmap-shard-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try Self.writeSyntheticSafetensors(at: tmp)

        let shard = try MappedSafetensorsShard(path: tmp, noCacheIO: true)
        #expect(shard.noCacheIOEnabled)
        let range = try #require(shard.byteRange(for: "expert_1"))
        let bytes = try shard.readBytes(range: range)
        #expect(Array(bytes) == (0..<32).map { UInt8($0 + 100) })
    }

    @Test("sniffTensorNames returns names without mmap'ing data")
    func sniffNames() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mmap-shard-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try Self.writeSyntheticSafetensors(at: tmp)

        let names = try #require(MappedSafetensorsShard.sniffTensorNames(at: tmp))
        #expect(names.count == 2)
        #expect(names.contains("expert_0"))
        #expect(names.contains("expert_1"))
        // Sniff filters __metadata__ even if present.
        #expect(!names.contains("__metadata__"))
    }

    @Test("unknown tensor returns nil from byteRange")
    func unknownTensorReturnsNil() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mmap-shard-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try Self.writeSyntheticSafetensors(at: tmp)

        let shard = try MappedSafetensorsShard(path: tmp)
        #expect(shard.byteRange(for: "nonexistent_tensor") == nil)
        #expect(shard.descriptor(for: "nonexistent_tensor") == nil)
    }
}
