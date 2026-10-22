import Foundation
import MLX
import Testing
@testable import MLXLMCommon

@Suite(.serialized)
struct ResidentSafetensorsReaderTests {
    @Test func cancellationPreventsAnyReadOrAllocation() async {
        let task = Task {
            while !Task.isCancelled { await Task.yield() }
            _ = try ResidentSafetensorsReader.load(
                url: URL(fileURLWithPath: "/nonexistent-cancelled-resident-proof"), excludingKeys: [])
        }
        task.cancel()
        do {
            _ = try await task.value
            Issue.record("cancelled reader unexpectedly returned")
        } catch {
            #expect(error is CancellationError)
        }
    }
    @Test func unalignedPayloadCrossesReadChunkAndPartialEOF() throws {
        try MLXMetalTestLock.withLock {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("resident-chunks-\(UUID()).safetensors")
            defer { try? FileManager.default.removeItem(at: url) }
            // A little over the 8-MiB physical read chunk, with a partial EOF page.
            let expected = MLXArray((0..<(2 * 1024 * 1024 + 3)).map { UInt32($0) })
            try MLX.save(arrays: ["w": expected], url: url)
            let (loaded, _) = try ResidentSafetensorsReader.load(url: url, excludingKeys: [])
            let actual = try #require(loaded["w"])
            #expect(actual.shape == expected.shape && actual.dtype == expected.dtype)
            #expect(MLX.all(actual .== expected).item(Bool.self))
        }
    }

    @Test func directRegionsOwnBytesAfterSourceMutation() throws {
        try MLXMetalTestLock.withLock {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("resident-regions-\(UUID())")
            defer { try? FileManager.default.removeItem(at: url) }
            let original = (0..<(96 * 1024 + 19)).map { UInt8(truncatingIfNeeded: $0 * 17 + 3) }
            try Data(original).write(to: url)
            let file = try FileHandle(forReadingFrom: url)
            var loaded: [(Int, Int, MLXArray)] = []
            for (offset, count) in [(0, 1), (1, 33), (8191, 32777), (16384, 32768), (65537, 32786)] {
                let array = try ResidentSafetensorsReader.loadRegion(file: file,
                    offset: UInt64(offset), length: count, shape: [count], dtype: .uint8)
                #expect(!WeightBacking.sharesMappedShard(array))
                loaded.append((offset, count, array))
            }
            try file.close()
            let writer = try FileHandle(forWritingTo: url)
            try writer.write(contentsOf: Data(repeating: 0, count: original.count))
            try writer.close()
            for (offset, count, array) in loaded {
                #expect(array.asArray(UInt8.self) == Array(original[offset..<(offset + count)]))
            }
        }
    }

    @Test func directRegionsRejectInvalidSpans() throws {
        try MLXMetalTestLock.withLock {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("resident-bounds-\(UUID())")
            defer { try? FileManager.default.removeItem(at: url) }
            try Data(repeating: 42, count: 32).write(to: url)
            for (offset, length, shape) in [(UInt64.max, 4, [1]), (31, 4, [1]), (0, 8, [1]), (0, 4, [-1])] {
                #expect(throws: (any Error).self) {
                    _ = try ResidentSafetensorsReader.loadRegion(url: url, offset: offset,
                        length: length, shape: shape, dtype: .uint32)
                }
            }
        }
    }

    @Test func nullMetadataMatchesAbsentMetadata() throws {
        // Real Flash Next shards 6, 7 and 13 carry __metadata__: null.
        // Exclude the payload: this regression exercises header parsing only.
        for metadata in [nil, NSNull(), [:] as NSDictionary] as [Any?] {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("resident-metadata-\(UUID()).safetensors")
            defer { try? FileManager.default.removeItem(at: url) }
            var fields: [String: Any] = [
                "w": ["dtype": "U32", "shape": [1], "data_offsets": [0, 4]],
            ]
            if let metadata { fields["__metadata__"] = metadata }
            let header = try JSONSerialization.data(withJSONObject: fields)
            var length = UInt64(header.count).littleEndian
            var data = withUnsafeBytes(of: &length) { Data($0) }
            data.append(header)
            data.append(Data(repeating: 0, count: 4))
            try data.write(to: url)
            let (arrays, result) = try ResidentSafetensorsReader.load(url: url, excludingKeys: ["w"])
            #expect(arrays.isEmpty && result.isEmpty)
        }
    }

    @Test func mixedTypesAndExclusions() throws {
        try MLXMetalTestLock.withLock {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("resident-reader-\(UUID()).safetensors")
            defer { try? FileManager.default.removeItem(at: url) }
            let arrays: [String: MLXArray] = [
                "packed": MLXArray([UInt32(0), UInt32.max, 123]),
                "scale": MLXArray([Float(0.125), -2, 3]).asType(.float16),
                "bf": MLXArray([Float(0.5), -7, 9]).asType(.bfloat16),
                "scalar": MLXArray(Int64(42)),
                "excluded": MLXArray([Float(123)]),
            ]
            try MLX.save(arrays: arrays, metadata: ["format": "pt"], url: url)
            let before = try Data(contentsOf: url)
            let (result, metadata) = try ResidentSafetensorsReader.load(url: url, excludingKeys: ["excluded"])
            #expect(Set(result.keys) == Set(arrays.keys).subtracting(["excluded"]))
            #expect(metadata == ["format": "pt"])
            #expect(try Data(contentsOf: url) == before)
            for (key, value) in result {
                let reference = arrays[key]!
                #expect(value.shape == reference.shape && value.dtype == reference.dtype)
                #expect(MLX.all(value.reshaped([-1]).view(dtype: .uint8) .== reference.reshaped([-1]).view(dtype: .uint8)).item(Bool.self))
            }
        }
    }

    @Test func emptyTensorPreservesShapeAndDtype() throws {
        try MLXMetalTestLock.withLock {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("resident-empty-\(UUID()).safetensors")
            defer { try? FileManager.default.removeItem(at: url) }
            let header = try JSONSerialization.data(withJSONObject: [
                "empty": ["dtype": "BF16", "shape": [2, 0, 3], "data_offsets": [0, 0]],
            ])
            var length = UInt64(header.count).littleEndian
            var data = withUnsafeBytes(of: &length) { Data($0) }
            data.append(header)
            try data.write(to: url)
            let (arrays, _) = try ResidentSafetensorsReader.load(url: url, excludingKeys: [])
            let empty = try #require(arrays["empty"])
            #expect(empty.shape == [2, 0, 3] && empty.dtype == .bfloat16 && empty.size == 0)
        }
    }

    @Test func malformedHeadersThrowBeforeAllocation() throws {
        let entries: [[String: Any]] = [
            ["dtype": "U32", "shape": [-1], "data_offsets": [0, 4]],
            ["dtype": "U32", "shape": [Int.max, 4], "data_offsets": [0, 4]],
            ["dtype": "U32", "shape": [0, Int(Int32.max) + 1], "data_offsets": [0, 0]],
            ["dtype": "U32", "shape": [2], "data_offsets": [0, 4]],
            ["dtype": "U32", "shape": [1], "data_offsets": [0, 8]],
            ["dtype": "UNKNOWN", "shape": [1], "data_offsets": [0, 4]],
            ["dtype": "U32", "shape": [1], "data_offsets": [4, 0]],
        ]
        for entry in entries {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("resident-invalid-\(UUID()).safetensors")
            defer { try? FileManager.default.removeItem(at: url) }
            let header = try JSONSerialization.data(withJSONObject: ["w": entry])
            var length = UInt64(header.count).littleEndian
            var data = withUnsafeBytes(of: &length) { Data($0) }
            data.append(header)
            data.append(Data(repeating: 0, count: 4))
            try data.write(to: url)
            #expect(throws: (any Error).self) {
                _ = try ResidentSafetensorsReader.load(url: url, excludingKeys: [])
            }
        }
    }
}
