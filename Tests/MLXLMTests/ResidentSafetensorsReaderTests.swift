import Foundation
import MLX
import Testing
@testable import MLXLMCommon

@Suite(.serialized)
struct ResidentSafetensorsReaderTests {
    @Test func nullMetadataMatchesAbsentMetadata() throws {
        // Real Flash Next shards 6, 7 and 13 carry __metadata__: null.
        // Exclude the payload: this regression exercises header parsing only.
        for metadata in [NSNull(), [:] as NSDictionary] as [Any] {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("resident-metadata-\(UUID()).safetensors")
            defer { try? FileManager.default.removeItem(at: url) }
            let header = try JSONSerialization.data(withJSONObject: [
                "__metadata__": metadata,
                "w": ["dtype": "U32", "shape": [1], "data_offsets": [0, 4]],
            ])
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
            let (result, metadata) = try ResidentSafetensorsReader.load(url: url, excludingKeys: ["excluded"])
            #expect(Set(result.keys) == Set(arrays.keys).subtracting(["excluded"]))
            #expect(metadata == ["format": "pt"])
            for (key, value) in result {
                let reference = arrays[key]!
                #expect(value.shape == reference.shape && value.dtype == reference.dtype)
                #expect(MLX.all(value.reshaped([-1]).view(dtype: .uint8) .== reference.reshaped([-1]).view(dtype: .uint8)).item(Bool.self))
            }
        }
    }

    @Test func malformedHeadersThrowBeforeAllocation() throws {
        let entries: [[String: Any]] = [
            ["dtype": "U32", "shape": [-1], "data_offsets": [0, 4]],
            ["dtype": "U32", "shape": [Int.max, 4], "data_offsets": [0, 4]],
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
