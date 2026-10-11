import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// `WeightBacking` decides whether a runtime re-layout of a weight (fused decode projection, lane
/// tiling) can REPLACE it. Windows into a mapped safetensors shard cannot be released, so a copy of
/// them is a permanent duplicate; arrays with their own allocation can be replaced.
@Suite(.serialized)
struct WeightBackingTests {
    private func withMappedLoad<T>(_ body: () throws -> T) rethrows -> T {
        let settings = [
            "MLX_SAFETENSORS_MMAP": "1", "MLX_SAFETENSORS_MMAP_START_COLD": "0",
            "MLX_SAFETENSORS_MMAP_TENSOR_BUFFERS": "0", "VMLX_COPY_MAPPED_WEIGHTS": "0",
        ]
        let previous = settings.keys.map { ($0, getenv($0).map { String(cString: $0) }) }
        for (key, value) in settings { setenv(key, value, 1) }
        defer {
            for (key, value) in previous {
                if let value { setenv(key, value, 1) } else { unsetenv(key) }
            }
        }
        return try body()
    }

    /// A shard laid out like a converted bundle: page-aligned data segment and tensor offsets, so
    /// the mmap loader maps it instead of realigning (copying) the tensors. Values are 0 ..< count.
    private func writeAlignedShard(at url: URL, tensors: [(String, Int)]) throws {
        let page = 16_384
        var header: [String: Any] = [:]
        var offset = 0
        var payload = Data()
        for (name, count) in tensors {
            let bytes = count * 4
            header[name] = ["dtype": "U32", "shape": [count / 256, 256], "data_offsets": [offset, offset + bytes]]
            var values = (0 ..< count).map { UInt32($0) }
            payload.append(Data(bytes: &values, count: bytes))
            offset += bytes
            let padded = (offset + page - 1) / page * page
            payload.append(Data(count: padded - offset))
            offset = padded
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        let dataStart = (8 + json.count + page - 1) / page * page
        json.append(Data(repeating: 0x20, count: dataStart - 8 - json.count))
        var length = UInt64(json.count).littleEndian
        var file = Data(bytes: &length, count: 8)
        file.append(json)
        file.append(payload)
        try file.write(to: url)
    }

    @Test func mappedShardWindowsAreNotReplaceableAndOwnedArraysAre() throws {
        try MLXMetalTestLock.withLock {
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("weight-backing-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: dir) }
            let url = dir.appendingPathComponent("shard.safetensors")
            // One small and one large tensor in the same shard: the small one is a window into a
            // mapping several times its size, as in a real multi-GB shard.
            try writeAlignedShard(at: url, tensors: [("a.weight", 65_536), ("b.weight", 1_048_576)])

            try withMappedLoad {
                let loaded = try MLX.loadArrays(url: url)
                let mapped = try #require(loaded["a.weight"])
                #expect(mapped.asArray(UInt32.self)[300] == 300)
                #expect(WeightBacking.sharesMappedShard(mapped))
                #expect(!WeightBacking.copyWouldReplace([mapped]))

                // A materialized copy owns its allocation.
                let owned = mapped + 0
                eval(owned)
                #expect(!WeightBacking.sharesMappedShard(owned))
                #expect(WeightBacking.copyWouldReplace([owned]))
                #expect(!WeightBacking.copyWouldReplace([owned, mapped]))

                setenv("VMLX_COPY_MAPPED_WEIGHTS", "1", 1)
                #expect(WeightBacking.copyWouldReplace([mapped]))
            }
            // A non-mapped load owns every tensor.
            let resident = try #require(try MLX.loadArrays(url: url)["a.weight"])
            #expect(!WeightBacking.sharesMappedShard(resident))
        }
    }
}
