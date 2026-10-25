import Foundation
import MLX
import Testing
@testable import MLXLMCommon
@testable import MLXVLM

#if canImport(Darwin)
@Suite("Auxiliary compute owns its loaded bytes", .serialized)
struct ResidentComputeAuxiliaryLoadingTests {
    private func directory() throws -> URL {
        let result = FileManager.default.temporaryDirectory
            .appendingPathComponent("resident-auxiliary-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: result, withIntermediateDirectories: true)
        return result
    }

    private func overwriteThenRemove(_ url: URL) throws {
        let file = try FileHandle(forWritingTo: url)
        let length = try file.seekToEnd()
        try file.seek(toOffset: 0)
        // Same inode and same length: unlink alone leaves mmap pages alive.
        // Mutate the actual payload before any loaded-array evaluation.
        try file.write(contentsOf: Data(repeating: 0xA5, count: Int(length)))
        try file.synchronize()
        try file.close()
        try FileManager.default.removeItem(at: url)
    }

    @Test("Legacy JANG repacking and companions survive source overwrite/removal")
    func legacyJANGWeights() throws {
        try MLXMetalTestLock.withLock {
            let root = try directory()
            defer { try? FileManager.default.removeItem(at: root) }
            let file = root.appendingPathComponent("model.jang.safetensors")
            try MLX.save(arrays: [
                "packed": MLXArray([UInt8(1), 2, 3, 4, 5, 6, 7, 8]).reshaped(2, 4),
                "scales": MLXArray([Float(0.125), -2]).asType(.float16),
                "padding": MLXArray.zeros([16_384], type: Float.self),
            ], url: file)
            let loaded = try JangLoader.loadV1Weights(at: root)
            try overwriteThenRemove(file)
            let packed = try #require(loaded["packed"])
            #expect(!WeightBacking.sharesMappedShard(packed))
            #expect(packed.dtype == .uint32 && packed.shape == [2, 1])
            #expect(packed.asArray(UInt32.self) == [0x0403_0201, 0x0807_0605])
            let scales = try #require(loaded["scales"])
            #expect(!WeightBacking.sharesMappedShard(scales))
            #expect(scales.dtype == .float16)
            #expect(scales.asType(.float32).asArray(Float.self) == [0.125, -2])
        }
    }

    @Test("JANGTQ signs and codebooks survive source overwrite/removal")
    func runtimeSidecar() throws {
        try MLXMetalTestLock.withLock {
            let root = try directory()
            defer { try? FileManager.default.removeItem(at: root) }
            let file = root.appendingPathComponent("jangtq_runtime.safetensors")
            let width = Int.random(in: 100_000...200_000)
            let seed = Int.random(in: 100_000...200_000)
            try MLX.save(arrays: [
                "signs.\(width).\(seed)": MLXArray([Float(1), -1, 1, -1]),
                "codebook.\(width).4": MLXArray([Float(-0.5), 0.5]),
                "ignored.padding": MLXArray.zeros([16_384], type: Float.self),
            ], url: file)
            #expect(JANGTQRuntimeCache.sniffCodebookBits(at: file) == 4)
            try JANGTQRuntimeCache.shared.loadSidecar(from: file)
            try overwriteThenRemove(file)
            let signs = try #require(JANGTQRuntimeCache.shared.signs(inFeatures: width, seed: seed))
            let codebook = try #require(JANGTQRuntimeCache.shared.codebook(inFeatures: width, bits: 4))
            #expect(!WeightBacking.sharesMappedShard(signs) && !WeightBacking.sharesMappedShard(codebook))
            #expect(signs.dtype == .float32 && codebook.dtype == .float32)
            #expect(signs.asArray(Float.self) == [1, -1, 1, -1])
            #expect(codebook.asArray(Float.self) == [-0.5, 0.5])
        }
    }

    @Test("MiMo selected media weights own bytes with and without an index", arguments: [false, true])
    func mimoMediaSelection(_ indexed: Bool) throws {
        try MLXMetalTestLock.withLock {
            let root = try directory()
            defer { try? FileManager.default.removeItem(at: root) }
            let file = root.appendingPathComponent("model.safetensors")
            let originals: [String: MLXArray] = [
                "visual.packed": MLXArray([UInt32(1), 17, UInt32.max]),
                "visual.fp16": MLXArray([Float(0.125), -2]).asType(.float16),
                "visual.bf16": MLXArray([Float(0.5), 9]).asType(.bfloat16),
                "visual.fp32": MLXArray([Float(-0.25), 3]),
            ]
            var shard = originals
            shard["model.unselected"] = MLXArray.zeros([16_384], type: Float.self)
            try MLX.save(arrays: shard, url: file)
            if indexed {
                var index = shard.mapValues { _ in file.lastPathComponent }
                index["model.unselected.missing"] = "missing-unselected.safetensors"
                try JSONSerialization.data(withJSONObject: ["weight_map": index])
                    .write(to: root.appendingPathComponent("model.safetensors.index.json"))
            }
            let loaded = try MiMoV26AuxiliaryLoader.selectedWeights(in: root) { $0.hasPrefix("visual.") }
            #expect(Set(loaded.keys) == Set(originals.keys))
            // No eval before mutation: a lazy read or shared mapped inode
            // cannot pass by reading original checkpoint bytes afterward.
            try overwriteThenRemove(file)
            for (key, expected) in originals {
                let actual = try #require(loaded[key])
                #expect(!WeightBacking.sharesMappedShard(actual))
                #expect(actual.dtype == expected.dtype && actual.shape == expected.shape)
                #expect(arrayEqual(actual, expected).item(Bool.self))
            }
        }
    }
}
#endif
