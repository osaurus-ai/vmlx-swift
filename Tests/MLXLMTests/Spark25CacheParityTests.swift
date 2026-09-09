import Foundation
import MLX
import Testing
import MLXLMCommon
@testable import MLXLLM

/// Strict numerical diagnostic runs in a fresh process with TF32 disabled;
/// production defaults are never changed by this test.
@Suite("Spark mixed-window cache parity", .serialized,
       .enabled(if: ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0"))
struct Spark25CacheParityTests {
    @Test func longChunkedPrefillAndDiskContinuationMatchFreshAttention() throws {
        try MLXMetalTestLock.withLock {
            let model = Spark25Model(try Spark25Tests.config(["sliding_window": 512]))
            #expect(model.parameters().flattened().reduce(0) { $0 + $1.1.size } < 250_000)
            let tokens = MLXArray((0..<525).map { Int32(($0 * 7 + 3) % 127) }).reshaped(1, 525)
            let baseline = model(tokens, cache: nil)
            eval(baseline)
            #expect(all(isFinite(baseline)).item(Bool.self))
            let cache = model.newCache()
            var offset = 0
            for count in [509, 3, 1] {
                let out = model(tokens[0..., offset..<(offset + count)], cache: cache)
                eval(out, cache)
                let expected = baseline[0..., (offset + count - 1)..<(offset + count), 0...]
                let actual = out[0..., (count - 1)..<count, 0...]
                let error = max(abs(actual - expected)).item(Float.self)
                print("SPARK_CACHE chunk=\(count) boundary=\(offset + count) maxAbs=\(error)")
                #expect(error < 1e-4)
                offset += count
            }
            #expect(cache.allSatisfy { $0.offset == 513 })
            let copied = cache.map { $0.copy() }
            let file = FileManager.default.temporaryDirectory
                .appendingPathComponent("spark-cache-\(UUID().uuidString).safetensors")
            defer { try? FileManager.default.removeItem(at: file) }
            try savePromptCache(url: file, cache: cache, metadata: ["case": "spark-513"])
            let (restored, metadata) = try loadPromptCache(url: file)
            #expect(metadata["case"] == "spark-513")
            #expect(restored.prefix(3).allSatisfy { $0 is RotatingKVCache })
            #expect(restored[3] is KVCacheSimple)
            #expect(restored.map(\.metaState) == cache.map(\.metaState))
            for count in [7, 5] {
                let input = tokens[0..., offset..<(offset + count)]
                let continuous = model(input, cache: cache)
                let fromCopy = model(input, cache: copied)
                let fromDisk = model(input, cache: restored)
                eval(continuous, fromCopy, fromDisk)
                let expected = baseline[0..., (offset + count - 1)..<(offset + count), 0...]
                let actual = continuous[0..., (count - 1)..<count, 0...]
                let error = max(abs(actual - expected)).item(Float.self)
                print("SPARK_CACHE continuation=\(count) boundary=\(offset + count) maxAbs=\(error)")
                #expect(error < 1e-4)
                #expect(all(continuous .== fromCopy).item(Bool.self))
                #expect(all(continuous .== fromDisk).item(Bool.self))
                offset += count
            }
            #expect(restored.allSatisfy { $0.offset == 525 })
        }
    }
}
