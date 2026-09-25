import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Exercise real files and arrays retained by an active consumer across quota
/// deletion. A hit counter alone cannot prove that a later decode can still
/// read the restored state, particularly integer metadata left lazy by the
/// floating-point integrity scan.
@Suite(.serialized)
struct DiskCacheActiveReaderTests {
    @Test func retainedArraysSurviveQuotaEvictionAndClear() throws {
        try MLXMetalTestLock.withLock {
            let root = FileManager.default.temporaryDirectory
                .appendingPathComponent("active-cache-reader-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: root) }
            let cache = DiskCache(cacheDir: root, maxSizeBytes: 110_000, modelKey: "active-reader")
            let tokens = [1, 2, 3]
            let expected = Array(0 ..< 8_192).map(Int32.init)
            cache.store(
                tokens: tokens,
                arrays: [
                    "keys": MLXArray(expected).asType(.float32),
                    "positions": MLXArray(expected),
                ])
            let retained = try #require(cache.fetch(tokens: tokens))
            // Do not evaluate the retained integer array before removal.
            Thread.sleep(forTimeInterval: 1.1)
            cache.store(
                tokens: [4, 5, 6],
                arrays: [
                    "keys": MLXArray.ones([16_384], dtype: .float32)
                ])
            #expect(cache.snapshotStats().evictions > 0)
            #expect(!cache.hasDurableEntry(tokens: tokens))
            #expect(cache.fetch(tokens: tokens) == nil)
            #expect(try #require(retained["positions"]).asArray(Int32.self) == expected)
            #expect(try #require(retained["keys"]).asArray(Float.self) == expected.map(Float.init))
            let second = try #require(cache.fetch(tokens: [4, 5, 6]))
            cache.clear()
            #expect(cache.snapshotStats().currentEntryCount == 0)
            #expect(try #require(second["keys"]).sum().item(Float.self) == 16_384)
        }
    }

    @Test func mediaIdentityRemainsIsolatedAfterEvictionAndRestart() throws {
        try MLXMetalTestLock.withLock {
            let root = FileManager.default.temporaryDirectory
                .appendingPathComponent("media-cache-eviction-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: root) }
            let tokens = [11, 12, 13]
            let writer = DiskCache(cacheDir: root, maxSizeBytes: 90_000, modelKey: "media-model")
            writer.store(
                tokens: tokens, arrays: ["keys": MLXArray.ones([16_384])], mediaSalt: "image-a")
            Thread.sleep(forTimeInterval: 1.1)
            writer.store(
                tokens: tokens, arrays: ["keys": MLXArray.ones([16_384]) * 2], mediaSalt: "image-b")
            #expect(writer.snapshotStats().evictions > 0)
            let restarted = DiskCache(cacheDir: root, maxSizeBytes: 90_000, modelKey: "media-model")
            #expect(restarted.fetch(tokens: tokens, mediaSalt: "image-a") == nil)
            #expect(restarted.fetch(tokens: tokens, mediaSalt: nil) == nil)
            let b = try #require(restarted.fetch(tokens: tokens, mediaSalt: "image-b"))
            #expect(try #require(b["keys"]).sum().item(Float.self) == 32_768)
            // This proves salt/retention identity, not vision-encoder companion
            // correctness. Actual media inference remains a separate live row.
        }
    }
}