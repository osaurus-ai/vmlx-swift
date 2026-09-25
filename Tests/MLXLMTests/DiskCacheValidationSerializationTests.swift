import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct DiskCacheValidationSerializationTests {
    @Test func invalidStoreWaitsForGlobalMLXIOGate() throws {
        try MLXMetalTestLock.withLock {
            let root = FileManager.default.temporaryDirectory
                .appendingPathComponent("cache-validation-gate-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: root) }
            let cache = DiskCache(cacheDir: root, maxSizeBytes: 1_048_576, modelKey: "validation")
            // Materialize before starting the worker. The lock holder performs
            // no GPU work: this detects the scope bug without racing encoders.
            nonisolated(unsafe) let arrays = ["keys": MLXArray([Float.nan])]
            MLX.eval(Array(arrays.values))
            let started = DispatchSemaphore(value: 0)
            let finished = DispatchSemaphore(value: 0)
            MLXDiskCacheIOLock.shared.lock()
            DispatchQueue.global().async {
                started.signal()
                cache.store(tokens: [1, 2, 3], arrays: arrays)
                finished.signal()
            }
            let didStart = started.wait(timeout: .now() + 5)
            let beforeRelease = finished.wait(timeout: .now() + 0.25)
            MLXDiskCacheIOLock.shared.unlock()
            #expect(didStart == .success)
            #expect(beforeRelease == .timedOut,
                    "Even rejected payload validation must wait for global MLX IO ownership")
            if beforeRelease == .timedOut {
                #expect(finished.wait(timeout: .now() + 5) == .success)
            }
            #expect(cache.snapshotStats().currentEntryCount == 0)
            #expect(!cache.hasDurableEntry(tokens: [1, 2, 3]))
            // Exercise another operation after rejection to catch a leaked lock.
            cache.store(tokens: [4, 5, 6], arrays: ["keys": MLXArray([Float(7)])])
            let valid = try #require(cache.fetch(tokens: [4, 5, 6]))
            #expect(try #require(valid["keys"]).item(Float.self) == 7)
        }
    }
}
