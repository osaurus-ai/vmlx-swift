import Foundation
import MLX
@testable import MLXLMCommon
import Testing

@Test func diskCacheStoreAndFetch() async throws {
    try await MLXMetalTestLock.withLock {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx_test_\(UUID())")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let cache = DiskCache(cacheDir: tempDir, maxSizeGB: 0.1)

        let tokens = [1, 2, 3, 4, 5]
        let arrays: [String: MLXArray] = [
            "keys": MLXArray.ones([2, 4, 8]),
            "values": MLXArray.zeros([2, 4, 8]),
        ]

        cache.store(tokens: tokens, arrays: arrays)

        // Wait for background write to complete
        try await Task.sleep(nanoseconds: 500_000_000)

        let result = cache.fetch(tokens: tokens)
        #expect(result != nil)
        #expect(result?.keys.sorted() == ["keys", "values"])
        #expect(cache.hits == 1)
        #expect(cache.stores == 1)
    }
}

@Test func diskCacheMiss() {
    let tempDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("vmlx_test_\(UUID())")
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let cache = DiskCache(cacheDir: tempDir, maxSizeGB: 0.1)

    let result = cache.fetch(tokens: [99, 100, 101])
    #expect(result == nil)
    #expect(cache.misses == 1)
    #expect(cache.hits == 0)
}

@Test func diskCacheClear() async throws {
    try await MLXMetalTestLock.withLock {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx_test_\(UUID())")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let cache = DiskCache(cacheDir: tempDir, maxSizeGB: 0.1)

        let tokens = [10, 20, 30]
        let arrays: [String: MLXArray] = [
            "data": MLXArray.ones([4, 4]),
        ]

        cache.store(tokens: tokens, arrays: arrays)

        // Wait for background write to complete
        try await Task.sleep(nanoseconds: 500_000_000)

        // Verify the entry exists
        let beforeClear = cache.fetch(tokens: tokens)
        #expect(beforeClear != nil)

        // Clear the cache
        cache.clear()

        // Verify the entry is gone
        let afterClear = cache.fetch(tokens: tokens)
        #expect(afterClear == nil)
    }
}

@Test func diskCacheCandidateTokenCountsAreDescendingAndBounded() async throws {
    try await MLXMetalTestLock.withLock {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx_test_\(UUID())")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let cache = DiskCache(cacheDir: tempDir, maxSizeGB: 0.1)
        let arrays = ["data": MLXArray.ones([1, 1])]

        cache.store(tokens: [1, 2, 3], arrays: arrays)
        cache.store(tokens: [1, 2, 3, 4, 5], arrays: arrays)
        cache.store(tokens: [9, 8, 7, 6, 5, 4, 3], arrays: arrays)

        let counts = cache.candidateTokenCounts(maxTokens: 6)
        #expect(counts == [5, 3])

        let limited = cache.candidateTokenCounts(maxTokens: 10, limit: 2)
        #expect(limited == [7, 5])
    }
}

@Test func diskCacheHashDeterminism() {
    let tokens = [42, 43, 44, 45]
    let hash1 = DiskCache.hashTokens(tokens)
    let hash2 = DiskCache.hashTokens(tokens)
    #expect(hash1 == hash2)
    #expect(hash1.count == 32)

    // Different tokens produce different hashes
    let hash3 = DiskCache.hashTokens([42, 43, 44, 46])
    #expect(hash1 != hash3)
}

@Test func coordinatorEnforcesOneQuotaAcrossKVAndCompanionPayloads() async throws {
    try await MLXMetalTestLock.withLock {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-combined-disk-quota-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let companionDir = root.appendingPathComponent("ssm_companion")
        let modelKey = "combined-quota-model"
        let tokens = [1, 2, 3, 4]

        let disk = DiskCache(cacheDir: root, maxSizeGB: 1, modelKey: modelKey)
        let companion = try SSMCompanionDiskStore(
            cacheDir: companionDir,
            modelKey: modelKey,
            maxBytes: 1_000_000)
        disk.store(
            tokens: tokens,
            arrays: ["data": MLXArray.ones([1_024])])
        try companion.store(
            ssmStates: [MLXArray.ones([1_024])],
            tokens: tokens,
            boundary: tokens.count)

        let kvEntry = try #require(disk.quotaEntries().first)
        let companionEntry = try #require(companion.quotaEntries().first)
        #expect(companionEntry.kvHash == kvEntry.hash)

        let smallerEntry = min(kvEntry.bytes, companionEntry.bytes)
        let combinedBytes = kvEntry.bytes + companionEntry.bytes
        let capBytes = combinedBytes - max(1, smallerEntry / 2)
        #expect(capBytes > kvEntry.bytes)
        #expect(capBytes > companionEntry.bytes)
        #expect(capBytes < combinedBytes)

        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheMaxGB: Float(capBytes) / 1_073_741_824,
            diskCacheDir: root,
            modelKey: modelKey))
        coordinator.enforceCombinedDiskQuota()

        #expect(coordinator.diskCache?.quotaEntries().isEmpty == true)
        #expect(coordinator.ssmStateCache.diskStore?.quotaEntries().isEmpty == true)
        #expect(disk.fetch(tokens: tokens) == nil)
        #expect(companion.fetch(tokens: tokens, boundary: tokens.count) == nil)
    }
}

@Test func combinedQuotaRetiresUnlinkedLegacyCompanionBeforeIndexedKV() async throws {
    try await MLXMetalTestLock.withLock {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-legacy-companion-quota-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let companionDir = root.appendingPathComponent("ssm_companion")
        let modelKey = "legacy-companion-quota-model"
        let tokens = [8, 6, 7, 5, 3, 0, 9]

        let disk = DiskCache(cacheDir: root, maxSizeGB: 1, modelKey: modelKey)
        let companion = try SSMCompanionDiskStore(
            cacheDir: companionDir,
            modelKey: modelKey,
            maxBytes: 1_000_000)
        disk.store(
            tokens: tokens,
            arrays: ["data": MLXArray.ones([1_024])])
        try companion.store(
            ssmStates: [MLXArray.ones([1_024])],
            tokens: tokens,
            boundary: tokens.count)

        let kvEntry = try #require(disk.quotaEntries().first)
        let companionEntry = try #require(companion.quotaEntries().first)
        let sidecarURL = companionDir
            .appendingPathComponent("ssm-\(companionEntry.hash).json")
        let sidecarData = try Data(contentsOf: sidecarURL)
        var sidecar = try #require(
            JSONSerialization.jsonObject(with: sidecarData) as? [String: Any])
        sidecar.removeValue(forKey: "kv_hash")
        try JSONSerialization.data(withJSONObject: sidecar, options: [.sortedKeys])
            .write(to: sidecarURL, options: [.atomic])

        let legacyEntry = try #require(companion.quotaEntries().first)
        #expect(legacyEntry.kvHash == nil)
        let combinedBytes = kvEntry.bytes + legacyEntry.bytes
        let capBytes = combinedBytes - max(1, min(kvEntry.bytes, legacyEntry.bytes) / 2)
        #expect(capBytes > kvEntry.bytes)
        #expect(capBytes > legacyEntry.bytes)

        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheMaxGB: Float(capBytes) / 1_073_741_824,
            diskCacheDir: root,
            modelKey: modelKey))

        #expect(coordinator.diskCache?.quotaEntries().count == 1)
        #expect(coordinator.ssmStateCache.diskStore?.quotaEntries().isEmpty == true)
        #expect(disk.fetch(tokens: tokens) != nil)
        #expect(companion.fetch(tokens: tokens, boundary: tokens.count) == nil)
    }
}
