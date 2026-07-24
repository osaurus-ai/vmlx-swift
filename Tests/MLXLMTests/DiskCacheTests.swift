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

@Test func diskCacheFetchRefreshesRecencyWithoutRewritingPayload() async throws {
    try await MLXMetalTestLock.withLock {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-disk-fetch-recency-\(UUID())")
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let modelKey = "disk-fetch-recency-model"
        let hotTokens = [1, 2, 3, 4]
        let coldTokens = [5, 6, 7, 8]
        let hotHash = DiskCache.hashTokens(hotTokens, modelKey: modelKey)
        let cache = DiskCache(
            cacheDir: tempDir, maxSizeGB: 0.1, modelKey: modelKey)
        let arrays = ["data": MLXArray.ones([64])]

        cache.store(tokens: hotTokens, arrays: arrays)
        try await Task.sleep(nanoseconds: 50_000_000)
        cache.store(tokens: coldTokens, arrays: arrays)

        let hotBefore = try #require(
            cache.quotaEntries().first { $0.hash == hotHash })
        let coldBefore = try #require(
            cache.quotaEntries().first {
                $0.hash == DiskCache.hashTokens(coldTokens, modelKey: modelKey)
            })
        #expect(hotBefore.createdAt < coldBefore.createdAt)

        let payloadURL = tempDir.appendingPathComponent("\(hotHash).safetensors")
        let payloadBefore = try Data(contentsOf: payloadURL)
        try await Task.sleep(nanoseconds: 50_000_000)

        #expect(cache.fetch(tokens: hotTokens) != nil)

        let hotAfter = try #require(
            cache.quotaEntries().first { $0.hash == hotHash })
        let coldAfter = try #require(
            cache.quotaEntries().first {
                $0.hash == DiskCache.hashTokens(coldTokens, modelKey: modelKey)
            })
        #expect(hotAfter.createdAt > coldAfter.createdAt)
        #expect(try Data(contentsOf: payloadURL) == payloadBefore)
    }
}

@Test func diskCacheSkipsRewriteOnlyAfterCurrentProcessValidation() async throws {
    try await MLXMetalTestLock.withLock {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx_dedup_\(UUID())")
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let tokens = [31, 41, 59, 26]
        let arrays = ["data": MLXArray.ones([8, 8])]
        let hash = DiskCache.hashTokens(tokens, modelKey: "dedup-model")
        let file = tempDir.appendingPathComponent("\(hash).safetensors")

        do {
            let first = DiskCache(
                cacheDir: tempDir, maxSizeGB: 0.1, modelKey: "dedup-model")
            first.store(tokens: tokens, arrays: arrays)
            #expect(first.snapshotStats().storeSkips == 0)
        }

        // A fresh process/cache instance must validate an inherited payload
        // before it is eligible for the no-rewrite path.
        let warm = DiskCache(
            cacheDir: tempDir, maxSizeGB: 0.1, modelKey: "dedup-model")
        let inheritedModification = try #require(
            (try FileManager.default.attributesOfItem(atPath: file.path))[.modificationDate]
                as? Date)
        #expect(warm.fetch(tokens: tokens) != nil)
        warm.store(tokens: tokens, arrays: arrays)
        let deduplicatedModification = try #require(
            (try FileManager.default.attributesOfItem(atPath: file.path))[.modificationDate]
                as? Date)
        #expect(deduplicatedModification == inheritedModification)
        #expect(warm.snapshotStats().stores == 1)
        #expect(warm.snapshotStats().storeSkips == 1)

        // External replacement invalidates the fingerprint and forces a real
        // healing write instead of preserving the changed file.
        let changedDate = Date(timeIntervalSince1970: 1)
        try FileManager.default.setAttributes(
            [.modificationDate: changedDate], ofItemAtPath: file.path)
        warm.store(tokens: tokens, arrays: arrays)
        let healedModification = try #require(
            (try FileManager.default.attributesOfItem(atPath: file.path))[.modificationDate]
                as? Date)
        #expect(healedModification != changedDate)
        #expect(warm.snapshotStats().stores == 2)
        #expect(warm.snapshotStats().storeSkips == 1)
    }
}

@Test func ssmCompanionFetchRefreshesRecencyWithoutRewritingPayload() async throws {
    try await MLXMetalTestLock.withLock {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-ssm-fetch-recency-\(UUID())")
        defer { try? FileManager.default.removeItem(at: dir) }
        let modelKey = "ssm-fetch-recency-model"
        let hotTokens = [11, 12, 13, 14]
        let coldTokens = [21, 22, 23, 24]
        let store = try SSMCompanionDiskStore(
            cacheDir: dir,
            modelKey: modelKey,
            maxBytes: 10_000_000)
        let states = [MLXArray.ones([64])]

        try store.store(
            ssmStates: states,
            tokens: hotTokens,
            boundary: hotTokens.count)
        try await Task.sleep(nanoseconds: 50_000_000)
        try store.store(
            ssmStates: states,
            tokens: coldTokens,
            boundary: coldTokens.count)

        let hotHash = SSMCompanionDiskStore.keyFor(
            tokens: hotTokens,
            boundary: hotTokens.count,
            modelKey: modelKey)
        let coldHash = SSMCompanionDiskStore.keyFor(
            tokens: coldTokens,
            boundary: coldTokens.count,
            modelKey: modelKey)
        let hotBefore = try #require(
            store.quotaEntries().first { $0.hash == hotHash })
        let coldBefore = try #require(
            store.quotaEntries().first { $0.hash == coldHash })
        #expect(hotBefore.modifiedAt < coldBefore.modifiedAt)

        let tensorURL = dir.appendingPathComponent("ssm-\(hotHash).safetensors")
        let sidecarURL = dir.appendingPathComponent("ssm-\(hotHash).json")
        let tensorBefore = try Data(contentsOf: tensorURL)
        let sidecarBefore = try Data(contentsOf: sidecarURL)
        try await Task.sleep(nanoseconds: 50_000_000)

        #expect(
            store.fetch(
                tokens: hotTokens,
                boundary: hotTokens.count) != nil)

        let hotAfter = try #require(
            store.quotaEntries().first { $0.hash == hotHash })
        let coldAfter = try #require(
            store.quotaEntries().first { $0.hash == coldHash })
        #expect(hotAfter.modifiedAt > coldAfter.modifiedAt)
        #expect(try Data(contentsOf: tensorURL) == tensorBefore)
        #expect(try Data(contentsOf: sidecarURL) == sidecarBefore)
    }
}

@Test func successfulHybridDiskRestoreTouchesWholeGroupForLRUEviction() async throws {
    try await MLXMetalTestLock.withLock {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-hybrid-group-lru-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let modelKey = "hybrid-group-lru-model"
        let hotTokens = [31, 32, 33, 34]
        let coldTokens = [41, 42, 43, 44]
        let newTokens = [51, 52, 53, 54]
        let kvArrays = ["data": MLXArray.ones([8])]
        let ssmStates = [MLXArray.ones([1_024])]
        let hotKVHash = DiskCache.hashTokens(hotTokens, modelKey: modelKey)
        let coldKVHash = DiskCache.hashTokens(coldTokens, modelKey: modelKey)
        let newKVHash = DiskCache.hashTokens(newTokens, modelKey: modelKey)
        let hotSSMHash = SSMCompanionDiskStore.keyFor(
            tokens: hotTokens,
            boundary: hotTokens.count,
            modelKey: modelKey)
        let coldSSMHash = SSMCompanionDiskStore.keyFor(
            tokens: coldTokens,
            boundary: coldTokens.count,
            modelKey: modelKey)
        let newSSMHash = SSMCompanionDiskStore.keyFor(
            tokens: newTokens,
            boundary: newTokens.count,
            modelKey: modelKey)

        var capBytes: Int64 = 0
        do {
            let seed = CacheCoordinator(config: CacheCoordinatorConfig(
                usePagedCache: false,
                enableDiskCache: true,
                diskCacheMaxGB: 1,
                diskCacheDir: root,
                modelKey: modelKey))
            seed.setHybrid(true, requiresRecurrentSSMCompanion: true)
            let disk = try #require(seed.diskCache)

            disk.store(tokens: hotTokens, arrays: kvArrays)
            seed.ssmStateCache.store(
                ssmStates: ssmStates,
                tokens: hotTokens,
                boundary: hotTokens.count)
            try await Task.sleep(nanoseconds: 50_000_000)
            disk.store(tokens: coldTokens, arrays: kvArrays)
            seed.ssmStateCache.store(
                ssmStates: ssmStates,
                tokens: coldTokens,
                boundary: coldTokens.count)

            let hotKV = try #require(
                disk.quotaEntries().first { $0.hash == hotKVHash })
            let coldKV = try #require(
                disk.quotaEntries().first { $0.hash == coldKVHash })
            let companion = try #require(seed.ssmStateCache.diskStore)
            let hotSSM = try #require(
                companion.quotaEntries().first { $0.hash == hotSSMHash })
            let coldSSM = try #require(
                companion.quotaEntries().first { $0.hash == coldSSMHash })
            let hotGroupBytes = hotKV.bytes + hotSSM.bytes
            let coldGroupBytes = coldKV.bytes + coldSSM.bytes
            // A + B fit before the access. Adding C must exceed the cap so
            // combined quota has to choose exactly one linked group to evict.
            capBytes = hotGroupBytes + coldGroupBytes
        }

        let capGB = Float(capBytes) / 1_073_741_824
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheMaxGB: capGB,
            diskCacheDir: root,
            modelKey: modelKey))
        coordinator.setHybrid(true, requiresRecurrentSSMCompanion: true)
        let disk = try #require(coordinator.diskCache)
        let companion = try #require(coordinator.ssmStateCache.diskStore)

        // Keep hot recurrent state in L1 so the accepted disk restore does not
        // call companion.fetch. The coordinator-level group touch must still
        // refresh the linked on-disk sidecar.
        coordinator.ssmStateCache.store(
            ssmStates: ssmStates,
            tokens: hotTokens,
            boundary: hotTokens.count,
            persistToDisk: false)

        let hotTensorURL = root
            .appendingPathComponent("ssm_companion")
            .appendingPathComponent("ssm-\(hotSSMHash).safetensors")
        let hotSidecarURL = root
            .appendingPathComponent("ssm_companion")
            .appendingPathComponent("ssm-\(hotSSMHash).json")
        let hotTensorBefore = try Data(contentsOf: hotTensorURL)
        let hotSidecarBefore = try Data(contentsOf: hotSidecarURL)
        try await Task.sleep(nanoseconds: 50_000_000)

        switch coordinator.fetch(tokens: hotTokens) {
        case .hit(
            let matchedTokens,
            let remainingTokens,
            .disk,
            _,
            let restoredStates,
            _):
            #expect(matchedTokens == hotTokens.count)
            #expect(remainingTokens.isEmpty)
            #expect(restoredStates?.count == 1)
        default:
            Issue.record("expected accepted hybrid disk restore for hot group")
        }

        let hotKVAfter = try #require(
            disk.quotaEntries().first { $0.hash == hotKVHash })
        let coldKVAfter = try #require(
            disk.quotaEntries().first { $0.hash == coldKVHash })
        let hotSSMAfter = try #require(
            companion.quotaEntries().first { $0.hash == hotSSMHash })
        let coldSSMAfter = try #require(
            companion.quotaEntries().first { $0.hash == coldSSMHash })
        let hotGroupRecency = min(
            hotKVAfter.createdAt, hotSSMAfter.modifiedAt)
        let coldGroupRecency = min(
            coldKVAfter.createdAt, coldSSMAfter.modifiedAt)
        #expect(
            abs(hotKVAfter.createdAt.timeIntervalSince(hotSSMAfter.modifiedAt))
                < 0.01)
        #expect(hotGroupRecency > coldGroupRecency)
        #expect(try Data(contentsOf: hotTensorURL) == hotTensorBefore)
        #expect(try Data(contentsOf: hotSidecarURL) == hotSidecarBefore)

        try await Task.sleep(nanoseconds: 50_000_000)
        disk.store(tokens: newTokens, arrays: kvArrays)
        coordinator.ssmStateCache.store(
            ssmStates: ssmStates,
            tokens: newTokens,
            boundary: newTokens.count)
        coordinator.enforceCombinedDiskQuota()

        let remainingKV = Set(disk.quotaEntries().map(\.hash))
        let remainingSSM = Set(companion.quotaEntries().map(\.hash))
        #expect(remainingKV.contains(hotKVHash))
        #expect(remainingKV.contains(newKVHash))
        #expect(!remainingKV.contains(coldKVHash))
        #expect(remainingSSM.contains(hotSSMHash))
        #expect(remainingSSM.contains(newSSMHash))
        #expect(!remainingSSM.contains(coldSSMHash))
    }
}

@Test func growingDenseRestoreKeepsStableCheckpointHotUnderQuota() async throws {
    try await MLXMetalTestLock.withLock {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-dense-stable-checkpoint-lru-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let modelKey = "dense-stable-checkpoint-lru-model"
        let stableTokens = [1, 2, 3, 4]
        let coldTokens = [21, 22, 23, 24]
        let growingTokens = [1, 2, 3, 4, 5, 6]
        let requestTokens = [1, 2, 3, 4, 5, 6, 7]
        let newTokens = [31, 32, 33, 34]
        let arrays = ["data": MLXArray.ones([8])]

        func hash(_ tokens: [Int]) -> String {
            DiskCache.hashTokens(tokens, modelKey: modelKey)
        }

        var capBytes: Int64 = 0
        do {
            let seed = CacheCoordinator(config: CacheCoordinatorConfig(
                usePagedCache: false,
                enableDiskCache: true,
                diskCacheMaxGB: 1,
                diskCacheDir: root,
                modelKey: modelKey))
            let disk = try #require(seed.diskCache)
            for tokens in [stableTokens, coldTokens, growingTokens] {
                disk.store(tokens: tokens, arrays: arrays)
                try await Task.sleep(nanoseconds: 50_000_000)
            }
            for tokens in [stableTokens, coldTokens, growingTokens] {
                capBytes += try #require(
                    disk.quotaEntries().first { $0.hash == hash(tokens) }).bytes
            }
            // Three existing payloads fit; four do not. Leave half an
            // average entry of rounding headroom for the Float GB setting.
            capBytes += capBytes / 6
        }

        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheMaxGB: Float(capBytes) / 1_073_741_824,
            diskCacheDir: root,
            modelKey: modelKey))
        let disk = try #require(coordinator.diskCache)

        switch coordinator.fetch(
            tokens: requestTokens,
            preferredDiskBoundaries: [stableTokens.count])
        {
        case .hit(let matched, let remaining, .disk, _, _, _):
            #expect(matched == growingTokens.count)
            #expect(remaining == [7])
        default:
            Issue.record("expected growing dense disk restore")
        }
        coordinator.touchStableDiskCheckpointsAfterRetainedRestore(
            requestTokens: requestTokens,
            matchedTokenCount: growingTokens.count,
            preferredDiskBoundaries: [stableTokens.count],
            skipExactDiskBoundary: false,
            mediaSalt: nil)

        try await Task.sleep(nanoseconds: 50_000_000)
        disk.store(tokens: newTokens, arrays: arrays)
        coordinator.enforceCombinedDiskQuota()

        let remaining = Set(disk.quotaEntries().map(\.hash))
        #expect(remaining.contains(hash(stableTokens)))
        #expect(remaining.contains(hash(growingTokens)))
        #expect(remaining.contains(hash(newTokens)))
        #expect(!remaining.contains(hash(coldTokens)))
    }
}

@Test func diskCacheTouchDoesNotRefreshMissingPayloadRow() async throws {
    try await MLXMetalTestLock.withLock {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-missing-payload-touch-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let modelKey = "missing-payload-touch-model"
        let tokens = [41, 42, 43, 44]
        let hash = DiskCache.hashTokens(tokens, modelKey: modelKey)
        let disk = DiskCache(cacheDir: root, maxSizeGB: 1, modelKey: modelKey)

        disk.store(tokens: tokens, arrays: ["data": MLXArray.ones([8])])
        let before = try #require(
            disk.quotaEntries().first { $0.hash == hash }).createdAt
        try FileManager.default.removeItem(
            at: root.appendingPathComponent("\(hash).safetensors"))
        try await Task.sleep(nanoseconds: 50_000_000)

        #expect(!disk.touchRecency(tokens: tokens, at: Date()))
        let after = try #require(
            disk.quotaEntries().first { $0.hash == hash }).createdAt
        #expect(after == before)
    }
}

@Test func growingHybridRestoreKeepsStableCheckpointHotUnderQuota() async throws {
    try await MLXMetalTestLock.withLock {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-stable-checkpoint-lru-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let modelKey = "stable-checkpoint-lru-model"
        let stableSeed = [1, 2, 3, 4]
        let coldTokens = [21, 22, 23, 24]
        let growingTokens = [1, 2, 3, 4, 5, 6]
        let requestTokens = [1, 2, 3, 4, 5, 6, 7]
        let newTokens = [31, 32, 33, 34]
        let stableBoundary = stableSeed.count + 1
        let kvArrays = ["data": MLXArray.ones([8])]
        let ssmStates = [MLXArray.ones([1_024])]

        func kvHash(_ tokens: [Int]) -> String {
            DiskCache.hashTokens(tokens, modelKey: modelKey)
        }
        func ssmHash(_ tokens: [Int]) -> String {
            SSMCompanionDiskStore.keyFor(
                tokens: tokens,
                boundary: tokens.count,
                modelKey: modelKey)
        }

        var capBytes: Int64 = 0
        do {
            let seed = CacheCoordinator(config: CacheCoordinatorConfig(
                usePagedCache: false,
                enableDiskCache: true,
                diskCacheMaxGB: 1,
                diskCacheDir: root,
                modelKey: modelKey))
            seed.setHybrid(true, requiresRecurrentSSMCompanion: true)
            let disk = try #require(seed.diskCache)
            let companion = try #require(seed.ssmStateCache.diskStore)

            for tokens in [stableSeed, coldTokens, growingTokens] {
                disk.store(tokens: tokens, arrays: kvArrays)
                seed.ssmStateCache.store(
                    ssmStates: ssmStates,
                    tokens: tokens,
                    boundary: tokens.count)
                try await Task.sleep(nanoseconds: 50_000_000)
            }

            let seededTokens = [stableSeed, coldTokens, growingTokens]
            for tokens in seededTokens {
                capBytes += try #require(
                    disk.quotaEntries().first { $0.hash == kvHash(tokens) }).bytes
                capBytes += try #require(
                    companion.quotaEntries().first { $0.hash == ssmHash(tokens) }).bytes
            }
            // Leave room for SQLite/Float rounding, but not another linked
            // cache group. Adding the fourth group must still force eviction.
            capBytes += 4_096
        }

        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheMaxGB: Float(capBytes) / 1_073_741_824,
            diskCacheDir: root,
            modelKey: modelKey))
        coordinator.setHybrid(true, requiresRecurrentSSMCompanion: true)
        let disk = try #require(coordinator.diskCache)
        let companion = try #require(coordinator.ssmStateCache.diskStore)

        // The current growing turn restores the longest six-token entry. The
        // four-token seed is the path-dependent N-1 form of the processor's
        // five-token stable system/tool boundary. It is not the served entry,
        // but it is logically reused by this request and must remain hot for a
        // subsequent new chat.
        switch coordinator.fetch(
            tokens: requestTokens,
            skipExactDiskBoundary: true,
            preferredDiskBoundaries: [stableBoundary])
        {
        case .hit(let matched, let remaining, .disk, _, _, _):
            #expect(matched == growingTokens.count)
            #expect(remaining == [7])
        default:
            Issue.record("expected growing hybrid disk restore")
        }
        coordinator.touchStableDiskCheckpointsAfterRetainedRestore(
            requestTokens: requestTokens,
            matchedTokenCount: growingTokens.count,
            preferredDiskBoundaries: [stableBoundary],
            skipExactDiskBoundary: true,
            mediaSalt: nil)

        try await Task.sleep(nanoseconds: 50_000_000)
        disk.store(tokens: newTokens, arrays: kvArrays)
        coordinator.ssmStateCache.store(
            ssmStates: ssmStates,
            tokens: newTokens,
            boundary: newTokens.count)
        coordinator.enforceCombinedDiskQuota()

        let remainingKV = Set(disk.quotaEntries().map(\.hash))
        let remainingSSM = Set(companion.quotaEntries().map(\.hash))
        #expect(remainingKV.contains(kvHash(stableSeed)))
        #expect(remainingKV.contains(kvHash(growingTokens)))
        #expect(remainingKV.contains(kvHash(newTokens)))
        #expect(!remainingKV.contains(kvHash(coldTokens)))
        #expect(remainingSSM.contains(ssmHash(stableSeed)))
        #expect(remainingSSM.contains(ssmHash(growingTokens)))
        #expect(remainingSSM.contains(ssmHash(newTokens)))
        #expect(!remainingSSM.contains(ssmHash(coldTokens)))
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
