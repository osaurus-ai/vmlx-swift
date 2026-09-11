import Foundation
import MLX
import MLXLLM
@testable import MLXLMCommon
import Testing

// MARK: - CacheCoordinator Tests

@Test func coordinatorDurableMambaBoundaryDoesNotRequireUnwrittenSidecar() throws {
    try MLXMetalTestLock.withLock {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("durable-mamba-boundary-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let config = CacheCoordinatorConfig(
            usePagedCache: false, enableDiskCache: true,
            diskCacheMaxGB: 0.1, diskCacheDir: tmp,
            modelKey: "durable-mamba-boundary")
        let tokens = [31, 32, 33]
        let cache = MambaCache()
        cache.offset = tokens.count
        cache[0] = MLXArray.ones([1, 2, 4])
        cache[1] = MLXArray.ones([1, 2, 4, 4]) * 2
        let topology = ModelCacheTopologySnapshot(cache: [cache])
        #expect(topology.requiresRecurrentSSMCompanionState)
        #expect(!topology.requiresSeparateRecurrentPayloadState)
        let writer = CacheCoordinator(config: config)
        writer.setHybrid(true,
            requiresRecurrentSSMCompanion: topology.requiresRecurrentSSMCompanionState,
            requiresSeparateRecurrentPayload: topology.requiresSeparateRecurrentPayloadState)
        writer.storeAfterGeneration(promptTokens: tokens, perLayerData: [],
            ssmStates: nil, cache: [cache])
        #expect(writer.diskCache?.hasDurableEntry(tokens: tokens) == true)
        #expect(writer.hasDurableDiskEntry(tokens: tokens),
            "the in-file Mamba payload must not be rebuilt for an intentionally absent sidecar")
        #expect(writer.hasValidatedDiskEntry(tokens: tokens))
        let restarted = CacheCoordinator(config: config)
        restarted.setHybrid(true, requiresRecurrentSSMCompanion: true,
            requiresSeparateRecurrentPayload: false)
        #expect(restarted.hasDurableDiskEntry(tokens: tokens))
        #expect(!restarted.hasValidatedDiskEntry(tokens: tokens))
        #expect(!restarted.hasDurableDiskEntry(tokens: [31, 32, 99]))
        #expect(!restarted.hasDurableDiskEntry(tokens: tokens, mediaSalt: "different-media"))
        guard case .hit(let matched, _, _, _, _, let arrays) =
            restarted.fetch(tokens: tokens + [34])
        else {
            Issue.record("the persisted Mamba prefix must remain consumable after restart")
            return
        }
        #expect(matched == tokens.count)
        let payload = try #require(arrays)
        var restored: [any KVCache] = [MambaCache()]
        _ = restoreFromDiskArrays(payload, into: &restored)
        #expect(restored[0].offset == tokens.count)
        #expect(restored[0].state.count == cache.state.count)
        for (actual, expected) in zip(restored[0].state, cache.state) {
            #expect(actual.asArray(Float.self) == expected.asArray(Float.self))
        }
        #expect(restarted.hasValidatedDiskEntry(tokens: tokens))
    }
}

@Test func coordinatorValidatedDiskEntryRequiresCurrentProcessProof() throws {
    try MLXMetalTestLock.withLock {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("validated-stable-prefix-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let config = CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheMaxGB: 1.0,
            diskCacheDir: tmp,
            modelKey: "validated-stable-prefix")
        let tokens = [71, 72, 73, 74]
        let keys = MLXArray.ones([1, 1, tokens.count, 4])
        let values = MLXArray.ones([1, 1, tokens.count, 4]) * 2.0

        let writer = CacheCoordinator(config: config)
        #expect(!writer.hasValidatedDiskEntry(tokens: tokens))
        writer.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: [(keys: keys, values: values)],
            ssmStates: nil,
            cache: nil)
        #expect(writer.hasValidatedDiskEntry(tokens: tokens))

        let restarted = CacheCoordinator(config: config)
        #expect(!restarted.hasValidatedDiskEntry(tokens: tokens),
            "an inherited filename/index is not current-process validation")
        guard case .hit = restarted.fetch(tokens: tokens) else {
            Issue.record("the inherited stable prefix should deserialize from L2")
            return
        }
        #expect(restarted.hasValidatedDiskEntry(tokens: tokens))

        let files = try FileManager.default.contentsOfDirectory(
            at: restarted.diskCache!.cacheDir,
            includingPropertiesForKeys: nil)
        let payload = try #require(files.first { $0.pathExtension == "safetensors" })
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSinceNow: 5)],
            ofItemAtPath: payload.path)
        #expect(!restarted.hasValidatedDiskEntry(tokens: tokens),
            "a changed file fingerprint must revoke the warm-store fast path")
    }
}

@Test func coordinatorValidatedHybridEntryRequiresCompleteCompanionPair() {
    MLXMetalTestLock.withLock {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("validated-hybrid-prefix-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheMaxGB: 1.0,
            diskCacheDir: tmp,
            modelKey: "validated-hybrid-prefix"))
        coordinator.setHybrid(true, requiresRecurrentSSMCompanion: true)
        let tokens = [81, 82, 83, 84]
        let keys = MLXArray.ones([1, 1, tokens.count, 4])
        let values = MLXArray.ones([1, 1, tokens.count, 4]) * 2.0
        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: [(keys: keys, values: values)],
            ssmStates: nil,
            cache: nil)
        #expect(!coordinator.hasValidatedDiskEntry(tokens: tokens),
            "validated KV alone must not hide a missing recurrent companion")
        #expect(!coordinator.hasDurableDiskEntry(tokens: tokens))

        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: [(keys: keys, values: values)],
            ssmStates: [MLXArray.ones([1, 4])],
            cache: nil)
        #expect(coordinator.hasValidatedDiskEntry(tokens: tokens))
        #expect(coordinator.hasDurableDiskEntry(tokens: tokens))
    }
}

@Test func coordinatorFindsValidDiskPrefixBelowFirst128IndexedLengths() throws {
    try MLXMetalTestLock.withLock {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("disk-prefix-pagination-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheMaxGB: 0.1,
            diskCacheDir: tmp,
            modelKey: "disk-prefix-pagination"))
        let disk = try #require(coordinator.diskCache)
        let arrays = ["data": MLXArray.ones([1, 1])]
        let requestTokens = Array(0 ..< 160)
        let validBoundary = 8
        disk.store(
            tokens: Array(requestTokens.prefix(validBoundary)),
            arrays: arrays)

        // Exactly 129 distinct larger lengths put 137 ... 10 on page one.
        // Count 9 and the valid count 8 require page two, and none of the
        // decoys matches the active request prefix.
        for count in 9 ... 137 {
            disk.store(tokens: Array(repeating: -count, count: count), arrays: arrays)
        }
        let firstPage = disk.candidateTokenCounts(
            maxTokens: requestTokens.count,
            limit: 128)
        #expect(firstPage.count == 128)
        #expect(firstPage.first == 137)
        #expect(firstPage.last == 10)
        #expect(!firstPage.contains(validBoundary))

        guard case .hit(let matched, let remaining, let detail, _, _, _) =
            coordinator.fetch(tokens: requestTokens)
        else {
            Issue.record("disk lookup must continue beyond its first 128 candidate lengths")
            return
        }
        #expect(matched == validBoundary)
        #expect(remaining == Array(requestTokens.dropFirst(validBoundary)))
        #expect(detail == .disk)
    }
}

/// 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
/// Verify the paged-incompatible short-circuit. With `setPagedIncompatible(true)`
/// a fetch should miss the paged tier even when an exact-prefix block
/// hash exists, so the disk tier (the only one that understands
/// `LayerKind.deepseekV4`) gets a chance to handle the hit.
@Test func coordinatorPagedIncompatibleSkipsPagedTier() {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let blockSize = 4
    let config = CacheCoordinatorConfig(
        usePagedCache: true,
        enableDiskCache: false,
        pagedBlockSize: blockSize,
        maxCacheBlocks: 20
    )
    let coordinator = CacheCoordinator(config: config)

    // Pre-populate the paged tier with one block.
    let tokens = [10, 11, 12, 13]
    let keys = MLXArray.ones([1, 1, blockSize, 4])
    let values = MLXArray.ones([1, 1, blockSize, 4]) * 2.0
    coordinator.storeAfterGeneration(
        promptTokens: tokens,
        perLayerData: [(keys: keys, values: values)],
        ssmStates: nil,
        cache: nil)

    // Sanity: paged tier hits without the flag.
    if case .hit(_, _, let detail, _, _, _) = coordinator.fetch(tokens: tokens) {
        #expect(detail == .paged, "baseline: paged tier should hit")
    } else {
        Issue.record("baseline: paged tier should hit before flipping incompatible flag")
    }

    // Flip the flag: subsequent fetch + store must skip the paged tier.
    coordinator.setPagedIncompatible(true)
    #expect(coordinator.isPagedIncompatible == true)
    #expect(coordinator.canPersistBoundaries == false,
        "an incompatible paged tier without disk L2 cannot persist a usable boundary")
    let incompatibleSnapshot = coordinator.snapshotStats()
    #expect(incompatibleSnapshot.pagedEnabled == false,
        "telemetry must report effective paged state, not the inert requested allocation")
    #expect(incompatibleSnapshot.pagedStats == nil,
        "inert paged counters must not be exposed as an active cache tier")
    let result = coordinator.fetch(tokens: tokens)
    if case .miss = result {
        // expected — paged tier skipped, no disk tier present
    } else {
        Issue.record(
            "paged-incompatible coordinator must miss when only paged tier holds the prefix")
    }

    // store must also skip the paged tier — replaying the store on the
    // flagged coordinator must not allocate new paged blocks.
    let beforeBlocks = coordinator.pagedCache?.stats.allocatedBlocks ?? 0
    coordinator.storeAfterGeneration(
        promptTokens: [20, 21, 22, 23],
        perLayerData: [(keys: keys, values: values)],
        ssmStates: nil,
        cache: nil)
    let afterBlocks = coordinator.pagedCache?.stats.allocatedBlocks ?? 0
    #expect(beforeBlocks == afterBlocks,
        "paged-incompatible store must not allocate new paged blocks")
}

@Test func coordinatorPagedIncompatibleStoresDeepseekV4InDiskTier() {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let blockSize = 4
    let tmp = FileManager.default.temporaryDirectory
        .appendingPathComponent("dsv4-disk-tier-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: tmp) }

    let config = CacheCoordinatorConfig(
        usePagedCache: true,
        enableDiskCache: true,
        pagedBlockSize: blockSize,
        maxCacheBlocks: 20,
        diskCacheMaxGB: 1.0,
        diskCacheDir: tmp,
        modelKey: "dsv4-cache-contract-test"
    )
    let coordinator = CacheCoordinator(config: config)
    coordinator.setPagedIncompatible(true)

    let tokens = [101, 102, 103, 104]
    let v4 = DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)
    _ = v4.update(
        keys: MLXArray.ones([1, 1, tokens.count, 8]),
        values: MLXArray.ones([1, 1, tokens.count, 8]) * 2.0)
    v4.setHybridPool(branch: .compressor, value: MLXArray.ones([1, 2, 8]) * 3.0)
    v4.setHybridPool(branch: .indexer, value: MLXArray.ones([1, 2, 8]) * 4.0)

    coordinator.storeAfterGeneration(
        promptTokens: tokens,
        perLayerData: extractLayerData(from: [v4]),
        ssmStates: nil,
        cache: [v4])

    #expect(coordinator.pagedCache?.stats.allocatedBlocks == 0,
        "DSV4 paged-incompatible store must not allocate generic paged blocks")

    switch coordinator.fetch(tokens: tokens) {
    case .hit(let matchedTokens, let remainingTokens, let detail, let blocks, _, let diskArrays):
        #expect(matchedTokens == tokens.count)
        #expect(remainingTokens.isEmpty)
        #expect(detail == .disk)
        #expect(blocks.isEmpty,
            "DSV4 L2 hits should return disk arrays, not pinned paged blocks")
        #expect(diskArrays?["dsv4_0_keys"] != nil)
        #expect(diskArrays?["dsv4_0_pool_comp"] != nil)
        #expect(diskArrays?["tq_0_ck_indices"] == nil,
            "DSV4 default path must not store TurboQuant cache blocks")
    case .miss:
        Issue.record("DSV4 paged-incompatible coordinator should hit disk tier")
    }
}

@Test func coordinatorReleaseVolatilePreservesDiskTier() {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let tmp = FileManager.default.temporaryDirectory
        .appendingPathComponent("coordinator-release-volatile-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: tmp) }

    let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
        usePagedCache: true,
        enableDiskCache: true,
        pagedBlockSize: 4,
        maxCacheBlocks: 8,
        diskCacheMaxGB: 1.0,
        diskCacheDir: tmp,
        modelKey: "release-volatile-preserves-l2"))

    let tokens = [201, 202, 203, 204]
    let keys = MLXArray.ones([1, 1, tokens.count, 4])
    let values = MLXArray.ones([1, 1, tokens.count, 4]) * 2.0
    coordinator.storeAfterGeneration(
        promptTokens: tokens,
        perLayerData: [(keys: keys, values: values)],
        ssmStates: nil,
        cache: nil)

    #expect(coordinator.diskCache?.candidateTokenCounts(maxTokens: tokens.count) == [tokens.count])

    coordinator.releaseVolatile()

    #expect(coordinator.pagedCache?.stats.allocatedBlocks == 0)
    #expect(coordinator.diskCache?.candidateTokenCounts(maxTokens: tokens.count) == [tokens.count])

    switch coordinator.fetch(tokens: tokens) {
    case .hit(let matchedTokens, let remainingTokens, let detail, let blocks, _, let diskArrays):
        #expect(matchedTokens == tokens.count)
        #expect(remainingTokens.isEmpty)
        #expect(detail == .disk)
        #expect(blocks.isEmpty)
        #expect(!(diskArrays?.isEmpty ?? true))
    case .miss:
        Issue.record("releaseVolatile must preserve L2 disk entries for later reuse")
    }
}

@Test func coordinatorHybridZayaCCADiskHitDoesNotRequireSeparateSSMCompanion() {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let tmp = FileManager.default.temporaryDirectory
        .appendingPathComponent("zaya-cca-disk-tier-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: tmp) }

    let config = CacheCoordinatorConfig(
        usePagedCache: true,
        enableDiskCache: true,
        pagedBlockSize: 4,
        maxCacheBlocks: 20,
        diskCacheMaxGB: 1.0,
        diskCacheDir: tmp,
        modelKey: "zaya-cca-cache-contract-test"
    )
    let coordinator = CacheCoordinator(config: config)
    coordinator.setHybrid(true, requiresRecurrentSSMCompanion: false)
    coordinator.setPagedIncompatible(true)

    let tokens = [201, 202, 203, 204]
    let cache = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
    _ = cache.update(
        keys: MLXArray.ones([1, 1, tokens.count, 8], dtype: .bfloat16),
        values: MLXArray.ones([1, 1, tokens.count, 8], dtype: .bfloat16) * 2)
    cache.writeCCA(
        conv: MLXArray.ones([1, 4, 2], dtype: .float32) * 3,
        prev: MLXArray.ones([1, 8], dtype: .float32) * 4)

    coordinator.storeAfterGeneration(
        promptTokens: tokens,
        perLayerData: extractLayerData(from: [cache]),
        ssmStates: nil,
        cache: [cache])

    switch coordinator.fetch(tokens: tokens) {
    case .hit(let matchedTokens, let remainingTokens, let detail, let blocks, _, let diskArrays):
        #expect(matchedTokens == tokens.count)
        #expect(remainingTokens.isEmpty)
        #expect(detail == .disk)
        #expect(blocks.isEmpty)
        #expect(diskArrays?["zaya_0_conv_state"] != nil)
        #expect(diskArrays?["zaya_0_prev_hs"] != nil)
    case .miss:
        Issue.record(
            """
            ZayaCCACache v2 disk payload already contains path-dependent state; \
            it must not require a separate SSM companion entry to hit.
            """)
    }
}

@Test func coordinatorMiss() {
    let config = CacheCoordinatorConfig(
        usePagedCache: true,
        enableDiskCache: false,
        pagedBlockSize: 4,
        maxCacheBlocks: 20
    )
    let coordinator = CacheCoordinator(config: config)

    let result = coordinator.fetch(tokens: [1, 2, 3, 4, 5, 6, 7, 8])

    switch result {
    case .miss:
        break  // expected
    case .hit:
        Issue.record("Empty coordinator should return .miss")
    }
}

@Test func coordinatorPagedHit() {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let blockSize = 4
    let config = CacheCoordinatorConfig(
        usePagedCache: true,
        enableDiskCache: false,
        pagedBlockSize: blockSize,
        maxCacheBlocks: 20
    )
    let coordinator = CacheCoordinator(config: config)

    // Store 8 tokens (2 full blocks of size 4)
    let tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    // Per-layer data covering the full 8-token sequence (coordinator splits into blocks)
    let perLayerData: [(keys: MLXArray, values: MLXArray)?] = [
        (keys: MLXArray.zeros([1, 1, tokens.count, 8]),
         values: MLXArray.zeros([1, 1, tokens.count, 8]))
    ]

    coordinator.storeAfterGeneration(
        promptTokens: tokens,
        perLayerData: perLayerData,
        ssmStates: nil
    )

    // Fetch with the same prefix plus extra tokens
    let query = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    let result = coordinator.fetch(tokens: query)

    switch result {
    case .hit(let matchedTokens, let remainingTokens, let detail, let blocks, let ssmStates, _):
        #expect(matchedTokens == 8)
        #expect(remainingTokens == [9, 10])
        #expect(detail == .paged)
        #expect(blocks.count == 2)
        #expect(ssmStates == nil)
    case .miss:
        Issue.record("Should have hit the paged cache")
    }
}

@Test func coordinatorDiskTierRestoresLongestStoredPrefixForGrowingPrompt() {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let tmp = FileManager.default.temporaryDirectory
        .appendingPathComponent("disk-prefix-tier-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: tmp) }

    let config = CacheCoordinatorConfig(
        usePagedCache: false,
        enableDiskCache: true,
        pagedBlockSize: 4,
        maxCacheBlocks: 20,
        diskCacheMaxGB: 1.0,
        diskCacheDir: tmp,
        modelKey: "disk-prefix-cache-contract-test"
    )
    let coordinator = CacheCoordinator(config: config)

    let storedTokens = [1, 2, 3, 4, 5]
    let keys = MLXArray.ones([1, 1, storedTokens.count, 4])
    let values = MLXArray.ones([1, 1, storedTokens.count, 4]) * 2

    coordinator.storeAfterGeneration(
        promptTokens: storedTokens,
        perLayerData: [(keys: keys, values: values)],
        ssmStates: nil)

    let query = [1, 2, 3, 4, 5, 6, 7, 8]
    switch coordinator.fetch(tokens: query) {
    case .hit(let matchedTokens, let remainingTokens, let detail, let blocks, _, let diskArrays):
        #expect(matchedTokens == storedTokens.count)
        #expect(remainingTokens == [6, 7, 8])
        #expect(detail == .disk)
        #expect(blocks.isEmpty)
        #expect(diskArrays != nil)
    case .miss:
        Issue.record(
            "Disk tier should restore the longest stored prompt prefix after app/model unload")
    }
}

@Test func coordinatorSkipExactDiskBoundaryAlsoAppliesToIndexedCandidates() {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let tmp = FileManager.default.temporaryDirectory
        .appendingPathComponent("disk-skip-exact-indexed-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: tmp) }

    let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
        usePagedCache: false,
        enableDiskCache: true,
        diskCacheMaxGB: 1.0,
        diskCacheDir: tmp,
        modelKey: "disk-skip-exact-indexed-contract"))
    let exact = [1, 2, 3, 4, 5]
    let safePrefix = Array(exact.prefix(3))

    func store(_ tokens: [Int]) {
        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: [(
                keys: MLXArray.ones([1, 1, tokens.count, 4]),
                values: MLXArray.ones([1, 1, tokens.count, 4]) * 2)],
            ssmStates: nil)
    }
    store(exact)
    store(safePrefix)

    switch coordinator.fetch(tokens: exact, skipExactDiskBoundary: true) {
    case .hit(let matched, let remaining, let detail, _, _, _):
        #expect(matched == safePrefix.count)
        #expect(remaining == [4, 5])
        #expect(detail == .disk)
    case .miss:
        Issue.record("skip-exact lookup should use the longest safe indexed prefix")
    }

    switch coordinator.fetch(tokens: exact) {
    case .hit(let matched, let remaining, let detail, _, _, _):
        #expect(matched == exact.count)
        #expect(remaining.isEmpty)
        #expect(detail == .disk)
    case .miss:
        Issue.record("ordinary lookup should still admit the exact disk boundary")
    }
}

@Test func coordinatorSSMCompanion() {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let blockSize = 4
    let config = CacheCoordinatorConfig(
        usePagedCache: true,
        enableDiskCache: false,
        pagedBlockSize: blockSize,
        maxCacheBlocks: 20,
        ssmMaxEntries: 10
    )
    let coordinator = CacheCoordinator(config: config)
    coordinator.setHybrid(true)

    #expect(coordinator.isHybrid == true)

    // Store 8 tokens with SSM states
    let tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    let perLayerData: [(keys: MLXArray, values: MLXArray)?] = [
        (keys: MLXArray.zeros([1, 1, tokens.count, 8]),
         values: MLXArray.zeros([1, 1, tokens.count, 8]))
    ]
    let ssmStates = [MLXArray.ones([2, 4]), MLXArray.zeros([2, 4])]

    coordinator.storeAfterGeneration(
        promptTokens: tokens,
        perLayerData: perLayerData,
        ssmStates: ssmStates
    )

    // Fetch should return SSM states alongside the paged hit
    let result = coordinator.fetch(tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    switch result {
    case .hit(let matchedTokens, _, let detail, _, let fetchedSSM, _):
        #expect(matchedTokens == 8)
        #expect(detail == .paged)
        #expect(fetchedSSM != nil)
        #expect(fetchedSSM?.count == 2)
    case .miss:
        Issue.record("Should have hit the paged cache with SSM companion")
    }
}

@Test func coordinatorClear() {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let blockSize = 4
    let config = CacheCoordinatorConfig(
        usePagedCache: true,
        enableDiskCache: false,
        pagedBlockSize: blockSize,
        maxCacheBlocks: 20
    )
    let coordinator = CacheCoordinator(config: config)

    // Store tokens
    let tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    let perLayerData: [(keys: MLXArray, values: MLXArray)?] = [
        (keys: MLXArray.zeros([1, 1, tokens.count, 8]),
         values: MLXArray.zeros([1, 1, tokens.count, 8]))
    ]
    coordinator.storeAfterGeneration(
        promptTokens: tokens,
        perLayerData: perLayerData,
        ssmStates: nil
    )

    // Verify the data is cached
    let beforeClear = coordinator.fetch(tokens: tokens)
    switch beforeClear {
    case .hit:
        break  // expected
    case .miss:
        Issue.record("Data should be cached before clear")
    }

    // Clear all caches
    coordinator.clear()

    // Verify the data is gone
    let afterClear = coordinator.fetch(tokens: tokens)
    switch afterClear {
    case .miss:
        break  // expected
    case .hit:
        Issue.record("Data should be gone after clear")
    }
}
