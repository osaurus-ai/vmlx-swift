// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("CacheCoordinator topology contracts", .serialized)
struct CacheCoordinatorTopologyFocusedTests {
    @Test("dense paged cache restores the longest prefix and leaves suffix tokens")
    func densePagedCacheRestoresPrefix() {
        FocusedMLXTestSupport.withLock {
        let coordinator = makeCoordinator(usePagedCache: true, enableDiskCache: false)
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8]

        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: fakeLayerData(tokenCount: tokens.count),
            ssmStates: nil)

        switch coordinator.fetch(tokens: tokens + [9, 10]) {
        case .hit(let matchedTokens, let remainingTokens, let detail, let blocks, let ssmStates, _):
            #expect(matchedTokens == tokens.count)
            #expect(remainingTokens == [9, 10])
            #expect(detail == .paged)
            #expect(blocks.count == 2)
            #expect(ssmStates == nil)
            coordinator.release(blocks: blocks)
        case .miss:
            Issue.record("dense paged cache should hit the stored prefix")
        }
        }
    }

    @Test("media salt isolates otherwise identical paged prefixes")
    func mediaSaltIsolatesPagedPrefixes() {
        FocusedMLXTestSupport.withLock {
        let coordinator = makeCoordinator(usePagedCache: true, enableDiskCache: false)
        let tokens = [11, 12, 13, 14, 15, 16, 17, 18]

        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: fakeLayerData(tokenCount: tokens.count),
            ssmStates: nil,
            mediaSalt: "image-a")

        switch coordinator.fetch(tokens: tokens, mediaSalt: "image-a") {
        case .hit(_, _, let detail, let blocks, _, _):
            #expect(detail == .paged)
            coordinator.release(blocks: blocks)
        case .miss:
            Issue.record("same media salt should read its own paged entry")
        }

        if case .hit = coordinator.fetch(tokens: tokens, mediaSalt: "image-b") {
            Issue.record("different media salt must not hit the same paged entry")
        }
        if case .hit = coordinator.fetch(tokens: tokens, mediaSalt: nil) {
            Issue.record("text-only nil media salt must not hit a salted media entry")
        }
        }
    }

    @Test("post-prepare media aliases resolve effective cache keys")
    func postPrepareMediaAliasesResolveEffectiveCacheKeys() {
        FocusedMLXTestSupport.withLock {
        let coordinator = makeCoordinator(usePagedCache: true, enableDiskCache: false, blockSize: 1)
        let rawVideoPrompt = [101, 27, 27, 27, 102]
        let effectiveVideoPrompt = [101, 27, 102]
        let growingTurn = rawVideoPrompt + [201, 202]

        #expect(
            coordinator.resolvePostPrepareCacheKeyAlias(
                rawTokens: rawVideoPrompt,
                mediaSalt: "video-a") == nil)

        coordinator.recordPostPrepareCacheKeyAlias(
            rawTokens: rawVideoPrompt,
            effectiveTokens: effectiveVideoPrompt,
            mediaSalt: "video-a")
        coordinator.storeAfterGeneration(
            promptTokens: effectiveVideoPrompt,
            perLayerData: fakeLayerData(tokenCount: effectiveVideoPrompt.count),
            ssmStates: nil,
            mediaSalt: "video-a")

        #expect(
            coordinator.resolvePostPrepareCacheKeyAlias(
                rawTokens: rawVideoPrompt,
                mediaSalt: "video-a") == effectiveVideoPrompt)
        #expect(
            coordinator.resolvePostPrepareCacheKeyAlias(
                rawTokens: growingTurn,
                mediaSalt: "video-a") == effectiveVideoPrompt + [201, 202])
        #expect(
            coordinator.resolvePostPrepareCacheKeyAlias(
                rawTokens: rawVideoPrompt,
                mediaSalt: "video-b") == nil)
        #expect(
            coordinator.resolvePostPrepareCacheKeyAlias(
                rawTokens: rawVideoPrompt,
                mediaSalt: nil) == nil)

        switch coordinator.fetch(
            tokens: coordinator.resolvePostPrepareCacheKeyAlias(
                rawTokens: growingTurn,
                mediaSalt: "video-a") ?? [],
            mediaSalt: "video-a")
        {
        case .hit(let matchedTokens, let remainingTokens, let detail, let blocks, _, _):
            #expect(matchedTokens == effectiveVideoPrompt.count)
            #expect(remainingTokens == [201, 202])
            #expect(detail == .paged)
            coordinator.release(blocks: blocks)
        case .miss:
            Issue.record("resolved post-prepare alias should fetch the stored effective prefix")
        }
        }
    }

    @Test("prompt tool-surface edits never return a full cached prompt hit")
    func promptToolSurfaceEditsNeverReturnFullPromptHit() {
        FocusedMLXTestSupport.withLock {
        let tmp = makeTempDir("tool-surface-edit")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let coordinator = makeCoordinator(
            usePagedCache: true,
            enableDiskCache: true,
            diskCacheDir: tmp,
            modelKey: "tool-surface-focused",
            blockSize: 1)

        // These stand in for the already-rendered chat-template token stream:
        // common system prefix, tool-schema token, then user prompt token. If
        // Osaurus shrinks/expands the tool surface, vmlx must reuse only the
        // shared prefix and re-prefill the modified schema/user suffix.
        let original = [101, 201, 301]
        let modifiedToolSchema = [101, 202, 301]

        coordinator.storeAfterGeneration(
            promptTokens: original,
            perLayerData: fakeLayerData(tokenCount: original.count),
            ssmStates: nil)

        switch coordinator.fetch(tokens: modifiedToolSchema) {
        case .hit(let matchedTokens, let remainingTokens, _, let blocks, _, _):
            #expect(matchedTokens < original.count)
            #expect(remainingTokens == Array(modifiedToolSchema.dropFirst(matchedTokens)))
            coordinator.release(blocks: blocks)
        case .miss:
            break
        }
        }
    }

    @Test("hybrid paged hit requires matching companion state")
    func hybridPagedHitRequiresSSMCompanion() {
        FocusedMLXTestSupport.withLock {
        let coordinator = makeCoordinator(usePagedCache: true, enableDiskCache: false)
        coordinator.setHybrid(true)

        let tokens = [21, 22, 23, 24, 25, 26, 27, 28]
        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: fakeLayerData(tokenCount: tokens.count),
            ssmStates: nil)

        if case .hit = coordinator.fetch(tokens: tokens + [29]) {
            Issue.record("hybrid paged hit without SSM companion state is a false positive")
        }

        let ssmStates = [
            MLXArray.ones([1, 4], dtype: .float32),
            MLXArray.zeros([1, 4], dtype: .float32),
        ]
        evalArrays(ssmStates)
        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: fakeLayerData(tokenCount: tokens.count),
            ssmStates: ssmStates)

        switch coordinator.fetch(tokens: tokens + [29]) {
        case .hit(let matchedTokens, let remainingTokens, let detail, let blocks, let fetchedSSM, _):
            #expect(matchedTokens == tokens.count)
            #expect(remainingTokens == [29])
            #expect(detail == .paged)
            #expect(fetchedSSM?.count == ssmStates.count)
            coordinator.release(blocks: blocks)
        case .miss:
            Issue.record("hybrid paged cache should hit once SSM companion state exists")
        }
        }
    }

    @Test("hybrid paged hit rejects partial companion state")
    func hybridPagedHitRejectsPartialSSMCompanion() {
        FocusedMLXTestSupport.withLock {
        let coordinator = makeCoordinator(usePagedCache: true, enableDiskCache: false)
        coordinator.setHybrid(true)

        let tokens = [71, 72, 73, 74, 75, 76, 77, 78]
        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: fakeLayerData(tokenCount: tokens.count),
            ssmStates: nil)
        coordinator.ssmStateCache.store(
            ssmStates: [MLXArray.ones([1, 4], dtype: .float32)],
            tokens: tokens,
            boundary: tokens.count,
            isComplete: false)

        if case .hit = coordinator.fetch(tokens: tokens + [79]) {
            Issue.record("hybrid paged cache must not extend partial SSM companion state")
        }
        }
    }

    @Test("disk tier restores the longest stored prefix after paged cache is unavailable")
    func diskTierRestoresLongestStoredPrefix() {
        FocusedMLXTestSupport.withLock {
        let tmp = makeTempDir("disk-prefix")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let coordinator = makeCoordinator(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheDir: tmp,
            modelKey: "disk-prefix-focused")
        let tokens = [31, 32, 33, 34, 35]

        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: fakeLayerData(tokenCount: tokens.count),
            ssmStates: nil)

        switch coordinator.fetch(tokens: tokens + [36, 37, 38]) {
        case .hit(let matchedTokens, let remainingTokens, let detail, let blocks, _, let diskArrays):
            #expect(matchedTokens == tokens.count)
            #expect(remainingTokens == [36, 37, 38])
            #expect(detail == .disk)
            #expect(blocks.isEmpty)
            #expect(diskArrays != nil)
        case .miss:
            Issue.record("disk tier should restore the longest stored prompt boundary")
        }
        }
    }

    @Test("hybrid disk hit rejects partial companion state")
    func hybridDiskHitRejectsPartialSSMCompanion() {
        FocusedMLXTestSupport.withLock {
        let tmp = makeTempDir("hybrid-disk-partial-ssm")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let coordinator = makeCoordinator(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheDir: tmp,
            modelKey: "hybrid-disk-partial-ssm-focused")
        coordinator.setHybrid(true)
        let tokens = [81, 82, 83, 84]

        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: fakeLayerData(tokenCount: tokens.count),
            ssmStates: nil)
        coordinator.ssmStateCache.store(
            ssmStates: [MLXArray.ones([1, 4], dtype: .float32)],
            tokens: tokens,
            boundary: tokens.count,
            isComplete: false)

        if case .hit = coordinator.fetch(tokens: tokens + [85]) {
            Issue.record("hybrid disk cache must not extend partial SSM companion state")
        }
        }
    }

    @Test("hybrid disk hit rejects legacy KV-only payload without companion state")
    func hybridDiskHitRejectsLegacyKVOnlyPayloadWithoutSSMCompanion() {
        FocusedMLXTestSupport.withLock {
        let tmp = makeTempDir("hybrid-disk-legacy-kv-only")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let coordinator = makeCoordinator(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheDir: tmp,
            modelKey: "nemotron-ultra-hybrid-legacy-kv-only")
        coordinator.setHybrid(true)
        let tokens = [91, 92, 93, 94, 95]

        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: fakeLayerData(tokenCount: tokens.count),
            ssmStates: nil)

        if case .hit = coordinator.fetch(tokens: tokens + [96]) {
            Issue.record(
                "Nemotron-style hybrid disk cache must not accept legacy KV-only L2 payloads without complete SSM companion state")
        }
        #expect(coordinator.ssmStateCache.snapshotStats().hits == 0)
        #expect(coordinator.ssmStateCache.snapshotStats().misses > 0)
        }
    }

    @Test("DSV4 paged-incompatible cache skips paged blocks and restores CSA HSA pools from disk")
    func dsv4PagedIncompatibleUsesDiskWithPools() {
        FocusedMLXTestSupport.withLock {
        let tmp = makeTempDir("dsv4-disk")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let coordinator = makeCoordinator(
            usePagedCache: true,
            enableDiskCache: true,
            diskCacheDir: tmp,
            modelKey: "dsv4-focused")
        coordinator.setPagedIncompatible(true)

        let tokens = [41, 42, 43, 44]
        let cache = DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)
        _ = cache.update(
            keys: MLXArray.ones([1, 1, tokens.count, 8], dtype: .bfloat16),
            values: MLXArray.ones([1, 1, tokens.count, 8], dtype: .bfloat16) * Float(2))
        cache.setHybridPool(
            branch: .compressor,
            value: MLXArray.ones([1, 2, 8], dtype: .bfloat16) * Float(3))
        cache.setHybridPool(
            branch: .indexer,
            value: MLXArray.ones([1, 2, 8], dtype: .bfloat16) * Float(4))

        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: extractLayerData(from: [cache]),
            ssmStates: nil,
            cache: [cache])

        #expect(coordinator.pagedCache?.stats.allocatedBlocks == 0)

        switch coordinator.fetch(tokens: tokens) {
        case .hit(let matchedTokens, let remainingTokens, let detail, let blocks, _, let diskArrays):
            #expect(matchedTokens == tokens.count)
            #expect(remainingTokens.isEmpty)
            #expect(detail == .disk)
            #expect(blocks.isEmpty)
            #expect(diskArrays?["dsv4_0_keys"] != nil)
            #expect(diskArrays?["dsv4_0_pool_comp"] != nil)
            #expect(diskArrays?["dsv4_0_pool_idx"] != nil)
        case .miss:
            Issue.record("DSV4 paged-incompatible coordinator should hit disk tier")
        }
        }
    }

    @Test("ZAYA CCA format-v2 disk payload restores growing prompt boundary")
    func zayaCCADiskHitRestoresGrowingPromptBoundary() {
        FocusedMLXTestSupport.withLock {
        let tmp = makeTempDir("zaya-cca-disk")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let coordinator = makeCoordinator(
            usePagedCache: true,
            enableDiskCache: true,
            diskCacheDir: tmp,
            modelKey: "zaya-cca-focused")
        coordinator.setHybrid(true, requiresRecurrentSSMCompanion: false)
        coordinator.setPagedIncompatible(true)

        let tokens = [51, 52, 53, 54]
        let cache = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        _ = cache.update(
            keys: MLXArray.ones([1, 1, tokens.count, 8], dtype: .bfloat16),
            values: MLXArray.ones([1, 1, tokens.count, 8], dtype: .bfloat16) * Float(2))
        cache.writeCCA(
            conv: MLXArray.ones([1, 4, 2], dtype: .float32) * Float(3),
            prev: MLXArray.ones([1, 8], dtype: .float32) * Float(4))

        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: extractLayerData(from: [cache]),
            ssmStates: nil,
            cache: [cache])

        switch coordinator.fetch(tokens: tokens + [55]) {
        case .hit(let matchedTokens, let remainingTokens, let detail, let blocks, _, let diskArrays):
            #expect(matchedTokens == tokens.count)
            #expect(remainingTokens == [55])
            #expect(detail == .disk)
            #expect(blocks.isEmpty)
            #expect(diskArrays?["zaya_0_conv_state"] != nil)
            #expect(diskArrays?["zaya_0_prev_hs"] != nil)
        case .miss:
            Issue.record("ZAYA CCA format-v2 payload should restore a complete growing prompt boundary")
        }
        }
    }

    @Test("ZAYA CCA state is not duplicated into SSM companion cache")
    func zayaCCADoesNotUseSSMCompanionExtraction() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Cache/CacheHelpers.swift",
            encoding: .utf8)

        #expect(source.contains("} else if layer is ZayaCCACache {"))
        #expect(source.contains("LayerKind.zayaCCA stores keys, values,"))
        #expect(source.contains("ZAYA CCA restore is owned by the LayerKind.zayaCCA disk payload"))
        #expect(!source.contains("let (conv, prev) = zaya.readCCA()"))
        #expect(!source.contains("zaya.writeCCA(conv: conv, prev: prev)"))
    }

    @Test("ZAYA CCA topology advertises CCA companion, not recurrent SSM")
    func zayaCCATopologyUsesCCACompanionTag() {
        let topology = ModelCacheTopologySnapshot(
            layerCount: 2,
            kvLayerCount: 1,
            zayaCCALayerCount: 1
        )

        #expect(topology.requiresSSMCompanionState)
        #expect(topology.requiresZayaCCACompanionState)
        #expect(!topology.requiresRecurrentSSMCompanionState)
        #expect(topology.topologyTags.contains("zayaCCALayers=1"))
        #expect(topology.topologyTags.contains("companion=zaya-cca"))
        #expect(!topology.topologyTags.contains("companion=ssm"))
    }

    @Test("hybrid disk media-salt prompt boundary returns exact hit")
    func hybridDiskMediaSaltPromptBoundaryReturnsExactHit() {
        FocusedMLXTestSupport.withLock {
        let tmp = makeTempDir("zaya-cca-salted-exact")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let coordinator = makeCoordinator(
            usePagedCache: true,
            enableDiskCache: true,
            diskCacheDir: tmp,
            modelKey: "zaya-cca-salted-exact-focused")
        coordinator.setHybrid(true, requiresRecurrentSSMCompanion: false)
        coordinator.setPagedIncompatible(true)

        let tokens = [61, 62, 63, 64]
        let cache = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        _ = cache.update(
            keys: MLXArray.ones([1, 1, tokens.count, 8], dtype: .bfloat16),
            values: MLXArray.ones([1, 1, tokens.count, 8], dtype: .bfloat16) * Float(2))
        cache.writeCCA(
            conv: MLXArray.ones([1, 4, 2], dtype: .float32) * Float(3),
            prev: MLXArray.ones([1, 8], dtype: .float32) * Float(4))

        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: extractLayerData(from: [cache]),
            ssmStates: nil,
            cache: [cache],
            mediaSalt: "image-a")

        switch coordinator.fetch(tokens: tokens, mediaSalt: "image-a") {
        case .hit(let matchedTokens, let remainingTokens, let detail, let blocks, _, let diskArrays):
            #expect(matchedTokens == tokens.count)
            #expect(remainingTokens.isEmpty)
            #expect(detail == .disk)
            #expect(blocks.isEmpty)
            #expect(diskArrays?["zaya_0_conv_state"] != nil)
            #expect(diskArrays?["zaya_0_prev_hs"] != nil)
        case .miss:
            Issue.record("hybrid disk-backed media-salt prompt boundary must hit exactly")
        }

        if case .hit = coordinator.fetch(tokens: tokens, mediaSalt: "image-b") {
            Issue.record("different media salt must not hit exact hybrid disk boundary")
        }
        }
    }

    @Test("typed disk need and paged incompatibility are distinct cache contracts")
    func pathDependentAndSlidingCachesRequireDiskBackedRestore() {
        FocusedMLXTestSupport.withLock {
        prepareMLXMetallibForCacheTopologyTests()

        #expect(cacheRequiresDiskBackedCoordinatorRestore([MambaCache(), KVCacheSimple()]))
        #expect(cacheRequiresDiskBackedCoordinatorRestore([ArraysCache(size: 2), KVCacheSimple()]))
        #expect(cacheRequiresDiskBackedCoordinatorRestore([ZayaCCACache(), KVCacheSimple()]))
        #expect(cacheRequiresDiskBackedCoordinatorRestore([RotatingKVCache(maxSize: 32), KVCacheSimple()]))
        #expect(cacheRequiresDiskBackedCoordinatorRestore([DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)]))
        #expect(!cacheRequiresDiskBackedCoordinatorRestore([KVCacheSimple(), KVCacheSimple()]))

        #expect(!cacheCannotUsePagedCoordinatorRestore([MambaCache(), KVCacheSimple()]))
        #expect(!cacheCannotUsePagedCoordinatorRestore([ArraysCache(size: 2), KVCacheSimple()]))
        #expect(!cacheCannotUsePagedCoordinatorRestore([
            MambaCache(), TurboQuantKVCache(keyBits: 4, valueBits: 4),
        ]))
        #expect(cacheCannotUsePagedCoordinatorRestore([ZayaCCACache(), KVCacheSimple()]))
        #expect(cacheCannotUsePagedCoordinatorRestore([RotatingKVCache(maxSize: 32), KVCacheSimple()]))
        #expect(cacheCannotUsePagedCoordinatorRestore([DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)]))
        #expect(!cacheCannotUsePagedCoordinatorRestore([KVCacheSimple(), KVCacheSimple()]))
        #expect(cacheCannotUsePagedCoordinatorRestore([BaseKVCache()]))
        }
    }

    @Test("prompt snapshot detaches Ornith GDN state from later in-place updates")
    func promptSnapshotDetachesArraysCacheState() {
        FocusedMLXTestSupport.withLock {
            let live = ArraysCache(size: 1)
            live[0] = MLXArray.ones([1, 4], dtype: .float32)
            live.offset = 4
            MLX.eval(live)

            let snapshot = makePromptBoundaryCacheSnapshot(from: [live])
            let detached = snapshot[0] as? ArraysCache
            #expect(detached != nil)
            #expect(detached?.offset == 4)

            // Ornith's GatedDeltaNet path uses the same setter after every
            // forward; with the old ellipsis-view copy this overwrote the
            // prompt snapshot as well.
            live[0] = MLXArray.ones([1, 4], dtype: .float32) * Float(9)
            live.offset = 5
            MLX.eval(live)

            #expect(detached?.offset == 4)
            #expect(detached?.state[0].sum().item(Float.self) == 4)
            #expect(live.state[0].sum().item(Float.self) == 36)
        }
    }

    @Test("Nemotron Omni Mamba plus TurboQuant topology restores a paged partial prefix atomically")
    func nemotronOmniTurboQuantPagedPartialRestore() {
        FocusedMLXTestSupport.withLock {
        let tokens = Array(1...8)
        let coordinator = makeCoordinator(
            usePagedCache: true,
            enableDiskCache: false,
            modelKey: "nemotron-omni-tq-paged-focused")
        coordinator.setHybrid(true, requiresRecurrentSSMCompanion: true)

        let source = makeNemotronOmniCache(tokenCount: tokens.count, populated: true)
        let topology = ModelCacheTopologySnapshot(cache: source)
        #expect(topology.layerCount == 29)
        #expect(topology.mambaLayerCount == 23)
        #expect(topology.turboQuantKVLayerCount == 6)
        #expect(!cacheCannotUsePagedCoordinatorRestore(source))

        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            // Match the real generation call sites for disk-backed hybrid
            // topologies: the coordinator must derive the paged KV payload
            // itself when the user explicitly enables paged caching.
            perLayerData: [],
            ssmStates: extractSSMStates(from: source),
            cache: source)

        switch coordinator.fetch(tokens: tokens + [9, 10, 11]) {
        case .hit(
            let matchedTokens, let remainingTokens, let detail,
            let blocks, let ssmStates, _):
            #expect(matchedTokens == tokens.count)
            #expect(remainingTokens == [9, 10, 11])
            #expect(detail == .paged)
            #expect(ssmStates?.count == 46)

            let restored = makeNemotronOmniCache(tokenCount: 0, populated: false)
            #expect(restoreLayerData(from: blocks, into: restored) == tokens.count)
            coordinator.release(blocks: blocks)
            if let ssmStates {
                restoreSSMStates(ssmStates, into: restored, boundary: matchedTokens)
            }

            var mambaCount = 0
            var tqCount = 0
            for layer in restored {
                if let mamba = layer as? MambaCache {
                    mambaCount += 1
                    #expect(mamba.state.count == 2)
                    #expect(mamba.offset == tokens.count)
                } else if let tq = layer as? TurboQuantKVCache {
                    tqCount += 1
                    if case .compressed = tq.phase {
                        // Expected: decoded paged KV stays in the compressed
                        // TQ lifecycle without a second lossy encode.
                    } else {
                        Issue.record("paged TQ restore must remain compressed")
                    }
                    #expect(tq.offset == tokens.count)
                    #expect(tq.state.first?.dim(2) == tokens.count)
                }
            }
            #expect(mambaCount == 23)
            #expect(tqCount == 6)
        case .miss:
            Issue.record(
                "Nemotron hybrid TQ paged prefix should restore with a complete same-boundary companion"
            )
        }
        }
    }

    @Test("paged eviction falls through to the persisted disk boundary")
    func pagedEvictionFallsThroughToDisk() {
        FocusedMLXTestSupport.withLock {
        let tmp = makeTempDir("paged-eviction-disk-fallback")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let coordinator = makeCoordinator(
            usePagedCache: true,
            enableDiskCache: true,
            diskCacheDir: tmp,
            modelKey: "paged-eviction-disk-fallback-focused",
            maxCacheBlocks: 3)
        coordinator.setHybrid(true)

        let first = Array(101...108)
        let second = Array(201...208)
        let companion = [
            MLXArray.ones([1, 4], dtype: .float32),
            MLXArray.zeros([1, 4], dtype: .float32),
        ]
        evalArrays(companion)
        coordinator.storeAfterGeneration(
            promptTokens: first,
            perLayerData: fakeLayerData(tokenCount: first.count),
            ssmStates: companion)
        coordinator.storeAfterGeneration(
            promptTokens: second,
            perLayerData: fakeLayerData(tokenCount: second.count),
            ssmStates: companion)

        #expect((coordinator.pagedCache?.snapshotStats().evictions ?? 0) >= 2)
        switch coordinator.fetch(tokens: first + [109, 110]) {
        case .hit(let matched, let remaining, let detail, let blocks, let ssm, let arrays):
            #expect(matched == first.count)
            #expect(remaining == [109, 110])
            #expect(detail == .disk)
            #expect(blocks.isEmpty)
            #expect(ssm?.count == companion.count)
            #expect(arrays != nil)
            #expect((coordinator.diskCache?.snapshotStats().hits ?? 0) > 0)
        case .miss:
            Issue.record("evicted paged prefix should fall through to its L2 disk record")
        }
        }
    }

    @Test("paged-off fresh coordinator restores a partial hybrid prefix from disk and companion L2")
    func pagedOffFreshCoordinatorRestoresHybridDiskPartialPrefix() {
        FocusedMLXTestSupport.withLock {
        let tmp = makeTempDir("disk-only-partial-restart")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let modelKey = "disk-only-partial-restart-focused"
        let tokens = Array(301...308)
        let companion = [
            MLXArray.ones([1, 4], dtype: .float32) * Float(3),
            MLXArray.ones([1, 4], dtype: .float32) * Float(4),
        ]
        evalArrays(companion)

        do {
            let writer = makeCoordinator(
                usePagedCache: false,
                enableDiskCache: true,
                diskCacheDir: tmp,
                modelKey: modelKey)
            writer.setHybrid(true)
            writer.storeAfterGeneration(
                promptTokens: tokens,
                perLayerData: fakeLayerData(tokenCount: tokens.count),
                ssmStates: companion)
            #expect(writer.pagedCache == nil)
            #expect((writer.diskCache?.snapshotStats().stores ?? 0) > 0)
        }

        let reader = makeCoordinator(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheDir: tmp,
            modelKey: modelKey)
        reader.setHybrid(true)
        switch reader.fetch(tokens: tokens + [309, 310, 311]) {
        case .hit(let matched, let remaining, let detail, let blocks, let ssm, let arrays):
            #expect(matched == tokens.count)
            #expect(remaining == [309, 310, 311])
            #expect(detail == .disk)
            #expect(blocks.isEmpty)
            #expect(ssm?.count == companion.count)
            #expect(arrays != nil)
            #expect((reader.diskCache?.snapshotStats().hits ?? 0) > 0)
            #expect(reader.ssmStateCache.snapshotStats().hits > 0)
        case .miss:
            Issue.record("fresh paged-off coordinator should restore the longest partial L2 prefix")
        }
        }
    }

    @Test("state-only hybrid cache never publishes a token-only paged hit")
    func stateOnlyHybridSkipsPagedAndFallsThroughToDisk() {
        FocusedMLXTestSupport.withLock {
        let tmp = makeTempDir("state-only-paged-false-hit")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let coordinator = makeCoordinator(
            usePagedCache: true,
            enableDiskCache: true,
            diskCacheDir: tmp,
            modelKey: "state-only-paged-false-hit-focused")
        coordinator.setHybrid(true, requiresRecurrentSSMCompanion: true)

        let tokens = Array(401...408)
        let mamba = MambaCache()
        mamba.state = [
            MLXArray.ones([1, 2, 4], dtype: .float32),
            MLXArray.ones([1, 2, 4], dtype: .float32) * Float(2),
        ]
        mamba.offset = tokens.count
        evalArrays(mamba.state)
        let cache: [any KVCache] = [mamba]

        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: [],
            ssmStates: extractSSMStates(from: cache),
            cache: cache)

        #expect((coordinator.pagedCache?.snapshotStats().allocatedBlocks ?? -1) == 0)
        switch coordinator.fetch(tokens: tokens + [409]) {
        case .hit(let matched, let remaining, let detail, let blocks, let ssm, let arrays):
            #expect(matched == tokens.count)
            #expect(remaining == [409])
            #expect(detail == .disk)
            #expect(blocks.isEmpty)
            #expect(ssm?.count == 2)
            #expect(arrays != nil)
        case .miss:
            Issue.record("state-only hybrid should bypass paged and retain its valid typed disk hit")
        }
        }
    }

    @Test("dynamic reasoning scope isolates every coordinator hash tier")
    func dynamicReasoningScopeIsolatesCoordinatorHashTiers() {
        let tokens = [101, 102, 103, 104]
        let reasoningOff = "bundle-a|kv=fp16|reasoning=off"
        let reasoningOn = "bundle-a|kv=fp16|reasoning=on"

        #expect(
            DiskCache.hashTokens(tokens, modelKey: reasoningOff)
            != DiskCache.hashTokens(tokens, modelKey: reasoningOn))
        #expect(
            CacheBlock.computeBlockHash(
                parentHash: nil,
                tokenIds: tokens,
                modelKey: reasoningOff)
            != CacheBlock.computeBlockHash(
                parentHash: nil,
                tokenIds: tokens,
                modelKey: reasoningOn))
        #expect(
            SSMStateCache.makeKey(
                tokens: tokens,
                boundary: tokens.count,
                modelKey: reasoningOff)
            != SSMStateCache.makeKey(
                tokens: tokens,
                boundary: tokens.count,
                modelKey: reasoningOn))
    }

    @Test("SSM companion disk key stays identical to memory key with model and media salt")
    func ssmCompanionDiskKeyMatchesMemoryKeyWithModelAndMediaSalt() {
        let tokens = [111, 112, 113, 114, 115]
        let modelKey = "nemotron-ultra|reasoning=deepseek_r1|tools=nemotron|kv=tq3x3"
        let mediaSalt = "text-only|no-media-processors"

        let memoryKey = SSMStateCache.makeKey(
            tokens: tokens,
            boundary: 4,
            mediaSalt: mediaSalt,
            modelKey: modelKey)
        let diskKey = SSMCompanionDiskStore.keyFor(
            tokens: tokens,
            boundary: 4,
            mediaSalt: mediaSalt,
            modelKey: modelKey)

        #expect(diskKey == memoryKey)
        #expect(
            SSMCompanionDiskStore.keyFor(
                tokens: tokens,
                boundary: 4,
                mediaSalt: "text-only|different-policy",
                modelKey: modelKey) != diskKey)
        #expect(
            SSMCompanionDiskStore.keyFor(
                tokens: tokens,
                boundary: 4,
                mediaSalt: mediaSalt,
                modelKey: "other-model") != diskKey)
    }

    @Test("cache scope salt includes semantic reasoning and tool-choice keys")
    func cacheScopeSaltIncludesSemanticReasoningAndToolChoiceKeys() {
        #expect(cacheScopeSalt(from: ["reasoning_effort": "high"]) == "effort=high")
        #expect(cacheScopeSalt(from: ["reasoning_effort": " No_Think "]) == "effort=no_think")
        #expect(cacheScopeSalt(from: [
            "enable_thinking": true,
            "reasoning_effort": "low",
        ]) == "reasoning=on|effort=low")
        #expect(cacheScopeSalt(from: [
            "enable_thinking": false,
            "reasoning_effort": "max",
        ]) == "reasoning=off|effort=max")
        #expect(cacheScopeSalt(from: [
            "tool_choice": "required",
            "tool_choice_name": "Line_Count",
        ]) == "tool=required|tool_name=line_count")
        #expect(cacheScopeSalt(from: [
            "enable_thinking": false,
            "tool_choice": "required",
            "tool_choice_name": "file_read",
        ]) == "reasoning=off|tool=required|tool_name=file_read")
        #expect(cacheScopeSalt(from: [
            "tool_choice": "auto",
            "ui_panel": "visible",
            "temperature_source": "default",
        ]) == nil)
    }

    @Test("cache policy salt always scopes text-only requests")
    func cachePolicySaltAlwaysScopesTextOnlyRequests() {
        let tokenArray = MLXArray([Int32(701), Int32(702), Int32(703)])
            .expandedDimensions(axis: 0)
        let text = LMInput.Text(tokens: tokenArray)
        let input = LMInput(text: text)

        #expect(computeCacheSalt(for: input) == nil)
        #expect(computeCacheSalt(for: input, parameters: GenerateParameters()) != nil)
        #expect(
            computeCacheSalt(for: input, parameters: GenerateParameters())
            != computeCacheSalt(
                for: LMInput(text: text, cacheScopeSalt: "reasoning=on"),
                parameters: GenerateParameters()))
    }

    @Test("Known unsafe required-tool rows skip disk-backed prompt seed boundary")
    func knownUnsafeRequiredToolRowsSkipDiskBackedPromptSeedBoundary() throws {
        let batchSource = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)
        let evaluateSource = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)

        #expect(batchSource.contains("shouldSkipDiskBackedToolPromptSeedBoundary"))
        #expect(batchSource.contains("slot.disablesGeneratedCacheBoundary"))
        #expect(batchSource.contains(#"modelName.contains("lfm2.5")"#))
        #expect(batchSource.contains(#"modelName.contains("mxfp8")"#))
        #expect(batchSource.contains(#"modelName.contains("gemma-4")"#))
        #expect(batchSource.contains(#"modelName.contains("mxfp4")"#))
        #expect(batchSource.contains("!shouldSkipDiskBackedToolPromptSeedBoundary(for: slot)"))
        #expect(batchSource.contains("shouldDisableDiskBackedRequiredToolRestore"))
        #expect(batchSource.contains("disableDiskBackedRequiredToolRestore: deferredDisableRestore"))
        #expect(batchSource.contains("Skipped disk-backed required-tool cache restore"))
        #expect(batchSource.contains("Skipped disk-backed tool prompt seed boundary"))
        #expect(evaluateSource.contains("disableDiskBackedRequiredToolRestore"))
        #expect(evaluateSource.contains("TokenIterator: skipped disk-backed required-tool cache restore"))
        #expect(evaluateSource.contains("requiresDiskBackedRestore && disableDiskBackedRequiredToolRestore"))
    }

    @Test("KV policy changes dynamic cache salt")
    func kvPolicyChangesDynamicCacheSalt() {
        let tokenArray = MLXArray([Int32(801), Int32(802), Int32(803)])
            .expandedDimensions(axis: 0)
        let input = LMInput(text: LMInput.Text(tokens: tokenArray))

        let plain = GenerateParameters()
        let affine = GenerateParameters(kvBits: 4, kvGroupSize: 64)
        let turboQuant = GenerateParameters(
            kvMode: .turboQuant(keyBits: 3, valueBits: 3))
        let rotating = GenerateParameters(maxKVSize: 4096)

        let plainSalt = computeCacheSalt(for: input, parameters: plain)
        let affineSalt = computeCacheSalt(for: input, parameters: affine)
        let turboSalt = computeCacheSalt(for: input, parameters: turboQuant)
        let rotatingSalt = computeCacheSalt(for: input, parameters: rotating)

        #expect(plainSalt != nil)
        #expect(plainSalt != affineSalt)
        #expect(plainSalt != turboSalt)
        #expect(plainSalt != rotatingSalt)
        #expect(affineSalt != turboSalt)
    }

    private func makeCoordinator(
        usePagedCache: Bool,
        enableDiskCache: Bool,
        diskCacheDir: URL? = nil,
        modelKey: String = "cache-topology-focused",
        blockSize: Int = 4,
        maxCacheBlocks: Int = 40
    ) -> CacheCoordinator {
        prepareMLXMetallibForCacheTopologyTests()

        return CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: usePagedCache,
            enableDiskCache: enableDiskCache,
            pagedBlockSize: blockSize,
            maxCacheBlocks: maxCacheBlocks,
            diskCacheMaxGB: 1.0,
            diskCacheDir: diskCacheDir,
            modelKey: modelKey))
    }

    private func fakeLayerData(tokenCount: Int) -> [(keys: MLXArray, values: MLXArray)?] {
        let keys = MLXArray.ones([1, 1, tokenCount, 4], dtype: .bfloat16)
        let values = MLXArray.ones([1, 1, tokenCount, 4], dtype: .bfloat16) * Float(0.5)
        MLX.eval(keys, values)
        return [(keys: keys, values: values)]
    }

    /// Exact cache-bearing layer order from the current local
    /// Nemotron-Omni-Nano JANGTQ4 bundle's `hybrid_override_pattern`:
    /// `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME`.
    /// MLP/MoE (`E`) positions compact away in `NemotronH.newCache`.
    private func makeNemotronOmniCache(
        tokenCount: Int,
        populated: Bool
    ) -> [any KVCache] {
        let pattern = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
        return pattern.compactMap { symbol -> (any KVCache)? in
            switch symbol {
            case "M":
                let mamba = MambaCache()
                if populated {
                    mamba.state = [
                        MLXArray.ones([1, 2, 4], dtype: .float32),
                        MLXArray.ones([1, 2, 4], dtype: .float32) * Float(2),
                    ]
                    mamba.offset = tokenCount
                    evalArrays(mamba.state)
                }
                return mamba
            case "*":
                let tq = TurboQuantKVCache(
                    keyBits: 4, valueBits: 4, sinkTokens: 0, residualTokens: 0)
                if populated {
                    let keys = MLXArray.ones(
                        [1, 1, tokenCount, 8], dtype: .bfloat16)
                    let values = MLXArray.ones(
                        [1, 1, tokenCount, 8], dtype: .bfloat16) * Float(0.5)
                    tq.restoreFromDecodedKV(
                        keys: keys, values: values, sourceOffset: tokenCount)
                }
                return tq
            default:
                return nil
            }
        }
    }

    private func evalArrays(_ arrays: [MLXArray]) {
        switch arrays.count {
        case 0:
            return
        case 1:
            MLX.eval(arrays[0])
        case 2:
            MLX.eval(arrays[0], arrays[1])
        default:
            MLX.eval(arrays)
        }
    }

    private func makeTempDir(_ prefix: String) -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-\(prefix)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test("Gemma4 processor normalizes tool schemas before applying chat template")
    func gemma4ProcessorNormalizesToolSchemasBeforeChatTemplate() throws {
        let sourceURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Libraries/MLXVLM/Models/Gemma4.swift")
        let source = try String(contentsOf: sourceURL)
        #expect(source.contains("normalizedToolsForChatTemplate(input.tools)"))
        #expect(source.contains("tools: chatTemplateTools"))
    }
}

@Suite("BatchArraysCache focused contracts", .serialized)
struct BatchArraysCacheFocusedTests {
    @Test("splitBack propagates model-mutated recurrent offset")
    func splitBackPropagatesModelMutatedOffset() {
        FocusedMLXTestSupport.withLock {
            let cache0 = MambaCache()
            let cache1 = MambaCache()
            cache0.offset = 10
            cache1.offset = 8
            cache0[0] = MLXArray.ones([1, 2, 4], dtype: .float32)
            cache0[1] = MLXArray.ones([1, 2, 4], dtype: .float32)
            cache1[0] = MLXArray.ones([1, 2, 4], dtype: .float32) * 3
            cache1[1] = MLXArray.ones([1, 2, 4], dtype: .float32) * 5

            let batch = BatchArraysCache(slotCaches: [cache0, cache1])
            batch[0] = MLXArray.ones([2, 2, 4], dtype: .float32) * 7
            batch[1] = MLXArray.ones([2, 2, 4], dtype: .float32) * 11

            // Mirrors Qwen35GatedDeltaNet: the model mutates the wrapper's
            // scalar offset directly instead of calling BatchArraysCache.advance.
            batch.offset += 1
            batch.splitBack()

            #expect(batch.offset == 11)
            #expect(batch.offsetArray.asArray(Int32.self) == [11, 9])
            #expect(cache0.offset == 11)
            #expect(cache1.offset == 9)
            #expect(cache0[0]?.shape == [1, 2, 4])
            #expect(cache0[1]?.shape == [1, 2, 4])
            #expect(cache1[0]?.shape == [1, 2, 4])
            #expect(cache1[1]?.shape == [1, 2, 4])
        }
    }
}

@Suite("Gemma4 cache topology focused contracts")
struct Gemma4CacheTopologyFocusedTests {
    private static func mixedRotatingSimpleCache(slidingWindow: Int = 64) -> [KVCache] {
        [
            RotatingKVCache(maxSize: slidingWindow, keep: 0),
            KVCacheSimple(),
            RotatingKVCache(maxSize: slidingWindow, keep: 0),
            KVCacheSimple(),
        ]
    }

    private static func allRotatingCache(slidingWindow: Int = 64, maxKVSize: Int = 2048) -> [KVCache] {
        [
            RotatingKVCache(maxSize: slidingWindow, keep: 0),
            RotatingKVCache(maxSize: maxKVSize, keep: 4),
            RotatingKVCache(maxSize: slidingWindow, keep: 0),
            RotatingKVCache(maxSize: maxKVSize, keep: 4),
        ]
    }

    @Test("Mixed Rotating+Simple Gemma4 cache classifies as heterogeneous")
    func cacheWithoutMaxKVSizeIsHeterogeneous() {
        let cache = Self.mixedRotatingSimpleCache()

        #expect(cache.count == 4)
        #expect(cache[0] is RotatingKVCache)
        #expect(cache[1] is KVCacheSimple)
        #expect(cache[2] is RotatingKVCache)
        #expect(cache[3] is KVCacheSimple)

        let family = CacheFamily.classify(cache)
        #expect(family == .heterogeneous)
        #expect(family.isCompileEligibleAtCurrentStage == false)
    }

    @Test("All-Rotating Gemma4 cache classifies as rotating")
    func cacheWithMaxKVSizeIsRotating() {
        let cache = Self.allRotatingCache()

        #expect(cache.count == 4)
        for layer in cache {
            #expect(layer is RotatingKVCache)
        }

        let family = CacheFamily.classify(cache)
        #expect(family == .rotating)
        #expect(family.isCompileEligibleAtCurrentStage == true)
    }

    @Test("Gemma4 compile policy follows actual cache topology")
    func compilePolicyFollowsActualCacheTopology() throws {
        let config = try JSONDecoder().decode(
            Gemma4TextConfiguration.self,
            from: Data(Self.minimalGemma4Config.utf8))
        let model = Gemma4TextModel(config)

        let defaultFamily = CacheFamily.classify(model.newCache(parameters: nil))
        #expect(defaultFamily == .heterogeneous)
        #expect(defaultFamily.isCompileEligibleAtCurrentStage == false)

        var params = GenerateParameters()
        params.maxKVSize = 2048
        let boundedFamily = CacheFamily.classify(model.newCache(parameters: params))
        #expect(boundedFamily == .rotating)
        #expect(boundedFamily.isCompileEligibleAtCurrentStage == true)
    }

    @Test("Full-attention rotating cache uses the attention-sink shape")
    func attentionSinkContract() {
        let sliding = RotatingKVCache(maxSize: 1024, keep: 0)
        let full = RotatingKVCache(maxSize: 4096, keep: 4)

        #expect(sliding.maxSize == 1024)
        #expect(full.maxSize == 4096)
    }

    @Test("actual Gemma4TextModel newCache matches mixed and maxKVSize topologies")
    func actualModelNewCacheMatchesTopologyContract() throws {
        let config = try JSONDecoder().decode(
            Gemma4TextConfiguration.self,
            from: Data(Self.minimalGemma4Config.utf8))
        let model = Gemma4TextModel(config)

        let defaultCache = model.newCache(parameters: nil)
        #expect(defaultCache.count == 4)
        #expect(defaultCache[0] is RotatingKVCache)
        #expect(defaultCache[1] is KVCacheSimple)
        #expect(defaultCache[2] is RotatingKVCache)
        #expect(defaultCache[3] is KVCacheSimple)
        #expect(CacheFamily.classify(defaultCache) == .heterogeneous)

        var params = GenerateParameters()
        params.maxKVSize = 2048
        let boundedCache = model.newCache(parameters: params)
        #expect(boundedCache.count == 4)
        for layer in boundedCache {
            #expect(layer is RotatingKVCache)
        }
        #expect(CacheFamily.classify(boundedCache) == .rotating)
    }

    @Test("TokenIterator compiled decode promotes all-rotating SWA caches")
    func tokenIteratorCompiledDecodePromotesAllRotatingCaches() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)
        guard let start = source.range(of: "mutating func setupCompiledDecode("),
              let end = source.range(
                of: "\n    }\n\n    /// Evaluate the next token",
                range: start.upperBound..<source.endIndex)
        else {
            Issue.record("Could not locate TokenIterator setupCompiledDecode helper")
            return
        }
        let helper = String(source[start.lowerBound..<end.lowerBound])

        #expect(helper.contains("cache.allSatisfy"))
        #expect(helper.contains("layer is RotatingKVCache"))
        #expect(helper.contains("CompilableRotatingKVCache(from: rotating)"))
        #expect(helper.contains("layer is KVCacheSimple"))
        #expect(helper.contains("CompilableKVCache(from: layer, maxLength: maxCacheLength)"))
    }

    private static let minimalGemma4Config = #"""
    {
      "model_type": "gemma4_text",
      "hidden_size": 64,
      "num_hidden_layers": 4,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      "global_head_dim": 16,
      "head_dim": 16,
      "intermediate_size": 128,
      "vocab_size": 256,
      "rms_norm_eps": 1e-6,
      "sliding_window": 64,
      "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
      "tie_word_embeddings": true,
      "attention_k_eq_v": false
    }
    """#
}

@Suite("Bailing/Ling hybrid cache topology focused contracts")
struct BailingLingHybridCacheTopologyFocusedTests {
    @Test("actual BailingHybridModel uses ArraysCache for linear layers and KV for global layers")
    func bailingHybridNewCacheMatchesAttentionTopology() throws {
        let config = try JSONDecoder().decode(
            BailingHybridConfiguration.self,
            from: Data(Self.minimalBailingHybridConfig.utf8))
        let model = BailingHybridModel(config)

        let defaultCache = model.newCache(parameters: nil)
        #expect(defaultCache.count == 5)
        #expect(defaultCache[0] is ArraysCache)
        #expect(defaultCache[1] is KVCacheSimple)
        #expect(defaultCache[2] is ArraysCache)
        #expect(defaultCache[3] is KVCacheSimple)
        #expect(defaultCache[4] is KVCacheSimple)
        #expect(cacheRequiresDiskBackedCoordinatorRestore(defaultCache))

        var params = GenerateParameters()
        params.maxKVSize = 2048
        let boundedCache = model.newCache(parameters: params)
        #expect(boundedCache[0] is ArraysCache)
        #expect(boundedCache[1] is RotatingKVCache)
        #expect(boundedCache[2] is ArraysCache)
        #expect(boundedCache[3] is RotatingKVCache)
        #expect(boundedCache[4] is RotatingKVCache)
        #expect(cacheRequiresDiskBackedCoordinatorRestore(boundedCache))
    }

    @Test("trailing partial Bailing/Ling layer group is global attention")
    func trailingPartialLayerGroupIsGlobalAttention() throws {
        let config = try JSONDecoder().decode(
            BailingHybridConfiguration.self,
            from: Data(Self.minimalBailingHybridConfig.utf8))

        #expect(config.isGlobalLayer(0) == false)
        #expect(config.isGlobalLayer(1) == true)
        #expect(config.isGlobalLayer(2) == false)
        #expect(config.isGlobalLayer(3) == true)
        #expect(config.isGlobalLayer(4) == true)
    }

    private static let minimalBailingHybridConfig = #"""
    {
      "model_type": "bailing_hybrid",
      "hidden_size": 32,
      "intermediate_size": 64,
      "max_position_embeddings": 4096,
      "moe_intermediate_size": 16,
      "num_experts": 4,
      "num_shared_experts": 1,
      "num_attention_heads": 4,
      "num_experts_per_tok": 2,
      "num_hidden_layers": 5,
      "num_key_value_heads": 2,
      "rms_norm_eps": 1e-6,
      "rope_theta": 1000000.0,
      "vocab_size": 256,
      "first_k_dense_replace": 1,
      "layer_group_size": 2,
      "group_norm_size": 4,
      "q_lora_rank": 8,
      "qk_rope_head_dim": 4,
      "qk_nope_head_dim": 4,
      "v_head_dim": 4,
      "kv_lora_rank": 8,
      "rope_interleave": true,
      "num_nextn_predict_layers": 1,
      "norm_topk_prob": true,
      "routed_scaling_factor": 1.0,
      "n_group": 1,
      "topk_group": 1,
      "score_function": "sigmoid",
      "moe_router_enable_expert_bias": true,
      "moe_router_enable_routed_scaling": true,
      "moe_router_enable_shared_expert": true,
      "rope_traditional": false,
      "use_bias": false,
      "use_qkv_bias": false,
      "use_qk_norm": true,
      "tie_word_embeddings": false,
      "partial_rotary_factor": 0.5,
      "head_dim": 8,
      "attention_bias": false,
      "weight_format": "mxtq",
      "mxtq_bits": {"routed_expert": 2, "attention": 8},
      "mxtq_seed": 42
    }
    """#
}

@Suite("BatchKVCache rotating-slot focused contracts", .serialized)
struct BatchKVCacheRotatingSlotFocusedTests {
    @Test("makeMask last axis matches update key count after ring wrap")
    func maskMatchesUpdatedKeyShape() {
        FocusedMLXTestSupport.withLock {
            prepareMLXMetallibForCacheTopologyTests()

            let maxSize = 16
            let prompt = 40
            let heads = 4
            let headDim = 8
            let rotating = RotatingKVCache(maxSize: maxSize, keep: 0)
            _ = rotating.update(
                keys: MLXArray.ones([1, heads, prompt, headDim]),
                values: MLXArray.ones([1, heads, prompt, headDim]))
            #expect(rotating.offset == prompt)

            let batchCache = BatchKVCache(slotCaches: [rotating])
            let mask = batchCache.makeMask(n: 1, windowSize: maxSize, returnArray: false)

            let newK = MLXArray.ones([1, heads, 1, headDim])
            let newV = MLXArray.ones([1, heads, 1, headDim])
            let (rotatingKeys, _) = batchCache.update(keys: newK, values: newV)
            #expect(rotatingKeys.shape == [1, heads, maxSize, headDim])

            guard case .array(let maskArray) = mask else {
                Issue.record("Expected .array mask, got \(mask)")
                return
            }
            #expect(maskArray.shape.last == rotatingKeys.shape[2])
            #expect(maskArray.shape == [1, 1, 1, maxSize])
            for column in 0..<maxSize {
                #expect(maskArray[0, 0, 0, column].item(Bool.self) == true)
            }
        }
    }

    @Test("pre-wrap rotating slot still gets standard causal mask")
    func preWrapMaskUnchanged() {
        FocusedMLXTestSupport.withLock {
            prepareMLXMetallibForCacheTopologyTests()

            let maxSize = 32
            let prompt = 8
            let rotating = RotatingKVCache(maxSize: maxSize, keep: 0)
            _ = rotating.update(
                keys: MLXArray.ones([1, 2, prompt, 4]),
                values: MLXArray.ones([1, 2, prompt, 4]))

            let batchCache = BatchKVCache(slotCaches: [rotating])
            let mask = batchCache.makeMask(n: 1, windowSize: maxSize, returnArray: false)

            guard case .array(let maskArray) = mask else {
                Issue.record("Expected .array mask, got \(mask)")
                return
            }
            #expect(maskArray.shape == [1, 1, 1, prompt + 1])
            for column in 0..<(prompt + 1) {
                #expect(maskArray[0, 0, 0, column].item(Bool.self) == true)
            }
        }
    }

    @Test("mixed wrapped rotating and unbounded slot produces compatible mask")
    func mixedWrappedAndUnboundedMask() {
        FocusedMLXTestSupport.withLock {
            prepareMLXMetallibForCacheTopologyTests()

            let maxSize = 16
            let rotating = RotatingKVCache(maxSize: maxSize, keep: 0)
            _ = rotating.update(
                keys: MLXArray.ones([1, 2, 40, 4]),
                values: MLXArray.ones([1, 2, 40, 4]))

            let simple = KVCacheSimple()
            _ = simple.update(
                keys: MLXArray.ones([1, 2, 5, 4]),
                values: MLXArray.ones([1, 2, 5, 4]))

            let batch = BatchKVCache(slotCaches: [rotating, simple])
            let mask = batch.makeMask(n: 1, windowSize: maxSize, returnArray: false)

            guard case .array(let maskArray) = mask else {
                Issue.record("Expected .array mask, got \(mask)")
                return
            }
            #expect(maskArray.shape == [2, 1, 1, maxSize])

            for column in 0..<maxSize {
                #expect(maskArray[0, 0, 0, column].item(Bool.self) == true)
            }
            for column in 0..<6 {
                #expect(maskArray[1, 0, 0, column].item(Bool.self) == true)
            }
            for column in 6..<maxSize {
                #expect(maskArray[1, 0, 0, column].item(Bool.self) == false)
            }
        }
    }

    @Test("explicit effectiveKeyLens caps batch causal mask maxTotal")
    func explicitEffectiveKeyLensCapsMaxTotal() {
        FocusedMLXTestSupport.withLock {
            prepareMLXMetallibForCacheTopologyTests()

            let mask = createBatchCausalMask(
                queryLen: 1,
                offsets: [100, 5],
                effectiveKeyLens: [16, 6],
                windowSize: nil)

            #expect(mask.shape == [2, 1, 1, 16])
            for column in 0..<16 {
                #expect(mask[0, 0, 0, column].item(Bool.self) == true)
            }
            for column in 0..<6 {
                #expect(mask[1, 0, 0, column].item(Bool.self) == true)
            }
            for column in 6..<16 {
                #expect(mask[1, 0, 0, column].item(Bool.self) == false)
            }
        }
    }
}

private let cacheTopologyTestRepoRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .standardizedFileURL

private let cacheTopologyMetallibSourceDirectory: URL? = {
    let sourceDirectories = [
        cacheTopologyTestRepoRoot.appendingPathComponent(".build/arm64-apple-macosx/debug"),
        cacheTopologyTestRepoRoot.appendingPathComponent(".build/debug"),
    ]
    return sourceDirectories.first {
        FileManager.default.fileExists(atPath: $0.appendingPathComponent("default.metallib").path)
    }
}()

private final class CacheTopologyTestBundleProbe {}

private let cacheTopologyMetallibPrepared: Void = {
    guard let sourceDirectory = cacheTopologyMetallibSourceDirectory else { return }
    let fileManager = FileManager.default
    let source = sourceDirectory.appendingPathComponent("default.metallib")

    var targetDirectories: [URL] = []
    if let executableURL = Bundle.main.executableURL {
        targetDirectories.append(executableURL.deletingLastPathComponent())
    }
    if let resourceURL = Bundle.main.resourceURL {
        targetDirectories.append(resourceURL)
    }
    let testBundle = Bundle(for: CacheTopologyTestBundleProbe.self)
    if let executableURL = testBundle.executableURL {
        targetDirectories.append(executableURL.deletingLastPathComponent())
    }
    if let resourceURL = testBundle.resourceURL {
        targetDirectories.append(resourceURL)
    }
    if let firstArgument = CommandLine.arguments.first, !firstArgument.isEmpty {
        targetDirectories.append(URL(fileURLWithPath: firstArgument).deletingLastPathComponent())
    }

    var scanned = Set<String>()
    for candidate in targetDirectories {
        var directory = candidate.standardizedFileURL
        for _ in 0..<4 {
            if scanned.insert(directory.path).inserted {
                try? fileManager.copyCacheTopologyMetallibsIfMissing(from: source, into: directory)
            }
            directory.deleteLastPathComponent()
        }
    }
}()

private func prepareMLXMetallibForCacheTopologyTests() {
    _ = cacheTopologyMetallibPrepared
}

private extension FileManager {
    func copyCacheTopologyMetallibsIfMissing(from source: URL, into directory: URL) throws {
        try createDirectory(at: directory, withIntermediateDirectories: true)
        for name in ["default.metallib", "mlx.metallib"] {
            let destination = directory.appendingPathComponent(name)
            if !fileExists(atPath: destination.path) {
                try copyItem(at: source, to: destination)
            }
        }
    }
}
