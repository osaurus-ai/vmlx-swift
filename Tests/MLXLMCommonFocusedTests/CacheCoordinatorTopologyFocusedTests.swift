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

    @Test("ZAYA CCA format-v2 disk payload is enough for hybrid cache hit")
    func zayaCCADiskHitDoesNotRequireSeparateSSMCompanion() {
        FocusedMLXTestSupport.withLock {
        let tmp = makeTempDir("zaya-cca-disk")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let coordinator = makeCoordinator(
            usePagedCache: true,
            enableDiskCache: true,
            diskCacheDir: tmp,
            modelKey: "zaya-cca-focused")
        coordinator.setHybrid(true)
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
            Issue.record("ZAYA CCA v2 disk payload already carries path-dependent state")
        }
        }
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
        coordinator.setHybrid(true)
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

    @Test("path-dependent and sliding caches require disk-backed coordinator restore")
    func pathDependentAndSlidingCachesRequireDiskBackedRestore() {
        FocusedMLXTestSupport.withLock {
        prepareMLXMetallibForCacheTopologyTests()

        #expect(cacheRequiresDiskBackedCoordinatorRestore([MambaCache(), KVCacheSimple()]))
        #expect(cacheRequiresDiskBackedCoordinatorRestore([ArraysCache(size: 2), KVCacheSimple()]))
        #expect(cacheRequiresDiskBackedCoordinatorRestore([ZayaCCACache(), KVCacheSimple()]))
        #expect(cacheRequiresDiskBackedCoordinatorRestore([RotatingKVCache(maxSize: 32), KVCacheSimple()]))
        #expect(cacheRequiresDiskBackedCoordinatorRestore([DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)]))
        #expect(!cacheRequiresDiskBackedCoordinatorRestore([KVCacheSimple(), KVCacheSimple()]))
        }
    }

    private func makeCoordinator(
        usePagedCache: Bool,
        enableDiskCache: Bool,
        diskCacheDir: URL? = nil,
        modelKey: String = "cache-topology-focused",
        blockSize: Int = 4
    ) -> CacheCoordinator {
        prepareMLXMetallibForCacheTopologyTests()

        return CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: usePagedCache,
            enableDiskCache: enableDiskCache,
            pagedBlockSize: blockSize,
            maxCacheBlocks: 40,
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
