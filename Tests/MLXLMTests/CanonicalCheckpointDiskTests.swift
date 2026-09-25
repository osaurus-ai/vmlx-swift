import Foundation
import MLX
import SQLite3
import Testing

@testable import MLXLMCommon

@Suite(.serialized) struct CanonicalCheckpointDiskTests {
    @Test func rejectedCheckpointIsRewrittenWithoutRemovingOrdinaryHit() throws {
        try MLXMetalTestLock.withLock {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let disk = DiskCache(cacheDir: root, maxSizeBytes: 1_000_000, modelKey: "model")
            let tokens = Array(0 ..< 32)
            let contract = try #require(CanonicalPrefillCheckpoint(chunkSize: 16))
            let cache: [KVCache] = [KVCacheSimple()]
            let keys = MLXArray.zeros([1, 1, 32, 1], dtype: .bfloat16)
            _ = cache[0].update(keys: keys, values: keys)
            let good = TQDiskSerializer.serialize(
                cache: cache, preserveStandardKVStorageDType: true)
            var bad = good
            bad[TQDiskSerializer.preserveStandardKVStorageDTypeKey] = MLXArray([Int32(0)])
            disk.storeCanonicalCheckpoint(
                tokens: tokens, arrays: bad, contract: contract,
                requestSalt: nil, chainId: "chat", enforceQuota: true)
            disk.store(tokens: tokens, arrays: good)
            _ = try #require(disk.fetch(tokens: tokens))
            let hitsBefore = disk.snapshotStats().hits
            _ = try #require(
                disk.fetchCanonicalCheckpoint(
                    targetTokens: tokens + [32], contract: contract, requestSalt: nil))
            #expect(
                disk.markRestoreRejected(
                    tokens: tokens, mediaSalt: contract.storageSalt(requestSalt: nil),
                    countedHit: false))
            #expect(disk.snapshotStats().hits == hitsBefore)
            disk.storeCanonicalCheckpoint(
                tokens: tokens, arrays: good, contract: contract,
                requestSalt: nil, chainId: "chat", enforceQuota: true)
            let repaired = try #require(
                disk.fetchCanonicalCheckpoint(
                    targetTokens: tokens + [32], contract: contract, requestSalt: nil))
            #expect(
                repaired.arrays[TQDiskSerializer.preserveStandardKVStorageDTypeKey]?.item(
                    Int32.self) == 1)
        }
    }

    @Test func checkpointDoesNotResolveLostResumeStateWarning() throws {
        try MLXMetalTestLock.withLock {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let disk = DiskCache(cacheDir: root, maxSizeBytes: 1_000_000, modelKey: "model")
            let tokens = Array(0 ..< 32)
            let contract = try #require(CanonicalPrefillCheckpoint(chunkSize: 16))
            disk.storeCanonicalCheckpoint(
                tokens: tokens, arrays: ["data": MLXArray([Int32(22)])],
                contract: contract, requestSalt: nil,
                chainId: "chat", enforceQuota: true)
            disk.recordQuotaPass(
                evictedGroups: 1, evictedBytes: 100, milliseconds: 0,
                event: DiskCachePressureEvent(
                    kind: .activeTipDropped, chainId: "chat",
                    tipBytes: 2_000_000, capBytes: 1_000_000),
                tipTokenCount: 16)
            disk.reconcileCapacityPressure(chainId: "chat", requiresCompanion: false)
            #expect(disk.snapshotStats().capacityPressureByChain["chat"] != nil)
            disk.store(
                tokens: tokens, arrays: ["data": MLXArray([Int32(11)])], enforceQuota: true,
                chainId: "chat", isResumeBoundary: true)
            disk.reconcileCapacityPressure(chainId: "chat", requiresCompanion: false)
            #expect(disk.snapshotStats().capacityPressureByChain["chat"] == nil)
        }
    }

    @Test func malformedChunkMetadataDoesNotCreateAReusableCheckpoint() throws {
        try MLXMetalTestLock.withLock {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let disk = DiskCache(cacheDir: root, maxSizeBytes: 1_000_000, modelKey: "model")
            let tokens = Array(0 ..< 32)
            let contract = try #require(CanonicalPrefillCheckpoint(chunkSize: 16))
            disk.storeCanonicalCheckpoint(
                tokens: tokens, arrays: ["data": MLXArray([Int32(22)])],
                contract: contract, requestSalt: nil,
                chainId: "chat", enforceQuota: true)
            var db: OpaquePointer?
            try #require(
                sqlite3_open(root.appendingPathComponent("cache_index.db").path, &db) == SQLITE_OK)
            defer { sqlite3_close(db) }
            for value in ["NULL", "0", "-16", "'bad'", "1e30", "9223372036854775807", "3"] {
                try #require(
                    sqlite3_exec(
                        db, "UPDATE cache_entries SET replay_chunk_size = \(value)", nil, nil, nil)
                        == SQLITE_OK)
                #expect(
                    disk.fetchCanonicalCheckpoint(
                        targetTokens: tokens + [32], contract: contract, requestSalt: nil) == nil)
                #expect(disk.quotaEntries().allSatisfy { !$0.isCanonicalCheckpoint })
            }
            // An invalid marker never deletes an otherwise readable payload.
            #expect(
                disk.fetch(tokens: tokens, mediaSalt: contract.storageSalt(requestSalt: nil)) != nil
            )
        }
    }

    @Test func smallCapSkipsCheckpointWriteAndKeepsResumePayload() throws {
        try MLXMetalTestLock.withLock {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let coordinator = CacheCoordinator(
                config: CacheCoordinatorConfig(
                    usePagedCache: false, enableDiskCache: true,
                    diskCacheMaxGB: Float(65_536.0 / 1_073_741_824.0),
                    diskCacheDir: root, modelKey: "small-cap"))
            let disk = try #require(coordinator.diskCache)
            let tokens = Array(0 ..< 32)
            coordinator.storePersistentBoundary(
                tokens: tokens, diskArrays: ["data": MLXArray([Int32(11)])],
                ssmStates: nil, chainId: "chat", isResumeBoundary: true)
            let before = disk.quotaEntries()
            let cache: [KVCache] = [KVCacheSimple()]
            let keys = MLXArray.zeros([1, 1, 32, 1], dtype: .bfloat16)
            _ = cache[0].update(keys: keys, values: keys)
            MLX.eval(cache)
            coordinator.storeCanonicalCheckpoint(
                tokens: tokens, cache: cache, chunkSize: 16,
                requestSalt: nil, chainId: "chat")
            let after = disk.quotaEntries()
            #expect(before.count == 1 && after.count == 1)
            #expect(before.map(\.hash) == after.map(\.hash))
            #expect(after.allSatisfy { !$0.isCanonicalCheckpoint })
            #expect(disk.fetch(tokens: tokens)?["data"]?.item(Int32.self) == 11)
            #expect(disk.snapshotStats().capacityPressureByChain.isEmpty)
        }
    }

    @Test func canonicalAndOrdinaryPayloadsNeverAliasAcrossReopen() throws {
        try MLXMetalTestLock.withLock {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let tokens = Array(0 ..< 32)
            let contract = try #require(CanonicalPrefillCheckpoint(chunkSize: 16))
            do {
                let disk = DiskCache(cacheDir: root, maxSizeBytes: 1_000_000, modelKey: "model")
                disk.store(
                    tokens: tokens, arrays: ["data": MLXArray([Int32(11)])], mediaSalt: "request")
                disk.storeCanonicalCheckpoint(
                    tokens: tokens, arrays: ["data": MLXArray([Int32(22)])],
                    contract: contract, requestSalt: "request",
                    chainId: "chat", enforceQuota: true)
                #expect(disk.quotaEntries().filter(\.isCanonicalCheckpoint).count == 1)
                #expect(disk.quotaEntries().count == 2)
            }
            let disk = DiskCache(cacheDir: root, maxSizeBytes: 1_000_000, modelKey: "model")
            let ordinary = try #require(disk.fetch(tokens: tokens, mediaSalt: "request"))
            #expect(ordinary["data"]?.item(Int32.self) == 11)
            let hit = try #require(
                disk.fetchCanonicalCheckpoint(
                    targetTokens: tokens + [32, 33], contract: contract, requestSalt: "request"))
            #expect(hit.tokens == tokens)
            #expect(hit.arrays["data"]?.item(Int32.self) == 22)
            #expect(
                disk.fetchCanonicalCheckpoint(
                    targetTokens: tokens + [32], contract: contract, requestSalt: "other") == nil)
            #expect(
                disk.fetchCanonicalCheckpoint(
                    targetTokens: tokens + [32],
                    contract: try #require(CanonicalPrefillCheckpoint(chunkSize: 8)),
                    requestSalt: "request") == nil)
            let other = DiskCache(cacheDir: root, maxSizeBytes: 1_000_000, modelKey: "other-model")
            #expect(
                other.fetchCanonicalCheckpoint(
                    targetTokens: tokens + [32], contract: contract, requestSalt: "request") == nil)
        }
    }

    @Test func coordinatorPreservesDTypeAndClassifiesQuotaRows() throws {
        try MLXMetalTestLock.withLock {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let coordinator = CacheCoordinator(
                config: CacheCoordinatorConfig(
                    usePagedCache: false, enableDiskCache: true, diskCacheMaxGB: 0.01,
                    diskCacheDir: root, modelKey: "canonical-test"))
            let cache: [KVCache] = [
                KVCacheSimple(), RotatingKVCache(maxSize: 16, keep: 0, step: 16),
            ]
            let keys = MLXArray(Array(0 ..< 32).map(Float.init)).asType(.bfloat16).reshaped(
                1, 1, 32, 1)
            for layer in cache { _ = layer.update(keys: keys, values: keys) }
            MLX.eval(cache)
            let tokens = Array(0 ..< 32)
            coordinator.storeCanonicalCheckpoint(
                tokens: tokens, cache: cache, chunkSize: 16,
                requestSalt: "request", chainId: "chat")
            let disk = try #require(coordinator.diskCache)
            let contract = try #require(CanonicalPrefillCheckpoint(chunkSize: 16))
            let hit = try #require(
                disk.fetchCanonicalCheckpoint(
                    targetTokens: tokens + [32],
                    contract: contract, requestSalt: "request"))
            var restored: [KVCache] = [
                KVCacheSimple(), RotatingKVCache(maxSize: 16, keep: 0, step: 16),
            ]
            #expect(
                restoreFromDiskArrays(hit.arrays, into: &restored, requirePromptBoundary: true)
                    == 32)
            for (a, b) in zip(cache, restored) {
                #expect(a.metaState == b.metaState)
                for (x, y) in zip(a.state, b.state) {
                    #expect(x.dtype == y.dtype)
                    #expect(x.shape == y.shape)
                    #expect(
                        x.asType(.float32).asArray(Float.self)
                            == y.asType(.float32).asArray(Float.self))
                }
            }
            #expect(disk.quotaEntries().filter(\.isCanonicalCheckpoint).count == 1)
        }
    }
}
