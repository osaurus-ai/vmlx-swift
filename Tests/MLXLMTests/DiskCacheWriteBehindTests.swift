import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Write-behind disk stores: a store publishes at once, its file reaches disk on a background writer
/// without MLX calls, and a lookup of an entry still being written is served from memory.
@Suite(.serialized)
struct DiskCacheWriteBehindTests {
    private func root(_ name: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("write-behind-\(name)-\(UUID().uuidString)")
    }

    @Test func byteWriterOutputLoadsThroughMLX() throws {
        try MLXMetalTestLock.withLock {
            let dir = root("bytes")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: dir) }
            let arrays: [String: MLXArray] = [
                "keys": (MLXArray(0 ..< 24).reshaped(2, 3, 4).asType(.float32) / 7).asType(.bfloat16),
                "values": MLXArray(0 ..< 5).asType(.float32) * 0.5,
                "offsets": MLXArray([Int32(3), 1, 4, 1, 5, 9]).reshaped(3, 2),
                "flags": MLXArray([UInt8(1), 0, 1]),
                "empty": MLXArray.zeros([1, 0, 3], dtype: .float16),
            ]
            eval(Array(arrays.values))
            let tensors = try #require(SafetensorsByteWriter.tensors(from: arrays))
            let url = dir.appendingPathComponent("payload.safetensors")
            let size = try SafetensorsByteWriter.write(tensors, metadata: ["format": "mlx"], to: url)
            #expect(size == (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int))
            #expect(DiskCache.isCompleteSafetensors(url: url))
            let loaded = try loadArrays(url: url)
            #expect(Set(loaded.keys) == Set(arrays.keys))
            for (name, original) in arrays {
                let back = try #require(loaded[name])
                #expect(back.dtype == original.dtype, "\(name)")
                #expect(back.shape == original.shape, "\(name)")
                if original.size > 0 {
                    #expect(
                        allClose(back.asType(.float32), original.asType(.float32)).item(Bool.self),
                        "\(name)")
                }
            }
        }
    }

    @Test func inFlightEntryIsServedFromMemoryThenPersisted() throws {
        try MLXMetalTestLock.withLock {
            let dir = root("inflight")
            defer { try? FileManager.default.removeItem(at: dir) }
            let cache = DiskCache(cacheDir: dir, maxSizeBytes: 50_000_000, modelKey: "wb")
            cache.writeBehind = true
            cache.writeDelayForTesting = 0.4
            let tokens = Array(100 ..< 137)
            let keys = MLXArray(0 ..< 4_096).asType(.float32) / 3
            cache.store(tokens: tokens, arrays: ["keys": keys])
            #expect(cache.pendingWriteCount == 1)
            // In flight counts as durable: it must not be produced (rederived) again.
            #expect(cache.hasDurableEntry(tokens: tokens))
            #expect(cache.candidateTokenCounts(maxTokens: 1_000).contains(tokens.count))
            let served = try #require(cache.fetch(tokens: tokens))
            #expect(allClose(try #require(served["keys"]), keys).item(Bool.self))
            #expect(cache.inflightFetches == 1)
            #expect(cache.waitForPendingWrites(timeout: 10))
            #expect(cache.pendingWriteCount == 0)
            #expect(cache.hasDurableEntry(tokens: tokens))
            // A new process (instance) reads the persisted entry.
            let restarted = DiskCache(cacheDir: dir, maxSizeBytes: 50_000_000, modelKey: "wb")
            let fromDisk = try #require(restarted.fetch(tokens: tokens))
            #expect(allClose(try #require(fromDisk["keys"]), keys).item(Bool.self))
        }
    }

    @Test func secondStoreOfAnInFlightPayloadIsSkipped() throws {
        try MLXMetalTestLock.withLock {
            let dir = root("dup")
            defer { try? FileManager.default.removeItem(at: dir) }
            let cache = DiskCache(cacheDir: dir, maxSizeBytes: 50_000_000, modelKey: "wb")
            cache.writeBehind = true
            cache.writeDelayForTesting = 0.4
            let tokens = Array(1 ..< 20)
            let payload = ["keys": MLXArray.ones([2_048], dtype: .float32)]
            cache.store(tokens: tokens, arrays: payload)
            cache.store(tokens: tokens, arrays: payload)
            #expect(cache.writeBehindStores == 1)
            #expect(cache.snapshotStats().storeSkips == 1)
            #expect(cache.waitForPendingWrites(timeout: 10))
            #expect(cache.hasDurableEntry(tokens: tokens))
        }
    }

    @Test func clearWaitsForPendingWrites() throws {
        try MLXMetalTestLock.withLock {
            let dir = root("clear")
            defer { try? FileManager.default.removeItem(at: dir) }
            let cache = DiskCache(cacheDir: dir, maxSizeBytes: 50_000_000, modelKey: "wb")
            cache.writeBehind = true
            cache.writeDelayForTesting = 0.3
            cache.store(tokens: Array(5 ..< 40), arrays: ["keys": MLXArray.ones([4_096])])
            cache.clear()
            #expect(cache.pendingWriteCount == 0)
            #expect(cache.snapshotStats().currentEntryCount == 0)
            let payloads = (try FileManager.default.contentsOfDirectory(atPath: dir.path))
                .filter { $0.hasSuffix(".safetensors") }
            #expect(payloads.isEmpty, "\(payloads)")
        }
    }

    /// The serving configuration opts the coordinator in: a persistent boundary store returns
    /// before its file exists, the entry already counts as durable (it must not be produced again),
    /// and quota enforcement still happens, on the writer, once each row has landed.
    @Test func coordinatorWritesBehindAndStillEnforcesTheQuota() throws {
        try MLXMetalTestLock.withLock {
            let dir = root("coordinator")
            defer { try? FileManager.default.removeItem(at: dir) }
            // Room for one 32 KB payload, not two.
            var config = CacheCoordinatorConfig(
                usePagedCache: false, enableDiskCache: true,
                diskCacheMaxGB: Float(50_000) / 1_073_741_824, diskCacheDir: dir, modelKey: "wb")
            config.diskWriteBehind = true
            let coordinator = CacheCoordinator(config: config)
            let disk = try #require(coordinator.diskCache)
            disk.writeDelayForTesting = 0.2
            let first = Array(200 ..< 260)
            coordinator.storePersistentBoundary(
                tokens: first, diskArrays: ["keys": MLXArray.ones([8_192], dtype: .float32)],
                ssmStates: nil, chainId: "A", isResumeBoundary: true)
            #expect(disk.pendingWriteCount == 1)
            #expect(coordinator.hasDurableDiskEntry(tokens: first))
            #expect(coordinator.waitForPendingDiskWrites())
            #expect(disk.snapshotStats().currentEntryCount == 1)
            Thread.sleep(forTimeInterval: 1.1)  // distinct recency
            let second = Array(300 ..< 370)
            coordinator.storePersistentBoundary(
                tokens: second, diskArrays: ["keys": MLXArray.ones([8_192], dtype: .float32) * 2],
                ssmStates: nil, chainId: "B", isResumeBoundary: true)
            #expect(coordinator.waitForPendingDiskWrites())
            #expect(disk.snapshotStats().currentEntryCount == 1)
            #expect(disk.snapshotStats().evictions > 0)
            #expect(coordinator.hasDurableDiskEntry(tokens: second))
            #expect(!coordinator.hasDurableDiskEntry(tokens: first))
        }
    }

    /// Without the serving opt-in a coordinator keeps synchronous stores.
    @Test func coordinatorStoresSynchronouslyByDefault() throws {
        try MLXMetalTestLock.withLock {
            let dir = root("coordinator-sync")
            defer { try? FileManager.default.removeItem(at: dir) }
            let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
                usePagedCache: false, enableDiskCache: true, diskCacheMaxGB: 1, diskCacheDir: dir,
                modelKey: "wb"))
            coordinator.storePersistentBoundary(
                tokens: Array(1 ..< 30), diskArrays: ["keys": MLXArray.ones([1_024])],
                ssmStates: nil, chainId: "A", isResumeBoundary: true)
            let disk = try #require(coordinator.diskCache)
            #expect(disk.pendingWriteCount == 0)
            #expect(disk.snapshotStats().currentEntryCount == 1)
        }
    }

    @Test func servingConfigurationTurnsWriteBehindOn() {
        #expect(VMLXServerRuntimeSettings().cacheCoordinatorConfig(modelKey: "m").diskWriteBehind)
    }

    /// The one-sum screen catches every non-finite element and never refuses a finite payload, even
    /// one whose half-precision sum overflows (the exact recount clears it).
    @Test func nonFiniteScreenIsExact() throws {
        try MLXMetalTestLock.withLock {
            func names(_ values: [Float], _ dtype: DType = .float32) -> [String] {
                let a = MLXArray(values).asType(dtype)
                eval(a)
                return DiskCache.nonFiniteTensorNames(in: ["t": a])
            }
            #expect(names([1, 2, 3]).isEmpty)
            #expect(names([1, .nan, 3]) == ["t(1)"])
            #expect(names([1, .infinity, 3]) == ["t(1)"])
            #expect(names([-.infinity, 2, 3]) == ["t(1)"])
            #expect(names([.infinity, -.infinity, 3]) == ["t(2)"])
            #expect(names([1, .nan, 3], .bfloat16) == ["t(1)"])
            // 4 x 60000 is finite in fp16, but its fp16 sum overflows to inf.
            #expect(names([60_000, 60_000, 60_000, 60_000], .float16).isEmpty)
            #expect(DiskCache.nonFiniteTensorNames(in: [
                "a": MLXArray([Float(1), 2]), "b": MLXArray([Float.nan, 1]), "c": MLXArray([Float(3)]),
            ]) == ["b(1)"])
        }
    }
}

