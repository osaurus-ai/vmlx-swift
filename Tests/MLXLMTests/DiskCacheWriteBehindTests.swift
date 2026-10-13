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

    /// A `..<offset` slice of a step-padded KV buffer (what a live or trimmed cache's `state` returns)
    /// is written run by run from the buffer itself, no host copy, and the file is byte-for-byte the
    /// one a contiguous copy produces. Offsets deliberately not multiples of the 256-token step.
    @Test func stridedKVSliceWritesTheSameBytesAsAContiguousCopy() throws {
        try MLXMetalTestLock.withLock {
            let dir = root("strided")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: dir) }
            for (capacity, offset) in [(512, 300), (1024, 777), (768, 257)] {
                let buffer = (MLXRandom.normal([1, 4, capacity, 128]) * 3).asType(.bfloat16)
                let values = (MLXRandom.normal([1, 4, capacity, 128])).asType(.bfloat16)
                let keys = buffer[.ellipsis, ..<offset, 0...]
                let vals = values[.ellipsis, ..<offset, 0...]
                eval(keys, vals)
                let arrays = ["k": keys, "v": vals, "meta": MLXArray([Int32(offset)])]
                let tensors = try #require(SafetensorsByteWriter.tensors(from: arrays))
                let strided = tensors.filter {
                    if case .strided = $0.payload { return true }
                    return false
                }
                #expect(strided.map(\.name).sorted() == ["k", "v"], "capacity \(capacity) offset \(offset)")
                let viewURL = dir.appendingPathComponent("view-\(offset).safetensors")
                try SafetensorsByteWriter.write(tensors, metadata: ["format": "mlx"], to: viewURL)

                let owned = ["k": keys * 1, "v": vals * 1, "meta": MLXArray([Int32(offset)])]
                eval(Array(owned.values))
                let ownedTensors = try #require(SafetensorsByteWriter.tensors(from: owned))
                #expect(ownedTensors.allSatisfy {
                    if case .contiguous = $0.payload { return true }
                    return false
                })
                let copyURL = dir.appendingPathComponent("copy-\(offset).safetensors")
                try SafetensorsByteWriter.write(ownedTensors, metadata: ["format": "mlx"], to: copyURL)

                #expect(try Data(contentsOf: viewURL) == Data(contentsOf: copyURL), "offset \(offset)")
                let loaded = try loadArrays(url: viewURL)
                #expect(arrayEqual(try #require(loaded["k"]), keys).item(Bool.self))
                #expect(arrayEqual(try #require(loaded["v"]), vals).item(Bool.self))
            }
        }
    }

    /// The gathered (`writev`) writer produces exactly the per-tensor writer's bytes for tiny inline
    /// tensors, large contiguous ones and strided KV views, including when a call must stop mid-segment
    /// (forced 4 KB / 7,777-byte call limits so every path of the resume loop runs).
    @Test func gatheredWritesMatchPerTensorWritesByteForByte() throws {
        try MLXMetalTestLock.withLock {
            let dir = root("gathered")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: dir) }
            let saved = SafetensorsByteWriter.maxGatheredCallBytes
            defer { SafetensorsByteWriter.maxGatheredCallBytes = saved }
            let buffer = (MLXRandom.normal([1, 4, 640, 128]) * 2).asType(.bfloat16)
            let arrays: [String: MLXArray] = [
                "kv_0_keys": buffer[.ellipsis, ..<333, 0...],
                "kv_0_values": MLXRandom.normal([1, 4, 333, 128]).asType(.bfloat16),
                "__layer_kind_0__": MLXArray([Int32(1)]),
                "__offset__": MLXArray([Int32(333)]),
                "flags": MLXArray([UInt8(1), 0, 1]),
                "empty": MLXArray.zeros([1, 0, 3], dtype: .float16),
            ]
            eval(Array(arrays.values))
            let tensors = try #require(SafetensorsByteWriter.tensors(from: arrays))
            let reference = dir.appendingPathComponent("per-tensor.safetensors")
            try SafetensorsByteWriter.writePerTensor(tensors, metadata: ["format": "mlx"], to: reference)
            let expected = try Data(contentsOf: reference)
            for limit in [1 << 30, 4096, 7_777] {
                SafetensorsByteWriter.maxGatheredCallBytes = limit
                let url = dir.appendingPathComponent("gathered-\(limit).safetensors")
                let size = try SafetensorsByteWriter.writeGathered(tensors, metadata: ["format": "mlx"], to: url)
                #expect(size == expected.count, "limit \(limit)")
                #expect(try Data(contentsOf: url) == expected, "limit \(limit)")
            }
            let loaded = try loadArrays(url: dir.appendingPathComponent("gathered-4096.safetensors"))
            #expect(arrayEqual(try #require(loaded["kv_0_keys"]), arrays["kv_0_keys"]!).item(Bool.self))
        }
    }

    /// The Swift counterpart of the Python bf16 store fix ("bf16 KV converted through float32 on every
    /// save"): a bf16 KV cache stored through the real coordinator path lands on disk as BF16 with the
    /// source's exact bit patterns (-0.0, subnormal, 1e30, 3e38, max), never F32/F16, at a
    /// partial last step (333 tokens, step 256), and restores bit-identically.
    @Test func bf16KVStoresBitExactWithoutFloat32() throws {
        try MLXMetalTestLock.withLock {
            let dir = root("bf16-bits")
            defer { try? FileManager.default.removeItem(at: dir) }
            var config = CacheCoordinatorConfig(
                usePagedCache: false, enableDiskCache: true, diskCacheMaxGB: 1, diskCacheDir: dir,
                modelKey: "bf16")
            config.diskWriteBehind = true
            let coordinator = CacheCoordinator(config: config)
            let tokens = Array(1000 ..< 1333)
            let specials: [UInt16] = [0x8000, 0x0001, 0x7149, 0x7F61, 0x7F7F, 0x3F80, 0xBF80, 0x0000]
            var caches: [KVCacheSimple] = []
            var sources: [(MLXArray, MLXArray)] = []
            for layer in 0 ..< 2 {
                let count = 1 * 4 * tokens.count * 64
                var bits = (0 ..< count).map { UInt16(truncatingIfNeeded: ($0 &* 2_654_435_761 &+ layer) >> 7) & 0x7F7F }
                for (i, v) in specials.enumerated() { bits[i * 97] = v }
                let keys = MLXArray(bits).reshaped(1, 4, tokens.count, 64).view(dtype: .bfloat16)
                let values = MLXArray(bits.reversed()).reshaped(1, 4, tokens.count, 64).view(dtype: .bfloat16)
                let cache = KVCacheSimple()
                _ = cache.update(keys: keys, values: values)
                eval(cache.state)
                caches.append(cache)
                sources.append((keys, values))
            }
            #expect(caches[0].offset == tokens.count)
            coordinator.storeAfterGeneration(
                promptTokens: tokens, perLayerData: [], ssmStates: nil, cache: caches)
            #expect(coordinator.waitForPendingDiskWrites())
            let files = (FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil)?
                .compactMap { $0 as? URL } ?? []).filter { $0.pathExtension == "safetensors" }
            let file = try #require(files.first)
            let raw = try Data(contentsOf: file)
            let headerLength = Int(raw.prefix(8).withUnsafeBytes { $0.load(as: UInt64.self) })
            let header = try #require(
                try JSONSerialization.jsonObject(with: raw.subdata(in: 8 ..< 8 + headerLength)) as? [String: Any])
            let dataStart = 8 + headerLength
            for (name, entry) in header where name != "__metadata__" {
                let info = try #require(entry as? [String: Any])
                let dtype = try #require(info["dtype"] as? String)
                #expect(dtype != "F32" && dtype != "F16", "\(name) is \(dtype)")
            }
            for layer in 0 ..< 2 {
                for (suffix, source) in [("keys", sources[layer].0), ("values", sources[layer].1)] {
                    let info = try #require(header["kv_\(layer)_\(suffix)"] as? [String: Any])
                    #expect(info["dtype"] as? String == "BF16")
                    let offsets = try #require(info["data_offsets"] as? [Int])
                    let onDisk = raw.subdata(in: dataStart + offsets[0] ..< dataStart + offsets[1])
                    let expected = source.view(dtype: .uint16).asData(access: .copy).data
                    #expect(onDisk == expected, "layer \(layer) \(suffix) bytes")
                }
            }
            let restored = try #require(coordinator.diskCache?.fetch(tokens: tokens))
            for layer in 0 ..< 2 {
                let keys = try #require(restored["kv_\(layer)_keys"])
                #expect(keys.dtype == .bfloat16)
                #expect(arrayEqual(keys.view(dtype: .uint16), sources[layer].0.view(dtype: .uint16)).item(Bool.self))
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

    @Test func clearDoesNotRepublishAfterSlowPendingWrite() throws {
        try MLXMetalTestLock.withLock {
            let dir = root("slow-clear")
            defer { try? FileManager.default.removeItem(at: dir) }
            let cache = DiskCache(cacheDir: dir, maxSizeBytes: 50_000_000, modelKey: "wb")
            cache.writeBehind = true
            cache.writeDelayForTesting = 31
            let tokens = Array(5 ..< 40)
            cache.store(tokens: tokens, arrays: ["keys": MLXArray.ones([4_096])])
            cache.clear()
            #expect(cache.waitForPendingWrites(timeout: 10))
            #expect(cache.snapshotStats().currentEntryCount == 0)
            #expect(!cache.hasDurableEntry(tokens: tokens))
        }
    }

    @Test func externalPurgeInvalidatesOnlyItsRootAndAllowsReplacement() throws {
        try MLXMetalTestLock.withLock {
            let dir = root("external-clear")
            let otherDir = root("external-clear-other")
            defer {
                try? FileManager.default.removeItem(at: dir)
                try? FileManager.default.removeItem(at: otherDir)
            }
            let cache = DiskCache(cacheDir: dir, maxSizeBytes: 50_000_000, modelKey: "wb")
            let other = DiskCache(cacheDir: otherDir, maxSizeBytes: 50_000_000, modelKey: "wb")
            for item in [cache, other] {
                item.writeBehind = true
                item.writeDelayForTesting = 0.3
            }
            let tokens = Array(5 ..< 40)
            let original = ["keys": MLXArray.ones([4_096])]
            cache.store(tokens: tokens, arrays: original)
            other.store(tokens: tokens, arrays: original)
            MLXCacheIOLock.withSerializedMLXCacheIO {
                DiskCache.invalidatePendingWrites(in: dir)
            }
            #expect(!cache.hasDurableEntry(tokens: tokens))
            #expect(other.hasDurableEntry(tokens: tokens))
            // Re-store the same hash before the invalidated writer finishes. Its completion must
            // not remove the replacement's buffers, publish stale bytes or corrupt accounting.
            cache.store(tokens: tokens, arrays: ["keys": MLXArray.ones([4_096]) * 2])
            #expect(cache.pendingWriteCount == 2)
            #expect(cache.waitForPendingWrites(timeout: 10))
            #expect(other.waitForPendingWrites(timeout: 10))
            #expect(cache.pendingWriteCount == 0)
            #expect(cache.hasWriteBehindRoom)
            let restored = try #require(cache.fetch(tokens: tokens)?["keys"])
            #expect(restored[0].item(Float.self) == 2)
            #expect(other.hasDurableEntry(tokens: tokens))
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

    /// Regression (live hang, Diplodocus K2 in the osaurus app, 2026-10-11): with the in-flight budget
    /// full and the queue over it by more than the next write, a store waited for the writer while
    /// holding `CombinedDiskCacheQuotaLock`, and the writer's `afterPublish` waited for that lock.
    /// Sizes: 4 KB then two 8 KB stores fill a 12 KB budget to 20 KB; finishing the 4 KB write leaves
    /// 16 KB, still over, so the fourth store keeps waiting while the writer needs the lock.
    @Test func fullWriteBehindBudgetDoesNotDeadlockCoordinatorStores() throws {
        try MLXMetalTestLock.withLock {
            let dir = root("budget-deadlock")
            defer { try? FileManager.default.removeItem(at: dir) }
            DiskCache.inflightByteBudgetOverrideForTesting = 12 * 1024
            defer { DiskCache.inflightByteBudgetOverrideForTesting = nil }
            var config = CacheCoordinatorConfig(
                usePagedCache: false, enableDiskCache: true, diskCacheMaxGB: 1, diskCacheDir: dir,
                modelKey: "wb")
            config.diskWriteBehind = true
            let coordinator = CacheCoordinator(config: config)
            let disk = try #require(coordinator.diskCache)
            disk.writeDelayForTesting = 0.3
            let stores: [([Int], Int)] = [
                (Array(1 ..< 20), 1_024), (Array(21 ..< 40), 2_048), (Array(41 ..< 60), 2_048),
                (Array(61 ..< 80), 2_048),
            ]
            let done = DispatchSemaphore(value: 0)
            Thread.detachNewThread {
                for (tokens, count) in stores {
                    coordinator.storePersistentBoundary(
                        tokens: tokens,
                        diskArrays: ["keys": MLXArray.ones([count], dtype: .float32) * Float(count)],
                        ssmStates: nil, chainId: "A", isResumeBoundary: true)
                }
                done.signal()
            }
            #expect(done.wait(timeout: .now() + 20) == .success, "stores deadlocked on a full budget")
            #expect(coordinator.waitForPendingDiskWrites(timeout: 20))
            for (tokens, _) in stores {
                #expect(coordinator.hasDurableDiskEntry(tokens: tokens))
            }
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

