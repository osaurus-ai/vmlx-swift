import Foundation
import MLX
@testable import MLXLMCommon
import Testing

@Suite(.serialized)
struct MambaPersistentDiskTests {
    @Test
    func sameKeyPayloadLayoutChangesAreNotDeduplicated() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("disk-layout-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let disk = DiskCache(cacheDir: root, maxSizeGB: 0.1)
        let tokens = [2, 3, 5]
        for change in 0..<4 {
            let arrays: [String: MLXArray] = [
                "data": change < 2 ? MLXArray.ones([2, 2])
                    : MLXArray.ones([2, 4]).asType(change == 2 ? .float32 : .float16),
                "__test_geometry__": MLXArray([Int32(change == 0 ? 2 : 4)]),
            ]
            let skips = disk.snapshotStats().storeSkips
            disk.store(tokens: tokens, arrays: arrays)
            #expect(disk.snapshotStats().storeSkips == skips)
            let restored = try #require(disk.fetch(tokens: tokens))
            #expect(restored["data"]?.shape == arrays["data"]?.shape)
            #expect(restored["data"]?.dtype == arrays["data"]?.dtype)
            #expect(restored["__test_geometry__"]?.item(Int32.self) == (change == 0 ? 2 : 4))
            disk.store(tokens: tokens, arrays: arrays)
            #expect(disk.snapshotStats().storeSkips == skips + 1)
        }
    }

    @Test
    func legacyDiskRecordCanBeReplacedByCompletePersistentState() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mamba-migration-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let disk = DiskCache(cacheDir: root, maxSizeGB: 0.1)
        let source = MambaCache(slots: 6, persistentSlotCount: 4)
        for slot in 0..<4 { source[slot] = MLXArray([Int32(slot + 10)]) }
        source.offset = 8
        let complete = TQDiskSerializer.serialize(cache: [attention(), source])
        var legacy = complete
        for key in ["__mamba_1_slots__", "__mamba_1_occupied__",
                    "mamba_1_state2", "mamba_1_state3"] {
            legacy.removeValue(forKey: key)
        }
        let tokens = Array(0..<8)
        disk.store(tokens: tokens, arrays: legacy)
        #expect(!disk.hasValidatedEntry(tokens: tokens))
        #expect(!disk.hasDurableEntry(tokens: tokens))
        let reopened = DiskCache(cacheDir: root, maxSizeGB: 0.1)
        #expect(!reopened.hasDurableEntry(tokens: tokens))
        let candidate = try #require(disk.fetch(tokens: tokens))
        var rejected: [any KVCache] = [KVCacheSimple(), MambaCache(slots: 6, persistentSlotCount: 4)]
        #expect(restoreFromDiskArrays(candidate, into: &rejected) == 0)
        // A real full prefill republishes the same token key with complete state.
        // A file-level validation must not suppress this format migration.
        disk.store(tokens: tokens, arrays: complete)
        #expect(disk.hasValidatedEntry(tokens: tokens))
        #expect(reopened.hasDurableEntry(tokens: tokens))
        let migrated = try #require(disk.fetch(tokens: tokens))
        var restored: [any KVCache] = [KVCacheSimple(), MambaCache(slots: 6, persistentSlotCount: 4)]
        #expect(restoreFromDiskArrays(migrated, into: &restored) == 8)
        let skips = disk.snapshotStats().storeSkips
        disk.store(tokens: tokens, arrays: complete)
        #expect(disk.snapshotStats().storeSkips == skips + 1)
    }

    @Test
    func malformedPromptGeometryThrowsWithoutAllocatingDeclaredCapacity() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let source = MambaCache(slots: 6, persistentSlotCount: 4)
        for slot in 0..<4 { source[slot] = MLXArray([Int32(slot)]) }
        source.offset = 8
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("mamba-malformed-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: path) }
        try savePromptCache(url: path, cache: [source])
        let (arrays, originalMetadata) = try loadArraysAndMetadata(url: path)
        for damage in 0..<5 {
            var metadata = originalMetadata
            switch damage {
            case 0: metadata["0.0.1"] = "2147483647"
            case 1: metadata["0.0.4"] = "0,1,2,2"
            case 2: metadata["0.0.3"] = "-1"
            case 3: metadata["0.0.4"] = "0,1,2,9"
            default: metadata["2.0"] = "MambaCache"
            }
            try MLX.save(arrays: arrays, metadata: metadata, url: path)
            #expect(throws: KVCacheError.self) { try loadPromptCache(url: path) }
        }
    }

    @Test
    func promptCacheFilePreservesDeclaredGeometryAndHoles() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let source = MambaCache(slots: 6, persistentSlotCount: 4)
        source[0] = MLXArray([Int32(10)])
        source[2] = MLXArray([Int32(12)])
        source[3] = MLXArray([Int32(13)])
        source[4] = MLXArray([Int32(999)])
        source.offset = 8
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("mamba-prompt-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: path) }
        try savePromptCache(url: path, cache: [source], metadata: ["test": "preserve"])
        let (restored, metadata) = try loadPromptCache(url: path)
        let result = try #require(restored.first as? MambaCache)
        #expect(metadata == ["test": "preserve"])
        #expect(result.slotCount == 6 && result.persistentSlotCount == 4)
        #expect(result.offset == 8)
        if result.slotCount >= 6 {
            #expect(result[1] == nil && result[4] == nil && result[5] == nil)
            #expect(result[2]?.item(Int32.self) == 12)
            #expect(result[3]?.item(Int32.self) == 13)
        }
    }

    private func attention() -> KVCacheSimple {
        let cache = KVCacheSimple()
        _ = cache.update(keys: MLXArray.ones([1, 1, 8, 4]), values: MLXArray.ones([1, 1, 8, 4]))
        return cache
    }

    @Test
    func compiledPromotionPreservesHolesAndExcludesScratch() {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let source = MambaCache(slots: 6, persistentSlotCount: 4)
        source[0] = MLXArray([Int32(10)])
        source[2] = MLXArray([Int32(12)])
        source[3] = MLXArray([Int32(13)])
        source[4] = MLXArray([Int32(999)])
        source[5] = MLXArray([Int32(998)])
        source.offset = 8
        let promoted = CompilableMambaCache(from: source)
        let copied = promoted.copy() as! CompilableMambaCache
        #expect(copied.persistentSlotCount == 4)
        #expect(copied.state.count == 3)
        #expect(copied.innerState().count == 5)
        let encoded = TQDiskSerializer.serialize(cache: [attention(), copied])
        #expect(encoded["mamba_1_state1"] == nil)
        #expect(encoded["mamba_1_state4"] == nil && encoded["mamba_1_state5"] == nil)
        let target = CompilableMambaCache(slots: 6, persistentSlotCount: 4)
        target[1] = MLXArray([Int32(7)])
        target[4] = MLXArray([Int32(8)])
        var restored: [any KVCache] = [KVCacheSimple(), target]
        #expect(restoreFromDiskArrays(encoded, into: &restored) == 8)
        #expect(target[1] == nil && target[4] == nil && target[5] == nil)
        #expect(target[2]?.item(Int32.self) == 12)
        #expect(target[3]?.item(Int32.self) == 13)
    }

    @Test
    func damagedMetadataRefusesBeforeSiblingMutation() {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let source = MambaCache(slots: 6, persistentSlotCount: 4)
        for slot in 0..<4 { source[slot] = MLXArray([Int32(slot + 10)]) }
        source.offset = 8
        let original = TQDiskSerializer.serialize(cache: [attention(), source])
        for damage in 0..<7 {
            var encoded = original
            switch damage {
            case 0: encoded.removeValue(forKey: "mamba_1_state3")
            case 1: encoded.removeValue(forKey: "__mamba_1_slots__")
            case 2: encoded.removeValue(forKey: "__mamba_1_occupied__")
            case 3: encoded["__mamba_1_occupied__"] = MLXArray([Int32(0), 1, 2, 2])
            case 4: encoded["__mamba_1_occupied__"] = MLXArray([Int32(0), 1, 2, 7])
            case 5: encoded["__mamba_1_slots__"] = MLXArray(Int32(5))
            default: encoded.removeValue(forKey: "__mamba_1_offset__")
            }
            let sibling = KVCacheSimple()
            let target = MambaCache(slots: 6, persistentSlotCount: 4)
            var restored: [any KVCache] = [sibling, target]
            #expect(restoreFromDiskArrays(encoded, into: &restored) == 0, "damage=\(damage)")
            #expect(sibling.offset == 0 && target.state.isEmpty, "damage=\(damage)")
        }
    }

    @Test
    func nestedRecurrentChildrenRestoreByIndexAndRejectDamage() {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let first = MambaCache()
        first[0] = MLXArray([Int32(7)])
        first.offset = 8
        let second = MambaCache(slots: 6, persistentSlotCount: 4)
        for slot in 0..<4 { second[slot] = MLXArray([Int32(slot + 20)]) }
        second.offset = 8
        let original = TQDiskSerializer.serialize(cache: [attention(), CacheList(first, second)])
        for damaged in [false, true] {
            var encoded = original
            if damaged { encoded.removeValue(forKey: "mamba_1_sub_1_state3") }
            let sibling = KVCacheSimple()
            let a = MambaCache()
            let b = MambaCache(slots: 6, persistentSlotCount: 4)
            var restored: [any KVCache] = [sibling, CacheList(a, b)]
            #expect(restoreFromDiskArrays(encoded, into: &restored) == (damaged ? 0 : 8))
            if damaged {
                #expect(sibling.offset == 0 && a.state.isEmpty && b.state.isEmpty)
            } else {
                #expect(a[0]?.item(Int32.self) == 7 && a[1] == nil)
                #expect(b[0]?.item(Int32.self) == 20 && b[3]?.item(Int32.self) == 23)
            }
        }
    }

    @Test
    func ordinaryLegacyOneAndTwoOccupiedSlotsRemainReadable() {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        for occupied in [1, 2] {
            let source = MambaCache()
            for slot in 0..<occupied { source[slot] = MLXArray([Int32(slot + 30)]) }
            source.offset = 8
            var encoded = TQDiskSerializer.serialize(cache: [attention(), source])
            encoded.removeValue(forKey: "__mamba_1_slots__")
            encoded.removeValue(forKey: "__mamba_1_occupied__")
            let target = MambaCache()
            var restored: [any KVCache] = [KVCacheSimple(), target]
            #expect(restoreFromDiskArrays(encoded, into: &restored) == 8)
            #expect(target.state.count == occupied)
            #expect(target[0]?.item(Int32.self) == 30)
        }
    }

    @Test
    func extendedStateSurvivesWithoutCompanion() {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let recurrent = MambaCache(slots: 6, persistentSlotCount: 4)
        for slot in 0..<4 { recurrent[slot] = MLXArray([Int32(slot + 10)]) }
        recurrent.offset = 8
        let attention = KVCacheSimple()
        _ = attention.update(
            keys: MLXArray.ones([1, 1, 8, 4]), values: MLXArray.ones([1, 1, 8, 4]))
        let encoded = TQDiskSerializer.serialize(cache: [attention, recurrent])
        let target = MambaCache(slots: 6, persistentSlotCount: 4)
        var restored: [any KVCache] = [KVCacheSimple(), target]
        #expect(restoreFromDiskArrays(encoded, into: &restored) == 8)
        for slot in 0..<4 {
            #expect(target[slot]?.item(Int32.self) == Int32(slot + 10))
        }
        #expect(target.slotCount == 6)
        #expect(target[4] == nil && target[5] == nil)
    }

    @Test
    func legacyTwoSlotPayloadCannotSeatExtendedCache() {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let old = MambaCache()
        old[0] = MLXArray([Int32(10)])
        old[1] = MLXArray([Int32(11)])
        old.offset = 8
        let attention = KVCacheSimple()
        _ = attention.update(
            keys: MLXArray.ones([1, 1, 8, 4]), values: MLXArray.ones([1, 1, 8, 4]))
        var encoded = TQDiskSerializer.serialize(cache: [attention, old])
        encoded.removeValue(forKey: "__mamba_1_slots__")
        encoded.removeValue(forKey: "__mamba_1_occupied__")
        let target = MambaCache(slots: 6, persistentSlotCount: 4)
        let untouched = KVCacheSimple()
        var restored: [any KVCache] = [untouched, target]
        #expect(restoreFromDiskArrays(encoded, into: &restored) == 0)
        #expect(untouched.offset == 0)
        #expect(target.state.isEmpty)
    }
}
