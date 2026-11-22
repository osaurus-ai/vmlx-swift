import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Windowed KV cache: every key row holds its own absolute position (values hold position + 0.5), so each test asserts
/// exactly which positions are held, returned, stored and restored, at lengths that do not divide the window, the
/// rollback slack or the 256-row growth step.
@Suite struct TemporalWindowKVCacheTests {
    private let window = 32

    private func rows(_ positions: Range<Int>, heads: Int = 1, dim: Int = 1) -> (MLXArray, MLXArray) {
        let tags = MLXArray(positions.map { Float($0) }).reshaped(1, 1, positions.count, 1)
        let k = MLX.broadcast(tags, to: [1, heads, positions.count, dim])
        return (k, k + 0.5)
    }

    private func positions(_ array: MLXArray) -> [Int] {
        array[0, 0, 0..., 0].asArray(Float.self).map { Int($0) }
    }

    private func valuePositions(_ array: MLXArray) -> [Int] {
        array[0, 0, 0..., 0].asArray(Float.self).map { Int($0 - 0.5) }
    }

    private func filled(_ cache: TemporalWindowKVCache, to count: Int, chunk: Int) {
        var at = cache.offset
        while at < count {
            let end = min(at + chunk, count)
            let (k, v) = rows(at ..< end)
            _ = cache.update(keys: k, values: v)
            at = end
        }
    }

    /// The held rows are exactly the positions just below the offset, and enough of them for the window.
    private func expectHeldInvariant(_ cache: TemporalWindowKVCache, comment: Comment? = nil) {
        #expect(cache.heldRows <= cache.offset, comment)
        #expect(cache.heldRows >= min(cache.offset, window - 1), comment)
        #expect(cache.firstPosition == cache.offset - cache.heldRows, comment)
        if let state = cache.state.first {
            let stored = min(cache.heldRows, cache.keep)
            #expect(state.dim(2) == stored, comment)
            #expect(positions(state) == Array((cache.offset - stored) ..< cache.offset), comment)
            #expect(valuePositions(cache.state[1]) == Array((cache.offset - stored) ..< cache.offset), comment)
        }
    }

    @Test(arguments: [(31, 1), (32, 7), (95, 1), (96, 13), (97, 5), (300, 300), (1023, 7), (1025, 512), (5000, 300)])
    func updateReturnsTheRecentRowsInOrder(length: Int, chunk: Int) {
        let cache = TemporalWindowKVCache(window: window)
        var at = 0
        while at < length {
            let end = min(at + chunk, length)
            let (k, v) = rows(at ..< end)
            let (keys, values) = cache.update(keys: k, values: v)
            let returned = positions(keys)
            // Every row of this update plus the full window of its first query, contiguous, ending at the last.
            #expect(returned.last == end - 1)
            #expect(returned == Array((end - returned.count) ..< end))
            #expect(returned.count >= min(end, window - 1 + (end - at)))
            #expect(valuePositions(values) == returned)
            at = end
            expectHeldInvariant(cache, comment: "after \(end)")
        }
        #expect(cache.offset == length)
        // RAM is bounded: never more than keep + headroom + one update.
        #expect(cache.capacity <= cache.keep + TemporalWindowKVCache.growth + chunk + TemporalWindowKVCache.growth)
    }

    @Test func copyAndStateHoldExactlyTheKeptRows() {
        let cache = TemporalWindowKVCache(window: window)
        filled(cache, to: 1025, chunk: 9)
        let copy = cache.copy() as! TemporalWindowKVCache
        #expect(copy.offset == 1025)
        #expect(copy.heldRows == cache.keep)
        #expect(copy.capacity == cache.keep)
        #expect(positions(copy.state[0]) == Array((1025 - cache.keep) ..< 1025))
        let (k, v) = rows(1025 ..< 1026)
        _ = cache.update(keys: k, values: v)
        #expect(positions(copy.state[0]).last == 1024)
    }

    @Test(arguments: [1, 2, 9, 40, 300])
    func speculativeRollbackIsExactAcrossACompaction(width: Int) {
        // Fill to just below a growth boundary so the verify update compacts.
        let cache = TemporalWindowKVCache(window: window)
        filled(cache, to: 1000, chunk: 1)
        while cache.heldRows + width <= cache.capacity {
            let (k, v) = rows(cache.offset ..< cache.offset + 1)
            _ = cache.update(keys: k, values: v)
        }
        let before = cache.offset
        let checkpoint = cache.speculativeCheckpoint()
        let (k, v) = rows(before ..< before + width)
        _ = cache.update(keys: k, values: v)
        // Reject all but the first verified row.
        #expect(cache.trim(width - 1) == width - 1)
        #expect(cache.offset == before + 1)
        expectHeldInvariant(cache, comment: "after rollback of \(width - 1)")
        let (k2, v2) = rows(9000 ..< 9001)
        let (keys, _) = cache.update(keys: k2, values: v2)
        let read = positions(keys)
        #expect(read.last == 9000)
        #expect(Array(read.dropLast().suffix(window - 1)) == Array((before + 1 - (window - 1)) ..< before + 1))
        // The checkpoint restores the pre-verify cache exactly.
        cache.restoreSpeculativeCheckpoint(checkpoint)
        #expect(cache.offset == before)
        expectHeldInvariant(cache, comment: "after checkpoint restore")
        #expect(cache.speculativeCheckpointIsExact)
        #expect(SpeculativeCacheListCheckpoint([cache]) != nil)
    }

    @Test func trimIsAllOrNothing() {
        let cache = TemporalWindowKVCache(window: window)
        filled(cache, to: 1025, chunk: 1025)
        let restored = TemporalWindowKVCache(window: window)
        #expect(restored.restoreDiskCacheState(cache.state, metadata: cache.metaState, offset: 1025))
        // A restored entry holds window - 1 + rollback rows: trims within the slack work, beyond it nothing moves.
        #expect(!restored.canTrim(cache.rollbackRows + 1))
        #expect(restored.trim(cache.rollbackRows + 1) == 0)
        #expect(restored.offset == 1025)
        #expect(restored.trim(cache.rollbackRows) == cache.rollbackRows)
        #expect(restored.offset == 1025 - cache.rollbackRows)
        expectHeldInvariant(restored)
        // Below the window everything is held, so any trim works.
        let short = TemporalWindowKVCache(window: window)
        filled(short, to: 20, chunk: 3)
        #expect(short.trim(19) == 19)
        #expect(short.offset == 1)
    }

    @Test(arguments: [1, 31, 33, 95, 96, 97, 1023, 1025, 5000])
    func diskStateRoundTripsAtItsLogicalPosition(length: Int) {
        let cache = TemporalWindowKVCache(window: window, heads: 1, headDim: 1)
        filled(cache, to: length, chunk: 7)
        let restored = TemporalWindowKVCache(window: window, heads: 1, headDim: 1)
        #expect(restored.restoreDiskCacheState(cache.state, metadata: cache.metaState, offset: length))
        #expect(restored.offset == length)
        expectHeldInvariant(restored)
        // The first update after a restore must not write into the restored (possibly shared) arrays.
        let seated = restored.state[0]
        let (k, v) = rows(length ..< length + 3)
        let (keys, _) = restored.update(keys: k, values: v)
        #expect(positions(keys).suffix(3) == [length, length + 1, length + 2])
        #expect(positions(seated).last == length - 1)
    }

    @Test func restoreRefusesEntriesThatDoNotFit() {
        let cache = TemporalWindowKVCache(window: window, heads: 1, headDim: 1)
        let (k, v) = rows(0 ..< (window - 2))
        let id = cache.diskCacheStateIdentifier
        // Too few rows for the window of position 300.
        #expect(!cache.restoreDiskCacheState([k, v], metadata: [id, "300"], offset: 300))
        // Foreign identifier / window, mismatched metadata, more rows than positions, wrong heads.
        #expect(!cache.restoreDiskCacheState([k, v], metadata: ["temporal-window-kv-v1-window-64", "30"], offset: 30))
        #expect(!cache.restoreDiskCacheState([k, v], metadata: [id, "31"], offset: 30))
        #expect(!cache.restoreDiskCacheState([k, v], metadata: [id, "29"], offset: 29))
        let (k2, v2) = rows(0 ..< 30, heads: 2)
        #expect(!cache.restoreDiskCacheState([k2, v2], metadata: [id, "30"], offset: 30))
        #expect(cache.offset == 0)
        #expect(cache.restoreDiskCacheState([k, v], metadata: [id, "30"], offset: 30))
    }

    @Test(arguments: [33, 1025, 5000])
    func serializedEntryRestoresThroughTheDiskTier(length: Int) {
        let source = TemporalWindowKVCache(window: window, heads: 1, headDim: 1)
        filled(source, to: length, chunk: 11)
        let full = KVCacheSimple()
        let (fk, fv) = rows(0 ..< length)
        _ = full.update(keys: fk, values: fv)
        let arrays = TQDiskSerializer.serialize(cache: [source, full])
        var target: [any KVCache] = [TemporalWindowKVCache(window: window, heads: 1, headDim: 1), KVCacheSimple()]
        #expect(restoreFromDiskArrays(arrays, into: &target) == length)
        let windowed = target[0] as! TemporalWindowKVCache
        #expect(windowed.offset == length)
        #expect(positions(windowed.state[0]) == Array((length - min(length, windowed.keep)) ..< length))
        #expect(validateRestoredCacheBoundary(target, matchedTokens: length, restoredTokens: length))
    }

    @Test func fullHistoryRecordIsRefusedByAWindowedSlot() {
        // An entry written before the slot became windowed (plain K/V rows) must miss cleanly, not seat nothing.
        let full = KVCacheSimple()
        let (k, v) = rows(0 ..< 100)
        _ = full.update(keys: k, values: v)
        let arrays = TQDiskSerializer.serialize(cache: [full])
        var target: [any KVCache] = [TemporalWindowKVCache(window: window)]
        #expect(restoreFromDiskArrays(arrays, into: &target) == 0)
        #expect(target[0].offset == 0)
    }

    @Test func promptCacheFileRoundTrips() throws {
        let cache = TemporalWindowKVCache(window: window)
        filled(cache, to: 1025, chunk: 100)
        let short = TemporalWindowKVCache(window: window)
        filled(short, to: 5, chunk: 2)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("twkv-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [cache, short], metadata: ["k": "v"])
        let (loaded, metadata) = try loadPromptCache(url: url)
        #expect(metadata["k"] == "v")
        let windowed = try #require(loaded[0] as? TemporalWindowKVCache)
        #expect(windowed.window == window)
        #expect(windowed.offset == 1025)
        #expect(positions(windowed.state[0]) == Array((1025 - cache.keep) ..< 1025))
        let shortLoaded = try #require(loaded[1] as? TemporalWindowKVCache)
        #expect(shortLoaded.offset == 5)
        #expect(positions(shortLoaded.state[0]) == Array(0 ..< 5))
    }
}
