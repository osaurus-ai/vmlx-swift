import Foundation
import MLX
@testable import MLXLMCommon
import Testing

/// SSMStateCache tests are wrapped in a `.serialized` suite because MLX's
/// Metal command buffer state is process-global. When swift-testing runs
/// multiple `@Test` bodies concurrently and each allocates its own MLXArray
/// / runs `.eval()`, the default shared command encoder can get two
/// concurrent "start encoding" calls and trip the AGX driver's
/// `tryCoalescingPreviousComputeCommandEncoderWithConfig:nextEncoderClass:`
/// assertion ("A command encoder is already encoding to this command
/// buffer"). That's an MLX-level parallelism issue, not a bug in these
/// tests — each test passes cleanly in isolation. Serializing within the
/// suite makes `swift test` reliable without needing `--no-parallel`.
@Suite(.serialized)
struct SSMStateCacheTests {
@Test func ssmStateCacheStoreAndFetch() {
    let cache = SSMStateCache(maxEntries: 10)
    let tokens = [1, 2, 3, 4, 5]
    let boundary = 3

    // Store two state arrays
    let state0 = MLXArray.ones([2, 4])
    let state1 = MLXArray.zeros([2, 4])
    cache.store(ssmStates: [state0, state1], tokens: tokens, boundary: boundary)

    // Fetch should return non-nil with correct count
    let fetched = cache.fetch(tokens: tokens, boundary: boundary)
    #expect(fetched != nil)
    #expect(fetched?.count == 2)
    #expect(cache.hits == 1)

    // Different tokens should miss
    let missed = cache.fetch(tokens: [9, 8, 7, 6, 5], boundary: boundary)
    #expect(missed == nil)
    #expect(cache.misses == 1)
}

@Test func ssmStateCacheLRUEviction() {
    let cache = SSMStateCache(maxEntries: 2)

    let stateA = [MLXArray.ones([1, 2])]
    let stateB = [MLXArray.ones([1, 2]) * 2]
    let stateC = [MLXArray.ones([1, 2]) * 3]

    cache.store(ssmStates: stateA, tokens: [10, 20], boundary: 2)
    cache.store(ssmStates: stateB, tokens: [30, 40], boundary: 2)
    cache.store(ssmStates: stateC, tokens: [50, 60], boundary: 2)

    // Oldest entry (tokens [10, 20]) should have been evicted
    let evicted = cache.fetch(tokens: [10, 20], boundary: 2)
    #expect(evicted == nil)

    // Second and third entries should still be present
    let second = cache.fetch(tokens: [30, 40], boundary: 2)
    #expect(second != nil)

    let third = cache.fetch(tokens: [50, 60], boundary: 2)
    #expect(third != nil)
}

@Test func ssmStateCacheDeepCopy() {
    let cache = SSMStateCache(maxEntries: 10)
    let tokens = [1, 2, 3]
    let boundary = 3

    let original = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float])
    cache.store(ssmStates: [original], tokens: tokens, boundary: boundary)

    // Fetch and modify the returned copy
    if let fetched = cache.fetch(tokens: tokens, boundary: boundary) {
        #expect(fetched.count == 1)
        // The fetched array should be independent — mutating it should not
        // affect a subsequent fetch.
        let modified = fetched[0] + 100
        MLX.eval(modified)
        _ = modified  // silence unused warning

        // Re-fetch: values should be unmodified
        if let refetched = cache.fetch(tokens: tokens, boundary: boundary) {
            let values = refetched[0].asArray(Float.self)
            #expect(values == [1.0, 2.0, 3.0, 4.0])
        } else {
            Issue.record("Re-fetch should not be nil")
        }
    } else {
        Issue.record("First fetch should not be nil")
    }
}

@Test func ssmStateCacheEmptyStatesIsMiss() {
    let cache = SSMStateCache(maxEntries: 10)
    let tokens = [1, 2, 3]
    let boundary = 3

    // Store an empty states array
    cache.store(ssmStates: [], tokens: tokens, boundary: boundary)

    // Fetch should treat empty states as a miss
    let result = cache.fetch(tokens: tokens, boundary: boundary)
    #expect(result == nil)
    #expect(cache.misses == 1)
    #expect(cache.hits == 0)
}

@Test func ssmStateCacheKeyDeterminism() {
    let tokens = [100, 200, 300, 400, 500]
    let boundary = 3

    // Same tokens + boundary should always produce the same key
    let keys = (0..<10).map { _ in
        SSMStateCache.makeKey(tokens: tokens, boundary: boundary)
    }

    let first = keys[0]
    #expect(first.count == 64)  // SHA-256 = 64 hex chars

    for key in keys {
        #expect(key == first)
    }

    // Different boundary should produce a different key
    let differentBoundary = SSMStateCache.makeKey(tokens: tokens, boundary: 4)
    #expect(differentBoundary != first)

    // Different tokens (same boundary) should produce a different key
    let differentTokens = SSMStateCache.makeKey(tokens: [100, 200, 999], boundary: 3)
    #expect(differentTokens != first)
}

@Test func ssmCompanionDiskStoreRoundTrip() throws {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("ssm-companion-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: dir) }

    let store = try SSMCompanionDiskStore(
        cacheDir: dir,
        modelKey: "test-model",
        maxBytes: 10_000_000)
    let tokens = [11, 22, 33, 44]
    let state0 = MLXArray([1.0, 2.0, 3.0] as [Float])
    let state1 = MLXArray.ones([1, 2])

    try store.store(
        ssmStates: [state0, state1],
        tokens: tokens,
        boundary: 3,
        mediaSalt: "audio-a",
        isComplete: false)

    let fetched = store.fetch(tokens: tokens, boundary: 3, mediaSalt: "audio-a")
    #expect(fetched != nil)
    #expect(fetched?.states.count == 2)
    #expect(fetched?.isComplete == false)
    #expect(fetched?.states[0].asArray(Float.self) == [1.0, 2.0, 3.0])

    let saltedMiss = store.fetch(tokens: tokens, boundary: 3, mediaSalt: "audio-b")
    #expect(saltedMiss == nil)
}

@Test func ssmCompanionDiskStoreEvictsOverCap() throws {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("ssm-companion-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: dir) }

    let store = try SSMCompanionDiskStore(
        cacheDir: dir,
        modelKey: "test-model",
        maxBytes: 1)

    try store.store(
        ssmStates: [MLXArray.ones([4, 4])],
        tokens: [1, 2, 3, 4],
        boundary: 4)

    let fetched = store.fetch(tokens: [1, 2, 3, 4], boundary: 4)
    #expect(fetched == nil)
}

@Test func transientPagedCompanionStateDoesNotWriteThroughToDisk() throws {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("ssm-companion-transient-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: dir) }

    let disk = try SSMCompanionDiskStore(
        cacheDir: dir,
        modelKey: "test-model",
        maxBytes: 10_000_000)
    let cache = SSMStateCache(maxEntries: 4, modelKey: "test-model")
    cache.diskStore = disk
    let tokens = [1, 2, 3, 4]

    cache.store(
        ssmStates: [MLXArray.ones([4, 4])],
        tokens: tokens,
        boundary: tokens.count,
        persistToDisk: false)

    #expect(cache.contains(tokens: tokens, boundary: tokens.count))
    #expect(disk.fetch(tokens: tokens, boundary: tokens.count) == nil)
}

@Test func coordinatorClearRemovesPersistentCompanionPayloads() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ssm-companion-clear-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let tokens = [5, 6, 7, 8]

    let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
        usePagedCache: false,
        enableDiskCache: true,
        diskCacheMaxGB: 0.1,
        diskCacheDir: root,
        modelKey: "clear-test"))
    coordinator.ssmStateCache.store(
        ssmStates: [MLXArray.ones([4, 4])],
        tokens: tokens,
        boundary: tokens.count)
    let disk = try #require(coordinator.ssmStateCache.diskStore)
    #expect(disk.fetch(tokens: tokens, boundary: tokens.count) != nil)

    coordinator.clear()

    #expect(disk.fetch(tokens: tokens, boundary: tokens.count) == nil)
}
}
