import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import Testing

/// Regression for the GLM-5.3 (`glm5_next`) first-generation crash: snapshotting
/// a FRESH cache at the prompt boundary called `Glm5NextIndexedKVCache.copy()`,
/// which round-tripped the inner `KVCacheSimple`'s EMPTY state through a setter
/// that traps unless it receives exactly `[keys, values]`. The app crashed
/// (EXC_BREAKPOINT in `KVCacheSimple.state.setter`) before producing a token.
/// Every other cache's `copy()` already guards the empty case; GLM-5.3's did
/// not. These tests pin that a fresh cache copies (and restores) without a trap.
@Suite("GLM-5.3 indexed KV cache copy", .serialized)
struct Glm5NextCacheCopyTests {
    @Test("copy() of a FRESH (empty) cache does not trap in the inner state setter")
    func copyOfFreshCacheDoesNotTrap() {
        let cache = Glm5NextIndexedKVCache()

        // The crash path: makePromptBoundaryCacheSnapshot -> copy() on a cache
        // with no tokens yet.
        let copy = cache.copy() as! Glm5NextIndexedKVCache

        #expect(copy.offset == 0)
        #expect(copy.state.isEmpty)
        #expect(copy.indexerPacked == nil)
    }

    @Test("a populated cache still round-trips keys, values, and the indexer buffer")
    func copyOfPopulatedCachePreservesState() {
        let cache = Glm5NextIndexedKVCache()

        // Shape [B, H, L, D]: one head, four positions, small head dim.
        let keys = MLXArray(0..<32, [1, 1, 4, 8]).asType(.float32)
        let values = MLXArray(100..<132, [1, 1, 4, 8]).asType(.float32)
        _ = cache.update(keys: keys, values: values)

        #expect(cache.offset == 4)
        #expect(cache.state.count == 2)

        let copy = cache.copy() as! Glm5NextIndexedKVCache
        #expect(copy.offset == 4)
        #expect(copy.state.count == 2)

        let originalKeys = cache.state[0]
        let copiedKeys = copy.state[0]
        #expect(copiedKeys.shape == originalKeys.shape)
        #expect(allClose(copiedKeys, originalKeys, atol: 0).all().item(Bool.self))
    }

    @Test("assigning an empty state to a fresh cache does not trap")
    func emptyStateRestoreDoesNotTrap() {
        let cache = Glm5NextIndexedKVCache()
        cache.state = []
        #expect(cache.offset == 0)
        #expect(cache.state.isEmpty)
    }
}
