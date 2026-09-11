import Foundation
import MLX
@testable import MLXLMCommon
import Testing

// MARK: - extractLayerData Tests

@Test func extractLayerDataFromSimpleCaches() {
    // Create 2 KVCacheSimple layers with known state
    let cache1 = KVCacheSimple()
    let cache2 = KVCacheSimple()

    // Simulate populating via update (shape: [B=1, H=2, T=4, D=8])
    let keys1 = MLXArray.ones([1, 2, 4, 8])
    let values1 = MLXArray.zeros([1, 2, 4, 8])
    _ = cache1.update(keys: keys1, values: values1)

    let keys2 = MLXArray.ones([1, 2, 4, 8]) * 2
    let values2 = MLXArray.zeros([1, 2, 4, 8]) + 1
    _ = cache2.update(keys: keys2, values: values2)

    let result = extractLayerData(from: [cache1, cache2])

    #expect(result.count == 2)
    #expect(result[0] != nil)
    #expect(result[1] != nil)

    // Verify shapes match
    #expect(result[0]!.keys.shape == [1, 2, 4, 8])
    #expect(result[0]!.values.shape == [1, 2, 4, 8])
    #expect(result[1]!.keys.shape == [1, 2, 4, 8])
    #expect(result[1]!.values.shape == [1, 2, 4, 8])
}

@Test func extractLayerDataSkipsMambaCache() {
    let kvCache = KVCacheSimple()
    let keys = MLXArray.ones([1, 2, 4, 8])
    let values = MLXArray.zeros([1, 2, 4, 8])
    _ = kvCache.update(keys: keys, values: values)

    let mambaCache = MambaCache()

    let result = extractLayerData(from: [kvCache, mambaCache])

    #expect(result.count == 2)
    #expect(result[0] != nil)  // KVCacheSimple extracted
    #expect(result[1] == nil)  // MambaCache returns nil
}

@Test func extractLayerDataFromEmptyCache() {
    let cache = KVCacheSimple()
    let result = extractLayerData(from: [cache])

    #expect(result.count == 1)
    #expect(result[0] == nil)  // Empty cache has no state
}

@Test func extractLayerDataFromCacheList() {
    // CacheList with MambaCache + KVCacheSimple (hybrid layer)
    let mamba = MambaCache()
    let kv = KVCacheSimple()
    let keys = MLXArray.ones([1, 2, 3, 8])
    let values = MLXArray.zeros([1, 2, 3, 8])
    _ = kv.update(keys: keys, values: values)

    let cacheList = CacheList(mamba, kv)

    let result = extractLayerData(from: [cacheList])

    #expect(result.count == 1)
    #expect(result[0] != nil)  // Should find KVCacheSimple inside CacheList
    #expect(result[0]!.keys.shape == [1, 2, 3, 8])
}

// MARK: - restoreLayerData Tests

@Test func restoreLayerDataSingleBlock() {
    // Create a block with KV data for 2 layers
    let block = CacheBlock(blockId: 0, blockSize: 32)
    block.tokenIds = [1, 2, 3, 4]
    block.cacheData = [
        (keys: MLXArray.ones([1, 2, 4, 8]), values: MLXArray.zeros([1, 2, 4, 8])),
        (keys: MLXArray.ones([1, 2, 4, 8]) * 2, values: MLXArray.zeros([1, 2, 4, 8]) + 1),
    ]

    let cache1 = KVCacheSimple()
    let cache2 = KVCacheSimple()

    let restored = restoreLayerData(from: [block], into: [cache1, cache2])

    #expect(restored == 4)
    #expect(cache1.offset == 4)
    #expect(cache2.offset == 4)

    // Verify state was restored
    let state1 = cache1.state
    #expect(state1.count == 2)
    #expect(state1[0].shape == [1, 2, 4, 8])
}

@Test func restoreLayerDataMultipleBlocks() {
    // Two blocks, each with 2 tokens
    let block1 = CacheBlock(blockId: 0, blockSize: 32)
    block1.tokenIds = [1, 2]
    block1.cacheData = [
        (keys: MLXArray.ones([1, 2, 2, 8]), values: MLXArray.zeros([1, 2, 2, 8])),
    ]

    let block2 = CacheBlock(blockId: 1, blockSize: 32)
    block2.tokenIds = [3, 4]
    block2.cacheData = [
        (keys: MLXArray.ones([1, 2, 2, 8]) * 3, values: MLXArray.zeros([1, 2, 2, 8]) + 3),
    ]

    let cache = KVCacheSimple()
    let restored = restoreLayerData(from: [block1, block2], into: [cache])

    #expect(restored == 4)
    #expect(cache.offset == 4)

    // Should have concatenated along axis 2 -> T=4
    let state = cache.state
    #expect(state[0].shape == [1, 2, 4, 8])
}

@Test func restoreLayerDataEmptyBlocks() {
    let cache = KVCacheSimple()
    let restored = restoreLayerData(from: [], into: [cache])
    #expect(restored == 0)
    #expect(cache.offset == 0)
}

@Test func restoreLayerDataLayerCountMismatch() {
    // Block has 2 layers but cache has 1
    let block = CacheBlock(blockId: 0, blockSize: 32)
    block.tokenIds = [1]
    block.cacheData = [
        (keys: MLXArray.ones([1, 2, 1, 8]), values: MLXArray.zeros([1, 2, 1, 8])),
        (keys: MLXArray.ones([1, 2, 1, 8]), values: MLXArray.zeros([1, 2, 1, 8])),
    ]

    let cache = KVCacheSimple()
    let restored = restoreLayerData(from: [block], into: [cache])
    #expect(restored == 0)  // Mismatch -> no restoration
}

// MARK: - extractSSMStates Tests

@Test func extractSSMStatesFromMambaCache() {
    let mamba = MambaCache()
    // Populate with conv_state and hidden_state
    let convState = MLXArray.ones([1, 4, 16])
    let hiddenState = MLXArray.zeros([1, 4, 16])
    mamba.state = [convState, hiddenState]

    let states = extractSSMStates(from: [mamba])

    #expect(states.count == 2)
    #expect(states[0].shape == [1, 4, 16])
    #expect(states[1].shape == [1, 4, 16])
}

@Test func extractSSMStatesSkipsKVCache() {
    let kv = KVCacheSimple()
    let keys = MLXArray.ones([1, 2, 4, 8])
    let values = MLXArray.zeros([1, 2, 4, 8])
    _ = kv.update(keys: keys, values: values)

    let states = extractSSMStates(from: [kv])
    #expect(states.isEmpty)
}

@Test func extractSSMStatesFromHybrid() {
    // Mixed: KVCacheSimple layer, MambaCache layer
    let kv = KVCacheSimple()
    _ = kv.update(keys: MLXArray.ones([1, 2, 4, 8]), values: MLXArray.zeros([1, 2, 4, 8]))

    let mamba = MambaCache()
    mamba.state = [MLXArray.ones([1, 4, 16]), MLXArray.zeros([1, 4, 16])]

    let states = extractSSMStates(from: [kv, mamba])
    #expect(states.count == 2)  // Only from MambaCache
}

@Test func extractSSMStatesFromCacheListComposite() {
    let mamba = MambaCache()
    mamba.state = [MLXArray.ones([1, 4, 16]), MLXArray.zeros([1, 4, 16])]

    let kv = KVCacheSimple()
    _ = kv.update(keys: MLXArray.ones([1, 2, 3, 8]), values: MLXArray.zeros([1, 2, 3, 8]))

    let cacheList = CacheList(mamba, kv)

    let states = extractSSMStates(from: [cacheList])
    #expect(states.count == 2)  // MambaCache's 2 states extracted from inside CacheList
}

// MARK: - restoreSSMStates Tests

@Test func restoreSSMStatesIntoMambaCache() {
    let mamba = MambaCache()
    // Pre-populate so existingCount > 0
    mamba.state = [MLXArray.zeros([1, 4, 16]), MLXArray.zeros([1, 4, 16])]

    let states = [MLXArray.ones([1, 4, 16]), MLXArray.ones([1, 4, 16]) * 2]
    restoreSSMStates(states, into: [mamba])

    let restored = mamba.state
    #expect(restored.count == 2)
}

@Test func restoreSSMStatesIntoFreshMambaCache() {
    let mamba = MambaCache()
    // Fresh cache — state is empty

    let states = [MLXArray.ones([1, 4, 16]), MLXArray.ones([1, 4, 16]) * 2]
    restoreSSMStates(states, into: [mamba])

    let restored = mamba.state
    #expect(restored.count == 2)
}

@Test func restoreSSMStatesSupportsExtendedPLEMambaAlongsideOrdinaryMamba() {
    let sourcePLE = MambaCache(slots: 6, persistentSlotCount: 4)
    for index in 0 ..< 4 {
        sourcePLE[index] = MLXArray([Int32(index + 10)])
    }
    let sourceOrdinary = MambaCache()
    sourceOrdinary[0] = MLXArray([Int32(20)])
    sourceOrdinary[1] = MLXArray([Int32(21)])

    let states = extractSSMStates(from: [sourcePLE, sourceOrdinary])
    #expect(states.count == 6)

    let restoredPLE = MambaCache(slots: 6, persistentSlotCount: 4)
    let restoredOrdinary = MambaCache()
    restoreSSMStates(states, into: [restoredPLE, restoredOrdinary], boundary: 17)

    #expect(restoredPLE.slotCount == 6)
    #expect(restoredPLE.state.count == 4)
    #expect(restoredPLE[2]?.item(Int32.self) == 12)
    #expect(restoredPLE[4] == nil)
    #expect(restoredPLE[5] == nil)
    #expect(restoredPLE.offset == 17)
    #expect(restoredOrdinary[1]?.item(Int32.self) == 21)
    #expect(restoredOrdinary.offset == 17)
}

@Test func restoreSSMStatesSkipsKVCache() {
    let kv = KVCacheSimple()
    let states = [MLXArray.ones([1, 4, 16])]
    restoreSSMStates(states, into: [kv])

    // KVCacheSimple should be untouched
    #expect(kv.offset == 0)
    #expect(kv.state.isEmpty)
}

// MARK: - CacheList.count Tests

@Test func cacheListCountProperty() {
    let kv1 = KVCacheSimple()
    let kv2 = KVCacheSimple()
    let mamba = MambaCache()

    let list2 = CacheList(kv1, kv2)
    #expect(list2.count == 2)

    let list3 = CacheList(mamba, kv1, kv2)
    #expect(list3.count == 3)

    let list1 = CacheList(kv1)
    #expect(list1.count == 1)
}

// MARK: - Round-trip Tests

@Test func roundTripKVExtractRestore() {
    // Create caches, populate, extract, restore into fresh caches
    let original = KVCacheSimple()
    let keys = MLXArray.ones([1, 2, 5, 8])
    let values = MLXArray.zeros([1, 2, 5, 8]) + 0.5
    _ = original.update(keys: keys, values: values)

    // Extract
    let layerData = extractLayerData(from: [original])
    #expect(layerData[0] != nil)

    // Package into a block
    let block = CacheBlock(blockId: 0, blockSize: 32)
    block.tokenIds = [10, 20, 30, 40, 50]
    block.cacheData = layerData

    // Restore into fresh cache
    let fresh = KVCacheSimple()
    let restored = restoreLayerData(from: [block], into: [fresh])

    #expect(restored == 5)
    #expect(fresh.offset == 5)

    // Compare states
    let origState = original.state
    let freshState = fresh.state
    #expect(origState.count == freshState.count)
    #expect(origState[0].shape == freshState[0].shape)
    #expect(origState[1].shape == freshState[1].shape)
}

@Test func roundTripSSMExtractRestore() {
    let mamba = MambaCache()
    let convState = MLXArray.ones([1, 4, 16]) * 3
    let hiddenState = MLXArray.zeros([1, 4, 16]) + 7
    mamba.state = [convState, hiddenState]

    // Extract
    let states = extractSSMStates(from: [mamba])
    #expect(states.count == 2)

    // Restore into fresh MambaCache
    let fresh = MambaCache()
    restoreSSMStates(states, into: [fresh])

    let restoredState = fresh.state
    #expect(restoredState.count == 2)
    #expect(restoredState[0].shape == [1, 4, 16])
    #expect(restoredState[1].shape == [1, 4, 16])
}
