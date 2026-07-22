import Foundation
import MLX
@testable import MLXLMCommon
import Testing

// MARK: - PagedCacheManager Tests

@Test func pagedCacheAllocationAndFree() {
    let manager = PagedCacheManager(blockSize: 4, maxBlocks: 8)

    // Allocate a block
    let block = manager.allocateBlock()
    #expect(block != nil)
    #expect(block!.blockId != 0, "Should never allocate the null sentinel (block 0)")
    #expect(block!.refCount == 1)

    // Stats reflect the allocation
    #expect(manager.stats.allocatedBlocks == 1)
    #expect(manager.stats.freeBlocks == 6) // 8 - 1 sentinel - 1 allocated

    // Free the block
    manager.freeBlock(block!)
    #expect(manager.stats.allocatedBlocks == 0)
    #expect(manager.stats.freeBlocks == 7) // back to 8 - 1 sentinel
}

@Test func pagedCacheExhaustion() {
    let manager = PagedCacheManager(blockSize: 4, maxBlocks: 4)
    // 4 total, block 0 = sentinel, so 3 allocatable

    var allocated: [CacheBlock] = []
    for _ in 0..<3 {
        let block = manager.allocateBlock()
        #expect(block != nil)
        allocated.append(block!)
    }

    // Pool exhausted — should return nil
    let overflow = manager.allocateBlock()
    #expect(overflow == nil)
    #expect(manager.stats.freeBlocks == 0)

    // Free one block and verify we can allocate again
    manager.freeBlock(allocated[0])
    #expect(manager.stats.freeBlocks == 1)

    let recycled = manager.allocateBlock()
    #expect(recycled != nil)
    #expect(recycled!.blockId == allocated[0].blockId)
}

@Test func pagedCachePrefixMatch() {
    let blockSize = 4
    let manager = PagedCacheManager(blockSize: blockSize, maxBlocks: 20)

    // Store 10 tokens — produces 2 full blocks (tokens 1-4, 5-8), remainder (9,10) is partial
    let tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    // Create dummy layer data for the 2 full blocks
    let dummyLayer: [(keys: MLXArray, values: MLXArray)] = [
        (keys: MLXArray.zeros([1, 1, blockSize, 8]), values: MLXArray.zeros([1, 1, blockSize, 8]))
    ]
    let layerData = [dummyLayer, dummyLayer]

    manager.storeTokenSequence(tokens: tokens, layerData: layerData)

    // Fetch with the same prefix plus extra tokens
    let query = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    let result = manager.fetchPrefix(tokens: query)

    #expect(result != nil)
    #expect(result!.matchedTokens == 8, "Should match 2 full blocks = 8 tokens")
    #expect(result!.remainingTokens == [9, 10, 11, 12])
    #expect(result!.blocks.count == 2)
}

@Test func pagedCachePartialMatch() {
    let blockSize = 4
    let manager = PagedCacheManager(blockSize: blockSize, maxBlocks: 20)

    // Store tokens [1,2,3,4, 5,6,7,8]
    let tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    let dummyLayer: [(keys: MLXArray, values: MLXArray)] = [
        (keys: MLXArray.zeros([1, 1, blockSize, 8]), values: MLXArray.zeros([1, 1, blockSize, 8]))
    ]
    let layerData = [dummyLayer, dummyLayer]
    manager.storeTokenSequence(tokens: tokens, layerData: layerData)

    // Query with same first block but different second block
    let query = [1, 2, 3, 4, 99, 98, 97, 96, 10, 11, 12, 13]
    let result = manager.fetchPrefix(tokens: query)

    #expect(result != nil)
    #expect(result!.matchedTokens == 4, "Only the first block should match")
    #expect(result!.remainingTokens == [99, 98, 97, 96, 10, 11, 12, 13])
    #expect(result!.blocks.count == 1)
}

@Test func pagedCacheReusesStoredPartialLeafForGrowingPrompt() throws {
    let blockSize = 4
    let manager = PagedCacheManager(blockSize: blockSize, maxBlocks: 20)
    let fullLayer: [(keys: MLXArray, values: MLXArray)] = [
        (keys: MLXArray.zeros([1, 1, 4, 8]),
         values: MLXArray.zeros([1, 1, 4, 8]))
    ]
    let partialLayer: [(keys: MLXArray, values: MLXArray)] = [
        (keys: MLXArray.zeros([1, 1, 2, 8]),
         values: MLXArray.zeros([1, 1, 2, 8]))
    ]

    manager.storeTokenSequence(
        tokens: [1, 2, 3, 4, 5, 6],
        layerData: [fullLayer, partialLayer])

    let result = try #require(manager.fetchPrefix(
        tokens: [1, 2, 3, 4, 5, 6, 7, 8]))
    #expect(result.matchedTokens == 6)
    #expect(result.remainingTokens == [7, 8])
    #expect(result.blocks.count == 2)
    manager.freeBlock(result.blocks[1])
    manager.freeBlock(result.blocks[0])
}

@Test func pagedCacheMiss() {
    let manager = PagedCacheManager(blockSize: 4, maxBlocks: 20)

    // Store some tokens
    let tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    let dummyLayer: [(keys: MLXArray, values: MLXArray)] = [
        (keys: MLXArray.zeros([1, 1, 4, 8]), values: MLXArray.zeros([1, 1, 4, 8]))
    ]
    manager.storeTokenSequence(tokens: tokens, layerData: [dummyLayer, dummyLayer])

    // Query with completely different tokens — no prefix match
    let query = [99, 98, 97, 96, 95, 94, 93, 92]
    let result = manager.fetchPrefix(tokens: query)

    #expect(result == nil, "Completely different tokens should produce no match")
    #expect(manager.stats.cacheMisses > 0)
}

@Test func pagedCacheTinyPoolKeepsLongestUsablePrefixInsteadOfOrphanLeaf() {
    let blockSize = 2
    let manager = PagedCacheManager(blockSize: blockSize, maxBlocks: 2)
    let dummyLayer: [(keys: MLXArray, values: MLXArray)] = [
        (keys: MLXArray.zeros([1, 1, blockSize, 8]),
         values: MLXArray.zeros([1, 1, blockSize, 8]))
    ]

    manager.storeTokenSequence(
        tokens: [1, 2, 3, 4, 5, 6],
        layerData: [dummyLayer, dummyLayer, dummyLayer])

    let result = manager.fetchPrefix(tokens: [1, 2, 3, 4, 5, 6, 7])
    #expect(result?.matchedTokens == blockSize)
    #expect(result?.remainingTokens == [3, 4, 5, 6, 7])
    #expect(result?.blocks.count == 1)
}

@Test func pagedCacheCoordinatorReleasesLeavesBeforeRootsUnderEviction() throws {
    let blockSize = 2
    let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
        usePagedCache: true,
        enableDiskCache: false,
        pagedBlockSize: blockSize,
        maxCacheBlocks: 4,
        modelKey: "leaf-first-release"))
    let manager = try #require(coordinator.pagedCache)
    let dummyLayer: [(keys: MLXArray, values: MLXArray)] = [
        (keys: MLXArray.zeros([1, 1, blockSize, 8]),
         values: MLXArray.zeros([1, 1, blockSize, 8]))
    ]
    let original = [1, 2, 3, 4, 5, 6]

    manager.storeTokenSequence(
        tokens: original,
        layerData: [dummyLayer, dummyLayer, dummyLayer])
    let pinned = try #require(manager.fetchPrefix(tokens: original))
    coordinator.release(blocks: pinned.blocks)

    manager.storeTokenSequence(tokens: [9, 10], layerData: [dummyLayer])

    let retained = manager.fetchPrefix(tokens: original)
    #expect(retained?.matchedTokens == 4)
    #expect(retained?.remainingTokens == [5, 6])
}
