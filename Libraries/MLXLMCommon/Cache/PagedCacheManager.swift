// Copyright © 2025 Apple Inc. All rights reserved.

import Foundation
import MLX
#if canImport(os)
    import os
#endif

// MARK: - CacheStats

/// Statistics for cache utilisation and hit/miss tracking.
public struct CacheStats: Sendable {
    public var totalBlocks: Int = 0
    public var allocatedBlocks: Int = 0
    public var freeBlocks: Int = 0
    public var cacheHits: Int = 0
    public var cacheMisses: Int = 0
    public var evictions: Int = 0
    /// Conservative retained storage charge including compact-buffer slack.
    public var residentBytes: Int = 0
    public var companionBytes: Int = 0
    public var maxResidentBytes: Int? = nil
}

// MARK: - PrefixFetchResult

/// The result of a prefix-match query against the paged cache.
public struct PrefixFetchResult: Sendable {
    /// Number of tokens that were matched from the cache.
    public let matchedTokens: Int
    /// Tokens that still need to be computed.
    public let remainingTokens: [Int]
    /// Cache blocks covering the matched prefix, in order.
    public let blocks: [CacheBlock]
}

// MARK: - PagedCacheManager

/// Manages a fixed-size pool of ``CacheBlock`` instances with hash-based
/// prefix matching for KV cache reuse.
///
/// Thread safety is provided by an `OSAllocatedUnfairLock`. All public
/// methods acquire the lock before accessing shared state.
public final class PagedCacheManager: @unchecked Sendable {

    // MARK: - Properties

    /// Number of tokens per block.
    public let blockSize: Int

    /// Total number of blocks in the pool (including the null sentinel at index 0).
    public let maxBlocks: Int

    /// Optional retained KV/companion storage budget, not an allocator limit.
    public let maxBytes: Int?

    private let retainedAccount: RetainedCacheBudget.Account
    // Charges follow manager admissions, not mutable public CacheBlock fields.
    private var storageCharges: [Int: (kv: Int, companion: Int)] = [:]

    /// Model key for cache isolation (prevents cross-model hash collisions).
    public let modelKey: String?

    /// Lock for thread safety.
    private let lock = OSAllocatedUnfairLock()

    /// Pre-allocated block pool. Index 0 is the null sentinel.
    private var blocks: [CacheBlock]

    /// Free block queue (LRU eviction order).
    private let freeQueue = FreeBlockQueue()

    /// Hash-to-block mapping for prefix lookup.
    private let hashMap = BlockHashMap()

    /// Cache statistics.
    public private(set) var stats = CacheStats()

    /// Thread-safe copy of current cache statistics.
    public func snapshotStats() -> CacheStats {
        lock.lock()
        defer { lock.unlock() }
        return stats
    }

    // MARK: - Initialization

    /// Create a paged cache manager.
    ///
    /// - Parameters:
    ///   - blockSize: Number of tokens per block (default 64).
    ///   - maxBlocks: Total pool size including the null sentinel (default 1000).
    public convenience init(
        blockSize: Int = 64, maxBlocks: Int = 1000, maxBytes: Int? = nil,
        modelKey: String? = nil
    ) {
        self.init(blockSize: blockSize, maxBlocks: maxBlocks, modelKey: modelKey,
                  retainedBudget: RetainedCacheBudget(limit: maxBytes))
    }

    init(
        blockSize: Int, maxBlocks: Int, modelKey: String?,
        retainedBudget: RetainedCacheBudget
    ) {
        self.blockSize = blockSize
        self.maxBlocks = maxBlocks
        self.maxBytes = retainedBudget.limit
        self.retainedAccount = retainedBudget.makeAccount()
        self.modelKey = modelKey

        // Pre-allocate all blocks.
        self.blocks = (0..<maxBlocks).map { CacheBlock(blockId: $0, blockSize: blockSize) }

        // Block 0 is the null sentinel — never allocate it.
        // Blocks 1..<maxBlocks go into the free queue.
        for i in 1..<maxBlocks {
            freeQueue.append(blocks[i])
        }

        stats.maxResidentBytes = maxBytes
        stats.totalBlocks = maxBlocks
        stats.freeBlocks = maxBlocks - 1
    }

    // MARK: - Core Methods

    /// Allocate a block from the free queue.
    ///
    /// - Returns: A freshly-reset ``CacheBlock`` with `refCount == 1`,
    ///   or `nil` if no blocks are available.
    public func allocateBlock() -> CacheBlock? {
        lock.lock()
        defer { lock.unlock() }
        return _allocateBlock()
    }

    /// Internal allocate (caller must hold the lock).
    ///
    /// Cached blocks live in the free queue at `refCount == 0` after
    /// `storeTokenSequence` registers them. If allocation reuses one,
    /// remove its old hash before reset so future prefix probes do not
    /// return stale tensors.
    private func _allocateBlock() -> CacheBlock? {
        guard let block = freeQueue.popFirst() else { return nil }
        if block.blockHash != nil {
            hashMap.remove(block)
            stats.evictions += 1
        }
        removeStorageCharge(block)
        block.reset()
        block.incrementRef()
        stats.allocatedBlocks += 1
        stats.freeBlocks -= 1
        return block
    }

    // MLX contiguous may share an already-contiguous source only when its
    // backing allocation exceeds logical bytes by at most 16 KiB (the pinned
    // backend's compact-view allowance). Charge that slack per tensor so views
    // cannot hide larger source allocations behind tiny logical slices.
    private static func storageCharge(_ arrays: [MLXArray]) -> Int {
        RetainedCacheBudget.storageCharge(arrays)
    }

    private static func kvArrays(_ block: CacheBlock) -> [MLXArray] {
        (block.cacheData ?? []).compactMap { $0 }.flatMap { [$0.keys, $0.values] }
    }

    private func removeStorageCharge(_ block: CacheBlock) {
        guard let charge = storageCharges.removeValue(forKey: block.blockId) else { return }
        let bytes = charge.kv + charge.companion
        retainedAccount.release(bytes)
        stats.residentBytes -= bytes
        stats.companionBytes -= charge.companion
    }

    /// Make room using only unpinned LRU payloads. An admitted fetch owns its
    /// blocks until release, so a concurrent store must never recycle them.
    private func reserveStorage(_ bytes: Int) -> Bool {
        if let maxBytes, bytes > maxBytes { return false }
        while !retainedAccount.reserve(bytes) {
            guard let victim = freeQueue.popFirst(where: {
                $0.refCount == 0 && ($0.cacheData != nil || $0.boundaryCompanionData != nil)
            }) else { return false }
            hashMap.remove(victim)
            removeStorageCharge(victim)
            victim.reset()
            freeQueue.append(victim)
            stats.evictions += 1
        }
        return true
    }

    private static func compact(_ array: MLXArray) -> MLXArray {
        contiguous(array)
    }

    /// Release a block back to the pool.
    ///
    /// Decrements the reference count; when it reaches zero the block is
    /// removed from the hash map, reset, and returned to the free queue.
    ///
    /// - Parameter block: The block to free.
    public func freeBlock(_ block: CacheBlock) {
        lock.lock()
        defer { lock.unlock() }
        _freeBlock(block)
    }

    /// Internal free (caller must hold the lock).
    private func _freeBlock(_ block: CacheBlock) {
        let wasReferenced = block.refCount > 0
        block.decrementRef()
        if block.refCount == 0 {
            if block.blockHash == nil || block.cacheData == nil || block.tokenIds.isEmpty {
                hashMap.remove(block)
                removeStorageCharge(block)
                block.reset()
            }
            freeQueue.append(block)
            if wasReferenced {
                stats.allocatedBlocks -= 1
                stats.freeBlocks += 1
            }
        }
    }

    /// Look up a cached block by its content hash.
    ///
    /// Updates hit/miss statistics.
    ///
    /// - Parameter hash: The chain hash to look up.
    /// - Returns: The matching ``CacheBlock``, or `nil` on a miss.
    public func findCachedBlock(hash: String) -> CacheBlock? {
        lock.lock()
        defer { lock.unlock() }
        return _findCachedBlock(hash: hash)
    }

    /// Internal find (caller must hold the lock).
    private func _findCachedBlock(hash: String) -> CacheBlock? {
        if let block = hashMap.find(hash: hash) {
            stats.cacheHits += 1
            return block
        }
        stats.cacheMisses += 1
        return nil
    }

    /// Register a block in the hash map under the given hash.
    ///
    /// - Parameters:
    ///   - block: The block to register.
    ///   - hash: The chain hash to use as the key.
    public func registerBlock(_ block: CacheBlock, hash: String) {
        lock.lock()
        defer { lock.unlock() }
        block.blockHash = hash
        hashMap.insert(block)
    }

    /// Reset the manager to its initial state, freeing all blocks.
    public func clear() {
        lock.lock()
        defer { lock.unlock() }

        hashMap.removeAll()

        // Drain the free queue so we can rebuild it cleanly.
        while freeQueue.popFirst() != nil {}

        for i in 0..<maxBlocks {
            blocks[i].reset()
        }
        storageCharges.removeAll()
        retainedAccount.releaseAll()

        // Re-enqueue blocks 1..<maxBlocks.
        for i in 1..<maxBlocks {
            freeQueue.append(blocks[i])
        }

        stats = CacheStats()
        stats.maxResidentBytes = maxBytes
        stats.totalBlocks = maxBlocks
        stats.freeBlocks = maxBlocks - 1
    }

    // MARK: - Prefix Matching

    /// Store a token sequence as a chain of cache blocks.
    ///
    /// Tokens are split into ``blockSize``-sized chunks. Each chunk's chain
    /// hash incorporates the previous block's hash so that identical prefixes
    /// always produce the same hash chain. Chunks whose hash already exists in
    /// the cache are skipped (prefix sharing).
    ///
    /// - Parameters:
    ///   - tokens: The full token sequence.
    ///   - layerData: Per-chunk, per-layer KV tensors. `layerData[chunkIndex]`
    ///     contains one `(keys, values)` tuple per transformer layer.
    public func storeTokenSequence(
        tokens: [Int],
        layerData: [[(keys: MLXArray, values: MLXArray)]],
        boundaryCompanionData: [String: MLXArray]? = nil,
        mediaSalt: String? = nil
    ) {
        lock.lock()
        defer { lock.unlock() }

        // Keep the whole chain pinned until the store is complete. Releasing
        // each block immediately lets a small/pressured pool recycle a parent
        // while its child is still being built, leaving only an orphaned chain
        // hash that can never satisfy a prefix walk.
        var pinnedBlocks: [CacheBlock] = []
        defer {
            // Leaves are least useful without their parents, so place them at
            // the front of the LRU queue and keep roots newest.
            for block in pinnedBlocks.reversed() {
                _freeBlock(block)
            }
        }

        var parentHash: String? = nil
        var chunkIndex = 0
        var offset = 0

        while offset < tokens.count {
            let end = min(offset + blockSize, tokens.count)
            let chunk = Array(tokens[offset..<end])
            let hash = CacheBlock.computeBlockHash(
                parentHash: parentHash, tokenIds: chunk,
                modelKey: modelKey, mediaSalt: mediaSalt)

            // Reuse and pin an existing chain member. It must leave the free
            // queue while descendants are stored, otherwise allocation could
            // reset it before the chain is complete.
            if let existing = hashMap.find(hash: hash) {
                if existing.refCount == 0 {
                    _ = freeQueue.remove(existing)
                    stats.allocatedBlocks += 1
                    stats.freeBlocks -= 1
                }
                existing.incrementRef()
                pinnedBlocks.append(existing)
                parentHash = hash
                offset = end
                chunkIndex += 1
                continue
            }

            // Do not register empty payload blocks. Some cache topologies
            // are represented by another tier (for example hybrid-pool
            // DSV4 state); registering a token-only block would create a
            // false paged hit that restores no cache data.
            guard chunkIndex < layerData.count, !layerData[chunkIndex].isEmpty else {
                break
            }

            let incoming = layerData[chunkIndex]
            let bytes = Self.storageCharge(incoming.flatMap { [$0.keys, $0.values] })
            guard reserveStorage(bytes) else { break }
            guard let block = _allocateBlock() else {
                retainedAccount.release(bytes)
                break
            }

            block.tokenIds = chunk
            block.cacheData = incoming.map {
                Optional((keys: Self.compact($0.keys), values: Self.compact($0.values)))
            }
            // Batch-evaluate one block to sever lazy source dependencies
            // without one GPU synchronization per individual tensor.
            eval(Self.kvArrays(block))
            storageCharges[block.blockId] = (kv: bytes, companion: 0)
            stats.residentBytes += bytes

            block.blockHash = hash
            hashMap.insert(block)
            pinnedBlocks.append(block)

            parentHash = hash
            offset = end
            chunkIndex += 1
        }

        // A typed companion describes the complete prompt boundary, not an
        // independently reusable token block. Attach it only when the entire
        // sequence was admitted. If the fixed pool ran out part-way through,
        // the partial chain remains useful to ordinary KV models but must not
        // be advertised as a complete mixed rotating/KV boundary.
        if offset == tokens.count, let leaf = pinnedBlocks.last {
            let previous = storageCharges[leaf.blockId]?.companion ?? 0
            let next = Self.storageCharge(Array((boundaryCompanionData ?? [:]).values))
            if reserveStorage(max(0, next - previous)) {
                leaf.boundaryCompanionData = boundaryCompanionData?.mapValues(Self.compact)
                eval(Array((leaf.boundaryCompanionData ?? [:]).values))
                storageCharges[leaf.blockId] = (
                    kv: storageCharges[leaf.blockId]?.kv ?? 0, companion: next)
                stats.residentBytes += next - previous
                stats.companionBytes += next - previous
                if next < previous { retainedAccount.release(previous - next) }
            }
        }
    }

    /// Fetch the longest cached prefix for a token sequence.
    ///
    /// Walks the tokens in ``blockSize``-sized chunks, computing chain hashes
    /// and looking up each in the hash map. Stops at the first miss.
    ///
    /// - Parameter tokens: The full token sequence to match.
    /// - Returns: A ``PrefixFetchResult`` describing how many tokens matched
    ///   and which blocks hold the cached data, or `nil` if no prefix matches.
    public func fetchPrefix(tokens: [Int], mediaSalt: String? = nil) -> PrefixFetchResult? {
        lock.lock()
        defer { lock.unlock() }

        var parentHash: String? = nil
        var matchedBlocks: [CacheBlock] = []
        var offset = 0

        while offset < tokens.count {
            let end = min(offset + blockSize, tokens.count)
            let chunk = Array(tokens[offset..<end])
            let hash = CacheBlock.computeBlockHash(
                parentHash: parentHash, tokenIds: chunk,
                modelKey: modelKey, mediaSalt: mediaSalt)
            let exact = hashMap.find(hash: hash)
            let block = exact ?? hashMap.findLongestPartialPrefix(
                parentHash: parentHash,
                tokenIds: chunk,
                modelKey: modelKey,
                mediaSalt: mediaSalt)

            if let block {
                stats.cacheHits += 1
                // Pin the cached-free block while the caller restores KV.
                // If it remains in the free queue, a concurrent allocator
                // could pop and reset it while restore reads `cacheData`.
                if block.refCount == 0 {
                    _ = freeQueue.remove(block)
                    stats.allocatedBlocks += 1
                    stats.freeBlocks -= 1
                }
                block.incrementRef()
                matchedBlocks.append(block)
                parentHash = block.blockHash
                offset += block.tokenCount
                // A stored partial leaf is a complete cache boundary. It can
                // seed the remaining prompt, but no later fixed-size block can
                // be chained after it because block partitioning restarts from
                // token zero for every stored sequence.
                if exact == nil { break }
            } else {
                stats.cacheMisses += 1
                break
            }
        }

        guard !matchedBlocks.isEmpty else { return nil }

        let matchedTokens = matchedBlocks.reduce(0) { $0 + $1.tokenCount }
        let remainingTokens = Array(tokens[offset...])

        return PrefixFetchResult(
            matchedTokens: matchedTokens,
            remainingTokens: remainingTokens,
            blocks: matchedBlocks
        )
    }
}
