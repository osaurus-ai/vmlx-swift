// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// In-place speculative checkpoints for a target cache list, and the draft-window hidden tail that a
// DeepSeek-V4.1 cache carries.
//
// A verify forward appends rows the speculation may reject. The native-MTP path undid that by
// replacing every cache object with a deep copy (`cache.map { $0.copy() }`), which costs a copy of
// every layer per cycle and changes cache identity under anything keyed on it. A checkpoint here is
// a reference snapshot instead (the Python reference's `Session.checkpoint`): each layer saves its
// array references and scalar metadata, and `restore` writes them back into the SAME objects. MLX
// arrays are values, so a later forward cannot change what a saved reference holds: cache writes
// either assign new arrays or slice-update into a buffer that the saved reference keeps from being
// donated. The saved references are separate `MLXArray` wrappers, because a slice update rebinds
// the wrapper object it is applied to.

import Foundation
import MLX

// MARK: - Speculative checkpoint

/// A cache that can take an O(1) reference snapshot and restore it in place.
public protocol SpeculativeCheckpointCache: KVCache {
    /// Snapshot of every field a forward can change. Opaque to callers.
    func speculativeCheckpoint() -> Any
    /// Write a snapshot taken from this same cache back into it. The cache object is unchanged.
    func restoreSpeculativeCheckpoint(_ checkpoint: Any)
    /// False for a subclass whose extra state the snapshot would miss.
    var speculativeCheckpointIsExact: Bool { get }
}

/// One snapshot per layer of a cache list; restores into the same layer objects.
public struct SpeculativeCacheListCheckpoint {
    private let layers: [(cache: any SpeculativeCheckpointCache, state: Any)]

    /// nil when some layer cannot be checkpointed in place.
    public init?(_ cache: [KVCache]) {
        var layers: [(cache: any SpeculativeCheckpointCache, state: Any)] = []
        layers.reserveCapacity(cache.count)
        for layer in cache {
            guard let c = layer as? any SpeculativeCheckpointCache, c.speculativeCheckpointIsExact
            else { return nil }
            layers.append((c, c.speculativeCheckpoint()))
        }
        self.layers = layers
    }

    /// True when every layer of `cache` supports ``SpeculativeCheckpointCache``.
    public static func supports(_ cache: [KVCache]) -> Bool {
        !cache.isEmpty
            && cache.allSatisfy {
                ($0 as? any SpeculativeCheckpointCache)?.speculativeCheckpointIsExact == true
            }
    }

    public func restore() {
        for layer in layers {
            layer.cache.restoreSpeculativeCheckpoint(layer.state)
        }
    }
}

/// A separate wrapper for the same array value (no op, no copy): `x[...]` over the full range is
/// the array itself. Keeps a saved reference out of reach of a slice update that rebinds `x`.
@inline(__always)
func speculativeReference(_ x: MLXArray?) -> MLXArray? {
    x.map { $0[.ellipsis] }
}

extension KVCacheSimple: SpeculativeCheckpointCache {
    private struct Checkpoint {
        let keys: MLXArray?
        let values: MLXArray?
        let offset: Int
        let step: Int
    }

    public var speculativeCheckpointIsExact: Bool { type(of: self) == KVCacheSimple.self }

    public func speculativeCheckpoint() -> Any {
        Checkpoint(
            keys: speculativeReference(keys), values: speculativeReference(values),
            offset: offset, step: step)
    }

    public func restoreSpeculativeCheckpoint(_ checkpoint: Any) {
        guard let c = checkpoint as? Checkpoint else {
            preconditionFailure("KVCacheSimple: foreign speculative checkpoint")
        }
        keys = c.keys
        values = c.values
        offset = c.offset
        step = c.step
    }
}

extension RotatingKVCache: SpeculativeCheckpointCache {
    struct Checkpoint {
        let keys: MLXArray?
        let values: MLXArray?
        let offset: Int
        let idx: Int
        let keep: Int
        let maxCacheSize: Int
        let step: Int
    }

    public var speculativeCheckpointIsExact: Bool { type(of: self) == RotatingKVCache.self }

    public func speculativeCheckpoint() -> Any {
        Checkpoint(
            keys: speculativeReference(keys), values: speculativeReference(values),
            offset: offset, idx: idx, keep: keep, maxCacheSize: maxCacheSize, step: step)
    }

    public func restoreSpeculativeCheckpoint(_ checkpoint: Any) {
        guard let c = checkpoint as? Checkpoint else {
            preconditionFailure("RotatingKVCache: foreign speculative checkpoint")
        }
        keys = c.keys
        values = c.values
        offset = c.offset
        idx = c.idx
        keep = c.keep
        maxCacheSize = c.maxCacheSize
        step = c.step
    }
}

// MARK: - Draft-window hidden tail

/// The hidden rows of the last `capacity` positions of a sequence, kept with its cache
/// (``HybridPoolCache/auxiliaryHiddenTail``). DeepSeek-V4.1 records the dSpark target-layer hidden
/// here on every forward, so the draft windows can be rebuilt from the cache alone: after an
/// in-place restore, after a prefix-cache hit (the Python reference's `dspark_tail`), or after a
/// stretch of plain decode steps.
///
/// Rows are kept as the chunks the forwards produced (no per-token concatenation). Rows at or after
/// a new record's start position are replaced, so a rejected verify tail never survives a later
/// forward at the same positions; `truncate` follows cache trims.
public struct HybridPoolHiddenTail {
    public struct Chunk {
        public let start: Int
        /// `[n, width]`, consecutive positions starting at `start`.
        public let rows: MLXArray
        public var end: Int { start + rows.dim(0) }
    }

    public let capacity: Int
    public private(set) var chunks: [Chunk] = []

    /// Chunks beyond this count are joined into one (bounds the graph a later read builds).
    static let maxChunks = 48

    public init(capacity: Int) {
        precondition(capacity > 0, "hidden tail capacity must be positive")
        self.capacity = capacity
    }

    public init(capacity: Int, start: Int, rows: MLXArray) {
        self.init(capacity: capacity)
        record(rows: rows, startPosition: start)
    }

    /// One past the last position held (0 when empty).
    public var endPosition: Int { chunks.last?.end ?? 0 }
    /// First position held (== `endPosition` when empty).
    public var startPosition: Int { chunks.first?.start ?? endPosition }
    public var count: Int { endPosition - startPosition }
    public var isEmpty: Bool { chunks.isEmpty }

    /// Record the rows of positions `startPosition ..< startPosition + rows.dim(0)`. Anything held at
    /// or after `startPosition` is dropped first; a record that leaves a gap after the held rows, or
    /// rewrites positions before them, starts over.
    public mutating func record(rows: MLXArray, startPosition start: Int) {
        var rows = rows
        var start = start
        let n = rows.dim(0)
        guard n > 0 else { return }
        if n > capacity {
            rows = rows[(n - capacity)...]
            start += n - capacity
        }
        if !chunks.isEmpty, start > endPosition || start < startPosition {
            chunks.removeAll()
        } else {
            truncate(toEnd: start)
        }
        chunks.append(Chunk(start: start, rows: rows))
        prune()
    }

    /// Keep only positions before `end`.
    public mutating func truncate(toEnd end: Int) {
        while let last = chunks.last, last.start >= end {
            chunks.removeLast()
        }
        if let last = chunks.last, last.end > end {
            chunks[chunks.count - 1] = Chunk(
                start: last.start, rows: last.rows[0 ..< (end - last.start)])
        }
    }

    /// Rows of positions `lo ..< hi`, or nil when the tail does not hold all of them.
    public func rows(from lo: Int, to hi: Int) -> MLXArray? {
        guard lo < hi, lo >= startPosition, hi <= endPosition else { return nil }
        var pieces: [MLXArray] = []
        for chunk in chunks where chunk.end > lo && chunk.start < hi {
            let a = Swift.max(lo, chunk.start) - chunk.start
            let b = Swift.min(hi, chunk.end) - chunk.start
            pieces.append(a == 0 && b == chunk.rows.dim(0) ? chunk.rows : chunk.rows[a ..< b])
        }
        return pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 0)
    }

    /// Every held row as one array (for persistence), with its first position.
    public func joined() -> (start: Int, rows: MLXArray)? {
        guard !chunks.isEmpty, let rows = rows(from: startPosition, to: endPosition) else {
            return nil
        }
        return (startPosition, rows)
    }

    private mutating func prune() {
        let floor = endPosition - capacity
        while let first = chunks.first, first.end <= floor {
            chunks.removeFirst()
        }
        if chunks.count > Self.maxChunks, let all = joined() {
            let lo = Swift.max(all.start, floor)
            chunks = [Chunk(start: lo, rows: all.rows[(lo - all.start)...])]
        }
    }
}
