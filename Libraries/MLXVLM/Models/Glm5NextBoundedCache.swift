import Foundation
import MLX
import MLXLMCommon

/// A prefix and recent suffix, or three spans in physical ring order. Host work stays constant per append;
/// slot lookup and position vectors are device arithmetic, without readbacks.
struct Glm5NextPositions {
    var ranges: [Range<Int>] = []
    init(_ ranges: [Range<Int>] = []) {
        for range in ranges where !range.isEmpty {
            if let last = self.ranges.last, range.lowerBound >= last.lowerBound, range.lowerBound <= last.upperBound {
                self.ranges[self.ranges.count - 1] = last.lowerBound..<max(last.upperBound, range.upperBound)
            } else { self.ranges.append(range) }
        }
    }
    var count: Int { ranges.reduce(0) { $0 + $1.count } }
    var array: MLXArray {
        let arrays = ranges.map { arange($0.lowerBound, $0.upperBound, dtype: .int32) }
        return arrays.count == 1 ? arrays[0] : concatenated(arrays, axis: 0)
    }
    func indices(of absolute: MLXArray) -> MLXArray {
        var result = MLXArray.full(absolute.shape, values: MLXArray(Int32(-1)))
        var base = 0
        for range in ranges {
            let valid = (absolute .>= MLXArray(Int32(range.lowerBound))) .&& (absolute .< MLXArray(Int32(range.upperBound)))
            result = MLX.where(valid, absolute - MLXArray(Int32(range.lowerBound - base)), result)
            base += range.count
        }
        return result
    }
}

/// Chronological bounded storage for GLM's KV and raw indexer rows. Indexer pools
/// retain their original absolute membership, including evicted boundary members;
/// this costs at most two partial pools beyond the KV window, never full history.
final class Glm5NextBoundedCache {
    let capacity: Int
    let keep: Int
    let poolSize: Int
    private(set) var offset = 0
    private var kv: RotatingKVCache
    private(set) var packed: MLXArray?
    private(set) var packedPositions = Glm5NextPositions()
    private(set) var positions = Glm5NextPositions()
    private(set) var attentionPositions = Glm5NextPositions()
    private(set) var attentionStoragePositions = Glm5NextPositions()
    private var pendingCount = 0

    init(capacity: Int, poolSize: Int) {
        precondition(poolSize > 0)
        self.capacity = capacity
        // Always reserve a slot for the newest token, including very small caps.
        self.keep = capacity > 0 ? min(4, capacity - 1) : 0
        self.poolSize = poolSize
        self.kv = RotatingKVCache(maxSize: max(1, capacity), keep: self.keep)
    }

    func retainedPositions(end: Int) -> Glm5NextPositions {
        let prefix = min(keep, end)
        let start = max(prefix, end - (capacity - keep))
        return Glm5NextPositions([0..<prefix, start..<end])
    }

    func plannedPositions(count: Int) -> Glm5NextPositions {
        // Union of the per-query windows, bounded by capacity + count - 1.
        let previous = retainedPositions(end: offset + 1).ranges.map { $0.lowerBound..<min($0.upperBound, offset) }
        return Glm5NextPositions(previous + [offset..<(offset + count)])
    }

    private func poolPositions(for positions: Glm5NextPositions, end: Int) -> Glm5NextPositions {
        Glm5NextPositions(positions.ranges.map {
            ($0.lowerBound / poolSize * poolSize)..<min((($0.upperBound + poolSize - 1) / poolSize) * poolSize, end)
        })
    }

    private func rowIndices(wanted: Glm5NextPositions, available: Glm5NextPositions) -> MLXArray {
        // Coverage is proven from two host ranges, never by evaluating an MLX value.
        precondition(wanted.ranges.allSatisfy { w in available.ranges.contains { $0.lowerBound <= w.lowerBound && $0.upperBound >= w.upperBound } },
            "GLM bounded companion history is incomplete")
        return available.indices(of: wanted.array)
    }

    private func selectRows(_ array: MLXArray, wanted: Glm5NextPositions, available: Glm5NextPositions, axis: Int) -> MLXArray {
        if wanted.ranges == available.ranges { return array }
        return take(array, rowIndices(wanted: wanted, available: available), axis: axis)
    }

    func updateIndexer(_ new: MLXArray) -> MLXArray {
        precondition(pendingCount == 0, "GLM indexer append must be committed by KV append")
        let count = new.dim(1)
        attentionPositions = plannedPositions(count: count)
        attentionStoragePositions = attentionPositions
        let wanted = poolPositions(for: attentionPositions, end: offset + count)
        // Slice surviving old spans BEFORE concat. Gathering the concatenated
        // payload copied the entire companion twice at each pool eviction.
        var pieces: [MLXArray] = []
        if let packed {
            for range in wanted.ranges where range.lowerBound < offset {
                let old = range.lowerBound..<min(range.upperBound, offset)
                var base = 0
                var found = false
                for available in packedPositions.ranges {
                    if available.lowerBound <= old.lowerBound && available.upperBound >= old.upperBound {
                        let start = base + old.lowerBound - available.lowerBound
                        pieces.append(packed[0..., start..<(start + old.count), 0...])
                        found = true
                        break
                    }
                    base += available.count
                }
                precondition(found, "GLM bounded companion history is incomplete")
            }
        }
        pieces.append(new)
        packed = pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 1)
        packedPositions = wanted
        pendingCount = count
        return packed!
    }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let count = newKeys.dim(2)
        precondition(count == pendingCount, "GLM KV/indexer append lengths disagree")
        let (attentionKeys, attentionValues) = kv.update(keys: newKeys, values: newValues)
        offset += count
        positions = retainedPositions(end: offset)
        if count == 1, offset > capacity {
            // The rotating cache returns physical ring order for decode. Keep
            // the indexer's chronological layout, and translate its selected
            // slots into this order instead of copying the latent KV payload.
            let next = Int(kv.metaState[4])!
            let recentStart = offset - (capacity - keep)
            let split = recentStart + capacity - next
            attentionStoragePositions = Glm5NextPositions([
                0..<keep, split..<offset, recentStart..<split,
            ])
        }
        if count > 1, attentionKeys.dim(2) > capacity {
            // A chunk needs the union of its windows during attention. Commit
            // only the final window, so dormant cache state respects the cap.
            let k = selectRows(attentionKeys, wanted: positions, available: attentionPositions, axis: 2)
            let v = selectRows(attentionValues, wanted: positions, available: attentionPositions, axis: 2)
            kv = RotatingKVCache(maxSize: capacity, keep: keep)
            kv.state = [k, v]
            kv.metaState = [String(keep), String(capacity), "256", String(offset), String(positions.count)]
        }
        let wantedPacked = poolPositions(for: positions, end: offset)
        packed = selectRows(packed!, wanted: wantedPacked, available: packedPositions, axis: 1)
        packedPositions = wantedPacked
        pendingCount = 0
        return (attentionKeys, attentionValues)
    }

    func storageIndices(_ selected: MLXArray) -> MLXArray {
        if attentionPositions.ranges == attentionStoragePositions.ranges { return selected }
        let safe = maximum(selected, MLXArray(Int32(0)))
        let absolute = take(attentionPositions.array, safe, axis: 0)
        return MLX.where(selected .>= MLXArray(Int32(0)), attentionStoragePositions.indices(of: absolute), MLXArray(Int32(-1)))
    }

    func visibility(positions: Glm5NextPositions, queryOffset: Int, count: Int) -> MLXArray {
        let key = positions.array[.newAxis, .newAxis, 0...]
        let query = arange(queryOffset, queryOffset + count, dtype: .int32)[.newAxis, 0..., .newAxis]
        return (key .<= query) .&& ((key .< MLXArray(Int32(keep))) .|| (key .> query - MLXArray(Int32(capacity - keep))))
    }

    var innerState: [MLXArray] { kv.innerState() + [packed].compactMap { $0 } }
    var state: [MLXArray] {
        guard let (keys, values) = kv.temporallyOrderedKV() else { return [] }
        return [keys, values] + [packed].compactMap { $0 }
    }
    var metadata: [String] { ["glm5-bounded-v1", String(capacity), String(keep), String(poolSize), String(offset)] }
    var isTrimmable: Bool { offset <= capacity }

    func restore(_ arrays: [MLXArray], metadata: [String], expectedOffset: Int) -> Bool {
        guard capacity > 0, metadata.count == 5, metadata[0] == "glm5-bounded-v1",
            Int(metadata[1]) == capacity, Int(metadata[2]) == keep,
            Int(metadata[3]) == poolSize, Int(metadata[4]) == expectedOffset,
            expectedOffset > 0, arrays.count == 3 else { return false }
        let k = arrays[0], v = arrays[1], p = arrays[2]
        let pos = retainedPositions(end: expectedOffset)
        let ipos = poolPositions(for: pos, end: expectedOffset)
        guard k.ndim == 4, v.ndim == 4, p.ndim == 3,
            k.dim(0) == 1, v.dim(0) == 1, p.dim(0) == 1,
            k.dim(1) > 0, v.dim(1) == k.dim(1), k.dim(3) > 0, v.dim(3) > 0,
            k.dim(2) == pos.count, v.dim(2) == pos.count, p.dim(1) == ipos.count,
            p.dim(2) >= 3, p.dim(2) % 2 == 1,
            k.dtype.isFloatingPoint, k.dtype == v.dtype, k.dtype == p.dtype else { return false }
        kv = RotatingKVCache(maxSize: capacity, keep: keep)
        kv.state = [k, v]
        kv.metaState = [String(keep), String(capacity), "256", String(expectedOffset), String(pos.count)]
        packed = p
        offset = expectedOffset; positions = pos; packedPositions = ipos
        attentionPositions = pos; attentionStoragePositions = pos; pendingCount = 0
        return true
    }

    func reset() {
        offset = 0; packed = nil
        kv = RotatingKVCache(maxSize: max(1, capacity), keep: keep)
        positions = Glm5NextPositions(); packedPositions = Glm5NextPositions()
        attentionPositions = Glm5NextPositions(); attentionStoragePositions = Glm5NextPositions(); pendingCount = 0
    }

    func trim(_ n: Int) -> Int {
        guard isTrimmable, pendingCount == 0 else { return 0 }
        let trimmed = min(max(n, 0), offset)
        let end = offset - trimmed
        if end == 0 { reset(); return trimmed }
        _ = kv.trim(trimmed)
        packed = packed?[0..., ..<end, 0...]
        offset = end; positions = Glm5NextPositions([0..<end]); packedPositions = positions
        attentionPositions = positions; attentionStoragePositions = positions
        return trimmed
    }

    func copy() -> Glm5NextBoundedCache {
        precondition(pendingCount == 0)
        let result = Glm5NextBoundedCache(capacity: capacity, poolSize: poolSize)
        if offset > 0 {
            precondition(result.restore(state.map { $0 * 1 }, metadata: metadata, expectedOffset: offset))
        }
        return result
    }
}
