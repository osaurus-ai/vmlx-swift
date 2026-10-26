// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DSV4 Compressor + Indexer + per-layer DeepseekV4Cache.
//
// These power the "compressed global context" path that augments the
// 128-token local sliding window with pooled summaries of older tokens.
// Every decoder layer with `compress_ratio > 0` (~41 of 43 layers in
// DSV4-Flash) carries a Compressor; layers with `compress_ratio == 4`
// ALSO carry an Indexer that picks the top-k most relevant pooled
// entries per query position.
//
// Reference:
//   - jang-tools/jang_tools/dsv4/mlx_model.py lines 410-489
//   - DSV4-RUNTIME-ARCHITECTURE.md §1 ("Compressor + Indexer")

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - DSV4 pool storage

/// Segmented affine storage for DSV4's compressor/indexer pools.
///
/// The first 64 MiB of each branch stays in its attention-ready dtype. Once a
/// branch crosses that byte threshold, rows are encoded independently with
/// UInt8 codes and fp16 scale/min metadata. Adjacent encoded rows are joined
/// losslessly into bounded 16K-row slabs so near-million-token decode does not
/// traverse thousands of tiny 64-row objects. The full pool is materialized
/// only as a temporary compatibility input; it is never retained beside the
/// encoded representation.
final class DeepseekV4PoolStorage {
    static let segmentRows = 64
    static let slabMaxRows = 16 * 1024
    static let groupSize = 32
    /// DeepSeek-V4.1 4-bit contiguous pools: one fp16 scale/minimum per 64 elements, as the
    /// reference `QuantGrowBuffer` (`mx.quantize`, GROUP_SIZE 64) - 288 B per 512-wide row,
    /// 900 B/token across the four kv-source pools (~0.9 GB at 1M). DSV4-0731 keeps 32.
    static let v41GroupSize = 64
    /// DSV4-0731 pool code width (unchanged default).
    static let bits = 8
    /// Widths the storage (and the disk schema) can encode. 4-bit codes are
    /// packed two per byte (low nibble = even element).
    static let supportedBits: Set<Int> = [4, 8]

    let quantizationEnabled: Bool
    let hotByteLimit: Int
    /// Code width for newly encoded rows. DeepSeek-V4.1 uses 4 (its reference
    /// keeps compressed KV in FP4); DSV4-0731 keeps 8.
    let bits: Int
    /// DeepSeek-V4.1 layout (reference `QuantGrowBuffer`): every row is encoded
    /// at append (no BF16 hot tier) into ONE growable slab written in place, so
    /// the retained pool is the packed 4-bit form at every context length, a
    /// selected-row gather is one device `take` (no host sync), and the disk
    /// serializer sees a single compact segment. DSV4-0731 keeps the segmented
    /// hot-tier layout (`false`).
    let contiguous: Bool
    /// Affine group for newly encoded rows (``v41GroupSize`` for the V4.1 contiguous 4-bit
    /// layout, ``groupSize`` otherwise). Encoded segments carry their own group, so records of
    /// another group still decode; restore re-encodes them once.
    let groupSize: Int
    private(set) var hotPool: MLXArray?
    private(set) var segmentStore: [HybridPoolQuantizedSegment] = []
    private var emptySpec: (shape: [Int], dtype: DType)?
    /// Contiguous layout only: arrays sized to `originalShape[1]` = capacity
    /// rows; rows `0 ..< slabRows` are live. Never handed out directly except
    /// for evaluation (rows are written in place).
    private var slab: HybridPoolQuantizedSegment?
    private var slabRows = 0
    /// Contiguous growth step (reference `QuantGrowBuffer.STEP`).
    static let contiguousGrowRows = 2048

    init(
        quantizationEnabled: Bool, hotByteLimit: Int, bits: Int = DeepseekV4PoolStorage.bits,
        contiguous: Bool = false, groupSize: Int? = nil
    ) {
        precondition(Self.supportedBits.contains(bits), "DSV4 pool bits must be 4 or 8")
        self.quantizationEnabled = quantizationEnabled
        self.hotByteLimit = max(0, hotByteLimit)
        self.bits = bits
        self.contiguous = contiguous && quantizationEnabled
        self.groupSize = groupSize
            ?? ((self.contiguous && bits == 4) ? Self.v41GroupSize : Self.groupSize)
    }

    /// Encoded segments as the disk serializer and inspection code read them.
    /// Contiguous layout: one segment holding exactly the live rows.
    var quantizedSegments: [HybridPoolQuantizedSegment] {
        if contiguous { return liveSlab().map { [$0] } ?? [] }
        return segmentStore
    }

    /// Live rows of the contiguous slab as a fresh segment (new array handles,
    /// so later in-place writes to the slab never alias it).
    private func liveSlab() -> HybridPoolQuantizedSegment? {
        guard let slab, slabRows > 0 else { return nil }
        return Self.sliceRows(slab, from: 0, to: slabRows)
    }

    var rowCount: Int {
        if contiguous { return slabRows }
        if let hotPool { return hotPool.dim(1) }
        return segmentStore.reduce(0) { $0 + $1.rowCount }
    }

    /// Resident bytes. The contiguous slab counts its full capacity (at most
    /// ``contiguousGrowRows`` rows of slack), not just the live rows.
    var retainedByteCount: Int {
        if contiguous { return slab?.retainedByteCount ?? 0 }
        return (hotPool?.nbytes ?? 0)
            + segmentStore.reduce(0) { $0 + $1.retainedByteCount }
    }

    var retainedArrays: [MLXArray] {
        if contiguous { return slab.map { [$0.codes, $0.scales, $0.biases] } ?? [] }
        if let hotPool { return [hotPool] }
        return segmentStore.flatMap { [$0.codes, $0.scales, $0.biases] }
    }

    var featureDim: Int? {
        if contiguous, let slab { return slab.originalShape.last }
        if let hotPool { return hotPool.dim(2) }
        if let first = segmentStore.first { return first.originalShape.last }
        if let emptySpec { return emptySpec.shape.last }
        return nil
    }

    var poolDType: DType? {
        if contiguous, let slab { return slab.originalDType }
        if let hotPool { return hotPool.dtype }
        if let first = segmentStore.first { return first.originalDType }
        if let emptySpec { return emptySpec.dtype }
        return nil
    }

    func materialized() -> MLXArray? {
        if contiguous {
            if let live = liveSlab() { return Self.dequantize(live) }
            if let emptySpec { return MLXArray.zeros(emptySpec.shape, dtype: emptySpec.dtype) }
            return nil
        }
        if let hotPool { return hotPool }
        if segmentStore.isEmpty {
            if let emptySpec {
                return MLXArray.zeros(emptySpec.shape, dtype: emptySpec.dtype)
            }
            return nil
        }
        let parts = segmentStore.map(Self.dequantize)
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
    }

    /// Visit `(globalRowStart, dequantized tile)` pairs under a strict row
    /// bound. Only one bounded tile's graph is handed out per callback so
    /// attention/top-k consumers never materialize the complete pool
    /// (Python mirror: `_StateProxy.iter_dequantized_tiles`). BF16 hot
    /// pools are sliced without any quantization round-trip.
    func forEachDequantizedTile(maxRows: Int, _ body: (Int, MLXArray) -> Void) {
        let maxRows = max(1, maxRows)
        if contiguous {
            guard let slab else { return }
            var start = 0
            while start < slabRows {
                let end = min(start + maxRows, slabRows)
                body(start, Self.dequantize(Self.sliceRows(slab, from: start, to: end)))
                start = end
            }
            return
        }
        if let hotPool {
            let rows = hotPool.dim(1)
            var start = 0
            while start < rows {
                let end = min(start + maxRows, rows)
                body(start, hotPool[0..., start ..< end, 0...])
                start = end
            }
            return
        }

        var cursor = 0
        var pending: [HybridPoolQuantizedSegment] = []
        var pendingStart = 0
        var pendingRows = 0

        func flush() {
            guard !pending.isEmpty else { return }
            let parts = pending.map(Self.dequantize)
            let tile = parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
            pending = []
            pendingRows = 0
            body(pendingStart, tile)
        }

        for segment in segmentStore {
            let segmentRows = segment.rowCount
            var segmentOffset = 0
            while segmentOffset < segmentRows {
                let take = min(maxRows - pendingRows, segmentRows - segmentOffset)
                let piece =
                    (segmentOffset == 0 && take == segmentRows)
                    ? segment
                    : Self.sliceRows(
                        segment, from: segmentOffset, to: segmentOffset + take)
                if pending.isEmpty { pendingStart = cursor + segmentOffset }
                pending.append(piece)
                pendingRows += take
                segmentOffset += take
                if pendingRows == maxRows { flush() }
            }
            cursor += segmentRows
        }
        flush()
    }

    /// Gather CSA-selected rows without decoding a single unselected pool
    /// row. `indices` holds global pool-row indices with shape `(B, ..., K)`;
    /// the result is `(*indices.shape, D)`. Selected indices are already
    /// materialized at the top-k boundary, so one small device-to-host copy
    /// maps each occurrence to its owning q8 segment (Python mirror:
    /// `_StateProxy.gather_dequantized_rows`).
    func gatherDequantizedRows(_ indices: MLXArray) -> MLXArray {
        precondition(
            indices.ndim >= 2,
            "DSV4 pool gather indices must include batch and row axes")
        let outShape = indices.shape
        let width = featureDim ?? 0
        let dtype = poolDType ?? .float32

        if contiguous {
            // One device gather from the slab: no host copy of the indices and
            // no per-segment bookkeeping. Bit-identical to gathering rows of
            // `materialized()` (dequantization is per row and group).
            guard let slab, slabRows > 0, indices.size > 0 else {
                return MLXArray.zeros(outShape + [width], dtype: dtype)
            }
            let B = indices.dim(0)
            let capacity = slab.originalShape[1]
            var flat = indices.reshaped(B, -1).asType(.int32)
            if B > 1 {
                flat = flat + (MLXArray(Int32(0) ..< Int32(B)) * Int32(capacity))
                    .expandedDimensions(axis: 1)
            }
            let linear = flat.reshaped(-1)
            func rows(_ a: MLXArray) -> MLXArray {
                take(a.reshaped([a.dim(0) * a.dim(1)] + Array(a.shape.dropFirst(2))), linear, axis: 0)
            }
            let codes = Self.elementCodes(rows(slab.codes), bits: slab.bits)
            return (codes.asType(.float32) * rows(slab.scales).asType(.float32)
                + rows(slab.biases).asType(.float32))
                .reshaped(outShape + [width])
                .asType(slab.originalDType)
        }

        if let hotPool {
            let B = hotPool.dim(0)
            let D = hotPool.dim(2)
            let flat = indices.reshaped(B, -1).asType(.int32)
            let idxBroad = broadcast(
                flat.expandedDimensions(axis: -1), to: [B, flat.dim(1), D])
            return takeAlong(hotPool, idxBroad, axis: 1)
                .reshaped(outShape + [D])
        }

        let rows = rowCount
        let B = indices.dim(0)
        let hostIndices = indices.asType(.int32).asArray(Int32.self)
        if hostIndices.isEmpty || rows == 0 {
            return MLXArray.zeros(outShape + [width], dtype: dtype)
        }
        let perBatch = hostIndices.count / B

        var segmentEnds: [Int] = []
        var acc = 0
        for segment in segmentStore {
            acc += segment.rowCount
            segmentEnds.append(acc)
        }
        let segmentStarts = [0] + segmentEnds.dropLast()

        var batchBySegment: [Int: [Int32]] = [:]
        var rowsBySegment: [Int: [Int32]] = [:]
        var positionsBySegment: [Int: [Int]] = [:]
        for (position, rawIndex) in hostIndices.enumerated() {
            let global = Int(rawIndex)
            precondition(
                global >= 0 && global < rows,
                "DSV4 pool gather index outside retained rows: \(global) of \(rows)")
            var lo = 0
            var hi = segmentEnds.count - 1
            while lo < hi {
                let mid = (lo + hi) / 2
                if segmentEnds[mid] > global { hi = mid } else { lo = mid + 1 }
            }
            batchBySegment[lo, default: []].append(Int32(position / perBatch))
            rowsBySegment[lo, default: []].append(Int32(global - segmentStarts[lo]))
            positionsBySegment[lo, default: []].append(position)
        }

        var decodedParts: [MLXArray] = []
        var positions: [Int] = []
        for segmentId in batchBySegment.keys.sorted() {
            decodedParts.append(
                Self.dequantizeSelected(
                    segmentStore[segmentId],
                    batchIndices: batchBySegment[segmentId]!,
                    rowIndices: rowsBySegment[segmentId]!))
            positions.append(contentsOf: positionsBySegment[segmentId]!)
        }
        var decoded =
            decodedParts.count == 1
            ? decodedParts[0] : concatenated(decodedParts, axis: 0)
        if positions.count > 1 {
            // positions is a permutation of 0..<N: restore[j] = decoded row
            // holding output occurrence j.
            var restore = [Int32](repeating: 0, count: positions.count)
            for (decodedRow, position) in positions.enumerated() {
                restore[position] = Int32(decodedRow)
            }
            decoded = take(decoded, MLXArray(restore), axis: 0)
        }
        return decoded.reshaped(outShape + [width])
    }

    func replace(with value: MLXArray?) {
        noteDrop(toRows: 0)
        hotPool = nil
        segmentStore = []
        emptySpec = nil
        slab = nil
        slabRows = 0
        guard let value else { return }
        precondition(value.ndim == 3, "DSV4 pooled state must have shape [batch, rows, features]")
        if value.dim(1) == 0 {
            emptySpec = (value.shape, value.dtype)
        } else if contiguous {
            // A BF16 state (older record, generic `state` setter) is encoded
            // exactly as appending its rows would have encoded them.
            slab = Self.quantize(value, bits: bits, groupSize: groupSize)
            slabRows = value.dim(1)
        } else if !quantizationEnabled || value.nbytes <= hotByteLimit {
            hotPool = value
        } else {
            replaceQuantized(with: value)
        }
    }

    func append(_ value: MLXArray) {
        guard value.ndim == 3, value.dim(1) > 0 else { return }
        emptySpec = nil

        if contiguous {
            appendContiguous(Self.quantize(value, bits: bits, groupSize: groupSize))
            return
        }

        if !quantizationEnabled {
            hotPool = hotPool.map { concatenated([$0, value], axis: 1) } ?? value
            return
        }

        if hotPool != nil || segmentStore.isEmpty {
            let combined = hotPool.map { concatenated([$0, value], axis: 1) } ?? value
            if combined.nbytes <= hotByteLimit {
                hotPool = combined
            } else {
                replaceQuantized(with: combined)
            }
            return
        }

        var offset = 0
        let rows = value.dim(1)
        let tail = trailingPartialSegmentRows()
        if tail > 0 {
            let take = min(Self.segmentRows - tail, rows)
            segmentStore.append(
                Self.quantize(value[0..., 0..<take, 0...], bits: bits, groupSize: groupSize))
            offset += take
            compactCompletedTail()
        }

        while rows - offset >= Self.segmentRows {
            let fullRows = ((rows - offset) / Self.segmentRows) * Self.segmentRows
            var take = Self.segmentRows
            while take * 2 <= min(fullRows, Self.slabMaxRows) {
                take *= 2
            }
            segmentStore.append(
                Self.quantize(value[0..., offset..<(offset + take), 0...], bits: bits, groupSize: groupSize))
            offset += take
            compactSlabCascade()
        }
        if offset < rows {
            segmentStore.append(
                Self.quantize(value[0..., offset..<rows, 0...], bits: bits, groupSize: groupSize))
        }
    }

    func trimTrailingRows(_ count: Int) {
        var remaining = max(0, count)
        guard remaining > 0 else { return }
        emptySpec = nil
        if contiguous {
            // Dropped rows stay in the slab as slack and are overwritten by the
            // next append.
            slabRows = max(0, slabRows - remaining)
            noteDrop(toRows: slabRows)
            return
        }
        if let hotPool {
            let keep = max(0, hotPool.dim(1) - remaining)
            self.hotPool = keep == 0 ? nil : hotPool[0..., 0..<keep, 0...]
            return
        }

        while remaining > 0, let last = segmentStore.last {
            if last.rowCount <= remaining {
                segmentStore.removeLast()
                remaining -= last.rowCount
            } else {
                segmentStore[segmentStore.count - 1] = Self.slice(
                    last, keepingRows: last.rowCount - remaining)
                remaining = 0
            }
        }
    }

    func replace(withQuantizedSegments segments: [HybridPoolQuantizedSegment]) {
        hotPool = nil
        emptySpec = nil
        if contiguous {
            noteDrop(toRows: 0)
            slab = nil
            slabRows = 0
            guard let first = segments.first else { return }
            let joinable = segments.allSatisfy { Self.compatible(first, $0) } && first.bits == bits
                && first.groupSize == Self.effectiveGroupSize(features: first.originalShape.last ?? 0,
                                                               groupSize: groupSize)
            let joined: HybridPoolQuantizedSegment
            if joinable {
                joined = segments.count == 1 ? first : Self.concatenate(segments)
            } else {
                // A record of another width or group: re-encode once.
                let parts = segments.map(Self.dequantize)
                joined = Self.quantize(
                    parts.count == 1 ? parts[0] : concatenated(parts, axis: 1), bits: bits,
                    groupSize: groupSize)
            }
            MLX.eval([joined.codes, joined.scales, joined.biases])
            slab = joined
            slabRows = joined.rowCount
            return
        }
        if quantizationEnabled {
            segmentStore = segments
            compactSegmentsToSlabs()
        } else {
            segmentStore = []
            guard !segments.isEmpty else { return }
            let parts = segments.map(Self.dequantize)
            hotPool = parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
        }
    }

    /// Snapshot for an in-place speculative restore (dSpark verify).
    ///
    /// Segmented layout: rows are only ever appended as new arrays or segments (never written in
    /// place), so the saved references stay valid.
    ///
    /// Contiguous layout: O(1) and holds no array. (A saved slab reference would stop MLX donating
    /// the slab, so every verify's first append would copy the whole slab.) Appends write only rows
    /// at or past the live count, and a growth copies the live rows unchanged, so the rows that were
    /// live at the checkpoint are still the slab's first rows: a restore only resets the live
    /// count. Rows a rejected tail wrote past it become slack that no reader sees and the next
    /// append overwrites. Only a drop below the checkpoint's count (trim, replace, re-encode, a
    /// restore to fewer rows) lets later appends rewrite its rows; restoring it after that traps.
    struct Snapshot {
        let hotPool: MLXArray?
        let segments: [HybridPoolQuantizedSegment]
        let emptySpec: (shape: [Int], dtype: DType)?
        let slabRows: Int
        let dropSerial: Int
    }

    func snapshot() -> Snapshot {
        Snapshot(
            hotPool: hotPool, segments: segmentStore, emptySpec: emptySpec,
            slabRows: slabRows, dropSerial: dropSerial)
    }

    func load(_ snapshot: Snapshot) {
        emptySpec = snapshot.emptySpec
        if !contiguous {
            hotPool = snapshot.hotPool
            segmentStore = snapshot.segments
            return
        }
        precondition(
            snapshotRowsIntact(snapshot),
            "DSV4 contiguous pool: speculative checkpoint of \(snapshot.slabRows) rows restored "
                + "after its rows were dropped (live rows \(slabRows))")
        if snapshot.slabRows < slabRows { noteDrop(toRows: snapshot.slabRows) }
        slabRows = snapshot.slabRows
    }

    /// True when the rows live at `snapshot` are still the slab's first rows.
    func snapshotRowsIntact(_ snapshot: Snapshot) -> Bool {
        guard contiguous else { return true }
        if snapshot.slabRows == 0 { return true }
        guard snapshot.dropSerial >= forgottenDropSerial,
              snapshot.slabRows <= slabRows, slab != nil
        else { return false }
        // `drops` rows increase with serial: the first drop after the snapshot is the lowest.
        let firstLater = drops.first { $0.serial > snapshot.dropSerial }
        return (firstLater?.rows ?? Int.max) >= snapshot.slabRows
    }

    /// Contiguous layout: drops of live rows as `(serial, rows left)`, rows strictly increasing with
    /// serial (a drop to `m` subsumes every earlier drop that left `>= m`). A snapshot taken at
    /// serial `s` with `n` rows is intact iff every later drop left at least `n` rows.
    private var drops: [(serial: Int, rows: Int)] = []
    private var dropSerial = 0
    /// Snapshots older than this serial can no longer be validated (``drops`` is bounded).
    private var forgottenDropSerial = 0
    private static let maxTrackedDrops = 64

    private func noteDrop(toRows rows: Int) {
        guard contiguous else { return }
        dropSerial += 1
        while let last = drops.last, last.rows >= rows { drops.removeLast() }
        drops.append((dropSerial, rows))
        if drops.count > Self.maxTrackedDrops {
            forgottenDropSerial = drops.removeFirst().serial
        }
    }

    func copy() -> DeepseekV4PoolStorage {
        let result = DeepseekV4PoolStorage(
            quantizationEnabled: quantizationEnabled,
            hotByteLimit: hotByteLimit,
            bits: bits,
            contiguous: contiguous,
            groupSize: groupSize)
        if contiguous {
            // The copy holds only the live rows (fresh handles over the same
            // buffer). The source keeps appending in place; MLX refuses to
            // donate a buffer another array still references, so the first
            // such write copies instead of changing this snapshot.
            result.emptySpec = emptySpec
            result.slab = liveSlab()
            result.slabRows = slabRows
            return result
        }
        result.hotPool = hotPool?[.ellipsis]
        result.emptySpec = emptySpec
        result.segmentStore = segmentStore.map { segment in
            HybridPoolQuantizedSegment(
                codes: segment.codes[.ellipsis],
                scales: segment.scales[.ellipsis],
                biases: segment.biases[.ellipsis],
                originalShape: segment.originalShape,
                groupSize: segment.groupSize,
                bits: segment.bits,
                originalDType: segment.originalDType)
        }
        return result
    }

    /// Contiguous layout: write freshly encoded rows after the live rows,
    /// growing the slab in ``contiguousGrowRows`` steps.
    private func appendContiguous(_ rows: HybridPoolQuantizedSegment) {
        let k = rows.rowCount
        guard k > 0 else { return }
        if let current = slab, !Self.compatible(current, rows) {
            // Width/group change (never produced by one cache): re-encode.
            noteDrop(toRows: 0)
            let live = liveSlab().map(Self.dequantize)
            let merged = Self.dequantize(rows)
            slab = Self.quantize(
                live.map { concatenated([$0, merged], axis: 1) } ?? merged, bits: bits, groupSize: groupSize)
            slabRows = slabRows + k
            return
        }
        let needed = slabRows + k
        let capacity = slab?.originalShape[1] ?? 0
        if slab == nil || needed > capacity {
            let step = Self.contiguousGrowRows
            let newCapacity = ((needed + step - 1) / step) * step
            func grown(_ live: MLXArray?, _ fresh: MLXArray) -> MLXArray {
                var shape = fresh.shape
                shape[1] = newCapacity - needed
                var parts: [MLXArray] = []
                if let live { parts.append(live) }
                parts.append(fresh)
                if shape[1] > 0 { parts.append(MLXArray.zeros(shape, dtype: fresh.dtype)) }
                return concatenated(parts, axis: 1)
            }
            let live = liveSlab()
            var shape = rows.originalShape
            shape[1] = newCapacity
            let next = HybridPoolQuantizedSegment(
                codes: grown(live?.codes, rows.codes),
                scales: grown(live?.scales, rows.scales),
                biases: grown(live?.biases, rows.biases),
                originalShape: shape,
                groupSize: rows.groupSize,
                bits: rows.bits,
                originalDType: rows.originalDType)
            asyncEval([next.codes, next.scales, next.biases])
            slab = next
            slabRows = needed
            return
        }
        guard let slab else { return }
        // In place: the slab arrays are private to this storage (copies and
        // views hold fresh handles), so MLX donates the buffer when nothing
        // else references it.
        slab.codes[0..., slabRows ..< needed, 0..., 0...] = rows.codes
        slab.scales[0..., slabRows ..< needed, 0..., 0...] = rows.scales
        slab.biases[0..., slabRows ..< needed, 0..., 0...] = rows.biases
        asyncEval([slab.codes, slab.scales, slab.biases])
        slabRows = needed
    }

    private func replaceQuantized(with value: MLXArray) {
        hotPool = nil
        segmentStore = []
        var start = 0
        while start < value.dim(1) {
            let remaining = value.dim(1) - start
            var take = min(Self.segmentRows, remaining)
            if remaining >= Self.segmentRows {
                let fullRows = (remaining / Self.segmentRows) * Self.segmentRows
                while take * 2 <= min(fullRows, Self.slabMaxRows) {
                    take *= 2
                }
            }
            let end = start + take
            segmentStore.append(Self.quantize(value[0..., start..<end, 0...], bits: bits, groupSize: groupSize))
            start = end
            compactSlabCascade()
        }
    }

    private func trailingPartialSegmentRows() -> Int {
        var rows = 0
        for segment in segmentStore.reversed() {
            guard segment.rowCount < Self.segmentRows else { break }
            rows += segment.rowCount
        }
        return rows
    }

    private func compactCompletedTail() {
        var start = segmentStore.count
        var rows = 0
        while start > 0 {
            let segment = segmentStore[start - 1]
            guard segment.rowCount < Self.segmentRows else { break }
            start -= 1
            rows += segment.rowCount
        }
        let parts = Array(segmentStore[start...])
        if parts.count >= 2 {
            segmentStore.replaceSubrange(start..., with: [Self.concatenate(parts)])
        }
        if rows >= Self.segmentRows {
            compactSlabCascade()
        }
    }

    /// Binary-counter compaction: equal trailing slabs merge until their
    /// combined row count reaches the 16K ceiling. Codes and affine metadata
    /// are concatenated directly, so this is bit-exact and never requantizes.
    private func compactSlabCascade() {
        while segmentStore.count >= 2 {
            let previous = segmentStore[segmentStore.count - 2]
            let last = segmentStore[segmentStore.count - 1]
            guard previous.rowCount == last.rowCount,
                  previous.rowCount + last.rowCount <= Self.slabMaxRows,
                  Self.compatible(previous, last)
            else { return }
            segmentStore.replaceSubrange(
                (segmentStore.count - 2)...,
                with: [Self.concatenate([previous, last])])
        }
    }

    /// Imported disk/delta state can still use the legacy 64-row wire shape.
    /// Regroup it once so every later token sees O(rows/16K) slabs.
    private func compactSegmentsToSlabs() {
        guard segmentStore.count > 1 else { return }
        var regrouped: [HybridPoolQuantizedSegment] = []
        var pending: [HybridPoolQuantizedSegment] = []
        var pendingRows = 0

        func flushPending() {
            guard !pending.isEmpty else { return }
            if pending.count == 1 {
                regrouped.append(pending[0])
            } else {
                let slab = Self.concatenate(pending)
                // Concatenate only async-schedules; a near-1M restore would
                // otherwise push the whole regroup into the Metal stream with
                // no sync, which can exhaust device memory beside the weights.
                MLX.eval([slab.codes, slab.scales, slab.biases])
                regrouped.append(slab)
            }
            pending = []
            pendingRows = 0
        }

        for segment in segmentStore {
            if let first = pending.first,
               pendingRows + segment.rowCount > Self.slabMaxRows
                    || !Self.compatible(first, segment)
            {
                flushPending()
            }
            pending.append(segment)
            pendingRows += segment.rowCount
        }
        flushPending()
        segmentStore = regrouped
    }

    /// The group a feature width actually encodes with (the whole row when it doesn't divide).
    static func effectiveGroupSize(features: Int, groupSize: Int) -> Int {
        features > 0 && features % groupSize == 0 ? groupSize : features
    }

    static func quantize(
        _ value: MLXArray, bits: Int = DeepseekV4PoolStorage.bits,
        groupSize: Int = DeepseekV4PoolStorage.groupSize
    ) -> HybridPoolQuantizedSegment {
        let shape = value.shape
        let featureCount = shape.last ?? 0
        precondition(featureCount > 0, "DSV4 pool feature dimension must be non-zero")
        let effectiveGroupSize = effectiveGroupSize(features: featureCount, groupSize: groupSize)
        // Nibble packing needs an even group; an odd fallback group stays 8-bit.
        let bits = (bits == 4 && effectiveGroupSize % 2 != 0) ? 8 : bits
        let groupCount = featureCount / effectiveGroupSize
        let groupedShape = Array(shape.dropLast()) + [groupCount, effectiveGroupSize]
        let grouped = value.asType(.float32).reshaped(groupedShape)
        let minimums = grouped.min(axis: -1, keepDims: true)
        let maximums = grouped.max(axis: -1, keepDims: true)
        let qmax = Float((1 << bits) - 1)
        let rawScales = (maximums - minimums) / qmax
        let scales32 = MLX.where(
            rawScales .> Float(1e-8),
            rawScales,
            MLXArray.ones(like: rawScales))
        var codes = clip(
            MLX.round((grouped - minimums) / scales32),
            min: Float(0), max: qmax).asType(.uint8)
        if bits == 4 {
            codes = packNibbles(codes)
        }
        let scales = scales32.asType(.float16)
        let biases = minimums.asType(.float16)
        asyncEval([codes, scales, biases])
        return HybridPoolQuantizedSegment(
            codes: codes,
            scales: scales,
            biases: biases,
            originalShape: shape,
            groupSize: effectiveGroupSize,
            bits: bits,
            originalDType: value.dtype)
    }

    /// `[..., g]` uint8 codes in 0...15 -> `[..., g/2]` bytes, element 2k in the
    /// low nibble and 2k+1 in the high nibble.
    static func packNibbles(_ codes: MLXArray) -> MLXArray {
        let shape = codes.shape
        let pairs = codes.reshaped(Array(shape.dropLast()) + [shape.last! / 2, 2])
        let lo = pairs[.ellipsis, 0]
        let hi = pairs[.ellipsis, 1]
        return (lo | (hi << UInt8(4))).asType(.uint8)
    }

    /// Inverse of ``packNibbles(_:)``: `[..., g/2]` bytes -> `[..., g]` codes.
    static func unpackNibbles(_ packed: MLXArray) -> MLXArray {
        let shape = packed.shape
        let lo = packed & UInt8(0x0F)
        let hi = (packed >> UInt8(4)) & UInt8(0x0F)
        return stacked([lo, hi], axis: -1)
            .reshaped(Array(shape.dropLast()) + [shape.last! * 2])
    }

    /// Codes as one value per element regardless of the storage width.
    fileprivate static func elementCodes(_ codes: MLXArray, bits: Int) -> MLXArray {
        bits == 4 ? unpackNibbles(codes) : codes
    }

    static func dequantize(_ segment: HybridPoolQuantizedSegment) -> MLXArray {
        (elementCodes(segment.codes, bits: segment.bits).asType(.float32)
            * segment.scales.asType(.float32)
            + segment.biases.asType(.float32))
            .reshaped(segment.originalShape)
            .asType(segment.originalDType)
    }

    /// Lazy row-range view of a segment for bounded tile iteration. No eval:
    /// the slices are transient graph nodes consumed by one dequantize.
    private static func sliceRows(
        _ segment: HybridPoolQuantizedSegment, from start: Int, to end: Int
    ) -> HybridPoolQuantizedSegment {
        var shape = segment.originalShape
        shape[1] = end - start
        return HybridPoolQuantizedSegment(
            codes: segment.codes[0..., start ..< end, 0..., 0...],
            scales: segment.scales[0..., start ..< end, 0..., 0...],
            biases: segment.biases[0..., start ..< end, 0..., 0...],
            originalShape: shape,
            groupSize: segment.groupSize,
            bits: segment.bits,
            originalDType: segment.originalDType)
    }

    /// Decode exactly the requested `(batch, row)` occurrences of one q8
    /// segment. Returns `(N, D)` in occurrence order (Python mirror:
    /// `_dequant_qpool_selected`).
    private static func dequantizeSelected(
        _ segment: HybridPoolQuantizedSegment,
        batchIndices: [Int32],
        rowIndices: [Int32]
    ) -> MLXArray {
        let rows = segment.rowCount
        let linear = MLXArray(
            zip(batchIndices, rowIndices).map { $0 * Int32(rows) + $1 })

        func flatten(_ arr: MLXArray) -> MLXArray {
            let shape = arr.shape
            return arr.reshaped([shape[0] * shape[1]] + Array(shape.dropFirst(2)))
        }
        let codes = elementCodes(
            take(flatten(segment.codes), linear, axis: 0), bits: segment.bits)
        let scales = take(flatten(segment.scales), linear, axis: 0)
        let biases = take(flatten(segment.biases), linear, axis: 0)
        let width = segment.originalShape.last ?? 0
        return (codes.asType(.float32) * scales.asType(.float32)
            + biases.asType(.float32))
            .reshaped(batchIndices.count, width)
            .asType(segment.originalDType)
    }

    private static func compatible(
        _ lhs: HybridPoolQuantizedSegment,
        _ rhs: HybridPoolQuantizedSegment
    ) -> Bool {
        lhs.originalShape.count == rhs.originalShape.count
            && lhs.originalShape.first == rhs.originalShape.first
            && Array(lhs.originalShape.dropFirst(2)) == Array(rhs.originalShape.dropFirst(2))
            && lhs.groupSize == rhs.groupSize
            && lhs.bits == rhs.bits
            && lhs.originalDType == rhs.originalDType
    }

    private static func concatenate(
        _ segments: [HybridPoolQuantizedSegment]
    ) -> HybridPoolQuantizedSegment {
        precondition(!segments.isEmpty, "cannot concatenate an empty DSV4 pool segment list")
        let first = segments[0]
        precondition(
            segments.dropFirst().allSatisfy { compatible(first, $0) },
            "cannot concatenate incompatible DSV4 pool segments")
        var shape = first.originalShape
        shape[1] = segments.reduce(0) { $0 + $1.rowCount }
        let result = HybridPoolQuantizedSegment(
            codes: concatenated(segments.map(\.codes), axis: 1),
            scales: concatenated(segments.map(\.scales), axis: 1),
            biases: concatenated(segments.map(\.biases), axis: 1),
            originalShape: shape,
            groupSize: first.groupSize,
            bits: first.bits,
            originalDType: first.originalDType)
        asyncEval([result.codes, result.scales, result.biases])
        return result
    }

    private static func slice(
        _ segment: HybridPoolQuantizedSegment,
        keepingRows rows: Int
    ) -> HybridPoolQuantizedSegment {
        var shape = segment.originalShape
        shape[1] = rows
        let result = HybridPoolQuantizedSegment(
            codes: segment.codes[0..., 0..<rows, 0..., 0...],
            scales: segment.scales[0..., 0..<rows, 0..., 0...],
            biases: segment.biases[0..., 0..<rows, 0..., 0...],
            originalShape: shape,
            groupSize: segment.groupSize,
            bits: segment.bits,
            originalDType: segment.originalDType)
        asyncEval([result.codes, result.scales, result.biases])
        return result
    }
}

// MARK: - DSV4 pool view

/// Attention-facing representation of a compressor/indexer pool.
///
/// Python mirror: `_QuantizedPoolView` + `update_pool_view`. A pool stays a
/// dense array while its BF16 hot tier is small — decode keeps the fused-SDPA
/// fast path, prefill keeps dense masks. Once the branch promotes to
/// segmented q8 storage, or a prefill's hot tier crosses the dense-row cap,
/// attention receives the bounded form instead and must iterate tiles or
/// gather selected rows — the complete historical pool is never materialized
/// on the token path.
public struct DeepseekV4PoolView {
    private enum Backing {
        case dense(MLXArray)
        case storage(DeepseekV4PoolStorage)
    }

    private let backing: Backing

    init(dense: MLXArray) {
        self.backing = .dense(dense)
    }

    init(storage: DeepseekV4PoolStorage) {
        self.backing = .storage(storage)
    }

    /// Non-nil when attention may use the dense fused-SDPA / dense-mask path.
    public var denseArray: MLXArray? {
        if case .dense(let array) = backing { return array }
        return nil
    }

    public var rowCount: Int {
        switch backing {
        case .dense(let array): return array.dim(1)
        case .storage(let storage): return storage.rowCount
        }
    }

    public var featureDim: Int {
        switch backing {
        case .dense(let array): return array.dim(2)
        case .storage(let storage): return storage.featureDim ?? 0
        }
    }

    public var dtype: DType {
        switch backing {
        case .dense(let array): return array.dtype
        case .storage(let storage): return storage.poolDType ?? .float32
        }
    }

    public func forEachDequantizedTile(
        maxRows: Int, _ body: (Int, MLXArray) -> Void
    ) {
        switch backing {
        case .dense(let array):
            let rows = array.dim(1)
            let maxRows = max(1, maxRows)
            var start = 0
            while start < rows {
                let end = min(start + maxRows, rows)
                body(start, array[0..., start ..< end, 0...])
                start = end
            }
        case .storage(let storage):
            storage.forEachDequantizedTile(maxRows: maxRows, body)
        }
    }

    public func gatherDequantizedRows(_ indices: MLXArray) -> MLXArray {
        switch backing {
        case .dense(let array):
            let B = array.dim(0)
            let D = array.dim(2)
            let flat = indices.reshaped(B, -1).asType(.int32)
            let idxBroad = broadcast(
                flat.expandedDimensions(axis: -1), to: [B, flat.dim(1), D])
            return takeAlong(array, idxBroad, axis: 1)
                .reshaped(indices.shape + [D])
        case .storage(let storage):
            return storage.gatherDequantizedRows(indices)
        }
    }

    /// Compatibility-only full materialization; never used on the token path.
    public func materialized() -> MLXArray {
        switch backing {
        case .dense(let array):
            return array
        case .storage(let storage):
            return storage.materialized()
                ?? MLXArray.zeros([1, 0, 1], dtype: .float32)
        }
    }
}

// MARK: - DeepseekV4Cache
//
// Per-layer composite cache. Wraps a `RotatingKVCache` for the local
// sliding window plus persistent buffer state for the compressor and
// indexer. Multi-call stateful: on each prefill step it accumulates
// raw-token windows until a full `compress_ratio`-sized chunk is ready,
// then pools and stores. The pooled sequence grows across calls so
// turn 2 and beyond see the full history summary.
//
// For short prompts (L < compress_ratio) and no V4Cache provided, the
// attention forward takes a fast-path that skips the compressor
// entirely (Python mirror: `if v4_cache is None and L < compress_ratio
// → skip`).
public final class DeepseekV4Cache: QuantizedHybridPoolCache, CacheRetainedByteCountProviding {
    /// Default (DSV4-0731) bit width the pool storage encodes with.
    public static var poolQuantizationBits: Int { DeepseekV4PoolStorage.bits }
    /// Every width the pool storage can encode. The disk serializer's
    /// quantized-pool schema accepts exactly these widths — pinned by
    /// `DeepseekV4CacheDiskRoundTripTests.serializerBitsMatchPoolStorage`
    /// so neither side can drift alone.
    public static var supportedPoolQuantizationBits: Set<Int> { DeepseekV4PoolStorage.supportedBits }
    /// This cache's pool code width: 8 for DSV4-0731, 4 for DeepSeek-V4.1
    /// (its reference stores compressed KV in FP4).
    public var hybridPoolQuantizationBits: Int { compPool.bits }
    /// True for DeepSeek-V4.1 caches: pools are encoded at append into one
    /// contiguous slab each (see ``DeepseekV4PoolStorage/contiguous``).
    public var contiguousPools: Bool { compPool.contiguous }
    /// DeepSeek-V4.1: after a multi-token update keep only the last
    /// `slidingWindow` local rows (``RotatingKVCache/compactToWindow()``), like the
    /// reference `cache.window = raw[:, -win:]`.
    public let compactLocalWindow: Bool

    /// Expose the inner rotating cache so `TQDiskSerializer` and
    /// `restoreRotatingLayer` can round-trip the sliding-window state.
    /// Compressor/Indexer pool tensors and their incomplete-window
    /// buffers ALSO round-trip via `state` / `metaState` so prefix-cache
    /// reuse for multi-turn chat doesn't have to re-derive the pool
    /// from prompt tokens every turn.
    public var rotating: RotatingKVCache { local }
    /// Local sliding-window cache (compress_ratio-agnostic).
    public let local: RotatingKVCache
    public let slidingWindow: Int
    /// Per-layer compress_ratio. Required (no nil): the proportional
    /// pool-row truncation in `trim(_:)` needs it, the paged cache
    /// uses it as part of the block hash so `cr=4` and `cr=128` layer
    /// blocks never collide, and the disk serializer stamps it as a
    /// metaState entry.
    public let compressRatio: Int
    /// Compressor buffer state (raw kv/gate not yet ready to pool)
    /// and pooled summary so far.
    fileprivate var compBufferKV: MLXArray?
    fileprivate var compBufferGate: MLXArray?
    private var compPool: DeepseekV4PoolStorage
    /// Indexer's own buffer state (separate branch — compressor inside
    /// the indexer uses its own buffers).
    fileprivate var idxBufferKV: MLXArray?
    fileprivate var idxBufferGate: MLXArray?
    private var idxPool: DeepseekV4PoolStorage

    /// Raw single-token hidden states waiting for the next compression-window
    /// boundary. Holding these refs avoids two fp32 projections plus buffer
    /// concat/slice bookkeeping on every non-boundary decode token. External
    /// state readers flush them before snapshot/export; trim drops them just
    /// like it drops incomplete projected buffers.
    private final class PendingPoolSlot {
        var inputs: [MLXArray] = []
        var basePosition = 0
        var lastPool: DeepseekV4PoolView?
        var flush: ((MLXArray, Int) -> DeepseekV4PoolView)?
    }
    private let compPending = PendingPoolSlot()
    private let idxPending = PendingPoolSlot()

    /// Family-native DSV4 pool quantization. This does not enable TurboQuant
    /// KV: the rotating local K/V cache remains in its model-selected dtype.
    public let hybridPoolQuantizationEnabled: Bool

    /// DeepSeek-V4.1 Engram n-gram history (compressed token ids) for the
    /// sequence this cache belongs to — the Swift analog of the reference
    /// `Session.history`. Only the designated holder layer of a V4.1 cache
    /// list carries it; it is nil for every DSV4-0731 cache. Bounded to the
    /// most recent ``auxiliaryTokenHistoryCapacity`` positions (an n-gram
    /// reaches back `max_ngram_size - 1` tokens; the slack absorbs trims).
    public var auxiliaryTokenHistory: HybridPoolTokenHistory?
    public let requiresAuxiliaryTokenHistory: Bool

    /// DeepSeek-V4.1 chunked prefill (history holder only): token ids of the NEXT
    /// forward on this sequence, set by `DeepseekV4Model.prepare` so that forward's
    /// engram reads start while the current chunk computes. Transient: never copied,
    /// serialized or restored; any trim drops it.
    var engramLookaheadTokens: [Int32]?
    /// The started lookahead (`EngramRuntime.Lookahead`), validated by the next forward.
    var engramLookahead: EngramRuntime.Lookahead?
    public static let auxiliaryTokenHistoryCapacity = 1024

    /// DeepSeek-V4.1 dSpark draft-window tail: the target-layer hidden rows of the last committed
    /// positions, recorded by every forward while a dSpark head is attached (holder layer only).
    /// Travels with copies, trims, speculative restores and disk records so the draft windows can
    /// be rebuilt from the cache alone. Speed-only: nil or stale just means a shorter window.
    public var auxiliaryHiddenTail: HybridPoolHiddenTail?
    /// dSpark verify capture requested for the next forward of this sequence (holder layer only).
    /// Transient: never copied, checkpointed or persisted.
    var dsparkVerifyCapture: DSV41VerifyCapture?

    public init(
        slidingWindow: Int,
        compressRatio: Int,
        poolQuantizationEnabled: Bool? = nil,
        poolBF16HotByteLimit: Int? = nil,
        requiresAuxiliaryTokenHistory: Bool = false,
        poolQuantizationBits: Int = 8,
        contiguousPools: Bool = false,
        compactLocalWindow: Bool = false
    ) {
        precondition(compressRatio > 0, "DeepseekV4Cache requires compressRatio > 0; cr=0 layers should use a plain RotatingKVCache.")
        let quantizationEnabled = poolQuantizationEnabled
            ?? Self.resolvePoolQuantizationDefault()
        self.slidingWindow = slidingWindow
        self.compressRatio = compressRatio
        self.hybridPoolQuantizationEnabled = quantizationEnabled
        let hotByteLimit = poolBF16HotByteLimit
            ?? Self.resolvePoolBF16HotByteLimit()
        self.compPool = DeepseekV4PoolStorage(
            quantizationEnabled: quantizationEnabled,
            hotByteLimit: hotByteLimit,
            bits: poolQuantizationBits,
            contiguous: contiguousPools)
        self.idxPool = DeepseekV4PoolStorage(
            quantizationEnabled: quantizationEnabled,
            hotByteLimit: hotByteLimit,
            bits: poolQuantizationBits,
            contiguous: contiguousPools)
        self.local = RotatingKVCache(maxSize: slidingWindow, keep: 0)
        self.compactLocalWindow = compactLocalWindow
        self.requiresAuxiliaryTokenHistory = requiresAuxiliaryTokenHistory
    }

    /// Append compressed ids for positions `startPos ..< startPos + tokens.count`,
    /// dropping any stale tail at or beyond `startPos` and keeping the bounded
    /// window. `previous` is what the caller already validated for the
    /// positions just before `startPos`.
    public func recordAuxiliaryTokens(_ tokens: [Int32], startPos: Int) {
        var kept: [Int32] = []
        if let h = auxiliaryTokenHistory, h.endPosition >= startPos, h.startPosition <= startPos {
            kept = Array(h.tokens.prefix(startPos - h.startPosition))
        }
        kept.append(contentsOf: tokens)
        let cap = Self.auxiliaryTokenHistoryCapacity
        if kept.count > cap { kept.removeFirst(kept.count - cap) }
        auxiliaryTokenHistory = HybridPoolTokenHistory(
            endPosition: startPos + tokens.count, tokens: kept)
    }

    /// Compressed ids at positions `startPos - count ..< startPos`, or nil
    /// when the carried history does not cover that range.
    public func auxiliaryTokens(before startPos: Int, count: Int) -> [Int32]? {
        guard count > 0 else { return [] }
        guard let h = auxiliaryTokenHistory,
              startPos <= h.endPosition, startPos - count >= h.startPosition
        else { return nil }
        let lo = startPos - count - h.startPosition
        return Array(h.tokens[lo ..< (lo + count)])
    }

    /// Pool quantization is the production default for DSV4 long context.
    /// `DSV4_POOL_QUANT=0|false|off|no` remains an explicit diagnostic A/B.
    public static func resolvePoolQuantizationDefault(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        guard let raw = environment["DSV4_POOL_QUANT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        else { return true }
        return !["0", "false", "off", "no"].contains(raw)
    }

    /// Attention-ready BF16 pool budget per compressor/indexer branch.
    /// At ratio 4 the 512-wide compressor pool consumes about 1 MiB per 4K
    /// prompt tokens, so the 64 MiB production tier keeps ordinary long
    /// contexts off the much slower q8 gather path while leaving q8 active for
    /// genuinely huge contexts. The override is diagnostic/benchmark-only.
    public static func resolvePoolBF16HotByteLimit(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        if let raw = environment["DSV4_POOL_BF16_MAX_BYTES"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           let parsed = Int(raw)
        {
            return max(0, parsed)
        }
        return 64 * 1024 * 1024
    }

    /// Conservative retained-byte projection for one batch of the complete
    /// DSV4 hybrid cache at a requested context. It includes UInt8 pool codes,
    /// fp16 affine metadata, every layer's bounded rotating K/V, and a full
    /// incomplete-window buffer allowance for both compressor branches.
    public static func projectedQuantizedCacheUpperBoundBytes(
        contextLength: Int,
        compressRatios: [Int],
        headDim: Int,
        indexerHeadDim: Int,
        slidingWindow: Int,
        activationBytes: Int = 2,
        poolBF16HotByteLimit: Int? = nil,
        poolQuantizationBits: Int = 8
    ) -> Int {
        precondition(contextLength >= 0)
        precondition(headDim > 0 && indexerHeadDim > 0)
        precondition(slidingWindow > 0 && activationBytes > 0)

        func encodedPoolBytes(rows: Int, features: Int) -> Int {
            guard rows > 0 else { return 0 }
            let group = features % DeepseekV4PoolStorage.groupSize == 0
                ? DeepseekV4PoolStorage.groupSize
                : features
            let elements = rows * features
            let affineGroups = rows * (features / group)
            // Packed code bytes (8-bit: one per element, 4-bit: one per two)
            // + fp16 scale + fp16 minimum per affine group.
            let codeBytes = poolQuantizationBits == 4 && group % 2 == 0
                ? elements / 2 : elements
            return codeBytes + affineGroups * 2 * MemoryLayout<Float16>.size
        }

        let hotByteLimit = poolBF16HotByteLimit
            ?? resolvePoolBF16HotByteLimit()
        func retainedPoolBytes(rows: Int, features: Int) -> Int {
            let rawBytes = rows * features * activationBytes
            return rawBytes <= hotByteLimit
                ? rawBytes
                : encodedPoolBytes(rows: rows, features: features)
        }

        var total = compressRatios.count
            * 2 * slidingWindow * headDim * activationBytes
        for ratio in compressRatios where ratio > 0 {
            let rows = (contextLength + ratio - 1) / ratio
            total += retainedPoolBytes(rows: rows, features: headDim)
            let compressorWidth = ratio == 4 ? 2 * headDim : headDim
            total += 2 * ratio * compressorWidth * activationBytes
            if ratio == 4 {
                total += retainedPoolBytes(rows: rows, features: indexerHeadDim)
                total += 2 * ratio * indexerHeadDim * activationBytes
            }
        }
        return total
    }

    // KVCache protocol implementation — delegate everything to `local`.
    public var offset: Int { local.offset }
    public var maxSize: Int? { local.maxSize }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let window = local.update(keys: keys, values: values)
        if compactLocalWindow, keys.dim(2) > 1 {
            // The returned window still feeds this forward's attention.
            local.compactToWindow()
        }
        return window
    }

    /// Round-trip layout (kept stable for `TQDiskSerializer`):
    ///   index 0..<localCount       — `local.state` (rotating window)
    ///   index localCount + 0       — compPooled (or empty zero-row tensor)
    ///   index localCount + 1       — compBufferKV (or empty)
    ///   index localCount + 2       — compBufferGate (or empty)
    ///   index localCount + 3       — idxPooled
    ///   index localCount + 4       — idxBufferKV
    ///   index localCount + 5       — idxBufferGate
    /// `metaState` carries the entry tag list + `compress_ratio` + the
    /// number of `local.state` arrays so deserialization knows where to
    /// split.
    public var state: [MLXArray] {
        get {
            flushPoolPending()
            let localState = local.state
            return localState
                + [serializableArray(compPool.materialized()),
                   serializableArray(compBufferKV),
                   serializableArray(compBufferGate),
                   serializableArray(idxPool.materialized()),
                   serializableArray(idxBufferKV),
                   serializableArray(idxBufferGate)]
        }
        set {
            dropPoolPending()
            // Last 6 slots are the DSV4 pool/buffer arrays; everything
            // before them is `local.state`.
            precondition(newValue.count >= 6,
                "DeepseekV4Cache.state setter expects at least 6 trailing pool slots")
            let split = newValue.count - 6
            local.state = Array(newValue[0..<split])
            compPool.replace(with: nullableFromArray(newValue[split + 0]))
            compBufferKV = nullableFromArray(newValue[split + 1])
            compBufferGate = nullableFromArray(newValue[split + 2])
            idxPool.replace(with: nullableFromArray(newValue[split + 3]))
            idxBufferKV = nullableFromArray(newValue[split + 4])
            idxBufferGate = nullableFromArray(newValue[split + 5])
        }
    }

    public var metaState: [String] {
        get {
            // First entries are `local.metaState`; we tack on dsv4-
            // specific scalars so the disk serializer + restore path
            // can reconstruct without guessing.
            var meta = local.metaState + [
                "dsv4_cache_v1",
                String(compressRatio),
                String(slidingWindow),
            ]
            // DeepSeek-V4.1 only: Engram history trailer. 0731 caches never
            // carry history, so their metaState is unchanged.
            if let h = auxiliaryTokenHistory {
                meta += [
                    "dsv41_aux_hist_v1",
                    String(h.endPosition),
                    h.tokens.map(String.init).joined(separator: ","),
                ]
            }
            return meta
        }
        set {
            var newValue = newValue
            if newValue.count >= 3, newValue[newValue.count - 3] == "dsv41_aux_hist_v1" {
                let end = Int(newValue[newValue.count - 2]) ?? 0
                let raw = newValue[newValue.count - 1]
                let tokens = raw.isEmpty ? [] : raw.split(separator: ",").compactMap { Int32($0) }
                auxiliaryTokenHistory = HybridPoolTokenHistory(endPosition: end, tokens: tokens)
                newValue.removeLast(3)
            } else {
                auxiliaryTokenHistory = nil
            }
            // Strip the DSV4 trailer if present and pass the rest to
            // local. We don't need to re-read compressRatio from the
            // trailer because it's an init-time invariant of the
            // surrounding layer's attention module — but accept the
            // legacy code path that didn't write it.
            if newValue.count >= 3,
               newValue[newValue.count - 3] == "dsv4_cache_v1"
            {
                local.metaState = Array(newValue[0..<(newValue.count - 3)])
            } else {
                local.metaState = newValue
            }
        }
    }

    public var isTrimmable: Bool { local.isTrimmable }

    /// 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
    /// Proportional pool-row truncation matching `llama.cpp dsv4_clear_rows`.
    /// Pre-fix this only delegated to `local.trim(n)` — the contaminated
    /// pool rows survived multi-turn prefix-cache reuse and produced the
    /// "polite-assistant attractor loops" reported on /v1/chat/completions.
    @discardableResult
    public func trim(_ n: Int) -> Int {
        // Pending entries have not completed a compression window and cannot
        // have contributed a pool row. A trim invalidates their positions in
        // exactly the same way as the projected incomplete-window buffers.
        dropPoolPending()
        engramLookaheadTokens = nil
        engramLookahead = nil
        let rv = local.trim(n)
        if var h = auxiliaryTokenHistory, h.endPosition > local.offset {
            let drop = min(h.tokens.count, h.endPosition - local.offset)
            h.tokens.removeLast(drop)
            h.endPosition = local.offset
            auxiliaryTokenHistory = h
        }
        if var tail = auxiliaryHiddenTail, tail.endPosition > local.offset {
            tail.truncate(toEnd: local.offset)
            auxiliaryHiddenTail = tail.isEmpty ? nil : tail
        }
        // Incomplete-window buffers are start_pos-keyed and invalidated
        // by ANY trim — clear unconditionally (Python comment
        // mlx_model.py L497-501).
        compBufferKV = nil
        compBufferGate = nil
        idxBufferKV = nil
        idxBufferGate = nil
        if n <= 0 || compressRatio <= 0 {
            return rv
        }
        // Drop the trailing `max(1, n / compress_ratio)` rows from each
        // pool. Most-recently-appended row may have been computed from
        // a window overlapping output tokens — keeping it would
        // re-introduce contamination (Python L532-537).
        let rowsToDrop = max(1, n / compressRatio)
        compPool.trimTrailingRows(rowsToDrop)
        idxPool.trimTrailingRows(rowsToDrop)
        return rv
    }

    public func innerState() -> [MLXArray] {
        local.innerState()
            + compPool.retainedArrays
            + [compBufferKV, compBufferGate, idxBufferKV, idxBufferGate].compactMap { $0 }
            + idxPool.retainedArrays
            + compPending.inputs
            + idxPending.inputs
    }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        local.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    public func copy() -> any KVCache {
        flushPoolPending()
        // Deep-copy: clone local KV plus all four pool/buffer slots so
        // the snapshot is fully independent of the original cache.
        // Pre-fix `copy()` constructed a fresh cache and only set
        // `state = local.state`, which silently dropped the pool
        // entirely — so any caller that relied on `copy()` to
        // checkpoint multi-turn state lost the long-context summary.
        let dup = DeepseekV4Cache(
            slidingWindow: slidingWindow,
            compressRatio: compressRatio,
            poolQuantizationEnabled: hybridPoolQuantizationEnabled,
            poolBF16HotByteLimit: compPool.hotByteLimit,
            requiresAuxiliaryTokenHistory: requiresAuxiliaryTokenHistory,
            poolQuantizationBits: compPool.bits,
            contiguousPools: compPool.contiguous,
            compactLocalWindow: compactLocalWindow)
        dup.auxiliaryTokenHistory = auxiliaryTokenHistory
        dup.auxiliaryHiddenTail = auxiliaryHiddenTail
        let localState = local.state
        if !localState.isEmpty {
            // RotatingKVCache writes decode tokens into its ring in place.
            // An ellipsis slice is only a view, so the retained prompt-minus-one
            // checkpoint would otherwise be overwritten before the synchronous
            // disk store at terminal state. Create independent buffers; the
            // snapshot caller batches their materialization in one MLX.eval.
            dup.local.state = localState.map { $0 * 1 }
        }
        dup.local.metaState = local.metaState
        dup.compPool = compPool.copy()
        dup.idxPool = idxPool.copy()
        dup.compBufferKV = compBufferKV?[.ellipsis]
        dup.compBufferGate = compBufferGate?[.ellipsis]
        dup.idxBufferKV = idxBufferKV?[.ellipsis]
        dup.idxBufferGate = idxBufferGate?[.ellipsis]
        return dup
    }

    public var retainedCacheByteCount: Int {
        local.state.reduce(0) { $0 + $1.nbytes }
            + compPool.retainedByteCount
            + idxPool.retainedByteCount
            + [compBufferKV, compBufferGate, idxBufferKV, idxBufferGate]
                .compactMap { $0 }
                .reduce(0) { $0 + $1.nbytes }
            + compPending.inputs.reduce(0) { $0 + $1.nbytes }
            + idxPending.inputs.reduce(0) { $0 + $1.nbytes }
    }

    // MARK: - State (de)serialization helpers

    /// MLX state arrays cannot be `nil`, so we substitute a zero-row
    /// sentinel and recover it on the way back. The sentinel is shape
    /// (1, 0, 1) fp32 — distinguishable from any real pool tensor by
    /// having axis-1 size 0.
    private func serializableArray(_ arr: MLXArray?) -> MLXArray {
        guard let arr else {
            return MLXArray.zeros([1, 0, 1], dtype: .float32)
        }
        return arr
    }

    private func nullableFromArray(_ arr: MLXArray) -> MLXArray? {
        // Sentinel for nil: any tensor whose axis-1 dim is 0.
        if arr.ndim >= 2 && arr.dim(1) == 0 {
            return nil
        }
        return arr
    }

    // State accessors for Compressor/Indexer. Public so disk round-trip
    // tests and any future cache-inspection code can verify the
    // ephemeral buffers are cleared on restore (they recompute from
    // prompt tokens on the next prefill).
    public func getBuffers(_ key: BranchKey) -> (kv: MLXArray?, gate: MLXArray?) {
        switch key {
        case .compressor: return (compBufferKV, compBufferGate)
        case .indexer: return (idxBufferKV, idxBufferGate)
        }
    }

    public func setBuffers(_ key: BranchKey, kv: MLXArray?, gate: MLXArray?) {
        switch key {
        case .compressor:
            compBufferKV = kv
            compBufferGate = gate
        case .indexer:
            idxBufferKV = kv
            idxBufferGate = gate
        }
    }

    public func getPooled(_ key: BranchKey) -> MLXArray? {
        key == .compressor ? compPool.materialized() : idxPool.materialized()
    }

    public func setPooled(_ key: BranchKey, value: MLXArray) {
        if key == .compressor {
            compPool.replace(with: value)
        } else {
            idxPool.replace(with: value)
        }
    }

    /// Prefill keys-per-query transients scale with pool rows on the dense
    /// path and Metal-OOM past ~180K tokens even while the hot tier is BF16.
    /// Above this row cap prefill gets the bounded tiled view instead; decode
    /// keeps its dense hot tier. Rows, not bytes, so the 512-wide compressor
    /// and 128-wide indexer branches flip together (Python mirror:
    /// `_POOL_PREFILL_DENSE_MAX_ROWS`).
    public static let poolPrefillDenseMaxRows: Int = {
        if let raw = ProcessInfo.processInfo.environment[
            "DSV4_POOL_PREFILL_DENSE_MAX_ROWS"],
            let parsed = Int(raw)
        {
            return max(0, parsed)
        }
        return 16384
    }()

    /// Active attention representation for one branch (Python mirror:
    /// `update_pool_view` return rules).
    func poolView(
        _ key: BranchKey, isPrefill: Bool, emptyLike: MLXArray
    ) -> DeepseekV4PoolView {
        let storage = key == .compressor ? compPool : idxPool
        if storage.contiguous {
            // DeepSeek-V4.1 (reference `Shared.view`): the owner layer dequantizes
            // a pool once per forward and its consumer layers share that view.
            // Index keys are always scored densely; compressed KV is dense for a
            // bounded prefill and gathered row by row otherwise (decode decodes
            // only the selected rows).
            let rows = storage.rowCount
            if rows > 0,
               key == .indexer || (isPrefill && rows <= Self.poolPrefillDenseMaxRows),
               let dense = storage.materialized()
            {
                return DeepseekV4PoolView(dense: dense)
            }
            if rows > 0 { return DeepseekV4PoolView(storage: storage) }
            return DeepseekV4PoolView(
                dense: MLXArray.zeros(
                    [emptyLike.dim(0), 0, emptyLike.dim(2)], dtype: emptyLike.dtype))
        }
        if let hot = storage.hotPool {
            if !isPrefill || hot.dim(1) <= Self.poolPrefillDenseMaxRows {
                return DeepseekV4PoolView(dense: hot)
            }
            return DeepseekV4PoolView(storage: storage)
        }
        if !storage.quantizedSegments.isEmpty {
            return DeepseekV4PoolView(storage: storage)
        }
        return DeepseekV4PoolView(
            dense: MLXArray.zeros(
                [emptyLike.dim(0), 0, emptyLike.dim(2)], dtype: emptyLike.dtype))
    }

    public func appendPooled(
        _ key: BranchKey, value: MLXArray, isPrefill: Bool
    ) -> DeepseekV4PoolView {
        let storage = key == .compressor ? compPool : idxPool
        storage.append(value)
        return poolView(key, isPrefill: isPrefill, emptyLike: value)
    }

    public enum BranchKey: Hashable { case compressor, indexer }

    private func pendingSlot(_ key: BranchKey) -> PendingPoolSlot {
        key == .compressor ? compPending : idxPending
    }

    /// Buffer one decode token and run the real compressor advancement only
    /// when its next window completes. `flush` bypasses this entry point and
    /// therefore cannot recursively defer itself.
    func deferPoolToken(
        _ x: MLXArray,
        branch: BranchKey,
        startPos: Int,
        ratio: Int,
        overlap: Bool,
        flush: @escaping (MLXArray, Int) -> DeepseekV4PoolView
    ) -> DeepseekV4PoolView {
        let slot = pendingSlot(branch)
        if slot.inputs.isEmpty {
            slot.basePosition = startPos
        }
        slot.inputs.append(x)
        slot.flush = flush

        var bufferedRows = getBuffers(branch).kv?.dim(1) ?? 0
        if overlap && bufferedRows >= ratio {
            // The first ratio rows are the previous completed overlap window;
            // only the rows after it count toward the next boundary.
            bufferedRows -= ratio
        }
        if slot.lastPool == nil || bufferedRows + slot.inputs.count >= ratio {
            return flushPoolPending(branch: branch)
                ?? DeepseekV4PoolView(
                    dense: MLXArray.zeros(
                        [x.dim(0), 0, x.dim(2)], dtype: x.dtype))
        }
        return slot.lastPool!
    }

    func notePoolResult(_ value: DeepseekV4PoolView, branch: BranchKey) {
        pendingSlot(branch).lastPool = value
    }

    @discardableResult
    func flushPoolPending(branch: BranchKey) -> DeepseekV4PoolView? {
        let slot = pendingSlot(branch)
        guard !slot.inputs.isEmpty else { return slot.lastPool }
        let inputs = slot.inputs
        let basePosition = slot.basePosition
        let flush = slot.flush
        slot.inputs = []
        guard let flush else { return slot.lastPool }
        let x = inputs.count == 1 ? inputs[0] : concatenated(inputs, axis: 1)
        let result = flush(x, basePosition)
        slot.lastPool = result
        return result
    }

    func flushPoolPending() {
        _ = flushPoolPending(branch: .compressor)
        _ = flushPoolPending(branch: .indexer)
    }

    private func dropPoolPending() {
        for slot in [compPending, idxPending] {
            slot.inputs = []
            slot.lastPool = nil
            slot.flush = nil
        }
    }

    func pendingPoolTokenCount(_ branch: BranchKey) -> Int {
        pendingSlot(branch).inputs.count
    }

    // MARK: - HybridPoolCache conformance
    //
    // 2026-05-04: forward the protocol API used by the disk serializer
    // and the paged cache to the existing `getPooled`/`setPooled` /
    // `getBuffers`/`setBuffers` accessors. The protocol exists so the
    // MLXLMCommon disk-cache subsystem can persist DSV4 layers without
    // an MLXLLM dependency.
    private static func bridge(_ branch: HybridPoolBranch) -> BranchKey {
        switch branch {
        case .compressor: return .compressor
        case .indexer: return .indexer
        }
    }

    public func hybridPool(branch: HybridPoolBranch) -> MLXArray? {
        flushPoolPending()
        return getPooled(Self.bridge(branch))
    }

    public func setHybridPool(branch: HybridPoolBranch, value: MLXArray?) {
        let key = Self.bridge(branch)
        if let value {
            setPooled(key, value: value)
        } else {
            switch key {
            case .compressor: compPool.replace(with: nil)
            case .indexer: idxPool.replace(with: nil)
            }
        }
    }

    public func hybridBuffers(branch: HybridPoolBranch) -> (kv: MLXArray?, gate: MLXArray?) {
        flushPoolPending()
        return getBuffers(Self.bridge(branch))
    }

    public func setHybridBuffers(branch: HybridPoolBranch, kv: MLXArray?, gate: MLXArray?) {
        setBuffers(Self.bridge(branch), kv: kv, gate: gate)
    }

    public func hybridPoolQuantizedSegments(
        branch: HybridPoolBranch
    ) -> [HybridPoolQuantizedSegment]? {
        flushPoolPending()
        let segments = branch == .compressor
            ? compPool.quantizedSegments
            : idxPool.quantizedSegments
        return segments.isEmpty ? nil : segments
    }

    public func setHybridPoolQuantizedSegments(
        branch: HybridPoolBranch,
        segments: [HybridPoolQuantizedSegment]
    ) {
        if branch == .compressor {
            compPool.replace(withQuantizedSegments: segments)
        } else {
            idxPool.replace(withQuantizedSegments: segments)
        }
    }

    public func hybridPoolRetainedByteCount(branch: HybridPoolBranch) -> Int {
        branch == .compressor
            ? compPool.retainedByteCount
            : idxPool.retainedByteCount
    }

    // MARK: - In-place speculative checkpoint (dSpark verify)

    private struct PendingSnapshot {
        let inputs: [MLXArray]
        let basePosition: Int
        let lastPool: DeepseekV4PoolView?
        let flush: ((MLXArray, Int) -> DeepseekV4PoolView)?
    }

    private struct SpeculativeSnapshot {
        let local: Any
        let compBufferKV: MLXArray?
        let compBufferGate: MLXArray?
        let idxBufferKV: MLXArray?
        let idxBufferGate: MLXArray?
        let compPool: DeepseekV4PoolStorage.Snapshot
        let idxPool: DeepseekV4PoolStorage.Snapshot
        let compPending: PendingSnapshot
        let idxPending: PendingSnapshot
        let history: HybridPoolTokenHistory?
        let hiddenTail: HybridPoolHiddenTail?
    }

    private static func pendingSnapshot(_ slot: PendingPoolSlot) -> PendingSnapshot {
        PendingSnapshot(
            inputs: slot.inputs, basePosition: slot.basePosition, lastPool: slot.lastPool,
            flush: slot.flush)
    }

    private static func loadPending(_ snapshot: PendingSnapshot, into slot: PendingPoolSlot) {
        slot.inputs = snapshot.inputs
        slot.basePosition = snapshot.basePosition
        slot.lastPool = snapshot.lastPool
        slot.flush = snapshot.flush
    }
}

extension DeepseekV4Cache: SpeculativeCheckpointCache {
    public var speculativeCheckpointIsExact: Bool { true }

    /// Every field a forward can change, by reference: the local window, both pools, the
    /// incomplete-window buffers, the deferred decode inputs, the Engram n-gram history and the
    /// dSpark hidden tail. Restoring writes them back into this object, so anything holding the
    /// cache (or keyed on it) sees the pre-verify state.
    public func speculativeCheckpoint() -> Any {
        SpeculativeSnapshot(
            local: local.speculativeCheckpoint(),
            compBufferKV: compBufferKV?[.ellipsis], compBufferGate: compBufferGate?[.ellipsis],
            idxBufferKV: idxBufferKV?[.ellipsis], idxBufferGate: idxBufferGate?[.ellipsis],
            compPool: compPool.snapshot(), idxPool: idxPool.snapshot(),
            compPending: Self.pendingSnapshot(compPending),
            idxPending: Self.pendingSnapshot(idxPending),
            history: auxiliaryTokenHistory, hiddenTail: auxiliaryHiddenTail)
    }

    public func restoreSpeculativeCheckpoint(_ checkpoint: Any) {
        guard let s = checkpoint as? SpeculativeSnapshot else {
            preconditionFailure("DeepseekV4Cache: foreign speculative checkpoint")
        }
        local.restoreSpeculativeCheckpoint(s.local)
        compBufferKV = s.compBufferKV
        compBufferGate = s.compBufferGate
        idxBufferKV = s.idxBufferKV
        idxBufferGate = s.idxBufferGate
        compPool.load(s.compPool)
        idxPool.load(s.idxPool)
        Self.loadPending(s.compPending, into: compPending)
        Self.loadPending(s.idxPending, into: idxPending)
        auxiliaryTokenHistory = s.history
        auxiliaryHiddenTail = s.hiddenTail
        // A restore moves the sequence back like a trim: the next-chunk Engram lookahead was keyed
        // to the forward that followed the checkpoint, so it is dropped (speed-only, transient).
        engramLookaheadTokens = nil
        engramLookahead = nil
    }
}

// MARK: - Compressor
//
// Projects input x through `wkv` + `wgate`, accumulates raw windows
// until a full `compress_ratio`-chunk is ready, then pools the chunk
// via softmax(gate)-weighted sum. For compress_ratio=4 the output is
// 2× widened (overlap mode) so each pool spans TWO adjacent chunks —
// strictly increases context coverage. After pooling, applies the
// absolute position embedding (APE) and partial RoPE at the chunk
// positions. Updates the per-layer V4Cache pool if provided.
/// Gate for the DSA dense-indexer score matmul input dtype. Default runs in
/// the pool's native dtype (no per-step full-pool fp32 copy); the env var
/// restores the historical fp32-input matmul for A/B and fallback.
enum DeepseekV4IndexerRuntime {
    nonisolated(unsafe) static var scoreFP32Inputs: Bool = {
        ProcessInfo.processInfo.environment["VMLX_DSA_INDEX_FP32"] == "1"
    }()
}

enum DeepseekV4CompressorQATKind {
    case attentionKV
    case indexer
}

public final class DeepseekV4Compressor: Module {
    let compressRatio: Int
    let headDim: Int
    let outDim: Int
    let overlap: Bool
    let rmsNormEps: Float
    let qatKind: DeepseekV4CompressorQATKind
    let activationQATEnabled: Bool
    let deferPoolDuringDecode: Bool

    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "wgate") var wgate: Linear
    private let regionLock = NSLock()
    private var frontOverlapRegion: DeepseekV4CompiledRegion? = nil
    private var frontRegion: DeepseekV4CompiledRegion? = nil
    /// APE: (compress_ratio, out_dim) learned positional bias inside
    /// each pool window.
    @ParameterInfo(key: "ape") var ape: MLXArray
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(
        config: DeepseekV4Configuration,
        compressRatio: Int,
        headDim: Int,
        qatKind: DeepseekV4CompressorQATKind = .attentionKV,
        deferPoolDuringDecode: Bool? = nil
    ) {
        self.compressRatio = compressRatio
        self.headDim = headDim
        self.rmsNormEps = config.rmsNormEps
        self.qatKind = qatKind
        self.activationQATEnabled = config.activationQATEnabled
        self.deferPoolDuringDecode = deferPoolDuringDecode
            ?? Self.resolvePoolDeferralDefault()
        self.overlap = compressRatio == 4
        self.outDim = headDim * (overlap ? 2 : 1)
        self._wkv.wrappedValue = Linear(config.hiddenSize, outDim, bias: false)
        self._wgate.wrappedValue = Linear(config.hiddenSize, outDim, bias: false)
        self._ape.wrappedValue = zeros([compressRatio, outDim])
        self._norm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    static func resolvePoolDeferralDefault(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        guard let raw = environment["DSV4_DEFER_POOL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        else { return true }
        return !["0", "false", "off", "no"].contains(raw)
    }

    /// Forward. Returns the branch's active pool representation (dense while
    /// small, bounded view once promoted). When `v4Cache` is provided, newly
    /// completed windows are appended to the cache pool first.
    func callAsFunction(
        _ x: MLXArray,
        rope: DeepseekV4RoPE,
        v4Cache: DeepseekV4Cache?,
        startPos: Int,
        branch: DeepseekV4Cache.BranchKey = .compressor,
        latentSink: ((MLXArray, Int) -> Void)? = nil
    ) -> DeepseekV4PoolView {
        if deferPoolDuringDecode,
           x.dim(0) == 1,
           x.dim(1) == 1,
           let cache = v4Cache
        {
            return cache.deferPoolToken(
                x,
                branch: branch,
                startPos: startPos,
                ratio: compressRatio,
                overlap: overlap
            ) { [unowned self, weak cache] pending, basePosition in
                guard let cache else {
                    fatalError("DSV4 cache released while flushing deferred pool state")
                }
                // Deferred tokens are decode tokens: the flush may carry up
                // to `ratio` of them, but the consuming forward has L == 1,
                // so the view decision must stay on the decode rules.
                return self.advance(
                    pending,
                    rope: rope,
                    v4Cache: cache,
                    startPos: basePosition,
                    branch: branch,
                    isPrefill: false,
                    latentSink: latentSink)
            }
        }

        // A multi-token continuation begins after any uncommitted decode
        // tail. Flush both compressor branches first to preserve window order.
        v4Cache?.flushPoolPending()
        let result = advance(
            x, rope: rope, v4Cache: v4Cache,
            startPos: startPos, branch: branch,
            isPrefill: x.dim(1) > 1,
            latentSink: latentSink)
        v4Cache?.notePoolResult(result, branch: branch)
        return result
    }

    /// Real compressor advancement. Deferred decode flushes call this entry
    /// directly so the batched boundary update cannot recursively defer.
    private func advance(
        _ x: MLXArray,
        rope: DeepseekV4RoPE,
        v4Cache: DeepseekV4Cache?,
        startPos: Int,
        branch: DeepseekV4Cache.BranchKey,
        isPrefill: Bool,
        latentSink: ((MLXArray, Int) -> Void)? = nil
    ) -> DeepseekV4PoolView {
        let B = x.dim(0)
        let L = x.dim(1)
        // Python `_decode_front(_overlap)` parity: compiled wkv/wgate front
        // (+ APE fold for overlap) for single-token decode and the small
        // ≤ratio flushes pool deferral emits. Each distinct flush length adds
        // one trace (k ≤ 4 for overlap); ratio-128 flushes stay eager — up to
        // 128 distinct shapes would each retrace.
        let useCompiledFront =
            DeepseekV4Math.compileRegionsEnabled
            && v4Cache != nil
            && (L == 1 || (overlap && L <= compressRatio))
            && !CompiledDecodeTrace.isActive

        var kv: MLXArray
        var gate: MLXArray
        var apeFolded = false
        if useCompiledFront, overlap {
            let ratio = Int32(compressRatio)
            let region = deepseekV4CachedRegion(regionLock, &frontOverlapRegion) {
                vmlxTrustedCompile { [unowned self] (args: [MLXArray]) -> [MLXArray] in
                    CompiledDecodeTrace.withActive {
                        // Official 0731 stages the compressor projections in
                        // fp32 — the cast must precede the linears.
                        let x32 = args[0].asType(.float32)
                        let kvOut = self.wkv(x32)
                        let gateOut = self.wgate(x32)
                        let apeRows = self.ape.asType(gateOut.dtype)[args[1] % ratio]
                        return [kvOut, gateOut + apeRows.expandedDimensions(axis: 0)]
                    }
                }
            }
            let pos = MLXArray(Int32(startPos)..<Int32(startPos + L))
            let outs = region([x, pos])
            if outs.count == 2 {
                kv = outs[0]
                gate = outs[1]
                apeFolded = true
            } else {
                let projectionInput = x.asType(.float32)
                kv = wkv(projectionInput)
                gate = wgate(projectionInput)
            }
        } else if useCompiledFront {
            let region = deepseekV4CachedRegion(regionLock, &frontRegion) {
                vmlxTrustedCompile { [unowned self] (args: [MLXArray]) -> [MLXArray] in
                    CompiledDecodeTrace.withActive {
                        let x32 = args[0].asType(.float32)
                        return [self.wkv(x32), self.wgate(x32)]
                    }
                }
            }
            let outs = region([x])
            if outs.count == 2 {
                kv = outs[0]
                gate = outs[1]
            } else {
                let projectionInput = x.asType(.float32)
                kv = wkv(projectionInput)
                gate = wgate(projectionInput)
            }
        } else {
            // Official 0731 stages the compressor projections and pooling
            // state in fp32. This must happen before the linears, not merely
            // after them.
            let projectionInput = x.asType(.float32)
            kv = wkv(projectionInput)
            gate = wgate(projectionInput)
        }

        var poolBase = startPos
        let alreadyWindowed: Bool
        if let cache = v4Cache, overlap {
            if !apeFolded {
                let positions = MLXArray(
                    Int32(startPos)..<Int32(startPos + gate.dim(1)))
                let apeRows = ape.asType(gate.dtype)[positions % Int32(compressRatio)]
                gate = gate + apeRows.expandedDimensions(axis: 0)
            }
            let accumulated = accumulateOverlapWindows(
                kv: kv, gate: gate, cache: cache, branch: branch,
                ratio: compressRatio, startPos: startPos)
            kv = accumulated.kvRows
            gate = accumulated.gateRows
            poolBase = accumulated.poolBase
            alreadyWindowed = true
        } else {
            // Accumulate windows. When cache present, prepend unused-tail
            // buffers from prior calls.
            if let cache = v4Cache {
                let (bufKV, bufGate) = cache.getBuffers(branch)
                if let bKV = bufKV, bKV.dim(1) > 0, let bG = bufGate {
                    kv = concatenated([bKV, kv], axis: 1)
                    gate = concatenated([bG, gate], axis: 1)
                    poolBase -= bKV.dim(1)
                }
                let total = kv.dim(1)
                let usable = (total / compressRatio) * compressRatio
                // Stash the tail for the next call.
                let tailKV = usable < total ? kv[0..., usable..., 0...] : nil
                let tailGate = usable < total ? gate[0..., usable..., 0...] : nil
                cache.setBuffers(branch, kv: tailKV, gate: tailGate)
                kv = kv[0..., 0..<usable, 0...]
                gate = gate[0..., 0..<usable, 0...]
            } else {
                let total = kv.dim(1)
                let usable = (total / compressRatio) * compressRatio
                kv = kv[0..., 0..<usable, 0...]
                gate = gate[0..., 0..<usable, 0...]
            }
            alreadyWindowed = false
        }

        if kv.dim(1) == 0 {
            let empty = MLXArray.zeros([B, 0, headDim], dtype: x.dtype)
            if let cache = v4Cache {
                return cache.poolView(branch, isPrefill: isPrefill, emptyLike: empty)
            }
            return DeepseekV4PoolView(dense: empty)
        }

        let W: Int
        var kvWin: MLXArray
        var gateWin: MLXArray
        if alreadyWindowed {
            W = kv.dim(1)
            kvWin = kv
            gateWin = gate
        } else {
            W = kv.dim(1) / compressRatio
            kvWin = kv.reshaped(B, W, compressRatio, outDim)
            gateWin =
                gate.reshaped(B, W, compressRatio, outDim) + ape.asType(gate.dtype)

            if overlap {
                kvWin = overlapTransform(kvWin, fillValue: 0.0)
                // For gate, the pre-allocated fill is -inf so softmax assigns
                // zero mass to the padding half.
                gateWin = overlapTransform(
                    gateWin, fillValue: -Float.infinity)
            }
        }

        let weights =
            softmax(gateWin.asType(.float32), axis: 2, precise: true).asType(
                kvWin.dtype)
        var pooled = (kvWin * weights).sum(axis: 2)
        pooled = norm(pooled.asType(x.dtype))
        // DeepSeek-V4.1: the pre-RoPE latents of the newly completed groups feed the shared
        // lightning indexer's key projection (reference `Indexer.__call__`: k = k_norm(wk(latent))).
        latentSink?(pooled, poolBase)

        // Apply RoPE at the chunk centers (position = chunk_idx * ratio
        // + pool_base). Shared strided table: same-ratio compressor
        // instances all ask for the identical window per forward pass.
        let (cosT, sinT) = rope.cosSin(
            base: poolBase, count: pooled.dim(1), stride: compressRatio)
        let cosP = cosT.expandedDimensions(axes: [0])
        let sinP = sinT.expandedDimensions(axes: [0])
        pooled = DeepseekV4Math.applyPartialRoPE(
            pooled, cos: cosP, sin: sinP, ropeDim: rope.dim)

        if activationQATEnabled {
            switch qatKind {
            case .attentionKV:
                let nopeDim = headDim - rope.dim
                if nopeDim >= 64 {
                    pooled = DeepseekV4Math.e4m3KVActivationRoundTrip(
                        pooled, ropeDim: rope.dim)
                }
            case .indexer:
                if headDim == 128 {
                    pooled = DeepseekV4Math.indexerActivationRoundTrip(pooled)
                }
            }
        }

        if let cache = v4Cache {
            return cache.appendPooled(branch, value: pooled, isPrefill: isPrefill)
        }
        return DeepseekV4PoolView(dense: pooled)
    }

    /// Ratio-4 overlap accumulation for the stateful decode path.
    ///
    /// DSV4's overlap compressor needs the previous complete window for the
    /// left half of the next pooled row. A plain remainder buffer works during
    /// large prefill chunks but corrupts single-token decode: every completed
    /// decode row would otherwise get a zero left half at the call boundary.
    func accumulateOverlapWindows(
        kv: MLXArray,
        gate: MLXArray,
        cache: DeepseekV4Cache,
        branch: DeepseekV4Cache.BranchKey,
        ratio: Int,
        startPos: Int
    ) -> (kvRows: MLXArray, gateRows: MLXArray, poolBase: Int) {
        let B = kv.dim(0)

        func emptyRows(_ poolBase: Int) -> (kvRows: MLXArray, gateRows: MLXArray, poolBase: Int) {
            (
                MLXArray.zeros([B, 0, 2 * ratio, headDim], dtype: kv.dtype),
                MLXArray.zeros([B, 0, 2 * ratio, headDim], dtype: gate.dtype),
                poolBase
            )
        }

        func makeRow(
            prevKV: MLXArray?,
            prevGate: MLXArray?,
            curKV: MLXArray,
            curGate: MLXArray
        ) -> (MLXArray, MLXArray) {
            let leftKV: MLXArray
            let leftGate: MLXArray
            if let prevKV, let prevGate {
                leftKV = prevKV[0..., 0..., 0..<headDim]
                leftGate = prevGate[0..., 0..., 0..<headDim]
            } else {
                leftKV = MLXArray.zeros([B, ratio, headDim], dtype: kv.dtype)
                leftGate = MLXArray.full(
                    [B, ratio, headDim],
                    values: MLXArray(-Float.infinity).asType(gate.dtype))
            }
            let rightKV = curKV[0..., 0..., headDim...]
            let rightGate = curGate[0..., 0..., headDim...]
            return (
                concatenated([leftKV, rightKV], axis: 1).expandedDimensions(axis: 1),
                concatenated([leftGate, rightGate], axis: 1).expandedDimensions(axis: 1)
            )
        }

        if startPos == 0 {
            let usable = (kv.dim(1) / ratio) * ratio
            let remainderKV = usable < kv.dim(1) ? kv[0..., usable..., 0...] : nil
            let remainderGate = usable < gate.dim(1) ? gate[0..., usable..., 0...] : nil
            if usable >= ratio {
                let lastKV = kv[0..., (usable - ratio)..<usable, 0...]
                let lastGate = gate[0..., (usable - ratio)..<usable, 0...]
                cache.setBuffers(
                    branch,
                    kv: remainderKV != nil
                        ? concatenated([lastKV, remainderKV!], axis: 1)
                        : lastKV,
                    gate: remainderGate != nil
                        ? concatenated([lastGate, remainderGate!], axis: 1)
                        : lastGate)
            } else {
                cache.setBuffers(branch, kv: remainderKV, gate: remainderGate)
            }
            guard usable > 0 else { return emptyRows(startPos) }
            let W = usable / ratio
            let fullKV = kv[0..., 0..<usable, 0...].reshaped(B, W, ratio, outDim)
            let fullGate = gate[0..., 0..<usable, 0...].reshaped(B, W, ratio, outDim)
            return (
                overlapTransform(fullKV, fillValue: 0.0),
                overlapTransform(fullGate, fillValue: -Float.infinity),
                startPos
            )
        }

        let (bufKV, bufGate) = cache.getBuffers(branch)
        var prevKV: MLXArray? = nil
        var prevGate: MLXArray? = nil
        var partialKV = bufKV
        var partialGate = bufGate
        if let bKV = bufKV, bKV.dim(1) >= ratio, let bGate = bufGate {
            prevKV = bKV[0..., 0..<ratio, 0...]
            prevGate = bGate[0..., 0..<ratio, 0...]
            partialKV = bKV.dim(1) > ratio ? bKV[0..., ratio..., 0...] : nil
            partialGate = bGate.dim(1) > ratio ? bGate[0..., ratio..., 0...] : nil
        }

        let priorPartialLen = partialKV?.dim(1) ?? 0
        var currentKV =
            if let partialKV, partialKV.dim(1) > 0 {
                concatenated([partialKV, kv], axis: 1)
            } else {
                kv
            }
        var currentGate =
            if let partialGate, partialGate.dim(1) > 0 {
                concatenated([partialGate, gate], axis: 1)
            } else {
                gate
            }

        var kvRows: [MLXArray] = []
        var gateRows: [MLXArray] = []
        while currentKV.dim(1) >= ratio {
            let curKV = currentKV[0..., 0..<ratio, 0...]
            let curGate = currentGate[0..., 0..<ratio, 0...]
            let row = makeRow(prevKV: prevKV, prevGate: prevGate, curKV: curKV, curGate: curGate)
            kvRows.append(row.0)
            gateRows.append(row.1)
            prevKV = curKV
            prevGate = curGate
            currentKV = currentKV.dim(1) > ratio ? currentKV[0..., ratio..., 0...] :
                MLXArray.zeros([B, 0, outDim], dtype: kv.dtype)
            currentGate = currentGate.dim(1) > ratio ? currentGate[0..., ratio..., 0...] :
                MLXArray.zeros([B, 0, outDim], dtype: gate.dtype)
        }

        if let prevKV, let prevGate {
            cache.setBuffers(
                branch,
                kv: currentKV.dim(1) > 0 ? concatenated([prevKV, currentKV], axis: 1) : prevKV,
                gate: currentGate.dim(1) > 0 ? concatenated([prevGate, currentGate], axis: 1) : prevGate)
        } else {
            cache.setBuffers(
                branch,
                kv: currentKV.dim(1) > 0 ? currentKV : nil,
                gate: currentGate.dim(1) > 0 ? currentGate : nil)
        }

        let poolBase = max(0, startPos - priorPartialLen)
        guard !kvRows.isEmpty else { return emptyRows(poolBase) }
        return (concatenated(kvRows, axis: 1), concatenated(gateRows, axis: 1), poolBase)
    }

    /// Overlap transform for compress_ratio=4. Expands (B, W, R, D) to
    /// (B, W, 2R, D) where the first R columns are the first half of
    /// the previous window's output and the last R columns are the
    /// current window's second half — gives each pool access to both
    /// chunks.
    private func overlapTransform(_ x: MLXArray, fillValue: Float) -> MLXArray {
        let B = x.dim(0)
        let W = x.dim(1)
        let R = x.dim(2)
        // Build output in two halves via concatenate; avoids
        // mutable-assignment patterns that mlx-swift Module doesn't
        // provide a clean API for.
        //
        // Layout:
        //   out[:, 0, :R]   = fill
        //   out[:, 1:, :R]  = x[:, :-1, :, :headDim]
        //   out[:,  :, R:]  = x[:, :, :, headDim:]
        let firstHalfAll = x[0..., 0..., 0..., 0..<headDim]  // (B, W, R, hd)
        // Shift: prepend a fill-window at position 0.
        let fillWindow = MLXArray.full(
            [B, 1, R, headDim], values: MLXArray(fillValue).asType(x.dtype))
        let shifted = concatenated(
            [fillWindow, firstHalfAll[0..., 0..<(W - 1), 0..., 0...]],
            axis: 1)  // (B, W, R, hd)
        let secondHalfAll = x[0..., 0..., 0..., headDim...]  // (B, W, R, hd)
        // Concat along the R axis: (B, W, 2R, hd).
        return concatenated([shifted, secondHalfAll], axis: 2)
    }
}

// MARK: - Indexer

/// Per-query top-k selector over the Compressor's pooled output.
/// Only present on compress_ratio=4 layers. Given `x` and `q_residual`
/// (the post-`q_norm` low-rank Q), projects Q into `n_heads`×`head_dim`,
/// scores against the Compressor's pooled keys, weights by a per-head
/// coefficient from `weights_proj`, and returns the top-`index_topk`
/// indices per query position.
public final class DeepseekV4Indexer: Module {
    let nHeads: Int
    let headDim: Int
    let topK: Int
    let compressRatio: Int
    let scale: Float
    let activationQATEnabled: Bool

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear
    @ModuleInfo(key: "compressor") var compressor: DeepseekV4Compressor

    init(config: DeepseekV4Configuration, compressRatio: Int) {
        self.nHeads = config.indexNHeads
        self.headDim = config.indexHeadDim
        self.topK = config.indexTopk
        self.compressRatio = compressRatio
        self.scale = 1.0 / sqrt(Float(headDim))
        self.activationQATEnabled = config.activationQATEnabled
        self._wqB.wrappedValue = Linear(
            config.qLoraRank, nHeads * headDim, bias: false)
        self._weightsProj.wrappedValue = Linear(
            config.hiddenSize, nHeads, bias: false)
        self._compressor.wrappedValue = DeepseekV4Compressor(
            config: config, compressRatio: compressRatio, headDim: headDim,
            qatKind: .indexer)
    }

    /// Always advances the indexer's private compressor branch. Returns top-k
    /// indices shape (B, L, k) only after the pool exceeds `topK`; while every
    /// row remains visible it returns nil without running the score path.
    func callAsFunction(
        _ x: MLXArray,
        qResidual: MLXArray,
        rope: DeepseekV4RoPE,
        positionRope: DeepseekV4RoPE,
        v4Cache: DeepseekV4Cache?,
        startPos: Int
    ) -> MLXArray? {
        let pooledView = compressor(
            x, rope: rope, v4Cache: v4Cache,
            startPos: startPos, branch: .indexer)
        let pooledLen = pooledView.rowCount
        if pooledLen == 0 { return nil }

        let B = x.dim(0)
        let L = x.dim(1)
        // Keep A3's score-path optimization without skipping the stateful
        // compressor above. Returning nil tells attention to use every pooled
        // row directly; no Q projection, score matmul, or arg-partition runs.
        // This branch must occur only after `compressor(...)` so incremental
        // 2048 -> 2052 token growth has all 512 historical index rows when row
        // 513 first activates learned top-k selection.
        if pooledLen <= topK {
            return nil
        }

        var q = wqB(qResidual)
            .reshaped(B, L, nHeads, headDim)
            .transposed(0, 2, 1, 3)
        // Partial RoPE on Q using the plain (non-compressor) RoPE.
        let (cosT, sinT) = positionRope.cosSin(offset: startPos, length: L)
        let cosQ = cosT.expandedDimensions(axes: [0, 1])
        let sinQ = sinT.expandedDimensions(axes: [0, 1])
        q = DeepseekV4Math.applyPartialRoPE(
            q, cos: cosQ, sin: sinQ, ropeDim: rope.dim)
        if activationQATEnabled && headDim == 128 {
            q = DeepseekV4Math.indexerActivationRoundTrip(q)
        }

        // weights: (B, L, nHeads) * n_heads^-0.5.
        let wRaw = weightsProj(x).asType(.float32)
            * MLXArray(1.0 / sqrt(Float(nHeads)))

        guard let pooled = pooledView.denseArray else {
            // Segmented q8 pool, or an oversized BF16 hot tier during
            // prefill: exact global top-k over bounded dequantized tiles so
            // the fp32 score matrix never spans the whole pool (Python
            // mirror: `_dsv4_tiled_index_topk`).
            return DeepseekV4Math.tiledIndexTopk(
                queries: q,
                headWeights: wRaw,
                pooled: pooledView,
                scale: scale,
                topK: topK,
                offset: startPos,
                ratio: compressRatio)
        }

        // scores: (B, nHeads, L, pooledLen). Match Python shape.
        // q is (B, nHeads, L, headDim); pooled is (B, pooledLen, headDim).
        // Expand pooled to (B, 1, pooledLen, headDim) for broadcast.
        //
        // The matmul runs in the pool's native dtype: casting `pooled` to
        // fp32 materialized a full-pool fp32 copy on EVERY decode step
        // (linear-in-context traffic per token — the same tax class as the
        // MLA decode cast fixed in AttentionUtils). Metal GEMM accumulates
        // in fp32 internally, and these are SELECTION logits: bf16 storage
        // rounding (~2^-8 relative) can only flip near-tie block choices,
        // and the fp32 head-weighted sum below happens after the cast.
        // `VMLX_DSA_INDEX_FP32=1` restores the fp32-input matmul for A/B.
        let pooledBroad = pooled.expandedDimensions(axis: 1)
        var scores: MLXArray
        if DeepseekV4IndexerRuntime.scoreFP32Inputs {
            scores = q.asType(.float32).matmul(
                pooledBroad.asType(.float32).swappedAxes(-1, -2))
        } else {
            scores = q.asType(pooled.dtype).matmul(
                pooledBroad.swappedAxes(-1, -2)).asType(.float32)
        }
        scores = maximum(scores, MLXArray(0.0)) * MLXArray(scale)

        // Reshape scores sum axis: (B, 1, L, nHeads) multiply → sum.
        let wExpanded = wRaw.swappedAxes(-1, -2).expandedDimensions(axis: -1)
        scores = (scores * wExpanded).sum(axis: 1)  // (B, L, pooledLen)
        scores = DeepseekV4Math.causalMaskedIndexerScores(
            scores, offset: startPos, ratio: compressRatio)

        let k = min(topK, pooled.dim(1))
        let topIdx = argPartition(-scores, kth: k - 1, axis: -1)[
            .ellipsis, 0..<k]
        return topIdx
    }
}

extension DeepseekV4Cache: MetalBufferRetainingCache {
    /// Measured on DSV4-Flash (Python vMLX R21-13): the cumulative
    /// compressor/indexer pool state retains ~one live Metal buffer per
    /// layer per decode token, and `clear_cache` frees none of it.
    public var retainedBuffersPerDecodeToken: Double { 1.0 }
}
