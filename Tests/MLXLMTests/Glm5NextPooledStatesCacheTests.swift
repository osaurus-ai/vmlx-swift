// Copyright © 2026 osaurus-eval contributors
// SPDX-License-Identifier: MIT
//
// GLM-5.3's DSA indexer: the derived pooled-key lane in `Glm5NextIndexedKVCache`.
//
// `pooledStates` used to re-pool the WHOLE indexer history on every decode step, a cost that grows
// with context in each of the 11 DSA layers. The lane keeps the pooled keys of COMPLETE pools, so
// a step pools only the rows it lacks. Every pooled result is compared against
// `frozenUpstreamPooledStates`, a verbatim copy of the function as it was before the lane existed,
// so the code under test cannot grade itself. Every comparison is BITWISE: the lane is exact by
// construction, and a tolerance would hide the one failure that matters.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

// MARK: - The oracle

extension Glm5NextIndexer {

    /// `pooledStates(packed:)` exactly as it is at osaurus-ai/vmlx-swift `de579d1f`
    /// (`Libraries/MLXVLM/Models/Glm5Next.swift`, lines 1226-1303), before the pooled-key lane
    /// existed.
    ///
    /// FROZEN. Do not edit, except to re-freeze it against a newer upstream commit. Both the lane
    /// and the refactored full path are compared against it, so it must share code with neither.
    /// The formatter is told to leave it alone, so it keeps upstream's lines exactly as they were.
    // swift-format-ignore
    func frozenUpstreamPooledStates(
        packed: MLXArray
    ) throws -> (keys: MLXArray, indices: MLXArray, valid: MLXArray) {
        let B = packed.dim(0)
        let N = packed.dim(1)
        let keys = packed[.ellipsis, ..<headDim]
        let gate = packed[.ellipsis, headDim ..< (2 * headDim)]
        let valid = packed[.ellipsis, 2 * headDim] .!= MLXArray(Float(0))

        // Pool layout is shared across the batch only when every sequence starts at the same slot.
        // Without padding that is trivially true; with it, per-sequence gathers would be required,
        // and producing one shared layout anyway would quietly mis-pool the padded rows.
        let anyValid = valid.any(axis: -1)
        let firstKey = MLX.where(anyValid, valid.asType(.int32).argMax(axis: -1), MLXArray(Int32(N)))
        let start: MLXArray
        if B == 1 {
            // A single sequence always has a shared layout. Keep its offset on the device:
            // reading it on the host would evaluate the preceding model graph at every indexer.
            start = firstKey[0].asType(.int32)
        } else {
            let firstValues = firstKey.asType(.int32).asArray(Int32.self)
            guard let first = firstValues.first, firstValues.allSatisfy({ $0 == first }) else {
                throw Glm5NextDecoderUnavailable(
                    detail:
                        "the sequences in this batch begin at different offsets (padding); the indexer's "
                        + "pool layout is shared across the batch and cannot express that")
            }
            start = MLXArray(first)
        }

        let poolCount = (N + poolSize - 1) / poolSize
        let offsets = Glm5NextIndexerRuntime.layoutRange(poolCount * poolSize)
            .reshaped(poolCount, poolSize)
        let poolIndices = offsets + start
        let safe = clip(poolIndices, min: MLXArray(Int32(0)), max: MLXArray(Int32(N - 1)))
        let flatSafe = safe.reshaped(-1)

        let groupedKeys = take(keys, flatSafe, axis: 1)
            .reshaped(B, poolCount, poolSize, headDim)
        let groupedGate = take(gate, flatSafe, axis: 1)
            .reshaped(B, poolCount, poolSize, headDim)
        var groupedValid = take(valid, flatSafe, axis: 1)
            .reshaped(B, poolCount, poolSize)
        groupedValid = groupedValid .&& (poolIndices .< MLXArray(Int32(N)))

        let poolValid = groupedValid.all(axis: -1)

        guard let ape = poolPositionalCode else {
            throw Glm5NextDecoderUnavailable(
                detail: "index_kpool_compress is off; pooled selection needs the compression code")
        }
        // A LEARNED weighted average within each pool: softmax over the POOL-MEMBER axis, per
        // feature channel — not one scalar weight per member.
        //
        // The gathers above span the ENTIRE packed index history and this
        // runs every decode step, so fp32 casts here materialized two
        // full-context fp32 tensors per token (the MLA-cast tax class).
        // The softmax reduces over the tiny pool-member axis only, and the
        // pooled keys feed SELECTION scores, so native-dtype math changes
        // at most near-tie pool choices. `VMLX_GLM5_INDEX_FP32=1` restores
        // the fp32 pipeline for A/B.
        let poolMathDtype: DType =
            Glm5NextIndexerRuntime.poolFP32 ? .float32 : packed.dtype
        var logits = groupedGate.asType(poolMathDtype)
            + ape.asType(poolMathDtype)[.newAxis, .newAxis]
        logits = MLX.where(
            groupedValid.expandedDimensions(axis: -1), logits,
            MLXArray(-Float.greatestFiniteMagnitude).asType(poolMathDtype))
        var probabilities = softmax(logits, axis: 2)
        // A fully invalid pool softmaxes -inf against -inf and yields NaN; it is masked out of the
        // selection anyway, but a NaN here would poison the pooled key and, through it, the scores
        // of every VALID pool in the same matmul.
        probabilities = MLX.where(
            isNaN(probabilities), MLXArray(Float(0)).asType(poolMathDtype), probabilities)
        let pooled = (probabilities * groupedKeys.asType(poolMathDtype)).sum(axis: 2)

        return (pooled.asType(packed.dtype), poolIndices, poolValid)
    }
}

// MARK: - Suite

@Suite("GLM-5.3 indexer pooled-key lane", .serialized)
struct Glm5NextPooledStatesCacheTests {

    typealias Pooled = (keys: MLXArray, indices: MLXArray, valid: MLXArray)

    // MARK: Fixtures

    /// A small indexer with RANDOM parameters, the pooling ones included.
    ///
    /// `Glm5NextIndexer` initialises `index_kpool_compress_ape` and `_gate` to zeros, which makes
    /// every member softmax uniform, so a test on the defaults could not see a wrong member order
    /// or axis. The update is verified, so a misspelt key fails here instead of staying zero.
    static func makeIndexer(
        seed: UInt64, dtype: DType = .float32, headDim: Int = 32, numHeads: Int = 4,
        topK: Int = 16, hidden: Int = 64, qLoraRank: Int = 48
    ) throws -> Glm5NextIndexer {
        let poolSize = 4
        let ix = Glm5NextIndexer(
            numHeads: numHeads, headDim: headDim, topK: topK, poolSize: poolSize,
            poolCompressed: true, alwaysSelectTail: true, hiddenSize: hidden,
            qLoraRank: qLoraRank)
        func random(_ shape: [Int], _ salt: UInt64, scale: Float) -> MLXArray {
            (MLXRandom.normal(shape, key: MLXRandom.key(seed &* 100 &+ salt)) * scale)
                .asType(dtype)
        }
        let perHidden = 1 / Float(hidden).squareRoot()
        let perRank = 1 / Float(qLoraRank).squareRoot()
        try ix.update(
            parameters: ModuleParameters.unflattened([
                "wq_b.weight": random([numHeads * headDim, qLoraRank], 1, scale: perRank),
                "wk.weight": random([headDim, hidden], 2, scale: perHidden),
                "k_norm.weight": (MLXArray.ones([headDim]) + random([headDim], 3, scale: 0.1))
                    .asType(dtype),
                "k_norm.bias": random([headDim], 4, scale: 0.1),
                "weights_proj.weight": random([numHeads, hidden], 5, scale: perHidden),
                "index_kpool_compress_ape": random([poolSize, headDim], 6, scale: 1),
                "index_kpool_compress_gate": random([headDim, hidden], 7, scale: perHidden),
            ]), verify: [.noUnusedKeys, .shapeMismatch])
        return ix
    }

    // MARK: Comparison

    static func bits(_ a: MLXArray) -> [UInt32] {
        a.asType(.float32).asArray(Float.self).map(\.bitPattern)
    }

    /// Where two arrays first differ, or nil when they are identical. Reports one element, not
    /// two arrays of thousands.
    static func firstMismatch<T: Equatable>(_ x: [T], _ y: [T]) -> String? {
        guard x != y else { return nil }
        guard x.count == y.count else { return "\(x.count) elements against \(y.count)" }
        let i = x.indices.first { x[$0] != y[$0] }!
        return "element \(i) of \(x.count): \(x[i]) against \(y[i])"
    }

    /// The same for float32 bit patterns, shown as the values they encode and as hex.
    static func firstMismatch(_ x: [UInt32], _ y: [UInt32]) -> String? {
        guard x != y else { return nil }
        guard x.count == y.count else { return "\(x.count) elements against \(y.count)" }
        let i = x.indices.first { x[$0] != y[$0] }!
        func show(_ b: UInt32) -> String { "\(Float(bitPattern: b)) (0x\(String(b, radix: 16)))" }
        return "element \(i) of \(x.count): \(show(x[i])) against \(show(y[i]))"
    }

    static func expectIdentical(_ got: Pooled, _ want: Pooled, _ context: String) {
        #expect(got.keys.shape == want.keys.shape, "\(context): key shape")
        #expect(got.keys.dtype == want.keys.dtype, "\(context): key dtype")
        #expect(got.indices.shape == want.indices.shape, "\(context): index shape")
        #expect(got.indices.dtype == want.indices.dtype, "\(context): index dtype")
        #expect(got.valid.shape == want.valid.shape, "\(context): validity shape")
        #expect(got.valid.dtype == want.valid.dtype, "\(context): validity dtype")
        guard got.keys.shape == want.keys.shape, got.indices.shape == want.indices.shape,
            got.valid.shape == want.valid.shape
        else { return }
        #expect(
            firstMismatch(bits(got.keys), bits(want.keys)) == nil,
            "\(context): pooled keys (float32 bit patterns)")
        #expect(
            firstMismatch(got.indices.asArray(Int32.self), want.indices.asArray(Int32.self))
                == nil, "\(context): pool indices")
        #expect(
            firstMismatch(got.valid.asArray(Bool.self), want.valid.asArray(Bool.self)) == nil,
            "\(context): pool validity")
    }

    // MARK: T0, the full path

    /// The validity patterns of `Glm5NextIndexerSelectionTests`'
    /// `singleRowDeviceOffsetsMatchValidatedBatch`, verbatim.
    static let upstreamMasks: [[Float]] = [
        [1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0], [1], [0],
    ]

    /// The same kinds of pattern at lengths where the pool count, and with it an MLX kernel's
    /// strategy, could matter. The producer only ever writes the first.
    static func patterns(_ n: Int) -> [(name: String, mask: [Float])] {
        var out: [(name: String, mask: [Float])] = [("all valid", Array(repeating: 1, count: n))]
        if n > 2 {
            out.append(("padded prefix", (0 ..< n).map { $0 < 2 ? 0 : 1 }))
            out.append(("interior gaps", (0 ..< n).map { $0 % 3 == 1 ? 0 : 1 }))
        }
        if n >= 8 {
            out.append(("one invalid pool", (0 ..< n).map { (4 ..< 8).contains($0) ? 0 : 1 }))
        }
        out.append(("all invalid", Array(repeating: 0, count: n)))
        return out
    }

    /// T0. The full path is the frozen function bit for bit, across every validity pattern,
    /// dtype, `VMLX_GLM5_INDEX_FP32` setting and batch width.
    ///
    /// The case no earlier test pins is a fully invalid pool under float32 pool math. There the
    /// mask value `-greatestFiniteMagnitude` is FINITE, so the softmax is uniform and the pooled
    /// key is the mean of its members' keys. In float16 and bfloat16 the mask rounds to -inf, and
    /// the NaN guard zeroes the result.
    @Test("T0: the full path is the frozen upstream function, bit for bit")
    func fullPathMatchesFrozenUpstream() throws {
        try MLXMetalTestLock.withLock {
            let saved = Glm5NextIndexerRuntime.poolFP32
            defer { Glm5NextIndexerRuntime.poolFP32 = saved }
            let ix = try Self.makeIndexer(seed: 1)
            let D = ix.headDim
            var cases = Self.upstreamMasks.map { (name: "upstream mask \($0)", mask: $0) }
            for n in [1, 3, 4, 5, 64, 4097, 16384] {
                cases += Self.patterns(n).map { (name: "N=\(n) \($0.name)", mask: $0.mask) }
            }
            #expect(cases.count == 40)
            var compared = 0
            for (i, c) in cases.enumerated() {
                let n = c.mask.count
                // Two DIFFERENT sequences with the SAME mask. Unequal first-valid slots would
                // make both arms throw (the batch-consistency check) instead of comparing.
                let data = MLXRandom.normal([2, n, 2 * D], key: MLXRandom.key(UInt64(1000 + i)))
                let validity = MLXArray(c.mask, [1, n, 1])
                let rows = concatenated(
                    [data, concatenated([validity, validity], axis: 0)], axis: -1)
                for dtype: DType in [.float32, .float16, .bfloat16] {
                    for fp32 in [false, true] {
                        Glm5NextIndexerRuntime.poolFP32 = fp32
                        for batch in [1, 2] {
                            let packed = rows[0 ..< batch].asType(dtype)
                            let want = try ix.frozenUpstreamPooledStates(packed: packed)
                            let got = try ix.pooledStates(packed: packed)
                            Self.expectIdentical(
                                got, want, "\(c.name) \(dtype) poolFP32=\(fp32) B=\(batch)")
                            compared += 1
                        }
                    }
                }
            }
            #expect(compared == 480)
        }
    }

    // MARK: Cache state (C1–C8): the lane's lifecycle, without the indexer

    /// The KV half for `rows` new positions. The indexer never reads it, but `trim` acts only on
    /// what the KV half reports trimmed, and the `state` setter traps when it is empty, so every
    /// test that trims or snapshots keeps it filled step for step.
    static func appendKV(_ cache: Glm5NextIndexedKVCache, batch: Int, rows: Int) {
        let stub = MLXArray.zeros([batch, 1, rows, 8])
        _ = cache.update(keys: stub, values: stub)
    }

    /// `n` rows of a tiny `[k | gate | valid]` layout (8 + 8 + 1 channels), all valid or none.
    static func tinyRows(_ n: Int, valid: Bool) -> MLXArray {
        concatenated(
            [
                MLXRandom.normal([1, n, 16], key: MLXRandom.key(UInt64(n))),
                MLXArray(Array(repeating: valid ? Float(1) : 0, count: n), [1, n, 1]),
            ], axis: -1)
    }

    /// A cache holding `rows` rows appended the way the indexer appends them, KV half filled.
    static func producerCache(rows: Int) -> Glm5NextIndexedKVCache {
        let cache = Glm5NextIndexedKVCache()
        _ = cache.updateIndexer(tinyRows(rows, valid: true), allRowsValid: true)
        appendKV(cache, batch: 1, rows: rows)
        return cache
    }

    /// A cache whose row 0 is invalid, KV half filled, checked once: `.notAllValid`.
    static func refusedCache(rows: Int) -> Glm5NextIndexedKVCache {
        let cache = Glm5NextIndexedKVCache()
        _ = cache.updateIndexer(
            concatenated([tinyRows(1, valid: false), tinyRows(rows - 1, valid: true)], axis: 1))
        appendKV(cache, batch: 1, rows: rows)
        #expect(!cache.historyIsAllValid())
        #expect(cache.historyValidity == .notAllValid)
        return cache
    }

    /// A stand-in lane. These tests are about WHEN the lane is kept or dropped, not its values.
    static func plantLane(_ cache: Glm5NextIndexedKVCache) {
        let count = (cache.indexerPacked?.dim(1) ?? 0) / 4
        cache.storeDerivedPooledKeys(MLXArray.zeros([1, count, 8]), count: count)
    }

    @Test("C1: a fresh cache is all-valid, with no lane and no checks")
    func freshCacheState() {
        let cache = Glm5NextIndexedKVCache()
        #expect(cache.historyValidity == .allRowsValid)
        #expect(cache.derivedPooledKeys == nil)
        #expect(cache.derivedPooledCount == 0)
        #expect(cache.historyValidityChecks == 0)
    }

    @Test("C2: only the indexer's own appends keep a history all-valid")
    func appendsAndValidity() {
        MLXMetalTestLock.withLock {
            let cache = Self.producerCache(rows: 8)
            Self.plantLane(cache)
            _ = cache.updateIndexer(Self.tinyRows(4, valid: true), allRowsValid: true)
            #expect(cache.historyValidity == .allRowsValid)
            #expect(cache.derivedPooledCount == 2, "a producer append keeps the lane")
            _ = cache.updateIndexer(Self.tinyRows(4, valid: true))
            #expect(cache.historyValidity == .unknown, "any other append is unverified")
            #expect(cache.derivedPooledKeys == nil)
            #expect(cache.derivedPooledCount == 0)
            _ = cache.updateIndexer(Self.tinyRows(4, valid: true), allRowsValid: true)
            #expect(cache.historyValidity == .unknown, "new valid rows vouch for nothing older")
            _ = cache.updateIndexer(Self.tinyRows(4, valid: true))
            #expect(cache.historyValidity == .unknown, "an unverified history stays unverified")
            #expect(cache.indexerPacked?.dim(1) == 24)

            let refused = Self.refusedCache(rows: 8)
            _ = refused.updateIndexer(Self.tinyRows(4, valid: true), allRowsValid: true)
            #expect(refused.historyValidity == .notAllValid, "valid rows cannot cure a bad one")
            _ = refused.updateIndexer(Self.tinyRows(4, valid: true))
            #expect(refused.historyValidity == .notAllValid)
            #expect(refused.historyValidityChecks == 1)
        }
    }

    @Test("C3: setting state drops the lane and adopts its own wrapper; empty state is fresh")
    func stateSetter() throws {
        try MLXMetalTestLock.withLock {
            let cache = Self.producerCache(rows: 8)
            Self.plantLane(cache)
            let snapshot = Self.producerCache(rows: 12).state.map { $0 * 1 }
            cache.state = snapshot
            #expect(cache.derivedPooledKeys == nil)
            #expect(cache.derivedPooledCount == 0)
            #expect(cache.historyValidity == .unknown)
            #expect(cache.indexerPacked?.dim(1) == 12)
            // The history is recognised by object identity, so the cache adopts its OWN wrapper:
            // a later in-place write to the caller's array must not reach it.
            #expect(cache.indexerPacked !== snapshot[2], "adopted the caller's wrapper")
            let adopted = Self.bits(try #require(cache.indexerPacked))
            let callers = snapshot[2]
            callers[0..., 0, 0...] = MLXArray.zeros([1, 17])
            // Evaluate the write first, the order a real caller would hit: it must land in the
            // caller's array, and must not be donated into the buffer the cache still holds.
            #expect(Self.bits(callers[0..., 0, 0...]).allSatisfy { $0 == 0 }, "the write landed")
            #expect(
                Self.bits(try #require(cache.indexerPacked)) == adopted,
                "an in-place write to the caller's array reached the cache")
            // An empty state is a fresh cache, whatever was known before.
            cache.state = []
            #expect(cache.historyValidity == .allRowsValid)
            #expect(cache.indexerPacked == nil)
            #expect(cache.offset == 0)
            // Only an all-valid history holds a lane, so the lane half needs a producer cache.
            let laned = Self.producerCache(rows: 8)
            Self.plantLane(laned)
            laned.state = []
            #expect(laned.derivedPooledKeys == nil)
            #expect(laned.derivedPooledCount == 0)
            // A restore forgets a refusal as well: nobody has checked the new rows.
            let refused = Self.refusedCache(rows: 8)
            refused.state = Self.producerCache(rows: 12).state.map { $0 * 1 }
            #expect(refused.historyValidity == .unknown)
        }
    }

    @Test("C4: a trim drops the lane only if it removed rows, and never changes the validity")
    func trimDropsTheLane() {
        MLXMetalTestLock.withLock {
            let cache = Self.producerCache(rows: 12)
            Self.plantLane(cache)
            #expect(cache.trim(0) == 0)
            #expect(cache.derivedPooledCount == 3, "trimming nothing keeps the lane")
            #expect(cache.trim(2) == 2)
            #expect(cache.derivedPooledKeys == nil)
            #expect(cache.derivedPooledCount == 0)
            #expect(cache.historyValidity == .allRowsValid, "a prefix of an all-valid history")
            #expect(cache.indexerPacked?.dim(1) == 10)

            let unknown = Glm5NextIndexedKVCache()
            unknown.state = Self.producerCache(rows: 12).state.map { $0 * 1 }
            #expect(unknown.trim(2) == 2)
            #expect(unknown.historyValidity == .unknown, "a trim does not vouch for the rest")

            let refused = Self.refusedCache(rows: 12)
            #expect(refused.trim(2) == 2)
            #expect(refused.historyValidity == .notAllValid, "row 0 is still invalid")
            #expect(refused.historyValidityChecks == 1)
        }
    }

    @Test("C5: a copy carries the validity, not the lane")
    func copyCarriesValidityNotLane() throws {
        try MLXMetalTestLock.withLock {
            let cache = Self.producerCache(rows: 8)
            Self.plantLane(cache)
            let copy = try #require(cache.copy() as? Glm5NextIndexedKVCache)
            #expect(copy.historyValidity == .allRowsValid)
            #expect(copy.derivedPooledKeys == nil)
            #expect(copy.derivedPooledCount == 0)
            #expect(cache.derivedPooledCount == 2, "the source keeps its lane")

            _ = cache.updateIndexer(Self.tinyRows(4, valid: true))
            Self.appendKV(cache, batch: 1, rows: 4)
            let unverified = try #require(cache.copy() as? Glm5NextIndexedKVCache)
            #expect(unverified.historyValidity == .unknown)

            let refused = try #require(
                Self.refusedCache(rows: 8).copy() as? Glm5NextIndexedKVCache)
            #expect(refused.historyValidity == .notAllValid)
            #expect(!refused.historyIsAllValid(), "the copy inherits the refusal")
            #expect(refused.historyValidityChecks == 0, "and answers it without reading back")
        }
    }

    @Test("C6: the history is read back at most once, then the answer is remembered")
    func verificationRunsOnce() {
        MLXMetalTestLock.withLock {
            let good = Glm5NextIndexedKVCache()
            _ = good.updateIndexer(Self.tinyRows(8, valid: true))
            #expect(good.historyIsAllValid())
            #expect(good.historyIsAllValid())
            #expect(good.historyValidity == .allRowsValid)
            #expect(good.historyValidityChecks == 1)

            let bad = Glm5NextIndexedKVCache()
            _ = bad.updateIndexer(
                concatenated(
                    [Self.tinyRows(1, valid: false), Self.tinyRows(7, valid: true)], axis: 1))
            #expect(!bad.historyIsAllValid())
            #expect(!bad.historyIsAllValid())
            #expect(bad.historyValidity == .notAllValid)
            #expect(bad.historyValidityChecks == 1)

            // A restored state without indexer rows. There is nothing to read, and an empty
            // history is all-valid, as `state = []` already says.
            let kvOnly = Glm5NextIndexedKVCache()
            Self.appendKV(kvOnly, batch: 1, rows: 4)
            let restored = Glm5NextIndexedKVCache()
            restored.state = kvOnly.state.map { $0 * 1 }
            #expect(restored.historyValidity == .unknown)
            #expect(restored.indexerPacked == nil)
            #expect(restored.historyIsAllValid())
            #expect(restored.historyValidity == .allRowsValid)

            let producer = Self.producerCache(rows: 8)
            #expect(producer.historyIsAllValid())
            #expect(producer.historyValidityChecks == 0, "a producer history needs no check")
        }
    }

    @Test("C7: innerState and the memory probe count the lane; state never does")
    func laneIsDerivedState() {
        MLXMetalTestLock.withLock {
            let cache = Self.producerCache(rows: 8)
            let inner = cache.innerState().count
            let serialized = cache.state.count
            #expect(Glm5NextPrefillMemoryProbe.derivedBytes([cache]) == 0)
            Self.plantLane(cache)
            #expect(cache.innerState().count == inner + 1)
            #expect(cache.state.count == serialized)
            // `kvcache=` counts serialized state only, so the lane is reported apart: two pools
            // of eight float32 values.
            #expect(Glm5NextPrefillMemoryProbe.derivedBytes([cache, KVCacheSimple()]) == 2 * 8 * 4)
        }
    }

    @Test("C8: the first append adopts its own wrapper, so the caller's array stays the caller's")
    func firstAppendAdoptsItsOwnWrapper() throws {
        try MLXMetalTestLock.withLock {
            let cache = Glm5NextIndexedKVCache()
            let rows = Self.tinyRows(8, valid: true)
            let returned = cache.updateIndexer(rows, allRowsValid: true)
            #expect(returned === cache.indexerPacked, "the producer passes on the cache's object")
            #expect(returned !== rows, "adopted the caller's wrapper")
            let adopted = Self.bits(try #require(cache.indexerPacked))
            rows[0..., 0, 0...] = MLXArray.zeros([1, 17])
            #expect(Self.bits(rows[0..., 0, 0...]).allSatisfy { $0 == 0 }, "the write landed")
            #expect(
                Self.bits(try #require(cache.indexerPacked)) == adopted,
                "an in-place write to the caller's array reached the cache")
        }
    }

    // MARK: Decoding through the real producer

    typealias Step = (hidden: MLXArray, qResid: MLXArray, queryOffset: Int, selected: MLXArray?)

    /// One model step, the way `Glm5NextSparseAttention` takes it. The indexer appends its rows
    /// through its own producer and selects with the cache, BEFORE the KV half advances, and it
    /// selects only once the context exceeds `index_topk`.
    @discardableResult
    static func step(
        _ ix: Glm5NextIndexer, _ cache: Glm5NextIndexedKVCache, batch: Int = 1, rows: Int,
        dtype: DType = .float32, key: UInt64
    ) throws -> Step {
        let hiddenSize = ix.wk.weight.dim(1)
        let qLoraRank = ix.wqB.weight.dim(1)
        let hidden = MLXRandom.normal([batch, rows, hiddenSize], key: MLXRandom.key(key))
            .asType(dtype)
        let qResid = MLXRandom.normal([batch, rows, qLoraRank], key: MLXRandom.key(key &+ 1))
            .asType(dtype)
        let queryOffset = cache.offset
        let selecting = queryOffset + rows > ix.topK
        let selected = try ix(
            hidden: hidden, qResid: qResid, cache: cache, selecting: selecting)
        appendKV(cache, batch: batch, rows: rows)
        return (hidden, qResid, queryOffset, selected)
    }

    /// The step's history, checked four ways.
    ///
    /// First, the lane as the MODEL's own selection left it, read before anything else touches the
    /// cache. That is the property this change exists for, and a check made after the direct call
    /// below would be satisfied by the checker itself. It is decisive at pool-completing steps and
    /// at the first selecting step after a reset; at other steps the previous direct call already
    /// left the right count. Then the pooled keys, indices and validity, bitwise against the
    /// oracle; the selection, against the same selection with no cache; and the lane's coverage
    /// after the direct call.
    static func checkStep(
        _ ix: Glm5NextIndexer, _ cache: Glm5NextIndexedKVCache, _ s: Step, expectLane: Bool,
        _ context: String
    ) throws {
        let packed = try #require(cache.indexerPacked)
        let complete = packed.dim(1) / ix.poolSize
        if !expectLane {
            #expect(cache.derivedPooledCount == 0, "\(context): the model stored no lane")
            #expect(cache.derivedPooledKeys == nil, "\(context): the model stored no lane")
        } else if s.selected != nil {
            #expect(
                cache.derivedPooledCount == complete,
                "\(context): the model's own selection kept the lane current")
        }
        let want = try ix.frozenUpstreamPooledStates(packed: packed)
        let got = try ix.pooledStates(packed: packed, cache: cache)
        expectIdentical(got, want, context)
        if let selected = s.selected {
            let B = packed.dim(0)
            let S = s.hidden.dim(1)
            let q = ix.wqB(s.qResid).reshaped(B, S, ix.numHeads, ix.headDim)
            let alone = try ix.selectTopK(
                packed: packed, queries: q, hidden: s.hidden, queryOffset: s.queryOffset)
            #expect(
                firstMismatch(selected.asArray(Int32.self), alone.asArray(Int32.self)) == nil,
                "\(context): selection")
        }
        if expectLane {
            #expect(cache.derivedPooledCount == complete, "\(context): lane coverage")
            if let lane = cache.derivedPooledKeys {
                #expect(lane.dtype == packed.dtype, "\(context): lane dtype")
                #expect(lane.dim(0) == packed.dim(0), "\(context): lane batch")
            }
        } else {
            #expect(cache.derivedPooledCount == 0, "\(context): no lane")
            #expect(cache.derivedPooledKeys == nil, "\(context): no lane")
        }
    }

    // MARK: T1, the lane against the oracle

    /// T1. Decoding with the lane equals the frozen rebuild at every step: a prefill past one pool,
    /// then 41 single-token steps across eleven pool boundaries. It runs at B = 1 and 2, in float32
    /// and bfloat16, with float32 pool math off and on, which also covers the lane's cast back to
    /// the history's dtype.
    @Test("T1: decode with the lane equals the frozen rebuild at every step")
    func decodeMatchesOracle() throws {
        try MLXMetalTestLock.withLock {
            let saved = Glm5NextIndexerRuntime.poolFP32
            defer { Glm5NextIndexerRuntime.poolFP32 = saved }
            for fp32 in [false, true] {
                Glm5NextIndexerRuntime.poolFP32 = fp32
                for dtype: DType in [.float32, .bfloat16] {
                    for batch in [1, 2] {
                        let ix = try Self.makeIndexer(seed: 2, dtype: dtype)
                        let cache = Glm5NextIndexedKVCache()
                        for (t, rows) in ([7] + Array(repeating: 1, count: 41)).enumerated() {
                            let s = try Self.step(
                                ix, cache, batch: batch, rows: rows, dtype: dtype,
                                key: UInt64(200 + 10 * t))
                            try Self.checkStep(
                                ix, cache, s, expectLane: true,
                                "poolFP32=\(fp32) \(dtype) B=\(batch) step \(t)")
                        }
                        #expect(cache.historyValidityChecks == 0)
                    }
                }
            }
        }
    }

    /// T1 at scale: a 7-row first chunk, then 16384 rows in 512-row prefill chunks (4100 pools in
    /// all), then single-token steps. Every 512-row chunk starts mid-pool (N ≡ 3 mod 4), as a
    /// follow-up turn's prefill does, so the lane meets chunks that complete a pool it lacks. A
    /// kernel whose strategy depends on the pool count would break exactness here, where no smaller
    /// test can see it.
    @Test("T1: thousands of pools, in prefill chunks and then decode")
    func longHistoryMatchesOracle() throws {
        try MLXMetalTestLock.withLock {
            let ix = try Self.makeIndexer(seed: 3, dtype: .bfloat16)
            let cache = Glm5NextIndexedKVCache()
            let schedule =
                [7] + Array(repeating: 512, count: 32) + Array(repeating: 1, count: 9)
            for (t, rows) in schedule.enumerated() {
                let s = try Self.step(
                    ix, cache, rows: rows, dtype: .bfloat16, key: UInt64(5000 + 10 * t))
                try Self.checkStep(ix, cache, s, expectLane: true, "long step \(t)")
            }
            #expect(cache.indexerPacked?.dim(1) == 16400)
        }
    }

    /// T1, the identity guard. After the lane covers history H, a call with a DIFFERENT history G
    /// must answer for G and leave the cache exactly as it was. G's rows differ inside pools the
    /// lane already covers, and G is one pool longer. Without the guard, the lane built from H is
    /// spliced into G's result, and G's pools then replace the lane.
    @Test("T1: a history that is not the cache's own never touches the lane")
    func foreignHistoryIsNotSplicedIn() throws {
        try MLXMetalTestLock.withLock {
            let ix = try Self.makeIndexer(seed: 4)
            let cache = Glm5NextIndexedKVCache()
            for (t, rows) in [24, 1, 1, 1, 1].enumerated() {  // N = 28: seven complete pools
                let s = try Self.step(ix, cache, rows: rows, key: UInt64(400 + 10 * t))
                try Self.checkStep(ix, cache, s, expectLane: true, "H step \(t)")
            }
            let h = try #require(cache.indexerPacked)
            let laneBefore = try #require(cache.derivedPooledKeys)
            let countBefore = cache.derivedPooledCount
            // G: H with the key and gate channels of its first 8 rows replaced, validity kept, and
            // one more complete pool of valid rows.
            let D = ix.headDim
            let replaced = MLXRandom.normal([1, 8, 2 * D], key: MLXRandom.key(4242))
            let extra = concatenated(
                [
                    MLXRandom.normal([1, 4, 2 * D], key: MLXRandom.key(4343)),
                    MLXArray.ones([1, 4, 1]),
                ], axis: -1)
            let g = concatenated(
                [
                    concatenated([replaced, h[0..., ..<8, (2 * D)...]], axis: -1),
                    h[0..., 8..., 0...],
                    extra,
                ], axis: 1)
            let want = try ix.frozenUpstreamPooledStates(packed: g)
            let got = try ix.pooledStates(packed: g, cache: cache)
            Self.expectIdentical(got, want, "foreign history")
            #expect(cache.derivedPooledKeys === laneBefore)
            #expect(cache.derivedPooledCount == countBefore)
            #expect(cache.historyValidity == .allRowsValid)
            #expect(cache.historyValidityChecks == 0)
        }
    }

    // MARK: T2–T8, the lane's lifecycle end to end

    /// A cache restored from a producer history of `rows` rows with row 0 marked invalid, as an
    /// external restore could deliver it: `.unknown` until checked. Also returns that history as
    /// the producer wrote it, all valid, which is FOREIGN to the restored cache.
    static func restoredWithInvalidFirstRow(
        _ ix: Glm5NextIndexer, rows: Int, key: UInt64
    ) throws -> (cache: Glm5NextIndexedKVCache, validHistory: MLXArray) {
        let source = Glm5NextIndexedKVCache()
        try step(ix, source, rows: rows, key: key)
        let snapshot = source.state.map { $0 * 1 }
        let D = ix.headDim
        let validity = concatenated(
            [MLXArray.zeros([1, 1, 1]), MLXArray.ones([1, rows - 1, 1])], axis: 1
        ).asType(snapshot[2].dtype)
        let restored = Glm5NextIndexedKVCache()
        restored.state = [
            snapshot[0], snapshot[1],
            concatenated([snapshot[2][.ellipsis, ..<(2 * D)], validity], axis: -1),
        ]
        #expect(restored.historyValidity == .unknown)
        return (restored, snapshot[2])
    }

    /// T2. Any trim clears the lane, and decoding afterwards rebuilds it exactly. The trim cuts
    /// into a pool the lane covers (rows 25–27 of pool 6), and the rows that replace them differ,
    /// so a lane that survived the trim would be caught by the oracle, not only by the count.
    @Test("T2: trim clears the lane, and decoding rebuilds it exactly")
    func trimClearsTheLane() throws {
        try MLXMetalTestLock.withLock {
            let ix = try Self.makeIndexer(seed: 5)
            let cache = Glm5NextIndexedKVCache()
            for (t, rows) in ([22] + Array(repeating: 1, count: 8)).enumerated() {  // N = 30
                let s = try Self.step(ix, cache, rows: rows, key: UInt64(500 + 10 * t))
                try Self.checkStep(ix, cache, s, expectLane: true, "before trim, step \(t)")
            }
            #expect(cache.derivedPooledCount == 7)
            #expect(cache.trim(0) == 0)
            #expect(cache.derivedPooledCount == 7, "trimming nothing keeps the lane")
            #expect(cache.trim(5) == 5)
            #expect(cache.derivedPooledCount == 0)
            #expect(cache.derivedPooledKeys == nil)
            #expect(cache.indexerPacked?.dim(1) == 25)
            #expect(cache.historyValidity == .allRowsValid)
            for t in 0 ..< 10 {  // N = 26 ... 35: past the rewritten pool 6, and beyond
                let s = try Self.step(ix, cache, rows: 1, key: UInt64(600 + 10 * t))
                try Self.checkStep(ix, cache, s, expectLane: true, "after trim, step \(t)")
            }
        }
    }

    /// T3. A DIFFERENT history restored over a cache with a populated lane. The lane is dropped,
    /// the model's next selection checks the restored rows once, and every later step matches the
    /// oracle on the restored history without checking again.
    @Test("T3: a restore clears a stale lane and verifies once")
    func restoreOverAStaleLane() throws {
        try MLXMetalTestLock.withLock {
            let ix = try Self.makeIndexer(seed: 6)
            let a = Glm5NextIndexedKVCache()
            for (t, rows) in ([20] + Array(repeating: 1, count: 7)).enumerated() {  // H1, N = 27
                let s = try Self.step(ix, a, rows: rows, key: UInt64(610 + 10 * t))
                try Self.checkStep(ix, a, s, expectLane: true, "H1 step \(t)")
            }
            #expect(a.derivedPooledCount == 6)
            let b = Glm5NextIndexedKVCache()
            try Self.step(ix, b, rows: 33, key: 700)  // H2, N = 33, different rows
            // A snapshot that owns its buffers, the way `copy()` takes one: `b`'s own arrays would
            // be written in place by `a`'s next KV update.
            a.state = b.state.map { $0 * 1 }
            #expect(a.derivedPooledCount == 0)
            #expect(a.derivedPooledKeys == nil)
            #expect(a.historyValidity == .unknown)
            for t in 0 ..< 10 {
                let s = try Self.step(ix, a, rows: 1, key: UInt64(800 + 10 * t))
                try Self.checkStep(ix, a, s, expectLane: true, "H2 step \(t)")
                #expect(a.historyValidityChecks == 1, "verified once, and never again")
                #expect(a.historyValidity == .allRowsValid)
            }
        }
    }

    /// T4. A restored history whose first row is not valid. A call with a FOREIGN history must not
    /// check this cache: the check comes after the identity guard, so the foreign call costs no
    /// sync and leaves `.unknown` alone. The model's own selection then finds the bad row with one
    /// check, no lane is ever stored, and every answer is the oracle's, including the grid shifted
    /// to row 1 that a lane, which assumes row 0, gets wrong.
    @Test("T4: a history with an invalid row is verified once and never laned")
    func invalidHistoryIsRefused() throws {
        try MLXMetalTestLock.withLock {
            let ix = try Self.makeIndexer(seed: 7)
            let (restored, validHistory) = try Self.restoredWithInvalidFirstRow(
                ix, rows: 21, key: 900)
            let wantForeign = try ix.frozenUpstreamPooledStates(packed: validHistory)
            let foreign = try ix.pooledStates(packed: validHistory, cache: restored)
            Self.expectIdentical(foreign, wantForeign, "foreign all-valid history")
            #expect(restored.historyValidityChecks == 0, "a foreign history triggers no check")
            #expect(restored.historyValidity == .unknown)
            for t in 0 ..< 6 {
                let s = try Self.step(ix, restored, rows: 1, key: UInt64(1000 + 10 * t))
                try Self.checkStep(
                    ix, restored, s, expectLane: false, "invalid row 0, step \(t)")
                #expect(restored.historyValidity == .notAllValid)
                #expect(restored.historyValidityChecks == 1)
            }
        }
    }

    /// T5. Rows that arrive through `updateIndexer(_:)` from anyone but the indexer. A fresh cache
    /// whose row 0 is invalid becomes `.unknown`, then `.notAllValid` after the model's own
    /// selection checks it once. On a cache that already has a lane, an external append drops the
    /// lane. If those rows turn out valid, one check restores `.allRowsValid` and the lane is
    /// rebuilt.
    @Test("T5: an external append is checked once before the lane is trusted")
    func externalAppendIsVerified() throws {
        try MLXMetalTestLock.withLock {
            let ix = try Self.makeIndexer(seed: 8)
            let D = ix.headDim
            func rows(_ n: Int, invalidFirst: Bool, key: UInt64) -> MLXArray {
                let validity = (0 ..< n).map { invalidFirst && $0 == 0 ? Float(0) : 1 }
                return concatenated(
                    [
                        MLXRandom.normal([1, n, 2 * D], key: MLXRandom.key(key)),
                        MLXArray(validity, [1, n, 1]),
                    ], axis: -1)
            }

            // 17 rows: every following step is past `index_topk` (16), so it selects.
            let fresh = Glm5NextIndexedKVCache()
            _ = fresh.updateIndexer(rows(17, invalidFirst: true, key: 1100))
            Self.appendKV(fresh, batch: 1, rows: 17)
            #expect(fresh.historyValidity == .unknown)
            for t in 0 ..< 5 {
                let s = try Self.step(ix, fresh, rows: 1, key: UInt64(1200 + 10 * t))
                #expect(s.selected != nil)
                try Self.checkStep(ix, fresh, s, expectLane: false, "external, invalid, step \(t)")
                #expect(fresh.historyValidity == .notAllValid)
                #expect(fresh.historyValidityChecks == 1)
            }

            let laned = Glm5NextIndexedKVCache()
            for (t, n) in [20, 1, 1, 1, 1].enumerated() {
                let s = try Self.step(ix, laned, rows: n, key: UInt64(1300 + 10 * t))
                try Self.checkStep(ix, laned, s, expectLane: true, "before external, step \(t)")
            }
            _ = laned.updateIndexer(rows(3, invalidFirst: false, key: 1400))
            Self.appendKV(laned, batch: 1, rows: 3)
            #expect(laned.derivedPooledKeys == nil)
            #expect(laned.historyValidity == .unknown)
            for t in 0 ..< 5 {
                let s = try Self.step(ix, laned, rows: 1, key: UInt64(1500 + 10 * t))
                try Self.checkStep(ix, laned, s, expectLane: true, "after external, step \(t)")
                #expect(laned.historyValidity == .allRowsValid)
                #expect(laned.historyValidityChecks == 1)
            }
        }
    }

    /// T6. With the switch off, no lane is ever stored and nothing is ever checked, even on an
    /// `.unknown` history, and every answer is the oracle's.
    @Test("T6: the switch turns the lane off, and nothing is ever checked")
    func switchOff() throws {
        try MLXMetalTestLock.withLock {
            let saved = Glm5NextIndexerRuntime.poolCache
            defer { Glm5NextIndexerRuntime.poolCache = saved }
            Glm5NextIndexerRuntime.poolCache = false
            let ix = try Self.makeIndexer(seed: 9)
            let cache = Glm5NextIndexedKVCache()
            let first = try Self.step(ix, cache, rows: 7, key: 1600)
            try Self.checkStep(ix, cache, first, expectLane: false, "switch off, prefill")
            // Unverified on purpose. If the check were ordered before the switch, this restored
            // history would be read back although the lane is off.
            cache.state = cache.state.map { $0 * 1 }
            #expect(cache.historyValidity == .unknown)
            for t in 1 ... 20 {
                let s = try Self.step(ix, cache, rows: 1, key: UInt64(1600 + 10 * t))
                try Self.checkStep(ix, cache, s, expectLane: false, "switch off, step \(t)")
                #expect(cache.historyValidity == .unknown)
                #expect(cache.historyValidityChecks == 0)
            }
        }
    }

    /// T7. `copy()` carries the history's validity but not the lane. A copy of an all-valid cache
    /// rebuilds its own lane on first use, without a check, and decoding it leaves the source's
    /// lane alone. A copy of a refused cache stays refused, without checking again.
    @Test("T7: a copy rebuilds its own lane, and inherits a refusal")
    func copyRebuildsItsOwnLane() throws {
        try MLXMetalTestLock.withLock {
            let ix = try Self.makeIndexer(seed: 10)
            let source = Glm5NextIndexedKVCache()
            for (t, rows) in ([20] + Array(repeating: 1, count: 7)).enumerated() {  // N = 27
                let s = try Self.step(ix, source, rows: rows, key: UInt64(1700 + 10 * t))
                try Self.checkStep(ix, source, s, expectLane: true, "source step \(t)")
            }
            let sourceLane = try #require(source.derivedPooledKeys)
            let copy = try #require(source.copy() as? Glm5NextIndexedKVCache)
            #expect(copy.historyValidity == .allRowsValid)
            #expect(copy.derivedPooledKeys == nil)
            #expect(copy.derivedPooledCount == 0)
            for t in 0 ..< 10 {
                let s = try Self.step(ix, copy, rows: 1, key: UInt64(1800 + 10 * t))
                try Self.checkStep(ix, copy, s, expectLane: true, "copy step \(t)")
            }
            #expect(copy.historyValidityChecks == 0)
            #expect(source.derivedPooledKeys === sourceLane)
            #expect(source.derivedPooledCount == 6)

            let (refused, _) = try Self.restoredWithInvalidFirstRow(ix, rows: 21, key: 1760)
            _ = try ix.pooledStates(packed: try #require(refused.indexerPacked), cache: refused)
            #expect(refused.historyValidity == .notAllValid)
            let refusedCopy = try #require(refused.copy() as? Glm5NextIndexedKVCache)
            #expect(refusedCopy.historyValidity == .notAllValid)
            for t in 0 ..< 5 {
                let s = try Self.step(ix, refusedCopy, rows: 1, key: UInt64(1900 + 10 * t))
                try Self.checkStep(
                    ix, refusedCopy, s, expectLane: false, "refused copy, step \(t)")
                #expect(refusedCopy.historyValidityChecks == 0, "the copy never checks again")
            }
        }
    }

    /// T8. A lane that cannot be spliced in, with a dtype or batch width other than the
    /// history's, is treated as absent. The answer is the oracle's, and a usable lane replaces it.
    /// Without the guard, the `reused == poolCount` branch would hand back the planted zeros.
    @Test("T8: an unusable lane is replaced, never spliced in")
    func unusableLaneIsReplaced() throws {
        try MLXMetalTestLock.withLock {
            let ix = try Self.makeIndexer(seed: 12)
            let planted: [(String, MLXArray)] = [
                ("dtype", MLXArray.zeros([1, 2, ix.headDim], dtype: .bfloat16)),
                ("batch", MLXArray.zeros([2, 2, ix.headDim])),
            ]
            for (name, lane) in planted {
                let cache = Glm5NextIndexedKVCache()
                try Self.step(ix, cache, rows: 8, key: 1950)  // N = 8, float32, not selecting
                cache.storeDerivedPooledKeys(lane, count: 2)
                let packed = try #require(cache.indexerPacked)
                let want = try ix.frozenUpstreamPooledStates(packed: packed)
                let got = try ix.pooledStates(packed: packed, cache: cache)
                Self.expectIdentical(got, want, "unusable lane (\(name))")
                let replacement = try #require(cache.derivedPooledKeys)
                #expect(replacement.dtype == packed.dtype, "\(name): replacement dtype")
                #expect(replacement.dim(0) == 1, "\(name): replacement batch width")
                #expect(cache.derivedPooledCount == 2)
            }
        }
    }

    /// T9. The lane is REUSED, not merely kept current. A sentinel lane, planted where the model
    /// would have stored one, comes back bit for bit in the pooled keys, and only the pool the
    /// lane lacks is computed. At N = 8 the lane covers every pool, the `reused == poolCount`
    /// branch. The lane is trusted without a re-check, which is why every transition that could
    /// leave it stale clears it. Every other test would pass if the lane were stored but never
    /// read back.
    @Test("T9: the lane is reused, not only kept current")
    func laneIsReused() throws {
        try MLXMetalTestLock.withLock {
            let ix = try Self.makeIndexer(seed: 13)
            for rows in [9, 8] {
                let cache = Glm5NextIndexedKVCache()
                try Self.step(ix, cache, rows: rows, key: UInt64(2000 + rows))  // not selecting
                let packed = try #require(cache.indexerPacked)
                let sentinel = MLXArray.ones([1, 2, ix.headDim]) * 7
                cache.storeDerivedPooledKeys(sentinel, count: 2)
                let got = try ix.pooledStates(packed: packed, cache: cache)
                #expect(
                    Self.bits(got.keys[0..., ..<2, 0...]) == Self.bits(sentinel),
                    "N = \(rows): the lane's pools come back as stored")
                if rows == 9 {
                    let want = try ix.frozenUpstreamPooledStates(packed: packed)
                    #expect(
                        Self.bits(got.keys[0..., 2..., 0...])
                            == Self.bits(want.keys[0..., 2..., 0...]),
                        "N = 9: the pool the lane lacks is pooled")
                }
                #expect(cache.derivedPooledKeys === sentinel, "N = \(rows): nothing new to store")
                #expect(cache.derivedPooledCount == 2)
            }
        }
    }

    /// The shipped indexer geometry (`index_head_dim` 128, `index_n_heads` 32, `index_topk` 2048,
    /// hidden 4096, `q_lora_rank` 1536) in bfloat16, prefilled past `index_topk` so selection is
    /// live, then decoded across a pool boundary.
    @Test("shipped geometry: selection live, lane exact")
    func shippedGeometry() throws {
        try MLXMetalTestLock.withLock {
            let ix = try Self.makeIndexer(
                seed: 11, dtype: .bfloat16, headDim: 128, numHeads: 32, topK: 2048,
                hidden: 4096, qLoraRank: 1536)
            let cache = Glm5NextIndexedKVCache()
            for (t, rows) in ([1024, 1024, 52] + Array(repeating: 1, count: 6)).enumerated() {
                let s = try Self.step(
                    ix, cache, rows: rows, dtype: .bfloat16, key: UInt64(2000 + 10 * t))
                try Self.checkStep(ix, cache, s, expectLane: true, "shipped step \(t)")
                if t >= 2 { #expect(s.selected != nil, "selection is live past index_topk") }
            }
            #expect(cache.indexerPacked?.dim(1) == 2106)
        }
    }
}
