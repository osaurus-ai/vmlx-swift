// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// End-to-end disk-tier round trip on the tiny synthetic Ling 3.0 model
/// (BailingMoeV3: MLA attention on `KVCacheSimple` + KDA on `MambaCache`).
///
/// `HybridRestoreBoundaryInvariantTests` (MLXLMCommonFocusedTests) pins
/// `validateRestoredCacheBoundary` against hand-built caches. This suite runs
/// the REAL pipeline — model prefill, `TQDiskSerializer.serialize` (v2),
/// `restoreFromDiskArrays`, the validator, then the suffix prefill — and
/// proves two things:
///
/// 1. A partial hit (same 37-token prefix, 5-token suffix) restored from a
///    v2 record reproduces one-shot prefill: every offset, the last-position
///    logits, and every recurrent state.
/// 2. A record whose two halves disagree (recurrent offset tampered, or one
///    attention layer's KV truncated) is refused by the validator.
///
/// 37 is deliberately not a multiple of any step/page size (16, 64, 256): the
/// dots3 `cached_tokens % 256` corruption was invisible at aligned lengths.
@Suite("Hybrid restore boundary round trip (Ling 3 tiny)", .serialized)
struct HybridRestoreBoundaryRoundTripTests {

    static let prefixLength = 37
    static let suffixLength = 5
    static var totalLength: Int { prefixLength + suffixLength }  // 42

    /// Deterministic prompt ids in `0..<vocab_size` (128 on the fixture).
    /// A fixed LCG rather than `MLXRandom` so the ids do not depend on the
    /// MLX RNG stream that model init also draws from.
    private static func promptTokens(count: Int, seed: UInt64 = 0x5EED_2026) -> [Int32] {
        var s = seed
        return (0 ..< count).map { _ in
            s = s &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int32((s >> 33) % 128)
        }
    }

    private struct Fixture {
        let model: BailingMoeV3Model
        let tokens: MLXArray  // [42] int32
        let mambaIndices: [Int]
        let attentionIndices: [Int]

        private typealias T = HybridRestoreBoundaryRoundTripTests

        var prefix: MLXArray { tokens[0 ..< T.prefixLength].reshaped(1, T.prefixLength) }
        var suffix: MLXArray {
            tokens[T.prefixLength ..< T.totalLength].reshaped(1, T.suffixLength)
        }
        var whole: MLXArray { tokens.reshaped(1, T.totalLength) }
    }

    /// Must be called under `MLXMetalTestLock`.
    private static func makeFixture() throws -> Fixture {
        MLXRandom.seed(11)
        let config = try JSONDecoder().decode(
            BailingMoeV3Configuration.self, from: BailingMoeV3Tests.v3ConfigJSON)
        let model = BailingMoeV3Model(config)
        MLX.eval(model.parameters())

        let probe = model.newCache(parameters: nil)
        let mamba = probe.indices.filter { probe[$0] is MambaCache }
        let attention = probe.indices.filter { probe[$0] is KVCacheSimple }
        // Fail closed: the hybrid shape is the whole point of the test.
        try #require(!mamba.isEmpty, "fixture has no MambaCache layers")
        try #require(!attention.isEmpty, "fixture has no KVCacheSimple layers")
        try #require(mamba.count + attention.count == probe.count)

        return Fixture(
            model: model,
            tokens: MLXArray(promptTokens(count: totalLength)),
            mambaIndices: mamba,
            attentionIndices: attention)
    }

    /// Prefill the 37-token prefix into a fresh cache and serialize it as a
    /// v2 disk record (Mamba topology, no `ssmStates` companion).
    private static func prefixRecord(_ fx: Fixture) throws -> [String: MLXArray] {
        let cache = fx.model.newCache(parameters: nil)
        let out = fx.model(fx.prefix, cache: cache)
        MLX.eval(out)
        MLX.eval(cache.flatMap(\.state))
        for (i, layer) in cache.enumerated() {
            try #require(layer.offset == prefixLength, "prefill: layer \(i) offset \(layer.offset)")
        }
        let record = TQDiskSerializer.serialize(cache: cache, ssmStates: nil)

        // The record must actually carry both halves, or the round trip below
        // would compare a miss against a miss.
        try #require(TQDiskSerializer.formatVersion(of: record) >= 2)
        for m in fx.mambaIndices {
            try #require(record["mamba_\(m)_state0"] != nil, "missing mamba_\(m)_state0")
            let off = try #require(record["__mamba_\(m)_offset__"], "missing __mamba_\(m)_offset__")
            #expect(off.shape == [1] && off.dtype == .int32, "offset meta shape \(off.shape) \(off.dtype)")
            #expect(off[0].item(Int32.self) == Int32(prefixLength))
        }
        for a in fx.attentionIndices {
            let keys = try #require(record["kv_\(a)_keys"], "missing kv_\(a)_keys")
            try #require(record["kv_\(a)_values"] != nil, "missing kv_\(a)_values")
            #expect(keys.dim(2) == prefixLength, "kv_\(a)_keys seq dim \(keys.dim(2))")
        }
        return record
    }

    private static func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    // MARK: - (a) partial hit reproduces one-shot prefill

    @Test("disk-tier partial hit (37 + 5) reproduces one-shot prefill of 42")
    func roundTripPartialHitMatchesOneShot() throws {
        try MLXMetalTestLock.withLock {
            let fx = try Self.makeFixture()

            // One shot: fresh cache, 42 tokens in a single prefill.
            let oneShotCache = fx.model.newCache(parameters: nil)
            let oneShot = fx.model(fx.whole, cache: oneShotCache)
            MLX.eval(oneShot)
            MLX.eval(oneShotCache.flatMap(\.state))
            let oneShotLast = oneShot[0, Self.totalLength - 1]
            for (i, layer) in oneShotCache.enumerated() {
                #expect(layer.offset == Self.totalLength, "one-shot: layer \(i) offset \(layer.offset)")
            }
            // Non-empty baseline: random-init logits must not be degenerate.
            try #require(abs(oneShotLast.asType(.float32)).max().item(Float.self) > 0)
            var oneShotStates: [Int: [MLXArray]] = [:]
            for m in fx.mambaIndices {
                let s = oneShotCache[m].state
                try #require(s.count == 2, "one-shot: mamba layer \(m) has \(s.count) slots")
                oneShotStates[m] = s
            }

            // Cached: prefill A, serialize (v2), restore into a fresh cache.
            let record = try Self.prefixRecord(fx)
            var restoredCache = fx.model.newCache(parameters: nil)
            let restoredTokens = restoreFromDiskArrays(record, into: &restoredCache)
            MLX.eval(restoredCache.flatMap(\.state))

            #expect(restoredTokens == Self.prefixLength, "restore returned \(restoredTokens)")
            #expect(
                validateRestoredCacheBoundary(
                    restoredCache, matchedTokens: Self.prefixLength,
                    restoredTokens: restoredTokens, detail: "round-trip"),
                "validator refused a consistent restore")
            for (i, layer) in restoredCache.enumerated() {
                #expect(layer.offset == Self.prefixLength, "restored: layer \(i) offset \(layer.offset)")
            }

            // Continue with the suffix through the restored cache.
            let step = fx.model(fx.suffix, cache: restoredCache)
            MLX.eval(step)
            MLX.eval(restoredCache.flatMap(\.state))
            for (i, layer) in restoredCache.enumerated() {
                #expect(layer.offset == Self.totalLength, "after suffix: layer \(i) offset \(layer.offset)")
            }

            // Same tolerance as BailingMoeV3Tests.cachedDecodeParity.
            let stepLast = step[0, Self.suffixLength - 1]
            let logitDiff = Self.maxAbsDiff(oneShotLast, stepLast)
            #expect(logitDiff < 2e-2, "restored-path logits diverged from one-shot: \(logitDiff)")

            // Every recurrent state must match one-shot slot for slot.
            for m in fx.mambaIndices {
                let restored = restoredCache[m].state
                let reference = try #require(oneShotStates[m])
                try #require(restored.count == reference.count, "mamba layer \(m) slot count")
                for (slot, (r, o)) in zip(restored, reference).enumerated() {
                    #expect(r.shape == o.shape, "mamba layer \(m) slot \(slot) shape \(r.shape) vs \(o.shape)")
                    let scale = max(1, abs(o.asType(.float32)).max().item(Float.self))
                    let diff = Self.maxAbsDiff(r, o)
                    #expect(
                        diff <= 2e-2 * scale,
                        "mamba layer \(m) slot \(slot) state diverged: \(diff) (scale \(scale))")
                }
            }
        }
    }

    // MARK: - (b) desynchronised records are refused

    @Test("a record whose recurrent offset is 38 against 37 attention tokens is refused")
    func tamperedRecurrentOffsetIsRefused() throws {
        try MLXMetalTestLock.withLock {
            let fx = try Self.makeFixture()
            var record = try Self.prefixRecord(fx)

            // Same shape/dtype `metaInt32` writes (1-element 1D int32) so the
            // deserializer's `readMetaInt32` accepts it as a valid offset.
            let tampered = fx.mambaIndices[0]
            record["__mamba_\(tampered)_offset__"] = MLXArray([Int32(Self.prefixLength + 1)])

            var cache = fx.model.newCache(parameters: nil)
            let restoredTokens = restoreFromDiskArrays(record, into: &cache)

            // What the code does today: the deserializer trusts a readable
            // offset and `restoreMambaLayer` seats it verbatim, while
            // `totalTokens` comes from the attention keys (37). So restore
            // reports 37 and leaves the tampered layer at 38 — exactly the
            // production desync shape — and ONLY the validator catches it.
            #expect(restoredTokens == Self.prefixLength, "restore returned \(restoredTokens)")
            #expect(cache[tampered].offset == Self.prefixLength + 1)
            for a in fx.attentionIndices {
                #expect(cache[a].offset == Self.prefixLength)
            }
            #expect(
                !validateRestoredCacheBoundary(
                    cache, matchedTokens: Self.prefixLength,
                    restoredTokens: restoredTokens, detail: "tampered-recurrent-offset"),
                "validator accepted a recurrent layer at 38 against a 37-token match")
        }
    }

    @Test("a record with one attention layer's KV truncated to 36 tokens is refused")
    func truncatedKVIsRefused() throws {
        try MLXMetalTestLock.withLock {
            let fx = try Self.makeFixture()
            let record = try Self.prefixRecord(fx)
            let firstAttention = try #require(fx.attentionIndices.first)

            // Truncate each attention layer in turn. Truncating the FIRST one
            // also shrinks the restore count (36 != 37); truncating a later
            // one leaves the count at 37 and is caught only by the per-layer
            // offset check. Both properties must refuse.
            for victim in fx.attentionIndices {
                var damaged = record
                let keys = try #require(damaged["kv_\(victim)_keys"])
                let values = try #require(damaged["kv_\(victim)_values"])
                // KVCacheSimple layout is [B, H, T, D]; slice the T axis the
                // same way `KVCacheSimple.state` does.
                let shorter = Self.prefixLength - 1
                damaged["kv_\(victim)_keys"] = keys[.ellipsis, ..<shorter, 0...]
                damaged["kv_\(victim)_values"] = values[.ellipsis, ..<shorter, 0...]

                var cache = fx.model.newCache(parameters: nil)
                let restoredTokens = restoreFromDiskArrays(damaged, into: &cache)

                // What the code does today: `totalTokens` is seeded from the
                // first `.standard` layer's key length, so it is 36 only when
                // the first attention layer is the truncated one.
                let expectedCount = victim == firstAttention
                    ? Self.prefixLength - 1 : Self.prefixLength
                #expect(
                    restoredTokens == expectedCount,
                    "victim \(victim): restore returned \(restoredTokens), expected \(expectedCount)")
                #expect(cache[victim].offset == Self.prefixLength - 1, "victim \(victim) offset \(cache[victim].offset)")
                for m in fx.mambaIndices {
                    #expect(cache[m].offset == Self.prefixLength, "mamba \(m) offset \(cache[m].offset)")
                }
                #expect(
                    !validateRestoredCacheBoundary(
                        cache, matchedTokens: Self.prefixLength,
                        restoredTokens: restoredTokens, detail: "truncated-kv-\(victim)"),
                    "validator accepted attention layer \(victim) at 36 against a 37-token match")
            }
        }
    }
}
