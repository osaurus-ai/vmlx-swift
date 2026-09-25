// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Fail-closed disk restore, proven per ARCHITECTURE rather than per cache
// class.
//
// The defect this pins: a damaged on-disk payload used to restore SOME layers
// and report success. The coordinator was told "n tokens restored" while a
// layer sat at offset 0, so attention ran against an empty layer — coherent,
// confident, wrong text. A miss costs one re-prefill. A bad restore costs the
// answer, silently, and looks like a model quality problem.
//
// Scoping by cache CLASS is the mistake that let this survive: the existing
// coverage tested DSV4 and rotating topologies, and both the shipped families
// whose caches are neither (hybrid SSM, Zaya CCA) and the families that use a
// bare `KVCacheSimple` went unexercised. So the matrix below is indexed by the
// shipped model families and built from the topology each one's `newCache`
// actually returns:
//
//   Muse (MuseGlimmerText)          RotatingKVCache
//   Ornith 1.5 / Qwen 3.5-3.6       MambaCache + RotatingKVCache   (hybrid SSM)
//   DeepSeek-V4                     DeepseekV4Cache + RotatingKVCache
//   Nanbeige / LagunaM1             KVCacheSimple      (no newCache override)
//   LFM2 / Jamba / GraniteHybrid    MambaCache + KVCacheSimple
//   FalconH1                       CacheList(MambaCache, KVCacheSimple)
//   Gemma-4 / Mistral / GPTOSS      RotatingKVCache
//
// Every case runs BOTH directions. The positive control is not optional: a
// fail-closed assertion passes vacuously if restore never returns anything,
// so each topology must first be shown to restore fully when undamaged.

import Foundation
import MLX
import MLXLLM

@testable import MLXLMCommon
import Testing

@Suite("Disk restore fails closed across architectures", .serialized)
struct DiskRestoreFailClosedMatrixTests {

    // MARK: - Topologies, named by the families that ship them

    private struct Topology {
        let name: String
        let families: String
        let make: () -> [any KVCache]
    }

    /// Computed rather than stored: the entries carry factory closures, and a
    /// stored static would be shared mutable global state under strict
    /// concurrency. Rebuilding the list per access also guarantees each test
    /// gets fresh cache objects.
    private static var topologies: [Topology] {
        [
        Topology(
            name: "rotating",
            families: "Muse Glimmer, Gemma-4, Mistral, GPT-OSS, AfMoE",
            make: {
                [
                    RotatingKVCache(maxSize: 16, keep: 0),
                    RotatingKVCache(maxSize: 16, keep: 4),
                    RotatingKVCache(maxSize: 16, keep: 0),
                ]
            }),
        Topology(
            name: "hybrid-ssm",
            families: "Ornith 1.5, Qwen 3.5/3.6, Qwen3Next, NemotronH",
            make: {
                [
                    MambaCache(),
                    RotatingKVCache(maxSize: 16, keep: 4),
                    MambaCache(),
                    RotatingKVCache(maxSize: 16, keep: 4),
                ]
            }),
        Topology(
            name: "deepseek-v4",
            families: "DeepSeek-V4, DeepSeek-V4 JANGTQ",
            make: {
                [
                    RotatingKVCache(maxSize: 16, keep: 0),
                    DeepseekV4Cache(slidingWindow: 16, compressRatio: 4),
                    RotatingKVCache(maxSize: 16, keep: 0),
                    DeepseekV4Cache(slidingWindow: 16, compressRatio: 128),
                ]
            }),
        Topology(
            name: "plain-kv",
            families: "Nanbeige, LagunaM1 (default cache, no newCache override)",
            make: { [KVCacheSimple(), KVCacheSimple(), KVCacheSimple()] }),
        Topology(
            name: "hybrid-plain-kv",
            families: "LFM2, LFM2MoE, Jamba, GraniteMoeHybrid mixed configurations",
            make: { [MambaCache(), KVCacheSimple(), MambaCache()] }),
        Topology(
            name: "composite-hybrid",
            families: "FalconH1",
            make: {
                [CacheList(MambaCache(), KVCacheSimple()),
                 CacheList(MambaCache(), KVCacheSimple())]
            }),
        Topology(
            name: "composite-duplicate-kv",
            families: "synthetic duplicate-type CacheList contract",
            make: { [CacheList(KVCacheSimple(), KVCacheSimple())] }),
        Topology(
            name: "composite-tq-fill",
            families: "synthetic full-precision TQ composite contract",
            make: { [CacheList(MambaCache(), TurboQuantKVCache())] }),
        Topology(
            name: "composite-rotating",
            families: "synthetic rotating CacheList contract",
            make: {
                [CacheList(MambaCache(), RotatingKVCache(maxSize: 16, keep: 0)),
                 CacheList(RotatingKVCache(maxSize: 16, keep: 0),
                           RotatingKVCache(maxSize: 16, keep: 0))]
            }),
        Topology(
            name: "pure-ssm",
            families: "synthetic all-recurrent configuration",
            make: { [MambaCache(), MambaCache(), MambaCache()] }),
        ]
    }

    // MARK: - Helpers

    private static func tempDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("failclosed-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Advances every layer, dispatching on what the layer actually is.
    ///
    /// A `MambaCache` holds recurrent state in `ArraysCache` slots and is not
    /// advanced by `update(keys:values:)` the way an attention cache is —
    /// driving it through the attention path would leave its offset at 0 and
    /// make every assertion below vacuously true.
    private static func advance(_ caches: [any KVCache], steps: Int, headDim: Int = 8) {
        for s in 0..<steps {
            for c in leaves(caches) {
                if let mamba = c as? MambaCache {
                    mamba[0] = MLXArray.ones([1, 4, headDim]) * Float(s + 1)
                    mamba[1] = MLXArray.ones([1, 4, headDim]) * Float(s + 2)
                    mamba.offset = 2 * (s + 1)
                } else {
                    let k = MLXArray.ones([1, 1, 2, headDim]) * Float(s + 1)
                    let v = MLXArray.ones([1, 1, 2, headDim]) * Float(s + 2)
                    _ = c.update(keys: k, values: v)
                }
            }
        }
        MLX.eval(caches)
    }

    /// Composite layers must validate every child, not only the wrapper's
    /// inherited offset (which does not represent the contained states).
    private static func leaves(_ caches: [any KVCache]) -> [any KVCache] {
        caches.flatMap { cache -> [any KVCache] in
            if let list = cache as? CacheList {
                return leaves((0 ..< list.count).map { list[$0] })
            }
            return [cache]
        }
    }

    private static func offsets(_ caches: [any KVCache]) -> [Int] {
        leaves(caches).map(\.offset)
    }

    /// The invariant, stated once.
    ///
    /// Restore may report a miss (0). What it may NOT do is report tokens
    /// while some layer stayed at offset 0 — that is the fail-OPEN shape, and
    /// it is the one that produces confident wrong text.
    private static func assertFailedClosed(
        restoredTokens: Int,
        offsets: [Int],
        topology: String,
        damage: String
    ) {
        if restoredTokens == 0 {
            return  // an honest miss: the caller re-prefills
        }
        let empty = offsets.enumerated().filter { $0.element == 0 }.map(\.offset)
        #expect(
            empty.isEmpty,
            """
            \(topology): restore reported \(restoredTokens) tokens after \(damage), \
            but layer(s) \(empty) are still at offset 0. Attention would run \
            against an empty layer — worse than a miss.
            """)
    }

    // MARK: - Positive control
    //
    // Runs first and for every topology. Without it the fail-closed tests
    // below could all pass on a restore path that is simply broken.

    @Test("every shipped topology restores fully when the payload is intact")
    func undamagedPayloadRestoresEveryLayer() throws {
        try MLXMetalTestLock.withLock {
            for topo in Self.topologies {
                let dir = Self.tempDir()
                defer { try? FileManager.default.removeItem(at: dir) }

                let tokens = [1, 2, 3, 4, 5]
                let original = topo.make()
                Self.advance(original, steps: 3)
                let expected = Self.offsets(original)
                #expect(
                    expected.allSatisfy { $0 > 0 },
                    "\(topo.name): setup failed to advance — the whole suite would be vacuous")

                let writer = DiskCache(
                    cacheDir: dir, maxSizeBytes: 64 * 1_048_576, modelKey: topo.name)
                writer.store(tokens: tokens, arrays: TQDiskSerializer.serialize(cache: original))

                let reader = DiskCache(
                    cacheDir: dir, maxSizeBytes: 64 * 1_048_576, modelKey: topo.name)
                let arrays = try #require(
                    reader.fetch(tokens: tokens), "\(topo.name): nothing came back from disk")

                var target = topo.make()
                let restored = restoreFromDiskArrays(arrays, into: &target)

                if topo.name == "pure-ssm" {
                    // KNOWN GAP, deliberately pinned rather than hidden.
                    //
                    // `restoreFromV2Arrays` seeds `totalTokens` only from
                    // layers that carry a sequence dimension, and its `.mamba`
                    // branch says so outright: "no sequence dim to measure ...
                    // the attention side already provides that number." That
                    // holds for hybrids and is FALSE for an all-recurrent
                    // topology. This synthetic row is not proof that the named
                    // shipped mixed-attention families always miss.
                    //
                    // This is a PERFORMANCE defect, not a correctness one — it
                    // misses, which is the safe direction — so it is out of
                    // scope for the fail-closed work and tracked separately.
                    // The `.tq` and `.qkv` branches already show the fix shape
                    // (`if totalTokens == 0 { totalTokens = comp.offset }`),
                    // but enabling reuse for a family that has never had it
                    // needs its own proof that the newly-allowed path is
                    // correct, not a drive-by change here.
                    #expect(
                        restored == 0,
                        """
                        pure-SSM now restores (\(restored) tokens) — the known gap \
                        has been closed. Remove this branch and assert full restore.
                        """)
                    continue
                }

                #expect(restored > 0, "\(topo.name) [\(topo.families)]: intact payload missed")
                for (i, (before, after)) in zip(expected, Self.offsets(target)).enumerated() {
                    #expect(
                        after == before,
                        "\(topo.name): layer \(i) offset \(before) -> \(after)")
                }
            }
        }
    }

    // MARK: - Damage: a dropped layer-kind tag

    /// A truncated or partially-written payload loses trailing keys. Every
    /// layer is tagged, so dropping one tag is the minimal realistic damage.
    @Test("a dropped layer-kind tag fails closed on every architecture")
    func droppedKindTagFailsClosed() throws {
        try MLXMetalTestLock.withLock {
            for topo in Self.topologies {
                let original = topo.make()
                Self.advance(original, steps: 2)

                for victim in 0..<original.count {
                    var arrays = TQDiskSerializer.serialize(cache: original)
                    guard arrays.removeValue(forKey: "__layer_kind_\(victim)__") != nil else {
                        Issue.record("\(topo.name): layer \(victim) carried no kind tag")
                        continue
                    }

                    var target = topo.make()
                    let restored = restoreFromDiskArrays(arrays, into: &target)
                    Self.assertFailedClosed(
                        restoredTokens: restored,
                        offsets: Self.offsets(target),
                        topology: "\(topo.name) [\(topo.families)]",
                        damage: "dropping the kind tag for layer \(victim)")
                }
            }
        }
    }

    // MARK: - Damage: a dropped data payload

    /// The nastier case. The tag survives — so the record still LOOKS
    /// well-formed and routes to a real per-kind decoder — but the tensor it
    /// needs is gone. This is the shape that used to decode against an empty
    /// layer rather than refuse.
    @Test("a dropped data tensor fails closed on every architecture")
    func droppedDataPayloadFailsClosed() throws {
        try MLXMetalTestLock.withLock {
            for topo in Self.topologies {
                let original = topo.make()
                Self.advance(original, steps: 2)
                let intact = TQDiskSerializer.serialize(cache: original)

                // Every non-tag key, one at a time: no assumption about which
                // tensor names a given kind happens to use.
                let dataKeys = intact.keys.filter { !$0.hasPrefix("__layer_kind_") }.sorted()
                #expect(!dataKeys.isEmpty, "\(topo.name): serializer emitted no data tensors")

                for key in dataKeys {
                    var arrays = intact
                    arrays.removeValue(forKey: key)

                    var target = topo.make()
                    let restored = restoreFromDiskArrays(arrays, into: &target)
                    Self.assertFailedClosed(
                        restoredTokens: restored,
                        offsets: Self.offsets(target),
                        topology: "\(topo.name) [\(topo.families)]",
                        damage: "dropping data tensor '\(key)'")
                }
            }
        }
    }

    @Test("invalid composite metadata misses before mutating any child")
    func invalidCompositeMetadataIsAtomic() throws {
        try MLXMetalTestLock.withLock {
            let topo = try #require(Self.topologies.first { $0.name == "composite-hybrid" })
            let original = topo.make()
            Self.advance(original, steps: 2)
            let intact = TQDiskSerializer.serialize(cache: original)
            // Small invalid values exercise the same bounds as a corrupt huge
            // count without ever requesting an unbounded baseline allocation.
            let mutations: [(String, Int32)] = [
                ("__cache_list_1_count__", 0),
                ("__cache_list_1_count__", -1),
                ("__cache_list_1_count__", Int32(intact.count + 1)),
                ("__cache_list_1_sub_1_kind__", -1),
                (TQDiskSerializer.formatVersionKey, 0),
            ]
            var malformedShape = intact
            malformedShape["kv_1_sub_1_keys"] = MLXArray.ones([2, 8])
            var malformedTarget = topo.make()
            #expect(restoreFromDiskArrays(malformedShape, into: &malformedTarget) == 0)
            #expect(Self.offsets(malformedTarget).allSatisfy { $0 == 0 })
            for (key, value) in mutations {
                var arrays = intact
                arrays[key] = MLXArray([value])
                var target = topo.make()
                #expect(restoreFromDiskArrays(arrays, into: &target) == 0, "\(key)=\(value)")
                #expect(Self.offsets(target).allSatisfy { $0 == 0 }, "\(key)=\(value)")
            }
        }
    }

    @Test("composite rotating snapshots retain wrapped-window offsets")
    func rotatingCompositeWindowRoundTrip() throws {
        try MLXMetalTestLock.withLock {
            let topo = try #require(Self.topologies.first { $0.name == "composite-rotating" })
            let original = topo.make()
            Self.advance(original, steps: 12)
            let arrays = TQDiskSerializer.serialize(cache: original)
            var target = topo.make()
            #expect(restoreFromDiskArrays(arrays, into: &target) == 24)
            #expect(Self.offsets(target) == Self.offsets(original))
            for field in [0, 1, 4] {
                var damaged = arrays
                let key = "__rot_1_sub_1_meta__"
                var meta = try #require(arrays[key]).asArray(Int32.self)
                meta[field] = 999
                damaged[key] = MLXArray(meta)
                var fresh = topo.make()
                #expect(restoreFromDiskArrays(damaged, into: &fresh) == 0)
                #expect(Self.offsets(fresh).allSatisfy { $0 == 0 })
            }
        }
    }

    @Test("legacy attention snapshots cannot restore hybrid companion state")
    func legacyHybridMissKeepsPlainKVCompatibility() throws {
        try MLXMetalTestLock.withLock {
            let keys = MLXArray.ones([1, 1, 4, 8])
            let arrays = ["b0_l0_keys": keys, "b0_l0_values": keys * 2]
            var plain: [any KVCache] = [KVCacheSimple()]
            #expect(restoreFromDiskArrays(arrays, into: &plain) == 4)
            #expect(Self.offsets(plain) == [4])
            var hybrid: [any KVCache] = [MambaCache(), KVCacheSimple()]
            #expect(restoreFromDiskArrays(arrays, into: &hybrid) == 0)
            #expect(Self.offsets(hybrid) == [0, 0])
            for requireBoundary in [false, true] {
                var sparse: [any KVCache] = [QSAKVCache()]
                #expect(restoreFromDiskArrays(
                    arrays, into: &sparse, requirePromptBoundary: requireBoundary) == 0)
                #expect(Self.offsets(sparse) == [0])
            }
        }
    }

    // MARK: - Damage: truncation

    /// Whole-tail loss, as a partially-flushed write leaves behind. Distinct
    /// from single-key removal because it takes tags and data together.
    @Test("a truncated payload fails closed on every architecture")
    func truncatedPayloadFailsClosed() throws {
        try MLXMetalTestLock.withLock {
            for topo in Self.topologies {
                let original = topo.make()
                Self.advance(original, steps: 2)
                let intact = TQDiskSerializer.serialize(cache: original)
                let ordered = intact.keys.sorted()

                // Keep the first half; drop the rest.
                for keep in [1, ordered.count / 2, max(1, ordered.count - 1)] {
                    var arrays: [String: MLXArray] = [:]
                    for k in ordered.prefix(keep) { arrays[k] = intact[k] }

                    var target = topo.make()
                    let restored = restoreFromDiskArrays(arrays, into: &target)
                    Self.assertFailedClosed(
                        restoredTokens: restored,
                        offsets: Self.offsets(target),
                        topology: "\(topo.name) [\(topo.families)]",
                        damage: "truncating to the first \(keep) of \(ordered.count) keys")
                }
            }
        }
    }

    // MARK: - Damage: an empty payload

    /// The degenerate end of truncation. Must be a miss, never a success.
    @Test("an empty payload is always a miss")
    func emptyPayloadIsAMiss() throws {
        try MLXMetalTestLock.withLock {
            for topo in Self.topologies {
                var target = topo.make()
                let restored = restoreFromDiskArrays([:], into: &target)
                #expect(restored == 0, "\(topo.name): empty payload reported \(restored) tokens")
                #expect(
                    Self.offsets(target).allSatisfy { $0 == 0 },
                    "\(topo.name): empty payload advanced a layer")
            }
        }
    }
}
