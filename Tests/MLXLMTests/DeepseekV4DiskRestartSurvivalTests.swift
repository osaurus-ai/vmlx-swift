// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DSV4 0731 cache: does it actually survive a RESTART, and does it fail
// closed when a payload is damaged?
//
// An audit of the existing DSV4 disk coverage found every test stopping short
// of the failure mode that matters:
//
//   - `DeepseekV4CacheDiskRoundTripTests` round-trips `state`/`metaState`
//     in memory but never writes a real safetensors file.
//   - `DeepseekV4PartialRestoreContinuationTests` continues after restore but
//     only calls `cache.update(kv,kv)`, so compressor/indexer state is never
//     advanced.
//   - `dsv4PagedIncompatibleUsesDiskWithPools` writes a real entry and reads
//     it back, but stops at the returned `diskArrays` — it never restores them
//     into a FRESH cache, so nothing proves the restored object behaves like
//     the original.
//
// The gap they leave is precisely "cold vs warm after a process restart",
// which is what a user hits when they quit the app and come back to a long
// conversation. These tests write through a real `DiskCache` on a real
// directory, then restore into caches built from scratch — the closest thing
// to a relaunch that a unit test can be — and assert the restored state
// matches, rather than merely being present.
//
// The negative cases matter as much as the positive one: a cache that restores
// SOMETHING for a damaged payload is worse than one that misses, because the
// miss costs a re-prefill while a bad restore silently corrupts the answer.

import Foundation
import MLX
import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("DSV4 disk restart survival", .serialized)
struct DeepseekV4DiskRestartSurvivalTests {

    private static func tempDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dsv4-restart-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// A mixed 0731-shaped topology: rotating layers interleaved with DSV4
    /// layers at two different compression ratios. Mixed is the case that
    /// catches kind-tag drift — a uniform array can restore correctly even if
    /// the per-layer tag is being ignored.
    private static func makeMixedCache() -> [any KVCache] {
        [
            RotatingKVCache(maxSize: 16, keep: 0),
            DeepseekV4Cache(slidingWindow: 16, compressRatio: 4),
            RotatingKVCache(maxSize: 16, keep: 0),
            DeepseekV4Cache(slidingWindow: 16, compressRatio: 128),
        ]
    }

    private static func advance(_ caches: [any KVCache], steps: Int, headDim: Int = 8) {
        for s in 0..<steps {
            for c in caches {
                let k = MLXArray.ones([1, 1, 2, headDim]) * Float(s + 1)
                let v = MLXArray.ones([1, 1, 2, headDim]) * Float(s + 2)
                _ = c.update(keys: k, values: v)
            }
        }
        MLX.eval(caches)
    }

    // MARK: - The restart case

    /// Write through a real DiskCache, then restore into caches constructed
    /// from scratch — no object from the writing side survives. This is the
    /// unit-test analogue of quitting the app and reopening a long chat.
    @Test("DSV4 mixed topology survives a write/restore across fresh caches")
    func mixedTopologySurvivesRestart() throws {
        try MLXMetalTestLock.withLock {
            let dir = Self.tempDir()
            defer { try? FileManager.default.removeItem(at: dir) }

            let tokens = [11, 22, 33, 44, 55, 66]
            let original = Self.makeMixedCache()
            Self.advance(original, steps: 3)

            let expectedOffsets = original.map(\.offset)
            #expect(expectedOffsets.allSatisfy { $0 > 0 }, "setup did not advance the caches")

            // Real file, real SQLite index — not an in-memory dictionary.
            let writer = DiskCache(cacheDir: dir, maxSizeBytes: 64 * 1_048_576, modelKey: "dsv4")
            writer.store(tokens: tokens, arrays: TQDiskSerializer.serialize(cache: original))

            let onDisk = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
            #expect(
                onDisk.contains { $0.hasSuffix(".safetensors") },
                "nothing was actually written to disk: \(onDisk)")

            // A brand-new DiskCache over the same directory: the writing
            // instance contributes nothing beyond the bytes it left behind.
            let reader = DiskCache(cacheDir: dir, maxSizeBytes: 64 * 1_048_576, modelKey: "dsv4")
            let restoredArrays = try #require(
                reader.fetch(tokens: tokens), "entry did not come back from disk")

            var target = Self.makeMixedCache()
            // NOTE: the return is a TOKEN count ("total number of tokens
            // restored... or 0 if nothing matched"), not a layer count.
            let restoredTokens = restoreFromDiskArrays(restoredArrays, into: &target)
            #expect(restoredTokens > 0, "restore reported a miss for an entry that is on disk")

            // Offsets must match layer for layer. A restore that returns the
            // right number of layers with the wrong offsets is the shape of
            // bug that produces coherent-but-wrong text.
            for (i, (before, after)) in zip(expectedOffsets, target.map(\.offset)).enumerated() {
                #expect(after == before, "layer \(i) offset drifted: \(before) -> \(after)")
            }
        }
    }

    /// The restored cache must keep ADVANCING correctly, not merely load. A
    /// restored-then-continued cache is what turn N+1 actually uses.
    @Test("a restored DSV4 cache continues in step with one that never left memory")
    func restoredCacheContinuesInStep() throws {
        try MLXMetalTestLock.withLock {
            let dir = Self.tempDir()
            defer { try? FileManager.default.removeItem(at: dir) }

            let tokens = [7, 8, 9, 10]
            let live = Self.makeMixedCache()
            Self.advance(live, steps: 2)

            let cache = DiskCache(cacheDir: dir, maxSizeBytes: 64 * 1_048_576, modelKey: "dsv4-cont")
            cache.store(tokens: tokens, arrays: TQDiskSerializer.serialize(cache: live))
            let arrays = try #require(cache.fetch(tokens: tokens))

            var restored = Self.makeMixedCache()
            _ = restoreFromDiskArrays(arrays, into: &restored)

            // Same continuation applied to both.
            Self.advance(live, steps: 2)
            Self.advance(restored, steps: 2)

            for (i, (a, b)) in zip(live.map(\.offset), restored.map(\.offset)).enumerated() {
                #expect(
                    a == b,
                    "layer \(i) diverged after continuing: live=\(a) restored=\(b)")
            }
        }
    }

    // MARK: - Fail-closed cases
    //
    // A miss costs a re-prefill. A bad restore costs a wrong answer. These
    // pin that damaged payloads produce the former.

    @Test("a missing layer-kind tag does not silently half-restore")
    func missingKindTagDoesNotHalfRestore() throws {
        try MLXMetalTestLock.withLock {
            let original = Self.makeMixedCache()
            Self.advance(original, steps: 2)
            var arrays = TQDiskSerializer.serialize(cache: original)

            // Drop the kind tag for one DSV4 layer, as a truncated or partially
            // written payload would.
            let removed = arrays.removeValue(forKey: "__layer_kind_1__")
            #expect(removed != nil, "test assumes layer 1 is kind-tagged")

            var target = Self.makeMixedCache()
            let restoredTokens = restoreFromDiskArrays(arrays, into: &target)
            let offsets = target.map(\.offset)

            // The invariant that matters: if the restore reports tokens, EVERY
            // layer must carry them. A layer left at 0 while the coordinator is
            // told "n tokens restored" means attention runs against an empty
            // layer -- coherent-but-wrong output, which is strictly worse than
            // a miss and a re-prefill.
            if restoredTokens > 0 {
                #expect(
                    offsets.allSatisfy { $0 > 0 },
                    "FAIL-OPEN: reported \(restoredTokens) tokens restored but layer offsets are \(offsets)")
            }
        }
    }

    @Test("a missing rotating payload does not report a full restore")
    func missingRotatingPayloadDoesNotReportFullRestore() throws {
        try MLXMetalTestLock.withLock {
            let original = Self.makeMixedCache()
            Self.advance(original, steps: 2)
            var arrays = TQDiskSerializer.serialize(cache: original)

            let removed = arrays.removeValue(forKey: "rot_0_keys")
            #expect(removed != nil, "test assumes layer 0 writes rot_0_keys")

            var target = Self.makeMixedCache()
            let restoredTokens = restoreFromDiskArrays(arrays, into: &target)
            let offsets = target.map(\.offset)
            if restoredTokens > 0 {
                #expect(
                    offsets.allSatisfy { $0 > 0 },
                    "FAIL-OPEN: reported \(restoredTokens) tokens restored but layer offsets are \(offsets)")
            }
        }
    }

    @Test("an empty payload restores nothing rather than a zeroed cache")
    func emptyPayloadRestoresNothing() throws {
        try MLXMetalTestLock.withLock {
            var target = Self.makeMixedCache()
            let restoredTokens = restoreFromDiskArrays([:], into: &target)
            #expect(restoredTokens == 0)
            #expect(
                target.allSatisfy { $0.offset == 0 },
                "an empty payload advanced a cache offset")
        }
    }
}
