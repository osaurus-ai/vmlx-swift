// SPDX-License-Identifier: Apache-2.0
//
// Regression tests for RotatingKVCache L2 disk persistence in
// `CacheCoordinator.storeAfterGeneration`.
//
// HISTORY: this file originally pinned a SKIP guard — the central
// `hasRotatingLayer` check that suppressed disk + memory writes when
// any layer was a `RotatingKVCache`. That guard existed because
// `TQDiskSerializer` v2 had no `.rotating` LayerKind tag and would
// emit `.skip` placeholders that lost the wrap state silently.
//
// SLIDING-1 (2026-04-15) added the missing `.rotating` LayerKind to
// the v2 schema. RotatingKVCache now round-trips cleanly via
// `serializeRotatingLayer` / `restoreRotatingLayer`, with the full
// 5-tuple metaState `(keep, maxSize, step, offset, idx)` captured in
// `__rot_{i}_meta__`. This file now pins the OPPOSITE contract:
// disk store MUST happen when a RotatingKVCache is present, AND the
// stored entry MUST be retrievable from disk afterwards.

import XCTest
import MLX
@testable import MLXLMCommon

final class CacheCoordinatorRotatingGuardTests: XCTestCase {

    private func makeCoordWithDisk() -> (CacheCoordinator, URL) {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx_rotating_guard_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: tmp, withIntermediateDirectories: true)
        var cfg = CacheCoordinatorConfig()
        cfg.enableDiskCache = true
        cfg.diskCacheDir = tmp
        cfg.diskCacheMaxGB = 1.0
        cfg.modelKey = "rotating-guard-test"
        return (CacheCoordinator(config: cfg), tmp)
    }

    /// SLIDING-1: cache lists containing a `RotatingKVCache` MUST persist
    /// to disk. The previous skip guard is gone — the v2 schema now has
    /// a `.rotating` LayerKind that captures both the ring buffer and
    /// the wrap-state metaState.
    func testStorePersistsRotatingCacheToDisk() {
        let (coord, dir) = makeCoordWithDisk()
        defer { try? FileManager.default.removeItem(at: dir) }

        // Build a small cache list: one KVCacheSimple + one RotatingKVCache,
        // BOTH prefilled to the same offset as the boundary key.
        //
        // This fixture predates #208 ("Refuse disk-cache stores whose cache
        // offset disagrees with the boundary key"), which is why it left `kv`
        // at offset 0 while `rot` sat at 8 and the store was — correctly —
        // refused, leaving this suite red on main. A real cache list advances
        // every layer together; a mismatched one is the poisoned-snapshot shape
        // that #208 exists to reject, and is pinned separately by
        // ``testStoreIsRefusedWhenLayerOffsetsDisagreeWithTheKey``.
        let kv = KVCacheSimple()
        let rot = RotatingKVCache(maxSize: 1024, keep: 0)
        let k = MLXArray.ones([1, 4, 8, 16], dtype: .bfloat16)
        let v = MLXArray.ones([1, 4, 8, 16], dtype: .bfloat16) * Float(0.5)
        _ = kv.update(keys: k, values: v)
        _ = rot.update(keys: k, values: v)
        XCTAssertEqual(kv.offset, 8)
        XCTAssertEqual(rot.offset, 8)

        coord.storeAfterGeneration(
            promptTokens: [1, 2, 3, 4, 5, 6, 7, 8],
            perLayerData: [],
            ssmStates: nil,
            cache: [kv, rot],
            mediaSalt: nil
        )

        let fetched = coord.diskCache?.fetch(
            tokens: [1, 2, 3, 4, 5, 6, 7, 8], mediaSalt: nil)
        XCTAssertNotNil(
            fetched,
            "SLIDING-1: disk store must include cache lists containing a " +
            "RotatingKVCache — the v2 .rotating LayerKind round-trips the " +
            "ring buffer + wrap state. If this fails, the central skip " +
            "guard returned. See CacheCoordinator.swift line 390 (gone).")

        // Verify the rotating layer's kind tag is present in the dict.
        if let arrays = fetched {
            XCTAssertTrue(
                arrays.keys.contains("__layer_kind_1__"),
                "RotatingKVCache layer must be tagged in the disk dict.")
            if let kindArr = arrays["__layer_kind_1__"] {
                let kind = kindArr.shape.isEmpty
                    ? kindArr.item(Int32.self)
                    : kindArr[0].item(Int32.self)
                XCTAssertEqual(
                    kind,
                    TQDiskSerializer.LayerKind.rotating.rawValue,
                    "Rotating layer must be tagged .rotating, got \(kind).")
            }
            XCTAssertTrue(
                arrays.keys.contains("rot_1_keys"),
                "Ring buffer keys must be present at rot_1_keys.")
            XCTAssertTrue(
                arrays.keys.contains("__rot_1_meta__"),
                "5-tuple metaState must be present at __rot_1_meta__.")
        }
    }

    /// The other half of the contract, and the reason the fixture above had to
    /// be repaired rather than the guard relaxed: a cache list whose layers do
    /// not all sit at the boundary key's token count must be refused outright.
    ///
    /// #208's rationale, verbatim from the call site: a poisoned entry
    /// content-matches the key on a later fetch, restores one more token of
    /// state than the key claims, and the engine re-feeds a token the cache
    /// already holds — every later position shifts by one and each following
    /// turn stores a seed derived from the corrupted state. A skipped entry
    /// costs one prefill; a poisoned one costs correctness for the rest of the
    /// conversation. Nothing pinned that behaviour until now.
    func testStoreIsRefusedWhenLayerOffsetsDisagreeWithTheKey() {
        let (coord, dir) = makeCoordWithDisk()
        defer { try? FileManager.default.removeItem(at: dir) }

        let kv = KVCacheSimple()
        let rot = RotatingKVCache(maxSize: 1024, keep: 0)
        let k = MLXArray.ones([1, 4, 8, 16], dtype: .bfloat16)
        let v = MLXArray.ones([1, 4, 8, 16], dtype: .bfloat16) * Float(0.5)
        // Only the rotating layer advances: offsets are {0, 8} against a
        // boundary key that claims 8.
        _ = rot.update(keys: k, values: v)

        coord.storeAfterGeneration(
            promptTokens: [1, 2, 3, 4, 5, 6, 7, 8],
            perLayerData: [],
            ssmStates: nil,
            cache: [kv, rot],
            mediaSalt: nil
        )

        XCTAssertNil(
            coord.diskCache?.fetch(tokens: [1, 2, 3, 4, 5, 6, 7, 8], mediaSalt: nil),
            "a cache list whose offsets disagree with the boundary key must not "
                + "reach disk — the entry would restore more state than its key claims")
    }

    /// SLIDING-1 edge case: `CacheList` wrapping a `RotatingKVCache`
    /// (BaichuanM1 / FalconH1 sliding+mamba mix). The serializer's
    /// outer dispatch sees the CacheList not the inner rotating, so
    /// this currently lands on `.skip` — restore re-prefills the
    /// wrapped layer naturally on the next turn. Pinned to confirm
    /// the disk write does NOT throw and the outer store returns
    /// successfully (no central skip guard remains).
    func testStoreCacheListWrappedRotatingDoesNotThrow() {
        let (coord, dir) = makeCoordWithDisk()
        defer { try? FileManager.default.removeItem(at: dir) }

        let rot = RotatingKVCache(maxSize: 1024, keep: 0)
        let kv = KVCacheSimple()
        let wrapped = CacheList(rot, kv)

        // Call must not throw; the outer CacheList layer lands on
        // `.skip` in v2 (no LayerKind for CacheList), but the OTHER
        // layers in the cache (none here) would still serialize.
        coord.storeAfterGeneration(
            promptTokens: [7, 8, 9],
            perLayerData: [],
            ssmStates: nil,
            cache: [wrapped],
            mediaSalt: nil
        )
        // No assertion on the disk fetch shape — empty stores can fall
        // through to no-op. The contract here is "doesn't crash".
    }
}
