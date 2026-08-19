// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// `CompilableKVCache.copy()` must produce a SNAPSHOT, not a live view.
//
// This cache mutates in place on purpose: `compile()` captures the array
// objects as tracked state and expects them to be updated, not rebound.
// That makes a reference-sharing `copy()` silently wrong — the "snapshot"
// keeps following the cache. It matters because `copy()` is what the
// prefix-cache store path uses, so a drifting copy persists whatever the
// cache held later instead of what was captured.

import Foundation
import MLX
import XCTest
@testable import MLXLMCommon

final class CompilableKVCacheSnapshotTests: XCTestCase {

    private func makeCache(offset: Int) -> CompilableKVCache {
        let cache = CompilableKVCache(maxLength: 64)
        let keys = MLXArray.zeros([1, 2, 4, 8], dtype: .float32)
        let values = MLXArray.zeros([1, 2, 4, 8], dtype: .float32)
        _ = cache.update(keys: keys, values: values)
        cache.offset = offset
        return cache
    }

    func testCopyDoesNotFollowLaterOffsetWrites() {
        let cache = makeCache(offset: 12)
        let snapshot = cache.copy()
        XCTAssertEqual(snapshot.offset, 12)

        cache.offset = 40
        XCTAssertEqual(cache.offset, 40, "live cache should advance")
        XCTAssertEqual(
            snapshot.offset, 12,
            "snapshot followed the live cache — copy() shared the offset array")
    }

    func testCopyDoesNotFollowLaterTrims() {
        let cache = makeCache(offset: 30)
        let snapshot = cache.copy()

        _ = cache.trim(10)
        XCTAssertEqual(cache.offset, 20, "trim should rewind the live cache")
        XCTAssertEqual(
            snapshot.offset, 30,
            "snapshot followed a trim — copy() shared the offset array")
    }

    func testCopyDoesNotFollowLaterKeyWrites() {
        let cache = makeCache(offset: 0)
        let snapshot = cache.copy()
        let before = snapshot.state.first.map { MLX.sum($0).item(Float.self) } ?? 0

        // A second update mutates the SAME key buffer in place.
        _ = cache.update(
            keys: MLXArray.ones([1, 2, 4, 8], dtype: .float32),
            values: MLXArray.ones([1, 2, 4, 8], dtype: .float32))

        let after = snapshot.state.first.map { MLX.sum($0).item(Float.self) } ?? 0
        XCTAssertEqual(
            before, after, accuracy: 0.0001,
            "snapshot's keys changed when the live cache was updated")
    }
}
