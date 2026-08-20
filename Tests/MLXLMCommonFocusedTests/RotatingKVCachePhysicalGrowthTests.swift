// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Pins RotatingKVCache growth sizing to the buffer's PHYSICAL fill.
//
// Restore paths force a large absolute `offset` onto a fresh (or short)
// drafter cache for RoPE-position continuity. Growth that sizes new buffer
// chunks from `maxCacheSize - offset` goes negative in that state and dies
// inside MLX as "[full] Negative dimensions not allowed" — the residual
// DFlash2 drafter failure past the sliding window that #281 could only
// contain. Growth keyed on physical fill is behavior-identical in the
// healthy path (offset == fill) and correct under forced offsets.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("RotatingKVCache physical-fill growth")
struct RotatingKVCachePhysicalGrowthTests {

    private func kv(_ s: Int) -> (MLXArray, MLXArray) {
        (
            MLXArray.zeros([1, 2, s, 8], dtype: .float16),
            MLXArray.zeros([1, 2, s, 8], dtype: .float16)
        )
    }

    @Test("fresh cache with a restore-forced offset past the window still grows and writes")
    func forcedOffsetOnFreshCache() {
        let cache = RotatingKVCache(maxSize: 2048, keep: 0, step: 256)
        // Restore-style: absolute conversation offset forced onto an empty
        // buffer. 3123 matches the live ctx where the drafter died.
        cache.offset = 3123

        let (k, v) = kv(1)
        let (outK, _) = cache.update(keys: k, values: v)

        #expect(outK.dim(2) >= 1)
        #expect(cache.offset == 3124)

        // And it keeps accepting single-token decode writes.
        for _ in 0 ..< 8 {
            let (k1, v1) = kv(1)
            _ = cache.update(keys: k1, values: v1)
        }
        #expect(cache.offset == 3132)
    }

    @Test("forced offset onto a SHORT existing buffer grows from fill, not offset")
    func forcedOffsetOnShortBuffer() {
        let cache = RotatingKVCache(maxSize: 2048, keep: 0, step: 256)
        // Seed a small physical buffer the normal way.
        let (k0, v0) = kv(4)
        _ = cache.update(keys: k0, values: v0)
        // Force the logical offset far past both the fill and the window.
        cache.offset = 2500

        let (k, v) = kv(1)
        _ = cache.update(keys: k, values: v)
        #expect(cache.offset == 2501)
    }

    @Test("healthy growth path is unchanged: fill tracks offset through the window")
    func healthyPathUnchanged() {
        let cache = RotatingKVCache(maxSize: 64, keep: 0, step: 16)
        for i in 0 ..< 96 {
            let (k, v) = kv(1)
            let (outK, _) = cache.update(keys: k, values: v)
            #expect(cache.offset == i + 1)
            #expect(outK.dim(2) <= 64)
        }
    }
}
