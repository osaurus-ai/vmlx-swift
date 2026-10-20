// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// `Qwen35CausalPrefillMask` swaps the model's explicit causal mask for SDPA's `.causal` mode on
// prefill chunks that continue a cached prefix. That is only valid if `.causal` is lower-right
// aligned (the last query row sees every key) for query length < key length, on both the fused
// (NAX, >= 1,024 rows, head dim 256) and the fallback kernels.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import Testing

@testable import MLXVLM

@Suite("Qwen3.5 causal prefill mask", .serialized)
struct Qwen35CausalPrefillMaskTests {
    @Test("causal mode equals the explicit offset-aware mask", arguments: [(64, 300), (1024, 1500), (2048, 2560)])
    func causalMatchesExplicitMask(rows: Int, keys: Int) throws {
        try MLXMetalTestLock.withLock {
            let q = (MLXRandom.normal([1, 32, rows, 256], key: MLXRandom.key(1)) * 0.5).asType(.bfloat16)
            let k = (MLXRandom.normal([1, 2, keys, 256], key: MLXRandom.key(2)) * 0.5).asType(.bfloat16)
            let v = (MLXRandom.normal([1, 2, keys, 256], key: MLXRandom.key(3)) * 0.5).asType(.bfloat16)
            let explicit = createCausalMask(n: rows, offset: keys - rows)
            let a = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: 0.0625, mask: .array(explicit))
            let b = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: 0.0625, mask: .causal)
            let err = MLX.max(abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
            #expect(err < 2e-2, "max |causal - explicit| = \(err)")
            // And not trivially: an upper-left aligned mask would differ.
            let upperLeft = createCausalMask(n: rows, offset: 0)
            let padded = concatenated(
                [upperLeft, MLXArray.zeros([rows, keys - rows], dtype: .bool)], axis: 1)
            let c = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: 0.0625, mask: .array(padded))
            #expect(MLX.max(abs(c.asType(.float32) - b.asType(.float32))).item(Float.self) > 0.1)
        }
    }

    @Test("applies only to plain single-sequence prefill chunks")
    func scope() {
        let cache = KVCacheSimple()
        _ = cache.update(keys: MLXArray.zeros([1, 2, 100, 8]), values: MLXArray.zeros([1, 2, 100, 8]))
        let mask = createCausalMask(n: 64, offset: 100)
        #expect(Qwen35CausalPrefillMask.applies(mask: mask, rows: 64, batch: 1, cache: cache, fullBufferCache: false))
        #expect(!Qwen35CausalPrefillMask.applies(mask: createCausalMask(n: 8, offset: 100), rows: 8, batch: 1, cache: cache, fullBufferCache: false))
        #expect(!Qwen35CausalPrefillMask.applies(mask: mask, rows: 64, batch: 2, cache: cache, fullBufferCache: false))
        #expect(!Qwen35CausalPrefillMask.applies(mask: mask, rows: 64, batch: 1, cache: cache, fullBufferCache: true))
        #expect(!Qwen35CausalPrefillMask.applies(mask: createCausalMask(n: 64, offset: 90), rows: 64, batch: 1, cache: cache, fullBufferCache: false))
        #expect(!Qwen35CausalPrefillMask.applies(mask: mask, rows: 64, batch: 1, cache: nil, fullBufferCache: false))
    }
}
