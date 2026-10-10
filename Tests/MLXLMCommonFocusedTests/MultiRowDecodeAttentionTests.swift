// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import Testing

/// The multi-row decode attention kernel against an fp32 SDPA reference: bottom-right causal over a cached prefix,
/// GQA, a KV view sliced from a larger buffer (as KVCacheSimple returns it), and a key count that is not a multiple of
/// the 32-key tile or of the block split.
@Suite("Multi-row decode attention", .serialized)
struct MultiRowDecodeAttentionTests {
    func relativeError(rows: Int, gqa: Int, keys: Int, blocks: Int?) -> Float {
        let hkv = 2, hq = hkv * gqa
        MLXRandom.seed(UInt64(rows * 1000 + keys))
        let cap = (keys / 256 + 1) * 256
        let kBuffer = MLXRandom.normal([1, hkv, cap, 128]).asType(.bfloat16)
        let vBuffer = MLXRandom.normal([1, hkv, cap, 128]).asType(.bfloat16)
        let k = kBuffer[0..., 0..., ..<keys, 0...], v = vBuffer[0..., 0..., ..<keys, 0...]
        let q = (MLXRandom.normal([1, hq, rows, 128]) * 1.5).asType(.bfloat16)
        let scale = Float(1 / 128.0.squareRoot())
        let reference = MLXFast.scaledDotProductAttention(
            queries: q.asType(.float32), keys: k.asType(.float32), values: v.asType(.float32), scale: scale,
            mask: .causal)
        let out = MultiRowDecodeAttention.attend(queries: q, keys: k, values: v, scale: scale, blocks: blocks)
        #expect(out.shape == [1, hq, rows, 128])
        #expect(out.dtype == .bfloat16)
        let d = out.asType(.float32) - reference
        return (sqrt((d * d).mean()) / sqrt((reference * reference).mean())).item(Float.self)
    }

    @Test("matches fp32 SDPA within bf16 output rounding", arguments: [
        (2, 4, 777, 32), (8, 4, 1_000, 64), (12, 4, 4_099, nil), (16, 4, 5_003, 64), (5, 8, 333, 32),
    ] as [(Int, Int, Int, Int?)])
    func matchesReference(rows: Int, gqa: Int, keys: Int, blocks: Int?) {
        // MLX's own bf16 SDPA lands at ~2.3e-3 on these shapes; the kernel accumulates in fp32 (~1.7e-3).
        #expect(relativeError(rows: rows, gqa: gqa, keys: keys, blocks: blocks) < 4e-3)
    }

    @Test("bottom-right causal: the first row never sees the last rows' keys")
    func causalVisibility() {
        let keys = 300, rows = 4
        let k = MLXRandom.normal([1, 1, keys, 128]).asType(.bfloat16)
        var vRows = [MLXArray](repeating: MLXArray.zeros([1, 1, 1, 128], dtype: .bfloat16), count: keys)
        // Only the last key carries signal: rows 0...rows-2 must not attend to it.
        vRows[keys - 1] = MLXArray.ones([1, 1, 1, 128], dtype: .bfloat16) * 100
        let v = concatenated(vRows, axis: 2)
        let q = MLXRandom.normal([1, 4, rows, 128]).asType(.bfloat16)
        let out = MultiRowDecodeAttention.attend(queries: q, keys: k, values: v, scale: 0.0884, blocks: 32)
        let firstRows = abs(out[0..., 0..., ..<(rows - 1), 0...]).max().item(Float.self)
        let lastRow = abs(out[0..., 0..., (rows - 1)..., 0...]).max().item(Float.self)
        #expect(firstRows == 0)
        #expect(lastRow > 0)
    }

    @Test("eligibility: 2-16 rows, head_dim 128, bf16, gqa*rows <= 64; array masks only when declared causal")
    func eligibility() {
        let q = MLXArray.zeros([1, 32, 16, 128], dtype: .bfloat16)
        let kv = MLXArray.zeros([1, 8, 100, 128], dtype: .bfloat16)
        #expect(MultiRowDecodeAttention.eligible(queries: q, keys: kv, values: kv))
        #expect(!MultiRowDecodeAttention.eligible(queries: q[0..., 0..., ..<1, 0...], keys: kv, values: kv))
        #expect(!MultiRowDecodeAttention.eligible(
            queries: MLXArray.zeros([1, 32, 17, 128], dtype: .bfloat16), keys: kv, values: kv))
        #expect(!MultiRowDecodeAttention.eligible(queries: q.asType(.float16), keys: kv, values: kv))
        let mask = MLXArray.ones([16, 100], type: Bool.self)
        let previous = MultiRowDecodeAttention.arrayMaskIsCausal
        MultiRowDecodeAttention.arrayMaskIsCausal = false
        #expect(!MultiRowDecodeAttention.maskAllowed(.array(mask), queryLength: 16, keyLength: 100))
        #expect(MultiRowDecodeAttention.maskAllowed(.causal, queryLength: 16, keyLength: 100))
        MultiRowDecodeAttention.withLinearChainMask {
            #expect(MultiRowDecodeAttention.maskAllowed(.array(mask), queryLength: 16, keyLength: 100))
            #expect(!MultiRowDecodeAttention.maskAllowed(.array(mask), queryLength: 8, keyLength: 100))
        }
        MultiRowDecodeAttention.arrayMaskIsCausal = previous
    }
}
