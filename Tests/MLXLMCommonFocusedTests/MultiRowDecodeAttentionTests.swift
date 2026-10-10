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
    func relativeError(rows: Int, gqa: Int, keys: Int, blocks: Int?, dim: Int = 128) -> Float {
        let hkv = 2, hq = hkv * gqa
        MLXRandom.seed(UInt64(rows * 1000 + keys))
        let cap = (keys / 256 + 1) * 256
        let kBuffer = MLXRandom.normal([1, hkv, cap, dim]).asType(.bfloat16)
        let vBuffer = MLXRandom.normal([1, hkv, cap, dim]).asType(.bfloat16)
        let k = kBuffer[0..., 0..., ..<keys, 0...], v = vBuffer[0..., 0..., ..<keys, 0...]
        let q = (MLXRandom.normal([1, hq, rows, dim]) * 1.5).asType(.bfloat16)
        let scale = Float(1 / Double(dim).squareRoot())
        let reference = MLXFast.scaledDotProductAttention(
            queries: q.asType(.float32), keys: k.asType(.float32), values: v.asType(.float32), scale: scale,
            mask: .causal)
        let out = MultiRowDecodeAttention.attend(queries: q, keys: k, values: v, scale: scale, blocks: blocks)
        #expect(out.shape == [1, hq, rows, dim])
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

    @Test("head_dim 256 and pair groups beyond 64 (Qwen3.8 27B gqa 6, Flash-Next gqa 12)", arguments: [
        (16, 6, 1_037, 32), (8, 12, 2_049, 64), (2, 6, 300, nil), (16, 12, 700, 32),
    ] as [(Int, Int, Int, Int?)])
    func matchesReferenceDim256(rows: Int, gqa: Int, keys: Int, blocks: Int?) {
        #expect(relativeError(rows: rows, gqa: gqa, keys: keys, blocks: blocks, dim: 256) < 4e-3)
    }

    @Test("draft-tree verify: prefix visible, window by root path (DFlash2 tree plan mask)", arguments: [
        (128, 4, 1_500), (256, 6, 2_100), (256, 12, 640),
    ])
    func treeMatchesReference(dim: Int, gqa: Int, prefix: Int) {
        // 12-row tree: root, two children, grandchildren on both branches.
        let parents = [-1, 0, 0, 1, 1, 2, 3, 3, 5, 6, 8, 9]
        let plan = DFlash2TreePlan(tokens: Array(0 ..< parents.count), parents: parents)
        let rows = plan.rows, hkv = 2, hq = hkv * gqa, n = prefix + rows
        MLXRandom.seed(UInt64(dim * 10 + gqa))
        let k = MLXRandom.normal([1, hkv, n, dim]).asType(.bfloat16)
        let v = MLXRandom.normal([1, hkv, n, dim]).asType(.bfloat16)
        let q = (MLXRandom.normal([1, hq, rows, dim]) * 1.5).asType(.bfloat16)
        let scale = Float(1 / Double(dim).squareRoot())
        let mask = plan.attentionMask(prefix: prefix, dtype: .float32)
        let reference = MLXFast.scaledDotProductAttention(
            queries: q.asType(.float32), keys: k.asType(.float32), values: v.asType(.float32), scale: scale,
            mask: .array(mask))
        let out = MultiRowDecodeAttention.attend(
            queries: q, keys: k, values: v, scale: scale,
            windowBits: MultiRowDecodeAttention.windowBits(plan.windowVisibility))
        let d = out.asType(.float32) - reference
        let rel = (sqrt((d * d).mean()) / sqrt((reference * reference).mean())).item(Float.self)
        #expect(rel < 4e-3)
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

    @Test("eligibility: 2-16 rows, head_dim 128/256, bf16, gqa*rows <= 256; array masks only when declared causal")
    func eligibility() {
        let q = MLXArray.zeros([1, 32, 16, 128], dtype: .bfloat16)
        let kv = MLXArray.zeros([1, 8, 100, 128], dtype: .bfloat16)
        #expect(MultiRowDecodeAttention.eligible(queries: q, keys: kv, values: kv))
        #expect(!MultiRowDecodeAttention.eligible(queries: q[0..., 0..., ..<1, 0...], keys: kv, values: kv))
        #expect(!MultiRowDecodeAttention.eligible(
            queries: MLXArray.zeros([1, 32, 17, 128], dtype: .bfloat16), keys: kv, values: kv))
        #expect(!MultiRowDecodeAttention.eligible(queries: q.asType(.float16), keys: kv, values: kv))
        let q256 = MLXArray.zeros([1, 24, 16, 256], dtype: .bfloat16)
        let kv256 = MLXArray.zeros([1, 4, 100, 256], dtype: .bfloat16)
        #expect(MultiRowDecodeAttention.eligible(queries: q256, keys: kv256, values: kv256))
        #expect(!MultiRowDecodeAttention.eligible(
            queries: MLXArray.zeros([1, 32, 4, 64], dtype: .bfloat16),
            keys: MLXArray.zeros([1, 8, 100, 64], dtype: .bfloat16),
            values: MLXArray.zeros([1, 8, 100, 64], dtype: .bfloat16)))
        let mask = MLXArray.ones([16, 100], type: Bool.self)
        let previous = MultiRowDecodeAttention.arrayMaskIsCausal
        MultiRowDecodeAttention.arrayMaskIsCausal = false
        #expect(!MultiRowDecodeAttention.maskAllowed(.array(mask), queryLength: 16, keyLength: 100))
        #expect(MultiRowDecodeAttention.maskAllowed(.causal, queryLength: 16, keyLength: 100))
        MultiRowDecodeAttention.withLinearChainMask {
            #expect(MultiRowDecodeAttention.maskAllowed(.array(mask), queryLength: 16, keyLength: 100))
            #expect(!MultiRowDecodeAttention.maskAllowed(.array(mask), queryLength: 8, keyLength: 100))
            let mask4 = MLXArray.ones([1, 1, 16, 100], type: Bool.self)
            #expect(MultiRowDecodeAttention.maskAllowed(.array(mask4), queryLength: 16, keyLength: 100))
            let perHead = MLXArray.ones([1, 4, 16, 100], type: Bool.self)
            #expect(!MultiRowDecodeAttention.maskAllowed(.array(perHead), queryLength: 16, keyLength: 100))
        }
        MultiRowDecodeAttention.arrayMaskIsCausal = previous
    }
}
