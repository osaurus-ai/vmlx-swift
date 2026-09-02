// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Precision gate for the MLA decode SDPA dtype change.
//
// The decode step (S == 1) historically cast Q/K/V to float32 before SDPA,
// which materializes a full-context fp32 copy of the KV cache per token per
// MLA layer — the dominant cost of long-context MLA decode. The fast path
// runs SDPA in the cache's own dtype (bf16). MLXFast's SDPA accumulates in
// fp32 internally either way, so the bf16 path must stay numerically inside
// bf16 rounding of the fp32 reference. This suite pins that at real MLA
// decode shapes (asymmetric K/V head dims, DeepSeek/Bailing geometry) and
// long context, including an adversarial large-logit case.

import Foundation
import MLX
import MLXFast
import MLXRandom
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct MLADecodeSDPADtypeTests {

    /// Runs both arms of the decode SDPA on identical inputs and returns
    /// (bf16Out, fp32Out) as float32 for comparison.
    private func runBothArms(
        heads: Int, context: Int, kDim: Int, vDim: Int, scale: Float,
        querySpread: Float = 1.0
    ) throws -> (MLXArray, MLXArray) {
        try FocusedMLXTestSupport.withLock {
            MLXRandom.seed(11)
            let q = (MLXRandom.normal([1, heads, 1, kDim]) * querySpread)
                .asType(.bfloat16)
            let k = MLXRandom.normal([1, heads, context, kDim]).asType(.bfloat16)
            let v = MLXRandom.normal([1, heads, context, vDim]).asType(.bfloat16)
            MLX.eval(q, k, v)

            let saved = MLAAttentionRuntime.decodeFP32Cast
            defer { MLAAttentionRuntime.decodeFP32Cast = saved }

            MLAAttentionRuntime.decodeFP32Cast = false
            let bf16 = mlaScaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale)
            MLX.eval(bf16)

            MLAAttentionRuntime.decodeFP32Cast = true
            let fp32 = mlaScaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale)
            MLX.eval(fp32)

            return (bf16.asType(.float32), fp32.asType(.float32))
        }
    }

    private func assertClose(
        _ bf16: MLXArray, _ fp32: MLXArray, label: String
    ) {
        // Attention outputs are convex combinations of unit-scale values, so
        // an absolute tolerance at bf16 rounding scale is the right gate:
        // bf16 has ~2-3 decimal digits (eps ~= 0.0078); allow a few ulps of
        // accumulated difference between the two softmax evaluations.
        let maxDiff = MLX.abs(bf16 - fp32).max().item(Float.self)
        #expect(
            maxDiff < 0.05,
            Comment(rawValue: "\(label): max |bf16 - fp32| = \(maxDiff)"))
    }

    @Test("DeepSeek/Bailing MLA decode geometry matches fp32 within bf16 rounding")
    func mlaGeometryLongContext() throws {
        // qk 192 (nope 128 + rope 64), v 128, 16 heads — Raptor/Ling/DSv3
        // decode shape — at a long context where the fp32 copy dominated.
        let (bf16, fp32) = try runBothArms(
            heads: 16, context: 4096, kDim: 192, vDim: 128,
            scale: pow(192, -0.5))
        assertClose(bf16, fp32, label: "192/128 @ 4096")
    }

    @Test("32k context: bf16 softmax reduction does not collapse at MLA dims")
    func longContextReduction() throws {
        // Gemma-4's global layers (head_dim 512 + softcap) collapse to <pad>
        // past ~26k when the unfused SDPA fallback reduces softmax in bf16 —
        // that is why ITS fp32 upcast is a correctness workaround, not a bug.
        // This pins that MLA dims (192/128) do NOT live in that regime: at
        // 32k keys the bf16 decode output must stay within bf16 rounding of
        // the fp32 arm. Live corroboration: Raptor at 25.9k context decodes
        // a coherent greedy summary (110 tok/s, clean stop, no loops).
        let (bf16, fp32) = try runBothArms(
            heads: 16, context: 32_768, kDim: 192, vDim: 128,
            scale: pow(192, -0.5))
        assertClose(bf16, fp32, label: "192/128 @ 32768")
    }

    @Test("large-magnitude queries stay bounded (softmax saturation case)")
    func largeLogitSaturation() throws {
        // 8x query spread pushes pre-softmax logits far from zero — the
        // regime where a naive low-precision softmax would diverge first.
        // Both arms consume the SAME bf16-rounded inputs and the prefill
        // path already runs this exact bf16 kernel for every MLA family, so
        // the gate here is "bounded, not divergent": measured deltas sit
        // around 10 bf16 ulps (~0.08) under deliberate saturation, far from
        // the O(1) blowup an unstable softmax would produce on unit-scale
        // values.
        let (bf16, fp32) = try runBothArms(
            heads: 16, context: 2048, kDim: 192, vDim: 128,
            scale: pow(192, -0.5), querySpread: 8.0)
        let maxDiff = MLX.abs(bf16 - fp32).max().item(Float.self)
        let meanDiff = MLX.abs(bf16 - fp32).mean().item(Float.self)
        #expect(
            maxDiff < 0.25,
            Comment(rawValue: "saturated 192/128 @ 2048: max |diff| = \(maxDiff)"))
        #expect(
            meanDiff < 0.01,
            Comment(rawValue: "saturated 192/128 @ 2048: mean |diff| = \(meanDiff)"))
    }
}
