// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXVLM

/// `Glm5NextSparseAttention.kbFactors(dtype:)` hydrates the absorbed-MLA factors of `kv_b_proj` in
/// the dtype the caller asks for; the decode path asks for the query's.
///
/// `dequantized` without a dtype returns its scales' dtype. The GLM-5.3 bundle this was found on
/// stores `kv_b_proj`'s scales and biases as float16 while the activations are bfloat16, so factors
/// hydrated at the default were float16, and the use site's cast to the query dtype converted them
/// on every decode step.
@Suite("GLM-5.3 absorbed-MLA factor dtype")
struct Glm5NextKbFactorDTypeTests {

    /// GLM-5.3's MLA widths (qk_nope 256, v_head 256, kv_lora_rank 512) with 4 heads instead of 64.
    static func layer() throws -> Glm5NextSparseAttention {
        let json = Glm5NextConstructionTests.configJSON
            .replacingOccurrences(of: #""num_attention_heads":64"#, with: #""num_attention_heads":4"#)
            .replacingOccurrences(of: #""num_key_value_heads":64"#, with: #""num_key_value_heads":4"#)
        let config = try Glm5NextConstructionTests.config(json)
        try #require(config.textConfig.numAttentionHeads == 4)
        return Glm5NextSparseAttention(config.textConfig)
    }

    /// The same layer with `kv_b_proj` quantized as the bundle ships it: 8 bits, group size 64,
    /// float16 scales and biases.
    static func quantizedLayer() throws -> (Glm5NextSparseAttention, QuantizedLinear) {
        let module = try layer()
        module.update(
            parameters: ModuleParameters.unflattened([
                "kv_b_proj.weight": module.kvBProj.weight.asType(.float16)
            ]))
        quantize(
            model: module, groupSize: 64, bits: 8, mode: .affine,
            filter: { path, _ in path == "kv_b_proj" })
        let q = try #require(module.kvBProj as? QuantizedLinear)
        try #require(q.scales.dtype == .float16)
        try #require(q.biases?.dtype == .float16)
        return (module, q)
    }

    @Test("float16 scales, bfloat16 request: the factors are bfloat16 and hold the weight's values")
    func quantizedFactorsTakeTheRequestedDType() throws {
        let (module, q) = try Self.quantizedLayer()
        let (k, v) = module.kbFactors(dtype: .bfloat16)
        #expect(k.dtype == .bfloat16)
        #expect(v.dtype == .bfloat16)
        #expect(k.shape == [4, 256, 512])
        #expect(v.shape == [4, 256, 512])

        // The values are the dequantized weight's: hydrate at the default (float16) and compare.
        let w = dequantized(
            q.weight, scales: q.scales, biases: q.biases,
            groupSize: q.groupSize, bits: q.bits, mode: q.mode
        ).reshaped(4, 512, -1)
        #expect(w.dtype == .float16, "the default is the scales' dtype, which is what made this a bug")
        let wk = w[0..., ..<256, 0...].asType(.float32)
        let wv = w[0..., 256..., 0...].asType(.float32)
        #expect(allClose(k.asType(.float32), wk, rtol: 1e-2, atol: 1e-2).item(Bool.self))
        #expect(allClose(v.asType(.float32), wv, rtol: 1e-2, atol: 1e-2).item(Bool.self))
    }

    @Test("the cache is kept for one dtype and rebuilt for another")
    func cacheIsKeyedByDType() throws {
        let (module, _) = try Self.quantizedLayer()
        let (k1, v1) = module.kbFactors(dtype: .bfloat16)
        let (k2, v2) = module.kbFactors(dtype: .bfloat16)
        #expect(k1 === k2)
        #expect(v1 === v2)

        let (k3, v3) = module.kbFactors(dtype: .float32)
        #expect(k3.dtype == .float32)
        #expect(v3.dtype == .float32)
        #expect(k3 !== k1)
    }

    @Test("an unquantized kv_b_proj is cast to the requested dtype")
    func unquantizedFactorsTakeTheRequestedDType() throws {
        let module = try Self.layer()
        try #require(module.kvBProj.weight.dtype == .float32)
        let (k, v) = module.kbFactors(dtype: .bfloat16)
        #expect(k.dtype == .bfloat16)
        #expect(v.dtype == .bfloat16)
    }
}
