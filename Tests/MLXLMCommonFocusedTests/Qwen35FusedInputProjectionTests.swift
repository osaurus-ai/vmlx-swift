// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Bit-identity gate for the Qwen3.5 GDN fused decode input projections.
//
// The fusion concatenates affine-quantized projection weights row-wise and
// replaces N quantized matmuls with one per scheme-compatible group. Row
// independence of (de)quantization and matmul makes the fused output
// bit-identical to the separate calls — this suite pins that for BOTH
// grouping shapes: a uniform stamp (all four projections fuse, 4→1) and the
// 27B's real mixed stamp (qkv+z at one width, b+a at another, 4→2). The
// baseline instance runs with `VMLX_GDN_FUSE_DECODE_INPUTS=0`, the fused
// instance with identical weights (same seed) and fusion on; outputs must
// match exactly.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct Qwen35FusedInputProjectionTests {

    private static func makeConfig() throws -> Qwen35TextConfiguration {
        let json = """
            {
              "hidden_size": 64,
              "linear_num_value_heads": 4,
              "linear_num_key_heads": 2,
              "linear_key_head_dim": 8,
              "linear_value_head_dim": 8,
              "linear_conv_kernel_dim": 4,
              "rms_norm_eps": 1e-6
            }
            """
        return try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
    }

    /// Deterministic layer with the four input projections quantized:
    /// qkv/z at `wideBits`, b/a at `narrowBits` (group size 32 throughout).
    private static func makeLayer(
        wideBits: Int, narrowBits: Int
    ) throws -> Qwen35GatedDeltaNet {
        MLXRandom.seed(42)
        let layer = Qwen35GatedDeltaNet(try makeConfig())
        quantize(
            model: layer,
            filter: { path, module in
                guard module is Linear, !(module is QuantizedLinear) else { return nil }
                if path.contains("in_proj_qkv") || path.contains("in_proj_z") {
                    return (groupSize: 32, bits: wideBits, mode: .affine)
                }
                if path.contains("in_proj_b") || path.contains("in_proj_a") {
                    return (groupSize: 32, bits: narrowBits, mode: .affine)
                }
                return nil
            })
        MLX.eval(layer.parameters())
        return layer
    }

    private static func run(
        _ layer: Qwen35GatedDeltaNet, input: MLXArray
    ) -> MLXArray {
        let out = layer(input, cache: MambaCache())
        MLX.eval(out)
        return out
    }

    private func assertBitIdentical(wideBits: Int, narrowBits: Int) throws {
        try FocusedMLXTestSupport.withLock {
            MLXRandom.seed(7)
            let input = MLXRandom.normal([1, 1, 64]).asType(.bfloat16)
            MLX.eval(input)

            setenv("VMLX_GDN_FUSE_DECODE_INPUTS", "0", 1)
            defer { unsetenv("VMLX_GDN_FUSE_DECODE_INPUTS") }
            let baselineLayer = try Self.makeLayer(
                wideBits: wideBits, narrowBits: narrowBits)
            let baseline = Self.run(baselineLayer, input: input)

            setenv("VMLX_GDN_FUSE_DECODE_INPUTS", "1", 1)
            let fusedLayer = try Self.makeLayer(
                wideBits: wideBits, narrowBits: narrowBits)
            let fused = Self.run(fusedLayer, input: input)

            let identical = MLX.all(baseline .== fused).item(Bool.self)
            #expect(
                identical,
                Comment(
                    rawValue:
                        "fused GDN input projections must be bit-identical "
                        + "(wide=\(wideBits) narrow=\(narrowBits))"))
        }
    }

    @Test("uniform quantization fuses all four projections bit-identically")
    func uniformSchemeFusesAllFour() throws {
        try assertBitIdentical(wideBits: 4, narrowBits: 4)
    }

    @Test("mixed quantization fuses compatible pairs bit-identically")
    func mixedSchemeFusesPairs() throws {
        // The 27B's real shape: wide projections at one bit width, the tiny
        // b/a pair at another — the all-or-nothing guard would refuse this.
        try assertBitIdentical(wideBits: 8, narrowBits: 4)
    }
}
