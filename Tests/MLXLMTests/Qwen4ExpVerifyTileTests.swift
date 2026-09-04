import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// Workplan W2a: verify-tile M-padding must be inert by default, must only
/// engage for 1 < S < tile at batch 1, and must return exactly the real
/// rows' values — padded rows are a kernel-shape vehicle, never data.
@Suite("Qwen4Exp verify-tile M padding", .serialized)
struct Qwen4ExpVerifyTileTests {
    @Test("gate parses off/absent/invalid to disabled and 2-64 to rows")
    func gateParsing() {
        #expect(Qwen4ExpVerifyTile.parse(nil) == nil)
        #expect(Qwen4ExpVerifyTile.parse("") == nil)
        #expect(Qwen4ExpVerifyTile.parse("off") == nil)
        #expect(Qwen4ExpVerifyTile.parse("OFF") == nil)
        #expect(Qwen4ExpVerifyTile.parse("0") == nil)
        #expect(Qwen4ExpVerifyTile.parse("1") == nil)
        #expect(Qwen4ExpVerifyTile.parse("65") == nil)
        #expect(Qwen4ExpVerifyTile.parse("banana") == nil)
        #expect(Qwen4ExpVerifyTile.parse("8") == 8)
        #expect(Qwen4ExpVerifyTile.parse(" 16 ") == 16)
        #expect(Qwen4ExpVerifyTile.parse("64") == 64)
    }

    @Test("pads only 1 < S < tile at batch 1, slices back to real rows")
    func paddingScope() throws {
        try MLXMetalTestLock.withLock {
            var seenRows: [Int] = []
            func run(batch: Int, s: Int, tile: Int?) -> Bool {
                let x = MLXArray(0 ..< (batch * s * 4), [batch, s, 4])
                    .asType(.float32)
                let y = Qwen4ExpVerifyTile.padded(x, tile: tile) { inner in
                    seenRows.append(inner.dim(1))
                    return inner * 2
                }
                #expect(y.shape == [batch, s, 4])
                return allClose(y, x * 2).item(Bool.self)
            }
            // Verify width: padded to the tile, values exact after the slice.
            #expect(run(batch: 1, s: 4, tile: 16))
            #expect(seenRows.last == 16)
            // AR decode width: untouched.
            #expect(run(batch: 1, s: 1, tile: 16))
            #expect(seenRows.last == 1)
            // At/above the tile (prefill): untouched.
            #expect(run(batch: 1, s: 16, tile: 16))
            #expect(seenRows.last == 16)
            #expect(run(batch: 1, s: 40, tile: 16))
            #expect(seenRows.last == 40)
            // Batch > 1: untouched.
            #expect(run(batch: 2, s: 4, tile: 16))
            #expect(seenRows.last == 4)
            // Gate off: untouched.
            #expect(run(batch: 1, s: 4, tile: nil))
            #expect(seenRows.last == 4)
        }
    }

    @Test("padded quantized matmul rows match the unpadded rows")
    func quantizedRowEquivalence() throws {
        try MLXMetalTestLock.withLock {
            let k = 512
            let n = 1024
            let s = 4
            let x =
                (MLXArray(0 ..< (s * k), [1, s, k]).asType(.float32)
                / Float(s * k) - 0.5).asType(.bfloat16)
            let dense =
                (MLXArray(0 ..< (n * k), [n, k]).asType(.float32)
                % 97 / 96 - 0.5).asType(.float16)
            let (weight, scales, biases) = quantized(
                dense, groupSize: 64, bits: 4, mode: .affine)

            func project(_ input: MLXArray) -> MLXArray {
                quantizedMM(
                    input, weight, scales: scales, biases: biases,
                    transpose: true, groupSize: 64, bits: 4, mode: .affine)
            }

            let direct = project(x)
            let tiled = Qwen4ExpVerifyTile.padded(x, tile: 16) { project($0) }
            #expect(tiled.shape == direct.shape)
            #expect(
                allClose(
                    tiled.asType(.float32), direct.asType(.float32),
                    rtol: 1e-2, atol: 1e-2
                ).item(Bool.self))
        }
    }

    @Test("two-input form pads both arrays and slices the projected result")
    func gatedOutputEquivalence() throws {
        try MLXMetalTestLock.withLock {
            let s = 3
            let x = MLXArray(0 ..< (s * 8), [1, s, 2, 4]).asType(.float32) / 24
            let gate = MLXArray(0 ..< (s * 8), [1, s, 2, 4]).asType(.float32) / 48
            var seenRows: Int? = nil
            let result = Qwen4ExpVerifyTile.padded(x, gate, tile: 8) { out, g in
                seenRows = out.dim(1)
                #expect(g.dim(1) == out.dim(1))
                return (out * sigmoid(g)).reshaped(out.dim(0), out.dim(1), -1)
            }
            #expect(seenRows == 8)
            #expect(result.shape == [1, s, 8])
            let expected = (x * sigmoid(gate)).reshaped(1, s, -1)
            #expect(allClose(result, expected).item(Bool.self))
        }
    }

    @Test("GDN verify-width forward matches padded vs unpadded")
    func gdnVerifyForwardEquivalence() throws {
        let text = try JSONDecoder.json5().decode(
            Qwen35Configuration.self,
            from: """
                {
                  "model_type": "qwen3_5_moe",
                  "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "hidden_size": 32,
                    "num_hidden_layers": 4,
                    "intermediate_size": 64,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "linear_num_value_heads": 4,
                    "linear_num_key_heads": 2,
                    "linear_key_head_dim": 8,
                    "linear_value_head_dim": 8,
                    "linear_conv_kernel_dim": 2,
                    "head_dim": 8,
                    "full_attention_interval": 4,
                    "vocab_size": 100,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {
                      "rope_type": "default",
                      "rope_theta": 100000.0,
                      "partial_rotary_factor": 0.25,
                      "mrope_section": [1, 1, 1]
                    }
                  },
                  "vocab_size": 100
                }
                """.data(using: .utf8)!
        ).textConfiguration
        try MLXMetalTestLock.withLock {
            let gdn = Qwen35Language.GatedDeltaNet(
                text, outputGateSigmoid: true,
                fuseDecodeInputProjections: true,
                verifyTilePadding: true)
            let input =
                (MLXArray(0 ..< (4 * 32), [1, 4, 32]).asType(.float32)
                    / 128 - 0.5)

            let saved = Qwen4ExpVerifyTile.rows
            defer { Qwen4ExpVerifyTile.rows = saved }

            Qwen4ExpVerifyTile.rows = nil
            let baseCache = MambaCache()
            let base = gdn(input, cache: baseCache)

            Qwen4ExpVerifyTile.rows = 16
            let tiledCache = MambaCache()
            let tiled = gdn(input, cache: tiledCache)
            eval(base, tiled)

            #expect(tiled.shape == base.shape)
            #expect(
                allClose(tiled, base, rtol: 1e-4, atol: 1e-5).item(Bool.self))
            // Padded rows must never reach the cache: identical committed
            // shapes and offsets either way.
            #expect(tiledCache.offset == baseCache.offset)
            #expect(tiledCache[0]?.shape == baseCache[0]?.shape)
            #expect(tiledCache[1]?.shape == baseCache[1]?.shape)
        }
    }

    @Test("mismatched gate rows fall through unpadded")
    func mismatchedGateRows() throws {
        try MLXMetalTestLock.withLock {
            let x = MLXArray(0 ..< 12, [1, 3, 4]).asType(.float32)
            let gate = MLXArray(0 ..< 8, [1, 2, 4]).asType(.float32)
            var seenRows: Int? = nil
            _ = Qwen4ExpVerifyTile.padded(x, gate, tile: 8) { out, _ in
                seenRows = out.dim(1)
                return out
            }
            #expect(seenRows == 3)
        }
    }
}
