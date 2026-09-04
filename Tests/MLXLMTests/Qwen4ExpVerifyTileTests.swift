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

    // MARK: - qwen3_5 (Qwen3.8-27B JANG family)

    /// Tiny DENSE qwen3_5 text trunk: 4 layers, layer 4 full attention, the
    /// rest GatedDeltaNet; untied lm_head. Mirrors the 27B layout in kind
    /// (GDN + gated attention + Linear head), not in size.
    private func tinyDenseQwen35Configuration() throws -> Qwen35Configuration {
        try JSONDecoder.json5().decode(
            Qwen35Configuration.self,
            from: """
                {
                  "model_type": "qwen3_5",
                  "text_config": {
                    "model_type": "qwen3_5_text",
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
                    "tie_word_embeddings": false,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {
                      "rope_type": "default",
                      "rope_theta": 100000.0,
                      "partial_rotary_factor": 0.25,
                      "mrope_section": [1, 1, 1]
                    }
                  },
                  "vision_config": {
                    "model_type": "qwen3_vl",
                    "depth": 1,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "out_hidden_size": 32,
                    "num_heads": 4,
                    "patch_size": 2,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 1,
                    "num_position_embeddings": 32
                  },
                  "vocab_size": 100,
                  "image_token_id": 98,
                  "video_token_id": 97
                }
                """.data(using: .utf8)!)
    }

    @Test("qwen3_5 GDN (family-tagged) verify-width forward matches padded vs unpadded")
    func qwen35GDNVerifyForwardEquivalence() throws {
        let text = try tinyDenseQwen35Configuration().textConfiguration
        try MLXMetalTestLock.withLock {
            // Exactly what Qwen35Language.DecoderLayer constructs for a
            // linear layer.
            let gdn = Qwen35Language.GatedDeltaNet(
                text,
                fuseDecodeInputProjections: true,
                verifyTilePadding: true,
                verifyTileFamily: Qwen4ExpVerifyTile.Family.qwen35)
            #expect(gdn.verifyTileFamily == "Qwen35")
            let input =
                (MLXArray(0 ..< (5 * 32), [1, 5, 32]).asType(.float32)
                    / 160 - 0.5)

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
            #expect(tiledCache.offset == baseCache.offset)
            #expect(tiledCache[0]?.shape == baseCache[0]?.shape)
            #expect(tiledCache[1]?.shape == baseCache[1]?.shape)
        }
    }

    @Test("qwen3_5 attention: prefill then verify width, padded vs unpadded")
    func qwen35AttentionVerifyForwardEquivalence() throws {
        let text = try tinyDenseQwen35Configuration().textConfiguration
        try MLXMetalTestLock.withLock {
            let attention = Qwen35Language.Attention(text)
            let prefill =
                (MLXArray(0 ..< (20 * 32), [1, 20, 32]).asType(.float32)
                    / 640 - 0.5)
            // 5 rows: 1 < S < 16, and not a divisor of the tile.
            let verify =
                (MLXArray(0 ..< (5 * 32), [1, 5, 32]).asType(.float32)
                    / 160 - 0.25)

            let saved = Qwen4ExpVerifyTile.rows
            defer { Qwen4ExpVerifyTile.rows = saved }

            // The mask the DecoderLayer hands the attention: built from the
            // cache offset BEFORE the K/V update, exactly like the model.
            func mask(_ h: MLXArray, _ cache: KVCache) -> MLXArray? {
                if case .array(let m) = createAttentionMask(
                    h: h, cache: cache, returnArray: true)
                {
                    return m
                }
                return nil
            }

            func run(rows: Int?) -> (MLXArray, KVCacheSimple) {
                let cache = KVCacheSimple()
                Qwen4ExpVerifyTile.rows = nil
                _ = attention(
                    prefill, mask: mask(prefill, cache), cache: cache, positionIds: nil)
                Qwen4ExpVerifyTile.rows = rows
                let out = attention(
                    verify, mask: mask(verify, cache), cache: cache, positionIds: nil)
                eval(out)
                return (out, cache)
            }

            let (base, baseCache) = run(rows: nil)
            let (tiled, tiledCache) = run(rows: 16)

            #expect(tiled.shape == [1, 5, 32])
            #expect(tiled.shape == base.shape)
            #expect(
                allClose(tiled, base, rtol: 1e-4, atol: 1e-5).item(Bool.self))
            // The KV cache only ever sees the real rows: 20 + 5, never 20 + 16.
            #expect(baseCache.offset == 25)
            #expect(tiledCache.offset == baseCache.offset)
            #expect(tiledCache.state.map(\.shape) == baseCache.state.map(\.shape))
            #expect(
                allClose(
                    tiledCache.state[0][.ellipsis, ..<25, 0...],
                    baseCache.state[0][.ellipsis, ..<25, 0...],
                    rtol: 1e-4, atol: 1e-5
                ).item(Bool.self))
        }
    }

    @Test("qwen3_5 model: verify-width logits and caches match padded vs unpadded")
    func qwen35ModelVerifyLogitsEquivalence() throws {
        let config = try tinyDenseQwen35Configuration()
        try MLXMetalTestLock.withLock {
            let model = Qwen35(config)
            // Untied head: the lm_head Linear is what the 27B stamps carry.
            #expect(
                model.parameters().flattened().contains { key, _ in
                    key.hasSuffix("lm_head.weight")
                })
            let prefill = MLXArray(Array(1 ..< 21))[.newAxis, .ellipsis]
            let verify = MLXArray([7, 3, 9, 4, 11])[.newAxis, .ellipsis]

            let saved = Qwen4ExpVerifyTile.rows
            defer { Qwen4ExpVerifyTile.rows = saved }

            func run(rows: Int?) -> (MLXArray, [any KVCache]) {
                let cache = model.newCache(parameters: nil)
                Qwen4ExpVerifyTile.rows = nil
                let prefillLogits = model(prefill, cache: cache)
                eval(prefillLogits)
                Qwen4ExpVerifyTile.rows = rows
                let logits = model(verify, cache: cache)
                eval(logits)
                return (logits, cache)
            }

            let (base, baseCache) = run(rows: nil)
            let (tiled, tiledCache) = run(rows: 16)

            #expect(tiled.shape == [1, 5, 100])
            #expect(tiled.shape == base.shape)
            #expect(
                allClose(tiled, base, rtol: 1e-4, atol: 1e-5).item(Bool.self))
            #expect(baseCache.count == 4)
            for (layer, (b, t)) in zip(baseCache, tiledCache).enumerated() {
                #expect(t.offset == b.offset, "layer \(layer) offset")
                #expect(b.offset == 25, "layer \(layer) committed rows")
                #expect(
                    t.state.map(\.shape) == b.state.map(\.shape),
                    "layer \(layer) state shapes")
            }
        }
    }

    @Test("qwen3_5 model: S == 1 decode and prefill never pad")
    func qwen35ScopeNeverPadsDecodeOrPrefill() throws {
        let config = try tinyDenseQwen35Configuration()
        try MLXMetalTestLock.withLock {
            let model = Qwen35(config)
            let saved = Qwen4ExpVerifyTile.rows
            defer { Qwen4ExpVerifyTile.rows = saved }
            Qwen4ExpVerifyTile.rows = 16

            let cache = model.newCache(parameters: nil)
            // Prefill at S >= tile: untouched by construction; the verify
            // gate is on the whole time, so the only thing keeping the pad
            // out is the scope rule.
            let prefillLogits = model(MLXArray(Array(1 ..< 17))[.newAxis, .ellipsis], cache: cache)
            eval(prefillLogits)
            #expect(prefillLogits.shape == [1, 16, 100])
            #expect(cache.allSatisfy { $0.offset == 16 })
            // AR decode: S == 1, untouched.
            let decodeLogits = model(MLXArray([5])[.newAxis, .ellipsis], cache: cache)
            eval(decodeLogits)
            #expect(decodeLogits.shape == [1, 1, 100])
            #expect(cache.allSatisfy { $0.offset == 17 })
        }
    }
}
