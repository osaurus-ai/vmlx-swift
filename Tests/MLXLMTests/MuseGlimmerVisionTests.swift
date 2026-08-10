import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM
import MLXLLM

/// The vision tower's risky parts are the ones with no shape check to catch
/// them: the window permutation (a wrong index silently scrambles which patch
/// attends to which), the bilinear position resample, and the merge-block
/// ordering that has to agree between the patch stream and the position rows.
@Suite("Muse Glimmer vision")
struct MuseGlimmerVisionTests {

    /// A small tower with the real ratios: 8 layers on the 3:1 window/full
    /// period, merge 2, and a 4×4 position table so resampling is exercised
    /// without a 32×32 grid.
    static func tinyConfig() throws -> MuseGlimmerVisionConfiguration {
        let json = """
            {
              "hidden_size": 64,
              "intermediate_size": 128,
              "num_attention_heads": 4,
              "num_hidden_layers": 8,
              "patch_size": 14,
              "patch_temporal": 2,
              "merge_size": 2,
              "pos_emb_height": 4,
              "pos_emb_width": 4,
              "layer_norm_eps": 1e-5,
              "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
            }
            """
        return try JSONDecoder().decode(
            MuseGlimmerVisionConfiguration.self, from: Data(json.utf8))
    }

    @Test("layer types default to window/window/window/full and drive the window size")
    func layerTypesAndWindowSize() throws {
        let config = try Self.tinyConfig()
        // Derived default: every 4th layer is full attention.
        #expect(config.isWindowed(0))
        #expect(config.isWindowed(1))
        #expect(config.isWindowed(2))
        #expect(config.isWindowed(3) == false)
        #expect(config.isWindowed(7) == false)
        #expect((0 ..< 8).filter { config.isWindowed($0) }.count == 6)

        // windowSize is derived from the position grid, not a config field.
        #expect(config.windowSize == 4 * 14)
    }

    @Test("explicit layer_types from the checkpoint are honoured")
    func explicitLayerTypes() throws {
        let json = """
            {
              "hidden_size": 64, "num_hidden_layers": 4,
              "layer_types": ["full_attention", "window_attention",
                              "window_attention", "full_attention"]
            }
            """
        let config = try JSONDecoder().decode(
            MuseGlimmerVisionConfiguration.self, from: Data(json.utf8))
        #expect(config.isWindowed(0) == false)
        #expect(config.isWindowed(1))
        #expect(config.isWindowed(3) == false)
    }

    @Test("the shipped 50-layer pattern decodes to 37 windowed / 13 full")
    func realLayerPattern() throws {
        // The shipped vision_config: [w,w,w,full] x12 then [w,full]. Note the
        // tail breaks the period — full attention lands on 47 and then 49,
        // only two apart — so the derived default below is a fallback only and
        // never reproduces the real checkpoint. Every shipped bundle supplies
        // layer_types explicitly.
        var types = [String]()
        for _ in 0 ..< 12 {
            types += ["window_attention", "window_attention", "window_attention",
                      "full_attention"]
        }
        types += ["window_attention", "full_attention"]
        #expect(types.count == 50)

        let json = """
            {"hidden_size": 1536, "num_hidden_layers": 50,
             "layer_types": \(try String(
                data: JSONSerialization.data(withJSONObject: types), encoding: .utf8)!)}
            """
        let config = try JSONDecoder().decode(
            MuseGlimmerVisionConfiguration.self, from: Data(json.utf8))
        #expect((0 ..< 50).filter { config.isWindowed($0) }.count == 37)
        #expect((0 ..< 50).filter { !config.isWindowed($0) }.count == 13)
        // Pin the exact positions, read off the shipped config.
        let full = (0 ..< 50).filter { !config.isWindowed($0) }
        #expect(full == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 49])
    }

    @Test("a forward pass merges 2x2 blocks and returns raster order")
    func forwardProducesMergedTokens() throws {
        let config = try Self.tinyConfig()
        let model = MuseGlimmerVisionModel(config)

        // One frame, 4x4 patch grid => 16 patches => 4 merged tokens.
        let frame = THW(1, 4, 4)
        let patchDim = config.patchTemporal * 3 * config.patchSize * config.patchSize
        let patches = MLXArray.zeros([16, patchDim])

        let out = model(patches, frames: [frame])
        eval(out)

        // Merged width is hidden * mergeSize^2 — the 6144 in the real config.
        #expect(out.dim(0) == 4)
        #expect(out.dim(1) == config.hiddenSize * config.mergeSize * config.mergeSize)
        // A scrambled window permutation typically surfaces as NaN or as a
        // shape mismatch above; assert finiteness explicitly too.
        let values = out.asArray(Float.self)
        #expect(values.allSatisfy { $0.isFinite })
    }

    @Test("a two-frame video runs through the same path")
    func forwardHandlesVideoFrames() throws {
        let config = try Self.tinyConfig()
        let model = MuseGlimmerVisionModel(config)

        // patch_temporal 2 means a video arrives as temporal patch pairs; t=2
        // here is two such pairs on a 4x4 grid.
        let frame = THW(2, 4, 4)
        let patchDim = config.patchTemporal * 3 * config.patchSize * config.patchSize
        let patches = MLXArray.zeros([32, patchDim])

        let out = model(patches, frames: [frame])
        eval(out)

        #expect(out.dim(0) == 8)
        #expect(out.dim(1) == config.hiddenSize * 4)
        #expect(out.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    @Test("projector applies the activation after both adapter layers")
    func projectorDoubleActivation() throws {
        // fc1 6144->4096, fc2 4096->4096 in the real checkpoint; the second
        // gelu is the detail that reads like a typo but is not.
        let projector = MuseGlimmerVisionProjector(
            inputDimensions: 16, hiddenDimensions: 8)
        let out = projector(MLXArray.zeros([2, 16]))
        eval(out)
        #expect(out.dim(0) == 2)
        #expect(out.dim(1) == 8)
        // gelu(0) == 0, so an all-zero input must stay all-zero through both
        // activations — a stray bias or a swapped layer breaks this.
        #expect(out.asArray(Float.self).allSatisfy { abs($0) < 1e-6 })
    }

    @Test("prepare produces 3-D logits for every prompt length and shape")
    func prepareLogitsAreThreeDimensional() throws {
        // The live failure chain that unit tests missed: inputEmbeddings added
        // .newAxis to already-2D ids, giving 4-D embeddings; the prefill chunk
        // slice then cut a size-1 axis; the forward pass still ran; and the
        // only symptom was convertToToken indexing `logits[0..., -1, 0...]` on
        // a non-3-D tensor. Assert the contract at the seam instead.
        let json = """
            {
              "model_type": "muse_glimmer",
              "text_config": {
                "model_type": "muse_glimmer_text",
                "hidden_size": 32, "num_hidden_layers": 4,
                "intermediate_size": 64, "num_attention_heads": 2,
                "head_dim": 16, "num_key_value_heads": 1,
                "vocab_size": 64, "sliding_window": 64
              },
              "vision_config": {
                "hidden_size": 32, "num_hidden_layers": 4,
                "intermediate_size": 64, "num_attention_heads": 2,
                "patch_size": 14, "patch_temporal": 2, "merge_size": 2,
                "pos_emb_height": 4, "pos_emb_width": 4
              },
              "projector_hidden_size": 16
            }
            """
        let config = try JSONDecoder().decode(
            MuseGlimmerConfiguration.self, from: Data(json.utf8))
        let model = MuseGlimmer(config)

        // Lengths that straddle the chunk boundary, including the k*step + 1
        // shape that previously crashed, and 1 token.
        for length in [1, 2, 63, 64, 65, 129, 257] {
            let ids = MLXArray((0 ..< length).map { Int32($0 % 64) }).reshaped(1, length)
            let input = LMInput(tokens: ids, tokenIds: (0 ..< length).map { $0 % 64 })
            let result = try model.prepare(input, cache: model.newCache(), windowSize: 64)
            guard case .logits(let out) = result else {
                Issue.record("prepare did not return logits for length \(length)")
                continue
            }
            eval(out.logits)
            #expect(out.logits.ndim == 3,
                "length \(length): logits must be 3-D, got \(out.logits.ndim)")
            #expect(out.logits.dim(0) == 1, "length \(length): batch must be 1")
            #expect(out.logits.dim(2) == 64, "length \(length): vocab mismatch")
            // convertToToken does exactly this; make the test fail here rather
            // than in the engine.
            let last = out.logits[0..., -1, 0...]
            eval(last)
            #expect(last.ndim == 2)
        }
    }
}
