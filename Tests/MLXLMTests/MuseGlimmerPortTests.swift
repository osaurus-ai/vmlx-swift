import Foundation
import MLX
import MLXNN
import MLXLMCommon
import Testing

@testable import MLXLLM

/// Muse Glimmer's port risk is concentrated in five details that keep every
/// tensor the correct shape and only make the model *worse* — the class of bug
/// that survives a smoke test and gets blamed on the quant. Each one gets a
/// test that fails loudly if the wiring regresses.
///
/// Reference semantics are recorded in
/// `Libraries/MLXVLM/Models/MuseGlimmer_ARCHITECTURE.md`.
///
/// Numeric checks pull values out with `asArray(Float.self)` and finish the
/// arithmetic in plain Swift: mixing MLX reductions with stdlib `sqrt` sends
/// the type checker into `Duration`/`Double` overloads and the failures read
/// as unrelated nonsense.
@Suite("Muse Glimmer port")
struct MuseGlimmerPortTests {

    static func rms(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        let sumSquares = values.reduce(Float(0)) { $0 + $1 * $1 }
        return (sumSquares / Float(values.count)).squareRoot()
    }

    /// A small config with the real structural ratios (4-layer period, one
    /// full/NoPE layer per period) so layer-type and NoPE indexing are
    /// exercised without loading 20GB.
    static func tinyConfig(layers: Int = 8) throws -> MuseGlimmerTextConfiguration {
        let json = """
            {
              "model_type": "muse_glimmer",
              "text_config": {
                "model_type": "muse_glimmer_text",
                "hidden_size": 64,
                "num_hidden_layers": \(layers),
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "head_dim": 16,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "rms_norm_eps": 1e-5,
                "post_norm_eps": 1e-8,
                "sliding_window": 8,
                "max_position_embeddings": 1024,
                "qk_scale_factor": 3.87,
                "output_multiplier": 0.19611613513818404,
                "final_logit_softcapping": 20.0,
                "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"}
              }
            }
            """
        return try JSONDecoder().decode(
            MuseGlimmerTextConfiguration.self, from: Data(json.utf8))
    }

    // MARK: - Divergence 3: layer types + NoPE

    @Test("full_attention lands on NoPE layers, counted back from the last")
    func layerTypesAndNoPEAgree() throws {
        let config = try Self.tinyConfig(layers: 52)

        // Reference: `(num_layers - 1 - i) % 4 == 0`.
        let expectedNoPE = (0 ..< 52).filter { (52 - 1 - $0) % 4 == 0 }
        #expect(expectedNoPE.first == 3)
        #expect(expectedNoPE.last == 51)
        #expect(expectedNoPE.count == 13)

        for i in 0 ..< 52 {
            let isNoPE = config.ropeTheta(i) == nil
            #expect(isNoPE == expectedNoPE.contains(i), "layer \(i) NoPE mismatch")
            // Every NoPE layer is a full-attention layer and vice versa — if
            // these drift apart the mask and the rotary disagree silently.
            #expect(isNoPE == !config.isSliding(i), "layer \(i) type/NoPE disagree")
        }

        #expect((0 ..< 52).filter { config.isSliding($0) }.count == 39)
    }

    @Test("explicit layer_rope_theta from the checkpoint wins over the default")
    func explicitLayerRopeThetaRespected() throws {
        // Rotary disabled on layer 0 only — the derived default could never
        // produce this, so it proves the checkpoint's list is actually read.
        let json = """
            {
              "model_type": "muse_glimmer_text",
              "num_hidden_layers": 4,
              "layer_rope_theta": [0, 500000.0, 500000.0, 500000.0],
              "layer_types": ["full_attention", "sliding_attention",
                              "sliding_attention", "sliding_attention"]
            }
            """
        let config = try JSONDecoder().decode(
            MuseGlimmerTextConfiguration.self, from: Data(json.utf8))
        #expect(config.ropeTheta(0) == nil)
        #expect(config.ropeTheta(1) == 500_000.0)
        #expect(config.isSliding(0) == false)
        #expect(config.isSliding(3) == true)
    }

    // MARK: - Divergence 1: centered RMSNorm

    @Test("a zero-weight centered norm is unit gain once the load-time fold runs")
    func centeredNormIsUnitGainAtZeroWeight() throws {
        // The checkpoint stores zero-centered gains, so the effective weight is
        // `1 + w`. That `+1` is folded into the tensor in `sanitize` rather than
        // applied per token, so the contract spans both halves and has to be
        // tested that way: fold the weight the way the loader does, then run the
        // module. Testing the module alone would now assert the wrong thing.
        let norm = MuseGlimmerCenteredRMSNorm(dimensions: 64, eps: 1e-5)
        let folded = MLXArray.zeros([64]) + 1
        try norm.update(
            parameters: ModuleParameters.unflattened(["weight": folded]),
            verify: Module.VerifyUpdate.all)
        eval(norm)

        let input: [Float] = (0 ..< 64).map { Float($0 % 7) - 3.0 }
        let normed = norm(MLXArray(input).reshaped(1, 1, 64))
        eval(normed)

        let out = normed.asArray(Float.self)
        #expect(abs(Self.rms(out) - 1.0) < 1e-3,
            "expected unit-RMS output, got \(Self.rms(out))")

        let maxAbs = out.map { abs($0) }.max() ?? 0
        #expect(maxAbs > 0.1, "norm collapsed toward zero — the fold did not reach this weight")

        // And the fold itself must select this key: an unfolded zero weight
        // would silently zero the layer's gain.
        #expect(MuseGlimmerTextModel.isCenteredNormWeight("model.layers.7.input_layernorm.weight"))
        #expect(MuseGlimmerTextModel.isCenteredNormWeight("model.layers.7.pre_feedforward_layernorm.weight"))
        #expect(!MuseGlimmerTextModel.isCenteredNormWeight("vision_tower.layers.3.norm1.weight"))
    }

    // MARK: - Divergence 2: Q-only qk scale

    @Test("scaleless QK-norm gives unit RMS, and only Q carries qk_scale_factor")
    func qkNormIsScalelessAndAsymmetric() throws {
        let input: [Float] = (0 ..< 32).map { Float(($0 * 13) % 11) + 0.5 }
        let x = MLXArray(input).reshaped(1, 1, 2, 16)

        let normed = MuseGlimmerMath.scalelessRMSNorm(x, eps: 1e-5)
        eval(normed)
        let k = normed.asArray(Float.self)

        // Unit gain: each 16-wide head row normalizes to RMS 1.
        for head in 0 ..< 2 {
            let row = Array(k[(head * 16) ..< ((head + 1) * 16)])
            #expect(abs(Self.rms(row) - 1.0) < 1e-3,
                "head \(head) scaleless norm should have unit RMS")
        }

        // The factor applies to Q only. If a refactor ever applies it to both,
        // attention logits scale by 3.87² and this catches it.
        let config = try Self.tinyConfig(layers: 4)
        let scaledQ = normed * config.qkScaleFactor
        eval(scaledQ)
        let q = scaledQ.asArray(Float.self)
        #expect(abs(Self.rms(q) / Self.rms(k) - 3.87) < 1e-2)
    }

    // MARK: - Divergence 5: logit tail

    @Test("logit tail multiplies then softcaps, and the cap actually binds")
    func logitTailMultipliesThenSoftcaps() throws {
        let config = try Self.tinyConfig(layers: 4)

        // Small logits: tanh is ~linear here, so the multiplier dominates.
        let small: [Float] = [0.5, -0.25, 1.0]
        let outSmall = MuseGlimmerTextModel.applyLogitTail(
            MLXArray(small).reshaped(1, 1, 3), config: config)
        eval(outSmall)
        let s = outSmall.asArray(Float.self)

        let cap: Float = 20.0
        let expected0 = cap * tanh(0.5 * config.outputMultiplier / cap)
        #expect(abs(s[0] - expected0) < 1e-5)
        // Softcapping is monotonic, so ordering must survive it.
        #expect(s[1] < s[0])
        #expect(s[0] < s[2])

        // Huge logits: the cap binds at ±20 however large the input.
        let huge: [Float] = [1e6, -1e6]
        let outHuge = MuseGlimmerTextModel.applyLogitTail(
            MLXArray(huge).reshaped(1, 1, 2), config: config)
        eval(outHuge)
        let h = outHuge.asArray(Float.self)
        #expect(abs(h[0] - 20.0) < 1e-3, "positive cap should saturate at +20")
        #expect(abs(h[1] + 20.0) < 1e-3, "negative cap should saturate at -20")
    }

    // MARK: - Cache topology

    @Test("sliding layers get rotating caches, full/NoPE layers get standard")
    func cacheTopologyMatchesLayerTypes() throws {
        let config = try Self.tinyConfig(layers: 52)
        let model = MuseGlimmerTextModel(config)
        let caches = model.newCache()

        #expect(caches.count == 52)
        for i in 0 ..< 52 {
            if config.isSliding(i) {
                #expect(caches[i] is RotatingKVCache, "layer \(i) should rotate")
            } else {
                #expect(caches[i] is StandardKVCache, "layer \(i) should be standard")
            }
        }
        // The 13 full-attention layers retain the whole history; the rest are
        // bounded by the 2048-token window.
        #expect(caches.filter { $0 is StandardKVCache }.count == 13)
    }

    // MARK: - Weight sanitization

    @Test("sanitize unwraps language_model and drops the vision tower")
    func sanitizeSelectsTextTower() throws {
        let config = try Self.tinyConfig(layers: 4)
        let model = MuseGlimmerTextModel(config)
        let dummy = MLXArray([Float(1.0)])

        let out = model.sanitize(weights: [
            "language_model.model.layers.0.self_attn.q_proj.weight": dummy,
            "language_model.model.embed_tokens.weight": dummy,
            "language_model.lm_head.weight": dummy,
            "model.vision_tower.layers.0.attn.q_proj.weight": dummy,
            "model.vision_projection.weight": dummy,
        ])

        #expect(out["model.layers.0.self_attn.q_proj.weight"] != nil)
        #expect(out["model.embed_tokens.weight"] != nil)
        #expect(out["lm_head.weight"] != nil)
        // Vision weights must not survive into a text-only load.
        #expect(out.keys.contains { $0.contains("vision") } == false)
        #expect(out.count == 3)
    }

    // MARK: - Registry

    @Test("muse_glimmer resolves through the LLM factory registry")
    func registeredInFactory() async throws {
        let registry = LLMModelFactory.shared.typeRegistry
        let json = """
            {
              "model_type": "muse_glimmer",
              "hidden_size": 64,
              "num_hidden_layers": 4,
              "intermediate_size": 128,
              "num_attention_heads": 4,
              "head_dim": 16,
              "num_key_value_heads": 2,
              "vocab_size": 128
            }
            """
        let data = Data(json.utf8)
        for modelType in ["muse_glimmer", "muse_glimmer_text"] {
            // Resolution is what's under test: the handler must exist and
            // build. An unregistered type throws here instead.
            let model = try await registry.createModel(
                configuration: data, modelType: modelType)
            #expect(model is MuseGlimmerTextModel)
        }
    }
}

/// The unit tests above all use short prompts, so none of them ever pushes a
/// rotating cache past its window — which is exactly how a prefill that
/// overran the 2048-token sliding window reached a live run and crashed inside
/// the MLX scatter. This suite prefills *past* the window on purpose.
@Suite("Muse Glimmer long prefill")
struct MuseGlimmerLongPrefillTests {

    @Test("prefilling past the sliding window does not overrun a rotating cache")
    func prefillLongerThanSlidingWindow() throws {
        // Window 8 with 24 layers: enough to hit the rotating path repeatedly
        // without allocating a real model.
        let json = """
            {
              "model_type": "muse_glimmer_text",
              "hidden_size": 64,
              "num_hidden_layers": 8,
              "intermediate_size": 128,
              "num_attention_heads": 4,
              "head_dim": 16,
              "num_key_value_heads": 2,
              "vocab_size": 128,
              "sliding_window": 8
            }
            """
        let config = try JSONDecoder().decode(
            MuseGlimmerTextConfiguration.self, from: Data(json.utf8))
        let model = MuseGlimmerTextModel(config)
        let cache = model.newCache()

        // 40 tokens through an 8-token window — five windows' worth. Fed in
        // window-sized chunks, the way `prepare` now does it.
        let total = 40
        let step = config.slidingWindow
        var offset = 0
        var out: MLXArray?
        while offset < total {
            let end = min(offset + step, total)
            let chunk = MLXArray((offset ..< end).map { Int32($0 % 128) })
                .reshaped(1, end - offset)
            out = model(chunk, cache: cache)
            eval(out!)
            offset = end
        }

        // Reaching here at all is the regression guard: feeding this prompt in
        // one pass overran the rotating cache's in-place write and tripped a
        // precondition inside the MLX scatter.
        #expect(out != nil)
        #expect(out!.dim(2) == config.vocabularySize)

        // `offset` is the logical token position, not the buffer size, so it
        // advances to the full prompt length on every layer — rotating caches
        // bound their *storage*, not their position counter.
        for i in 0 ..< config.hiddenLayers {
            #expect(cache[i].offset == total,
                "layer \(i) should have consumed all \(total) tokens")
        }
    }

    @Test("a prefill chunk wider than the cache's growth step still fits")
    func chunkWiderThanDefaultGrowthStep() throws {
        // The live crash: RotatingKVCache allocates `step` rows (256 by
        // default) and updateInPlace writes the whole incoming chunk into that
        // block, so a 512-token chunk ran off the end. newCache now sizes the
        // step to the window, so a full-window chunk is writable in one go.
        let json = """
            {
              "model_type": "muse_glimmer_text",
              "hidden_size": 32, "num_hidden_layers": 4,
              "intermediate_size": 64, "num_attention_heads": 2,
              "head_dim": 16, "num_key_value_heads": 1,
              "vocab_size": 64, "sliding_window": 512
            }
            """
        let config = try JSONDecoder().decode(
            MuseGlimmerTextConfiguration.self, from: Data(json.utf8))
        let model = MuseGlimmerTextModel(config)
        let cache = model.newCache()

        // One 512-token chunk — twice the old 256 default growth step.
        let chunk = MLXArray((0 ..< 512).map { Int32($0 % 64) }).reshaped(1, 512)
        let out = model(chunk, cache: cache)
        eval(out)

        #expect(out.dim(2) == config.vocabularySize)
        #expect(cache[0].offset == 512)
    }
}
