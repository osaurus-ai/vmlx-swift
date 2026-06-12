// Copyright © 2026 Jinho Jang (eric@jangq.ai)
//
// DiffusionGemma block-diffusion engine tests.
//
// Covers the ordered KV read APIs the diffusion decoder relies on, the
// block-diffusion sampling primitives, the DiffusionGemma model family,
// and the BlockDiffusionTokenIterator generation flow.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

// MARK: - Ordered KV read APIs (Task 1)

@Test(.serialized)
func testRotatingKVCacheTemporallyOrderedKV() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let cache = RotatingKVCache(maxSize: 4, keep: 0)

    // 6 single-token updates whose key payload encodes the position.
    for position in 0 ..< 6 {
        let keys = MLXArray.full([1, 1, 1, 2], values: MLXArray(Float(position)))
        let values = MLXArray.full([1, 1, 1, 2], values: MLXArray(Float(position) + 100))
        _ = cache.update(keys: keys, values: values)
    }
    #expect(cache.offset == 6)

    let ordered = try #require(cache.temporallyOrderedKV())
    #expect(ordered.keys.dim(2) == 4)
    let keyPositions = ordered.keys[0, 0, 0..., 0].asArray(Float.self)
    #expect(keyPositions == [2, 3, 4, 5])
    let valuePositions = ordered.values[0, 0, 0..., 0].asArray(Float.self)
    #expect(valuePositions == [102, 103, 104, 105])

    // Read must not mutate rotation state.
    #expect(cache.offset == 6)
    let again = try #require(cache.temporallyOrderedKV())
    #expect(again.keys[0, 0, 0..., 0].asArray(Float.self) == [2, 3, 4, 5])
}

@Test(.serialized)
func testRotatingKVCacheTemporallyOrderedKVBeforeWrap() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let cache = RotatingKVCache(maxSize: 8, keep: 0)
    for position in 0 ..< 3 {
        let keys = MLXArray.full([1, 1, 1, 2], values: MLXArray(Float(position)))
        let values = MLXArray.full([1, 1, 1, 2], values: MLXArray(Float(position)))
        _ = cache.update(keys: keys, values: values)
    }

    let ordered = try #require(cache.temporallyOrderedKV())
    #expect(ordered.keys.dim(2) == 3)
    #expect(ordered.keys[0, 0, 0..., 0].asArray(Float.self) == [0, 1, 2])
    #expect(cache.offset == 3)
}

@Test(.serialized)
func testKVCacheSimpleReadKV() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let cache = KVCacheSimple()
    for position in 0 ..< 3 {
        let keys = MLXArray.full([1, 1, 1, 2], values: MLXArray(Float(position)))
        let values = MLXArray.full([1, 1, 1, 2], values: MLXArray(Float(position)))
        _ = cache.update(keys: keys, values: values)
    }

    let read = try #require(cache.readKV())
    #expect(read.keys.dim(2) == 3)
    #expect(read.keys[0, 0, 0..., 0].asArray(Float.self) == [0, 1, 2])
    #expect(cache.offset == 3)

    let empty = KVCacheSimple()
    #expect(empty.readKV() == nil)
    #expect(RotatingKVCache(maxSize: 4).temporallyOrderedKV() == nil)
}

// MARK: - Block-diffusion primitives (Task 2)

@Test
func testBlockDiffusionTemperatureSchedule() {
    // First denoising step (curStep == maxSteps) runs at tMax.
    #expect(
        blockDiffusionTemperature(curStep: 48, maxSteps: 48, tMin: 0.4, tMax: 0.8) == 0.8)
    // Annealing toward tMin as curStep approaches 0.
    let nearEnd = blockDiffusionTemperature(curStep: 1, maxSteps: 48, tMin: 0.4, tMax: 0.8)
    #expect(abs(nearEnd - (0.4 + 0.4 / 48)) < 1e-6)
    // Midpoint.
    let mid = blockDiffusionTemperature(curStep: 24, maxSteps: 48, tMin: 0.4, tMax: 0.8)
    #expect(abs(mid - 0.6) < 1e-6)
}

@Test(.serialized)
func testCanvasTokenEntropy() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    // Near-one-hot logits → entropy ≈ 0; uniform logits → entropy ≈ ln(V).
    let vocab = 16
    var oneHot = [Float](repeating: -1e9, count: vocab)
    oneHot[3] = 0
    let uniform = [Float](repeating: 0, count: vocab)
    let logits = MLXArray(oneHot + uniform).reshaped(1, 2, vocab)

    let entropy = canvasTokenEntropy(processedLogits: logits)
    #expect(entropy.shape == [1, 2])
    let values = entropy[0].asArray(Float.self)
    #expect(values[0] < 1e-3)
    #expect(abs(values[1] - log(Float(vocab))) < 1e-3)
}

@Test(.serialized)
func testEntropyBoundAcceptMask() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    // Acceptance at sorted rank k requires the SUM of all lower entropies
    // to stay within the bound (cum_k − sorted_k = Σ_{i<k}).
    // Entropies [0.0, 5.0, 0.2]: ranks give Σ_{i<k} = [0, 0, 0.2]
    // → with bound 0.1 accept positions 0 and 2, reject the 5.0 token.
    let entropy = MLXArray([Float(0.0), 5.0, 0.2]).reshaped(1, 3)
    let mask = entropyBoundAcceptMask(tokenEntropy: entropy, entropyBound: 0.1)
    #expect(mask.shape == [1, 3])
    let accepted = mask[0].asArray(Bool.self)
    #expect(accepted == [true, false, true])

    // The lowest-entropy token is always accepted, even when above the bound.
    let high = MLXArray([Float(3.0), 4.0]).reshaped(1, 2)
    let highMask = entropyBoundAcceptMask(tokenEntropy: high, entropyBound: 0.1)
    #expect(highMask[0].asArray(Bool.self) == [true, false])
}

@Test
func testStableConfidentStopper() {
    var stopper = StableConfidentStopper(stabilityThreshold: 1, confidenceThreshold: 0.005)

    let canvasA: [Int32] = [1, 2, 3]
    let canvasB: [Int32] = [1, 2, 4]

    // First step: no history yet → not stable, even when confident.
    #expect(stopper.shouldStop(argmaxCanvas: canvasA, meanEntropy: 0.001) == false)
    // Second step, same canvas, confident → stop.
    #expect(stopper.shouldStop(argmaxCanvas: canvasA, meanEntropy: 0.001) == true)

    stopper.reset()
    #expect(stopper.shouldStop(argmaxCanvas: canvasA, meanEntropy: 0.001) == false)
    // Canvas changed → not stable.
    #expect(stopper.shouldStop(argmaxCanvas: canvasB, meanEntropy: 0.001) == false)
    // Stable but not confident → keep denoising.
    #expect(stopper.shouldStop(argmaxCanvas: canvasB, meanEntropy: 0.5) == false)
    // Stable and confident → stop.
    #expect(stopper.shouldStop(argmaxCanvas: canvasB, meanEntropy: 0.0001) == true)
}

// MARK: - DiffusionGemma model family (Task 3)

/// Tiny configuration that exercises every diffusion code path: one sliding
/// + one full-attention layer, MoE routing, K=V full attention, canvas of 8.
private func tinyDiffusionGemmaConfiguration() throws -> DiffusionGemmaConfiguration {
    let json = """
        {
          "model_type": "diffusion_gemma",
          "canvas_length": 8,
          "eos_token_id": [1, 7],
          "image_token_id": 60,
          "text_config": {
            "model_type": "diffusion_gemma_text",
            "vocab_size": 64,
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 4,
            "global_head_dim": 4,
            "num_experts": 4,
            "top_k_experts": 2,
            "sliding_window": 4,
            "layer_types": ["sliding_attention", "full_attention"],
            "final_logit_softcapping": 30.0,
            "rms_norm_eps": 1e-6,
            "pad_token_id": 0,
            "rope_parameters": {
              "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
              "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0
              }
            }
          }
        }
        """
    return try JSONDecoder().decode(
        DiffusionGemmaConfiguration.self, from: Data(json.utf8))
}

@Test(.serialized)
func testDiffusionGemmaCacheTopologyAndDefaults() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let config = try tinyDiffusionGemmaConfiguration()
    let model = DiffusionGemmaModel(config)

    let cache = model.newCache(parameters: nil)
    #expect(cache.count == 2)
    let sliding = try #require(cache[0] as? RotatingKVCache)
    #expect(sliding.maxSize == 4)
    #expect(cache[1] is KVCacheSimple)

    let defaults = model.blockDiffusionDefaults
    #expect(defaults.canvasLength == 8)
    #expect(defaults.eosTokenIds == Set([1, 7]))
    #expect(defaults.padTokenId == 0)
    #expect(model.diffusionVocabularySize == 64)
}

@Test(.serialized)
func testDiffusionGemmaPrepareThrowsARGuard() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let config = try tinyDiffusionGemmaConfiguration()
    let model = DiffusionGemmaModel(config)
    let input = LMInput(text: .init(tokens: MLXArray([1, 2, 3].map { Int32($0) })))

    #expect(throws: BlockDiffusionModelError.self) {
        _ = try model.prepare(input, cache: model.newCache(parameters: nil), windowSize: 512)
    }
}

@Test(.serialized)
func testDiffusionGemmaSanitize() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let weights: [String: MLXArray] = [
        // Fused experts (E=4, out=2*8, in=8) → split into gate/up halves.
        "model.decoder.layers.0.experts.gate_up_proj.weight": MLXArray.zeros([4, 16, 8]),
        "model.decoder.layers.0.experts.gate_up_proj.scales": MLXArray.zeros([4, 16, 2]),
        "model.decoder.layers.0.experts.down_proj.weight": MLXArray.zeros([4, 8, 8]),
        // Encoder side: keep scalars, drop vision tower / embedder.
        "model.encoder.language_model.layers.0.layer_scalar": MLXArray.ones([1]),
        "model.encoder.vision_tower.patch_embedder.input_proj.weight": MLXArray.zeros([4, 4]),
        "model.encoder.embed_vision.embedding_projection.weight": MLXArray.zeros([4, 4]),
        // Ordinary decoder weight passes through unchanged.
        "model.decoder.layers.0.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
    ]

    let config = try tinyDiffusionGemmaConfiguration()
    let model = DiffusionGemmaModel(config)
    let sanitized = model.sanitize(weights: weights)

    let gate = try #require(
        sanitized["model.decoder.layers.0.experts.switch_glu.gate_proj.weight"])
    #expect(gate.shape == [4, 8, 8])
    let up = try #require(
        sanitized["model.decoder.layers.0.experts.switch_glu.up_proj.weight"])
    #expect(up.shape == [4, 8, 8])
    let gateScales = try #require(
        sanitized["model.decoder.layers.0.experts.switch_glu.gate_proj.scales"])
    #expect(gateScales.shape == [4, 8, 2])
    #expect(sanitized["model.decoder.layers.0.experts.switch_glu.down_proj.weight"] != nil)
    #expect(sanitized["model.decoder.layers.0.experts.gate_up_proj.weight"] == nil)
    #expect(sanitized["model.decoder.layers.0.experts.down_proj.weight"] == nil)

    #expect(sanitized["model.encoder.language_model.layers.0.layer_scalar"] != nil)
    #expect(
        sanitized["model.encoder.vision_tower.patch_embedder.input_proj.weight"] == nil)
    #expect(
        sanitized["model.encoder.embed_vision.embedding_projection.weight"] == nil)
    #expect(sanitized["model.decoder.layers.0.self_attn.q_proj.weight"] != nil)
}

@Test(.serialized)
func testDiffusionGemmaEncoderDecoderForward() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let config = try tinyDiffusionGemmaConfiguration()
    let model = DiffusionGemmaModel(config)
    let cache = model.newCache(parameters: nil)

    // Encoder prefill: 6 tokens > sliding window 4, so the rotating cache
    // wraps and the decoder read path must reorder it.
    let prompt = MLXArray([3, 4, 5, 6, 7, 8].map { Int32($0) }).expandedDimensions(axis: 0)
    model.encoderForward(prompt, cache: cache)
    #expect(cache[0].offset == 6)
    #expect(cache[1].offset == 6)

    // Decoder forward returns [1, C, V] and must NOT mutate the cache.
    let canvas = MLXArray((0 ..< 8).map { Int32($0 % 64) }).expandedDimensions(axis: 0)
    let logits = model.decoderForward(
        canvas: canvas, cache: cache, selfConditioningLogits: nil)
    #expect(logits.shape == [1, 8, 64])
    // Softcap bound holds (forces materialization of the graph).
    #expect(abs(logits).max().item(Float.self) <= 30.0)
    #expect(cache[0].offset == 6)
    #expect(cache[1].offset == 6)

    // Self-conditioning must influence the logits.
    let conditioned = model.decoderForward(
        canvas: canvas, cache: cache, selfConditioningLogits: logits)
    #expect(conditioned.shape == [1, 8, 64])
    let difference = abs(conditioned - logits).max().item(Float.self)
    #expect(difference > 1e-4)

    // A second encoder append (the finalized canvas) advances the cache.
    model.encoderForward(canvas, cache: cache)
    #expect(cache[0].offset == 14)
    #expect(cache[1].offset == 14)
}

// MARK: - BlockDiffusionTokenIterator (Task 5)

@Test(.serialized)
func testBlockDiffusionTokenIteratorEndToEnd() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let config = try tinyDiffusionGemmaConfiguration()
    let model = DiffusionGemmaModel(config)

    var parameters = GenerateParameters()
    parameters.maxTokens = 16

    let prompt = LMInput(
        text: .init(tokens: MLXArray([10, 11, 12, 13, 14].map { Int32($0) })))
    var iterator = try BlockDiffusionTokenIterator(
        input: prompt,
        model: model,
        parameters: parameters,
        options: model.blockDiffusionDefaults)

    var emitted: [Int] = []
    while let token = iterator.next() {
        emitted.append(token)
    }

    // At least one token; never beyond maxTokens; everything within vocab.
    #expect(!emitted.isEmpty)
    #expect(emitted.count <= 16)
    #expect(emitted.allSatisfy { $0 >= 0 && $0 < 64 })
    #expect(iterator.tokenCount == emitted.count)

    // Denoising loop budget: at most maxDenoisingSteps per canvas, and the
    // canvas cycle ran at least once.
    #expect(iterator.denoisingForwardCount >= 1)
    #expect(iterator.denoisingForwardCount <= 48 * 2)

    // End-of-turn cache contract: the encoder cache holds the prompt plus
    // exactly the emitted reply, minus a trailing EOS (mirrors the AR
    // iterator, whose final sampled token never re-enters the cache). This
    // is what makes multi-turn live-cache reuse coherent.
    var keptTokens = emitted
    if let last = keptTokens.last, [1, 7].contains(last) {
        keptTokens.removeLast()
    }
    let offset = iterator.cache[0].offset
    #expect(offset == 5 + keptTokens.count)
    #expect(iterator.cache[1].offset == offset)
}

@Test(.serialized)
func testARTokenIteratorRejectsDiffusionModel() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let config = try tinyDiffusionGemmaConfiguration()
    let model = DiffusionGemmaModel(config)
    let prompt = LMInput(
        text: .init(tokens: MLXArray([10, 11, 12].map { Int32($0) })))

    // The AR TokenIterator must fail loudly (model.prepare throws) instead
    // of silently next-token decoding a diffusion model.
    #expect(throws: BlockDiffusionModelError.self) {
        _ = try TokenIterator(
            input: prompt, model: model, parameters: GenerateParameters())
    }
}

// MARK: - Factory registration (Task 6)

@Test(.serialized)
func testFactoryRegistersDiffusionGemma() async throws {
    let configData = """
        {
          "model_type": "diffusion_gemma",
          "canvas_length": 8,
          "eos_token_id": [1, 7],
          "text_config": {
            "model_type": "diffusion_gemma_text",
            "vocab_size": 64,
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 4,
            "global_head_dim": 4,
            "num_experts": 4,
            "top_k_experts": 2,
            "sliding_window": 4,
            "layer_types": ["sliding_attention", "full_attention"],
            "rms_norm_eps": 1e-6,
            "pad_token_id": 0
          }
        }
        """.data(using: .utf8)!

    let modelTypeName = try await MLXMetalTestLock.withLock {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: configData, modelType: "diffusion_gemma")
        return String(describing: type(of: model))
    }
    #expect(modelTypeName.contains("DiffusionGemma"))
}

// MARK: - generation_config.json diffusion fields (Task 4)

@Test
func testGenerationConfigFileDecodesDiffusionFields() throws {
    // Verbatim from the local DiffusionGemma bundles.
    let json = """
        {
          "confidence_threshold": 0.005,
          "eos_token_id": [1, 106, 50],
          "max_denoising_steps": 48,
          "max_new_tokens": 256,
          "pad_token_id": 0,
          "sampler_config": {
            "_cls_name": "EntropyBoundSamplerConfig",
            "entropy_bound": 0.1
          },
          "stability_threshold": 1,
          "t_max": 0.8,
          "t_min": 0.4,
          "transformers_version": "5.8.0.dev0"
        }
        """
    let config = try JSONDecoder().decode(
        GenerationConfigFile.self, from: Data(json.utf8))
    #expect(config.eosTokenIds?.values == [1, 106, 50])
    #expect(config.maxNewTokens == 256)
    #expect(config.maxDenoisingSteps == 48)
    #expect(config.samplerConfig?.entropyBound == 0.1)
    #expect(config.tMin == 0.4)
    #expect(config.tMax == 0.8)
    #expect(config.stabilityThreshold == 1)
    #expect(config.confidenceThreshold == 0.005)
    #expect(config.padTokenId == 0)

    let defaults = BlockDiffusionParameters(canvasLength: 256)
        .resolving(generationConfig: config)
    #expect(defaults.canvasLength == 256)
    #expect(defaults.maxDenoisingSteps == 48)

    // Per-request speed/quality override (app slider path).
    var fast = GenerateParameters()
    fast.diffusionMaxDenoisingSteps = 16
    #expect(defaults.overriding(parameters: fast).maxDenoisingSteps == 16)
    var clamped = GenerateParameters()
    clamped.diffusionMaxDenoisingSteps = 0
    #expect(defaults.overriding(parameters: clamped).maxDenoisingSteps == 1)
    #expect(defaults.overriding(parameters: GenerateParameters()).maxDenoisingSteps == 48)
    #expect(defaults.entropyBound == 0.1)
    #expect(defaults.eosTokenIds == Set([1, 106, 50]))
    #expect(defaults.padTokenId == 0)
}
