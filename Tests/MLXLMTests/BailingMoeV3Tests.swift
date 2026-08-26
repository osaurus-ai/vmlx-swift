// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// Ling 3.0 (BailingMoeV3): the same `model_type` string as Ling 2.6 resolves
/// to a DIFFERENT architecture. These tests pin the dispatch, the layer
/// typing rule, the KDA kernel/ops numeric parity, and cached-decode parity
/// on a tiny synthetic model.
@Suite("BailingMoeV3 (Ling 3.0)", .serialized)
struct BailingMoeV3Tests {

    /// Field set copied from the real Ling-3.0-tiny `config.json` (values
    /// reduced only where irrelevant to dispatch). Schema-shaped on purpose:
    /// a hand-invented fixture once hid a decode bug behind defaults.
    static let v3ConfigJSON = """
        {
          "model_type": "bailing_hybrid",
          "architectures": ["BailingMoeV3ForCausalLM"],
          "hidden_size": 64,
          "num_hidden_layers": 8,
          "intermediate_size": 128,
          "num_attention_heads": 2,
          "num_key_value_heads": 2,
          "head_dim": 32,
          "vocab_size": 128,
          "rms_norm_eps": 1e-6,
          "rope_theta": 6000000,
          "tie_word_embeddings": false,
          "layer_group_size": 4,
          "first_k_dense_replace": 1,
          "short_conv_kernel_size": 4,
          "no_kda_lora": true,
          "kda_safe_gate": true,
          "kda_lower_bound": -5,
          "linear_attention": "kda",
          "q_lora_rank": 32,
          "kv_lora_rank": 32,
          "qk_rope_head_dim": 16,
          "qk_nope_head_dim": 32,
          "v_head_dim": 32,
          "rope_interleave": true,
          "use_qkv_bias": false,
          "gated_attention_proj_granularity_type": "head_wise",
          "num_experts": 8,
          "num_experts_per_tok": 2,
          "num_shared_experts": 1,
          "moe_intermediate_size": 32,
          "moe_shared_expert_intermediate_size": 32,
          "n_group": 2,
          "topk_group": 1,
          "routed_scaling_factor": 2.5,
          "norm_topk_prob": true,
          "score_function": "sigmoid",
          "moe_router_enable_expert_bias": true
        }
        """.data(using: .utf8)!

    /// Ling 2.6 (GLA) still resolves to the legacy runtime — a minimal
    /// field set the legacy configuration accepts, with NO V3 markers.
    static let legacyConfigJSON = """
        {
          "model_type": "bailing_hybrid",
          "architectures": ["BailingMoeV2ForCausalLM"],
          "max_position_embeddings": 32768,
          "hidden_size": 64,
          "num_hidden_layers": 4,
          "intermediate_size": 128,
          "num_attention_heads": 2,
          "num_key_value_heads": 2,
          "head_dim": 32,
          "vocab_size": 128,
          "rms_norm_eps": 1e-6,
          "rope_theta": 600000,
          "tie_word_embeddings": true,
          "num_experts": 8,
          "num_experts_per_tok": 2,
          "num_shared_experts": 1,
          "moe_intermediate_size": 32,
          "first_k_dense_replace": 1,
          "q_lora_rank": 32,
          "kv_lora_rank": 32,
          "qk_rope_head_dim": 16,
          "qk_nope_head_dim": 32,
          "v_head_dim": 32
        }
        """.data(using: .utf8)!

    @Test("architectures=BailingMoeV3ForCausalLM dispatches to the V3 runtime")
    func v3Dispatch() async throws {
        let name = try await MLXMetalTestLock.withLock {
            let model = try await LLMTypeRegistry.shared.createModel(
                configuration: Self.v3ConfigJSON, modelType: "bailing_hybrid")
            return String(describing: type(of: model))
        }
        #expect(name.contains("BailingMoeV3"))
    }

    @Test("a V2 config keeps resolving to the legacy GLA runtime")
    func legacyDispatch() async throws {
        let name = try await MLXMetalTestLock.withLock {
            let model = try await LLMTypeRegistry.shared.createModel(
                configuration: Self.legacyConfigJSON, modelType: "bailing_hybrid")
            return String(describing: type(of: model))
        }
        #expect(name.contains("BailingHybridModel"))
        #expect(!name.contains("V3"))
    }

    @Test("layer typing matches the reference rule for 24 layers / group 4")
    func layerTyping() throws {
        var config = try JSONDecoder().decode(
            BailingMoeV3Configuration.self, from: Self.v3ConfigJSON)
        config.numHiddenLayers = 24
        config.layerGroupSize = 4
        let full = (0 ..< 24).filter { config.isFullAttentionLayer($0) }
        #expect(full == [3, 7, 11, 15, 19, 23])

        // Trailing partial group is full attention (reference line 1006-1008).
        config.numHiddenLayers = 26
        let fullTail = (0 ..< 26).filter { config.isFullAttentionLayer($0) }
        #expect(fullTail == [3, 7, 11, 15, 19, 23, 24, 25])
    }

    @Test("KDA Metal kernel matches the ops fallback numerically")
    func kdaKernelOpsParity() throws {
        try MLXMetalTestLock.withLock {
            MLXRandom.seed(7)
            let (B, T, H, Dk, Dv) = (1, 9, 2, 32, 32)
            let q = MLXRandom.normal([B, T, H, Dk]) * 0.3
            let k = MLXRandom.normal([B, T, H, Dk]) * 0.3
            let v = MLXRandom.normal([B, T, H, Dv]) * 0.3
            let fRaw = MLXRandom.normal([B, T, H, Dk])
            let beta = sigmoid(MLXRandom.normal([B, T, H]))
            let aLog = MLXRandom.normal([H]) * 0.5
            let dtBias = MLXRandom.normal([H * Dk]) * 0.1

            // Dk = 32 → kernel path.
            let (yKernel, sKernel) = kdaUpdate(
                q: q, k: k, v: v, fRaw: fRaw, beta: beta,
                aLog: aLog, dtBias: dtBias, safeGate: true, lowerBound: -5)

            // Same math through the ops fallback: precompute the decay and
            // call gatedDeltaOps directly, exactly as kdaUpdate's fallback
            // branch does.
            let g = computeKDADecay(
                fRaw: fRaw, aLog: aLog, dtBias: dtBias,
                safeGate: true, lowerBound: -5)
            let state = MLXArray.zeros([B, H, Dv, Dk], dtype: .float32)
            let (yOps, sOps) = gatedDeltaOps(
                q: q, k: k, v: v, g: g, beta: beta.asType(.float32), state: state)

            let yDiff = abs(yKernel.asType(.float32) - yOps.asType(.float32)).max()
                .item(Float.self)
            let sDiff = abs(sKernel.asType(.float32) - sOps.asType(.float32)).max()
                .item(Float.self)
            #expect(yDiff < 2e-3, "kernel/ops output diverge: \(yDiff)")
            #expect(sDiff < 2e-3, "kernel/ops state diverge: \(sDiff)")
        }
    }

    @Test("safe-gate decay is bounded to (exp(lower_bound), 1)")
    func safeGateBounds() throws {
        try MLXMetalTestLock.withLock {
            let fRaw = MLXRandom.normal([1, 4, 2, 8]) * 10  // extreme inputs
            let aLog = MLXArray(converting: [0.5, -0.5])
            let dtBias = MLXArray.zeros([16])
            let g = computeKDADecay(
                fRaw: fRaw, aLog: aLog, dtBias: dtBias, safeGate: true, lowerBound: -5)
            let mn = g.min().item(Float.self)
            let mx = g.max().item(Float.self)
            #expect(mn >= exp(Float(-5)) - 1e-6)
            #expect(mx <= 1.0 + 1e-6)
        }
    }

    @Test("tiny model: one-shot prefill equals cached prefill + decode step")
    func cachedDecodeParity() throws {
        try MLXMetalTestLock.withLock {
            MLXRandom.seed(11)
            let config = try JSONDecoder().decode(
                BailingMoeV3Configuration.self, from: Self.v3ConfigJSON)
            let model = BailingMoeV3Model(config)
            MLX.eval(model.parameters())

            let tokens = MLXArray([3, 17, 42, 5, 99, 7].map(Int32.init))

            // One shot, no cache.
            let full = model(tokens.reshaped(1, 6), cache: nil)
            let fullLast = full[0, 5]

            // Prefill 5, then decode token 6 through the cache.
            let cache = model.newCache(parameters: nil)
            _ = model(tokens[0 ..< 5].reshaped(1, 5), cache: cache)
            let step = model(tokens[5 ..< 6].reshaped(1, 1), cache: cache)
            let stepLast = step[0, 0]

            let diff = abs(
                fullLast.asType(.float32) - stepLast.asType(.float32)
            ).max().item(Float.self)
            #expect(diff < 2e-2, "cached decode diverged from one-shot: \(diff)")
        }
    }
}

@Suite("per_tensor quantization map decode")
struct PerTensorQuantizationDecodeTests {
    /// The JANG stamper writes per-tensor overrides as ONE `per_tensor` map.
    /// The dynamic key scan used to parse the map itself as a single
    /// Quantization and threw "Missing field 'quantization.per_tensor.bits'",
    /// blocking every load of Ling-3.0-tiny-JANG_6M.
    @Test("Ling 6M quantization dict decodes; entries become per-layer overrides")
    func perTensorMapDecodes() throws {
        let json = """
            {
              "model_type": "bailing_hybrid",
              "quantization": {
                "group_size": 64,
                "bits": 8,
                "per_tensor": {
                  "lm_head": {"mode": "affine", "bits": 8, "group_size": 64},
                  "model.layers.0.attention.k_proj": {"mode": "affine", "bits": 6, "group_size": 64}
                }
              }
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder().decode(BaseConfiguration.self, from: json)
        #expect(config.quantization?.bits == 8)
        let lmHead = config.perLayerQuantization?.quantization(layer: "lm_head")
        #expect(lmHead?.bits == 8)
        let kProj = config.perLayerQuantization?.quantization(
            layer: "model.layers.0.attention.k_proj")
        #expect(kProj?.bits == 6)
    }
}

extension BailingMoeV3Tests {
    /// Zero disk-cache stores, measured live: the eligibility gate requires
    /// every layer's offset to advance, and the KDA layers stayed at 0.
    @Test("KDA forward advances the MambaCache offset by the segment length")
    func kdaAdvancesOffset() throws {
        try MLXMetalTestLock.withLock {
            let config = try JSONDecoder().decode(
                BailingMoeV3Configuration.self, from: Self.v3ConfigJSON)
            let model = BailingMoeV3Model(config)
            MLX.eval(model.parameters())
            let cache = model.newCache(parameters: nil)
            _ = model(MLXArray([1, 2, 3, 4, 5].map(Int32.init)).reshaped(1, 5), cache: cache)
            for (i, c) in cache.enumerated() {
                #expect(c.offset == 5, "layer \(i) offset \(c.offset) != 5")
            }
            _ = model(MLXArray([7].map(Int32.init)).reshaped(1, 1), cache: cache)
            for (i, c) in cache.enumerated() {
                #expect(c.offset == 6, "layer \(i) offset \(c.offset) != 6 after decode step")
            }
        }
    }
}
