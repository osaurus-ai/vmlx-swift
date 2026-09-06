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

    /// Ling 2.6 (GLA) shaped config — a minimal field set naming the 2.6
    /// architecture, with NO V3 markers. The architecture name is the ONLY
    /// route to the GLA runtime (osaurus#2652: a genuine Ling 2.6 bundle
    /// must load; #424: a marker-less config must never reach GLA).
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

    @Test("a config naming the Ling 2.6 architecture dispatches to the GLA runtime (osaurus#2652)")
    func legacyArchitectureDispatchesToGLA() async throws {
        // Ling 2.6 flash JANGTQ (`architectures: [BailingMoeV2_5ForCausalLM]`,
        // no KDA marker) is a real bundle users run; it must reach
        // `BailingHybridModel`, never the V3 decoder and never a refusal.
        let name = try await MLXMetalTestLock.withLock {
            let model = try await LLMTypeRegistry.shared.createModel(
                configuration: Self.legacyConfigJSON, modelType: "bailing_hybrid")
            return String(describing: type(of: model))
        }
        #expect(name.contains("BailingHybridModel"), Comment(rawValue: name))
        #expect(!name.contains("BailingMoeV3"))
    }

    @Test("a config naming the 2.6 architecture WITH a Ling 3 marker is Ling 3 (markers win)")
    func v3MarkerBeatsLegacyArchitectureName() async throws {
        // Belt and braces: a converter that leaves a stale V2 architecture
        // string on a Ling 3 config still lands on the KDA runtime.
        var root = try #require(
            try JSONSerialization.jsonObject(with: Self.v3ConfigJSON) as? [String: Any])
        root["architectures"] = ["BailingMoeV2_5ForCausalLM"]
        let data = try JSONSerialization.data(withJSONObject: root)
        let name = try await MLXMetalTestLock.withLock {
            let model = try await LLMTypeRegistry.shared.createModel(
                configuration: data, modelType: "bailing_hybrid")
            return String(describing: type(of: model))
        }
        #expect(name.contains("BailingMoeV3"), Comment(rawValue: name))
    }

    @Test("no separate model_type reaches the Ling 2.6 GLA runtime (bailing_moe_v2_5 is unregistered)")
    func noRouteToGLA() async throws {
        let outcome: String = try await MLXMetalTestLock.withLock {
            do {
                let model = try await LLMTypeRegistry.shared.createModel(
                    configuration: Self.legacyConfigJSON, modelType: "bailing_moe_v2_5")
                return "loaded:" + String(describing: type(of: model))
            } catch {
                return "refused:" + String(describing: error)
            }
        }
        #expect(outcome.hasPrefix("refused:"), Comment(rawValue: outcome))
        #expect(!outcome.contains("BailingHybridModel"))
    }

    @Test("bailing_hybrid dispatches to V3 with every Ling 3 marker stripped from the config")
    func v3DispatchWithoutMarkers() async throws {
        var dict = try JSONSerialization.jsonObject(with: Self.v3ConfigJSON) as! [String: Any]
        dict.removeValue(forKey: "architectures")
        dict.removeValue(forKey: "linear_attention")
        dict.removeValue(forKey: "kda_lower_bound")
        let data = try JSONSerialization.data(withJSONObject: dict)
        let name = try await MLXMetalTestLock.withLock {
            let model = try await LLMTypeRegistry.shared.createModel(
                configuration: data, modelType: "bailing_hybrid")
            return String(describing: type(of: model))
        }
        #expect(name.contains("BailingMoeV3"))
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

    /// The kernel used to receive the decay narrowed to the activation dtype:
    /// in bf16 every per-channel decay above ~0.998 is exactly 1.0 (that
    /// channel never forgets) and 0.997 becomes 0.996. Nine steps of random
    /// gates never show it; a long horizon of slow-decay channels does. The
    /// ops fallback keeps everything fp32 and is the reference here.
    @Test("KDA Metal kernel keeps slow decays over a long horizon (fp32 gate into the kernel)")
    func kdaKernelSlowDecayLongHorizon() throws {
        try MLXMetalTestLock.withLock {
            MLXRandom.seed(11)
            let (B, T, H, Dk, Dv) = (1, 4096, 1, 32, 32)
            let q = (MLXRandom.normal([B, T, H, Dk]) * 0.3).asType(.bfloat16)
            let k = (MLXRandom.normal([B, T, H, Dk]) * 0.3).asType(.bfloat16)
            let v = (MLXRandom.normal([B, T, H, Dv]) * 0.3).asType(.bfloat16)
            // Very negative pre-activations → sigmoid ≈ 1.7e-5 → g ≈ -8e-5 →
            // decay ≈ 0.99992: representable in fp32, exactly 1.0 in bf16
            // (the largest bf16 below 1 is 0.99609). Over 4096 steps the
            // fp32 reference forgets ~28% of early content; the narrowed
            // kernel forgets nothing.
            let fRaw = (MLXRandom.normal([B, T, H, Dk]) * 0.2 - 11.0).asType(.bfloat16)
            let beta = sigmoid(MLXRandom.normal([B, T, H])).asType(.bfloat16)
            let aLog = MLXArray(converting: [0.0])
            let dtBias = MLXArray.zeros([H * Dk])

            let (yKernel, sKernel) = kdaUpdate(
                q: q, k: k, v: v, fRaw: fRaw, beta: beta,
                aLog: aLog, dtBias: dtBias, safeGate: true, lowerBound: -5)
            let g = computeKDADecay(fRaw: fRaw, aLog: aLog, dtBias: dtBias, safeGate: true, lowerBound: -5)
            #expect(g.min().item(Float.self) > 0.9997, "fixture must exercise decays bf16 cannot represent")
            #expect(g.max().item(Float.self) < 1.0, "fixture must actually decay")
            let state = MLXArray.zeros([B, H, Dv, Dk], dtype: .float32)
            let (yOps, sOps) = gatedDeltaOps(
                q: q, k: k, v: v, g: g, beta: beta.asType(.float32), state: state)

            let sScale = abs(sOps).max().item(Float.self)
            let sDiff = abs(sKernel.asType(.float32) - sOps.asType(.float32)).max().item(Float.self)
            let yDiff = abs(yKernel.asType(.float32)[0, -1] - yOps.asType(.float32)[0, -1]).max().item(Float.self)
            #expect(sDiff / max(sScale, 1e-6) < 1e-3, "final state diverges from the fp32 reference: \(sDiff) (scale \(sScale)); the narrowed kernel measured 1.1e-2 here, the fp32 one 4e-7")
            #expect(yDiff < 1e-3, "last-step output diverges from the fp32 reference: \(yDiff); the narrowed kernel measured 1.6e-2 here")
        }
    }

    /// fla's l2norm puts eps on the SUM of squares; rmsNorm puts it on the
    /// MEAN. The equivalent rmsNorm eps is therefore eps / D. With a bare
    /// 1e-6 the stabiliser is D× too large — invisible on ordinary
    /// activations, measurable on small ones.
    @Test("KDA q/k normalisation reproduces fla's l2norm (eps on the sum of squares)")
    func kdaQKNormMatchesL2Norm() throws {
        try MLXMetalTestLock.withLock {
            let D = 128
            MLXRandom.seed(3)
            let x = (MLXRandom.normal([1, 4, 2, D]) * 0.002).asType(.float32)  // small activations
            let invScale = pow(Float(D), -0.5)
            let l2 = x * rsqrt((x * x).sum(axis: -1, keepDims: true) + 1e-6)  // fla l2norm
            let viaRms = MLXArray(invScale) * MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: 1e-6 / Float(D))
            let viaRmsBareEps = MLXArray(invScale) * MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: 1e-6)
            let good = abs(viaRms - l2).max().item(Float.self)
            let bad = abs(viaRmsBareEps - l2).max().item(Float.self)
            #expect(good < 1e-5, "eps/D form must equal l2norm: \(good)")
            #expect(bad > 1e-3, "the bare-eps form is measurably different on small activations: \(bad)")
        }
    }

    /// The reference combines routed expert outputs with float32 routing
    /// weights and sums in float32; rounding the weights to bf16 and summing
    /// the top-k products in bf16 is a different result.
    @Test("routed-expert combination is float32 (weights and sum), cast back once")
    func moeCombineIsFloat32() throws {
        try MLXMetalTestLock.withLock {
            MLXRandom.seed(5)
            let y = MLXRandom.normal([2, 3, 8, 64]).asType(.bfloat16)          // [B, T, topk, D]
            let scores = softmax(MLXRandom.normal([2, 3, 8]), axis: -1)          // float32
            let combined = BailingV3SparseMoE.combineExpertOutputs(y, scores: scores)
            #expect(combined.dtype == .bfloat16)
            let reference = (y.asType(.float32) * scores[.ellipsis, .newAxis]).sum(axis: -2)
            let bf16Path = (y * scores[.ellipsis, .newAxis].asType(.bfloat16)).sum(axis: -2).asType(.float32)
            let ours = abs(combined.asType(.float32) - reference).max().item(Float.self)
            let old = abs(bf16Path - reference).max().item(Float.self)
            // Ours differs from the float32 reference only by the final bf16 rounding of the result.
            #expect(ours <= abs(reference).max().item(Float.self) * 0.008, "combine deviates from the float32 reference: \(ours)")
            #expect(old > ours, "the bf16 path must be measurably worse (old \(old) vs ours \(ours))")
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
