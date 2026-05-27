import Foundation
import MLX
import MLXLMCommon
@testable import MLXLLM
import Testing

@Suite("MiMo V2.5 hybrid full/SWA cache topology")
struct MiMoV2FlashCacheTopologyTests {
    private static let minimalMiMoV25Config = #"""
    {
      "model_type": "mimo_v2",
      "num_experts_per_tok": 2,
      "hybrid_layer_pattern": [0, 1, 1, 0],
      "moe_layer_freq": [0, 1, 1, 1],
      "add_swa_attention_sink_bias": true,
      "add_full_attention_sink_bias": false,
      "sliding_window_size": 128,
      "vocab_size": 256,
      "hidden_size": 32,
      "intermediate_size": 64,
      "moe_intermediate_size": 16,
      "num_hidden_layers": 4,
      "num_attention_heads": 8,
      "num_key_value_heads": 4,
      "n_shared_experts": 1,
      "n_routed_experts": 4,
      "routed_scaling_factor": 1.0,
      "topk_method": "noaux_tc",
      "scoring_func": "sigmoid",
      "norm_topk_prob": true,
      "n_group": 1,
      "topk_group": 1,
      "max_position_embeddings": 4096,
      "layernorm_epsilon": 1e-6,
      "rope_theta": 10000000,
      "swa_rope_theta": 10000,
      "swa_num_attention_heads": 8,
      "swa_num_key_value_heads": 8,
      "head_dim": 4,
      "v_head_dim": 4,
      "swa_head_dim": 4,
      "swa_v_head_dim": 4,
      "partial_rotary_factor": 0.5,
      "attention_value_scale": 0.707
    }
    """#

    @Test("model_type=mimo_v2 dispatches to MiMoV2FlashModel")
    func modelTypeAliasDispatches() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(Self.minimalMiMoV25Config.utf8),
            modelType: "mimo_v2")

        #expect(model is MiMoV2FlashModel)
    }

    @Test("per-layer KV heads and cache classes match full/SWA layer pattern")
    func perLayerKVHeadsAndCacheTopologyFollowHybridPattern() throws {
        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Data(Self.minimalMiMoV25Config.utf8))
        let model = MiMoV2FlashModel(config)

        #expect(model.modelType == "mimo_v2")
        #expect(model.kvHeads == [4, 8, 8, 4])

        let cache = model.newCache(parameters: GenerateParameters(
            maxTokens: 8,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3)))
        #expect(cache.count == 4)
        #expect(cache[0] is KVCacheSimple)
        #expect(cache[1] is RotatingKVCache)
        #expect(cache[2] is RotatingKVCache)
        #expect(cache[3] is KVCacheSimple)
        #expect(!cache.contains { $0 is TurboQuantKVCache })
    }

    @Test("configuration decodes MiMo V2.5 attention value scale")
    func configurationDecodesAttentionValueScale() throws {
        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Data(Self.minimalMiMoV25Config.utf8))

        #expect(config.attentionValueScale == 0.707)
    }

    @Test("sanitize splits fused MiMo qkv projection by layer-specific KV shape")
    func sanitizeSplitsFusedQKVProjection() throws {
        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Data(Self.minimalMiMoV25Config.utf8))
        let model = MiMoV2FlashModel(config)

        let sanitized = model.sanitize(weights: [
            "model.layers.0.self_attn.qkv_proj.weight": MLXArray.ones([64, 2]),
            "model.layers.0.self_attn.qkv_proj.scales": MLXArray.ones([64, 1]),
            "model.layers.0.self_attn.qkv_proj.biases": MLXArray.zeros([64, 1]),
            "model.layers.1.self_attn.qkv_proj.weight": MLXArray.ones([96, 2]),
            "model.layers.1.self_attn.qkv_proj.scales": MLXArray.ones([96, 1]),
            "model.layers.1.self_attn.qkv_proj.biases": MLXArray.zeros([96, 1]),
        ])

        #expect(sanitized["model.layers.0.self_attn.qkv_proj.weight"] == nil)
        #expect(sanitized["model.layers.0.self_attn.q_proj.weight"]?.shape == [32, 2])
        #expect(sanitized["model.layers.0.self_attn.k_proj.weight"]?.shape == [16, 2])
        #expect(sanitized["model.layers.0.self_attn.v_proj.weight"]?.shape == [16, 2])
        #expect(sanitized["model.layers.0.self_attn.q_proj.scales"]?.shape == [32, 1])
        #expect(sanitized["model.layers.0.self_attn.k_proj.biases"]?.shape == [16, 1])

        #expect(sanitized["model.layers.1.self_attn.qkv_proj.weight"] == nil)
        #expect(sanitized["model.layers.1.self_attn.q_proj.weight"]?.shape == [32, 2])
        #expect(sanitized["model.layers.1.self_attn.k_proj.weight"]?.shape == [32, 2])
        #expect(sanitized["model.layers.1.self_attn.v_proj.weight"]?.shape == [32, 2])
        #expect(sanitized["model.layers.1.self_attn.v_proj.scales"]?.shape == [32, 1])
    }

    @Test("TurboQuant KV only promotes full-attention cache layers")
    func turboQuantOnlyPromotesFullAttentionKVLayers() throws {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Data(Self.minimalMiMoV25Config.utf8))
        let model = MiMoV2FlashModel(config)
        var cache = model.newCache(parameters: GenerateParameters(
            maxTokens: 8,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3)))
        let tokenCount = 4
            + TurboQuantKVCache.defaultResidualTokens
            + TurboQuantKVCache.minimumCompressedTokens
            + 8
        Self.fill(cache: cache, tokenCount: tokenCount)

        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: nil,
            quantizedKVStart: 0,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3))

        #expect(cache[0] is TurboQuantKVCache)
        #expect(cache[1] is RotatingKVCache)
        #expect(cache[2] is RotatingKVCache)
        #expect(cache[3] is TurboQuantKVCache)
    }

    @Test("cache topology snapshot marks MiMo full/SWA KV as disk-backed rotating topology")
    func cacheTopologySnapshotMarksMiMoHybridFullSWA() throws {
        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Data(Self.minimalMiMoV25Config.utf8))
        let model = MiMoV2FlashModel(config)

        let snapshot = ModelCacheTopologySnapshot(cache: model.newCache(parameters: nil))

        #expect(snapshot.layerCount == 4)
        #expect(snapshot.kvLayerCount == 2)
        #expect(snapshot.rotatingKVLayerCount == 2)
        #expect(snapshot.requiresSSMCompanionState == false)
        #expect(snapshot.requiresDiskBackedCoordinatorRestore)
        #expect(snapshot.topologyTags.contains("kvLayers=2"))
        #expect(snapshot.topologyTags.contains("rotatingLayers=2"))
        #expect(snapshot.topologyTags.contains("restore=disk-backed"))
        #expect(!snapshot.topologyTags.contains("companion=ssm"))
    }

    @Test("L2 disk round trip preserves hybrid full/SWA cache kinds")
    func l2DiskRoundTripPreservesHybridCacheKinds() throws {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Data(Self.minimalMiMoV25Config.utf8))
        let model = MiMoV2FlashModel(config)
        let cache = model.newCache(parameters: GenerateParameters(maxTokens: 8))
        Self.fill(cache: cache, tokenCount: 12)

        let arrays = TQDiskSerializer.serialize(cache: cache)
        #expect(TQDiskSerializer.formatVersion(of: arrays) == 2)
        #expect(Self.layerKind(arrays, 0) == TQDiskSerializer.LayerKind.kv.rawValue)
        #expect(Self.layerKind(arrays, 1) == TQDiskSerializer.LayerKind.rotating.rawValue)
        #expect(Self.layerKind(arrays, 2) == TQDiskSerializer.LayerKind.rotating.rawValue)
        #expect(Self.layerKind(arrays, 3) == TQDiskSerializer.LayerKind.kv.rawValue)

        var restored = model.newCache(parameters: GenerateParameters(maxTokens: 8))
        let restoredTokens = restoreFromDiskArrays(arrays, into: &restored)
        #expect(restoredTokens == 12)
        #expect(restored[0] is KVCacheSimple)
        #expect(restored[1] is RotatingKVCache)
        #expect(restored[2] is RotatingKVCache)
        #expect(restored[3] is KVCacheSimple)
        #expect(restored[0].state[0].dim(2) == 12)
        #expect((restored[1] as? RotatingKVCache)?.offset == 12)
        #expect((restored[2] as? RotatingKVCache)?.offset == 12)
        #expect(restored[3].state[0].dim(2) == 12)
    }

    @Test("L2 disk round trip preserves TurboQuant full layers plus rotating SWA")
    func l2DiskRoundTripPreservesTurboQuantFullAndRotatingSWA() throws {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Data(Self.minimalMiMoV25Config.utf8))
        let model = MiMoV2FlashModel(config)
        var cache = model.newCache(parameters: GenerateParameters(
            maxTokens: 8,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3)))
        let tokenCount = 4
            + TurboQuantKVCache.defaultResidualTokens
            + TurboQuantKVCache.minimumCompressedTokens
            + 8
        Self.fill(cache: cache, tokenCount: tokenCount)

        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: nil,
            quantizedKVStart: 0,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3))

        let arrays = TQDiskSerializer.serialize(cache: cache)
        #expect(TQDiskSerializer.formatVersion(of: arrays) == 2)
        #expect(Self.layerKind(arrays, 0) == TQDiskSerializer.LayerKind.tq.rawValue)
        #expect(Self.layerKind(arrays, 1) == TQDiskSerializer.LayerKind.rotating.rawValue)
        #expect(Self.layerKind(arrays, 2) == TQDiskSerializer.LayerKind.rotating.rawValue)
        #expect(Self.layerKind(arrays, 3) == TQDiskSerializer.LayerKind.tq.rawValue)

        var restored = model.newCache(parameters: GenerateParameters(maxTokens: 8))
        let restoredTokens = restoreFromDiskArrays(arrays, into: &restored)
        #expect(restoredTokens == tokenCount)
        #expect(restored[0] is TurboQuantKVCache)
        #expect(restored[1] is RotatingKVCache)
        #expect(restored[2] is RotatingKVCache)
        #expect(restored[3] is TurboQuantKVCache)
        #expect((restored[0] as? TurboQuantKVCache)?.offset == tokenCount)
        #expect((restored[1] as? RotatingKVCache)?.offset == tokenCount)
        #expect((restored[2] as? RotatingKVCache)?.offset == tokenCount)
        #expect((restored[3] as? TurboQuantKVCache)?.offset == tokenCount)
    }

    private static func fill(cache: [any KVCache], tokenCount: Int) {
        for (index, layer) in cache.enumerated() {
            let heads = index == 1 || index == 2 ? 8 : 4
            let keys = MLXArray.ones([1, heads, tokenCount, 4], dtype: .bfloat16)
            let values = MLXArray.ones([1, heads, tokenCount, 4], dtype: .bfloat16) * Float(index + 1)
            _ = layer.update(keys: keys, values: values)
        }
        MLX.eval(cache.flatMap(\.state))
    }

    private static func layerKind(_ arrays: [String: MLXArray], _ index: Int) -> Int32? {
        arrays["__layer_kind_\(index)__"]?.item(Int32.self)
    }
}
