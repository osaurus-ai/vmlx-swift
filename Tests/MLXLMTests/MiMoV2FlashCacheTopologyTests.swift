import Foundation
import MLX
import MLXDistributedTP
import MLXLMCommon
@testable import MLXLLM
import MLXNN
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

    @Test("JANGTQ config builds TurboQuant switch MLP and drops tq_bits metadata")
    func jangtqConfigBuildsTurboQuantSwitchMLPAndDropsTQBitsMetadata() throws {
        let configJSON = Self.minimalMiMoV25Config.replacingOccurrences(
            of: #""attention_value_scale": 0.707"#,
            with: """
      "attention_value_scale": 0.707,
      "weight_format": "mxtq",
      "mxtq_bits": {
        "routed_expert": {
          "gate_proj": 2,
          "up_proj": 2,
          "down_proj": 4
        }
      },
      "mxtq_seed": 7
""")
        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Data(configJSON.utf8))
        #expect(config.weightFormat == "mxtq")
        #expect(config.usesTurboQuantRoutedExperts)
        #expect(config.routedExpertQuantization(layerIndex: 1, projection: "gate_proj").bits == 2)
        #expect(config.routedExpertQuantization(layerIndex: 1, projection: "up_proj").bits == 2)
        #expect(config.routedExpertQuantization(layerIndex: 1, projection: "down_proj").bits == 4)
        let model = MiMoV2FlashModel(config)

        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        let gate = try #require(
            leaves["model.layers.1.mlp.switch_mlp.gate_proj"] as? TurboQuantSwitchLinear)
        let up = try #require(
            leaves["model.layers.1.mlp.switch_mlp.up_proj"] as? TurboQuantSwitchLinear)
        let down = try #require(
            leaves["model.layers.1.mlp.switch_mlp.down_proj"] as? TurboQuantSwitchLinear)

        #expect(gate.bits == 2)
        #expect(up.bits == 2)
        #expect(down.bits == 4)

        let sanitized = model.sanitize(weights: [
            "model.layers.1.mlp.switch_mlp.gate_proj.tq_packed": MLXArray.ones(
                [4, 2, 2], dtype: .uint32),
            "model.layers.1.mlp.switch_mlp.gate_proj.tq_norms": MLXArray.ones(
                [4, 2, 1], dtype: .float32),
            "model.layers.1.mlp.switch_mlp.gate_proj.tq_bits": MLXArray(Int32(2)),
            "model.layers.1.mlp.switch_mlp.up_proj.tq_packed": MLXArray.ones(
                [4, 2, 2], dtype: .uint32),
            "model.layers.1.mlp.switch_mlp.up_proj.tq_norms": MLXArray.ones(
                [4, 2, 1], dtype: .float32),
            "model.layers.1.mlp.switch_mlp.up_proj.tq_bits": MLXArray(Int32(2)),
            "model.layers.1.mlp.switch_mlp.down_proj.tq_packed": MLXArray.ones(
                [4, 2, 2], dtype: .uint32),
            "model.layers.1.mlp.switch_mlp.down_proj.tq_norms": MLXArray.ones(
                [4, 2, 1], dtype: .float32),
            "model.layers.1.mlp.switch_mlp.down_proj.tq_bits": MLXArray(Int32(4)),
            "audio_encoder.input_local_transformer.layers.0.input_layernorm.weight":
                MLXArray.ones([32], dtype: .float32),
            "encoder.layers.0.self_attn.q_proj.weight": MLXArray.ones([32, 32], dtype: .float32),
            "speech_embeddings.0.weight": MLXArray.ones([16, 32], dtype: .float32),
            "visual.blocks.0.attn.qkv.weight": MLXArray.ones([32, 32], dtype: .float32),
        ])

        #expect(sanitized["model.layers.1.mlp.switch_mlp.gate_proj.tq_packed"] != nil)
        #expect(sanitized["model.layers.1.mlp.switch_mlp.gate_proj.tq_norms"] != nil)
        #expect(sanitized["model.layers.1.mlp.switch_mlp.gate_proj.tq_bits"] == nil)
        #expect(sanitized["model.layers.1.mlp.switch_mlp.up_proj.tq_bits"] == nil)
        #expect(sanitized["model.layers.1.mlp.switch_mlp.down_proj.tq_bits"] == nil)
        #expect(sanitized["audio_encoder.input_local_transformer.layers.0.input_layernorm.weight"] == nil)
        #expect(sanitized["encoder.layers.0.self_attn.q_proj.weight"] == nil)
        #expect(sanitized["speech_embeddings.0.weight"] == nil)
        #expect(sanitized["visual.blocks.0.attn.qkv.weight"] == nil)
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

    @Test("CacheCoordinator L2 prefix hit preserves MiMo full/SWA cache topology")
    func cacheCoordinatorL2PrefixHitPreservesHybridFullSWA() throws {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-mimo-v25-l2-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Data(Self.minimalMiMoV25Config.utf8))
        let model = MiMoV2FlashModel(config)
        let promptTokens = [701, 702, 703, 704, 705, 706]
        let cache = model.newCache(parameters: GenerateParameters(maxTokens: 8))
        Self.fill(cache: cache, tokenCount: promptTokens.count)

        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: true,
            enableDiskCache: true,
            pagedBlockSize: 2,
            maxCacheBlocks: 16,
            diskCacheMaxGB: 1.0,
            diskCacheDir: tmp,
            modelKey: "mimo-v25-hybrid-full-swa"))
        coordinator.setPagedIncompatible(cacheRequiresDiskBackedCoordinatorRestore(cache))
        coordinator.storeAfterGeneration(
            promptTokens: promptTokens,
            perLayerData: extractLayerData(from: cache),
            ssmStates: nil,
            cache: cache)

        #expect(coordinator.isPagedIncompatible)
        #expect(coordinator.pagedCache?.stats.allocatedBlocks == 0)

        switch coordinator.fetch(tokens: promptTokens + [707, 708]) {
        case .hit(let matchedTokens, let remainingTokens, let detail, let blocks, _, let diskArrays):
            #expect(matchedTokens == promptTokens.count)
            #expect(remainingTokens == [707, 708])
            #expect(detail == .disk)
            #expect(blocks.isEmpty)
            #expect(diskArrays != nil)
            guard let diskArrays else {
                Issue.record("MiMo disk-backed L2 hit must include serialized cache arrays")
                return
            }
            #expect(TQDiskSerializer.formatVersion(of: diskArrays) == 2)
            #expect(Self.layerKind(diskArrays, 0) == TQDiskSerializer.LayerKind.kv.rawValue)
            #expect(Self.layerKind(diskArrays, 1) == TQDiskSerializer.LayerKind.rotating.rawValue)
            #expect(Self.layerKind(diskArrays, 2) == TQDiskSerializer.LayerKind.rotating.rawValue)
            #expect(Self.layerKind(diskArrays, 3) == TQDiskSerializer.LayerKind.kv.rawValue)

            var restored = model.newCache(parameters: GenerateParameters(maxTokens: 8))
            let restoredTokens = restoreFromDiskArrays(diskArrays, into: &restored)
            #expect(restoredTokens == promptTokens.count)
            #expect(restored[0] is KVCacheSimple)
            #expect(restored[1] is RotatingKVCache)
            #expect(restored[2] is RotatingKVCache)
            #expect(restored[3] is KVCacheSimple)
            #expect(restored[0].state[0].dim(2) == promptTokens.count)
            #expect((restored[1] as? RotatingKVCache)?.offset == promptTokens.count)
            #expect((restored[2] as? RotatingKVCache)?.offset == promptTokens.count)
            #expect(restored[3].state[0].dim(2) == promptTokens.count)
        case .miss:
            Issue.record("MiMo hybrid full/SWA coordinator should hit L2 disk for stored prefix")
        }
    }

    @Test("MiMo exact L2 full hit can trim every restored cache layer before seed token")
    func exactL2FullHitTrimPreservesMiMoPromptBoundary() throws {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-mimo-v25-full-hit-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Data(Self.minimalMiMoV25Config.utf8))
        let model = MiMoV2FlashModel(config)
        let promptTokens = [701, 702, 703, 704, 705, 706]
        let promptCache = model.newCache(parameters: GenerateParameters(maxTokens: 8))
        Self.fill(cache: promptCache, tokenCount: promptTokens.count)

        var diskStoreCache = promptCache.map { $0.copy() }
        maybeQuantizeKVCache(
            cache: &diskStoreCache,
            kvBits: nil,
            kvGroupSize: 64,
            quantizedKVStart: 0,
            kvMode: .turboQuant(keyBits: 4, valueBits: 4))
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: true,
            enableDiskCache: true,
            pagedBlockSize: 2,
            maxCacheBlocks: 16,
            diskCacheMaxGB: 1.0,
            diskCacheDir: tmp,
            modelKey: "mimo-v25-full-hit-tqkv"))
        coordinator.setPagedIncompatible(true)
        coordinator.storeAfterGeneration(
            promptTokens: promptTokens,
            perLayerData: [],
            ssmStates: nil,
            cache: diskStoreCache)

        guard case .hit(let matchedTokens, let remainingTokens, let detail, _, _, let diskArrays) =
            coordinator.fetch(tokens: promptTokens)
        else {
            Issue.record("MiMo exact prompt should hit disk L2")
            return
        }
        #expect(matchedTokens == promptTokens.count)
        #expect(remainingTokens.isEmpty)
        #expect(detail == .disk)

        let arrays = try #require(diskArrays)
        var restored = model.newCache(parameters: GenerateParameters(maxTokens: 8))
        let restoredTokens = restoreFromDiskArrays(arrays, into: &restored)
        #expect(restoredTokens == promptTokens.count)
        #expect(restored.allSatisfy { $0.isTrimmable })

        for layer in restored where layer.isTrimmable {
            _ = layer.trim(1)
        }
        MLX.eval(restored)

        #expect(restored.map(\.offset) == Array(repeating: promptTokens.count - 1, count: 4))
    }

    @Test("MiMo TP plan shards attention, SWA sinks, and SwitchGLU expert projections")
    func mimoTPPlanShardsAttentionSinksAndSwitchGLU() throws {
        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Data(Self.minimalMiMoV25Config.utf8))
        let model = MiMoV2FlashModel(config)
        let group = Group.singleProcessTest(rank: 2, size: 4)

        let replaced = ShardingPlan.mimoV2.apply(to: model, group: group)

        #expect(replaced.contains("model.layers.0.self_attn.q_proj"))
        #expect(replaced.contains("model.layers.0.self_attn.o_proj"))
        #expect(replaced.contains("model.layers.1.mlp.switch_mlp.gate_proj"))
        #expect(replaced.contains("model.layers.1.mlp.switch_mlp.down_proj"))
        #expect(replaced.count >= 24)

        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        let q0 = try #require(leaves["model.layers.0.self_attn.q_proj"] as? AllToShardedLinear)
        let k0 = try #require(leaves["model.layers.0.self_attn.k_proj"] as? AllToShardedLinear)
        let v0 = try #require(leaves["model.layers.0.self_attn.v_proj"] as? AllToShardedLinear)
        let o0 = try #require(leaves["model.layers.0.self_attn.o_proj"] as? ShardedToAllLinear)
        #expect(q0.weight.shape == [8, 32])
        #expect(k0.weight.shape == [4, 32])
        #expect(v0.weight.shape == [4, 32])
        #expect(o0.weight.shape == [32, 8])

        let q1 = try #require(leaves["model.layers.1.self_attn.q_proj"] as? AllToShardedLinear)
        let k1 = try #require(leaves["model.layers.1.self_attn.k_proj"] as? AllToShardedLinear)
        let v1 = try #require(leaves["model.layers.1.self_attn.v_proj"] as? AllToShardedLinear)
        let o1 = try #require(leaves["model.layers.1.self_attn.o_proj"] as? ShardedToAllLinear)
        #expect(q1.weight.shape == [8, 32])
        #expect(k1.weight.shape == [8, 32])
        #expect(v1.weight.shape == [8, 32])
        #expect(o1.weight.shape == [32, 8])

        let gate = try #require(leaves["model.layers.1.mlp.switch_mlp.gate_proj"] as? AllToShardedSwitchLinear)
        let up = try #require(leaves["model.layers.1.mlp.switch_mlp.up_proj"] as? AllToShardedSwitchLinear)
        let down = try #require(leaves["model.layers.1.mlp.switch_mlp.down_proj"] as? ShardedToAllSwitchLinear)
        #expect(gate.weight.shape == [4, 4, 32])
        #expect(up.weight.shape == [4, 4, 32])
        #expect(down.weight.shape == [4, 32, 4])

        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        #expect(parameters["model.layers.0.self_attn.attention_sink_bias"]?.shape == [2])
        #expect(parameters["model.layers.1.self_attn.attention_sink_bias"]?.shape == [2])
    }

    @Test("MiMo TP plan preserves quantized SwitchGLU expert projections")
    func mimoTPPlanPreservesQuantizedSwitchGLU() throws {
        let root = QuantizedSwitchProbeRoot()
        let plan = ShardingPlan(directives: [
            "gate": .allToSharded(segments: 1),
            "up": .allToSharded(segments: 1),
            "down": .shardedToAll(segments: 1),
        ])
        let group = Group.singleProcessTest(rank: 2, size: 4)
        let replaced = plan.apply(to: root, group: group)

        #expect(replaced.contains("gate"))
        #expect(replaced.contains("up"))
        #expect(replaced.contains("down"))

        let leaves = Dictionary(uniqueKeysWithValues: root.leafModules().flattened())
        let shardedGate = try #require(leaves["gate"] as? QuantizedAllToShardedSwitchLinear)
        let shardedUp = try #require(leaves["up"] as? QuantizedAllToShardedSwitchLinear)
        let shardedDown = try #require(leaves["down"] as? QuantizedShardedToAllSwitchLinear)

        #expect(shardedGate.weight.shape == [4, 32, 4])
        #expect(shardedGate.scales.shape == [4, 32, 1])
        #expect(shardedGate.biases?.shape == [4, 32, 1])
        #expect(shardedGate.groupSize == 32)
        #expect(shardedGate.bits == 4)
        #expect(shardedGate.mode == .affine)

        #expect(shardedUp.weight.shape == [4, 32, 4])
        #expect(shardedUp.scales.shape == [4, 32, 1])
        #expect(shardedDown.weight.shape == [4, 32, 4])
        #expect(shardedDown.scales.shape == [4, 32, 1])
        #expect(shardedDown.biases?.shape == [4, 32, 1])
        #expect(shardedDown.groupSize == 32)
        #expect(shardedDown.bits == 4)
        #expect(shardedDown.mode == .affine)
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

private final class QuantizedSwitchProbeRoot: Module {
    @ModuleInfo(key: "gate") var gate: SwitchLinear
    @ModuleInfo(key: "up") var up: SwitchLinear
    @ModuleInfo(key: "down") var down: SwitchLinear

    override init() {
        _gate.wrappedValue = QuantizedSwitchLinear(
            SwitchLinear(inputDims: 32, outputDims: 128, numExperts: 4, bias: false),
            groupSize: 32,
            bits: 4,
            mode: .affine)
        _up.wrappedValue = QuantizedSwitchLinear(
            SwitchLinear(inputDims: 32, outputDims: 128, numExperts: 4, bias: false),
            groupSize: 32,
            bits: 4,
            mode: .affine)
        _down.wrappedValue = QuantizedSwitchLinear(
            SwitchLinear(inputDims: 128, outputDims: 32, numExperts: 4, bias: false),
            groupSize: 32,
            bits: 4,
            mode: .affine)
        super.init()
    }
}
