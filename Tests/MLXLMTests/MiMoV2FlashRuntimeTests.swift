// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("MiMo V2.5 runtime and cache contracts", .serialized)
struct MiMoV2FlashRuntimeTests {
    private static func configJSON(modelType: String = "mimo_v2") -> Data {
        """
        {
          "model_type": "\(modelType)",
          "vocab_size": 128,
          "hidden_size": 8,
          "intermediate_size": 16,
          "moe_intermediate_size": 16,
          "num_hidden_layers": 4,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "swa_num_attention_heads": 2,
          "swa_num_key_value_heads": 2,
          "head_dim": 4,
          "v_head_dim": 2,
          "swa_head_dim": 4,
          "swa_v_head_dim": 2,
          "hybrid_layer_pattern": [0, 1, 1, 0],
          "moe_layer_freq": [0, 1, 1, 0],
          "add_full_attention_sink_bias": false,
          "add_swa_attention_sink_bias": true,
          "sliding_window": 128,
          "attention_value_scale": 0.707,
          "partial_rotary_factor": 0.334,
          "rope_theta": 10000000,
          "swa_rope_theta": 10000,
          "n_routed_experts": 4,
          "num_experts_per_tok": 2,
          "n_group": 1,
          "topk_group": 1,
          "norm_topk_prob": true,
          "topk_method": "noaux_tc",
          "scoring_func": "sigmoid",
          "max_position_embeddings": 1024,
          "layernorm_epsilon": 1e-5
        }
        """.data(using: .utf8)!
    }

    @Test("mimo_v2 source config decodes and registry dispatches native model")
    func registryRecognizesMimoV2SourceModelType() throws {
        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Self.configJSON())
        #expect(config.modelType == "mimo_v2")

        let registryPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Libraries/MLXLLM/LLMModelFactory.swift")
        let source = try String(contentsOf: registryPath, encoding: .utf8)
        #expect(
            source.contains(
                #""mimo_v2": create(MiMoV2FlashConfiguration.self, MiMoV2FlashModel.init)"#))
    }

    @Test("mimo_v2 autodetects think XML reasoning and XML function tools")
    func parserAutodetectMatchesMiMoTemplate() {
        #expect(reasoningStampFromModelType("mimo_v2") == "think_xml")
        #expect(ReasoningParser.fromCapabilityName(reasoningStampFromModelType("mimo_v2")) != nil)
        #expect(ToolCallFormat.infer(from: "mimo_v2") == .xmlFunction)
        #expect(ToolCallFormat.fromCapabilityName("xml_function") == .xmlFunction)
    }

    @Test("per-layer KV heads and cache topology follow full/SWA pattern")
    func kvHeadsAndCacheTopologyFollowHybridPattern() throws {
        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Self.configJSON())

        #expect((0 ..< config.hiddenLayers).map { config.kvHeadsForLayer($0) } == [1, 2, 2, 1])
        #expect((0 ..< config.hiddenLayers).map { config.isSlidingLayer($0) }
            == [false, true, true, false])
        #expect(config.slidingWindowSize == 128)
    }

    @Test("fused qkv projection splits by full and SWA source row contract")
    func sanitizeSplitsFusedQKVByLayerKind() throws {
        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Self.configJSON())

        #expect(config.qkvProjectionRows(layerIndex: 0).q == 8)
        #expect(config.qkvProjectionRows(layerIndex: 0).k == 4)
        #expect(config.qkvProjectionRows(layerIndex: 0).v == 2)
        #expect(config.qkvProjectionRows(layerIndex: 1).q == 8)
        #expect(config.qkvProjectionRows(layerIndex: 1).k == 8)
        #expect(config.qkvProjectionRows(layerIndex: 1).v == 4)
    }

    @Test("cache topology exposes full KV plus rotating SWA layers")
    func cacheTopologyExposesFullKVPlusRotatingSWA() throws {
        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Self.configJSON())
        #expect(config.hybridLayerPattern == [0, 1, 1, 0])

        let source = try String(contentsOfFile: #filePath)
        let modelSourcePath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Libraries/MLXLLM/Models/MiMoV2Flash.swift")
        let modelSource = try String(contentsOf: modelSourcePath, encoding: .utf8)
        #expect(source.contains("ModelCacheTopologySnapshot(cache: cache)"))
        #expect(modelSource.contains("public func newCache(parameters: GenerateParameters?)"))
        #expect(modelSource.contains("if configuration.isSlidingLayer(layerIndex)"))
        #expect(modelSource.contains("RotatingKVCache(maxSize: configuration.slidingWindowSize)"))
        #expect(modelSource.contains("KVCacheSimple()"))

        let cache: [KVCache] = [
            KVCacheSimple(),
            RotatingKVCache(maxSize: config.slidingWindowSize),
            RotatingKVCache(maxSize: config.slidingWindowSize),
            KVCacheSimple(),
        ]
        let topology = ModelCacheTopologySnapshot(cache: cache)
        #expect(topology.layerCount == 4)
        #expect(topology.kvLayerCount == 2)
        #expect(topology.rotatingKVLayerCount == 2)
        #expect(!topology.requiresSSMCompanionState)
        #expect(topology.requiresDiskBackedCoordinatorRestore)
        #expect(topology.topologyTags.contains("restore=disk-backed"))
    }

    @Test("TurboQuant source contract compresses only full-attention KV layers")
    func turboQuantSourceContractOnlyWrapsFullAttentionKVLayers() throws {
        let kvCacheSourcePath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Libraries/MLXLMCommon/KVCache.swift")
        let source = try String(contentsOf: kvCacheSourcePath, encoding: .utf8)
        #expect(source.contains("let firstSimple = cache.first { $0 is KVCacheSimple }"))
        #expect(source.contains("if let simpleCache = cache[i] as? KVCacheSimple"))
        #expect(source.contains("TurboQuantKVCache.fromSimpleCache"))
        // The skip list gained `ZayaMoEPlaceholderCache` and the comment wrapped, so the old
        // one-line literal stopped matching text that is present and correct. Match against
        // whitespace-collapsed source, and name the types individually so ADDING a skipped cache
        // does not break the test while REMOVING one still does.
        let flat = source.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        for skipped in [
            "RotatingKVCache", "DeepseekV4Cache", "MambaCache", "CacheList",
            "ZayaMoEPlaceholderCache",
        ] {
            #expect(
                flat.contains("\(skipped),") || flat.contains("\(skipped):"),
                "\(skipped) must stay in the TurboQuant skip list")
        }
        #expect(flat.contains("ZayaMoEPlaceholderCache: skip."))

        let topology = ModelCacheTopologySnapshot(cache: [
            TurboQuantKVCache(),
            RotatingKVCache(maxSize: 128),
            RotatingKVCache(maxSize: 128),
            TurboQuantKVCache(),
        ])
        #expect(topology.turboQuantKVLayerCount == 2)
        #expect(topology.rotatingKVLayerCount == 2)
        #expect(topology.requiresDiskBackedCoordinatorRestore)
    }

    @Test("L2 disk serializer source supports TurboQuant full KV and rotating SWA")
    func diskSerializerSourceSupportsTurboQuantAndRotatingSWA() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let serializer = try String(
            contentsOf: root.appendingPathComponent("Libraries/MLXLMCommon/Cache/TQDiskSerializer.swift"),
            encoding: .utf8)
        let helpers = try String(
            contentsOf: root.appendingPathComponent("Libraries/MLXLMCommon/Cache/CacheHelpers.swift"),
            encoding: .utf8)

        #expect(serializer.contains("case tq = 1"))
        #expect(serializer.contains("case rotating = 6"))
        #expect(serializer.contains("serializeTQLayer"))
        #expect(serializer.contains("serializeRotatingLayer"))
        #expect(serializer.contains("layer is KVCacheSimple || layer is TurboQuantKVCache"))
        #expect(helpers.contains("case .tq(let comp):"))
        #expect(helpers.contains("restoreTQLayer(comp, into: &cache[i])"))
        #expect(helpers.contains("case .rotating(let comp):"))
        #expect(helpers.contains("restoreRotatingLayer(comp, into: cache[i])"))
    }
}
