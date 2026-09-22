// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import MLXRandom
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("MiMo V2.6 mixed quantization and native cache", .serialized)
struct MiMoV26RuntimeTests {
    /// The pinned MLX backend enables TF32 for multi-row F32 GEMM on M5,
    /// while GEMV remains F32. Keep the strict assertion and record this
    /// attributed backend difference; the same matrix MUST also run in a
    /// separate MLX_ENABLE_TF32=0 process, where no known issue is allowed.
    static func expectChunkParity(_ actual: MLXArray, _ expected: MLXArray,
                                  rtol: Double = 1e-4, atol: Double = 1e-4) {
        let check = {
            #expect(allClose(actual, expected, rtol: rtol, atol: atol).item(Bool.self),
                "max absolute chunk difference: \(abs(actual - expected).max().item(Float.self))")
        }
        if ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] != "0" {
            withKnownIssue("Pinned MLX M5 TF32 GEMM versus F32 GEMV; strict separate-process matrix required",
                           isIntermittent: true) { check() }
        } else { check() }
    }
    static func configuration(moe: Bool = false) throws -> Data {
        try JSONSerialization.data(withJSONObject: [
            "model_type": "mimo_v2", "attention_projection_layout": "fused_qkv",
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
            "moe_intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "swa_num_attention_heads": 2, "swa_num_key_value_heads": 2,
            "head_dim": 32, "v_head_dim": 16, "swa_head_dim": 32, "swa_v_head_dim": 16,
            "partial_rotary_factor": 0.5, "sliding_window": 4,
            "hybrid_layer_pattern": [0, 1], "moe_layer_freq": [0, moe ? 1 : 0],
            "n_routed_experts": 4, "num_experts_per_tok": 2,
            "layernorm_epsilon": 1e-6, "attention_value_scale": 0.707,
            "quantization": [
                "bits": 8, "group_size": 64, "mode": "affine",
                "model.layers.0.self_attn.o_proj": ["bits": 8, "group_size": 32, "mode": "affine"],
                "model.layers.1.self_attn.o_proj": ["bits": 8, "group_size": 32, "mode": "affine"],
                "model.layers.1.mlp.switch_mlp.gate_proj": ["bits": 4, "group_size": 32, "mode": "mxfp4"],
                "model.layers.1.mlp.switch_mlp.up_proj": ["bits": 2, "group_size": 64, "mode": "affine"],
            ],
        ])
    }

    static func model(moe: Bool = false) throws -> MiMoV26TextModel {
        try MiMoV26TextModel(JSONDecoder().decode(MiMoV2FlashConfiguration.self, from: configuration(moe: moe)))
    }

    @Test("Fresh representation selects the fused runtime; legacy keeps its own runtime")
    func dispatch() async throws {
        let data = try Self.configuration()
        #expect(MiMoV26Contract.matches(data))
        let fresh = try await LLMTypeRegistry.shared.createModel(configuration: data, modelType: "mimo_v2")
        #expect(fresh is MiMoV26TextModel)
        #expect(!fresh.supportsWholeForwardCompilation)
        var old = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        old.removeValue(forKey: "attention_projection_layout")
        let oldData = try JSONSerialization.data(withJSONObject: old)
        #expect(!MiMoV26Contract.matches(oldData))
        let legacy = try await LLMTypeRegistry.shared.createModel(configuration: oldData, modelType: "mimo_v2")
        #expect(legacy is MiMoV2FlashModel)
    }

    @Test("Fused QKV survives sanitization with original scales and native expert bytes")
    func preserveConvertedWeights() throws {
        let model = try Self.model()
        let names = ["model.layers.0.self_attn.qkv_proj.weight", "model.layers.0.self_attn.qkv_proj.scales",
                     "model.layers.1.mlp.switch_mlp.gate_proj.weight", "lm_head.weight"]
        var weights = Dictionary(uniqueKeysWithValues: names.map { ($0, MLXArray([UInt32(17)])) })
        weights["visual.blocks.0.weight"] = MLXArray([1])
        weights["audio_encoder.projection.fc1.weight"] = MLXArray([1])
        weights["model.mtp.layers.0.weight"] = MLXArray([1])
        #expect(model.requiresExactTensorMmapBuffers)
        for name in weights.keys {
            #expect(model.excludeFromGenericSafetensorsLoad(key: name) == !names.contains(name))
        }
        let sanitized = model.sanitize(weights: weights)
        #expect(Set(sanitized.keys) == Set(names))
        for name in names { #expect(sanitized[name]?.item(UInt32.self) == 17) }
    }

    @Test("Native cache keeps asymmetric K/V in the activation dtype", arguments: [DType.float16, .bfloat16])
    func cacheDtype(_ dtype: DType) throws {
        let model = try Self.model(moe: true)
        model.update(parameters: ModuleParameters.unflattened(
            model.parameters().flattened().map { ($0.0, $0.1.asType(dtype)) }))
        let cache = model.newCache(parameters: nil)
        #expect(cache[0] is KVCacheSimple)
        #expect(cache[1] is RotatingKVCache)
        let logits = model(MLXArray(Array(0..<9)).reshaped(1, 9), cache: cache)
        eval(logits, cache)
        #expect(logits.dtype == dtype)
        for slot in cache {
            #expect(slot.offset == 9)
            #expect(slot.state[0].dtype == dtype)
            #expect(slot.state[1].dtype == dtype)
            #expect(slot.state[0].dim(-1) == 32)
            #expect(slot.state[1].dim(-1) == 16)
        }
        eval(model(MLXArray([9]).reshaped(1, 1), cache: cache))
        #expect(cache.allSatisfy { $0.offset == 10 })
    }

    @Test("Full and split prefill agree across a rotated window and multi-token tail")
    func chunkedCausality() throws {
        MLXRandom.seed(26)
        let model = try Self.model()
        let tokens = MLXArray(Array(0..<15)).reshaped(1, 15)
        let wholeCache = model.newCache(parameters: nil)
        let whole = model(tokens, cache: wholeCache)
        let splitCache = model.newCache(parameters: nil)
        eval(model(tokens[0..., ..<9], cache: splitCache))
        let tail = model(tokens[0..., 9...], cache: splitCache)
        eval(whole, tail)
        #expect(allClose(whole[0..., 9...], tail, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        #expect(splitCache.allSatisfy { $0.offset == 15 })
    }

    @Test("Router correction changes selection but not the selected sigmoid weights")
    func routerMath() throws {
        let config = try JSONDecoder().decode(MiMoV2FlashConfiguration.self, from: Self.configuration(moe: true))
        let gate = MiMoV26Router(config)
        try gate.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray.zeros([4, 64], dtype: .bfloat16),
            "e_score_correction_bias": MLXArray([Float(0), 10, 20, 0]),
        ]), verify: .all)
        let (indices, scores) = gate(MLXArray.ones([1, 1, 64], dtype: .bfloat16))
        eval(indices, scores)
        #expect(Set(indices.asArray(Int32.self)) == Set([Int32(1), 2]))
        #expect(scores.dtype == .float32)
        #expect(allClose(scores, MLXArray.full([1, 1, 2], values: MLXArray(Float(0.5))), atol: 1e-7).item(Bool.self))
    }

    @Test("Single-token and mixed chunk tails preserve sliding-window attention", arguments: [1, 2, 3, 4, 5, 7])
    func singleTokenTail(step: Int) throws {
        MLXRandom.seed(26)
        let model = try Self.model()
        let tokens = MLXArray(Array(0..<13)).reshaped(1, 13)
        let wholeCache = model.newCache(parameters: nil)
        let whole = model(tokens, cache: wholeCache)
        eval(whole)
        let cache = model.newCache(parameters: nil)
        var final: MLXArray?
        for start in stride(from: 0, to: 13, by: step) {
            final = model(tokens[0..., start..<min(start + step, 13)], cache: cache)
            eval(final!, cache)
        }
        let actual = try #require(final)[0, -1]
        Self.expectChunkParity(actual, whole[0, -1])
    }

    @Test("Real safetensors load preserves mixed expert formats and FP32 router parameters")
    func mixedCheckpointRoundTrip() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let data = try Self.configuration(moe: true)
        try data.write(to: directory.appendingPathComponent("config.json"))
        try Data("""
            {"version":2,"weight_format":"mixed_affine_mxfp4","profile":"JANG_2L",
             "quantization":{"method":"mixed","modes":["affine","mxfp4"]}}
            """.utf8).write(to: directory.appendingPathComponent("jang_config.json"))
        let original = try Self.model(moe: true)
        let parameters = original.parameters().flattened().map { key, value in
            let router = key.contains(".mlp.gate.")
            return (key, router ? value + Float(0.001337) : value.asType(.bfloat16))
        }
        try original.update(parameters: ModuleParameters.unflattened(parameters), verify: .all)
        MLXNN.quantize(model: original, filter: { path, module in
            guard module is Quantizable else { return nil }
            if path.hasSuffix("switch_mlp.gate_proj") { return (32, 4, .mxfp4) }
            if path.hasSuffix("switch_mlp.up_proj") { return (64, 2, .affine) }
            if path.hasSuffix("self_attn.o_proj") { return (32, 8, .affine) }
            return (64, 8, .affine)
        })
        let weights = Dictionary(uniqueKeysWithValues: original.parameters().flattened())
        var checkpoint = weights
        checkpoint["visual.blocks.0.weight"] = MLXArray([Float(99)])
        checkpoint["audio_encoder.projection.fc1.weight"] = MLXArray([Float(98)])
        checkpoint["model.mtp.layers.0.weight"] = MLXArray([Float(97)])
        try MLX.save(arrays: checkpoint, url: directory.appendingPathComponent("model.safetensors"))
        let loaded = try Self.model(moe: true)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        try loadWeights(modelDirectory: directory, model: loaded,
                        perLayerQuantization: base.perLayerQuantization,
                        jangConfig: try JangLoader.loadConfig(at: directory))
        let after = Dictionary(uniqueKeysWithValues: loaded.parameters().flattened())
        #expect(Set(after.keys) == Set(weights.keys))
        for (name, before) in weights {
            let value = try #require(after[name])
            #expect(value.dtype == before.dtype, "\(name) changed dtype")
            #expect(arrayEqual(value, before).item(Bool.self), "\(name) changed checkpoint values")
        }
        let loadedModules = Dictionary(uniqueKeysWithValues: loaded.leafModules().flattened())
        for (name, module) in original.leafModules().flattened() {
            guard let before = module as? Quantized else { continue }
            let after = try #require(loadedModules[name] as? Quantized)
            try #require(after.mode == before.mode && after.bits == before.bits
                         && after.groupSize == before.groupSize, "\(name) changed quantization format")
        }
        let tokens = MLXArray([1, 2, 3]).reshaped(1, 3)
        let cache = loaded.newCache(parameters: nil)
        let expected = original(tokens, cache: original.newCache(parameters: nil))
        let actual = loaded(tokens, cache: cache)
        eval(expected, actual, cache)
        #expect(arrayEqual(actual, expected).item(Bool.self))
        #expect(cache.allSatisfy { $0.state.allSatisfy { $0.dtype == .bfloat16 } })
    }
}
