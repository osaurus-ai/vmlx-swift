import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

@Suite(.serialized)
struct Spark25Tests {
    static func config(_ overrides: [String: Any] = [:]) throws -> Spark25Configuration {
        var fields: [String: Any] = [
            "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 96,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
            "vocab_size": 128, "rms_norm_eps": 0.000001, "hidden_act": "gelu",
            "gate_attn_act_mode": "sigmoid", "headwise_attn_output_gate": true,
            "attention_bias": false, "mlp_bias": false, "tie_word_embeddings": true,
            "sliding_window": 4,
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            ],
            "rope_parameters": [
                "full_attention": ["rope_theta": 5_000_000.0, "partial_rotary_factor": 0.25],
                "sliding_attention": ["rope_theta": 10_000.0, "partial_rotary_factor": 1.0],
            ],
        ]
        fields.merge(overrides) { _, new in new }
        return try JSONDecoder().decode(
            Spark25Configuration.self, from: JSONSerialization.data(withJSONObject: fields))
    }

    @Test func configRejectsIncompatibleGeometryBeforeConstruction() throws {
        for bad: [String: Any] in [
            ["num_key_value_heads": 3], ["head_dim": 0], ["hidden_act": "silu"],
            ["layer_types": ["full_attention"]], ["layer_types": ["unknown"]],
            ["sliding_window": 0], ["gate_attn_act_mode": "unknown"],
            [
                "rope_parameters": [
                    "full_attention": ["rope_theta": 10000, "partial_rotary_factor": 0.1]
                ]
            ],
        ] {
            #expect(throws: DecodingError.self) { try Self.config(bad) }
        }
    }

    @Test func tensorNamesAndGeometryMatchFusedCheckpoint() throws {
        try MLXMetalTestLock.withLock {
            let model = Spark25Model(try Self.config())
            let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            #expect(parameters.values.reduce(0) { $0 + $1.size } < 250_000)
            #expect(parameters["model.embedding.weight"]?.shape == [128, 64])
            #expect(parameters["model.layers.0.self_attn.q_k_v_proj.weight"]?.shape == [256, 64])
            #expect(parameters["model.layers.0.self_attn.out_proj.weight"]?.shape == [64, 128])
            #expect(parameters["model.layers.0.self_attn.g_proj.weight"]?.shape == [4, 64])
            #expect(parameters["lm_head.weight"] == nil)
            #expect(parameters["model.embed_tokens.weight"] == nil)
            #expect(model.config.rotary(for: .full).partialFactor == 0.25)
            #expect(model.config.rotary(for: .sliding).partialFactor == 1)
        }
    }

    @Test func tiedProjectionAndResidualStayFloatingAfterQuantization() throws {
        try MLXMetalTestLock.withLock {
            let model = Spark25Model(try Self.config())
            quantize(
                model: model, groupSize: 32, bits: 8,
                filter: { path, _ in
                    !path.hasSuffix("g_proj")
                })
            #expect(model.model.embedding is QuantizedEmbedding)
            #expect(model.model.layers[0].attention.qkv is QuantizedLinear)
            #expect(!(model.model.layers[0].attention.gate is QuantizedLinear))
            let tokens = MLXArray([Int32(1), 5, 7]).reshaped(1, 3)
            let cache = model.newCache()
            let logits = model(tokens, cache: cache)
            eval(logits)
            #expect(logits.shape == [1, 3, 128])
            #expect(logits.dtype == .float32)
            #expect(all(isFinite(logits)).item(Bool.self))
            #expect((max(logits) - min(logits)).item(Float.self) > 0)
            let hidden = model.model(tokens, cache: nil)
            #expect(
                all(model.model.embedding.asLinear(hidden) .== model(tokens, cache: nil)).item(
                    Bool.self))
        }
    }

    @Test func mixedCacheUsesOwnOffsetsAndBounds() throws {
        try MLXMetalTestLock.withLock {
            let model = Spark25Model(try Self.config())
            let cache = model.newCache()
            #expect(cache.prefix(3).allSatisfy { $0 is RotatingKVCache })
            #expect(cache[3] is KVCacheSimple)
            for token in 0 ..< 11 {
                let logits = model(MLXArray([Int32(token)]).reshaped(1, 1), cache: cache)
                eval(logits)
                eval(cache)
                #expect(all(isFinite(logits)).item(Bool.self))
                #expect(cache.allSatisfy { $0.offset == token + 1 })
            }
            for layer in cache.prefix(3) {
                #expect(layer.state.allSatisfy { $0.dim(2) <= 4 })
            }
            #expect(cache[3].state.allSatisfy { $0.dim(2) == 11 })
        }
    }
}
