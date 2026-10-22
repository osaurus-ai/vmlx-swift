import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXLLM
@testable import MLXLMCommon

@Suite("Flint architecture and SSD cache contract", .serialized)
struct FlintRuntimeTests {
    static func config() throws -> FlintConfiguration {
        try JSONDecoder().decode(FlintConfiguration.self, from: Data("""
        {"model_type":"flint","hidden_size":32,"num_hidden_layers":3,
        "intermediate_size":64,"moe_intermediate_size":32,"shared_expert_intermediate_size":32,
        "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"num_experts":4,
        "num_experts_per_tok":2,"vocab_size":64,"rms_norm_eps":0.000001,"rope_theta":10000,
        "sliding_window":5,"layer_types":["full_attention","sliding_attention","sliding_attention"],
        "no_rope_layer_indices":[0,2],"mlp_only_layers":[0],"norm_topk_prob":true,
        "router_score_function":"sigmoid","softmax_type":"learnable","tie_word_embeddings":false}
        """.utf8))
    }
    static func model() throws -> FlintModel {
        let model = try FlintModel(config())
        let weights = model.parameters().flattened().map { name, array in
            let count = array.size
            let tensor: MLXArray
            if name.hasSuffix("expert_bias") { tensor = MLXArray.zeros(array.shape) }
            else if name.contains("norm") { tensor = MLXArray.ones(array.shape) }
            else { tensor = (sin(MLXArray(0..<count).asType(.float32) * 0.17) * 0.1).reshaped(array.shape) }
            return (name, tensor.asType(name.hasSuffix("expert_bias") ? .float32 : .bfloat16))
        }
        model.update(parameters: .unflattened(weights))
        return model
    }

    @Test func pythonReferenceLogits() throws {
        try MLXMetalTestLock.withLock {
            let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                .appendingPathComponent("Fixtures/Flint/reference.json")
            struct Reference: Decodable { let logits: [Float]; let shape: [Int] }
            let ref = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))
            let m = try Self.model()
            let actual = m(MLXArray(Array(0..<13)).reshaped(1,13), cache: nil)
            let expected = MLXArray(ref.logits, ref.shape).asType(.bfloat16)
            #expect(arrayEqual(actual, expected).item(Bool.self),
                    "Reference maximum difference: \(abs(actual - expected).max().item(Float.self))")
        }
    }

    @Test func developerRoleAndReasoningHistorySurvive() {
        MLXMetalTestLock.withLock {
            let generator = DefaultMessageGenerator()
            let developer = generator.generate(message: Chat.Message(role: .developer, content: "Use metric units."))
            #expect(developer["role"] as? String == "developer")
            let assistant = generator.generate(message: Chat.Message(role: .assistant, content: "Answer",
                                                                     reasoningContent: "Reason"))
            #expect(assistant["reasoning_content"] as? String == "Reason")
        }
    }

    @Test func configAndTopology() throws {
        try MLXMetalTestLock.withLock {
            let m = try Self.model()
            #expect(m.preservesCheckpointParameterDTypes)
            #expect(m.model.layers[0].attention.rope == nil)
            #expect(m.model.layers[1].attention.rope != nil)
            // Deliberately sliding+NoPE: do not derive positional encoding from layer type.
            #expect(m.model.layers[2].attention.rope == nil)
            let cache = m.newCache(parameters: nil)
            #expect(cache[0] is AttentionSinkKVCache)
            #expect(cache[1].maxSize == 5)
        }
    }

    @Test func chunkedAndDiskRestore() throws {
        try MLXMetalTestLock.withLock {
            let m = try Self.model()
            let tokens = MLXArray(Array(0..<13)).reshaped(1, 13)
            let full = m(tokens, cache: nil)
            let cache = m.newCache(parameters: nil)
            var pieces: [MLXArray] = []
            for range in [0..<4, 4..<9, 9..<13] {
                pieces.append(m(tokens[0..., range], cache: cache))
                eval(pieces.last!, cache)
            }
            let joined = concatenated(pieces, axis: 1)
            #expect(allClose(full, joined, rtol: 0.02, atol: 0.02).item(Bool.self),
                    "BF16 chunk difference: \(abs(full - joined).max().item(Float.self))")
            let serialized = TQDiskSerializer.serialize(cache: cache, preserveStandardKVStorageDType: true)
            var restored = m.newCache(parameters: nil)
            #expect(restoreFromDiskArrays(serialized, into: &restored, requirePromptBoundary: true) == 13)
            #expect(restored[0] is AttentionSinkKVCache)
            let next = MLXArray([Int32(14)]).reshaped(1, 1)
            let expected = m(next, cache: cache)
            let actual = m(next, cache: restored)
            #expect(arrayEqual(expected, actual).item(Bool.self))
        }
    }

    @Test func sinkCacheNeverPromotes() throws {
        try MLXMetalTestLock.withLock {
            let m = try Self.model()
            var cache = m.newCache(parameters: nil)
            let k = MLXArray.ones([1, 2, 1024, 8], dtype: .bfloat16)
            _ = cache[0].update(keys: k, values: k)
            for mode: KVQuantizationMode in [.affine(bits: 4, groupSize: 32), .turboQuant(keyBits: 4, valueBits: 4), .none] {
                maybeQuantizeKVCache(cache: &cache, kvBits: 4, kvMode: mode)
                #expect(cache[0] is AttentionSinkKVCache)
                #expect(cache[0].copy() is AttentionSinkKVCache)
            }
        }
    }

    @Test func routerBiasSelectsButDoesNotWeight() throws {
        try MLXMetalTestLock.withLock {
            let c = try Self.config(), router = FlintRouter(c)
            router.update(parameters: .unflattened([
                "weight": MLXArray.zeros([4,32], dtype: .bfloat16),
                "expert_bias": MLXArray([Float(0), 1, 2, 0]),
            ]))
            let (indices, weights) = router(MLXArray.ones([1,1,32], dtype: .bfloat16))
            #expect(Set(indices.asArray(Int32.self)) == Set([Int32(1), Int32(2)]))
            #expect(weights.asArray(Float.self) == [0.5, 0.5])
        }
    }

    @Test func batchDecodeMatchesIndependentCaches() throws {
        try MLXMetalTestLock.withLock {
            let m = try Self.model()
            let left = m.newCache(parameters: nil), right = m.newCache(parameters: nil)
            eval(m(MLXArray([Int32(1),2,3]).reshaped(1,3), cache: left), left)
            eval(m(MLXArray([Int32(4),5,6,7,8,9,10]).reshaped(1,7), cache: right), right)
            let lc = left.map { $0.copy() }, rc = right.map { $0.copy() }
            let batch: [KVCache] = left.indices.map { BatchKVCache(slotCaches: [left[$0], right[$0]]) }
            let result = m(MLXArray([Int32(11),12]).reshaped(2,1), cache: batch)
            let expected = concatenated([
                m(MLXArray([Int32(11)]).reshaped(1,1), cache: lc),
                m(MLXArray([Int32(12)]).reshaped(1,1), cache: rc),
            ], axis: 0)
            #expect(allClose(result, expected, rtol: 0.02, atol: 0.02).item(Bool.self))
        }
    }
}
