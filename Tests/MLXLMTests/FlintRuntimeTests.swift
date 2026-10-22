import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXLLM
@testable import MLXLMCommon

private final class UnmarkedCacheIdentityModel: Module, LanguageModel, PrefixCacheInvariantModel {
    var statesBeforeFirstMediaTokenAreMediaIndependent: Bool { true }
    var exactPrefixRestoreIsSuffixIndependent: Bool { true }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray { inputs }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [KVCacheSimple()] }
}

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

    @Test func generationProjectsOnlyLastRowWithoutChangingCache() throws {
        try MLXMetalTestLock.withLock {
            let m = try Self.model()
            let tokens = MLXArray(Array(0..<13)).reshaped(1, 13)
            let fullCache = m.newCache(parameters: nil)
            let generationCache = m.newCache(parameters: nil)
            let full = m(tokens, cache: fullCache)
            let generated = m(LMInput.Text(tokens: tokens), cache: generationCache, state: nil).logits
            eval(full, generated, fullCache, generationCache)
            #expect(generated.shape == [1, 1, m.vocabularySize])
            #expect(allClose(generated, full[0..., (full.dim(1) - 1)..<full.dim(1), 0...], rtol: 0.02, atol: 0.02).item(Bool.self))
            #expect(fullCache.map(\.offset) == generationCache.map(\.offset))
            let next = MLXArray([Int32(14)]).reshaped(1, 1)
            #expect(arrayEqual(m(next, cache: fullCache), m(next, cache: generationCache)).item(Bool.self))
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

    @Test func maximumKVSizeDoesNotChangeFlintStateOrSalt() throws {
        try MLXMetalTestLock.withLock {
            let m = try Self.model()
            let tokens = MLXArray(Array(0..<13)).reshaped(1, 13)
            let input = LMInput(text: LMInput.Text(tokens: tokens))
            let plain = GenerateParameters()
            let capped = GenerateParameters(maxKVSize: 2)
            let a = m.newCache(parameters: plain), b = m.newCache(parameters: capped)
            let left = m(tokens, cache: a), right = m(tokens, cache: b)
            eval(left, right, a, b)
            #expect(arrayEqual(left, right).item(Bool.self))
            #expect(a[0].offset == 13 && b[0].offset == 13)
            #expect(a[0].state[0].dim(2) == 13 && b[0].state[0].dim(2) == 13)
            for (x, y) in zip(a, b) {
                #expect(x.metaState == y.metaState)
                for (xs, ys) in zip(x.state, y.state) {
                    #expect(arrayEqual(xs, ys).item(Bool.self))
                }
            }
            #expect(computeCacheSalt(for: input, parameters: plain, model: m)
                    == computeCacheSalt(for: input, parameters: capped, model: m))
            #expect(capped.maxKVSize == 2, "Identity normalization must not mutate request settings")
            // Unknown models and the legacy API retain conservative identity.
            let unmarked = UnmarkedCacheIdentityModel()
            #expect(!unmarked.maximumKVSizeIsCacheStateInvariant)
            #expect(computeCacheSalt(for: input, parameters: plain, model: unmarked)
                    != computeCacheSalt(for: input, parameters: capped, model: unmarked))
            #expect(computeCacheSalt(for: input, parameters: plain)
                    != computeCacheSalt(for: input, parameters: capped))
            let quantized = GenerateParameters(kvMode: .turboQuant(keyBits: 4, valueBits: 4))
            #expect(computeCacheSalt(for: input, parameters: plain, model: m)
                    != computeCacheSalt(for: input, parameters: quantized, model: m))
        }
    }

    @Test func automaticWindowThresholdKeepsFlintIdentityWithoutChangingPolicy() throws {
        try MLXMetalTestLock.withLock {
            let m = try Self.model()
            let input = LMInput(text: LMInput.Text(tokens: MLXArray([Int32(1), 2, 3])))
            let coordinator = CacheCoordinatorConfig(defaultMaxKVSize: 65536)
            let before = coordinator.resolveKVPolicy(kvMode: .none, maxKVSize: nil,
                                                      promptTokenCount: 131072)
            let after = coordinator.resolveKVPolicy(kvMode: .none, maxKVSize: nil,
                                                     promptTokenCount: 131073)
            #expect(before.maxKVSize == nil)
            #expect(after.maxKVSize == 65536, "Preserve coordinator memory-safety policy")
            let a = GenerateParameters(maxKVSize: before.maxKVSize)
            let b = GenerateParameters(maxKVSize: after.maxKVSize)
            #expect(computeCacheSalt(for: input, parameters: a, model: m)
                    == computeCacheSalt(for: input, parameters: b, model: m))
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
            for suffix: [Int32] in [[14], [14, 15, 16]] {
                var restored = m.newCache(parameters: nil)
                #expect(restoreFromDiskArrays(serialized, into: &restored, requirePromptBoundary: true) == 13)
                #expect(restored[0] is AttentionSinkKVCache)
                #expect(!requiredToolSelectionSkipsDiskBackedRestore(
                    model: m, cache: restored, restorePolicy: .freshRequiredToolSelection))
                let next = MLXArray(suffix).reshaped(1, suffix.count)
                let expected = m(next, cache: cache.map { $0.copy() })
                let actual = m(next, cache: restored)
                #expect(arrayEqual(expected, actual).item(Bool.self))
            }
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

    @Test func ordinaryCacheStillQuantizesAndCannotRestoreIntoSinkCache() {
        MLXMetalTestLock.withLock {
            let sink = AttentionSinkKVCache(headCount: 2, headDimension: 64)
            let ordinary = KVCacheSimple()
            let values = MLXArray.ones([1,2,64,64], dtype: .bfloat16)
            _ = sink.update(keys: values, values: values)
            _ = ordinary.update(keys: values, values: values)
            var mixed: [KVCache] = [sink, ordinary]
            maybeQuantizeKVCache(cache: &mixed, kvBits: 4, kvGroupSize: 32)
            #expect(mixed[0] is AttentionSinkKVCache)
            #expect(mixed[1] is QuantizedKVCache)
            let record = TQDiskSerializer.serialize(cache: [mixed[1]])
            var destination: [KVCache] = [AttentionSinkKVCache(headCount: 2, headDimension: 64)]
            #expect(restoreFromDiskArrays(record, into: &destination, requirePromptBoundary: true) == 0)
            #expect(destination[0].offset == 0)
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
