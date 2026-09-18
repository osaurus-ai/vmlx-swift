import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite("Bonsai2 FP16 attention and unchanged recurrent state", .serialized)
struct Bonsai2AttentionPrecisionTests {
    private func tensor(_ shape: [Int], seed: Int = 0) -> MLXArray {
        let count = shape.reduce(1, *)
        return MLXArray((0 ..< count).map { Float(($0 * 17 + seed) % 101 - 50) / 37 }, shape)
    }

    private func close(_ actual: MLXArray, _ expected: MLXArray) {
        #expect(actual.shape == expected.shape)
        #expect(MLX.all(isFinite(actual)).item(Bool.self))
        let error = MLX.max(abs(actual.asType(.float32) - expected.asType(.float32))).item(
            Float.self)
        let scale = MLX.max(abs(expected.asType(.float32))).item(Float.self)
        #expect(error <= 0.002 + 0.003 * scale)
    }

    @Test("F32 norm promotion is real; FP16 attention halves stored KV bytes")
    func coldAndDecode() throws {
        try MLXMetalTestLock.withLock {
            let normalized = MLXFast.rmsNorm(
                tensor([1, 2, 11, 64]).asType(.float16),
                weight: MLXArray.ones([64], dtype: .float32), eps: 1e-6)
            #expect(normalized.dtype == .float32)
            let old = KVCacheSimple()
            let new = KVCacheSimple()
            for length in [11, 1, 3, 1] {
                let q = tensor([1, 4, length, 64], seed: old.offset)
                let k = tensor([1, 2, length, 64], seed: old.offset + 1)
                let v = tensor([1, 2, length, 64], seed: old.offset + 2)
                let mask = old.makeMask(n: length, windowSize: nil, returnArray: true)
                let reference = JangHadamardAttention.attention(
                    queries: q, keys: k, values: v, cache: old, scale: 0.125,
                    mask: mask, enabled: false)
                let actual = JangHadamardAttention.attention(
                    queries: q, keys: k, values: v, cache: new, scale: 0.125,
                    mask: mask, enabled: true)
                MLX.eval(actual, reference, old, new)
                #expect(reference.dtype == .float32 && actual.dtype == .float32)
                #expect(old.state.allSatisfy { $0.dtype == .float32 })
                #expect(new.state.allSatisfy { $0.dtype == .float16 })
                let oldBytes: Int = old.state.reduce(0) { $0 + $1.nbytes }
                let newBytes: Int = new.state.reduce(0) { $0 + $1.nbytes }
                #expect(oldBytes == 2 * newBytes)
                #expect(old.offset == new.offset)
                close(actual, reference)
            }
        }
    }

    @Test("existing simple, rotating and compiled buffers preserve counters and capacity")
    func restoredStorage() throws {
        try MLXMetalTestLock.withLock {
            let simple = KVCacheSimple()
            _ = simple.update(keys: tensor([1, 2, 7, 64]), values: tensor([1, 2, 7, 64], seed: 1))
            let ring = RotatingKVCache(maxSize: 8, keep: 2, step: 8)
            for _ in 0 ..< 13 {
                _ = ring.update(
                    keys: tensor([1, 2, 1, 64]), values: tensor([1, 2, 1, 64], seed: 1))
            }
            let compiled = CompilableKVCache(from: simple, maxLength: 32)
            let compiledRing = CompilableRotatingKVCache(from: ring)
            for cache in [simple, ring, compiled, compiledRing] as [any KVCache] {
                let offset = cache.offset
                let metadata = cache.metaState
                let shapes = cache.innerState().map(\.shape)
                let old = cache.copy()
                MLX.eval(old)
                #expect(JangHadamardAttention.prepareStorage(cache))
                MLX.eval(cache)
                #expect(cache.offset == offset && cache.metaState == metadata)
                #expect(cache.innerState().map(\.shape) == shapes)
                #expect(
                    cache.innerState().filter { $0.ndim == 4 }.allSatisfy { $0.dtype == .float16 })
                #expect(
                    old.innerState().filter { $0.ndim == 4 }.allSatisfy { $0.dtype == .float32 })
                let q = tensor([1, 4, 1, 64])
                let k = tensor([1, 2, 1, 64])
                let v = tensor([1, 2, 1, 64], seed: 1)
                let expected = attentionWithCacheUpdate(
                    queries: q, keys: k, values: v, cache: old, scale: 0.125,
                    mask: old.makeMask(n: 1, windowSize: nil, returnArray: true))
                let actual = JangHadamardAttention.attention(
                    queries: q, keys: k, values: v, cache: cache, scale: 0.125,
                    mask: cache.makeMask(n: 1, windowSize: nil, returnArray: true), enabled: true)
                close(actual, expected)
                #expect(cache.offset == offset + 1)
            }
        }
    }

    @Test("batch owners convert together; explicit quantized storage is not reinterpreted")
    func batchAndQuantized() throws {
        try MLXMetalTestLock.withLock {
            let slots = [KVCacheSimple(), KVCacheSimple()]
            for (i, slot) in slots.enumerated() {
                _ = slot.update(
                    keys: tensor([1, 2, i + 2, 64]), values: tensor([1, 2, i + 2, 64], seed: 1))
            }
            let batch = BatchKVCache(slotCaches: slots)
            let mask = batch.makeMask(n: 1, windowSize: nil, returnArray: true)
            let output = JangHadamardAttention.attention(
                queries: tensor([2, 4, 1, 64]), keys: tensor([2, 2, 1, 64]),
                values: tensor([2, 2, 1, 64], seed: 1), cache: batch, scale: 0.125,
                mask: mask, enabled: true)
            #expect(MLX.all(isFinite(output)).item(Bool.self))
            #expect(slots.map(\.offset) == [3, 4])
            #expect(slots.flatMap(\.state).allSatisfy { $0.dtype == .float16 })
            let quantized = QuantizedKVCache(groupSize: 64, bits: 4)
            #expect(!JangHadamardAttention.prepareStorage(quantized))
            let untouched = KVCacheSimple()
            _ = untouched.update(keys: tensor([1, 2, 2, 64]), values: tensor([1, 2, 2, 64]))
            #expect(
                !JangHadamardAttention.prepareStorage(
                    BatchKVCache(slotCaches: [untouched, quantized])))
            #expect(untouched.state.allSatisfy { $0.dtype == .float32 })
        }
    }

    @Test(
        "actual Hadamard VLM vision prefill and decode keep FP16 KV and F32 recurrence",
        arguments: [false, true])
    func visionAndDecode(packed: Bool) throws {
        try MLXMetalTestLock.withLock {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(
                "bonsai-fp16-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let model = try Qwen35HadamardCacheTests.writeFixture(
                at: root, packed: packed, vision: true)
            #expect(
                JangHadamardAttention.cacheKeyComponent(model: model) == "bonsai-attention-fp16-v1")
            let tokens: [Int32] = [1, 2, 96, 98, 98, 98, 98, 95, 3, 4, 5, 6, 7]
            let input = LMInput(
                text: .init(tokens: MLXArray(tokens, [1, tokens.count])),
                image: .init(pixels: tensor([16, 12]), frames: [THW(1, 4, 4)]))
            let cache = model.newCache(parameters: nil)
            let prepared = try model.prepare(input, cache: cache, windowSize: 3)
            guard case .logits(let first) = prepared else {
                Issue.record("vision route did not return prepared logits")
                return
            }
            #expect(MLX.all(isFinite(first.logits)).item(Bool.self))
            #expect(cache.last?.state.allSatisfy { $0.dtype == .float16 } == true)
            #expect(cache.prefix(3).allSatisfy { $0.state[1].dtype == .float32 })
            for token: Int32 in [19, 20] {
                let output = model(MLXArray([token], [1, 1]), cache: cache)
                #expect(MLX.all(isFinite(output)).item(Bool.self))
            }
            #expect(cache.allSatisfy { $0.offset == tokens.count + 2 })
            #expect(cache.last?.state.allSatisfy { $0.dtype == .float16 } == true)
            #expect(cache.prefix(3).allSatisfy { $0.state[1].dtype == .float32 })
        }
    }

    @Test("compiled replay preserves FP16 buffers and advancing ring/graph offsets")
    func compiledReplay() throws {
        try MLXMetalTestLock.withLock {
            for rotating in [false, true] {
                let seed = RotatingKVCache(maxSize: 8, keep: 2, step: 8)
                _ = seed.update(
                    keys: tensor([1, 2, 7, 64]).asType(.float16),
                    values: tensor([1, 2, 7, 64], seed: 1).asType(.float16))
                nonisolated(unsafe) let cache: any KVCache =
                    rotating
                    ? CompilableRotatingKVCache(from: seed)
                    : CompilableKVCache(from: seed, maxLength: 32)
                let eager = cache.copy()
                MLX.eval(cache, eager)
                let forward: @Sendable ([MLXArray]) -> [MLXArray] = compile(
                    inputs: [cache], outputs: [cache]
                ) { args in
                    CompiledDecodeTrace.withActive {
                        [
                            JangHadamardAttention.attention(
                                queries: args[0], keys: args[1], values: args[2], cache: cache,
                                scale: 0.125, mask: .none, enabled: true)
                        ]
                    }
                }
                for step in 0 ..< 5 {
                    let q = tensor([1, 4, 1, 64], seed: step)
                    let k = tensor([1, 2, 1, 64], seed: step + 1)
                    let v = tensor([1, 2, 1, 64], seed: step + 2)
                    let expected = JangHadamardAttention.attention(
                        queries: q, keys: k, values: v, cache: eager,
                        scale: 0.125, mask: .none, enabled: true)
                    let actual = forward([q, k, v])[0]
                    close(actual, expected)
                    #expect(cache.offset == 8 + step && eager.offset == cache.offset)
                    #expect(
                        cache.innerState().filter { $0.ndim == 4 }.allSatisfy {
                            $0.dtype == .float16
                        })
                }
            }
        }
    }

    @Test("both container caching entrypoints isolate the new numerical policy")
    func containerCacheNamespace() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "bonsai-namespace-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let container = try await MLXMetalTestLock.withLock {
            let model = try Qwen35HadamardCacheTests.writeFixture(at: root, packed: false)
            let processor = TestInputProcessor()
            return ModelContainer(
                context: ModelContext(
                    configuration: ModelConfiguration(id: "not-a-model-name-heuristic"),
                    model: model,
                    processor: processor, tokenizer: processor.tokenizer))
        }
        let config = CacheCoordinatorConfig(
            usePagedCache: false, enableDiskCache: false, modelKey: "same-bundle-and-media")
        container.enableCaching(config: config)
        let syncKey = try #require(container.cacheCoordinator?.config.modelKey)
        #expect(syncKey == "same-bundle-and-media|bonsai-attention-fp16-v1")
        await container.enableCachingAsync(config: config)
        #expect(container.cacheCoordinator?.config.modelKey == syncKey)
    }
}
