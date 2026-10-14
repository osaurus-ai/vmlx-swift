import Cmlx
import Foundation
import MLX
import MLXLMCommon
import MLXNN
@testable import MLXVLM
import Testing

@Suite("GLM bounded indexed cache", .serialized)
struct Glm5NextBoundedCacheTests {
    private static let metalFixture: Void = {
        if let path = ProcessInfo.processInfo.environment["VMLX_TEST_METALLIB_PATH"] {
            precondition(mlx_metal_set_metallib_path(path) == 0)
        }
    }()
    init() { _ = Self.metalFixture }

    private func attention(topK: Int = 4, pool: Int = 4) throws -> Glm5NextSparseAttention {
        let json = Self.tinyJSON.replacingOccurrences(of: "\"index_topk\":2048", with: "\"index_topk\":\(topK)")
            .replacingOccurrences(of: "\"index_kpool\":4", with: "\"index_kpool\":\(pool)")
        let config = try JSONDecoder().decode(Glm5NextConfiguration.self, from: Data(json.utf8))
        return Glm5NextSparseAttention(config.textConfig)
    }

    private func hidden(_ count: Int) -> MLXArray {
        let phase = arange(count * 64, dtype: .float32).reshaped(1, count, 64)
        return sin(phase * 0.137) + cos(phase * 0.071)
    }

    private func close(_ a: MLXArray, _ b: MLXArray, tolerance: Float = 2e-5) {
        let error = abs(a - b).max().item(Float.self)
        #expect(error < tolerance, "maxabs=\(error)")
    }

    @Test("cap keeps indexer history before eviction in both MLA representations", arguments: [true, false])
    func beforeEviction(absorbed: Bool) throws {
        let module = try attention()
        let x = hidden(5)
        let bounded = Glm5NextIndexedKVCache(absorbed: absorbed, maxSize: 32, poolSize: 4)
        let full = Glm5NextIndexedKVCache(absorbed: absorbed)
        close(try module(x[0..., ..<4], cache: bounded), try module(x[0..., ..<4], cache: full))
        close(try module(x[0..., 4..<5], cache: bounded), try module(x[0..., 4..<5], cache: full))
        #expect(bounded.indexerPacked?.dim(1) == 5)
    }

    @Test("capped chunked prefill matches single token appends through repeated eviction", arguments: [
        (7, 4, 4), (9, 4, 4), (3, 4, 4), (7, 64, 4), (7, 8, 8),
    ])
    func chunking(cap: Int, topK: Int, pool: Int) throws {
        let module = try attention(topK: topK, pool: pool)
        let x = hidden(37)
        let single = Glm5NextIndexedKVCache(maxSize: cap, poolSize: pool)
        let chunks = Glm5NextIndexedKVCache(maxSize: cap, poolSize: pool)
        var reference: [MLXArray] = []
        for i in 0..<37 { reference.append(try module(x[0..., i..<(i + 1)], cache: single)) }
        var output: [MLXArray] = []
        var start = 0
        for count in [4, 3, 9, 1, 12, 8] {
            output.append(try module(x[0..., start..<(start + count)], cache: chunks))
            start += count
            #expect(chunks.state[0].dim(2) <= cap)
            #expect(chunks.state[2].dim(1) <= cap + 2 * (pool - 1))
        }
        close(concatenated(reference, axis: 1), concatenated(output, axis: 1), tolerance: 1e-4)
    }

    @Test("original learned pool membership survives nonaligned windows and the prefix gap")
    func originalPools() throws {
        let module = try attention(topK: 8, pool: 8)
        let x = hidden(41)
        let bounded = Glm5NextIndexedKVCache(maxSize: 7, poolSize: 8)
        let full = Glm5NextIndexedKVCache()
        for i in 0..<41 {
            _ = try module(x[0..., i..<(i + 1)], cache: bounded)
            _ = try module(x[0..., i..<(i + 1)], cache: full)
            let boundedPools = try module.indexer.pooledStates(packed: bounded.indexerPacked!)
            let fullPools = try module.indexer.pooledStates(packed: full.indexerPacked!)
            let positions = bounded.bounded!.packedPositions.array.asArray(Int32.self)
            let poolIDs = stride(from: 0, to: positions.count, by: 8).map { positions[$0] / 8 }
            close(boundedPools.keys, take(fullPools.keys, MLXArray(poolIDs), axis: 1))
        }
    }

    @Test("copy, in-memory disk-state restore, malformed restore, and trim preserve companion state")
    func persistence() throws {
        let module = try attention()
        let x = hidden(26)
        let source = Glm5NextIndexedKVCache(maxSize: 7, poolSize: 4)
        _ = try module(x[0..., ..<4], cache: source)
        let early = source.copy() as! Glm5NextIndexedKVCache
        #expect(early.trim(2) == 2)
        #expect(source.offset == 4)
        #expect(early.offset == 2)
        #expect(early.indexerPacked?.dim(1) == 2)
        _ = try module(x[0..., 4..<24], cache: source)
        _ = try module(x[0..., 24..<25], cache: source)
        let copy = source.copy() as! Glm5NextIndexedKVCache
        let restored = Glm5NextIndexedKVCache(maxSize: 7, poolSize: 4)
        #expect(restored.restoreDiskCacheState(source.state, metadata: source.metaState, offset: source.offset))
        let wrongCap = Glm5NextIndexedKVCache(maxSize: 8, poolSize: 4)
        #expect(!wrongCap.restoreDiskCacheState(source.state, metadata: source.metaState, offset: source.offset))
        #expect(!restored.restoreDiskCacheState(Array(source.state.prefix(2)), metadata: source.metaState, offset: source.offset))
        #expect(!copy.isTrimmable)
        #expect(copy.trim(1) == 0)
        let expected = try module(x[0..., 25..<26], cache: source)
        close(expected, try module(x[0..., 25..<26], cache: copy))
        close(expected, try module(x[0..., 25..<26], cache: restored))
    }
    @Test("factory preserves indexed companion under a finite cap")
    func factory() throws {
        let config = try JSONDecoder().decode(Glm5NextConfiguration.self, from: Data(Self.tinyJSON.utf8))
        let model = try Glm5Next(config, requesting: [.text])
        let caches = model.newCache(parameters: GenerateParameters(maxKVSize: 7))
        #expect(caches[1] is Glm5NextIndexedKVCache)
        #expect(caches[1].maxSize == 7)
        #expect(caches[0] is MambaCache)
    }

    @Test("bounded storage retains native dtype and generic state round trips", arguments: [DType.bfloat16, .float16])
    func nativeStorage(dtype: DType) {
        let cache = Glm5NextIndexedKVCache(maxSize: 7, poolSize: 4)
        for start in stride(from: 0, to: 24, by: 3) {
            let rows = arange(start, start + 3, dtype: .int32).asType(dtype)
            _ = cache.updateIndexer(repeated(rows.reshaped(1, 3, 1), count: 5, axis: 2))
            _ = cache.update(keys: rows.reshaped(1, 1, 3, 1), values: rows.reshaped(1, 1, 3, 1))
        }
        #expect(cache.state.allSatisfy { $0.dtype == dtype })
        for metadataFirst in [false, true] {
            let restored = Glm5NextIndexedKVCache(maxSize: 7, poolSize: 4)
            if metadataFirst { restored.metaState = cache.metaState; restored.state = cache.state }
            else { restored.state = cache.state; restored.metaState = cache.metaState }
            #expect(restored.offset == cache.offset)
            for (a, b) in zip(restored.state, cache.state) { close(a, b) }
        }
    }

    @Test("gather and masked attention agree with uncapped controls and after repeated eviction")
    func gatherMaskControls() throws {
        let previous = Glm5NextIndexerRuntime.gatherSelected
        defer { Glm5NextIndexerRuntime.gatherSelected = previous }
        let module = try attention()
        let x = hidden(29)
        for cap in [nil, 7] as [Int?] {
            let gatherCache = Glm5NextIndexedKVCache(absorbed: true, maxSize: cap, poolSize: 4)
            let maskCache = Glm5NextIndexedKVCache(absorbed: true, maxSize: cap, poolSize: 4)
            var start = 0
            for count in [4, 3, 1, 8, 1, 12] {
                let chunk = x[0..., start..<(start + count)]
                Glm5NextIndexerRuntime.gatherSelected = true
                let gathered = try module(chunk, cache: gatherCache)
                Glm5NextIndexerRuntime.gatherSelected = false
                let masked = try module(chunk, cache: maskCache)
                close(gathered, masked, tolerance: 1e-4)
                start += count
            }
        }
    }

    @Test("nonpositive finite caps fail explicitly before attention", arguments: [0, -1])
    func invalidCap(cap: Int) throws {
        let module = try attention()
        let cache = Glm5NextIndexedKVCache(maxSize: cap, poolSize: 4)
        #expect(throws: Glm5NextDecoderUnavailable.self) {
            _ = try module(hidden(1), cache: cache)
        }
    }

    @Test("uncapped chunk arithmetic control")
    func uncappedChunkControl() throws {
        let module = try attention()
        let x = hidden(37)
        let single = Glm5NextIndexedKVCache()
        let chunks = Glm5NextIndexedKVCache()
        var a: [MLXArray] = [], b: [MLXArray] = []
        for i in 0..<37 { a.append(try module(x[0..., i..<(i + 1)], cache: single)) }
        var start = 0
        for count in [4, 3, 9, 1, 12, 8] {
            b.append(try module(x[0..., start..<(start + count)], cache: chunks)); start += count
        }
        let error = abs(concatenated(a, axis: 1) - concatenated(b, axis: 1)).max().item(Float.self)
        print("GLM_UNCAPPED_CHUNK_CONTROL maxabs=\(error)")
        close(concatenated(a, axis: 1), concatenated(b, axis: 1), tolerance: 1e-4)
    }

    @Test("exact selected positions and windows match an independent scalar oracle", arguments: [3, 7, 9])
    func selectedPositionOracle(cap: Int) throws {
        let indexer = Glm5NextIndexer(numHeads: 1, headDim: 2, topK: 4, poolSize: 4,
            poolCompressed: true, alwaysSelectTail: true, hiddenSize: 2, qLoraRank: 2)
        indexer.update(parameters: ModuleParameters.unflattened([
            "weights_proj.weight": MLXArray.ones([1, 2]),
            "index_kpool_compress_ape": MLXArray.zeros([4, 2]),
        ]))
        let cache = Glm5NextBoundedCache(capacity: cap, poolSize: 4)
        let keep = min(4, cap - 1)
        var offset = 0
        for count in [4, 3, 9, 1, 12, 8] + Array(repeating: 1, count: 14) {
            // Equal channel keys increase strictly with absolute position; zero
            // gates/APE make each complete pool's score strictly increase too.
            let key = arange(offset, offset + count, dtype: .int32).asType(.float32).reshaped(1, count, 1)
            let packedNew = concatenated([key, key, MLXArray.zeros([1, count, 2]), MLXArray.ones([1, count, 1])], axis: 2)
            let packed = cache.updateIndexer(packedNew)
            let selected = try indexer.selectTopK(packed: packed,
                queries: MLXArray.ones([1, count, 1, 2]), hidden: MLXArray.ones([1, count, 2]),
                queryOffset: offset, bounded: cache).asArray(Int32.self)
            let stored = cache.attentionPositions.array.asArray(Int32.self).map(Int.init)
            let visibility = cache.visibility(positions: cache.attentionPositions, queryOffset: offset, count: count).asArray(Bool.self)
            for row in 0..<count {
                let query = offset + row
                let allowed: (Int) -> Bool = { $0 <= query && ($0 < keep || $0 > query - (cap - keep)) }
                let completePools = (0..<((query + 1) / 4)).filter { pool in (pool * 4..<(pool * 4 + 4)).contains(where: allowed) }
                var expected: [Int] = []
                if let best = completePools.last { expected += (best * 4..<(best * 4 + 4)).filter(allowed) }
                let tailCount = (query + 1) % 4
                expected += ((query + 1 - tailCount)..<(query + 1)).filter(allowed)
                let actual = selected[(row * 7)..<((row + 1) * 7)].filter { $0 >= 0 }.map { stored[Int($0)] }
                #expect(Set(actual) == Set(expected), "cap=\(cap) query=\(query) actual=\(actual) expected=\(expected)")
                #expect(Array(visibility[(row * stored.count)..<((row + 1) * stored.count)]) == stored.map(allowed))
            }
            let (storageKeys, _) = cache.update(keys: key.reshaped(1, 1, count, 1), values: key.reshaped(1, 1, count, 1))
            let mapped = cache.storageIndices(MLXArray(selected, [1, count, 7])).asArray(Int32.self)
            let storedAbsolute = storageKeys.reshaped(-1).asArray(Float.self).map(Int.init)
            for i in selected.indices where selected[i] >= 0 {
                #expect(storedAbsolute[Int(mapped[i])] == stored[Int(selected[i])])
            }
            offset += count
        }
    }

    @Test("diagnostic 64k cache append cost")
    func appendCostProbe() {
        guard ProcessInfo.processInfo.environment["VMLX_GLM_CACHE_APPEND_PROBE"] == "1" else { return }
        let rows = 65_536
        let keys = MLXArray.ones([1, 1, rows, 512], dtype: .bfloat16)
        let values = MLXArray.ones([1, 1, rows, 1], dtype: .bfloat16)
        let packed = MLXArray.ones([1, rows, 257], dtype: .bfloat16)
        let nextKeys = MLXArray.ones([1, 1, 1, 512], dtype: .bfloat16)
        let nextValues = MLXArray.ones([1, 1, 1, 1], dtype: .bfloat16)
        let nextPacked = MLXArray.ones([1, 1, 257], dtype: .bfloat16)
        let bounded = Glm5NextIndexedKVCache(maxSize: rows, poolSize: 4)
        let indexed = Glm5NextIndexedKVCache()
        let rotating = RotatingKVCache(maxSize: rows, keep: 4)
        for cache in [bounded, indexed, rotating] as [any KVCache] {
            if let indexedCache = cache as? Glm5NextIndexedKVCache { _ = indexedCache.updateIndexer(packed) }
            _ = cache.update(keys: keys, values: values)
            eval(cache.innerState())
        }
        func run(_ cache: any KVCache, label: String) {
            func append() {
                if let indexedCache = cache as? Glm5NextIndexedKVCache { _ = indexedCache.updateIndexer(nextPacked) }
                _ = cache.update(keys: nextKeys, values: nextValues)
                eval(cache.innerState())
            }
            for _ in 0..<4 { append() }
            let start = DispatchTime.now().uptimeNanoseconds
            for _ in 0..<20 { append() }
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000 / 20
            print("GLM_CACHE_APPEND label=\(label) rows=\(rows) rank=512 dtype=bf16 iterations=20 ms_per_append=\(elapsed)")
        }
        run(rotating, label: "rotating_kv_only")
        run(indexed, label: "unbounded_indexed")
        run(bounded, label: "bounded_indexed")
    }

    @Test("serialized safetensors disk round trip continues a wrapped indexed cache")
    func actualDiskRoundTrip() throws {
        let module = try attention()
        let x = hidden(29)
        let source = Glm5NextIndexedKVCache(maxSize: 7, poolSize: 4)
        _ = try module(x[0..., ..<24], cache: source)
        _ = try module(x[0..., 24..<25], cache: source)
        eval(source.innerState())
        let payload = TQDiskSerializer.serialize(cache: [source])
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("glm-bounded-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("cache.safetensors")
        try save(arrays: payload, url: file)
        let reloaded = try loadArrays(url: file)
        var restored: [any KVCache] = [Glm5NextIndexedKVCache(maxSize: 7, poolSize: 4)]
        #expect(restoreFromDiskArrays(reloaded, into: &restored, requirePromptBoundary: true) == 25)
        #expect(restored[0].offset == 25)
        #expect(restored[0].maxSize == 7)
        let expected = try module(x[0..., 25..<29], cache: source)
        let actual = try module(x[0..., 25..<29], cache: restored[0])
        let error = abs(expected - actual).max().item(Float.self)
        print("GLM_BOUNDED_DISK_ROUNDTRIP restored=25 continuation=4 maxabs=\(error)")
        #expect(error == 0)
    }

    static let tinyJSON = #"""
        {"model_type":"glm5_next","image_token_id":9,"video_token_id":10,
         "text_config":{"model_type":"glm5_next_text","hidden_size":64,
           "num_hidden_layers":4,"intermediate_size":128,"num_attention_heads":4,
           "num_key_value_heads":4,"vocab_size":128,"rms_norm_eps":1e-05,
           "max_position_embeddings":4096,"kv_lora_rank":16,"q_lora_rank":32,
           "qk_nope_head_dim":16,"qk_rope_head_dim":0,"v_head_dim":16,"mla_use_nope":true,
           "n_routed_experts":8,"n_shared_experts":1,"num_experts_per_tok":2,
           "moe_intermediate_size":32,"first_k_dense_replace":2,"scoring_func":"sigmoid",
           "topk_method":"noaux_tc","routed_scaling_factor":2.5,"norm_topk_prob":true,
           "n_group":1,"topk_group":1,
           "mhc":true,"hc_mult":4,"hc_sinkhorn_iters":20,"hc_eps":1e-06,
           "index_head_dim":16,"index_n_heads":2,"index_topk":2048,"index_kpool":4,
           "index_kpool_compress":true,"index_kpool_always_select_tail":true,
           "num_nextn_predict_layers":1,"swiglu_limit":10.0,"tie_word_embeddings":false,
           "linear_attn_config":{"num_heads":4,"gate_lower_bound":-5.0,"head_dim":16,
             "short_conv_kernel_size":4,"kda_layers":[0,2,3],"full_attn_layers":[1]},
           "layer_types":["linear_attention","deepseek_sparse_attention","linear_attention",
             "linear_attention"],
           "mlp_layer_types":["dense","dense","sparse","sparse"]},
         "vision_config":{"model_type":"glm5_next_vision","depth":2,"hidden_size":32,
           "intermediate_size":64,"out_hidden_size":64,"num_heads":2,"in_channels":3,
           "image_size":56,"patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2,
           "rms_norm_eps":1e-05,"projection_intermediate_size":128,"swiglu_limit":10.0}}
        """#


}
