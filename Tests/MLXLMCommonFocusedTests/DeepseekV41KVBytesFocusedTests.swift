// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 retained KV bytes at the real cache geometry, against the Python
// reference (jang_tools/dsv41/model.py `QuantGrowBuffer`, dsv4_1/mlx_model.py
// `DeepseekV41Cache.nbytes`): only the kv-source layers [2, 8, 14, 20] hold
// compressed-KV (512-wide) and index-key (128-wide) pools, 4-bit from the first row,
// plus one 128-row sliding window per layer.
//
// The caches are built by the real `DeepseekV4Model.makeCaches` and filled through
// the cache API the way a prefill fills them (no model weights, CPU device only).

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("DeepSeek-V4.1 KV bytes per token (real geometry, CPU)", .serialized)
struct DeepseekV41KVBytesFocusedTests {

    static let bundleConfig = URL(fileURLWithPath:
        "/Users/eric/models/DeepSeek-V4.1-Flash-1017-JANG-Q2Q3-IMATRIX/config.json")

    /// The bundle's text config, or the same geometry inline when it is absent.
    static func config() throws -> DeepseekV4Configuration {
        if let data = try? Data(contentsOf: bundleConfig) {
            return try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
        }
        let ratios = [0, 0] + Array(repeating: 2, count: 18) + Array(repeating: 1, count: 20) + [0, 0, 0]
        let json = """
            {"model_type":"deepseek_v41","text_config":{"model_type":"deepseek_v41_text",
             "num_hidden_layers":40,"head_dim":512,"qk_rope_head_dim":64,"sliding_window":128,
             "index_head_dim":128,"index_n_heads":32,"index_topk":512,
             "compress_ratios":\(ratios),"kv_source_layer_ids":[2,8,14,20],
             "index_source_layer_ids":[2,8,14,20,24,28,32,36],"candidate_source_layer_id":20,
             "engram_layer_ids":[1,14]}}
            """
        return try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(json.utf8))
    }

    /// Reference row bytes: 4-bit codes + bf16 scale and bias per 64 (mx.quantize, GROUP_SIZE 64).
    static func pythonRowBytes(_ d: Int) -> Int { d / 2 + (d / 64) * 4 }
    /// vmlx row bytes: 4-bit codes + fp16 scale and minimum per 64 (`v41GroupSize`, same bytes as
    /// the reference's bf16 scale and bias per 64).
    static func swiftRowBytes(_ d: Int) -> Int { d / 2 + (d / DeepseekV4PoolStorage.v41GroupSize) * 4 }

    struct Geometry {
        let layers: Int
        let window: Int
        let headDim: Int
        let indexDim: Int
        /// (layer, ratio) of every kv-source layer.
        let sources: [(Int, Int)]

        init(_ c: DeepseekV4Configuration) {
            layers = c.numHiddenLayers
            window = c.slidingWindow
            headDim = c.headDim
            indexDim = c.indexHeadDim
            sources = c.kvSourceLayerIds.map { ($0, c.compressRatios[$0]) }
        }

        func rows(_ tokens: Int, _ ratio: Int) -> Int { tokens / ratio }

        /// Python resident bytes at `tokens` (live rows; one window copy per layer).
        func pythonBytes(_ tokens: Int) -> Int {
            sources.reduce(layers * window * headDim * 2) { total, s in
                total + rows(tokens, s.1)
                    * (pythonRowBytes(headDim) + pythonRowBytes(indexDim))
            }
        }

        /// vmlx resident bytes at `tokens`: slab capacity (2048-row steps) for the
        /// pools and the K and V window buffers of every layer.
        func swiftBytes(_ tokens: Int) -> Int {
            let step = DeepseekV4PoolStorage.contiguousGrowRows
            return sources.reduce(layers * 2 * window * headDim * 2) { total, s in
                let capacity = (rows(tokens, s.1) + step - 1) / step * step
                return total + capacity * (swiftRowBytes(headDim) + swiftRowBytes(indexDim))
            }
        }
    }

    /// Fill `caches` as a chunked prefill does: every layer's window takes the chunk's
    /// KV, and each kv-source layer appends the groups the chunk completed to its
    /// compressed pool and its index-key pool. Returns the appended rows per source
    /// layer (for value checks).
    @discardableResult
    static func prefill(
        _ caches: [KVCache], geometry g: Geometry, tokens: Int, chunk: Int = 2048,
        keepRows: Bool = false
    ) -> [Int: (comp: [MLXArray], idx: [MLXArray])] {
        var kept: [Int: (comp: [MLXArray], idx: [MLXArray])] = [:]
        var done = 0
        var seed: UInt64 = 1
        while done < tokens {
            let L = min(chunk, tokens - done)
            let kv = MLXRandom.normal([1, 1, L, g.headDim], key: MLXRandom.key(seed)).asType(.bfloat16)
            seed += 1
            for layer in caches {
                _ = layer.update(keys: kv, values: kv)
                if let v4 = layer as? DeepseekV4Cache, v4.requiresAuxiliaryTokenHistory {
                    v4.recordAuxiliaryTokens(
                        (0 ..< L).map { Int32(($0 + done) % 997) }, startPos: done)
                }
            }
            DeepseekV4Model.compactSlidingWindows(caches)
            for (layer, ratio) in g.sources {
                let fresh = g.rows(done + L, ratio) - g.rows(done, ratio)
                guard fresh > 0, let cache = caches[layer] as? DeepseekV4Cache else { continue }
                let comp = MLXRandom.normal([1, fresh, g.headDim], key: MLXRandom.key(seed))
                    .asType(.bfloat16)
                let idx = MLXRandom.normal([1, fresh, g.indexDim], key: MLXRandom.key(seed + 1))
                    .asType(.bfloat16)
                seed += 2
                _ = cache.appendPooled(.compressor, value: comp, isPrefill: true)
                _ = cache.appendPooled(.indexer, value: idx, isPrefill: true)
                if keepRows {
                    kept[layer, default: ([], [])].comp.append(comp)
                    kept[layer, default: ([], [])].idx.append(idx)
                }
            }
            MLX.eval(caches)
            done += L
        }
        return kept
    }

    @Test("resident KV follows the reference formula (~1 KB/token, 1M ≈ 1.0 GB); the old layout was 3.2 KB/token")
    func residentBytesPerToken() throws {
        try Device.withDefaultDevice(.cpu) {
            let cfg = try Self.config()
            #expect(cfg.isDeepseekV41)
            let g = Geometry(cfg)
            #expect(g.sources.map(\.1) == [2, 2, 2, 1])

            let caches = DeepseekV4Model.makeCaches(config: cfg, environment: [:])
            #expect(caches.count == g.layers)
            let v41 = caches.compactMap { $0 as? DeepseekV4Cache }
            #expect(v41.allSatisfy { $0.contiguousPools && $0.hybridPoolQuantizationBits == 4 })

            let tokens = 42_250
            Self.prefill(caches, geometry: g, tokens: tokens)
            let bytes = CacheStoreBudget.cacheBytes(caches)
            #expect(bytes == g.swiftBytes(tokens), "\(bytes) vs formula \(g.swiftBytes(tokens))")

            // Live per-token pool bytes against the reference: identical (group 64 both).
            let swiftPerToken = g.sources.reduce(0.0) {
                $0 + (Double(Self.swiftRowBytes(g.headDim) + Self.swiftRowBytes(g.indexDim)) / Double($1.1))
            }
            let pythonPerToken = g.sources.reduce(0.0) {
                $0 + (Double(Self.pythonRowBytes(g.headDim) + Self.pythonRowBytes(g.indexDim)) / Double($1.1))
            }
            #expect(pythonPerToken == 900)
            #expect(swiftPerToken == 900)
            let swiftWindows = g.layers * 2 * g.window * g.headDim * 2
            let pythonWindows = g.layers * g.window * g.headDim * 2
            let swiftPools = bytes - swiftWindows
            let pythonPools = g.pythonBytes(tokens) - pythonWindows
            #expect(
                Double(swiftPools) <= 1.07 * Double(pythonPools),
                "pools \(swiftPools) B vs reference \(pythonPools) B")
            #expect(Double(g.swiftBytes(1 << 20)) < 0.97e9, "1M tokens: \(g.swiftBytes(1 << 20)) B")

            // Root cause: the segmented layout kept every pool under 64 MiB in BF16.
            let old = DeepseekV4Model.makeCaches(
                config: cfg, environment: ["VMLX_DSV41_POOL_LAYOUT": "segmented"])
            Self.prefill(old, geometry: g, tokens: tokens)
            let oldBytes = CacheStoreBudget.cacheBytes(old)
            let oldPerToken = Double(oldBytes - swiftWindows) / Double(tokens)
            #expect(oldPerToken > 3_000, "segmented layout \(oldPerToken) B/token")
            #expect(oldBytes - swiftWindows > 3 * swiftPools)
        }
    }

    @Test("the disk record stores the same compact 4-bit form and restores bit-exactly")
    func diskRecordIsCompact() throws {
        try Device.withDefaultDevice(.cpu) {
            let cfg = try Self.config()
            let g = Geometry(cfg)
            let caches = DeepseekV4Model.makeCaches(config: cfg, environment: [:])
            let tokens = 9_000
            Self.prefill(caches, geometry: g, tokens: tokens, chunk: 2048)

            let record = TQDiskSerializer.serialize(cache: caches)
            let poolBytes = record
                .filter { $0.key.contains("_pool_") && $0.key.contains("_q_") && !$0.key.hasPrefix("__") }
                .reduce(0) { $0 + $1.value.nbytes }
            let livePoolBytes = g.sources.reduce(0) {
                $0 + g.rows(tokens, $1.1) * (Self.swiftRowBytes(g.headDim) + Self.swiftRowBytes(g.indexDim))
            }
            #expect(poolBytes == livePoolBytes, "record pools \(poolBytes) vs live \(livePoolBytes)")
            let recordBytes = record.reduce(0) { $0 + $1.value.nbytes }
            #expect(recordBytes <= g.swiftBytes(tokens))

            var restored = DeepseekV4Model.makeCaches(config: cfg, environment: [:])
            #expect(restoreFromDiskArrays(record, into: &restored) == tokens)
            for (layer, _) in g.sources {
                let a = caches[layer] as! DeepseekV4Cache
                let b = restored[layer] as! DeepseekV4Cache
                #expect(b.contiguousPools)
                for branch in [DeepseekV4Cache.BranchKey.compressor, .indexer] {
                    #expect(MLX.abs(a.getPooled(branch)! - b.getPooled(branch)!).max().item(Float.self) == 0)
                }
            }
            // A restored cache keeps growing in place.
            Self.prefill(restored, geometry: g, tokens: 4096)
            let l20 = restored[20] as! DeepseekV4Cache
            #expect(l20.getPooled(.compressor)!.dim(1) == tokens + 4096)
        }
    }

    @Test("contiguous storage keeps exactly the 4-bit values, gathers on device and never aliases copies")
    func contiguousValues() {
        Device.withDefaultDevice(.cpu) {
            let storage = DeepseekV4PoolStorage(
                quantizationEnabled: true, hotByteLimit: 0, bits: 4, contiguous: true)
            var parts: [MLXArray] = []
            var seed: UInt64 = 100
            for rows in [1, 7, 2048, 3000, 1, 1, 5] {
                let p = MLXRandom.normal([1, rows, 512], key: MLXRandom.key(seed)).asType(.bfloat16)
                seed += 1
                parts.append(p)
                storage.append(p)
            }
            let all = concatenated(parts, axis: 1)
            let n = all.dim(1)
            #expect(storage.rowCount == n)
            #expect(storage.quantizedSegments.count == 1)
            #expect(storage.hotPool == nil)
            let expected = DeepseekV4PoolStorage.dequantize(
                DeepseekV4PoolStorage.quantize(all, bits: 4, groupSize: DeepseekV4PoolStorage.v41GroupSize))
            let full = storage.materialized()!
            #expect(MLX.abs(full - expected).max().item(Float.self) == 0)
            #expect(storage.retainedByteCount
                == 6144 * Self.swiftRowBytes(512))  // capacity: 5063 rows -> 3 x 2048

            let ids: [Int32] = [0, 1, 7, 2055, 5062, 4000, 0]
            let gathered = storage.gatherDequantizedRows(MLXArray(ids, [1, ids.count]))
            #expect(MLX.abs(gathered - take(full, MLXArray(ids), axis: 1)).max().item(Float.self) == 0)

            var tiles: [MLXArray] = []
            storage.forEachDequantizedTile(maxRows: 1000) { _, tile in tiles.append(tile) }
            #expect(MLX.abs(concatenated(tiles, axis: 1) - full).max().item(Float.self) == 0)

            // A copy is a snapshot: in-place writes to the source never reach it.
            let snapshot = storage.copy()
            let extra = MLXRandom.normal([1, 3, 512], key: MLXRandom.key(7)).asType(.bfloat16)
            storage.append(extra)
            #expect(MLX.abs(snapshot.materialized()! - full).max().item(Float.self) == 0)
            #expect(snapshot.rowCount == n)
            let grown = storage.materialized()!
            #expect(grown.dim(1) == n + 3)
            #expect(MLX.abs(grown[0..., 0 ..< n, 0...] - full).max().item(Float.self) == 0)

            // Trim then append overwrites the dropped rows.
            storage.trimTrailingRows(4)
            #expect(storage.rowCount == n - 1)
            let tail = MLXRandom.normal([1, 2, 512], key: MLXRandom.key(8)).asType(.bfloat16)
            storage.append(tail)
            let after = storage.materialized()!
            let want = concatenated([all[0..., 0 ..< (n - 1), 0...], tail], axis: 1)
            #expect(MLX.abs(after - DeepseekV4PoolStorage.dequantize(
                DeepseekV4PoolStorage.quantize(want, bits: 4, groupSize: DeepseekV4PoolStorage.v41GroupSize)))
                .max().item(Float.self) == 0)
        }
    }

    @Test("window compaction leaves every later update identical")
    func windowCompactionIsInvisible() {
        Device.withDefaultDevice(.cpu) {
            let plain = RotatingKVCache(maxSize: 128, keep: 0)
            let compacted = RotatingKVCache(maxSize: 128, keep: 0)
            var seed: UInt64 = 300
            func step(_ L: Int) {
                let kv = MLXRandom.normal([1, 1, L, 64], key: MLXRandom.key(seed))
                seed += 1
                let (a, _) = plain.update(keys: kv, values: kv)
                let (b, _) = compacted.update(keys: kv, values: kv)
                #expect(a.shape == b.shape)
                #expect(MLX.abs(a - b).max().item(Float.self) == 0)
                if L > 1 { compacted.compactToWindow() }
            }
            step(300)
            #expect(compacted.state[0].dim(2) == 128)
            #expect(plain.state[0].dim(2) == 300)
            for _ in 0 ..< 140 { step(1) }
            step(50)
            for _ in 0 ..< 3 { step(1) }
            step(2048)
            step(1)
            #expect(compacted.offset == plain.offset)
            #expect(MLX.abs(compacted.state[0] - plain.state[0]).max().item(Float.self) == 0)

            // A short prompt (no overflow) is left alone.
            let short = RotatingKVCache(maxSize: 128, keep: 0)
            _ = short.update(keys: MLXArray.ones([1, 1, 40, 8]), values: MLXArray.ones([1, 1, 40, 8]))
            short.compactToWindow()
            #expect(short.state[0].dim(2) == 40)
        }
    }

    @Test("DSV4-0731 caches keep the segmented hot-tier layout and their windows")
    func dsv4_0731Unchanged() throws {
        Device.withDefaultDevice(.cpu) {
            var cfg = DeepseekV4Configuration()
            cfg.numHiddenLayers = 4
            cfg.compressRatios = [0, 4, 128, 0]
            #expect(!cfg.isDeepseekV41)
            let caches = DeepseekV4Model.makeCaches(config: cfg, environment: [:])
            let hybrid = caches.compactMap { $0 as? DeepseekV4Cache }
            #expect(hybrid.count == 2)
            #expect(hybrid.allSatisfy {
                !$0.contiguousPools && !$0.compactLocalWindow && $0.hybridPoolQuantizationBits == 8
            })
            let kv = MLXArray.ones([1, 1, 300, 8])
            _ = hybrid[0].update(keys: kv, values: kv)
            #expect(hybrid[0].local.state[0].dim(2) == 300)
        }
    }
}
