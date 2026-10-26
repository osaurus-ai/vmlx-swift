// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 dSpark pieces in MLXLLM (CPU only, tiny synthetic weights, no bundle, no GPU):
//   - DeepseekV4Cache in-place speculative restore: local window, pools, incomplete-window buffers,
//     deferred decode inputs, the Engram n-gram history, the hidden tail and the offset all return
//     to the pre-verify state in the SAME object;
//   - the dSpark hidden tail travels with copy, trim and the disk record;
//   - the draft head loads from a synthetic `dspark.safetensors` with every parameter verified, the
//     windows synced incrementally from the cache tail equal a one-shot sync, and the Markov chain
//     equals the reference host loop;
//   - gating: only DeepSeek-V4.1 bundles with `dspark/` resolve `.dspark`, Off and
//     `VMLX_DSV41_DSPARK=0` turn it off, a DSV4-0731 model never attaches a head, and native MTP
//     cold-start stays off for both DSV4 families.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("DeepSeek-V4.1 dSpark (CPU)", .serialized)
struct DeepseekV41DSparkFocusedTests {

    static func random(_ shape: [Int], seed: UInt64, scale: Float = 0.05) -> MLXArray {
        var rng = SplitMix64(seed: seed)
        let n = shape.reduce(1, *)
        let values = (0 ..< n).map { _ in Float(Double(rng.next() % 20_001) / 10_000.0 - 1.0) * scale }
        return MLXArray(values, shape)
    }

    static func equal(_ a: MLXArray?, _ b: MLXArray?) -> Bool {
        switch (a, b) {
        case (nil, nil): return true
        case (let a?, let b?):
            guard a.shape == b.shape else { return false }
            if a.size == 0 { return true }
            return MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self) == 0
        default: return false
        }
    }

    // MARK: Pool layouts

    /// The two DeepSeek-V4.1 pool layouts `DeepseekV4Model.makeCaches` builds: one contiguous 4-bit
    /// slab per pool written in place (default), and `VMLX_DSV41_POOL_LAYOUT=segmented`.
    enum Layout: String, CaseIterable, Sendable, CustomTestStringConvertible {
        case segmented, contiguous
        var testDescription: String { rawValue }
    }

    /// A V4.1 holder-layer cache with `layout`'s pool storage, as `makeCaches` configures it
    /// (`hotByteLimit` overrides the segmented layout's BF16 hot tier).
    static func v41Cache(
        slidingWindow: Int, compressRatio: Int, layout: Layout, hotByteLimit: Int? = nil
    ) -> DeepseekV4Cache {
        DeepseekV4Cache(
            slidingWindow: slidingWindow, compressRatio: compressRatio,
            poolBF16HotByteLimit: layout == .contiguous ? 0 : hotByteLimit,
            requiresAuxiliaryTokenHistory: true, poolQuantizationBits: 4,
            contiguousPools: layout == .contiguous, compactLocalWindow: true)
    }

    @Test("pool checkpoint survives in-place appends and slab growth; restore hides the rejected rows",
          arguments: Layout.allCases, [0, 64 << 20])
    func poolStorageCheckpoint(layout: Layout, hotByteLimit: Int) {
        Device.withDefaultDevice(.cpu) {
            let contiguous = layout == .contiguous
            func storage() -> DeepseekV4PoolStorage {
                DeepseekV4PoolStorage(
                    quantizationEnabled: true, hotByteLimit: hotByteLimit, bits: 4,
                    contiguous: contiguous)
            }
            #expect(storage().contiguous == contiguous)
            let grow = DeepseekV4PoolStorage.contiguousGrowRows
            let base = Self.random([1, grow - 2, 64], seed: 600, scale: 1)
            let rejected = Self.random([1, 5, 64], seed: 601, scale: 1)
            let kept = Self.random([1, 3, 64], seed: 602, scale: 1)
            let baseOnly = storage()
            baseOnly.append(base)
            // The reference never saw the rejected rows.
            let want = storage()
            want.append(base)
            want.append(kept)

            let s = storage()
            s.append(base)
            let ck = s.snapshot()
            let capacityBefore = s.retainedByteCount
            s.append(rejected[0..., 0 ..< 1])  // contiguous: in place, inside the slab
            s.append(rejected[0..., 1...])  // contiguous: past capacity, the slab grows
            #expect(s.rowCount == grow + 3)
            if contiguous { #expect(s.retainedByteCount > capacityBefore) }
            s.load(ck)
            // Only the checkpoint's rows are visible, through every reader.
            #expect(s.rowCount == grow - 2)
            #expect(Self.equal(s.materialized(), baseOnly.materialized()))
            #expect(s.quantizedSegments.reduce(0) { $0 + $1.rowCount }
                == baseOnly.quantizedSegments.reduce(0) { $0 + $1.rowCount })
            var tiled = 0
            s.forEachDequantizedTile(maxRows: 512) { start, tile in tiled = max(tiled, start + tile.dim(1)) }
            #expect(tiled == grow - 2)
            s.load(ck)  // restoring the same checkpoint again changes nothing
            #expect(s.rowCount == grow - 2)

            // Appends after the restore overwrite the rejected rows (contiguous: grown slab, in place).
            s.append(kept)
            #expect(Self.equal(s.materialized(), want.materialized()))
            let idx = MLXArray([Int32(grow - 3), Int32(grow - 2), Int32(grow)]).reshaped(1, 3)
            #expect(Self.equal(s.gatherDequantizedRows(idx), want.gatherDequantizedRows(idx)))

            // A second verify inside the grown slab's capacity.
            let ck2 = s.snapshot()
            s.append(rejected[0..., 0 ..< 2])
            s.load(ck2)
            s.append(kept[0..., 0 ..< 1])
            want.append(kept[0..., 0 ..< 1])
            #expect(Self.equal(s.materialized(), want.materialized()))
            #expect(s.snapshotRowsIntact(ck))

            if contiguous {
                // A drop below a checkpoint's rows lets later appends rewrite them: that checkpoint
                // is no longer intact; one at or below the drop still is.
                let ck3 = s.snapshot()
                s.trimTrailingRows(2)
                s.append(rejected[0..., 0 ..< 2])
                #expect(!s.snapshotRowsIntact(ck3))
                #expect(!s.snapshotRowsIntact(ck2))
                #expect(s.snapshotRowsIntact(ck))
                // Replacing the content invalidates every non-empty checkpoint.
                let ck4 = s.snapshot()
                s.replace(with: base)
                #expect(!s.snapshotRowsIntact(ck4))
            }
        }
    }

    // MARK: DeepseekV4Cache in-place restore

    @Test("DeepseekV4Cache restore returns every field to the pre-verify state in place",
          arguments: Layout.allCases)
    func dsv4CacheRestore(layout: Layout) {
        Device.withDefaultDevice(.cpu) {
            let cache = Self.v41Cache(slidingWindow: 8, compressRatio: 2, layout: layout, hotByteLimit: 0)
            #expect(cache.contiguousPools == (layout == .contiguous))
            // Committed state past the window (not trimmable any more).
            _ = cache.update(keys: Self.random([1, 1, 10, 64], seed: 1), values: Self.random([1, 1, 10, 64], seed: 2))
            for i in 0 ..< 3 {
                _ = cache.update(keys: Self.random([1, 1, 1, 64], seed: 10 + UInt64(i)),
                                 values: Self.random([1, 1, 1, 64], seed: 20 + UInt64(i)))
            }
            #expect(!cache.isTrimmable)
            _ = cache.appendPooled(.compressor, value: Self.random([1, 70, 64], seed: 3), isPrefill: true)
            _ = cache.appendPooled(.indexer, value: Self.random([1, 70, 64], seed: 4), isPrefill: true)
            cache.setBuffers(.indexer, kv: Self.random([1, 1, 64], seed: 5), gate: Self.random([1, 1, 64], seed: 6))
            cache.recordAuxiliaryTokens((0 ..< 13).map { Int32($0 * 3) }, startPos: 0)
            cache.auxiliaryHiddenTail = HybridPoolHiddenTail(
                capacity: 8, start: 5, rows: Self.random([8, 12], seed: 7))
            // The flush stands in for the compressor; it leaves the pools alone.
            let flush: (MLXArray, Int) -> DeepseekV4PoolView = { x, _ in DeepseekV4PoolView(dense: x) }
            _ = cache.deferPoolToken(
                Self.random([1, 1, 64], seed: 8), branch: .compressor, startPos: 13, ratio: 2,
                overlap: false, flush: flush)  // no previous pool view: flushed at once

            let before = cache.copy() as! DeepseekV4Cache
            _ = cache.deferPoolToken(
                Self.random([1, 1, 64], seed: 9), branch: .compressor, startPos: 13, ratio: 2,
                overlap: false, flush: flush)  // stays deferred
            let pendingBefore = cache.pendingPoolTokenCount(.compressor)
            #expect(pendingBefore == 1)
            let ck = cache.speculativeCheckpoint()
            let poolRows = cache.getPooled(.compressor)?.dim(1)

            // A rejected verify block: every field moves.
            cache.engramLookaheadTokens = [7, 8, 9]
            _ = cache.update(keys: Self.random([1, 1, 5, 64], seed: 30), values: Self.random([1, 1, 5, 64], seed: 31))
            _ = cache.appendPooled(.compressor, value: Self.random([1, 3, 64], seed: 32), isPrefill: false)
            _ = cache.appendPooled(.indexer, value: Self.random([1, 3, 64], seed: 33), isPrefill: false)
            cache.setBuffers(.indexer, kv: Self.random([1, 2, 64], seed: 34), gate: nil)
            _ = cache.deferPoolToken(
                Self.random([1, 1, 64], seed: 36), branch: .compressor, startPos: 14, ratio: 2,
                overlap: false, flush: flush)
            cache.recordAuxiliaryTokens([99, 98, 97, 96, 95], startPos: 13)
            var tail = cache.auxiliaryHiddenTail!
            tail.record(rows: Self.random([5, 12], seed: 35), startPosition: 13)
            cache.auxiliaryHiddenTail = tail
            #expect(cache.offset == 18)

            cache.restoreSpeculativeCheckpoint(ck)
            #expect(cache.offset == 13)
            #expect(cache.getPooled(.compressor)?.dim(1) == poolRows)
            // The next-chunk Engram lookahead belonged to the rejected forward.
            #expect(cache.engramLookaheadTokens == nil && cache.engramLookahead == nil)
            #expect(cache.pendingPoolTokenCount(.compressor) == pendingBefore)
            #expect(cache.auxiliaryTokenHistory == before.auxiliaryTokenHistory)
            #expect(cache.auxiliaryHiddenTail?.endPosition == 13)
            #expect(Self.equal(cache.auxiliaryHiddenTail?.joined()?.rows, before.auxiliaryHiddenTail?.joined()?.rows))
            #expect(Self.equal(cache.getBuffers(.indexer).kv, before.getBuffers(.indexer).kv))
            // Flushing the restored pending input reproduces the copy's state exactly.
            let a = cache.state, b = before.state
            #expect(a.count == b.count)
            for (x, y) in zip(a, b) { #expect(Self.equal(x, y)) }
            #expect(cache.metaState == before.metaState)
        }
    }

    @Test("the hidden tail travels with copy, trim and the disk record", arguments: Layout.allCases)
    func tailTravelsWithCache(layout: Layout) {
        Device.withDefaultDevice(.cpu) {
            let src = Self.v41Cache(slidingWindow: 16, compressRatio: 2, layout: layout)
            _ = src.local.update(keys: MLXArray.ones([1, 1, 6, 8]), values: MLXArray.ones([1, 1, 6, 8]))
            _ = src.appendPooled(.compressor, value: Self.random([1, 3, 64], seed: 42), isPrefill: true)
            _ = src.appendPooled(.indexer, value: Self.random([1, 3, 64], seed: 43), isPrefill: true)
            src.recordAuxiliaryTokens([1, 2, 3, 4, 5, 6], startPos: 0)
            let rows = Self.random([6, 12], seed: 41)
            src.auxiliaryHiddenTail = HybridPoolHiddenTail(capacity: 8, start: 0, rows: rows)
            let pool = src.getPooled(.compressor)

            let dup = src.copy() as! DeepseekV4Cache
            #expect(Self.equal(dup.auxiliaryHiddenTail?.joined()?.rows, rows))
            #expect(Self.equal(dup.getPooled(.compressor), pool))

            let encoded = TQDiskSerializer.serialize(cache: [src])
            let dst = Self.v41Cache(slidingWindow: 16, compressRatio: 2, layout: layout)
            var target: [any KVCache] = [dst]
            #expect(restoreFromDiskArrays(encoded, into: &target) == 6)
            #expect(dst.auxiliaryHiddenTail?.startPosition == 0)
            #expect(dst.auxiliaryHiddenTail?.capacity == 8)
            #expect(Self.equal(dst.auxiliaryHiddenTail?.joined()?.rows, rows))
            #expect(Self.equal(dst.getPooled(.compressor), pool))
            // The restored cache checkpoints and restores in place like a live one.
            let ck = dst.speculativeCheckpoint()
            _ = dst.appendPooled(.compressor, value: Self.random([1, 2, 64], seed: 44), isPrefill: false)
            var tail = dst.auxiliaryHiddenTail!
            tail.record(rows: Self.random([2, 12], seed: 45), startPosition: 6)
            dst.auxiliaryHiddenTail = tail
            dst.restoreSpeculativeCheckpoint(ck)
            #expect(Self.equal(dst.getPooled(.compressor), pool))
            #expect(Self.equal(dst.auxiliaryHiddenTail?.joined()?.rows, rows))

            src.trim(2)
            #expect(src.auxiliaryHiddenTail?.endPosition == 4)
            #expect(Self.equal(src.auxiliaryHiddenTail?.joined()?.rows, rows[0 ..< 4]))
            #expect(src.getPooled(.compressor)?.dim(1) == 2)
        }
    }

    // MARK: In-place prefix commit (Python commit_prefix)

    /// One V4.1 kv-source (and index-owner) attention layer at `ratio`.
    static func commitLayer(ratio: Int) -> DeepseekV4Attention {
        var c = tinyConfig()
        c.numHiddenLayers = 1
        c.compressRatios = [ratio]
        c.kvSourceLayerIds = [0]
        c.indexSourceLayerIds = [0]
        c.indexNHeads = 2
        c.indexHeadDim = 64
        c.slidingWindow = 4
        return DeepseekV4Attention(config: c, layerIdx: 0)
    }

    static func commitCache(_ layout: Layout) -> DeepseekV4Cache {
        v41Cache(slidingWindow: 4, compressRatio: 2, layout: layout)
    }

    static func run(_ attn: DeepseekV4Attention, _ x: MLXArray, _ cache: DeepseekV4Cache,
                    capture: DSV41VerifyCapture? = nil) {
        let share = DSV41CompressShare()
        if let capture {
            capture.begin(tokens: x.dim(1))
            share.capture = capture
        }
        let out = attn(x, mask: createAttentionMask(h: x, cache: cache), cache: cache, compressShare: share)
        eval(out)
    }

    /// Prompt prefix, then two single-token steps (deferred pool tokens, wrapped window).
    /// `nearCapacity` pads both pools to just under the contiguous slab's capacity, so the verify
    /// forward's pool rows grow (reallocate) the slab after the checkpoint.
    static func primed(
        _ attn: DeepseekV4Attention, _ xs: MLXArray, _ layout: Layout, nearCapacity: Bool = false
    ) -> DeepseekV4Cache {
        let cache = commitCache(layout)
        run(attn, xs[0..., 0 ..< 7], cache)
        if nearCapacity {
            for (branch, seed) in [(DeepseekV4Cache.BranchKey.compressor, UInt64(70)), (.indexer, 71)] {
                let rows = cache.getPooled(branch)?.dim(1) ?? 0
                _ = cache.appendPooled(
                    branch,
                    value: random([1, DeepseekV4PoolStorage.contiguousGrowRows - 1 - rows, 64], seed: seed),
                    isPrefill: true)
            }
        }
        run(attn, xs[0..., 7 ..< 8], cache)
        run(attn, xs[0..., 8 ..< 9], cache)
        return cache
    }

    static func snapshot(_ cache: DeepseekV4Cache) -> (offset: Int, state: [MLXArray], meta: [String]) {
        let state = cache.state  // flushes deferred decode inputs
        eval(state)
        return (cache.offset, state, cache.metaState)
    }

    @Test("commit_prefix(all) equals the captured verify's own state bit for bit; partial keeps match a forward of only the kept tokens",
          arguments: [2, 1], Layout.allCases)
    func commitPrefixMatchesForward(ratio: Int, layout: Layout) {
        commitPrefixCase(ratio: ratio, layout: layout, nearCapacity: false)
    }

    @Test("commit_prefix across a contiguous slab growth matches a forward of only the kept tokens",
          arguments: Layout.allCases)
    func commitPrefixAcrossSlabGrowth(layout: Layout) {
        commitPrefixCase(ratio: 2, layout: layout, nearCapacity: true)
    }

    func commitPrefixCase(ratio: Int, layout: Layout, nearCapacity: Bool) {
        Device.withDefaultDevice(.cpu) {
            let attn = Self.commitLayer(ratio: ratio)
            let xs = Self.random([1, 15, 64], seed: 500 + UInt64(ratio), scale: 1)
            let verify = xs[0..., 9 ..< 15]  // [p0, d1..d5]

            // Reference: the verify forward's own state.
            let full = Self.primed(attn, xs, layout, nearCapacity: nearCapacity)
            let grow = DeepseekV4PoolStorage.contiguousGrowRows
            if nearCapacity {
                // The verify forward's pool rows cross the slab capacity.
                #expect((full.getPooled(.compressor)?.dim(1) ?? 0) <= grow)
            }
            Self.run(attn, verify, full)
            if nearCapacity {
                #expect((full.getPooled(.compressor)?.dim(1) ?? 0) > grow)
                #expect((full.getPooled(.indexer)?.dim(1) ?? 0) > grow)
            }
            let fullState = Self.snapshot(full)

            for keep in 1 ... 6 {
                let cache = Self.primed(attn, xs, layout, nearCapacity: nearCapacity)
                let objects = ObjectIdentifier(cache)
                let ck = SpeculativeCacheListCheckpoint([cache])!
                let capture = DSV41VerifyCapture()
                Self.run(attn, verify, cache, capture: capture)
                #expect(capture.valid && capture.layers[0] != nil)
                ck.restore()
                attn.commitCapturedPrefix(capture.layers[0]!, keep: keep, cache: cache)
                #expect(ObjectIdentifier(cache) == objects)
                let got = Self.snapshot(cache)

                if keep == 6 {
                    #expect(got.offset == fullState.offset)
                    #expect(got.meta == fullState.meta)
                    #expect(got.state.count == fullState.state.count)
                    for (a, b) in zip(got.state, fullState.state) { #expect(Self.equal(a, b)) }
                }
                // A forward of only the kept tokens.
                let only = Self.primed(attn, xs, layout, nearCapacity: nearCapacity)
                Self.run(attn, verify[0..., 0 ..< keep], only)
                let want = Self.snapshot(only)
                #expect(got.offset == want.offset, "keep \(keep)")
                #expect(got.meta == want.meta, "keep \(keep)")
                #expect(got.state.count == want.state.count)
                for (a, b) in zip(got.state, want.state) {
                    #expect(a.shape == b.shape, "keep \(keep)")
                    if a.shape == b.shape, a.size > 0 {
                        #expect(allClose(a.asType(.float32), b.asType(.float32), rtol: 2e-2, atol: 2e-2)
                            .item(Bool.self), "keep \(keep)")
                    }
                }
                #expect(cache.getPooled(.indexer)?.dim(1) == only.getPooled(.indexer)?.dim(1))
            }
        }
    }

    // MARK: Draft head

    static let blockSize = 5

    static func tinyConfig(v41: Bool = true) -> DeepseekV4Configuration {
        var c = DeepseekV4Configuration()
        c.vocabSize = 128
        c.hiddenSize = 64
        c.numHiddenLayers = 4
        c.numAttentionHeads = 2
        c.headDim = 64
        c.qkRopeHeadDim = 16
        c.qLoraRank = 64
        c.oLoraRank = 64
        c.oGroups = 2
        c.slidingWindow = 8
        c.hcMult = 2
        c.hcSinkhornIters = 4
        c.moeIntermediateSize = 64
        c.rmsNormEps = 1e-6
        c.nRoutedExperts = 4
        c.numExpertsPerTok = 2
        c.compressRatios = [0, 0, 0, 0]
        if v41 { c.modelType = "deepseek_v41_text" }
        return c
    }

    nonisolated(unsafe) static let raw: [String: Any] = [
        "dspark_block_size": blockSize, "dspark_noise_token_id": 127,
        "dspark_target_layer_ids": [3, 1, 2], "dspark_markov_rank": 16,
        "dspark_n_routed_experts": 4, "dspark_num_experts_per_tok": 2, "num_nextn_predict_layers": 3,
    ]

    /// A synthetic bundle (`config.json` + `dspark/`) with the bundle's tensor names and layouts.
    /// Returns the `dspark/` directory; remove its parent when done.
    static func writeDraft(_ cfg: DSparkV41Config) throws -> URL {
        let bundle = FileManager.default.temporaryDirectory
            .appendingPathComponent("dsv41-bundle-\(UUID().uuidString)", isDirectory: true)
        let dir = bundle.appendingPathComponent("dspark", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: ["model_type": "deepseek_v41", "text_config": raw])
            .write(to: bundle.appendingPathComponent("config.json"))
        let d = cfg.hiddenSize, h = cfg.nHeads, hd = cfg.headDim
        let mix = (2 + cfg.hcMult) * cfg.hcMult
        var t: [String: MLXArray] = [:]
        var seed: UInt64 = 100
        func r(_ shape: [Int], scale: Float = 0.05) -> MLXArray {
            seed += 1
            return random(shape, seed: seed, scale: scale)
        }
        for i in 0 ..< cfg.nStages {
            let p = "mtp.\(i)"
            t["\(p).attn.wq_a.weight"] = r([cfg.qLoraRank, d]).asType(.bfloat16)
            t["\(p).attn.wq_b.weight"] = r([h * hd, cfg.qLoraRank]).asType(.bfloat16)
            t["\(p).attn.wkv.weight"] = r([hd, d]).asType(.bfloat16)
            t["\(p).attn.wo_a.weight"] = r([cfg.oGroups * cfg.oLoraRank, h * hd / cfg.oGroups]).asType(.bfloat16)
            t["\(p).attn.wo_b.weight"] = r([d, cfg.oGroups * cfg.oLoraRank]).asType(.bfloat16)
            t["\(p).attn.q_norm.weight"] = MLXArray.ones([cfg.qLoraRank], dtype: .bfloat16)
            t["\(p).attn.kv_norm.weight"] = MLXArray.ones([hd], dtype: .bfloat16)
            t["\(p).attn.attn_sink"] = r([h], scale: 0.5)
            t["\(p).attn_norm.weight"] = MLXArray.ones([d], dtype: .bfloat16)
            t["\(p).ffn_norm.weight"] = MLXArray.ones([d], dtype: .bfloat16)
            for w in ["attn", "ffn"] {
                t["\(p).hc_\(w)_fn"] = r([mix, cfg.hcMult * d])
                t["\(p).hc_\(w)_base"] = r([mix], scale: 0.2)
                t["\(p).hc_\(w)_scale"] = r([3], scale: 0.2) + 1
            }
            t["\(p).ffn.gate.weight"] = r([cfg.nRoutedExperts, d]).asType(.bfloat16)
            t["\(p).ffn.gate.bias"] = r([cfg.nRoutedExperts])
            t["\(p).ffn.gate.bias_vl"] = r([cfg.nRoutedExperts])
            t["\(p).ffn.shared_experts.w1.weight"] = r([cfg.moeInterDim, d]).asType(.bfloat16)
            t["\(p).ffn.shared_experts.w3.weight"] = r([cfg.moeInterDim, d]).asType(.bfloat16)
            t["\(p).ffn.shared_experts.w2.weight"] = r([d, cfg.moeInterDim]).asType(.bfloat16)
            for (proj, shape) in [
                ("gate_proj", [cfg.nRoutedExperts, cfg.moeInterDim, d]),
                ("up_proj", [cfg.nRoutedExperts, cfg.moeInterDim, d]),
                ("down_proj", [cfg.nRoutedExperts, d, cfg.moeInterDim]),
            ] {
                let q = quantized(r(shape), groupSize: 64, bits: 4)
                t["\(p).ffn.experts.\(proj).weight"] = q.wq
                t["\(p).ffn.experts.\(proj).scales"] = q.scales.asType(.bfloat16)
                t["\(p).ffn.experts.\(proj).biases"] = q.biases!.asType(.bfloat16)
            }
            if i == 0 {
                t["\(p).main_proj.weight"] = r([d, d * cfg.targetLayerIds.count]).asType(.bfloat16)
                t["\(p).main_norm.weight"] = MLXArray.ones([d], dtype: .bfloat16)
            }
            if i == cfg.nStages - 1 {
                t["\(p).norm.weight"] = MLXArray.ones([d], dtype: .bfloat16)
                t["\(p).markov_head.embed.weight"] = r([cfg.vocabSize, cfg.markovRank], scale: 1).asType(.bfloat16)
                t["\(p).markov_head.head.weight"] = r([cfg.vocabSize, cfg.markovRank], scale: 1).asType(.bfloat16)
                t["\(p).confidence_head.proj.weight"] = r([1, d + cfg.markovRank]).asType(.bfloat16)
            }
        }
        try save(arrays: t, url: dir.appendingPathComponent("dspark.safetensors"))
        return dir
    }

    static func loadHead() throws -> (DSparkV41Head, URL) {
        let cfg = DSparkV41Config(tinyConfig(), raw: raw)
        let dir = try writeDraft(cfg)
        let embedW = random([cfg.vocabSize, cfg.hiddenSize], seed: 900, scale: 1)
        let headW = random([cfg.hiddenSize, cfg.vocabSize], seed: 901, scale: 1)
        let head = try DSparkV41Head.load(
            directory: dir, config: cfg,
            embed: { take(embedW, $0.reshaped(-1), axis: 0).reshaped($0.dim(0), $0.dim(1), -1) },
            head: { matmul($0.asType(.float32), headW) })
        return (head, dir)
    }

    /// A V4.1-shaped cache list at offset `end` whose holder carries `tail`.
    static func cacheAt(_ end: Int, tail: HybridPoolHiddenTail?) -> [KVCache] {
        let holder = DeepseekV4Cache(
            slidingWindow: 8, compressRatio: 2, requiresAuxiliaryTokenHistory: true,
            poolQuantizationBits: 4)
        if end > 0 {
            _ = holder.local.update(
                keys: MLXArray.zeros([1, 1, end, 4]), values: MLXArray.zeros([1, 1, end, 4]))
        }
        holder.auxiliaryHiddenTail = tail
        return [RotatingKVCache(maxSize: 8), holder]
    }

    @Test("draft head loads with every parameter verified; incremental window sync equals one-shot")
    func draftHeadWindows() throws {
        try Device.withDefaultDevice(.cpu) {
            let (head, dir) = try Self.loadHead()
            defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }
            let cfg = head.config
            #expect(cfg.targetLayerIds == [3, 1, 2] && cfg.blockSize == 5 && cfg.nStages == 3)
            let width = cfg.hiddenSize * cfg.targetLayerIds.count
            let all = Self.random([30, width], seed: 77, scale: 1)

            // Incremental: commit 12 positions, then 3 more, then 1 more.
            let inc = DSparkV41Session(head: head)
            var tail = HybridPoolHiddenTail(capacity: cfg.slidingWindow)
            tail.record(rows: all[0 ..< 12], startPosition: 0)
            let d1 = inc.draft(seed: 9, cache: Self.cacheAt(12, tail: tail))
            tail.record(rows: all[12 ..< 15], startPosition: 12)
            _ = inc.draft(seed: 4, cache: Self.cacheAt(15, tail: tail))
            tail.record(rows: all[15 ..< 16], startPosition: 15)
            let dInc = inc.draft(seed: 7, cache: Self.cacheAt(16, tail: tail))
            // One-shot from the same tail (a prefix-cache hit seeds from the persisted tail).
            let once = DSparkV41Session(head: head)
            let dOnce = once.draft(seed: 7, cache: Self.cacheAt(16, tail: tail))
            #expect(inc.winEnd == 16 && once.winEnd == 16)
            for (a, b) in zip(inc.windows, once.windows) {
                #expect(a?.dim(1) == cfg.slidingWindow)
                #expect(allClose(a!.asType(.float32), b!.asType(.float32), rtol: 1e-2, atol: 1e-2).item(Bool.self))
            }
            #expect(dInc == dOnce)
            #expect(d1.count == 5 && d1.allSatisfy { (0 ..< cfg.vocabSize).contains($0) })
            // Deterministic.
            #expect(DSparkV41Session(head: head).draft(seed: 7, cache: Self.cacheAt(16, tail: tail)) == dOnce)
            // A tail that does not end at the cache offset is not trusted: empty windows.
            let stale = DSparkV41Session(head: head)
            _ = stale.draft(seed: 7, cache: Self.cacheAt(17, tail: tail))
            #expect(stale.windows.allSatisfy { $0 == nil })
        }
    }

    @Test("markov chain equals the reference host loop")
    func markovChainMatchesReference() {
        Device.withDefaultDevice(.cpu) {
            let vocab = 97, rank = 16, bs = 5
            let emb = Self.random([vocab, rank], seed: 3, scale: 1).asType(.bfloat16)
            let headW = Self.random([vocab, rank], seed: 4, scale: 1).asType(.bfloat16)
            let logits = Self.random([bs, vocab], seed: 5, scale: 1)
            var out = [11]
            for i in 0 ..< bs {
                let me = emb[out[i]].asType(.float32)
                out.append(argMax(logits[i] + matmul(me, headW.asType(.float32).transposed())).item(Int.self))
            }
            let got = DSparkV41Head.markovChain(
                logits: logits, markovEmbed: emb, headT: headW.asType(.float32).transposed(), seed: 11)
            #expect([11] + got.asArray(Int32.self).map(Int.init) == out)
        }
    }

    @Test("a draft with a missing tensor fails to load")
    func missingTensorFails() throws {
        try Device.withDefaultDevice(.cpu) {
            let cfg = DSparkV41Config(Self.tinyConfig(), raw: Self.raw)
            let dir = try Self.writeDraft(cfg)
            defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }
            let url = dir.appendingPathComponent("dspark.safetensors")
            var arrays = try loadArrays(url: url)
            arrays.removeValue(forKey: "mtp.1.attn.wkv.weight")
            try save(arrays: arrays, url: url)
            #expect(throws: (any Error).self) {
                _ = try DSparkV41Head.load(directory: dir, config: cfg, embed: { $0 }, head: { $0 })
            }
        }
    }

    // MARK: Gating

    static func bundle(modelType: String, textType: String?, withDSpark: Bool) throws -> (URL, Data) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dsv4-bundle-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("dspark"), withIntermediateDirectories: true)
        if withDSpark {
            FileManager.default.createFile(
                atPath: dir.appendingPathComponent("dspark/dspark.safetensors").path, contents: Data())
        }
        var config: [String: Any] = ["model_type": modelType]
        if let textType { config["text_config"] = ["model_type": textType] }
        return (dir, try JSONSerialization.data(withJSONObject: config))
    }

    @Test("only V4.1 bundles with dspark/ resolve .dspark; Off and the env switch turn it off")
    func strategyGate() throws {
        let (v41, v41Config) = try Self.bundle(modelType: "deepseek_v41", textType: "deepseek_v41_text", withDSpark: true)
        let (v4, v4Config) = try Self.bundle(modelType: "deepseek_v4", textType: nil, withDSpark: true)
        let (bare, bareConfig) = try Self.bundle(modelType: "deepseek_v41", textType: nil, withDSpark: false)
        defer { for d in [v41, v4, bare] { try? FileManager.default.removeItem(at: d) } }

        var settings = VMLXServerRuntimeSettings()
        #expect(settings.mtp.mode == .familyDefault)
        #expect(settings.bundledDSparkDrafter(configData: v41Config, bundleDirectory: v41, environment: [:])
            == v41.appendingPathComponent("dspark", isDirectory: true))
        let strategy = settings.resolvedMTPDraftStrategy(
            configData: v41Config, jangConfig: nil, status: nil, bundleDirectory: v41)
        #expect(strategy?.usesDSpark == true)
        #expect(strategy?.dsparkDrafterPath?.lastPathComponent == "dspark")

        // DSV4-0731 (even with a dspark/ folder) and a V4.1 bundle without the head: no dSpark.
        #expect(settings.bundledDSparkDrafter(configData: v4Config, bundleDirectory: v4, environment: [:]) == nil)
        #expect(settings.resolvedMTPDraftStrategy(
            configData: v4Config, jangConfig: nil, status: nil, bundleDirectory: v4)?.usesDSpark != true)
        #expect(settings.bundledDSparkDrafter(configData: bareConfig, bundleDirectory: bare, environment: [:]) == nil)

        // The env switch can only turn it off.
        #expect(settings.bundledDSparkDrafter(
            configData: v41Config, bundleDirectory: v41, environment: ["VMLX_DSV41_DSPARK": "0"]) == nil)
        #expect(DSparkEnvironment.disabled(["VMLX_DSV41_DSPARK": "off"]))
        #expect(!DSparkEnvironment.disabled(["VMLX_DSV41_DSPARK": "1"]))

        // Explicit Off wins over everything, including VMLX_DSV41_DSPARK=1.
        settings.mtp.mode = .off
        #expect(settings.bundledDSparkDrafter(
            configData: v41Config, bundleDirectory: v41, environment: ["VMLX_DSV41_DSPARK": "1"]) == nil)
        #expect(settings.resolvedMTPDraftStrategy(
            configData: v41Config, jangConfig: nil, status: nil, bundleDirectory: v41) == nil)
    }

    @Test("native MTP cold start stays off for both DSV4 families")
    func nativeMTPUntouched() {
        for type in ["deepseek_v4", "deepseek_v41"] {
            #expect(NativeMTPAutoDecodePolicy.measuredFamilyColdStartDepth(
                config: ["model_type": type], fallback: nil) == nil)
        }
    }

    @Test("a DSV4-0731 model never attaches a head or records a tail; a V4.1 model does")
    func modelAttachGate() throws {
        try Device.withDefaultDevice(.cpu) {
            let cfg41 = DSparkV41Config(Self.tinyConfig(), raw: Self.raw)
            let dir = try Self.writeDraft(cfg41)
            defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

            let m0731 = DeepseekV4Model(Self.tinyConfig(v41: false))
            #expect(!m0731.config.isDeepseekV41)
            #expect(try m0731.attachDSparkDraft(directory: dir) == false)
            #expect(m0731.dsparkBlockSize == nil)
            #expect(m0731.makeDSparkDraftSession() == nil)
            #expect(m0731.model.dsparkTapLayers.isEmpty)

            let m41 = DeepseekV4Model(Self.tinyConfig())
            #expect(m41.config.isDeepseekV41)
            #expect(try m41.attachDSparkDraft(directory: dir))
            #expect(m41.dsparkBlockSize == 5)
            #expect(m41.model.dsparkTapLayers == [1, 2, 3])
            #expect(m41.makeDSparkDraftSession() != nil)
            #expect(try m41.attachDSparkDraft(directory: dir))  // idempotent
            // The head is not a Module child: the backbone's parameter tree is unchanged.
            #expect(!m41.parameters().flattened().contains { $0.0.contains("stages") })
            let cache = Self.cacheAt(4, tail: nil)
            #expect(m41.dsparkCacheUnsupportedReason(cache) == nil)
            #expect(m41.dsparkCacheUnsupportedReason([KVCacheSimple()]) != nil)
            // No capture recorded: the in-place commit refuses and leaves the cache alone.
            let ck = SpeculativeCacheListCheckpoint(cache)!
            #expect(!m41.dsparkCommitPrefix(cache: cache, checkpoint: ck, keep: 1))
            #expect(cache[1].offset == 4)
        }
    }
}
