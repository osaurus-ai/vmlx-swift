// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

@Suite("DSV4 segmented pool quantization", .serialized)
struct DeepseekV4PoolQuantizationTests {
    private static let hotLimit = 2 * 1024 * 1024

    @Test("production default is enabled with an explicit diagnostic opt-out")
    func defaultPolicy() {
        #expect(DeepseekV4Cache.resolvePoolQuantizationDefault(environment: [:]))
        #expect(DeepseekV4Cache.resolvePoolQuantizationDefault(
            environment: ["DSV4_POOL_QUANT": "1"]))
        #expect(!DeepseekV4Cache.resolvePoolQuantizationDefault(
            environment: ["DSV4_POOL_QUANT": "0"]))
        #expect(!DeepseekV4Cache.resolvePoolQuantizationDefault(
            environment: ["DSV4_POOL_QUANT": "off"]))
        #expect(DeepseekV4Cache.resolvePoolBF16HotByteLimit(environment: [:])
            == 64 * 1024 * 1024)
        #expect(DeepseekV4Cache.resolvePoolBF16HotByteLimit(
            environment: ["DSV4_POOL_BF16_MAX_BYTES": "1048576"])
            == 1024 * 1024)

        let cache = DeepseekV4Cache(slidingWindow: 128, compressRatio: 4)
        #expect(cache.hybridPoolQuantizationEnabled)
        let diagnostic = DeepseekV4Cache(
            slidingWindow: 128, compressRatio: 4, poolQuantizationEnabled: false)
        #expect(!diagnostic.hybridPoolQuantizationEnabled)
    }

    @Test("small pools stay hot; promotion is segmented, bounded, and high fidelity")
    func adaptivePromotionAndQuality() {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }

        let cache = DeepseekV4Cache(
            slidingWindow: 128, compressRatio: 4,
            poolQuantizationEnabled: true,
            poolBF16HotByteLimit: Self.hotLimit)
        let short = Self.signal(rows: 1024, features: 512)
        #expect(short.nbytes < Self.hotLimit)
        cache.setPooled(.compressor, value: short)
        #expect(cache.hybridPoolQuantizedSegments(branch: .compressor) == nil)
        #expect(cache.hybridPoolRetainedByteCount(branch: .compressor) == short.nbytes)

        let raw = Self.signal(rows: 2049, features: 512)
        #expect(raw.nbytes > Self.hotLimit)
        cache.setPooled(.compressor, value: raw)
        guard let segments = cache.hybridPoolQuantizedSegments(branch: .compressor),
              let restored = cache.getPooled(.compressor)
        else {
            Issue.record("pool did not promote to encoded storage")
            return
        }
        MLX.eval(restored)
        #expect(segments.count == 2)
        #expect(segments.map(\.rowCount) == [2048, 1])
        #expect(segments.allSatisfy { $0.rowCount > 0 && $0.rowCount <= 16 * 1024 })
        #expect(segments.allSatisfy { $0.bits == 8 && $0.groupSize == 32 })
        #expect(segments.allSatisfy { $0.codes.dtype == .uint8 })
        let retainedBefore = cache.hybridPoolRetainedByteCount(branch: .compressor)
        #expect(retainedBefore < Int(Double(raw.nbytes) * 0.60))
        #expect(Self.cosine(raw, restored) >= 0.999)

        // A materialized attention read must not be retained beside the codes.
        _ = cache.getPooled(.compressor)
        #expect(cache.hybridPoolRetainedByteCount(branch: .compressor) == retainedBefore)
        #expect(CacheStoreBudget.cacheBytes([cache as any KVCache])
            == cache.retainedCacheByteCount)
    }

    @Test("quantized trim keeps the prefix encoded")
    func trimWithoutTierChange() {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }

        let raw = Self.signal(rows: 2075, features: 512)
        let cache = DeepseekV4Cache(
            slidingWindow: 128, compressRatio: 4,
            poolQuantizationEnabled: true,
            poolBF16HotByteLimit: Self.hotLimit)
        cache.setPooled(.compressor, value: raw)
        #expect(cache.hybridPoolQuantizedSegments(branch: .compressor) != nil)
        _ = cache.trim(68) // max(1, 68 / 4) == 17 pool rows
        guard let kept = cache.getPooled(.compressor),
              let segments = cache.hybridPoolQuantizedSegments(branch: .compressor)
        else {
            Issue.record("trim unexpectedly expanded or cleared encoded pool")
            return
        }
        MLX.eval(kept)
        #expect(kept.shape == [1, 2058, 512])
        #expect(!segments.isEmpty)
        #expect(Self.cosine(raw[0..., 0..<2058, 0...], kept) >= 0.999)
    }

    @Test("SSD serializer preserves encoded segments instead of BF16 expansion")
    func encodedDiskRoundTrip() {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }

        let raw = Self.signal(rows: 2049, features: 512)
        let source = DeepseekV4Cache(
            slidingWindow: 128, compressRatio: 4,
            poolQuantizationEnabled: true,
            poolBF16HotByteLimit: Self.hotLimit)
        Self.fillRotating(source.local)
        source.setPooled(.compressor, value: raw)
        let sourceRetained = source.hybridPoolRetainedByteCount(branch: .compressor)

        let encoded = TQDiskSerializer.serialize(cache: [source])
        #expect(encoded["__dsv4_0_pool_comp_qcount__"] != nil)
        #expect(encoded["dsv4_0_pool_comp_q_0_codes"] != nil)
        #expect(encoded["dsv4_0_pool_comp_q_0_scales"] != nil)
        #expect(encoded["dsv4_0_pool_comp_q_0_biases"] != nil)

        let target = DeepseekV4Cache(
            slidingWindow: 128, compressRatio: 4,
            poolQuantizationEnabled: true,
            poolBF16HotByteLimit: Self.hotLimit)
        var targetLayers: [any KVCache] = [target]
        let restoredTokens = restoreFromDiskArrays(encoded, into: &targetLayers)
        #expect(restoredTokens == source.offset)
        guard let segments = target.hybridPoolQuantizedSegments(branch: .compressor),
              let restored = target.getPooled(.compressor)
        else {
            Issue.record("disk restore did not preserve encoded pool segments")
            return
        }
        MLX.eval(restored)
        #expect(segments.count == 1)
        #expect(segments[0].rowCount == 2049)
        #expect(target.hybridPoolRetainedByteCount(branch: .compressor) == sourceRetained)
        #expect(Self.cosine(raw, restored) >= 0.999)
    }

    @Test("one-million-token DSV4-Flash retained cache projection is below 10 GiB")
    func oneMillionTokenProjection() {
        let ratios = [0, 0] + (0..<41).map { $0.isMultiple(of: 2) ? 4 : 128 }
        let bytes = DeepseekV4Cache.projectedQuantizedCacheUpperBoundBytes(
            contextLength: 1_048_576,
            compressRatios: ratios,
            headDim: 512,
            indexerHeadDim: 128,
            slidingWindow: 128,
            poolBF16HotByteLimit: 64 * 1024 * 1024)
        #expect(ratios.count == 43)
        #expect(ratios.filter { $0 == 4 }.count == 21)
        #expect(ratios.filter { $0 == 128 }.count == 20)
        #expect(bytes > 3 * 1024 * 1024 * 1024)
        #expect(bytes < 10 * 1024 * 1024 * 1024)
    }

    // MARK: - Tiled / selected pool attention parity
    //
    // The tiled online-softmax paths (Python `_dsv4_tiled_pool_attention`
    // port) must be numerically interchangeable with a dense softmax over
    // the SAME dequantized values. 70 000 rows forces at least two bounded
    // tiles through the 64Ki-row cap, so cross-tile accumulator merging and
    // the segment-pending flush in `forEachDequantizedTile` are both on the
    // hot path.

    private static let parityRows = 70_000
    private static let parityRatio = 4
    private static let parityOffset = 140_000
    private static let parityDim = 64
    private static let parityWindow = 8

    /// Deterministic continuous data. `bias` keeps dot products positive so
    /// the indexer relu is the identity and scores stay tie-free.
    private static func continuous(
        _ shape: [Int], step: Float, bias: Float = 0
    ) -> MLXArray {
        let count = shape.reduce(1, *)
        return (MLX.sin(MLXArray(0..<count).asType(.float32) * step) * 0.5
            + MLXArray(bias))
            .reshaped(shape)
    }

    private static func q8Storage(rows: Int, dim: Int) -> DeepseekV4PoolStorage {
        let storage = DeepseekV4PoolStorage(
            quantizationEnabled: true, hotByteLimit: 4096)
        storage.append(
            continuous([1, rows, dim], step: 0.917, bias: 0.9).asType(.bfloat16))
        return storage
    }

    /// Dense fp32 softmax-with-sinks over `kvAll` = local ++ pool rows.
    private static func referenceAttention(
        q: MLXArray, kvAll: MLXArray, mask: MLXArray, scale: Float,
        sinks: MLXArray
    ) -> MLXArray {
        let B = q.dim(0)
        let H = q.dim(1)
        let S = q.dim(2)
        let q32 = q.asType(.float32)
        let kv32 = kvAll.asType(.float32).expandedDimensions(axis: 1)
        var scores = (q32 * MLXArray(scale)).matmul(kv32.swappedAxes(-1, -2))
        scores = MLX.where(mask, scores, MLXArray(-Float.infinity))
        let sinkCol = broadcast(
            sinks.asType(.float32).reshaped(1, H, 1, 1), to: [B, H, S, 1])
        let all = concatenated([scores, sinkCol], axis: -1)
        let weights = softmax(all, axis: -1)[.ellipsis, 0..<kvAll.dim(1)]
        return weights.matmul(kv32)
    }

    @Test("tiled pool attention matches the dense softmax reference on q8 storage")
    func tiledPoolAttentionParity() {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }

        let (rows, ratio, offset, D, W) = (
            Self.parityRows, Self.parityRatio, Self.parityOffset,
            Self.parityDim, Self.parityWindow)
        let (H, S) = (4, 5)
        let storage = Self.q8Storage(rows: rows, dim: D)
        #expect(storage.hotPool == nil)
        #expect(storage.quantizedSegments.count > 1)
        let view = DeepseekV4PoolView(storage: storage)
        let q = Self.continuous([1, H, S, D], step: 0.311, bias: 0.5)
        let localKV = Self.continuous([1, 1, W, D], step: 0.157, bias: 0.2)
        let sinks = Self.continuous([H], step: 0.703)
        let scale = 1.0 / Float(D).squareRoot()

        let tiled = DeepseekV4Math.tiledPoolAttention(
            queries: q, localKV: localKV, pooled: view,
            offset: offset, window: W, ratio: ratio,
            scale: scale, sinks: sinks, topK: nil)

        let pool = view.materialized().asType(.float32)
        let kvAll = concatenated(
            [localKV.squeezed(axis: 1).asType(.float32), pool], axis: 1)
        let mask = concatenated(
            [
                DeepseekV4Math.buildWindowMask(
                    batch: 1, queryLen: S, offset: offset,
                    window: W, windowLen: W),
                DeepseekV4Math.compressedVisibility(
                    batch: 1, queryLen: S, offset: offset,
                    compressedLen: rows, ratio: ratio),
            ], axis: -1)
        let reference = Self.referenceAttention(
            q: q, kvAll: kvAll, mask: mask, scale: scale, sinks: sinks)
        let diff = MLX.abs(tiled.asType(.float32) - reference).max()
        MLX.eval(diff)
        #expect(diff.item(Float.self) < 1e-3)
    }

    @Test("selected pool attention with fixed indices matches the gathered reference")
    func selectedPoolAttentionParity() {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }

        let (rows, ratio, offset, D, W) = (
            Self.parityRows, Self.parityRatio, Self.parityOffset,
            Self.parityDim, Self.parityWindow)
        let (H, S, K) = (4, 5, 7)
        let storage = Self.q8Storage(rows: rows, dim: D)
        let view = DeepseekV4PoolView(storage: storage)
        let q = Self.continuous([1, H, S, D], step: 0.311, bias: 0.5)
        let localKV = Self.continuous([1, 1, W, D], step: 0.157, bias: 0.2)
        let sinks = Self.continuous([H], step: 0.703)
        let scale = 1.0 / Float(D).squareRoot()

        // Fixed indices remove all top-k tie sensitivity: visible rows,
        // duplicates, and one causally INVISIBLE row per query (35_000 + s
        // fails `(k+1)*ratio <= q+1`) so mask parity is exercised too.
        var raw: [Int32] = []
        for s in 0..<S {
            for j in 0..<(K - 1) {
                raw.append(Int32((s * 4999 + j * 613) % 34_000))
            }
            raw.append(Int32(35_000 + s))
        }
        let topk = MLXArray(raw).reshaped(1, S, K)

        let tiled = DeepseekV4Math.tiledPoolAttention(
            queries: q, localKV: localKV, pooled: view,
            offset: offset, window: W, ratio: ratio,
            scale: scale, sinks: sinks, topK: topk)

        let pool = view.materialized().asType(.float32).squeezed(axis: 0)
        let sel = take(pool, topk.reshaped(-1), axis: 0).reshaped(1, S, K, D)
        let q32 = q.asType(.float32)
        var selScores = einsum(
            "bhqd,bqkd->bhqk", q32 * MLXArray(scale), sel)
        let qPos = MLXArray(Int32(offset)..<Int32(offset + S)).reshaped(1, S, 1)
        let visible =
            ((topk + MLXArray(Int32(1))) * MLXArray(Int32(ratio)))
            .<= (qPos + MLXArray(Int32(1)))
        selScores = MLX.where(
            visible.expandedDimensions(axis: 1), selScores,
            MLXArray(-Float.infinity))
        let local32 = localKV.asType(.float32)
        var localScores = (q32 * MLXArray(scale)).matmul(
            local32.swappedAxes(-1, -2))
        localScores = MLX.where(
            DeepseekV4Math.buildWindowMask(
                batch: 1, queryLen: S, offset: offset,
                window: W, windowLen: W),
            localScores, MLXArray(-Float.infinity))
        let sinkCol = broadcast(
            sinks.asType(.float32).reshaped(1, H, 1, 1), to: [1, H, S, 1])
        let weights = softmax(
            concatenated([localScores, selScores, sinkCol], axis: -1), axis: -1)
        let reference =
            weights[.ellipsis, 0..<W].matmul(local32)
            + einsum(
                "bhqk,bqkd->bhqd", weights[.ellipsis, W..<(W + K)], sel)
        let diff = MLX.abs(tiled.asType(.float32) - reference).max()
        MLX.eval(diff)
        #expect(diff.item(Float.self) < 1e-3)
    }

    @Test("tiled index top-k selects the same score mass as dense argpartition")
    func tiledIndexTopkParity() {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }

        let (rows, ratio, offset, D) = (
            Self.parityRows, Self.parityRatio, Self.parityOffset, Self.parityDim)
        let (H, S, topK) = (2, 5, 11)
        let storage = Self.q8Storage(rows: rows, dim: D)
        let view = DeepseekV4PoolView(storage: storage)
        let q = Self.continuous([1, H, S, D], step: 0.311, bias: 0.5)
        let wRaw = Self.continuous([1, S, H], step: 0.507, bias: 0.4)
        let scale = 1.0 / Float(D).squareRoot()

        let tiledIdx = DeepseekV4Math.tiledIndexTopk(
            queries: q, headWeights: wRaw, pooled: view,
            scale: scale, topK: topK, offset: offset, ratio: ratio)
        #expect(tiledIdx.shape == [1, S, topK])

        // Dense reference over the SAME dequantized values.
        let pool = view.materialized().asType(.float32)
        var scores = q.asType(.float32).matmul(
            pool.expandedDimensions(axis: 1).swappedAxes(-1, -2))
        scores = maximum(scores, MLXArray(Float(0))) * MLXArray(scale)
        let wExp = wRaw.asType(.float32).swappedAxes(-1, -2)
            .expandedDimensions(axis: -1)
        var reduced = (scores * wExp).sum(axis: 1)
        reduced = MLX.where(
            DeepseekV4Math.indexVisibility(
                batch: 1, seqLen: S, offset: offset,
                rowStart: 0, rowCount: rows, ratio: ratio),
            reduced, MLXArray(-Float.infinity, dtype: reduced.dtype))
        let denseIdx = argPartition(-reduced, kth: topK - 1, axis: -1)[
            .ellipsis, 0..<topK]

        // Compare the SELECTED SCORE multisets per query (tie-robust: any
        // tie-break choosing different indices still selects equal scores).
        let tiledScores = takeAlong(reduced, tiledIdx.asType(.int64), axis: -1)
        let denseScores = takeAlong(reduced, denseIdx, axis: -1)
        let tiledSorted = sorted(tiledScores, axis: -1)
        let denseSorted = sorted(denseScores, axis: -1)
        let diff = MLX.abs(tiledSorted - denseSorted).max()
        MLX.eval(diff)
        #expect(diff.item(Float.self) < 1e-4)
    }

    @Test("gather decodes exactly the selected q8 rows")
    func gatherSelectedRowsParity() {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }

        let rows = 20_000
        let D = Self.parityDim
        let storage = Self.q8Storage(rows: rows, dim: D)
        #expect(storage.quantizedSegments.count > 1)
        let view = DeepseekV4PoolView(storage: storage)
        // Segment boundaries, duplicates, first and last rows.
        let raw: [Int32] = [
            0, 16_383, 16_384, Int32(rows - 1), 42, 42,
            9_999, 16_385, 1, Int32(rows - 2), 12_345, 0,
        ]
        let indices = MLXArray(raw).reshaped(1, 3, 4)
        let gathered = view.gatherDequantizedRows(indices)
        #expect(gathered.shape == [1, 3, 4, D])
        let reference = take(
            view.materialized().squeezed(axis: 0), MLXArray(raw), axis: 0
        ).reshaped(1, 3, 4, D)
        let exact = MLX.allClose(gathered, reference, rtol: 0, atol: 0)
        MLX.eval(exact)
        #expect(exact.item(Bool.self))
    }

    // MARK: - Heads-16 indexed prefill kernel

    private static func heads16Values(
        _ count: Int, frequency: Float, phase: Float
    ) -> MLXArray {
        MLX.sin(MLXArray(0..<count).asType(.float32) * frequency + phase)
            * Float(0.3)
    }

    @Test("heads16 kernel self-test passes (bf16 + f16 vs fp32 reference)")
    func heads16SelfTestPasses() {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }
        #expect(DeepseekV4Math.heads16SelfTest() == nil)
    }

    @Test("heads16 kernel matches reference on a wider non-self-test shape")
    func heads16KernelWiderShapeParity() {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }

        let seqLen = 24
        let rows = 48
        let poolRows = 64
        let k = 12
        let offset = 200
        let window = 32
        let ratio = 4
        let scale = Float(pow(Double(512), -0.5))
        // Distinct per query: stride 5 is coprime with 64.
        var topkValues = [Int32]()
        for t in 0..<seqLen {
            for i in 0..<k {
                topkValues.append(Int32((t * 7 + i * 5) % poolRows))
            }
        }
        let topk2d = MLXArray(topkValues).reshaped(seqLen, k)
        let sinks32 = Self.heads16Values(64, frequency: 0.83, phase: 1.7)
        for dtype in [DType.bfloat16, DType.float16] {
            let q = Self.heads16Values(
                64 * seqLen * 512, frequency: 0.37, phase: 0.1
            ).reshaped(1, 64, seqLen, 512).asType(dtype)
            let kv2d = Self.heads16Values(
                rows * 512, frequency: 0.53, phase: 0.9
            ).reshaped(rows, 512).asType(dtype)
            let pool2d = Self.heads16Values(
                poolRows * 512, frequency: 0.71, phase: 2.3
            ).reshaped(poolRows, 512).asType(dtype)
            let got = DeepseekV4Math.heads16RunKernel(
                q: q, kv2d: kv2d, pool2d: pool2d, topk2d: topk2d,
                sinks32: sinks32,
                offset: offset, window: window, ratio: ratio, scale: scale
            ).asType(.float32)
            let ref = DeepseekV4Math.heads16Reference(
                q: q, kv2d: kv2d, pool2d: pool2d, topk2d: topk2d,
                sinks32: sinks32,
                offset: offset, window: window, ratio: ratio, scale: scale
            ).asType(.float32)
            MLX.eval(got, ref)
            let denom = max(MLX.abs(ref).max().item(Float.self), Float(1e-6))
            let rel = MLX.abs(got - ref).max().item(Float.self) / denom
            #expect(rel.isFinite && rel <= 2.5e-2, "rel \(rel) dtype \(dtype)")
        }
    }

    @Test("heads16 matches reference at cold-prefill production shape")
    func heads16ColdPrefillProductionShapeParity() {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }

        // The other heads16 cases all run offset > 0 with a pool that is
        // almost entirely causally visible, and shapes (seq <= 24, window
        // <= 32) far below anything DSV4-Flash decodes. A cold prefill is
        // the opposite regime: offset == 0, window 128 from the bundle, and
        // early queries where EVERY selected pool row is causally invisible
        // so the kernel's threadgroup-uniform `continue` carries the result.
        let seqLen = 512
        let rows = 512
        let poolRows = 128
        let k = 96
        let window = 128
        let ratio = 4
        let scale = Float(pow(Double(512), -0.5))
        var topkValues = [Int32]()
        for t in 0..<seqLen {
            for i in 0..<k {
                topkValues.append(Int32((t * 13 + i * 11) % poolRows))
            }
        }
        let topk2d = MLXArray(topkValues).reshaped(seqLen, k)
        let sinks32 = Self.heads16Values(64, frequency: 0.61, phase: 0.3)
        for offset in [0, 2048] {
            let q = Self.heads16Values(
                64 * seqLen * 512, frequency: 0.29, phase: 0.7
            ).reshaped(1, 64, seqLen, 512).asType(.bfloat16)
            let kv2d = Self.heads16Values(
                rows * 512, frequency: 0.47, phase: 1.3
            ).reshaped(rows, 512).asType(.bfloat16)
            let pool2d = Self.heads16Values(
                poolRows * 512, frequency: 0.73, phase: 2.1
            ).reshaped(poolRows, 512).asType(.bfloat16)
            let got = DeepseekV4Math.heads16RunKernel(
                q: q, kv2d: kv2d, pool2d: pool2d, topk2d: topk2d,
                sinks32: sinks32,
                offset: offset, window: window, ratio: ratio, scale: scale
            ).asType(.float32)
            let ref = DeepseekV4Math.heads16Reference(
                q: q, kv2d: kv2d, pool2d: pool2d, topk2d: topk2d,
                sinks32: sinks32,
                offset: offset, window: window, ratio: ratio, scale: scale
            ).asType(.float32)
            MLX.eval(got, ref)
            let denom = max(MLX.abs(ref).max().item(Float.self), Float(1e-6))
            let rel = MLX.abs(got - ref).max().item(Float.self) / denom
            #expect(rel.isFinite && rel <= 2.5e-2, "rel \(rel) offset \(offset)")
        }
    }

    @Test("heads16 is bit-stable across repeated runs at production shape")
    func heads16RepeatedRunsAreStable() {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }

        // Live DSV4 produced token soup on one cold prefill and clean prose
        // on the next with an identical prompt and build. MLX ops are
        // deterministic, so a run-to-run difference here would localize that
        // to the kernel's threadgroup staging rather than to sampling.
        let seqLen = 512
        let rows = 512
        let poolRows = 128
        let k = 96
        var topkValues = [Int32]()
        for t in 0..<seqLen {
            for i in 0..<k {
                topkValues.append(Int32((t * 13 + i * 11) % poolRows))
            }
        }
        let topk2d = MLXArray(topkValues).reshaped(seqLen, k)
        let sinks32 = Self.heads16Values(64, frequency: 0.61, phase: 0.3)
        let q = Self.heads16Values(
            64 * seqLen * 512, frequency: 0.29, phase: 0.7
        ).reshaped(1, 64, seqLen, 512).asType(.bfloat16)
        let kv2d = Self.heads16Values(
            rows * 512, frequency: 0.47, phase: 1.3
        ).reshaped(rows, 512).asType(.bfloat16)
        let pool2d = Self.heads16Values(
            poolRows * 512, frequency: 0.73, phase: 2.1
        ).reshaped(poolRows, 512).asType(.bfloat16)

        func run() -> MLXArray {
            DeepseekV4Math.heads16RunKernel(
                q: q, kv2d: kv2d, pool2d: pool2d, topk2d: topk2d,
                sinks32: sinks32,
                offset: 0, window: 128, ratio: 4,
                scale: Float(pow(Double(512), -0.5)))
        }
        let baseline = run()
        MLX.eval(baseline)
        for attempt in 1...8 {
            let again = run()
            MLX.eval(again)
            let identical = MLX.all(again .== baseline)
            MLX.eval(identical)
            #expect(
                identical.item(Bool.self),
                "attempt \(attempt) diverged from the first run")
        }
    }

    @Test("heads16 entry path reshapes/casts correctly and matches reference")
    func heads16EntryMatchesReference() {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }

        let seqLen = 16
        let rows = 40
        let poolRows = 48
        let k = 8
        let offset = 120
        let window = 24
        let ratio = 4
        let scale = Float(pow(Double(512), -0.5))
        var topkValues = [Int32]()
        for t in 0..<seqLen {
            for i in 0..<k {
                topkValues.append(Int32((t * 11 + i * 7) % poolRows))
            }
        }
        let q = Self.heads16Values(
            64 * seqLen * 512, frequency: 0.41, phase: 0.4
        ).reshaped(1, 64, seqLen, 512).asType(.bfloat16)
        // Entry contract shapes: localKV (B, 1, rows, D), pooled dense
        // (B, W, D) in fp32 (entry casts to q dtype), topk (1, S, K).
        let localKV = Self.heads16Values(
            rows * 512, frequency: 0.59, phase: 1.1
        ).reshaped(1, 1, rows, 512).asType(.bfloat16)
        let pooled = Self.heads16Values(
            poolRows * 512, frequency: 0.67, phase: 2.9
        ).reshaped(1, poolRows, 512)
        let topK = MLXArray(topkValues).reshaped(1, seqLen, k)
        let sinks = Self.heads16Values(64, frequency: 0.79, phase: 0.6)
            .asType(.bfloat16)

        let out = DeepseekV4Math.heads16PrefillAttention(
            queries: q, localKV: localKV, pooled: pooled, topK: topK,
            offset: offset, window: window, ratio: ratio, scale: scale,
            sinks: sinks)
        #expect(out != nil)
        guard let out else { return }
        #expect(out.shape == [1, 64, seqLen, 512])

        let ref = DeepseekV4Math.heads16Reference(
            q: q,
            kv2d: localKV.reshaped(rows, 512),
            pool2d: pooled.reshaped(poolRows, 512).asType(q.dtype),
            topk2d: topK.reshaped(seqLen, k),
            sinks32: sinks.asType(.float32),
            offset: offset, window: window, ratio: ratio, scale: scale)
        let got32 = out.asType(.float32)
        let ref32 = ref.asType(.float32)
        MLX.eval(got32, ref32)
        let denom = max(MLX.abs(ref32).max().item(Float.self), Float(1e-6))
        let rel = MLX.abs(got32 - ref32).max().item(Float.self) / denom
        #expect(rel.isFinite && rel <= 2.5e-2, "rel \(rel)")
    }

    @Test("heads16 entry declines out-of-contract layouts")
    func heads16EntryContractFallbacks() {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }

        let q = MLXArray.zeros([1, 64, 4, 512], dtype: .bfloat16)
        let localKV = MLXArray.zeros([1, 1, 8, 512], dtype: .bfloat16)
        let pooled = MLXArray.zeros([1, 16, 512], dtype: .bfloat16)
        let topK = MLXArray.zeros([1, 4, 8], dtype: .int32)
        let sinks = MLXArray.zeros([64], dtype: .bfloat16)

        // Missing sinks → stock path.
        #expect(
            DeepseekV4Math.heads16PrefillAttention(
                queries: q, localKV: localKV, pooled: pooled, topK: topK,
                offset: 0, window: 128, ratio: 4, scale: 1.0, sinks: nil)
                == nil)
        // Non-DSV4 head count → stock path.
        #expect(
            DeepseekV4Math.heads16PrefillAttention(
                queries: MLXArray.zeros([1, 32, 4, 512], dtype: .bfloat16),
                localKV: localKV, pooled: pooled, topK: topK,
                offset: 0, window: 128, ratio: 4, scale: 1.0, sinks: sinks)
                == nil)
        // fp32 queries → stock path.
        #expect(
            DeepseekV4Math.heads16PrefillAttention(
                queries: MLXArray.zeros([1, 64, 4, 512], dtype: .float32),
                localKV: localKV, pooled: pooled, topK: topK,
                offset: 0, window: 128, ratio: 4, scale: 1.0, sinks: sinks)
                == nil)
        // Topk seq-length mismatch → stock path.
        #expect(
            DeepseekV4Math.heads16PrefillAttention(
                queries: q, localKV: localKV, pooled: pooled,
                topK: MLXArray.zeros([1, 3, 8], dtype: .int32),
                offset: 0, window: 128, ratio: 4, scale: 1.0, sinks: sinks)
                == nil)
        // Empty local KV → stock path.
        #expect(
            DeepseekV4Math.heads16PrefillAttention(
                queries: q,
                localKV: MLXArray.zeros([1, 1, 0, 512], dtype: .bfloat16),
                pooled: pooled, topK: topK,
                offset: 0, window: 128, ratio: 4, scale: 1.0, sinks: sinks)
                == nil)
    }

    @Test("strided RoPE table matches the manual angle build and memoizes")
    func stridedCosSinParity() {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }

        let rope = DeepseekV4RoPE(
            dim: 64, base: 160000, factor: 40, origMaxPos: 4096)
        let base = 3712
        let count = 17
        let stride = 4

        let (gotCos, gotSin) = rope.cosSin(
            base: base, count: count, stride: stride)

        let positions =
            MLXArray(Int32(0)..<Int32(count)).asType(.float32) * Float(stride)
            + Float(base)
        let angles =
            positions.expandedDimensions(axis: -1)
            * rope.invFreq.expandedDimensions(axis: 0)
        let refCos = MLX.cos(angles)
        let refSin = MLX.sin(angles)

        #expect(gotCos.shape == [count, 32])
        #expect(MLX.allClose(gotCos, refCos).item(Bool.self))
        #expect(MLX.allClose(gotSin, refSin).item(Bool.self))

        // Memoized: an equal-frequency sibling instance asking for the same
        // strided window must get the cached arrays back.
        let sibling = DeepseekV4RoPE(
            dim: 64, base: 160000, factor: 40, origMaxPos: 4096)
        let (again, _) = sibling.cosSin(base: base, count: count, stride: stride)
        #expect(again === gotCos)

        // A different stride must not evict or collide.
        let (other, _) = rope.cosSin(base: base, count: count, stride: 128)
        #expect(other !== gotCos)
        let (backAgain, _) = rope.cosSin(base: base, count: count, stride: stride)
        #expect(backAgain === gotCos)
    }

    private static func signal(rows: Int, features: Int) -> MLXArray {
        let count = rows * features
        return MLX.sin(MLXArray(0..<count).asType(.float32) * Float(0.013))
            .reshaped(1, rows, features)
            .asType(.bfloat16)
    }

    private static func cosine(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let a = lhs.asType(.float32).reshaped(-1)
        let b = rhs.asType(.float32).reshaped(-1)
        let numerator = (a * b).sum()
        let aa = (a * a).sum()
        let bb = (b * b).sum()
        MLX.eval([numerator, aa, bb])
        return numerator.item(Float.self)
            / max(sqrt(aa.item(Float.self) * bb.item(Float.self)), 1e-9)
    }

    private static func fillRotating(_ rotating: RotatingKVCache) {
        let keys = MLXArray.ones([1, 1, 5, 8], dtype: .bfloat16)
        let values = MLXArray.ones([1, 1, 5, 8], dtype: .bfloat16) * Float(2)
        _ = rotating.update(keys: keys, values: values)
    }
}
