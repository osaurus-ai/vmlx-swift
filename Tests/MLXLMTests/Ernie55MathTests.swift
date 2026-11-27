import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

@Suite("ERNIE 5.5 numerical and HCA cache contract", .serialized)
struct Ernie55MathTests {
    @Test("W4A8 rounding uses independent power-of-two E4M3 scales")
    func expertActivationRounding() throws {
        try MLXMetalTestLock.withLock {
            let input = MLXArray(
                [Float(400), 0.002, 0.001, -0.002] + Array(repeating: Float(0), count: 124)
                    + [Float(0.78125), 0.001, -0.001] + Array(repeating: Float(0), count: 125),
                [1, 256])
            let result = Ernie55Math.roundExpertActivations(input)
            #expect(result.shape == [1, 256])
            #expect(result[0, 0].item(Float.self) == 384)
            #expect(result[0, 1].item(Float.self) == 0.001953125)
            #expect(result[0, 2].item(Float.self) == 0.001953125)
            #expect(result[0, 3].item(Float.self) == -0.001953125)
            #expect(result[0, 128].item(Float.self) == 0.75)
            #expect(abs(result[0, 129].item(Float.self) - 0.001) < 0.0001)
            #expect(result[0, 130].item(Float.self) == -result[0, 129].item(Float.self))
        }
    }

    @Test("Fused expert activation matches the reference across E4M3 ties and scales")
    func batchedExpertActivationRounding() throws {
        try MLXMetalTestLock.withLock {
            let exponents = -24...18
            let samples = exponents.flatMap { exponent in
                (0..<128).map { position -> Float in
                    let value = Float(position - 64) * (1.0 / 64.0)
                    return value * Float(sign: .plus, exponent: exponent, significand: 448)
                }
            }
            for dtype in [DType.float32, .bfloat16] {
                let input = MLXArray(samples, [exponents.count, 128]).asType(dtype)
                let expected = Ernie55Math.roundExpertActivationsGraph(input)
                let actual = Ernie55Math.roundExpertActivations(input)
                #expect(arrayEqual(actual, expected).item(Bool.self))
            }
        }
    }

    @Test("HCA windows cross a chunk and cache copy without shifting rotary positions")
    func hcaChunkBoundary() throws {
        try MLXMetalTestLock.withLock {
            let ratio = 128
            let compressor = Ernie55HCACompressor(hiddenSize: 8, headDimension: 8, ratio: ratio, epsilon: 1e-6)
            let rope = DeepseekV4RoPE(dim: 4, base: 160_000, factor: 16)
            let samples: [Float] = (0 ..< 2104).map { index in
                let centered = (index * 17) % 113 - 56
                return Float(centered) / 128
            }
            let values = MLXArray(samples, [1, 263, 8])
            let expected = compressor(values, rope: rope, cache: nil, start: 0)
            #expect(expected.shape == [1, 2, 8])

            let cache = DeepseekV4Cache(slidingWindow: 128, compressRatio: ratio, poolQuantizationEnabled: false)
            let before = compressor(values[0..., ..<127, 0...], rope: rope, cache: cache, start: 0)
            #expect(before.dim(1) == 0)
            #expect(cache.getBuffers(.compressor).kv?.shape == [1, 127, 8])
            #expect(cache.getBuffers(.compressor).kv?.dtype == .float32)

            let restored = try #require(cache.copy() as? DeepseekV4Cache)
            let first = compressor(values[0..., 127 ..< 128, 0...], rope: rope, cache: restored, start: 127)
            #expect(first.dim(1) == 1)
            #expect(restored.getBuffers(.compressor).kv == nil)
            let actual = compressor(values[0..., 128..., 0...], rope: rope, cache: restored, start: 128)
            #expect(restored.getBuffers(.compressor).kv?.shape == [1, 7, 8])
            #expect(actual.shape == expected.shape)
            #expect(max(abs(actual - expected)).item(Float.self) < 1e-5)
        }
    }

    @Test("Disk serializer round-trips HCA local and pending state with MLA and index companions")
    func hybridDiskRoundTrip() throws {
        try MLXMetalTestLock.withLock {
            let hca = DeepseekV4Cache(
                slidingWindow: 128, compressRatio: 128, poolQuantizationEnabled: false, compactLocalWindow: true,
                sharesLocalKeysAndValues: true)
            let latent = KVCacheSimple()
            let index = KVCacheSimple()
            let length = 129
            let kv = MLXArray.ones([1, 1, length, 8], dtype: .bfloat16)
            _ = hca.updateSharedKV(kv)
            hca.setPooled(.compressor, value: MLXArray.ones([1, 1, 8], dtype: .bfloat16) * 3)
            hca.setBuffers(
                .compressor, kv: MLXArray.ones([1, 1, 8]) * 2.5, gate: MLXArray.ones([1, 1, 8]) * -0.125)
            _ = latent.update(keys: kv * 2, values: kv[.ellipsis, ..<4])
            _ = index.update(keys: kv * 7, values: MLXArray.zeros([1, 1, length, 0], dtype: .bfloat16))
            let arrays = TQDiskSerializer.serialize(
                cache: [hca, CacheList(latent, index)], preserveStandardKVStorageDType: true)
            #expect(arrays["dsv4_0_values"] == nil)
            #expect(arrays["__dsv4_0_shared_local__"] != nil)
            let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
            defer { try? FileManager.default.removeItem(at: url) }
            try MLX.save(arrays: arrays, url: url)
            let (onDisk, _) = try MLX.loadArraysAndMetadata(url: url)
            let hca2 = DeepseekV4Cache(
                slidingWindow: 128, compressRatio: 128, poolQuantizationEnabled: false, compactLocalWindow: true,
                sharesLocalKeysAndValues: true)
            let mla2 = CacheList(KVCacheSimple(), KVCacheSimple())
            var fresh: [any KVCache] = [hca2, mla2]
            #expect(restoreFromDiskArrays(onDisk, into: &fresh, requirePromptBoundary: true) == length)
            #expect(hca2.offset == length)
            #expect(hca2.local.state[0].dim(2) == 128)
            #expect(arrayEqual(hca2.local.state[0], hca2.local.state[1]).item(Bool.self))
            #expect(hca2.getBuffers(.compressor).kv?.dtype == .float32)
            #expect(hca2.getBuffers(.compressor).gate?.asArray(Float.self) == Array(repeating: -0.125, count: 8))
            #expect(mla2[0].offset == length)
            #expect(mla2[1].offset == length)
            #expect(mla2[1].state.last?.dim(-1) == 0)
        }
    }

    @Test("HCA rows become visible exactly at the completed window")
    func completedWindowVisibility() {
        let mask = Ernie55Math.hcaPoolMask(start: 126, length: 3, columns: 2, ratio: 128)
        #expect(mask.shape == [3, 2])
        #expect(mask.asArray(Bool.self) == [false, false, true, false, true, false])
    }

    @Test("Fused ERNIE Metal kernels match reference ops for R=1 and R=7")
    func fusedKernelsParity() throws {
        try MLXMetalTestLock.withLock {
            MLXRandom.seed(42)
            // 0. compressorProject tiled path (M=37, K=256, N=64)
            let tx = MLXRandom.normal([1, 37, 256]).asType(.bfloat16)
            let twk = MLXRandom.normal([64, 256]).asType(.bfloat16) * 0.1
            let twg = MLXRandom.normal([64, 256]).asType(.bfloat16) * 0.1
            let (tk, tg) = Ernie55Kernels.compressorProject(tx, keyWeight: twk, gateWeight: twg)
            let refTK = matmul(tx.asType(.float32), twk.asType(.float32).transposed())
            let refTG = matmul(tx.asType(.float32), twg.asType(.float32).transposed())
            #expect(max(abs(tk - refTK)).item(Float.self) < 1e-3)
            #expect(max(abs(tg - refTG)).item(Float.self) < 1e-3)

            for rows in [1, 7] {
                // 1. compressorProject fast path (K=256, N=32)
                let cx = MLXRandom.normal([1, rows, 256]).asType(.bfloat16)
                let wk = MLXRandom.normal([32, 256]).asType(.bfloat16) * 0.1
                let wg = MLXRandom.normal([32, 256]).asType(.bfloat16) * 0.1
                let (ck, cg) = Ernie55Kernels.compressorProject(cx, keyWeight: wk, gateWeight: wg)
                let refK = matmul(cx.asType(.float32), wk.asType(.float32).transposed())
                let refG = matmul(cx.asType(.float32), wg.asType(.float32).transposed())
                #expect(max(abs(ck - refK)).item(Float.self) < 1e-3)
                #expect(max(abs(cg - refG)).item(Float.self) < 1e-3)

                // 2. fusedHCCollapseNorm (without and with prevOut)
                let h = MLXRandom.normal([1, rows, 4, 4096]).asType(.bfloat16) * 0.5
                let fn = MLXRandom.normal([24, 16384]).asType(.float32) * 0.01
                let scale = MLXArray([Float(0.1), 0.1, 0.1])
                let base = MLXRandom.normal([24]).asType(.float32) * 0.1
                let nw = (MLXRandom.normal([4096]) * 0.1 + 1.0).asType(.bfloat16)
                let eps: Float = 1e-6
                let fused0 = try #require(Ernie55Kernels.fusedHCCollapseNorm(
                    h: h, fn: fn, scale: scale, base: base, normWeight: nw,
                    iterations: 20, mixingEpsilon: 1e-6, normEpsilon: 1e-4, rmsEpsilon: eps))
                let ref0 = Ernie55Math.collapse(
                    h, projection: fn, scales: scale, bias: base,
                    iterations: 20, mixingEpsilon: 1e-6, normEpsilon: 1e-4)
                let refNorm0 = MLXFast.rmsNorm(ref0.hidden, weight: nw, eps: eps)
                #expect(max(abs(fused0.normed.asType(.float32) - refNorm0.asType(.float32))).item(Float.self) < 5e-2)
                #expect(max(abs(fused0.post - ref0.post)).item(Float.self) < 1e-4)
                #expect(max(abs(fused0.comb - ref0.residualMix)).item(Float.self) < 1e-4)

                let batched0 = try #require(Ernie55Kernels.batchedHCCollapseNorm(
                    h: h, fn: fn, scale: scale, base: base, normWeight: nw,
                    iterations: 20, mixingEpsilon: 1e-6, normEpsilon: 1e-4, rmsEpsilon: eps))
                #expect(max(abs(batched0.normed.asType(.float32) - refNorm0.asType(.float32))).item(Float.self) < 5e-2)
                #expect(max(abs(batched0.post - ref0.post)).item(Float.self) < 2e-3)
                #expect(max(abs(batched0.comb - ref0.residualMix)).item(Float.self) < 2e-3)

                let prevOut = MLXRandom.normal([1, rows, 4096]).asType(.bfloat16) * 0.25
                let fused1 = try #require(Ernie55Kernels.fusedHCCollapseNorm(
                    h: h, prevOut: prevOut, prevPost: ref0.post, prevComb: ref0.residualMix,
                    fn: fn, scale: scale, base: base, normWeight: nw,
                    iterations: 20, mixingEpsilon: 1e-6, normEpsilon: 1e-4, rmsEpsilon: eps))
                let refStream = Ernie55Math.expand(prevOut, residual: h, post: ref0.post, mix: ref0.residualMix)
                let ref1 = Ernie55Math.collapse(
                    refStream, projection: fn, scales: scale, bias: base,
                    iterations: 20, mixingEpsilon: 1e-6, normEpsilon: 1e-4)
                let refNorm1 = MLXFast.rmsNorm(ref1.hidden, weight: nw, eps: eps)
                #expect(max(abs(fused1.stream.asType(.float32) - refStream.asType(.float32))).item(Float.self) < 4e-2)
                #expect(max(abs(fused1.normed.asType(.float32) - refNorm1.asType(.float32))).item(Float.self) < 5e-2)

                // 3. routeFused (512 experts, K=256, top-10)
                let rx = MLXRandom.normal([1, rows, 256]).asType(.bfloat16) * 0.2
                let r1 = MLXRandom.normal([512, 256]).asType(.bfloat16) * 0.2
                let r2 = MLXRandom.normal([512, 256]).asType(.bfloat16) * 0.2
                let corr = MLXRandom.normal([512]).asType(.float32) * 0.05
                let es = MLXArray.ones([512], dtype: .float32) * 2.5
                let rf = try #require(Ernie55Kernels.routeFused(
                    rx, first: r1, second: r2, correction: corr, expertScale: es, count: 10, normalize: true))
                let refScores = sigmoid(matmul(rx.asType(.float32), r1.asType(.float32).transposed()))
                    + sigmoid(matmul(rx.asType(.float32), r2.asType(.float32).transposed()))
                let refOrder = argSort(-(refScores + corr), axis: -1)[.ellipsis, ..<10]
                let refPicked = takeAlong(refScores, refOrder, axis: -1)
                let refWeights = (refPicked / maximum(refPicked.sum(axis: -1, keepDims: true), 1e-20))
                    * take(es, refOrder)
                #expect(arrayEqual(rf.indices, refOrder.asType(.uint32)).item(Bool.self))
                #expect(max(abs(rf.weights - refWeights)).item(Float.self) < 1e-4)

                // 4. lmHeadMultiMax (V=257, K=256)
                let lx = MLXRandom.normal([1, rows, 256]).asType(.bfloat16) * 0.2
                let lw = MLXRandom.normal([257, 256]).asType(.bfloat16) * 0.2
                let ranges = MLXArray([Float(-10), 10, -20, 20])
                let slopes = MLXArray([Float(0.1), -0.1, 0.01, -0.01])
                let lf = try #require(Ernie55Kernels.lmHeadMultiMax(lx, weight: lw, ranges: ranges, slopes: slopes))
                let lRef = Ernie55Math.multiMax(
                    matmul(lx.asType(.float32), lw.asType(.float32).transposed()),
                    ranges: ranges, slopes: slopes)
                #expect(max(abs(lf - lRef)).item(Float.self) < 1e-3)

                // 5. invRoPEVHAGroup + prepareHCARows
                let rope = DeepseekV4RoPE(dim: 64, base: 160_000, factor: 16)
                let phase = rope.cosSin(offset: 13, length: rows)
                let attended = MLXRandom.normal([1, 64, rows, 512]).asType(.bfloat16) * 0.5
                let vhaMix = (eye(64) + MLXRandom.normal([64, 64]) * 0.02).asType(.bfloat16)
                let gf = try #require(Ernie55Kernels.invRoPEVHAGroup(
                    attended, cos: phase.cos, sin: phase.sin, vhaMix: vhaMix, ropeWidth: 64))
                let unrot = DeepseekV4Math.applyPartialRoPE(
                    attended, cos: phase.cos, sin: -phase.sin, ropeDim: 64)
                let refVha = matmul(vhaMix, unrot.swappedAxes(1, 2))
                    .reshaped(rows, 8, 4096).swappedAxes(0, 1)
                #expect(max(abs(gf.asType(.float32) - refVha.asType(.float32))).item(Float.self) < 4e-2)

                let qIn = MLXRandom.normal([1, rows, 64, 512]).asType(.bfloat16)
                let kIn = MLXRandom.normal([1, rows, 512]).asType(.bfloat16)
                let kw = (MLXRandom.normal([512]) * 0.1 + 1.0).asType(.bfloat16)
                let (qf, kf) = try #require(Ernie55Kernels.prepareHCARows(
                    qIn, rawKV: kIn, keyWeight: kw, cos: phase.cos, sin: phase.sin,
                    heads: 64, ropeWidth: 64, epsilon: eps))
                let qRef = DeepseekV4Math.applyPartialRoPE(
                    MLXFast.rmsNorm(qIn, weight: MLXArray.ones([512], dtype: .bfloat16), eps: eps).swappedAxes(1, 2),
                    cos: phase.cos, sin: phase.sin, ropeDim: 64)
                let kRef = DeepseekV4Math.applyPartialRoPE(
                    MLXFast.rmsNorm(kIn, weight: kw, eps: eps).expandedDimensions(axis: 1),
                    cos: phase.cos, sin: phase.sin, ropeDim: 64)
                #expect(max(abs(qf.asType(.float32) - qRef.asType(.float32))).item(Float.self) < 2e-2)
                #expect(max(abs(kf.asType(.float32) - kRef.asType(.float32))).item(Float.self) < 2e-2)

                // 6. fusedSiTU
                let sg = MLXRandom.normal([1, rows, 256]).asType(.bfloat16) * 2
                let su = MLXRandom.normal([1, rows, 256]).asType(.bfloat16) * 2
                let sf = try #require(Ernie55Kernels.fusedSiTU(sg, su, gateCap: 20, upCap: 20))
                let sRef = (20 * tanh(sg.asType(.float32) / 20) * sigmoid(sg.asType(.float32))
                    * (20 * tanh(su.asType(.float32) / 20))).asType(.bfloat16)
                #expect(max(abs(sf.asType(.float32) - sRef.asType(.float32))).item(Float.self) < 2e-2)
            }
        }
    }

    @Test("CacheList supports in-place speculative checkpoint/restore and ERNIE DSpark discovery")
    func dsparkCheckpointAndDiscovery() throws {
        try MLXMetalTestLock.withLock {
            let latent = KVCacheSimple()
            let index = KVCacheSimple()
            let list = CacheList(latent, index)
            let k1 = MLXArray.ones([1, 1, 3, 8], dtype: .bfloat16)
            let v1 = MLXArray.ones([1, 1, 3, 4], dtype: .bfloat16)
            _ = latent.update(keys: k1, values: v1)
            _ = index.update(keys: k1 * 2, values: MLXArray.zeros([1, 1, 3, 0], dtype: .bfloat16))
            #expect(SpeculativeCacheListCheckpoint.supports([list]))
            let ckpt = try #require(SpeculativeCacheListCheckpoint([list]))
            _ = latent.update(keys: k1 * 5, values: v1 * 5)
            _ = index.update(keys: k1 * 9, values: MLXArray.zeros([1, 1, 3, 0], dtype: .bfloat16))
            #expect(latent.offset == 6)
            #expect(index.offset == 6)
            ckpt.restore()
            #expect(latent.offset == 3)
            #expect(index.offset == 3)

            let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            let dsparkDir = tmp.appendingPathComponent("dspark", isDirectory: true)
            try FileManager.default.createDirectory(at: dsparkDir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: tmp) }
            try Data("{}".utf8).write(to: dsparkDir.appendingPathComponent("draft-manifest.json"))
            try Data().write(to: dsparkDir.appendingPathComponent("draft-head.safetensors"))
            try Data().write(to: dsparkDir.appendingPathComponent("draft-layer-0.safetensors"))
            let cfg = Data(#"{"model_type":"ernie5_5_moe"}"#.utf8)
            var settings = VMLXServerRuntimeSettings()
            #expect(settings.bundledDSparkDrafter(configData: cfg, bundleDirectory: tmp, environment: [:]) == dsparkDir)
            #expect(settings.bundledDSparkDrafter(configData: cfg, bundleDirectory: tmp, environment: ["VMLX_DSPARK": "0"]) == nil)
            settings.mtp.mode = .off
            #expect(settings.bundledDSparkDrafter(configData: cfg, bundleDirectory: tmp, environment: [:]) == nil)
        }
    }

    @Test("ERNIE hybrid HCA + dual-KV CacheList survives SSD disk cache round-trip with DSpark tail")
    func hybridCacheDiskRoundTrip() throws {
        try MLXMetalTestLock.withLock {
            let hca = DeepseekV4Cache(
                slidingWindow: 128, compressRatio: 128,
                poolQuantizationEnabled: false, compactLocalWindow: true, sharesLocalKeysAndValues: true)
            let kv = MLXRandom.normal([1, 1, 5, 16]).asType(.bfloat16)
            _ = hca.update(keys: kv, values: kv)
            hca.auxiliaryHiddenTail = HybridPoolHiddenTail(
                capacity: 32, start: 0, rows: MLXRandom.normal([5, 24]).asType(.bfloat16))

            let mlaLatent = KVCacheSimple()
            let mlaIndex = KVCacheSimple()
            let latentK = MLXRandom.normal([1, 1, 5, 32]).asType(.bfloat16)
            let latentV = MLXArray.zeros([1, 1, 5, 0], dtype: .bfloat16)
            let indexK = MLXRandom.normal([1, 1, 5, 12]).asType(.bfloat16)
            let indexV = MLXArray.zeros([1, 1, 5, 0], dtype: .bfloat16)
            _ = mlaLatent.update(keys: latentK, values: latentV)
            _ = mlaIndex.update(keys: indexK, values: indexV)
            let mla = CacheList(mlaLatent, mlaIndex)

            let src: [any KVCache] = [hca, mla]
            let serialized = TQDiskSerializer.serialize(cache: src)

            let dstHca = DeepseekV4Cache(
                slidingWindow: 128, compressRatio: 128,
                poolQuantizationEnabled: false, compactLocalWindow: true, sharesLocalKeysAndValues: true)
            let dstMla = CacheList(KVCacheSimple(), KVCacheSimple())
            var dst: [any KVCache] = [dstHca, dstMla]
            let restored = restoreFromDiskArrays(serialized, into: &dst)
            #expect(restored == 5)
            #expect(validateRestoredCacheBoundary(dst, matchedTokens: 5, restoredTokens: restored))
            #expect(dstMla[0].offset == 5)
            #expect(dstMla[1].offset == 5)
            #expect(dstMla[0].state[0].shape == [1, 1, 5, 32])
            #expect(dstMla[1].state[0].shape == [1, 1, 5, 12])
            #expect(max(abs(dstMla[0].state[0].asType(.float32) - latentK.asType(.float32))).item(Float.self) == 0)
            #expect(max(abs(dstMla[1].state[0].asType(.float32) - indexK.asType(.float32))).item(Float.self) == 0)
            let restoredTail = try #require(dstHca.auxiliaryHiddenTail?.joined())
            #expect(restoredTail.start == 0)
            #expect(restoredTail.rows.shape == [5, 24])
        }
    }

    @Test("Fused sparse MLA indexer scores and sparse absorbed MLA prefill match dense reference")
    func sparseMLALongContextPrefillMatchesReference() throws {
        try MLXMetalTestLock.withLock {
            MLXRandom.seed(123)
            let (B, H, T, S, D) = (1, 32, 16, 128, 128)
            let q = (MLXRandom.normal([B, H, T, D]) * 0.2).asType(.bfloat16)
            let k = (MLXRandom.normal([B, 1, S, D]) * 0.2).asType(.bfloat16)
            let w = (MLXRandom.normal([B, T, H]) * 0.1).asType(.float32)
            let gotScores = try #require(Ernie55Kernels.mlaIndexScores(q: q, k: k, weights: w))
            let refDots = maximum(matmul(q, k.swappedAxes(-1, -2)), 0).asType(.float32)
            let refScores = (refDots * w.transposed(0, 2, 1).expandedDimensions(axis: -1))
                .sum(axis: 1, keepDims: true)
            eval(gotScores, refScores)
            #expect(max(abs(gotScores - refScores)).item(Float.self) < 1e-2)

            // Sparse absorbed MLA prefill with per-query selected indices including -1 masked slots.
            let (NQ, latent, rope, M) = (16, 128, 64, 24)
            let scale: Float = 1 / sqrt(Float(latent + rope))
            let qAbs = (MLXRandom.normal([B, NQ, T, latent]) * 0.25).asType(.bfloat16)
            let qPos = (MLXRandom.normal([B, NQ, T, rope]) * 0.25).asType(.bfloat16)
            let kv = (MLXRandom.normal([B, 1, S, latent]) * 0.25).asType(.bfloat16)
            let kPos = (MLXRandom.normal([B, 1, S, rope]) * 0.25).asType(.bfloat16)
            let sinks = (MLXRandom.normal([NQ]) * 0.5).asType(.bfloat16)
            var idxValues: [Int32] = []
            idxValues.reserveCapacity(T * M)
            for t in 0 ..< T {
                for m in 0 ..< M {
                    idxValues.append(m < 2 && t < 3 ? -1 : Int32((t * 7 + m * 5) % S))
                }
            }
            let indices = MLXArray(idxValues, [B, 1, T, M])
            let gotAttn = try #require(Ernie55Kernels.sparseMLAPrefill(
                qAbs: qAbs, qPos: qPos, kv: kv, kPos: kPos,
                indices: indices, sinks: sinks, scale: scale))

            // Reference: explicit per-row gathered SDPA over the exact same indices.
            var refRows: [MLXArray] = []
            for t in 0 ..< T {
                let rowIdx = indices[0, 0, t]
                let valid = rowIdx .>= Int32(0)
                let safe = maximum(rowIdx, Int32(0))
                let kvRow = kv[0, 0, safe].reshaped([1, 1, M, latent])
                let kpRow = kPos[0, 0, safe].reshaped([1, 1, M, rope])
                let qA = qAbs[0..., 0..., t ..< (t + 1), 0...]
                let qP = qPos[0..., 0..., t ..< (t + 1), 0...]
                let pos = matmul(qP * scale, kpRow.swappedAxes(-1, -2))
                let bias = `where`(valid.reshaped([1, 1, 1, M]), pos, MLXArray(-Float.infinity).asType(pos.dtype))
                let a = MLXFast.scaledDotProductAttention(
                    queries: qA, keys: kvRow, values: kvRow, scale: scale, mask: bias, sinks: sinks)
                refRows.append(a)
            }
            let refAttn = concatenated(refRows, axis: 2)
            eval(gotAttn, refAttn)
            let maxDiff = max(abs(gotAttn.asType(.float32) - refAttn.asType(.float32))).item(Float.self)
            #expect(maxDiff < 0.03, "sparseMLAPrefill maxDiff=\(maxDiff)")

            // Exact 3-pass radix Top-K + recent window vs argPartition reference (including ties & edge N <= K).
            let (topK, win, bigS) = (64, 16, 4096)
            let bigScores = MLXRandom.normal([1, 1, 19, bigS]).asType(.float32)
            // Test both T=1 (1024-thread TG) and T=19 (256-thread TG) and small startQPos (N <= K).
            for (subT, startPos) in [(1, bigS - 1), (19, bigS - 19), (19, 30)] {
                let sIn = bigScores[0..., 0..., ..<subT, 0...]
                let gotIdx = try #require(Ernie55Kernels.mlaTopKSelect(
                    scores: sIn, startQPos: startPos, topK: topK, window: win))
                eval(gotIdx, sIn)
                let idxArr = gotIdx.asArray(Int32.self)
                let scoreArr = sIn.asArray(Float.self)
                let strideM = topK + win
                for row in 0 ..< subT {
                    let qPos = startPos + row
                    let nOlder = min(bigS, max(0, qPos - win + 1))
                    let rowStart = row * strideM
                    // Check recent window slots exactly.
                    for wIdx in 0 ..< win {
                        let expected = qPos - (win - 1) + wIdx
                        let expVal = (expected >= 0 && expected < bigS) ? Int32(expected) : Int32(-1)
                        #expect(idxArr[rowStart + topK + wIdx] == expVal)
                    }
                    // Check selected older set matches exact top-K cutoff.
                    let picked = (0 ..< topK).map { Int(idxArr[rowStart + $0]) }
                    if nOlder <= topK {
                        for k in 0 ..< topK {
                            #expect(picked[k] == (k < nOlder ? k : -1))
                        }
                    } else {
                        #expect(Set(picked).count == topK)
                        #expect(picked.allSatisfy { $0 >= 0 && $0 < nOlder })
                        let rowScores = (0 ..< nOlder).map { scoreArr[row * bigS + $0] }
                        let sortedDesc = rowScores.sorted(by: >)
                        let kthVal = sortedDesc[topK - 1]
                        let minPicked = picked.map { rowScores[$0] }.min()!
                        #expect(minPicked >= kthVal, "minPicked=\(minPicked) < kthVal=\(kthVal)")
                        // Every strictly-greater-than-kth index must be in picked.
                        let pickedSet = Set(picked)
                        for c in 0 ..< nOlder where rowScores[c] > kthVal {
                            #expect(pickedSet.contains(c))
                        }
                    }
                }
            }

            // Fused MMA HCA prefill matches reference SDPA over local window + completed pooled KV.
            let (hcaNQ, hcaT, hcaD, hcaWin, hcaRatio) = (64, 37, 512, 16, 16)
            let hcaOffset = 64
            let hcaPrior = 16
            let hcaPoolCount = (hcaOffset + hcaT) / hcaRatio
            let hcaQ = (MLXRandom.normal([1, hcaNQ, hcaT, hcaD]) * 0.2).asType(.bfloat16)
            let hcaLocal = (MLXRandom.normal([1, 1, hcaPrior + hcaT, hcaD]) * 0.2).asType(.bfloat16)
            let hcaPool = (MLXRandom.normal([1, hcaPoolCount, hcaD]) * 0.2).asType(.bfloat16)
            let hcaSinks = (MLXRandom.normal([hcaNQ]) * 0.5).asType(.bfloat16)
            let hcaScale: Float = 1 / sqrt(Float(hcaD))
            let gotHCA = try #require(Ernie55Kernels.hcaPrefillAttention(
                q: hcaQ, localKV: hcaLocal, pooled: hcaPool, sinks: hcaSinks,
                offset: hcaOffset, priorLocal: hcaPrior, window: hcaWin, ratio: hcaRatio, scale: hcaScale))
            let qPosArr = MLXArray(Int32(hcaOffset) ..< Int32(hcaOffset + hcaT)).reshaped(hcaT, 1)
            let kPosArr = MLXArray(Int32(hcaOffset - hcaPrior) ..< Int32(hcaOffset + hcaT)).reshaped(1, hcaPrior + hcaT)
            let locMask = (kPosArr .<= qPosArr) & (kPosArr .> (qPosArr - Int32(hcaWin)))
            let poolMask = Ernie55Math.hcaPoolMask(start: hcaOffset, length: hcaT, columns: hcaPoolCount, ratio: hcaRatio)
            let fullMask = concatenated([locMask, poolMask], axis: -1)
            let fullKV = concatenated([hcaLocal, hcaPool.expandedDimensions(axis: 1)], axis: 2)
            let refHCA = MLXFast.scaledDotProductAttention(
                queries: hcaQ, keys: fullKV, values: fullKV, scale: hcaScale,
                mask: fullMask, sinks: hcaSinks)
            eval(gotHCA, refHCA)
            let hcaMaxDiff = max(abs(gotHCA.asType(.float32) - refHCA.asType(.float32))).item(Float.self)
            #expect(hcaMaxDiff < 0.03, "hcaPrefillAttention maxDiff=\(hcaMaxDiff)")
        }
    }
}
