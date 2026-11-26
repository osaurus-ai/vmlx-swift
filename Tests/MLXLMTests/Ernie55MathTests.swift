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
}
