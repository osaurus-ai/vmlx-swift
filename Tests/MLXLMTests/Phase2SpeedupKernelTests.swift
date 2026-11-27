// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast
@testable import MLXLLM
@testable import MLXLMCommon
import MLXNN
import XCTest

final class Phase2SpeedupKernelTests: XCTestCase {

    // MARK: - ERNIE 5.5 Zero-Spill MXFP8 2D/3D GEMV

    func testErnie55MXFP8LinearMatchesQuantizedMM() {
        guard Device.defaultDevice().deviceType == .gpu else { return }
        for (rows, n, k) in [(1, 1024, 4096), (6, 1024, 4096), (1, 4096, 2048), (6, 4096, 2048)] {
            let wDense = (MLXRandom.normal([n, k]) * 0.25).asType(.bfloat16)
            let (wq, scales, _) = quantized(wDense, groupSize: 32, bits: 8, mode: .mxfp8)
            let x = (MLXRandom.normal([1, rows, k]) * 0.5).asType(.bfloat16)
            let ref = quantizedMM(
                x, wq, scales: scales, biases: nil, transpose: true,
                groupSize: 32, bits: 8, mode: .mxfp8)
            guard let got = Ernie55Kernels.mxfp8Rows(x, weight: wq, scales: scales) else {
                XCTFail("Expected mxfp8Rows to handle rows=\(rows) n=\(n) k=\(k)")
                return
            }
            eval(ref, got)
            let maxErr = abs(got.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
            XCTAssertLessThan(
                maxErr, 0.25,
                "MXFP8 2D mismatch for rows=\(rows) n=\(n) k=\(k): maxErr=\(maxErr)")
        }

        // 3D grouped projection [8, 6, 4096] x [8, 1024, 1024] -> [8, 6, 1024]
        let groups = 8, rows = 6, k = 4096, n = 1024
        let wDense3 = (MLXRandom.normal([groups, n, k]) * 0.25).asType(.bfloat16)
        let (wq3, scales3, _) = quantized(wDense3, groupSize: 32, bits: 8, mode: .mxfp8)
        let x3 = (MLXRandom.normal([groups, rows, k]) * 0.5).asType(.bfloat16)
        let ref3 = quantizedMM(
            x3, wq3, scales: scales3, biases: nil, transpose: true,
            groupSize: 32, bits: 8, mode: .mxfp8)
        guard let got3 = Ernie55Kernels.mxfp8Rows(x3, weight: wq3, scales: scales3) else {
            XCTFail("Expected mxfp8Rows to handle 3D grouped shape")
            return
        }
        eval(ref3, got3)
        let maxErr3 = abs(got3.asType(.float32) - ref3.asType(.float32)).max().item(Float.self)
        XCTAssertLessThan(maxErr3, 0.25, "MXFP8 3D mismatch: maxErr=\(maxErr3)")
    }

    // MARK: - ERNIE 5.5 Multi-Row Router Top-K & Shared SiTU + LatentNorm + Concat

    func testErnie55MultiRouterTopKMatchesReference() {
        guard Device.defaultDevice().deviceType == .gpu else { return }
        let experts = 512, k = 10
        let corr = (MLXRandom.normal([experts]) * 0.05).asType(.float32)
        let eScale = (MLXRandom.uniform(0.5 ..< 1.5, [experts])).asType(.float32)

        for rows in [1, 6, 32] {
            let raw = sigmoid((MLXRandom.normal([1, rows, experts]) * 1.5).asType(.float32))
            let biased = raw + corr
            let order = argSort(-biased, axis: -1)[.ellipsis, ..<k]
            let picked = takeAlong(raw, order, axis: -1)
            let refWeights = (picked / maximum(picked.sum(axis: -1, keepDims: true), MLXArray(Float(1e-20))))
                * eScale[order]

            guard let got = Ernie55Kernels.routeTopK(
                raw, correction: corr, expertScale: eScale, count: k, normalize: true)
            else {
                XCTFail("Expected routeTopK to handle rows=\(rows)")
                return
            }
            eval(order, refWeights, got.indices, got.weights)
            let gotIds = got.indices.reshaped(rows, k).asType(.int32)
            let gotW = got.weights.reshaped(rows, k)
            let refSum = refWeights.reshaped(rows, k).sum(axis: -1)
            let gotSum = gotW.sum(axis: -1)
            let sumErr = abs(refSum - gotSum).max().item(Float.self)
            XCTAssertLessThan(sumErr, 1e-3, "routeTopK weight sum mismatch rows=\(rows): \(sumErr)")
            let refSorted = sorted(order.reshaped(rows, k).asType(.int32), axis: -1)
            let gotSorted = sorted(gotIds, axis: -1)
            XCTAssertTrue(
                all(refSorted .== gotSorted).item(Bool.self),
                "routeTopK selected experts mismatch rows=\(rows)")
        }
    }

    func testErnie55SharedSiTULatentNormConcatMatchesReference() {
        guard Device.defaultDevice().deviceType == .gpu else { return }
        let w = 256, latent = 256
        let normWeight = (MLXRandom.normal([latent]) * 0.2 + 1.0).asType(.bfloat16)
        for rows in [1, 6, 24] {
            let fc1 = (MLXRandom.normal([1, rows, 2 * w + latent]) * 0.5).asType(.bfloat16)
            let combined = (MLXRandom.normal([1, rows, latent]) * 0.5).asType(.bfloat16)
            let refAct = Ernie55Math.situ(
                fc1[.ellipsis, ..<w], fc1[.ellipsis, w ..< (2 * w)], gateCap: 4.0, upCap: 25.0)
            let refNorm = MLXFast.rmsNorm(combined, weight: normWeight, eps: 1e-6)
            let ref = concatenated([refAct, refNorm], axis: -1)
            guard let got = Ernie55Kernels.sharedSiTULatentNormConcat(
                proj: fc1, combined: combined, normWeight: normWeight,
                sharedWidth: w, gateCap: 4.0, upCap: 25.0, normEps: 1e-6)
            else {
                XCTFail("Expected sharedSiTULatentNormConcat for rows=\(rows)")
                return
            }
            eval(ref, got)
            let maxErr = abs(got.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
            XCTAssertLessThan(maxErr, 0.05, "sharedSiTULatentNormConcat mismatch rows=\(rows): \(maxErr)")
        }
    }

    // MARK: - ERNIE 5.5 MXFP4 Fused MoE Megakernels

    func testErnie55MXFP4FusedDecodeVerifyMatchesReference() {
        guard Device.defaultDevice().deviceType == .gpu else { return }
        let experts = 16, h = 256, n = 256, slots = 10
        let gwDense = (MLXRandom.normal([experts, n, h]) * 0.25).asType(.bfloat16)
        let uwDense = (MLXRandom.normal([experts, n, h]) * 0.25).asType(.bfloat16)
        let dwDense = (MLXRandom.normal([experts, h, n]) * 0.25).asType(.bfloat16)
        let (gw, gs, _) = quantized(gwDense, groupSize: 32, bits: 4, mode: .mxfp4)
        let (uw, us, _) = quantized(uwDense, groupSize: 32, bits: 4, mode: .mxfp4)
        let (dw, ds, _) = quantized(dwDense, groupSize: 32, bits: 4, mode: .mxfp4)
        let layer = Ernie55MXFP4RoutedExpertLayer(
            gateWeight: gw, gateScales: gs,
            upWeight: uw, upScales: us,
            downWeight: dw, downScales: ds,
            hiddenSize: h, intermediateSize: n)

        for rows in [1, 6] {
            let x = (MLXRandom.normal([1, rows, h]) * 0.5).asType(.bfloat16)
            let ids = MLXRandom.randInt(0 ..< experts, [1, rows, slots]).asType(.uint32)
            let scores = MLXRandom.uniform(0.05 ..< 0.25, [1, rows, slots]).asType(.float32)

            let got = layer.callRouted(x, indices: ids, scores: scores, caps: (4.0, 25.0))
            // Reference via MLX native gatherQuantizedMM (.mxfp4) + Ernie55Math.situ.
            let xExp = x.reshaped(rows, 1, 1, h)
            let idxFlat = ids.reshaped(rows, slots)
            let gRef = gatherQuantizedMM(
                xExp, gw, scales: gs, biases: nil, rhsIndices: idxFlat,
                transpose: true, groupSize: 32, bits: 4, mode: .mxfp4)
            let uRef = gatherQuantizedMM(
                xExp, uw, scales: us, biases: nil, rhsIndices: idxFlat,
                transpose: true, groupSize: 32, bits: 4, mode: .mxfp4)
            let actRef = Ernie55Math.situ(gRef, uRef, gateCap: 4.0, upCap: 25.0)
            let dRef = gatherQuantizedMM(
                actRef, dw, scales: ds, biases: nil, rhsIndices: idxFlat,
                transpose: true, groupSize: 32, bits: 4, mode: .mxfp4).reshaped(rows, slots, h)
            let ref = (dRef.asType(.float32) * scores.reshaped(rows, slots, 1)).sum(axis: 1)
                .asType(.bfloat16).reshaped(1, rows, h)
            eval(got, ref)
            let maxErr = abs(got.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
            XCTAssertLessThan(maxErr, 0.25, "MXFP4 fused MoE mismatch rows=\(rows): maxErr=\(maxErr)")
        }
    }

    // MARK: - Fused Metal FWHT-1024 / FWHT-512 & HadamardProjectionGroup

    func testFusedHadamardFWHTMatchesMLXReference() {
        guard Device.defaultDevice().deviceType == .gpu else { return }
        for blockSize in [512, 1024] {
            let width = blockSize * 2
            let signBits = MLXRandom.randInt(0 ..< 2, [width])
            let signs = `where`(signBits .== 0, MLXArray(Float(-1)), MLXArray(Float(1)))
            for dtype in [DType.float16, DType.bfloat16, DType.float32] {
                let x = (MLXRandom.normal([1, 4, width]) * 0.5).asType(dtype)
                for inverse in [false, true] {
                    let got = jangHadamardActivation(x, signs: signs, blockSize: blockSize, inverse: inverse)
                    let src = x.asType(.float32)
                    let signed = inverse ? src : src * signs
                    let transformed = MLX.hadamardTransform(
                        signed.reshaped(-1, blockSize), scale: 1 / sqrt(Float(blockSize))
                    ).reshaped(x.shape)
                    let ref = (inverse ? transformed * signs : transformed).asType(dtype)
                    eval(got, ref)
                    let maxErr = abs(got.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
                    let tol: Float = dtype == .float32 ? 1e-4 : 2e-2
                    XCTAssertLessThan(
                        maxErr, tol,
                        "Fused FWHT mismatch block=\(blockSize) dtype=\(dtype) inv=\(inverse): maxErr=\(maxErr)")
                }
            }
        }
    }

    func testHadamardProjectionGroupMatchesSeparateModules() {
        guard Device.defaultDevice().deviceType == .gpu else { return }
        let blockSize = 512, k = 1024
        let signBits = MLXRandom.randInt(0 ..< 2, [k])
        let signs = `where`(signBits .== 0, MLXArray(Float(-1)), MLXArray(Float(1)))

        func makeHadamardLinear(outDim: Int) -> HadamardQuantizedLinear {
            let dense = (MLXRandom.normal([outDim, k]) * 0.25).asType(.float16)
            let (w, s, b) = quantized(dense, groupSize: 128, bits: 2, mode: .affine)
            eval(w, s, b!)
            let q = QuantizedLinear(weight: w, bias: nil, scales: s, biases: b, groupSize: 128, bits: 2, mode: .affine)
            let h = HadamardQuantizedLinear(q, blockSize: blockSize)
            try! h.update(parameters: ModuleParameters.unflattened(["signs": signs]), verify: .none)
            eval(h)
            return h
        }

        let m1 = makeHadamardLinear(outDim: 256)
        let m2 = makeHadamardLinear(outDim: 128)
        let m3 = makeHadamardLinear(outDim: 128)

        let x1 = (MLXRandom.normal([1, 1, k]) * 0.5).asType(.float16)
        let ref1 = [m1(x1), m2(x1), m3(x1)]
        let x4 = (MLXRandom.normal([1, 4, k]) * 0.5).asType(.float16)
        let ref4 = FlashVerificationScope.withVerification(inputShape: [1, 4]) {
            [m1(x4), m2(x4), m3(x4)]
        }
        eval(ref1 + ref4)

        guard let group = HadamardProjectionGroup.make(from: [m1, m2, m3]) else {
            XCTFail("Expected HadamardProjectionGroup.make to succeed for owned sibling modules")
            return
        }
        let got1 = group(x1)
        let got4 = FlashVerificationScope.withVerification(inputShape: [1, 4]) {
            group(x4)
        }
        eval(got1 + got4)

        for i in 0 ..< 3 {
            let err1 = abs(got1[i].asType(.float32) - ref1[i].asType(.float32)).max().item(Float.self)
            XCTAssertEqual(err1, 0, "Decode HadamardProjectionGroup part \(i) diverged: \(err1)")
            let err4 = abs(got4[i].asType(.float32) - ref4[i].asType(.float32)).max().item(Float.self)
            XCTAssertEqual(err4, 0, "Row-exact verify HadamardProjectionGroup part \(i) diverged: \(err4)")
        }
    }

    func testRowExact2BitQ2ChunkIsBitExactAcrossRows() throws {
        MLXRandom.seed(77)
        let k = 1024
        let n = 256
        let w = MLXRandom.normal([n, k]).asType(.float16) * Float(0.25)
        let (wq, scales, biases) = quantized(w, groupSize: 128, bits: 2, mode: .affine)
        let b = biases!
        eval(wq, scales, b)
        for rows in [2, 3, 4, 5, 8] {
            let x = MLXRandom.normal([rows, k]).asType(.float16) * Float(0.5)
            eval(x)
            let refFn: (MLXArray) -> MLXArray = { input in
                let rCount = input.dim(0)
                return concatenated((0 ..< rCount).map { r in
                    quantizedMM(input[r ..< (r + 1)], wq, scales: scales, biases: b, transpose: true, groupSize: 128, bits: 2, mode: .affine)
                }, axis: 0)
            }
            let ref = refFn(x)
            let fast = try XCTUnwrap(RowExactAffineQMV.project(x, weight: wq, scales: scales, biases: b, groupSize: 128, bits: 2, reference: refFn))
            eval(ref, fast)
            let diff = abs(fast.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
            XCTAssertEqual(diff, 0, "2-bit rx_q2_chunk diverged at rows=\(rows): \(diff)")
        }
    }

    func testErnie55ReleaseMXFP8DoesNotCrash() {
        let lin = Linear(256, 128, bias: false)
        let q = QuantizedLinear(lin, groupSize: 32, bits: 8, mode: .mxfp8)
        eval(q.weight, q.scales)
        Ernie55Kernels.releaseMXFP8(q)
        XCTAssertEqual(q.weight.shape, [1, 1])
        XCTAssertEqual(q.scales.shape, [1, 1])
    }

    func testRoundE4M3AndRot128AndH32MatchReference() throws {
        MLXRandom.seed(99)
        let rows = 6
        let width = 2048
        let x = MLXRandom.normal([rows, width]).asType(.bfloat16) * Float(2.0)
        let signs = MLX.where(MLXRandom.uniform(0 ..< 1, [width]) .> Float(0.5), MLXArray(Float(1)), MLXArray(Float(-1)))
        eval(x, signs)

        let (xq, xr) = JANGTKernels.roundE4M3AndRot128(x, signs: signs)
        let refXq = Ernie55Math.roundExpertActivations(x)
        let refXr = JANGTKernels.rotateRows(refXq, signs: signs)
        eval(xq, xr, refXq, refXr)

        let diffXq = abs(xq.asType(.float32) - refXq.asType(.float32)).max().item(Float.self)
        let diffXr = abs(xr - refXr).max().item(Float.self)
        XCTAssertLessThan(diffXq, 1e-3, "roundE4M3AndRot128 xq diverged: \(diffXq)")
        XCTAssertLessThan(diffXr, 1e-3, "roundE4M3AndRot128 xr diverged: \(diffXr)")

        let h32Fused = JANGTKernels.roundE4M3AndH32(x, outputDType: .float32)
        // Reference H32 via reshaped [rows, width/32, 32] butterfly or sign-less rotate comparison
        let h32Kernel = MLXFast.metalKernel(
            name: "test_h32_ref", inputNames: ["x"], outputNames: ["out"],
            source: """
                uint lane = thread_index_in_simdgroup;
                size_t offset = size_t(threadgroup_position_in_grid.z) * K
                    + size_t(threadgroup_position_in_grid.y) * 32u + lane;
                float value = float(x[offset]);
                for (ushort stage = 1; stage < 32; stage <<= 1) {
                    float other = simd_shuffle_xor(value, stage);
                    value = (lane & stage) ? other - value : value + other;
                }
                out[offset] = value * 0.17677669529663687f;
                """)
        let refH32 = h32Kernel(
            [contiguous(refXq)], template: [("K", width)],
            grid: (32, width / 32, rows), threadGroup: (32, 1, 1),
            outputShapes: [[rows, width]], outputDTypes: [.float32])[0]
        eval(h32Fused, refH32)
        let diffH32 = abs(h32Fused - refH32).max().item(Float.self)
        XCTAssertLessThan(diffH32, 1e-3, "roundE4M3AndH32 diverged: \(diffH32)")
    }

    func testErnie55DSparkMarkovArgmaxMatchesReference() throws {
        MLXRandom.seed(123)
        let vocab = 8192
        let rank = 256
        let baseRow = MLXRandom.normal([vocab]).asType(.float32) * Float(3.0)
        let w2 = (MLXRandom.normal([vocab, rank]) * Float(0.2)).asType(.bfloat16)
        let trans = (MLXRandom.normal([rank]) * Float(0.5)).asType(.bfloat16)
        let ranges = MLXArray([Float(-10), Float(10), Float(-15), Float(15)])
        let ts = MLXArray([Float(0.1), Float(0.1), Float(0.05), Float(0.05)])
        eval(baseRow, w2, trans, ranges, ts)

        let fast = try XCTUnwrap(Ernie55Kernels.dsparkMarkovArgmax(
            baseRow: baseRow, markovHead: w2, transition: trans, ranges: ranges, ts: ts))
        let refBias = matmul(w2, trans).asType(.float32)
        let refScore = Ernie55Math.multiMax(baseRow + refBias, ranges: ranges, slopes: ts)
        let ref = argMax(refScore).asType(.int32)
        eval(fast, ref)
        XCTAssertEqual(fast.item(Int32.self), ref.item(Int32.self))
    }

    func testNativeCheckpointFormatsAdmitsErnieMXFP4() {
        let mxfp4 = Data(#"{"model_type":"ernie5_5_moe","quantization_config":{"quant_method":"fp8","expert_dtype":"mxfp4"}}"#.utf8)
        let plainBF16 = Data(#"{"model_type":"ernie5_5_moe"}"#.utf8)
        XCTAssertTrue(NativeCheckpointFormats.loadsUpstreamCheckpoint(configData: mxfp4))
        XCTAssertFalse(NativeCheckpointFormats.loadsUpstreamCheckpoint(configData: plainBF16))
    }

    func testInklingLoopedDFlash2CompatibilityAndPreference() throws {
        let inklingConfig = Data("""
        {
          "model_type": "inkling2_loop",
          "text_config": {
            "hidden_size": 2304,
            "num_hidden_layers": 24,
            "vocab_size": 201024,
            "unpadded_vocab_size": 200058,
            "loop_config": {
              "core_start": 6,
              "core_end": 18,
              "num_passes": 3
            }
          }
        }
        """.utf8)
        let info = VMLXDFlash2DrafterInfo(
            path: "/tmp/nonexistent-inkling-dflash2",
            blockSize: 8,
            vocabularySize: 201024,
            targetLayerCount: 48,
            targetLayerIDs: [5, 23, 41, 48],
            weightBytes: 989_975_616)
        // Without a real config.json at path, drafterHiddenSize is nil; test on a temp dir with real config.json
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("inkling-dflash2-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let dflashCfg = """
        {
          "architectures": ["DFlash2DraftModel"],
          "hidden_size": 2304,
          "intermediate_size": 6912,
          "num_attention_heads": 32,
          "num_hidden_layers": 5,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "num_target_layers": 48,
          "vocab_size": 201024,
          "layer_types": ["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],
          "sliding_window": 2048,
          "dflash_config": {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "final_logit_softcapping": 30.0,
            "mask_token_id": 200056,
            "selector_rank": 256,
            "selector_softcap_mode": "after_correction",
            "selector_top_k": 16,
            "target_layer_ids": [5, 23, 41, 48]
          }
        }
        """
        try Data(dflashCfg.utf8).write(to: tmp.appendingPathComponent("config.json"))
        let cfgObj = try JSONSerialization.jsonObject(with: Data(dflashCfg.utf8)) as! [String: Any]
        let parsed = try DFlash2Configuration(json: cfgObj)
        XCTAssertEqual(parsed.dflash.selectorSoftcapMode, "after_correction")
        XCTAssertEqual(parsed.dflash.finalLogitSoftcapping, 30.0)
        XCTAssertEqual(parsed.targetLayerIds, [5, 23, 41, 48])

        let validInfo = VMLXDFlash2DrafterInfo(
            path: tmp.path,
            blockSize: 8,
            vocabularySize: 201024,
            targetLayerCount: 48,
            targetLayerIDs: [5, 23, 41, 48],
            weightBytes: 989_975_616)
        XCTAssertNil(validInfo.mismatchReason(configData: inklingConfig))
        XCTAssertNotNil(info.mismatchReason(configData: inklingConfig))

        let wrongLayerCount = Data("""
        {
          "model_type": "qwen3",
          "hidden_size": 2304,
          "num_hidden_layers": 24,
          "vocab_size": 201024
        }
        """.utf8)
        XCTAssertNotNil(validInfo.mismatchReason(configData: wrongLayerCount))

        // If the real local Inkling-2-Mini-JANGHT4.2 bundle with dflash2/ is present, verify
        // resolvedMTPDraftStrategy prefers dflash2/ over inline DSpark!
        let localBundle = URL(fileURLWithPath: "/Volumes/EricMLWork/models/OsaurusAI/Inkling-2-Mini-JANGHT4.2")
        let localDFlash2 = localBundle.appendingPathComponent("dflash2")
        if FileManager.default.fileExists(atPath: localDFlash2.appendingPathComponent("model.safetensors").path),
            let bundleCfg = try? Data(contentsOf: localBundle.appendingPathComponent("config.json"))
        {
            XCTAssertNil(DFlash2ArtifactMetadata.rejectionReason(at: localDFlash2))
            let settings = VMLXServerRuntimeSettings()
            let strategy = settings.resolvedMTPDraftStrategy(
                configData: bundleCfg, jangConfig: nil, status: nil, bundleDirectory: localBundle)
            if case .dflash2(let drafterPath, _)? = strategy {
                XCTAssertEqual(drafterPath.lastPathComponent, "dflash2")
            } else {
                XCTFail("Expected .dflash2 to win over inline DSpark on Inkling-2-Mini-JANGHT4.2, got \(String(describing: strategy))")
            }
        }
    }

    func testBonsai2BitQ2TimingDiagnostic() throws {
        guard Device.defaultDevice().deviceType == .gpu else { return }
        MLXRandom.seed(11)
        let k = 5120
        let n = 34816
        let w = (MLXRandom.normal([n, k]) * Float(0.25)).asType(.float16)
        let (wq, scales, biases) = quantized(w, groupSize: 128, bits: 2, mode: .affine)
        let b = biases!
        let x1 = (MLXRandom.normal([1, k]) * Float(0.5)).asType(.float16)
        let x3 = (MLXRandom.normal([3, k]) * Float(0.5)).asType(.float16)
        eval(wq, scales, b, x1, x3)

        let refFn: (MLXArray) -> MLXArray = { input in
            MLX.quantizedMM(
                input.expandedDimensions(axis: 1), wq.expandedDimensions(axis: 0),
                scales: scales.expandedDimensions(axis: 0), biases: b.expandedDimensions(axis: 0),
                transpose: true, groupSize: 128, bits: 2, mode: .affine
            ).squeezed(axis: 1)
        }
        let warm1 = quantizedMM(x1, wq, scales: scales, biases: b, transpose: true, groupSize: 128, bits: 2, mode: .affine)
        let warm3Fast = try XCTUnwrap(RowExactAffineQMV.project(x3, weight: wq, scales: scales, biases: b, groupSize: 128, bits: 2, reference: refFn))
        let warm3Ref = refFn(x3)
        eval(warm1, warm3Fast, warm3Ref)

        let iters = 15
        let t0 = Date()
        var o1 = warm1
        for _ in 0 ..< iters { o1 = quantizedMM(x1, wq, scales: scales, biases: b, transpose: true, groupSize: 128, bits: 2, mode: .affine) }
        eval(o1)
        let t1Ms = Date().timeIntervalSince(t0) * 1000.0 / Double(iters)

        let t1 = Date()
        var o3Fast = warm3Fast
        for _ in 0 ..< iters {
            o3Fast = RowExactAffineQMV.project(x3, weight: wq, scales: scales, biases: b, groupSize: 128, bits: 2, reference: refFn)!
        }
        eval(o3Fast)
        let t3FastMs = Date().timeIntervalSince(t1) * 1000.0 / Double(iters)

        let t2 = Date()
        var o3Ref = warm3Ref
        for _ in 0 ..< iters { o3Ref = refFn(x3) }
        eval(o3Ref)
        let t3RefMs = Date().timeIntervalSince(t2) * 1000.0 / Double(iters)

        print(String(format: "[BONSAI_Q2_BENCH] K=%d N=%d T=1=%.3fms T=3(rx_q2_chunk)=%.3fms T=3(batched_qmv)=%.3fms", k, n, t1Ms, t3FastMs, t3RefMs))
    }
}
