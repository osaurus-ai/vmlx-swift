// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast
@testable import MLXLLM
@testable import MLXLMCommon
import MLXNN
import XCTest

final class Phase4SpeedupKernelTests: XCTestCase {

    // MARK: - DFlash 2 Kernels

    func testDFlash2GroupedDynamicConvolveK2MatchesReference() throws {
        guard Device.defaultDevice().deviceType == .gpu else { return }
        MLXRandom.seed(41)
        let B = 1, L = 8, H = 2304, G = 16
        let groups = H / G
        let hidden = (MLXRandom.normal([B, L, H]) * 0.5).asType(.bfloat16)
        let dynamic = (MLXRandom.normal([B, L, 2, groups]) * 0.25).asType(.bfloat16)
        let base = (MLXRandom.normal([2, H]) * 0.25).asType(.bfloat16)

        let got = try XCTUnwrap(
            DFlash2Kernels.groupedDynamicConvolveK2(
                hidden: hidden, dynamic: dynamic, base: base, groupSize: G))

        let blocks = hidden.reshaped(B, L, groups, G)
        let dyn = dynamic.reshaped(B, L, 2, groups, 1)
        let pad = MLXArray.zeros([B, 1, groups, G], dtype: .bfloat16)
        let shifted = concatenated([pad, blocks[0..., ..<(L - 1), 0..., 0...]], axis: 1)
        let k0 = base[0].reshaped(1, 1, groups, G) + dyn[0..., 0..., 0, 0..., 0...]
        let k1 = base[1].reshaped(1, 1, groups, G) + dyn[0..., 0..., 1, 0..., 0...]
        let ref = (k0.asType(.float32) * blocks.asType(.float32)
            + k1.asType(.float32) * shifted.asType(.float32)).asType(.bfloat16).reshaped(B, L, H)
        eval(got, ref)
        let maxErr = abs(got.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
        XCTAssertLessThan(maxErr, 2e-2, "DFlash2 gdconv_k2 mismatch: \(maxErr)")
    }

    func testDFlash2TopKMatchesExactTopKOver200KVocab() throws {
        guard Device.defaultDevice().deviceType == .gpu else { return }
        MLXRandom.seed(42)
        let P = 7, V = 200_058, K = 16
        let logits = (MLXRandom.normal([1, P, V]) * 3.0).asType(.float32)

        let got = try XCTUnwrap(DFlash2Kernels.topK(logits: logits, topK: K))
        let refOrder = argSort(-logits, axis: -1)[.ellipsis, ..<K].asType(.int32)
        let refVals = takeAlong(logits, refOrder, axis: -1)
        eval(got.unary, got.candidates, refVals, refOrder)

        let valErr = abs(got.unary - refVals).max().item(Float.self)
        XCTAssertEqual(valErr, 0, "DFlash2 topK values mismatch: \(valErr)")
        XCTAssertTrue(
            all(got.candidates .== refOrder).item(Bool.self),
            "DFlash2 topK indices mismatch")
    }

    func testDFlash2MarkovSelectorsMatchReference() throws {
        guard Device.defaultDevice().deviceType == .gpu else { return }
        MLXRandom.seed(43)
        let P = 7, K = 16, R = 256, V = 4096
        let unary = (MLXRandom.normal([1, P, K]) * 2.0).asType(.float32)
        let candidates = MLXRandom.randInt(0 ..< V, [1, P, K]).asType(.int32)
        let hiddenProj = (MLXRandom.normal([1, P, R]) * 0.3).asType(.bfloat16)
        let predW = (MLXRandom.normal([V, R]) * 0.25).asType(.bfloat16)
        let succW = (MLXRandom.normal([V, R]) * 0.25).asType(.bfloat16)
        let anchor = MLXArray([Int32(137)])

        for cap in [Float(0), Float(30.0)] {
            let gotPath = try XCTUnwrap(
                DFlash2Kernels.markovSelectGreedy(
                    unary: unary, candidates: candidates, hiddenProj: hiddenProj,
                    predWeight: predW, succWeight: succW, anchorIds: anchor, cap: cap))

            var pred = anchor
            var refPath: [MLXArray] = []
            for pos in 0 ..< P {
                let pVec = predW[pred].asType(.float32).expandedDimensions(axis: 1)
                let hVec = hiddenProj[0..., pos, 0...].asType(.float32).expandedDimensions(axis: 1)
                let cRow = candidates[0..., pos, 0...]
                let sVec = succW[cRow].asType(.float32)
                var scores = unary[0..., pos, 0...] + (pVec * hVec * sVec).sum(axis: -1)
                if cap > 0 {
                    scores = tanh(scores / cap) * cap
                }
                let stepGot = try XCTUnwrap(
                    DFlash2Kernels.markovStepScores(
                        unaryRow: unary[0..., pos, 0...], candRow: cRow,
                        hRow: hiddenProj[0..., pos, 0...],
                        predWeight: predW, succWeight: succW, predId: pred, cap: cap))
                eval(stepGot, scores)
                let stepErr = abs(stepGot - scores).max().item(Float.self)
                XCTAssertLessThan(stepErr, 1e-4, "markovStepScores mismatch cap=\(cap) pos=\(pos): \(stepErr)")
                let idx = argMax(scores, axis: -1)
                pred = takeAlong(cRow, idx.expandedDimensions(axis: -1), axis: -1)[0..., 0]
                refPath.append(pred)
            }
            let ref = stacked(refPath, axis: 1)
            eval(gotPath, ref)
            XCTAssertTrue(
                all(gotPath .== ref).item(Bool.self),
                "markovSelectGreedy mismatch for cap=\(cap)")
        }
    }

    // MARK: - Inkling-2 Multi-Vector Affine-8 g=64 GEMV

    func testInkling2Affine8G64MultiGemvMatchesReferenceAndIsRowExact() throws {
        guard Device.defaultDevice().deviceType == .gpu else { return }
        MLXRandom.seed(44)
        let K = 2304, N = 2048, outCols = 1997
        let wDense = (MLXRandom.normal([N, K]) * 0.25).asType(.bfloat16)
        let (wq, scales, biasesOpt) = quantized(wDense, groupSize: 64, bits: 8, mode: .affine)
        let biases = try XCTUnwrap(biasesOpt)
        eval(wq, scales, biases)

        let x8 = (MLXRandom.normal([1, 8, K]) * 0.5).asType(.bfloat16)
        eval(x8)

        for cap in [Float(0), Float(30.0)] {
            let got8 = try XCTUnwrap(
                Inkling2Kernels.affine8G64MultiGemv(
                    x8, weight: wq, scales: scales, biases: biases,
                    outDType: .float32, softcap: cap, outCols: outCols))
            let got1 = try XCTUnwrap(
                Inkling2Kernels.affine8G64MultiGemv(
                    x8[0..., 3 ..< 4, 0...], weight: wq, scales: scales, biases: biases,
                    outDType: .float32, softcap: cap, outCols: outCols))

            let wDequant = dequantized(
                wq, scales: scales.asType(.float32), biases: biases.asType(.float32),
                groupSize: 64, bits: 8, mode: .affine, dtype: .float32
            )
            var ref = matmul(x8.asType(.float32), wDequant.T, stream: .cpu)
            if cap > 0 {
                ref = tanh(ref / cap) * cap
            }
            ref = ref[.ellipsis, ..<outCols]
            eval(got8, got1, ref)

            let refErr = abs(got8 - ref).max().item(Float.self)
            XCTAssertLessThan(refErr, 1e-3, "affine8G64MultiGemv vs FP32 ref cap=\(cap): \(refErr) (wDequant.dtype=\(wDequant.dtype))")

            // Row 3 in T=8 must be bit-identical to T=1!
            let rowDiff = abs(got8[0..., 3 ..< 4, 0...] - got1).max().item(Float.self)
            XCTAssertEqual(rowDiff, 0, "affine8G64MultiGemv T=1 vs T=8 row parity broke: \(rowDiff)")
        }
    }

    // MARK: - Inkling-2 Fused MoE Router

    func testInkling2FusedRouterMatchesReference() throws {
        guard Device.defaultDevice().deviceType == .gpu else { return }
        MLXRandom.seed(45)
        let E = 128, K = 2304, topK = 8
        let routeScale: Float = 2.5
        let w = (MLXRandom.normal([E, K]) * 0.1).asType(.bfloat16)
        let bias = (MLXRandom.normal([E]) * 0.05).asType(.float32)

        for T in [1, 8, 12] {
            let x = (MLXRandom.normal([T, K]) * 0.5).asType(.float32)
            let got = try XCTUnwrap(
                Inkling2Kernels.routerTopK(
                    x, gateWeight: w, bias: bias, topK: topK, routeScale: routeScale))
            let scores = sigmoid(matmul(x, w.asType(.float32).T))
            let biased = scores + bias
            let refIds = argSort(-biased, axis: -1)[.ellipsis, ..<topK].asType(.uint32)
            let picked = takeAlong(scores, refIds.asType(.int32), axis: -1)
            let refW = (picked / (picked.sum(axis: -1, keepDims: true) + 1e-20)) * routeScale
            eval(got.indices, got.weights, refIds, refW)

            XCTAssertTrue(
                all(got.indices .== refIds).item(Bool.self),
                "Inkling2 routerTopK ids mismatch at T=\(T)")
            let wErr = abs(got.weights - refW).max().item(Float.self)
            XCTAssertLessThan(wErr, 1e-4, "Inkling2 routerTopK weights mismatch at T=\(T): \(wErr)")
        }
    }

    // MARK: - Inkling-2 Zero-Mask Verify Attention, GQA MPP, and Prefill MPP

    private func referenceInklingAttention(
        q: MLXArray, k: MLXArray, v: MLXArray, r: MLXArray, projT: MLXArray,
        sinks: MLXArray, seen: Int, kStart: Int, window: Int, relExtent: Int, scale: Float
    ) -> MLXArray {
        let H = q.dim(1), LQ = q.dim(2), Hkv = k.dim(1), S = k.dim(2)
        let gqa = H / Hkv
        let kRep = repeated(k, count: gqa, axis: 1).asType(.float32)
        let vRep = repeated(v, count: gqa, axis: 1).asType(.float32)
        let qf = q.asType(.float32)
        var scores = matmul(qf, kRep.swappedAxes(-1, -2)) * scale  // [1, H, LQ, S]

        let qPos = MLXArray(Int32(seen) ..< Int32(seen + LQ)).expandedDimensions(axis: 1)
        let kPos = MLXArray(Int32(kStart) ..< Int32(kStart + S)).expandedDimensions(axis: 0)
        let dist = qPos - kPos  // [LQ, S]
        var valid = dist .>= 0
        if window > 0 { valid = valid .&& (dist .< Int32(window)) }
        let inBand = (dist .>= 0) .&& (dist .< Int32(relExtent))
        let clipped = clip(dist, min: 0, max: relExtent - 1)
        let rel = matmul(r.asType(.float32), projT.asType(.float32).T).asType(.bfloat16)
            .asType(.float32).transposed(0, 2, 1, 3)  // [1, H, LQ, E]
        let bias = takeAlong(
            rel, broadcast(clipped.reshaped(1, 1, LQ, S), to: [1, H, LQ, S]), axis: -1)
        scores = scores + `where`(inBand.reshaped(1, 1, LQ, S), bias, MLXArray(Float(0)))
        scores = `where`(valid.reshaped(1, 1, LQ, S), scores, MLXArray(-Float.infinity))

        let sinkCol = sinks.reshaped(1, LQ, H).transposed(0, 2, 1).expandedDimensions(axis: -1)
            .asType(.float32)
        let allScores = concatenated([sinkCol, scores], axis: -1)
        let probs = softmax(allScores, axis: -1)[0..., 0..., 0..., 1...]
        return matmul(probs, vRep).asType(.bfloat16)
    }

    func testInkling2VerifyAndGQAAndPrefillAttentionMatchReference() throws {
        guard Device.defaultDevice().deviceType == .gpu else { return }
        MLXRandom.seed(46)
        let H = 24, Hkv = 8, D = 128, DR = 16, relExtent = 64
        let scale: Float = 1.0 / sqrt(Float(D))
        let projT = (MLXRandom.normal([relExtent, DR]) * 0.2).asType(.bfloat16)

        // 1. Verify attention: single-split (S=64), multi-split (S=300), and GQA MPP (S=640 sliding & S=1024 full)
        for (LQ, S, window) in [(8, 64, 1024), (8, 300, 1024), (8, 640, 512), (8, 1024, 0), (1, 1024, 0)] {
            let q = (MLXRandom.normal([1, H, LQ, D]) * 0.3).asType(.bfloat16)
            let k = (MLXRandom.normal([1, Hkv, S, D]) * 0.3).asType(.bfloat16)
            let v = (MLXRandom.normal([1, Hkv, S, D]) * 0.3).asType(.bfloat16)
            let r = (MLXRandom.normal([1, LQ, H, DR]) * 0.3).asType(.bfloat16)
            let sinks = (MLXRandom.normal([LQ, H]) * 0.5).asType(.bfloat16)

            let got = try XCTUnwrap(
                Inkling2Kernels.verifyAttention(
                    q: q, k: k, v: v, r: r, projT: projT, sinks: sinks,
                    window: window, relExtent: relExtent, scale: scale))
            let ref = referenceInklingAttention(
                q: q, k: k, v: v, r: r, projT: projT, sinks: sinks,
                seen: S - LQ, kStart: 0, window: window, relExtent: relExtent, scale: scale)
            eval(got, ref)
            let maxErr = abs(got.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
            XCTAssertLessThan(
                maxErr, 2e-2,
                "verifyAttention mismatch LQ=\(LQ) S=\(S) W=\(window): maxErr=\(maxErr)")
        }

        // 2. Metal 4 MPP prefill attention (both padded LQ=16 and multi-tile LQ=48, sliding & full)
        if Inkling2Kernels.mppAvailable() {
            for (LQ, S, window) in [(16, 256, 0), (48, 512, 256), (48, 512, 0)] {
                let q = (MLXRandom.normal([1, H, LQ, D]) * 0.3).asType(.bfloat16)
                let k = (MLXRandom.normal([1, Hkv, S, D]) * 0.3).asType(.bfloat16)
                let v = (MLXRandom.normal([1, Hkv, S, D]) * 0.3).asType(.bfloat16)
                let r = (MLXRandom.normal([1, LQ, H, DR]) * 0.3).asType(.bfloat16)
                let sinks = (MLXRandom.normal([1, LQ, H]) * 0.5).asType(.bfloat16)

                let got = try XCTUnwrap(
                    Inkling2Kernels.prefillAttentionMPP(
                        q: q, k: k, v: v, r: r, projT: projT, sinks: sinks,
                        seen: S - LQ, kStart: 0, window: window, relExtent: relExtent, scale: scale))
                let ref = referenceInklingAttention(
                    q: q, k: k, v: v, r: r, projT: projT, sinks: sinks.reshaped(LQ, H),
                    seen: S - LQ, kStart: 0, window: window, relExtent: relExtent, scale: scale)
                eval(got, ref)
                let maxErr = abs(got.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
                XCTAssertLessThan(
                    maxErr, 2e-2,
                    "prefillAttentionMPP mismatch LQ=\(LQ) S=\(S) W=\(window): maxErr=\(maxErr)")
            }
        }
    }

    // MARK: - Inkling-2 Fused NVFP4 Weighted Down (JANGHT mixed native banks)

    func testInkling2NVFP4DownWeightedMatchesReference() throws {
        guard Device.defaultDevice().deviceType == .gpu else { return }
        MLXRandom.seed(48)
        let E = 16, H = 256, I = 128, k = 8
        let down = Inkling2NativeSwitchLinear(inputDims: I, outputDims: H, experts: E)
        let dense = (MLXRandom.normal([E, H, I]) * 0.25).asType(.bfloat16)
        let (codes, scales, _) = quantized(dense, groupSize: 16, bits: 4, mode: .nvfp4)
        let es = MLXRandom.uniform(0.5 ..< 1.5, [E]).asType(.float32)
        try down.update(
            parameters: ModuleParameters.unflattened([
                "weight": codes, "scales": scales, "weight_scale_2": es,
            ]), verify: .none)
        eval(down)

        for T in [1, 8] {
            let act = (MLXRandom.normal([T * k, I]) * 0.5).asType(.float32)
            let inds = MLXRandom.randInt(0 ..< E, [T, k]).asType(.uint32)
            let rw = MLXRandom.uniform(0.05 ..< 0.3, [T, k]).asType(.float32)

            let got = Inkling2DecodeKernels.nvfp4DownWeighted(
                act32: act, inds: inds, weights: rw, down: down, outDType: .float32)
            let raw = down(act.reshaped([T, k, 1, I]), inds, sorted: false).reshaped([T, k, H])
            let ref = (raw.asType(.float32) * es[inds].expandedDimensions(axis: -1)
                * rw.expandedDimensions(axis: -1)).sum(axis: 1)
            eval(got, ref)
            let maxErr = abs(got - ref).max().item(Float.self)
            XCTAssertLessThan(maxErr, 2e-2, "nvfp4DownWeighted mismatch at T=\(T): \(maxErr)")
        }
    }

    func testInkling2NVFP4GateUpActV2MatchesReference() throws {
        guard Device.defaultDevice().deviceType == .gpu else { return }
        MLXRandom.seed(49)
        let E = 16, H = 256, I = 128, k = 8
        let gateUp = Inkling2NativeSwitchLinear(inputDims: H, outputDims: 2 * I, experts: E)
        let dense = (MLXRandom.normal([E, 2 * I, H]) * 0.25).asType(.bfloat16)
        let (codes, scales, _) = quantized(dense, groupSize: 16, bits: 4, mode: .nvfp4)
        let es = MLXRandom.uniform(0.5 ..< 1.5, [E]).asType(.float32)
        try gateUp.update(
            parameters: ModuleParameters.unflattened([
                "weight": codes, "scales": scales, "weight_scale_2": es,
            ]), verify: .none)
        eval(gateUp)
        let caps = MLXArray([Float(7.0), Float(7.0)])

        for T in [1, 8] {
            let x = (MLXRandom.normal([T, H]) * 0.5).asType(.float32)
            let inds = MLXRandom.randInt(0 ..< E, [T, k]).asType(.uint32)
            let got = Inkling2DecodeKernels.nvfp4GateUpAct(
                x32: x, inds: inds, gateUp: gateUp, caps: caps, topK: k)
            let raw = gateUp(x.reshaped([T, 1, 1, H]), inds, sorted: false).reshaped([T * k, 2 * I])
            let ref = Inkling2Kernels.moeActivation(
                [raw, inds.reshaped([T * k]), es, caps],
                grid: (I, T * k, 1), threadGroup: (min(256, I), 1, 1),
                outputShapes: [[T * k, I]], outputDTypes: [.float32])[0]
            eval(got, ref)
            let maxErr = abs(got - ref).max().item(Float.self)
            XCTAssertLessThan(maxErr, 2e-2, "nvfp4GateUpAct v2 mismatch at T=\(T): \(maxErr)")
        }
    }

    // MARK: - ERNIE 5.5 Split-K Hardware-MMA HCA Decode & Verify Attention

    func testErnie55HCASplitKDecodeVerifyMatchesReference() throws {
        guard Device.defaultDevice().deviceType == .gpu else { return }
        MLXRandom.seed(50)
        let NQ = 64, D = 128, window = 128, ratio = 128, pooledRows = 512
        let offset = pooledRows * ratio + 37
        let scale: Float = 1.0 / sqrt(Float(D))
        let pooled = (MLXRandom.normal([1, pooledRows, D]) * 0.4).asType(.bfloat16)
        let sinks = (MLXRandom.normal([NQ]) * 0.5).asType(.bfloat16)

        for T in [1, 6] {
            let priorLocal = window - 1
            let localKV = (MLXRandom.normal([1, 1, priorLocal + T, D]) * 0.4).asType(.bfloat16)
            let q = (MLXRandom.normal([1, NQ, T, D]) * 0.4).asType(.bfloat16)

            let got = try XCTUnwrap(
                Ernie55Kernels.hcaDecodeVerifyAttention(
                    q: q, localKV: localKV, pooled: pooled, sinks: sinks,
                    offset: offset, priorLocal: priorLocal, window: window, ratio: ratio, scale: scale))
            let ref = try XCTUnwrap(
                Ernie55Kernels.hcaPrefillAttention(
                    q: q, localKV: localKV, pooled: pooled, sinks: sinks,
                    offset: offset, priorLocal: priorLocal, window: window, ratio: ratio, scale: scale))
            eval(got, ref)
            let maxErr = abs(got.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
            XCTAssertLessThan(
                maxErr, 5e-3, "Ernie55 hcaDecodeVerifyAttention Split-K mismatch T=\(T): maxErr=\(maxErr)")
        }
    }

    // MARK: - ERNIE 5.5 Metal 4 MPP DSA Indexer (Prefill & L=1 Decode)

    func testErnie55MPPIndexScoresDecodeAndPrefillMatchReference() throws {
        guard Device.defaultDevice().deviceType == .gpu else { return }
        MLXRandom.seed(47)
        let H = 32, D = 128, S = 600

        for T in [1, 40] {
            let q = (MLXRandom.normal([1, H, T, D]) * 0.4).asType(.bfloat16)
            let k = (MLXRandom.normal([1, 1, S, D]) * 0.4).asType(.bfloat16)
            let w = (MLXRandom.normal([1, T, H]) * 0.5).asType(.float32)

            let got = try XCTUnwrap(Ernie55Kernels.mlaIndexScores(q: q, k: k, weights: w))
            let dots = maximum(
                matmul(q.asType(.float32), k.asType(.float32).swappedAxes(-1, -2)),
                MLXArray(Float(0)))
            let ref = (dots * w.transposed(0, 2, 1).expandedDimensions(axis: -1))
                .sum(axis: 1, keepDims: true)
            eval(got, ref)
            let peak = abs(ref).max().item(Float.self)
            let relErr = abs(got - ref).max().item(Float.self) / max(peak, 1e-6)
            XCTAssertLessThan(relErr, 2e-3, "Ernie55 mlaIndexScores MPP mismatch T=\(T): relErr=\(relErr)")
        }
    }
}
