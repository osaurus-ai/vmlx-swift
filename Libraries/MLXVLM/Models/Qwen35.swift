//
//  Qwen35.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/25.
//
//  Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen3_5
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Compiled sigmoid multiply: fuses x * sigmoid(gate) into one Metal dispatch.
/// Used in GatedDeltaNet output projection (per-layer, ~30 layers per forward).
private let compiledSigmoidMultiply: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (x: MLXArray, gate: MLXArray) -> MLXArray in
        x * sigmoid(gate)
    }
    guard HardwareInfo.isCompiledDecodeSupported else { return body }
    let compiled = compile(shapeless: true, body)
    // Inside the outer compiled-decode trace a separately-compiled function is
    // an illegal nested compile (see `safeGeluApproximate` in SwitchLayers);
    // run the plain body there — its ops fuse into the outer graph anyway.
    return { x, g in CompiledDecodeTrace.isActive ? body(x, g) : compiled(x, g) }
}()

/// Compiled shared expert gate: sigmoid(gate_output) * expert_output → 1 fused op.
private let compiledSigmoidGate: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gateOutput: MLXArray, expertOutput: MLXArray) -> MLXArray in
        sigmoid(gateOutput) * expertOutput
    }
    guard HardwareInfo.isCompiledDecodeSupported else { return body }
    let compiled = compile(shapeless: true, body)
    // Plain body inside the outer compiled-decode trace — nested compile is illegal.
    return { g, e in CompiledDecodeTrace.isActive ? body(g, e) : compiled(g, e) }
}()

private enum Qwen35VLError: Error {
    case featureTokenMismatch(expected: Int, actual: Int)
}

// MARK: - Gated Delta Helpers

/// Compiled compute_g — fuses exp+softplus+mul into 1 Metal dispatch.
private let _vlmCompiledComputeG: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = {
        (aLog: MLXArray, a: MLXArray, dtBias: MLXArray) -> MLXArray in
        let decay = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
        return decay
    }
    guard HardwareInfo.isCompiledDecodeSupported else { return body }
    let compiled = compile(shapeless: true, body)
    // Plain body inside the outer compiled-decode trace — nested compile is illegal.
    return { l, a, d in CompiledDecodeTrace.isActive ? body(l, a, d) : compiled(l, a, d) }
}()

/// Compiled swiglu: silu(gate) * x → 1 fused Metal dispatch instead of 2.
/// Matches Python mlx_lm/models/activations.py @partial(mx.compile, shapeless=True) swiglu.
/// Called 40×/step in Qwen3.5 MoE (shared_expert MLP).
private let _vlmCompiledSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, x: MLXArray) -> MLXArray in
        silu(gate) * x
    }
    let compiled = compile(shapeless: true, body)
    // Plain body inside the outer compiled-decode trace — nested compile is illegal.
    return { g, x in CompiledDecodeTrace.isActive ? body(g, x) : compiled(g, x) }
}()

/// Compiled precise swiglu: casts to float32, does silu+mul, casts back.
/// Matches Python mlx_lm/models/qwen3_next.py @partial(mx.compile, shapeless=True) _precise_swiglu.
/// Called 30×/step in Qwen3.5 (linear_attention RMSNormGated output path).
private let _vlmCompiledPreciseSwiGLU: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = {
        (h: MLXArray, gate: MLXArray, x: MLXArray) -> MLXArray in
        let gateF32 = silu(gate.asType(.float32))
        let xF32 = x.asType(.float32)
        return (gateF32 * xF32).asType(h.dtype)
    }
    let compiled = compile(shapeless: true, body)
    // Plain body inside the outer compiled-decode trace — nested compile is illegal.
    return { h, g, x in CompiledDecodeTrace.isActive ? body(h, g, x) : compiled(h, g, x) }
}()

/// Qwen3.8 Flash Next uses the same four-way GDN projection shape in every
/// linear-attention layer.  Pass each layer's packed affine tensors as graph
/// inputs so one trusted BF16 decode graph can be reused without retaining
/// layer-specific weight constants.
enum Qwen4ExpCompiledGDNInputs {
    typealias Region = @Sendable ([MLXArray]) -> [MLXArray]

    private static let enabled: Bool = {
        let value =
            ProcessInfo.processInfo.environment["VMLINUX_QWEN4_EXP_COMPILE_GDN"]
            ?? "1"
        return value != "0" && value.lowercased() != "false"
    }()
    private static let allowFP32Tail =
        ProcessInfo.processInfo.environment[
            "VMLINUX_QWEN4_EXP_COMPILE_GDN_FP32_TAIL"] == "1"
    private static let lock = NSLock()
    nonisolated(unsafe) private static var regions: [String: Region] = [:]
    nonisolated(unsafe) private static var didReport = false
    nonisolated(unsafe) private static var didReportTail = false
    nonisolated(unsafe) private static var didReportFrontRejection = false
    nonisolated(unsafe) private static var didReportTailRejection = false

    static func call(
        input: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        splitIndices: [Int],
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    ) -> [MLXArray]? {
        let metadataDType = scales.dtype
        guard enabled, !CompiledDecodeTrace.isActive, input.dim(1) == 1,
            input.dtype == .bfloat16,
            metadataDType == .bfloat16 || metadataDType == .float16,
            biases.dtype == metadataDType
        else { return nil }

        let key =
            ([
                String(describing: metadataDType), String(groupSize),
                String(bits), String(describing: mode),
            ]
            + splitIndices.map(String.init)).joined(separator: "|")
        lock.lock()
        var region = regions[key]
        if region == nil {
            region = vmlxTrustedCompile { (args: [MLXArray]) -> [MLXArray] in
                let combined = quantizedMM(
                    args[0], args[1], scales: args[2], biases: args[3],
                    transpose: true, groupSize: groupSize, bits: bits, mode: mode)
                return MLX.split(combined, indices: splitIndices, axis: -1)
            }
            regions[key] = region
        }
        if !didReport {
            didReport = true
            FileHandle.standardError.write(
                Data(
                    "[Qwen4Exp] compiled_gdn_input_projection=active shared_weight_inputs=true input=bfloat16 metadata=\(metadataDType)\n"
                        .utf8))
        }
        lock.unlock()

        let outputs = region!([input, weight, scales, biases])
        return outputs.count == 4 ? outputs : nil
    }

    static func callFront(
        input: MLXArray,
        convState: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        convWeight: MLXArray,
        splitIndices: [Int],
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode,
        convDim: Int,
        convKernelSize: Int,
        keyDim: Int,
        numKHeads: Int,
        headKDim: Int,
        numVHeads: Int,
        headVDim: Int
    ) -> [MLXArray]? {
        let metadataDType = scales.dtype
        let supported =
            enabled && !CompiledDecodeTrace.isActive && input.dim(1) == 1
            && input.dtype == .bfloat16
            && (convState.dtype == .bfloat16 || convState.dtype == .float32)
            && (metadataDType == .bfloat16 || metadataDType == .float16)
            && biases.dtype == metadataDType && convWeight.dtype == .bfloat16
        guard supported
        else {
            lock.lock()
            if !didReportFrontRejection {
                didReportFrontRejection = true
                FileHandle.standardError.write(
                    Data(
                        ("[Qwen4Exp] compiled_gdn_front=rejected"
                            + " enabled=\(enabled) trace=\(CompiledDecodeTrace.isActive)"
                            + " input=\(input.shape):\(input.dtype)"
                            + " conv_state=\(convState.shape):\(convState.dtype)"
                            + " scales=\(metadataDType) biases=\(biases.dtype)"
                            + " conv_weight=\(convWeight.dtype)\n").utf8))
            }
            lock.unlock()
            return nil
        }

        let key =
            ([
                "front", String(describing: metadataDType), String(groupSize),
                String(bits), String(describing: mode),
                String(describing: convState.dtype),
                String(convDim), String(convKernelSize), String(keyDim),
                String(numKHeads), String(headKDim), String(numVHeads),
                String(headVDim),
            ] + splitIndices.map(String.init)).joined(separator: "|")
        lock.lock()
        var region = regions[key]
        if region == nil {
            region = vmlxTrustedCompile { (args: [MLXArray]) -> [MLXArray] in
                let input = args[0]
                let B = input.dim(0)
                let S = input.dim(1)
                let combined = quantizedMM(
                    input, args[2], scales: args[3], biases: args[4],
                    transpose: true, groupSize: groupSize, bits: bits, mode: mode)
                let projections = MLX.split(combined, indices: splitIndices, axis: -1)
                let z = projections[1].reshaped(B, S, numVHeads, headVDim)

                let convInput = concatenated([args[1], projections[0]], axis: 1)
                    .reshaped(B, args[1].dim(1) + S, convDim)
                let convEnd = convInput.dim(1)
                let convTail = convInput[
                    0..., (convEnd - (convKernelSize - 1)) ..< convEnd, 0...]
                let convOut = silu(
                    conv1d(
                        convInput, args[5], stride: 1, padding: 0,
                        dilation: 1, groups: convDim))
                let qkv = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
                let q = qkv[0].reshaped(B, S, numKHeads, headKDim)
                let k = qkv[1].reshaped(B, S, numKHeads, headKDim)
                let v = qkv[2].reshaped(B, S, numVHeads, headVDim)
                let invScale = pow(Float(headKDim), -0.5)
                let qNormed =
                    MLXArray(pow(invScale, 2), dtype: q.dtype)
                    * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
                let kNormed =
                    MLXArray(invScale, dtype: k.dtype)
                    * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)
                return [
                    z, projections[2], projections[3], qNormed, kNormed, v,
                    convTail,
                ]
            }
            regions[key] = region
        }
        if !didReport {
            didReport = true
            FileHandle.standardError.write(
                Data(
                    "[Qwen4Exp] compiled_gdn_front=active shared_weight_inputs=true input=bfloat16 metadata=\(metadataDType) conv_state=\(convState.dtype) recurrence=float32\n"
                        .utf8))
        }
        lock.unlock()

        let outputs = region!([
            input, convState, weight, scales, biases, convWeight,
        ])
        return outputs.count == 7 ? outputs : nil
    }

    static func callTail(
        output: MLXArray,
        gate: MLXArray,
        sigmoidGate: Bool,
        normWeight: MLXArray,
        outWeight: MLXArray,
        outScales: MLXArray,
        outBiases: MLXArray,
        eps: Float,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    ) -> MLXArray? {
        let metadataDType = outScales.dtype
        let supported =
            enabled && !CompiledDecodeTrace.isActive && output.dim(1) == 1
            && (output.dtype == .bfloat16
                || (output.dtype == .float32 && allowFP32Tail))
            && (gate.dtype == .bfloat16 || gate.dtype == .float32)
            && normWeight.dtype == .bfloat16
            && (metadataDType == .bfloat16 || metadataDType == .float16)
            && outBiases.dtype == metadataDType
        guard supported
        else {
            lock.lock()
            if output.dim(1) == 1, !didReportTailRejection {
                didReportTailRejection = true
                FileHandle.standardError.write(
                    Data(
                        ("[Qwen4Exp] compiled_gdn_tail=rejected"
                            + " enabled=\(enabled) trace=\(CompiledDecodeTrace.isActive)"
                            + " output=\(output.shape):\(output.dtype)"
                            + " gate=\(gate.shape):\(gate.dtype) norm=\(normWeight.dtype)"
                            + " scales=\(metadataDType) biases=\(outBiases.dtype)\n").utf8))
            }
            lock.unlock()
            return nil
        }

        let key = [
            "tail", String(output.dim(-1)), String(gate.dim(-1)),
            sigmoidGate ? "sigmoid" : "silu",
            String(describing: output.dtype),
            String(describing: gate.dtype),
            String(describing: metadataDType),
            String(groupSize), String(bits), String(describing: mode), String(eps),
        ].joined(separator: "|")
        lock.lock()
        var region = regions[key]
        if region == nil {
            region = vmlxTrustedCompile { (args: [MLXArray]) -> [MLXArray] in
                let normalized = MLXFast.rmsNorm(args[0], weight: args[2], eps: eps)
                let gate =
                    sigmoidGate
                    ? sigmoid(args[1].asType(.float32))
                    : silu(args[1].asType(.float32))
                let gated = (normalized.asType(.float32) * gate)
                    .asType(args[0].dtype)
                let projected = quantizedMM(
                    gated.reshaped(gated.dim(0), gated.dim(1), -1),
                    args[3], scales: args[4], biases: args[5],
                    transpose: true, groupSize: groupSize, bits: bits, mode: mode)
                return [projected]
            }
            regions[key] = region
        }
        if !didReportTail {
            didReportTail = true
            FileHandle.standardError.write(
                Data(
                    "[Qwen4Exp] compiled_gdn_tail=active gate_mode=\(sigmoidGate ? "sigmoid" : "silu") recurrence_output=\(output.dtype) gate=\(gate.dtype) metadata=\(metadataDType) shared_weight_inputs=true\n"
                        .utf8))
        }
        lock.unlock()

        return region!([
            output, gate, normWeight, outWeight, outScales, outBiases,
        ]).first
    }
}

private enum Qwen4ExpCompiledMoE {
    typealias Region = @Sendable ([MLXArray]) -> [MLXArray]

    static let enabled: Bool = {
        let value =
            ProcessInfo.processInfo.environment["VMLINUX_QWEN4_EXP_COMPILE_MOE"]
            ?? "1"
        return value != "0" && value.lowercased() != "false"
    }()
    private static let lock = NSLock()
    nonisolated(unsafe) private static var routerRegions: [String: Region] = [:]
    nonisolated(unsafe) private static var sharedRegions: [String: Region] = [:]
    nonisolated(unsafe) private static var didReportRouter = false
    nonisolated(unsafe) private static var didReportShared = false

    static func router(
        _ x: MLXArray,
        weight: MLXArray, scales: MLXArray, biases: MLXArray,
        groupSize: Int, bits: Int, mode: QuantizationMode,
        topK: Int, normTopK: Bool
    ) -> (indices: MLXArray, scores: MLXArray)? {
        guard enabled, !CompiledDecodeTrace.isActive, x.dim(1) == 1,
            x.dtype == .bfloat16, scales.dtype == .bfloat16,
            biases.dtype == .bfloat16
        else { return nil }
        let experts = weight.dim(0)
        let key = "\(experts)|\(topK)|\(normTopK)|\(groupSize)|\(bits)|\(mode)"
        lock.lock()
        var region = routerRegions[key]
        if region == nil {
            region = vmlxTrustedCompile { (args: [MLXArray]) -> [MLXArray] in
                let logits = quantizedMM(
                    args[0], args[1], scales: args[2], biases: args[3],
                    transpose: true, groupSize: groupSize, bits: bits, mode: mode)
                let gates = MLX.softmax(logits, axis: -1, precise: true)
                let kth = experts - topK
                let indices = MLX.argPartition(gates, kth: kth, axis: -1)[
                    .ellipsis, kth...]
                var scores = MLX.takeAlong(gates, indices, axis: -1)
                if normTopK {
                    scores = scores / scores.sum(axis: -1, keepDims: true)
                }
                return [indices, scores]
            }
            routerRegions[key] = region
        }
        if !didReportRouter {
            didReportRouter = true
            FileHandle.standardError.write(
                Data(
                    "[Qwen4Exp] compiled_moe_router=active shared_weight_inputs=true dtype=bfloat16\n"
                        .utf8))
        }
        lock.unlock()
        let outputs = region!([x, weight, scales, biases])
        guard outputs.count == 2 else { return nil }
        return (outputs[0], outputs[1])
    }

    static func denseRouter(
        _ x: MLXArray, weight: MLXArray, topK: Int, normTopK: Bool
    ) -> (indices: MLXArray, scores: MLXArray)? {
        guard enabled, !CompiledDecodeTrace.isActive, x.dim(1) == 1,
            x.dtype == .bfloat16, weight.dtype == .bfloat16
        else { return nil }
        let experts = weight.dim(0)
        let key = "dense|\(experts)|\(topK)|\(normTopK)"
        lock.lock()
        var region = routerRegions[key]
        if region == nil {
            region = vmlxTrustedCompile { (args: [MLXArray]) -> [MLXArray] in
                let gates = MLX.softmax(
                    matmul(args[0], args[1].transposed()), axis: -1, precise: true)
                let kth = experts - topK
                let indices = MLX.argPartition(gates, kth: kth, axis: -1)[
                    .ellipsis, kth...]
                var scores = MLX.takeAlong(gates, indices, axis: -1)
                if normTopK {
                    scores = scores / scores.sum(axis: -1, keepDims: true)
                }
                return [indices, scores]
            }
            routerRegions[key] = region
        }
        if !didReportRouter {
            didReportRouter = true
            FileHandle.standardError.write(
                Data(
                    "[Qwen4Exp] compiled_moe_router=active kind=dense shared_weight_inputs=true dtype=bfloat16\n"
                        .utf8))
        }
        lock.unlock()
        let outputs = region!([x, weight])
        guard outputs.count == 2 else { return nil }
        return (outputs[0], outputs[1])
    }

    static func sharedExpert(
        _ x: MLXArray,
        gateWeight: MLXArray, gateScales: MLXArray, gateBiases: MLXArray,
        upWeight: MLXArray, upScales: MLXArray, upBiases: MLXArray,
        downWeight: MLXArray, downScales: MLXArray, downBiases: MLXArray,
        sharedGateWeight: MLXArray,
        groupSize: Int, bits: Int, mode: QuantizationMode
    ) -> MLXArray? {
        guard enabled, !CompiledDecodeTrace.isActive, x.dim(1) == 1,
            x.dtype == .bfloat16,
            [
                gateScales, gateBiases, upScales, upBiases, downScales,
                downBiases,
            ]
            .allSatisfy({ $0.dtype == .bfloat16 })
                && sharedGateWeight.dtype == .bfloat16
        else { return nil }
        let key = "shared|\(x.dim(-1))|\(gateWeight.dim(0))|\(groupSize)|\(bits)|\(mode)"
        lock.lock()
        var region = sharedRegions[key]
        if region == nil {
            region = vmlxTrustedCompile { (args: [MLXArray]) -> [MLXArray] in
                let gate = quantizedMM(
                    args[0], args[1], scales: args[2], biases: args[3],
                    transpose: true, groupSize: groupSize, bits: bits, mode: mode)
                let up = quantizedMM(
                    args[0], args[4], scales: args[5], biases: args[6],
                    transpose: true, groupSize: groupSize, bits: bits, mode: mode)
                let activated = silu(gate) * up
                let shared = quantizedMM(
                    activated, args[7], scales: args[8], biases: args[9],
                    transpose: true, groupSize: groupSize, bits: bits, mode: mode)
                let gateOutput = matmul(args[0], args[10].transposed())
                return [sigmoid(gateOutput) * shared]
            }
            sharedRegions[key] = region
        }
        if !didReportShared {
            didReportShared = true
            FileHandle.standardError.write(
                Data(
                    "[Qwen4Exp] compiled_moe_shared_expert=active shared_weight_inputs=true dtype=bfloat16\n"
                        .utf8))
        }
        lock.unlock()
        let outputs = region!([
            x,
            gateWeight, gateScales, gateBiases,
            upWeight, upScales, upBiases,
            downWeight, downScales, downBiases,
            sharedGateWeight,
        ])
        return outputs.first
    }

    private static let postRegion = vmlxTrustedCompile(shapeless: true) {
        (routed: MLXArray, scores: MLXArray, shared: MLXArray) -> MLXArray in
        let combined = (routed * scores[.ellipsis, .newAxis]).sum(axis: -2)
        return combined + shared
    }

    static func post(
        routed: MLXArray, scores: MLXArray, shared: MLXArray
    ) -> MLXArray? {
        guard enabled, !CompiledDecodeTrace.isActive, routed.dtype == .bfloat16,
            scores.dtype == .bfloat16, shared.dtype == .bfloat16
        else { return nil }
        return postRegion(routed, scores, shared)
    }
}

private func computeGatedDeltaG(_ aLog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray)
    -> MLXArray
{
    _vlmCompiledComputeG(aLog, a, dtBias)
}

/// Raw step — called by compiled wrapper or directly for masked prefill.
private func _vlmRawStepOps(
    q: MLXArray, k: MLXArray, v: MLXArray,
    g: MLXArray, beta: MLXArray, state: MLXArray
) -> (MLXArray, MLXArray) {
    let decay: MLXArray
    if g.ndim == 2 {
        decay = expandedDimensions(g, axes: [2, 3])
    } else if g.ndim == 3 {
        decay = expandedDimensions(g, axis: -2)
    } else {
        fatalError("Unsupported gating shape \(g.shape)")
    }

    var state = state * decay
    let kvMem = (state * expandedDimensions(k, axis: -2)).sum(axis: -1)
    let delta = (v - kvMem) * expandedDimensions(beta, axis: -1)
    state = state + expandedDimensions(k, axis: -2) * expandedDimensions(delta, axis: -1)
    let y = (state * expandedDimensions(q, axis: -2)).sum(axis: -1)
    return (y, state)
}

/// Compiled GatedDelta step — fuses ~10 ops into 1 Metal dispatch.
private let _vlmCompiledStepOps: @Sendable ([MLXArray]) -> [MLXArray] =
    compile { (args: [MLXArray]) -> [MLXArray] in
        let (y, state) = _vlmRawStepOps(
            q: args[0], k: args[1], v: args[2],
            g: args[3], beta: args[4], state: args[5])
        return [y, state]
    }

private func gatedDeltaStepOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    if let mask {
        let oldState = state
        let (y, newState) = _vlmRawStepOps(q: q, k: k, v: v, g: g, beta: beta, state: state)
        let expandedMask: MLXArray
        if mask.ndim == 1 {
            expandedMask = expandedDimensions(mask, axes: [1, 2, 3])
        } else if mask.ndim == 2 {
            expandedMask = expandedDimensions(mask, axes: [2, 3])
        } else if mask.ndim == 3 {
            expandedMask = expandedDimensions(mask, axis: -1)
        } else {
            fatalError("Unsupported mask shape \(mask.shape)")
        }
        return (y.asType(q.dtype), MLX.where(expandedMask, newState, oldState))
    }

    let result = _vlmCompiledStepOps([q, k, v, g, beta, state])
    return (result[0].asType(q.dtype), result[1])
}

private func gatedDeltaOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil,
    roundStateEachStep: Bool = false
) -> (MLXArray, MLXArray) {
    let B = q.dim(0)
    let T = q.dim(1)
    let Hk = q.dim(2)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    var q = q
    var k = k

    let repeatFactor = Hv / Hk
    if repeatFactor > 1 {
        q = repeated(q, count: repeatFactor, axis: -2)
        k = repeated(k, count: repeatFactor, axis: -2)
    }

    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)

    var ys = [MLXArray]()
    ys.reserveCapacity(T)

    for t in 0 ..< T {
        let qT = q[0..., t]
        let kT = k[0..., t]
        let vT = v[0..., t]
        let gT = g[0..., t]
        let betaT = beta[0..., t]
        let maskT = mask == nil ? nil : mask![0..., t]

        let (y, newState) = gatedDeltaStepOps(
            q: qT,
            k: kT,
            v: vT,
            g: gT,
            beta: betaT,
            state: state,
            mask: maskT
        )
        ys.append(y)
        state =
            roundStateEachStep
            ? newState.asType(q.dtype).asType(.float32)
            : newState
    }

    let y = MLX.stacked(ys, axis: 1)
    return (y, state)
}

// MARK: - Fused Metal kernel for gated delta step
//
// Replaces the ops-based `gatedDeltaOps` fallback (7+ Metal dispatches per layer
// per token) with a single fused kernel matching Python mlx_lm's gated_delta.py.
// For Qwen3.5 with 30 linear_attention layers this eliminates ~210 dispatches
// per decode step, closing most of the Python ↔ Swift gap on SSM+MoE models.

private func makeVLMGatedDeltaKernel(
    hasMask: Bool,
    roundStateEachStep: Bool = false
) -> MLXFast.MLXFastKernel? {
    let maskSource = hasMask ? "mask[b_idx * T + t]" : "true"
    let stepRoundSource =
        roundStateEachStep
        ? """
                for (int i = 0; i < n_per_t; ++i) {
                  state[i] = static_cast<float>(static_cast<InT>(state[i]));
                }
        """
        : ""
    let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            auto g_ = g + b_idx * T * Hv;
            auto b_ = b + b_idx * T * Hv;

            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            for (int t = 0; t < T; ++t) {
              if (\(maskSource)) {
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] * g_[hv_idx];
                  kv_mem += state[i] * k_[s_idx];
                }
                kv_mem = simd_sum(kv_mem);

                // `b` remains in the model's BF16 activation dtype. Compute
                // sigmoid directly in the FP32 recurrence register instead
                // of materializing a separate BF16 or FP32 beta tensor.
                auto beta_value = 1.0f
                    / (1.0f + metal::fast::exp(-static_cast<float>(b_[hv_idx])));
                auto delta = (v_[dv_idx] - kv_mem)
                    * beta_value;

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] + k_[s_idx] * delta;
                  out += state[i] * q_[s_idx];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                  y[dv_idx] = static_cast<InT>(out);
                }
        \(stepRoundSource)
              } else {
                y[dv_idx] = static_cast<InT>(0);
              }
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              b_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<StT>(state[i]);
            }
        """
    var inputNames = ["q", "k", "v", "g", "b", "state_in", "T"]
    if hasMask { inputNames.append("mask") }
    let suffix = (hasMask ? "_mask" : "") + (roundStateEachStep ? "_strict" : "_fast")
    return MLXFast.metalKernel(
        name: "vlm_gated_delta_step\(suffix)",
        inputNames: inputNames,
        outputNames: ["y", "state_out"],
        source: source
    )
}

private final class VLMGatedDeltaKernelManager: @unchecked Sendable {
    static let shared = VLMGatedDeltaKernelManager()
    let kernel: MLXFast.MLXFastKernel?
    let kernelMasked: MLXFast.MLXFastKernel?
    let strictKernel: MLXFast.MLXFastKernel?
    let strictKernelMasked: MLXFast.MLXFastKernel?
    private init() {
        kernel = makeVLMGatedDeltaKernel(hasMask: false, roundStateEachStep: false)
        kernelMasked = makeVLMGatedDeltaKernel(hasMask: true, roundStateEachStep: false)
        strictKernel = makeVLMGatedDeltaKernel(hasMask: false, roundStateEachStep: true)
        strictKernelMasked = makeVLMGatedDeltaKernel(hasMask: true, roundStateEachStep: true)
    }

    func kernel(hasMask: Bool, roundStateEachStep: Bool) -> MLXFast.MLXFastKernel? {
        switch (hasMask, roundStateEachStep) {
        case (false, false): return kernel
        case (true, false): return kernelMasked
        case (false, true): return strictKernel
        case (true, true): return strictKernelMasked
        }
    }
}

private func vlmGatedDeltaKernel(
    q: MLXArray, k: MLXArray, v: MLXArray,
    g: MLXArray, b: MLXArray, state: MLXArray,
    mask: MLXArray? = nil,
    roundStateEachStep: Bool = false
) -> (MLXArray, MLXArray) {
    let B = k.dim(0)
    let T = k.dim(1)
    let Hk = k.dim(2)
    let Dk = k.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let inputType = q.dtype
    let stateType = state.dtype

    let selectedKernel: MLXFast.MLXFastKernel?
    var inputs: [MLXArray] = [q, k, v, g, b, state, MLXArray(T)]
    if let mask {
        selectedKernel = VLMGatedDeltaKernelManager.shared.kernel(
            hasMask: true,
            roundStateEachStep: roundStateEachStep)
        inputs.append(mask)
    } else {
        selectedKernel = VLMGatedDeltaKernelManager.shared.kernel(
            hasMask: false,
            roundStateEachStep: roundStateEachStep)
    }
    guard let kernel = selectedKernel else {
        fatalError("VLM gated delta kernel not available")
    }
    let outputs = kernel(
        inputs,
        template: [
            ("InT", inputType),
            ("StT", stateType),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: (32, Dv, B * Hv),
        threadGroup: (32, 4, 1),
        outputShapes: [[B, T, Hv, Dv], state.shape],
        outputDTypes: [inputType, stateType]
    )
    return (outputs[0], outputs[1])
}

private func gatedDeltaUpdate(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil,
    roundStateEachStep: Bool = false
) -> (MLXArray, MLXArray) {
    let g = computeGatedDeltaG(aLog, a, dtBias)

    let B = q.dim(0)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    // Keep recurrent state in float32 so cached and cold prefill partitions
    // preserve the same GatedDelta recurrence. This mirrors the text path and
    // upstream mlx-lm / mlx-swift-lm.
    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)
    if state.dtype != .float32 {
        state = state.asType(.float32)
    }
    QwenGatedDeltaDTypeTrace.reportOnce(
        q: q, k: k, v: v, a: a, b: b, decay: g, state: state)

    // Prefer fused Metal kernel (Python parity, ~210 fewer dispatches per token).
    // The kernel tiles Dk in 32-wide SIMD chunks; unsupported head widths must
    // use the ops fallback instead of compiling an invalid zero/remainder tile.
    let manager = VLMGatedDeltaKernelManager.shared
    let selectedKernel = manager.kernel(
        hasMask: mask != nil,
        roundStateEachStep: roundStateEachStep)
    if selectedKernel != nil && Dk >= 32 && Dk % 32 == 0 {
        return vlmGatedDeltaKernel(
            q: q, k: k, v: v, g: g, b: b, state: state, mask: mask,
            roundStateEachStep: roundStateEachStep)
    }
    let beta = sigmoid(b)
    return gatedDeltaOps(
        q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask,
        roundStateEachStep: roundStateEachStep)
}

private enum QwenGatedDeltaDTypeTrace {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var didReport = false

    static func reportOnce(
        q: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray, b: MLXArray,
        decay: MLXArray, state: MLXArray
    ) {
        lock.lock()
        defer { lock.unlock() }
        guard !didReport else { return }
        didReport = true
        FileHandle.standardError.write(
            Data(
                "[QwenGDN] live_dtype_boundaries q=\(q.dtype) k=\(k.dtype) v=\(v.dtype) a=\(a.dtype) b=\(b.dtype) beta=fused_from_\(b.dtype) decay=\(decay.dtype) recurrent_state=\(state.dtype) output=\(q.dtype)\n"
                    .utf8))
    }
}

// MARK: - Configuration

public struct Qwen35Configuration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public var modelType: String = ""
        public var hiddenSize: Int = 4096
        public var hiddenLayers: Int = 32
        public var intermediateSize: Int = 14_336
        public var attentionHeads: Int = 32
        public var kvHeads: Int = 8
        public var linearNumValueHeads: Int = 64
        public var linearNumKeyHeads: Int = 16
        public var linearKeyHeadDim: Int = 192
        public var linearValueHeadDim: Int = 128
        public var linearConvKernelDim: Int = 4
        public var rmsNormEps: Float = 1e-6
        public var vocabularySize: Int = 248_320
        public var ropeTheta: Float = 100_000.0
        public var partialRotaryFactor: Float = 0.25
        public var maxPositionEmbeddings: Int = 131_072
        public var tieWordEmbeddings: Bool = false
        public var attentionBias: Bool = false
        public var headDim: Int?
        public var ropeParameters: [String: StringOrNumber]?
        public var fullAttentionInterval: Int = 4
        public var mtpNumHiddenLayers: Int = 0
        public var weightFormat: String = ""
        /// Authoritative RMSNorm convention declared by the bundle (config.json `norm_convention`).
        /// When set it overrides the architecture default; nil when the bundle declares none.
        public var normConvention: String? = nil
        public var mxtqBits: Int = 2
        public var mxtqGateUpBits: Int = 2
        public var mxtqDownBits: Int = 2
        public var mxtqSeed: Int = 42

        // MoE fields
        public var numExperts: Int = 0
        public var numExpertsPerTok: Int = 0
        public var decoderSparseStep: Int = 1
        public var sharedExpertIntermediateSize: Int = 0
        public var moeIntermediateSize: Int = 0
        public var normTopkProb: Bool = true

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case linearNumValueHeads = "linear_num_value_heads"
            case linearNumKeyHeads = "linear_num_key_heads"
            case linearKeyHeadDim = "linear_key_head_dim"
            case linearValueHeadDim = "linear_value_head_dim"
            case linearConvKernelDim = "linear_conv_kernel_dim"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
            case maxPositionEmbeddings = "max_position_embeddings"
            case tieWordEmbeddings = "tie_word_embeddings"
            case attentionBias = "attention_bias"
            case headDim = "head_dim"
            case ropeParameters = "rope_parameters"
            case fullAttentionInterval = "full_attention_interval"
            case mtpNumHiddenLayers = "mtp_num_hidden_layers"
            case weightFormat = "weight_format"
            case normConvention = "norm_convention"
            case mxtqBits = "mxtq_bits"
            case mxtqGateUpBits = "mxtq_gate_up_bits"
            case mxtqDownBits = "mxtq_down_bits"
            case mxtqSeed = "mxtq_seed"
            case numExperts = "num_experts"
            case numExpertsPerTok = "num_experts_per_tok"
            case decoderSparseStep = "decoder_sparse_step"
            case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
            case moeIntermediateSize = "moe_intermediate_size"
            case normTopkProb = "norm_topk_prob"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)

            self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
            self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
            self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
            self.intermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14_336
            self.attentionHeads =
                try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
            self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
            self.linearNumValueHeads =
                try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
            self.linearNumKeyHeads =
                try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
            self.linearKeyHeadDim =
                try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 192
            self.linearValueHeadDim =
                try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
            self.linearConvKernelDim =
                try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
            self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            self.vocabularySize =
                try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 248_320
            self.maxPositionEmbeddings =
                try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
            self.tieWordEmbeddings =
                try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
            self.attentionBias =
                try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
            self.fullAttentionInterval =
                try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
            self.mtpNumHiddenLayers =
                try container.decodeIfPresent(Int.self, forKey: .mtpNumHiddenLayers) ?? 0
            self.weightFormat =
                try container.decodeIfPresent(String.self, forKey: .weightFormat) ?? ""
            self.normConvention =
                try container.decodeIfPresent(String.self, forKey: .normConvention)
            if let flatBits = try? container.decodeIfPresent(Int.self, forKey: .mxtqBits) {
                self.mxtqBits = flatBits
            }
            self.mxtqGateUpBits =
                try container.decodeIfPresent(Int.self, forKey: .mxtqGateUpBits) ?? self.mxtqBits
            self.mxtqDownBits =
                try container.decodeIfPresent(Int.self, forKey: .mxtqDownBits) ?? self.mxtqBits
            self.mxtqSeed =
                try container.decodeIfPresent(Int.self, forKey: .mxtqSeed) ?? 42

            self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
            self.numExpertsPerTok =
                try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
            self.decoderSparseStep =
                try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
            self.sharedExpertIntermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
            self.moeIntermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
            self.normTopkProb =
                try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true

            let defaultRopeParameters: [String: StringOrNumber] = [
                "type": .string("default"),
                "mrope_section": .ints([11, 11, 10]),
                "rope_theta": .float(100_000.0),
                "partial_rotary_factor": .float(0.25),
            ]

            var decodedRope = try container.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeParameters)

            if decodedRope == nil {
                let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                let partial = try container.decodeIfPresent(
                    Float.self, forKey: .partialRotaryFactor)
                if ropeTheta != nil || partial != nil {
                    decodedRope = defaultRopeParameters
                    if let ropeTheta {
                        decodedRope?["rope_theta"] = .float(ropeTheta)
                    }
                    if let partial {
                        decodedRope?["partial_rotary_factor"] = .float(partial)
                    }
                }
            }

            if var decodedRope {
                if decodedRope["type"] == nil, let ropeType = decodedRope["rope_type"] {
                    decodedRope["type"] = ropeType
                }
                self.ropeParameters = decodedRope
                self.ropeTheta = decodedRope["rope_theta"]?.asFloat() ?? 100_000.0
                self.partialRotaryFactor = decodedRope["partial_rotary_factor"]?.asFloat() ?? 0.25
            } else {
                self.ropeParameters = defaultRopeParameters
                self.ropeTheta = 100_000.0
                self.partialRotaryFactor = 0.25
            }

            if self.headDim == nil {
                self.headDim = self.hiddenSize / self.attentionHeads
            }
        }
    }

    public typealias VisionConfiguration = Qwen3VLConfiguration.VisionConfiguration

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _ignoreIndex: Int?
    public var ignoreIndex: Int { _ignoreIndex ?? -100 }
    private let _imageTokenId: Int?
    public var imageTokenId: Int { _imageTokenId ?? 248_056 }
    private let _videoTokenId: Int?
    public var videoTokenId: Int { _videoTokenId ?? 248_057 }
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int { _imageTokenIndex ?? imageTokenId }
    private let _videoTokenIndex: Int?
    public var videoTokenIndex: Int { _videoTokenIndex ?? videoTokenId }
    private let _visionStartTokenId: Int?
    public var visionStartTokenId: Int { _visionStartTokenId ?? 248_045 }
    private let _visionEndTokenId: Int?
    public var visionEndTokenId: Int { _visionEndTokenId ?? 248_046 }
    private let _vocabSize: Int?
    public var vocabSize: Int { _vocabSize ?? textConfiguration.vocabularySize }
    private let _eosTokenId: IntOrIntArray?
    public var eosTokenId: [Int]? { _eosTokenId?.values }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _ignoreIndex = "ignore_index"
        case _imageTokenId = "image_token_id"
        case _videoTokenId = "video_token_id"
        case _imageTokenIndex = "image_token_index"
        case _videoTokenIndex = "video_token_index"
        case _visionStartTokenId = "vision_start_token_id"
        case _visionEndTokenId = "vision_end_token_id"
        case _vocabSize = "vocab_size"
        case _eosTokenId = "eos_token_id"
    }
}

// MARK: - Language

enum Qwen35Language {
    static func shouldCompileDecodeRegions(
        _ args: Qwen35Configuration.TextConfiguration,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        if let override = environment["VMLINUX_QWEN35_COMPILE_DECODE_REGIONS"] {
            return override != "0" && override.lowercased() != "false"
        }
        return args.modelType == "qwen3_5_moe_text"
            && args.hiddenSize == 2048
            && args.hiddenLayers == 40
            && args.fullAttentionInterval == 4
            && args.numExperts == 256
            && args.numExpertsPerTok == 8
            && args.moeIntermediateSize == 512
            && args.linearNumKeyHeads == 16
            && args.linearNumValueHeads == 32
            && args.linearKeyHeadDim == 128
            && args.linearValueHeadDim == 128
    }

    final class RotaryEmbedding {
        private let invFreq: MLXArray
        private let mropeSection: [Int]

        init(dim: Int, base: Float, mropeSection: [Int]) {
            let safeDim = max(1, dim)
            var freq = MLXArray(stride(from: 0, to: safeDim, by: 2)).asType(.float32)
            freq = freq / Float(safeDim)
            self.invFreq = 1.0 / pow(MLXArray(base), freq)
            self.mropeSection =
                mropeSection.count >= 3 ? mropeSection : [11, 11, 10]
        }

        private func applyInterleavedMRope(_ freqs: MLXArray) -> MLXArray {
            let freqsT = freqs[0, 0..., 0..., 0...]
            let dims = freqsT.dim(-1)
            var slices: [MLXArray] = []
            slices.reserveCapacity(dims)

            for idx in 0 ..< dims {
                var slice = freqsT[0..., 0..., idx]
                for (dim, offset) in [(1, 1), (2, 2)] {
                    let length = min(mropeSection[dim] * 3, dims)
                    if idx >= offset && idx < length && ((idx - offset) % 3 == 0) {
                        slice = freqs[dim, 0..., 0..., idx]
                        break
                    }
                }
                slices.append(slice)
            }

            return stacked(slices, axis: -1)
        }

        func callAsFunction(x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
            var positionIds = positionIds
            if positionIds.ndim == 2 {
                positionIds = broadcast(
                    positionIds[.newAxis, 0..., 0...],
                    to: [3, positionIds.dim(0), positionIds.dim(1)])
            }

            let pos = positionIds.asType(.float32)
            var inv = invFreq.asType(.float32)
            inv = inv[.newAxis, .newAxis, .newAxis, 0...]
            var freqs = pos[0..., 0..., 0..., .newAxis] * inv
            freqs = applyInterleavedMRope(freqs)

            let emb = concatenated([freqs, freqs], axis: -1)
            return (cos(emb).asType(x.dtype), sin(emb).asType(x.dtype))
        }
    }

    static func applyMultimodalRotaryPosEmb(
        q: MLXArray,
        k: MLXArray,
        cos: MLXArray,
        sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        let cos = expandedDimensions(cos, axis: 1)
        let sin = expandedDimensions(sin, axis: 1)

        let rotaryDim = cos.dim(-1)
        let qDim = q.dim(-1)
        let kDim = k.dim(-1)

        let qRot = q[.ellipsis, ..<rotaryDim]
        let kRot = k[.ellipsis, ..<rotaryDim]

        let qEmbedded = (qRot * cos) + (QwenVL.rotateHalf(qRot) * sin)
        let kEmbedded = (kRot * cos) + (QwenVL.rotateHalf(kRot) * sin)

        let qOut: MLXArray
        if rotaryDim < qDim {
            qOut = concatenated([qEmbedded, q[.ellipsis, rotaryDim...]], axis: -1)
        } else {
            qOut = qEmbedded
        }

        let kOut: MLXArray
        if rotaryDim < kDim {
            kOut = concatenated([kEmbedded, k[.ellipsis, rotaryDim...]], axis: -1)
        } else {
            kOut = kEmbedded
        }

        return (qOut, kOut)
    }

    final class RMSNormGated: Module {
        @ParameterInfo(key: "weight") var weight: MLXArray
        let eps: Float
        /// qwen4_exp declares `output_gate_type: "sigmoid"` — the gate is
        /// `sigmoid(z)` instead of the qwen3_5 family's `silu(z)`.
        let sigmoidGate: Bool

        init(dimensions: Int, eps: Float = 1e-6, sigmoidGate: Bool = false) {
            self.eps = eps
            self.sigmoidGate = sigmoidGate
            _weight.wrappedValue = MLXArray.ones([dimensions])
            super.init()
        }

        func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray? = nil) -> MLXArray {
            let x = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: eps)
            if let gate {
                if sigmoidGate {
                    return (x.asType(.float32) * sigmoid(gate.asType(.float32)))
                        .asType(hiddenStates.dtype)
                }
                // Fused precise swiglu matching Python's _precise_swiglu (1 dispatch vs 2+casts).
                return _vlmCompiledPreciseSwiGLU(hiddenStates, gate, x)
            }
            return x.asType(hiddenStates.dtype)
        }
    }

    final class Attention: Module {
        let numKeyValueHeads: Int
        let numAttentionHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        let rotaryEmbedding: RotaryEmbedding

        init(_ args: Qwen35Configuration.TextConfiguration) {
            self.numKeyValueHeads = args.kvHeads
            self.numAttentionHeads = args.attentionHeads
            self.headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
            self.scale = pow(Float(headDim), -0.5)

            _qProj.wrappedValue = Linear(
                args.hiddenSize, numAttentionHeads * headDim * 2, bias: args.attentionBias)
            _kProj.wrappedValue = Linear(
                args.hiddenSize, numKeyValueHeads * headDim, bias: args.attentionBias)
            _vProj.wrappedValue = Linear(
                args.hiddenSize, numKeyValueHeads * headDim, bias: args.attentionBias)
            _oProj.wrappedValue = Linear(
                numAttentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

            let mrope = args.ropeParameters?["mrope_section"]?.asInts() ?? [11, 11, 10]
            let rotaryDim = Int(Float(headDim) * args.partialRotaryFactor)
            self.rotaryEmbedding = RotaryEmbedding(
                dim: rotaryDim, base: args.ropeTheta, mropeSection: mrope)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let B = x.dim(0)
            let L = x.dim(1)

            let qProjOutput = qProj(x)
            let qSplit = qProjOutput.reshaped(B, L, numAttentionHeads, -1).split(parts: 2, axis: -1)
            var queries = qSplit[0]
            let gate = qSplit[1].reshaped(B, L, -1)

            var keys = kProj(x)
            var values = vProj(x)

            queries = qNorm(queries).transposed(0, 2, 1, 3)
            keys = kNorm(keys.reshaped(B, L, numKeyValueHeads, -1)).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)

            var kvSeqLen = keys.dim(-2)
            var positionIds = positionIds

            // A Compilable cache (compiled decode) returns the FULL static
            // buffer from `update` and its `makeMask` already covers it, so
            // the kvSeqLen mask slice must not run — and reading `.offset`
            // (Int) is an illegal `.item()` inside the compile trace.
            let fullBufferCache =
                !(cache is BatchKVCache)
                && graphOffsetArray(for: cache) != nil

            if positionIds == nil {
                // Build position IDs from cache offset. For batched decode with
                // BatchKVCache, use per-sequence offsets for correct positional encoding.
                if let batchCache = cache as? BatchKVCache {
                    let offsets = batchCache.offsetArray  // [B]
                    kvSeqLen += batchCache.offset + 1
                    // For L=1 decode: each sequence's position is its offset
                    // Shape: [3, B, 1] — 3 for the 3D rope dimensions
                    let base = offsets.reshaped(1, B, 1)
                    positionIds = tiled(base, repetitions: [3, 1, L])
                } else if fullBufferCache, let graphOffset = graphOffsetArray(for: cache) {
                    var base =
                        graphOffset.reshaped([]).asType(.int32)
                        + MLXArray(0 ..< L).asType(.int32)
                    base = tiled(base[.newAxis, 0...], repetitions: [B, 1])
                    positionIds = tiled(base[.newAxis, 0..., 0...], repetitions: [3, 1, 1])
                } else {
                    let offset = cache?.offset ?? 0
                    kvSeqLen += offset + 1
                    var base = MLXArray(stride(from: offset, to: offset + L, by: 1)).asType(.int32)
                    base = tiled(base[.newAxis, 0...], repetitions: [B, 1])
                    positionIds = base[.newAxis, 0..., 0...]
                    positionIds = tiled(positionIds!, repetitions: [3, 1, 1])
                }
            } else if let cache, !fullBufferCache {
                kvSeqLen += cache.offset + 1
            }

            let (cosValues, sinValues) = rotaryEmbedding(x: values, positionIds: positionIds!)
            (queries, keys) = applyMultimodalRotaryPosEmb(
                q: queries, k: keys, cos: cosValues, sin: sinValues)

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                // Full-buffer (Compilable) caches: mask and K/V both span the
                // whole static buffer — pass through unsliced.
                attentionMask =
                    fullBufferCache
                    ? .array(mask) : .array(mask[.ellipsis, 0 ..< kvSeqLen])
            } else {
                attentionMask = .none
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: attentionMask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return oProj(compiledSigmoidMultiply(output, gate))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            // Fused silu(gate) * up via compiled swiglu (1 Metal dispatch vs 2).
            downProj(_vlmCompiledSwiGLU(gateProj(x), upProj(x)))
        }
    }

    final class GatedDeltaNet: Module {
        private static let fusionDiagnosticLock = NSLock()
        private nonisolated(unsafe) static var didReportDecodeInputFusion = false

        private struct FusedDecodeInputProjection {
            let weight: MLXArray
            let scales: MLXArray
            let biases: MLXArray
            let groupSize: Int
            let bits: Int
            let mode: QuantizationMode
        }

        let hiddenSize: Int
        let numVHeads: Int
        let numKHeads: Int
        let headKDim: Int
        let headVDim: Int
        let keyDim: Int
        let valueDim: Int
        let convKernelSize: Int
        let convDim: Int
        let fuseDecodeInputProjections: Bool
        private var attemptedDecodeInputFusion = false
        private var fusedDecodeInputProjection: FusedDecodeInputProjection?

        @ModuleInfo(key: "conv1d") var conv1d: Conv1d
        @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
        @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
        @ModuleInfo(key: "in_proj_b") var inProjB: Linear
        @ModuleInfo(key: "in_proj_a") var inProjA: Linear

        @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
        @ParameterInfo(key: "A_log") var aLog: MLXArray

        @ModuleInfo(key: "norm") var norm: RMSNormGated
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(
            _ args: Qwen35Configuration.TextConfiguration,
            outputGateSigmoid: Bool = false,
            fuseDecodeInputProjections: Bool = false
        ) {
            self.hiddenSize = args.hiddenSize
            self.numVHeads = args.linearNumValueHeads
            self.numKHeads = args.linearNumKeyHeads
            self.headKDim = args.linearKeyHeadDim
            self.headVDim = args.linearValueHeadDim
            self.keyDim = headKDim * numKHeads
            self.valueDim = headVDim * numVHeads
            self.convKernelSize = args.linearConvKernelDim
            self.convDim = keyDim * 2 + valueDim
            self.fuseDecodeInputProjections = fuseDecodeInputProjections

            precondition(
                numVHeads % numKHeads == 0,
                "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))"
            )

            _conv1d.wrappedValue = Conv1d(
                inputChannels: convDim,
                outputChannels: convDim,
                kernelSize: convKernelSize,
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: convDim,
                bias: false
            )

            _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
            _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
            _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
            _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

            _dtBias.wrappedValue = MLXArray.ones([numVHeads])
            let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
            _aLog.wrappedValue = log(a)

            _norm.wrappedValue = RMSNormGated(
                dimensions: headVDim, eps: args.rmsNormEps, sigmoidGate: outputGateSigmoid)
            _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)
            super.init()
        }

        private func ensureFusedDecodeInputProjection(
            _ inputs: MLXArray
        ) -> FusedDecodeInputProjection? {
            guard
                ProcessInfo.processInfo.environment[
                    "VMLINUX_QWEN4_EXP_FUSE_DECODE_INPUTS"] != "0",
                fuseDecodeInputProjections, inputs.dim(1) == 1,
                !CompiledDecodeTrace.isActive
            else { return nil }

            if !attemptedDecodeInputFusion {
                attemptedDecodeInputFusion = true
                let modules = [inProjQKV, inProjZ, inProjB, inProjA]
                guard let first = modules[0] as? QuantizedLinear,
                    let firstBiases = first.biases,
                    first.bias == nil,
                    modules.allSatisfy({ module in
                        guard let quantized = module as? QuantizedLinear,
                            quantized.biases != nil, quantized.bias == nil
                        else { return false }
                        return quantized.groupSize == first.groupSize
                            && quantized.bits == first.bits
                            && quantized.mode == first.mode
                            && quantized.scales.dtype == first.scales.dtype
                            && quantized.biases?.dtype == firstBiases.dtype
                            && quantized.scales.dtype == quantized.biases?.dtype
                    })
                else { return nil }

                let quantized = modules.map { $0 as! QuantizedLinear }
                let weight = concatenated(quantized.map(\.weight), axis: 0)
                let scales = concatenated(quantized.map(\.scales), axis: 0)
                let biases = concatenated(
                    quantized.map { $0.biases ?? firstBiases }, axis: 0)
                MLX.eval(weight, scales, biases)
                fusedDecodeInputProjection = FusedDecodeInputProjection(
                    weight: weight, scales: scales, biases: biases,
                    groupSize: first.groupSize, bits: first.bits, mode: first.mode)
                Self.fusionDiagnosticLock.lock()
                if !Self.didReportDecodeInputFusion {
                    Self.didReportDecodeInputFusion = true
                    FileHandle.standardError.write(
                        Data(
                            "[Qwen4Exp] fused_gdn_decode_input_projections=active dtype=\(inputs.dtype)\n"
                                .utf8))
                }
                Self.fusionDiagnosticLock.unlock()
            }

            return fusedDecodeInputProjection
        }

        private func fusedDecodeInputs(_ inputs: MLXArray) -> [MLXArray]? {
            guard let fusedDecodeInputProjection = ensureFusedDecodeInputProjection(inputs)
            else { return nil }
            let splitIndices = [
                convDim, convDim + valueDim, convDim + valueDim + numVHeads,
            ]
            let combined = Qwen4ExpBF16Affine.dense(
                inputs, fusedDecodeInputProjection.weight,
                scales: fusedDecodeInputProjection.scales,
                biases: fusedDecodeInputProjection.biases,
                groupSize: fusedDecodeInputProjection.groupSize,
                bits: fusedDecodeInputProjection.bits,
                mode: fusedDecodeInputProjection.mode)
            return MLX.split(
                combined,
                indices: splitIndices,
                axis: -1)
        }

        private func compiledDecodeFront(
            _ inputs: MLXArray, convState: MLXArray
        ) -> [MLXArray]? {
            guard let fused = ensureFusedDecodeInputProjection(inputs) else { return nil }
            return Qwen4ExpCompiledGDNInputs.callFront(
                input: inputs,
                convState: convState,
                weight: fused.weight,
                scales: fused.scales,
                biases: fused.biases,
                convWeight: conv1d.weight,
                splitIndices: [
                    convDim, convDim + valueDim, convDim + valueDim + numVHeads,
                ],
                groupSize: fused.groupSize,
                bits: fused.bits,
                mode: fused.mode,
                convDim: convDim,
                convKernelSize: convKernelSize,
                keyDim: keyDim,
                numKHeads: numKHeads,
                headKDim: headKDim,
                numVHeads: numVHeads,
                headVDim: headVDim)
        }

        private func compiledDecodeTail(_ output: MLXArray, gate: MLXArray) -> MLXArray? {
            guard fuseDecodeInputProjections,
                let quantized = outProj as? QuantizedLinear,
                let biases = quantized.biases,
                quantized.bias == nil
            else { return nil }
            return Qwen4ExpCompiledGDNInputs.callTail(
                output: output, gate: gate, sigmoidGate: norm.sigmoidGate,
                normWeight: norm.weight,
                outWeight: quantized.weight, outScales: quantized.scales,
                outBiases: biases, eps: norm.eps,
                groupSize: quantized.groupSize, bits: quantized.bits,
                mode: quantized.mode)
        }

        /// One-shot (per process) diagnostic for a discarded incompatible
        /// cache slot — the reset self-heals, but a silent reset would hide
        /// the upstream restore bug that produced the bad shape.
        private static let discardedCacheWarningLock = NSLock()
        private nonisolated(unsafe) static var didWarnDiscardedCache = false

        static func warnDiscardedCacheState(slot: String, shape: [Int]) {
            discardedCacheWarningLock.lock()
            defer { discardedCacheWarningLock.unlock() }
            guard !didWarnDiscardedCache else { return }
            didWarnDiscardedCache = true
            print(
                "[Qwen35] GatedDeltaNet discarded incompatible \(slot) cache state "
                    + "(shape \(shape)); resetting linear-attention state")
        }

        func callAsFunction(
            _ inputs: MLXArray,
            mask: MLXArray? = nil,
            cache: MambaCache? = nil,
            recordPrefixCommitStates: Bool = false
        ) -> MLXArray {
            let B = inputs.dim(0)
            let S = inputs.dim(1)

            // A restored cache (paged/hybrid restore, prefix commit) can hand
            // back a slot whose shape no longer matches this layer — an
            // over- or under-rank array here trips the MLXArray subscript
            // rank precondition below and aborts the app. Discard the slot
            // and restart from zero state instead: one degraded generation
            // beats a crash.
            let convState: MLXArray
            if let cacheState = cache?[0],
                cacheState.ndim == 3, cacheState.dim(0) == B, cacheState.dim(2) == convDim
            {
                convState = cacheState
            } else {
                if let cacheState = cache?[0] {
                    Self.warnDiscardedCacheState(slot: "conv", shape: cacheState.shape)
                }
                convState = MLXArray.zeros(
                    [B, max(0, convKernelSize - 1), convDim], dtype: inputs.dtype)
            }

            // Staged verify (compiled DFlash 2): committed cache slots and
            // offset stay UNTOUCHED for the whole forward; everything the
            // post-acceptance commit needs goes into fixed staging slots.
            let stageVerify =
                cache != nil && S > 1 && mask == nil
                && NativeMTPVerifierStatePolicy.shouldStageVerifyInputs
            let front = compiledDecodeFront(inputs, convState: convState)
            let z: MLXArray
            let b: MLXArray
            let a: MLXArray
            let qNormed: MLXArray
            let kNormed: MLXArray
            let v: MLXArray
            let convInput: MLXArray
            if let front {
                z = front[0]
                b = front[1]
                a = front[2]
                qNormed = front[3]
                kNormed = front[4]
                v = front[5]
                convInput = front[6]
                if let cache, convKernelSize > 1 { cache[0] = front[6] }
            } else {
                let fusedInputs = fusedDecodeInputs(inputs)
                var mixedQKV = fusedInputs?[0] ?? inProjQKV(inputs)
                z = (fusedInputs?[1] ?? inProjZ(inputs)).reshaped(
                    B, S, numVHeads, headVDim)
                b = fusedInputs?[2] ?? inProjB(inputs)
                a = fusedInputs?[3] ?? inProjA(inputs)

                if let mask {
                    mixedQKV = MLX.where(mask[.ellipsis, .newAxis], mixedQKV, 0)
                }
                convInput = concatenated([convState, mixedQKV], axis: 1)
                    .reshaped(B, convState.dim(1) + S, convDim)
                if let cache, convKernelSize > 1 {
                    let end = convInput.dim(1)
                    let start = max(0, end - (convKernelSize - 1))
                    let tail = convInput[0..., start ..< end, 0...]
                    if stageVerify {
                        cache.stageVerifySlot(7, tail)
                    } else {
                        cache[0] = tail
                    }
                }

                let convOut = silu(conv1d(convInput))
                let split = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
                let q = split[0].reshaped(B, S, numKHeads, headKDim)
                let k = split[1].reshaped(B, S, numKHeads, headKDim)
                v = split[2].reshaped(B, S, numVHeads, headVDim)
                let invScale = pow(Float(headKDim), -0.5)
                qNormed =
                    MLXArray(pow(invScale, 2), dtype: q.dtype)
                    * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
                kNormed =
                    MLXArray(invScale, dtype: k.dtype)
                    * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)
            }

            // Same defense as the conv slot: a mis-restored recurrent state
            // with the wrong rank/head dims crashes inside `gatedDeltaUpdate`
            // subscripts. Expected shape is [B, Hv, Dv, Dk].
            let initialState: MLXArray?
            if let cachedState = cache?[1] {
                if cachedState.ndim == 4, cachedState.dim(0) == B,
                    cachedState.dim(1) == numVHeads, cachedState.dim(2) == headVDim,
                    cachedState.dim(3) == headKDim
                {
                    initialState = cachedState
                } else {
                    Self.warnDiscardedCacheState(slot: "recurrent", shape: cachedState.shape)
                    initialState = nil
                }
            } else {
                initialState = nil
            }
            var state = initialState

            var out: MLXArray
            (out, state) = gatedDeltaUpdate(
                q: qNormed,
                k: kNormed,
                v: v,
                a: a,
                b: b,
                aLog: aLog,
                dtBias: dtBias,
                state: state,
                mask: mask,
                roundStateEachStep: recordPrefixCommitStates
                    && NativeMTPVerifierStatePolicy.shouldRoundGDNStateEachVerifierStep
            )
            let finalState = state!

            if let cache {
                if stageVerify {
                    // The first staged verify must run eagerly to allocate
                    // the slots — assigning a trace tracer as a persistent
                    // slot would pin it into every later replay.
                    if CompiledDecodeTrace.isActive, !cache.verifyStagingReady {
                        fatalError(
                            "[Qwen35] staged verify traced before an eager "
                                + "warm-up allocated the staging slots")
                    }
                    cache.stageVerifySlot(0, qNormed)
                    cache.stageVerifySlot(1, kNormed)
                    cache.stageVerifySlot(2, v)
                    cache.stageVerifySlot(3, a)
                    cache.stageVerifySlot(4, b)
                    cache.stageVerifySlot(5, convInput)
                    cache.stageVerifySlot(6, finalState)
                    // cache[0]/cache[1]/offset untouched — committed by
                    // commitVerifyStaged after acceptance.
                } else {
                    if recordPrefixCommitStates, S > 1,
                        NativeMTPVerifierStatePolicy.shouldRecordAcceptedPrefixStates
                    {
                        self.recordPrefixCommitStates(
                            cache: cache,
                            convInput: convInput,
                            q: qNormed,
                            k: kNormed,
                            v: v,
                            a: a,
                            b: b,
                            initialState: initialState,
                            mask: mask,
                            baseOffset: cache.offset)
                    }
                    // DFlash 2 lazy rollback: references only, replayed once
                    // on rejection. See the LLM twin for the measured cost of
                    // the per-prefix recording this replaces.
                    if S > 1, mask == nil,
                        NativeMTPVerifierStatePolicy.shouldStashVerifyInputs
                    {
                        cache.verifyInputStash = MambaCache.VerifyInputStash(
                            arrays: [qNormed, kNormed, v, a, b, convInput],
                            baseOffset: cache.offset,
                            initialState: initialState.map { $0 * 1 },
                            initialConvState: nil)
                    }
                    cache[1] = finalState
                    cache.offset += S
                }
            }

            if let tail = compiledDecodeTail(out, gate: z) { return tail }
            out = norm(out, gate: z)
            return outProj(out.reshaped(B, S, -1))
        }

        /// Commit for the STAGED (compile-compatible) verify: the forward
        /// left cache[0]/cache[1]/offset untouched and wrote everything
        /// into the staging slots. Runs on EVERY staged cycle — a full
        /// accept just adopts the staged final state; a partial accept
        /// replays the accepted rows from the still-intact pre-verify
        /// state in cache[1].
        func commitVerifyStaged(
            cache: MambaCache, acceptedInputs: Int, blockLength: Int
        ) -> Bool {
            guard cache.verifyStagingReady else { return false }
            let n = acceptedInputs
            guard n > 0, n <= blockLength else { return false }
            let slots = cache.verifyStagingSlots
            if n == blockLength {
                cache[1] = slots[6]!
                if convKernelSize > 1 { cache[0] = slots[7]! }
            } else {
                let q = slots[0]!
                let k = slots[1]!
                let v = slots[2]!
                let a = slots[3]!
                let b = slots[4]!
                let convInput = slots[5]!
                // Pre-verify state: still exactly what the verify forward
                // started from, because the staged forward never wrote it.
                let initialState: MLXArray? = cache[1]
                let (_, prefixState) = gatedDeltaUpdate(
                    q: q[0..., ..<n, 0..., 0...],
                    k: k[0..., ..<n, 0..., 0...],
                    v: v[0..., ..<n, 0..., 0...],
                    a: a[0..., ..<n, 0...],
                    b: b[0..., ..<n, 0...],
                    aLog: aLog,
                    dtBias: dtBias,
                    state: initialState,
                    mask: nil)
                cache[1] = prefixState
                if convKernelSize > 1 {
                    let convEnd = convInput.dim(1) - blockLength + n
                    let convStart = max(0, convEnd - max(0, convKernelSize - 1))
                    cache[0] = convInput[0..., convStart ..< convEnd, 0...]
                }
            }
            cache.offset += n
            return true
        }

        /// One-shot lazy rollback — see the LLM twin for rationale.
        func commitVerifyStash(cache: MambaCache, acceptedInputs: Int) -> Bool {
            guard let stash = cache.verifyInputStash, stash.arrays.count == 6 else { return false }
            defer { cache.clearVerifyInputStash() }
            let q = stash.arrays[0]
            let k = stash.arrays[1]
            let v = stash.arrays[2]
            let a = stash.arrays[3]
            let b = stash.arrays[4]
            let convInput = stash.arrays[5]
            let blockLength = q.dim(1)
            guard acceptedInputs > 0, acceptedInputs <= blockLength else { return false }
            if acceptedInputs == blockLength { return true }

            let n = acceptedInputs
            let audit = ProcessInfo.processInfo.environment["VMLX_DFLASH2_ROLLBACK_AUDIT"] == "1"
            let (_, prefixState) = gatedDeltaUpdate(
                q: q[0..., ..<n, 0..., 0...],
                k: k[0..., ..<n, 0..., 0...],
                v: v[0..., ..<n, 0..., 0...],
                a: a[0..., ..<n, 0...],
                b: b[0..., ..<n, 0...],
                aLog: aLog,
                dtBias: dtBias,
                state: stash.initialState,
                mask: nil)
            let convEnd = convInput.dim(1) - blockLength + n
            let convStart = max(0, convEnd - max(0, convKernelSize - 1))
            if audit {
                var chained: MLXArray? = stash.initialState
                for step in 0 ..< n {
                    let r = step ..< (step + 1)
                    let (_, s) = gatedDeltaUpdate(
                        q: q[0..., r, 0..., 0...], k: k[0..., r, 0..., 0...],
                        v: v[0..., r, 0..., 0...], a: a[0..., r, 0...], b: b[0..., r, 0...],
                        aLog: aLog, dtBias: dtBias, state: chained, mask: nil)
                    chained = s
                }
                let diff = abs(prefixState - chained!).max().item(Float.self)
                let scale = abs(chained!).max().item(Float.self)
                FileHandle.standardError.write(
                    Data(
                        String(
                            format:
                                "[rollback-audit] n=%d replay-vs-chained maxdiff=%.3e scale=%.3e\n",
                            n, diff, scale
                        ).utf8))
            }
            cache[1] = prefixState
            cache[0] = convInput[0..., convStart ..< convEnd, 0...]
            cache.offset = stash.baseOffset + n
            return true
        }

        private func recordPrefixCommitStates(
            cache: MambaCache,
            convInput: MLXArray,
            q: MLXArray,
            k: MLXArray,
            v: MLXArray,
            a: MLXArray,
            b: MLXArray,
            initialState: MLXArray?,
            mask: MLXArray?,
            baseOffset: Int
        ) {
            let sequenceLength = q.dim(1)
            guard sequenceLength > 1 else { return }

            let replayStart = Date.timeIntervalSinceReferenceDate
            var recurrentState = initialState
            for prefixLength in 1 ..< sequenceLength {
                let tokenRange = (prefixLength - 1) ..< prefixLength
                let (_, prefixState) = gatedDeltaUpdate(
                    q: q[0..., tokenRange, 0..., 0...],
                    k: k[0..., tokenRange, 0..., 0...],
                    v: v[0..., tokenRange, 0..., 0...],
                    a: a[0..., tokenRange, 0...],
                    b: b[0..., tokenRange, 0...],
                    aLog: aLog,
                    dtBias: dtBias,
                    state: recurrentState,
                    mask: stepMask(mask, index: prefixLength - 1),
                    roundStateEachStep: NativeMTPVerifierStatePolicy
                        .shouldRoundGDNStateEachVerifierStep)
                recurrentState = prefixState

                let convEnd = convInput.dim(1) - sequenceLength + prefixLength
                let convStart = max(0, convEnd - max(0, convKernelSize - 1))
                let convState = convInput[0..., convStart ..< convEnd, 0...]

                cache.recordPrefixCommitState(
                    length: prefixLength,
                    arrays: [convState, prefixState],
                    offset: baseOffset + prefixLength)
            }
            NativeMTPGDNReplayDiagnostics.recordPrefixReplay(
                prefixStates: sequenceLength - 1,
                seconds: Date.timeIntervalSinceReferenceDate - replayStart)
        }

        private func stepMask(_ mask: MLXArray?, index: Int) -> MLXArray? {
            guard let mask else { return nil }
            let range = index ..< (index + 1)
            if mask.ndim == 1 {
                return mask[range]
            }
            if mask.ndim == 2 {
                return mask[0..., range]
            }
            return mask[0..., range, 0...]
        }
    }

    final class SparseMoeBlock: Module, UnaryLayer {
        let normTopkProb: Bool
        let numExperts: Int
        let topK: Int
        let compileDecodeRegions: Bool

        @ModuleInfo(key: "gate") var gate: Linear
        @ModuleInfo(key: "switch_mlp") var switchMLP: Module

        @ModuleInfo(key: "shared_expert") var sharedExpert: MLP
        @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

        init(
            _ args: Qwen35Configuration.TextConfiguration,
            layerIdx: Int = -1,
            allowFusedGateUpCache: Bool = true,
            compileDecodeRegions: Bool = false
        ) {
            self.normTopkProb = args.normTopkProb
            self.numExperts = args.numExperts
            self.topK = args.numExpertsPerTok
            self.compileDecodeRegions = compileDecodeRegions

            _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
            let weightFormat = args.weightFormat.lowercased()
            let usesTurboQuant =
                weightFormat == "mxtq"
                || weightFormat.hasPrefix("jangtq")
                || weightFormat == "turboquant"
            if usesTurboQuant {
                if JANGTQStreamingExperts.isEnabled && layerIdx >= 0 {
                    _switchMLP.wrappedValue = StreamingTurboQuantSwitchGLU(
                        inputDims: args.hiddenSize,
                        hiddenDims: args.moeIntermediateSize,
                        numExperts: args.numExperts,
                        gateUpBits: args.mxtqGateUpBits,
                        downBits: args.mxtqDownBits,
                        seed: args.mxtqSeed,
                        layerIdx: layerIdx)
                } else {
                    _switchMLP.wrappedValue = TurboQuantSwitchGLU(
                        inputDims: args.hiddenSize,
                        hiddenDims: args.moeIntermediateSize,
                        numExperts: args.numExperts,
                        gateUpBits: args.mxtqGateUpBits,
                        downBits: args.mxtqDownBits,
                        seed: args.mxtqSeed)
                }
            } else {
                _switchMLP.wrappedValue = SwitchGLU(
                    inputDims: args.hiddenSize,
                    hiddenDims: args.moeIntermediateSize,
                    numExperts: args.numExperts,
                    allowFusedGateUpCache: allowFusedGateUpCache,
                    compileSeparatedDecode: compileDecodeRegions
                )
            }

            _sharedExpert.wrappedValue = MLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.sharedExpertIntermediateSize
            )
            _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
            super.init()
        }

        private func compiledRouter(_ x: MLXArray) -> (MLXArray, MLXArray)? {
            guard compileDecodeRegions else { return nil }
            if let quantized = gate as? QuantizedLinear,
                let biases = quantized.biases
            {
                return Qwen4ExpCompiledMoE.router(
                    x, weight: quantized.weight, scales: quantized.scales,
                    biases: biases, groupSize: quantized.groupSize,
                    bits: quantized.bits, mode: quantized.mode,
                    topK: topK, normTopK: normTopkProb)
            }
            guard gate.bias == nil else { return nil }
            return Qwen4ExpCompiledMoE.denseRouter(
                x, weight: gate.weight, topK: topK, normTopK: normTopkProb)
        }

        private func compiledSharedExpert(_ x: MLXArray) -> MLXArray? {
            guard compileDecodeRegions,
                let gate = sharedExpert.gateProj as? QuantizedLinear,
                let up = sharedExpert.upProj as? QuantizedLinear,
                let down = sharedExpert.downProj as? QuantizedLinear,
                let gateBiases = gate.biases, let upBiases = up.biases,
                let downBiases = down.biases,
                sharedExpertGate.bias == nil,
                [up, down].allSatisfy({
                    $0.groupSize == gate.groupSize && $0.bits == gate.bits
                        && $0.mode == gate.mode
                })
            else { return nil }
            return Qwen4ExpCompiledMoE.sharedExpert(
                x,
                gateWeight: gate.weight, gateScales: gate.scales,
                gateBiases: gateBiases,
                upWeight: up.weight, upScales: up.scales, upBiases: upBiases,
                downWeight: down.weight, downScales: down.scales,
                downBiases: downBiases,
                sharedGateWeight: sharedExpertGate.weight,
                groupSize: gate.groupSize, bits: gate.bits, mode: gate.mode)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let inds: MLXArray
            let scores: MLXArray
            if let compiled = compiledRouter(x) {
                (inds, scores) = compiled
            } else {
                var gates = gate(x)
                gates = MLX.softmax(gates, axis: -1, precise: true)
                let kth = gates.dim(-1) - topK
                inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, kth...]
                var selected = MLX.takeAlong(gates, inds, axis: -1)
                if normTopkProb {
                    selected = selected / selected.sum(axis: -1, keepDims: true)
                }
                scores = selected
            }

            let routed: MLXArray?
            let eagerCombined: MLXArray?
            if let streaming = switchMLP as? StreamingTurboQuantSwitchGLU {
                routed = nil
                eagerCombined = streaming.reduced(x, indices: inds, scores: scores)
            } else if let turbo = switchMLP as? TurboQuantSwitchGLU {
                let y = turbo(x, inds)
                routed = nil
                eagerCombined = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
            } else if let affine = switchMLP as? SwitchGLU {
                if let combined = affine.qwen4ExpReduced(
                    x, indices: inds, scores: scores)
                {
                    routed = nil
                    eagerCombined = combined
                } else {
                    routed = affine(x, inds)
                    eagerCombined = nil
                }
            } else {
                fatalError("Unsupported Qwen35 VLM MoE switch_mlp: \(type(of: switchMLP))")
            }

            if let routed, let shared = compiledSharedExpert(x),
                let result = Qwen4ExpCompiledMoE.post(
                    routed: routed, scores: scores, shared: shared)
            {
                return result
            }
            if let eagerCombined, let shared = compiledSharedExpert(x) {
                return eagerCombined + shared
            }
            let combined =
                eagerCombined
                ?? (routed! * scores[.ellipsis, .newAxis]).sum(axis: -2)
            let sharedY = sharedExpert(x)
            let gatedSharedY = compiledSigmoidGate(sharedExpertGate(x), sharedY)

            return combined + gatedSharedY
        }
    }

    final class DecoderLayer: Module {
        let isLinear: Bool

        @ModuleInfo(key: "self_attn") var selfAttn: Attention?
        @ModuleInfo(key: "linear_attn") var linearAttn: GatedDeltaNet?

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        @ModuleInfo(key: "mlp") var mlp: Module

        init(_ args: Qwen35Configuration.TextConfiguration, layerIdx: Int) {
            self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0
            let compiledDecodeRegions = shouldCompileDecodeRegions(args)

            if isLinear {
                _linearAttn.wrappedValue = GatedDeltaNet(
                    args,
                    fuseDecodeInputProjections: compiledDecodeRegions)
            } else {
                _selfAttn.wrappedValue = Attention(args)
            }

            if args.numExperts > 0 {
                _mlp.wrappedValue = SparseMoeBlock(
                    args,
                    layerIdx: layerIdx,
                    compileDecodeRegions: compiledDecodeRegions)
            } else {
                _mlp.wrappedValue = MLP(
                    dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            }

            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)

            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionMask: MLXArray?,
            ssmMask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?,
            recordPrefixCommitStates: Bool = false
        ) -> MLXArray {
            let r: MLXArray
            if NativeMTPPhaseDiagnostics.enabled {
                let phase = isLinear ? "vlm_gdn" : "vlm_attention"
                let start = Date.timeIntervalSinceReferenceDate
                if isLinear {
                    r = linearAttn!(
                        inputLayerNorm(x),
                        mask: ssmMask,
                        cache: cache as? MambaCache,
                        recordPrefixCommitStates: recordPrefixCommitStates)
                } else {
                    r = selfAttn!(
                        inputLayerNorm(x), mask: attentionMask, cache: cache,
                        positionIds: positionIds)
                }
                MLX.eval(r)
                NativeMTPPhaseDiagnostics.record(
                    phase,
                    seconds: Date.timeIntervalSinceReferenceDate - start)
            } else if isLinear {
                r = linearAttn!(
                    inputLayerNorm(x),
                    mask: ssmMask,
                    cache: cache as? MambaCache,
                    recordPrefixCommitStates: recordPrefixCommitStates)
            } else {
                r = selfAttn!(
                    inputLayerNorm(x), mask: attentionMask, cache: cache, positionIds: positionIds)
            }

            let h = x + r
            if NativeMTPPhaseDiagnostics.enabled {
                let start = Date.timeIntervalSinceReferenceDate
                let out = (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
                MLX.eval(out)
                NativeMTPPhaseDiagnostics.record(
                    "vlm_mlp",
                    seconds: Date.timeIntervalSinceReferenceDate - start)
                return h + out
            }
            return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
        }
    }

    final class MTPDecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var selfAttn: Attention
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
        @ModuleInfo(key: "mlp") var mlp: Module

        init(_ args: Qwen35Configuration.TextConfiguration) {
            _selfAttn.wrappedValue = Attention(args)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            if args.numExperts > 0 {
                _mlp.wrappedValue = SparseMoeBlock(args)
            } else {
                _mlp.wrappedValue = MLP(
                    dimensions: args.hiddenSize,
                    hiddenDimensions: args.intermediateSize)
            }
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionMask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let h =
                x
                + selfAttn(
                    inputLayerNorm(x),
                    mask: attentionMask,
                    cache: cache,
                    positionIds: positionIds)
            return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
        }
    }

    final class MTPModule: Module {
        @ModuleInfo(key: "pre_fc_norm_hidden") var preFCNormHidden: RMSNorm
        @ModuleInfo(key: "pre_fc_norm_embedding") var preFCNormEmbedding: RMSNorm
        @ModuleInfo(key: "fc") var fc: Linear
        @ModuleInfo(key: "layers") var layers: [MTPDecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm

        init(_ args: Qwen35Configuration.TextConfiguration) {
            _preFCNormHidden.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            _preFCNormEmbedding.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
            _layers.wrappedValue = (0 ..< max(0, args.mtpNumHiddenLayers)).map { _ in
                MTPDecoderLayer(args)
            }
            _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
            super.init()
        }

        func preNormHidden(
            hiddenStates: MLXArray,
            nextTokenIds: MLXArray,
            embedTokens: Embedding,
            cache: [KVCache]?
        ) -> MLXArray {
            let embeds = embedTokens(nextTokenIds)
            let fusedInput = concatenated(
                [
                    preFCNormEmbedding(embeds),
                    preFCNormHidden(hiddenStates),
                ], axis: -1)
            var hiddenStates = fc(fusedInput)

            var cacheArray = cache
            if cacheArray == nil {
                cacheArray = (0 ..< layers.count).map { _ in KVCacheSimple() as KVCache }
            }
            let maskMode = createAttentionMask(
                h: hiddenStates, cache: cacheArray?.first, returnArray: true)
            let attentionMask: MLXArray?
            if case .array(let mask) = maskMode {
                attentionMask = mask
            } else {
                attentionMask = nil
            }
            let positionIds = textPositionIds(for: hiddenStates, cache: cacheArray?.first)
            for (index, layer) in layers.enumerated() {
                hiddenStates = layer(
                    hiddenStates,
                    attentionMask: attentionMask,
                    cache: cacheArray?[index],
                    positionIds: positionIds)
            }
            return hiddenStates
        }

        func callAsFunction(
            hiddenStates: MLXArray,
            nextTokenIds: MLXArray,
            embedTokens: Embedding,
            cache: [KVCache]?
        ) -> MLXArray {
            norm(
                preNormHidden(
                    hiddenStates: hiddenStates,
                    nextTokenIds: nextTokenIds,
                    embedTokens: embedTokens,
                    cache: cache))
        }

        func makeCache() -> [KVCache] {
            (0 ..< layers.count).map { _ in KVCacheSimple() as KVCache }
        }

        private func textPositionIds(for hiddenStates: MLXArray, cache: KVCache?) -> MLXArray {
            let batchSize = hiddenStates.dim(0)
            let seqLength = hiddenStates.dim(1)
            let offset = cache?.offset ?? 0
            var base = MLXArray(stride(from: offset, to: offset + seqLength, by: 1))
                .asType(.int32)
            base = tiled(base[.newAxis, 0...], repetitions: [batchSize, 1])
            return tiled(base[.newAxis, 0..., 0...], repetitions: [3, 1, 1])
        }
    }

    final class Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") fileprivate var layers: [DecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm

        let ssmIdx: Int
        let faIdx: Int

        init(_ args: Qwen35Configuration.TextConfiguration) {
            precondition(args.vocabularySize > 0)
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
            _layers.wrappedValue = (0 ..< args.hiddenLayers).map {
                DecoderLayer(args, layerIdx: $0)
            }
            _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

            self.ssmIdx = 0
            self.faIdx = args.fullAttentionInterval - 1
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            positionIds: MLXArray? = nil
        ) -> MLXArray {
            let (h, _) = callAsFunctionCapturing(
                inputs, inputsEmbeds: inputsEmbeds, cache: cache,
                positionIds: positionIds, captureLayerIDs: [])
            return h
        }

        func callAsFunctionPreNorm(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            positionIds: MLXArray? = nil,
            recordPrefixCommitStates: Bool = false
        ) -> MLXArray {
            var hiddenStates: MLXArray
            if let inputsEmbeds {
                hiddenStates = inputsEmbeds
            } else {
                hiddenStates = embedTokens(inputs)
            }

            var cacheArray = cache
            if cacheArray == nil {
                cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
            }

            let faMaskMode = createAttentionMask(
                h: hiddenStates, cache: cacheArray?[faIdx], returnArray: true)
            let faMask: MLXArray?
            if case .array(let arrayMask) = faMaskMode {
                faMask = arrayMask
            } else {
                faMask = nil
            }
            let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

            for (index, layer) in layers.enumerated() {
                let layerSSMMask = layer.isLinear ? ssmMask : nil
                hiddenStates = layer(
                    hiddenStates,
                    attentionMask: faMask,
                    ssmMask: layerSSMMask,
                    cache: cacheArray?[index],
                    positionIds: positionIds,
                    recordPrefixCommitStates: recordPrefixCommitStates)
            }

            return hiddenStates
        }

        /// Forward with optional per-block hidden-state capture. Mirrors
        /// the Qwen35 LLM-path conformance — empty `captureLayerIDs`
        /// yields byte-identical logits to the plain forward. Required
        /// for DFlash / DDTree speculative decoding through the VLM
        /// wrapper (iter 17).
        func callAsFunctionCapturing(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            positionIds: MLXArray? = nil,
            captureLayerIDs: Set<Int>,
            recordPrefixCommitStates: Bool = false
        ) -> (MLXArray, [Int: MLXArray]) {
            var hiddenStates: MLXArray
            if let inputsEmbeds {
                hiddenStates = inputsEmbeds
            } else {
                hiddenStates = embedTokens(inputs)
            }

            var cacheArray = cache
            if cacheArray == nil {
                cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
            }

            let faMaskMode = createAttentionMask(
                h: hiddenStates, cache: cacheArray?[faIdx], returnArray: true)
            let faMask: MLXArray?
            if case .array(let arrayMask) = faMaskMode {
                faMask = arrayMask
            } else {
                faMask = nil
            }
            let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

            var captured: [Int: MLXArray] = [:]
            captured.reserveCapacity(captureLayerIDs.count)

            for (index, layer) in layers.enumerated() {
                let layerSSMMask = layer.isLinear ? ssmMask : nil
                hiddenStates = layer(
                    hiddenStates,
                    attentionMask: faMask,
                    ssmMask: layerSSMMask,
                    cache: cacheArray?[index],
                    positionIds: positionIds,
                    recordPrefixCommitStates: recordPrefixCommitStates
                )
                if captureLayerIDs.contains(index) {
                    captured[index] = hiddenStates
                }
            }

            return (norm(hiddenStates), captured)
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?
        @ModuleInfo(key: "mtp") var mtp: MTPModule?

        let config: Qwen35Configuration
        let textConfig: Qwen35Configuration.TextConfiguration
        let modelType: String
        let kvHeads: [Int]

        fileprivate var precomputedPositionIds: MLXArray? = nil
        fileprivate var ropeDeltas: MLXArray? = nil

        init(_ config: Qwen35Configuration) {
            self.config = config
            self.textConfig = config.textConfiguration
            self.modelType = config.textConfiguration.modelType
            self.model = Model(config.textConfiguration)
            self.kvHeads = Array(
                repeating: config.textConfiguration.kvHeads,
                count: config.textConfiguration.hiddenLayers
            )

            if !config.textConfiguration.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.textConfiguration.hiddenSize,
                    config.textConfiguration.vocabularySize,
                    bias: false)
            }
            if config.textConfiguration.mtpNumHiddenLayers > 0 {
                _mtp.wrappedValue = MTPModule(config.textConfiguration)
            }
            super.init()
        }

        func resetPositionState() {
            precomputedPositionIds = nil
            ropeDeltas = nil
        }

        private func resolvedPositionIds(
            inputs: MLXArray,
            cache: [KVCache?]?,
            mask: MLXArray?,
            providedPositionIds: MLXArray?,
            imageGridTHW: [THW]?,
            videoGridTHW: [THW]?,
            resetForMedia: Bool
        ) -> MLXArray? {
            if resetForMedia {
                precomputedPositionIds = nil
                ropeDeltas = nil
            }

            // Compiled decode: the promoted caches keep their offset as an
            // MLXArray. Reading `.offset` (Int) forces `.item()` — illegal
            // inside the compile trace and, worse, would bake the trace-time
            // offset into the graph as a constant for every later token.
            // Build the positions from the graph offset instead.
            var cacheOffset = 0
            var graphOffset: MLXArray? = nil
            if let cache, let faCache = cache[model.faIdx] {
                if let g = graphOffsetArray(for: faCache) {
                    graphOffset = g
                } else {
                    cacheOffset = faCache.offset
                }
            }

            var ropeMask = mask
            if let mask, mask.dim(-1) != inputs.dim(-1) {
                ropeMask = nil
            }

            var positionIds = providedPositionIds
            if positionIds == nil, let graphOffset,
                ropeMask == nil || ropeMask?.ndim == 2
            {
                // Graph-visible twin of the `else` (delta) branch below. A
                // Compilable cache only exists after prefill, so the offset
                // is non-zero and text positions are offset + arange —
                // shifted by ropeDeltas when the prompt carried media.
                let batchSize = inputs.dim(0)
                let seqLength = inputs.dim(1)

                // Compiled single-sequence caches expose `[1]`, while
                // `BatchKVCache` exposes one offset per live sequence as
                // `[B]`. Preserve that batch dimension: reshaping `[B]` to a
                // scalar is only valid for B == 1 and process-fatals as soon
                // as continuous batching decodes two Qwen3.5/Ornith slots.
                var delta = graphOffset.asType(.int32)
                if let ropeDeltas {
                    delta = delta + ropeDeltas.asType(.int32)
                }

                var base = MLXArray(0 ..< seqLength).asType(.int32)
                base = broadcast(base[.newAxis, 0...], to: [batchSize, seqLength])

                if delta.ndim == 0 {
                    delta = broadcast(delta, to: [batchSize])
                } else if delta.dim(0) < batchSize {
                    delta = repeated(delta, count: batchSize, axis: 0)
                } else if delta.dim(0) > batchSize {
                    delta = delta[0 ..< batchSize]
                }

                base = base + delta[0..., .newAxis]
                return broadcast(
                    base[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])
            }
            if positionIds == nil && (ropeMask == nil || ropeMask?.ndim == 2) {
                if (cache != nil && cache?[model.faIdx] != nil && cacheOffset == 0)
                    || ropeDeltas == nil
                    || cache == nil
                {
                    if let precomputedPositionIds {
                        let seqLength = inputs.dim(1)
                        positionIds =
                            precomputedPositionIds[
                                0..., 0..., cacheOffset ..< (cacheOffset + seqLength)]
                    } else {
                        let (computed, deltas) = Qwen3VLLanguage.getRopeIndex(
                            inputIds: inputs,
                            imageGridTHW: imageGridTHW,
                            videoGridTHW: videoGridTHW,
                            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
                            imageTokenId: config.imageTokenId,
                            videoTokenId: config.videoTokenId,
                            visionStartTokenId: config.visionStartTokenId,
                            attentionMask: ropeMask)
                        positionIds = computed
                        precomputedPositionIds = computed
                        ropeDeltas = deltas
                    }
                } else {
                    let batchSize = inputs.dim(0)
                    let seqLength = inputs.dim(1)

                    var delta = MLXArray(cacheOffset).asType(.int32)
                    if let ropeDeltas {
                        delta = delta + ropeDeltas.asType(.int32)
                    }

                    var base = MLXArray(0 ..< seqLength).asType(.int32)
                    base = broadcast(base[.newAxis, 0...], to: [batchSize, seqLength])

                    if delta.ndim == 0 {
                        delta = broadcast(delta, to: [batchSize])
                    } else if delta.dim(0) < batchSize {
                        delta = repeated(delta, count: batchSize, axis: 0)
                    } else if delta.dim(0) > batchSize {
                        delta = delta[0 ..< batchSize]
                    }

                    base = base + delta[0..., .newAxis]
                    positionIds = broadcast(
                        base[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])
                }
            }

            return positionIds
        }

        func callAsFunction(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            mask: MLXArray? = nil,
            positionIds providedPositionIds: MLXArray? = nil,
            pixelValues: MLXArray? = nil,
            imageGridTHW: [THW]? = nil,
            videoGridTHW: [THW]? = nil
        ) -> LMOutput {
            let positionIds = resolvedPositionIds(
                inputs: inputs,
                cache: cache,
                mask: mask,
                providedPositionIds: providedPositionIds,
                imageGridTHW: imageGridTHW,
                videoGridTHW: videoGridTHW,
                resetForMedia: pixelValues != nil)

            var out = model(
                inputs,
                inputsEmbeds: inputsEmbeds,
                cache: cache,
                positionIds: positionIds
            )

            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }

            return LMOutput(logits: out)
        }

        /// Text-only `(MLXArray, [KVCache]?) -> MLXArray` forward used by
        /// SpecDec — bypasses the vision RoPE-index bookkeeping and
        /// the VLM-only `LMOutput` wrapper. No pixelValues / imageGrid.
        /// Drafters feed plain text through this path.
        func textOnlyForward(
            _ inputs: MLXArray,
            cache: [KVCache]?
        ) -> MLXArray {
            let (logits, _) = textOnlyForwardCapturing(
                inputs, cache: cache, captureLayerIDs: [])
            return logits
        }

        /// Same as `textOnlyForward` but records per-block hidden
        /// states at the requested 0-based layer indices. Required for
        /// DFlash / DDTree speculative decoding.
        func textOnlyForwardCapturing(
            _ inputs: MLXArray,
            cache: [KVCache]?,
            captureLayerIDs: Set<Int>,
            recordPrefixCommitStates: Bool = false
        ) -> (MLXArray, [Int: MLXArray]) {
            let cacheOpt: [KVCache?]? = cache?.map { $0 as KVCache? }
            // Media-aware RoPE continuation ONLY when this process actually
            // ran a media prefill (`ropeDeltas` exists). Forcing position
            // resolution without that state is wrong twice on a
            // cache-restored text path: `resolvedPositionIds` computes a
            // table from the CHUNK's own rows starting at position 0 —
            // ignoring the restored offset — and the next chunk then
            // slices past the end of that short table
            // (measured: restored offset 1375, chunk 33, slice [1408..<1412]
            // → broadcast abort). `nil` positions make attention derive
            // them from the cache offset, exactly like the plain forward.
            let positionIds: MLXArray? =
                ropeDeltas != nil
                ? resolvedPositionIds(
                    inputs: inputs,
                    cache: cacheOpt,
                    mask: nil,
                    providedPositionIds: nil,
                    imageGridTHW: nil,
                    videoGridTHW: nil,
                    resetForMedia: false)
                : nil
            let (hidden, captured) = model.callAsFunctionCapturing(
                inputs,
                cache: cacheOpt,
                positionIds: positionIds,
                captureLayerIDs: captureLayerIDs,
                recordPrefixCommitStates: recordPrefixCommitStates)
            let logits: MLXArray
            if let lmHead {
                logits = lmHead(hidden)
            } else {
                logits = model.embedTokens.asLinear(hidden)
            }
            return (logits, captured)
        }

        func textOnlyForwardWithHidden(
            _ inputs: MLXArray,
            cache: [KVCache]?,
            recordPrefixCommitStates: Bool = false
        ) -> NativeMTPForwardResult {
            let cacheOpt: [KVCache?]? = cache?.map { $0 as KVCache? }
            // Native MTP enters this path after VLM `prepare` has already run
            // media prefill through `callAsFunction`, which seeds Qwen3VL
            // `ropeDeltas`. Reuse that state for verifier/bridge positions so
            // image/video MRoPE continuation matches normal decode.
            let positionIds = resolvedPositionIds(
                inputs: inputs,
                cache: cacheOpt,
                mask: nil,
                providedPositionIds: nil,
                imageGridTHW: nil,
                videoGridTHW: nil,
                resetForMedia: false)
            let hidden = model.callAsFunctionPreNorm(
                inputs,
                cache: cacheOpt,
                positionIds: positionIds,
                recordPrefixCommitStates: recordPrefixCommitStates)
            let logits: MLXArray
            if NativeMTPPhaseDiagnostics.enabled {
                let start = Date.timeIntervalSinceReferenceDate
                if let lmHead {
                    logits = lmHead(model.norm(hidden))
                } else {
                    logits = model.embedTokens.asLinear(model.norm(hidden))
                }
                MLX.eval(logits)
                NativeMTPPhaseDiagnostics.record(
                    "vlm_lm_head",
                    seconds: Date.timeIntervalSinceReferenceDate - start)
            } else if let lmHead {
                logits = lmHead(model.norm(hidden))
            } else {
                logits = model.embedTokens.asLinear(model.norm(hidden))
            }
            return NativeMTPForwardResult(logits: logits, hiddenStates: hidden)
        }

        func nativeMTPForward(
            hiddenStates: MLXArray,
            nextTokenIds: MLXArray,
            cache: [KVCache]?
        ) -> NativeMTPForwardResult {
            guard let mtp else {
                fatalError("Qwen35 VLM nativeMTPForward called without an MTP module")
            }
            if NativeMTPPhaseDiagnostics.enabled {
                let blockStart = Date.timeIntervalSinceReferenceDate
                let hidden = mtp.preNormHidden(
                    hiddenStates: hiddenStates,
                    nextTokenIds: nextTokenIds,
                    embedTokens: model.embedTokens,
                    cache: cache)
                MLX.eval(hidden)
                NativeMTPPhaseDiagnostics.record(
                    "vlm_mtp_block",
                    seconds: Date.timeIntervalSinceReferenceDate - blockStart)

                let headStart = Date.timeIntervalSinceReferenceDate
                let logits: MLXArray
                if let lmHead {
                    logits = lmHead(mtp.norm(hidden))
                } else {
                    logits = model.embedTokens.asLinear(mtp.norm(hidden))
                }
                MLX.eval(logits)
                NativeMTPPhaseDiagnostics.record(
                    "vlm_mtp_lm_head",
                    seconds: Date.timeIntervalSinceReferenceDate - headStart)
                return NativeMTPForwardResult(logits: logits, hiddenStates: hidden)
            }
            let hidden = mtp.preNormHidden(
                hiddenStates: hiddenStates,
                nextTokenIds: nextTokenIds,
                embedTokens: model.embedTokens,
                cache: cache)
            let logits: MLXArray
            if let lmHead {
                logits = lmHead(mtp.norm(hidden))
            } else {
                logits = model.embedTokens.asLinear(mtp.norm(hidden))
            }
            return NativeMTPForwardResult(logits: logits, hiddenStates: hidden)
        }

        func makeCache(maxKVSize: Int?) -> [KVCache] {
            model.layers.map { layer in
                if layer.isLinear {
                    return MambaCache()
                }
                if let maxKVSize {
                    return RotatingKVCache(maxSize: maxKVSize, keep: 4)
                }
                return KVCacheSimple()
            }
        }
    }
}

// MARK: - Model

public class Qwen35: Module, VLMModel, HiddenStateCaptureModel, TokenEmbedderModel, NativeMTPModel,
    DFlash2StagedVerifyRollbackModel
{
    @ModuleInfo(key: "vision_tower") private var visionModel: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "language_model") fileprivate var languageModel: Qwen35Language.LanguageModel

    public let config: Qwen35Configuration

    public init(_ config: Qwen35Configuration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = Qwen35Language.LanguageModel(config)
        super.init()
    }

    public var vocabularySize: Int { config.vocabSize }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.makeCache(maxKVSize: parameters?.maxKVSize)
    }

    /// Scatters vision features onto their placeholder positions.
    ///
    /// Image and video features are scattered SEPARATELY, each onto its own
    /// placeholder kind. Pooling `imageMask .|| videoMask` into one scatter is
    /// only correct while the feature rows happen to appear in the same order
    /// as the placeholders, and they do not: `prepare` always concatenates
    /// image pixels before video pixels, while the placeholders appear in
    /// conversation order. A chat whose earlier turn carried a video and whose
    /// current turn carries an image therefore has video pads FIRST and image
    /// pads second, so a pooled scatter lays the image's rows onto the video's
    /// pads and vice versa — the two blocks swap.
    ///
    /// That failed silently rather than loudly: the total element count still
    /// matches, so the size guard passes and the model simply answers about
    /// the wrong medium. Measured on Qwen3.8 27B — a red "3" image sent after
    /// a video turn was described as the video's last frame, twice, with and
    /// without tools in the turn.
    ///
    /// Confirmed by an A/B on the live app, same model, byte-identical files,
    /// same prompts: turn 1 a 4-frame video (7 blue, 3 red, 5 green, 9 gold),
    /// turn 2 a purple "4" — a digit/colour pair in no frame, so a leak is
    /// unambiguous. Before this change the answer was "the digit 9 … on a
    /// solid orange/amber (golden) background" (the video's LAST frame);
    /// after, "the digit 4 on a blue (indigo) background".
    ///
    /// Note the bug needs media split ACROSS turns. Within one message
    /// `Qwen3VLMessageGenerator` emits image content before video content, so
    /// pads and rows agree and a single mixed turn cannot expose it — a
    /// combined-attachment turn passes either way and proves nothing.
    ///
    /// Splitting by kind is correct for any interleaving of the two kinds:
    /// `LMInput` carries one image batch and one video batch, so within a kind
    /// the rows are already in placeholder order.
    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray,
        imageRowCount: Int,
        inputEmbeds: MLXArray,
        inputIds: MLXArray,
        imageTokenIndex: Int,
        videoTokenIndex: Int
    ) throws -> (MLXArray, MLXArray) {
        let imageMask = (inputIds .== MLXArray(imageTokenIndex))
        let videoMask = (inputIds .== MLXArray(videoTokenIndex))
        let specialMask = imageMask .|| videoMask

        let nImageTokens = specialMask.sum().item(Int.self)
        let nFeatureRows = imageFeatures.dim(0)

        // Total-count check stays: a genuine mismatch must still fail loudly.
        guard nImageTokens == nFeatureRows else {
            throw Qwen35VLError.featureTokenMismatch(expected: nImageTokens, actual: nFeatureRows)
        }

        let originalShape = inputEmbeds.shape
        var result = inputEmbeds.flattened()
        let width = inputEmbeds.dim(-1)

        // Rows [0, imageRowCount) belong to the image batch, the remainder to
        // the video batch — the order `prepare` concatenated them in.
        func scatter(mask: MLXArray, rows: MLXArray) throws {
            let positions = nonZero(mask.flattened().asType(.bool))
            guard !positions.isEmpty else { return }
            guard positions.count == rows.dim(0) else {
                throw Qwen35VLError.featureTokenMismatch(
                    expected: positions.count, actual: rows.dim(0))
            }
            var flatIndices: [UInt32] = []
            flatIndices.reserveCapacity(positions.count * width)
            for position in positions {
                let base = position * width
                for offset in 0 ..< width { flatIndices.append(UInt32(base + offset)) }
            }
            result[MLXArray(flatIndices)] = rows.flattened()
        }

        if imageRowCount > 0 {
            try scatter(mask: imageMask, rows: imageFeatures[0 ..< imageRowCount])
        }
        if nFeatureRows > imageRowCount {
            try scatter(mask: videoMask, rows: imageFeatures[imageRowCount ..< nFeatureRows])
        }

        result = result.reshaped(originalShape)
        let visualMask = specialMask.asType(.bool)
        return (result, visualMask)
    }

    /// Number of merged vision rows a frame list contributes, matching the
    /// vision tower's output rows: each `THW` frame yields
    /// `t*h*w / spatialMergeSize^2` rows.
    private func mergedRowCount(for frames: [THW]?) -> Int {
        guard let frames else { return 0 }
        let merge = config.visionConfiguration.spatialMergeSize
        let divisor = max(1, merge * merge)
        return frames.reduce(0) { $0 + $1.product / divisor }
    }

    private func nonZero(_ mask: MLXArray) -> [Int] {
        let values = mask.asArray(Bool.self)
        var indices: [Int] = []
        indices.reserveCapacity(values.count)
        for (idx, value) in values.enumerated() where value {
            indices.append(idx)
        }
        return indices
    }

    private func combinedFrames(imageFrames: [THW]?, videoFrames: [THW]?) -> [THW] {
        var frames: [THW] = []
        if let imageFrames { frames.append(contentsOf: imageFrames) }
        if let videoFrames { frames.append(contentsOf: videoFrames) }
        return frames
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        windowSize: Int?
    ) throws -> PrepareResult {
        let inputIds = input.text.tokens

        var pixelValues: MLXArray?
        var imageFrames: [THW]?
        var videoFrames: [THW]?

        let visionDType = visionModel.patchEmbed.proj.weight.dtype
        var pixelParts: [MLXArray] = []

        if let image = input.image {
            pixelParts.append(image.pixels.asType(visionDType))
            imageFrames = image.frames
        }
        if let video = input.video {
            pixelParts.append(video.pixels.asType(visionDType))
            videoFrames = video.frames
        }
        if !pixelParts.isEmpty {
            pixelValues = concatenated(pixelParts)
        }

        var inputEmbeddings: MLXArray?

        if let pixelValues,
            let frames = combinedFrames(imageFrames: imageFrames, videoFrames: videoFrames)
                .nilIfEmpty
        {
            let textEmbeds = languageModel.model.embedTokens(inputIds)
            let (visionHidden, _) = visionModel(pixelValues, gridTHW: frames)
            let visionFeatures = visionHidden.asType(textEmbeds.dtype)

            let (mergedEmbeds, _) = try mergeInputIdsWithImageFeatures(
                imageFeatures: visionFeatures,
                imageRowCount: mergedRowCount(for: imageFrames),
                inputEmbeds: textEmbeds,
                inputIds: inputIds,
                imageTokenIndex: config.imageTokenIndex,
                videoTokenIndex: config.videoTokenIndex
            )
            inputEmbeddings = mergedEmbeds
        } else {
            languageModel.resetPositionState()
        }

        let typedCache = castCache(cache)

        // Chunked text-only prefill so the UI prefill counter advances instead
        // of freezing at "0/N". The single-shot forward below emits no
        // `PrefillProgress` frames, so a long hybrid (Ornith / qwen3_5) prompt
        // showed a frozen counter until first token. Only the pure-text,
        // causal-mask path is chunked: image/video prefill and custom masks keep
        // the single-shot path because mrope position ids are derived from the
        // full image grid. Chunking is numerically identical to single-shot —
        // the hybrid cache (GatedDeltaNet conv+recurrent state + KV) carries
        // state across forwards, and the language model derives position ids and
        // the causal mask from `cache.offset` (the same invariant that makes
        // token-by-token decode correct). Mirrors Gemma4's VLM prefill.
        // No `input.text.mask == nil` guard: each chunk passes `mask: nil` and
        // the model rebuilds the causal mask from `cache.offset` (identical to
        // how token-by-token decode runs), so a causal/padding prefill mask is
        // reconstructed correctly per chunk — same as Gemma4's chunked VLM
        // prefill, which also drops the incoming mask.
        let prefillStepSize = windowSize ?? 512
        let promptTokenCount = inputIds.dim(1)
        if inputEmbeddings == nil, pixelValues == nil,
            prefillStepSize > 0, promptTokenCount > prefillStepSize
        {
            var offset = 0
            while offset + prefillStepSize < promptTokenCount {
                // Bound the orphan-producer window on client disconnect —
                // see LLMModel.prepare. Prefill has no other cancellation
                // points and an orphan producer racing a follow-up request
                // on the shared GPU command queue aborts the process.
                try Task.checkCancellation()
                let end = offset + prefillStepSize
                _ = languageModel(
                    inputIds[0..., offset ..< end],
                    inputsEmbeds: nil,
                    cache: typedCache,
                    mask: nil,
                    positionIds: nil,
                    pixelValues: nil,
                    imageGridTHW: nil,
                    videoGridTHW: nil
                )
                MLX.eval(cache)
                PrefillProgressReporter.reportCompletedUnits(end)
                offset = end
                MLX.Memory.clearCache()
            }
            let output = languageModel(
                inputIds[0..., offset...],
                inputsEmbeds: nil,
                cache: typedCache,
                mask: nil,
                positionIds: nil,
                pixelValues: nil,
                imageGridTHW: nil,
                videoGridTHW: nil
            )
            return .logits(output)
        }

        let output = languageModel(
            inputIds,
            inputsEmbeds: inputEmbeddings,
            cache: typedCache,
            mask: input.text.mask,
            positionIds: nil,
            pixelValues: pixelValues,
            imageGridTHW: imageFrames,
            videoGridTHW: videoFrames
        )

        return .logits(output)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let typedCache = castCacheOptional(cache)
        let result = languageModel(
            inputs,
            inputsEmbeds: nil,
            cache: typedCache,
            mask: nil,
            positionIds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil
        )
        return result.logits
    }

    // MARK: - SpecDec conformance (iter 17)

    /// `HiddenStateCaptureModel` conformance — routes through the
    /// nested LanguageModel's text-only capturing forward so DFlash /
    /// DDTree speculative decoding can read per-block hidden states
    /// without going through the vision RoPE-bookkeeping path.
    public func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        captureLayerIDs: Set<Int>
    ) -> (logits: MLXArray, capturedHiddenStates: [Int: MLXArray]) {
        languageModel.textOnlyForwardCapturing(
            inputs, cache: cache, captureLayerIDs: captureLayerIDs)
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        captureLayerIDs: Set<Int>,
        recordPrefixCommitStates: Bool
    ) -> (logits: MLXArray, capturedHiddenStates: [Int: MLXArray]) {
        languageModel.textOnlyForwardCapturing(
            inputs, cache: cache, captureLayerIDs: captureLayerIDs,
            recordPrefixCommitStates: recordPrefixCommitStates)
    }

    /// The GatedDeltaNet layers record their per-step state, so DFlash 2
    /// can roll a rejected block back without replaying the prefix.
    public var supportsCapturingPrefixCommitRecording: Bool { true }

    // MARK: - DFlash2VerifyRollbackModel

    public func commitVerifiedBlock(cache: [KVCache], acceptedInputs: Int) -> Bool {
        for (index, layer) in languageModel.model.layers.enumerated() where layer.isLinear {
            guard index < cache.count, let mamba = cache[index] as? MambaCache,
                let gdn = layer.linearAttn,
                gdn.commitVerifyStash(cache: mamba, acceptedInputs: acceptedInputs)
            else { return false }
        }
        return true
    }

    // MARK: - DFlash2StagedVerifyRollbackModel

    public func commitStagedVerifiedBlock(
        cache: [KVCache], acceptedInputs: Int, blockLength: Int
    ) -> Bool {
        for (index, layer) in languageModel.model.layers.enumerated() where layer.isLinear {
            guard index < cache.count, let mamba = cache[index] as? MambaCache,
                let gdn = layer.linearAttn,
                gdn.commitVerifyStaged(
                    cache: mamba, acceptedInputs: acceptedInputs,
                    blockLength: blockLength)
            else { return false }
        }
        return true
    }

    /// `TokenEmbedderModel` conformance — exposes the target's shared
    /// embedding and LM head for DFlash drafters.
    public func embed(_ tokenIds: MLXArray) -> MLXArray {
        languageModel.model.embedTokens(tokenIds)
    }

    public func projectToLogits(_ hidden: MLXArray) -> MLXArray {
        if let head = languageModel.lmHead {
            return head(hidden)
        }
        return languageModel.model.embedTokens.asLinear(hidden)
    }

    // MARK: - NativeMTPModel

    public var nativeMTPAvailable: Bool { languageModel.mtp != nil }

    public func makeNativeMTPCache() -> [KVCache] {
        languageModel.mtp?.makeCache() ?? []
    }

    public func nativeBackboneForward(
        _ inputs: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPForwardResult {
        languageModel.textOnlyForwardWithHidden(inputs, cache: cache)
    }

    public func nativeBackboneMTPVerifyForward(
        _ inputs: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPForwardResult {
        languageModel.textOnlyForwardWithHidden(
            inputs,
            cache: cache,
            recordPrefixCommitStates: true)
    }

    public func nativeMTPForward(
        hiddenStates: MLXArray,
        nextTokenIds: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPForwardResult {
        languageModel.nativeMTPForward(
            hiddenStates: hiddenStates,
            nextTokenIds: nextTokenIds,
            cache: cache)
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        let isMLXFormat = metadata["format"]?.lowercased() == "mlx"
        return sanitize(
            weights: weights,
            isMLXFormat: isMLXFormat,
            normConvention: Self.normConvention(metadata))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        sanitize(weights: weights, isMLXFormat: false, normConvention: nil)
    }

    private func sanitize(
        weights: [String: MLXArray],
        isMLXFormat: Bool,
        normConvention: String?
    ) -> [String: MLXArray] {
        let loadNativeMTP = config.textConfiguration.mtpNumHiddenLayers > 0
        var weights = loadNativeMTP ? weights : weights.filter { !Self.isMTPWeightKey($0.key) }
        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }

        // Resolve the (1 + weight) RMSNorm shift via the shared resolver: a per-bundle
        // metadata/config declaration wins; otherwise (this arch uses the convention) the
        // order-independent majority vote decides raw (→ shift) vs already-shifted (→ leave it).
        // The same architecture ships both — JangQ stores raw, MXFP4 stores already-shifted — so
        // this MUST be measured per bundle, not declared as always-on.
        let shouldShiftNormWeights = NormConventionResolver.shouldApplyPlusOneShift(
            metadataConvention: normConvention,
            configConvention: config.textConfiguration.normConvention,
            declaredConvention: declaredNormConvention,
            weights: weights,
            probeSuffixes: [".input_layernorm.weight", ".post_attention_layernorm.weight"],
            fallbackWhenNoProbe: { !Self.isMLXFormatLike(weights) })
        let shouldShiftMTPNormWeights =
            loadNativeMTP
            && (shouldShiftNormWeights || Self.mtpNormWeightsNeedShift(weights))

        if config.textConfiguration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        // MLX-native models with no unsanitized conv1d may still need key remapping.
        if isMLXFormat && !hasUnsanitizedConv1d && !shouldShiftNormWeights
            && !shouldShiftMTPNormWeights
        {
            let needsRemap = weights.keys.contains {
                $0.contains("model.language_model") || $0.contains("model.visual")
            }
            if !needsRemap {
                return weights
            }
        }

        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        for (key, originalValue) in weights {
            var key = key
            var value = originalValue

            if key.contains("model") {
                if key.contains("model.language_model") {
                    key = key.replacingOccurrences(
                        of: "model.language_model", with: "language_model.model")
                } else if key.contains("model.visual") {
                    key = key.replacingOccurrences(of: "model.visual", with: "vision_tower")
                }
            } else if key.contains("lm_head") {
                key = key.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            }
            if loadNativeMTP {
                if key.hasPrefix("mtp.") {
                    key = "language_model." + key
                } else if key.hasPrefix("model.mtp_layers.") {
                    key = key.replacingOccurrences(
                        of: "model.mtp_layers.",
                        with: "language_model.mtp.layers.")
                }
            }

            if key.contains("conv1d.weight") && value.dim(-1) != 1 {
                value = value.movedAxis(source: 2, destination: 1)
            }
            let isMTPNorm =
                loadNativeMTP
                && Self.isMTPWeightKey(key)
                && key.hasSuffix(".weight")
                && (key.contains("norm") || key.contains("q_norm") || key.contains("k_norm"))
            let shouldShiftBaseNorm =
                shouldShiftNormWeights
                && normKeys.contains(where: { key.hasSuffix($0) })
            let shouldShiftMTPNorm = isMTPNorm && shouldShiftMTPNormWeights
            if (shouldShiftBaseNorm || shouldShiftMTPNorm) && value.ndim == 1 {
                value = value + MLXArray(1, dtype: value.dtype)
            }

            sanitized[key] = value
        }

        return visionModel.sanitize(weights: sanitized)
    }

    private static func mtpNormWeightsNeedShift(_ weights: [String: MLXArray]) -> Bool {
        let probeSuffixes = [
            "mtp.layers.0.input_layernorm.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.pre_fc_norm_embedding.weight",
        ]
        for suffix in probeSuffixes {
            for (key, value) in weights where value.ndim == 1 {
                guard Self.isMTPWeightKey(key), key.hasSuffix(suffix) else {
                    continue
                }
                return value.asType(.float32).mean().item(Float.self) < 0.5
            }
        }
        return false
    }

    private static func normConvention(_ metadata: [String: String]) -> String? {
        let value = metadata["norm_convention"] ?? metadata["runtime.norm_convention"]
        return value?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    // NB: qwen3.5 deliberately does NOT declare a class-level `norm_convention`. This architecture
    // uses the (1 + weight) convention, but its bundles are stored in BOTH states — JangQ stores the
    // norms raw (needs +1), MXFP4 stores them already-shifted (must not be shifted again) — so no
    // truthful architecture-level claim exists. A per-bundle `config.json` / metadata declaration or,
    // failing that, the order-independent vote decides. Do NOT override `declaredNormConvention`
    // here: an authoritative class declaration would wrongly short-circuit the vote and degrade one
    // of the two storage states. See ``NormConventionResolver``.

    private static func isMTPWeightKey(_ key: String) -> Bool {
        key.hasPrefix("mtp.")
            || key.hasPrefix("model.mtp_layers.")
            || key.contains(".mtp.")
            || key.contains(".mtp_layers.")
    }

    // The order-independent "are these norms already shifted?" fallback now lives in
    // `NormConventionResolver.weightsAppearUnshifted` (MLXLMCommon), invoked via
    // `shouldApplyPlusOneShift` above. Architectures share that one implementation.

    private static func isMLXFormatLike(_ weights: [String: MLXArray]) -> Bool {
        weights.keys.contains { key in
            key.hasPrefix("language_model.") || key.hasPrefix("vision_tower.")
        }
    }
}

extension Array where Element == THW {
    fileprivate var nilIfEmpty: [THW]? { isEmpty ? nil : self }
}

extension Qwen35 {
    fileprivate func castCache(_ cache: [any KVCache]) -> [KVCache]? {
        guard !cache.isEmpty else { return nil }
        return cache.map { $0 }
    }

    fileprivate func castCacheOptional(_ cache: [any KVCache]?) -> [KVCache]? {
        guard let cache else { return nil }
        return castCache(cache)
    }
}
