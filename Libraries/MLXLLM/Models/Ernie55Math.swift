// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX
import MLXNN

/// Numerical primitives for the Baidu ERNIE-5.5-Flash target.
/// The attention constructors keep HCA and MLA projection/rotary layouts separate.
enum Ernie55Math {
    /// Released W4A8 expert contract. Quantize each group before the gate/up
    /// projections and again, after SiTU and its dtype conversion, before down.
    static func roundExpertActivations(_ input: MLXArray) -> MLXArray {
        precondition(input.dim(-1).isMultiple(of: 128))
        #if canImport(Metal)
        if input.size > 0, Device.defaultDevice().deviceType == .gpu {
            return expertActivationKernel(
                [contiguous(input), MLXArray(UInt32(input.size / 128))],
                template: [("Out", input.dtype)], grid: (input.size / 4, 1, 1),
                threadGroup: (256, 1, 1), outputShapes: [input.shape], outputDTypes: [input.dtype])[0]
        }
        #endif
        return roundExpertActivationsGraph(input)
    }

    static func roundExpertActivationsGraph(_ input: MLXArray) -> MLXArray {
        let blocks = input.asType(.float32).reshaped(-1, 128)
        let absMax = maximum(abs(blocks).max(axis: -1, keepDims: true), 1e-10)
        let scale = pow(MLXArray(Float(2)), ceil(log2(absMax * (1.0 / 448.0))))
        let codes = toFP8(clip(blocks / scale, min: -448, max: 448))
        return (fromFP8(codes, dtype: .float32) * scale).reshaped(input.shape).asType(input.dtype)
    }

    #if canImport(Metal)
    private static let expertActivationKernel = MLXFast.metalKernel(
        name: "vmlx_ernie55_e4m3_roundtrip", inputNames: ["src", "groups"], outputNames: ["dst"],
        source: """
            uint group = thread_position_in_grid.x / 32u;
            if (group >= groups) return;
            uint lane = thread_index_in_simdgroup;
            uint base = group * 128u + lane;
            float values[4];
            float maximum = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                values[i] = float(src[base + i * 32u]);
                maximum = metal::max(maximum, metal::abs(values[i]));
            }
            float ratio = metal::max(simd_max(maximum), 1e-10f) * (1.0f / 448.0f);
            int exponent;
            float fraction = metal::frexp(ratio, exponent);
            float scale = metal::ldexp(1.0f, exponent - int(fraction == 0.5f));
            for (uint i = 0; i < 4; ++i) {
                float normalized = metal::clamp(values[i] / scale, -448.0f, 448.0f);
                int magnitudeExponent;
                metal::frexp(metal::abs(normalized), magnitudeExponent);
                int precision = metal::max(magnitudeExponent - 1, -6) - 3;
                if (normalized == 0.0f) precision = -9;
                float nearest = metal::rint(metal::ldexp(normalized, -precision));
                dst[base + i * 32u] = Out(metal::ldexp(nearest, precision) * scale);
            }
            """)
    #endif

    /// SiTU is the same softcapped SwiGLU shape as Inkling. The release rounds
    /// the product back to the gate projection dtype before the next operation.
    static func situ(_ gate: MLXArray, _ up: MLXArray, gateCap: Float, upCap: Float) -> MLXArray {
        if let fused = Ernie55Kernels.fusedSiTU(gate, up, gateCap: gateCap, upCap: upCap) {
            return fused
        }
        let g = gate.asType(.float32)
        let u = up.asType(.float32)
        let gateValue = gateCap * tanh(g / gateCap) * sigmoid(g)
        return (gateValue * (upCap * tanh(u / upCap))).asType(gate.dtype)
    }

    /// ERNIE places RMS epsilon outside the square root in its sublayer mHC.
    /// DeepSeek's mHC pre-projection uses a different denominator.
    static func collapse(
        _ input: MLXArray, projection: MLXArray, scales: MLXArray, bias: MLXArray,
        iterations: Int, mixingEpsilon: Float, normEpsilon: Float
    ) -> (hidden: MLXArray, post: MLXArray, residualMix: MLXArray) {
        let flat = input.reshaped(input.dim(0), input.dim(1), -1).asType(.float32)
        let rms = sqrt((flat * flat).mean(axis: -1, keepDims: true)) + normEpsilon
        let proj = projection.dtype == .float32 ? projection : projection.asType(.float32)
        let mixed = matmul(flat, proj.transposed()) / rms
        let (pre, post, comb) = DeepseekV4Math.hcSplitSinkhorn(
            mixes: mixed, scale: scales, base: bias, hcMult: input.dim(-2),
            iters: iterations, eps: mixingEpsilon)
        return (DeepseekV4Math.hcCollapse(input, pre: pre), post, comb)
    }

    static func expand(_ hidden: MLXArray, residual: MLXArray, post: MLXArray, mix: MLXArray)
        -> MLXArray
    {
        if hidden.dim(0) == 1 && hidden.dim(1) > 0 && Device.defaultDevice().deviceType == .gpu {
            return DeepseekV4Math.hcPostRows(blockOut: hidden, residual: residual, post: post, comb: mix)
        }
        let mixed = DeepseekV4Math.hcExpandResidual(comb: mix, residual: residual)
        return (mixed + post.expandedDimensions(axis: -1)
            * hidden.asType(.float32).expandedDimensions(axis: -2)).asType(residual.dtype)
    }

    static func route(
        _ input: MLXArray, first: MLXArray, second: MLXArray, correction: MLXArray,
        expertScale: MLXArray, count: Int, normalize: Bool
    ) -> (indices: MLXArray, weights: MLXArray) {
        if let fused = Ernie55Kernels.routeFused(
            input, first: first, second: second, correction: correction,
            expertScale: expertScale, count: count, normalize: normalize)
        {
            return fused
        }
        let x = input.asType(.float32)
        let scores = sigmoid(matmul(x, first.asType(.float32).transposed()))
            + sigmoid(matmul(x, second.asType(.float32).transposed()))
        if let fastTopK = Ernie55Kernels.routeTopK(
            scores, correction: correction, expertScale: expertScale, count: count, normalize: normalize)
        {
            return fastTopK
        }
        let indices = argPartition(-(scores + correction), kth: count - 1, axis: -1)[.ellipsis, ..<count]
        let picked = takeAlong(scores, indices, axis: -1)
        let weights = normalize ? picked / maximum(picked.sum(axis: -1, keepDims: true), 1e-20) : picked
        return (indices, weights * take(expertScale, indices))
    }

    static func multiMax(_ input: MLXArray, ranges: MLXArray, slopes: MLXArray) -> MLXArray {
        let logits = input.asType(.float32)
        let low = maximum(ranges[2] - logits, 0)
        let high = maximum(logits - ranges[3], 0)
        return logits + slopes[0] * maximum(ranges[0] - logits, 0)
            + slopes[1] * maximum(logits - ranges[1], 0)
            + slopes[2] * square(low) + slopes[3] * square(high)
    }

    /// One pooled column becomes visible when its entire window is committed.
    /// Local and pooled masks use the same uncompressed token coordinates.
    static func hcaPoolMask(start: Int, length: Int, columns: Int, ratio: Int) -> MLXArray {
        let queries = MLXArray(Int32(start + 1)..<Int32(start + length + 1))
        let ready = MLXArray(Int32(1)..<Int32(columns + 1)) * Int32(ratio)
        return queries.expandedDimensions(axis: 1) .>= ready.expandedDimensions(axis: 0)
    }
}

/// Non-overlapping HCA pooling over FP32 projection and gate buffers. The
/// matching local window and pending FP32 state travel in the typed pool cache.
final class Ernie55HCACompressor: Module {
    @ModuleInfo(key: "wkv") var keyProjection: Linear
    @ModuleInfo(key: "wgate") var gateProjection: Linear
    @ModuleInfo var norm: RMSNorm
    @ParameterInfo var ape: MLXArray

    let ratio: Int
    let headDimension: Int

    init(hiddenSize: Int, headDimension: Int, ratio: Int, epsilon: Float) {
        precondition(ratio > 0)
        self.ratio = ratio
        self.headDimension = headDimension
        _keyProjection.wrappedValue = Linear(hiddenSize, headDimension, bias: false)
        _gateProjection.wrappedValue = Linear(hiddenSize, headDimension, bias: false)
        _norm.wrappedValue = RMSNorm(dimensions: headDimension, eps: epsilon)
        _ape.wrappedValue = MLXArray.zeros([ratio, headDimension])
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray, rope: DeepseekV4RoPE, cache: DeepseekV4Cache?, start: Int
    ) -> MLXArray {
        if let cache, input.dim(1) == 1 {
            let view = cache.deferPoolToken(
                input, branch: .compressor, startPos: start, ratio: ratio, overlap: false
            ) { [weak self] batchInput, baseStart in
                guard let self else {
                    return DeepseekV4PoolView(
                        dense: MLXArray.zeros([batchInput.dim(0), 0, 1], dtype: batchInput.dtype))
                }
                let pooled = self.advance(batchInput, rope: rope, cache: cache, start: baseStart)
                return DeepseekV4PoolView(dense: pooled)
            }
            return view.denseArray ?? view.materialized()
        }
        cache?.flushPoolPending(branch: .compressor)
        let pooled = advance(input, rope: rope, cache: cache, start: start)
        cache?.notePoolResult(DeepseekV4PoolView(dense: pooled), branch: .compressor)
        return pooled
    }

    private func advance(
        _ input: MLXArray, rope: DeepseekV4RoPE, cache: DeepseekV4Cache?, start: Int
    ) -> MLXArray {
        var (keys, gates) = Ernie55Kernels.compressorProject(
            input, keyWeight: keyProjection.weight, gateWeight: gateProjection.weight)
        var base = start
        if let pending = cache?.getBuffers(.compressor), let priorKeys = pending.kv,
            let priorGates = pending.gate
        {
            base -= priorKeys.dim(1)
            keys = concatenated([priorKeys, keys], axis: 1)
            gates = concatenated([priorGates, gates], axis: 1)
        }

        let rows = keys.dim(1) / ratio
        let complete = rows * ratio
        cache?.setBuffers(
            .compressor, kv: complete < keys.dim(1) ? contiguous(keys[0..., complete..., 0...]) : nil,
            gate: complete < gates.dim(1) ? contiguous(gates[0..., complete..., 0...]) : nil)
        if rows == 0 {
            return cache?.getPooled(.compressor)
                ?? MLXArray.zeros([input.dim(0), 0, headDimension], dtype: input.dtype)
        }

        let batch = input.dim(0)
        let pooledKeys = keys[0..., ..<complete, 0...].reshaped(batch, rows, ratio, headDimension)
        let pooledGates = gates[0..., ..<complete, 0...].reshaped(batch, rows, ratio, headDimension)
        let poolWeights = softmax(pooledGates + ape.asType(.float32), axis: -2, precise: true)
        let pooled = norm((pooledKeys * poolWeights).sum(axis: -2))
        let phase = rope.cosSin(base: base, count: rows, stride: ratio)
        let rotated = DeepseekV4Math.applyPartialRoPE(
            pooled.expandedDimensions(axis: 1), cos: phase.cos, sin: phase.sin, ropeDim: rope.dim)
            .squeezed(axis: 1).asType(input.dtype)
        guard let cache else { return rotated }
        _ = cache.appendPooled(.compressor, value: rotated, isPrefill: input.dim(1) > 1)
        return cache.getPooled(.compressor) ?? rotated
    }
}
