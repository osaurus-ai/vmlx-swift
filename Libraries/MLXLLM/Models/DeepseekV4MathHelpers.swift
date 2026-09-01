// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Pure-math building blocks for the DeepSeek-V4 forward pass.
// Each helper is pure (no module state, no cache) so it's trivially
// unit-testable with synthetic tensors.
//
// Reference:
//   - `jang/research/DSV4-RUNTIME-ARCHITECTURE.md` §2 (per-layer forward)
//   - `jang-tools/jang_tools/dsv4_prune/mlx_model.py` —
//       * `_hc_split_sinkhorn_ops` (lines 79-110)
//       * `_apply_partial_rope` (lines 355-362)
//       * `_dsv4_swiglu` (lines 799-814)
//       * `sqrtsoftplus_select` (lines 736-757)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private let _deepseekV4SwiGLUClampedBody:
    @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
{ gate, up, limit in
    let outputType = gate.dtype
    let gateF32 = minimum(gate.asType(.float32), limit)
    let upF32 = minimum(maximum(up.asType(.float32), -limit), limit)
    return (silu(gateF32) * upF32).asType(outputType)
}

private let _deepseekV4SwiGLUUnclampedBody:
    @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
{ gate, up, _ in
    let outputType = gate.dtype
    return (silu(gate.asType(.float32)) * up.asType(.float32)).asType(outputType)
}

private let _deepseekV4ScoredSwiGLUClampedBody:
    @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray =
{ gate, up, scores, limit in
    let outputType = gate.dtype
    let gateF32 = minimum(gate.asType(.float32), limit)
    let upF32 = minimum(maximum(up.asType(.float32), -limit), limit)
    let scoreF32 = scores.asType(.float32)[.ellipsis, .newAxis, .newAxis]
    return (silu(gateF32) * upF32 * scoreF32).asType(outputType)
}

private let _deepseekV4ScoredSwiGLUUnclampedBody:
    @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray =
{ gate, up, scores, _ in
    let outputType = gate.dtype
    let scoreF32 = scores.asType(.float32)[.ellipsis, .newAxis, .newAxis]
    return (silu(gate.asType(.float32)) * up.asType(.float32) * scoreF32)
        .asType(outputType)
}

private let _deepseekV4ScoredSwiGLUClampedArrayBody:
    @Sendable ([MLXArray]) -> [MLXArray] =
{ args in
    [_deepseekV4ScoredSwiGLUClampedBody(args[0], args[1], args[2], args[3])]
}

private let _deepseekV4ScoredSwiGLUUnclampedArrayBody:
    @Sendable ([MLXArray]) -> [MLXArray] =
{ args in
    [_deepseekV4ScoredSwiGLUUnclampedBody(args[0], args[1], args[2], args[3])]
}

// The authoritative affine runtime compiles this exact stateless activation.
// Keep it separate from whole-model compiled decode: it captures no model or
// cache state, and the wrapper below avoids an illegal nested compile trace.
private let _compiledDeepseekV4SwiGLUClamped =
    vmlxTrustedCompile(shapeless: true, _deepseekV4SwiGLUClampedBody)
private let _compiledDeepseekV4SwiGLUUnclamped =
    vmlxTrustedCompile(shapeless: true, _deepseekV4SwiGLUUnclampedBody)
// NOT shapeless: the `[.ellipsis, .newAxis, .newAxis]` score expansion bakes a
// concrete reshape target into the trace, so a shapeless trace recorded at one
// L fatals ("[reshape] cannot reshape…") when a different L arrives. Per-shape
// retrace is cached by signature and decode always re-hits the L==1 trace.
private let _compiledDeepseekV4ScoredSwiGLUClamped =
    vmlxTrustedCompile(_deepseekV4ScoredSwiGLUClampedArrayBody)
private let _compiledDeepseekV4ScoredSwiGLUUnclamped =
    vmlxTrustedCompile(_deepseekV4ScoredSwiGLUUnclampedArrayBody)

typealias DeepseekV4CompiledRegion = @Sendable ([MLXArray]) -> [MLXArray]

/// Resolve a per-instance compiled decode region, building it once.
///
/// The modules holding these caches live on the shared model object, so two
/// concurrent requests reach the same uninitialised slot together. Without the
/// lock both threads write the property at once; a closure is a (function,
/// context) pair, and a torn read pairs one thread's function with the other's
/// context. The lock is held across the build so the region is compiled once —
/// that happens on the first decode step of a layer and never again, while the
/// *call* stays outside the lock so decode is never serialised.
@inline(__always)
func deepseekV4CachedRegion(
    _ lock: NSLock,
    _ storage: inout DeepseekV4CompiledRegion?,
    build: () -> DeepseekV4CompiledRegion
) -> DeepseekV4CompiledRegion {
    lock.lock()
    defer { lock.unlock() }
    if let cached = storage { return cached }
    let built = build()
    storage = built
    return built
}

public enum DeepseekV4Math {

    private static let e4m3KVActivationRoundTripKernel = MLXFast.metalKernel(
        name: "deepseek_v4_e4m3_kv_activation_roundtrip",
        inputNames: ["x"],
        outputNames: ["y"],
        source: """
            const uint gid = thread_position_in_grid.x;
            const uint lane = thread_position_in_threadgroup.x;
            const uint group = gid >> 6;
            const uint block = group % NBT;
            const uint row = group / NBT;
            const uint idx = row * N + block * 64 + lane;

            if (block >= NBQ) {
                y[idx] = static_cast<outT>(x[idx]);
            } else {
                threadgroup float scratch[64];
                const float input_value = static_cast<float>(x[idx]);
                scratch[lane] = metal::abs(input_value);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = 32; stride > 0; stride >>= 1) {
                    if (lane < stride) {
                        scratch[lane] = metal::max(
                            scratch[lane], scratch[lane + stride]);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                const float amax = metal::max(scratch[0], 1.0e-4f);
                const float raw_scale = amax / 448.0f;
                const uint raw_bits = as_type<uint>(raw_scale);
                const int raw_exp = int((raw_bits >> 23) & 0xffu) - 127;
                const bool has_mantissa = (raw_bits & 0x7fffffu) != 0u;
                const int scale_exp = raw_exp + int(has_mantissa);
                const float scale = as_type<float>(uint(scale_exp + 127) << 23);

                const float normalized = metal::clamp(
                    input_value / scale, -448.0f, 448.0f);
                const float sign = normalized < 0.0f ? -1.0f : 1.0f;
                const float absolute = metal::min(metal::abs(normalized), 448.0f);
                int low = 0;
                int high = 126;
                while (low < high) {
                    const int middle = (low + high + 1) >> 1;
                    const int exponent = (middle >> 3) & 0x0f;
                    const int mantissa = middle & 0x07;
                    const float candidate = exponent == 0
                        ? float(mantissa) * 0.001953125f
                        : (1.0f + float(mantissa) * 0.125f)
                            * metal::fast::exp2(float(exponent - 7));
                    if (candidate <= absolute) low = middle;
                    else high = middle - 1;
                }

                int best = low;
                const int best_exponent = (best >> 3) & 0x0f;
                const int best_mantissa = best & 0x07;
                float best_value = best_exponent == 0
                    ? float(best_mantissa) * 0.001953125f
                    : (1.0f + float(best_mantissa) * 0.125f)
                        * metal::fast::exp2(float(best_exponent - 7));
                if (best < 126) {
                    const int next = best + 1;
                    const int next_exponent = (next >> 3) & 0x0f;
                    const int next_mantissa = next & 0x07;
                    const float next_value = next_exponent == 0
                        ? float(next_mantissa) * 0.001953125f
                        : (1.0f + float(next_mantissa) * 0.125f)
                            * metal::fast::exp2(float(next_exponent - 7));
                    const float best_diff = metal::abs(absolute - best_value);
                    const float next_diff = metal::abs(absolute - next_value);
                    if (next_diff < best_diff ||
                        (next_diff == best_diff && (next & 1) == 0 && (best & 1) != 0)) {
                        best_value = next_value;
                    }
                }
                y[idx] = static_cast<outT>(sign * best_value * scale);
            }
        """)

    private static let indexerActivationRoundTripKernel = MLXFast.metalKernel(
        name: "deepseek_v4_indexer_hadamard128_e2m1_roundtrip",
        inputNames: ["x"],
        outputNames: ["y"],
        source: """
            const uint gid = thread_position_in_grid.x;
            const uint lane = thread_position_in_threadgroup.x;
            const uint row = gid >> 7;
            const uint idx = row * 128 + lane;
            threadgroup float values[128];
            threadgroup float magnitudes[128];

            values[lane] = static_cast<float>(x[idx]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 1; stride < 128; stride <<= 1) {
                if (lane < 64) {
                    const uint block = lane / stride;
                    const uint offset = lane % stride;
                    const uint low_idx = block * 2 * stride + offset;
                    const uint high_idx = low_idx + stride;
                    const float low = values[low_idx];
                    const float high = values[high_idx];
                    values[low_idx] = low + high;
                    values[high_idx] = low - high;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const float rotated = values[lane] * 0.08838834764831845f;
            magnitudes[lane] = metal::abs(rotated);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 16; stride > 0; stride >>= 1) {
                if ((lane & 31u) < stride) {
                    magnitudes[lane] = metal::max(
                        magnitudes[lane], magnitudes[lane + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const uint block_start = lane & ~31u;
            const float amax = metal::max(
                magnitudes[block_start], 7.052966104933725e-38f);
            const float raw_scale = amax / 6.0f;
            const uint raw_bits = as_type<uint>(raw_scale);
            const int raw_exp = int((raw_bits >> 23) & 0xffu) - 127;
            const bool has_mantissa = (raw_bits & 0x7fffffu) != 0u;
            const int scale_exp = raw_exp + int(has_mantissa);
            const float scale = as_type<float>(uint(scale_exp + 127) << 23);

            const float normalized = metal::clamp(rotated / scale, -6.0f, 6.0f);
            const float absolute = metal::abs(normalized);
            constexpr float codebook[8] = {
                0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
            };
            int best = 0;
            float best_diff = absolute;
            for (int code = 1; code < 8; ++code) {
                const float diff = metal::abs(absolute - codebook[code]);
                if (diff < best_diff ||
                    (diff == best_diff && (code & 1) == 0 && (best & 1) != 0)) {
                    best = code;
                    best_diff = diff;
                }
            }
            const float sign = normalized < 0.0f ? -1.0f : 1.0f;
            y[idx] = static_cast<outT>(sign * codebook[best] * scale);
        """)

    /// Apply the pinned 0731 post-RoPE KV activation-QAT graph while copying
    /// the RoPE suffix exactly. `ropeDim` and the non-RoPE prefix are both
    /// 64-aligned in the released model (512 = 448 + 64).
    public static func e4m3KVActivationRoundTrip(
        _ x: MLXArray, ropeDim: Int
    ) -> MLXArray {
        let width = x.dim(-1)
        let nopeWidth = width - ropeDim
        precondition(
            width > 0 && ropeDim >= 0 && nopeWidth >= 0
                && width.isMultiple(of: 64) && nopeWidth.isMultiple(of: 64),
            "DSV4 E4M3 KV QAT requires 64-aligned full and non-RoPE widths")
        if nopeWidth == 0 { return x }
        let input = contiguous(x)
        let rows = input.size / width
        return e4m3KVActivationRoundTripKernel(
            [input],
            template: [
                ("N", width),
                ("NBQ", nopeWidth / 64),
                ("NBT", width / 64),
                ("outT", input.dtype),
            ],
            grid: (rows * width, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [input.shape],
            outputDTypes: [input.dtype]
        )[0]
    }

    /// Pinned 0731 indexer activation graph: normalized Hadamard-128 followed
    /// by block-32 power-of-two E2M1 fake quantization.
    public static func indexerActivationRoundTrip(_ x: MLXArray) -> MLXArray {
        precondition(x.dim(-1) == 128, "DSV4 indexer QAT requires 128-wide rows")
        let input = contiguous(x)
        let rows = input.size / 128
        return indexerActivationRoundTripKernel(
            [input],
            template: [("outT", input.dtype)],
            grid: (rows * 128, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [input.shape],
            outputDTypes: [input.dtype]
        )[0]
    }

    /// Routed expert outputs already include their per-route weights before
    /// the quantized down projection. Accumulate the expert axis in fp32; the
    /// caller adds the shared expert in fp32 and casts only once afterward.
    public static func reduceRoutedExpertsFP32(_ routed: MLXArray) -> MLXArray {
        routed.asType(.float32).sum(axis: -2)
    }

    public static func addSharedExpertFP32(
        _ routed: MLXArray, shared: MLXArray, outputDType: DType
    ) -> MLXArray {
        (routed.asType(.float32) + shared.asType(.float32)).asType(outputDType)
    }

    /// Official hyper-connection expansion. PyTorch broadcasts the first
    /// `comb` axis against the residual HC axis and sums that axis, which is
    /// algebraically `combᵀ @ residual`, not `comb @ residual`.
    public static func hcExpandResidual(
        comb: MLXArray, residual: MLXArray
    ) -> MLXArray {
        comb.asType(.float32).swappedAxes(-1, -2)
            .matmul(residual.asType(.float32))
    }

    /// Apply DSV4's official mHC post contraction. Single-token GPU decode
    /// uses the fused Metal kernel; multi-token prefill retains the batched
    /// MLX path where large matrix operations have better throughput.
    public static func hcPost(
        blockOut: MLXArray,
        residual: MLXArray,
        post: MLXArray,
        comb: MLXArray
    ) -> MLXArray {
        let hcMult = comb.dim(-1)
        let hiddenSize = blockOut.dim(-1)
        let isSingleTokenDecode =
            blockOut.ndim >= 3 && blockOut.dim(-2) == 1
                && residual.dim(-2) == hcMult
                && comb.dim(-2) == hcMult
                && blockOut.size > 0
        if Device.defaultDevice().deviceType == .gpu && isSingleTokenDecode {
            let x = contiguous(blockOut)
            let residual = contiguous(residual)
            let post = contiguous(post)
            let comb = contiguous(comb)
            let outputShape = Array(x.shape.dropLast()) + [hcMult, hiddenSize]
            return hcPostDecodeKernel(
                [x, residual, post, comb],
                template: [
                    ("HC", hcMult),
                    ("D", hiddenSize),
                    ("outT", x.dtype),
                ],
                grid: (x.size * hcMult, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [outputShape],
                outputDTypes: [x.dtype]
            )[0]
        }

        let combResid = hcExpandResidual(comb: comb, residual: residual)
        return (
            post.asType(.float32).expandedDimensions(axis: -1)
                * blockOut.asType(.float32).expandedDimensions(axis: -2)
                + combResid
        ).asType(blockOut.dtype)
    }

    /// Python `_DSV4_COMPILE_REGIONS` parity gate. Decode-only compiled
    /// regions collapse per-token host graph construction; prefill keeps the
    /// raw graph so trace count stays bounded.
    public static let compileRegionsEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["DSV4_COMPILE_REGIONS"]?
            .lowercased()
        return raw != "0" && raw != "false" && raw != "off" && raw != "no"
    }()

    /// `VMLX_DSV4_LAYER_TRACE=1` reports the last position's magnitude after
    /// every decoder layer. Diagnostic only.
    public static let layerTraceEnabled: Bool = {
        ProcessInfo.processInfo.environment["VMLX_DSV4_LAYER_TRACE"] == "1"
    }()

    /// `absmax`/`absmean` of the final position of `h`, as an *unevaluated*
    /// scalar pair. Both statistics are needed: a NaN/Inf blowup moves `absmax`
    /// alone, while a routing or weight fault that keeps the state finite but
    /// wrong moves `absmean`.
    ///
    /// Deliberately lazy. Reading these per layer would insert a barrier into
    /// exactly the lazy cross-layer graph under suspicion, so a laziness or
    /// concurrency fault could vanish under its own instrumentation. Collect
    /// here; evaluate once at the end of the forward pass.
    public static func layerStat(_ tag: String, _ h: MLXArray) -> (String, MLXArray)? {
        guard layerTraceEnabled, h.ndim >= 2 else { return nil }
        let last = MLX.take(h, MLXArray([Int32(h.dim(1) - 1)]), axis: 1)
        let a = MLX.abs(last.asType(.float32))
        return (tag, MLX.stacked([a.max(), a.mean()]))
    }

    public static func flushLayerStats(_ stats: [(String, MLXArray)], length: Int) {
        guard layerTraceEnabled, !stats.isEmpty else { return }
        MLX.eval(stats.map(\.1))
        for (tag, stat) in stats {
            let v = stat.asArray(Float.self)
            print("[vmlx][dsv4/layer] \(tag) L=\(length) absmax=\(v[0]) absmean=\(v[1])")
        }
    }

    /// `VMLX_DSV4_WEIGHT_CHECKSUM=1` fingerprints layers 14-16 weight tensors
    /// once per process, after the first forward. Separates "weights corrupted
    /// at load" from "weights fine, compute path corrupted" on a souped boot.
    public static let weightChecksumEnabled: Bool = {
        ProcessInfo.processInfo.environment["VMLX_DSV4_WEIGHT_CHECKSUM"] == "1"
    }()

    private static let weightChecksumOnceLock = NSLock()
    nonisolated(unsafe) private static var weightChecksumDone = false

    public static func runWeightChecksumOnce(_ body: () -> Void) {
        weightChecksumOnceLock.lock()
        let shouldRun = !weightChecksumDone
        weightChecksumDone = true
        weightChecksumOnceLock.unlock()
        if shouldRun { body() }
    }

    /// Python `_dsv4_attn_subchunk_tokens` parity: attention sub-chunk length
    /// for wide-chunk prefill. SWA/CSA attention cost grows super-linearly
    /// with chunk width while MoE gather_qmm throughput grows with batch, so
    /// attention runs in 512-token sub-slices against the layer cache while
    /// MoE consumes the full chunk. 0 disables.
    public static let attnSubchunkTokens: Int = {
        guard let raw = ProcessInfo.processInfo.environment["DSV4_ATTN_SUBCHUNK"],
            let v = Int(raw)
        else { return 512 }
        return max(0, v)
    }()

    /// Python `_layerwise_prefill_materialization_enabled` parity: bound the
    /// lazy cross-layer graph during multi-token prefill once the final
    /// context exceeds the standalone-proven safe curve. Barriers cost
    /// 25-30% prefill throughput, so with sub-chunking active they engage
    /// only past `DSV4_LAYERWISE_PREFILL_AUTO_TOKENS` (default 24,576).
    public static func layerwisePrefillEnabled(
        chunkTokens: Int, finalContextTokens: Int
    ) -> Bool {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["DSV4_LAYERWISE_PREFILL"]?.lowercased(),
            ["0", "false", "no", "off"].contains(raw)
        {
            return false
        }
        let minTokens = max(2, env["DSV4_LAYERWISE_PREFILL_MIN_TOKENS"].flatMap(Int.init) ?? 256)
        if chunkTokens < minTokens { return false }
        let explicit = env["DSV4_LAYERWISE_PREFILL"] != nil
        if !explicit && attnSubchunkTokens > 0 {
            let auto = env["DSV4_LAYERWISE_PREFILL_AUTO_TOKENS"].flatMap(Int.init) ?? 24576
            if auto > 0 && finalContextTokens <= auto { return false }
        }
        return true
    }

    /// Raw mHC-pre math (Python `_hc_pre_impl`): fp32 flatten, scalar
    /// reciprocal RMS applied after the projection, split-Sinkhorn, four-way
    /// residual mix. Shared by the eager path and the compiled decode region.
    /// - Parameters:
    ///   - eps: `hc_eps`. The epsilon of the sigmoid / softmax / Sinkhorn arithmetic.
    ///   - normEps: `rms_norm_eps`. The epsilon of the UNWEIGHTED input RMS norm, which the
    ///     reference takes from a different config field. DSV4 ships both at 1e-6, so reusing one
    ///     for the other is invisible there; GLM-5.3 ships 1e-5 and 1e-6, where it is not.
    public static func hcPreGraph(
        _ h: MLXArray,
        fn: MLXArray,
        scale: MLXArray,
        base: MLXArray,
        hcMult: Int,
        hiddenSize: Int,
        iters: Int,
        eps: Float,
        normEps: Float
    ) -> (x: MLXArray, post: MLXArray, comb: MLXArray) {
        let B = h.dim(0)
        let L = h.dim(1)
        let xFlat = h.reshaped(B, L, hcMult * hiddenSize).asType(.float32)
        let reciprocalRMS = rsqrt(
            (xFlat * xFlat).mean(axis: -1, keepDims: true) + normEps)
        let mixes = xFlat.matmul(fn.asType(.float32).transposed()) * reciprocalRMS
        let (pre, post, comb) = hcSplitSinkhorn(
            mixes: mixes, scale: scale, base: base,
            hcMult: hcMult, iters: iters, eps: eps)
        let xReshape = xFlat.reshaped(B, L, hcMult, hiddenSize)
        let y = (pre.expandedDimensions(axis: -1) * xReshape).sum(axis: -2)
        return (x: y.asType(h.dtype), post: post, comb: comb)
    }

    private static let hcPreCompiledLock = NSLock()
    nonisolated(unsafe) private static var hcPreCompiledCache:
        [String: ([MLXArray]) -> [MLXArray]] = [:]

    /// Python `_get_hc_pre_compiled` parity: ONE compiled mHC-pre region
    /// shared by every layer, with weights as inputs. The Sinkhorn Metal
    /// kernel traces cleanly inside the region; fusing the surrounding
    /// cast/RMS/mix/reduce glue collapses ~10 host dispatches per call
    /// (86 calls per generated token) into one.
    public static func hcPreCompiled(
        _ h: MLXArray,
        fn: MLXArray,
        scale: MLXArray,
        base: MLXArray,
        hcMult: Int,
        hiddenSize: Int,
        iters: Int,
        eps: Float,
        normEps: Float
    ) -> (x: MLXArray, post: MLXArray, comb: MLXArray) {
        // `normEps` belongs in the key. The cached region is a function of all FIVE quantities, but
        // the key named only four — so two families agreeing on hcMult/hiddenSize/iters/eps and
        // differing only in the norm epsilon would collide, and the second one silently gets the
        // first one's region. DSV4 alone cannot expose this: it derives both epsilons from a
        // configuration that always makes them equal, so its key is unambiguous by accident.
        let key = "\(hcMult)|\(hiddenSize)|\(iters)|\(eps)|\(normEps)"
        hcPreCompiledLock.lock()
        var region = hcPreCompiledCache[key]
        if region == nil {
            region = vmlxTrustedCompile { (args: [MLXArray]) -> [MLXArray] in
                let out = hcPreGraph(
                    args[0], fn: args[1], scale: args[2], base: args[3],
                    hcMult: hcMult, hiddenSize: hiddenSize,
                    iters: iters, eps: eps, normEps: normEps)
                return [out.x, out.post, out.comb]
            }
            hcPreCompiledCache[key] = region
        }
        hcPreCompiledLock.unlock()
        let outs = region!([h, fn, scale, base])
        return (x: outs[0], post: outs[1], comb: outs[2])
    }

    // MARK: - Fused mHC split-Sinkhorn
    //
    // DSV4 executes this twice per transformer layer. Expressing twenty
    // Sinkhorn iterations as ordinary MLX ops creates more than forty graph
    // nodes per call (roughly 3,400 nodes per token across 43 layers). The
    // reference DSV4 MLX runtime uses one Metal dispatch instead. Keep the
    // same fp32 arithmetic and exact normalization order here.
    private static let hcPostDecodeKernel = MLXFast.metalKernel(
        name: "deepseek_v4_hc_post_decode",
        inputNames: ["x", "residual", "post", "comb"],
        outputNames: ["y"],
        source: """
            const uint gid = thread_position_in_grid.x;
            const uint d = gid % D;
            const uint residual_row = gid / D;
            const uint target_hc = residual_row % HC;
            const uint batch_row = residual_row / HC;

            float residual_mix =
                static_cast<float>(comb[batch_row * HC * HC + target_hc])
                * static_cast<float>(residual[batch_row * HC * D + d]);
            for (uint source_hc = 1; source_hc < HC; ++source_hc) {
                const float term =
                    static_cast<float>(
                        comb[batch_row * HC * HC + source_hc * HC + target_hc])
                    * static_cast<float>(
                        residual[batch_row * HC * D + source_hc * D + d]);
                residual_mix = residual_mix + term;
            }

            const float direct =
                static_cast<float>(post[batch_row * HC + target_hc])
                * static_cast<float>(x[batch_row * D + d]);
            y[gid] = static_cast<outT>(direct + residual_mix);
        """)

    private static let hcSplitSinkhornKernel = MLXFast.metalKernel(
        name: "deepseek_v4_hc_split_sinkhorn",
        inputNames: ["mixes", "scale", "base", "eps"],
        outputNames: ["pre", "post", "comb"],
        source: """
            uint idx = thread_position_in_grid.x;
            constexpr int MIX = (2 + HC) * HC;
            float epsv = static_cast<float>(eps[0]);

            auto mix = mixes + idx * MIX;
            auto pre_out = pre + idx * HC;
            auto post_out = post + idx * HC;
            auto comb_out = comb + idx * HC * HC;

            float pre_scale = static_cast<float>(scale[0]);
            float post_scale = static_cast<float>(scale[1]);
            float comb_scale = static_cast<float>(scale[2]);

            for (int i = 0; i < HC; ++i) {
                float z = static_cast<float>(mix[i]) * pre_scale
                    + static_cast<float>(base[i]);
                pre_out[i] = 1.0f / (1.0f + metal::fast::exp(-z)) + epsv;
            }
            for (int i = 0; i < HC; ++i) {
                int off = HC + i;
                float z = static_cast<float>(mix[off]) * post_scale
                    + static_cast<float>(base[off]);
                post_out[i] = 2.0f / (1.0f + metal::fast::exp(-z));
            }

            float c[HC * HC];
            for (int i = 0; i < HC; ++i) {
                float row_max = -INFINITY;
                for (int j = 0; j < HC; ++j) {
                    int cidx = i * HC + j;
                    int off = 2 * HC + cidx;
                    float v = static_cast<float>(mix[off]) * comb_scale
                        + static_cast<float>(base[off]);
                    c[cidx] = v;
                    row_max = metal::max(row_max, v);
                }
                float row_sum = 0.0f;
                for (int j = 0; j < HC; ++j) {
                    int cidx = i * HC + j;
                    float v = metal::fast::exp(c[cidx] - row_max);
                    c[cidx] = v;
                    row_sum += v;
                }
                float inv_sum = 1.0f / row_sum;
                for (int j = 0; j < HC; ++j) {
                    int cidx = i * HC + j;
                    c[cidx] = c[cidx] * inv_sum + epsv;
                }
            }

            for (int j = 0; j < HC; ++j) {
                float col_sum = 0.0f;
                for (int i = 0; i < HC; ++i) {
                    col_sum += c[i * HC + j];
                }
                float inv_denom = 1.0f / (col_sum + epsv);
                for (int i = 0; i < HC; ++i) {
                    c[i * HC + j] *= inv_denom;
                }
            }

            for (int iter = 1; iter < ITERS; ++iter) {
                for (int i = 0; i < HC; ++i) {
                    float row_sum = 0.0f;
                    for (int j = 0; j < HC; ++j) {
                        row_sum += c[i * HC + j];
                    }
                    float inv_denom = 1.0f / (row_sum + epsv);
                    for (int j = 0; j < HC; ++j) {
                        c[i * HC + j] *= inv_denom;
                    }
                }
                for (int j = 0; j < HC; ++j) {
                    float col_sum = 0.0f;
                    for (int i = 0; i < HC; ++i) {
                        col_sum += c[i * HC + j];
                    }
                    float inv_denom = 1.0f / (col_sum + epsv);
                    for (int i = 0; i < HC; ++i) {
                        c[i * HC + j] *= inv_denom;
                    }
                }
            }

            for (int i = 0; i < HC * HC; ++i) {
                comb_out[i] = c[i];
            }
        """)

    private static let scalarArrayLock = NSLock()
    nonisolated(unsafe) private static var scalarArrays: [Float: MLXArray] = [:]

    private static func scalarArray(_ value: Float) -> MLXArray {
        scalarArrayLock.lock()
        defer { scalarArrayLock.unlock() }
        if let cached = scalarArrays[value] { return cached }
        let array = MLXArray([value])
        scalarArrays[value] = array
        return array
    }

    // MARK: - Per-head Q RMSNorm ones cache
    //
    // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
    // The DSV4 attention applies a unit-weight RMSNorm-like rescale per
    // head before partial RoPE: `q * rsqrt((q^2).mean(-1) + eps)`. The
    // Python reference fuses this into a single `mx.fast.rms_norm` kernel
    // with a cached ones tensor. The cache is keyed on `(headDim, dtype)`
    // and shared across all 64 attention heads × 43 layers. NSLock-guarded
    // so concurrent SwitchGLU/attention forwards never race the cache.
    private static let qNormOnesLock = NSLock()
    nonisolated(unsafe) private static var qNormOnesCache: [String: MLXArray] = [:]

    public static func qNormOnes(headDim: Int, dtype: DType) -> MLXArray {
        let key = "\(headDim)|\(dtype)"
        qNormOnesLock.lock(); defer { qNormOnesLock.unlock() }
        if let w = qNormOnesCache[key] { return w }
        let w = MLXArray.ones([headDim], dtype: dtype)
        // `ones` is a pending `full` node; publishing it unevaluated would let
        // two concurrent requests drive `eval` on the same array_desc.
        w.eval()
        qNormOnesCache[key] = w
        return w
    }


    // MARK: - mHC split-Sinkhorn (collapse matrices)
    //
    // Given `mixes` of shape (..., 3*hcMult) and per-block scale/base
    // parameters, produce the three matrices needed by the HC collapse
    // kernel:
    //
    //   pre   = sigmoid(mixes * scale[0] + base[:hcMult]) + eps
    //           (no normalization — used to weight residual copies)
    //
    //   post  = 2 * sigmoid(mixes * scale[1] + base[hcMult:2*hcMult])
    //           (no eps — used to scale block output before add-back)
    //
    //   comb  = softmax(mixes * scale[2] + base[2*hcMult:3*hcMult], axis=-1) + eps
    //           col-normalize
    //           repeat (iters-1)× { row-normalize; col-normalize }
    //
    // `comb` is the sinkhorn doubly-stochastic mixing matrix that
    // preserves residual norm when used for the `expand` step.
    //
    // Shape contract:
    //   mixes: (..., 3*hcMult)
    //   scale: (3,)       one learned scalar per field
    //   base:  (3*hcMult,) learned bias concatenated across fields
    //   → pre:  (..., hcMult)
    //   → post: (..., hcMult)
    //   → comb: (..., hcMult, hcMult)
    public static func hcSplitSinkhorn(
        mixes: MLXArray,
        scale: MLXArray,
        base: MLXArray,
        hcMult: Int,
        iters: Int = 20,
        eps: Float = 1e-6
    ) -> (pre: MLXArray, post: MLXArray, comb: MLXArray) {
        if Device.defaultDevice().deviceType == .gpu {
            let leadShape = Array(mixes.shape.dropLast())
            let rows = mixes.size / ((2 + hcMult) * hcMult)
            let outputs = hcSplitSinkhornKernel(
                [mixes, scale, base, scalarArray(eps)],
                template: [("HC", hcMult), ("ITERS", iters)],
                grid: (rows, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [
                    leadShape + [hcMult],
                    leadShape + [hcMult],
                    leadShape + [hcMult, hcMult],
                ],
                outputDTypes: [.float32, .float32, .float32])
            return (pre: outputs[0], post: outputs[1], comb: outputs[2])
        }
        return hcSplitSinkhornOps(
            mixes: mixes,
            scale: scale,
            base: base,
            hcMult: hcMult,
            iters: iters,
            eps: eps)
    }

    /// Pure-op reference and non-Metal fallback for the fused kernel above.
    public static func hcSplitSinkhornOps(
        mixes: MLXArray,
        scale: MLXArray,
        base: MLXArray,
        hcMult: Int,
        iters: Int = 20,
        eps: Float = 1e-6
    ) -> (pre: MLXArray, post: MLXArray, comb: MLXArray) {
        // Match Python `_hc_split_sinkhorn_ops` exactly. Mixes width is
        // `(2 + hcMult) * hcMult`, not `3 * hcMult`. The first hcMult
        // elements form `pre`, the next hcMult form `post`, and the
        // remaining `hcMult * hcMult` are reshaped into the (hc, hc)
        // doubly-stochastic mixing matrix `comb`.
        let mh = hcMult
        let mixHc = (2 + mh) * mh
        precondition(
            mixes.shape.last == mixHc,
            "mixes last dim must be (2+hcMult)*hcMult = \(mixHc), got \(mixes.shape.last ?? -1)")

        // Bring everything to fp32 for numerical stability — the
        // sinkhorn iterations are sensitive to fp16 underflow on the
        // post-softmax row/col normalizations.
        let mixesF = mixes.asType(.float32)
        let scaleF = scale.asType(.float32)
        let baseF = base.asType(.float32)

        let preScale = scaleF[0]
        let postScale = scaleF[1]
        let combScale = scaleF[2]

        let basePre = baseF[0..<mh]
        let basePost = baseF[mh..<(2 * mh)]
        let baseComb = baseF[(2 * mh)...]  // length mh*mh

        let mixPre = mixesF[.ellipsis, 0..<mh]
        let mixPost = mixesF[.ellipsis, mh..<(2 * mh)]
        let mixCombFlat = mixesF[.ellipsis, (2 * mh)...]  // (..., mh*mh)

        let pre = sigmoid(mixPre * preScale + basePre) + eps
        let post = 2.0 * sigmoid(mixPost * postScale + basePost)

        // Reshape last axis (mh*mh) into (mh, mh) and add bias also
        // reshaped to (mh, mh).
        let leadShape = Array(mixCombFlat.shape.dropLast())
        var combRaw = mixCombFlat * combScale
        combRaw = combRaw.reshaped(leadShape + [mh, mh])
            + baseComb.reshaped([mh, mh])
        var comb = softmax(combRaw, axis: -1) + eps
        // Initial col-normalize, then (iters-1) × {row, col}.
        comb = sinkhornColNormalize(comb, eps: eps)
        for _ in 0..<max(iters - 1, 0) {
            comb = sinkhornRowNormalize(comb, eps: eps)
            comb = sinkhornColNormalize(comb, eps: eps)
        }

        return (pre: pre, post: post, comb: comb)
    }

    private static func sinkhornRowNormalize(_ x: MLXArray, eps: Float) -> MLXArray {
        let rowSum = x.sum(axis: -1, keepDims: true)
        return x / (rowSum + eps)
    }

    private static func sinkhornColNormalize(_ x: MLXArray, eps: Float) -> MLXArray {
        let colSum = x.sum(axis: -2, keepDims: true)
        return x / (colSum + eps)
    }

    // MARK: - Partial RoPE
    //
    // DSV4 applies rotary ONLY to the last `ropeDim` (default 64) of
    // the head-dim=512 Q/K vector — the first 448 dims are "no-position".
    // Forward (token -> position-rotated): standard RoPE rotate.
    // Inverse (position-rotated -> token, used on attention OUTPUT):
    //   undo the rotation via negative-angle cos/sin, so the residual
    //   stream contribution is position-agnostic.
    public static func applyPartialRoPE(
        _ x: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        ropeDim: Int,
        inverse: Bool = false
    ) -> MLXArray {
        let headDim = x.shape.last!
        precondition(ropeDim <= headDim, "ropeDim must be ≤ headDim")
        let noPoseDim = headDim - ropeDim
        if noPoseDim == 0 {
            return rotateHalf(x, cos: cos, sin: sin, inverse: inverse)
        }
        // Split last axis: [..., :noPoseDim] keep; [..., noPoseDim:] rotate.
        let nope = x[.ellipsis, 0..<noPoseDim]
        let pe = x[.ellipsis, noPoseDim...]
        let rotated = rotateHalf(pe, cos: cos, sin: sin, inverse: inverse)
        return concatenated([nope, rotated], axis: -1)
    }

    /// Apply traditional/interleaved RoPE — DSV4 uses
    /// `traditional=True` (mx.fast.rope) which rotates ADJACENT pairs:
    /// `(x[…,0], x[…,1])`, `(x[…,2], x[…,3])`, etc. NOT split-half
    /// `(x[…,:D/2], x[…,D/2:])`. Mirror Python `_call_manual` in
    /// jang_tools/dsv4/mlx_model.py:DeepseekV4RoPE — using the wrong
    /// convention scrambles positional information across the head
    /// dim and the model decodes a repeating-token loop (verified
    /// 2026-04-24).
    ///
    /// `cos`/`sin` shape must broadcast over the leading axes and
    /// match `(L, ropeDim/2)`. `inverse=true` flips sin sign
    /// (equivalent to multiplying by conjugate of the rotation).
    private static func rotateHalf(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, inverse: Bool
    ) -> MLXArray {
        // The frequency table is intentionally built in fp32, but the
        // authoritative DSV4 runtime casts cos/sin to the activation dtype
        // before applying RoPE. Without this cast Swift promotes Q/K to fp32
        // in layer 0; the fp32 attention result then promotes the entire mHC
        // residual stream. Every later affine expert call consequently casts
        // its 67-134 MiB fp16 scale/bias tensors to fp32 on every token.
        // Preserve the model dtype here, matching DeepseekV4RoPE.__call__ in
        // jang_tools/dsv4/mlx_model.py.
        let c = cos.asType(x.dtype)
        let sinForDirection = inverse ? -sin : sin
        let s = sinForDirection.asType(x.dtype)
        let lastDim = x.shape.last!
        let halfDim = lastDim / 2
        // Reshape last axis from D to (D/2, 2) so the trailing pair
        // is the (real, imag) tuple of each rotation.
        let xPaired = x.reshaped(x.shape.dropLast() + [halfDim, 2])
        let x0 = xPaired[.ellipsis, 0]  // (..., D/2)
        let x1 = xPaired[.ellipsis, 1]  // (..., D/2)
        let r0 = x0 * c - x1 * s
        let r1 = x0 * s + x1 * c
        // Stack along a new last axis (D/2, 2) then collapse → D.
        let stacked = stacked([r0, r1], axis: -1)
        return stacked.reshaped(x.shape)
    }

    // MARK: - DSV4 SwiGLU activation with `limit`
    //
    // silu(min(gate, limit)) * clip(up, -limit, +limit). The authoritative
    // affine runtime evaluates the clamp, SiLU, and multiply in fp32 and only
    // then casts back to the incoming dtype. This is both a precision contract
    // across the 43-layer MoE stack and a compiled one-dispatch microkernel.
    public static func dsv4SwiGLU(
        gate: MLXArray,
        up: MLXArray,
        limit: Float
    ) -> MLXArray {
        let body = limit > 0
            ? _deepseekV4SwiGLUClampedBody
            : _deepseekV4SwiGLUUnclampedBody
        let compiled = limit > 0
            ? _compiledDeepseekV4SwiGLUClamped
            : _compiledDeepseekV4SwiGLUUnclamped
        let limitArray = scalarArray(limit)
        if CompiledDecodeTrace.isActive {
            return body(gate, up, limitArray)
        }
        return compiled(gate, up, limitArray)
    }

    /// Routed DSV4 expert activation. The score is multiplied while the
    /// limited-SwiGLU result is still fp32, then cast to the expert activation
    /// dtype immediately before the quantized down projection.
    public static func dsv4ScoredSwiGLU(
        gate: MLXArray,
        up: MLXArray,
        scores: MLXArray,
        limit: Float
    ) -> MLXArray {
        let body = limit > 0
            ? _deepseekV4ScoredSwiGLUClampedArrayBody
            : _deepseekV4ScoredSwiGLUUnclampedArrayBody
        let compiled = limit > 0
            ? _compiledDeepseekV4ScoredSwiGLUClamped
            : _compiledDeepseekV4ScoredSwiGLUUnclamped
        let limitArray = scalarArray(limit)
        let args = [gate, up, scores, limitArray]
        if CompiledDecodeTrace.isActive {
            return body(args)[0]
        }
        return compiled(args).first ?? body(args)[0]
    }

    // MARK: - sqrtsoftplus (MoE gate scoring)
    //
    // scores = sqrt(log1p(exp(logits))) — replaces softmax for DSV4's
    // routing. Monotonic, smoother gradient in the tail than softmax,
    // and doesn't require the sum-to-1 constraint that makes hash
    // routing incompatible.
    //
    // Numerical guard: log1p(exp(x)) is `softplus(x)` — mlx exposes
    // it directly and handles the overflow branch for large x.
    public static func sqrtSoftplus(_ logits: MLXArray) -> MLXArray {
        sqrt(logAddExp(logits, MLXArray(0.0)))
    }

    // MARK: - Top-k over sqrtsoftplus with bias + norm
    //
    // Production gate path (for non-hash layers):
    //   biased = scores + noauxBias
    //   topKIdx = argpartition(-biased, k)[:k]
    //   topKWeights = take_along_axis(scores, topKIdx)   — UNBIASED!
    //   normalized = topKWeights / sum(topKWeights) * routedScalingFactor
    //
    // Critical: `noauxBias` is used ONLY to pick the indices — once
    // picked, the UNBIASED score is what gets used as the expert
    // weight. This was bug #6 in the DSV-EXHAUSTIVE-VARIABLES-GUIDE;
    // using biased weights broke coherence.
    public static func sqrtSoftplusSelect(
        scores: MLXArray,
        noauxBias: MLXArray?,
        k: Int,
        normalize: Bool,
        scalingFactor: Float
    ) -> (indices: MLXArray, weights: MLXArray) {
        let biased = noauxBias != nil ? (scores + noauxBias!) : scores
        // argpartition returns unordered top-k; sort indices for
        // determinism (matters for cache-hit byte equivalence).
        let topKIdx = argPartition(-biased, kth: k - 1, axis: -1)[.ellipsis, 0..<k]
        // Gather the UNBIASED scores at those indices.
        let gathered = takeAlong(scores, topKIdx, axis: -1)
        var weights = gathered
        if normalize {
            let denom = weights.sum(axis: -1, keepDims: true) + 1e-20
            weights = weights / denom * scalingFactor
        } else {
            weights = weights * scalingFactor
        }
        return (indices: topKIdx, weights: weights)
    }

    // MARK: - YaRN RoPE freq table
    //
    // `rope_factor=16`, `original_seq_len=65536`, `beta_fast=32`,
    // `beta_slow=1` are the DSV4 defaults when compress_ratio>0.
    // Layers with compress_ratio==0 use plain (non-YaRN) RoPE with
    // `rope_theta=10000`.
    public static func yarnInvFreq(
        dim: Int,
        base: Float,
        maxPos: Int,
        origMaxPos: Int,
        factor: Float,
        betaFast: Float,
        betaSlow: Float
    ) -> MLXArray {
        _ = maxPos  // reserved for future extrapolation logic

        // Standard inv-freq table.
        let dimF = Float(dim)
        let halfDim = dim / 2
        var invFreq = [Float]()
        invFreq.reserveCapacity(halfDim)
        for i in 0..<halfDim {
            let exponent = Float(2 * i) / dimF
            invFreq.append(1.0 / pow(base, exponent))
        }
        let invFreqArr = MLXArray(invFreq)

        if factor == 1.0 {
            return invFreqArr
        }

        // YaRN: ramp mask smooths the transition between full and
        // scaled frequencies for dims that correspond to wavelengths
        // between betaFast and betaSlow. Match the pinned DSV4 source's
        // `find_correction_range(beta_fast, beta_slow, ...)`: betaFast owns
        // the low index and betaSlow owns the high index. Reversing them
        // collapses the intended ramp into a clamped step for the production
        // 0731 parameters. `high = min(..., dim - 1)` follows the source.
        let twoPi = Float.pi * 2
        func correctionDim(_ beta: Float) -> Float {
            dimF * log(Float(origMaxPos) / (beta * twoPi)) / (2.0 * log(base))
        }
        let low = max(0.0, floor(correctionDim(betaFast)))
        let high = min(Float(dim - 1), ceil(correctionDim(betaSlow)))
        let rangeWidth = max(high - low, 0.001)

        var ramp = [Float]()
        ramp.reserveCapacity(halfDim)
        for i in 0..<halfDim {
            let t = (Float(i) - low) / rangeWidth
            ramp.append(max(0.0, min(1.0, t)))
        }
        let rampArr = MLXArray(ramp)
        let smooth = MLXArray(1.0) - rampArr
        let scaled = invFreqArr / factor
        return scaled * (MLXArray(1.0) - smooth) + invFreqArr * smooth
    }

    // MARK: - LM head fp32 matmul
    //
    // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
    // DeepSeek-V4 reference inference (`inference/model.py`
    // `ParallelHead.get_logits`) explicitly performs the lm_head matmul
    // in fp32 — the weights are stored as fp32 and the activations are
    // cast to fp32 before `F.linear`. Hidden contraction is 4096 wide,
    // and bf16/fp16 accumulation drops ~0.5 ULP per logit which
    // empirically flips arithmetic-style next-token answers.
    //
    // Default path is now the fused quantized matmul with fp32
    // activations: the kernel dequantizes per-group in-registers, so
    // per token it reads only the packed weights (~0.55 GB vs ~1.97 GB
    // for a materialized fp32 head at 4096×vocab), while accumulation
    // stays fp32. Python vmlx (23d2ac294) measured greedy-identical
    // outputs over 160 steps, logits rel diff 4.8e-04, −0.62 ms/tok.
    // VMLX_DSV4_LM_HEAD_MODE=exact restores the materialized path.
    static let lmHeadMode: String =
        (ProcessInfo.processInfo.environment["VMLX_DSV4_LM_HEAD_MODE"] ?? "qmm").lowercased()

    public static func lmHeadFp32(_ h: MLXArray, lmHead: Linear) -> MLXArray {
        if let q = lmHead as? QuantizedLinear {
            if lmHeadMode != "exact" {
                let hF32 = h.asType(.float32)
                var out = MLX.quantizedMM(
                    hF32, q.weight, scales: q.scales, biases: q.biases,
                    transpose: true,
                    groupSize: q.groupSize, bits: q.bits, mode: q.mode
                )
                if let b = q.bias {
                    out = out + b.asType(.float32)
                }
                return out
            }
            let wF32 = MLX.dequantized(
                q.weight, scales: q.scales, biases: q.biases,
                groupSize: q.groupSize, bits: q.bits, mode: q.mode
            ).asType(.float32)
            let hF32 = h.asType(.float32)
            var out = hF32.matmul(wF32.transposed())
            if let b = q.bias {
                out = out + b.asType(.float32)
            }
            return out
        }
        let wF32 = lmHead.weight.asType(.float32)
        let hF32 = h.asType(.float32)
        var out = hF32.matmul(wF32.transposed())
        if let b = lmHead.bias {
            out = out + b.asType(.float32)
        }
        return out
    }

    // MARK: - Compressor + Indexer attention masks (PR #1195 port)
    //
    // The DSV4 paper §9-13 attention path is a hybrid of:
    //   1. A LOCAL sliding-window over the last `window` raw tokens
    //      (kept in a RotatingKVCache).
    //   2. A GLOBAL pooled context over compressor chunks (kept as
    //      a single (B, P, head_dim) tensor in an ArraysCache slot).
    //
    // Visibility from query at raw position `q` to a key:
    //
    //   - Window key at raw position `r`:
    //       q - window < r <= q
    //
    //   - Compressed key at chunk index `k` covering raw range
    //     [k*ratio, (k+1)*ratio):
    //       (k + 1) * ratio <= q + 1
    //
    // For compress_ratio==4 layers the Indexer adds a top-k
    // selection — only the K compressed chunks the indexer scored
    // highest are visible, ANDed with the causal staircase above.
    //
    // Both helpers return 4D bool arrays of shape (B, 1, S, L_kv)
    // that broadcast onto SDPA attention scores (B, H, S, L_kv).
    // Building 4D directly avoids the SDPA broadcast bugs the
    // previous staircase attempts hit.

    /// Per-query visibility into the local sliding-window cache.
    ///
    /// `windowLen` is the number of slots currently filled in the
    /// RotatingKVCache (== window once the buffer wraps). The
    /// trailing `windowLen` raw positions in the cache map to raw
    /// token indices `(offset + S) - windowLen + i` for slot `i`.
    /// Returns shape `(B, 1, S, windowLen)`.
    public static func buildWindowMask(
        batch B: Int, queryLen S: Int,
        offset: Int, window: Int, windowLen: Int
    ) -> MLXArray {
        // q_pos: (B, S) — broadcasted absolute raw positions of each
        // query slot. The PR #1195 Python builds (B, S) by broadcasting
        // (1, S) to (B, S); we do the same.
        let qPos =
            MLXArray(Int32(offset)..<Int32(offset + S))
            .expandedDimensions(axis: 0)        // (1, S)
            .reshaped(1, S)
        // raw_pos_at_k: (windowLen,) → (1, 1, windowLen)
        let cacheK = MLXArray(Int32(0)..<Int32(windowLen))
        let rawPosAtK = MLXArray(Int32((offset + S) - windowLen)) + cacheK
        // qPos: (1, S) → (1, S, 1) then broadcast against (1, 1, windowLen)
        let qPos3 = qPos.expandedDimensions(axis: -1)
        let raw3 = rawPosAtK.expandedDimensions(axes: [0, 1])
        let lower = raw3 .> (qPos3 - MLXArray(Int32(window)))
        let upper = raw3 .<= qPos3
        let visible = MLX.logicalAnd(lower, upper)
        // (1, S, windowLen) → broadcast to (B, 1, S, windowLen)
        let v4 = visible.expandedDimensions(axis: 1)
        let bArr = MLX.broadcast(v4, to: [B, 1, S, windowLen])
        return bArr
    }

    /// Per-query causal visibility into the compressor's pooled
    /// chunks. Chunk `k` covers raw positions `[k*ratio, (k+1)*ratio)`
    /// and is visible to query `q` once that whole chunk has been
    /// observed: `(k+1)*ratio <= q+1`.
    /// Returns shape `(B, 1, S, compressedLen)`.
    public static func compressedVisibility(
        batch B: Int, queryLen S: Int,
        offset: Int, compressedLen: Int, ratio: Int
    ) -> MLXArray {
        let qPos =
            MLXArray(Int32(offset)..<Int32(offset + S))
            .expandedDimensions(axis: 0)
            .reshaped(1, S)
        let k = MLXArray(Int32(0)..<Int32(compressedLen))
        // (k+1) * ratio <= (qPos + 1)
        let lhs =
            (k + MLXArray(Int32(1))) * MLXArray(Int32(ratio))
        let rhs = qPos + MLXArray(Int32(1))
        // lhs: (compressedLen,) → (1, 1, compressedLen)
        let lhs3 = lhs.expandedDimensions(axes: [0, 1])
        // rhs: (1, S) → (1, S, 1)
        let rhs3 = rhs.expandedDimensions(axis: -1)
        let visible = lhs3 .<= rhs3
        let v4 = visible.expandedDimensions(axis: 1)
        return MLX.broadcast(v4, to: [B, 1, S, compressedLen])
    }

    /// Apply DSV4's block-causal compressed-pool visibility before
    /// HSA indexer top-k selection. The later SDPA mask is still the
    /// final authority, but masking here prevents argpartition from
    /// spending the whole top-k budget on future chunks that are later
    /// filtered out.
    ///
    /// `scores` shape: `(B, S, compressedLen)`.
    public static func causalMaskedIndexerScores(
        _ scores: MLXArray, offset: Int, ratio: Int
    ) -> MLXArray {
        let B = scores.dim(0)
        let S = scores.dim(1)
        let compressedLen = scores.dim(2)
        guard S > 1, compressedLen > 0 else { return scores }

        let visible = compressedVisibility(
            batch: B, queryLen: S, offset: offset,
            compressedLen: compressedLen, ratio: ratio
        ).squeezed(axis: 1)
        let negLarge = MLXArray(Float(-1.0e30), dtype: scores.dtype)
        return MLX.where(visible, scores, negLarge)
    }

    /// AND the per-query indexer top-k selection onto a compressed
    /// visibility mask. `topk` is the indexer's `(B, S, K)` int array
    /// of selected chunk indices; returns `(B, 1, S, compressedLen)`
    /// bool — true only when chunk index `c` appears in `topk[b, s, :]`.
    public static func indexerSelectionMask(
        topk: MLXArray, compressedLen: Int
    ) -> MLXArray {
        // topk: (B, S, K) → (B, S, K, 1)
        let topk4 = topk.expandedDimensions(axis: -1)
        // k_range: (compressedLen,) → (1, 1, 1, compressedLen)
        let kRange =
            MLXArray(Int32(0)..<Int32(compressedLen))
            .expandedDimensions(axes: [0, 1, 2])
        // (B, S, K, compressedLen) → (B, S, compressedLen) via any over K
        let eq = topk4 .== kRange
        let selected = eq.any(axis: -2)  // (B, S, compressedLen)
        return selected.expandedDimensions(axis: 1)  // (B, 1, S, compressedLen)
    }

    // MARK: - Tiled pool attention (Python `_dsv4_tiled_pool_attention` port)
    //
    // Once a pool outgrows its BF16 hot tier (segmented q8) or exceeds the
    // prefill dense-row cap, materializing the whole pool per layer per call
    // makes each round of prefill slower than the last. These helpers keep
    // exactly one bounded dequantized tile and its fp32 score matrix alive
    // at a time, folding tiles into an online softmax with a non-blocking
    // `asyncEval` bound between tiles.

    static let poolTileTargetBytes = 128 * 1024 * 1024
    static let poolTileMaxRows = 64 * 1024
    static let poolTileMinRows = 64

    /// Choose a bounded pool row tile from query-score and BF16-view
    /// geometry (Python `_dsv4_pool_tile_rows`).
    static func poolTileRows(queries q: MLXArray, valueDim: Int) -> Int {
        let B = q.dim(0)
        let H = q.dim(1)
        let S = q.dim(2)
        let scoreBytesPerRow = B * H * S * 4
        let valueBytesPerRow = B * valueDim * 2
        let maskBytesPerRow = B * S
        let bytesPerRow = max(
            1, scoreBytesPerRow + valueBytesPerRow + maskBytesPerRow)
        var rows = poolTileTargetBytes / bytesPerRow
        rows = min(poolTileMaxRows, max(poolTileMinRows, rows))
        if rows >= poolTileMinRows {
            rows = (rows / poolTileMinRows) * poolTileMinRows
        }
        return max(1, rows)
    }

    /// Rows eligible for indexer top-k before ranking: row `rowStart + i`
    /// becomes causal only after all `ratio` source tokens in that row have
    /// been observed. Returns `(B, S, rowCount)` bool.
    static func indexVisibility(
        batch B: Int, seqLen S: Int, offset: Int,
        rowStart: Int, rowCount: Int, ratio: Int
    ) -> MLXArray {
        if rowCount <= 0 {
            return MLXArray.zeros([B, S, 0], dtype: .bool)
        }
        let qPos = MLXArray(Int32(offset)..<Int32(offset + S))
        let rowIdx = MLXArray(Int32(rowStart)..<Int32(rowStart + rowCount))
        let lhs = (rowIdx + MLXArray(Int32(1))) * MLXArray(Int32(ratio))
        let rhs = qPos + MLXArray(Int32(1))
        // (1, rowCount) <= (S, 1) → (S, rowCount)
        let visible =
            lhs.expandedDimensions(axis: 0) .<= rhs.expandedDimensions(axis: -1)
        return broadcast(visible.expandedDimensions(axis: 0), to: [B, S, rowCount])
    }

    /// Exact global top-k over a segmented (or oversized-BF16) indexer pool
    /// (Python `_dsv4_tiled_index_topk`). Candidate scores are reduced to
    /// the running global top-k before the next tile is consumed.
    static func tiledIndexTopk(
        queries q: MLXArray,
        headWeights: MLXArray,
        pooled: DeepseekV4PoolView,
        scale: Float,
        topK: Int,
        offset: Int,
        ratio: Int
    ) -> MLXArray {
        let totalRows = pooled.rowCount
        let k = min(max(1, topK), totalRows)
        let tileRows = poolTileRows(queries: q, valueDim: pooled.featureDim)
        let q32 = q.asType(.float32)
        // (B, L, H) → (B, H, L, 1)
        let weights = headWeights.swappedAxes(-1, -2)
            .expandedDimensions(axis: -1).asType(.float32)
        var bestScores: MLXArray? = nil
        var bestIndices: MLXArray? = nil
        pooled.forEachDequantizedTile(maxRows: tileRows) { start, tile in
            if let bs = bestScores, let bi = bestIndices {
                // Bound the lazy graph across tiles WITHOUT a GPU
                // round-trip stall: a blocking eval here serialized decode
                // (one stall per layer per token) once pools promoted to q8.
                asyncEval(bs, bi)
            }
            let tile32 = tile.asType(.float32)
            var scores = q32.matmul(
                tile32.expandedDimensions(axis: 1).swappedAxes(-1, -2))
            scores = maximum(scores, MLXArray(Float(0))) * MLXArray(scale)
            var reduced = (scores * weights).sum(axis: 1)  // (B, S, rows)
            let rows = tile.dim(1)
            let visible = indexVisibility(
                batch: reduced.dim(0), seqLen: reduced.dim(1), offset: offset,
                rowStart: start, rowCount: rows, ratio: ratio)
            let negInf = MLXArray(-Float.infinity, dtype: reduced.dtype)
            reduced = MLX.where(visible, reduced, negInf)
            var indices = broadcast(
                MLXArray(Int32(start)..<Int32(start + rows))
                    .expandedDimensions(axes: [0, 1]),
                to: reduced.shape)
            if let bs = bestScores, let bi = bestIndices {
                reduced = concatenated([bs, reduced], axis: -1)
                indices = concatenated([bi, indices], axis: -1)
            }
            let keep = min(k, reduced.dim(-1))
            if reduced.dim(-1) > keep {
                let selected = argPartition(-reduced, kth: keep - 1, axis: -1)[
                    .ellipsis, 0..<keep]
                bestScores = takeAlong(reduced, selected, axis: -1)
                bestIndices = takeAlong(indices, selected, axis: -1)
            } else {
                bestScores = reduced
                bestIndices = indices
            }
        }
        guard let best = bestIndices else {
            return MLXArray.zeros([q.dim(0), q.dim(2), 0], dtype: .int32)
        }
        return best.asType(.int32)
    }

    /// Online-softmax accumulator seeded from attention sinks: the sink
    /// logit is the initial running max and contributes weight 1 with a
    /// zero value (Python parity). Without sinks (tiny test configs) the
    /// accumulator starts empty.
    static func initPoolAccumulator(
        sinks: MLXArray?, batch B: Int, heads H: Int, seqLen S: Int,
        headDim D: Int
    ) -> (max: MLXArray, sum: MLXArray, value: MLXArray) {
        let runningMax: MLXArray
        let runningSum: MLXArray
        if let sinks {
            runningMax = broadcast(
                sinks.asType(.float32).reshaped(1, H, 1), to: [B, H, S])
            runningSum = MLXArray.ones([B, H, S], dtype: .float32)
        } else {
            runningMax = broadcast(MLXArray(Float(-1.0e30)), to: [B, H, S])
            runningSum = MLXArray.zeros([B, H, S], dtype: .float32)
        }
        let runningValue = MLXArray.zeros([B, H, S, D], dtype: .float32)
        return (runningMax, runningSum, runningValue)
    }

    /// Merge one `(B, R, D)` key/value tile into the online softmax
    /// accumulator (Python `_dsv4_attention_accumulate`). DSV4 pool rows
    /// and window slots serve as both keys and values.
    static func poolAttentionAccumulate(
        q32: MLXArray, kv: MLXArray, mask: MLXArray, scale: Float,
        runningMax: MLXArray, runningSum: MLXArray, runningValue: MLXArray
    ) -> (max: MLXArray, sum: MLXArray, value: MLXArray) {
        let kv32 = kv.expandedDimensions(axis: 1).asType(.float32)  // (B,1,R,D)
        var scores = (q32 * MLXArray(scale)).matmul(kv32.swappedAxes(-1, -2))
        scores = MLX.where(mask, scores, MLXArray(-Float.infinity))
        let tileMax = scores.max(axis: -1)
        let nextMax = maximum(runningMax, tileMax)
        let priorScale = exp(runningMax - nextMax)
        let tileWeights = exp(scores - nextMax.expandedDimensions(axis: -1))
        let nextSum = runningSum * priorScale + tileWeights.sum(axis: -1)
        let nextValue =
            runningValue * priorScale.expandedDimensions(axis: -1)
            + tileWeights.matmul(kv32)
        return (nextMax, nextSum, nextValue)
    }

    /// Merge per-query CSA-selected values `(B, Q, K, D)` into the online
    /// softmax state (Python `_dsv4_selected_attention_accumulate`).
    static func selectedPoolAttentionAccumulate(
        q32: MLXArray, selectedKV: MLXArray, mask: MLXArray, scale: Float,
        runningMax: MLXArray, runningSum: MLXArray, runningValue: MLXArray
    ) -> (max: MLXArray, sum: MLXArray, value: MLXArray) {
        let kv32 = selectedKV.asType(.float32)
        var scores = einsum("bhqd,bqkd->bhqk", q32 * MLXArray(scale), kv32)
        scores = MLX.where(mask, scores, MLXArray(-Float.infinity))
        let tileMax = scores.max(axis: -1)
        let nextMax = maximum(runningMax, tileMax)
        let priorScale = exp(runningMax - nextMax)
        let tileWeights = exp(scores - nextMax.expandedDimensions(axis: -1))
        let nextSum = runningSum * priorScale + tileWeights.sum(axis: -1)
        let nextValue =
            runningValue * priorScale.expandedDimensions(axis: -1)
            + einsum("bhqk,bqkd->bhqd", tileWeights, kv32)
        return (nextMax, nextSum, nextValue)
    }

    /// Bound selected-value and score temporaries during multi-token
    /// prefill (Python `_dsv4_selected_query_rows`): per query, selected
    /// BF16 values, fp32 per-head scores/masks, and the fp32 output
    /// accumulator must stay within the 128 MiB tile budget.
    static func selectedQueryRows(queries q: MLXArray, selectedRows: Int) -> Int {
        let B = q.dim(0)
        let H = q.dim(1)
        let S = q.dim(2)
        let D = q.dim(3)
        let bytesPerQuery = B * (selectedRows * (D * 2 + H * 4 + 1) + H * D * 4)
        return max(1, min(S, poolTileTargetBytes / max(1, bytesPerQuery)))
    }

    /// Attend to the CSA top-k by decoding exactly the selected q8 rows
    /// (Python `_dsv4_selected_pool_attention`). The 512-dim compressor
    /// pool is never scanned again after indexer selection: each query
    /// tile gathers only its selected row occurrences and performs the
    /// local+selected online softmax.
    private static func selectedPoolAttention(
        queries q: MLXArray,
        localKV: MLXArray,
        pooled: DeepseekV4PoolView,
        offset: Int,
        window: Int,
        ratio: Int,
        scale: Float,
        sinks: MLXArray?,
        topK: MLXArray
    ) -> MLXArray {
        let B = q.dim(0)
        let H = q.dim(1)
        let S = q.dim(2)
        let D = q.dim(3)
        let localRows = localKV.dim(2)
        let localMask = buildWindowMask(
            batch: B, queryLen: S, offset: offset,
            window: window, windowLen: localRows)
        let localFlat = localKV.squeezed(axis: 1)
        let queryRows = selectedQueryRows(queries: q, selectedRows: topK.dim(-1))
        var outputs: [MLXArray] = []
        var start = 0
        while start < S {
            if let last = outputs.last {
                // Materialize the previous query tile without blocking
                // (memory bound for long prefill; decode has exactly one
                // tile and pays nothing).
                asyncEval(last)
            }
            let end = min(S, start + queryRows)
            let tileLen = end - start
            let qTile = q[0..., 0..., start..<end]
            let q32 = qTile.asType(.float32)
            let selectedIndices = topK[0..., start..<end].asType(.int32)
            let selectedKV = pooled.gatherDequantizedRows(selectedIndices)
            var (runningMax, runningSum, runningValue) = initPoolAccumulator(
                sinks: sinks, batch: B, heads: H, seqLen: tileLen, headDim: D)
            if localRows > 0 {
                (runningMax, runningSum, runningValue) = poolAttentionAccumulate(
                    q32: q32, kv: localFlat,
                    mask: localMask[0..., 0..., start..<end],
                    scale: scale,
                    runningMax: runningMax, runningSum: runningSum,
                    runningValue: runningValue)
            }
            let qPos = MLXArray(Int32(offset + start)..<Int32(offset + end))
                .reshaped(1, tileLen, 1)
            let visible =
                ((selectedIndices + MLXArray(Int32(1))) * MLXArray(Int32(ratio)))
                .<= (qPos + MLXArray(Int32(1)))
            (runningMax, runningSum, runningValue) =
                selectedPoolAttentionAccumulate(
                    q32: q32, selectedKV: selectedKV,
                    mask: visible.expandedDimensions(axis: 1),
                    scale: scale,
                    runningMax: runningMax, runningSum: runningSum,
                    runningValue: runningValue)
            outputs.append(
                (runningValue / runningSum.expandedDimensions(axis: -1))
                    .asType(q.dtype))
            start = end
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 2)
    }

    /// DSV4 local + compressed attention without a full BF16 pool view
    /// (Python `_dsv4_tiled_pool_attention`). Replaces MLXFast SDPA on
    /// pool-backed layers: sinks fold into the initial accumulator, the
    /// local window accumulates first, then bounded dequantized pool tiles.
    static func tiledPoolAttention(
        queries q: MLXArray,
        localKV: MLXArray,
        pooled: DeepseekV4PoolView,
        offset: Int,
        window: Int,
        ratio: Int,
        scale: Float,
        sinks: MLXArray?,
        topK: MLXArray?
    ) -> MLXArray {
        if let topK {
            return selectedPoolAttention(
                queries: q, localKV: localKV, pooled: pooled,
                offset: offset, window: window, ratio: ratio,
                scale: scale, sinks: sinks, topK: topK)
        }
        let B = q.dim(0)
        let H = q.dim(1)
        let S = q.dim(2)
        let D = q.dim(3)
        let q32 = q.asType(.float32)
        var (runningMax, runningSum, runningValue) = initPoolAccumulator(
            sinks: sinks, batch: B, heads: H, seqLen: S, headDim: D)
        let localRows = localKV.dim(2)
        if localRows > 0 {
            let localMask = buildWindowMask(
                batch: B, queryLen: S, offset: offset,
                window: window, windowLen: localRows)
            (runningMax, runningSum, runningValue) = poolAttentionAccumulate(
                q32: q32, kv: localKV.squeezed(axis: 1), mask: localMask,
                scale: scale,
                runningMax: runningMax, runningSum: runningSum,
                runningValue: runningValue)
        }
        let tileRows = poolTileRows(queries: q, valueDim: pooled.featureDim)
        var firstTile = true
        pooled.forEachDequantizedTile(maxRows: tileRows) { start, tile in
            if !firstTile {
                // Non-blocking graph bound between tiles (see
                // `tiledIndexTopk`).
                asyncEval(runningMax, runningSum, runningValue)
            }
            firstTile = false
            let poolMask = indexVisibility(
                batch: B, seqLen: S, offset: offset,
                rowStart: start, rowCount: tile.dim(1), ratio: ratio)
                .expandedDimensions(axis: 1)
            (runningMax, runningSum, runningValue) = poolAttentionAccumulate(
                q32: q32, kv: tile, mask: poolMask, scale: scale,
                runningMax: runningMax, runningSum: runningSum,
                runningValue: runningValue)
        }
        return (runningValue / runningSum.expandedDimensions(axis: -1))
            .asType(q.dtype)
    }

    // MARK: - Heads-16 indexed prefill attention

    // Python `indexed_prefill_attention.py` parity: replaces the CSA prefill
    // dense-membership-mask branch (topk-selected pool rows) with one Metal
    // kernel that touches at most `window + topk` rows per query regardless
    // of context length. One 256-thread threadgroup per (query token,
    // 16-head group); 8 simdgroups each own heads h0 = g*16 + sg and
    // h1 = h0 + 8; each KV row is staged once into threadgroup memory and
    // consumed by all 16 heads (valid because DSV4 is MQA — K == V). fp32
    // online softmax with the attention sink folded into the initial state
    // (M = sink, S = 1, O = 0). Topk indices are unsorted; invisible rows
    // are skipped with `continue` — the branch is threadgroup-uniform (idx
    // depends only on the token), so the staging barriers stay convergent.
    // Dynamic dims arrive via an int32 params buffer so pool-shape math
    // never enters the compiled source: one compile total.
    private static let heads16PrefillKernel = MLXFast.metalKernel(
        name: "vmlx_dsv4_heads16_prefill",
        inputNames: ["q", "kv", "pool", "topk", "sinks", "fscale", "params"],
        outputNames: ["out"],
        source: """
            uint t = threadgroup_position_in_grid.x;   // query token in [0, S)
            uint g = threadgroup_position_in_grid.y;   // 16-head group in [0, H/16)
            uint tid = thread_position_in_threadgroup.x;
            uint sg = simdgroup_index_in_threadgroup;  // 0..7
            uint lane = thread_index_in_simdgroup;     // 0..31

            const int S = params[0];
            const int R = params[1];
            const int P = params[2];
            const int K = params[3];
            const int offset = params[4];
            const int window = params[5];
            const int ratio = params[6];
            const float scale = fscale[0];

            const int qpos = offset + int(t);
            const int h0 = int(g) * 16 + int(sg);
            const int h1 = h0 + 8;

            // Per-lane query fragments: unit u covers elements [4*lane + 128*u, +4).
            const device T* q0p = q + ((size_t)h0 * (size_t)S + (size_t)t) * 512;
            const device T* q1p = q + ((size_t)h1 * (size_t)S + (size_t)t) * 512;
            float4 q0[4];
            float4 q1[4];
            for (int u = 0; u < 4; ++u) {
                int e = 4 * int(lane) + 128 * u;
                q0[u] = float4(float(q0p[e]), float(q0p[e + 1]),
                               float(q0p[e + 2]), float(q0p[e + 3]));
                q1[u] = float4(float(q1p[e]), float(q1p[e + 1]),
                               float(q1p[e + 2]), float(q1p[e + 3]));
            }

            // Online softmax state, sink folded in: M = sink, S = 1, O = 0.
            float M0 = sinks[h0];
            float M1 = sinks[h1];
            float S0 = 1.0f;
            float S1 = 1.0f;
            float4 O0[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
            float4 O1[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};

            threadgroup float4 kv_shared[128];

            // ---- local sliding-window rows -------------------------------------
            // Row j sits at absolute position (offset + S) - R + j; visible iff
            // pos <= qpos && pos > qpos - window  =>  contiguous j range.
            int j_lo = int(t) + R - S - (window - 1);
            if (j_lo < 0) j_lo = 0;
            int j_hi = int(t) + R - S;
            if (j_hi > R - 1) j_hi = R - 1;
            for (int j = j_lo; j <= j_hi; ++j) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < 128) {
                    const device T* src = kv + (size_t)j * 512 + (size_t)tid * 4;
                    kv_shared[tid] = float4(float(src[0]), float(src[1]),
                                            float(src[2]), float(src[3]));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float4 k0 = kv_shared[lane];
                float4 k1 = kv_shared[lane + 32];
                float4 k2 = kv_shared[lane + 64];
                float4 k3 = kv_shared[lane + 96];
                float s0 = dot(q0[0], k0) + dot(q0[1], k1)
                         + dot(q0[2], k2) + dot(q0[3], k3);
                float s1 = dot(q1[0], k0) + dot(q1[1], k1)
                         + dot(q1[2], k2) + dot(q1[3], k3);
                s0 = simd_sum(s0) * scale;
                s1 = simd_sum(s1) * scale;
                {
                    float mn = max(M0, s0);
                    float c = exp(M0 - mn);
                    float p = exp(s0 - mn);
                    S0 = S0 * c + p;
                    O0[0] = O0[0] * c + p * k0;
                    O0[1] = O0[1] * c + p * k1;
                    O0[2] = O0[2] * c + p * k2;
                    O0[3] = O0[3] * c + p * k3;
                    M0 = mn;
                }
                {
                    float mn = max(M1, s1);
                    float c = exp(M1 - mn);
                    float p = exp(s1 - mn);
                    S1 = S1 * c + p;
                    O1[0] = O1[0] * c + p * k0;
                    O1[1] = O1[1] * c + p * k1;
                    O1[2] = O1[2] * c + p * k2;
                    O1[3] = O1[3] * c + p * k3;
                    M1 = mn;
                }
            }

            // ---- indexer-selected compressed pool rows -------------------------
            const device int* trow = topk + (size_t)t * (size_t)K;
            for (int i = 0; i < K; ++i) {
                int idx = trow[i];
                // Threadgroup-uniform skip (idx depends only on t): argpartition
                // fills short visible sets with causally invisible rows; drop them.
                if (idx < 0 || idx >= P) continue;
                if ((idx + 1) * ratio > qpos + 1) continue;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < 128) {
                    const device T* src = pool + (size_t)idx * 512 + (size_t)tid * 4;
                    kv_shared[tid] = float4(float(src[0]), float(src[1]),
                                            float(src[2]), float(src[3]));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float4 k0 = kv_shared[lane];
                float4 k1 = kv_shared[lane + 32];
                float4 k2 = kv_shared[lane + 64];
                float4 k3 = kv_shared[lane + 96];
                float s0 = dot(q0[0], k0) + dot(q0[1], k1)
                         + dot(q0[2], k2) + dot(q0[3], k3);
                float s1 = dot(q1[0], k0) + dot(q1[1], k1)
                         + dot(q1[2], k2) + dot(q1[3], k3);
                s0 = simd_sum(s0) * scale;
                s1 = simd_sum(s1) * scale;
                {
                    float mn = max(M0, s0);
                    float c = exp(M0 - mn);
                    float p = exp(s0 - mn);
                    S0 = S0 * c + p;
                    O0[0] = O0[0] * c + p * k0;
                    O0[1] = O0[1] * c + p * k1;
                    O0[2] = O0[2] * c + p * k2;
                    O0[3] = O0[3] * c + p * k3;
                    M0 = mn;
                }
                {
                    float mn = max(M1, s1);
                    float c = exp(M1 - mn);
                    float p = exp(s1 - mn);
                    S1 = S1 * c + p;
                    O1[0] = O1[0] * c + p * k0;
                    O1[1] = O1[1] * c + p * k1;
                    O1[2] = O1[2] * c + p * k2;
                    O1[3] = O1[3] * c + p * k3;
                    M1 = mn;
                }
            }

            // ---- normalize + write ---------------------------------------------
            device T* o0p = out + ((size_t)h0 * (size_t)S + (size_t)t) * 512;
            device T* o1p = out + ((size_t)h1 * (size_t)S + (size_t)t) * 512;
            float inv0 = 1.0f / S0;
            float inv1 = 1.0f / S1;
            for (int u = 0; u < 4; ++u) {
                int e = 4 * int(lane) + 128 * u;
                float4 v0 = O0[u] * inv0;
                float4 v1 = O1[u] * inv1;
                o0p[e] = T(v0.x);
                o0p[e + 1] = T(v0.y);
                o0p[e + 2] = T(v0.z);
                o0p[e + 3] = T(v0.w);
                o1p[e] = T(v1.x);
                o1p[e + 1] = T(v1.y);
                o1p[e + 2] = T(v1.z);
                o1p[e + 3] = T(v1.w);
            }
        """)

    /// `VMLX_DSV4_HEADS16_PREFILL` gate (default ON). Cached once — per-call
    /// `ProcessInfo.environment` reads rebuild the entire dictionary.
    public static let heads16PrefillEnvEnabled: Bool = {
        guard
            let raw = ProcessInfo.processInfo.environment[
                "VMLX_DSV4_HEADS16_PREFILL"]?
                .trimmingCharacters(in: .whitespaces).lowercased()
        else { return true }
        return ["1", "on", "true", "yes"].contains(raw)
    }()

    nonisolated(unsafe) private static var heads16Disabled = false
    nonisolated(unsafe) private static var heads16SelfTested = false

    static func heads16RunKernel(
        q: MLXArray, kv2d: MLXArray, pool2d: MLXArray,
        topk2d: MLXArray, sinks32: MLXArray,
        offset: Int, window: Int, ratio: Int, scale: Float
    ) -> MLXArray {
        let heads = q.dim(1)
        let seqLen = q.dim(2)
        let headDim = q.dim(3)
        let params = MLXArray(
            [
                Int32(seqLen), Int32(kv2d.dim(0)), Int32(pool2d.dim(0)),
                Int32(topk2d.dim(1)), Int32(offset), Int32(window),
                Int32(ratio),
            ])
        let fscale = MLXArray([scale])
        return heads16PrefillKernel(
            [q, kv2d, pool2d, topk2d, sinks32, fscale, params],
            template: [("T", q.dtype)],
            grid: (256 * seqLen, heads / 16, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[1, heads, seqLen, headDim]],
            outputDTypes: [q.dtype])[0]
    }

    /// fp32 replica of the stock dense-mask branch (mask + softmax + sink).
    /// Used by the self-test and by parity tests; never on the hot path.
    static func heads16Reference(
        q: MLXArray, kv2d: MLXArray, pool2d: MLXArray,
        topk2d: MLXArray, sinks32: MLXArray,
        offset: Int, window: Int, ratio: Int, scale: Float
    ) -> MLXArray {
        let heads = q.dim(1)
        let seqLen = q.dim(2)
        let rows = kv2d.dim(0)
        let poolRows = pool2d.dim(0)
        let q32 = q.asType(.float32)
        let keys = concatenated(
            [kv2d.asType(.float32), pool2d.asType(.float32)], axis: 0)
        var scores =
            q32.matmul(keys.transposed().expandedDimensions(axes: [0, 1]))
            * MLXArray(scale)

        let qPos = MLXArray(Int32(offset)..<Int32(offset + seqLen))
            .reshaped(seqLen, 1)
        let kPos = MLXArray(
            Int32(offset + seqLen - rows)..<Int32(offset + seqLen)
        ).reshaped(1, rows)
        let localVis = MLX.logicalAnd(
            kPos .<= qPos, kPos .> (qPos - MLXArray(Int32(window))))
        let kIdx = MLXArray(Int32(0)..<Int32(poolRows))
        var compVis =
            ((kIdx.reshaped(1, poolRows) + MLXArray(Int32(1)))
                * MLXArray(Int32(ratio))) .<= (qPos + MLXArray(Int32(1)))
        let selected = (topk2d.expandedDimensions(axis: 2)
            .== kIdx.reshaped(1, 1, poolRows)).any(axis: 1)
        compVis = MLX.logicalAnd(compVis, selected)
        let mask = concatenated([localVis, compVis], axis: -1)
            .expandedDimensions(axes: [0, 1])

        scores = MLX.where(mask, scores, MLXArray(-Float.infinity))
        let sink = sinks32.reshaped(1, heads, 1, 1)
        let m = maximum(scores.max(axis: -1, keepDims: true), sink)
        let p = exp(scores - m)
        let denom = p.sum(axis: -1, keepDims: true) + exp(sink - m)
        return (p.matmul(keys.expandedDimensions(axes: [0, 1])) / denom)
            .asType(q.dtype)
    }

    private static func heads16TestValues(
        _ count: Int, frequency: Float, phase: Float, amplitude: Float
    ) -> MLXArray {
        let idx = MLXArray(Int32(0)..<Int32(count)).asType(.float32)
        return sin(idx * MLXArray(frequency) + MLXArray(phase))
            * MLXArray(amplitude)
    }

    /// Kernel vs fp32-reference check on a small deterministic shape,
    /// mirroring Python `_self_test` (seq 8, rows 16, pool 32, k 8) with
    /// sin-derived values so the global RNG state is untouched. Topk rows
    /// are DISTINCT per query (rotations of a fixed permutation): the
    /// kernel accumulates a duplicated row twice while the reference
    /// membership mask counts it once — production argPartition indices
    /// are distinct by construction, so duplicates are out of contract.
    static func heads16SelfTest() -> String? {
        let seqLen = 8
        let rows = 16
        let poolRows = 32
        let k = 8
        let offset = 100
        let window = 12
        let ratio = 4
        let scale = Float(pow(Double(512), -0.5))
        let basePerm: [Int32] = [
            7, 19, 3, 28, 11, 0, 24, 15, 30, 5, 22, 9, 17, 2, 26, 13,
            31, 6, 20, 1, 25, 10, 29, 4, 18, 12, 27, 8, 21, 14, 23, 16,
        ]
        var topkValues = [Int32]()
        for t in 0..<seqLen {
            for i in 0..<k {
                topkValues.append(basePerm[(t * 5 + i) % basePerm.count])
            }
        }
        let topk2d = MLXArray(topkValues).reshaped(seqLen, k)
        let sinks32 = heads16TestValues(
            64, frequency: 0.83, phase: 1.7, amplitude: 0.5)
        for dtype in [DType.bfloat16, DType.float16] {
            let q = heads16TestValues(
                64 * seqLen * 512, frequency: 0.37, phase: 0.1, amplitude: 0.3
            ).reshaped(1, 64, seqLen, 512).asType(dtype)
            let kv2d = heads16TestValues(
                rows * 512, frequency: 0.53, phase: 0.9, amplitude: 0.3
            ).reshaped(rows, 512).asType(dtype)
            let pool2d = heads16TestValues(
                poolRows * 512, frequency: 0.71, phase: 2.3, amplitude: 0.3
            ).reshaped(poolRows, 512).asType(dtype)
            let got = heads16RunKernel(
                q: q, kv2d: kv2d, pool2d: pool2d, topk2d: topk2d,
                sinks32: sinks32,
                offset: offset, window: window, ratio: ratio, scale: scale
            ).asType(.float32)
            let ref = heads16Reference(
                q: q, kv2d: kv2d, pool2d: pool2d, topk2d: topk2d,
                sinks32: sinks32,
                offset: offset, window: window, ratio: ratio, scale: scale
            ).asType(.float32)
            // MLX graph materialization (not code eval).
            MLX.eval(got, ref)
            let denom = max(abs(ref).max().item(Float.self), Float(1e-6))
            let rel = abs(got - ref).max().item(Float.self) / denom
            if !rel.isFinite || rel > 2.5e-2 {
                return "self-test rel diff \(rel) > 2.5e-2 (\(dtype))"
            }
        }
        return nil
    }

    /// Kernel path for the CSA prefill topk branch; `nil` → caller uses the
    /// stock dense-mask SDPA. Layout contract: B == 1, H == 64, D == 512,
    /// dtype f16/bf16, dense pool `(B, W, D)`, topk `(1, S, K)`. Returns
    /// `(1, H, S, D)` matching stock SDPA output. First enabled use runs a
    /// live self-test against fp32 reference math; on any failure the kernel
    /// permanently disables itself for this process.
    static func heads16PrefillAttention(
        queries q: MLXArray,
        localKV: MLXArray,
        pooled: MLXArray,
        topK: MLXArray,
        offset: Int,
        window: Int,
        ratio: Int,
        scale: Float,
        sinks: MLXArray?
    ) -> MLXArray? {
        if heads16Disabled || !heads16PrefillEnvEnabled { return nil }
        guard let sinks, ratio > 0 else { return nil }
        let B = q.dim(0)
        let H = q.dim(1)
        let S = q.dim(2)
        let D = q.dim(3)
        guard B == 1, H == 64, D == 512, S > 0 else { return nil }
        guard q.dtype == .float16 || q.dtype == .bfloat16 else { return nil }
        guard topK.ndim == 3, topK.dim(0) == 1, topK.dim(1) == S,
            topK.dim(2) > 0
        else { return nil }
        let rows = localKV.dim(2)
        let poolRows = pooled.dim(1)
        guard rows > 0, poolRows > 0 else { return nil }

        if !heads16SelfTested {
            heads16SelfTested = true
            if let err = heads16SelfTest() {
                heads16Disabled = true
                FileHandle.standardError.write(
                    Data(
                        "DSV4 heads16 indexed-prefill kernel self-test FAILED (\(err)); using stock indexed prefill for this process\n"
                            .utf8))
                return nil
            }
        }

        let kv2d = localKV.reshaped(rows, D)
        let pool2d = pooled.reshaped(poolRows, D).asType(q.dtype)
        let topk2d = topK.reshaped(S, topK.dim(2)).asType(.int32)
        return heads16RunKernel(
            q: q, kv2d: kv2d, pool2d: pool2d, topk2d: topk2d,
            sinks32: sinks.asType(.float32),
            offset: offset, window: window, ratio: ratio, scale: scale)
    }
}
