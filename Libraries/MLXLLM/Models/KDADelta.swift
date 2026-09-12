// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// KDA (Kimi Delta Attention) recurrence — the linear-attention path of
// BailingMoeV3 / Ling 3.0. Structurally the GatedDelta recurrence with one
// difference that forces a separate kernel: the decay gate is PER K-CHANNEL
// ([B, T, H, Dk]), not per head ([B, T, H]). The state update per step is
//
//     S     ← S ⊙ g_t            (g_t broadcast over the Dv axis)
//     v'_t  = β_t (v_t − Sᵀ k_t)
//     S     ← S + k_t ⊗ v'_t
//     o_t   = Sᵀ q_t
//
// with g_t already the multiplicative decay (exp applied on the host side).
// Reference: fla `fused_recurrent_kda` (USE_GATE_IN_KERNEL branch) as invoked
// by `modeling_bailing_moe_v3.BailingMoeV3KimiDeltaAttention`.

/// Multiplicative per-channel decay from the raw f-projection.
///
/// `safe_gate` (Ling 3.0 ships `kda_safe_gate = true`): the log-decay is
/// `lower_bound * sigmoid(exp(A_log) * (f + dt_bias))`, bounded to
/// `(lower_bound, 0)`. The unsafe branch is `-exp(A_log) * softplus(f + dt_bias)`.
func computeKDADecay(
    fRaw: MLXArray,  // [B, T, H, Dk] — f_proj output + nothing else
    aLog: MLXArray,  // [H]
    dtBias: MLXArray,  // [H * Dk]
    safeGate: Bool,
    lowerBound: Float
) -> MLXArray {
    let H = aLog.dim(0)
    let dk = fRaw.dim(-1)
    let bias = dtBias.reshaped(H, dk).asType(.float32)
    let x = fRaw.asType(.float32) + bias
    let aExp = exp(aLog.asType(.float32)).reshaped(1, 1, H, 1)
    let logDecay: MLXArray
    if safeGate {
        logDecay = MLXArray(lowerBound) * sigmoid(aExp * x)
    } else {
        logDecay = -aExp * logAddExp(x, MLXArray(Float(0)))  // softplus
    }
    return exp(logDecay)
}

// MARK: - Metal kernel (whole-sequence, one dispatch per layer)

private func makeKDAKernel(hasMask: Bool) -> MLXFast.MLXFastKernel? {
    let maskSource = hasMask ? "mask[b_idx * T + t]" : "true"

    // Identical loop structure to `gated_delta_step`, but `g` is indexed per
    // K channel: g[b, t, hv, dk] with stride Hv * Dk per step.
    let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // g: [B, T, Hv, Dk] — per-channel decay
            auto g_ = g + b_idx * T * Hv * Dk + hv_idx * Dk;
            // beta: [B, T, Hv]
            auto beta_ = beta + b_idx * T * Hv;

            // state_in, state_out: [B, Hv, Dv, Dk]
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
                  state[i] = state[i] * static_cast<float>(g_[s_idx]);
                  kv_mem += state[i] * k_[s_idx];
                }
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

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
              } else {
                y[dv_idx] = static_cast<InT>(0);
              }
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv * Dk;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<StT>(state[i]);
            }
        """

    var inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if hasMask { inputNames.append("mask") }

    return MLXFast.metalKernel(
        name: "kda_delta_step" + (hasMask ? "_mask" : ""),
        inputNames: inputNames,
        outputNames: ["y", "state_out"],
        source: source
    )
}

private final class KDAKernelManager: Sendable {
    static let shared = KDAKernelManager()
    let kernel: MLXFast.MLXFastKernel?
    let kernelMasked: MLXFast.MLXFastKernel?
    private init() {
        kernel = makeKDAKernel(hasMask: false)
        kernelMasked = makeKDAKernel(hasMask: true)
    }
}

private func kdaKernel(
    q: MLXArray, k: MLXArray, v: MLXArray,
    g: MLXArray, beta: MLXArray, state: MLXArray,
    mask: MLXArray?
) -> (MLXArray, MLXArray) {
    let B = k.dim(0)
    let T = k.dim(1)
    let Hk = k.dim(2)
    let Dk = k.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    let selected =
        mask != nil ? KDAKernelManager.shared.kernelMasked : KDAKernelManager.shared.kernel
    guard let kernel = selected else { fatalError("KDA kernel not available") }

    // The decay `g` (exp of the fp32 gate, values in (0, 1]) and `beta` stay
    // float32 into the kernel: narrowed to bf16, any per-channel decay above
    // ~0.998 rounds to exactly 1.0 (no decay) and 0.997 to 0.996, an error
    // the recurrence compounds every token. Upstream FLA computes the gate
    // in fp32 inside its kernel; the Metal source reads both through
    // `static_cast<float>`, so the array dtype is the only thing that
    // decides the precision.
    var inputs: [MLXArray] = [q, k, v, g.asType(.float32), beta.asType(.float32), state, MLXArray(T)]
    if let mask { inputs.append(mask) }

    let outputs = kernel(
        inputs,
        template: [
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: (32, Dv, B * Hv),
        threadGroup: (32, 4, 1),
        outputShapes: [[B, T, Hv, Dv], state.shape],
        outputDTypes: [q.dtype, state.dtype]
    )
    return (outputs[0], outputs[1])
}

// MARK: - Fused decode-step short convolution

/// One kernel for the whole of a KDA decode step's input stage: three depthwise short convolutions,
/// three SiLUs, and the three state shifts.
///
/// The unfused path, per layer per token, is a `concatenated` to prepend the conv state, three
/// `Conv1d`s, three `silu`s and a slice-store to put the state back — roughly eight dispatches and
/// as many intermediate allocations. GLM-5.3 runs KDA in 34 of its 45 layers, so that is ~270
/// dispatches per decoded token where this is 34. The allocations matter as much as the dispatches
/// on a model this size: past MLX's `gc_limit_` every allocation triggers `release_cached_buffers`,
/// so removing seven per layer per step removes them from the steady state too.
///
/// DECODE ONLY, and deliberately so — `B == 1, T == 1`. A prefill has real arithmetic per token and
/// the stock `Conv1d` is the better kernel for it; this exists because a decode step's arithmetic is
/// trivial and its cost is almost all launch and allocation overhead. Every other shape, dtype or
/// geometry returns `nil` and the caller keeps its existing path, so this can only ever be a FAST
/// path, never a semantic one.
///
/// Our conv state is PACKED as `[B, K-1, 3 * proj]` (q, then k, then v), because that is what
/// `BailingV3KDAAttention` already keeps in `cache[0]`; the reference keeps three separate states.
/// Packing costs one integer division per thread to recover which projection a channel belongs to,
/// and saves changing a cache layout that is serialized, snapshotted and restored.
private final class KDAConvKernelManager: Sendable {
    static let shared = KDAConvKernelManager()
    let kernel: MLXFast.MLXFastKernel?

    private init() {
        // `P`, `C3` and `K` arrive as template parameters, so MLX specializes and caches one variant
        // per geometry rather than this file rebuilding a source string per shape.
        let source = """
                uint c = thread_position_in_grid.x;
                if (c >= (uint)C3) return;

                uint lane = c / (uint)P;
                uint ch = c - lane * (uint)P;
                device const T* w = lane == 0u ? q_weight : (lane == 1u ? k_weight : v_weight);

                // Taps 0..K-2 read the carried state; the last tap is this step's own token. That is
                // exactly what a stock conv1d with zero padding computes over `concat(state, token)`,
                // which is what makes this a fast path and not a variant.
                float acc = 0.0f;
                for (uint tap = 0; tap < (uint)K - 1u; ++tap) {
                    acc += (float)state[(size_t)tap * (size_t)C3 + c]
                         * (float)w[(size_t)ch * (size_t)K + tap];
                }
                float cur = (float)token[c];
                acc += cur * (float)w[(size_t)ch * (size_t)K + ((uint)K - 1u)];

                // Shift the window: drop the oldest tap, append this token. Writing `next` from
                // `state` rather than in place keeps the two disjoint, so the caller may pass the
                // live cache buffer.
                for (uint tap = 0; tap + 2u < (uint)K; ++tap) {
                    next[(size_t)tap * (size_t)C3 + c] =
                        state[(size_t)(tap + 1u) * (size_t)C3 + c];
                }
                next[(size_t)((uint)K - 2u) * (size_t)C3 + c] = (T)cur;

                // SiLU, folded in: x * sigmoid(x), written as the division the reference uses.
                out[c] = (T)(acc / (1.0f + metal::exp(-acc)));
            """
        kernel = MLXFast.metalKernel(
            name: "vmlx_glm5_kda_qkv_conv_packed",
            inputNames: ["token", "state", "q_weight", "k_weight", "v_weight"],
            outputNames: ["out", "next"],
            source: source)
    }
}

public enum KDAConvRuntime {
    /// Default ON, matching the reference, with the same shape of explicit rollback.
    nonisolated(unsafe) public static var enabled: Bool = {
        let v = ProcessInfo.processInfo.environment["VMLX_GLM5_FUSED_KDA_CONV"]?
            .trimmingCharacters(in: .whitespaces).lowercased()
        return !(["0", "false", "off", "no", ""].contains(v ?? "1"))
    }()
}

/// Convolve + activate + shift one decode step, or return `nil` meaning "use the stock path".
///
/// Returns `nil` for every shape, dtype or geometry it does not handle EXACTLY, rather than trying
/// to cope: a fast path that silently accepts a case it gets subtly wrong is worse than none, and
/// the caller already holds a correct implementation.
public func glm5KDAConvDecode(
    token: MLXArray,  // [1, 1, 3 * proj] — this step's q|k|v projections, packed
    state: MLXArray,  // [1, kernelSize - 1, 3 * proj]
    qWeight: MLXArray,  // [proj, kernelSize, 1] — Conv1d's depthwise layout
    kWeight: MLXArray,
    vWeight: MLXArray,
    proj: Int,
    kernelSize: Int,
    /// `nil` reads the process-wide default. Callers that hold their own copy pass it explicitly —
    /// which is what the layer does, so the choice is fixed when a module is built rather than
    /// re-read per call. A flag that can change between two calls on the same cache is a race in
    /// waiting, and one already cost a session's debugging on the MLA representation switch.
    enabled: Bool? = nil
) -> (out: MLXArray, next: MLXArray)? {
    guard enabled ?? KDAConvRuntime.enabled, kernelSize >= 2 else { return nil }
    guard let kernel = KDAConvKernelManager.shared.kernel else { return nil }

    let channels = 3 * proj
    guard token.ndim == 3, token.shape == [1, 1, channels],
        state.ndim == 3, state.shape == [1, kernelSize - 1, channels]
    else { return nil }
    // float32 is excluded for the reason the reference excludes it: the production decode path is
    // half precision, and a variant nobody runs is a variant nobody tests.
    guard token.dtype == .float16 || token.dtype == .bfloat16, state.dtype == token.dtype
    else { return nil }
    for w in [qWeight, kWeight, vWeight] {
        guard w.dtype == token.dtype, w.size == proj * kernelSize else { return nil }
    }

    let outputs = kernel(
        [token, state, qWeight, kWeight, vWeight],
        template: [("T", token.dtype), ("P", proj), ("C3", channels), ("K", kernelSize)],
        grid: (channels, 1, 1),
        threadGroup: (min(256, channels), 1, 1),
        outputShapes: [token.shape, state.shape],
        outputDTypes: [token.dtype, token.dtype])
    return (outputs[0], outputs[1])
}

// MARK: - Public entry

/// KDA recurrence over a full segment. `fRaw` is the raw f-projection output
/// (decay computed here); `beta` must already be `sigmoid(b_proj(x))`.
public func kdaUpdate(
    q: MLXArray,  // [B, T, H, Dk] — L2-normalized, scaled
    k: MLXArray,  // [B, T, H, Dk] — L2-normalized
    v: MLXArray,  // [B, T, H, Dv]
    fRaw: MLXArray,  // [B, T, H, Dk]
    beta: MLXArray,  // [B, T, H]
    aLog: MLXArray,  // [H]
    dtBias: MLXArray,  // [H * Dk]
    safeGate: Bool,
    lowerBound: Float,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = q.dim(0)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    let g = computeKDADecay(
        fRaw: fRaw, aLog: aLog, dtBias: dtBias,
        safeGate: safeGate, lowerBound: lowerBound)

    // Recurrent state stays float32 for the same cached-prefix-boundary
    // rounding reason as GatedDelta (see gatedDeltaUpdate).
    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)
    if state.dtype != .float32 { state = state.asType(.float32) }

    // Same SIMD tile constraint as the GatedDelta kernel: Dk must be a
    // multiple of 32 (Ling 3.0 ships Dk = 128). Narrow synthetic configs
    // take the ops fallback, whose per-step decay branch broadcasts a
    // 3-D per-channel gate.
    let manager = KDAKernelManager.shared
    let available = (mask != nil ? manager.kernelMasked : manager.kernel) != nil
    if available && Dk >= 32 && Dk % 32 == 0 {
        return kdaKernel(
            q: q, k: k, v: v, g: g, beta: beta,
            state: state, mask: mask)
    }
    return gatedDeltaOps(
        q: q, k: k, v: v, g: g, beta: beta.asType(.float32),
        state: state, mask: mask)
}

/// The KDA attention module, named for the mechanism rather than the first family to ship it.
///
/// The recurrence above already had a file of its own; the module that feeds it did not, and lived
/// inside `BailingMoeV3`. GLM-5.3 (`glm5_next`) runs the identical mechanism — same
/// `linear_attn_config` quantities, same weight names — so it uses this rather than carrying a
/// second copy that could drift from the kernel it calls.
public typealias KDAAttention = BailingV3KDAAttention
