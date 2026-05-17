//
//  GatedDelta.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Compute G

/// Compiled compute_g — fuses exp+exp+softplus+mul+neg into 1 Metal dispatch.
/// Matches Python's @mx.compile(shapeless=True) def compute_g().
/// Called per GatedDeltaNet layer per token (~12 SSM layers in Qwen 3.5).
///
/// Always compiled, regardless of `HardwareInfo.isCompiledDecodeSupported`. The broad
/// disable of compile() targeted shape-variable prefill graphs that caused trace-cache
/// misses; compute_g operates on tiny [numVHeads] tensors with stable shapes across
/// every decode step, so the trace cache hits 100% of the time and there is no prefill
/// stall to worry about. Python's mlx_lm reference compiles this exact function and
/// gets ~17 tok/s more on Qwen 3.5-35B vs Swift's eager path.
private let _compiledComputeG: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { (aLog: MLXArray, a: MLXArray, dtBias: MLXArray) -> MLXArray in
        let decay = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
        return decay.asType(a.dtype)
    }

func computeGatedDeltaG(_ aLog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray) -> MLXArray {
    _compiledComputeG(aLog, a, dtBias)
}

// MARK: - Metal Kernel

private func makeGatedDeltaKernel(
    hasMask: Bool,
    roundStateEachStep: Bool = false
) -> MLXFast.MLXFastKernel? {
    let maskSource = hasMask ? "mask[b_idx * T + t]" : "true"
    let stepRoundSource = roundStateEachStep
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

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // g: [B, T, Hv]
            auto g_ = g + b_idx * T * Hv;
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
                  state[i] = state[i] * g_[hv_idx];
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
        \(stepRoundSource)
              }
              // Increment data pointers to next time step
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<InT>(state[i]);
            }
        """

    var inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if hasMask {
        inputNames.append("mask")
    }

    let suffix = (hasMask ? "_mask" : "") + (roundStateEachStep ? "_strict" : "_fast")

    return MLXFast.metalKernel(
        name: "gated_delta_step\(suffix)",
        inputNames: inputNames,
        outputNames: ["y", "state_out"],
        source: source
    )
}

private final class GatedDeltaKernelManager: Sendable {
    static let shared = GatedDeltaKernelManager()

    let kernel: MLXFast.MLXFastKernel?
    let kernelMasked: MLXFast.MLXFastKernel?
    let strictKernel: MLXFast.MLXFastKernel?
    let strictKernelMasked: MLXFast.MLXFastKernel?

    private init() {
        kernel = makeGatedDeltaKernel(hasMask: false, roundStateEachStep: false)
        kernelMasked = makeGatedDeltaKernel(hasMask: true, roundStateEachStep: false)
        strictKernel = makeGatedDeltaKernel(hasMask: false, roundStateEachStep: true)
        strictKernelMasked = makeGatedDeltaKernel(hasMask: true, roundStateEachStep: true)
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

// MARK: - Kernel Dispatch

func gatedDeltaKernel(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
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

    let selectedKernel: MLXFast.MLXFastKernel?
    var inputs: [MLXArray] = [q, k, v, g, beta, state, MLXArray(T)]
    if let mask {
        selectedKernel = GatedDeltaKernelManager.shared.kernel(
            hasMask: true,
            roundStateEachStep: roundStateEachStep)
        inputs.append(mask)
    } else {
        selectedKernel = GatedDeltaKernelManager.shared.kernel(
            hasMask: false,
            roundStateEachStep: roundStateEachStep)
    }

    guard let kernel = selectedKernel else {
        fatalError("Gated delta kernel not available")
    }

    let outputs = kernel(
        inputs,
        template: [
            ("InT", inputType),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: (32, Dv, B * Hv),
        threadGroup: (32, 4, 1),
        outputShapes: [[B, T, Hv, Dv], state.shape],
        outputDTypes: [inputType, inputType]
    )

    return (outputs[0], outputs[1])
}

// MARK: - Ops Fallback

/// Raw step implementation — called by compiled wrapper or directly for masked prefill.
private func _rawStepOps(
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

/// Compiled GatedDelta step — fuses ~10 ops (decay, kv_mem, delta, state update, output)
/// into one Metal dispatch. Matches Python's @mx.compile _gated_delta_step_ops.
/// Called per SSM layer per token (~30 layers). Without compile: 300 extra kernel launches/token.
private let _compiledStepOps: @Sendable ([MLXArray]) -> [MLXArray] =
    compile { (args: [MLXArray]) -> [MLXArray] in
        let (y, state) = _rawStepOps(
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
        // Masked path (prefill) — can't compile due to conditional where
        let oldState = state
        let (y, newState) = _rawStepOps(q: q, k: k, v: v, g: g, beta: beta, state: state)
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
        return (y, MLX.where(expandedMask, newState, oldState))
    }

    // Decode path — compiled, fuses ~10 ops into 1 Metal dispatch
    let result = _compiledStepOps([q, k, v, g, beta, state])
    return (result[0], result[1])
}

func gatedDeltaOps(
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

    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: q.dtype)

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
        state = roundStateEachStep ? newState.asType(q.dtype) : newState
    }

    let y = MLX.stacked(ys, axis: 1)
    return (y, state)
}

// MARK: - Public API

func gatedDeltaUpdate(
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
    let beta = sigmoid(b)
    let g = computeGatedDeltaG(aLog, a, dtBias)

    let B = q.dim(0)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    let state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: q.dtype)

    // Mirror `MLXVLM/Models/Qwen35.swift` `gatedDeltaUpdate`:
    // (1) select the masked vs unmasked kernel by `mask` presence so a
    //     compile failure in only one direction can't crash the unrelated
    //     direction, and (2) guard Dk against the 32-wide SIMD tile shape
    //     the kernel source assumes — `n_per_t = Dk / 32` becomes 0 or a
    //     fractional remainder for Dk < 32 / Dk not a multiple of 32, and
    //     Metal rejects the resulting `float state[n_per_t]` zero-length
    //     array. Real qwen3_next bundles use Dk=128 so the fast path is
    //     always taken there; only narrow synthetic configs hit the ops
    //     fallback.
    let manager = GatedDeltaKernelManager.shared
    let selectedKernel = manager.kernel(
        hasMask: mask != nil,
        roundStateEachStep: roundStateEachStep)
    if selectedKernel != nil && Dk >= 32 && Dk % 32 == 0 {
        return gatedDeltaKernel(
            q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask,
            roundStateEachStep: roundStateEachStep)
    }

    return gatedDeltaOps(
        q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask,
        roundStateEachStep: roundStateEachStep)
}
