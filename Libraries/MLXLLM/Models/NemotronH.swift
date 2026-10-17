//
//  NemotronH.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/nemotron_h.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - JANGTQ context

/// Propagated through ``NemotronHBackbone`` / ``NemotronHBlock`` /
/// ``NemotronHMoE`` to opt the routed-expert MoE projections into the
/// JANGTQ codebook kernels (``TurboQuantSwitchLinear``) instead of the
/// affine ``SwitchLinear``. Set non-nil at ``NemotronHModel`` init for
/// JANGTQ bundles (Cascade-2 JANG_2L variants, Nemotron-Omni JANGTQ4 / JANGTQ2).
public struct NemotronHJANGTQContext: Sendable {
    public let bits: Int
    public let mxtqSeed: Int
    public init(bits: Int = 2, mxtqSeed: Int = 42) {
        self.bits = bits
        self.mxtqSeed = mxtqSeed
    }
}

// MARK: - Block Type

internal enum NemotronHBlockType {
    case mamba  // "M"
    case attention  // "*"
    case mlp  // "-"
    case moe  // "E"

    init(from char: Character) {
        switch char {
        case "M": self = .mamba
        case "*": self = .attention
        case "-": self = .mlp
        case "E": self = .moe
        default: fatalError("Unknown NemotronH block type: \(char)")
        }
    }

    /// `mtp_layers_block_type` names the blocks in words rather than the
    /// single letters `hybrid_override_pattern` uses.
    init(fromName name: String) {
        switch name.trimmingCharacters(in: .whitespaces).lowercased() {
        case "mamba": self = .mamba
        case "attention": self = .attention
        case "mlp": self = .mlp
        case "moe": self = .moe
        default: fatalError("Unknown NemotronH block type name: \(name)")
        }
    }

    var profileName: String {
        switch self {
        case .mamba: return "mamba"
        case .attention: return "attention"
        case .mlp: return "mlp"
        case .moe: return "moe"
        }
    }
}

// MARK: - Mixer Protocol

/// Protocol for all mixer types in NemotronH blocks
internal protocol NemotronHMixer: Module {
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray
}

/// Type-erased switch-MLP forward used by both ``NemotronHSwitchMLP``
/// (affine `SwitchLinear`) and ``NemotronHJANGTQSwitchMLP``
/// (TurboQuant codebook). Lets ``NemotronHMoE`` hold its switch_mlp
/// as a generic `Module` and dispatch via this protocol at forward time.
internal protocol NemotronHSwitchMLPLayer: Module {
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray
    func weightedDecode(_ x: MLXArray, _ indices: MLXArray, scores: MLXArray) -> MLXArray?
}

extension NemotronHSwitchMLPLayer {
    func weightedDecode(_ x: MLXArray, _ indices: MLXArray, scores: MLXArray) -> MLXArray? {
        nil
    }
}

private func nemotronHEnvFlag(_ name: String) -> Bool {
    let raw = ProcessInfo.processInfo.environment[name]?.lowercased()
    return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
}

private let nemotronHActivationBF16RetentionFlag =
    RuntimeEnvironment.flag("VMLX_ENABLE_NEMOTRON_ACTIVATION_BF16")
    && !nemotronHEnvFlag("JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16")
private let nemotronHWeightedMoEFastPathFlag =
    nemotronHEnvFlag("JANGTQ_ENABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH")
    && !nemotronHEnvFlag("JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH")
/// Native MTP for Nemotron-H, **default OFF**.
///
/// The head is implemented for correctness and parity, not for speed. Measured
/// on Lightning 30B-A3B (JANG_4M, 200 tokens x 3): D2 runs at 0.84x — 16 %
/// SLOWER than plain autoregressive — at a 72.4 % accept rate, and D3 at 0.48x
/// with 54.3 % accept, while staying token-identical to D1. A 3 B-active model
/// is small enough that the forward is not purely bandwidth-bound, so the
/// two-token verify batch costs +46 % rather than ~0 %: 1.72 tokens for 1.46x
/// the work is a loss once per-step overhead lands.
///
/// So this must never switch itself on. Enabling it is an explicit,
/// per-machine decision, and any speed claim has to be re-measured on the
/// bundle in question first.
private let nemotronHNativeMTPEnvFlag = nemotronHEnvFlag("VMLX_NEMOTRON_MTP")
private let nemotronHNativeMTPKillSwitch = nemotronHEnvFlag("VMLX_DISABLE_NEMOTRON_MTP")

/// Whether the Nemotron native MTP head should be instantiated and its weights
/// admitted by `sanitize`.
///
/// This used to be a process-level `let` reading only `VMLX_NEMOTRON_MTP`,
/// which meant the host's Speculative Decoding setting could not reach it at
/// all: Off / Auto / Force-On all produced the same headless model, so the
/// control was inert for this family and no restart could change that within a
/// session.
///
/// It now follows ``NativeMTPActivation/isExplicitlyRequested`` — the same
/// per-load task-local the host sets from `LoadConfiguration.nativeMTP` — so
/// the decision travels with the load that asked for it. The env var stays as a
/// per-machine opt-in for benchmarking, and the kill switch still wins over
/// everything.
///
/// Still fail-closed by default: with no host request and no env var this is
/// `false`, so the head is never built and costs nothing, which is the D1
/// behaviour the measurements above call for.
internal func nemotronHNativeMTPEnabled() -> Bool {
    if nemotronHNativeMTPKillSwitch { return false }
    return NativeMTPActivation.isExplicitlyRequested || nemotronHNativeMTPEnvFlag
}

private let nemotronHLayerProfileFlag =
    RuntimeEnvironment.flag("VMLX_NEMOTRON_LAYER_PROFILE")
private let nemotronHMambaConvFastPathDisabledFlag =
    RuntimeEnvironment.flag("VMLX_DISABLE_NEMOTRON_MAMBA_CONV_FASTPATH")

private func nemotronHActivationBF16RetentionEnabled() -> Bool {
    nemotronHActivationBF16RetentionFlag
}

private func nemotronHWeightedMoEFastPathEnabled() -> Bool {
    nemotronHWeightedMoEFastPathFlag
}

private func nemotronHLayerProfileEnabled() -> Bool {
    nemotronHLayerProfileFlag
}

private final class NemotronHLayerProfiler: @unchecked Sendable {
    static let shared = NemotronHLayerProfiler()

    private let lock = NSLock()
    private var rows: [String: (count: Int, totalMs: Double)] = [:]

    func record(_ label: String, seconds: Double) {
        lock.lock()
        let current = rows[label] ?? (0, 0)
        rows[label] = (current.count + 1, current.totalMs + seconds * 1000)
        lock.unlock()
    }

    func printSummary() {
        lock.lock()
        let snapshot = rows
        rows.removeAll()
        lock.unlock()

        for key in snapshot.keys.sorted() {
            guard let value = snapshot[key], value.count > 0 else { continue }
            let avg = value.totalMs / Double(value.count)
            FileHandle.standardError.write(Data(
                String(
                    format: "NEMOTRON_LAYER_PROFILE label=%@ count=%d total_ms=%.3f avg_ms=%.3f\n",
                    key, value.count, value.totalMs, avg
                ).utf8))
        }
    }
}

// MARK: - Activations

/// Squared ReLU activation: relu(x)^2
internal func relu2(_ x: MLXArray) -> MLXArray {
    let y = MLX.maximum(x, MLXArray(0, dtype: x.dtype))
    return y * y
}

// MARK: - MambaRMSNormGated

internal class NemotronHRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let groupSize: Int

    init(dimensions: Int, eps: Float, groupSize: Int) {
        self.eps = eps
        self.groupSize = groupSize
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray?) -> MLXArray {
        var states = x
        if let gate {
            states = states * silu(gate)
        }

        // Python: x = mx.unflatten(x, axis=-1, shape=(-1, self.group_size))
        // Reshape [..., hidden] -> [..., nGroups, groupSize] for per-group normalization
        let shape = states.shape
        var newShape = Array(shape.dropLast())
        newShape.append(-1)
        newShape.append(groupSize)
        let unflattened = states.reshaped(newShape)

        // Python: x = mx.fast.rms_norm(x, weight=None, eps=self.eps)
        // Apply RMS norm per group WITHOUT scaling (pass ones as weight)
        // Swift rmsNorm doesn't accept nil, so we use identity weight
        let normed = MLXFast.rmsNorm(unflattened, weight: MLXArray.ones([groupSize], dtype: unflattened.dtype), eps: eps)

        // Python: return self.weight * x.flatten(-2)
        // Flatten back to [..., hidden] and apply learned weight
        let flattened = normed.reshaped(shape)
        return weight * flattened
    }
}

private func nemotronHMambaConvFastPathDisabled() -> Bool {
    nemotronHMambaConvFastPathDisabledFlag
}

private func makeNemotronHMambaDepthwiseDecodeConvKernel() -> MLXFast.MLXFastKernel? {
    let source = """
            auto c = thread_position_in_grid.x;
            auto b = thread_position_in_grid.y;

            if (c >= C || b >= Bsz) {
                return;
            }

            float acc = 0.0;
            auto state_base = (b * (K - 1)) * C + c;
            auto weight_base = c * K;

            for (int k = 0; k < K - 1; ++k) {
                acc += static_cast<float>(state[state_base + k * C])
                    * static_cast<float>(weight[weight_base + k]);
            }
            acc += static_cast<float>(input[b * C + c])
                * static_cast<float>(weight[weight_base + K - 1]);
            // Match conv1d -> bias -> SiLU, including each activation-dtype
            // rounding boundary. Keeping everything in fp32 changes bf16
            // Mamba state on every decode step despite an identical prefill.
            T value = static_cast<T>(static_cast<T>(acc) + bias[c]);
            // This is MLX's stable Sigmoid implementation. fast::exp(-x)
            // also differs from the reference for bf16 negative activations.
            auto tail = 1 / (1 + metal::precise::exp(metal::abs(value)));
            T sigmoid = (value < 0) ? tail : 1 - tail;
            out[b * C + c] = value * sigmoid;

            auto next_base = (b * (K - 1)) * C + c;
            for (int k = 0; k < K - 2; ++k) {
                next_state[next_base + k * C] = state[state_base + (k + 1) * C];
            }
            if (K > 1) {
                next_state[next_base + (K - 2) * C] = input[b * C + c];
            }
        """

    // METAL-ONLY: case 2. Metal kernel; the `conv1d` path of `applyConv`
    // computes the same with MLX ops, but `nemotronHMambaDepthwiseDecodeConv`
    // (on unless `VMLX_DISABLE_NEMOTRON_MAMBA_CONV_FASTPATH=1`) does not pick
    // it for the shapes this kernel handles, so this path needs a Metal device.
    return MLXFast.metalKernel(
        name: "nemotron_h_mamba_depthwise_decode_conv",
        inputNames: ["input", "state", "weight", "bias"],
        outputNames: ["out", "next_state"],
        source: source
    )
}

private final class NemotronHMambaConvKernelManager: Sendable {
    static let shared = NemotronHMambaConvKernelManager()

    let decodeKernel: MLXFast.MLXFastKernel?

    private init() {
        decodeKernel = makeNemotronHMambaDepthwiseDecodeConvKernel()
    }
}

internal func nemotronHMambaDepthwiseDecodeConv(
    input: MLXArray,
    state: MLXArray,
    weight: MLXArray,
    bias: MLXArray,
    channels: Int,
    kernelSize: Int
) -> (MLXArray, MLXArray)? {
    guard !nemotronHMambaConvFastPathDisabled(),
        input.dim(1) == 1,
        state.dim(1) == kernelSize - 1,
        state.dim(2) == channels,
        weight.dim(0) == channels,
        weight.dim(1) == kernelSize,
        weight.dim(2) == 1,
        bias.dim(0) == channels,
        let kernel = NemotronHMambaConvKernelManager.shared.decodeKernel
    else {
        return nil
    }

    let batch = input.dim(0)
    let outputDType = input.dtype
    let outputs = kernel(
        [input, state, weight, bias],
        template: [
            ("T", outputDType),
            ("Bsz", batch),
            ("C", channels),
            ("K", kernelSize),
        ],
        grid: (channels, batch, 1),
        threadGroup: (min(256, channels), 1, 1),
        outputShapes: [[batch, 1, channels], [batch, kernelSize - 1, channels]],
        outputDTypes: [outputDType, outputDType]
    )

    return (outputs[0], outputs[1])
}

// MARK: - Row-exact multi-row Mamba verify kernels

/// Native-MTP verify rows through the Mamba conv and SSM step in ONE launch each, with every row
/// running exactly the single-row decode statements of `nemotron_h_mamba_depthwise_decode_conv` /
/// `ssm_kernel` (SSM.swift). The per-row path ran R conv + R SSM launches per layer and read and wrote
/// the full recurrent state (128 x 64 x 128 fp32 = 4 MB on Super-VL) once per row; here the window
/// and the state stay in registers between rows (the state is re-read as its stored dtype, as the next
/// single-row step would), and only each row's state is written for staged partial accepts.
/// Batch 1 only (verify). `VMLX_NEMOTRON_FUSED_VERIFY_MAMBA=0` keeps the per-row launches.
enum NemotronHMambaRowsKernels {
    static let enabled =
        ProcessInfo.processInfo.environment["VMLX_NEMOTRON_FUSED_VERIFY_MAMBA"] != "0"

    static let conv: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "nemotron_h_mamba_depthwise_conv_rows",
        inputNames: ["input", "state", "weight", "bias"],
        outputNames: ["out"],
        source: """
            auto c = thread_position_in_grid.x;
            if (c >= C) {
                return;
            }
            auto weight_base = c * K;
            T window[K > 1 ? K - 1 : 1];
            for (int k = 0; k < K - 1; ++k) {
                window[k] = state[k * C + c];
            }
            for (int r = 0; r < R; ++r) {
                float acc = 0.0;
                for (int k = 0; k < K - 1; ++k) {
                    acc += static_cast<float>(window[k])
                        * static_cast<float>(weight[weight_base + k]);
                }
                T in_v = input[r * C + c];
                acc += static_cast<float>(in_v)
                    * static_cast<float>(weight[weight_base + K - 1]);
                T value = static_cast<T>(static_cast<T>(acc) + bias[c]);
                auto tail = 1 / (1 + metal::precise::exp(metal::abs(value)));
                T sigmoid = (value < 0) ? tail : 1 - tail;
                out[r * C + c] = value * sigmoid;
                for (int k = 0; k < K - 2; ++k) {
                    window[k] = window[k + 1];
                }
                if (K > 1) {
                    window[K - 2] = in_v;
                }
            }
            """)

    static let ssm: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "nemotron_h_ssm_rows",
        inputNames: ["X", "A_log", "B", "C", "D", "dt", "state_in"],
        outputNames: ["out", "states_out"],
        source: """
            auto n = thread_position_in_grid.z;
            auto h_idx = n % H;
            auto g_idx = n / G;
            constexpr int n_per_t = Ds / 32;
            auto ds_idx = thread_position_in_threadgroup.x;
            auto d_idx = thread_position_in_grid.y;
            auto i_state = state_in + n * Dh * Ds;
            auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
            float st[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
                st[i] = i_state[d_idx * Ds + n_per_t * ds_idx + i];
            }
            for (int r = 0; r < R; ++r) {
                auto x = X + (r * H + n) * Dh;
                auto C_ = C + (r * HB + g_idx) * Ds;
                auto B_ = B + (r * HB + g_idx) * Ds;
                auto o_state = states_out + (r * H + n) * Dh * Ds;
                auto dt_ = static_cast<float>(dt[r * H + n]);
                auto dA = fast::exp(A * dt_);
                float acc = 0.0;
                auto x_ = static_cast<float>(x[d_idx]);
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * ds_idx + i;
                    auto idx = d_idx * Ds + s_idx;
                    auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
                    auto state = dA * st[i] + dB_by_x;
                    U stored = static_cast<U>(state);
                    o_state[idx] = stored;
                    st[i] = stored;
                    acc += state * C_[s_idx];
                }
                acc = simd_sum(acc);
                if (thread_index_in_simdgroup == 0) {
                    out[(r * H + n) * Dh + d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
                }
            }
            """)

    /// conv rows: `input` [1, R, C], `state` [1, K-1, C] -> [1, R, C] (SiLU applied).
    static func convRows(
        input: MLXArray, state: MLXArray, weight: MLXArray, bias: MLXArray, channels: Int,
        kernelSize: Int
    ) -> MLXArray? {
        guard enabled, !nemotronHMambaConvFastPathDisabled(), input.dim(0) == 1,
            state.dim(0) == 1, state.dim(1) == kernelSize - 1, state.dim(2) == channels,
            weight.dim(0) == channels, weight.dim(1) == kernelSize, bias.dim(0) == channels,
            state.dtype == input.dtype
        else { return nil }
        let rows = input.dim(1)
        return conv(
            [input, state, weight, bias],
            template: [("T", input.dtype), ("C", channels), ("K", kernelSize), ("R", rows)],
            grid: (channels, 1, 1), threadGroup: (min(256, channels), 1, 1),
            outputShapes: [[1, rows, channels]], outputDTypes: [input.dtype])[0]
    }

    /// SSM rows: hidden [1, R, H, Dh], B/C [1, R, G, Ds], dt [1, R, H] (raw) -> (y [1, R, H, Dh],
    /// per-row states [R, 1, H, Dh, Ds]).
    static func ssmRows(
        hidden: MLXArray, ALog: MLXArray, B: MLXArray, C: MLXArray, D: MLXArray, dt: MLXArray,
        dtBias: MLXArray, state: MLXArray, timeStepLimit: (Float, Float)
    ) -> (MLXArray, MLXArray)? {
        guard enabled, hidden.ndim == 4, hidden.dim(0) == 1, state.ndim == 4, state.dim(0) == 1,
            B.dim(-1) % 32 == 0
        else { return nil }
        let (_, rows, h, d) = hidden.shape4
        let (hb, ds) = (B.dim(-2), B.dim(-1))
        let dt = computeDt(dt, dtBias, timeStepLimit)
        let outputs = ssm(
            [hidden, ALog, B, C, D, dt, state],
            template: [
                ("T", hidden.dtype), ("U", state.dtype), ("Dh", d), ("Ds", ds), ("H", h),
                ("G", h / hb), ("HB", hb), ("R", rows),
            ],
            grid: (32, d, h), threadGroup: (32, 8, 1),
            outputShapes: [[1, rows, h, d], [rows] + state.shape],
            outputDTypes: [hidden.dtype, state.dtype])
        return (outputs[0], outputs[1])
    }
}

// MARK: - Mamba2Mixer

/// Row-exact native-MTP verification. `FlashVerificationScope` admits 2...8 rows on the verify
/// forward ONLY; inside it every op of a verify row must reproduce the single-row AR step bit for
/// bit, or greedy MTP drifts from AR (vMLX Python C17: first divergence at layer 0, argmax flips
/// within ~48 tokens). Measured on this MLX fork (M5 Max): affine 4-bit `quantizedMM` at M >= 2
/// routes to `qmv_wide` and differs from M = 1 by up to 2.0; 8-bit differs at M = 3 and 8; fp32
/// GEMM differs from GEMV. The `[rows, 1, K] x [1, N, K]` batched form runs the M = 1 `qmv` per
/// row and is bitwise equal for 4/8-bit; dense matmuls run per row. DO NOT replace these with the
/// plain multi-row calls without re-running BENCH_ROWEXACT (VMLX_FLASH_ROW_EXACT_OFF=<site>
/// attributes the cost of each site).
enum NemotronHRowExact {
    static func active(_ x: MLXArray, site: String) -> Bool {
        FlashVerificationScope.usesRowExactVerification(inputShape: x.shape, site: site)
    }

    static func project(_ module: Linear, _ x: MLXArray, site: String) -> MLXArray {
        guard active(x, site: site) else { return module(x) }
        let rows = x.dim(1)
        if type(of: module) == QuantizedLinear.self, let q = module as? QuantizedLinear {
            // Batched M = 1 per row: exact, but streams the weight once PER ROW.
            func batched(_ input: MLXArray) -> MLXArray {
                quantizedMM(
                    input.reshaped(input.dim(0), 1, input.dim(-1)), q.weight[.newAxis],
                    scales: q.scales[.newAxis], biases: q.biases.map { $0[.newAxis] },
                    transpose: true, groupSize: q.groupSize, bits: q.bits, mode: q.mode
                ).reshaped(input.dim(0), -1)
            }
            let flat = x.reshaped(rows, x.dim(-1))
            // Same per-row arithmetic with the weight streamed once for all rows (admitted
            // bitwise against `batched` per shape; see RowExactAffineQMV).
            var y =
                (q.mode == .affine
                    ? RowExactAffineQMV.project(
                        flat, weight: q.weight, scales: q.scales, biases: q.biases,
                        groupSize: q.groupSize, bits: q.bits, reference: batched)
                    : nil) ?? batched(flat)
            if let bias = q.bias { y = y + bias }
            if y.dtype != x.dtype, x.dtype == .bfloat16 || x.dtype == .float16 {
                y = y.asType(x.dtype)
            }
            return y.reshaped(1, rows, -1)
        }
        return concatenated((0 ..< rows).map { module(x[0..., $0 ..< ($0 + 1)]) }, axis: 1)
    }
}

internal class NemotronHMamba2Mixer: Module, NemotronHMixer {
    let numHeads: Int
    let hiddenSize: Int
    let ssmStateSize: Int
    let convKernelSize: Int
    let intermediateSize: Int
    let numGroups: Int
    let headDim: Int
    let timeStepLimit: (Float, Float)
    let headsPerGroup: Int

    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "D") var D: MLXArray

    @ModuleInfo(key: "norm") var norm: NemotronHRMSNormGated

    init(_ args: NemotronHConfiguration) {
        self.numHeads = args.mambaNumHeads
        self.hiddenSize = args.hiddenSize
        self.ssmStateSize = args.ssmStateSize
        self.convKernelSize = args.convKernel
        self.intermediateSize = args.mambaNumHeads * args.mambaHeadDim
        self.numGroups = args.nGroups
        self.headDim = args.mambaHeadDim
        self.timeStepLimit = (args.timeStepLimitMin, args.timeStepLimitMax)
        self.headsPerGroup = numHeads / numGroups
        self.convDim = intermediateSize + 2 * numGroups * ssmStateSize

        self._conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            groups: convDim,
            bias: args.useConvBias
        )

        let projectionSize = intermediateSize + convDim + numHeads
        self._inProj.wrappedValue = Linear(hiddenSize, projectionSize, bias: args.mambaProjBias)

        self._dtBias.wrappedValue = MLXArray.ones([numHeads])
        let headsRange = (MLXArray(0 ..< numHeads).asType(.float32) + 1)
        self._aLog.wrappedValue = MLX.log(headsRange)
        self._D.wrappedValue = MLXArray.ones([numHeads])

        let groupSize = intermediateSize / numGroups
        self._norm.wrappedValue = NemotronHRMSNormGated(
            dimensions: intermediateSize,
            eps: args.layerNormEpsilon,
            groupSize: groupSize
        )

        self._outProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: args.mambaProjBias)

        super.init()
    }

    private func applyConv(
        _ input: MLXArray, mask: MLXArray?, cache: MambaCache?, stageVerify: Bool = false
    ) -> MLXArray {
        var convInput = input

        // Apply mask if present
        if let mask {
            let expandedMask = expandedDimensions(mask, axis: -1)
            convInput = MLX.where(expandedMask, convInput, MLXArray.zeros(like: convInput))
        }

        let batch = convInput.dim(0)
        let dtype = convInput.dtype
        var convState = cache?[0]

        if convState == nil {
            if convKernelSize > 1 {
                convState = MLXArray.zeros([batch, convKernelSize - 1, convDim], dtype: dtype)
            } else {
                convState = MLXArray.zeros([batch, 0, convDim], dtype: dtype)
            }
        }

        if convInput.dim(1) == 1,
            let bias = conv1d.bias,
            let (convOutput, nextState) = nemotronHMambaDepthwiseDecodeConv(
                input: convInput,
                state: convState!,
                weight: conv1d.weight,
                bias: bias,
                channels: convDim,
                kernelSize: convKernelSize)
        {
            if let cache {
                cache[0] = nextState
            }
            return convOutput
        }

        let padded = concatenated([convState!, convInput], axis: 1)

        if let cache {
            let end = padded.dim(1)
            let start = max(0, end - (convKernelSize - 1))
            let tail = padded[0..., start ..< end, 0...]
            if stageVerify {
                // Staged MTP verify: the committed conv state stays untouched;
                // commitVerifyStaged picks the accepted prefix's tail.
                cache.stageVerifySlot(5, padded)
                cache.stageVerifySlot(6, tail)
            } else {
                cache[0] = tail
            }
        }

        let convOutput = conv1d(padded)
        return silu(convOutput)
    }

    private func mambaForward(
        _ hiddenStates: MLXArray,
        mask: MLXArray?,
        cache: MambaCache?
    ) -> MLXArray {
        let profile = nemotronHLayerProfileEnabled()
        if let cache, hiddenStates.dim(1) > 1, mask == nil,
            NativeMTPVerifierStatePolicy.shouldStageVerifyInputs,
            NemotronHRowExact.active(hiddenStates, site: "mamba")
        {
            return mambaVerifyRowExact(hiddenStates, cache: cache)
        }
        let inProjStart = profile ? CFAbsoluteTimeGetCurrent() : 0
        let projected = NemotronHRowExact.project(inProj, hiddenStates, site: "mamba_in")
        if profile {
            MLX.eval(projected)
            NemotronHLayerProfiler.shared.record("mamba.in_proj", seconds: CFAbsoluteTimeGetCurrent() - inProjStart)
        }

        let splitStart = profile ? CFAbsoluteTimeGetCurrent() : 0
        let splits = split(
            projected, indices: [intermediateSize, intermediateSize + convDim], axis: -1)
        // A failed split (e.g. a checkpoint whose in_proj width disagrees with
        // intermediateSize + convDim + numHeads) records an MLX error and
        // returns an empty vector; subscripting it traps the process before
        // the recorded error can surface. Bail with the input — the enclosing
        // withError scope throws the real diagnostic at exit.
        guard splits.count == 3 else { return hiddenStates }
        let gate = splits[0]
        let convInput = splits[1]
        let dt = splits[2]
        if profile {
            MLX.eval(gate, convInput, dt)
            NemotronHLayerProfiler.shared.record("mamba.split", seconds: CFAbsoluteTimeGetCurrent() - splitStart)
        }

        // Staged native-MTP verify (`.inputCaptureStaged`): run the S verify
        // rows from the committed state but leave cache[0]/cache[1]/offset
        // untouched; the iterator commits the accepted prefix afterwards via
        // ``commitVerifyStaged(cache:acceptedInputs:blockLength:)``. Without
        // this a rejected draft costs a full backbone replay of the accepted
        // rows. End-to-end MTP benefit must still be measured per bundle.
        let stageVerify = cache != nil && hiddenStates.dim(1) > 1 && mask == nil
            && NativeMTPVerifierStatePolicy.shouldStageVerifyInputs
        let convStart = profile ? CFAbsoluteTimeGetCurrent() : 0
        let convOutput = applyConv(convInput, mask: mask, cache: cache, stageVerify: stageVerify)
        if profile {
            MLX.eval(convOutput)
            NemotronHLayerProfiler.shared.record("mamba.conv", seconds: CFAbsoluteTimeGetCurrent() - convStart)
        }

        let convSplitStart = profile ? CFAbsoluteTimeGetCurrent() : 0
        let convSplits = split(
            convOutput,
            indices: [intermediateSize, intermediateSize + numGroups * ssmStateSize],
            axis: -1
        )

        guard convSplits.count == 3 else { return hiddenStates }
        var hidden = convSplits[0]
        var B = convSplits[1]
        var C = convSplits[2]

        hidden = hidden.reshaped([hidden.dim(0), hidden.dim(1), numHeads, headDim])
        B = B.reshaped([B.dim(0), B.dim(1), numGroups, ssmStateSize])
        C = C.reshaped([C.dim(0), C.dim(1), numGroups, ssmStateSize])

        let dtArray = dt.reshaped([dt.dim(0), dt.dim(1), numHeads])
        if profile {
            MLX.eval(hidden, B, C, dtArray)
            NemotronHLayerProfiler.shared.record("mamba.conv_split", seconds: CFAbsoluteTimeGetCurrent() - convSplitStart)
        }

        let previousState = cache?[1]
        let ssmStart = profile ? CFAbsoluteTimeGetCurrent() : 0
        let (y, nextState) = ssmUpdate(
            hiddenStates: hidden,
            ALog: aLog,
            B: B,
            C: C,
            D: D.asType(hidden.dtype),
            dt: dtArray,
            dtBias: dtBias,
            state: previousState,
            timeStepLimit: timeStepLimit,
            mask: mask
        )
        if profile {
            MLX.eval(y, nextState)
            NemotronHLayerProfiler.shared.record("mamba.ssm_update", seconds: CFAbsoluteTimeGetCurrent() - ssmStart)
        }

        if let cache {
            if stageVerify {
                cache.stageVerifySlot(0, hidden)
                cache.stageVerifySlot(1, B)
                cache.stageVerifySlot(2, C)
                cache.stageVerifySlot(3, dtArray)
                cache.stageVerifySlot(4, nextState)
                cache.stageVerifySlot(7, MLXArray(Int32(hidden.dim(1))))
                cache.verifyStagedRowStates = nil
            } else {
                cache[1] = nextState
                cache.offset += y.dim(1)
            }
        }

        let flattenedY = y.flattened(start: 2)
        let normStart = profile ? CFAbsoluteTimeGetCurrent() : 0
        let normalized = norm(flattenedY, gate: gate)
        if profile {
            MLX.eval(normalized)
            NemotronHLayerProfiler.shared.record("mamba.norm", seconds: CFAbsoluteTimeGetCurrent() - normStart)
        }

        let outProjStart = profile ? CFAbsoluteTimeGetCurrent() : 0
        let output = NemotronHRowExact.project(outProj, normalized, site: "mamba_out")
        if profile {
            MLX.eval(output)
            NemotronHLayerProfiler.shared.record("mamba.out_proj", seconds: CFAbsoluteTimeGetCurrent() - outProjStart)
        }
        return output
    }

    /// One AR-step convolution: exactly the S = 1 branch of ``applyConv`` (fused decode kernel
    /// when it applies, else conv1d over [state, row] + SiLU).
    private func convStep(_ row: MLXArray, state: MLXArray) -> (MLXArray, MLXArray) {
        if let bias = conv1d.bias,
            let (out, next) = nemotronHMambaDepthwiseDecodeConv(
                input: row, state: state, weight: conv1d.weight, bias: bias,
                channels: convDim, kernelSize: convKernelSize)
        {
            return (out, next)
        }
        let padded = concatenated([state, row], axis: 1)
        let end = padded.dim(1)
        let start = max(0, end - (convKernelSize - 1))
        return (silu(conv1d(padded)), padded[0..., start ..< end, 0...])
    }

    /// Row-exact staged verify: conv and SSM advance one row at a time through the AR-step
    /// kernels (projections stay batched through the row-exact M = 1 form), and the state after
    /// every row is staged, so a partial accept commits exactly the AR state.
    private func mambaVerifyRowExact(_ hiddenStates: MLXArray, cache: MambaCache) -> MLXArray {
        let rows = hiddenStates.dim(1)
        let projected = NemotronHRowExact.project(inProj, hiddenStates, site: "mamba_in")
        let splits = split(
            projected, indices: [intermediateSize, intermediateSize + convDim], axis: -1)
        guard splits.count == 3 else { return hiddenStates }
        let gate = splits[0]
        let convInput = splits[1]
        let dt = splits[2]

        let batch = convInput.dim(0)
        let committedConv = cache[0]
            ?? MLXArray.zeros([batch, max(0, convKernelSize - 1), convDim], dtype: convInput.dtype)
        let convOutput: MLXArray
        if let bias = conv1d.bias,
            let fused = NemotronHMambaRowsKernels.convRows(
                input: convInput, state: committedConv, weight: conv1d.weight, bias: bias,
                channels: convDim, kernelSize: convKernelSize)
        {
            convOutput = fused
        } else {
            var convState = committedConv
            var convRows: [MLXArray] = []
            convRows.reserveCapacity(rows)
            for r in 0 ..< rows {
                let (out, next) = convStep(convInput[0..., r ..< (r + 1)], state: convState)
                convRows.append(out)
                convState = next
            }
            convOutput = concatenated(convRows, axis: 1)
        }
        let convSplits = split(
            convOutput, indices: [intermediateSize, intermediateSize + numGroups * ssmStateSize],
            axis: -1)
        guard convSplits.count == 3 else { return hiddenStates }
        let hidden = convSplits[0].reshaped([batch, rows, numHeads, headDim])
        let B = convSplits[1].reshaped([batch, rows, numGroups, ssmStateSize])
        let C = convSplits[2].reshaped([batch, rows, numGroups, ssmStateSize])
        let dtArray = dt.reshaped([batch, rows, numHeads])

        var state = cache[1]
        var ys: [MLXArray] = []
        var states: [MLXArray] = []
        ys.reserveCapacity(rows)
        states.reserveCapacity(rows)
        if let committed = state,
            let (y, rowStates) = NemotronHMambaRowsKernels.ssmRows(
                hidden: hidden, ALog: aLog, B: B, C: C, D: D.asType(hidden.dtype), dt: dtArray,
                dtBias: dtBias, state: committed, timeStepLimit: timeStepLimit)
        {
            ys = [y]
            states = (0 ..< rows).map { rowStates[$0] }
        }
        for r in 0 ..< (ys.isEmpty ? rows : 0) {
            let (y, next) = ssmUpdate(
                hiddenStates: hidden[0..., r ..< (r + 1)], ALog: aLog,
                B: B[0..., r ..< (r + 1)], C: C[0..., r ..< (r + 1)],
                D: D.asType(hidden.dtype), dt: dtArray[0..., r ..< (r + 1)], dtBias: dtBias,
                state: state, timeStepLimit: timeStepLimit, mask: nil)
            ys.append(y)
            states.append(next)
            state = next
        }
        let padded = concatenated([committedConv, convInput], axis: 1)
        let end = padded.dim(1)
        let tail = padded[0..., max(0, end - (convKernelSize - 1)) ..< end, 0...]
        cache.stageVerifySlot(0, hidden)
        cache.stageVerifySlot(1, B)
        cache.stageVerifySlot(2, C)
        cache.stageVerifySlot(3, dtArray)
        cache.stageVerifySlot(4, states[rows - 1])
        cache.stageVerifySlot(5, padded)
        cache.stageVerifySlot(6, tail)
        cache.stageVerifySlot(7, MLXArray(Int32(rows)))
        cache.verifyStagedRowStates = states

        let y = concatenated(ys, axis: 1)
        let normalized = norm(y.flattened(start: 2), gate: gate)
        return NemotronHRowExact.project(outProj, normalized, site: "mamba_out")
    }

    /// Commit `acceptedInputs` rows of the last staged verify block. The
    /// committed state is still the pre-verify state (the staged forward never
    /// wrote it): a full accept adopts the staged final state, a partial
    /// accept re-runs the scan over the accepted rows only — the in_proj and
    /// conv outputs come from the staging slots, so no backbone replay.
    func commitVerifyStaged(cache: MambaCache, acceptedInputs: Int, blockLength: Int) -> Bool {
        guard cache.verifyStagingReady else { return false }
        let n = acceptedInputs
        let slots = cache.verifyStagingSlots
        guard n > 0, n <= blockLength, let hidden = slots[0], hidden.dim(1) == blockLength
        else { return false }
        if n == blockLength {
            cache[1] = slots[4]!
            cache[0] = slots[6]!
        } else if let rowStates = cache.verifyStagedRowStates, rowStates.count == blockLength {
            cache[1] = rowStates[n - 1]
            let padded = slots[5]!
            let convEnd = padded.dim(1) - blockLength + n
            let convStart = max(0, convEnd - max(0, convKernelSize - 1))
            cache[0] = padded[0..., convStart ..< convEnd, 0...]
        } else {
            let (_, prefixState) = ssmUpdate(
                hiddenStates: hidden[0..., ..<n],
                ALog: aLog,
                B: slots[1]![0..., ..<n],
                C: slots[2]![0..., ..<n],
                D: D.asType(hidden.dtype),
                dt: slots[3]![0..., ..<n],
                dtBias: dtBias,
                state: cache[1],
                timeStepLimit: timeStepLimit,
                mask: nil)
            cache[1] = prefixState
            let padded = slots[5]!
            let convEnd = padded.dim(1) - blockLength + n
            let convStart = max(0, convEnd - max(0, convKernelSize - 1))
            cache[0] = padded[0..., convStart ..< convEnd, 0...]
        }
        cache.offset += n
        return true
    }

    // Protocol conformance
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        mambaForward(x, mask: ssmMask, cache: cache as? MambaCache)
    }
}

// MARK: - Attention

internal class NemotronHAttention: Module, NemotronHMixer {
    let args: NemotronHConfiguration
    let scale: Float
    let numHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    // NOTE: NemotronH attention does NOT use RoPE (unlike most transformer models)
    // The Python implementation at mlx_lm/models/nemotron_h.py lines 247-274 shows
    // direct attention without position embeddings

    init(_ args: NemotronHConfiguration) {
        self.args = args
        self.numHeads = args.numAttentionHeads
        self.numKeyValueHeads = args.numKeyValueHeads
        self.headDim = args.headDim ?? (args.hiddenSize / args.numAttentionHeads)
        self.scale = pow(Float(headDim), -0.5)

        let dim = args.hiddenSize
        let attentionBias = args.attentionBias

        self._wq.wrappedValue = Linear(dim, numHeads * headDim, bias: attentionBias)
        self._wk.wrappedValue = Linear(dim, numKeyValueHeads * headDim, bias: attentionBias)
        self._wv.wrappedValue = Linear(dim, numKeyValueHeads * headDim, bias: attentionBias)
        self._wo.wrappedValue = Linear(numHeads * headDim, dim, bias: attentionBias)

        super.init()
    }

    private func attentionForward(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let queries = NemotronHRowExact.project(wq, x, site: "attn_proj")
            .reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let keys = NemotronHRowExact.project(wk, x, site: "attn_proj")
            .reshaped(B, L, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
        let values = NemotronHRowExact.project(wv, x, site: "attn_proj")
            .reshaped(B, L, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)

        // No RoPE applied - NemotronH attention uses direct attention without position embeddings

        // Row-exact verify: the AR step's SDPA sees one query row; three or more query rows take
        // a different kernel, so the verify attends in pairs over the updated cache (Qwen4Exp
        // measured pairs bitwise equal to the single-row step).
        if NemotronHRowExact.active(x, site: "attn"), (3 ... 8).contains(L), B == 1,
            let simple = cache as? KVCacheSimple
        {
            let (cachedKeys, cachedValues) = simple.update(keys: keys, values: values)
            let keyLength = cachedKeys.dim(2)
            let past = keyLength - L
            let keyPositions = MLXArray(Int32(0) ..< Int32(keyLength)).reshaped(1, keyLength)
            var chunks: [MLXArray] = []
            for start in stride(from: 0, to: L, by: 2) {
                let end = min(start + 2, L)
                let queryPositions = MLXArray(Int32(past + start) ..< Int32(past + end))
                    .reshaped(end - start, 1)
                let visible = (keyPositions .<= queryPositions).reshaped(1, 1, end - start, keyLength)
                chunks.append(MLXFast.scaledDotProductAttention(
                    queries: queries[0..., 0..., start ..< end, 0...],
                    keys: cachedKeys, values: cachedValues, scale: scale,
                    mask: .array(visible)))
            }
            let attended = concatenated(chunks, axis: 2).transposed(0, 2, 1, 3).reshaped(B, L, -1)
            return NemotronHRowExact.project(wo, attended, site: "attn_proj")
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return NemotronHRowExact.project(wo, output, site: "attn_proj")
    }

    // Protocol conformance
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        attentionForward(x, mask: attentionMask, cache: cache)
    }
}

// MARK: - MLP

internal class NemotronHMLP: Module, UnaryLayer, NemotronHMixer {
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ args: NemotronHConfiguration, intermediateSize: Int? = nil) {
        let intermediate = intermediateSize ?? args.intermediateSize
        self._upProj.wrappedValue = Linear(args.hiddenSize, intermediate, bias: args.mlpBias)
        self._downProj.wrappedValue = Linear(intermediate, args.hiddenSize, bias: args.mlpBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        NemotronHRowExact.project(
            downProj, relu2(NemotronHRowExact.project(upProj, x, site: "mlp")), site: "mlp")
    }

    // Protocol conformance
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        callAsFunction(x)
    }
}

// MARK: - MoE Gate

internal func groupExpertSelect(
    gates: MLXArray,
    eSCB: MLXArray,  // e_score_correction_bias
    topK: Int,
    nGroup: Int,
    topkGroup: Int,
    routedScalingFactor: Float,
    normTopkProb: Bool
) -> (MLXArray, MLXArray) {
    let (bsz, seqLen) = (gates.dim(0), gates.dim(1))

    // Original scores using sigmoid. fp32 precision floor matches Python
    // `mlx_lm/models/dots1.py:116` and `mlx_lm/models/nemotron_h.py:324`
    // (both do `mx.sigmoid(gates.astype(mx.float32))`). NemotronH's caller
    // passes bf16 gates without a pre-cast, so without this fp32 cast the
    // sigmoid + downstream score-add ran in bf16 — the same class of
    // precision drift that the Hy3 fp32 lm_head fix addresses, but at
    // the MoE router instead of the logit head. Hy3's `Hy3MoEGate`
    // pre-casts gates to fp32 already, so this cast is idempotent for
    // Hy3 and load-bearing for NemotronH.
    let origScores = sigmoid(gates.asType(.float32))
    var scores = origScores + eSCB

    // Group-based selection if n_group > 1
    if nGroup > 1 {
        let numExperts = scores.dim(-1)
        let expertsPerGroup = numExperts / nGroup

        // Reshape to [batch, seq, n_group, experts_per_group]
        let groupScores = scores.reshaped(bsz, seqLen, nGroup, -1)

        // Get top-2 per group and sum for group scores (using sorted to get top values)
        let topKGroupScores = sorted(groupScores, axis: -1)[.ellipsis, ..<2].sum(
            axis: -1, keepDims: true)

        // Keep only top topkGroup groups (zero out the rest)
        let k = nGroup - topkGroup
        var groupIdx = argPartition(topKGroupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
        groupIdx = broadcast(groupIdx, to: [bsz, seqLen, k, expertsPerGroup])

        // Zero out scores from non-selected groups
        scores = putAlong(groupScores, groupIdx, values: MLXArray(0.0, dtype: groupScores.dtype), axis: -2)

        // Flatten back
        scores = flattened(scores, start: -2, end: -1)
    }

    // Get top-k experts
    let inds = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
    var finalScores = takeAlong(origScores, inds, axis: -1)

    // Normalize if needed
    if topK > 1 && normTopkProb {
        let denominator = finalScores.sum(axis: -1, keepDims: true) + MLXArray(1e-20, dtype: finalScores.dtype)
        finalScores = finalScores / denominator
    }

    // Apply scaling factor
    finalScores = finalScores * routedScalingFactor

    return (inds, finalScores)
}

internal class NemotronHMoEGate: Module {
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eSCB: MLXArray

    init(_ args: NemotronHConfiguration, experts: Int, topK: Int) {
        self.topK = topK
        self.nGroup = args.nGroup
        self.topkGroup = args.topkGroup
        self.routedScalingFactor = args.routedScalingFactor
        self.normTopkProb = args.normTopkProb

        self._weight.wrappedValue = MLXArray.zeros([experts, args.hiddenSize])
        self._eSCB.wrappedValue = MLXArray.zeros([experts])

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        // Router logits in fp32 (HF `NemotronHTopkRouter`: `F.linear(x.float(),
        // weight.float())`). A bf16 matmul rounds the logits before the sigmoid
        // and reorders near-tied experts; with 512 experts / top-22 (Nemotron
        // 3.5 Super) that changes the routed set on real tokens. Casting only
        // the sigmoid input (groupExpertSelect) is too late.
        // SPEED: the upcast is exact, so the first forward REPLACES the bf16
        // parameter with its fp32 copy (no second resident copy) instead of
        // re-converting the whole [E, hidden] matrix on every token — that
        // per-call cast moved ~10 B/param/token (Bolt: ~0.6 GB/token).
        if weight.dtype != .float32 {
            let upcast = weight.asType(.float32)
            MLX.eval(upcast)
            weight._updateInternal(upcast)
        }
        // Row-exact verify: the AR step's fp32 product is a GEMV; a multi-row GEMM reduces in a
        // different order, so each verify row gets its own product. (A multi-row transcription of
        // MLX's GEMVKernel streaming the weight once measured NO gain on Super-VL, 2026-10-17: the
        // 8 MB [512, 4096] fp32 router stays cache-resident across the consecutive per-row GEMVs.)
        let gates: MLXArray
        if NemotronHRowExact.active(x, site: "router") {
            let w = weight.transposed()
            gates = concatenated(
                (0 ..< x.dim(1)).map { MLX.matmul(x[0..., $0 ..< ($0 + 1)].asType(.float32), w) },
                axis: 1)
        } else {
            gates = MLX.matmul(x.asType(.float32), weight.transposed())
        }
        return groupExpertSelect(
            gates: gates,
            eSCB: eSCB,
            topK: topK,
            nGroup: nGroup,
            topkGroup: topkGroup,
            routedScalingFactor: routedScalingFactor,
            normTopkProb: normTopkProb
        )
    }
}

// MARK: - SwitchMLP for NemotronH (uses relu2 instead of silu/glu)

internal class NemotronHSwitchMLP: Module, NemotronHSwitchMLPLayer {
    @ModuleInfo(key: "fc1") var fc1: SwitchLinear
    @ModuleInfo(key: "fc2") var fc2: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int

    init(inputDims: Int, hiddenDims: Int, numExperts: Int) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts

        self._fc1.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: false)
        self._fc2.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        // Row-exact verify keeps the unsorted gather: per-row bitwise equal to the AR step.
        let doSort = indices.size > 64 && !NemotronHRowExact.active(x, site: "moe")
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        var y = fc1(x, idx, sortedIndices: doSort)
        y = relu2(y)
        y = fc2(y, idx, sortedIndices: doSort)

        if doSort {
            y = scatterUnsort(x: y, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(y, axis: -2)
    }
}

// MARK: - MoE

internal class NemotronHMoE: Module, UnaryLayer, NemotronHMixer {
    let layerIdx: Int
    let numExpertsPerTok: Int
    let hasSharedExperts: Bool
    let moeLatentSize: Int?

    @ModuleInfo(key: "gate") var gate: NemotronHMoEGate
    @ModuleInfo(key: "fc1_latent_proj") var fc1LatentProj: Linear?
    @ModuleInfo(key: "fc2_latent_proj") var fc2LatentProj: Linear?
    /// `switch_mlp` is type-erased so that JANGTQ bundles can swap in
    /// `NemotronHJANGTQSwitchMLP` (TurboQuant-backed fc1/fc2) without
    /// duplicating the surrounding MoE plumbing. Both variants implement
    /// `callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray`.
    @ModuleInfo(key: "switch_mlp") var switchMLP: Module
    @ModuleInfo(key: "shared_experts") var sharedExperts: NemotronHMLP?

    init(
        _ args: NemotronHConfiguration, layerIdx: Int, jangtq: NemotronHJANGTQContext? = nil,
        routed: (any WeightedRoutedExpertLayer)? = nil, geometryLayer: Int? = nil
    ) {
        // `geometryLayer` is the backbone block index used to look up per-block
        // MoE geometry (nil for the MTP head, which uses the global fields).
        let geometry = geometryLayer.map { args.moeGeometry(layer: $0) }
            ?? (experts: args.nRoutedExperts, intermediate: args.moeIntermediateSize,
                topK: args.numExpertsPerTok)
        self.layerIdx = layerIdx
        self.numExpertsPerTok = geometry.topK
        self.hasSharedExperts = args.nSharedExperts != nil && args.nSharedExperts! > 0
        self.moeLatentSize = args.moeLatentSize

        self._gate.wrappedValue = NemotronHMoEGate(
            args, experts: geometry.experts, topK: geometry.topK)
        let expertInputDims = args.moeLatentSize ?? args.hiddenSize
        if let latentSize = args.moeLatentSize {
            self._fc1LatentProj.wrappedValue = Linear(
                args.hiddenSize, latentSize, bias: false)
            self._fc2LatentProj.wrappedValue = Linear(
                latentSize, args.hiddenSize, bias: false)
        }
        if let routed {
            // Mapped custom banks (JANGH relu2): owned outside the parameter
            // tree, never touched by the affine quantizer or weight update.
            self._switchMLP.wrappedValue = routed
        } else if let jangtq {
            let streamJANGTQExperts =
                JANGTQStreamingExperts.isEnabled
                || JANGTQStreamingExperts.shouldAutoEnableNemotronUltra(
                    nRoutedExperts: args.nRoutedExperts,
                    numExpertsPerTok: args.numExpertsPerTok,
                    routedBits: jangtq.bits)
                    && JANGTQStreamingExperts.canUseNemotronUltraStreaming(layerIdx: layerIdx)
            if streamJANGTQExperts {
                self._switchMLP.wrappedValue = StreamingTurboQuantSwitchReLUSquaredMLP(
                    inputDims: expertInputDims,
                    hiddenDims: args.moeIntermediateSize,
                    numExperts: args.nRoutedExperts,
                    bits: jangtq.bits,
                    seed: jangtq.mxtqSeed,
                    layerIdx: layerIdx)
            } else {
                self._switchMLP.wrappedValue = NemotronHJANGTQSwitchMLP(
                    inputDims: expertInputDims,
                    hiddenDims: args.moeIntermediateSize,
                    numExperts: args.nRoutedExperts,
                    bits: jangtq.bits,
                    mxtqSeed: jangtq.mxtqSeed
                )
            }
        } else {
            self._switchMLP.wrappedValue = NemotronHSwitchMLP(
                inputDims: expertInputDims,
                hiddenDims: geometry.intermediate,
                numExperts: geometry.experts
            )
        }

        if hasSharedExperts {
            self._sharedExperts.wrappedValue = NemotronHMLP(
                args,
                intermediateSize: args.moeSharedExpertIntermediateSize
            )
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let profile = nemotronHLayerProfileEnabled()

        let gateStart = profile ? CFAbsoluteTimeGetCurrent() : 0
        let (inds, scores) = gate(x)
        if profile {
            MLX.eval(inds, scores)
            NemotronHLayerProfiler.shared.record("moe.gate", seconds: CFAbsoluteTimeGetCurrent() - gateStart)
        }
        JangPressCanonicalExpertAdvisor.shared.observe(layer: layerIdx, indices: inds)
        let fc1Start = profile ? CFAbsoluteTimeGetCurrent() : 0
        let expertInput = fc1LatentProj.map { NemotronHRowExact.project($0, x, site: "latent") } ?? x
        if profile {
            MLX.eval(expertInput)
            NemotronHLayerProfiler.shared.record("moe.fc1_latent", seconds: CFAbsoluteTimeGetCurrent() - fc1Start)
        }
        if let routed = switchMLP as? any WeightedRoutedExpertLayer {
            // JANGH relu2 banks reduce over the routes themselves (F32).
            let switchStart = profile ? CFAbsoluteTimeGetCurrent() : 0
            var y = routed.callRouted(
                expertInput, indices: inds.asType(.uint32), scores: scores.asType(.float32))
            if profile {
                MLX.eval(y)
                NemotronHLayerProfiler.shared.record("moe.switch_mlp", seconds: CFAbsoluteTimeGetCurrent() - switchStart)
            }
            if let fc2LatentProj { y = NemotronHRowExact.project(fc2LatentProj, y, site: "latent") }
            if let sharedExperts { y = y + sharedExperts(x) }
            return y
        }
        // Dispatch through the type-erased protocol — both
        // `NemotronHSwitchMLP` (affine) and `NemotronHJANGTQSwitchMLP`
        // (TurboQuant) implement `callAsFunction(_:_:)`.
        guard let layer = switchMLP as? NemotronHSwitchMLPLayer else {
            fatalError("NemotronHMoE.switch_mlp must conform to NemotronHSwitchMLPLayer (got \(type(of: switchMLP)))")
        }
        var y: MLXArray
        let switchStart = profile ? CFAbsoluteTimeGetCurrent() : 0
        if nemotronHWeightedMoEFastPathEnabled(),
            let weighted = layer.weightedDecode(expertInput, inds, scores: scores)
        {
            y = weighted
        } else {
            y = layer(expertInput, inds)
            // Router-weighted sum accumulated in fp32, then cast back — the
            // reference (`(y * scores[..., None]).sum(-2).astype(y.dtype)` with
            // fp32 scores) promotes before the reduction. Casting the scores to
            // bf16 first summed up to 22 bf16 products in bf16.
            y = (y.asType(.float32) * scores.asType(.float32)[.ellipsis, .newAxis])
                .sum(axis: -2).asType(y.dtype)
        }
        if profile {
            MLX.eval(y)
            NemotronHLayerProfiler.shared.record("moe.switch_mlp", seconds: CFAbsoluteTimeGetCurrent() - switchStart)
        }
        if let fc2LatentProj {
            let fc2Start = profile ? CFAbsoluteTimeGetCurrent() : 0
            y = NemotronHRowExact.project(fc2LatentProj, y, site: "latent")
            if profile {
                MLX.eval(y)
                NemotronHLayerProfiler.shared.record("moe.fc2_latent", seconds: CFAbsoluteTimeGetCurrent() - fc2Start)
            }
        }

        if let sharedExperts {
            let sharedStart = profile ? CFAbsoluteTimeGetCurrent() : 0
            y = y + sharedExperts(x)
            if profile {
                MLX.eval(y)
                NemotronHLayerProfiler.shared.record("moe.shared", seconds: CFAbsoluteTimeGetCurrent() - sharedStart)
            }
        }

        return y
    }

    // Protocol conformance
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        callAsFunction(x)
    }
}

// MARK: - Decoder Block

internal class NemotronHBlock: Module {
    let blockType: NemotronHBlockType

    @ModuleInfo(key: "norm") var norm: RMSNorm

    // Single mixer property with base Module type - concrete type assigned at init
    // This pattern is used by Jamba for feedForward: Module
    @ModuleInfo(key: "mixer") var mixer: Module

    init(
        _ args: NemotronHConfiguration,
        layerIdx: Int,
        blockType: Character,
        jangtq: NemotronHJANGTQContext? = nil,
        routed: (any WeightedRoutedExpertLayer)? = nil
    ) {
        self.blockType = NemotronHBlockType(from: blockType)

        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)

        // Assign the appropriate concrete mixer type. The MoE branch
        // optionally swaps in JANGTQ-flavored switch projections; every
        // other layer kind is unaffected by the quant choice.
        switch self.blockType {
        case .mamba:
            _mixer.wrappedValue = NemotronHMamba2Mixer(args)
        case .attention:
            _mixer.wrappedValue = NemotronHAttention(args)
        case .mlp:
            _mixer.wrappedValue = NemotronHMLP(args)
        case .moe:
            _mixer.wrappedValue = NemotronHMoE(
                args, layerIdx: layerIdx, jangtq: jangtq, routed: routed, geometryLayer: layerIdx)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let profile = nemotronHLayerProfileEnabled()
        let label = blockType.profileName

        let normStart = profile ? CFAbsoluteTimeGetCurrent() : 0
        let hidden = norm(x)
        if profile {
            MLX.eval(hidden)
            NemotronHLayerProfiler.shared.record("\(label).norm", seconds: CFAbsoluteTimeGetCurrent() - normStart)
        }

        // Cast to protocol and call - all mixer types conform to NemotronHMixer
        let mixerFunc = mixer as! NemotronHMixer
        let mixerStart = profile ? CFAbsoluteTimeGetCurrent() : 0
        let output = mixerFunc(hidden, attentionMask: attentionMask, ssmMask: ssmMask, cache: cache)
        if profile {
            MLX.eval(output)
            NemotronHLayerProfiler.shared.record("\(label).mixer", seconds: CFAbsoluteTimeGetCurrent() - mixerStart)
        }

        let residualStart = profile ? CFAbsoluteTimeGetCurrent() : 0
        let residual = x + output
        if nemotronHActivationBF16RetentionEnabled(),
            (x.dtype == .bfloat16 || x.dtype == .float16),
            residual.dtype != x.dtype
        {
            let casted = residual.asType(x.dtype)
            if profile {
                MLX.eval(casted)
                NemotronHLayerProfiler.shared.record("\(label).residual", seconds: CFAbsoluteTimeGetCurrent() - residualStart)
            }
            return casted
        }
        if profile {
            MLX.eval(residual)
            NemotronHLayerProfiler.shared.record("\(label).residual", seconds: CFAbsoluteTimeGetCurrent() - residualStart)
        }
        return residual
    }
}

// MARK: - MTP (multi-token prediction head)

/// One block of the Nemotron MTP head.
///
/// The shipped weights are the ordinary `NemotronHBlock` shape (`norm` +
/// `mixer`) with two extras that only the first block carries:
///
///     mtp.layers.0   enorm | hnorm | eh_proj | norm | mixer.{q,k,v,o}_proj
///     mtp.layers.1   norm | mixer.gate | mixer.experts.N | shared_experts
///                    | final_layernorm
///
/// so the block is composed rather than reimplemented — the mixer is the same
/// `NemotronHAttention` / `NemotronHMoE` used by the backbone, and every tensor
/// binds on the path the bundle already uses.
internal class NemotronHMTPBlock: Module {
    @ModuleInfo(key: "enorm") var enorm: RMSNorm?
    @ModuleInfo(key: "hnorm") var hnorm: RMSNorm?
    @ModuleInfo(key: "eh_proj") var ehProj: Linear?
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "mixer") var mixer: Module
    @ModuleInfo(key: "final_layernorm") var finalLayerNorm: RMSNorm?

    let blockType: NemotronHBlockType

    /// - Parameters:
    ///   - fuses: this block owns the embedding/hidden fusion (`mtp.layers.0`).
    ///   - finalNorm: this block owns the head's output norm (`mtp.layers.1`).
    init(
        _ args: NemotronHConfiguration,
        blockType: NemotronHBlockType,
        fuses: Bool,
        finalNorm: Bool,
        layerIdx: Int,
        jangtq: NemotronHJANGTQContext?
    ) {
        self.blockType = blockType
        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        switch blockType {
        case .attention: _mixer.wrappedValue = NemotronHAttention(args)
        case .moe: _mixer.wrappedValue = NemotronHMoE(args, layerIdx: layerIdx, jangtq: jangtq)
        case .mlp: _mixer.wrappedValue = NemotronHMLP(args)
        case .mamba: _mixer.wrappedValue = NemotronHMamba2Mixer(args)
        }
        if fuses {
            self._enorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
            self._hnorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
            // Concatenated [embedding, hidden] -> hidden.
            self._ehProj.wrappedValue = Linear(
                args.hiddenSize * 2, args.hiddenSize, bias: false)
        }
        if finalNorm {
            self._finalLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        }
        super.init()
    }

    /// Fuse the next-token embedding with the backbone hidden state. Only the
    /// fusing block implements this; the rest pass through.
    func fuse(embeddings: MLXArray, hiddenStates: MLXArray) -> MLXArray {
        guard let enorm, let hnorm, let ehProj else { return hiddenStates }
        return ehProj(concatenated([enorm(embeddings), hnorm(hiddenStates)], axis: -1))
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let mixerFunc = mixer as! NemotronHMixer
        return x + mixerFunc(norm(x), attentionMask: attentionMask, ssmMask: nil, cache: cache)
    }
}

/// The Nemotron MTP head: a short stack described by `mtp_layers_block_type`.
///
/// Depth beyond the shipped head count is produced by re-entering this same
/// stack with the previously drafted token, which is what makes d2/d3 possible
/// on a bundle that ships `num_nextn_predict_layers: 1`.
internal class NemotronHMTP: Module {
    @ModuleInfo(key: "layers") var layers: [NemotronHMTPBlock]

    init(_ args: NemotronHConfiguration, jangtq: NemotronHJANGTQContext?) {
        let kinds = args.mtpLayersBlockType
        _layers.wrappedValue = kinds.enumerated().map { index, name in
            NemotronHMTPBlock(
                args,
                blockType: NemotronHBlockType(fromName: name),
                fuses: index == 0,
                finalNorm: index == kinds.count - 1,
                layerIdx: index,
                jangtq: jangtq)
        }
        super.init()
    }

    /// Only attention blocks consume KV cache; the MoE/MLP blocks are
    /// stateless, so the cache array is positional and holds a placeholder for
    /// them. That keeps indices aligned with `layers` without allocating state
    /// nothing reads.
    func makeCache() -> [KVCache] {
        layers.map { _ in KVCacheSimple() as KVCache }
    }

    /// Run the head and return the hidden state BEFORE `final_layernorm`;
    /// the model applies ``finalNorm(_:)`` and hands that normed value both to
    /// `lm_head` and to the next recursive draft step (vLLM semantics).
    func preNormHidden(
        embeddings: MLXArray,
        hiddenStates: MLXArray,
        cache: [KVCache]?
    ) -> MLXArray {
        guard !layers.isEmpty else { return hiddenStates }
        var h = layers[0].fuse(embeddings: embeddings, hiddenStates: hiddenStates)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (index, layer) in layers.enumerated() {
            h = layer(h, attentionMask: mask, cache: cache?[index])
        }
        return h
    }

    /// `mtp.layers.<last>.final_layernorm`, applied only on the way to logits.
    func finalNorm(_ x: MLXArray) -> MLXArray {
        layers.last?.finalLayerNorm.map { $0(x) } ?? x
    }
}

// MARK: - Backbone (matches Python's NemotronHModel which is stored as self.backbone)

internal class NemotronHBackbone: Module {
    let args: NemotronHConfiguration

    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [NemotronHBlock]
    @ModuleInfo(key: "norm_f") var normF: RMSNorm

    // Cache indices (into the cache list, not pattern indices)
    // Python: fa_idx counts Mamba layers before first Attention
    // Python: ssm_idx counts Attention layers before first Mamba
    let firstAttentionCacheIndex: Int?
    let firstMambaCacheIndex: Int?

    /// Optional recorder of block OUTPUTS (before `norm_f`), e.g. the LiLiCorr drafter's target
    /// taps. Nil by default: the loop then pays one optional check per block and nothing else.
    var blockOutputTap: NemotronHBlockOutputTap? = nil

    init(
        _ args: NemotronHConfiguration, jangtq: NemotronHJANGTQContext? = nil,
        routed: [Int: any WeightedRoutedExpertLayer] = [:]
    ) {
        self.args = args
        precondition(args.vocabSize > 0)

        self._embeddings.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)

        let pattern = Array(args.hybridOverridePattern)
        self._layers.wrappedValue = pattern.enumerated().map { layerIdx, blockType in
            NemotronHBlock(
                args, layerIdx: layerIdx, blockType: blockType, jangtq: jangtq,
                routed: routed[layerIdx])
        }

        self._normF.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)

        // Calculate cache indices (only Mamba + Attention have caches)
        // fa_idx: count Mamba layers (M) before the first Attention layer (*)
        var faIdx: Int? = nil
        var mambaCount = 0
        for char in pattern {
            if char == "*" {
                faIdx = mambaCount
                break
            } else if char == "M" {
                mambaCount += 1
            }
        }
        self.firstAttentionCacheIndex = faIdx

        // ssm_idx: count Attention layers (*) before the first Mamba layer (M)
        var ssmIdx: Int? = nil
        var attnCount = 0
        for char in pattern {
            if char == "M" {
                ssmIdx = attnCount
                break
            } else if char == "*" {
                attnCount += 1
            }
        }
        self.firstMambaCacheIndex = ssmIdx

        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let hidden = embeddings(inputs)
        return forwardFromEmbeddings(hidden, cache: cache)
    }

    /// Forward starting from a pre-computed embedding tensor (for multimodal splice).
    func forwardFromEmbeddings(_ inputsEmbeds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        normF(preNormFromEmbeddings(inputsEmbeds, cache: cache))
    }

    /// The layer stack, stopping before `norm_f`. Both forwards share this
    /// single copy of the loop.
    func preNormFromEmbeddings(_ inputsEmbeds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden = inputsEmbeds

        // Create attention mask using the first attention layer's cache
        // A cache-less multi-token forward (teacher-forced scoring, a
        // stateless re-encode) still needs the causal mask: returning `.none`
        // there made every attention row see the future tokens.
        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = {
            guard firstAttentionCacheIndex != nil else { return .none }
            guard let cacheIdx = firstAttentionCacheIndex,
                let cache = cache,
                cacheIdx < cache.count
            else { return hidden.dim(1) > 1 ? .causal : .none }
            return createAttentionMask(h: hidden, cache: cache[cacheIdx])
        }()

        // Create SSM mask using the first Mamba layer's cache
        // Python: ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])
        let ssmMask: MLXArray? = {
            guard let cacheIdx = firstMambaCacheIndex,
                let cache = cache,
                cacheIdx < cache.count,
                let mambaCache = cache[cacheIdx] as? MambaCache
            else { return nil }
            return mambaCache.makeMask(N: hidden.dim(1))
        }()

        // Track which cache to use for each layer
        var cacheCounter = 0
        let tap: NemotronHBlockOutputTap? = blockOutputTap.flatMap { tap in
            (tap.begin?(cache, hidden.dim(1)) ?? true) ? tap : nil
        }
        defer { tap?.end?() }
        for (layerIndex, layer) in layers.enumerated() {
            let c: KVCache?
            if layer.blockType == .mamba || layer.blockType == .attention {
                c = cache?[cacheCounter]
                cacheCounter += 1
            } else {
                c = nil
            }

            hidden = layer(hidden, attentionMask: attentionMask, ssmMask: ssmMask, cache: c)
            if let tap, tap.layerIndices.contains(layerIndex) {
                tap.record(layerIndex, hidden)
            }
        }

        return hidden
    }

    /// Pre-final-norm hidden from token ids.
    func preNorm(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        preNormFromEmbeddings(embeddings(inputs), cache: cache)
    }

    /// Commit a staged verify block on every Mamba layer. Cache slots follow
    /// the forward's order (only Mamba and attention blocks own one);
    /// attention caches are trimmed by the iterator.
    func commitStagedVerifiedBlock(
        cache: [KVCache], acceptedInputs: Int, blockLength: Int
    ) -> Bool {
        var cacheCounter = 0
        for layer in layers {
            guard layer.blockType == .mamba || layer.blockType == .attention else { continue }
            defer { cacheCounter += 1 }
            guard layer.blockType == .mamba else { continue }
            guard cacheCounter < cache.count,
                let mamba = cache[cacheCounter] as? MambaCache,
                let mixer = layer.mixer as? NemotronHMamba2Mixer,
                mixer.commitVerifyStaged(
                    cache: mamba, acceptedInputs: acceptedInputs, blockLength: blockLength)
            else { return false }
        }
        return true
    }

}

// MARK: - Main Model (matches Python's Model class)

public class NemotronHModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel,
    TokenEmbedderModel, NativeMTPModel, DFlash2StagedVerifyRollbackModel,
    NemotronHBlockOutputTappable
{
    public let vocabularySize: Int
    public let kvHeads: [Int]

    /// `NemotronHBlockOutputTappable`: block-output recorder on the backbone (nil by default).
    public var blockOutputTap: NemotronHBlockOutputTap? {
        get { backbone.blockOutputTap }
        set { backbone.blockOutputTap = newValue }
    }

    @ModuleInfo(key: "backbone") private var backbone: NemotronHBackbone
    public let configuration: NemotronHConfiguration
    public let jangtqContext: NemotronHJANGTQContext?

    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    /// Native multi-token-prediction head. Present only when the bundle
    /// declares `mtp_layers_block_type`; absent everywhere else, which is what
    /// makes `nativeMTPAvailable` honest rather than aspirational.
    @ModuleInfo(key: "mtp") var mtp: NemotronHMTP?
    /// Raw checkpoint keys owned by mapped custom banks (JANGH); the generic
    /// safetensors load must not read them.
    let excludedSafetensorsKeys: Set<String>

    public var loraLayers: [Module] {
        backbone.layers
    }

    public convenience init(_ args: NemotronHConfiguration) {
        self.init(args, routedLayers: [:], excludedSafetensorsKeys: [])
    }

    /// JANGH relu2 bundles (Nemotron 3.5 Bolt): `routedLayers` holds the
    /// admitted mapped bank per routed block; `excludedSafetensorsKeys` the raw
    /// tensor names those banks own.
    public init(
        _ args: NemotronHConfiguration,
        routedLayers: [Int: any WeightedRoutedExpertLayer],
        excludedSafetensorsKeys: Set<String>
    ) {
        self.configuration = args
        self.jangtqContext = nil
        self.vocabularySize = args.vocabSize
        self.excludedSafetensorsKeys = excludedSafetensorsKeys

        // kvHeads array: non-zero for attention layers, zero for others
        let pattern = Array(args.hybridOverridePattern)
        self.kvHeads = pattern.compactMap { char -> Int? in
            let blockType = NemotronHBlockType(from: char)
            if blockType == .mamba || blockType == .attention {
                return blockType == .attention ? args.numKeyValueHeads : 0
            }
            return nil
        }

        self._backbone.wrappedValue = NemotronHBackbone(args, jangtq: nil, routed: routedLayers)
        if nemotronHNativeMTPEnabled(), !args.mtpLayersBlockType.isEmpty {
            self._mtp.wrappedValue = NemotronHMTP(args, jangtq: nil)
        }

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
        }
    }

    /// JANGTQ-flavored init — replaces routed-expert `SwitchLinear` fc1/fc2
    /// inside every MoE layer with `TurboQuantSwitchLinear` so the codebook
    /// kernels run instead of `gather_qmm`. Use for Cascade-2 / Nemotron-Omni
    /// JANGTQ bundles. The factory layer detects the bundle via
    /// `jang_config.weight_format == "mxtq"` and dispatches here.
    public init(jangtqContext: NemotronHJANGTQContext, configuration args: NemotronHConfiguration) {
        self.configuration = args
        self.jangtqContext = jangtqContext
        self.vocabularySize = args.vocabSize
        self.excludedSafetensorsKeys = []
        let pattern = Array(args.hybridOverridePattern)
        self.kvHeads = pattern.compactMap { char -> Int? in
            let blockType = NemotronHBlockType(from: char)
            if blockType == .mamba || blockType == .attention {
                return blockType == .attention ? args.numKeyValueHeads : 0
            }
            return nil
        }
        self._backbone.wrappedValue = NemotronHBackbone(args, jangtq: jangtqContext)
        if nemotronHNativeMTPEnabled(), !args.mtpLayersBlockType.isEmpty {
            self._mtp.wrappedValue = NemotronHMTP(args, jangtq: jangtqContext)
        }
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = backbone(inputs, cache: cache)
        if let lmHead {
            if nemotronHActivationBF16RetentionEnabled(),
                out.dtype != lmHead.computeDType
            {
                out = out.asType(lmHead.computeDType)
            }
            out = lmHead(out)
        } else {
            out = backbone.embeddings.asLinear(out)
        }
        if nemotronHLayerProfileEnabled() {
            MLX.eval(out)
            NemotronHLayerProfiler.shared.printSummary()
        }
        return out
    }

    /// Look up text token embeddings (used for multimodal splice).
    public func embedTokens(_ tokens: MLXArray) -> MLXArray {
        backbone.embeddings(tokens)
    }

    /// Forward starting from pre-computed embeddings (for multimodal splice).
    /// Returns logits in the same shape as ``callAsFunction(_:cache:)``.
    public func callAsFunction(inputsEmbeds: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = backbone.forwardFromEmbeddings(inputsEmbeds, cache: cache)
        if let lmHead {
            if nemotronHActivationBF16RetentionEnabled(),
                out.dtype != lmHead.computeDType
            {
                out = out.asType(lmHead.computeDType)
            }
            out = lmHead(out)
        } else {
            out = backbone.embeddings.asLinear(out)
        }
        if nemotronHLayerProfileEnabled() {
            MLX.eval(out)
            NemotronHLayerProfiler.shared.printSummary()
        }
        return out
    }

    // MARK: - Native MTP

    /// The head exists only when the bundle shipped one, so speculation can
    /// never engage on a Nemotron build without MTP weights.
    public var nativeMTPAvailable: Bool { mtp != nil || lilicorrLane != nil }

    /// Bolt's LiLiCorr block drafter (`lilicorr/`), attached only for an explicit
    /// native-MTP load of a bundle without a recursive head.
    public private(set) var lilicorrLane: NemotronLiLiCorrLane?

    public var hasRecursiveMTPHead: Bool { mtp != nil }

    public func attachLiLiCorr(drafter: NemotronLiLiCorrDrafter) throws {
        guard let attentionIndex = backbone.firstAttentionCacheIndex else {
            throw NemotronLiLiCorrError.malformedConfig("target has no attention layer for positions")
        }
        let layerCount = backbone.layers.count
        guard drafter.targetLayerIds.allSatisfy({ $0 >= 0 && $0 < layerCount }),
            drafter.tapDimension == drafter.targetLayerIds.count * configuration.hiddenSize
        else {
            throw NemotronLiLiCorrError.malformedConfig(
                "target_layer_ids \(drafter.targetLayerIds) / tap width \(drafter.tapDimension) do not fit a \(layerCount)-block target of hidden \(configuration.hiddenSize)")
        }
        // The block is embedded with the TARGET table: the mask id must be a target token, and
        // the block must fit the head's slot embedding (both read from the checkpoint).
        guard drafter.maskTokenId >= 0, drafter.maskTokenId < configuration.vocabSize,
            drafter.blockSize >= 2,
            drafter.lilicorr.map({ drafter.blockSize - 1 <= $0.slotCapacity }) ?? true
        else {
            throw NemotronLiLiCorrError.malformedConfig(
                "mask_token_id \(drafter.maskTokenId) / block_size \(drafter.blockSize) do not fit the target vocabulary \(configuration.vocabSize) or the head's slot embedding")
        }
        let lane = NemotronLiLiCorrLane(drafter: drafter, attentionCacheIndex: attentionIndex)
        backbone.blockOutputTap = lane.makeTap()
        lilicorrLane = lane
    }

    /// Per-request cache for the head alone. Never prefix, paged or L2 state —
    /// the draft must not be able to contaminate the base cache.
    public func makeNativeMTPCache() -> [KVCache] {
        mtp?.makeCache() ?? []
    }

    /// `TokenEmbedderModel`: the same embedding the backbone's own forward
    /// uses as its first-layer input. `embedTokens` already exists on this type
    /// for other callers, so this is a rename rather than a second path.
    public func embed(_ tokenIds: MLXArray) -> MLXArray {
        embedTokens(tokenIds)
    }

    public func projectToLogits(_ hidden: MLXArray) -> MLXArray {
        if let lmHead { return NemotronHRowExact.project(lmHead, hidden, site: "lm_head") }
        return backbone.embeddings.asLinear(hidden)
    }

    /// Nemotron-H's MTP head consumes the target's hidden state AFTER `norm_f`
    /// (the same vector `lm_head` reads): vLLM `nemotron_h_mtp.py` receives the
    /// model's normed output, and on Nemotron 3.5 Super the measured top-1 vs
    /// token t+2 is 85.1 % with the post-norm input and far lower with the
    /// pre-norm one (bundle `jang_config.mtp.heldout.h_input = "post"`). The
    /// iterator treats `hiddenStates` as opaque, so this family returns the
    /// normed value here and from ``nativeMTPForward(hiddenStates:nextTokenIds:cache:)``.
    public func nativeBackboneForward(
        _ inputs: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPForwardResult {
        let hidden = backbone.normF(backbone.preNorm(inputs, cache: cache))
        return NativeMTPForwardResult(
            logits: projectToLogits(hidden),
            hiddenStates: hidden)
    }

    /// Nemotron's hybrid stack carries Mamba convolution/SSM state that cannot
    /// be trimmed after the fact, so the verifier forward is the same forward.
    /// The iterator decides the accepted prefix from the returned logits; any
    /// rollback is the cache layer's job, not this method's.
    public func nativeBackboneMTPVerifyForward(
        _ inputs: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPForwardResult {
        // The verify forward ONLY opens the row-exact scope (2...8 rows): every verify row then
        // reproduces the AR step bit for bit (``NemotronHRowExact``), and small verify windows of
        // mapped JANGH experts keep the decode kernels instead of the sorted prefill path.
        FlashVerificationScope.withVerification(inputShape: inputs.shape) {
            nativeBackboneForward(inputs, cache: cache)
        }
    }

    public func nativeMTPForward(
        hiddenStates: MLXArray,
        nextTokenIds: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPForwardResult {
        guard let mtp else {
            fatalError("NemotronH nativeMTPForward called without an MTP head")
        }
        // Recursive depth feeds the head's own output AFTER `final_layernorm`,
        // as vLLM does (the MTP layer returns the end-normed state and
        // `compute_logits` applies only `lm_head`).
        let hidden = mtp.finalNorm(
            mtp.preNormHidden(
                embeddings: embedTokens(nextTokenIds),
                hiddenStates: hiddenStates,
                cache: cache))
        return NativeMTPForwardResult(
            logits: projectToLogits(hidden),
            hiddenStates: hidden)
    }

    // MARK: - Staged verify rollback (native MTP on the hybrid stack)

    /// Input-capture (unstaged) rollback is not implemented for this family;
    /// the iterator selects the staged mode for it.
    public func commitVerifiedBlock(cache: [KVCache], acceptedInputs: Int) -> Bool {
        false
    }

    public func commitStagedVerifiedBlock(
        cache: [KVCache], acceptedInputs: Int, blockLength: Int
    ) -> Bool {
        backbone.commitStagedVerifiedBlock(
            cache: cache, acceptedInputs: acceptedInputs, blockLength: blockLength)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        // 2026-05-01: honor `parameters.maxKVSize` for the attention slots.
        // Audit found NemotronH previously ignored maxKVSize, leaving the
        // CacheCoordinator's `defaultMaxKVSize` contract a silent no-op
        // for the Cascade-2 / Nemotron-Omni hybrid family. Mamba layers
        // ignore the bound by design (their hidden state is fixed-size).
        let pattern = Array(configuration.hybridOverridePattern)
        return pattern.compactMap { char -> KVCache? in
            let blockType = NemotronHBlockType(from: char)
            switch blockType {
            case .mamba:
                return MambaCache()
            case .attention:
                if let maxKVSize = parameters?.maxKVSize {
                    return RotatingKVCache(maxSize: maxKVSize, keep: 4)
                }
                return KVCacheSimple()
            case .mlp, .moe:
                return nil  // No cache needed for MLP/MoE layers
            }
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, value) in weights {
            // Nemotron-H bundles can ship MTP speculative-decoding heads
            // (`mtp.*`). They are dropped unless native MTP is switched on,
            // because instantiating the head is what makes the weights
            // bindable — admitting them while no module exists would fail
            // `verify` instead of being ignored.
            if key.hasPrefix("mtp.") {
                if !nemotronHNativeMTPEnabled() || mtp == nil { continue }
            }
            if key.hasSuffix(".importance") {
                continue
            }
            if key.hasPrefix("vision_model.")
                || key.hasPrefix("sound_encoder.")
                || key.hasPrefix("sound_projection.")
                || key.hasPrefix("mlp1.")
                || key.hasPrefix("vision_projector.")
            {
                continue
            }

            var finalValue = value

            // Handle conv1d weight axis swap
            if key.contains("conv1d.weight"), value.dim(-1) != 1 {
                finalValue = value.swappedAxes(1, 2)
            }

            // 2026-05-01: Tied-embeddings — drop redundant `lm_head.*`
            // when the LM shares weights with `backbone.embeddings`.
            // Init guard at line 783-785 already declines to allocate
            // `lmHead`; the redundant tensors otherwise survived only
            // because `verify: [.noUnusedKeys]` silently absorbed
            // them. Forward-compat: also drops scales/biases for
            // bundles that quantize lm_head despite tied embeddings.
            if configuration.tieWordEmbeddings,
                key == "lm_head.weight"
                    || key == "lm_head.scales"
                    || key == "lm_head.biases"
            {
                continue
            }

            sanitized[key] = finalValue
        }

        // JANG models have pre-stacked expert weights as switch_mlp.{down,up}_proj
        // but the module tree expects switch_mlp.{fc2,fc1}. Remap key names.
        let jangExpertRenames: [(String, String)] = [
            (".switch_mlp.down_proj.", ".switch_mlp.fc2."),
            (".switch_mlp.up_proj.", ".switch_mlp.fc1."),
        ]
        var remapped = [String: MLXArray]()
        for (key, value) in sanitized {
            var newKey = key
            for (from, to) in jangExpertRenames {
                if newKey.contains(from) {
                    newKey = newKey.replacingOccurrences(of: from, with: to)
                }
            }
            remapped[newKey] = value
        }
        sanitized = remapped

        // Stack per-expert weights (HF format): backbone.layers.{l}.mixer.experts.{e}.{proj}.weight
        for l in 0 ..< configuration.numHiddenLayers {
            let prefix = "backbone.layers.\(l).mixer"
            for (m, n) in [("down_proj", "fc2"), ("up_proj", "fc1")] {
                if sanitized["\(prefix).experts.0.\(m).weight"] != nil {
                    let toJoin = (0 ..< configuration.moeGeometry(layer: l).experts).compactMap { e in
                        sanitized.removeValue(forKey: "\(prefix).experts.\(e).\(m).weight")
                    }
                    if !toJoin.isEmpty {
                        sanitized["\(prefix).switch_mlp.\(n).weight"] =
                            loadTimeMaterializedStacked(toJoin)
                    }
                }
            }
        }

        // JANGTQ per-expert TurboQuant tensors:
        //   backbone.layers.{l}.mixer.experts.{e}.{up,down}_proj.{tq_packed,tq_norms,tq_bits}
        // → backbone.layers.{l}.mixer.switch_mlp.{fc1,fc2}.{tq_packed,tq_norms}
        // (tq_bits is metadata — dropped)
        // Mirrors `jang_tools.load_jangtq` § "MoE stacking" (nemo_pat).
        // Pre-stacked variants (MLX-native conversions where converter
        // already produced the stacked layout) skip the loop because
        // `experts.0.up_proj.tq_packed` won't exist; the keys go through
        // unchanged via the early loop above.
        let tqMap: [(String, String)] = [("up_proj", "fc1"), ("down_proj", "fc2")]
        var tqStackedCount = 0
        for l in 0 ..< configuration.numHiddenLayers {
            let prefix = "backbone.layers.\(l).mixer"
            for (proj, fc) in tqMap {
                let probe = "\(prefix).experts.0.\(proj).tq_packed"
                guard sanitized[probe] != nil else { continue }
                for kind in ["tq_packed", "tq_norms"] {
                    let target = "\(prefix).switch_mlp.\(fc).\(kind)"
                    let streamJANGTQExperts =
                        JANGTQStreamingExperts.isEnabled
                        || JANGTQStreamingExperts.shouldAutoEnableNemotronUltra(
                            nRoutedExperts: configuration.nRoutedExperts,
                            numExpertsPerTok: configuration.numExpertsPerTok,
                            routedBits: nil)
                    if streamJANGTQExperts {
                        for e in 0 ..< configuration.nRoutedExperts {
                            sanitized.removeValue(
                                forKey: "\(prefix).experts.\(e).\(proj).\(kind)")
                        }
                        continue
                    }
                    if sanitized[target] != nil {
                        for e in 0 ..< configuration.nRoutedExperts {
                            sanitized.removeValue(
                                forKey: "\(prefix).experts.\(e).\(proj).\(kind)")
                        }
                        continue
                    }
                    let stacked: [MLXArray] = (0 ..< configuration.nRoutedExperts).compactMap { e in
                        sanitized.removeValue(
                            forKey: "\(prefix).experts.\(e).\(proj).\(kind)")
                    }
                    if !stacked.isEmpty {
                        sanitized[target] = loadTimeMaterializedStacked(stacked)
                        tqStackedCount += 1
                    }
                }
                // Drop the per-expert tq_bits metadata.
                for e in 0 ..< configuration.nRoutedExperts {
                    sanitized.removeValue(forKey: "\(prefix).experts.\(e).\(proj).tq_bits")
                }
            }
        }
        // Strip any remaining `.tq_bits` keys anywhere in the tree.
        for key in Array(sanitized.keys) where key.hasSuffix(".tq_bits") {
            sanitized.removeValue(forKey: key)
        }

        if tqStackedCount > 0 {
            FileHandle.standardError.write(
                Data("[NemotronH.sanitize] stacked \(tqStackedCount) JANGTQ tensor groups (per-expert tq_packed/tq_norms → switch_mlp.{fc1,fc2}.{tq_*})\n".utf8))
        }
        _ = tqStackedCount  // silence unused warning when log is gated

        return sanitized
    }

    /// Predicate for casting: some parameters should stay in float32
    public var castPredicate: ((String) -> Bool)? {
        { key in
            !key.contains("e_score_correction_bias") && !key.contains("A_log")
        }
    }
}

// MARK: - Configuration

public struct NemotronHConfiguration: Codable, Sendable {
    public var modelType: String = "nemotron_h"
    public var vocabSize: Int
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var attentionBias: Bool
    public var mambaNumHeads: Int
    public var mambaHeadDim: Int
    public var mambaProjBias: Bool
    public var ssmStateSize: Int
    public var convKernel: Int
    public var nGroups: Int
    public var intermediateSize: Int
    public var moeIntermediateSize: Int
    public var moeLatentSize: Int?
    public var moeSharedExpertIntermediateSize: Int
    public var nRoutedExperts: Int
    public var nSharedExperts: Int?
    public var numExpertsPerTok: Int
    public var hybridOverridePattern: String
    public var layerNormEpsilon: Float
    public var mlpBias: Bool
    public var useBias: Bool
    public var useConvBias: Bool
    public var tieWordEmbeddings: Bool
    public var ropeTheta: Float
    public var headDim: Int?
    public var nGroup: Int
    public var topkGroup: Int
    public var normTopkProb: Bool
    public var routedScalingFactor: Float
    public var timeStepLimitMin: Float
    public var timeStepLimitMax: Float

    /// Number of next-token-prediction heads the bundle ships. Nemotron 3.5
    /// Lightning ships 1; depth beyond that is produced by re-entering the same
    /// head, exactly as the DeepSeek-style MTP module does.
    public var numNextnPredictLayers: Int
    /// Block kinds inside the MTP head, e.g. `["attention", "moe"]`. Mapped
    /// through the same letters as `hybridOverridePattern` so the head reuses
    /// `NemotronHBlock` and the weights land on their shipped key paths.
    public var mtpLayersBlockType: [String]

    /// Per-block MoE geometry (Nemotron 3.5 Bolt "puzzle" blocks). Each MoE
    /// block declares its own expert count, intermediate width and top-k; the
    /// config's global MoE fields are absent or stale (`num_experts: 512`) on
    /// such bundles. `nil` on uniform bundles.
    public var blockConfigs: [NemotronHBlockConfig]?

    /// Expert count, intermediate width and top-k of `layer`'s MoE mixer.
    public func moeGeometry(layer: Int) -> (experts: Int, intermediate: Int, topK: Int) {
        if let blockConfigs, layer < blockConfigs.count, blockConfigs[layer].blockType == "moe" {
            let block = blockConfigs[layer]
            return (
                block.nRoutedExperts ?? nRoutedExperts,
                block.moeIntermediateSize ?? moeIntermediateSize,
                RuntimeMoETopKOverride.effectiveTopK(
                    currentTopK: block.numExpertsPerTok ?? numExpertsPerTok,
                    modelType: modelType, field: "num_experts_per_tok"))
        }
        return (nRoutedExperts, moeIntermediateSize, numExpertsPerTok)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case attentionBias = "attention_bias"
        case mambaNumHeads = "mamba_num_heads"
        case mambaHeadDim = "mamba_head_dim"
        case mambaProjBias = "mamba_proj_bias"
        case ssmStateSize = "ssm_state_size"
        case convKernel = "conv_kernel"
        case nGroups = "n_groups"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeLatentSize = "moe_latent_size"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case hybridOverridePattern = "hybrid_override_pattern"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case mtpLayersBlockType = "mtp_layers_block_type"
        case layerNormEpsilon = "layer_norm_epsilon"
        case mlpBias = "mlp_bias"
        case useBias = "use_bias"
        case useConvBias = "use_conv_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case timeStepLimitMin = "time_step_limit_min"
        case timeStepLimitMax = "time_step_limit_max"
    }

    private enum RawCodingKeys: String, CodingKey {
        case timeStepLimit = "time_step_limit"
        case layersBlockType = "layers_block_type"
        case blockConfigs = "block_configs"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let rawContainer = try decoder.container(keyedBy: RawCodingKeys.self)

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "nemotron_h"
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mambaNumHeads = try container.decode(Int.self, forKey: .mambaNumHeads)
        mambaHeadDim = try container.decode(Int.self, forKey: .mambaHeadDim)
        mambaProjBias = try container.decodeIfPresent(Bool.self, forKey: .mambaProjBias) ?? false
        ssmStateSize = try container.decode(Int.self, forKey: .ssmStateSize)
        convKernel = try container.decode(Int.self, forKey: .convKernel)
        nGroups = try container.decode(Int.self, forKey: .nGroups)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        moeLatentSize = try container.decodeIfPresent(Int.self, forKey: .moeLatentSize)
        moeSharedExpertIntermediateSize = try container.decode(
            Int.self, forKey: .moeSharedExpertIntermediateSize)
        nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        // Puzzle bundles carry the MoE geometry per block only; the globals are
        // then filled from the first MoE block (vMLX Python PuzzleModelArgs).
        blockConfigs = try rawContainer.decodeIfPresent(
            [NemotronHBlockConfig].self, forKey: .blockConfigs)
        let firstMoE = blockConfigs?.first { $0.blockType == "moe" }
        guard let experts = try container.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
                ?? firstMoE?.nRoutedExperts,
            let moeIntermediate = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
                ?? firstMoE?.moeIntermediateSize,
            let topK = try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
                ?? firstMoE?.numExpertsPerTok
        else {
            throw DecodingError.dataCorruptedError(
                forKey: .nRoutedExperts, in: container,
                debugDescription: "n_routed_experts / moe_intermediate_size / num_experts_per_tok missing globally and in block_configs")
        }
        nRoutedExperts = experts
        moeIntermediateSize = moeIntermediate
        numExpertsPerTok = RuntimeMoETopKOverride.effectiveTopK(
            currentTopK: topK,
            modelType: modelType,
            field: CodingKeys.numExpertsPerTok.rawValue)
        layerNormEpsilon =
            try container.decodeIfPresent(Float.self, forKey: .layerNormEpsilon) ?? 1e-5
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        useBias = try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        useConvBias = try container.decodeIfPresent(Bool.self, forKey: .useConvBias) ?? true
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try container.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        routedScalingFactor =
            try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0

        // Absent on every non-MTP Nemotron bundle, so both default to "no head"
        // and the model simply reports `nativeMTPAvailable == false`.
        numNextnPredictLayers =
            try container.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 0
        mtpLayersBlockType =
            try container.decodeIfPresent([String].self, forKey: .mtpLayersBlockType) ?? []

        // Handle hybrid_override_pattern - can be string or array of strings.
        // Nemotron 3 Ultra instead ships mlx-lm's `layers_block_type`
        // string array: mamba / attention / moe / mlp.
        if let patternString = try? container.decode(String.self, forKey: .hybridOverridePattern) {
            hybridOverridePattern = patternString
        } else if let patternArray = try? container.decode(
            [String].self, forKey: .hybridOverridePattern)
        {
            hybridOverridePattern = patternArray.joined()
        } else if let layerTypes = try? rawContainer.decode([String].self, forKey: .layersBlockType) {
            hybridOverridePattern = try Self.pattern(fromLayerBlockTypes: layerTypes)
        } else {
            throw DecodingError.dataCorruptedError(
                forKey: .hybridOverridePattern, in: container,
                debugDescription: "hybrid_override_pattern must be string/array, or layers_block_type must be present")
        }
        numHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers)
            ?? hybridOverridePattern.count
        if let blockConfigs {
            let pattern = Array(hybridOverridePattern)
            guard blockConfigs.count == pattern.count,
                zip(blockConfigs, pattern).allSatisfy({ config, char in
                    (try? Self.pattern(fromLayerBlockTypes: [config.blockType])) == String(char)
                }),
                blockConfigs.allSatisfy({ $0.blockType != "moe"
                    || ($0.nRoutedExperts ?? 0) > 0 && ($0.moeIntermediateSize ?? 0) > 0
                        && ($0.numExpertsPerTok ?? 0) > 0 })
            else {
                throw DecodingError.dataCorruptedError(
                    forKey: .hybridOverridePattern, in: container,
                    debugDescription: "block_configs disagree with the layer pattern or lack MoE geometry")
            }
        }

        // mlx-lm stores Nemotron-H as `time_step_limit: [min, max]` and clips
        // dt to exactly that range (`compute_dt`, mlx-lm 0.31: a missing limit
        // defaults to (0, inf)). Raising a 0.0 floor to 0.001 here changed
        // every small time step relative to the reference.
        if let limits = try? rawContainer.decode([Float].self, forKey: .timeStepLimit),
           let first = limits.first
        {
            timeStepLimitMin = max(0, first)
            timeStepLimitMax = limits.count > 1 ? limits[1] : first
        } else {
            let decodedMin = try container.decodeIfPresent(Float.self, forKey: .timeStepLimitMin) ?? 0
            timeStepLimitMin = max(0, decodedMin)
            timeStepLimitMax =
                try container.decodeIfPresent(Float.self, forKey: .timeStepLimitMax)
                ?? Float.infinity
        }
    }

    private static func pattern(fromLayerBlockTypes layerTypes: [String]) throws -> String {
        try layerTypes.map { raw in
            switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
            case "mamba", "m": return "M"
            case "attention", "attn", "*": return "*"
            case "moe", "expert", "e": return "E"
            case "mlp", "dense", "-": return "-"
            default:
                throw DecodingError.dataCorrupted(
                    DecodingError.Context(
                        codingPath: [],
                        debugDescription: "Unknown NemotronH layers_block_type entry: \(raw)"))
            }
        }.joined()
    }

    /// Memberwise initializer for testing
    public init(
        vocabSize: Int,
        hiddenSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        mambaNumHeads: Int,
        mambaHeadDim: Int,
        ssmStateSize: Int,
        convKernel: Int,
        nGroups: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        moeLatentSize: Int? = nil,
        moeSharedExpertIntermediateSize: Int,
        nRoutedExperts: Int,
        numExpertsPerTok: Int,
        hybridOverridePattern: String,
        layerNormEpsilon: Float = 1e-5,
        attentionBias: Bool = false,
        mambaProjBias: Bool = false,
        mlpBias: Bool = false,
        useBias: Bool = false,
        useConvBias: Bool = true,
        tieWordEmbeddings: Bool = false,
        ropeTheta: Float = 10000.0,
        headDim: Int? = nil,
        nSharedExperts: Int? = nil,
        nGroup: Int = 1,
        topkGroup: Int = 1,
        normTopkProb: Bool = true,
        routedScalingFactor: Float = 1.0,
        timeStepLimitMin: Float = 0.0,
        timeStepLimitMax: Float = .infinity
    ) {
        self.modelType = "nemotron_h"
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.attentionBias = attentionBias
        self.mambaNumHeads = mambaNumHeads
        self.mambaHeadDim = mambaHeadDim
        self.mambaProjBias = mambaProjBias
        self.ssmStateSize = ssmStateSize
        self.convKernel = convKernel
        self.nGroups = nGroups
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.moeLatentSize = moeLatentSize
        self.moeSharedExpertIntermediateSize = moeSharedExpertIntermediateSize
        self.nRoutedExperts = nRoutedExperts
        self.nSharedExperts = nSharedExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.hybridOverridePattern = hybridOverridePattern
        self.layerNormEpsilon = layerNormEpsilon
        self.mlpBias = mlpBias
        self.useBias = useBias
        self.useConvBias = useConvBias
        self.tieWordEmbeddings = tieWordEmbeddings
        self.ropeTheta = ropeTheta
        self.headDim = headDim
        self.nGroup = nGroup
        self.topkGroup = topkGroup
        self.normTopkProb = normTopkProb
        self.routedScalingFactor = routedScalingFactor
        self.timeStepLimitMin = timeStepLimitMin
        self.timeStepLimitMax = timeStepLimitMax
        // Memberwise callers are non-MTP fixtures; the head stays absent
        // unless a bundle's JSON declares it.
        self.numNextnPredictLayers = 0
        self.mtpLayersBlockType = []
        self.blockConfigs = nil
    }
}

/// One entry of Nemotron-H `block_configs` (puzzle architectures).
public struct NemotronHBlockConfig: Codable, Sendable, Equatable {
    public var blockType: String
    public var nRoutedExperts: Int?
    public var moeIntermediateSize: Int?
    public var numExpertsPerTok: Int?

    enum CodingKeys: String, CodingKey {
        case blockType = "block_type"
        case nRoutedExperts = "n_routed_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case numExpertsPerTok = "num_experts_per_tok"
    }
}


extension NemotronHModel: NativeMTPAutoregressiveBackboneModel {
    /// The AR step is the ordinary single-row backbone forward; declaring it lets the
    /// row-exactness bench (BENCH_ROWEXACT) compare verify rows with it.
    public func nativeAutoregressiveBackboneForward(
        _ inputs: MLXArray, cache: [KVCache]?
    ) -> NativeMTPForwardResult {
        nativeBackboneForward(inputs, cache: cache)
    }
}

extension NemotronHModel: NativeMTPBlockDraftModel {
    public var nativeMTPUsesBlockDrafts: Bool { mtp == nil && lilicorrLane != nil }

    public var nativeMTPBlockDraftDepthLimit: Int {
        Swift.max(1, (lilicorrLane?.drafter.blockSize ?? 2) - 1)
    }

    public var nativeMTPBlockDraftVocabularySize: Int { configuration.vocabSize }

    public func nativeMTPBlockDraft(nextToken: MLXArray, depth: Int, targetCache: [KVCache]) -> MLXArray {
        guard let lane = lilicorrLane else {
            fatalError("NemotronH block draft requested without an attached LiLiCorr drafter")
        }
        return lane.draft(
            nextToken: nextToken, depth: depth, targetCache: targetCache,
            embed: { self.embedTokens($0) }, lmHead: { self.projectToLogits($0) })
    }
}

extension NemotronHModel: SafetensorsLoadKeyExcluding {
    public func excludeFromGenericSafetensorsLoad(key: String) -> Bool {
        excludedSafetensorsKeys.contains(key)
    }

    public var requiresExactTensorMmapBuffers: Bool { !excludedSafetensorsKeys.isEmpty }
}
