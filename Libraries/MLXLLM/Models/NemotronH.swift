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

private func nemotronHActivationBF16RetentionEnabled() -> Bool {
    !nemotronHEnvFlag("JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16")
}

private func nemotronHWeightedMoEFastPathEnabled() -> Bool {
    !nemotronHEnvFlag("JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH")
}

private func nemotronHLayerProfileEnabled() -> Bool {
    nemotronHEnvFlag("VMLINUX_NEMOTRON_LAYER_PROFILE")
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
    nemotronHEnvFlag("VMLINUX_DISABLE_NEMOTRON_MAMBA_CONV_FASTPATH")
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
            acc += static_cast<float>(bias[c]);

            auto sigmoid = 1.0f / (1.0f + fast::exp(-acc));
            out[b * C + c] = static_cast<T>(acc * sigmoid);

            auto next_base = (b * (K - 1)) * C + c;
            for (int k = 0; k < K - 2; ++k) {
                next_state[next_base + k * C] = state[state_base + (k + 1) * C];
            }
            if (K > 1) {
                next_state[next_base + (K - 2) * C] = input[b * C + c];
            }
        """

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

// MARK: - Mamba2Mixer

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

    private func applyConv(_ input: MLXArray, mask: MLXArray?, cache: MambaCache?) -> MLXArray {
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
            cache[0] = padded[0..., start ..< end, 0...]
        }

        let convOutput = conv1d(padded)
        return silu(convOutput)
    }

    private func mambaForward(
        _ hiddenStates: MLXArray,
        mask: MLXArray?,
        cache: MambaCache?
    ) -> MLXArray {
        let projected = inProj(hiddenStates)
        let splits = split(
            projected, indices: [intermediateSize, intermediateSize + convDim], axis: -1)
        let gate = splits[0]
        let convInput = splits[1]
        let dt = splits[2]

        let convOutput = applyConv(convInput, mask: mask, cache: cache)
        let convSplits = split(
            convOutput,
            indices: [intermediateSize, intermediateSize + numGroups * ssmStateSize],
            axis: -1
        )

        var hidden = convSplits[0]
        var B = convSplits[1]
        var C = convSplits[2]

        hidden = hidden.reshaped([hidden.dim(0), hidden.dim(1), numHeads, headDim])
        B = B.reshaped([B.dim(0), B.dim(1), numGroups, ssmStateSize])
        C = C.reshaped([C.dim(0), C.dim(1), numGroups, ssmStateSize])

        let dtArray = dt.reshaped([dt.dim(0), dt.dim(1), numHeads])

        let previousState = cache?[1]
        let (y, nextState) = ssmUpdate(
            hiddenStates: hidden,
            ALog: aLog,
            B: B,
            C: C,
            D: D,
            dt: dtArray,
            dtBias: dtBias,
            state: previousState,
            timeStepLimit: timeStepLimit,
            mask: mask
        )

        if let cache {
            cache[1] = nextState
        }

        let flattenedY = y.flattened(start: 2)
        return outProj(norm(flattenedY, gate: gate))
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

        let queries = wq(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let keys = wk(x).reshaped(B, L, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)

        // No RoPE applied - NemotronH attention uses direct attention without position embeddings

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

        return wo(output)
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
        downProj(relu2(upProj(x)))
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

    init(_ args: NemotronHConfiguration) {
        self.topK = args.numExpertsPerTok
        self.nGroup = args.nGroup
        self.topkGroup = args.topkGroup
        self.routedScalingFactor = args.routedScalingFactor
        self.normTopkProb = args.normTopkProb

        self._weight.wrappedValue = MLXArray.zeros([args.nRoutedExperts, args.hiddenSize])
        self._eSCB.wrappedValue = MLXArray.zeros([args.nRoutedExperts])

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let gates = MLX.matmul(x, weight.transposed())
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
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size > 64

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

    init(_ args: NemotronHConfiguration, layerIdx: Int, jangtq: NemotronHJANGTQContext? = nil) {
        self.layerIdx = layerIdx
        self.numExpertsPerTok = args.numExpertsPerTok
        self.hasSharedExperts = args.nSharedExperts != nil && args.nSharedExperts! > 0
        self.moeLatentSize = args.moeLatentSize

        self._gate.wrappedValue = NemotronHMoEGate(args)
        let expertInputDims = args.moeLatentSize ?? args.hiddenSize
        if let latentSize = args.moeLatentSize {
            self._fc1LatentProj.wrappedValue = Linear(
                args.hiddenSize, latentSize, bias: false)
            self._fc2LatentProj.wrappedValue = Linear(
                latentSize, args.hiddenSize, bias: false)
        }
        if let jangtq {
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
                hiddenDims: args.moeIntermediateSize,
                numExperts: args.nRoutedExperts
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
        let expertInput = fc1LatentProj.map { $0(x) } ?? x
        if profile {
            MLX.eval(expertInput)
            NemotronHLayerProfiler.shared.record("moe.fc1_latent", seconds: CFAbsoluteTimeGetCurrent() - fc1Start)
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
            y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        }
        if profile {
            MLX.eval(y)
            NemotronHLayerProfiler.shared.record("moe.switch_mlp", seconds: CFAbsoluteTimeGetCurrent() - switchStart)
        }
        if let fc2LatentProj {
            let fc2Start = profile ? CFAbsoluteTimeGetCurrent() : 0
            y = fc2LatentProj(y)
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
        jangtq: NemotronHJANGTQContext? = nil
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
            _mixer.wrappedValue = NemotronHMoE(args, layerIdx: layerIdx, jangtq: jangtq)
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

    init(_ args: NemotronHConfiguration, jangtq: NemotronHJANGTQContext? = nil) {
        self.args = args
        precondition(args.vocabSize > 0)

        self._embeddings.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)

        let pattern = Array(args.hybridOverridePattern)
        self._layers.wrappedValue = pattern.enumerated().map { layerIdx, blockType in
            NemotronHBlock(args, layerIdx: layerIdx, blockType: blockType, jangtq: jangtq)
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
        var hidden = inputsEmbeds

        // Create attention mask using the first attention layer's cache
        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = {
            guard let cacheIdx = firstAttentionCacheIndex,
                let cache = cache,
                cacheIdx < cache.count
            else { return .none }
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
        for layer in layers {
            let c: KVCache?
            if layer.blockType == .mamba || layer.blockType == .attention {
                c = cache?[cacheCounter]
                cacheCounter += 1
            } else {
                c = nil
            }

            hidden = layer(hidden, attentionMask: attentionMask, ssmMask: ssmMask, cache: c)
        }

        return normF(hidden)
    }
}

// MARK: - Main Model (matches Python's Model class)

public class NemotronHModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "backbone") private var backbone: NemotronHBackbone
    public let configuration: NemotronHConfiguration
    public let jangtqContext: NemotronHJANGTQContext?

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public var loraLayers: [Module] {
        backbone.layers
    }

    public init(_ args: NemotronHConfiguration) {
        self.configuration = args
        self.jangtqContext = nil
        self.vocabularySize = args.vocabSize

        // kvHeads array: non-zero for attention layers, zero for others
        let pattern = Array(args.hybridOverridePattern)
        self.kvHeads = pattern.compactMap { char -> Int? in
            let blockType = NemotronHBlockType(from: char)
            if blockType == .mamba || blockType == .attention {
                return blockType == .attention ? args.numKeyValueHeads : 0
            }
            return nil
        }

        self._backbone.wrappedValue = NemotronHBackbone(args, jangtq: nil)

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
        let pattern = Array(args.hybridOverridePattern)
        self.kvHeads = pattern.compactMap { char -> Int? in
            let blockType = NemotronHBlockType(from: char)
            if blockType == .mamba || blockType == .attention {
                return blockType == .attention ? args.numKeyValueHeads : 0
            }
            return nil
        }
        self._backbone.wrappedValue = NemotronHBackbone(args, jangtq: jangtqContext)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = backbone(inputs, cache: cache)
        if let lmHead {
            if nemotronHActivationBF16RetentionEnabled(),
                out.dtype != lmHead.weight.dtype
            {
                out = out.asType(lmHead.weight.dtype)
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
                out.dtype != lmHead.weight.dtype
            {
                out = out.asType(lmHead.weight.dtype)
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
            // (`mtp.*`). The normal causal runtime does not instantiate
            // or use them; Python/JANG loaders drop the same keys.
            if key.hasPrefix("mtp.") || key.hasSuffix(".importance") {
                continue
            }
            if key.hasPrefix("vision_model.")
                || key.hasPrefix("sound_encoder.")
                || key.hasPrefix("sound_projection.")
                || key.hasPrefix("mlp1.")
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
                    let toJoin = (0 ..< configuration.nRoutedExperts).compactMap { e in
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
        moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        moeLatentSize = try container.decodeIfPresent(Int.self, forKey: .moeLatentSize)
        moeSharedExpertIntermediateSize = try container.decode(
            Int.self, forKey: .moeSharedExpertIntermediateSize)
        nRoutedExperts = try container.decode(Int.self, forKey: .nRoutedExperts)
        nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        numExpertsPerTok = RuntimeMoETopKOverride.effectiveTopK(
            currentTopK: try container.decode(Int.self, forKey: .numExpertsPerTok),
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

        // mlx-lm stores Nemotron-H as `time_step_limit: [min, max]`.
        // A lower bound of 0.0 is normalized to 0.001 upstream before the
        // selective-scan dt clamp. Keep Swift decode and update passes aligned.
        if let limits = try? rawContainer.decode([Float].self, forKey: .timeStepLimit),
           let first = limits.first
        {
            timeStepLimitMin = first <= 0 ? 0.001 : first
            timeStepLimitMax = limits.count > 1 ? limits[1] : first
        } else {
            let decodedMin = try container.decodeIfPresent(Float.self, forKey: .timeStepLimitMin) ?? 0.001
            timeStepLimitMin = decodedMin <= 0 ? 0.001 : decodedMin
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
    }
}
