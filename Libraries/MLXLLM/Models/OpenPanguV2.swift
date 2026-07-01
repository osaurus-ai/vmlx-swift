// Copyright © 2026 Osaurus.
//
// OpenPangu 2.0 Flash (`openpangu_v2` / OpenPanguV2ForCausalLM).
// See OPENPANGU-V2-PORT-STATUS.md for the full reverse-engineered weight graph
// and the DSV3/DSV4 reuse map. Adapted from DeepseekV3 (MLA geometry + biased
// top-k MoE) and DeepseekV4 (sinks / indexer / hybrid cache / JANGTQ), with
// Pangu-specific: 3 stateful causal convs, prepended-KV sinks, MHC
// hyper-connections, sandwich norm.
//
// BUILD STATUS (incremental): attention (MLA + convs + prepended sinks) + MoE
// are in this pass. DSA indexer, MHC, sandwich-norm decoder layer, inner/outer
// model, hybrid cache, MTP head, and factory registration follow in later
// passes (see status matrix).

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Causal depthwise conv (qa_conv / compresskv_conv / o_conv, k=3, stateful)

/// A short causal depthwise conv over the sequence axis. Weight shape is
/// `[channels, 1, kernel]` (groups == channels). During a multi-token prefill it
/// is a normal left-padded causal conv1d; during L==1 decode the caller supplies
/// the trailing `kernel-1` tokens via `state` (the conv cache), Mamba-style.
///
/// This mirrors the `Qwen35GatedDeltaNet` conv pattern and the `ZayaCCACache`
/// conv-state contract: the state MUST round-trip with the KV or a KV-only
/// prefix-cache hit produces a silent false hit (garbled turn-2). The
/// state-carry is owned by the layer's cache (added in the cache pass).
final class OpenPanguCausalConv: Module {
    let channels: Int
    let kernelSize: Int
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(channels: Int, kernelSize: Int = 3) {
        self.channels = channels
        self.kernelSize = kernelSize
        // depthwise: groups == channels, no bias (weight [channels, 1, k]).
        self._conv.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: kernelSize, padding: 0, groups: channels, bias: false)
        super.init()
    }

    /// `x`: `(B, L, C)`. `state`: optional `(B, kernel-1, C)` trailing context
    /// from the previous step (nil → zero-pad, i.e. sequence start). Returns
    /// `(y, newState)` where `newState` is the last `kernel-1` input columns to
    /// carry forward.
    func callAsFunction(_ x: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        let B = x.dim(0)
        let pad = kernelSize - 1
        let left = state ?? MLXArray.zeros([B, pad, channels], dtype: x.dtype)
        let padded = concatenated([left, x], axis: 1)          // (B, pad+L, C)
        let y = conv(padded)                                    // (B, L, C) — 'valid'
        // carry the last `pad` input columns (from the padded stream) forward.
        let newState = padded[0..., (padded.dim(1) - pad)..., 0...]
        return (y, newState)
    }
}

// MARK: - MLA attention (+ 3 convs + 128 prepended-KV sinks)

/// Multi-head Latent Attention with OpenPangu's conv-augmented q_a / compressed
/// KV / output paths and 128 learned sink KV rows prepended to every layer's
/// attention. The DSA lightning indexer (16 dsa layers) is added in a later pass
/// via `indexer` (nil on SWA layers).
final class OpenPanguV2Attention: Module {
    let numHeads: Int
    let qkNopeHeadDim: Int
    let qkRopeHeadDim: Int
    let qHeadDim: Int
    let vHeadDim: Int
    let kvLoraRank: Int
    let qLoraRank: Int
    let scale: Float
    let sinkCount: Int

    let rope: RoPELayer

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "qa_conv") var qaConv: OpenPanguCausalConv
    @ModuleInfo(key: "compresskv_conv") var compressKvConv: OpenPanguCausalConv
    @ModuleInfo(key: "o_conv") var oConv: OpenPanguCausalConv

    // Learned sinks: prepended KV rows (see status doc). Named to match the graph.
    @ParameterInfo(key: "param_sink_compressed_kv") var paramSinkCompressedKv: MLXArray
    @ParameterInfo(key: "param_sink_k_pe") var paramSinkKPe: MLXArray

    init(_ config: OpenPanguV2Configuration) {
        self.numHeads = config.numAttentionHeads
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.qHeadDim = config.qkHeadDim  // nope + rope = 192
        self.vHeadDim = config.vHeadDim
        self.kvLoraRank = config.kvLoraRank
        self.qLoraRank = config.qLoraRank
        self.sinkCount = config.paramSinkNumber
        self.scale = pow(Float(qHeadDim), -0.5)

        self._qAProj.wrappedValue = Linear(config.hiddenSize, qLoraRank, bias: false)
        self._qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank, eps: config.rmsNormEps)
        self._qBProj.wrappedValue = Linear(qLoraRank, numHeads * qHeadDim, bias: false)
        self._kvAProjWithMqa.wrappedValue = Linear(
            config.hiddenSize, kvLoraRank + qkRopeHeadDim, bias: false)
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: config.rmsNormEps)
        self._kvBProj.wrappedValue = Linear(
            kvLoraRank, numHeads * (qkNopeHeadDim + vHeadDim), bias: false)
        self._oProj.wrappedValue = Linear(numHeads * vHeadDim, config.hiddenSize, bias: false)

        self._qaConv.wrappedValue = OpenPanguCausalConv(channels: qLoraRank)
        self._compressKvConv.wrappedValue = OpenPanguCausalConv(channels: kvLoraRank)
        self._oConv.wrappedValue = OpenPanguCausalConv(channels: numHeads * vHeadDim)

        self._paramSinkCompressedKv.wrappedValue =
            MLXArray.zeros([config.paramSinkNumber, kvLoraRank])
        self._paramSinkKPe.wrappedValue =
            MLXArray.zeros([config.paramSinkNumber, qkRopeHeadDim])

        // rope_interleave == false → split-half (non-traditional) rotation.
        self.rope = initializeRope(
            dims: config.qkRopeHeadDim, base: config.ropeTheta,
            traditional: config.ropeInterleave, scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
        super.init()
    }

    /// Build the 128 synthetic sink K (qHeadDim) / V (vHeadDim) rows shared by
    /// all queries. Sinks are position-free (no RoPE on their k_pe — they are
    /// always-visible), broadcast to `numHeads`.
    private func sinkKeysValues(_ B: Int, dtype: DType) -> (MLXArray, MLXArray) {
        // compressed_kv [S,512] -> kv_b_proj -> [S, numHeads*(nope+v)]
        let kv = kvBProj(kvALayerNorm(paramSinkCompressedKv.asType(dtype)))
            .reshaped(sinkCount, numHeads, qkNopeHeadDim + vHeadDim)
            .transposed(1, 0, 2)                                   // (H, S, nope+v)
        let sp = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let (sinkKNope, sinkV) = (sp[0], sp[1])                    // (H,S,nope),(H,S,v)
        // k_pe [S,64] -> broadcast to all heads, no rope (position-free sinks)
        let sinkKPe = broadcast(
            paramSinkKPe.asType(dtype).reshaped(1, sinkCount, qkRopeHeadDim),
            to: [numHeads, sinkCount, qkRopeHeadDim])
        let sinkK = concatenated([sinkKNope, sinkKPe], axis: -1)   // (H,S,192)
        return (
            sinkK.reshaped(1, numHeads, sinkCount, qHeadDim).asType(dtype),
            sinkV.reshaped(1, numHeads, sinkCount, vHeadDim).asType(dtype))
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: OpenPanguV2Cache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        // Q: low-rank + qa_conv (on the 1024 latent) + up-proj.
        var qLat = qALayerNorm(qAProj(x))                         // (B,L,1024)
        let (qConved, qaState) = qaConv(qLat, state: cache?.convState(.qa))
        qLat = qConved
        cache?.setConvState(.qa, qaState)
        var q = qBProj(qLat).reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        var (qNope, qPe) = (splitQ[0], splitQ[1])

        // KV: low-rank split into compressed_kv (512) + k_pe (64); conv on the 512.
        let kvA = kvAProjWithMqa(x)
        let splitKvA = split(kvA, indices: [kvLoraRank], axis: -1)
        var compressedKv = kvALayerNorm(splitKvA[0])              // (B,L,512)
        let (kvConved, ckvState) = compressKvConv(compressedKv, state: cache?.convState(.compressKv))
        compressedKv = kvConved
        cache?.setConvState(.compressKv, ckvState)
        var kPe = splitKvA[1].reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        var kv = kvBProj(compressedKv).reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let splitKv = split(kv, indices: [qkNopeHeadDim], axis: -1)
        var (kNope, values) = (splitKv[0], splitKv[1])

        qPe = applyRotaryPosition(rope, to: qPe, cache: cache?.kv)
        kPe = applyRotaryPosition(rope, to: kPe, cache: cache?.kv)
        kPe = repeated(kPe, count: numHeads, axis: 1)

        var keys: MLXArray
        if let kv = cache?.kv {
            (keys, values) = kv.update(
                keys: concatenated([kNope, kPe], axis: -1), values: values)
        } else {
            keys = concatenated([kNope, kPe], axis: -1)
        }
        let queries = concatenated([qNope, qPe], axis: -1)

        // Prepend the 128 learned sink K/V (always-visible, position-free).
        var effMask = mask
        if sinkCount > 0 {
            let (sinkK, sinkV) = sinkKeysValues(B, dtype: keys.dtype)
            keys = concatenated([sinkK, keys], axis: 2)
            values = concatenated([sinkV, values], axis: 2)
            effMask = OpenPanguV2Attention.prependSinkMask(mask, sinks: sinkCount, queryLen: L)
        }

        var output = mlaScaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: effMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, numHeads * vHeadDim)                      // (B,L,6144)

        // o_conv on the concatenated head output, before o_proj.
        let (oConved, oState) = oConv(output, state: cache?.convState(.o))
        output = oConved
        cache?.setConvState(.o, oState)
        return oProj(output)
    }

    /// Widen an array attention mask so every query additionally sees all `sinks`
    /// prepended key columns. For non-array masks (`.causal`/`.none`) the sinks
    /// are always-visible so we fall back to an explicit array built from the
    /// causal shape. (Refined in the cache pass to match RotatingKVCache offsets.)
    static func prependSinkMask(
        _ mask: MLXFast.ScaledDotProductAttentionMaskMode, sinks: Int, queryLen: Int
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        switch mask {
        case .array(let m):
            // m: (..., L, K) → prepend `sinks` all-visible columns.
            let ones = MLXArray.zeros(
                Array(m.shape.dropLast()) + [sinks], dtype: m.dtype)
            return .array(concatenated([ones, m], axis: -1))
        default:
            // causal/none: sinks visible to all; the (L,K) causal part is handled
            // by SDPA's causal mode over the non-sink columns via a materialized
            // mask in the cache pass. For now pass through (sinks widen keys only).
            return mask
        }
    }
}

// MARK: - MoE (DeepSeek-V3 style: sigmoid biased top-k, +1 shared, first-2 dense)

final class OpenPanguV2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// Sigmoid router with `e_score_correction_bias` used for SELECTION only; the
/// UNBIASED score is gathered for the weight, then normalized × routedScalingFactor.
final class OpenPanguV2Gate: Module {
    let topK: Int
    let normTopkProb: Bool
    let routedScalingFactor: Float
    let useExpertBias: Bool
    var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: OpenPanguV2Configuration) {
        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.routedScalingFactor = config.routedScalingFactor
        self.useExpertBias = config.routerEnableExpertBias
        self.weight = MLXArray.zeros([config.nRoutedExperts, config.hiddenSize])
        self._eScoreCorrectionBias.wrappedValue = MLXArray.zeros([config.nRoutedExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let logits = x.matmul(weight.T)
        let originalScores = sigmoid(logits.asType(.float32))
        let scoresForChoice =
            useExpertBias ? originalScores + eScoreCorrectionBias.asType(.float32) : originalScores
        let inds = argPartition(-scoresForChoice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var scores = takeAlong(originalScores, inds, axis: -1)
        if topK > 1, normTopkProb {
            let denom = scores.sum(axis: -1, keepDims: true) + MLXArray(1e-20, dtype: scores.dtype)
            scores = (scores / denom) * routedScalingFactor
        } else {
            scores = scores * routedScalingFactor
        }
        return (inds, scores.asType(x.dtype))
    }
}

final class OpenPanguV2MoE: Module, UnaryLayer {
    let numExpertsPerTok: Int
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    var gate: OpenPanguV2Gate
    @ModuleInfo(key: "shared_experts") var sharedExperts: OpenPanguV2MLP?

    init(_ config: OpenPanguV2Configuration) {
        self.numExpertsPerTok = config.numExpertsPerTok
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize, hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts, activation: silu)
        self.gate = OpenPanguV2Gate(config)
        if config.nSharedExperts > 0 {
            self._sharedExperts.wrappedValue = OpenPanguV2MLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize * config.nSharedExperts)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        var y = switchMLP(x, inds)
        y = (y * scores.expandedDimensions(axis: -1)).sum(axis: -2)
        if let shared = sharedExperts { y = y + shared(x) }
        return y
    }
}
