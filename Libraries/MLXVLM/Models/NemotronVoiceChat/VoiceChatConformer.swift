// NemotronLabs VoiceChat — full streaming-capable FastConformer encoder.
//
// Mirrors `mlx_audio.stt.models.nemotron_asr.conformer` (the exact module the
// reference runtime instantiates for this checkpoint), with the SAME parameter
// keys as the shipped bundle:
//
//   encoder.pre_encode.conv.{0,2,3,5,6}.{weight,bias}   causal dw_striding ×8
//   encoder.pre_encode.out.{weight,bias}
//   encoder.layers.{i}.norm_feed_forward1 / feed_forward1.linear{1,2}
//   encoder.layers.{i}.norm_self_att / self_attn.linear_{q,k,v,out,pos}
//   encoder.layers.{i}.self_attn.pos_bias_{u,v}         (untied, per layer)
//   encoder.layers.{i}.norm_conv / conv.pointwise_conv{1,2} / conv.depthwise_conv
//   encoder.layers.{i}.conv.batch_norm                  (a LayerNorm — NeMo
//                                                        keeps the name so
//                                                        checkpoint keys match)
//   encoder.layers.{i}.norm_feed_forward2 / feed_forward2.linear{1,2}
//   encoder.layers.{i}.norm_out
//
// Streaming properties this file is responsible for (the ones that make the
// encoder valid on a LIVE mic, not just on a whole file):
//   * causal depthwise conv — left-only padding, no future frames
//   * causal dw-striding subsampling — asymmetric pad (left K-1, right S-1)
//   * chunked_limited attention — NeMo's CHUNK-BASED mask, see below
//
// 🚨 `chunked_limited` is chunk-based, not a per-frame sliding window: frames
// group into chunks of `right_context + 1`, and a frame sees its own chunk plus
// `left_context / chunk_size` previous chunks. With the shipped `[[70, 0]]`
// the chunk size is 1 and it degenerates to the per-frame window, which is why
// the two formulations agree on THIS bundle — but the general form is the one
// NeMo trains against, so the general form is what we implement.

import Foundation
import MLX
import MLXFast
import MLXNN

/// Transformer-XL relative positional encoding (no learned parameters).
public final class VoiceChatRelPositionalEncoding {
    public let dModel: Int
    public private(set) var maxLen: Int
    private let scale: Float
    private var pe: MLXArray

    public init(dModel: Int, maxLen: Int = 5000, scaleInput: Bool = false) {
        self.dModel = dModel
        self.maxLen = maxLen
        self.scale = scaleInput ? Float(Foundation.sqrt(Double(dModel))) : 1.0
        self.pe = Self.computePE(dModel: dModel, maxLen: maxLen)
    }

    private static func computePE(dModel: Int, maxLen: Int) -> MLXArray {
        // positions maxLen-1 … -(maxLen-1), matching NeMo's center-out layout.
        let positions = MLXArray(stride(from: maxLen - 1, through: -(maxLen - 1), by: -1).map { Float($0) })
            .reshaped([2 * maxLen - 1, 1])
        let divTerm = MLX.exp(
            MLXArray(stride(from: 0, to: dModel, by: 2).map { Float($0) })
                * (-Float(Foundation.log(10000.0)) / Float(dModel)))
        let angles = positions * divTerm.reshaped([1, dModel / 2])
        let sin = MLX.sin(angles)
        let cos = MLX.cos(angles)
        // Interleave sin into even columns and cos into odd ones.
        let stacked = MLX.stacked([sin, cos], axis: -1)  // (2L-1, d/2, 2)
        return stacked.reshaped([2 * maxLen - 1, dModel]).expandedDimensions(axis: 0)
    }

    private func ensure(length: Int) {
        if length > maxLen {
            maxLen = length + 1
            pe = Self.computePE(dModel: dModel, maxLen: maxLen)
        }
    }

    /// - Returns: (scaled input, position embedding of length 2*T-1)
    public func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let T = x.dim(1)
        ensure(length: T)
        let buffer = pe.dim(1)
        let start = buffer / 2 - (T - 1)
        let end = buffer / 2 + (T - 1) + 1
        let posEmb = pe[0..., start ..< end, 0...].asType(x.dtype)
        return (scale == 1.0 ? x : x * MLXArray(scale), posEmb)
    }

    /// Position embedding for a key window of `length` frames (2*length-1 rows)
    /// — the cache-aware streaming path, where a short query chunk attends to a
    /// longer cache+chunk key window.
    public func posEmb(forWindow length: Int, dtype: DType) -> MLXArray {
        ensure(length: length)
        let center = pe.dim(1) / 2
        return pe[0..., (center - (length - 1)) ..< (center + length), 0...].asType(dtype)
    }
}

/// NeMo `chunked_limited` additive mask (0 visible, large-negative blocked).
public func voiceChatChunkedLimitedMask(
    sequenceLength T: Int, leftContext: Int, rightContext: Int, dtype: DType
) -> MLXArray {
    let chunkSize = rightContext + 1
    let leftChunks = leftContext >= 0 ? leftContext / chunkSize : Int(1e8)
    let chunkIdx = MLXArray(0 ..< T) / MLXArray(Int32(chunkSize))
    let diff = chunkIdx.reshaped([T, 1]) - chunkIdx.reshaped([1, T])
    let visible = MLX.logicalAnd(diff .>= MLXArray(0), diff .<= MLXArray(Int32(leftChunks)))
    return MLX.where(visible, MLXArray(Float(0)), MLXArray(Float(-1e30)))
        .asType(dtype)
        .reshaped([1, 1, T, T])
}

/// Rel-pos multi-head attention, Transformer-XL convention, `use_bias: false`.
public class VoiceChatRelPositionAttention: Module {
    let nHead: Int
    let nFeat: Int
    let headDim: Int
    let attnScale: Float

    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "linear_pos") var linearPos: Linear
    @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray

    public init(nHead: Int, nFeat: Int, bias: Bool = false) {
        precondition(nFeat % nHead == 0)
        self.nHead = nHead
        self.nFeat = nFeat
        self.headDim = nFeat / nHead
        self.attnScale = Float(1.0 / Foundation.sqrt(Double(nFeat / nHead)))
        self._linearQ.wrappedValue = Linear(nFeat, nFeat, bias: bias)
        self._linearK.wrappedValue = Linear(nFeat, nFeat, bias: bias)
        self._linearV.wrappedValue = Linear(nFeat, nFeat, bias: bias)
        self._linearOut.wrappedValue = Linear(nFeat, nFeat, bias: bias)
        self._linearPos.wrappedValue = Linear(nFeat, nFeat, bias: false)
        self._posBiasU.wrappedValue = MLXArray.zeros([nHead, nFeat / nHead])
        self._posBiasV.wrappedValue = MLXArray.zeros([nHead, nFeat / nHead])
    }

    /// Transformer-XL relative shift: (B, H, Tq, P) → same shape, rows realigned.
    func relShift(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), H = x.dim(1), Tq = x.dim(2), P = x.dim(3)
        var y = MLX.padded(x, widths: [.init((0, 0)), .init((0, 0)), .init((0, 0)), .init((1, 0))])
        y = y.reshaped([B, H, P + 1, Tq])
        y = y[0..., 0..., 1..., 0...]
        return y.reshaped([B, H, Tq, P])
    }

    /// Self-attention over `x` with an additive mask.
    public func callAsFunction(_ x: MLXArray, posEmb: MLXArray, mask: MLXArray?) -> MLXArray {
        attend(query: x, keyValue: x, posEmb: posEmb, mask: mask)
    }

    /// Cache-aware step: `query` (the new chunk) attends to `keyValue`
    /// (cache + chunk). No mask — the window IS the allowed context.
    public func stream(query: MLXArray, keyValue: MLXArray, posEmb: MLXArray) -> MLXArray {
        attend(query: query, keyValue: keyValue, posEmb: posEmb, mask: nil)
    }

    private func attend(
        query: MLXArray, keyValue: MLXArray, posEmb: MLXArray, mask: MLXArray?
    ) -> MLXArray {
        let B = query.dim(0), Tq = query.dim(1)
        let Tk = keyValue.dim(1)
        let q = linearQ(query).reshaped([B, Tq, nHead, headDim])
        let k = linearK(keyValue).reshaped([B, Tk, nHead, headDim]).transposed(0, 2, 1, 3)
        let v = linearV(keyValue).reshaped([B, Tk, nHead, headDim]).transposed(0, 2, 1, 3)
        let p = linearPos(posEmb)
        let P = p.dim(1)
        let pT = p.reshaped([p.dim(0), P, nHead, headDim]).transposed(0, 2, 1, 3)

        let qU = (q + posBiasU).transposed(0, 2, 1, 3)
        let qV = (q + posBiasV).transposed(0, 2, 1, 3)

        var matrixBD = MLX.matmul(qV, pT.transposed(0, 1, 3, 2))
        matrixBD = relShift(matrixBD)
        matrixBD = matrixBD[0..., 0..., 0..., 0 ..< Tk] * MLXArray(attnScale)
        if let mask {
            matrixBD = matrixBD + mask
        }

        let o = MLXFast.scaledDotProductAttention(
            queries: qU, keys: k, values: v, scale: attnScale, mask: matrixBD)
        return linearOut(o.transposed(0, 2, 1, 3).reshaped([B, Tq, nFeat]))
    }
}

/// linear1 → SiLU → linear2, `use_bias: false` on this checkpoint.
public class VoiceChatFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    public init(dModel: Int, dFF: Int, bias: Bool) {
        self._linear1.wrappedValue = Linear(dModel, dFF, bias: bias)
        self._linear2.wrappedValue = Linear(dFF, dModel, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

/// Conv module: GLU pointwise → CAUSAL depthwise → LayerNorm → SiLU → pointwise.
///
/// 🚨 The normaliser is a LayerNorm stored under the key `batch_norm` — NeMo
/// keeps the historical name so checkpoint keys line up. LayerNorm (not
/// BatchNorm) is what makes the module chunk-invariant: it normalises each
/// frame against itself instead of against whole-utterance running statistics.
public class VoiceChatConformerConvolution: Module {
    let kernelSize: Int
    /// Left-only padding: output at t reads inputs ≤ t, never the future.
    let padLeft: Int

    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo(key: "batch_norm") var batchNorm: LayerNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

    /// Trailing `K-1` input frames of the previous chunk (streaming only).
    private var carry: MLXArray?

    public init(dModel: Int, kernelSize: Int, bias: Bool) {
        self.kernelSize = kernelSize
        self.padLeft = kernelSize - 1
        self._pointwiseConv1.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: 2 * dModel, kernelSize: 1, bias: bias)
        self._depthwiseConv.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel, kernelSize: kernelSize,
            groups: dModel, bias: bias)
        self._batchNorm.wrappedValue = LayerNorm(dimensions: dModel)
        self._pointwiseConv2.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel, kernelSize: 1, bias: bias)
    }

    public func resetState() { carry = nil }

    public func callAsFunction(_ x: MLXArray, streaming: Bool = false) -> MLXArray {
        var h = pointwiseConv1(x)
        let parts = MLX.split(h, parts: 2, axis: -1)
        h = parts[0] * MLX.sigmoid(parts[1])  // GLU

        // Causal left context: carried tail when streaming, zeros at stream start.
        let left: MLXArray
        if streaming, let c = carry, c.dim(1) == padLeft {
            left = c.asType(h.dtype)
        } else {
            left = MLXArray.zeros([h.dim(0), padLeft, h.dim(2)], dtype: h.dtype)
        }
        let padded = MLX.concatenated([left, h], axis: 1)
        if streaming {
            carry = padded[0..., (padded.dim(1) - padLeft)..., 0...]
        }

        h = depthwiseConv(padded)
        h = batchNorm(h)
        h = silu(h)
        return pointwiseConv2(h)
    }
}

/// One conformer block: ½FF → rel-pos MHA → conv → ½FF → LayerNorm.
public class VoiceChatConformerBlock: Module {
    @ModuleInfo(key: "norm_feed_forward1") var normFeedForward1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var feedForward1: VoiceChatFeedForward
    @ModuleInfo(key: "norm_self_att") var normSelfAtt: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: VoiceChatRelPositionAttention
    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo(key: "conv") var conv: VoiceChatConformerConvolution
    @ModuleInfo(key: "norm_feed_forward2") var normFeedForward2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var feedForward2: VoiceChatFeedForward
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm

    public init(dModel: Int, nHeads: Int, ffExpansion: Int, convKernel: Int, bias: Bool) {
        let dFF = dModel * ffExpansion
        self._normFeedForward1.wrappedValue = LayerNorm(dimensions: dModel)
        self._feedForward1.wrappedValue = VoiceChatFeedForward(dModel: dModel, dFF: dFF, bias: bias)
        self._normSelfAtt.wrappedValue = LayerNorm(dimensions: dModel)
        self._selfAttn.wrappedValue = VoiceChatRelPositionAttention(nHead: nHeads, nFeat: dModel, bias: bias)
        self._normConv.wrappedValue = LayerNorm(dimensions: dModel)
        self._conv.wrappedValue = VoiceChatConformerConvolution(
            dModel: dModel, kernelSize: convKernel, bias: bias)
        self._normFeedForward2.wrappedValue = LayerNorm(dimensions: dModel)
        self._feedForward2.wrappedValue = VoiceChatFeedForward(dModel: dModel, dFF: dFF, bias: bias)
        self._normOut.wrappedValue = LayerNorm(dimensions: dModel)
    }

    public func resetState() { conv.resetState() }

    public func callAsFunction(
        _ x: MLXArray, posEmb: MLXArray, mask: MLXArray?, streaming: Bool = false
    ) -> MLXArray {
        var x = x
        x = x + 0.5 * feedForward1(normFeedForward1(x))
        x = x + selfAttn(normSelfAtt(x), posEmb: posEmb, mask: mask)
        x = x + conv(normConv(x), streaming: streaming)
        x = x + 0.5 * feedForward2(normFeedForward2(x))
        return normOut(x)
    }
}

/// Depthwise-striding ×8 subsampling with CAUSAL (asymmetric) padding:
/// left `K-1`, right `stride-1` on both time and frequency, matching NeMo's
/// `CausalConv2D`. Symmetric padding here would read future audio and add
/// silent latency to every emitted frame.
public class VoiceChatCausalSubsampling: Module {
    let samplingNum: Int
    let kernelSize = 3
    let stride = 2
    let padLeft: Int
    let padRight: Int
    let outputFreq: Int
    /// Indices of the stride-2 3×3 convs that receive causal padding.
    let stridedIndices: Set<Int>

    @ModuleInfo(key: "conv") var conv: [Module]
    @ModuleInfo(key: "out") var out: Linear

    public init(featIn: Int, channels: Int, subsamplingFactor: Int, dModel: Int) {
        self.samplingNum = Int(Foundation.log2(Double(subsamplingFactor)).rounded())
        self.padLeft = kernelSize - 1
        self.padRight = stride - 1

        var freq = featIn
        for _ in 0 ..< samplingNum {
            freq = (freq + padLeft + padRight - kernelSize) / stride + 1
        }
        self.outputFreq = freq

        // NeMo layer indices: ReLU at 1 / 4 / 7 carry no parameters, so the
        // parameterised entries land at conv.0, conv.2, conv.3, conv.5, conv.6.
        var layers: [Module] = [
            Conv2d(inputChannels: 1, outputChannels: channels, kernelSize: 3, stride: 2, padding: 0),
            ReLU(),
        ]
        var strided: Set<Int> = [0]
        for i in 0 ..< (samplingNum - 1) {
            strided.insert(2 + 3 * i)
            layers.append(
                Conv2d(
                    inputChannels: channels, outputChannels: channels, kernelSize: 3,
                    stride: 2, padding: 0, groups: channels))
            layers.append(
                Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1))
            layers.append(ReLU())
        }
        self.stridedIndices = strided
        self._conv.wrappedValue = layers
        self._out.wrappedValue = Linear(channels * freq, dModel)
    }

    public func outputLength(_ length: Int) -> Int {
        var length = length
        for _ in 0 ..< samplingNum {
            length = (length + padLeft + padRight - kernelSize) / stride + 1
        }
        return length
    }

    /// x: (B, T, F) mel frames → (B, T/8, dModel)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.expandedDimensions(axis: -1)  // (B, T, F, 1) channels-last
        for (i, layer) in conv.enumerated() {
            if stridedIndices.contains(i) {
                h = MLX.padded(
                    h,
                    widths: [
                        .init((0, 0)),
                        .init((padLeft, padRight)),  // time — causal
                        .init((padLeft, padRight)),  // freq
                        .init((0, 0)),
                    ])
            }
            h = (layer as! UnaryLayer)(h)
        }
        // (B, T', F', C) → (B, T', C·F') channel-major, matching NeMo.
        let B = h.dim(0), T = h.dim(1), F = h.dim(2), C = h.dim(3)
        h = h.transposed(0, 1, 3, 2).reshaped([B, T, C * F])
        return out(h)
    }
}

/// The full 24-layer streaming FastConformer encoder.
public class VoiceChatConformerEncoder: Module {
    public let dModel: Int
    public let leftContext: Int
    public let rightContext: Int
    public let posEnc: VoiceChatRelPositionalEncoding

    @ModuleInfo(key: "pre_encode") var preEncode: VoiceChatCausalSubsampling
    @ModuleInfo(key: "layers") var layers: [VoiceChatConformerBlock]

    public init(_ config: VoiceChatEncoderConfiguration) {
        self.dModel = config.dModel
        self.leftContext = config.leftContextFrames ?? 70
        self.rightContext = config.rightContextFrames ?? 0
        self.posEnc = VoiceChatRelPositionalEncoding(
            dModel: config.dModel,
            maxLen: config.posEmbMaxLen ?? 5000,
            scaleInput: config.xscaling ?? false)
        self._preEncode.wrappedValue = VoiceChatCausalSubsampling(
            featIn: config.featIn,
            channels: config.subsamplingConvChannels,
            subsamplingFactor: config.subsamplingFactor,
            dModel: config.dModel)
        self._layers.wrappedValue = (0 ..< config.nLayers).map { _ in
            VoiceChatConformerBlock(
                dModel: config.dModel,
                nHeads: config.nHeads,
                ffExpansion: config.ffExpansionFactor,
                convKernel: config.convKernelSize,
                bias: config.useBias ?? false)
        }
    }

    public func resetState() {
        for layer in layers { layer.resetState() }
    }

    /// Whole-sequence (mask-based) forward. `x`: (B, T, F) mel frames.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var (h, posEmb) = posEnc(preEncode(x))
        let mask = voiceChatChunkedLimitedMask(
            sequenceLength: h.dim(1),
            leftContext: leftContext,
            rightContext: rightContext,
            dtype: h.dtype)
        for layer in layers {
            h = layer(h, posEmb: posEmb, mask: mask, streaming: false)
        }
        return h
    }
}
