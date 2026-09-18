// Streaming Conformer pieces for NemotronLabs VoiceChat.
//
// The existing `NemotronHOmni/Parakeet.swift` Conformer is OFFLINE-correct but
// cannot run on a live mic. VoiceChat's encoder config asks for three things it
// does not do:
//
//   conv_norm_type:      layer_norm      (Parakeet uses NemotronHBatchNorm1d)
//   conv_context_size:   causal          (Parakeet pads SYMMETRICALLY)
//   att_context_style:   chunked_limited (Parakeet is full attention)
//   att_context_size:    [[70, 0]]       70 left, ZERO right — no lookahead
//
// These live in their own file rather than as flags on the Omni path: the Omni
// encoder is shipped and benchmarked (18/18 RunBench rows, 157–219 ms first
// delta) and symmetric padding is CORRECT for it. Changing it in place would
// alter a working model to serve a different one.
//
// 🚨 On the padding discrepancy: `Parakeet.swift`'s depthwise is commented
// "Depthwise causal conv" but computes `pad = (K-1)/2` on BOTH sides, i.e. it
// reads (K-1)/2 = 4 frames into the future. Harmless offline. On a live mic it
// means the encoder cannot emit frame t until t+4 exists — silent added latency
// that no offline test can observe.

import Foundation
import MLX
import MLXNN

/// Depthwise conv over time with **left-only** padding, so output at t depends
/// on inputs ≤ t and nothing later.
///
/// Optionally carries `K-1` frames of tail state across chunk boundaries: with
/// state, streaming a sequence in chunks is bit-identical to running it whole.
/// Without it, every chunk boundary silently zero-pads and injects an artifact
/// once per chunk — audible as periodic glitching, and invisible to a
/// whole-sequence test.
public class VoiceChatCausalDepthwiseConv: Module {
    let dim: Int
    let kernelSize: Int

    @ParameterInfo(key: "depthwise_conv_weight") var dwWeight: MLXArray

    /// Trailing `K-1` frames of the previous chunk. `nil` = start of stream.
    private var carry: MLXArray?

    public init(dim: Int, kernelSize: Int) {
        self.dim = dim
        self.kernelSize = kernelSize
        self._dwWeight.wrappedValue = MLXArray.zeros([dim, 1, kernelSize])
    }

    /// Drop streaming state — call between utterances, never mid-stream.
    public func resetState() { carry = nil }

    /// - Parameters:
    ///   - x: (B, T, D)
    ///   - streaming: when true, consume/produce cross-chunk carry state.
    public func callAsFunction(_ x: MLXArray, streaming: Bool = false) -> MLXArray {
        let B = x.dim(0), T = x.dim(1), D = x.dim(2)
        let K = kernelSize
        let padLeft = K - 1

        // Left context: carried tail if streaming, else zeros (start of stream).
        let left: MLXArray
        if streaming, let c = carry, c.dim(1) == padLeft {
            left = c.asType(x.dtype)
        } else {
            left = MLXArray.zeros([B, padLeft, D], dtype: x.dtype)
        }
        let padded = MLX.concatenated([left, x], axis: 1)

        if streaming {
            // Keep the last K-1 frames of THIS chunk for the next call.
            carry = padded[0..., (padded.dim(1) - padLeft)..., 0...]
        }

        let w = dwWeight.reshaped([D, K]).transposed(1, 0).asType(x.dtype)  // (K, D)
        var out = MLXArray.zeros([B, T, D], dtype: x.dtype)
        for i in 0 ..< K {
            let slice = padded[0..., i ..< (i + T), 0...]
            out = out + slice * w[i].reshaped([1, 1, D])
        }
        return out
    }
}

/// Conformer conv module with a **LayerNorm** normaliser and a causal depthwise.
///
/// LayerNorm rather than BatchNorm is not cosmetic for streaming: BatchNorm
/// normalises per channel using stored running statistics gathered over whole
/// utterances, whereas LayerNorm normalises each frame against itself. Under
/// chunked inference the former is a fixed global assumption and the latter is
/// chunk-invariant by construction.
public class VoiceChatConformerConvModule: Module {
    let dim: Int
    let kernelSize: Int

    @ModuleInfo(key: "pointwise_conv1") var pw1: Conv1d
    @ModuleInfo(key: "depthwise") var depthwise: VoiceChatCausalDepthwiseConv
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pointwise_conv2") var pw2: Conv1d

    public init(dim: Int, kernelSize: Int) {
        self.dim = dim
        self.kernelSize = kernelSize
        self._pw1.wrappedValue = Conv1d(
            inputChannels: dim, outputChannels: 2 * dim, kernelSize: 1, bias: false)
        self._depthwise.wrappedValue = VoiceChatCausalDepthwiseConv(
            dim: dim, kernelSize: kernelSize)
        self._norm.wrappedValue = LayerNorm(dimensions: dim)
        self._pw2.wrappedValue = Conv1d(
            inputChannels: dim, outputChannels: dim, kernelSize: 1, bias: false)
    }

    public func resetState() { depthwise.resetState() }

    /// x: (B, T, D)
    public func callAsFunction(_ x: MLXArray, streaming: Bool = false) -> MLXArray {
        var h = pw1(x)                                   // (B, T, 2D)
        let parts = MLX.split(h, parts: 2, axis: -1)     // GLU
        h = parts[0] * MLX.sigmoid(parts[1])
        h = depthwise(h, streaming: streaming)
        h = norm(h)
        h = silu(h)
        return pw2(h)
    }
}

/// Additive attention mask for `chunked_limited` context.
///
/// `att_context_size == [[70, 0]]`: each query may attend to 70 frames of
/// history and **zero** frames of future. Right context 0 is what allows the
/// encoder to emit as audio arrives; any positive value would make every
/// response wait for that many future frames.
public enum VoiceChatAttentionMask {

    /// - Returns: (T, T) additive mask, 0 where attention is allowed and
    ///   -inf where it is not.
    public static func chunkedLimited(
        queryCount T: Int, leftContext: Int, rightContext: Int, dtype: DType
    ) -> MLXArray {
        // idx[i][j] = j - i  → negative is history, positive is future.
        let rows = MLXArray(0 ..< T).reshaped([T, 1])
        let cols = MLXArray(0 ..< T).reshaped([1, T])
        let delta = cols - rows

        let tooOld = delta .< MLXArray(-leftContext)
        let tooNew = delta .> MLXArray(rightContext)
        let blocked = MLX.logicalOr(tooOld, tooNew)

        let neg = MLXArray(-Float.greatestFiniteMagnitude)
        return MLX.where(blocked, neg, MLXArray(Float(0))).asType(dtype)
    }
}
