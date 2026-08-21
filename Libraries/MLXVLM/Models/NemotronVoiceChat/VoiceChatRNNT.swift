// NemotronLabs VoiceChat — RNN-T prediction network, joint network, and the
// greedy streaming transducer decode.
//
// Mirrors `mlx_audio.stt.models.nemotron_asr.rnnt` (NeMo `RNNTDecoder` /
// `RNNTJoint` layout) so checkpoint keys map directly:
//
//   rnnt_decoder.prediction.embed.weight
//   rnnt_decoder.prediction.dec_rnn.lstm.{i}.{Wx,Wh,bias}
//   rnnt_joint.enc.{weight,bias} · rnnt_joint.pred.{weight,bias}
//   rnnt_joint.joint_net.2.{weight,bias}
//
// 🚨 The published bundles store the LSTM in RAW PyTorch layout
// (`weight_ih_l{i}`, `weight_hh_l{i}`, `bias_ih_l{i}` + `bias_hh_l{i}`).
// `VoiceChatRNNT.sanitized(_:)` converts: ih→Wx, hh→Wh, and the TWO bias
// vectors SUM into one (PyTorch keeps them separate; MLX folds them).
// Skipping the sum silently halves the recurrent bias.
//
// RNN-T decodes on a different algorithm from AR language decode: for each
// encoder frame the joint proposes a token; blank (id 1024 — one PAST the
// 1024-entry vocabulary, not a member of it) advances to the next frame,
// a non-blank token is emitted and re-fed to the prediction network WITHOUT
// advancing the frame, at most `max_symbols` (10) times per frame. The
// prediction state advances only on emission, so decode state must be carried
// across chunks for streaming ASR to equal whole-utterance ASR.

import Foundation
import MLX
import MLXNN

/// Multi-layer LSTM stack, one `MLXNN.LSTM` per layer under `lstm.{i}`.
public class VoiceChatLSTMStack: Module {
    public let hiddenSize: Int
    public let numLayers: Int

    @ModuleInfo(key: "lstm") var lstm: [LSTM]

    public init(inputSize: Int, hiddenSize: Int, numLayers: Int) {
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self._lstm.wrappedValue = (0 ..< numLayers).map { i in
            LSTM(inputSize: i == 0 ? inputSize : hiddenSize, hiddenSize: hiddenSize)
        }
    }

    /// Step the stack over `x` (B, L, D) with optional carried per-layer state.
    /// Returns the top-layer hidden sequence and next per-layer (h, c).
    public func callAsFunction(
        _ x: MLXArray, state: (h: [MLXArray], c: [MLXArray])?
    ) -> (MLXArray, (h: [MLXArray], c: [MLXArray])) {
        var seq = x
        var nextH = [MLXArray]()
        var nextC = [MLXArray]()
        for (i, layer) in lstm.enumerated() {
            let (allH, allC) = layer(seq, hidden: state?.h[i], cell: state?.c[i])
            seq = allH
            nextH.append(allH[.ellipsis, -1, 0...])
            nextC.append(allC[.ellipsis, -1, 0...])
        }
        return (seq, (nextH, nextC))
    }
}

/// NeMo `RNNTDecoder` — embedding + LSTM over previously emitted tokens.
public class VoiceChatPredictNetwork: Module {
    public class Prediction: Module {
        @ModuleInfo(key: "embed") var embed: Embedding
        @ModuleInfo(key: "dec_rnn") var decRNN: VoiceChatLSTMStack

        init(vocab: Int, hidden: Int, layers: Int) {
            self._embed.wrappedValue = Embedding(embeddingCount: vocab, dimensions: hidden)
            self._decRNN.wrappedValue = VoiceChatLSTMStack(
                inputSize: hidden, hiddenSize: hidden, numLayers: layers)
        }
    }

    public let predHidden: Int

    @ModuleInfo(key: "prediction") var prediction: Prediction

    public init(config: VoiceChatRNNTDecoderConfiguration) {
        let hidden = config.predHidden ?? 640
        let layers = config.predRNNLayers ?? 2
        // `blank_as_pad: true` — the embedding table has one extra row so the
        // blank id can be fed as a start-of-sequence pad.
        let vocab = (config.vocabSize ?? 1024) + ((config.blankAsPad ?? true) ? 1 : 0)
        self.predHidden = hidden
        self._prediction.wrappedValue = Prediction(vocab: vocab, hidden: hidden, layers: layers)
    }

    /// Step over token ids `y` (B, L); `y == nil` is the start-of-stream input
    /// (a zeros embedding, matching the reference).
    public func callAsFunction(
        _ y: MLXArray?, state: (h: [MLXArray], c: [MLXArray])?
    ) -> (MLXArray, (h: [MLXArray], c: [MLXArray])) {
        let embedded: MLXArray
        if let y {
            embedded = prediction.embed(y)
        } else {
            let batch = state?.h.first?.dim(0) ?? 1
            embedded = MLXArray.zeros([batch, 1, predHidden])
        }
        return prediction.decRNN(embedded, state: state)
    }
}

/// NeMo `RNNTJoint` — enc/pred projections plus the output projection at
/// `joint_net.2` (indices 0/1 are the activation and a dropout placeholder,
/// which carry no parameters — the array shape preserves checkpoint keys).
public class VoiceChatJointNetwork: Module {
    public let numClassesWithBlank: Int

    @ModuleInfo(key: "enc") var enc: Linear
    @ModuleInfo(key: "pred") var pred: Linear
    @ModuleInfo(key: "joint_net") var jointNet: [Module]

    public init(config: VoiceChatRNNTJointConfiguration) {
        let jointHidden = config.jointHidden ?? 640
        let numClasses = (config.numClasses ?? 1024) + 1  // + blank
        self.numClassesWithBlank = numClasses
        self._enc.wrappedValue = Linear(config.encoderHidden ?? 1024, jointHidden)
        self._pred.wrappedValue = Linear(config.predHidden ?? 640, jointHidden)
        self._jointNet.wrappedValue = [
            ReLU(),
            Identity(),
            Linear(jointHidden, numClasses),
        ]
    }

    /// `encFrame` (B, T, encHidden) × `predOut` (B, U, predHidden)
    /// → logits (B, T, U, classes+blank).
    public func callAsFunction(_ encFrame: MLXArray, _ predOut: MLXArray) -> MLXArray {
        let e = enc(encFrame).expandedDimensions(axis: 2)
        let p = pred(predOut).expandedDimensions(axis: 1)
        var x = e + p
        for layer in jointNet {
            x = (layer as! UnaryLayer)(x)
        }
        return x
    }
}

/// Carried greedy-decode state: the prediction network's LSTM state and the
/// last emitted token. Advances ONLY on non-blank emission, so it must be
/// threaded across chunks — resetting it at a chunk boundary silently degrades
/// streaming ASR relative to whole-utterance ASR.
public struct VoiceChatRNNTDecodeState {
    public var lstm: (h: [MLXArray], c: [MLXArray])?
    public var lastToken: Int?
    /// Cached prediction output for the current state (recomputed on emission).
    var predOut: MLXArray?

    public init() {
        self.lstm = nil
        self.lastToken = nil
        self.predOut = nil
    }
}

/// The transducer pair plus the greedy streaming decode loop.
public enum VoiceChatRNNT {

    /// Greedy decode of `encoded` (B=1, T, encHidden) continuing from `state`.
    /// Returns emitted token ids and the advanced state.
    ///
    /// `maxSymbols` (10 on this bundle) bounds emissions per frame so a
    /// pathological joint cannot loop forever on one frame.
    public static func greedyDecode(
        encoded: MLXArray,
        decoder: VoiceChatPredictNetwork,
        joint: VoiceChatJointNetwork,
        blankId: Int,
        maxSymbols: Int,
        state: inout VoiceChatRNNTDecodeState
    ) -> [Int] {
        var emitted = [Int]()
        let T = encoded.dim(1)

        // Prediction output for the current state: zeros-input at stream start,
        // else the cached output from the last emission.
        var predOut: MLXArray
        if let cached = state.predOut {
            // Resuming mid-stream: the cached output corresponds exactly to
            // the carried LSTM state, so chunked decode == whole-utterance.
            predOut = cached
        } else {
            // Start of stream: NeMo's greedy loop computes the SOS step from a
            // zeros embedding and DOES carry the resulting LSTM state.
            let (out, newState) = decoder(
                state.lastToken.map { MLXArray([Int32($0)]).reshaped([1, 1]) },
                state: state.lstm)
            state.lstm = newState
            predOut = out
        }

        for t in 0 ..< T {
            let frame = encoded[0..., t ..< (t + 1), 0...]
            var symbols = 0
            while symbols < maxSymbols {
                let logits = joint(frame, predOut)  // (1, 1, 1, C)
                let token = logits.reshaped([-1]).argMax().item(Int.self)
                if token == blankId {
                    break
                }
                emitted.append(token)
                state.lastToken = token
                let (out, newState) = decoder(
                    MLXArray([Int32(token)]).reshaped([1, 1]), state: state.lstm)
                state.lstm = newState
                predOut = out
                symbols += 1
            }
        }

        state.predOut = predOut
        return emitted
    }

    /// Convert the raw PyTorch LSTM tensor layout the published bundles ship
    /// into the module layout above. Non-LSTM keys pass through untouched.
    public static func sanitized(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var converted = [String: MLXArray]()
        var biasParts = [String: [MLXArray]]()

        for (key, value) in weights {
            guard key.contains(".dec_rnn.lstm.") else {
                converted[key] = value
                continue
            }
            let pieces = key.components(separatedBy: ".dec_rnn.lstm.")
            let stem = pieces[0] + ".dec_rnn.lstm"
            let suffix = pieces[1]
            if suffix.hasPrefix("weight_ih_l") {
                converted["\(stem).\(suffix.dropFirst("weight_ih_l".count)).Wx"] = value
            } else if suffix.hasPrefix("weight_hh_l") {
                converted["\(stem).\(suffix.dropFirst("weight_hh_l".count)).Wh"] = value
            } else if suffix.hasPrefix("bias_ih_l") || suffix.hasPrefix("bias_hh_l") {
                let layer = suffix.components(separatedBy: "_l").last ?? "0"
                biasParts["\(stem).\(layer).bias", default: []].append(value)
            } else {
                converted[key] = value
            }
        }
        for (key, parts) in biasParts {
            converted[key] = parts.dropFirst().reduce(parts[0], +)
        }
        return converted
    }
}
