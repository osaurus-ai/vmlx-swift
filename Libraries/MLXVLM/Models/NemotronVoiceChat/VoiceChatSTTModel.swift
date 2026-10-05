// NemotronLabs VoiceChat — the fused STT side: perception (streaming Conformer
// + projection), the nemotron_h backbone, BOTH output heads, and the RNN-T
// transcription branch. Mirrors `DuplexSTTModel` in
// `mlx_vlm/models/nemotron_voicechat/model.py`.
//
// 🚨 `function_head` is a FULL VOCAB-SIZED Linear (131072 × 4480 = 0.587 B —
// the same shape as `lm_head`, and its measured Hessian trace is identical).
// Tool calls are a weighted parallel output channel
// (`function_channel_weight: 2.0`) decoded from their OWN logits, never parsed
// out of the transcript. A runtime that reads only `lm_head` loses tool
// calling silently — the model will still SAY it did things.
//
// Key layout notes (all verified against the shipped MLX bundle):
//   * The bundle stores the backbone FLAT under `llm.` (`llm.layers.*`,
//     `llm.norm_f.*`) with `embed_tokens`/`lm_head` as stt-level siblings.
//     Swift's `NemotronHModel` nests everything under `backbone.` and owns its
//     own embedding/head slots, so `sanitized(_:)` remaps:
//       stt_model.llm.X                → stt_model.llm.backbone.X
//       stt_model.embed_tokens.weight  → stt_model.llm.backbone.embeddings.weight
//       stt_model.lm_head.weight       → stt_model.llm.lm_head.weight
//   * Perception conv weights ship in PyTorch layout (O,C,H,W / O,C,K) and are
//     transposed to MLX channels-last here, exactly like the reference
//     `Model.sanitize`.
//   * The RNN-T LSTM ships raw PyTorch keys; see `VoiceChatRNNT.sanitized`.
//   * `perception.preprocessor.featurizer.{fb,window}` are deterministic mel
//     buffers rebuilt by the front-end, dropped at load like the reference.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// One forward step's outputs: the shared hidden state and the two channels.
public struct VoiceChatSTTOutput {
    public let hiddenStates: MLXArray
    /// `lm_head` logits — the agent's own speech/text channel.
    public let textLogits: MLXArray
    /// `function_head` logits — the tool-call channel. A separate head over
    /// the SAME hidden state; consuming text logits alone drops tool calls.
    public let functionLogits: MLXArray
}

/// Perception: streaming Conformer encoder + projection into the backbone
/// hidden size (1024 → 4480; the config asserts the match at decode time).
public class VoiceChatPerception: Module {
    @ModuleInfo(key: "encoder") public var encoder: VoiceChatConformerEncoder
    @ModuleInfo(key: "proj") var proj: Linear

    public init(config: VoiceChatAudioConfiguration) {
        self._encoder.wrappedValue = VoiceChatConformerEncoder(config.encoder)
        self._proj.wrappedValue = Linear(config.encoder.dModel, config.outputDim, bias: true)
    }

    /// mel (B, T, F) → (projected (B, T/8, 4480), encoded (B, T/8, 1024)).
    /// The unprojected encoding feeds the RNN-T joint, which was trained
    /// against encoder width, not backbone width.
    public func callAsFunction(_ mel: MLXArray) -> (projected: MLXArray, encoded: MLXArray) {
        let encoded = encoder(mel)
        return (proj(encoded), encoded)
    }
}

/// The fused duplex STT model (everything under the bundle's `stt_model.`).
public class VoiceChatSTTModel: Module {
    public let config: NemotronVoiceChatConfiguration

    @ModuleInfo(key: "llm") public var llm: NemotronHModel
    @ModuleInfo(key: "function_head") var functionHead: Linear
    @ModuleInfo(key: "perception") public var perception: VoiceChatPerception
    @ModuleInfo(key: "rnnt_decoder") public var rnntDecoder: VoiceChatPredictNetwork
    @ModuleInfo(key: "rnnt_joint") public var rnntJoint: VoiceChatJointNetwork

    public init(_ config: NemotronVoiceChatConfiguration) {
        self.config = config
        self._llm.wrappedValue = NemotronHModel(config.textConfig)
        self._functionHead.wrappedValue = Linear(
            config.textConfig.hiddenSize, config.textConfig.vocabSize, bias: false)
        self._perception.wrappedValue = VoiceChatPerception(config: config.audioConfig)
        self._rnntDecoder.wrappedValue = VoiceChatPredictNetwork(
            config: config.audioConfig.decoder
                ?? VoiceChatRNNTDecoderConfiguration(
                    predHidden: nil, predRNNLayers: nil, vocabSize: nil, blankAsPad: nil))
        self._rnntJoint.wrappedValue = VoiceChatJointNetwork(
            config: config.audioConfig.joint
                ?? VoiceChatRNNTJointConfiguration(
                    jointHidden: nil, numClasses: nil, encoderHidden: nil,
                    predHidden: nil, activation: nil))
    }

    /// Token-id embedding via the backbone table (the bundle's
    /// `embed_tokens` is remapped onto it at load).
    public func embed(_ tokens: MLXArray) -> MLXArray {
        llm.embedTokens(tokens)
    }

    /// One fused forward over pre-mixed embeddings: hidden once, both heads.
    public func callAsFunction(inputsEmbeds: MLXArray, cache: [KVCache]?) -> VoiceChatSTTOutput {
        let hidden = llm.hiddenStatesFromEmbeddings(inputsEmbeds, cache: cache)
        return VoiceChatSTTOutput(
            hiddenStates: hidden,
            textLogits: llm.projectToLogits(hidden),
            functionLogits: functionHead(hidden))
    }

    /// Fresh per-session backbone cache (Mamba + attention slots, per the
    /// hybrid override pattern — the backbone builds its own layout).
    public func makeCache() -> [KVCache] {
        llm.newCache(parameters: nil)
    }

    /// Streaming ASR over already-encoded (UNPROJECTED) conformer frames.
    public func transcribe(
        encoded: MLXArray, state: inout VoiceChatRNNTDecodeState
    ) -> [Int] {
        VoiceChatRNNT.greedyDecode(
            encoded: encoded,
            decoder: rnntDecoder,
            joint: rnntJoint,
            blankId: config.rnntBlankId,
            maxSymbols: config.audioConfig.maxSymbols,
            state: &state)
    }

    /// Map the bundle's on-disk `stt_model.*` tensor layout onto this module
    /// tree. Input keys arrive WITHOUT the `stt_model.` prefix.
    public static func sanitized(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        for (key, value) in weights {
            // Deterministic mel-filter/window buffers — rebuilt by the
            // front-end, not model parameters.
            if key.hasPrefix("perception.preprocessor.") { continue }

            var key = key
            var value = value

            // Prefix (not exact `.weight`) matches: a QUANTIZED bundle ships
            // `.scales` and `.biases` beside every packed `.weight`, and
            // matching only the weight leaves those two behind as unhandled
            // keys — the load then fails on the strict update.
            if key.hasPrefix("embed_tokens.") {
                key = "llm.backbone.embeddings." + key.dropFirst("embed_tokens.".count)
            } else if key.hasPrefix("lm_head.") {
                key = "llm.lm_head." + key.dropFirst("lm_head.".count)
            } else if key.hasPrefix("llm.") {
                key = "llm.backbone." + key.dropFirst("llm.".count)
            }

            // PyTorch conv layouts → MLX channels-last, as in the reference:
            // 2D perception convs (O,C,H,W) → (O,H,W,C); 1D perception convs
            // and the backbone mamba conv1d (O,C,K) → (O,K,C).
            if key.hasPrefix("perception."), value.ndim == 4 {
                value = value.transposed(0, 2, 3, 1)
            } else if value.ndim == 3,
                key.hasPrefix("perception.") || (key.contains("llm.") && key.contains(".conv1d.weight"))
            {
                value = value.transposed(0, 2, 1)
            }

            out[key] = value
        }
        return VoiceChatRNNT.sanitized(out)
    }
}
