// NemotronLabs VoiceChat — the speech side assembled: codec + EAR-TTS +
// speaker prompt latents. Port of `SpeechDecoder` in tts.py, and the
// counterpart to `VoiceChatSTTModel` on the listening side.
//
// 🚨 Speaker identity is a LEARNED LATENT (`audio_prompt_latents.Aria`,
// [1, 37, 1152] = 83 KiB), not something baked into the weights. That is what
// makes custom voices (the Dinoki avatar) a matter of installing another
// latent rather than retraining — and it is why the tensor must stay fp and
// byte-exact through any quantization.

import Foundation
import MLX
import MLXNN

/// Named speaker prompt latents. The checkpoint ships `Aria`; additional
/// voices are additional keys of the same shape.
public class VoiceChatPromptLatents: Module {
    @ParameterInfo(key: "Aria") public var aria: MLXArray

    public init(hiddenSize: Int, frames: Int = 37) {
        self._aria.wrappedValue = MLXArray.zeros([1, frames, hiddenSize])
    }
}

/// Everything under the bundle's `tts_model.` prefix.
public class VoiceChatSpeechDecoder: Module {
    public let ttsConfig: VoiceChatTTSConfiguration
    public let codecConfig: VoiceChatCodecConfiguration

    @ModuleInfo(key: "audio_codec") public var audioCodec: VoiceChatCodec
    /// Doubled name is the checkpoint's own: `tts_model.tts_model.*`.
    @ModuleInfo(key: "tts_model") public var ttsModel: VoiceChatRVQEARTTSModel
    @ParameterInfo(key: "control_codes") var controlCodes: MLXArray
    @ModuleInfo(key: "audio_prompt_latents") public var audioPromptLatents: VoiceChatPromptLatents
    @ParameterInfo(key: "codec_silence_tokens") public var codecSilenceTokens: MLXArray

    public init(tts: VoiceChatTTSConfiguration, codec: VoiceChatCodecConfiguration) {
        self.ttsConfig = tts
        self.codecConfig = codec
        self._audioCodec.wrappedValue = VoiceChatCodec(codec)
        self._ttsModel.wrappedValue = VoiceChatRVQEARTTSModel(tts)
        self._controlCodes.wrappedValue = MLXArray.zeros([3], dtype: .int32)
        self._audioPromptLatents.wrappedValue = VoiceChatPromptLatents(
            hiddenSize: tts.hiddenSize)
        self._codecSilenceTokens.wrappedValue = MLXArray.zeros(
            [tts.numQuantizers], dtype: .int32)
    }

    /// Render one step of generated codes to 22.05 kHz audio.
    ///
    /// `cache` MUST be carried across steps for live speech: it holds every
    /// causal conv tail plus the iSTFT overlap frames, and without it each
    /// chunk restarts from zeros and clicks at the seam.
    public func renderCodes(
        _ codes: MLXArray, cache: VoiceChatCausalConv1dCache? = nil, flush: Bool = false
    ) -> MLXArray {
        audioCodec.decode(codes, cache: cache, flush: flush)
    }

    /// Map the bundle's on-disk `tts_model.*` layout onto this module tree.
    /// Input keys arrive WITHOUT the `tts_model.` prefix.
    public static func sanitized(
        _ weights: [String: MLXArray], codecConfig: VoiceChatCodecConfiguration
    ) -> [String: MLXArray] {
        var direct = [String: MLXArray]()
        var codec = [String: MLXArray]()

        for (key, value) in weights {
            if key == "_control_codes" {
                direct["control_codes"] = value
            } else if key.hasPrefix("audio_codec.") {
                codec[String(key.dropFirst("audio_codec.".count))] = value
            } else {
                direct[key] = value
            }
        }
        for (key, value) in VoiceChatCodec.sanitized(codec, config: codecConfig) {
            direct["audio_codec.\(key)"] = value
        }
        return direct
    }
}
