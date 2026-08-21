// NemotronLabs VoiceChat 11B — the assembled model and its turn loop.
//
// Port of `Model` (model.py) + `VoiceChatSession.generate` (session.py).
//
// The loop is the whole point of a duplex model, so it is worth stating
// plainly: on every 0.08 s frame the backbone consumes ONE fused embedding
// built from three channels —
//
//     fused = embed(previous text token)
//           + user audio embedding for this frame
//           + function_channel_weight × embed(previous function token)
//
// and produces this frame's text token (lm_head) and tool-call token
// (function_head) TOGETHER. The agent's speech is generated from the text
// token by the EAR-TTS model on the SAME clock, so listening, answering,
// speaking, and tool-calling all advance one frame at a time. That shared
// clock is what makes barge-in and turn-taking expressible at all.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// One completed VoiceChat turn.
public struct VoiceChatTurnResult {
    /// Agent text-channel token ids (PAD/silence included — decode filters).
    public let textTokens: [Int]
    /// Tool-call channel token ids. 🚨 Read THESE for tool calls, never the
    /// transcript: a model can say "I checked the weather" while emitting
    /// zero function tokens, and that exact bug has shipped before.
    public let functionTokens: [Int]
    /// (frames, quantizers) RVQ codes for the agent's speech.
    public let audioCodes: MLXArray
    /// (samples,) waveform at `sampleRate`.
    public let audio: MLXArray
    public let sampleRate: Int
    /// Unprojected encoder frames, for RNN-T transcription of the user.
    public let asrEmbeddings: MLXArray
    public let audioFrames: Int
}

/// The full duplex model: listening side + speaking side.
public class NemotronVoiceChatModel: Module {
    public let config: NemotronVoiceChatConfiguration

    @ModuleInfo(key: "stt_model") public var sttModel: VoiceChatSTTModel
    @ModuleInfo(key: "tts_model") public var ttsModel: VoiceChatSpeechDecoder

    public init(_ config: NemotronVoiceChatConfiguration) {
        self.config = config
        self._sttModel.wrappedValue = VoiceChatSTTModel(config)
        self._ttsModel.wrappedValue = VoiceChatSpeechDecoder(
            tts: config.ttsConfig, codec: config.codecConfig)
    }

    /// Split the bundle's flat tensor dict by side and hand each half to its
    /// own sanitizer. Keys arrive exactly as stored (`stt_model.…`,
    /// `tts_model.…`).
    public static func sanitized(
        _ weights: [String: MLXArray], config: NemotronVoiceChatConfiguration
    ) -> [String: MLXArray] {
        var stt = [String: MLXArray]()
        var tts = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("stt_model.") {
                stt[String(key.dropFirst("stt_model.".count))] = value
            } else if key.hasPrefix("tts_model.") {
                tts[String(key.dropFirst("tts_model.".count))] = value
            }
        }
        var out = [String: MLXArray]()
        for (key, value) in VoiceChatSTTModel.sanitized(stt) {
            out["stt_model.\(key)"] = value
        }
        for (key, value) in VoiceChatSpeechDecoder.sanitized(
            tts, codecConfig: config.codecConfig)
        {
            out["tts_model.\(key)"] = value
        }
        return out
    }

    /// The character vocabulary the TTS conditioning needs. Must be called
    /// once with the tokenizer's vocab before any speech is generated.
    public func setVocabulary(_ vocabulary: [String: Int]) throws {
        try ttsModel.ttsModel.setVocabulary(vocabulary)
    }

    /// Substitute the codec's silence codes wherever a control code appears —
    /// control codes are protocol markers, not audio, and rendering them
    /// directly produces artifacts.
    func replacingControlCodes(_ codes: MLXArray) -> MLXArray {
        let control = ttsModel.controlCodesValues
        guard !control.isEmpty else { return codes }
        var mask = MLXArray.zeros(codes.shape, dtype: .bool)
        for token in control {
            mask = MLX.logicalOr(mask, codes .== MLXArray(Int32(token)))
        }
        let silence = MLX.broadcast(
            ttsModel.codecSilenceTokens.reshaped([1, 1, -1]), to: codes.shape)
        return MLX.where(mask, silence, codes)
    }

    /// Which voice the agent speaks in.
    public enum Voice {
        /// The bundle's shipped speaker latent (`audio_prompt_latents.Aria`).
        case builtIn
        /// Clone the voice in a reference recording: 16 kHz-agnostic mono
        /// samples at the CODEC's rate (22.05 kHz), about
        /// `audio_prompt_duration` seconds.
        ///
        /// 🚨 This is not a separate feature bolted on — it is the path the
        /// reference runtime takes whenever no explicit latent is supplied:
        /// the prompt frames become `embed_code(depthSum(codes)) @
        /// audio_prompt_projection_W`, i.e. the frozen projection consumes the
        /// prompt audio's own CODEC CODE EMBEDDINGS. Feeding it a real
        /// recording instead of silence is what makes the speaker that
        /// recording's speaker. (This also settles the open question in the
        /// wiring spec: the projection consumes codec-derived embeddings, not
        /// raw mel.)
        case reference([Float])
    }

    /// Build the speaker-prompt warmup inputs for the TTS model.
    ///
    /// The prompt is a codec-encoded lead-in — silence for the built-in voice,
    /// the reference recording when cloning — whose last two positions carry
    /// the mask sentinel so the model starts speaking from a clean state
    /// rather than continuing the prompt's own codes.
    func ttsPrompt(batchSize: Int = 1, voice: Voice = .builtIn) -> (
        codes: MLXArray, subwords: MLXArray, subwordMask: MLXArray, audioMask: MLXArray,
        latent: MLXArray?
    ) {
        let ttsConfig = config.ttsConfig
        let frames = ttsModel.audioPromptLatents.aria.dim(1)
        let total = frames + 1
        let promptSamples = total * config.codecConfig.waveformToTokenRatio

        // The prompt waveform: silence for the built-in latent (which is
        // supplied separately), or the reference recording when cloning —
        // padded or trimmed to exactly the prompt length the model expects.
        let promptWaveform: MLXArray
        switch voice {
        case .builtIn:
            promptWaveform = MLXArray.zeros([batchSize, 1, promptSamples], dtype: .float32)
        case .reference(let samples):
            var fitted = samples
            if fitted.count > promptSamples {
                // Keep the TAIL: the end of a clip is more likely to be settled
                // speech than the attack at the start.
                fitted = Array(fitted.suffix(promptSamples))
            } else if fitted.count < promptSamples {
                fitted += [Float](repeating: 0, count: promptSamples - fitted.count)
            }
            promptWaveform = MLXArray(fitted).reshaped([1, 1, promptSamples])
                .asType(.float32)
        }

        let promptCodes = ttsModel.audioCodec.encode(promptWaveform).transposed(0, 2, 1)

        var pieces = (0 ..< total).map { promptCodes[0..., $0] }
        let maskCodes = MLXArray.full(
            [batchSize, ttsConfig.numQuantizers], values: MLXArray(Int32(ttsConfig.codebookSize)))
        pieces[0] = maskCodes
        pieces[total - 2] = maskCodes
        let codes = MLX.stacked(pieces, axis: 1)

        let subwords = MLXArray.full(
            [batchSize, frames], values: MLXArray(Int32(config.padTokenId)))
        var subwordMask = MLXArray.zeros([batchSize, frames], dtype: .bool)
        subwordMask[0..., (frames - 2)...] = MLXArray.ones(
            [batchSize, 2], dtype: .bool)
        var audioMask = MLXArray.zeros([batchSize, frames], dtype: .bool)
        audioMask[0..., (frames - 1)...] = MLXArray.ones([batchSize, 1], dtype: .bool)

        // Built-in voice: hand the shipped latent over directly. Cloning:
        // pass nil so `warmup` derives the speaker from the prompt codes
        // through the frozen projection.
        let latent: MLXArray?
        switch voice {
        case .builtIn:
            latent = MLX.broadcast(
                ttsModel.audioPromptLatents.aria[0 ..< 1],
                to: [batchSize, frames, ttsConfig.hiddenSize])
        case .reference:
            latent = nil
        }

        return (codes[0..., ..<(total - 1), 0...], subwords, subwordMask, audioMask, latent)
    }

    /// Run one turn over already-encoded user audio embeddings.
    ///
    /// - Parameters:
    ///   - audioEmbeds: (1, frames, hidden) projected perception output.
    ///   - asrEmbeds: (1, frames, encoderHidden) unprojected, for RNN-T.
    ///   - promptEmbeds: optional system-prompt embeddings prepended on the
    ///     user-audio channel; the output channels stay PAD across that prefix
    ///     and it is trimmed from the result.
    public func generateTurn(
        audioEmbeds: MLXArray,
        asrEmbeds: MLXArray,
        promptEmbeds: MLXArray? = nil,
        maxFrames: Int? = nil,
        voice: Voice = .builtIn
    ) -> VoiceChatTurnResult {
        var audioEmbeds = audioEmbeds
        var audioFrames = audioEmbeds.dim(1)
        if let maxFrames {
            precondition(maxFrames >= 2, "maxFrames must be at least 2")
            audioFrames = Swift.min(audioFrames, maxFrames)
            audioEmbeds = audioEmbeds[0..., ..<audioFrames, 0...]
        }

        var promptFrames = 0
        if let promptEmbeds {
            promptFrames = promptEmbeds.dim(1)
            audioEmbeds = MLX.concatenated(
                [promptEmbeds.asType(audioEmbeds.dtype), audioEmbeds], axis: 1)
        }
        let timelineFrames = promptFrames + audioFrames

        let padId = config.padTokenId
        var textTokens = [Int](repeating: padId, count: timelineFrames)
        var functionTokens = [Int](repeating: padId, count: timelineFrames)
        let sttCache = sttModel.makeCache()

        let prompt = ttsPrompt(voice: voice)
        // 🚨 Materialise the prompt BEFORE the warmup graph is built. The
        // speaker enters the turn only through these frames, and leaving them
        // lazy made cloning intermittent: the same reference recording changed
        // the voice on one run and produced byte-identical audio to the
        // built-in speaker on the next.
        MLX.eval(prompt.codes, prompt.subwords, prompt.subwordMask, prompt.audioMask)
        if let latent = prompt.latent { MLX.eval(latent) }

        var (warmHidden, ttsCache) = ttsModel.ttsModel.warmup(
            code: prompt.codes,
            subwordIds: prompt.subwords,
            subwordMask: prompt.subwordMask,
            audioMask: prompt.audioMask,
            audioPromptLatent: prompt.latent,
            guidance: true)
        // Force the warmup itself. Its hidden state is not otherwise consumed
        // — the speaker is carried in the KV cache it writes — so without this
        // the whole warmup can stay an unevaluated graph whose effect on the
        // cache never lands before the first step.
        MLX.eval(warmHidden)

        // A slice of an MLXArray aliases its parent, and this value is
        // reassigned every frame; copy it so the prompt's own codes cannot be
        // written through later.
        var previousCode = MLXArray(
            prompt.codes[0..., (prompt.codes.dim(1) - 1)..., 0...].asType(.int32)
                .asArray(Int32.self)
        ).reshaped([1, 1, config.ttsConfig.numQuantizers])

        var generatedFrames = [MLXArray]()
        generatedFrames.reserveCapacity(timelineFrames)
        let zeroFrame = MLXArray.zeros([1, 1, config.ttsConfig.numQuantizers], dtype: .int32)

        for time in 0 ..< timelineFrames {
            // NeMo's misleadingly named `_get_bos_embedding` initialises the
            // first agent position with PAD, not BOS.
            let previousTextId = time == 0 ? padId : textTokens[time - 1]
            let previousFunctionId = time == 0 ? padId : functionTokens[time - 1]
            let previousText = MLXArray([Int32(previousTextId)]).reshaped([1, 1])
            let previousFunction = MLXArray([Int32(previousFunctionId)]).reshaped([1, 1])

            let fused =
                sttModel.embed(previousText)
                + audioEmbeds[0..., time ..< (time + 1), 0...]
                + config.functionChannelWeight * sttModel.embed(previousFunction)

            let output = sttModel(inputsEmbeds: fused, cache: sttCache)
            if time >= promptFrames {
                textTokens[time] = output.textLogits[0..., -1].argMax().item(Int.self)
                functionTokens[time] = output.functionLogits[0..., -1].argMax().item(Int.self)
            }

            // Frame zero warms the language model but requests no TTS code,
            // matching the reference loop.
            if time == 0 {
                generatedFrames.append(zeroFrame)
                continue
            }

            let current = MLXArray([Int32(textTokens[time])]).reshaped([1, 1])
            if textTokens[time] == config.eosTokenId {
                previousCode = MLX.broadcast(
                    ttsModel.codecSilenceTokens.reshaped([1, 1, -1]), to: previousCode.shape)
            }
            let ttsOutput = ttsModel.ttsModel.step(
                code: previousCode,
                subwordIds: current,
                subwordMask: MLXArray.ones(current.shape, dtype: .bool),
                cache: ttsCache,
                guidance: true)
            previousCode = ttsOutput.codes
            generatedFrames.append(previousCode)
            MLX.eval(previousCode, output.textLogits, output.functionLogits)
        }

        var codes = MLX.concatenated(generatedFrames, axis: 1)
        codes = codes[0..., promptFrames..., 0...]
        codes = replacingControlCodes(codes)
        let decoded = ttsModel.audioCodec.decode(codes.transposed(0, 2, 1))
        MLX.eval(decoded)

        return VoiceChatTurnResult(
            textTokens: Array(textTokens[promptFrames...]),
            functionTokens: Array(functionTokens[promptFrames...]),
            audioCodes: codes[0],
            audio: decoded[0, 0],
            sampleRate: config.outputSampleRate,
            asrEmbeddings: asrEmbeds[0..., ..<audioFrames, 0...],
            audioFrames: audioFrames)
    }

    /// Greedy RNN-T transcription of the user's own audio for this turn.
    public func transcribeUser(_ result: VoiceChatTurnResult) -> [Int] {
        var state = VoiceChatRNNTDecodeState()
        return sttModel.transcribe(encoded: result.asrEmbeddings, state: &state)
    }
}

extension VoiceChatSpeechDecoder {
    /// Control-code ids as Swift ints (protocol markers substituted with
    /// silence before rendering).
    var controlCodesValues: [Int] {
        controlCodes.asArray(Int32.self).map(Int.init)
    }
}
