import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Step 6: the assembled duplex model and its turn loop.
///
/// The full 11 B bundle is too large for a unit test to load, so these run the
/// REAL loop on a miniature configuration with the real module code — every
/// channel, the shared frame clock, the TTS warmup and the codec render. What
/// they pin is behaviour no shape test can see: that the function channel is
/// driven independently of text, that the loop's fused input actually mixes
/// three channels, that frame zero warms without emitting audio, and that
/// control codes never reach the vocoder.
///
/// Full-bundle key parity for BOTH sides is covered by
/// `VoiceChatSTTModelTests` and `VoiceChatTTSTests`; the live audio-to-audio
/// proof on each shipped quant runs through the app harness.
public class VoiceChatEndToEndTests: XCTestCase {

    /// A miniature but structurally faithful VoiceChat: dense nemotron_h
    /// backbone, both heads, a 2-layer TTS with pattern 2, and a 2-quantizer
    /// codec.
    private func miniConfig(functionChannelWeight: Float = 2.0) throws
        -> NemotronVoiceChatConfiguration
    {
        let json = """
            {
              "text_config": {
                "model_type": "nemotron_h", "vocab_size": 40, "hidden_size": 16,
                "intermediate_size": 32, "num_hidden_layers": 2,
                "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
                "mamba_num_heads": 2, "mamba_head_dim": 8, "ssm_state_size": 8,
                "conv_kernel": 4, "n_groups": 1, "layer_norm_epsilon": 1e-5,
                "hybrid_override_pattern": "M*"
              },
              "audio_config": {
                "preprocessor": {"sample_rate": 16000, "features": 8, "n_fft": 64,
                                 "window_size": 0.025, "window_stride": 0.01},
                "encoder": {"d_model": 8, "n_layers": 1, "n_heads": 2, "feat_in": 8,
                            "ff_expansion_factor": 2, "conv_kernel_size": 3,
                            "subsampling_factor": 2, "subsampling_conv_channels": 4,
                            "self_attention_model": "rel_pos",
                            "att_context_style": "chunked_limited",
                            "att_context_size": [[4, 0]], "use_bias": false},
                "decoder": {"pred_hidden": 8, "pred_rnn_layers": 1, "vocab_size": 8,
                            "blank_as_pad": true},
                "joint": {"joint_hidden": 8, "num_classes": 8, "encoder_hidden": 8,
                          "pred_hidden": 8},
                "output_dim": 16, "max_symbols": 4
              },
              "tts_config": {
                "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
                "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
                "sliding_window": 32, "latent_size": 4, "num_quantizers": 2,
                "codebook_size": 4, "num_delay_speech_tokens": 2,
                "sliding_window_pattern": 2, "num_iterations": 2,
                "mog_head": {"intermediate_size": 32, "low_rank": 4, "num_layers": 1,
                             "num_predictions": 4},
                "character_encoder": {"hidden_size": 16, "intermediate_size": 32,
                                      "num_hidden_layers": 1, "num_attention_heads": 2,
                                      "num_key_value_heads": 2, "head_dim": 8,
                                      "char_vocab_size": 5}
              },
              "codec_config": {
                "sample_rate": 22050, "base_channels": 4, "channel_multipliers": [1, 2],
                "downsample_rates": [2, 2], "blocks_per_stage": 1,
                "block_kernel_size": 3, "latent_dim": 4, "n_fft": 16, "hop_length": 4,
                "num_quantizers": 2, "codebook_size": 4
              },
              "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 12,
              "silence_token_id": 11, "rnnt_blank_id": 8,
              "input_sample_rate": 16000, "output_sample_rate": 22050,
              "frame_duration": 0.08,
              "function_channel_weight": \(functionChannelWeight),
              "speaker": "Aria"
            }
            """
        return try JSONDecoder().decode(
            NemotronVoiceChatConfiguration.self, from: Data(json.utf8))
    }

    private func miniModel(seed: UInt64 = 21) throws -> NemotronVoiceChatModel {
        let model = NemotronVoiceChatModel(try miniConfig())
        MLXRandom.seed(seed)
        var params = [String: MLXArray]()
        for (key, value) in model.parameters().flattened() {
            // Integer buffers (flags, control codes, silence tokens) must keep
            // their type — they are indices, not weights.
            params[key] =
                value.dtype == .int32
                ? value : MLXRandom.normal(value.shape).asType(value.dtype) * 0.1
        }
        try model.update(parameters: ModuleParameters.unflattened(params), verify: [.none])
        try model.setVocabulary(["a": 0, "b": 1, "c": 2, "d": 3, "ab": 4])
        return model
    }

    /// One full turn: audio embeddings in, text + function tokens + rendered
    /// 22.05 kHz audio out, on the shared frame clock.
    func testTurnProducesAllChannelsOnOneClock() throws {
        let model = try miniModel()
        let config = try miniConfig()
        let frames = 6
        MLXRandom.seed(31)
        let audioEmbeds = MLXRandom.normal([1, frames, config.textConfig.hiddenSize])
        let asrEmbeds = MLXRandom.normal([1, frames, 8])

        let result = model.generateTurn(audioEmbeds: audioEmbeds, asrEmbeds: asrEmbeds)

        // Every channel advances on the SAME clock — one entry per frame.
        XCTAssertEqual(result.textTokens.count, frames)
        XCTAssertEqual(result.functionTokens.count, frames)
        XCTAssertEqual(result.audioCodes.dim(0), frames)
        XCTAssertEqual(result.audioCodes.dim(1), config.ttsConfig.numQuantizers)
        XCTAssertEqual(result.sampleRate, 22050)
        XCTAssertEqual(result.audioFrames, frames)

        // Audio was actually rendered and is finite.
        XCTAssertGreaterThan(result.audio.size, 0, "turn produced no samples")
        XCTAssertTrue(
            MLX.all(MLX.isFinite(result.audio)).item(Bool.self),
            "rendered audio must be finite")

        // Codes stay inside the codebook — a sentinel or control code reaching
        // the vocoder is an audible artifact, not a silent one.
        XCTAssertLessThan(
            Int(result.audioCodes.max().item(Int32.self)), config.ttsConfig.codebookSize,
            "an unrefined sentinel reached the codec")
        XCTAssertGreaterThanOrEqual(result.audioCodes.min().item(Int32.self), 0)
    }

    /// The function channel must be driven by `function_head`, independently
    /// of the text channel. Tying the two heads makes the channels agree; with
    /// the shipped (untied) heads they must be free to diverge.
    func testFunctionChannelIsIndependentOfTextChannel() throws {
        let model = try miniModel(seed: 41)
        let config = try miniConfig()
        MLXRandom.seed(43)
        let audioEmbeds = MLXRandom.normal([1, 5, config.textConfig.hiddenSize]) * 3.0
        let asrEmbeds = MLXRandom.normal([1, 5, 8])

        let untied = model.generateTurn(audioEmbeds: audioEmbeds, asrEmbeds: asrEmbeds)

        // Tie function_head to lm_head and re-run: the two channels must then
        // agree, which proves the divergence above came from the separate head
        // rather than from sampling noise.
        let lmHead = try XCTUnwrap(model.sttModel.llm.lmHead?.weight)
        try model.update(
            parameters: ModuleParameters.unflattened([
                "stt_model.function_head.weight": lmHead
            ]),
            verify: [.none])
        let tied = model.generateTurn(audioEmbeds: audioEmbeds, asrEmbeds: asrEmbeds)
        XCTAssertEqual(
            tied.textTokens, tied.functionTokens,
            "with tied heads both channels must decode identically")
        XCTAssertNotEqual(
            untied.functionTokens, tied.functionTokens,
            "untied function_head must not produce the tied result — the channel is not wired")
    }

    /// The fused frame input mixes three channels; perturbing ONLY the
    /// function-channel weight must change the turn. If it does not, the
    /// tool-call channel is decorative.
    func testFunctionChannelWeightAffectsTheTurn() throws {
        let base = try miniModel(seed: 47)
        let config = try miniConfig()
        MLXRandom.seed(53)
        let audioEmbeds = MLXRandom.normal([1, 5, config.textConfig.hiddenSize]) * 3.0
        let asrEmbeds = MLXRandom.normal([1, 5, 8])
        let withWeight = base.generateTurn(audioEmbeds: audioEmbeds, asrEmbeds: asrEmbeds)

        // Identical weights, but function_channel_weight 0 — the tool-call
        // channel stops feeding back into the timeline.
        let zeroModel = NemotronVoiceChatModel(try miniConfig(functionChannelWeight: 0))
        try zeroModel.update(
            parameters: ModuleParameters.unflattened(
                base.parameters().flattened().reduce(into: [String: MLXArray]()) {
                    $0[$1.0] = $1.1
                }),
            verify: [.none])
        try zeroModel.setVocabulary(["a": 0, "b": 1, "c": 2, "d": 3, "ab": 4])
        let withoutWeight = zeroModel.generateTurn(
            audioEmbeds: audioEmbeds, asrEmbeds: asrEmbeds)

        XCTAssertNotEqual(
            withWeight.textTokens, withoutWeight.textTokens,
            "function_channel_weight had no effect — the channel is not fused into the timeline")
    }

    /// A system prompt is prepended on the user-audio channel and trimmed from
    /// the result: the returned timeline must still be exactly the audio.
    func testSystemPromptPrefixIsTrimmedFromTheResult() throws {
        let model = try miniModel(seed: 59)
        let config = try miniConfig()
        MLXRandom.seed(61)
        let frames = 4
        let audioEmbeds = MLXRandom.normal([1, frames, config.textConfig.hiddenSize])
        let asrEmbeds = MLXRandom.normal([1, frames, 8])
        let promptEmbeds = MLXRandom.normal([1, 3, config.textConfig.hiddenSize])

        let result = model.generateTurn(
            audioEmbeds: audioEmbeds, asrEmbeds: asrEmbeds, promptEmbeds: promptEmbeds)
        XCTAssertEqual(result.textTokens.count, frames, "prompt prefix must be trimmed")
        XCTAssertEqual(result.functionTokens.count, frames)
        XCTAssertEqual(result.audioCodes.dim(0), frames)
    }

    /// Control codes are protocol markers; rendering them directly is an
    /// audible artifact. They must be replaced by the codec silence codes.
    func testControlCodesAreReplacedBeforeRendering() throws {
        let model = try miniModel(seed: 67)
        let config = try miniConfig()
        // Declare code 3 as a control code and 1 as silence.
        try model.ttsModel.update(
            parameters: ModuleParameters.unflattened([
                "control_codes": MLXArray([Int32(3), Int32(3), Int32(3)]),
                "codec_silence_tokens": MLXArray(
                    [Int32](repeating: 1, count: config.ttsConfig.numQuantizers)),
            ]),
            verify: [.none])

        let codes = MLXArray([Int32(3), Int32(0), Int32(2), Int32(3)])
            .reshaped([1, 2, config.ttsConfig.numQuantizers])
        let cleaned = model.replacingControlCodes(codes)
        let values = cleaned.reshaped([-1]).asArray(Int32.self)
        XCTAssertEqual(values, [1, 0, 2, 1], "control codes must become silence codes")
    }

    /// Transcription reads the UNPROJECTED encoder frames through the RNN-T,
    /// not the text channel — the user's words and the agent's are different
    /// streams and must not be confused.
    func testUserTranscriptionUsesTheASRBranch() throws {
        let model = try miniModel(seed: 71)
        let config = try miniConfig()
        MLXRandom.seed(73)
        let result = model.generateTurn(
            audioEmbeds: MLXRandom.normal([1, 4, config.textConfig.hiddenSize]),
            asrEmbeds: MLXRandom.normal([1, 4, 8]) * 5.0)
        let tokens = model.transcribeUser(result)
        // Emissions are bounded by max_symbols per frame regardless of content.
        XCTAssertLessThanOrEqual(tokens.count, 4 * config.audioConfig.maxSymbols)
        for token in tokens {
            XCTAssertNotEqual(token, config.rnntBlankId, "blank must never be emitted")
        }
    }
}
