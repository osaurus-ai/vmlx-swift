import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXVLM

/// Steps 3+4 of the VoiceChat integration: the fused STT model (both output
/// heads) and the RNN-T transducer.
///
/// The heavyweight test here is key parity against the REAL shipped bundle:
/// every tensor the bundle stores under `stt_model.` must land on exactly one
/// module parameter, and every module parameter must be fed by the bundle.
/// That is the test that catches a wrong key, a missed PyTorch→MLX transpose,
/// a forgotten LSTM bias sum, or a head silently left at random init — the
/// Ornith lesson being that a runtime can "load" and still have lost a whole
/// output channel.
public class VoiceChatSTTModelTests: XCTestCase {

    private static let bundlePath =
        ("~/models/nemotron-voicechat-src/VoiceChat-11B-mlx-bf16" as NSString)
        .expandingTildeInPath

    private func loadRealConfig() throws -> NemotronVoiceChatConfiguration? {
        let url = URL(fileURLWithPath: Self.bundlePath).appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try JSONDecoder().decode(
            NemotronVoiceChatConfiguration.self, from: Data(contentsOf: url))
    }

    /// All `stt_model.*` tensors from the real bundle, prefix stripped,
    /// loaded lazily (mmap) so nothing is materialised.
    private func loadRealSTTWeights() throws -> [String: MLXArray]? {
        let dir = URL(fileURLWithPath: Self.bundlePath)
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("config.json").path)
        else { return nil }
        var weights = [String: MLXArray]()
        let shards = try FileManager.default.contentsOfDirectory(atPath: dir.path)
            .filter { $0.hasSuffix(".safetensors") }
        for shard in shards {
            let arrays = try MLX.loadArrays(url: dir.appendingPathComponent(shard))
            for (key, value) in arrays where key.hasPrefix("stt_model.") {
                weights[String(key.dropFirst("stt_model.".count))] = value
            }
        }
        return weights.isEmpty ? nil : weights
    }

    // MARK: - Key parity with the shipped artifact

    func testSanitizedRealWeightsFullyLoadTheModule() throws {
        guard let config = try loadRealConfig(),
            let raw = try loadRealSTTWeights()
        else { throw XCTSkip("VoiceChat MLX bundle not present at \(Self.bundlePath)") }

        let model = VoiceChatSTTModel(config)
        let sanitized = VoiceChatSTTModel.sanitized(raw)

        // `.all` verifies BOTH directions at shape level: no unused bundle
        // tensor, no module parameter left at random init. Either failure is
        // exactly the silent-channel-loss class this model punishes.
        XCTAssertNoThrow(
            try model.update(
                parameters: ModuleParameters.unflattened(sanitized),
                verify: [.all]),
            "sanitized bundle tensors must land 1:1 on the module tree")
    }

    func testSanitizeMapsTheKnownTrapKeys() throws {
        // Value-independent key facts, checkable without the bundle.
        let dummy = MLXArray.zeros([2, 2])
        let mapped = VoiceChatSTTModel.sanitized([
            "embed_tokens.weight": dummy,
            "lm_head.weight": dummy,
            "llm.norm_f.weight": dummy,
            "rnnt_joint.enc.weight": dummy,
        ])
        XCTAssertNotNil(mapped["llm.backbone.embeddings.weight"])
        XCTAssertNotNil(mapped["llm.lm_head.weight"])
        XCTAssertNotNil(mapped["llm.backbone.norm_f.weight"])
        XCTAssertNotNil(mapped["rnnt_joint.enc.weight"])
        XCTAssertNil(mapped["embed_tokens.weight"], "flat key must be remapped, not duplicated")

        // The LSTM bias trap: PyTorch ships ih and hh bias separately; MLX
        // folds them into ONE vector. Dropping one silently halves the bias.
        let ih = MLXArray(converting: [1.0, 2.0])
        let hh = MLXArray(converting: [10.0, 20.0])
        let lstm = VoiceChatRNNT.sanitized([
            "rnnt_decoder.prediction.dec_rnn.lstm.weight_ih_l0": dummy,
            "rnnt_decoder.prediction.dec_rnn.lstm.weight_hh_l0": dummy,
            "rnnt_decoder.prediction.dec_rnn.lstm.bias_ih_l0": ih,
            "rnnt_decoder.prediction.dec_rnn.lstm.bias_hh_l0": hh,
        ])
        XCTAssertNotNil(lstm["rnnt_decoder.prediction.dec_rnn.lstm.0.Wx"])
        XCTAssertNotNil(lstm["rnnt_decoder.prediction.dec_rnn.lstm.0.Wh"])
        let bias = try XCTUnwrap(lstm["rnnt_decoder.prediction.dec_rnn.lstm.0.bias"])
        XCTAssertEqual(bias[0].item(Float.self), 11.0, accuracy: 1e-6)
        XCTAssertEqual(bias[1].item(Float.self), 22.0, accuracy: 1e-6)
    }

    // MARK: - RNN-T greedy decode properties (real transducer weights)

    private func realTransducer() throws
        -> (VoiceChatPredictNetwork, VoiceChatJointNetwork, NemotronVoiceChatConfiguration)?
    {
        guard let config = try loadRealConfig(),
            let raw = try loadRealSTTWeights()
        else { return nil }
        let decoder = VoiceChatPredictNetwork(config: config.audioConfig.decoder!)
        let joint = VoiceChatJointNetwork(config: config.audioConfig.joint!)
        let sanitized = VoiceChatRNNT.sanitized(raw)
        let decWeights = sanitized.filter { $0.key.hasPrefix("rnnt_decoder.") }
            .reduce(into: [String: MLXArray]()) {
                $0[String($1.key.dropFirst("rnnt_decoder.".count))] = $1.value
            }
        let jointWeights = sanitized.filter { $0.key.hasPrefix("rnnt_joint.") }
            .reduce(into: [String: MLXArray]()) {
                $0[String($1.key.dropFirst("rnnt_joint.".count))] = $1.value
            }
        try decoder.update(parameters: ModuleParameters.unflattened(decWeights), verify: [.all])
        try joint.update(parameters: ModuleParameters.unflattened(jointWeights), verify: [.all])
        return (decoder, joint, config)
    }

    /// Chunked decode with carried state must equal whole-sequence decode.
    /// The prediction LSTM advances only on emission, so a state reset at a
    /// chunk boundary degrades streaming ASR while every whole-file test
    /// stays green — the exact class of bug live audio exists to catch.
    func testGreedyDecodeChunkCarryEqualsWhole() throws {
        guard let (decoder, joint, config) = try realTransducer() else {
            throw XCTSkip("VoiceChat MLX bundle not present at \(Self.bundlePath)")
        }
        MLXRandom.seed(7)
        let encoded = MLXRandom.normal([1, 14, 1024]) * 2.0

        var whole = VoiceChatRNNTDecodeState()
        let wholeTokens = VoiceChatRNNT.greedyDecode(
            encoded: encoded, decoder: decoder, joint: joint,
            blankId: config.rnntBlankId, maxSymbols: config.audioConfig.maxSymbols,
            state: &whole)

        var chunked = VoiceChatRNNTDecodeState()
        var chunkedTokens = [Int]()
        var offset = 0
        for size in [5, 3, 4, 2] {
            let piece = encoded[0..., offset ..< (offset + size), 0...]
            chunkedTokens += VoiceChatRNNT.greedyDecode(
                encoded: piece, decoder: decoder, joint: joint,
                blankId: config.rnntBlankId, maxSymbols: config.audioConfig.maxSymbols,
                state: &chunked)
            offset += size
        }

        XCTAssertEqual(wholeTokens, chunkedTokens, "carried decode state must make chunking invisible")

        // Vacuity guard: a decode that never emits proves nothing about state
        // carry. Amplified random frames through the real joint must emit.
        XCTAssertFalse(wholeTokens.isEmpty, "test input produced no emissions — property not exercised")
        // And the bound that keeps a pathological joint from looping forever:
        XCTAssertLessThanOrEqual(
            wholeTokens.count, 14 * config.audioConfig.maxSymbols,
            "emissions must respect max_symbols per frame")
    }

    /// Blank must terminate a frame without touching the prediction state:
    /// feeding pure-silence-like frames (joint driven to blank) emits nothing.
    func testAllBlankFramesEmitNothing() throws {
        guard let (decoder, joint, config) = try realTransducer() else {
            throw XCTSkip("VoiceChat MLX bundle not present at \(Self.bundlePath)")
        }
        // Zero encoder frames: the trained joint overwhelmingly favours blank
        // on silence-shaped input. If this ever emits, the assertion tells us
        // the real distribution changed rather than silently absorbing it.
        let silent = MLXArray.zeros([1, 6, 1024])
        var state = VoiceChatRNNTDecodeState()
        let tokens = VoiceChatRNNT.greedyDecode(
            encoded: silent, decoder: decoder, joint: joint,
            blankId: config.rnntBlankId, maxSymbols: config.audioConfig.maxSymbols,
            state: &state)
        XCTAssertTrue(tokens.isEmpty, "zero frames should decode to blank only, got \(tokens)")
    }

    // MARK: - The function channel is a separate head, not parsed text

    func testFunctionChannelIsAnIndependentHead() throws {
        // Tiny synthetic dense nemotron_h so the whole model fits a unit test.
        let json = """
            {
              "text_config": {
                "model_type": "nemotron_h", "vocab_size": 48, "hidden_size": 16,
                "intermediate_size": 32, "num_hidden_layers": 2,
                "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
                "mamba_num_heads": 2, "mamba_head_dim": 8, "ssm_state_size": 8,
                "conv_kernel": 4, "n_groups": 1, "layer_norm_epsilon": 1e-5,
                "hybrid_override_pattern": "M-"
              },
              "audio_config": {
                "preprocessor": {"sample_rate": 16000, "features": 8, "n_fft": 64,
                                 "window_size": 0.025, "window_stride": 0.01},
                "encoder": {"d_model": 8, "n_layers": 1, "n_heads": 2, "feat_in": 8,
                            "ff_expansion_factor": 2, "conv_kernel_size": 3,
                            "subsampling_factor": 2, "subsampling_conv_channels": 4,
                            "self_attention_model": "rel_pos",
                            "att_context_size": [[4, 0]]},
                "decoder": {"pred_hidden": 8, "pred_rnn_layers": 1, "vocab_size": 8,
                            "blank_as_pad": true},
                "joint": {"joint_hidden": 8, "num_classes": 8, "encoder_hidden": 8,
                          "pred_hidden": 8},
                "output_dim": 16, "max_symbols": 4
              },
              "tts_config": {
                "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 4,
                "sliding_window": 16, "latent_size": 4, "num_quantizers": 2,
                "codebook_size": 8, "num_delay_speech_tokens": 2
              },
              "codec_config": {
                "sample_rate": 22050, "base_channels": 4, "channel_multipliers": [1, 1, 1],
                "downsample_rates": [2, 2, 2], "blocks_per_stage": 1,
                "block_kernel_size": 3, "latent_dim": 4, "n_fft": 16, "hop_length": 4,
                "num_quantizers": 2, "codebook_size": 8
              },
              "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 12,
              "silence_token_id": 11, "rnnt_blank_id": 8,
              "input_sample_rate": 16000, "output_sample_rate": 22050,
              "frame_duration": 0.08, "function_channel_weight": 2.0
            }
            """
        let config = try JSONDecoder().decode(
            NemotronVoiceChatConfiguration.self, from: Data(json.utf8))
        let model = VoiceChatSTTModel(config)

        MLXRandom.seed(11)
        let embeds = MLXRandom.normal([1, 3, 16])
        let out = model(inputsEmbeds: embeds, cache: nil)

        XCTAssertEqual(out.textLogits.shape, [1, 3, 48])
        XCTAssertEqual(out.functionLogits.shape, [1, 3, 48])

        // Independent heads at random init must disagree…
        let disagreement = MLX.abs(out.textLogits - out.functionLogits).max().item(Float.self)
        XCTAssertGreaterThan(
            disagreement, 1e-3,
            "function channel produced identical logits to text — one head is being read twice")

        // …and the vacuity guard: with the SAME weights in both heads the
        // channels must agree, proving the disagreement above comes from the
        // heads and not from nondeterminism elsewhere.
        let tied = model.llm.lmHead!.weight
        try model.update(
            parameters: ModuleParameters.unflattened(["function_head.weight": tied]),
            verify: [.none])
        let outTied = model(inputsEmbeds: embeds, cache: nil)
        let agreement = MLX.abs(outTied.textLogits - outTied.functionLogits).max().item(Float.self)
        XCTAssertLessThan(agreement, 1e-5, "tied heads must produce identical channels")
    }
}
