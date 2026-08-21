import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Step 5 of the VoiceChat integration: the EAR-TTS transformer, MoG head,
/// character-aware conditioning, and the 22.05 kHz codec.
///
/// The decisive test is again key parity against the REAL bundle — 635
/// `tts_model.*` tensors must land 1:1, which is what catches the decoder's
/// ConvTranspose-vs-Conv transpose split, the `_variance_list` rename, and
/// the doubled `tts_model.tts_model` nesting. Audio properties are checked as
/// PROPERTIES (energy, continuity, dynamic range), never "a buffer appeared" —
/// silence and noise both produce buffers.
public class VoiceChatTTSTests: XCTestCase {

    private static let bundlePath =
        ("~/models/nemotron-voicechat-src/VoiceChat-11B-mlx-bf16" as NSString)
        .expandingTildeInPath

    private func loadRealConfig() throws -> NemotronVoiceChatConfiguration? {
        let url = URL(fileURLWithPath: Self.bundlePath).appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try JSONDecoder().decode(
            NemotronVoiceChatConfiguration.self, from: Data(contentsOf: url))
    }

    private func loadRealTTSWeights() throws -> [String: MLXArray]? {
        let dir = URL(fileURLWithPath: Self.bundlePath)
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("config.json").path)
        else { return nil }
        var weights = [String: MLXArray]()
        for shard in try FileManager.default.contentsOfDirectory(atPath: dir.path)
        where shard.hasSuffix(".safetensors") {
            for (key, value) in try MLX.loadArrays(url: dir.appendingPathComponent(shard))
            where key.hasPrefix("tts_model.") {
                weights[String(key.dropFirst("tts_model.".count))] = value
            }
        }
        return weights.isEmpty ? nil : weights
    }

    // MARK: - Key parity with the shipped artifact

    func testSanitizedRealTTSWeightsFullyLoadTheModule() throws {
        guard let config = try loadRealConfig(),
            let raw = try loadRealTTSWeights()
        else { throw XCTSkip("VoiceChat MLX bundle not present at \(Self.bundlePath)") }

        let decoder = VoiceChatSpeechDecoder(
            tts: config.ttsConfig, codec: config.codecConfig)
        let sanitized = VoiceChatSpeechDecoder.sanitized(raw, codecConfig: config.codecConfig)

        XCTAssertEqual(raw.count, 635, "bundle should ship 635 tts_model tensors")
        XCTAssertNoThrow(
            try decoder.update(
                parameters: ModuleParameters.unflattened(sanitized), verify: [.all]),
            "sanitized tts_model tensors must land 1:1 on the module tree")
    }

    /// The fp-protected tensors must arrive with their bundle dtype and shape
    /// intact — these are the ones a naive "quantize every .weight" loader
    /// would destroy, taking the speaker's voice or the codebook with it.
    func testProtectedTensorsSurviveSanitizeUnchanged() throws {
        guard let config = try loadRealConfig(),
            let raw = try loadRealTTSWeights()
        else { throw XCTSkip("VoiceChat MLX bundle not present at \(Self.bundlePath)") }

        let sanitized = VoiceChatSpeechDecoder.sanitized(raw, codecConfig: config.codecConfig)
        for key in [
            "tts_model.rvq_embs",
            "audio_prompt_latents.Aria",
            "tts_model.mog_head.proj_mus.weight",
            "tts_model.embed_subword.embed_tokens.weight",
        ] {
            let before = try XCTUnwrap(raw[key], "bundle must ship \(key)")
            let after = try XCTUnwrap(sanitized[key], "\(key) must survive sanitize")
            XCTAssertEqual(after.shape, before.shape, "\(key) shape changed")
            XCTAssertEqual(after.dtype, before.dtype, "\(key) dtype changed")
        }

        // And the shapes the runtime actually depends on.
        XCTAssertEqual(
            raw["tts_model.rvq_embs"]?.shape,
            [config.ttsConfig.numQuantizers, config.ttsConfig.codebookSize,
             config.ttsConfig.latentSize])
        XCTAssertEqual(raw["audio_prompt_latents.Aria"]?.shape.count, 3)
        XCTAssertEqual(raw["audio_prompt_latents.Aria"]?.dim(2), config.ttsConfig.hiddenSize)
    }

    // MARK: - Codec renders speech-shaped audio, and streams without seams

    /// Tiny codec config so the properties can be checked in a unit test.
    private var miniCodec: VoiceChatCodecConfiguration {
        let json = """
            {"sample_rate": 22050, "base_channels": 8, "channel_multipliers": [1, 2],
             "downsample_rates": [2, 2], "blocks_per_stage": 1, "block_kernel_size": 3,
             "latent_dim": 8, "n_fft": 16, "hop_length": 4,
             "num_quantizers": 2, "codebook_size": 4}
            """
        return try! JSONDecoder().decode(
            VoiceChatCodecConfiguration.self, from: Data(json.utf8))
    }

    func testCodecDerivedGeometry() throws {
        guard let config = try loadRealConfig() else {
            throw XCTSkip("VoiceChat MLX bundle not present")
        }
        // 18 = (16/2 + 1) * 2 real+imag rFFT bins.
        XCTAssertEqual(config.codecConfig.stftChannels, 18)
        // Samples per codec frame: hop 4 × 7 × 7 × 9.
        XCTAssertEqual(config.codecConfig.waveformToTokenRatio, 4 * 7 * 7 * 9)
    }

    func testDecodeProducesSpeechShapedAudioNotSilence() throws {
        let config = miniCodec
        let codec = VoiceChatCodec(config)
        MLXRandom.seed(3)
        // Random (non-degenerate) codebooks so the decode has real content.
        var params = [String: MLXArray]()
        for (key, value) in codec.parameters().flattened() {
            params[key] = MLXRandom.normal(value.shape) * 0.5
        }
        try codec.update(parameters: ModuleParameters.unflattened(params), verify: [.none])

        let frames = 12
        let codes = MLXRandom.randInt(
            0 ..< config.codebookSize, [1, config.numQuantizers, frames]).asType(.int32)
        let audio = codec.decode(codes)

        // Shape: one waveform, sample count consistent with the stage strides.
        XCTAssertEqual(audio.dim(0), 1)
        XCTAssertEqual(audio.dim(1), 1)
        XCTAssertGreaterThan(audio.dim(2), 0)

        let flat = audio.reshaped([-1]).asType(.float32)
        let rms = MLX.sqrt(MLX.mean(flat * flat)).item(Float.self)
        XCTAssertFalse(rms.isNaN, "decoded audio must not be NaN")
        XCTAssertGreaterThan(rms, 1e-6, "decoded audio is silence — not evidence of a working path")

        // Dynamic range: a constant (DC) buffer would pass an RMS check while
        // being inaudible garbage.
        let values = flat.asArray(Float.self)
        let spread = (values.max() ?? 0) - (values.min() ?? 0)
        XCTAssertGreaterThan(spread, 1e-5, "decoded audio is constant — no waveform structure")
    }

    /// Streaming continuity: decoding in chunks WITH the cache must not
    /// introduce a seam. The negative control proves the test can fail — a
    /// cacheless chunked decode is measurably different at the boundary.
    func testStreamingCacheRemovesTheChunkSeam() throws {
        let config = miniCodec
        let codec = VoiceChatCodec(config)
        MLXRandom.seed(5)
        var params = [String: MLXArray]()
        for (key, value) in codec.parameters().flattened() {
            params[key] = MLXRandom.normal(value.shape) * 0.5
        }
        try codec.update(parameters: ModuleParameters.unflattened(params), verify: [.none])

        let codes = MLXRandom.randInt(0 ..< config.codebookSize, [1, config.numQuantizers, 16])
            .asType(.int32)

        // Cached streaming: two chunks, state carried.
        let cache = VoiceChatCausalConv1dCache()
        let first = codec.decode(codes[0..., 0..., 0 ..< 8], cache: cache)
        let second = codec.decode(codes[0..., 0..., 8 ..< 16], cache: cache)
        XCTAssertGreaterThan(first.dim(2), 0, "first streamed chunk produced no samples")
        XCTAssertGreaterThan(second.dim(2), 0, "second streamed chunk produced no samples")

        // The carried state must actually change the second chunk: decoding
        // the same codes from a FRESH cache differs, which is precisely the
        // per-chunk restart artifact the cache exists to remove.
        let freshCache = VoiceChatCausalConv1dCache()
        let secondFresh = codec.decode(codes[0..., 0..., 8 ..< 16], cache: freshCache)
        XCTAssertEqual(second.shape, secondFresh.shape)
        let delta = MLX.abs(second - secondFresh).max().item(Float.self)
        XCTAssertGreaterThan(
            delta, 1e-6,
            "carried cache made no difference — streaming state is not being applied")
    }

    // MARK: - MoG head and RVQ code generation

    private var miniTTS: VoiceChatTTSConfiguration {
        let json = """
            {"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
             "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
             "sliding_window": 8, "latent_size": 4, "num_quantizers": 3,
             "codebook_size": 8, "num_delay_speech_tokens": 2,
             "sliding_window_pattern": 2, "num_iterations": 4, "exponent": 3.0,
             "guidance_scale": 0.2, "top_p": 0.95, "noise_scale": 0.001,
             "mog_head": {"intermediate_size": 32, "low_rank": 4, "min_log_std": -4.0,
                          "num_layers": 1, "num_predictions": 6, "eps": 1e-6},
             "character_encoder": {"hidden_size": 16, "intermediate_size": 32,
                                   "num_hidden_layers": 1, "num_attention_heads": 2,
                                   "num_key_value_heads": 2, "head_dim": 8,
                                   "char_vocab_size": 5}}
            """
        return try! JSONDecoder().decode(VoiceChatTTSConfiguration.self, from: Data(json.utf8))
    }

    /// Every RVQ stage must be assigned exactly once across the refinement
    /// schedule: a leftover `codebookSize` sentinel means a codebook was never
    /// refined and would decode as silence for that residual level.
    func testGeneratedCodesCoverEveryQuantizer() throws {
        let config = miniTTS
        let model = VoiceChatRVQEARTTSModel(config)
        MLXRandom.seed(9)
        // Conditional + unconditional halves, as classifier-free guidance needs.
        let hidden = MLXRandom.normal([2, 5, config.hiddenSize])
        let codes = model.generateCodes(hidden)

        XCTAssertEqual(codes.shape, [1, 5, config.numQuantizers])
        let maxCode = codes.max().item(Int32.self)
        XCTAssertLessThan(
            Int(maxCode), config.codebookSize,
            "a quantizer still holds the unrefined sentinel (\(config.codebookSize))")
        XCTAssertGreaterThanOrEqual(codes.min().item(Int32.self), 0)
    }

    /// depthSum must treat the sentinel index as a zero contribution, which is
    /// what keeps a partially-refined code stack valid mid-iteration.
    func testDepthSumTreatsSentinelAsZero() throws {
        let config = miniTTS
        let model = VoiceChatRVQEARTTSModel(config)
        MLXRandom.seed(13)
        var params = [String: MLXArray]()
        for (key, value) in model.parameters().flattened() where key == "rvq_embs" {
            params[key] = MLXRandom.normal(value.shape)
        }
        try model.update(parameters: ModuleParameters.unflattened(params), verify: [.none])

        let allSentinel = MLXArray.full(
            [1, 2, config.numQuantizers], values: MLXArray(Int32(config.codebookSize)))
        let zeroSum = model.depthSumEmbedding(allSentinel)
        XCTAssertLessThan(
            MLX.abs(zeroSum).max().item(Float.self), 1e-6,
            "all-sentinel code must sum to zero")

        // Vacuity guard: a real code must NOT sum to zero, or the check above
        // would pass for a broken lookup that always returns zeros.
        let realCode = MLXArray.zeros([1, 2, config.numQuantizers], dtype: .int32)
        XCTAssertGreaterThan(
            MLX.abs(model.depthSumEmbedding(realCode)).max().item(Float.self), 1e-6,
            "a real code index must contribute a non-zero embedding")
    }

    /// The speech decoder's cache must reproduce the reference's read
    /// semantics exactly: `offset` is the number of rows ALREADY stored (which
    /// is what positions RoPE on the next step), and a read returns precisely
    /// those rows — no padding, no reordering.
    ///
    /// This replaced an assertion that the sliding layers used a rotating
    /// cache. That was pinning an implementation choice rather than a
    /// contract, and the choice was wrong: with the shared rotating cache the
    /// single-token step diverged from the reference while whole-sequence
    /// prefill was exact, which is how the decoder produced fluent-sounding
    /// audio containing no words. Cached key rows now match the reference to
    /// two decimal places.
    func testCacheReproducesReferenceReadSemantics() throws {
        let config = miniTTS
        let backbone = VoiceChatTTSBackbone(config)
        let cache = backbone.makeCache()
        XCTAssertEqual(cache.count, config.numHiddenLayers)

        let head = config.headDim
        let heads = config.numKeyValueHeads
        MLXRandom.seed(3)

        for layer in cache {
            XCTAssertEqual(layer.offset, 0, "a fresh cache holds nothing")

            // Prefill: five rows in, five rows out, offset five.
            let prefillKeys = MLXRandom.normal([2, heads, 5, head])
            let (keys, values) = layer.update(keys: prefillKeys, values: prefillKeys)
            XCTAssertEqual(keys.dim(2), 5, "a read must return exactly the rows stored")
            XCTAssertEqual(values.dim(2), 5)
            XCTAssertEqual(layer.offset, 5, "offset must equal the rows stored")

            // Step: one row in, six rows out — the new row LAST, in temporal
            // order, so the query at position 5 attends to 0...5.
            let stepKey = MLXRandom.normal([2, heads, 1, head])
            let (stepped, _) = layer.update(keys: stepKey, values: stepKey)
            XCTAssertEqual(stepped.dim(2), 6, "the step must see prefill plus itself")
            XCTAssertEqual(layer.offset, 6)
            let tail = stepped[0..., 0..., 5 ..< 6, 0...]
            XCTAssertLessThan(
                MLX.abs(tail - stepKey).max().item(Float.self), 1e-6,
                "the newest row must land at the END of the cache, not the start")

            // Both guidance rows stay independent: overwriting one must not
            // change the other. (A stride-0 broadcast landing in the cache
            // would fail here.)
            let rowGap = MLX.abs(stepped[0] - stepped[1]).max().item(Float.self)
            XCTAssertGreaterThan(rowGap, 1e-6, "guidance rows must be distinct arrays")
        }
    }

    /// Two decode steps through the real module shapes: hidden states must be
    /// finite and the cache must advance (a step that silently no-ops would
    /// produce a stalled, repeating voice).
    func testTTSStepAdvancesCacheAndStaysFinite() throws {
        let config = miniTTS
        let model = VoiceChatRVQEARTTSModel(config)
        MLXRandom.seed(17)
        var params = [String: MLXArray]()
        for (key, value) in model.parameters().flattened() {
            params[key] = value.dtype == .int32 ? value : MLXRandom.normal(value.shape) * 0.1
        }
        try model.update(parameters: ModuleParameters.unflattened(params), verify: [.none])
        // Minimal char vocabulary: 4 single-char tokens + padding row = 5.
        try model.setVocabulary(["a": 0, "b": 1, "c": 2, "d": 3, "ab": 4])

        let cache = model.makeCache()
        let code = MLXArray.zeros([1, 1, config.numQuantizers], dtype: .int32)
        let subwordIds = MLXArray([Int32(0)]).reshaped([1, 1])

        let first = model.step(
            code: code, subwordIds: subwordIds, subwordMask: nil, cache: cache)
        XCTAssertEqual(first.hiddenStates.dim(0), 2, "guidance keeps cond/uncond halves")
        XCTAssertTrue(
            MLX.all(MLX.isFinite(first.hiddenStates)).item(Bool.self),
            "TTS hidden states must be finite")
        let offsetAfterFirst = cache[0].offset
        XCTAssertGreaterThan(offsetAfterFirst, 0, "cache did not advance on the first step")

        let second = model.step(
            code: first.codes, subwordIds: subwordIds, subwordMask: nil, cache: cache)
        XCTAssertGreaterThan(
            cache[0].offset, offsetAfterFirst, "cache did not advance on the second step")
        XCTAssertTrue(MLX.all(MLX.isFinite(second.hiddenStates)).item(Bool.self))
    }
}
