// Pin Gemma4 audio-input contract + image-token-id resolution contracts via
// source-coverage tests (no MLX runtime required).
//
// Background (see `docs/GEMMA4-DEEP-TRACE-2026-05-10.md` §7.4 + §7.5):
//
// §7.5 Audio contract. Gemma4 bundles ship `embed_audio.embedding_projection`
// for pre-encoded early-fusion audio features. E-series bundles also ship an
// `audio_tower`, but raw audio feature extraction is not implemented in Swift.
// `Gemma4.prepare` must project pre-encoded audio at the configured width and
// typed-refuse raw audio instead of silently dropping the lane.
//
// §7.4 Image-token-id resolution. The processor used to call
// `tokenizer.encode("<|image|>").last ?? 258880` — fragile because
// `encode()` defaults to `addSpecialTokens: true`, which can prepend
// BOS on some tokenizers and silently change the encoded length.
// `convertTokenToId("<|image|>")` skips that special-token machinery
// entirely and looks the special token up directly in the vocab.
// This test pins the new lookup form so a future regression that
// reintroduces `encode().last` is caught at test time.
//
// Source-coverage instead of full forward: `Gemma4.init` triggers
// MLX Metal allocations (Embedding + layers + vision tower), and the
// SwiftPM test runner here cannot load the mlx-swift `default.metallib`
// resource. The guard contract is purely structural — pinning the
// source pattern is sufficient to catch regressions.

import Foundation
import Testing

@Suite("Gemma4 audio contract + image-token-id resolution source coverage")
struct Gemma4AudioGuardTests {

    /// Resolves the absolute path of `Libraries/MLXVLM/Models/Gemma4.swift`
    /// relative to the test source file.
    private static func gemma4Source() throws -> String {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // .../Tests/MLXLMTests
            .deletingLastPathComponent()  // .../Tests
            .deletingLastPathComponent()  // repo root
        let url = repo
            .appendingPathComponent("Libraries")
            .appendingPathComponent("MLXVLM")
            .appendingPathComponent("Models")
            .appendingPathComponent("Gemma4.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// Pins the audio/video contract at the top of `Gemma4.prepare`.
    @Test("Gemma4.prepare supports pre-encoded audio and typed-refuses unsupported media")
    func audioProjectionContractIsPresent() throws {
        let source = try Self.gemma4Source()

        #expect(source.contains("if input.video != nil {"),
            "Gemma4.prepare must reject unsupported video before proceeding.")
        #expect(!source.contains("if input.audio != nil || input.video != nil {"),
            "Gemma4.prepare must not reject the audio lane.")

        #expect(source.contains("throw VLMError.processing("),
            "Unsupported media must surface as VLMError.processing, not fatalError or silent fallthrough.")
        #expect(source.contains("@ModuleInfo(key: \"embed_audio\") private var embedAudio"),
            "Gemma4 must keep the audio projection module.")
        #expect(!source.contains("nk.hasPrefix(\"embed_audio.\")"),
            "Gemma4 must not drop embed_audio projection weights.")
        #expect(source.contains("audioFeatures.dim(-1) == config.audioEmbedDim"),
            "Gemma4 must validate audio feature width before projection.")
        #expect(source.contains("let projectedAudio = embedAudio(audioFeatures).asType(emb.dtype)"),
            "Gemma4 must project audio features into the text embedding dimension.")
        #expect(!source.contains("pre-encoded 640-dim"),
            "Gemma4 raw/pre-encoded audio errors must not hard-code 640; E-series bundles use a different configured width.")
    }

    /// Pins the unified raw-waveform chunking contract
    /// (Gemma4UnifiedAudioFeatureExtractor parity).
    @Test("Gemma4Processor chunks raw waveforms for unified bundles and gates the mel/tower pipeline")
    func unifiedRawAudioChunkingContract() throws {
        let source = try Self.gemma4Source()

        // The unified extractor is encoder-free: raw 16 kHz frames of
        // `audio_samples_per_token` samples are the soft-token features.
        #expect(source.contains("unifiedWaveformFeatures"),
            "Gemma4Processor must implement unified raw-waveform chunking.")
        #expect(source.contains("Gemma4UnifiedAudioFeatureExtractor"),
            "Unified chunking must be gated on the bundle's feature_extractor_type.")
        #expect(source.contains("audio_samples_per_token"),
            "Frame width must come from processor_config.json feature_extractor, not a hard-coded constant.")
        #expect(source.contains("tokens > config.audioSeqLength"),
            "Unified chunking must cap soft tokens at audio_seq_length like the upstream extractor.")
        #expect(source.contains("linearResamplePCM"),
            "Raw PCM at non-16kHz rates must be resampled, not rejected or misinterpreted.")
        // E-series mel + conformer bundles now route raw audio through
        // gemma4ExtractMelFeatures instead of a typed refusal.
        #expect(
            source.contains(
                "config.audioFeatureExtractorType == \"Gemma4AudioFeatureExtractor\""),
            "Gemma4Processor must route Gemma4AudioFeatureExtractor bundles through the mel pipeline.")
        #expect(source.contains("gemma4ExtractMelFeatures"),
            "E-series raw audio must run through the Gemma4 mel extractor.")
    }

    /// Resolves `Libraries/MLXVLM/Models/Gemma4Audio.swift` (the E-series
    /// conformer tower + mel extractor).
    private static func gemma4AudioSource() throws -> String {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // .../Tests/MLXLMTests
            .deletingLastPathComponent()  // .../Tests
            .deletingLastPathComponent()  // repo root
        let url = repo
            .appendingPathComponent("Libraries")
            .appendingPathComponent("MLXVLM")
            .appendingPathComponent("Models")
            .appendingPathComponent("Gemma4Audio.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// Pins the E-series conformer audio tower contract:
    /// tower gated on audio_config, sanitize keeps audio_tower.* when the
    /// tower exists, and placeholder counts use the post-subsampling formula.
    @Test("Gemma4 instantiates the conformer audio tower only for gemma4_audio configs")
    func conformerAudioTowerContract() throws {
        let source = try Self.gemma4Source()
        let audioSource = try Self.gemma4AudioSource()

        // Tower instantiation is gated on audio_config presence + type.
        #expect(source.contains("audioConfig.isConformerTower"),
            "Gemma4.init must gate the audio tower on audio_config.model_type == gemma4_audio.")
        #expect(audioSource.contains("modelType == \"gemma4_audio\""),
            "isConformerTower must require model_type gemma4_audio so unified 12B bundles stay tower-free.")
        #expect(source.contains("@ModuleInfo(key: \"audio_tower\") private var audioTower"),
            "Gemma4 must register the audio tower under the audio_tower checkpoint prefix.")

        // sanitize() must keep audio_tower.* when the tower exists and keep
        // discarding it otherwise (unified 12B has no tower weights anyway;
        // 26B/31B have no audio_config at all).
        #expect(source.contains("guard hasAudioTower else { continue }"),
            "sanitize() must discard audio_tower.* only when no tower is instantiated.")
        #expect(!source.contains("if nk.hasPrefix(\"audio_tower.\") { continue }"),
            "sanitize() must not unconditionally drop audio_tower weights.")

        // Clipped-linear sidecars are consumed by the tower, not dropped.
        #expect(audioSource.contains("@ParameterInfo(key: \"input_min\")"),
            "Gemma4ClippedLinear must load the input_min/input_max/output_min/output_max scalars.")

        // Placeholder count parity: <|audio|> expansion must use the
        // post-subsampling formula (two stride-2 convs => ceil(T/4)),
        // matching HF Gemma4Processor.replace_audio_token.
        #expect(audioSource.contains("func gemma4AudioSoftTokenCount(melFrameCount: Int) -> Int"),
            "The soft-token count formula must exist in Gemma4Audio.swift.")
        #expect(audioSource.contains("(melFrameCount + 3) / 4"),
            "Soft-token count must be ceil(melFrames / 4) per HF replace_audio_token.")
        #expect(source.contains("gemma4AudioSoftTokenCount(melFrameCount: mel.dim(0))"),
            "The processor must expand <|audio|> placeholders with the post-subsampling token count.")

        // The chat template needs audio content items to emit <|audio|>.
        #expect(source.contains("message.audios.map { _ in [\"type\": \"audio\"] }"),
            "Gemma4MessageGenerator must emit audio content parts so the template renders <|audio|>.")
    }

    /// Pins the image-token-id resolution path in `Gemma4Processor.prepare`.
    @Test("Gemma4Processor uses convertTokenToId for <|image|> instead of encode().last")
    func imageTokenIdUsesConvertTokenToId() throws {
        let source = try Self.gemma4Source()

        // The lookup must use the special-token map directly.
        #expect(
            source.contains("tokenizer.convertTokenToId(\"<|image|>\")"),
            "Gemma4Processor must look up `<|image|>` via `convertTokenToId` so the result is independent of encode()'s addSpecialTokens default.")

        // The fragile `encode("<|image|>").last` form must NOT come back.
        #expect(
            !source.contains("tokenizer.encode(text: \"<|image|>\").last"),
            "Gemma4Processor must not reintroduce `encode(text: \"<|image|>\").last` — see deep-trace §7.4 for why.")

        // Sanity: the 258880 fallback is preserved (covers tokenizers
        // that don't expose `<|image|>` as an addable special token).
        #expect(source.contains("?? 258880"),
            "Gemma4Processor must keep the 258880 fallback for tokenizers without an `<|image|>` special token.")
    }
}
