import Foundation
import MLXLMCommon
import Testing

@testable import MLXVLM

/// Exercises the actual directory loader without constructing/loading model weights.
@Suite("VLM processor sidecar selection")
struct VLMProcessorConfigFileSelectionTests {
    // numind/NuExtract3@c99dc8f5641b866aa0192b6ea78f84bf9f3535f1/processor_config.json.
    // The bundled fixture is semantically identical JSON, compacted for readability.
    static let officialNuExtract3 = #"{"image_processor":{"do_convert_rgb":true,"do_normalize":true,"do_rescale":true,"do_resize":true,"image_mean":[0.5,0.5,0.5],"image_processor_type":"Qwen2VLImageProcessor","image_std":[0.5,0.5,0.5],"merge_size":2,"patch_size":16,"resample":3,"rescale_factor":0.00392156862745098,"size":{"longest_edge":16777216,"shortest_edge":65536},"temporal_patch_size":2},"processor_class":"Qwen3VLProcessor","video_processor":{"do_convert_rgb":true,"do_normalize":true,"do_rescale":true,"do_resize":true,"do_sample_frames":true,"fps":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"max_frames":768,"merge_size":2,"min_frames":4,"patch_size":16,"resample":3,"rescale_factor":0.00392156862745098,"return_metadata":false,"size":{"longest_edge":25165824,"shortest_edge":4096},"temporal_patch_size":2,"video_processor_type":"Qwen3VLVideoProcessor"}}"#

    private func directory() throws -> URL {
        let result = FileManager.default.temporaryDirectory
            .appendingPathComponent("processor-sidecar-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: result, withIntermediateDirectories: true)
        return result
    }

    private func write(_ text: String, named name: String, in directory: URL) throws {
        let file = directory.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: file)
    }

    @Test func processorOnlyNuExtract3LoadsExactNestedImageAndVideoSettings() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        try write(Self.officialNuExtract3, named: "processor_config.json", in: root)
        let (data, base) = try await loadProcessorConfig(from: root)
        #expect(data == Data(Self.officialNuExtract3.utf8))
        #expect(base.processorClass == "Qwen3VLProcessor")
        let typed = try JSONDecoder().decode(Qwen3VLProcessorConfiguration.self, from: data)
        #expect(typed.imageMean == [0.5, 0.5, 0.5])
        #expect(typed.imageStd == [0.5, 0.5, 0.5])
        #expect(typed.patchSize == 16)
        #expect(typed.mergeSize == 2)
        #expect(typed.temporalPatchSize == 2)
        #expect(typed.minPixels == 65_536)
        #expect(typed.maxPixels == 16_777_216)
        #expect(typed.videoFPS == 2)
        #expect(typed.videoMinFrames == 4)
        #expect(typed.videoMaxFrames == 768)
        #expect(typed.videoMinPixels == 4_096)
        #expect(typed.videoMaxPixels == 25_165_824)
        _ = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: data, processorType: base.processorClass, tokenizer: TestTokenizer())
    }

    @Test func preferredPreprocessorWinsWithoutCombiningTheOtherFile() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        try write(Qwen3VLProcessorConfigurationLayoutTests.flat, named: "preprocessor_config.json", in: root)
        try write(Self.officialNuExtract3, named: "processor_config.json", in: root)
        let (data, _) = try await loadProcessorConfig(from: root)
        let typed = try JSONDecoder().decode(Qwen3VLProcessorConfiguration.self, from: data)
        #expect(data == Data(Qwen3VLProcessorConfigurationLayoutTests.flat.utf8))
        #expect(typed.imageMean == [0.48, 0.45, 0.4])
        #expect(typed.videoMaxPixels == typed.maxPixels)
    }

    @Test func malformedPreferredFileFailsRatherThanFallingBack() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        try write("{ broken", named: "preprocessor_config.json", in: root)
        try write(Self.officialNuExtract3, named: "processor_config.json", in: root)
        do {
            _ = try await loadProcessorConfig(from: root)
            Issue.record("Malformed preferred processor configuration must fail")
        } catch let error as ProcessorConfigError {
            #expect(error.filename == "preprocessor_config.json")
            #expect(error.underlying is DecodingError)
        }
    }

    @Test func unreadablePreferredFileRetainsItsFilenameAndDoesNotFallback() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        // A directory at the preferred file path deterministically fails Data reads,
        // without relying on process privileges or changing user filesystem access.
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("preprocessor_config.json"), withIntermediateDirectories: false)
        try write(Self.officialNuExtract3, named: "processor_config.json", in: root)
        do {
            _ = try await loadProcessorConfig(from: root)
            Issue.record("An unreadable preferred file must fail")
        } catch let error as ProcessorConfigError {
            #expect(error.filename == "preprocessor_config.json")
            #expect(!(error.underlying is DecodingError))
        }
    }

    @Test func malformedFallbackRetainsProcessorFilename() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        try write("{ broken", named: "processor_config.json", in: root)
        do {
            _ = try await loadProcessorConfig(from: root)
            Issue.record("Malformed fallback configuration must fail")
        } catch let error as ProcessorConfigError {
            #expect(error.filename == "processor_config.json")
            #expect(error.underlying is DecodingError)
        }
    }

    @Test func nestedVideoSidecarOverridesVideoButRetainsNestedImage() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        try write(Self.officialNuExtract3, named: "processor_config.json", in: root)
        try write(#"{"fps":3,"min_frames":2,"max_frames":12,"size":{"shortest_edge":8192,"longest_edge":1048576}}"#,
                  named: "video_preprocessor_config.json", in: root)
        let (data, _) = try await loadProcessorConfig(from: root)
        let typed = try JSONDecoder().decode(Qwen3VLProcessorConfiguration.self, from: data)
        #expect(typed.minPixels == 65_536)
        #expect(typed.maxPixels == 16_777_216)
        #expect(typed.videoFPS == 3)
        #expect(typed.videoMinFrames == 2)
        #expect(typed.videoMaxFrames == 12)
        #expect(typed.videoMinPixels == 8_192)
        #expect(typed.videoMaxPixels == 1_048_576)
    }

    @Test func malformedVideoSidecarReportsItsOwnFilename() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        try write(Self.officialNuExtract3, named: "processor_config.json", in: root)
        try write("{ broken", named: "video_preprocessor_config.json", in: root)
        do {
            _ = try await loadProcessorConfig(from: root)
            Issue.record("Malformed video sidecar must fail")
        } catch let error as ProcessorConfigError {
            #expect(error.filename == "video_preprocessor_config.json")
        }
    }

    @Test func audioFallbackRemainsLastAfterBothVisionNames() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        try write(#"{"processor_class":"AudioOnlyFixture"}"#,
                  named: "audio_preprocessor/preprocessor_config.json", in: root)
        let (_, audio) = try await loadProcessorConfig(from: root)
        #expect(audio.processorClass == "AudioOnlyFixture")
        try write(Self.officialNuExtract3, named: "processor_config.json", in: root)
        let (_, vision) = try await loadProcessorConfig(from: root)
        #expect(vision.processorClass == "Qwen3VLProcessor")
    }

    @Test func missingImageSettingsRemainARealTypedProcessorFailure() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        try write(#"{"processor_class":"Qwen3VLProcessor","video_processor":{"fps":2}}"#,
                  named: "processor_config.json", in: root)
        let (data, base) = try await loadProcessorConfig(from: root)
        #expect(base.processorClass == "Qwen3VLProcessor")
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(Qwen3VLProcessorConfiguration.self, from: data)
        }
        await #expect(throws: (any Error).self) {
            try await VLMProcessorTypeRegistry.shared.createModel(
                configuration: data, processorType: base.processorClass, tokenizer: TestTokenizer())
        }
    }
}
