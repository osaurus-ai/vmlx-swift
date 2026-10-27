import Foundation
import MLXLMCommon
import Testing

@testable import MLXVLM

/// `processor_config.json` comes in two layouts. Older conversions write the
/// image settings at the top level (`preprocessor_config.json` shape);
/// transformers v5 nests them under `image_processor` / `video_processor`.
/// unsloth's Qwen3.6 MLX repos ship only the nested file, which failed the
/// flat decode on `image_mean` and made the VLM load fall back to text-only.
@Suite("Qwen3-VL processor configuration layouts")
struct Qwen3VLProcessorConfigurationLayoutTests {
    /// Verbatim `processor_config.json` of unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit.
    static let unslothNested = """
        {"image_processor": {"do_convert_rgb": true, "do_normalize": true, "do_rescale": true, "do_resize": true,
          "image_mean": [0.5, 0.5, 0.5], "image_processor_type": "Qwen2VLImageProcessor", "image_std": [0.5, 0.5, 0.5],
          "merge_size": 2, "patch_size": 16, "resample": 3, "rescale_factor": 0.00392156862745098,
          "size": {"longest_edge": 16777216, "shortest_edge": 65536}, "temporal_patch_size": 2},
         "processor_class": "Qwen3VLProcessor",
         "video_processor": {"do_convert_rgb": true, "do_normalize": true, "do_rescale": true, "do_resize": true,
          "do_sample_frames": true, "fps": 2, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
          "max_frames": 768, "merge_size": 2, "min_frames": 4, "patch_size": 16, "resample": 3,
          "rescale_factor": 0.00392156862745098, "return_metadata": false,
          "size": {"longest_edge": 25165824, "shortest_edge": 4096}, "temporal_patch_size": 2,
          "video_processor_type": "Qwen3VLVideoProcessor"}}
        """

    static let flat = """
        {"image_mean": [0.48, 0.45, 0.4], "image_std": [0.26, 0.26, 0.27], "image_processor_type": "Qwen2VLImageProcessor",
         "merge_size": 2, "patch_size": 16, "temporal_patch_size": 2,
         "size": {"longest_edge": 16777216, "shortest_edge": 65536}, "processor_class": "Qwen3VLProcessor"}
        """

    private func decode(_ json: String) throws -> Qwen3VLProcessorConfiguration {
        try JSONDecoder().decode(Qwen3VLProcessorConfiguration.self, from: Data(json.utf8))
    }

    @Test func nestedTransformersV5LayoutDecodes() throws {
        let config = try decode(Self.unslothNested)
        #expect(config.patchSize == 16)
        #expect(config.mergeSize == 2)
        #expect(config.temporalPatchSize == 2)
        #expect(config.imageMean == [0.5, 0.5, 0.5])
        #expect(config.imageStd == [0.5, 0.5, 0.5])
        #expect(config.imageProcessorType == "Qwen2VLImageProcessor")
        #expect(config.minPixels == 65536)
        #expect(config.maxPixels == 16_777_216)
        #expect(config.videoFPS == 2)
        #expect(config.videoMinFrames == 4)
        #expect(config.videoMaxFrames == 768)
        #expect(config.videoMinPixels == 4096)
        #expect(config.videoMaxPixels == 25_165_824)
    }

    @Test func flatLayoutIsUnchanged() throws {
        let config = try decode(Self.flat)
        #expect(config.imageMean == [0.48, 0.45, 0.4])
        #expect(config.patchSize == 16)
        #expect(config.minPixels == 65536)
        #expect(config.maxPixels == 16_777_216)
        // No video settings: video falls back to the image bounds and defaults.
        #expect(config.videoMinPixels == 65536)
        #expect(config.videoFPS == 2)
    }

    @Test func topLevelSettingsWinOverANestedCopy() throws {
        let both = """
            {"image_mean": [0.1, 0.2, 0.3], "image_std": [0.5, 0.5, 0.5], "image_processor_type": "Qwen2VLImageProcessor",
             "merge_size": 2, "patch_size": 14, "temporal_patch_size": 2,
             "image_processor": {"image_mean": [0.9, 0.9, 0.9], "image_std": [0.5, 0.5, 0.5],
               "image_processor_type": "Qwen2VLImageProcessor", "merge_size": 2, "patch_size": 16, "temporal_patch_size": 2}}
            """
        let config = try decode(both)
        #expect(config.imageMean == [0.1, 0.2, 0.3])
        #expect(config.patchSize == 14)
    }

    @Test func missingImageSettingsStillFail() {
        #expect(throws: DecodingError.self) {
            try decode(#"{"processor_class": "Qwen3VLProcessor", "video_processor": {"fps": 2}}"#)
        }
    }

    @Test func registeredProcessorAcceptsTheNestedFile() async throws {
        _ = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: Data(Self.unslothNested.utf8), processorType: "Qwen3VLProcessor",
            tokenizer: TestTokenizer())
    }
}
