import Foundation

/// Text-only Qwen3.8-Flash-Next (qwen4_exp) bundles: no `vision_config`, no `visual.*` tensors, and no
/// image/video preprocessor files (the Qwen3.8-Flash-Next JANGHT bundle, 2026-10-16, ships none of them).
///
/// qwen4_exp is registered only in the VLM factory, and that factory always builds an input processor from
/// `preprocessor_config.json` / `processor_config.json`. A text-only bundle has neither, so the load failed after the
/// weights had already been read. For such a bundle the processor configuration is synthesized here: the Qwen3-VL
/// processor class the family uses, with the Qwen3.8 vendor image geometry (patch 16, merge 2, temporal 2, mean/std
/// 0.5). Those image values are never applied to a model: `Qwen4ExpConfiguration` without `vision_config` constructs
/// no vision tower, its modalities are `[.text]`, and `Qwen4Exp.prepare` refuses image or video input.
enum Qwen4ExpTextOnlyBundle {
    /// The synthesized processor configuration, or nil when the bundle is not a text-only qwen4_exp bundle (any
    /// processor file present, another model type, or a `vision_config`).
    static func processorConfiguration(modelDirectory: URL, configData: Data?) -> Data? {
        for name in ["preprocessor_config.json", "processor_config.json", "audio_preprocessor/preprocessor_config.json"] {
            if FileManager.default.fileExists(atPath: modelDirectory.appending(path: name).path) { return nil }
        }
        guard let configData,
              let root = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
              root["model_type"] as? String == "qwen4_exp",
              root["vision_config"] == nil || root["vision_config"] is NSNull
        else { return nil }
        let synthesized: [String: Any] = [
            "processor_class": "Qwen3VLProcessor",
            "image_processor_type": "Qwen2VLImageProcessorFast",
            "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
            "size": ["longest_edge": 16_777_216, "shortest_edge": 65_536],
        ]
        return try? JSONSerialization.data(withJSONObject: synthesized, options: [.sortedKeys])
    }
}
