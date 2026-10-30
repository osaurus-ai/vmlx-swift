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
///
/// A bundle that declares a vision tower but ships no processor file (a mixed 4/8-bit mlx-vlm-converted Qwen3.8 Flash-Next)
/// is made text-only first, by ``textOnlyConfiguration(_:modelDirectory:)``: its image geometry is undeclared, and
/// filling it in from the vendor would apply values the bundle never stated.
enum Qwen4ExpTextOnlyBundle {
    /// True when the bundle ships none of the processor files the VLM factory reads.
    static func shipsNoProcessorFile(_ modelDirectory: URL) -> Bool {
        ["preprocessor_config.json", "processor_config.json", "audio_preprocessor/preprocessor_config.json"]
            .allSatisfy { !FileManager.default.fileExists(atPath: modelDirectory.appending(path: $0).path) }
    }

    /// The synthesized processor configuration, or nil when the bundle ships a processor file or is another model
    /// type. A bundle with a `vision_config` gets it too, because it is constructed without that config.
    static func processorConfiguration(modelDirectory: URL, configData: Data?) -> Data? {
        guard shipsNoProcessorFile(modelDirectory),
              let configData,
              let root = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
              root["model_type"] as? String == "qwen4_exp"
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

    /// The configuration to construct from: without `vision_config` when a qwen4_exp bundle declares a vision tower
    /// but ships no processor file, so no tower is built, sanitize drops the `visual.*` weights and `prepare` refuses
    /// media. Any other configuration is returned unchanged, byte for byte.
    static func textOnlyConfiguration(_ configData: Data, modelDirectory: URL) -> Data {
        guard shipsNoProcessorFile(modelDirectory),
              var root = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
              root["model_type"] as? String == "qwen4_exp",
              let vision = root["vision_config"], !(vision is NSNull)
        else { return configData }
        root.removeValue(forKey: "vision_config")
        guard let scrubbed = try? JSONSerialization.data(withJSONObject: root) else { return configData }
        FileHandle.standardError.write(Data(
            ("[Qwen4Exp] \(modelDirectory.lastPathComponent) declares a vision tower but ships no "
                + "preprocessor_config.json: loaded text-only, image and video input refused\n").utf8))
        return scrubbed
    }
}
