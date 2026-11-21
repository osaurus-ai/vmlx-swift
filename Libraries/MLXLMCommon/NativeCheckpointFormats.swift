// Copyright © 2026 Jinho Jang. All rights reserved.

import Foundation

/// Architectures whose UPSTREAM checkpoints (transformers / ModelOpt safetensors, tagged
/// `__metadata__.format = "pt"`) this runtime loads directly: the model's `sanitize` converts the
/// published tensors at load (e.g. Inkling-2-Mini's ModelOpt NVFP4 / MXFP8 routed experts run
/// natively through `gatherQuantizedMM`). Hosts that gate loading on "is this an MLX bundle"
/// consult this so such a checkpoint is not refused before the loader ever sees it.
public enum NativeCheckpointFormats {
    public static let modelTypes: Set<String> = ["inkling2_loop"]

    /// `model_type` (top level, or `text_config.model_type`) of the bundle's config.json is one
    /// this runtime loads in its upstream format.
    public static func loadsUpstreamCheckpoint(configData: Data) -> Bool {
        guard let object = try? JSONSerialization.jsonObject(with: configData) as? [String: Any]
        else { return false }
        let text = object["text_config"] as? [String: Any]
        let types = [object["model_type"] as? String, text?["model_type"] as? String].compactMap { $0 }
        return types.contains { modelTypes.contains($0.lowercased()) }
    }

    public static func loadsUpstreamCheckpoint(at directory: URL) -> Bool {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("config.json"))
        else { return false }
        return loadsUpstreamCheckpoint(configData: data)
    }
}
