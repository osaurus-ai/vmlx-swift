import Foundation
@preconcurrency import MLXLMCommon

// MARK: - JANG support bridge
//
// Reuse vmlx-swift-lm's `JangLoader` / `JangConfig` instead of re-porting
// them. When a Flux model directory contains a `jang_config.json`, we
// load the per-layer quantization metadata and pass it through to the
// weight loader so 2-bit affine + MXTQ packed weights decode correctly.
//
// Why bridge vs re-export: vmlx-swift-lm already has 533 lines of
// JANG v2 parsing (JangConfig / JangQuantization / JangSourceModel /
// JangArchitecture / JangRuntime) plus per-model weight remap logic for
// Mistral 4 / Gemma 4 / Nemotron H / Qwen 3.5. All of that works on
// transformer-style checkpoints and is directly usable here.

/// Thin wrapper so callers in VMLXFluxModels don't need to import
/// MLXLMCommon themselves.
public enum JangBridge {

    /// True if the directory contains a JANG v2 config file.
    public static func isJangModel(at path: URL) -> Bool {
        MLXLMCommon.JangLoader.isJangModel(at: path)
    }

    /// Parse the JANG config from a model directory. Returns nil if
    /// not a JANG model. Rethrows parse errors.
    public static func loadConfig(at path: URL) throws -> MLXLMCommon.JangConfig? {
        guard MLXLMCommon.JangLoader.isJangModel(at: path) else { return nil }
        return try MLXLMCommon.JangLoader.loadConfig(at: path)
    }

    /// Convenience: `(isJang, config)` tuple for the single branch the
    /// model loaders always want — "detect + parse in one shot".
    public static func detect(at path: URL) throws -> (isJang: Bool, config: MLXLMCommon.JangConfig?) {
        guard MLXLMCommon.JangLoader.isJangModel(at: path) else {
            return (false, nil)
        }
        return (true, try MLXLMCommon.JangLoader.loadConfig(at: path))
    }
}
