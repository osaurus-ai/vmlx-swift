import Foundation

/// Shared by the loader and host admission checks; never inferred from a model name.
public enum MiMoV26BundleContract {
    public static func matches(_ data: Data) -> Bool {
        guard let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            config["model_type"] as? String == "mimo_v2",
            config["attention_projection_layout"] as? String == "fused_qkv",
            let quantization = config["quantization"] as? [String: Any]
        else { return false }
        let modes = Set(([quantization] + quantization.values.compactMap { $0 as? [String: Any] })
            .compactMap { $0["mode"] as? String })
        return modes.contains("affine") && modes.contains("mxfp4")
    }
}
