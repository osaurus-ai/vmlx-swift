import Foundation

/// Shared by the loader and host admission checks; never inferred from a model name.
public enum MiMoV26BundleContract {
    public static func matches(_ data: Data) -> Bool {
        guard let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            config["model_type"] as? String == "mimo_v2",
            config["attention_projection_layout"] as? String == "fused_qkv",
            let quantization = config["quantization"] as? [String: Any]
        else { return false }
        // Fused QKV is an architecture contract, not a particular quant mix.
        // The loaders validate individual formats and companion tensors. A
        // uniform affine or MX bundle must not fall into the legacy model.
        return !quantization.isEmpty
    }
}
