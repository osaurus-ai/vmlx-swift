import Foundation

/// The local, self-contained 3B export. Never downloads embeddings or guesses a variant.
public struct SeedVR2BundleMetadata: Sendable {
    public let directory: URL
    public let bits: Int?
    public let groupSize: Int

    public init(directory: URL) throws {
        self.directory = directory
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            raw["model_type"] as? String == "seedvr2",
            raw["variant"] as? String == "seedvr2-3b",
            raw["transformer_overrides"] == nil
                || (raw["transformer_overrides"] as? [String: Any])?.isEmpty == true
        else {
            throw FluxError.invalidRequest(
                "SeedVR2 requires a 3B export without architecture overrides")
        }
        if raw["quantization"] != nil && !(raw["quantization"] is NSNull) {
            guard let q = raw["quantization"] as? [String: Any] else {
                throw FluxError.invalidRequest("SeedVR2 quantization metadata must be an object")
            }
            guard q["bits"] as? Int == 8, q["group_size"] as? Int == 64 else {
                throw FluxError.invalidRequest(
                    "SeedVR2 supports float weights or affine int8/group64 exports")
            }
            bits = 8
            groupSize = 64
        } else {
            bits = nil
            groupSize = 64
        }
        for name in ["transformer.safetensors", "vae.safetensors", "pos_emb.safetensors"] {
            guard
                FileManager.default.fileExists(atPath: directory.appendingPathComponent(name).path)
            else {
                throw FluxError.weightsNotFound(directory.appendingPathComponent(name))
            }
        }
    }

}

extension SeedVR2BundleMetadata {
    /// Current single-image implementation envelope, not an architectural context limit.
    public static let maximumOutputPixels = 4096 * 4096

    public static func validateUpscale(
        sourceWidth: Int, sourceHeight: Int, scale: Int, format: ImageFormat
    ) throws {
        guard scale == 2 || scale == 4 else {
            throw FluxError.invalidField(field: "scale", value: String(scale), allowed: "2 or 4")
        }
        guard format == .png else {
            throw FluxError.invalidField(
                field: "outputFormat", value: format.rawValue, allowed: "png")
        }
        let width = sourceWidth.multipliedReportingOverflow(by: scale)
        let height = sourceHeight.multipliedReportingOverflow(by: scale)
        let pixels = width.partialValue.multipliedReportingOverflow(by: height.partialValue)
        guard sourceWidth > 0, sourceHeight > 0, !width.overflow, !height.overflow,
            !pixels.overflow,
            pixels.partialValue <= maximumOutputPixels
        else {
            throw FluxError.invalidField(
                field: "dimensions", value: "\(sourceWidth)x\(sourceHeight) at \(scale)x",
                allowed: "positive dimensions and at most \(maximumOutputPixels) output pixels")
        }
        // The transformer/VAE process a 16-aligned canvas, not just the crop.
        // Very thin images can otherwise exceed the allocation budget by 8x.
        let paddedWidth = (width.partialValue + 15) / 16 * 16
        let paddedHeight = (height.partialValue + 15) / 16 * 16
        guard paddedWidth <= maximumOutputPixels / paddedHeight else {
            throw FluxError.invalidField(
                field: "dimensions",
                value: "\(width.partialValue)x\(height.partialValue)",
                allowed: "at most \(maximumOutputPixels) pixels including 16-pixel canvas padding")
        }
    }
}
