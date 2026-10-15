import Foundation

/// Peak live MLX buffers, separate from allocator cache, host image decoding,
/// macOS file cache and the application's other models. Estimates do not change
/// allocator limits or admit a job on their own.
public struct ImageMemoryEstimate: Sendable, Equatable {
    public let weightBytes: Int
    public let conditioningBytes: Int
    public let denoisingBytes: Int
    public let decodingBytes: Int
    public var peakBytes: Int { weightBytes + conditioningBytes + max(denoisingBytes, decodingBytes) }
}

public enum ImageMemoryEstimator {
    public static func estimate(model: URL, family: String, width: Int, height: Int,
                                sourcePixels: Int = 0, parallelImages: Int = 1,
                                guidance: Float = 1, previewMaxPixel: Int? = nil,
                                adapters: [FluxAdapter] = []) throws -> ImageMemoryEstimate {
        guard width > 0, height > 0, parallelImages > 0, sourcePixels >= 0 else {
            throw FluxError.invalidRequest("memory estimate requires positive dimensions and batch")
        }
        let configURL = model.appendingPathComponent("transformer/config.json")
        let config = (try? Data(contentsOf: configURL)).flatMap {
            try? JSONSerialization.jsonObject(with: $0) as? [String: Any]
        } ?? [:]
        func integer(_ key: String, _ fallback: Int) -> Int { (config[key] as? NSNumber)?.intValue ?? fallback }
        func attentionWidth(heads: Int, dimension: Int) throws -> Int {
            let count = integer("num_attention_heads", heads)
            let size = integer("attention_head_dim", dimension)
            let product = count.multipliedReportingOverflow(by: size)
            guard count > 0, size > 0, !product.overflow else {
                throw FluxError.invalidRequest("invalid attention dimensions in model configuration")
            }
            return product.partialValue
        }
        let hidden: Int, latentScale: Int, textRows: Int, vaeChannels: Int
        switch family {
        case "flux1-schnell":
            hidden = try attentionWidth(heads: 24, dimension: 128)
            latentScale = 16; textRows = 256; vaeChannels = 128
        case "z-image-turbo":
            hidden = integer("dim", 3840); latentScale = 16; textRows = 512; vaeChannels = 128
        case "qwen-image", "qwen-image-edit":
            hidden = try attentionWidth(heads: 24, dimension: 128)
            latentScale = 16; textRows = 512; vaeChannels = 96
        case "qwen-image-2.1", "qwen-image-2.1-turbo":
            hidden = try attentionWidth(heads: 32, dimension: 128)
            latentScale = 16; textRows = 1024; vaeChannels = 144
        case "ideogram":
            hidden = integer("hidden_size", 2048); latentScale = 16; textRows = 512; vaeChannels = 128
        case "seedvr2":
            // VAE downsamples by 8; the transformer then packs 2x2 patches.
            hidden = integer("hidden_size", 2560); latentScale = 16; textRows = 58; vaeChannels = 128
        default:
            throw FluxError.notImplemented("peak memory estimate for \(family)")
        }
        guard hidden > 0, previewMaxPixel == nil || previewMaxPixel! > 0 else {
            throw FluxError.invalidRequest("memory estimate requires positive hidden and preview dimensions")
        }
        let pixels = Double(width) * Double(height)
        let source = Double(sourcePixels)
        let rows = ceil(pixels / Double(latentScale * latentScale))
        let referenceRows = ceil(source / Double(latentScale * latentScale))
        let bytesPerActivation = 4.0 // native pipelines retain fp32 residuals/VAE buffers
        let batch = Double(parallelImages)
        // Residual, Q/K/V, attention output, gated MLP expansion and output
        // buffers for one evaluated transformer block. CFG runs serially.
        let denoise = (rows * batch + referenceRows + Double(textRows)) * Double(hidden) * bytesPerActivation * 20
        let conditioning = (Double(textRows) + referenceRows) * Double(hidden) * bytesPerActivation
            * (guidance > 1 ? 2 : 1)
        // Untiled fp32 VAEs retain considerably more than the two logical
        // feature maps while Metal executes their convolution graphs.
        // Calibrated against the native 8-bit pipeline: 14.750 GB at 512² and
        // 25.141 GB at 1024², with 10.814 GB live weights. Account for the
        // measured size-dependent workspace; do not multiply the weights or
        // alter allocator limits to make an estimate look correct.
        let decodeMaps: Int
        switch family {
        case "z-image-turbo", "flux1-schnell": decodeMaps = 28
        // Native Flux peaks22.268/32.265GB at512/1024 with17.932GB live
        // weights. Ideogram peaks31.213/41.209/76.615GB at512/1024/2048
        // with26.880GB live weights. Legacy Qwen and Edit share the96-channel
        // decoder; measured1024 peaks45.405/46.675GB respectively.
        case "ideogram", "qwen-image", "qwen-image-edit": decodeMaps = 24
        // Native Qwen2.1 fp32 decoder uses144 channels, not the96-channel
        // encoder. 512² measured14.37GB and1536² measured41.26GB on the4bit
        // bundle; this workspace model is within8% at both sizes.
        case "qwen-image-2.1", "qwen-image-2.1-turbo": decodeMaps = 24
        default: decodeMaps = 2
        }
        var decode = pixels * batch * Double(vaeChannels * decodeMaps + 3) * bytesPerActivation
        if family == "seedvr2" {
            // SeedVR2VAETiling evaluates at most a 512px decoder tile, even
            // for a large output. Its convolution workspace dominates small
            // jobs: measured peaks are 15.491/15.606 GB at 512/1024 with
            // 4.723 GB of live weights. The calibrated tile workspace is
            // separate from the full-resolution assembly/output buffers.
            let tilePixels = Double(min(width, 512)) * Double(min(height, 512))
            decode = (tilePixels * Double(vaeChannels) * 80 + pixels * 36)
                * bytesPerActivation * batch
        }
        if let previewMaxPixel {
            decode += Double(previewMaxPixel) * Double(previewMaxPixel) * Double(vaeChannels * 2 + 3) * bytesPerActivation
        }
        let baseWeights = try WeightLoader.storedWeightBytes(in: model)
        let adapterWeights = try WeightLoader.storedWeightBytes(inFiles: adapters.filter { $0.scale != 0 }.map(\.url))
        let combined = baseWeights.addingReportingOverflow(adapterWeights)
        guard !combined.overflow else { throw FluxError.invalidRequest("adapter weight byte count overflow") }
        let weights = combined.partialValue
        let total = Double(weights) + conditioning + max(denoise, decode)
        guard total.isFinite, total < Double(Int.max) else {
            throw FluxError.invalidRequest("memory estimate exceeds representable size")
        }
        let conditioningBytes = Int(ceil(conditioning))
        let denoisingBytes = Int(ceil(denoise))
        let decodingBytes = Int(ceil(decode))
        let resident = weights.addingReportingOverflow(conditioningBytes)
        let peak = resident.partialValue.addingReportingOverflow(max(denoisingBytes, decodingBytes))
        guard !resident.overflow, !peak.overflow else {
            throw FluxError.invalidRequest("memory estimate exceeds representable size")
        }
        return ImageMemoryEstimate(weightBytes: weights, conditioningBytes: conditioningBytes,
            denoisingBytes: denoisingBytes, decodingBytes: decodingBytes)
    }
}
