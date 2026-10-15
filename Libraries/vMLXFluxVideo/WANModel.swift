import Foundation
@preconcurrency import MLX
import MLXRandom
import vMLXFluxKit

/// Native Wan 2.1 T2V/I2V using original tensor names in safetensors
/// components. See docs/FLUX-WORKFLOW-SPEC-IMPLEMENTATION.md for layout.
public final class WANModel: VideoGenerator, @unchecked Sendable {
    public static let _registerWan21: Void = {
        ModelRegistry.register(ModelEntry(name: "wan-2.1", displayName: "Wan 2.1",
            kind: .videoGen, defaultSteps: 50, defaultGuidance: 5,
            loader: { path, quant in try await WANModel(modelPath: path, quantize: quant, version: .wan21) }))
    }()
    public static let _registerWan22: Void = {
        ModelRegistry.register(ModelEntry(name: "wan-2.2", displayName: "Wan 2.2",
            kind: .videoGen, defaultSteps: 50, defaultGuidance: 5,
            loader: { _, _ in throw FluxError.notImplemented("Wan 2.2 dual-transformer routing") }))
    }()
    public enum Version: Sendable { case wan21, wan22 }
    public let modelPath: URL
    public let quantize: Int?
    public let version: Version
    private let transformer: WanNativeTransformer
    private let encoder: WanNativeTextEncoder
    private let vae: WanNativeVAE
    private let clip: WanNativeCLIP?
    private let shift: Float
    private let fixedSigmas: [Float]?

    public init(modelPath: URL, quantize: Int?, version: Version) async throws {
        guard version == .wan21 else { throw FluxError.notImplemented("Wan 2.2 dual-transformer routing") }
        self.modelPath = modelPath; self.quantize = quantize; self.version = version
        let configURL = modelPath.appendingPathComponent("config.json")
        guard let config = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any],
              let modelType = config["model_type"] as? String, ["t2v", "i2v"].contains(modelType),
              config["dim"] != nil, config["num_heads"] != nil, config["num_layers"] != nil else {
            throw FluxError.localModelIncomplete(modelPath, reasons: ["original Wan config.json with model_type, dim, num_heads and num_layers is required"])
        }
        let loaded = try WeightLoader.load(from: modelPath)
        guard let text = loaded.componentWeights["text_encoder"], let vaeWeights = loaded.componentWeights["vae"] else {
            throw FluxError.localModelIncomplete(modelPath, reasons: ["text_encoder/ and vae/ safetensor components are required"])
        }
        let transformerWeights = loaded.componentWeights["transformer"] ?? loaded.weights
        transformer = try WanNativeTransformer(weights: transformerWeights, config: config)
        encoder = try await WanNativeTextEncoder(weights: text, modelURL: modelPath)
        vae = try WanNativeVAE(weights: vaeWeights)
        if modelType == "i2v" {
            let directory = modelPath.appendingPathComponent("image_encoder")
            var arrays: [String: MLXArray] = [:]
            for file in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
                .filter({ $0.pathExtension == "safetensors" }).sorted(by: { $0.path < $1.path }) {
                for (key, value) in try MLX.loadArrays(url: file) {
                    guard arrays[key] == nil else { throw FluxError.invalidRequest("duplicate Wan image encoder tensor \(key)") }
                    arrays[key] = value
                }
            }
            clip = try WanNativeCLIP(arrays: arrays)
        } else { clip = nil }
        shift = (config["sample_shift"] as? NSNumber)?.floatValue ?? (modelType == "i2v" ? 5 : 3)
        fixedSigmas = (config["sample_sigmas"] as? [NSNumber])?.map(\.floatValue)
        guard shift.isFinite, shift > 0 else { throw FluxError.invalidRequest("Wan sample_shift must be positive") }
        if let fixedSigmas {
            guard !fixedSigmas.isEmpty, fixedSigmas.allSatisfy({ $0.isFinite && $0 > 0 && $0 <= 1 }),
                zip(fixedSigmas, fixedSigmas.dropFirst()).allSatisfy({ $0 > $1 }) else {
                throw FluxError.invalidRequest("Wan sample_sigmas must be strictly decreasing in (0,1]")
            }
        }
    }

    public func generate(_ request: VideoGenRequest) -> AsyncThrowingStream<VideoGenEvent, Error> {
        let interrupt = request.interrupt ?? FluxInterrupt()
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try self.run(request, interrupt: interrupt, continuation: continuation)
                    continuation.finish()
                } catch is CancellationError {
                    continuation.yield(.cancelled); continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { reason in
                if case .cancelled = reason { interrupt.cancel(); task.cancel() }
            }
        }
    }

    private func run(_ request: VideoGenRequest, interrupt: FluxInterrupt,
                     continuation: AsyncThrowingStream<VideoGenEvent, Error>.Continuation) throws {
        try interrupt.check()
        guard request.width >= 16, request.height >= 16, request.width % 16 == 0, request.height % 16 == 0,
              request.numFrames > 0, request.numFrames % 4 == 1, (1...240).contains(request.fps),
              request.steps > 0, request.guidance.isFinite, request.guidance >= 0 else {
            throw FluxError.invalidRequest("Wan requires positive multiples of 16, 4n+1 frames, positive steps, finite guidance and fps in 1...240")
        }
        let pixels = request.width.multipliedReportingOverflow(by: request.height)
        guard !pixels.overflow, pixels.partialValue <= Int(Int32.max) / request.numFrames else {
            throw FluxError.invalidRequest("Wan video dimensions exceed tensor indexing limits")
        }
        guard transformer.imageConditioned == (request.sourceImage != nil) else {
            throw FluxError.invalidField(field: "sourceImage", value: request.sourceImage == nil ? "missing" : "provided",
                allowed: transformer.imageConditioned ? "required by this I2V bundle" : "omitted for this T2V bundle")
        }
        let positive = try encoder.encode(request.prompt, interrupt: interrupt)
        let negative = request.guidance == 1 ? nil : try encoder.encode(request.negativePrompt ?? "", interrupt: interrupt)
        let frames = (request.numFrames - 1) / 4 + 1, h = request.height / 8, w = request.width / 8
        let seed = request.seed ?? UInt64.random(in: 0...UInt64.max)
        var latent = MLXRandom.normal([frames, h, w, 16], key: MLXRandom.key(seed)).asType(.bfloat16)
        var imageFeatures: MLXArray?, imageLatent: MLXArray?
        if let source = request.sourceImage, let clip {
            imageFeatures = try clip.encode(source, interrupt: interrupt)
            let size = try ImageIO.dimensions(of: source)
            let scale = max(Double(request.width) / Double(size.width), Double(request.height) / Double(size.height))
            let resizedW = max(request.width, Int((Double(size.width) * scale).rounded()))
            let resizedH = max(request.height, Int((Double(size.height) * scale).rounded()))
            let values = try ImageIO.readRGBValues(source, width: resizedW, height: resizedH, normalization: .minusOneToOne)
            let top = (resizedH - request.height) / 2, left = (resizedW - request.width) / 2
            let resized = MLXArray(values, [3, resizedH, resizedW])
            let image = resized[0..., top..<(top + request.height), left..<(left + request.width)].transposed(1, 2, 0)
            let video = concatenated([image.expandedDimensions(axis: 0),
                MLXArray.zeros([request.numFrames - 1, request.height, request.width, 3])], axis: 0)
            let encoded = try vae.encode(video, interrupt: interrupt)
            let mask = concatenated([MLXArray.ones([1, h, w, 4]), MLXArray.zeros([frames - 1, h, w, 4])], axis: 0)
            imageLatent = concatenated([mask, encoded], axis: -1).asType(latent.dtype)
        }
        let grid = fixedSigmas ?? (0..<request.steps).map { index -> Float in
            let sigma = 1 - Float(index) / Float(request.steps)
            return shift * sigma / (1 + (shift - 1) * sigma)
        }
        let sigmas = grid + [0]
        let start = Date()
        for step in grid.indices {
            try interrupt.check()
            let prediction = try transformer.predict(latent, time: grid[step] * 1000,
                context: positive, imageFeatures: imageFeatures, imageLatent: imageLatent, interrupt: interrupt)
            let velocity: MLXArray
            if let negative {
                let unconditioned = try transformer.predict(latent, time: grid[step] * 1000,
                    context: negative, imageFeatures: imageFeatures, imageLatent: imageLatent, interrupt: interrupt)
                velocity = unconditioned + request.guidance * (prediction - unconditioned)
            } else { velocity = prediction }
            latent = (latent.asType(.float32) + velocity.asType(.float32) * (sigmas[step + 1] - sigmas[step])).asType(latent.dtype)
            eval(latent)
            let elapsed = Date().timeIntervalSince(start)
            continuation.yield(.step(step: step + 1, total: grid.count,
                etaSeconds: elapsed / Double(step + 1) * Double(grid.count - step - 1)))
        }
        var chunks: [MLXArray] = []
        try vae.decode(latent.asType(.float32), interrupt: interrupt) { chunks.append(($0 + 1) / 2) }
        let video = concatenated(chunks, axis: 0)
        guard video.dim(0) == request.numFrames else { throw FluxError.invalidRequest("Wan VAE produced an unexpected frame count") }
        try interrupt.check()
        let url = request.outputDir.appendingPathComponent("wan-\(UUID().uuidString).mp4")
        try WanVideoIO.writeMP4(video.transposed(3, 0, 1, 2), outputURL: url, fps: request.fps, interrupt: interrupt)
        if interrupt.isCancelled { try? FileManager.default.removeItem(at: url); throw CancellationError() }
        continuation.yield(.completed(url: url, seed: seed, fps: request.fps, frameCount: request.numFrames,
            effective: EffectiveImageParameters(width: request.width, height: request.height,
                steps: grid.count, guidance: request.guidance, seed: seed, sampler: "euler",
                scheduler: fixedSigmas == nil ? "wan-flow" : "bundle-sigmas", shift: shift, sampleSigmas: fixedSigmas)))
    }
}

public enum VMLXFluxVideo {
    public static func registerAll() {
        _ = WANModel._registerWan21
        _ = WANModel._registerWan22
    }
}
