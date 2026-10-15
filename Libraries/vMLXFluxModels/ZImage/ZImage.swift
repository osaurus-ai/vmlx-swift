import Foundation
@preconcurrency import MLX
import MLXRandom
import vMLXFluxKit

// Z-Image-Turbo — single-encoder turbo model, ~2B params, 4-8 steps.
// Python source: `mflux.models.z_image.variants.z_image.ZImage`.
//
public final class ZImage: ImageGenerator, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "z-image-turbo",
            displayName: "Z-Image Turbo",
            kind: .imageGen,
            defaultSteps: 4,
            defaultGuidance: 0.0,
            supportsLoRA: true,
            limits: { _ in NativeImageRequestContracts.zImage },
            loader: { path, quant in
                _ = ZImage._register
                return try await ZImage(modelPath: path, quantize: quant)
            }
        ))
    }()

    public let modelPath: URL
    public let quantize: Int?
    public let loadedWeights: LoadedWeights
    private let pipeline: ZImageNativePipeline

    public init(modelPath: URL, quantize: Int?) async throws {
        self.modelPath = modelPath
        self.quantize = quantize
        _ = Self._register
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw FluxError.weightsNotFound(modelPath)
        }
        // Eagerly load weights via the WeightLoader so we surface any
        // JANG config / missing-shard errors at `.load` time rather than
        // on the first generate call.
        self.loadedWeights = try WeightLoader.load(from: modelPath)

        self.pipeline = try await ZImageNativePipeline(
            modelPath: modelPath,
            loadedWeights: loadedWeights)
    }

    public func generate(_ request: ImageGenRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        do { try NativeImageRequestContracts.zImage.validate(request) }
        catch { return AsyncThrowingStream { $0.finish(throwing: error) } }
        return AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    try await self.performGenerate(request, continuation: continuation)
                    continuation.finish()
                } catch is CancellationError {
                    continuation.yield(.cancelled)
                    continuation.finish()
                } catch {
                    let message = String(describing: error)
                    let hfAuth = message.contains("401") || message.contains("403")
                    continuation.yield(.failed(message: message, hfAuth: hfAuth))
                    continuation.finish()
                }
            }
        }
    }

    private func performGenerate(
        _ request: ImageGenRequest,
        continuation: AsyncThrowingStream<ImageGenEvent, Error>.Continuation
    ) async throws {
        guard request.steps > 0 else {
            throw FluxError.invalidRequest("Z-Image steps must be greater than zero")
        }
        let seed = request.seed ?? UInt64.random(in: 0...UInt64(UInt32.max))
                    let preview = ImagePreviewEmitter(interval: request.previewInterval, maxPixel: request.previewMaxPixel) { data, step in
                        continuation.yield(.preview(pngData: data, step: step))
                    }
                    defer { preview?.close() }
        let image = try await pipeline.generate(
            prompt: request.prompt,
            negativePrompt: request.negativePrompt,
            guidance: request.guidance,
            width: request.width,
            height: request.height,
            steps: request.steps,
            seed: seed, preview: preview
        ) { step, total, eta in
            continuation.yield(.step(step: step, total: total, etaSeconds: eta))
        }

        preview?.close()
                    try FluxInterruptScope.check()
                    let outURL = try ImageIO.writePNG(
            image,
            outputDir: request.outputDir,
            prefix: "z-image"
        )
        continuation.yield(.completed(url: outURL, seed: seed,
            effectiveParameters: try NativeImageRequestContracts.parameters(url: outURL,
                steps: request.steps, guidance: request.negativePrompt != nil && request.guidance > 0 ? request.guidance : 0,
                seed: seed, scheduler: "z-image-linear")))
    }

    // MARK: - Placeholder math (to be replaced by real transformer + VAE)

    /// Placeholder velocity predictor. Returns a scaled noise field that
    /// drives the scheduler to a stable-but-meaningless converged state
    /// so the end-to-end plumbing is testable today.
    private func velocityPlaceholder(
        latent: MLXArray,
        stepIndex: Int,
        scheduler: FlowMatchEulerScheduler
    ) -> MLXArray {
        let noise = MLXRandom.normal(latent.shape)
        let sigmaDelta = scheduler.sigmas[stepIndex + 1] - scheduler.sigmas[stepIndex]
        return noise * MLXArray(Float(sigmaDelta * 0.1))
    }

    /// Placeholder VAE decoder. Converts a (B, 4, H/8, W/8) latent into a
    /// (B, 3, H, W) image by averaging channel pairs, tanh-squashing to
    /// [0, 1], and nearest-neighbor upsampling. The output is a blocky
    /// gradient — not the prompt, but a valid PNG that exercises every
    /// pipeline stage. Replaced by the real VAE decoder when weights land.
    private func vaeDecodePlaceholder(
        latent: MLXArray,
        targetWidth: Int,
        targetHeight: Int
    ) -> MLXArray {
        let b = latent.dim(0)
        let h = latent.dim(2)
        let w = latent.dim(3)
        let r = mean(latent[0..<b, 0..<2, 0..<h, 0..<w], axis: 1, keepDims: true)
        let g = mean(latent[0..<b, 1..<3, 0..<h, 0..<w], axis: 1, keepDims: true)
        let bl = mean(latent[0..<b, 2..<4, 0..<h, 0..<w], axis: 1, keepDims: true)
        let rgb = concatenated([r, g, bl], axis: 1)

        let squashed = (MLX.tanh(rgb) + MLXArray(Float(1))) * MLXArray(Float(0.5))
        let upsampled = upsampleNearest(squashed, factor: 8)
        return cropToSize(upsampled, height: targetHeight, width: targetWidth)
    }

    /// Nearest-neighbor upsample by an integer factor. (B, C, H, W) →
    /// (B, C, H*f, W*f). Uses `MLX.repeated(_:count:axis:)` along H and W.
    private func upsampleNearest(_ x: MLXArray, factor: Int) -> MLXArray {
        // Repeat along H (axis 2) then W (axis 3). Each repeat expands
        // that dimension by `factor` via nearest-neighbor duplication.
        let expandedH = repeated(x, count: factor, axis: 2)
        let expandedW = repeated(expandedH, count: factor, axis: 3)
        return expandedW
    }

    /// Crop an image tensor to exact (height, width). Takes top-left crop
    /// if the input is larger; smaller-than-target cases fall through
    /// since upsampleNearest already pads up to `factor * latentDim`.
    private func cropToSize(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let currentH = x.dim(2)
        let currentW = x.dim(3)
        if currentH == height && currentW == width { return x }
        let h = min(currentH, height)
        let w = min(currentW, width)
        return x[0 ..< x.dim(0), 0 ..< x.dim(1), 0 ..< h, 0 ..< w]
    }
}
