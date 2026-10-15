import Foundation

// MARK: - ImageGenRequest

public struct ImageGenRequest: Sendable {
    public var prompt: String
    public var negativePrompt: String?
    public var width: Int
    public var height: Int
    public var steps: Int
    public var guidance: Float
    public var seed: UInt64?
    public var numImages: Int
    /// Explicit per-image seeds. Otherwise preserve seed + index (wrapping UInt64).
    public var seeds: [UInt64]?
    /// Requested parallel sampling width; implementations may lower this to fit.
    public var batchSize: Int
    public var interrupt: FluxInterrupt?
    public var outputDir: URL
    public var outputFormat: ImageFormat
    public var sampler: String?
    public var scheduler: String?
    public var shift: Float?
    public var previewInterval: Int?
    public var previewMaxPixel: Int
    public var adapters: [FluxAdapter]

    public init(
        prompt: String,
        negativePrompt: String? = nil,
        width: Int = 1024,
        height: Int = 1024,
        steps: Int = 20,
        guidance: Float = 3.5,
        seed: UInt64? = nil,
        numImages: Int = 1,
        seeds: [UInt64]? = nil,
        batchSize: Int = 1,
        interrupt: FluxInterrupt? = nil,
        outputDir: URL,
        outputFormat: ImageFormat = .png,
        sampler: String? = nil, scheduler: String? = nil, shift: Float? = nil,
        previewInterval: Int? = nil, previewMaxPixel: Int = 256, adapters: [FluxAdapter] = []
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.numImages = numImages
        self.seeds = seeds
        self.batchSize = batchSize
        self.interrupt = interrupt
        self.outputDir = outputDir
        self.outputFormat = outputFormat
        self.sampler = sampler
        self.scheduler = scheduler
        self.shift = shift
        self.previewInterval = previewInterval
        self.previewMaxPixel = previewMaxPixel
        self.adapters = adapters
    }
}

// MARK: - ImageEditRequest

public struct ImageEditRequest: Sendable {
    public var prompt: String
    public var sourceImage: URL
    public var sourceImages: [URL]
    public var mask: URL?          // optional PNG mask — white=edit, black=keep
    public var strength: Float?    // nil uses the backend's native editing behavior
    public var width: Int?         // nil uses the backend's reference-aspect/resolution policy
    public var height: Int?
    public var steps: Int
    public var guidance: Float
    public var seed: UInt64?
    public var outputDir: URL
    public var outputFormat: ImageFormat
    public var sampler: String?
    public var scheduler: String?
    public var shift: Float?
    public var previewInterval: Int?
    public var previewMaxPixel: Int
    public var adapters: [FluxAdapter]

    public init(
        prompt: String,
        sourceImage: URL,
        mask: URL? = nil,
        strength: Float? = nil,
        width: Int? = nil,
        height: Int? = nil,
        steps: Int = 20,
        guidance: Float = 3.5,
        seed: UInt64? = nil,
        outputDir: URL,
        outputFormat: ImageFormat = .png,
        sampler: String? = nil, scheduler: String? = nil, shift: Float? = nil,
        previewInterval: Int? = nil, previewMaxPixel: Int = 256, adapters: [FluxAdapter] = []
    ) {
        self.prompt = prompt
        self.sourceImage = sourceImage
        self.sourceImages = [sourceImage]
        self.mask = mask
        self.strength = strength
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.outputDir = outputDir
        self.outputFormat = outputFormat
        self.sampler = sampler
        self.scheduler = scheduler
        self.shift = shift
        self.previewInterval = previewInterval
        self.previewMaxPixel = previewMaxPixel
        self.adapters = adapters
    }

    public init(
        prompt: String,
        sourceImages: [URL],
        mask: URL? = nil,
        strength: Float? = nil,
        width: Int? = nil,
        height: Int? = nil,
        steps: Int = 20,
        guidance: Float = 3.5,
        seed: UInt64? = nil,
        outputDir: URL,
        outputFormat: ImageFormat = .png,
        sampler: String? = nil, scheduler: String? = nil, shift: Float? = nil,
        previewInterval: Int? = nil, previewMaxPixel: Int = 256, adapters: [FluxAdapter] = []
    ) throws {
        guard let sourceImage = sourceImages.first else {
            throw FluxError.invalidRequest("ImageEditRequest requires at least one source image")
        }
        self.prompt = prompt
        self.sourceImage = sourceImage
        self.sourceImages = sourceImages
        self.mask = mask
        self.strength = strength
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.outputDir = outputDir
        self.outputFormat = outputFormat
        self.sampler = sampler
        self.scheduler = scheduler
        self.shift = shift
        self.previewInterval = previewInterval
        self.previewMaxPixel = previewMaxPixel
        self.adapters = adapters
    }
}

// MARK: - UpscaleRequest (SeedVR2)

public struct UpscaleRequest: Sendable {
    public var interrupt: FluxInterrupt?
    public var sourceImage: URL
    public var scale: Int          // 2, 4
    public var steps: Int
    public var seed: UInt64?
    public var outputDir: URL
    public var outputFormat: ImageFormat

    public init(
        sourceImage: URL,
        scale: Int = 4,
        steps: Int = 10,
        seed: UInt64? = nil,
        outputDir: URL,
        outputFormat: ImageFormat = .png,
        interrupt: FluxInterrupt? = nil
    ) {
        self.interrupt = interrupt
        self.sourceImage = sourceImage
        self.scale = scale
        self.steps = steps
        self.seed = seed
        self.outputDir = outputDir
        self.outputFormat = outputFormat
    }
}

// MARK: - VideoGenRequest (future)

public struct VideoGenRequest: Sendable {
    public var prompt: String
    public var negativePrompt: String?
    public var width: Int
    public var height: Int
    public var numFrames: Int
    public var fps: Int
    public var steps: Int
    public var guidance: Float
    public var seed: UInt64?
    public var outputDir: URL

    public init(
        prompt: String,
        negativePrompt: String? = nil,
        width: Int = 1280,
        height: Int = 720,
        numFrames: Int = 121,   // WAN 2.1 default: 121 frames @ 24fps ≈ 5s
        fps: Int = 24,
        steps: Int = 50,
        guidance: Float = 5.0,
        seed: UInt64? = nil,
        outputDir: URL
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.fps = fps
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.outputDir = outputDir
    }
}

// MARK: - Events

/// Streamed progress event for gen / edit / upscale.
public enum ImageGenEvent: Sendable {
    /// Incremental progress. `step` is 1-indexed, `total` is the full step count.
    case step(step: Int, total: Int, etaSeconds: Double?)
    /// Optional in-progress preview (partially denoised latent decoded to a
    /// small PNG). Not every model emits previews — this fires at most
    /// every N steps if the scheduler supports it.
    case preview(pngData: Data, step: Int)
    /// Generation finished. `url` points to the saved image on disk.
    /// Metadata is nil for backends that have not implemented effective-parameter reporting.
    /// Callers must not substitute the requested parameters for missing runtime metadata.
    case completed(url: URL, seed: UInt64, effectiveParameters: EffectiveImageParameters? = nil)
    /// One durable output of a multi-image job; stream termination ends the batch.
    case imageCompleted(url: URL, seed: UInt64, index: Int, effectiveParameters: EffectiveImageParameters? = nil)
    /// Fatal error. `hfAuth=true` signals a HuggingFace 401/403 so the UI
    /// can show a "Add HF token" CTA instead of a generic error banner.
    case failed(message: String, hfAuth: Bool)
    /// User or system cancelled the job.
    case cancelled
}

/// Streamed progress event for video gen (future).
public enum VideoGenEvent: Sendable {
    case step(step: Int, total: Int, etaSeconds: Double?)
    case preview(pngData: Data, frame: Int)
    case completed(url: URL, seed: UInt64, fps: Int, frameCount: Int)
    case failed(message: String, hfAuth: Bool)
    case cancelled
}

// MARK: - Formats

public enum ImageFormat: String, Sendable, Codable {
    case png
    case jpeg
    case webp
}
