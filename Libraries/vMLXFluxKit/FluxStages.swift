import Foundation
@preconcurrency import MLX

public struct FluxConditioning: Sendable, Hashable {
    public let id: UUID
    public init(id: UUID = UUID()) { self.id = id }
}

public struct FluxLatent: Sendable, Hashable {
    public let id: UUID
    public let width: Int
    public let height: Int
    public init(id: UUID = UUID(), width: Int, height: Int) {
        self.id = id; self.width = width; self.height = height
    }
}

public enum FluxStageHandle: Sendable, Hashable {
    case conditioning(FluxConditioning)
    case latent(FluxLatent)
}

public enum FluxSampleEvent: Sendable {
    case step(step: Int, total: Int, etaSeconds: Double?)
    case preview(pngData: Data, step: Int)
    case completed(FluxLatent, effective: EffectiveImageParameters)
    case failed(message: String, hfAuth: Bool)
    case cancelled
}

/// Internal stage payload shared by native pipelines. Public handles never expose
/// arrays; the engine owns their lifetime. Spatial latents use BHWC throughout.
public struct FluxStageConditioning: @unchecked Sendable {
    public let arrays: [String: MLXArray]
    public init(arrays: [String: MLXArray]) { self.arrays = arrays }
}

public struct FluxStageSampling: Sendable {
    public var steps: Int
    public var guidance: Float
    public var seed: UInt64
    public var sampler: String?
    public var scheduler: String?
    public var shift: Float?
    public var denoise: Float
    public init(steps: Int, guidance: Float, seed: UInt64, sampler: String? = nil,
                scheduler: String? = nil, shift: Float? = nil, denoise: Float = 1) {
        self.steps = steps; self.guidance = guidance; self.seed = seed
        self.sampler = sampler; self.scheduler = scheduler; self.shift = shift; self.denoise = denoise
    }
}

/// Implemented against the same native pipeline used by generate/edit. A family
/// must opt in; generic synthetic latents or decoding are never substituted.
public protocol FluxStageModel: FluxModel {
    var stageSpatialScale: Int { get }
    var stageChannels: Int { get }
    func stageEncodePrompt(_ prompt: String, negativePrompt: String?) throws -> FluxStageConditioning
    func stageEncodeImage(_ image: URL, width: Int, height: Int) throws -> MLXArray
    func stageSample(_ conditioning: FluxStageConditioning, latent: MLXArray,
                     width: Int, height: Int, options: FluxStageSampling,
                     interrupt: FluxInterrupt?, progress: (Int, Int, Double?) -> Void)
        throws -> (MLXArray, EffectiveImageParameters)
    func stageDecode(_ latent: MLXArray) throws -> MLXArray
}
