import Foundation

/// Values used by the pipeline, rather than the caller's requested values.
/// Optional fields are absent when the operation does not use that parameter.
public struct EffectiveImageParameters: Sendable, Codable, Equatable {
    public var width: Int
    public var height: Int
    public var steps: Int
    public var guidance: Float
    public var seed: UInt64
    public var sampler: String?
    public var scheduler: String?
    public var shift: Float?
    public var strength: Float?
    public var scale: Int?
    public var modelRevision: String?
    /// A bundle's fixed sigma grid, excluding the terminal zero.
    public var sampleSigmas: [Float]?

    public init(
        width: Int, height: Int, steps: Int, guidance: Float, seed: UInt64,
        sampler: String? = nil, scheduler: String? = nil, shift: Float? = nil,
        strength: Float? = nil, scale: Int? = nil, modelRevision: String? = nil,
        sampleSigmas: [Float]? = nil
    ) {
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.sampler = sampler
        self.scheduler = scheduler
        self.shift = shift
        self.strength = strength
        self.scale = scale
        self.modelRevision = modelRevision
        self.sampleSigmas = sampleSigmas
    }
}
