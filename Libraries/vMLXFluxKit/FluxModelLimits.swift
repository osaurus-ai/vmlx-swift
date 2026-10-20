import Foundation

/// Architectural/implemented limits. A host may impose additional memory or UI limits.
/// Nil maxima mean no fixed architectural maximum, not unlimited available memory.
public struct FluxModelLimits: Sendable, Codable, Equatable {
    public let dimensionMultiple: Int
    public let minDimension: Int
    public let maxDimension: Int?
    public let maxPixels: Int?
    public let minimumSteps: Int
    public let maximumSteps: Int?
    public let fixedSteps: Int?
    /// Distilled families accept negative prompts without conditioning on them.
    public let ignoresNegativePrompt: Bool
    public let supportsNegativePrompt: Bool
    public let supportsEditNegativePrompt: Bool
    public let supportsStrength: Bool
    public let supportsMask: Bool
    public let maxSourceImages: Int
    public let supportedFormats: [ImageFormat]
    public let samplers: [String]
    public let schedulers: [String]
    public let defaultSampler: String?
    public let defaultScheduler: String?
    public let supportsShift: Bool
    public let maximumImages: Int
    public let maximumParallelImages: Int
    /// Nil means the pipeline always consumes negative conditioning, even at CFG <= 1.
    public let negativePromptRequiresGuidanceAbove: Float?

    public init(
        dimensionMultiple: Int, minDimension: Int, maxDimension: Int? = nil,
        maxPixels: Int? = nil, minimumSteps: Int, maximumSteps: Int? = nil,
        fixedSteps: Int? = nil, supportsNegativePrompt: Bool,
        supportsEditNegativePrompt: Bool, ignoresNegativePrompt: Bool = false, supportsStrength: Bool,
        supportsMask: Bool, maxSourceImages: Int, supportedFormats: [ImageFormat],
        maximumImages: Int = 1, maximumParallelImages: Int = 1,
        negativePromptRequiresGuidanceAbove: Float? = 1,
        samplers: [String] = [], schedulers: [String] = [],
        defaultSampler: String? = nil, defaultScheduler: String? = nil, supportsShift: Bool = false
    ) {
        self.dimensionMultiple = dimensionMultiple
        self.minDimension = minDimension
        self.maxDimension = maxDimension
        self.maxPixels = maxPixels
        self.minimumSteps = minimumSteps
        self.maximumSteps = maximumSteps
        self.fixedSteps = fixedSteps
        self.ignoresNegativePrompt = ignoresNegativePrompt
        self.supportsNegativePrompt = supportsNegativePrompt
        self.supportsEditNegativePrompt = supportsEditNegativePrompt
        self.supportsStrength = supportsStrength
        self.supportsMask = supportsMask
        self.maxSourceImages = maxSourceImages
        self.supportedFormats = supportedFormats
        self.samplers = samplers
        self.schedulers = schedulers
        self.defaultSampler = defaultSampler
        self.defaultScheduler = defaultScheduler
        self.supportsShift = supportsShift
        self.maximumImages = maximumImages
        self.maximumParallelImages = maximumParallelImages
        self.negativePromptRequiresGuidanceAbove = negativePromptRequiresGuidanceAbove
    }

    private enum CodingKeys: String, CodingKey {
        case samplers, schedulers, defaultSampler, defaultScheduler, supportsShift
        case dimensionMultiple, minDimension, maxDimension, maxPixels, minimumSteps, maximumSteps, fixedSteps
        case ignoresNegativePrompt, supportsNegativePrompt, supportsEditNegativePrompt, supportsStrength, supportsMask, maxSourceImages
        case supportedFormats, maximumImages, maximumParallelImages, negativePromptRequiresGuidanceAbove
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            dimensionMultiple: try c.decode(Int.self, forKey: .dimensionMultiple),
            minDimension: try c.decode(Int.self, forKey: .minDimension),
            maxDimension: try c.decodeIfPresent(Int.self, forKey: .maxDimension),
            maxPixels: try c.decodeIfPresent(Int.self, forKey: .maxPixels),
            minimumSteps: try c.decode(Int.self, forKey: .minimumSteps),
            maximumSteps: try c.decodeIfPresent(Int.self, forKey: .maximumSteps),
            fixedSteps: try c.decodeIfPresent(Int.self, forKey: .fixedSteps),
            supportsNegativePrompt: try c.decode(Bool.self, forKey: .supportsNegativePrompt),
            supportsEditNegativePrompt: try c.decode(Bool.self, forKey: .supportsEditNegativePrompt),
            ignoresNegativePrompt: try c.decodeIfPresent(Bool.self, forKey: .ignoresNegativePrompt) ?? false,
            supportsStrength: try c.decode(Bool.self, forKey: .supportsStrength),
            supportsMask: try c.decode(Bool.self, forKey: .supportsMask),
            maxSourceImages: try c.decode(Int.self, forKey: .maxSourceImages),
            supportedFormats: try c.decode([ImageFormat].self, forKey: .supportedFormats),
            maximumImages: try c.decodeIfPresent(Int.self, forKey: .maximumImages) ?? 1,
            maximumParallelImages: try c.decodeIfPresent(Int.self, forKey: .maximumParallelImages) ?? 1,
            negativePromptRequiresGuidanceAbove: c.contains(.negativePromptRequiresGuidanceAbove)
                ? try c.decodeIfPresent(Float.self, forKey: .negativePromptRequiresGuidanceAbove) : 1,
            samplers: try c.decodeIfPresent([String].self, forKey: .samplers) ?? [],
            schedulers: try c.decodeIfPresent([String].self, forKey: .schedulers) ?? [],
            defaultSampler: try c.decodeIfPresent(String.self, forKey: .defaultSampler),
            defaultScheduler: try c.decodeIfPresent(String.self, forKey: .defaultScheduler),
            supportsShift: try c.decodeIfPresent(Bool.self, forKey: .supportsShift) ?? false)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(samplers, forKey: .samplers)
        try c.encode(schedulers, forKey: .schedulers)
        try c.encode(defaultSampler, forKey: .defaultSampler)
        try c.encode(defaultScheduler, forKey: .defaultScheduler)
        try c.encode(supportsShift, forKey: .supportsShift)
        try c.encode(dimensionMultiple, forKey: .dimensionMultiple)
        try c.encode(minDimension, forKey: .minDimension)
        try c.encode(maxDimension, forKey: .maxDimension)
        try c.encode(maxPixels, forKey: .maxPixels)
        try c.encode(minimumSteps, forKey: .minimumSteps)
        try c.encode(maximumSteps, forKey: .maximumSteps)
        try c.encode(fixedSteps, forKey: .fixedSteps)
        try c.encode(ignoresNegativePrompt, forKey: .ignoresNegativePrompt)
        try c.encode(supportsNegativePrompt, forKey: .supportsNegativePrompt)
        try c.encode(supportsEditNegativePrompt, forKey: .supportsEditNegativePrompt)
        try c.encode(supportsStrength, forKey: .supportsStrength)
        try c.encode(supportsMask, forKey: .supportsMask)
        try c.encode(maxSourceImages, forKey: .maxSourceImages)
        try c.encode(supportedFormats, forKey: .supportedFormats)
        try c.encode(maximumImages, forKey: .maximumImages)
        try c.encode(maximumParallelImages, forKey: .maximumParallelImages)
        try c.encode(negativePromptRequiresGuidanceAbove, forKey: .negativePromptRequiresGuidanceAbove)
    }

    public func validate(_ request: ImageGenRequest) throws {
        try validatePreview(interval: request.previewInterval, maxPixel: request.previewMaxPixel)
        try validateSampling(sampler: request.sampler, scheduler: request.scheduler, shift: request.shift)
        try validateCommon(width: request.width, height: request.height, steps: request.steps,
                           guidance: request.guidance, format: request.outputFormat)
        try ImageBatchPlan.validate(request, maximumImages: maximumImages)
        if !ignoresNegativePrompt, let negative = request.negativePrompt, !negative.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           !supportsNegativePrompt || negativePromptRequiresGuidanceAbove.map({ request.guidance <= $0 }) == true {
            throw FluxError.invalidField(field: "negativePrompt", value: "provided", allowed: negativePromptRequiresGuidanceAbove.map { "a supported CFG path with guidance > \($0)" } ?? "a supported negative-conditioning path")
        }
    }

    public func validate(_ request: ImageEditRequest) throws {
        try validatePreview(interval: request.previewInterval, maxPixel: request.previewMaxPixel)
        try validateSampling(sampler: request.sampler, scheduler: request.scheduler, shift: request.shift)
        try validateEdit(width: request.width, height: request.height, steps: request.steps,
                         guidance: request.guidance, format: request.outputFormat,
                         sourceCount: request.sourceImages.count, hasMask: request.mask != nil, strength: request.strength)
    }

    /// The host can validate staged or in-memory sources before loading weights.
    public func validateEdit(
        width: Int?, height: Int?, steps: Int, guidance: Float, format: ImageFormat,
        sourceCount: Int, hasMask: Bool, strength: Float?, negativePrompt: String? = nil
    ) throws {
        try validateCommon(width: width, height: height, steps: steps, guidance: guidance, format: format)
        if !ignoresNegativePrompt, let negativePrompt, !negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            !supportsEditNegativePrompt {
            throw FluxError.invalidField(field: "negativePrompt", value: "provided", allowed: "no edit negative prompt")
        }
        guard sourceCount >= 1, sourceCount <= maxSourceImages else {
            throw FluxError.invalidField(field: "sourceImages", value: "\(sourceCount)", allowed: "1...\(maxSourceImages)")
        }
        if hasMask, !supportsMask {
            throw FluxError.invalidField(field: "mask", value: "provided", allowed: "no mask")
        }
        if let strength, !supportsStrength, strength != 1 {
            throw FluxError.invalidField(field: "strength", value: "\(strength)", allowed: "omitted or 1; partial denoising is not supported")
        }
    }

    public func validatePreview(interval: Int?, maxPixel: Int) throws {
        guard let interval else { return }
        guard interval > 0 else {
            throw FluxError.invalidField(field: "previewInterval", value: "\(interval)", allowed: "a positive step interval")
        }
        guard (16...1024).contains(maxPixel) else {
            throw FluxError.invalidField(field: "previewMaxPixel", value: "\(maxPixel)", allowed: "16...1024 pixels")
        }
    }

    public func validateSampling(sampler: String?, scheduler: String?, shift: Float?) throws {
        if let sampler, !samplers.contains(sampler) {
            throw FluxError.invalidField(field: "sampler", value: sampler, allowed: samplers.joined(separator: ", "))
        }
        if let scheduler, !schedulers.contains(scheduler) {
            throw FluxError.invalidField(field: "scheduler", value: scheduler, allowed: schedulers.joined(separator: ", "))
        }
        if let shift, !supportsShift || !shift.isFinite {
            throw FluxError.invalidField(field: "shift", value: "\(shift)", allowed: supportsShift ? "a finite timestep shift" : "the bundle's native schedule")
        }
    }

    private func validateCommon(width: Int?, height: Int?, steps: Int, guidance: Float, format: ImageFormat) throws {
        for (field, dimension) in [("width", width), ("height", height)] {
            if let dimension, dimension < minDimension || dimension % dimensionMultiple != 0
                || maxDimension.map({ dimension > $0 }) == true {
                throw FluxError.invalidField(field: field, value: "\(dimension)", allowed: "multiples of \(dimensionMultiple), minimum \(minDimension)\(maxDimension.map { ", maximum \($0)" } ?? "")")
            }
        }
        if let width, let height {
            let pixels = width.multipliedReportingOverflow(by: height)
            if pixels.overflow || maxPixels.map({ pixels.partialValue > $0 }) == true {
                throw FluxError.invalidField(field: "dimensions", value: "\(width)x\(height)", allowed: maxPixels.map { "at most \($0) pixels" } ?? "a representable pixel count")
            }
        }
        let actualSteps = fixedSteps ?? steps
        guard actualSteps >= minimumSteps, maximumSteps.map({ actualSteps <= $0 }) ?? true else {
            throw FluxError.invalidField(field: "steps", value: "\(actualSteps)", allowed: "at least \(minimumSteps)\(maximumSteps.map { ", at most \($0)" } ?? "")")
        }
        guard guidance.isFinite else {
            throw FluxError.invalidField(field: "guidance", value: "\(guidance)", allowed: "a finite number")
        }
        guard supportedFormats.contains(format) else {
            throw FluxError.invalidField(field: "outputFormat", value: format.rawValue, allowed: supportedFormats.map(\.rawValue).joined(separator: ", "))
        }
    }
}
