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
    public let supportsNegativePrompt: Bool
    public let supportsEditNegativePrompt: Bool
    public let supportsStrength: Bool
    public let supportsMask: Bool
    public let maxSourceImages: Int
    public let supportedFormats: [ImageFormat]

    public init(
        dimensionMultiple: Int, minDimension: Int, maxDimension: Int? = nil,
        maxPixels: Int? = nil, minimumSteps: Int, maximumSteps: Int? = nil,
        fixedSteps: Int? = nil, supportsNegativePrompt: Bool,
        supportsEditNegativePrompt: Bool, supportsStrength: Bool,
        supportsMask: Bool, maxSourceImages: Int, supportedFormats: [ImageFormat]
    ) {
        self.dimensionMultiple = dimensionMultiple
        self.minDimension = minDimension
        self.maxDimension = maxDimension
        self.maxPixels = maxPixels
        self.minimumSteps = minimumSteps
        self.maximumSteps = maximumSteps
        self.fixedSteps = fixedSteps
        self.supportsNegativePrompt = supportsNegativePrompt
        self.supportsEditNegativePrompt = supportsEditNegativePrompt
        self.supportsStrength = supportsStrength
        self.supportsMask = supportsMask
        self.maxSourceImages = maxSourceImages
        self.supportedFormats = supportedFormats
    }

    public func validate(_ request: ImageGenRequest) throws {
        try validateCommon(width: request.width, height: request.height, steps: request.steps,
                           guidance: request.guidance, format: request.outputFormat)
        guard request.numImages == 1 else {
            throw FluxError.invalidField(field: "numImages", value: "\(request.numImages)", allowed: "1 (native batching is not implemented)")
        }
        if let negative = request.negativePrompt, !negative.isEmpty,
           !supportsNegativePrompt || request.guidance <= 1 {
            throw FluxError.invalidField(field: "negativePrompt", value: "provided", allowed: "a supported CFG path with guidance > 1")
        }
    }

    public func validate(_ request: ImageEditRequest) throws {
        try validateEdit(width: request.width, height: request.height, steps: request.steps,
                         guidance: request.guidance, format: request.outputFormat,
                         sourceCount: request.sourceImages.count, hasMask: request.mask != nil, strength: request.strength)
    }

    /// The host can validate staged or in-memory sources before loading weights.
    public func validateEdit(
        width: Int?, height: Int?, steps: Int, guidance: Float, format: ImageFormat,
        sourceCount: Int, hasMask: Bool, strength: Float?
    ) throws {
        try validateCommon(width: width, height: height, steps: steps, guidance: guidance, format: format)
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
