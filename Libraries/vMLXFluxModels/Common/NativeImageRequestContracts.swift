import Foundation
import vMLXFluxKit

/// Constraints of implemented pipelines, independent of a host's RAM admission
/// or picker policy. Stub registry entries deliberately have no working contract.
enum NativeImageRequestContracts {
    static let fluxSchnell = FluxModelLimits(
        dimensionMultiple: 16, minDimension: 16, minimumSteps: 1,
        supportsNegativePrompt: false, supportsEditNegativePrompt: false,
        supportsStrength: false, supportsMask: false, maxSourceImages: 0, supportedFormats: [.png],
        samplers: ["euler"], schedulers: ["flux-flow"], defaultSampler: "euler", defaultScheduler: "flux-flow")
    static let zImage = FluxModelLimits(
        dimensionMultiple: 16, minDimension: 16, minimumSteps: 1,
        supportsNegativePrompt: true, supportsEditNegativePrompt: false,
        supportsStrength: false, supportsMask: false, maxSourceImages: 0, supportedFormats: [.png],
        negativePromptRequiresGuidanceAbove: 0,
        samplers: ["euler"], schedulers: ["z-image-linear"], defaultSampler: "euler", defaultScheduler: "z-image-linear")
    static let qwenImage = FluxModelLimits(
        dimensionMultiple: 16, minDimension: 16, minimumSteps: 1,
        supportsNegativePrompt: true, supportsEditNegativePrompt: false,
        supportsStrength: false, supportsMask: false, maxSourceImages: 0, supportedFormats: [.png],
        negativePromptRequiresGuidanceAbove: nil,
        samplers: ["euler"], schedulers: ["qwen-image"], defaultSampler: "euler", defaultScheduler: "qwen-image")
    // The edit preprocessor accepts any integer >=16 and floors it to a
    // multiple of16. Metadata comes from the written image, so replay uses
    // exactly the dimensions the VAE decoded.
    static let qwenEdit = FluxModelLimits(
        dimensionMultiple: 1, minDimension: 16, minimumSteps: 1,
        supportsNegativePrompt: false, supportsEditNegativePrompt: false,
        supportsStrength: false, supportsMask: false, maxSourceImages: 4, supportedFormats: [.png],
        samplers: ["euler"], schedulers: ["qwen-image"], defaultSampler: "euler", defaultScheduler: "qwen-image")
    static let ideogram = FluxModelLimits(
        dimensionMultiple: 16, minDimension: 256, maxDimension: 2048, minimumSteps: 1,
        supportsNegativePrompt: false, supportsEditNegativePrompt: false,
        supportsStrength: false, supportsMask: false, maxSourceImages: 0, supportedFormats: [.png],
        samplers: ["euler"], schedulers: ["ideogram-logit-normal"], defaultSampler: "euler", defaultScheduler: "ideogram-logit-normal")

    static func parameters(url: URL, steps: Int, guidance: Float, seed: UInt64,
                           scheduler: String, shift: Float? = nil) throws -> EffectiveImageParameters {
        let size = try ImageIO.dimensions(of: url)
        return EffectiveImageParameters(width: size.width, height: size.height,
            steps: steps, guidance: guidance, seed: seed, sampler: "euler",
            scheduler: scheduler, shift: shift)
    }
}
