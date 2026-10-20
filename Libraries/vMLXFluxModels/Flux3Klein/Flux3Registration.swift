import Foundation
import vMLXFluxKit

/// Metadata construction is independent of tensor allocation and the native loader.
enum Flux3Registration {
    static func entry(loader: @Sendable @escaping (URL, Int?) async throws -> any FluxModel) -> ModelEntry {
        ModelEntry(name: "flux3-image-klein", displayName: "FLUX 3 Image Klein", kind: .imageGen,
            defaultSteps: 16, defaultGuidance: 1, supportsLoRA: false,
            limits: { try Flux3Bundle(modelPath: $0).limits }, loader: loader)
    }
}
