import Foundation
import vMLXFluxKit

/// Scalar float32 operations mirror mflux's float32 linspace and flow-map Euler state.
/// Kept independent of MLX so parity tests never initialize a GPU device.
public struct Flux3Schedule: Sendable {
    public let timesteps: [Float]
    public init(steps: Int, alpha: Float) throws {
        guard steps >= 1, alpha.isFinite, alpha > 0 else {
            throw FluxError.invalidRequest("flow-map requires positive steps and alpha")
        }
        timesteps = (0...steps).map { i in
            let t = Float(1) - Float(i) / Float(steps)
            return alpha * t / (1 + (alpha - 1) * t)
        }
    }
    public func step(latents: [Float], velocity: [Float], index: Int) throws -> [Float] {
        guard latents.count == velocity.count, index >= 0, index + 1 < timesteps.count else {
            throw FluxError.invalidRequest("flow-map step shape/index mismatch")
        }
        let dt = timesteps[index + 1] - timesteps[index]
        return zip(latents, velocity).map { $0 + dt * $1 }
    }
}
