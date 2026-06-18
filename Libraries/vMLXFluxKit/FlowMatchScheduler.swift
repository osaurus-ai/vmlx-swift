import Foundation
import MLX

// MARK: - FlowMatchEulerScheduler
//
// Flow-matching (a.k.a. rectified flow) Euler scheduler used by FLUX.1,
// FLUX.2, ZImage, Qwen-Image, and FIBO. The sampling loop walks a set
// of monotonically decreasing "sigmas" from 1.0 → 0.0, calling the
// transformer at each step with the current latent + timestep embedding,
// and applying a first-order Euler update:
//
//     x_{t-1} = x_t + (σ_{t-1} - σ_t) * v_pred
//
// where `v_pred` is the velocity field predicted by the transformer.
//
// The sigma schedule is parameterized by `shift` — a scalar that skews
// the timestep distribution toward noisier (higher shift) or cleaner
// (lower shift) regions. The Python mflux reference uses:
//
//     shift = base_shift + (max_shift - base_shift) * (image_seq_len - 256) / (4096 - 256)
//
// which gives slightly more steps at low-noise regions for large images.
// Defaults match mflux/models/flux/flow_scheduler.py.

public struct FlowMatchEulerScheduler: Sendable {

    /// Number of sampling steps.
    public let steps: Int

    /// Sigma vector (length `steps + 1`, monotonically decreasing from 1→0).
    public let sigmas: [Float]

    /// Discrete timesteps (length `steps`), derived from sigmas for the
    /// transformer's timestep embedding.
    public let timesteps: [Float]

    public init(
        steps: Int,
        imageSeqLen: Int = 4096,
        baseShift: Float = 0.5,
        maxShift: Float = 1.15,
        baseSeqLen: Float = 256,
        maxSeqLen: Float = 4096,
        shiftTerminal: Float? = nil,
        exponentialShift: Bool = false
    ) {
        self.steps = steps

        // Compute the resolution-dependent shift.
        let shift = Self.computeShift(
            imageSeqLen: imageSeqLen,
            baseShift: baseShift,
            maxShift: maxShift,
            baseSeqLen: baseSeqLen,
            maxSeqLen: maxSeqLen
        )

        // Linspace sigmas from 1 → 0 over (steps + 1) points.
        var rawSigmas: [Float] = []
        for i in 0...steps {
            let t = Float(i) / Float(steps)
            rawSigmas.append(1.0 - t)
        }

        let shifted = rawSigmas.dropLast().map { sigma in
            if exponentialShift {
                let expShift = exp(shift)
                return expShift / (expShift + (1.0 / sigma - 1.0))
            }
            return shift * sigma / (1.0 + (shift - 1.0) * sigma)
        }
        if let shiftTerminal, let last = shifted.last {
            let scale = (1.0 - last) / (1.0 - shiftTerminal)
            self.sigmas = shifted.map { sigma in
                1.0 - ((1.0 - sigma) / scale)
            } + [0.0]
        } else {
            self.sigmas = shifted + [0.0]
        }

        // Timesteps match sigmas[0..<steps] (discrete conditioning value).
        self.timesteps = Array(sigmas.prefix(steps)).map { $0 * 1000.0 }
    }

    public static func qwenImage(
        steps: Int,
        imageSeqLen: Int
    ) -> FlowMatchEulerScheduler {
        FlowMatchEulerScheduler(
            steps: steps,
            imageSeqLen: imageSeqLen,
            baseShift: 0.5,
            maxShift: 0.9,
            baseSeqLen: 256,
            maxSeqLen: 8192,
            shiftTerminal: 0.02,
            exponentialShift: true)
    }

    /// Apply a single Euler update step. `latent` is the current noisy
    /// latent, `velocity` is the transformer's predicted velocity field,
    /// `stepIndex` is 0..<steps.
    public func step(
        latent: MLXArray,
        velocity: MLXArray,
        stepIndex: Int
    ) -> MLXArray {
        let sigmaCurrent = sigmas[stepIndex]
        let sigmaNext = sigmas[stepIndex + 1]
        let delta = sigmaNext - sigmaCurrent
        return latent + velocity * MLXArray(delta)
    }

    /// Compute the resolution-dependent timestep shift.
    public static func computeShift(
        imageSeqLen: Int,
        baseShift: Float = 0.5,
        maxShift: Float = 1.15,
        baseSeqLen: Float = 256,
        maxSeqLen: Float = 4096
    ) -> Float {
        let t = (Float(imageSeqLen) - baseSeqLen) / (maxSeqLen - baseSeqLen)
        let clamped = min(max(t, 0), 1)
        return baseShift + (maxShift - baseShift) * clamped
    }

    /// Convenience: number of actual Euler steps (same as `steps` but
    /// expressed as a function signature for clarity at call sites).
    public var stepCount: Int { steps }
}
