// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_euler_scheduler.py (1-step euler).
import MLX

public struct SeedVR2EulerScheduler {
    public let timesteps: MLXArray  // [T, ..., 0]
    let ts: [Float]
    let T: Float

    public init(numInferenceSteps: Int = 1, numTrainTimesteps: Int = 1000) {
        self.T = Float(numTrainTimesteps)
        let step = T / Float(numInferenceSteps)
        var t: [Float] = []
        for i in 0 ... numInferenceSteps { t.append(max(T - Float(i) * step, 0)) }
        self.ts = t
        self.timesteps = MLXArray(t)
    }

    public var numSteps: Int { ts.count - 1 }

    /// One euler step. For the single-step case (t=T, s=0) this reduces to `latents - noise`.
    public func step(noise: MLXArray, timestepIdx idx: Int, latents: MLXArray) -> MLXArray {
        let t = ts[idx]
        let s = ts[idx + 1]
        let tNorm = t / T
        let sNorm = s / T
        let predX0 = latents - tNorm * noise
        let predNoise = latents + (1 - tNorm) * noise
        return s > 0 ? (1 - sNorm) * predX0 + sNorm * predNoise : predX0
    }
}
