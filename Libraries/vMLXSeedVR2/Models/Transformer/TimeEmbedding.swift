// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_transformer/time_embedding.py.
import Foundation
import MLX
import MLXNN

public final class TimeEmbedding: Module {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "proj_hid") var projHid: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear
    let sinusoidalDim: Int

    public init(sinusoidalDim: Int = 256, hiddenDim: Int, outputDim: Int) {
        self.sinusoidalDim = sinusoidalDim
        self._projIn.wrappedValue = Linear(sinusoidalDim, hiddenDim)
        self._projHid.wrappedValue = Linear(hiddenDim, hiddenDim)
        self._projOut.wrappedValue = Linear(hiddenDim, outputDim)
        super.init()
    }

    public func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        let ts = timestep.ndim == 0 ? timestep.expandedDimensions(axis: 0) : timestep
        var emb = Self.sinusoid(ts, dim: sinusoidalDim)
        emb = silu(projIn(emb))
        emb = silu(projHid(emb))
        return projOut(emb)
    }

    /// `_get_timestep_embedding`: [sin(t·freqs), cos(t·freqs)] — a constant transform, not a parameter.
    static func sinusoid(_ timesteps: MLXArray, dim: Int) -> MLXArray {
        let half = dim / 2
        let freqs = exp(MLXArray(0 ..< half).asType(.float32) * Float(-log(10000.0) / Double(half)))
        let args = timesteps.expandedDimensions(axis: 1).asType(.float32) * freqs
        return concatenated([sin(args), cos(args)], axis: -1)
    }
}
