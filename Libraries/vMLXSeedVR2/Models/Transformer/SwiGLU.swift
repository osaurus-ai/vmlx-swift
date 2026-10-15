// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_transformer/swiglu_mlp.py + mm_swiglu.py.
import MLX
import MLXNN

/// SwiGLU MLP: `proj_out(silu(proj_in_gate(x)) * proj_in(x))`. All Linears bias=false.
public final class SwiGLUMLP: Module, UnaryLayer {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "proj_in_gate") var projInGate: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    public init(dim: Int, expandRatio: Int = 4, multipleOf: Int = 256) {
        var hidden = (2 * dim * expandRatio) / 3
        hidden = multipleOf * ((hidden + multipleOf - 1) / multipleOf)
        self._projIn.wrappedValue = Linear(dim, hidden, bias: false)
        self._projInGate.wrappedValue = Linear(dim, hidden, bias: false)
        self._projOut.wrappedValue = Linear(hidden, dim, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        projOut(silu(projInGate(x)) * projIn(x))
    }
}

/// Dual-stream MLP wrapper (video + text branches), mflux MMSwiGLU.
/// Shared blocks (i >= mm_layers) use a single `all` MLP for both streams.
public final class MMSwiGLU: Module {
    @ModuleInfo(key: "all") var all: SwiGLUMLP?
    @ModuleInfo(key: "vid") var vid: SwiGLUMLP?
    @ModuleInfo(key: "txt") var txt: SwiGLUMLP?
    let isLastLayer: Bool
    let shared: Bool

    public init(
        vidDim: Int, txtDim: Int, expandRatio: Int = 4, shared: Bool = false,
        isLastLayer: Bool = false
    ) {
        self.isLastLayer = isLastLayer
        self.shared = shared
        if shared {
            self._all.wrappedValue = SwiGLUMLP(dim: vidDim, expandRatio: expandRatio)
        } else {
            self._vid.wrappedValue = SwiGLUMLP(dim: vidDim, expandRatio: expandRatio)
            self._txt.wrappedValue =
                isLastLayer ? nil : SwiGLUMLP(dim: txtDim, expandRatio: expandRatio)
        }
        super.init()
    }

    public func callAsFunction(_ vidIn: MLXArray, _ txtIn: MLXArray) -> (MLXArray, MLXArray) {
        let mlpVid = shared ? all! : vid!
        let v = mlpVid(vidIn)
        if isLastLayer { return (v, txtIn) }
        let t = (shared ? all! : txt!)(txtIn)
        return (v, t)
    }
}
