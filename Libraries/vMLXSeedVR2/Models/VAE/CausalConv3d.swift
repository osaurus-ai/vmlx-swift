// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_vae/common/conv3d.py.
import MLX
import MLXNN

/// VAE activation precision after group-norms (mflux `ModelConfig.precision`).
enum VAEPrecision { static let dtype: DType = .bfloat16 }

/// Causal (in time) 3D conv. Weight layout [O, kt, kh, kw, I] (MLX NDHWC) — load directly.
public final class CausalConv3d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let k: (Int, Int, Int), s: (Int, Int, Int), p: (Int, Int, Int)
    let causalTemporal: Bool, usePaddingCausal: Bool

    public init(
        _ inCh: Int, _ outCh: Int, kernel: (Int, Int, Int) = (3, 3, 3),
        stride: (Int, Int, Int) = (1, 1, 1), padding: (Int, Int, Int) = (1, 1, 1),
        causalTemporal: Bool = true, usePaddingCausal: Bool = false
    ) {
        self.k = kernel
        self.s = stride
        self.p = padding
        self.causalTemporal = causalTemporal
        self.usePaddingCausal = usePaddingCausal
        self._weight.wrappedValue = MLXArray.zeros([outCh, kernel.0, kernel.1, kernel.2, inCh])
        self._bias.wrappedValue = MLXArray.zeros([outCh])
        super.init()
    }

    public func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        var x = xIn  // [B,C,T,H,W]
        var tPad = p.0
        if causalTemporal && k.0 > 1 {
            let causalPad = usePaddingCausal ? 2 * p.0 : k.0 - 1
            if causalPad > 0 {
                let first = x[0..., 0..., 0 ..< 1]
                let pad = repeated(first, count: causalPad, axis: 2)
                x = concatenated([pad, x], axis: 2)
            }
            tPad = 0
        }
        x = x.transposed(0, 2, 3, 4, 1)  // NDHWC
        var out = convGeneral(x, weight, strides: [s.0, s.1, s.2], padding: [tPad, p.1, p.2])
        out = out + bias
        return out.transposed(0, 4, 1, 2, 3)  // back to [B,O,T,H,W]
    }
}

/// GroupNorm over channels (input [B,C,T,H,W]) done in fp32, cast back to VAE precision.
func vaeGroupNorm(_ x: MLXArray, _ norm: GroupNorm) -> MLXArray {
    var h = x.transposed(0, 2, 3, 4, 1)  // NDHWC
    h = norm(h.asType(.float32)).asType(VAEPrecision.dtype)
    return h.transposed(0, 4, 1, 2, 3)
}
