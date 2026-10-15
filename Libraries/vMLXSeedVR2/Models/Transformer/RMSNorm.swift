// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
//
// Port of mflux seedvr2_transformer/rms_norm.py (1:1).
import Foundation
import MLX
import MLXFast
import MLXNN

/// RMSNorm matching mflux: `mx.fast.rms_norm(x, weight, eps)`.
public final class SeedVR2RMSNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") public var weight: MLXArray
    let eps: Float

    public init(_ dim: Int, eps: Float = 1e-5) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dim])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}
