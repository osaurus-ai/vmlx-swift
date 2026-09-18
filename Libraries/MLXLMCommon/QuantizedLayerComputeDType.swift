// Copyright © 2026 osaurus-eval contributors
// SPDX-License-Identifier: MIT

import MLX
import MLXNN

// A quantized layer's `weight` holds packed integer codes, so `weight.dtype` is uint32 there, not the
// dtype the layer computes in. An activation cast to it is truncated to integers, and a scalar typed
// by it loses its fraction. Code that needs a layer's dtype asks for `computeDType` instead.

extension Linear {
    /// The dtype this layer computes in: the one to cast an input to before calling it.
    ///
    /// For a float layer that is its weight's dtype. A quantized layer computes in the dtype its
    /// codes dequantize to: the scales' in affine mode, bfloat16 in the others, which is how
    /// `dequantized` types them when given no dtype.
    public var computeDType: DType {
        guard let quantized = self as? QuantizedLinear else { return weight.dtype }
        return quantized.mode == .affine ? quantized.scales.dtype : .bfloat16
    }
}

extension Embedding {
    /// The dtype of the rows this table returns.
    ///
    /// Read off a one-row lookup, whose dtype MLX infers without evaluating it. That covers a float
    /// table (its weight's dtype), a quantized one (the dtype its codes dequantize to) and one whose
    /// loader pinned `QuantizedEmbedding.outputDType`, without this code having to know which.
    public var computeDType: DType {
        callAsFunction(MLXArray([Int32(0)])).dtype
    }
}
