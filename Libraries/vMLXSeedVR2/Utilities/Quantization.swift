// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
//
// The int8 export quantizes the transformer; VAE precision is unchanged.
import MLX
import MLXNN

public enum SeedVR2Quant {
    /// Quantize the transformer's Linear layers in place. Skips any Linear whose input
    /// dimension isn't divisible by `groupSize` (e.g. `vid_in.proj` at in=132) — those
    /// stay fp16. Use the IDENTICAL call on save and load so the same layers are quantized.
    public static func quantizeTransformer(
        _ model: SeedVR2Transformer, groupSize: Int = 64, bits: Int = 8
    ) {
        quantize(model: model, groupSize: groupSize, bits: bits) { _, m in
            guard let lin = m as? Linear else { return false }
            return (lin.weight.shape.last ?? 0) % groupSize == 0
        }
    }
}
