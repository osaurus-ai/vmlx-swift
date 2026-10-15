// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_latent_creator.py.
import MLX
import MLXRandom

public enum SeedVR2LatentCreator {
    /// Seeded Gaussian noise latents. MLX-Swift and MLX-Python share the same RNG core,
    /// so `key(seed)` produces identical noise.
    public static func noiseLatents(
        seed: UInt64, height: Int, width: Int,
        batch: Int = 1, latentChannels: Int = 16
    ) -> MLXArray {
        MLXRandom.normal([batch, latentChannels, 1, height, width], key: MLXRandom.key(seed))
    }

    /// Append an all-ones mask channel to the encoded latent: [B,16,T,H,W] -> [B,17,T,H,W].
    public static func condition(_ encoded: MLXArray) -> MLXArray {
        let l = encoded.ndim == 4 ? encoded.expandedDimensions(axis: 2) : encoded
        let (h, w) = (l.shape[3], l.shape[4])
        let mask = MLXArray.ones([1, 1, 1, h, w]).asType(l.dtype)
        return concatenated([l, mask], axis: 1)
    }
}
