import Foundation
import MLX
import vMLXFluxKit

/// One-step 3B restoration. No prompt tokenizer: conditioning is the bundle's fixed embedding.
public final class SeedVR2Pipeline {
    private let transformer: SeedVR2Transformer
    private let vae: SeedVR2VAE
    private let text: MLXArray

    public init(bundle: SeedVR2Bundle) throws {
        (transformer, vae, text) = try bundle.load()
        try checkedEval(transformer, vae, text)
    }

    public func upscale(
        _ request: UpscaleRequest, seed: UInt64, checkpoint: () throws -> Void,
        step: () -> Void
    ) throws -> (MLXArray, Int, Int) {
        let image = try SeedVR2Preprocess.read(
            request.sourceImage, scale: request.scale, checkpoint: checkpoint)
        let encoded = try SeedVR2VAETiling.encode(image.tensor, vae: vae, checkpoint: checkpoint)
        let noise = SeedVR2LatentCreator.noiseLatents(
            seed: seed, height: encoded.dim(-2), width: encoded.dim(-1))
        let input = concatenated([noise, SeedVR2LatentCreator.condition(encoded)], axis: 1)
        let prediction = try transformer(
            input, text, timestep: MLXArray(Float(1000)), checkpoint: checkpoint)
        let latent = SeedVR2EulerScheduler().step(noise: prediction, timestepIdx: 0, latents: noise)
        try checkedEval(latent)
        try checkpoint()
        step()
        let decoded = try SeedVR2VAETiling.decode(latent, vae: vae, checkpoint: checkpoint)
        let content = decoded[0..., 0..., 0 ..< image.height, 0 ..< image.width]
        let style = image.tensor[0..., 0..., 0 ..< image.height, 0 ..< image.width]
        let corrected = try SeedVR2ColorCorrect.labTransfer(
            content: content, style: style, checkpoint: checkpoint)
        try checkpoint()
        return ((corrected + 1) * 0.5, image.width, image.height)
    }
}
