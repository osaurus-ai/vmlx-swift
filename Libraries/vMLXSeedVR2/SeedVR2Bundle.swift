import Foundation
import MLX
import MLXNN
import vMLXFluxKit

/// The local, self-contained 3B export. Never downloads embeddings or guesses a variant.
public struct SeedVR2Bundle: Sendable {
    public let directory: URL
    public let bits: Int?
    public let groupSize: Int

    public init(directory: URL) throws {
        let metadata = try SeedVR2BundleMetadata(directory: directory)
        self.directory = metadata.directory
        self.bits = metadata.bits
        self.groupSize = metadata.groupSize
    }

    public func load() throws -> (SeedVR2Transformer, SeedVR2VAE, MLXArray) {
        let transformer = SeedVR2Transformer()
        if let bits {
            SeedVR2Quant.quantizeTransformer(transformer, groupSize: groupSize, bits: bits)
        }
        let vae = SeedVR2VAE()
        let tw = try MLX.loadArrays(
            url: directory.appendingPathComponent("transformer.safetensors"))
        let vw = try MLX.loadArrays(url: directory.appendingPathComponent("vae.safetensors"))
        try transformer.update(parameters: ModuleParameters.unflattened(tw), verify: .all)
        try vae.update(parameters: ModuleParameters.unflattened(vw), verify: .all)
        let embeddings = try MLX.loadArrays(
            url: directory.appendingPathComponent("pos_emb.safetensors"))
        guard embeddings.count == 1, let embedding = embeddings["embedding"],
            embedding.shape == [58, 5120],
            [.float16, .bfloat16, .float32].contains(embedding.dtype)
        else {
            throw FluxError.invalidRequest(
                "SeedVR2 positive embedding must be [58,5120] floating point")
        }
        return (transformer, vae, embedding.expandedDimensions(axis: 0))
    }
}
