// Copyright © 2026 Jinho Jang (eric@jangq.ai)
//
// DiffusionGemma VLM wiring — attaches the Gemma4-unified vision tower and
// multimodal embedder to the MLXLLM block-diffusion engine.
//
// The text engine (`DiffusionGemmaModel`, MLXLLM) owns generation; this file
// only contributes the vision compute path:
//
//   pixels → VisionTower → MultimodalEmbedder → text-space features
//   features scattered over `<|image|>` placeholder positions in the prompt
//   embeddings (maskedScatter), image blocks attending bidirectionally
//   during the single-shot encoder prefill.
//
// The bundle ships `processor_class: Gemma4Processor` and a `gemma4_vision`
// tower config, so the processor and vision classes are reused verbatim.
// Audio is intentionally absent: the DiffusionGemma bundle has no
// audio_config/audio_token_id.
//
// Python reference: mlx_vlm/models/diffusion_gemma/{diffusion_gemma,language}.py
// (EncoderModel._embed_inputs, _vision_block_overlay).

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct DiffusionGemmaVLMConfiguration: Codable, Sendable {
    public let core: DiffusionGemmaConfiguration
    public let visionConfig: Gemma4VisionConfig?

    enum CodingKeys: String, CodingKey {
        case visionConfig = "vision_config"
    }

    public init(from decoder: Decoder) throws {
        core = try DiffusionGemmaConfiguration(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        visionConfig = try container.decodeIfPresent(
            Gemma4VisionConfig.self, forKey: .visionConfig)
    }
}

// MARK: - Factory

/// Build the block-diffusion engine and, when the bundle ships a vision
/// tower, install the vision modules + the prompt-embedding splice closure.
/// Text-only bundles (no `vision_config` / `image_token_id`) come back as
/// the plain text engine.
public func makeDiffusionGemmaVLM(
    _ config: DiffusionGemmaVLMConfiguration
) -> DiffusionGemmaModel {
    let model = DiffusionGemmaModel(config.core)
    guard let visionConfig = config.visionConfig,
        let imageTokenId = config.core.imageTokenId
    else {
        return model
    }

    let tower = makeGemma4VisionTower(visionConfig)
    let embedder = makeGemma4MultimodalEmbedder(
        embDim: visionConfig.hiddenSize,
        textDim: config.core.textHiddenSize)

    model.installVision(tower: tower.module, embedder: embedder.module) { input, model in
        guard let image = input.image else { return nil }
        let pixels = image.pixels

        // Per-image tower pass at original (unpadded) dimensions — mirrors
        // the Gemma4 VLM prepare() contract; every image yields exactly
        // defaultOutputLength features.
        let batch = pixels.dim(0)
        var featuresList = [MLXArray]()
        featuresList.reserveCapacity(batch)
        for i in 0 ..< batch {
            let single: MLXArray
            if let frames = image.frames, i < frames.count {
                single = pixels[i, 0..., ..<frames[i].h, ..<frames[i].w]
                    .expandedDimensions(axis: 0)
            } else {
                single = pixels[i].expandedDimensions(axis: 0)
            }
            featuresList.append(embedder.compute(tower.compute(single)))
        }
        let features = batch == 1 ? featuresList[0] : concatenated(featuresList)

        // Text embeddings with image placeholders masked to pad before
        // embedding, then features scattered over the placeholder slots.
        let tokens =
            input.text.tokens.ndim == 1
            ? input.text.tokens.expandedDimensions(axis: 0) : input.text.tokens
        let imageMask = MLX.equal(tokens, MLXArray(Int32(imageTokenId)))
        var embeds = model.embedPromptTokens(tokens)
        let maskExpanded = MLX.broadcast(
            expandedDimensions(imageMask, axis: -1), to: embeds.shape)
        embeds = try gemma4MaskedScatter(
            input: embeds, mask: maskExpanded, source: features.asType(embeds.dtype))

        // Contiguous image-token runs → block ids (−1 for text positions),
        // consumed by the encoder's bidirectional vision-block overlay.
        let ids = tokens.reshaped(-1).asArray(Int32.self)
        var blockIds = [Int32](repeating: -1, count: ids.count)
        var currentBlock: Int32 = -1
        var insideBlock = false
        for (index, token) in ids.enumerated() {
            if token == Int32(imageTokenId) {
                if !insideBlock {
                    currentBlock += 1
                    insideBlock = true
                }
                blockIds[index] = currentBlock
            } else {
                insideBlock = false
            }
        }

        return (embeddings: embeds, visionBlockIds: MLXArray(blockIds))
    }

    return model
}
