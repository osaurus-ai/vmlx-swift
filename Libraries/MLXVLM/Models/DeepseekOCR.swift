//
//  DeepseekOCR.swift
//  mlx-swift-lm
//
//  Top-level DeepSeek-OCR / Unlimited-OCR VLM (DeepseekOCRForCausalLM, top
//  model_type "deepseek_vl_v2"). Wires together the four already-ported
//  component towers:
//
//    - `vision_model`   (DeepseekOCRVisionModel)   CLIP-L/14 ViT
//    - `sam_model`      (DeepseekOCRSAMEncoder)     SAM-ViT-B
//    - `language_model` (DeepseekOCRLanguageModel)  DeepSeek-V2 MoE decoder
//    - `projector`      (MlpProjector)              vision feat dim -> n_embed
//
//  plus the two formatting parameters `image_newline` and `view_separator`.
//
//  Faithful port of mlx-vlm's deepseekocr/deepseekocr.py — the orchestration in
//  `get_input_embeddings` (SAM+CLIP fusion, projector, 2D image tiling with
//  per-row image_newline and trailing view_separator, scatter into the text
//  embeddings at image-token positions) is mirrored 1:1.
//
//  The decoder uses PLAIN 1D RoPE (handled inside DeepseekOCRLanguageModel);
//  there is NO M-RoPE / position-id pre-compute here (that is GlmOcr-specific).
//
//  Weight-key layout (after `sanitize`):
//    vision_model.*      sam_model.*      projector.*
//    language_model.model.*      language_model.lm_head.*
//    image_newline (n_embed,)    view_separator (n_embed,)
//

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - MlpProjector

/// Projects fused vision features (CLIP[:,1:] ++ SAM flatten) to the text
/// hidden size. DeepSeek-OCR uses projector_type "linear" — a single Linear.
/// Mirrors deepseekocr.py `MlpProjector` (linear branch). The
/// `downsample_mlp_gelu` branch is ported for fidelity but is not exercised by
/// the shipped config.
private class MlpProjector: Module, UnaryLayer {

    let projectorType: String
    let downsampleRatio: Int

    // The projector weights live under the `layers` attribute in the checkpoint
    // (`projector.layers.{weight,bias}` for the linear case). DeepSeek-OCR ships
    // projector_type "linear", so a single Linear keyed "layers" matches exactly.
    @ModuleInfo(key: "layers") var layers: Linear

    init(_ config: DeepseekOCRConfiguration.ProjectorConfiguration) {
        self.projectorType = config.projectorType
        self.downsampleRatio = config.downsampleRatio

        // NOTE: only the "linear" projector ships with DeepSeek-OCR /
        // Unlimited-OCR. The "downsample_mlp_gelu" branch in deepseekocr.py would
        // require a heterogeneous (Linear/GELU) module list under `layers`, which
        // MLX-Swift cannot key uniformly to `projector.layers.{i}`; it is omitted
        // here. Guard loudly so an unexpected config surfaces immediately.
        precondition(
            config.projectorType == "linear",
            "DeepseekOCR MlpProjector only supports projector_type=linear, got \(config.projectorType)"
        )
        self._layers.wrappedValue = Linear(config.inputDim, config.nEmbed)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Linear branch (deepseekocr.py): x = self.layers(x).
        layers(x)
    }
}

// MARK: - Model

/// Top-level DeepSeek-OCR VLM.
public class DeepseekOCR: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_model") private var visionModel: DeepseekOCRVisionModel
    @ModuleInfo(key: "sam_model") private var samModel: DeepseekOCRSAMEncoder
    @ModuleInfo(key: "language_model") private var languageModel: DeepseekOCRLanguageModel
    @ModuleInfo(key: "projector") private var projector: MlpProjector

    @ParameterInfo(key: "image_newline") private var imageNewline: MLXArray
    @ParameterInfo(key: "view_separator") private var viewSeparator: MLXArray

    public let config: DeepseekOCRConfiguration

    public var vocabularySize: Int { config.textConfiguration.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    // NOTE: DeepseekOCRTextModelInner.layers is `fileprivate` in the component
    // file (which we must not edit), so the decoder layers aren't reachable here.
    // LoRA is not part of the OCR inference path, so expose no adaptable layers.
    public var loraLayers: [Module] {
        []
    }

    public init(_ config: DeepseekOCRConfiguration) {
        self.config = config
        self._visionModel.wrappedValue = DeepseekOCRVisionModel(config.visionConfiguration)
        self._samModel.wrappedValue = DeepseekOCRSAMEncoder(config.samConfiguration)
        self._languageModel.wrappedValue = DeepseekOCRLanguageModel(config.textConfiguration)
        self._projector.wrappedValue = MlpProjector(config.projectorConfiguration)

        let nEmbed = config.projectorConfiguration.nEmbed
        self._imageNewline.wrappedValue = MLXArray.zeros([nEmbed])
        self._viewSeparator.wrappedValue = MLXArray.zeros([nEmbed])
    }

    /// Debug: print mean/std/shape of an intermediate tensor when DSOCR_DUMP=1.
    /// Used to localize numerical divergence vs the PyTorch reference (SAM /
    /// CLIP / fused / projector ranges). No-op in normal operation.
    static func dumpStat(_ name: String, _ x: MLXArray) {
        guard ProcessInfo.processInfo.environment["DSOCR_DUMP"] == "1" else { return }
        let xf = x.asType(.float32)
        let m = xf.mean().item(Float.self)
        let s = sqrt((xf * xf).mean().item(Float.self) - m * m)
        let mn = xf.min().item(Float.self)
        let mx = xf.max().item(Float.self)
        FileHandle.standardError.write(Data(
            "[stat] \(name): shape=\(x.shape) mean=\(m) std=\(s) min=\(mn) max=\(mx)\n".utf8))
    }

    // MARK: get_input_embeddings

    /// Port of deepseekocr.py `Model.get_input_embeddings`.
    ///
    /// `patches`    : local crop views  [N_crops, 3, image_size, image_size]   (may be nil)
    /// `imageOri`   : global views      [N_images, 3, base_size, base_size]
    /// `spatialCrop`: per-image (width_crop_num, height_crop_num)
    /// `seqMask`    : [B, L] bool image-token mask (carried from the processor)
    ///
    /// Returns the text embeddings with image features scattered into the
    /// image-token positions.
    private func inputEmbeddings(
        inputIds: MLXArray,
        patches: MLXArray?,
        imageOri: MLXArray?,
        spatialCrop: [[Int]],
        seqMask: MLXArray?
    ) -> MLXArray {
        var inputEmbeds = languageModel.model.embedTokens(inputIds)

        // No images, or autoregressive decode step (L == 1): plain embeddings.
        guard let imageOri, inputIds.dim(1) != 1 else {
            return inputEmbeds
        }
        // Mirror Python's `mx.sum(pixel_values[1]).item() != 0` guard: the global
        // views are always present on prefill, so this is effectively a "has any
        // pixel content" check.
        if imageOri.sum().item(Float.self) == 0 {
            return inputEmbeds
        }

        var idx = 0
        var patchIdx = 0

        for crop in spatialCrop {
            let widthCropNum = crop[0]
            let heightCropNum = crop[1]
            let hasCrops = widthCropNum > 1 || heightCropNum > 1
            let numPatches = hasCrops ? widthCropNum * heightCropNum : 0

            // Extract local crop patches for this image.
            var imagePatches: MLXArray?
            if hasCrops, numPatches > 0, let patches {
                imagePatches = patches[patchIdx ..< (patchIdx + numPatches)]
                patchIdx += numPatches
            } else {
                imagePatches = nil
            }

            // Global view for this image (one per batch item).
            let imageOriSingle = imageOri[idx ..< (idx + 1)]

            let globalLocalFeatures: MLXArray

            if let imagePatches, imagePatches.sum().item(Float.self) != 0 {
                // --- local crop features ---
                Self.dumpStat("crop.imagePatches", imagePatches)
                let localFeatures1 = samModel(imagePatches.transposed(0, 2, 3, 1))
                Self.dumpStat("crop.sam_out", localFeatures1)
                let localFeatures2 = visionModel(
                    imagePatches.transposed(0, 2, 3, 1), patchEmbeds: localFeatures1)
                Self.dumpStat("crop.clip_out", localFeatures2)
                var localFeatures = concatenated(
                    [
                        localFeatures2[0..., 1...],
                        flattened(localFeatures1, start: 1, end: 2),
                    ], axis: -1)
                localFeatures = projector(localFeatures)
                Self.dumpStat("crop.localProjected", localFeatures)

                // --- global view features ---
                let globalFeatures1 = samModel(imageOriSingle.transposed(0, 2, 3, 1))
                let globalFeatures2 = visionModel(
                    imageOriSingle.transposed(0, 2, 3, 1), patchEmbeds: globalFeatures1)
                var globalFeatures = concatenated(
                    [
                        globalFeatures2[0..., 1...],
                        flattened(globalFeatures1, start: 1, end: 2),
                    ], axis: -1)
                globalFeatures = projector(globalFeatures)

                // Drop batch dim: (hw, n_dim)
                globalFeatures = globalFeatures[0]
                let hw = globalFeatures.dim(0)
                let nDim = globalFeatures.dim(1)
                let h = Int(Double(hw).squareRoot())
                let w = h

                let hw2 = localFeatures.dim(1)
                let nDim2 = localFeatures.dim(2)
                let h2 = Int(Double(hw2).squareRoot())
                let w2 = h2

                // Global: append one image_newline column per row, flatten.
                globalFeatures = globalFeatures.reshaped(h, w, nDim)
                globalFeatures = concatenated(
                    [
                        globalFeatures,
                        broadcast(imageNewline[.newAxis, .newAxis, 0...], to: [h, 1, nDim]),
                    ], axis: 1)
                globalFeatures = globalFeatures.reshaped(-1, nDim)

                // Local: reassemble (H_crop, W_crop) grid of (h2, w2) tiles, append
                // one image_newline column per row, flatten.
                localFeatures =
                    localFeatures
                    .reshaped(heightCropNum, widthCropNum, h2, w2, nDim2)
                    .transposed(0, 2, 1, 3, 4)
                    .reshaped(heightCropNum * h2, widthCropNum * w2, nDim2)
                localFeatures = concatenated(
                    [
                        localFeatures,
                        broadcast(
                            imageNewline[.newAxis, .newAxis, 0...],
                            to: [heightCropNum * h2, 1, nDim2]),
                    ], axis: 1)
                localFeatures = localFeatures.reshaped(-1, nDim2)

                globalLocalFeatures = concatenated(
                    [localFeatures, globalFeatures, viewSeparator[.newAxis, 0...]], axis: 0)
            } else {
                // Global-only (no crops).
                let globalFeatures1 = samModel(imageOriSingle.transposed(0, 2, 3, 1))
                let globalFeatures2 = visionModel(
                    imageOriSingle.transposed(0, 2, 3, 1), patchEmbeds: globalFeatures1)
                var globalFeatures = concatenated(
                    [
                        globalFeatures2[0..., 1...],
                        flattened(globalFeatures1, start: 1, end: 2),
                    ], axis: -1)
                Self.dumpStat("imageOri", imageOriSingle)
                Self.dumpStat("sam_out(globalFeatures1)", globalFeatures1)
                Self.dumpStat("clip_out(globalFeatures2)", globalFeatures2)
                Self.dumpStat("fused", globalFeatures)
                globalFeatures = projector(globalFeatures)
                Self.dumpStat("projected", globalFeatures)
                Self.dumpStat("image_newline", imageNewline)
                Self.dumpStat("view_separator", viewSeparator)

                globalFeatures = globalFeatures[0]
                let hw = globalFeatures.dim(0)
                let nDim = globalFeatures.dim(1)
                let h = Int(Double(hw).squareRoot())
                let w = h

                globalFeatures = globalFeatures.reshaped(h, w, nDim)
                globalFeatures = concatenated(
                    [
                        globalFeatures,
                        broadcast(imageNewline[.newAxis, .newAxis, 0...], to: [h, 1, nDim]),
                    ], axis: 1)
                globalFeatures = globalFeatures.reshaped(-1, nDim)

                globalLocalFeatures = concatenated(
                    [globalFeatures, viewSeparator[.newAxis, 0...]], axis: 0)
            }

            // Scatter into the image-token positions of this batch row. Mirrors
            // `input_embeds[idx, image_indices] = images_in_this_batch`.
            if let seqMask {
                let rowMask: [Bool] = seqMask[idx].asArray(Int32.self).map { $0 != 0 }
                let imageIndices = rowMask.enumerated().compactMap { $0.element ? $0.offset : nil }
                if !imageIndices.isEmpty {
                    // Canonical scatter (mirrors QwenVL.mergeInputIdsWithImageFeatures):
                    // full slice on batch + channel axes, MLXArray index on seq axis.
                    let indexArray = MLXArray(imageIndices)
                    inputEmbeds[idx ..< (idx + 1), indexArray, 0...] =
                        globalLocalFeatures[.newAxis, 0..., 0...]
                }
            }

            idx += 1
        }

        return inputEmbeds
    }

    // MARK: - Pixel unpacking (decode the processor's packing contract)

    /// Decode the processor's `LMInput.ProcessedImage` packing contract back into
    /// `(patches, image_ori, images_spatial_crop)`.
    ///
    /// Per DeepseekOCRProcessor:
    ///   pixels      = image_ori.flatten() [++ patches.flatten() when N_crops > 0]
    ///   frames[0]   = THW(t: N_crops, h: N_images, w: 0)
    ///   frames[1..] = one THW(1, width_crop_num, height_crop_num) per image
    ///                 (= images_spatial_crop)
    ///
    /// image_ori is reshaped [N_images, 3, base, base]; patches (if any) is the
    /// remaining buffer reshaped [N_crops, 3, image, image].
    private func unpackPixels(_ image: LMInput.ProcessedImage, dtype: MLX.DType)
        -> (patches: MLXArray?, imageOri: MLXArray, spatialCrop: [[Int]])?
    {
        guard let frames = image.frames, let head = frames.first else { return nil }

        let nCrops = head.t
        let nImages = head.h
        let base = config.samConfiguration.imageSize  // 1024 global view side
        let imageSize = 640  // local crop tile side (processing_deepseekocr.py)

        let flat = image.pixels.reshaped(-1).asType(dtype)

        let oriCount = nImages * 3 * base * base
        let oriFlat = flat[0 ..< oriCount]
        let imageOri = oriFlat.reshaped(nImages, 3, base, base)

        var patches: MLXArray?
        if nCrops > 0 {
            let patchCount = nCrops * 3 * imageSize * imageSize
            let patchFlat = flat[oriCount ..< (oriCount + patchCount)]
            patches = patchFlat.reshaped(nCrops, 3, imageSize, imageSize)
        }

        // images_spatial_crop = frames[1...] as (width_crop_num, height_crop_num).
        let spatialCrop: [[Int]] = frames.dropFirst().map { [$0.h, $0.w] }

        return (patches, imageOri, spatialCrop)
    }

    // MARK: - VLMModel surface

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        // Working float dtype for the pixel buffer. Must NOT be taken from the
        // text embedding table: under quantization that weight is uint32 (a
        // QuantizedEmbedding), and casting normalized [-1,1] pixels to uint32
        // truncates every value to 0 (the whole global view became zero, which
        // tripped the `imageOri.sum()==0` guard and silently dropped image
        // injection). `image_newline` is a plain learned parameter in the
        // embedding space (bf16), so it gives the correct float dtype.
        let dtype = imageNewline.dtype

        let inputIds = input.text.tokens
        let seqMask = input.text.mask

        let embeds: MLXArray
        if let image = input.image,
            let (patches, imageOri, spatialCrop) = unpackPixels(image, dtype: dtype)
        {
            embeds = inputEmbeddings(
                inputIds: inputIds,
                patches: patches,
                imageOri: imageOri,
                spatialCrop: spatialCrop,
                seqMask: seqMask)
        } else {
            embeds = languageModel.model.embedTokens(inputIds)
        }

        let logits = languageModel(inputIds, cache: cache, inputsEmbeds: embeds)
        return .logits(.init(logits: logits))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Step 1: normalize HF prefixes (deepseekocr.py transform_key), applied
        // first so component-specific transposes see the final key layout.
        var transformed = [String: MLXArray]()
        for (key, value) in weights {
            var k = key

            if k.contains("model.layers"), !k.contains("language_model") {
                k = k.replacingOccurrences(
                    of: "model.layers", with: "language_model.model.layers")
            }
            if k.contains("model.embed_tokens"), !k.contains("language_model") {
                k = k.replacingOccurrences(
                    of: "model.embed_tokens", with: "language_model.model.embed_tokens")
            }
            if k.contains("model.norm"), !k.contains("language_model") {
                k = k.replacingOccurrences(of: "model.norm", with: "language_model.model.norm")
            }
            if k.contains("model.vision_model") {
                k = k.replacingOccurrences(of: "model.vision_model", with: "vision_model")
            }
            if k.contains("model.sam_model") {
                k = k.replacingOccurrences(of: "model.sam_model", with: "sam_model")
            }
            if k.contains("model.projector") {
                k = k.replacingOccurrences(of: "model.projector", with: "projector")
            }
            // Note the upstream typo "view_seperator" → our "view_separator".
            if k.contains("model.view_seperator") {
                k = k.replacingOccurrences(of: "model.view_seperator", with: "view_separator")
            }
            if k.contains("model.image_newline") {
                k = k.replacingOccurrences(of: "model.image_newline", with: "image_newline")
            }
            if k.contains("lm_head.weight"), !k.contains("language_model") {
                k = k.replacingOccurrences(
                    of: "lm_head.weight", with: "language_model.lm_head.weight")
            }

            transformed[k] = value
        }

        // Step 2: SAM conv-weight transposes (NCHW -> NHWC) on the sam_model.* slice.
        transformed = DeepseekOCRSAMEncoder.sanitize(weights: transformed, prefix: "sam_model.")

        // Step 3: stack per-expert MoE weights into the SwitchGLU switch_mlp layout
        // (operates on the language_model.model.* keys produced above).
        transformed = languageModel.sanitize(weights: transformed)

        return transformed
    }
}
