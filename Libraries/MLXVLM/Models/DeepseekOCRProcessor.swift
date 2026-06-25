//
//  DeepseekOCRProcessor.swift
//  mlx-swift-lm
//
//  `UserInputProcessor` for DeepSeek-OCR / Unlimited-OCR (DeepseekOCRForCausalLM,
//  top model_type "deepseek_vl_v2").
//
//  Port of the image-preprocessing + token-sequence logic from
//  https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/deepseekocr
//  (processing_deepseekocr.py + conversation.py). Covers
//  deepseek-ai/DeepSeek-OCR and baidu/Unlimited-OCR (identical arch).
//
//  The DeepEncoder consumes a PAIR of pixel tensors plus two side tensors:
//    - pixel_values:        [patches, image_ori]
//                           patches    = local crop views  (N_crops, 3, image_size, image_size)
//                           image_ori  = global views      (N_images, 3, base_size, base_size)
//    - images_spatial_crop: per-image [width_crop_num, height_crop_num]
//    - images_seq_mask:     bool mask of image-token positions in input_ids
//
//  `LMInput.ProcessedImage` only exposes `pixels: MLXArray` + `frames: [THW]?`,
//  so we pack the four model tensors into those two fields with a fixed layout
//  the model's `prepare(_:cache:windowSize:)` decodes back (see "Packing
//  contract" below). The token COUNT placed in `input_ids` (and the matching
//  `true` run in `images_seq_mask`) is computed to EXACTLY equal the number of
//  feature vectors `get_input_embeddings` emits per image, including the
//  per-row `image_newline` slots and the trailing `view_separator` slot.
//

import CoreImage
import Foundation
import MLX
import MLXLMCommon

// MARK: - Processor

public struct DeepseekOCRProcessor: UserInputProcessor {

    private let config: DeepseekOCRConfiguration
    private let tokenizer: any Tokenizer

    // Defaults matching mlx-vlm processing_deepseekocr.py / processor_config.json.
    // candidate_resolutions[0][0] == base_size (1024).
    private let baseSize: Int          // global view size (1024)
    private let imageSize: Int         // crop tile size (640)
    private let patchSize: Int         // 16
    private let downsampleRatio: Int   // 4
    private let cropping: Bool         // "Gundam"/Unlimited tiling for large images
    private let minNum: Int            // dynamic_preprocess min tiles (2)
    private let maxNum: Int            // dynamic_preprocess max tiles (9)

    // IMAGENET-style normalization for DeepSeek-OCR is mean = std = 0.5 per channel
    // (processor_config.json image_mean / image_std), i.e. pixel*2 - 1.
    private let imageMean: (CGFloat, CGFloat, CGFloat)
    private let imageStd: (CGFloat, CGFloat, CGFloat)

    /// `<image>` token id. Read from config (image_token_index / image_token_id,
    /// default 128815).
    private var imageTokenId: Int { config.imageTokenIndex }

    public init(_ config: DeepseekOCRConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer

        // candidate_resolutions[0][0] is the base/global size (1024 for DeepSeek-OCR).
        self.baseSize = 1024
        self.imageSize = 640
        self.patchSize = 16
        self.downsampleRatio = 4
        self.cropping = true
        self.minNum = 2
        self.maxNum = 9
        self.imageMean = (0.5, 0.5, 0.5)
        self.imageStd = (0.5, 0.5, 0.5)
    }

    // MARK: - Image transform

    /// Port of `ImageTransform.__call__`: scale to [0,1], normalize, return CHW.
    ///
    /// `MediaProcessing.asMLXArray` already renders to `[1, C, H, W]` Float32 in
    /// the [0,1] range and `.normalized(mean:std:)` applies `(x - mean) / std`
    /// in that same range (CIImage tone-curve pixels are 0…1) — matching the
    /// Python `mx.array(np.array(img)) / 255.0` then `(img - mean) / std`.
    private func imageTransform(_ image: CIImage) -> MLXArray {
        image
            .toSRGB()
            .normalized(mean: imageMean, std: imageStd)
            .asMLXArray()  // [1, 3, H, W]
            .squeezed(axis: 0)  // [3, H, W]
    }

    /// Background CIColor used by `ImageOps.pad` fill = mean * 255. With mean
    /// 0.5 this is mid-grey; in normalized space it maps to ~0.
    private var padColor: CIColor {
        CIColor(red: imageMean.0, green: imageMean.1, blue: imageMean.2)
    }

    // MARK: - Resolution / tiling math (port of dynamic_preprocess)

    /// Port of `find_closest_aspect_ratio`.
    private func findClosestAspectRatio(
        aspectRatio: Double, targetRatios: [(Int, Int)], width: Int, height: Int,
        imageSize: Int
    ) -> (Int, Int) {
        var bestRatioDiff = Double.greatestFiniteMagnitude
        var bestRatio = (1, 1)
        let area = Double(width * height)
        for ratio in targetRatios {
            let targetAspectRatio = Double(ratio.0) / Double(ratio.1)
            let ratioDiff = abs(aspectRatio - targetAspectRatio)
            if ratioDiff < bestRatioDiff {
                bestRatioDiff = ratioDiff
                bestRatio = ratio
            } else if ratioDiff == bestRatioDiff {
                if area > 0.5 * Double(imageSize) * Double(imageSize)
                    * Double(ratio.0) * Double(ratio.1)
                {
                    bestRatio = ratio
                }
            }
        }
        return bestRatio
    }

    /// Port of `dynamic_preprocess`: returns the local crop tiles plus the chosen
    /// `(width_crop_num, height_crop_num)` aspect ratio.
    private func dynamicPreprocess(
        image: CIImage, origWidth: Int, origHeight: Int
    ) -> (tiles: [CIImage], cropRatio: (Int, Int)) {
        let aspectRatio = Double(origWidth) / Double(origHeight)

        // target_ratios = { (i,j) : min_num <= i*j <= max_num }, deduped, sorted by area.
        var ratioSet = Set<[Int]>()
        for n in minNum ... maxNum {
            for i in 1 ... n {
                for j in 1 ... n {
                    let prod = i * j
                    if prod <= maxNum && prod >= minNum {
                        ratioSet.insert([i, j])
                    }
                }
            }
        }
        let targetRatios = ratioSet
            .map { ($0[0], $0[1]) }
            .sorted { ($0.0 * $0.1) < ($1.0 * $1.1) }

        let targetAspectRatio = findClosestAspectRatio(
            aspectRatio: aspectRatio, targetRatios: targetRatios,
            width: origWidth, height: origHeight, imageSize: imageSize)

        let targetWidth = imageSize * targetAspectRatio.0
        let targetHeight = imageSize * targetAspectRatio.1
        let blocks = targetAspectRatio.0 * targetAspectRatio.1

        // NOTE: PIL `image.resize((w,h))` is a plain (non-aspect-preserving) resize.
        // CIImage origin is bottom-left whereas PIL is top-left, but the crop box
        // math below mirrors PIL's left→right / top→bottom tiling; the model treats
        // tiles as a (height_crop_num × width_crop_num) grid reassembled in
        // get_input_embeddings, so consistent row-major ordering is what matters.
        let resized = resize(image, to: CGSize(width: targetWidth, height: targetHeight))

        let cols = targetWidth / imageSize
        var tiles: [CIImage] = []
        for i in 0 ..< blocks {
            let x = (i % cols) * imageSize
            let y = (i / cols) * imageSize
            // Crop an imageSize×imageSize tile. Flip y to PIL top-left convention.
            let flippedY = targetHeight - y - imageSize
            let cropRect = CGRect(x: x, y: flippedY, width: imageSize, height: imageSize)
            let tile = resized
                .cropped(to: cropRect)
                .transformed(by: CGAffineTransform(translationX: -CGFloat(x),
                                                   y: -CGFloat(flippedY)))
            tiles.append(tile)
        }
        return (tiles, targetAspectRatio)
    }

    /// Plain resize (non aspect-preserving) to an exact pixel size.
    private func resize(_ image: CIImage, to size: CGSize) -> CIImage {
        let extent = image.extent
        let sx = size.width / extent.width
        let sy = size.height / extent.height
        return image
            .transformed(by: CGAffineTransform(scaleX: sx, y: sy))
            .transformed(by: CGAffineTransform(
                translationX: -image.extent.origin.x * sx,
                y: -image.extent.origin.y * sy))
    }

    /// Aspect-preserving pad to a square `side`, mean-colored background.
    /// Port of `ImageOps.pad(image, (side, side), color=mean*255)`.
    private func padToSquare(_ image: CIImage, side: Int) -> MLXArray {
        let extent = image.extent
        let scale = min(
            CGFloat(side) / extent.width, CGFloat(side) / extent.height)
        let scaled = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let sExtent = scaled.extent
        let tx = (CGFloat(side) - sExtent.width) * 0.5 - sExtent.origin.x
        let ty = (CGFloat(side) - sExtent.height) * 0.5 - sExtent.origin.y
        let centered = scaled.transformed(by: CGAffineTransform(translationX: tx, y: ty))

        let background = CIImage(color: padColor)
            .cropped(to: CGRect(x: 0, y: 0, width: side, height: side))
        let composited = centered.composited(over: background)
            .cropped(to: CGRect(x: 0, y: 0, width: side, height: side))
        return imageTransform(composited)
    }

    // MARK: - Token sequence (port of tokenize_with_images)

    /// Per-image expansion result: token sequence + image-seq mask run.
    private struct ImageTokens {
        var tokens: [Int]
        var globalView: MLXArray      // [3, base, base]
        var cropViews: [MLXArray]     // [3, image, image] each (may be empty)
        var spatialCrop: (Int, Int)   // (width_crop_num, height_crop_num)
    }

    /// Build the image-token block for one image, matching the embedding count
    /// `get_input_embeddings` emits.
    ///
    /// num_queries      = ceil((image_size / patch_size) / downsample_ratio)  (=10)
    /// num_queries_base = ceil((base_size  / patch_size) / downsample_ratio)  (=16)
    ///
    /// Global view tokens (always):
    ///   ([img] * num_queries_base + [img]) * num_queries_base + [img]
    ///   = (base+1)*base + 1   slots
    ///   → per row: `base` feature columns + 1 image_newline; `base` rows;
    ///     + 1 trailing view_separator.
    ///
    /// Local crop tokens (only when width_crop_num > 1 OR height_crop_num > 1):
    ///   ([img] * (num_queries * width_crop_num) + [img]) * (num_queries * height_crop_num)
    ///   = (q*W + 1) * (q*H)   slots
    ///   → per local row: q*W feature columns + 1 image_newline; q*H rows.
    ///   (The view_separator is shared/global — emitted once via the +[img]
    ///    already accounted in the global block's trailing element.)
    private func processImage(_ image: CIImage) -> ImageTokens {
        let extent = image.extent
        let w = Int(extent.width.rounded())
        let h = Int(extent.height.rounded())

        var cropRatio = (1, 1)
        var cropTiles: [CIImage] = []

        if cropping {
            if w <= 640 && h <= 640 {
                cropRatio = (1, 1)
            } else {
                let (tiles, ratio) = dynamicPreprocess(
                    image: image, origWidth: w, origHeight: h)
                cropTiles = tiles
                cropRatio = ratio
            }
        }

        // Global view: aspect-preserving pad to base_size, normalized CHW.
        let globalView = padToSquare(image, side: baseSize)

        let widthCropNum = cropRatio.0
        let heightCropNum = cropRatio.1
        let hasCrops = widthCropNum > 1 || heightCropNum > 1

        let numQueries = Int(
            (Double(imageSize / patchSize) / Double(downsampleRatio)).rounded(.up))
        let numQueriesBase = Int(
            (Double(baseSize / patchSize) / Double(downsampleRatio)).rounded(.up))

        var tokens: [Int] = []
        // Global block: ([img]*numQueriesBase + [img]) * numQueriesBase + [img]
        let globalRow = Array(repeating: imageTokenId, count: numQueriesBase) + [imageTokenId]
        for _ in 0 ..< numQueriesBase { tokens += globalRow }
        tokens += [imageTokenId]

        // Local block (only with crops).
        var cropViews: [MLXArray] = []
        if hasCrops {
            let localRow =
                Array(repeating: imageTokenId, count: numQueries * widthCropNum)
                + [imageTokenId]
            let localRows = numQueries * heightCropNum
            for _ in 0 ..< localRows { tokens += localRow }
            cropViews = cropTiles.map { imageTransform($0) }
        }

        return ImageTokens(
            tokens: tokens, globalView: globalView, cropViews: cropViews,
            spatialCrop: (widthCropNum, heightCropNum))
    }

    // MARK: - Prompt formatting (port of conversation.py "deepseek" template)

    /// Extract the user text and image count from the structured input. DeepSeek-OCR
    /// has no useful HF chat template (the bundled one is a near-no-op), so we build
    /// the DeepSeek conversation prompt directly:
    ///
    ///   <|User|>: <image>\n{user text}\n\n<|Assistant|>:
    ///
    /// with one `<image>` marker per supplied image, and a leading BOS (id 0).
    private func renderPrompt(text: String, imageCount: Int) -> String {
        let imageMarkers = String(repeating: "<image>\n", count: imageCount)
        return "<|User|>: \(imageMarkers)\(text)\n\n<|Assistant|>:"
    }

    private func userText(from input: UserInput) -> String {
        switch input.prompt {
        case .text(let t):
            return t
        case .messages(let messages):
            // Concatenate user-role text content.
            return messages.compactMap { msg -> String? in
                guard let role = msg["role"] as? String, role == "user" else { return nil }
                if let content = msg["content"] as? String { return content }
                if let parts = msg["content"] as? [[String: Any]] {
                    return parts.compactMap { $0["text"] as? String }.joined()
                }
                return nil
            }.joined(separator: "\n")
        case .chat(let chatMessages):
            return chatMessages
                .filter { $0.role == .user }
                .map(\.content)
                .joined(separator: "\n")
        }
    }

    // MARK: - prepare

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = try input.images.map { try $0.asCIImage() }
        let text = userText(from: input)

        // Text-only fast path.
        if images.isEmpty {
            let prompt = renderPrompt(text: text, imageCount: 0)
            // tokenize_with_images uses add_special_tokens=False then prepends
            // a hardcoded bos_id = 0.
            var tokens = tokenizer.encode(text: prompt, addSpecialTokens: false)
            tokens.insert(0, at: 0)  // BOS id 0
            return LMInput(tokens: MLXArray(tokens), tokenIds: tokens)
        }

        // Build the prompt, splitting on the <image> marker exactly like
        // tokenize_with_images: text-around tokens get mask=false, each image
        // expands into its image-token block (mask=true).
        let prompt = renderPrompt(text: text, imageCount: images.count)
        let textSplits = prompt.components(separatedBy: "<image>")
        precondition(
            textSplits.count == images.count + 1,
            "image-marker count must match image count")

        var tokenizedStr: [Int] = []
        var imagesSeqMask: [Bool] = []

        var globalViews: [MLXArray] = []
        var cropViews: [MLXArray] = []
        var spatialCrops: [[Int]] = []

        for (i, image) in images.enumerated() {
            // Text separator before this image.
            let sep = tokenizer.encode(text: textSplits[i], addSpecialTokens: false)
            tokenizedStr += sep
            imagesSeqMask += Array(repeating: false, count: sep.count)

            // Image token block + pixel tensors.
            let processed = processImage(image)
            tokenizedStr += processed.tokens
            imagesSeqMask += Array(repeating: true, count: processed.tokens.count)

            globalViews.append(processed.globalView)
            cropViews += processed.cropViews
            spatialCrops.append([processed.spatialCrop.0, processed.spatialCrop.1])
        }

        // Trailing text after the last image.
        let tail = tokenizer.encode(text: textSplits.last!, addSpecialTokens: false)
        tokenizedStr += tail
        imagesSeqMask += Array(repeating: false, count: tail.count)

        // Leading BOS (id 0), matching tokenize_with_images.
        tokenizedStr.insert(0, at: 0)
        imagesSeqMask.insert(false, at: 0)

        precondition(
            tokenizedStr.count == imagesSeqMask.count,
            "tokenized_str length must equal images_seq_mask length")

        // Stack pixel tensors.
        // image_ori: [N_images, 3, base, base]  (global views, always present)
        let imageOri = MLX.stacked(globalViews, axis: 0)
        // patches: [N_crops, 3, image, image]   (local crop views, may be empty)
        let nCrops = cropViews.count
        let patches: MLXArray? = nCrops == 0 ? nil : MLX.stacked(cropViews, axis: 0)

        // Packing contract for LMInput.ProcessedImage (model decodes this in
        // prepare(_:cache:windowSize:)):
        //
        //   pixels = image_ori.flatten() [++ patches.flatten() when N_crops > 0]
        //
        //   frames[0]    = THW(t: N_crops, h: N_images, w: 0)
        //                  → counts used to slice `pixels`:
        //                    image_ori = first  N_images * 3 * base * base elements,
        //                                reshaped [N_images, 3, base, base];
        //                    patches   = remaining N_crops * 3 * image * image elements,
        //                                reshaped [N_crops, 3, image, image]
        //                                (empty when N_crops == 0 → model uses the
        //                                 zeros sentinel get_input_embeddings expects).
        //   frames[1...]  = one THW(1, width_crop_num, height_crop_num) per image
        //                  → this IS images_spatial_crop.
        //
        //   images_seq_mask is reconstructed model-side from input_ids == imageTokenId
        //   (carried via mediaTokenIds), and is also provided here as text.mask.
        //
        // NOTE: image_ori (base_size square) and patches (image_size square) have
        // different spatial dims, so they cannot form one rectangular MLXArray;
        // hence the flatten-and-concat layout with shapes recovered from frames[0]
        // plus the model's known base_size / image_size.
        var frames: [THW] = [THW(nCrops, images.count, 0)]
        for crop in spatialCrops {
            frames.append(THW(1, crop[0], crop[1]))
        }

        let pixels = packPixels(imageOri: imageOri, patches: patches)
        let processedImage = LMInput.ProcessedImage(pixels: pixels, frames: frames)

        let promptArray = MLXArray(tokenizedStr).expandedDimensions(axis: 0)
        let maskArray = MLXArray(imagesSeqMask.map { $0 ? Int32(1) : Int32(0) })
            .expandedDimensions(axis: 0)

        return LMInput(
            text: .init(tokens: promptArray, mask: maskArray, tokenIds: tokenizedStr),
            image: processedImage,
            mediaTokenIds: [imageTokenId])
    }

    /// Flatten image_ori and (optional) crop patches into a single 1-D pixel
    /// buffer the model unpacks using `frames[0]` (= THW(nCrops, nImages, 0)) and
    /// its known base_size / image_size.
    /// Layout: [ image_ori.flatten() (++ patches.flatten() if present) ].
    private func packPixels(imageOri: MLXArray, patches: MLXArray?) -> MLXArray {
        let oriFlat = imageOri.reshaped([-1])
        guard let patches else { return oriFlat }
        return concatenated([oriFlat, patches.reshaped([-1])], axis: 0)
    }
}
