import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen2_vl

// MARK: - Common Utilities for Qwen 2 VL and Qwen 2.5 VL

public struct QwenVL {
    /// Rotates half the hidden dims of the input
    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let index = x.dim(-1) / 2
        let x1 = x[.ellipsis, 0 ..< index]
        let x2 = x[.ellipsis, index...]
        return concatenated([-x2, x1], axis: -1)
    }

    static func mergeInputIdsWithImageFeatures(
        inputIds: MLXArray, inputEmbeds: MLXArray, imageFeatures: MLXArray,
        imageTokenId: Int, videoTokenId: Int
    ) -> MLXArray {
        var imageIndices = [Int]()
        for (i, v) in inputIds.asArray(Int.self).enumerated() {
            if v == imageTokenId || v == videoTokenId {
                imageIndices.append(i)
            }
        }

        // Make sure shapes match before assignment
        var result = inputEmbeds
        if result.ndim == 2 {
            result = result[.newAxis, 0..., 0...]
        }

        if imageFeatures.ndim == 2 {
            let reshapedFeatures = imageFeatures[.newAxis, 0..., 0...]
            result[0..., MLXArray(imageIndices), 0...] = reshapedFeatures
        } else {
            result[0..., MLXArray(imageIndices), 0...] = imageFeatures
        }

        return result
    }

    public class VisionRotaryEmbedding {
        let dimensions: Int
        let theta: Float
        let inverseFreq: MLXArray

        init(dimensions: Int, theta: Float) {
            self.dimensions = dimensions
            self.theta = theta
            let p = MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / dimensions
            self.inverseFreq = 1.0 / pow(theta, p)
        }

        func callAsFunction(sequenceLength: Int) -> MLXArray {
            let seq = MLXArray(0 ..< sequenceLength).asType(inverseFreq.dtype)
            let freqs = outer(seq, inverseFreq)
            return freqs
        }
    }

    public class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo var proj: Conv3d

        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let outputDimensions: Int

        // For Qwen 2 VL
        convenience init(
            patchSize: Int, temporalPatchSize: Int, inChannels: Int, embedDimensions: Int
        ) {
            self.init(
                patchSize: patchSize, temporalPatchSize: temporalPatchSize,
                inChannels: inChannels, outputDimensions: embedDimensions)
        }

        // For Qwen 2.5 VL
        convenience init(patchSize: Int, temporalPatchSize: Int, inChannels: Int, hiddenSize: Int) {
            self.init(
                patchSize: patchSize, temporalPatchSize: temporalPatchSize,
                inChannels: inChannels, outputDimensions: hiddenSize)
        }

        // Common initializer
        init(patchSize: Int, temporalPatchSize: Int, inChannels: Int, outputDimensions: Int) {
            self.patchSize = patchSize
            self.temporalPatchSize = temporalPatchSize
            self.inChannels = inChannels
            self.outputDimensions = outputDimensions

            let kernelSize = IntOrTriple([temporalPatchSize, patchSize, patchSize])
            self._proj.wrappedValue = Conv3d(
                inputChannels: inChannels,
                outputChannels: outputDimensions,
                kernelSize: kernelSize,
                stride: kernelSize,
                bias: false
            )
        }

        public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            var hiddenStates = hiddenStates.reshaped(
                -1, inChannels, temporalPatchSize, patchSize, patchSize
            ).movedAxis(source: 1, destination: 4)

            hiddenStates = proj(hiddenStates)
            hiddenStates = hiddenStates.reshaped(-1, outputDimensions)
            return hiddenStates
        }
    }

    /// Convert a `CGSize` (typically `CIImage.extent.size`) to a finite
    /// positive `(height, width)` integer pair, throwing `VLMError.imageProcessingFailure`
    /// for non-finite or non-positive extents.
    ///
    /// Why this exists: `CIImage(color:)` and other procedurally-generated
    /// images return `extent.size = (.infinity, .infinity)`. The naïve
    /// `Int(extent.height)` then traps with "Double value cannot be
    /// converted to Int because it is either infinite or NaN." Same trap
    /// fires for NaN (e.g. cropped-then-resampled corrupt sources).
    /// Guarding via `Int(exactly:)` lets us throw a clean
    /// `VLMError.imageProcessingFailure` instead of aborting the process.
    static func intExtent(_ size: CGSize) throws -> (height: Int, width: Int) {
        guard size.height.isFinite, size.width.isFinite else {
            throw VLMError.imageProcessingFailure(
                "Image extent is not finite (\(size.width) × \(size.height)); "
                + "likely a procedurally-generated CIImage (e.g. CIImage(color:)) "
                + "without a `cropped(to:)` rect.")
        }
        guard let h = Int(exactly: size.height.rounded()),
              let w = Int(exactly: size.width.rounded()),
              h > 0, w > 0
        else {
            throw VLMError.imageProcessingFailure(
                "Invalid image dimensions \(size.width) × \(size.height); both must be positive finite numbers.")
        }
        return (h, w)
    }

    // image_processing_qwen2_vl.smart_resize
    static func targetSize(height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int)
        throws
        -> (Int, Int)
    {


        if height < factor {
            throw VLMError.imageProcessingFailure(
                "Height: \(height) must be larger than factor: \(factor)")
        }
        if width < factor {
            throw VLMError.imageProcessingFailure(
                "Width: \(width) must be larger than factor: \(factor)")
        }
        if max(height, width) / min(height, width) > 200 {
            throw VLMError.imageProcessingFailure(
                "Absolute aspect ratio must be smaller than 200: \(width) × \(height)")
        }

        var hBar = max(factor, Int(round(Float(height) / Float(factor))) * factor)
        var wBar = max(factor, Int(round(Float(width) / Float(factor))) * factor)

        // Scale based on total pixel count
        if hBar * wBar > maxPixels {
            let beta = sqrt(Float(height * width) / Float(maxPixels))
            hBar = Int(floor(Float(height) / beta / Float(factor))) * factor
            wBar = Int(floor(Float(width) / beta / Float(factor))) * factor
        } else if hBar * wBar < minPixels {
            let beta = sqrt(Float(minPixels) / Float(height * width))
            hBar = Int(ceil(Float(height) * beta / Float(factor))) * factor
            wBar = Int(ceil(Float(width) * beta / Float(factor))) * factor
        }

        // Ensure dimensions are divisible by the factor
        hBar = (hBar / factor) * factor
        wBar = (wBar / factor) * factor

        // Final sanity check
        if hBar <= 0 || wBar <= 0 {
            throw VLMError.imageProcessingFailure(
                "Invalid target dimensions: \(wBar) × \(hBar)")
        }

        return (hBar, wBar)
    }

    static func replacePaddingTokens(
        in promptTokens: [Int], frames: [THW], paddingToken: String, mergeSize: Int,
        tokenizer: any Tokenizer
    ) throws -> [Int] {
        // Replace single padding token with correct number for each image or video frame
        let placeholderRanges = paddingPlaceholderRanges(
            in: promptTokens,
            paddingToken: paddingToken,
            tokenizer: tokenizer)
        guard placeholderRanges.count == frames.count else {
            throw VLMError.processing(
                "Number of placeholder tokens (\(placeholderRanges.count)) does not match number of frames (\(frames.count))")
        }
        let mergeLength = mergeSize * mergeSize
        let replacementSequences = frames.map { frame in
            let paddingCount = frame.product / mergeLength
            return tokenizer.encode(
                text:
                    "<|vision_start|>\(Array(repeating: paddingToken, count: paddingCount).joined())<|vision_end|>",
                addSpecialTokens: false)
        }
        // Build the final array
        var result: [Int] = []
        var currentIndex = promptTokens.startIndex
        for (range, replacement) in zip(placeholderRanges, replacementSequences) {
            // Add tokens before the placeholder
            result.append(contentsOf: promptTokens[currentIndex ..< range.lowerBound])
            // Add replacement sequence
            result.append(contentsOf: replacement)
            currentIndex = range.upperBound
        }
        // Add any remaining tokens after the last replacement
        if currentIndex < promptTokens.endIndex {
            result.append(contentsOf: promptTokens[currentIndex...])
        }
        return result
    }

    /// Placeholder token ids for `LMInput.mediaTokenIds`, resolved from the
    /// tokenizer rather than a model config the processor cannot see.
    ///
    /// Thin wrapper over ``MediaTokenIds/resolve(tokenizer:tokens:)`` — see
    /// there for why a token must round-trip and why this returns `nil` rather
    /// than `[]`.
    static func mediaTokenIds(tokenizer: any Tokenizer, tokens: [String]) -> [Int]? {
        MediaTokenIds.resolve(tokenizer: tokenizer, tokens: tokens)
    }

    static func paddingPlaceholderRanges(
        in promptTokens: [Int],
        paddingToken: String,
        tokenizer: any Tokenizer
    ) -> [Range<Int>] {
        let placeholderTokens = tokenizer.encode(
            text: "<|vision_start|>\(paddingToken)<|vision_end|>",
            addSpecialTokens: false)
        return promptTokens.ranges(of: placeholderTokens)
    }

    static func patchify(images: [MLXArray], mergeSize: Int, patchSize: Int, temporalPatchSize: Int)
        throws -> (
            MLXArray, THW
        )
    {
        guard let firstImage = images.first else {
            throw VLMError.imageProcessingFailure("No images in video sequence")
        }
        let resizedHeight = firstImage.dim(-2)
        let resizedWidth = firstImage.dim(-1)
        var patches = concatenated(images)

        // Pad to match temporal patch size if needed
        let mod = patches.dim(0) % temporalPatchSize
        if mod != 0 {
            let lastPatch = patches[-1, .ellipsis]
            let lastPatchRepeated = tiled(
                lastPatch, repetitions: [temporalPatchSize - mod, 1, 1, 1])
            patches = concatenated([patches, lastPatchRepeated])
        }
        let channel = patches.dim(1)
        let gridT = patches.dim(0) / temporalPatchSize
        let gridH = resizedHeight / patchSize
        let gridW = resizedWidth / patchSize

        patches = patches.reshaped(
            gridT,
            temporalPatchSize,
            channel,
            gridH / mergeSize,
            mergeSize,
            patchSize,
            gridW / mergeSize,
            mergeSize,
            patchSize
        )
        patches = patches.transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)

        let flattenedPatches = patches.reshaped(
            gridT * gridH * gridW,
            channel * temporalPatchSize * patchSize * patchSize
        )

        return (flattenedPatches, .init(gridT, gridH, gridW))
    }

}
