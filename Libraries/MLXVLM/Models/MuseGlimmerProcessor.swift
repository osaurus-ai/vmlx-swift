//
//  MuseGlimmerProcessor.swift
//  vmlx-swift
//
//  Image/video preprocessing and placeholder expansion for Muse Glimmer.
//

import CoreImage
import Foundation
import MLX
import MLXLMCommon

public struct MuseGlimmerProcessorConfiguration: Codable, Sendable {
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let patchSize: Int
    public let mergeSize: Int
    public let temporalPatchSize: Int
    /// Muse Glimmer budgets by *merged tokens* rather than by the pixel range
    /// Qwen uses, so the resize target is derived from this rather than from
    /// min/max pixels.
    public let maxImageTokens: Int
    public let videoFPS: Double

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    /// One merged token covers `mergeSize` patches on each axis.
    public var pixelsPerMergedToken: Int {
        patchSize * mergeSize * patchSize * mergeSize
    }

    enum ImageKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case patchSize = "patch_size"
        case mergeSize = "merge_size"
        case temporalPatchSize = "temporal_patch_size"
        case maxImageTokens = "max_image_tokens"
    }

    enum VideoKeys: String, CodingKey {
        case fps
    }

    enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor"
        case videoProcessor = "video_processor"
    }

    public init(from decoder: Decoder) throws {
        // `processor_config.json` nests the two sections; a flattened config
        // (hand-written, or a converter that hoisted the fields) is accepted
        // too so a bundle without the nesting still loads.
        let outer = try decoder.container(keyedBy: CodingKeys.self)
        let image =
            outer.contains(.imageProcessor)
            ? try outer.nestedContainer(keyedBy: ImageKeys.self, forKey: .imageProcessor)
            : try decoder.container(keyedBy: ImageKeys.self)

        imageMean = try image.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd = try image.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        patchSize = try image.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        mergeSize = try image.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
        temporalPatchSize =
            try image.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
        maxImageTokens = try image.decodeIfPresent(Int.self, forKey: .maxImageTokens) ?? 4096

        if outer.contains(.videoProcessor),
            let video = try? outer.nestedContainer(
                keyedBy: VideoKeys.self, forKey: .videoProcessor)
        {
            videoFPS = try video.decodeIfPresent(Double.self, forKey: .fps) ?? 2.0
        } else {
            videoFPS = 2.0
        }
    }

    public func encode(to encoder: Encoder) throws {
        var outer = encoder.container(keyedBy: CodingKeys.self)
        var image = outer.nestedContainer(keyedBy: ImageKeys.self, forKey: .imageProcessor)
        try image.encode(imageMean, forKey: .imageMean)
        try image.encode(imageStd, forKey: .imageStd)
        try image.encode(patchSize, forKey: .patchSize)
        try image.encode(mergeSize, forKey: .mergeSize)
        try image.encode(temporalPatchSize, forKey: .temporalPatchSize)
        try image.encode(maxImageTokens, forKey: .maxImageTokens)
        var video = outer.nestedContainer(keyedBy: VideoKeys.self, forKey: .videoProcessor)
        try video.encode(videoFPS, forKey: .fps)
    }
}

public struct MuseGlimmerProcessor: UserInputProcessor {
    private let config: MuseGlimmerProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: MuseGlimmerProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    private func preprocess(image: CIImage, resizedSize: CGSize) -> CIImage {
        image
            .toSRGB()
            .resampled(to: resizedSize, method: .bicubic)
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
    }

    /// Resize so the merged-token count fits `max_image_tokens`, keeping both
    /// sides a multiple of `patchSize * mergeSize`.
    ///
    /// `QwenVL.targetSize` speaks in pixels, so the token budget is converted
    /// into a pixel ceiling first. The floor of one merge block per side keeps
    /// a very small or very thin image from resizing to a zero-sized grid,
    /// which would produce an empty patch tensor rather than an error.
    private func targetSize(height: Int, width: Int) throws -> CGSize {
        let factor = config.patchSize * config.mergeSize
        let maxPixels = config.maxImageTokens * config.pixelsPerMergedToken
        let (h, w) = try QwenVL.targetSize(
            height: height, width: width,
            factor: factor,
            minPixels: factor * factor,
            maxPixels: maxPixels)
        return CGSize(width: w, height: h)
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        let images = images.map { MediaProcessing.apply($0, processing: processing) }
        let (extentH, extentW) = try QwenVL.intExtent(images[0].extent.size)
        let resizedSize = try targetSize(height: extentH, width: extentW)

        let processed = images.map { preprocess(image: $0, resizedSize: resizedSize).asMLXArray() }
        return try QwenVL.patchify(
            images: processed, mergeSize: config.mergeSize, patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = DefaultMessageGenerator().generate(from: input)

        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)

        if input.images.isEmpty, input.videos.isEmpty {
            return LMInput(
                tokens: MLXArray(promptTokens),
                tokenIds: promptTokens,
                cacheScopeSalt: cacheScopeSalt(from: input.additionalContext))
        }

        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let pixelsAndFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            processedImage = LMInput.ProcessedImage(
                pixels: concatenated(pixelsAndFrames.map { $0.0 }),
                frames: pixelsAndFrames.map { $0.1 })
            if let frames = processedImage?.frames {
                // The template emits a single `<|patch|>`; it has to become one
                // placeholder per merged token or the scatter and the feature
                // count disagree.
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens, frames: frames, paddingToken: "<|patch|>",
                    mergeSize: config.mergeSize, tokenizer: tokenizer)
            }
        }

        var processedVideo: LMInput.ProcessedVideo?
        if !input.videos.isEmpty {
            var sequences = [[MLXArray]]()
            for video in input.videos {
                var resizedSize: CGSize = .zero
                let sequence = try await MediaProcessing.asProcessedSequence(
                    video, targetFPS: { _ in config.videoFPS }
                ) { frame in
                    let resized = MediaProcessing.apply(frame.frame, processing: input.processing)
                    if resizedSize == .zero {
                        let (h, w) = try QwenVL.intExtent(resized.extent.size)
                        resizedSize = try targetSize(height: h, width: w)
                    }
                    return VideoFrame(
                        frame: preprocess(image: resized, resizedSize: resizedSize),
                        timeStamp: frame.timeStamp)
                }
                sequences.append(sequence.frames)
            }
            let pixelsAndFrames = try sequences.map {
                try QwenVL.patchify(
                    images: $0, mergeSize: config.mergeSize, patchSize: config.patchSize,
                    temporalPatchSize: config.temporalPatchSize)
            }
            processedVideo = LMInput.ProcessedVideo(
                pixels: concatenated(pixelsAndFrames.map { $0.0 }),
                frames: pixelsAndFrames.map { $0.1 })
            if let frames = processedVideo?.frames {
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens, frames: frames, paddingToken: "<|video|>",
                    mergeSize: config.mergeSize, tokenizer: tokenizer)
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask, tokenIds: promptTokens),
            image: processedImage,
            video: processedVideo,
            cacheScopeSalt: cacheScopeSalt(from: input.additionalContext))
    }
}
