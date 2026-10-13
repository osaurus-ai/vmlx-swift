// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// MiniCPM-V 4.7 (`minicpmv4_7`): a Qwen3.5 text model (dense 1B `qwen3_5_text`, MoE 35B-A3B `qwen3_5_moe_text`) plus
// MiniCPM's own SigLIP-style vision tower, window merger and resampler (`MiniCPMV47Vision`). The model is built by
// `Qwen35` / `Qwen35MoE` carrying `MiniCPMV47Vision.Tower` in its `vision_tower` slot (see the `minicpmv4_7` entry
// in VLMModelFactory); canvas M-RoPE positions come from `MiniCPMV47Canvas`.
//
// This processor is the port of transformers' `MiniCPMV4_7Processor` + `MiniCPMV4_6ImageProcessorPil`: the chat
// template renders one `<|image_pad|>` per image, which is expanded to
// `<image_id>N</image_id><image>pad*thumb</image>` plus rows of `<slice>pad*k</slice>` separated by `\n`, while the
// pixels are the thumbnail and slices as NaViT patches (`MiniCPMV47ImageProcessing`).
//
// Video is refused explicitly: the reference video processor samples frames and stacks several of them onto one
// canvas image with separator lines (`concat_frames_as_image`) before the same slicing — a different input path
// that is not implemented here.

import CoreImage
import Foundation
import MLX
import MLXLMCommon

public struct MiniCPMV47ProcessorConfiguration: Codable, Sendable {
    public var maxSliceNums: Int = 9
    public var scaleResolution: Int = 448
    public var patchSize: Int = 14
    public var sliceMode: Bool = true
    public var useImageId: Bool = true
    public var downsampleMode: String = "16x"
    public var imageMean: [Float] = [0.5, 0.5, 0.5]
    public var imageStd: [Float] = [0.5, 0.5, 0.5]
    public var mropeMode: String?

    /// Patch-to-LLM-token merge factor per axis (4 in `16x`, 2 in `4x`); placeholder counts use its square.
    public var tokensPerPatchDivisor: Int { downsampleMode == "4x" ? 4 : 16 }

    enum CodingKeys: String, CodingKey {
        case maxSliceNums = "max_slice_nums"
        case scaleResolution = "scale_resolution"
        case patchSize = "patch_size"
        case sliceMode = "slice_mode"
        case useImageId = "use_image_id"
        case downsampleMode = "downsample_mode"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case mropeMode = "mrope_mode"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        maxSliceNums = try c.decodeIfPresent(Int.self, forKey: .maxSliceNums) ?? 9
        scaleResolution = try c.decodeIfPresent(Int.self, forKey: .scaleResolution) ?? 448
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        sliceMode = try c.decodeIfPresent(Bool.self, forKey: .sliceMode) ?? true
        useImageId = try c.decodeIfPresent(Bool.self, forKey: .useImageId) ?? true
        downsampleMode = try c.decodeIfPresent(String.self, forKey: .downsampleMode) ?? "16x"
        imageMean = try c.decodeIfPresent([Float].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd = try c.decodeIfPresent([Float].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        mropeMode = try c.decodeIfPresent(String.self, forKey: .mropeMode)
    }
}

public struct MiniCPMV47Processor: UserInputProcessor {
    private let config: MiniCPMV47ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public static let imagePadToken = "<|image_pad|>"
    public static let videoPadToken = "<|video_pad|>"

    public init(_ config: MiniCPMV47ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    /// Marker ids come from the tokenizer, the same place the conversion script resolved the config's
    /// `image_start_id` etc. from.
    struct Markers {
        let pad: Int
        let imageStart: Int
        let imageEnd: Int
        let sliceStart: Int
        let sliceEnd: Int
        let newline: Int
        let imageIdStart: Int?
        let imageIdEnd: Int?

        var canvas: (imageStart: Int, imageEnd: Int, sliceStart: Int, sliceEnd: Int, newline: Int) {
            (imageStart, imageEnd, sliceStart, sliceEnd, newline)
        }
    }

    func markers() throws -> Markers {
        func id(_ token: String) throws -> Int {
            guard let id = tokenizer.convertTokenToId(token) else {
                throw VLMError.processing("MiniCPM-V 4.7 tokenizer has no \(token) token")
            }
            return id
        }
        // `\n` is the byte-level BPE newline (`Ċ`), not a special token: resolve it by encoding.
        let newline = tokenizer.encode(text: "\n", addSpecialTokens: false)
        guard newline.count == 1 else {
            throw VLMError.processing("MiniCPM-V 4.7 tokenizer does not encode a newline as one token")
        }
        return Markers(
            pad: try id(Self.imagePadToken), imageStart: try id("<image>"), imageEnd: try id("</image>"),
            sliceStart: try id("<slice>"), sliceEnd: try id("</slice>"), newline: newline[0],
            imageIdStart: tokenizer.convertTokenToId("<image_id>"),
            imageIdEnd: tokenizer.convertTokenToId("</image_id>"))
    }

    /// Thumbnail + slices for one image, as NaViT patches and patch grids.
    func preprocess(image: CIImage, processing: UserInput.Processing?) throws -> MiniCPMV47ImageProcessing.ProcessedCrops {
        let applied = MediaProcessing.apply(image, processing: processing)
        let rgb = try MiniCPMV47ImageProcessing.rgb8(from: applied)
        return MiniCPMV47ImageProcessing.process(
            rgb, maxSliceNums: config.maxSliceNums, scaleResolution: config.scaleResolution,
            patchSize: config.patchSize, sliceMode: config.sliceMode, mean: config.imageMean, std: config.imageStd)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty else {
            throw VLMError.processing(
                "MiniCPM-V 4.7 video input is not implemented in this runtime: the reference stacks sampled frames "
                + "onto one canvas image with separator lines before slicing, which this processor does not do. "
                + "Send images or text.")
        }
        let messages = Qwen3VLMessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )
        var cacheBoundaries = canonicalChatCacheBoundaries(
            tokenizer: tokenizer, messages: messages, tools: input.tools,
            additionalContext: input.additionalContext, promptTokens: promptTokens,
            staticSystemPrefix: input.cacheStableSystemPrefix)

        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let markers = try markers()
            let crops = try input.images.map { try preprocess(image: try $0.asCIImage(), processing: input.processing) }
            let replacements = crops.enumerated().map { index, crop in
                MiniCPMV47ImageProcessing.placeholderTokens(
                    crops: crop, localIndex: index, useImageId: config.useImageId, padTokenId: markers.pad,
                    tokensPerPatchDivisor: config.tokensPerPatchDivisor, markers: markers.canvas,
                    imageIdStart: markers.imageIdStart, imageIdEnd: markers.imageIdEnd,
                    digitTokens: { tokenizer.encode(text: String($0), addSpecialTokens: false) })
            }
            let expanded = try MiniCPMV47ImageProcessing.expand(
                promptTokens: promptTokens, placeholder: markers.pad, replacements: replacements,
                boundaries: cacheBoundaries)
            promptTokens = expanded.tokens
            cacheBoundaries = expanded.boundaries
            processedImage = .init(
                pixels: concatenated(crops.map { $0.patches }),
                frames: crops.flatMap { $0.grids })
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask, tokenIds: promptTokens),
            image: processedImage,
            mediaTokenIds: processedImage == nil
                ? nil : QwenVL.mediaTokenIds(tokenizer: tokenizer, tokens: [Self.imagePadToken]),
            cacheScopeSalt: cacheScopeSalt(from: input.additionalContext),
            cachePrefixTokenCounts: cacheBoundaries.all,
            cacheStablePrefixTokenCounts: cacheBoundaries.stable,
            toolSchemas: input.tools)
    }
}
