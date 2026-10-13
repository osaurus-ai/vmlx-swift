// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// MiniCPM-V 4.7 (`minicpmv4_7`): a Qwen3.5 text model (dense 1B `qwen3_5_text`, MoE 35B-A3B `qwen3_5_moe_text`) plus
// MiniCPM's own SigLIP-style vision tower and resampler. The text model is built by `Qwen35` / `Qwen35MoE` (see the
// `minicpmv4_7` entry in VLMModelFactory). The tower is not ported yet, so this processor serves TEXT and refuses
// images and video explicitly rather than dropping them.

import Foundation
import MLX
import MLXLMCommon

public struct MiniCPMV47ProcessorConfiguration: Codable, Sendable {
    public var maxSliceNums: Int?
    public var mropeMode: String?

    enum CodingKeys: String, CodingKey {
        case maxSliceNums = "max_slice_nums"
        case mropeMode = "mrope_mode"
    }
}

public struct MiniCPMV47Processor: UserInputProcessor {
    private let config: MiniCPMV47ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: MiniCPMV47ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.images.isEmpty, input.videos.isEmpty else {
            throw VLMError.processing(
                "MiniCPM-V 4.7 image and video input is not supported yet in this runtime (text only).")
        }
        let messages = Qwen3VLMessageGenerator().generate(from: input)
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )
        let cacheBoundaries = canonicalChatCacheBoundaries(
            tokenizer: tokenizer, messages: messages, tools: input.tools,
            additionalContext: input.additionalContext, promptTokens: promptTokens,
            staticSystemPrefix: input.cacheStableSystemPrefix)
        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask, tokenIds: promptTokens),
            cacheScopeSalt: cacheScopeSalt(from: input.additionalContext),
            cachePrefixTokenCounts: cacheBoundaries.all,
            cacheStablePrefixTokenCounts: cacheBoundaries.stable,
            toolSchemas: input.tools)
    }
}
