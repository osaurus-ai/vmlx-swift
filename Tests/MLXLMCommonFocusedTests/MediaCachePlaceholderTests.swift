// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("media cache placeholder policy")
struct MediaCachePlaceholderTests {
    @Test("known media tokens allow text-only cache-hit suffixes")
    func knownMediaTokensAllowTextOnlySuffixes() {
        FocusedMLXTestSupport.withLock {
        let waveform = MLXArray([Float](repeating: 0.0, count: 8)).reshaped([1, 8])
        MLX.eval(waveform)

        let unknown = LMInput(
            text: .init(tokens: MLXArray([Int32(1), 27, 2])),
            audio: .init(waveform: waveform))
        #expect(!unknown.cacheHitSuffixContainsMediaPlaceholder([]))
        #expect(unknown.cacheHitSuffixContainsMediaPlaceholder([1, 2]))

        let known = LMInput(
            text: .init(tokens: MLXArray([Int32(1), 27, 2])),
            audio: .init(waveform: waveform),
            mediaTokenIds: [27])
        #expect(known.cacheHitSuffixContainsMediaPlaceholder([27, 2]))
        #expect(!known.cacheHitSuffixContainsMediaPlaceholder([2, 3]))
        #expect(!LMInput(
            text: .init(tokens: MLXArray([Int32(1)])),
            mediaTokenIds: [27]
        ).cacheHitSuffixContainsMediaPlaceholder([27]))
        }
    }

    @Test("video EVS inputs require post-prepare cache keys")
    func videoEVSInputsRequirePostPrepareCacheKeys() {
        FocusedMLXTestSupport.withLock {
        let videoPixels = MLXArray([Float](repeating: 0.0, count: 3 * 4 * 4))
            .reshaped([1, 3, 4, 4])
        MLX.eval(videoPixels)

        let plainVideo = LMInput(
            text: .init(tokens: MLXArray([Int32(1), 27, 2])),
            video: .init(pixels: videoPixels))
        #expect(!plainVideo.requiresPostPrepareCacheKey)

        let evsVideo = LMInput(
            text: .init(tokens: MLXArray([Int32(1), 27, 2])),
            video: .init(pixels: videoPixels, embeddingTokenCount: 1))
        #expect(evsVideo.requiresPostPrepareCacheKey)
        }
    }

    @Test("post-prepare cache-key paths use effective prompt tokens")
    func postPrepareCacheKeyPathsUseEffectivePromptTokens() throws {
        let scheduler = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchScheduler.swift",
            encoding: .utf8)
        let batch = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)
        let evaluate = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)
        let nativeMTP = try String(
            contentsOfFile: "Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift",
            encoding: .utf8)

        #expect(scheduler.contains("var cachePromptTokenIds: [Int]"))
        #expect(batch.contains("slot.cachePromptTokenIds = effectivePromptTokens"))
        #expect(batch.contains("resolvePostPrepareCacheKeyAlias"))
        #expect(batch.contains("recordPostPrepareCacheKeyAlias"))
        #expect(batch.contains("let promptTokens = slot.cachePromptTokenIds"))
        #expect(batch.contains("!slot.originalInput.requiresPostPrepareCacheKey"))
        #expect(evaluate.contains("promptTokenIds = effectivePromptTokens"))
        #expect(evaluate.contains("resolvePostPrepareCacheKeyAlias"))
        #expect(evaluate.contains("recordPostPrepareCacheKeyAlias"))
        #expect(evaluate.contains("!input.requiresPostPrepareCacheKey"))
        #expect(evaluate.contains("!originalInput.requiresPostPrepareCacheKey"))
        #expect(nativeMTP.contains("var promptTokenIds: [Int]"))
        #expect(nativeMTP.contains("self.promptTokenIds = effectivePromptTokens"))
        #expect(nativeMTP.contains("resolvePostPrepareCacheKeyAlias"))
        #expect(nativeMTP.contains("recordPostPrepareCacheKeyAlias"))
        #expect(nativeMTP.contains("!input.requiresPostPrepareCacheKey"))
        #expect(nativeMTP.contains("!originalInput.requiresPostPrepareCacheKey"))
    }

    @Test("OmniBench carries live repeated-video post-prepare alias proof")
    func omniBenchCarriesRepeatedVideoAliasProof() throws {
        let omniBench = try String(
            contentsOfFile: "RunBench/OmniBench.swift",
            encoding: .utf8)

        #expect(omniBench.contains("5d. video repeated cache alias"))
        #expect(omniBench.contains("BENCH_OMNI_VIDEO_CACHE_ALIAS"))
        #expect(omniBench.contains("resolvePostPrepareCacheKeyAlias"))
        #expect(omniBench.contains("same-video post-prepare alias"))
    }
}
