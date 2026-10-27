// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 prefill: engram next-chunk lookahead (reference `DSV41._start_lookahead`)
// and the candidate-block scatter. CPU only; the engram case reads a few thousand table
// rows from the real bundle when it is present (no model weights are loaded).

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("DeepSeek-V4.1 prefill lookahead and candidate scatter (CPU)")
struct DeepseekV41PrefillLookaheadFocusedTests {
    @Test(
        "background (lookahead) reads return the same rows as the forward-time reads",
        .enabled(if: DSV41TestBundle.item("engram") != nil, "needs VMLX_DSV41_BUNDLE"))
    func lookaheadRowsAreIdentical() throws {
        let bundle = try #require(DSV41TestBundle.directory)
        try Device.withDefaultDevice(.cpu) {
            let config = try JSONDecoder().decode(
                DeepseekV4Configuration.self,
                from: Data(contentsOf: bundle.appendingPathComponent("config.json")))
            let runtime = try EngramRuntime(modelDirectory: bundle, config: config)
            var rng = SplitMix64(seed: 41)
            let first = (0 ..< 300).map { _ in Int32(rng.next() % 120_000) }
            let next = (0 ..< 257).map { _ in Int32(rng.next() % 120_000) }

            // Chunk k, then the history its forward leaves for chunk k+1.
            let comp0 = runtime.hasher.compress(first)
            _ = runtime.prepare(comp: comp0, startPos: 0, previous: [])
            let reach = runtime.hasher.historyReach
            let previous = Array(comp0.suffix(reach))
            let comp1 = runtime.hasher.compress(next)

            let direct = runtime.prepare(comp: comp1, startPos: comp0.count, previous: previous)
            let ahead = EngramRuntime.Lookahead(
                startPos: comp0.count, comp: comp1, previous: previous,
                chunk: runtime.prepare(
                    comp: comp1, startPos: comp0.count, previous: previous, background: true))
            #expect(ahead.matches(startPos: comp0.count, comp: comp1, previous: previous))
            #expect(!ahead.matches(startPos: comp0.count + 1, comp: comp1, previous: previous))
            #expect(!ahead.matches(startPos: comp0.count, comp: Array(comp1.dropLast()), previous: previous))
            #expect(!ahead.matches(startPos: comp0.count, comp: comp1, previous: Array(previous.dropFirst())))
            for layer in config.engramLayerIds {
                let a = try #require(direct.rows(layer: layer))
                let b = try #require(ahead.chunk.rows(layer: layer))
                #expect(a.shape == [1, next.count, runtime.aux.cols, config.engramHeadDim])
                #expect(MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self) == 0)
            }
        }
    }

    @Test("a cache trim drops any pending engram lookahead")
    func trimDropsLookahead() {
        let cache = DeepseekV4Cache(
            slidingWindow: 8, compressRatio: 2, requiresAuxiliaryTokenHistory: true,
            poolQuantizationBits: 4, contiguousPools: true, compactLocalWindow: true)
        cache.engramLookaheadTokens = [1, 2, 3]
        _ = cache.trim(1)
        #expect(cache.engramLookaheadTokens == nil)
        #expect(cache.engramLookahead == nil)
        let copy = cache.copy() as! DeepseekV4Cache
        #expect(copy.engramLookaheadTokens == nil)
    }

    @Test("candidate scatter matches put_along_axis at long-context block counts")
    func candidateScatterAtScale() {
        Device.withDefaultDevice(.cpu) {
            // nb = 6000 blocks > topkBlocks = 2048: the scatter branch (was a [C, k, nb] one-hot).
            let C = 3, block = 8, T = 6000 * block
            let logits = MLXRandom.normal([C, T], key: MLXRandom.key(5))
            let lens = MLXArray([Int32(T), Int32(T - 77), Int32(9_000)], [C, 1])
            let masked = DeepseekV41IndexerMath.maskInvisible(logits, lens: lens)
            let keep = DeepseekV41IndexerMath.candidateBlocks(
                masked, lens: lens, topkBlocks: 2048, blockSize: block)
            #expect(keep.shape == [C, T])
            let perRow = keep.asType(.int32).sum(axis: -1).asArray(Int32.self)
            // Row 0/1: exactly 2048 blocks kept (the forced last block is one of them).
            #expect(perRow[0] == 2048 * Int32(block))
            #expect(perRow[1] <= 2048 * Int32(block) && perRow[1] > 2047 * Int32(block))
            // Row 2 sees 9000 rows = 1125 blocks < 2048: every visible block is kept.
            #expect(perRow[2] == 9_000)
            // The kept blocks of row 0 are its 2048 best block maxima.
            let blockMax = masked[0].reshaped(6000, block).max(axis: -1).asArray(Float.self)
            let threshold = blockMax.sorted(by: >)[2047]
            let kept = keep[0].reshaped(6000, block)[0..., 0].asArray(Bool.self)
            for b in 0 ..< 6000 where blockMax[b] > threshold { #expect(kept[b]) }
        }
    }
}
