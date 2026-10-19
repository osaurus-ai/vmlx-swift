import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// A prefix restored from the cache and a suffix prefilled after it must reproduce a cold prefill
/// of the whole prompt when the prompt carries media (M-RoPE): text after a restored image, and an
/// image after restored text. `RestoredPrefixPositionModel` supplies the full prompt's positions;
/// without it the suffix is positioned from the suffix alone (the control case below).
@Suite("Qwen3.5 restored prefix M-RoPE positions", .serialized)
struct Qwen35RestoredMediaPositionTests {
    private func model() throws -> Qwen35 {
        let config = try JSONDecoder().decode(
            Qwen35Configuration.self,
            from: Data(
                """
                {
                  "model_type": "qwen3_5", "text_config": {
                    "model_type": "qwen3_5_text", "hidden_size": 32,
                    "num_hidden_layers": 4, "intermediate_size": 64,
                    "num_attention_heads": 4, "num_key_value_heads": 2,
                    "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                    "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                    "linear_conv_kernel_dim": 4, "head_dim": 8,
                    "full_attention_interval": 4, "vocab_size": 100,
                    "tie_word_embeddings": false, "rms_norm_eps": 1e-6,
                    "rope_parameters": {
                      "rope_type": "default", "rope_theta": 10000.0,
                      "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]
                    }
                  },
                  "vision_config": {
                    "model_type": "qwen3_vl", "depth": 1,
                    "hidden_size": 16, "intermediate_size": 32,
                    "out_hidden_size": 32, "num_heads": 4, "patch_size": 2,
                    "spatial_merge_size": 2, "temporal_patch_size": 1,
                    "num_position_embeddings": 16
                  },
                  "vocab_size": 100, "image_token_id": 98, "video_token_id": 97,
                  "vision_start_token_id": 96, "vision_end_token_id": 95
                }
                """.utf8))
        let model = Qwen35(config)
        // Explicit parameters make the row repeatable without changing a
        // process-global random seed or relying on random initialization order.
        var values: [String: MLXArray] = [:]
        for (name, parameter) in model.parameters().flattened() {
            let seed = name.utf8.reduce(0) { $0 + Int($1) }
            let data: [Float]
            if name.hasSuffix(".A_log") || name.hasSuffix(".dt_bias") {
                data = Array(repeating: 0, count: parameter.size)
            } else if name.contains("norm"), name.hasSuffix(".weight") {
                // 1, not 0, for every norm: a zero q/k norm weight makes every rotated key zero,
                // and then no position error could show.
                let gamma: Float = 1
                data = Array(repeating: gamma, count: parameter.size)
            } else {
                data = (0 ..< parameter.size).map { Float(($0 * 17 + seed) % 41 - 20) * 0.015 }
            }
            values[name] = MLXArray(data, parameter.shape).asType(parameter.dtype)
        }
        model.update(parameters: ModuleParameters.unflattened(values))
        MLX.eval(model)
        return model
    }

    private let pixels = MLXArray((0 ..< 16 * 12).map { Float($0 % 19) / 19 }, [16, 12])

    private func input(_ tokens: [Int32], image: Bool) -> LMInput {
        LMInput(
            text: .init(tokens: MLXArray(tokens, [1, tokens.count])),
            image: image ? .init(pixels: pixels, frames: [THW(1, 4, 4)]) : nil,
            mediaTokenIds: [98])
    }

    private func logits(_ prepared: PrepareResult) throws -> MLXArray {
        guard case .logits(let result) = prepared else {
            Issue.record("VLM prepare must return logits")
            throw CocoaError(.coderInvalidValue)
        }
        return result.logits[0..., -1, 0...]
    }

    private func maxError(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.max(abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
    }

    /// Cold logits for the whole prompt, then two decode steps.
    private func cold(_ full: LMInput) throws -> [MLXArray] {
        let m = try model()
        let cache = m.newCache(parameters: nil)
        var out = [try logits(m.prepare(full, cache: cache, windowSize: 0))]
        for token: Int32 in [31, 32] { out.append(m(MLXArray([token], [1, 1]), cache: cache)[0..., -1, 0...]) }
        out.append(attentionKeys(cache))
        MLX.eval(out)
        return out
    }

    /// Prefix forwarded as an earlier turn, suffix prepared after it (as a cache hit does), then decode.
    private func restored(
        _ full: LMInput, prefix: LMInput, suffix: LMInput, window: Int, hook: Bool
    ) throws -> [MLXArray] {
        let m = try model()
        let cache = m.newCache(parameters: nil)
        MLX.eval(try logits(m.prepare(prefix, cache: cache, windowSize: 0)), cache)
        if hook {
            m.prepareRestoredPrefixPositions(
                fullInput: full, restoredTokenCount: prefix.text.tokens.dim(1))
        }
        var out = [try logits(m.prepare(suffix, cache: cache, windowSize: window))]
        for token: Int32 in [31, 32] { out.append(m(MLXArray([token], [1, 1]), cache: cache)[0..., -1, 0...]) }
        out.append(attentionKeys(cache))
        MLX.eval(out)
        return out
    }

    /// Rotated keys of the full-attention layer (layer 4 of 4), every cached row: RoPE makes them a
    /// direct function of each row's position, so a mispositioned suffix shows here at full size.
    private func attentionKeys(_ cache: [any KVCache]) -> MLXArray {
        let layer = cache[3]
        return layer.state[0][0..., 0..., ..<layer.offset, 0...]
    }

    private let imageRun: [Int32] = [96, 98, 98, 98, 98, 95]
    private let tail: [Int32] = (0 ..< 6).map { Int32(4 + $0) }

    @Test("text after a restored image matches a cold prefill", arguments: [0, 3])
    func textAfterRestoredImage(window: Int) throws {
        try MLXMetalTestLock.withLock {
            let head: [Int32] = [1, 2] + imageRun
            let full = input(head + tail, image: true)
            let expected = try cold(full)
            let actual = try restored(
                full, prefix: input(head, image: true), suffix: input(tail, image: false),
                window: window, hook: true)
            for (a, e) in zip(actual, expected) { #expect(maxError(a, e) < 1e-4) }
            // Control: without the full-prompt plan the suffix is placed at offset + index, past
            // the image's compressed positions, and the logits move.
            let control = try restored(
                full, prefix: input(head, image: true), suffix: input(tail, image: false),
                window: window, hook: false)
            #expect(maxError(control.last!, expected.last!) > 1e-2)
            #expect(maxError(control[0], expected[0]) > 1e-4)
        }
    }

    @Test("an image after restored text matches a cold prefill", arguments: [0, 3])
    func imageAfterRestoredText(window: Int) throws {
        try MLXMetalTestLock.withLock {
            let head: [Int32] = [1, 2, 3, 4]
            let full = input(head + imageRun + tail, image: true)
            let expected = try cold(full)
            let suffix = mediaCarryingSuffixInput(
                from: full, remainingTokens: (imageRun + tail).map(Int.init), matchedTokens: head.count)
            let actual = try restored(
                full, prefix: input(head, image: false), suffix: suffix, window: window, hook: true)
            for (a, e) in zip(actual, expected) { #expect(maxError(a, e) < 1e-4) }
            let control = try restored(
                full, prefix: input(head, image: false), suffix: suffix, window: window, hook: false)
            #expect(maxError(control.last!, expected.last!) > 1e-2)
        }
    }

    @Test("pre-media boundaries are media-independent for Qwen-VL positions")
    func declaresPreMediaInvariant() throws {
        let m = try model()
        #expect(m.statesBeforeFirstMediaTokenAreMediaIndependent)
        #expect(!m.exactPrefixRestoreIsSuffixIndependent)
    }

    // MARK: - Two images: per-item keys

    private let pixels2 = MLXArray((0 ..< 16 * 12).map { Float(($0 * 7) % 23) / 23 }, [16, 12])

    private func twoImages(_ tokens: [Int32]) -> LMInput {
        LMInput(
            text: .init(tokens: MLXArray(tokens, [1, tokens.count]), tokenIds: tokens.map(Int.init)),
            image: .init(
                pixels: concatenated([pixels, pixels2], axis: 0), frames: [THW(1, 4, 4), THW(1, 4, 4)]),
            mediaTokenIds: [98, 97])
    }

    private var firstTurn: [Int32] { [1, 2] + imageRun + [5, 6, 7] }
    private var secondTurn: [Int32] { firstTurn + [8, 9] + imageRun + tail }

    @Test("the boundary between two images is keyed like the earlier one-image turn")
    func perItemSalt() throws {
        let m = try model()
        let full = twoImages(secondTurn)
        let spans = try #require(m.mediaItemSpans(in: full, tokenIds: secondTurn.map(Int.init)))
        #expect(spans.map(\.tokenRange) == [3 ..< 7, 14 ..< 18])
        #expect(spans.map(\.pixelRows) == [0 ..< 16, 16 ..< 32])
        let params = GenerateParameters()
        let plan = CacheBoundarySaltPlan.make(
            requestSalt: computeCacheSalt(for: full, parameters: params), input: full,
            lookupTokenIds: secondTurn.map(Int.init), model: m, parameters: params)
        let earlier = LMInput(
            text: .init(tokens: MLXArray(firstTurn, [1, firstTurn.count])),
            image: .init(pixels: pixels, frames: [THW(1, 4, 4)]), mediaTokenIds: [98, 97])
        let earlierSalt = computeCacheSalt(for: earlier, parameters: params)
        #expect(plan.salt(forBoundary: firstTurn.count) == earlierSalt)
        #expect(plan.salt(forBoundary: 3) == computeCacheSalt(for: full, parameters: params, includingMedia: false))
        #expect(plan.salt(forBoundary: secondTurn.count) == plan.requestSalt)
        #expect(plan.requestSalt != earlierSalt)
        #expect(plan.resumeBeforeMediaIsSafe(matchedTokens: firstTurn.count))
        #expect(plan.resumeBeforeMediaIsSafe(matchedTokens: 14))
        #expect(!plan.resumeBeforeMediaIsSafe(matchedTokens: 5))
        #expect(!plan.resumeBeforeMediaIsSafe(matchedTokens: 16))
        #expect(!plan.resumeBeforeMediaIsSafe(matchedTokens: 20))
        #expect(plan.mediaIndependentPrefix?.salt(forBoundary: 12) == .some(earlierSalt))
        // The suffix after the first image carries only the second image's pixels.
        let suffix = mediaCarryingSuffixInput(
            from: full, remainingTokens: Array(secondTurn[firstTurn.count...]).map(Int.init),
            matchedTokens: firstTurn.count, itemSpans: plan.itemSpans)
        #expect(suffix.image?.frames?.map(\.product) == [16])
        #expect(suffix.image.map { maxError($0.pixels, pixels2) } == 0)
    }

    @Test("a second image after a restored first image matches a cold prefill", arguments: [0, 3])
    func secondImageAfterRestoredFirst(window: Int) throws {
        try MLXMetalTestLock.withLock {
            let full = twoImages(secondTurn)
            let expected = try cold(full)
            let m = try model()
            let spans = try #require(m.mediaItemSpans(in: full, tokenIds: secondTurn.map(Int.init)))
            let prefix = LMInput(
                text: .init(tokens: MLXArray(firstTurn, [1, firstTurn.count])),
                image: .init(pixels: pixels, frames: [THW(1, 4, 4)]), mediaTokenIds: [98, 97])
            let suffix = mediaCarryingSuffixInput(
                from: full, remainingTokens: Array(secondTurn[firstTurn.count...]).map(Int.init),
                matchedTokens: firstTurn.count, itemSpans: spans)
            let actual = try restored(full, prefix: prefix, suffix: suffix, window: window, hook: true)
            for (a, e) in zip(actual, expected) { #expect(maxError(a, e) < 1e-4) }
        }
    }
}
