import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// A suffix prefilled after a cache restore must take the positions a cold prefill of the whole
/// prompt gives it. Qwen3-VL keeps its M-RoPE decode delta on the language model, so two restores
/// went wrong: a fresh model (a relaunch restoring a disk entry) numbered the suffix from 0, and a
/// model that had just served an image prompt shifted a later text-only suffix by that image's
/// delta. Both showed in the app as a model reciting its prompt after a disk-cache hit.
@Suite("Qwen3-VL restored prefix M-RoPE positions", .serialized)
struct Qwen3VLRestoredPrefixPositionTests {
    private func model() throws -> Qwen3VL {
        let config = try JSONDecoder().decode(
            Qwen3VLConfiguration.self,
            from: Data(
                """
                {
                  "model_type": "qwen3_vl",
                  "text_config": {
                    "model_type": "qwen3_vl_text", "hidden_size": 32,
                    "intermediate_size": 64, "num_hidden_layers": 2,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                    "max_position_embeddings": 4096, "vocab_size": 100,
                    "tie_word_embeddings": false, "rope_theta": 10000.0,
                    "rope_scaling": { "type": "default", "mrope_interleaved": true, "mrope_section": [1, 1, 2] }
                  },
                  "vision_config": {
                    "model_type": "qwen3_vl", "depth": 1, "hidden_size": 16,
                    "intermediate_size": 32, "out_hidden_size": 32, "num_heads": 4,
                    "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 1,
                    "num_position_embeddings": 16
                  },
                  "image_token_id": 98, "video_token_id": 97,
                  "vision_start_token_id": 96, "vision_end_token_id": 95
                }
                """.utf8))
        let model = Qwen3VL(config)
        var values: [String: MLXArray] = [:]
        for (name, parameter) in model.parameters().flattened() {
            let seed = name.utf8.reduce(0) { $0 + Int($1) }
            let data: [Float]
            if name.contains("norm"), name.hasSuffix(".weight") {
                data = Array(repeating: 1, count: parameter.size)
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
    private let imageRun: [Int32] = [96, 98, 98, 98, 98, 95]
    private let head: [Int32] = [1, 2, 3, 4, 5]
    private let tail: [Int32] = [6, 7, 8, 9, 10, 11]

    private func input(_ tokens: [Int32], image: Bool) -> LMInput {
        LMInput(
            text: .init(tokens: MLXArray(tokens, [1, tokens.count])),
            image: image ? .init(pixels: pixels, frames: [THW(1, 4, 4)]) : nil,
            mediaTokenIds: [98])
    }

    private func lastLogits(_ prepared: PrepareResult) throws -> MLXArray {
        guard case .logits(let result) = prepared else {
            Issue.record("VLM prepare must return logits")
            throw CocoaError(.coderInvalidValue)
        }
        return result.logits[0..., -1, 0...]
    }

    /// Rotated keys of the last layer, every cached row: RoPE makes them a direct function of each
    /// row's position, so a mispositioned suffix shows here at full size.
    private func keys(_ cache: [any KVCache]) -> MLXArray {
        let layer = cache[cache.count - 1]
        return layer.state[0][0..., 0..., ..<layer.offset, 0...]
    }

    private func maxError(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.max(abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
    }

    /// First-token logits, two decode steps and the final keys.
    private func run(_ m: Qwen3VL, prepare: LMInput, cache: [any KVCache]) throws -> [MLXArray] {
        var out = [try lastLogits(m.prepare(prepare, cache: cache, windowSize: 0))]
        for token: Int32 in [31, 32] {
            out.append(m(MLXArray([token], [1, 1]), cache: cache)[0..., -1, 0...])
        }
        out.append(keys(cache))
        MLX.eval(out)
        return out
    }

    /// What every prefill does before it hands the model a restored cache (`Evaluate.swift`, `BatchEngine`).
    private func armRestore(_ m: Qwen3VL, full: LMInput, restored: Int) {
        let model: any LanguageModel = m
        (model as? any RestoredPrefixPositionModel)?.prepareRestoredPrefixPositions(
            fullInput: full, restoredTokenCount: restored)
    }

    private func cold(_ full: LMInput) throws -> [MLXArray] {
        let m = try model()
        let cache = m.newCache(parameters: nil)
        armRestore(m, full: full, restored: 0)
        return try run(m, prepare: full, cache: cache)
    }

    /// The prefix's KV rows, written by another model instance, as a disk restore hands them over.
    private func restoredCache(prefix: LMInput) throws -> [any KVCache] {
        let writer = try model()
        let cache = writer.newCache(parameters: nil)
        MLX.eval(try lastLogits(writer.prepare(prefix, cache: cache, windowSize: 0)), cache)
        return cache
    }

    private func assertMatchesCold(_ actual: [MLXArray], _ expected: [MLXArray]) {
        #expect(maxError(actual[0], expected[0]) < 5e-3, "first-token logits")
        for (index, (a, e)) in zip(actual.dropFirst(), expected.dropFirst()).enumerated() {
            #expect(maxError(a, e) < 1e-4, "decode step / keys \(index): \(maxError(a, e))")
        }
    }

    @Test("a text prefix restored into a fresh model continues at the cache offset")
    func textPrefixRestoredIntoFreshModel() throws {
        try MLXMetalTestLock.withLock {
            let full = input(head + tail, image: false)
            let expected = try cold(full)
            let cache = try restoredCache(prefix: input(head, image: false))
            let fresh = try model()
            armRestore(fresh, full: full, restored: head.count)
            assertMatchesCold(
                try run(fresh, prepare: input(tail, image: false), cache: cache), expected)
        }
    }

    @Test("a text-only restore after an image prompt does not reuse the image's delta")
    func textRestoreAfterImagePrompt() throws {
        try MLXMetalTestLock.withLock {
            let full = input(head + tail, image: false)
            let expected = try cold(full)
            let cache = try restoredCache(prefix: input(head, image: false))

            let live = try model()
            let imagePrompt = input([1, 2] + imageRun + [5, 6, 7], image: true)
            armRestore(live, full: imagePrompt, restored: 0)
            MLX.eval(
                try lastLogits(
                    live.prepare(imagePrompt, cache: live.newCache(parameters: nil), windowSize: 0))
            )

            armRestore(live, full: full, restored: head.count)
            assertMatchesCold(
                try run(live, prepare: input(tail, image: false), cache: cache), expected)
        }
    }

    @Test("text after a restored image takes the full prompt's positions")
    func textAfterRestoredImage() throws {
        try MLXMetalTestLock.withLock {
            let prefixTokens: [Int32] = [1, 2] + imageRun
            let full = input(prefixTokens + tail, image: true)
            let expected = try cold(full)
            let cache = try restoredCache(prefix: input(prefixTokens, image: true))
            let fresh = try model()
            armRestore(fresh, full: full, restored: prefixTokens.count)
            assertMatchesCold(
                try run(fresh, prepare: input(tail, image: false), cache: cache), expected)
        }
    }

    @Test("an image after a restored text prefix matches a cold prefill")
    func imageAfterRestoredText() throws {
        try MLXMetalTestLock.withLock {
            let full = input(head + imageRun + tail, image: true)
            let expected = try cold(full)
            let cache = try restoredCache(prefix: input(head, image: false))
            let suffix = mediaCarryingSuffixInput(
                from: full, remainingTokens: (imageRun + tail).map(Int.init),
                matchedTokens: head.count)
            let fresh = try model()
            armRestore(fresh, full: full, restored: head.count)
            assertMatchesCold(try run(fresh, prepare: suffix, cache: cache), expected)
        }
    }

    /// The test above is what lets the prefix cache resume an image prompt from a text boundary:
    /// without the declaration that hit rolls back to a full prefill.
    @Test("an image prompt resumes from a boundary before its first placeholder")
    func preMediaBoundaryResumes() throws {
        try MLXMetalTestLock.withLock {
            let tokens = head + imageRun + tail
            let full = input(tokens, image: true)
            let plan = CacheBoundarySaltPlan.make(
                requestSalt: "image", input: full, lookupTokenIds: tokens.map(Int.init),
                model: try model(), parameters: GenerateParameters())
            let firstPlaceholder = head.count + 1
            #expect(plan.mediaIndependentTokenCount == firstPlaceholder)
            #expect(plan.salt(forBoundary: head.count) != "image")
            #expect(plan.salt(forBoundary: firstPlaceholder + 1) == "image")
            #expect(plan.resumeBeforeMediaIsSafe(matchedTokens: head.count))
            #expect(plan.prefixRowsAreMediaIndependent(boundary: firstPlaceholder))
            #expect(!plan.prefixRowsAreMediaIndependent(boundary: firstPlaceholder + 1))
        }
    }
}
