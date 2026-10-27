// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXHuggingFace
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM
@preconcurrency import VMLXTokenizers

/// LFM3-VL: tower geometry, placeholder expansion, and (opt-in) parity with a released bundle.
@Suite("LFM3-VL", .serialized)
struct LFM3VLTests {

    // MARK: - Geometry

    /// Tokens are 2x2 merge blocks row-major, patches inside each block row-major; the
    /// projector concatenates each consecutive four, and the conv reshapes to the grid.
    @Test func mergeOrderCoordinates() {
        let (rows, cols) = LFM3VLVision.mergeOrderedCoordinates(height: 4, width: 4, merge: 2)
        #expect(Array(zip(rows, cols).prefix(8)).map { [Int($0.0), Int($0.1)] } == [
            [0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3],
        ])
    }

    @Test func expansionReplacesOnlyTheInnerPlaceholderAndCarriesBoundaries() throws {
        // [bos, start, IMG, end, text, start, IMG, end, text]
        let tokens = [1, 8, 5, 9, 2, 8, 5, 9, 3]
        let out = try LFM3VLProcessor.expand(
            tokens, placeholder: 5, replacements: [[5, 5, 5], [5, 5]], boundaries: [1, 5, 9])
        #expect(out.tokens == [1, 8, 5, 5, 5, 9, 2, 8, 5, 5, 9, 3])
        #expect(out.boundaries == [1, 7, 12])
        #expect(throws: (any Error).self) {
            try LFM3VLProcessor.expand(tokens, placeholder: 5, replacements: [[5]])
        }
    }

    // MARK: - Released bundle parity (opt-in)

    /// `LFM3_VL_BUNDLE`, `LFM3_VL_REF` (JSON + `.safetensors` sidecar from the reference
    /// script) and `LFM3_VL_IMAGE`. Checks: processor ids identical to the reference; pixels
    /// within resampling tolerance; tower + projector features on the REFERENCE pixels; and an
    /// end-to-end greedy answer through the processor and `prepare`.
    @Test func releasedBundleMatchesReference() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let bundle = env["LFM3_VL_BUNDLE"], let refPath = env["LFM3_VL_REF"],
            let imagePath = env["LFM3_VL_IMAGE"]
        else { return }
        let fp32 = env["LFM3_FP32"] == "1"
        let ref = try JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: refPath))) as! [String: Any]
        let tensors = try MLX.loadArrays(url: URL(fileURLWithPath: refPath + ".safetensors"))
        let refIds = try #require(ref["ids"] as? [Int])
        let question = try #require(ref["question"] as? String)

        let context = try await VLMModelFactory.shared.load(
            from: URL(fileURLWithPath: bundle), using: #huggingFaceTokenizerLoader())
        let model = try #require(context.model as? LFM3VL)
        if fp32 {
            let upcast = model.parameters().flattened().map { key, value in
                (key, value.dtype == .bfloat16 ? value.asType(.float32) : value)
            }
            try model.update(parameters: ModuleParameters.unflattened(upcast), verify: .none)
            eval(model)
        }

        // Processor.
        let input = UserInput(
            chat: [.user(question, images: [.url(URL(fileURLWithPath: imagePath))])],
            additionalContext: ["enable_thinking": false])
        let prepared = try await context.processor.prepare(input: input)
        let ids = prepared.text.tokens.reshaped(-1).asArray(Int.self)
        #expect(ids == refIds, "processor ids differ: \(ids.count) vs \(refIds.count)")
        let refPixels = try #require(tensors["pixel_values"])
        let pixels = try #require(prepared.image?.pixels)
        #expect(pixels.shape == refPixels.shape)
        let pixelDiff = abs(pixels.asType(.float32) - refPixels).mean().item(Float.self)
        print("[LFM3-VL] pixels mean |diff| \(pixelDiff), grid \(prepared.image?.frames ?? [])")
        #expect(pixelDiff < 0.02, "pixel mean diff \(pixelDiff)")

        // Tower + projector on the reference's own pixels.
        let grids = try #require(prepared.image?.frames)
        let features = try model.mediaFeatures(pixels: refPixels, grids: grids).asType(.float32)
        let refFeatures = try #require(tensors["features"])
        let featureDiff = abs(features - refFeatures).max().item(Float.self)
        let featureScale = abs(refFeatures).max().item(Float.self)
        print("[LFM3-VL] features max |diff| \(featureDiff) (scale \(featureScale))")
        #expect(featureDiff <= (fp32 ? 1e-3 : 0.05) * max(1, featureScale))

        // End to end: our processor's input through prepare + decode.
        let refGreedy = try #require(ref["greedy"] as? [Int])
        var iterator = try TokenIterator(
            input: prepared, model: model,
            parameters: GenerateParameters(maxTokens: refGreedy.count, temperature: 0))
        var greedy: [Int] = []
        while greedy.count < refGreedy.count, let token = iterator.next() { greedy.append(token) }
        let prefix = zip(greedy, refGreedy).prefix { $0 == $1 }.count
        let text = context.tokenizer.decode(tokenIds: greedy)
        print("[LFM3-VL] greedy prefix \(prefix)/\(refGreedy.count): \(text)")
        #expect(text.contains("OSAURUS 42"), "answer must read the text in the image")
        #expect(prefix >= min(refGreedy.count, 12), "greedy diverged at \(prefix)")
    }
}
