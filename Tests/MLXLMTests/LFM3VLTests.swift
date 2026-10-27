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

    /// Expected sizes from the reference `_smart_resize_image` (factor 32, min 65536) at the
    /// LFM3-VL (16777216) and LFM3-Omni (2097152) pixel budgets. 720 / 32 = 22.5 rounds to 22
    /// (half to even), so a 1280x720 frame is 44x80 patches, not 46x80.
    @Test func imageResizeMatchesTheReference() throws {
        let cases: [((Int, Int), (Int, Int), (Int, Int))] = [
            ((720, 1280), (704, 1280), (704, 1280)),
            ((1280, 720), (1280, 704), (1280, 704)),
            ((480, 640), (480, 640), (480, 640)),
            ((10, 1000), (32, 2560), (32, 2560)),
            ((31, 31), (256, 256), (256, 256)),
            ((3000, 4000), (3008, 4000), (1248, 1664)),
            ((80, 80), (256, 256), (256, 256)),
            ((400, 336), (384, 320), (384, 320)),
            ((1080, 1920), (1088, 1920), (1088, 1920)),
            ((1, 50), (64, 1824), (64, 1824)),
            ((5000, 5000), (4096, 4096), (1440, 1440)),
        ]
        for (size, vl, omni) in cases {
            let a = try LFM3VLProcessor.imageTargetSize(
                height: size.0, width: size.1, factor: 32, minPixels: 65536, maxPixels: 16_777_216)
            let b = try LFM3VLProcessor.imageTargetSize(
                height: size.0, width: size.1, factor: 32, minPixels: 65536, maxPixels: 2_097_152)
            #expect(a.0 == vl.0 && a.1 == vl.1, "VL \(size): \(a)")
            #expect(b.0 == omni.0 && b.1 == omni.1, "Omni \(size): \(b)")
        }
        #expect(throws: (any Error).self) {
            try LFM3VLProcessor.imageTargetSize(
                height: 1, width: 300, factor: 32, minPixels: 65536, maxPixels: 16_777_216)
        }
        // Control: the shared Qwen-VL helper rounds 22.5 up, which is why LFM3 carries its own.
        let shared = try QwenVL.targetSize(
            height: 720, width: 1280, factor: 32, minPixels: 65536, maxPixels: 16_777_216)
        #expect(shared.0 == 736, "shared helper now \(shared); if it rounds half-even, reuse it")
    }

    /// Expected sizes from the reference `Lfm3VlVideoProcessor._process_one` rule (min 4096,
    /// max 25165824, temporal 2).
    @Test func videoResizeMatchesTheReference() throws {
        let cases: [((Int, Int, Int), (Int, Int))] = [
            ((8, 480, 640), (480, 640)), ((11, 720, 1280), (704, 1280)),
            ((768, 1080, 1920), (128, 224)), ((4, 20, 40), (32, 64)), ((9, 336, 400), (320, 384)),
        ]
        for (input, expected) in cases {
            let out = try LFM3VLProcessor.videoTargetSize(
                frames: input.0, height: input.1, width: input.2, temporal: 2, factor: 32,
                minPixels: 4096, maxPixels: 25_165_824)
            #expect(out.0 == expected.0 && out.1 == expected.1, "video \(input): \(out)")
        }
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

    /// `LFM3_VL_BUNDLE` + `LFM3_VL_VIDEO`: a 4 s clip of a red square moving left to right past
    /// a fixed blue circle. Not reference-exact (frame sampling is the shared vMLX sampler):
    /// checks the per-group token structure and that the answer reads the motion.
    @Test func releasedBundleReadsVideo() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let bundle = env["LFM3_VL_BUNDLE"], let videoPath = env["LFM3_VL_VIDEO"] else { return }
        let context = try await VLMModelFactory.shared.load(
            from: URL(fileURLWithPath: bundle), using: #huggingFaceTokenizerLoader())
        let input = UserInput(
            chat: [
                .user(
                    "In which direction does the red square move: left to right, or right to left? Answer in one sentence.",
                    videos: [.url(URL(fileURLWithPath: videoPath))])
            ],
            additionalContext: ["enable_thinking": false])
        let prepared = try await context.processor.prepare(input: input)
        let grid = try #require(prepared.video?.frames?.first)
        let ids = prepared.text.tokens.reshaped(-1).asArray(Int.self)
        let videoId = try #require(context.tokenizer.convertTokenToId("<|video|>"))
        let start = try #require(context.tokenizer.convertTokenToId("<|vision_start|>"))
        // One inner vision_start per frame group, plus the template's outer one.
        #expect(ids.filter { $0 == videoId }.count == grid.t * grid.h * grid.w / 4)
        #expect(ids.filter { $0 == start }.count == grid.t + 1)
        let rendered = context.tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
        #expect(rendered.contains("<0.0 seconds>") || rendered.contains("<0.2 seconds>"), "timestamps")

        var iterator = try TokenIterator(
            input: prepared, model: context.model,
            parameters: GenerateParameters(maxTokens: 40, temperature: 0))
        var tokens: [Int] = []
        while let token = iterator.next() {
            if context.configuration.eosTokenIds.contains(token) { break }
            tokens.append(token)
        }
        let answer = context.tokenizer.decode(tokenIds: tokens)
        print("[LFM3-VL] video grid \(grid), \(ids.count) prompt tokens: \(answer)")
        // The reversed clip (LFM3_VL_VIDEO_EXPECT="right to left") is the control: a model that
        // ignored the frame order would give the same answer to both.
        let expected = env["LFM3_VL_VIDEO_EXPECT"] ?? "left to right"
        #expect(answer.lowercased().contains(expected), "motion answer: \(answer)")
    }
}
