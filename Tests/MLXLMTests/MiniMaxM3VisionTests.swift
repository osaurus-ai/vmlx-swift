// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import CoreImage
import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// MiniMax-M3.1 image input (`MiniMaxM3Vision.swift`) against references computed by
/// `scripts/minimax_m3_fixtures/generate_vision_fixtures.py` (`Fixtures/MiniMaxM3/vision`): the bundle's own image
/// processor (torch uint8 antialiased bicubic) on seeded images taking every `smart_resize` branch, vMLX Python's vision
/// stack (e2bb5b771) at a tiny width with seeded weights in the bundle's names and layouts (bit for bit with
/// MLX_ENABLE_TF32=0), and the bundle tokenizer's
/// template renders of image conversations. Then the serving path on the synthetic M3: image runs never split across
/// prefill forwards, the image turn and its text follow-up through the disk prefix cache bit-identical to cold
/// prefills, another image never served the cached rows, and DSpark off while media is prefilled, on for the follow-up.
@Suite("MiniMax-M3.1 vision", .serialized)
struct MiniMaxM3VisionTests {
    typealias A = MiniMaxM3AlignedPrefixCacheTests
    typealias S = MiniMaxM3StructuralPrefixCacheTests
    typealias DS = MiniMaxM3DSparkTests

    static var directory: URL { DS.fixtures.appendingPathComponent("vision") }

    static func meta() throws -> [String: Any] {
        try #require(
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appendingPathComponent("vision-fixtures.json"))) as? [String: Any])
    }

    static func tensors() throws -> [String: MLXArray] {
        try loadArrays(url: directory.appendingPathComponent("vision-fixtures.safetensors"))
    }

    static func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    static func sha256(_ floats: [Float]) -> String { floats.withUnsafeBufferPointer { sha256(Data(buffer: $0)) } }

    static func rgb(_ array: MLXArray) -> DeepseekV41RGBImage {
        DeepseekV41RGBImage(width: array.dim(1), height: array.dim(0), pixels: array.asArray(UInt8.self))
    }

    /// The generator's formula image: rgb[y, x, c] = (7x + 13y + 61c + (x*y mod 31)) mod 256.
    static func formulaImage(width: Int, height: Int) -> DeepseekV41RGBImage {
        var px = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0 ..< height {
            for x in 0 ..< width {
                for c in 0 ..< 3 {
                    px[(y * width + x) * 3 + c] = UInt8((7 * x + 13 * y + 61 * c + (x * y) % 31) % 256)
                }
            }
        }
        return DeepseekV41RGBImage(width: width, height: height, pixels: px)
    }

    // MARK: - Preprocessing

    @Test("Preprocessing == the bundle's processor: grids, float32 pixel values bit for bit, bfloat16 payload digest")
    func preprocessingMatchesBundleProcessor() throws {
        let meta = try Self.meta()
        let arrays = try Self.tensors()
        let config = MiniMaxM3ImageProcessorConfiguration()
        var checked = 0
        for item in meta["images"] as! [[String: Any]] {
            let name = item["name"] as! String
            let w = (item["width"] as! NSNumber).intValue, h = (item["height"] as! NSNumber).intValue
            let image = arrays["\(name).rgb"].map(Self.rgb) ?? Self.formulaImage(width: w, height: h)
            let out = try MiniMaxM3ImagePreprocessor.process(image, config: config)
            let grid = (item["grid"] as! [NSNumber]).map(\.intValue)
            #expect([out.grid.t, out.grid.h, out.grid.w] == grid, "\(name): grid")
            #expect(Self.sha256(out.patches) == item["pixel_values_sha256"] as? String, "\(name): float32 pixels")
            if let expected = arrays["\(name).pixel_values"] {
                let e = expected.asArray(Float.self)
                let mismatches = zip(out.patches, e).filter { $0.0.bitPattern != $0.1.bitPattern }.count
                #expect(out.patches.count == e.count && mismatches == 0, "\(name): \(mismatches) values differ")
            }
            let payload = MiniMaxM3ImagePreprocessor.processedImage([out], rowWidth: 1176)
            let bits = payload.pixels.view(dtype: .uint16).asArray(UInt16.self)
            #expect(bits.withUnsafeBufferPointer { Self.sha256(Data(buffer: $0)) }
                == item["pixel_values_bf16_sha256"] as? String, "\(name): bfloat16 payload")
            checked += 1
        }
        #expect(checked == 4)
    }

    @Test("smart_resize: the processor's branches; an aspect ratio over 200 is the processor's own error")
    func smartResize() throws {
        let r = { try MiniMaxM3ImagePreprocessor.smartResize(height: $0, width: $1, factor: 28, minPixels: 3136, maxPixels: 451_584) }
        #expect(try r(97, 203) == (84, 196))
        #expect(try r(40, 50) == (56, 84))  // 28 x 56 = 1568 px < 3136: the min-pixels ceil branch
        #expect(try r(45, 61) == (56, 56))  // exactly 3136 px: not upscaled
        #expect(try r(700, 1100) == (532, 840))  // max-pixels floor branch
        #expect(try r(112, 112) == (112, 112))
        #expect(try r(1999, 1543) == (756, 588))  // the graded set's large document page
        // Python round(): half to even (126 / 28 = 4.5 -> 4).
        #expect(try r(126, 280).height == 112)
        #expect(throws: MiniMaxM3VisionError.self) { try r(10, 2010) }
    }

    @Test("PNG decode keeps the stored RGB bytes (no colour management), as PIL's convert('RGB')")
    func decodesPNG() throws {
        let decoded = try DeepseekV41ImagePreprocessor.decodeRGB(
            .url(Self.directory.appendingPathComponent("decode_61x45.png")))
        let expected = Self.rgb(try #require(try Self.tensors()["decode_61x45.rgb"]))
        #expect(decoded == expected)
    }

    // MARK: - Vision stack

    static func tinyVisionConfiguration(projector: Int, imageTokenId: Int = 200_025) -> MiniMaxM3VisionConfiguration {
        MiniMaxM3VisionConfiguration(
            hiddenSize: 64, attentionHeads: 4, layers: 2, intermediateSize: 96, patchSize: 14, channels: 3,
            layerNormEps: 1e-5, ropeTheta: 10_000, mergeSize: 2, temporalPatchSize: 2, projectorHiddenSize: projector,
            imageTokenId: imageTokenId, videoTokenId: imageTokenId + 1)
    }

    /// The tiny stack with the fixture weights loaded through the model's sanitize (bundle names and layouts).
    static func tinyStack() throws -> MiniMaxM3VisionStack {
        let stack = MiniMaxM3VisionStack(tinyVisionConfiguration(projector: 32))
        let raw = try loadArrays(url: directory.appendingPathComponent("vision-tiny-weights.safetensors"))
        var update: [(String, MLXArray)] = []
        for (key, value) in raw {
            let (name, tensor) = try #require(MiniMaxM3VisionStack.sanitize(key: key, value: value))
            update.append((String(name.dropFirst("vision.".count)), tensor))
        }
        try stack.update(parameters: ModuleParameters.unflattened(update), verify: .all)
        eval(stack)
        return stack
    }

    @Test("Vision stack == vMLX Python's (tiny width, seeded weights): one image, and two in one call")
    func visionStackMatchesPython() throws {
        try MLXMetalTestLock.withLock {
            try Device.withDefaultDevice(.gpu) {
                let meta = try Self.meta()
                let arrays = try Self.tensors()
                let stack = try Self.tinyStack()
                let grids = Dictionary(uniqueKeysWithValues: (meta["images"] as! [[String: Any]]).map {
                    ($0["name"] as! String, ($0["grid"] as! [NSNumber]).map(\.intValue))
                })
                func pixels(_ name: String) -> MLXArray { arrays["\(name).pixel_values"]!.asType(.bfloat16) }
                func thw(_ name: String) -> THW { THW(grids[name]![0], grids[name]![1], grids[name]![2]) }
                let cases: [(String, [String])] = [
                    ("round_203x97", ["round_203x97"]), ("same_112x112", ["same_112x112"]),
                    ("joint", ["round_203x97", "same_112x112"]),
                ]
                for (label, names) in cases {
                    let out = stack(concatenated(names.map(pixels), axis: 0), grids: names.map(thw))
                    #expect(out.dtype == .float32, "\(label): the float32 patch embedding promotes the stack")
                    let expected = try #require(arrays["features.\(label)"])
                    #expect(out.shape == expected.shape, "\(label): \(out.shape) vs \(expected.shape)")
                    let diff = abs(out - expected).max().item(Float.self)
                    let scale = abs(expected).max().item(Float.self)
                    let identical = arrayEqual(out, expected).item(Bool.self)
                    // The fixture ran with MLX_ENABLE_TF32=0: with it here too the features are Python's bit for bit
                    // (the rotary table on Metal's precise pow, as Python's GPU power). With MLX's default TF32 the
                    // float32 matmuls of the stack round their inputs to TF32 on NAX devices (~1e-3 relative).
                    let tf32Off = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0"
                    if tf32Off {
                        #expect(identical, "\(label): max |diff| \(diff) with TF32 off")
                    } else {
                        #expect(identical || diff <= 5e-3 * max(1, scale), "\(label): max |diff| \(diff) (scale \(scale))")
                    }
                    print("[m3-vision] \(label): identical \(identical), max |diff| \(diff), max |feature| \(scale)")
                }
            }
        }
    }

    // MARK: - Prompt

    @Test("Template: image items render as Python's preprocessing renders them (images first in their turn)")
    func templateRendersImagesLikeHF() throws {
        let renders = try Self.meta()["renders"] as! [String: String]
        let template = try String(
            contentsOf: DS.fixtures.appendingPathComponent("thinking-mode/chat_template.jinja"), encoding: .utf8)
        let image = UserInput.Image.ciImage(CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4)))
        let chats: [String: [Chat.Message]] = [
            "one_image": [.user("What is in this image?", images: [image])],
            "two_images": [.user("Which has more circles?", images: [image, image])],
            "follow_up": [
                .system("You are terse."), .user("Describe it.", images: [image]), .assistant("A red square."),
                .user("What color was it?"),
            ],
            "image_only": [.user("", images: [image])],
        ]
        for (name, chat) in chats {
            let anchored = MiniMaxM3VisionPrompt.anchorImages(
                chat: chat, looseImages: [], generated: DefaultMessageGenerator().generate(messages: chat))
            #expect(anchored.images.count == chat.flatMap(\.images).count)
            let rendered = try MiniMaxM3ThinkingModeTests.render(
                template: template, messages: anchored.messages, tools: nil, addGenerationPrompt: true, context: nil,
                special: ["bos_token": "]~b]", "eos_token": "[e~["])
            let expected = try #require(renders[name])
            #expect(rendered == expected, "\(name): \(MiniMaxM3ThinkingModeTests.firstDifference(rendered, expected))")
        }
    }

    @Test("Expansion: start, t h w / 4 image tokens, end per image; boundaries follow; a count mismatch throws")
    func expansion() throws {
        let img = 50, start = 51, end = 52
        let out = try MiniMaxM3VisionPrompt.expand(
            tokens: [1, 2, img, 3, img, 4], imageTokenId: img, startTokenId: start, endTokenId: end, counts: [3, 1])
        #expect(out.tokens == [1, 2, start, img, img, img, end, 3, start, img, end, 4])
        #expect(out.rawToExpanded == [0, 1, 2, 7, 8, 11, 12])
        #expect(MiniMaxM3VisionPrompt.remapBoundaries([2, 3, 6, 9], rawToExpanded: out.rawToExpanded) == [2, 7, 12])
        #expect(throws: MiniMaxM3VisionError.self) {
            try MiniMaxM3VisionPrompt.expand(tokens: [img], imageTokenId: img, startTokenId: start, endTokenId: end, counts: [1, 1])
        }
    }

    // MARK: - Serving on the synthetic M3

    static let imageToken = 1010, startToken = 1008, endToken = 1009

    /// The aligned / structural tests' synthetic M3 (`A.configuration()`, the M3.1 per-module plan) with a tiny vision
    /// stack whose projector emits the text width; the vision stack stays unquantized (the bundle's is BF16).
    static func visionModel(seed: UInt64 = 5) throws -> MiniMaxM3Model {
        let model = MiniMaxM3Model(
            try A.configuration(), vision: tinyVisionConfiguration(projector: 512, imageTokenId: imageToken))
        var key = MLXRandom.key(seed)
        var update: [(String, MLXArray)] = []
        for (name, value) in model.parameters().flattened() {
            let split = MLXRandom.split(key: key)
            key = split.0
            let r = MLXRandom.normal(value.shape, key: split.1)
            if name.hasSuffix("patch_embedding.weight") {
                update.append((name, (r * Float(0.05)).asType(.float32)))
            } else if name.hasPrefix("vision.") && name.contains("norm") && name.hasSuffix("weight") {
                update.append((name, (1 + r * Float(0.1)).asType(.bfloat16)))
            } else if name.hasPrefix("vision.") && name.hasSuffix("bias") {
                update.append((name, (r * Float(0.02)).asType(.bfloat16)))
            } else if name.hasSuffix("norm.weight") {
                update.append((name, (r * Float(0.1)).asType(.bfloat16)))
            } else if name.hasSuffix("e_score_correction_bias") {
                update.append((name, (r * Float(0.05)).asType(.float32)))
            } else if name.hasSuffix(".mlp.gate.weight") {
                update.append((name, (r * Float(0.3)).asType(.float32)))
            } else if name.contains("embed_tokens") {
                update.append((name, r.asType(.bfloat16)))
            } else {
                update.append((name, (r * Float(1.5 / Double(value.dim(-1)).squareRoot())).asType(.bfloat16)))
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(update), verify: .all)
        quantize(model: model) { path, module in path.hasPrefix("vision") ? nil : DS.plan(path, module) }
        eval(model)
        model.setAlignedPrefillTurnMarkers(S.markers)
        return model
    }

    /// Random bfloat16 patches for a grid (what the processor would hand the model).
    static func patches(_ grid: THW, seed: UInt64) -> MLXArray {
        MLXRandom.normal([grid.t * grid.h * grid.w, 1176], key: MLXRandom.key(seed)).asType(.bfloat16)
    }

    static func imageRun(_ grid: THW) -> [Int] {
        [startToken] + Array(repeating: imageToken, count: grid.t * grid.h * grid.w / 4) + [endToken]
    }

    /// One turn through `TokenIterator` (greedy) with the request's image, then the end-of-turn store.
    static func run(
        _ model: MiniMaxM3Model, _ prompt: [Int], image: LMInput.ProcessedImage?, coordinator: CacheCoordinator?,
        step: Int, recorder: A.Recorder, maxTokens: Int = A.maxTokens, drafter: URL? = nil
    ) throws -> A.Run {
        let before = (recorder.logits.count, recorder.restores.count, recorder.prefills.count)
        var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        parameters.prefillStepSize = step
        if let drafter { parameters.draftStrategy = .dspark(drafterPath: drafter) }
        let boundaries = [prompt.count - S.gen.count]
        let request = LMInput(
            text: .init(tokens: MLXArray(prompt.map { Int32($0) }).reshaped(1, prompt.count), tokenIds: prompt),
            image: image, mediaTokenIds: [imageToken], cachePrefixTokenCounts: boundaries)
        var iterator = try TokenIterator(
            input: request, model: model, parameters: parameters, cacheCoordinator: coordinator)
        var out: [Int] = []
        while let token = iterator.next() { out.append(token) }
        iterator.storeCacheAfterGeneration(generatedTokenIds: out, includeGeneratedBoundary: false)
        let logits = Array(recorder.logits[before.0...])
        eval(logits)
        return A.Run(
            prompt: prompt.count, tokens: out, logits: logits, restores: Array(recorder.restores[before.1...]),
            prefills: Array(recorder.prefills[before.2...]), cache: iterator.cache,
            dspark: iterator.dsparkGenerationStats)
    }

    /// system 301 + user [20 text, image run, 15 text]: the user message starts at 306, so the image run [315, 363)
    /// straddles the step 320.
    static let grid = THW(1, 12, 16)
    static func opening() -> [Int] {
        S.message(S.system, S.content(301, seed: 1))
            + S.message(S.user, S.content(6, seed: 2) + imageRun(grid) + S.content(15, seed: 3))
    }

    @Test("Image prefill: no forward splits an image run; logits == the model's own forward with the features spliced")
    func imageRunsStayWhole() throws {
        let step = 320
        try A.withRecorder { recorder in
            let model = try Self.visionModel()
            let pixels = Self.patches(Self.grid, seed: 11)
            let image = LMInput.ProcessedImage(pixels: pixels, frames: [Self.grid])
            let prompt = Self.opening() + S.gen
            let run = try Self.run(model, prompt, image: image, coordinator: nil, step: step, recorder: recorder)
            let runs = prompt.indices.filter { prompt[$0] == Self.imageToken }
            let lo = try #require(runs.first), hi = try #require(runs.last) + 1
            #expect(lo < step && step < hi, "the run [\(lo), \(hi)) must straddle the step")
            let pieces = try #require(run.prefills.first).pieces
            #expect(!pieces.dropLast().contains { lo < $0.upperBound && $0.upperBound < hi }, "pieces \(pieces)")
            // A text prefill of the same tokens cuts at the step; the image prefill does not.
            #expect(!pieces.map(\.upperBound).contains(step))
            #expect(pieces.map(\.upperBound).contains(S.k(prompt)), "the store boundary stays a cut")

            // Reference: the model's forward over the same pieces, features spliced by hand.
            let features = model.vision!(pixels, grids: [Self.grid])
            let cache = model.newCache(parameters: nil)
            var last = MLXArray(0)
            for piece in pieces {
                let ids = MLXArray(prompt[piece].map { Int32($0) }).reshaped(1, piece.count)
                var h = model.model.embedTokens(ids).reshaped(piece.count, -1)
                let rows = piece.filter { prompt[$0] == Self.imageToken }
                if !rows.isEmpty {
                    let at = MLXArray(rows.map { Int32($0 - piece.lowerBound) })
                    h[at] = features[MLXArray(rows.map { Int32($0 - lo) })].asType(h.dtype)
                }
                last = model.lmHead!(model.model(ids, cache: cache, inputEmbeddings: h.reshaped(1, piece.count, -1)))
                eval(cache.flatMap(\.state))
            }
            #expect(A.equal(run.logits[0], last[0, -1].asType(.float32)), "image prefill != spliced forward")
            // Control: the image changes the answer's logits (the features reach the model).
            let other = try Self.run(
                model, prompt, image: .init(pixels: Self.patches(Self.grid, seed: 12), frames: [Self.grid]),
                coordinator: nil, step: step, recorder: recorder)
            #expect(!A.equal(run.logits[0], other.logits[0]), "another image gave the same logits")
        }
    }

    @Test("Image turn then text follow-up through the disk cache: reuse == K_prev, cached == cold; another image misses")
    func followUpReusesImageTurn() throws {
        let step = 320
        let dir = A.temporaryDirectory("vision-followup")
        defer { try? FileManager.default.removeItem(at: dir) }
        try A.withRecorder { recorder in
            let model = try Self.visionModel()
            let coordinator = S.coordinator(dir, step: step)
            let image = LMInput.ProcessedImage(pixels: Self.patches(Self.grid, seed: 11), frames: [Self.grid])
            let first = Self.opening() + S.gen
            let warm1 = try Self.run(model, first, image: image, coordinator: coordinator, step: step, recorder: recorder)
            let cold1 = try Self.run(model, first, image: image, coordinator: nil, step: step, recorder: recorder)
            #expect(A.same(warm1, cold1), "image turn: warm != cold")
            #expect(warm1.cached == 0)

            let second = Self.opening() + S.message(S.ai, warm1.tokens.map { $0 % 990 })
                + S.message(S.user, S.content(23, seed: 4)) + S.gen
            let warm2 = try Self.run(model, second, image: image, coordinator: coordinator, step: step, recorder: recorder)
            let cold2 = try Self.run(model, second, image: image, coordinator: nil, step: step, recorder: recorder)
            #expect(warm2.cached == S.k(first), "follow-up reused \(warm2.cached), K_prev \(S.k(first))")
            #expect(A.same(warm2, cold2), "follow-up: warm != cold (first gap \(A.firstLogitGap(warm2, cold2)))")
            let problems = DS.cacheProblems(warm2.cache, cold2.cache)
            #expect(problems.isEmpty, "follow-up caches differ \(problems.prefix(3))")
            // The follow-up prefilled text only: the restored prefix holds the whole image run.
            let prefill = try #require(warm2.prefills.first)
            #expect(!second[prefill.start...].contains(Self.imageToken))

            // Same tokens, another image: the media salt differs, nothing is reused, and the answer is its own.
            let otherImage = LMInput.ProcessedImage(pixels: Self.patches(Self.grid, seed: 12), frames: [Self.grid])
            let other = try Self.run(
                model, second, image: otherImage, coordinator: coordinator, step: step, recorder: recorder)
            let otherCold = try Self.run(model, second, image: otherImage, coordinator: nil, step: step, recorder: recorder)
            #expect(other.cached == 0, "another image reused \(other.cached) rows")
            #expect(A.same(other, otherCold))
            #expect(!A.equal(other.logits[0], warm2.logits[0]))
        }
    }

    @Test("DSpark: off while the prefilled input carries the image, drafting on the text follow-up; tokens == plain")
    func dsparkAroundMedia() throws {
        let step = 320
        let dir = A.temporaryDirectory("vision-dspark")
        let drafter = A.temporaryDirectory("vision-dspark-drafter")
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: drafter)
        }
        try DS.writeReleaseDrafter(drafter)
        try A.withRecorder { recorder in
            let model = try Self.visionModel()
            #expect(try model.attachDSparkDraft(directory: drafter))
            let coordinator = S.coordinator(dir, step: step)
            let image = LMInput.ProcessedImage(pixels: Self.patches(Self.grid, seed: 11), frames: [Self.grid])
            let first = Self.opening() + S.gen
            let turn1 = try Self.run(
                model, first, image: image, coordinator: coordinator, step: step, recorder: recorder, maxTokens: 12,
                drafter: drafter)
            let plain1 = try Self.run(model, first, image: image, coordinator: nil, step: step, recorder: recorder, maxTokens: 12)
            #expect(turn1.dspark == nil, "DSpark ran on a media prefill")
            #expect(turn1.tokens == plain1.tokens)
            var history = Self.opening() + S.message(S.ai, turn1.tokens.map { $0 % 990 })
            var drafted = 0
            for turn in 0 ..< 3 {
                let prompt = history + S.message(S.user, S.content(40 + 17 * turn, seed: UInt64(20 + turn))) + S.gen
                let warm = try Self.run(
                    model, prompt, image: image, coordinator: coordinator, step: step, recorder: recorder, maxTokens: 12,
                    drafter: drafter)
                let plain = try Self.run(
                    model, prompt, image: image, coordinator: nil, step: step, recorder: recorder, maxTokens: 12)
                #expect(warm.cached > 0, "follow-up \(turn + 1) reused nothing")
                #expect(warm.dspark != nil, "follow-up \(turn + 1): DSpark was not set up on a text prefill")
                drafted += warm.dspark?.specSteps ?? 0
                #expect(warm.tokens == plain.tokens, "follow-up \(turn + 1): DSpark tokens != plain")
                history = Array(prompt.dropLast(S.gen.count)) + S.message(S.ai, warm.tokens.map { $0 % 990 })
            }
            #expect(drafted > 0, "no DSpark verify ran on the follow-ups")
        }
    }

    @Test("Model build: vision lane built from vision_config, refused text-only requests keep no stack; sanitize drops it")
    func modalities() throws {
        let configData = try JSONSerialization.data(withJSONObject: [
            "model_type": "minimax_m3_vl", "image_token_index": 1010, "projector_hidden_size": 512,
            "vision_config": [
                "hidden_size": 64, "num_attention_heads": 4, "num_hidden_layers": 2, "intermediate_size": 96,
                "patch_size": 14, "img_token_compression_config": ["spatial_merge_size": 2, "temporal_patch_size": 2],
            ],
        ])
        let vision = try #require(try MiniMaxM3VisionConfiguration.decode(configuration: configData))
        #expect(vision.imageTokenId == 1010 && vision.projectorHiddenSize == 512 && vision.headDim == 16)
        #expect(try MiniMaxM3VisionConfiguration.decode(configuration: Data(#"{"model_type":"minimax_m3"}"#.utf8)) == nil)
        // An incomplete vision_config builds the text model (no refusal): no stack.
        #expect(try MiniMaxM3VisionConfiguration.decode(configuration: Data(
            #"{"model_type":"minimax_m3_vl","vision_config":{"hidden_size":64,"num_hidden_layers":1}}"#.utf8)) == nil)
        let text = MiniMaxM3Model(try A.configuration())
        #expect(text.modalities == [.text] && text.vision == nil)
        let both = MiniMaxM3Model(try A.configuration(), vision: vision)
        #expect(both.modalities == [.text, .vision])
        let w = ["vision_tower.vision_model.embeddings.patch_embedding.weight": MLXArray.zeros([64, 3, 2, 14, 14])]
        #expect(text.sanitize(weights: w).isEmpty)
        let mapped = both.sanitize(weights: w)
        #expect(mapped["vision.vision_tower.vision_model.embeddings.patch_embedding.weight"]?.shape == [64, 2, 14, 14, 3])
    }
}
