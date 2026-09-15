import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

@Suite("GLM image input contracts", .serialized)
struct Glm5NextInputContractTests {
    struct MarkerTokenizer: MLXLMCommon.Tokenizer {
        let omitLastImage: Bool
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [10, 11] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "fixture" }
        func convertTokenToId(_ token: String) -> Int? {
            switch token {
            case "<|image|>": 99
            case "<|video|>": 98
            default: nil
            }
        }
        func convertIdToToken(_ id: Int) -> String? { id == 99 ? "<|image|>" : nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            var tokens = [10]
            for message in messages {
                let content = message["content"] as? [[String: String]] ?? []
                for item in content where item["type"] == "image" { tokens += [99, 20] }
                for item in content where item["type"] == "video" { tokens += [98, 20] }
            }
            if omitLastImage, let last = tokens.lastIndex(of: 99) { tokens.remove(at: last) }
            return tokens + [11]
        }
    }

    static func processor(omitLastImage: Bool = false) throws -> Glm5NextProcessor {
        let config = try JSONDecoder().decode(
            Glm5NextImageProcessorConfiguration.self,
            from: Data(
                """
                {"image_mean":[0,0,0],"image_std":[1,1,1],"patch_size":14,
                 "merge_size":2,"temporal_patch_size":2,"min_image_tokens":1,"max_image_tokens":16}
                """.utf8))
        return Glm5NextProcessor(
            config: config,
            tokenizer: MarkerTokenizer(omitLastImage: omitLastImage), imageToken: 99)
    }

    @Test("processor tokens accept exactly one generic generation batch dimension")
    func unbatchedText() async throws {
        try await MLXMetalTestLock.withLock {
            let input = try await Self.processor().prepare(input: UserInput(prompt: "hello"))
            #expect(input.text.tokens.shape == [2])
            #expect(input.text.tokens[.newAxis, 0...].shape == [1, 2])
        }
    }

    @Test("two images retain separate ordered placeholder runs and patch grids")
    func orderedImages() async throws {
        try await MLXMetalTestLock.withLock {
            let p = try Self.processor()
            let red = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 28, height: 28))
            let blue = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 56, height: 28))
            let first = try p.preprocess(image: red)
            let second = try p.preprocess(image: blue)
            let input = try await p.prepare(
                input: UserInput(prompt: "compare", images: [.ciImage(red), .ciImage(blue)]))
            let expected =
                [10] + Array(repeating: 99, count: first.tokenCount) + [20]
                + Array(repeating: 99, count: second.tokenCount) + [20, 11]
            #expect(input.text.tokens.shape == [expected.count])
            #expect(input.text.tokens.asArray(Int.self) == expected)
            #expect(input.mediaTokenIds == [99])
            let afterImages = try #require(expected.lastIndex(of: 99)) + 1
            #expect(
                input.canCaptureHybridStripBoundary(promptTokenIds: expected, boundary: afterImages)
            )
            #expect(
                !input.canCaptureHybridStripBoundary(
                    promptTokenIds: expected, boundary: afterImages - 1))
            #expect(!input.cacheHitSuffixContainsMediaPlaceholder(Array(expected[afterImages...])))
            #expect(
                input.cacheHitSuffixContainsMediaPlaceholder(Array(expected[(afterImages - 1)...])))
            let image = try #require(input.image)
            #expect(image.frames?.map(\.product) == [first.grid.product, second.grid.product])
            let expectedPixels = concatenated([first.patches, second.patches], axis: 0)
            #expect(image.pixels.shape == expectedPixels.shape)
            #expect(abs(image.pixels - expectedPixels).max().item(Float.self) < 1e-6)
        }
    }

    @Test("a template missing an image placeholder cannot discard one attachment")
    func missingImageMarker() async throws {
        try await MLXMetalTestLock.withLock {
            let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 28, height: 28))
            do {
                _ = try await Self.processor(omitLastImage: true).prepare(
                    input: UserInput(prompt: "compare", images: [.ciImage(image), .ciImage(image)]))
                Issue.record("Missing image placeholder was accepted")
            } catch {
                #expect(String(describing: error).contains("placeholder"))
            }
        }
    }

    @Test(
        "short text crosses the generic prefill boundary with finite nonconstant logits",
        arguments: [false, true])
    func shortTextForward(batchedFragment: Bool) async throws {
        try await MLXMetalTestLock.withLock {
            let config = try JSONDecoder().decode(
                Glm5NextConfiguration.self,
                from: Data(Glm5NextConstructionTests.tinyJSON.utf8))
            let model = try Glm5Next(config, requesting: [.text])
            var input = try await Self.processor().prepare(input: UserInput(prompt: "hello"))
            if batchedFragment {
                // Cache boundary splitting supplies [1, T] fragments to prepare.
                input = LMInput(
                    text: .init(
                        tokens: input.text.tokens[.newAxis, 0...],
                        mask: MLXArray.ones([1, 2]), tokenIds: [10, 11]))
            }
            let cache = model.newCache(parameters: nil)
            guard
                case .tokens(let remaining) = try model.prepare(
                    input, cache: cache, windowSize: 512)
            else {
                Issue.record("Short text did not exercise generic prefill")
                return
            }
            #expect(remaining.tokens.shape == [2])
            if batchedFragment {
                #expect(remaining.mask?.shape == [2])
                #expect(remaining.tokenIds == [10, 11])
            }
            let output = model(remaining[text: .newAxis], cache: cache, state: nil)
            let values = output.logits.asType(.float32).asArray(Float.self)
            #expect(output.logits.shape == [1, 2, config.textConfig.vocabSize])
            #expect(values.allSatisfy { $0.isFinite })
            #expect(Set(values).count > 1, "An invalid input must not pass as zero logits")
        }
    }

    @Test("processor image output reaches the vision tower and decoder without a second batch axis")
    func imageForward() async throws {
        try await MLXMetalTestLock.withLock {
            let config = try JSONDecoder().decode(
                Glm5NextConfiguration.self,
                from: Data(
                    Glm5NextConstructionTests.tinyJSON.replacingOccurrences(
                        of: "\"image_token_id\":9", with: "\"image_token_id\":99"
                    ).utf8))
            let model = try Glm5Next(config, requesting: [.vision])
            let red = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 28, height: 28))
            let blue = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 56, height: 28))
            for images in [[red], [red, blue]] {
                let input = try await Self.processor().prepare(
                    input: UserInput(
                        prompt: "compare", images: images.map { .ciImage($0) }))
                guard
                    case .logits(let output) = try model.prepare(
                        input, cache: model.newCache(parameters: nil), windowSize: 512)
                else {
                    Issue.record("Images did not exercise vision prefill")
                    return
                }
                let values = output.logits.asType(.float32).asArray(Float.self)
                #expect(
                    output.logits.shape == [1, input.text.tokens.size, config.textConfig.vocabSize])
                #expect(values.allSatisfy { $0.isFinite })
                #expect(Set(values).count > 1)
            }
        }
    }

    @Test("video frame expansion remains unbatched and carries every frame into prefill")
    func videoForward() async throws {
        try await MLXMetalTestLock.withLock {
            let config = try JSONDecoder().decode(
                Glm5NextConfiguration.self,
                from: Data(
                    Glm5NextConstructionTests.tinyJSON.replacingOccurrences(
                        of: "\"image_token_id\":9", with: "\"image_token_id\":99"
                    ).utf8))
            let model = try Glm5Next(config, requesting: [.vision])
            let red = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 28, height: 28))
            let blue = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 28, height: 28))
            let frames: [UserInput.VideoFrame] = [red, red, blue, blue].enumerated().map {
                .init(
                    frame: $0.element,
                    timeStamp: CMTime(seconds: Double($0.offset), preferredTimescale: 600))
            }
            let input = try await Self.processor().prepare(
                input: UserInput(prompt: "compare", videos: [.frames(frames)]))
            #expect(input.text.tokens.ndim == 1)
            #expect(input.image?.frames?.count == 2)
            #expect(!input.text.tokens.asArray(Int.self).contains(98))
            guard
                case .logits(let output) = try model.prepare(
                    input, cache: model.newCache(parameters: nil), windowSize: 512)
            else {
                Issue.record("Video did not exercise media prefill")
                return
            }
            let values = output.logits.asType(.float32).asArray(Float.self)
            #expect(
                output.logits.shape == [1, input.text.tokens.size, config.textConfig.vocabSize])
            #expect(values.allSatisfy { $0.isFinite })
            #expect(Set(values).count > 1)
        }
    }
}
