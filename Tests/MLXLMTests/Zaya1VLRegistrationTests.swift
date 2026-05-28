import Foundation
import CoreImage
import CoreMedia
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
@testable import MLXVLM
import Testing
@preconcurrency import VMLXTokenizers

@Suite("ZAYA1-VL registration and metadata", .serialized)
struct Zaya1VLRegistrationTests {
    static let bundleRoots: [String] = {
        if let override = ProcessInfo.processInfo.environment["VMLINUX_ZAYA_VL_BUNDLE_ROOT"],
           !override.isEmpty
        {
            return [override]
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        return [
            home.appendingPathComponent("models/Zyphra").path,
            home.appendingPathComponent("models/JANGQ").path,
            home.appendingPathComponent("models/Osaurus").path,
        ]
    }()

    static func bundlePath(_ name: String) -> String {
        for root in bundleRoots {
            let path = URL(fileURLWithPath: root).appendingPathComponent(name).path
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }
        return URL(fileURLWithPath: bundleRoots[0]).appendingPathComponent(name).path
    }

    static func fileData(_ bundle: String, _ file: String) throws -> Data {
        try Data(contentsOf: URL(fileURLWithPath: bundlePath(bundle)).appendingPathComponent(file))
    }

    static var hasJANGTQ2TokenizerBundle: Bool {
        let dir = URL(fileURLWithPath: bundlePath("ZAYA1-VL-8B-JANGTQ2"))
        return FileManager.default.fileExists(atPath: dir.appendingPathComponent("tokenizer.json").path)
            && FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("preprocessor_config.json").path)
    }

    static var hasJANGTQ4TokenizerBundle: Bool {
        let dir = URL(fileURLWithPath: bundlePath("ZAYA1-VL-8B-JANGTQ4"))
        return FileManager.default.fileExists(atPath: dir.appendingPathComponent("tokenizer.json").path)
            && FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("preprocessor_config.json").path)
    }

    struct VisionPadTokenizer: MLXLMCommon.Tokenizer {
        var forcedImageCount: Int? = nil
        var bosToken: String? = nil
        var eosToken: String? = nil
        var eosTokenId: Int? = 1_000_001
        var unknownToken: String? = nil
        var unknownTokenId: Int? = 1_000_002

        func encode(text: String, addSpecialTokens: Bool = true) -> [Int] {
            if text.contains("<|vision_start|>") {
                let spanCount = text.components(separatedBy: "<|vision_start|>").count - 1
                if spanCount > 1 {
                    let videoSpans = text.components(separatedBy: "<|video_pad|>").count - 1
                    let padToken = videoSpans > 0 ? 262_148 : 262_147
                    return (0..<spanCount).flatMap { _ in [255_999, padToken, 256_000] }
                }
                let zayaImages = text.components(separatedBy: "<image>").count - 1
                let imagePads = text.components(separatedBy: "<|image_pad|>").count - 1
                let videoPads = text.components(separatedBy: "<|video_pad|>").count - 1
                let padCount = max(zayaImages, max(imagePads, videoPads))
                let padToken = videoPads > 0 ? 262_148 : 262_147
                return [255_999] + Array(repeating: padToken, count: padCount) + [256_000]
            }
            return text.utf8.map { Int($0) % 251 + 10 }
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map(String.init).joined(separator: " ")
        }

        func convertTokenToId(_ token: String) -> Int? {
            switch token {
            case "<|vision_start|>": return 255_999
            case "<|vision_end|>": return 256_000
            case "<image>": return 262_147
            case "<|image_pad|>": return 262_147
            case "<|video_pad|>": return 262_148
            default: return nil
            }
        }

        func convertIdToToken(_ id: Int) -> String? { String(id) }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            let imageCount = forcedImageCount ?? messages.reduce(into: 0) { count, message in
                if let content = message["content"] as? [[String: any Sendable]] {
                    count += content.filter { $0["type"] as? String == "image" }.count
                } else if let content = message["content"] as? [[String: String]] {
                    count += content.filter { $0["type"] == "image" }.count
                }
            }
            if imageCount == 0 {
                return encode(text: "user:Describe.")
            }
            let imageSpans = String(
                repeating: "<|vision_start|><image><|vision_end|>",
                count: imageCount)
            return encode(text: "user:\(imageSpans)\nDescribe.")
        }
    }

    struct TrapTokenizerLoader: MLXLMCommon.TokenizerLoader {
        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
            Issue.record("ZAYA1-VL unsupported load gate should fire before tokenizer or weight load")
            return VisionPadTokenizer()
        }
    }

    static func solidImage(width: Int, height: Int, color: CIColor) -> CIImage {
        let filter = CIFilter(name: "CIConstantColorGenerator")!
        filter.setValue(color, forKey: "inputColor")
        return filter.outputImage!.cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
    }

    static func processorConfigurationData() -> Data {
        """
        {
          "processor_class": "Zaya1VLProcessor",
          "image_processor_type": "Qwen2VLImageProcessor",
          "image_mean": [0.48145466, 0.4578275, 0.40821073],
          "image_std": [0.26862954, 0.26130258, 0.27577711],
          "min_pixels": 3136,
          "max_pixels": 12845056,
          "merge_size": 2,
          "patch_size": 14,
          "temporal_patch_size": 1
        }
        """.data(using: .utf8)!
    }

    struct SafetensorsIndex: Decodable {
        let weightMap: [String: String]

        enum CodingKeys: String, CodingKey {
            case weightMap = "weight_map"
        }
    }

    struct CapabilityStampedConfig: Decodable {
        struct Capabilities: Decodable {
            let reasoningParser: String
            let toolParser: String
            let thinkInTemplate: Bool
            let supportsTools: Bool
            let supportsThinking: Bool
            let family: String
            let modality: String
            let cacheType: String

            enum CodingKeys: String, CodingKey {
                case reasoningParser = "reasoning_parser"
                case toolParser = "tool_parser"
                case thinkInTemplate = "think_in_template"
                case supportsTools = "supports_tools"
                case supportsThinking = "supports_thinking"
                case family
                case modality
                case cacheType = "cache_type"
            }
        }

        let capabilities: Capabilities
    }

    @Test("VLM registry recognizes zaya1_vl without silently routing to text-only zaya")
    func registryRecognizesZaya1VL() async throws {
        #expect(VLMTypeRegistry.supportedModelTypes.contains("zaya1_vl"))
        #expect(
            !VLMTypeRegistry.supportedModelTypes.contains("zaya"),
            "Text ZAYA (`model_type=zaya`) is served by MLXLLM, not MLXVLM; advertising it here makes app-side VLM detection route text-only bundles through the wrong factory."
        )

        let data = """
        {
          "model_type": "zaya1_vl",
          "architectures": ["Zaya1VLForConditionalGeneration"],
          "hidden_size": 2048,
          "num_hidden_layers": 40,
          "num_attention_heads": 8,
          "num_key_value_heads": 2,
          "vocab_size": 262272,
          "image_token_id": 262147,
          "vision_start_token_id": 255999,
          "vision_end_token_id": 256000,
          "vision_config": {
            "model_type": "qwen2_5_vl",
            "hidden_size": 1280,
            "out_hidden_size": 2048,
            "spatial_patch_size": 14,
            "temporal_patch_size": 1,
            "in_chans": 3
          }
        }
        """.data(using: .utf8)!

        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "zaya1_vl")
        #expect(model is Zaya1VL)
    }

    @Test("ZAYA1-VL native model exposes one CCA cache per sequential VL block")
    func nativeModelUsesZayaCCACachesForEveryBlock() async throws {
        let data = """
        {
          "model_type": "zaya1_vl",
          "architectures": ["Zaya1VLForConditionalGeneration"],
          "hidden_size": 2048,
          "num_hidden_layers": 40,
          "num_attention_heads": 8,
          "num_key_value_heads": 2,
          "vocab_size": 262272,
          "image_token_id": 262147,
          "vision_start_token_id": 255999,
          "vision_end_token_id": 256000,
          "vision_lora": true,
          "vision_config": {
            "model_type": "qwen2_5_vl",
            "hidden_size": 1280,
            "out_hidden_size": 2048,
            "spatial_patch_size": 14,
            "temporal_patch_size": 1,
            "in_chans": 3
          }
        }
        """.data(using: .utf8)!

        let model = try #require(try await VLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "zaya1_vl") as? Zaya1VL)
        let caches = model.newCache(parameters: nil)

        #expect(caches.count == 40)
        #expect(caches.allSatisfy { $0 is ZayaCCACache })
    }

    @Test("Zaya1VLProcessor uses ZAYA image placeholders with Qwen2.5-VL image geometry")
    func processorRegistryRecognizesZaya1VLProcessor() async throws {
        try await MLXMetalTestLock.withLock {
            let processor = try await VLMProcessorTypeRegistry.shared.createModel(
                configuration: Self.processorConfigurationData(),
                processorType: "Zaya1VLProcessor",
                tokenizer: TestTokenizer(vocabularySize: 262272))

            #expect(processor is Zaya1VLProcessor)

            let input = try await processor.prepare(input: UserInput(
                chat: [.user("Describe the image.")],
                additionalContext: ["enable_thinking": false]))

            #expect(input.image == nil)
            #expect(input.video == nil)
            #expect(input.cacheScopeSalt == "reasoning=off")
        }
    }

    @Test("Zaya1VLProcessor actually patchifies image input with ZAYA1-VL resolution settings")
    func processorPatchifiesImageInput() async throws {
        try await MLXMetalTestLock.withLock {
            let config = try JSONDecoder.json5().decode(
                Qwen25VLProcessorConfiguration.self,
                from: Self.processorConfigurationData())
            let processor = Zaya1VLProcessor(config, tokenizer: VisionPadTokenizer())

            let image = Self.solidImage(width: 56, height: 56, color: .red)
            let input = try await processor.prepare(input: UserInput(
                prompt: "Describe this image.",
                images: [.ciImage(image)],
                additionalContext: ["enable_thinking": false]))

            let processed = try #require(input.image)
            let frame = try #require(processed.frames?.first)
            #expect(frame.t == 1)
            #expect(frame.h == 4)
            #expect(frame.w == 4)
            #expect(processed.pixels.shape == [16, 588])
            #expect(input.text.tokens.shape == [6])
            #expect(input.text.mask == nil)
            #expect(input.cacheScopeSalt == "reasoning=off")
            #expect(computeMediaSalt(for: input) != nil)
            #expect(computeCacheSalt(for: input) != computeCacheSalt(for: LMInput(
                text: input.text, cacheScopeSalt: input.cacheScopeSalt)))
        }
    }

    @Test("ZAYA1-VL processor preserves Qwen2.5-VL grid math across legal aspect ratios")
    func processorPatchifiesNonSquareImageGrids() async throws {
        try await MLXMetalTestLock.withLock {
            let config = try JSONDecoder.json5().decode(
                Qwen25VLProcessorConfiguration.self,
                from: Self.processorConfigurationData())
            let processor = Zaya1VLProcessor(config, tokenizer: VisionPadTokenizer())

            let cases: [(width: Int, height: Int, expectedGrid: THW, expectedPatchRows: Int)] = [
                (56, 56, THW(1, 4, 4), 16),
                (112, 56, THW(1, 4, 8), 32),
                (56, 84, THW(1, 6, 4), 24),
            ]

            for testCase in cases {
                let input = try await processor.prepare(input: UserInput(
                    prompt: "Describe this image.",
                    images: [.ciImage(Self.solidImage(
                        width: testCase.width, height: testCase.height, color: .green))],
                    additionalContext: ["enable_thinking": false]))

                let processed = try #require(input.image)
                let frame = try #require(processed.frames?.first)
                #expect(frame.t == testCase.expectedGrid.t)
                #expect(frame.h == testCase.expectedGrid.h)
                #expect(frame.w == testCase.expectedGrid.w)
                #expect(processed.pixels.shape == [testCase.expectedPatchRows, 588])

                let expectedVisionPads = testCase.expectedGrid.product / (config.mergeSize * config.mergeSize)
                #expect(input.text.tokens.shape == [expectedVisionPads + 2])
                #expect(input.text.mask == nil)
                #expect(input.cacheScopeSalt == "reasoning=off")
                #expect(computeMediaSalt(for: input) != nil)
            }
        }
    }

    @Test("ZAYA1-VL processor preserves multiple image frames and expands each placeholder")
    func processorPreservesMultipleImageFrames() async throws {
        try await MLXMetalTestLock.withLock {
            let config = try JSONDecoder.json5().decode(
                Qwen25VLProcessorConfiguration.self,
                from: Self.processorConfigurationData())
            let processor = Zaya1VLProcessor(
                config, tokenizer: VisionPadTokenizer(forcedImageCount: 2))

            let input = try await processor.prepare(input: UserInput(
                prompt: "Compare these images.",
                images: [
                    .ciImage(Self.solidImage(width: 56, height: 56, color: .red)),
                    .ciImage(Self.solidImage(width: 112, height: 56, color: .blue)),
                ],
                additionalContext: ["enable_thinking": false]))

            let processed = try #require(input.image)
            let frames = try #require(processed.frames)
            #expect(frames.count == 2)
            #expect(frames[0].t == 1)
            #expect(frames[0].h == 4)
            #expect(frames[0].w == 4)
            #expect(frames[1].t == 1)
            #expect(frames[1].h == 4)
            #expect(frames[1].w == 8)
            #expect(processed.pixels.shape == [48, 588])

            // The tokenizer starts with two `<image>` placeholders. The ZAYA1-VL
            // processor must expand them independently using each frame grid:
            // 16 / merge^2 = 4 tokens, 32 / merge^2 = 8 tokens, plus each
            // image's vision_start/vision_end delimiters.
            #expect(input.text.tokens.shape == [16])
            #expect(input.cacheScopeSalt == "reasoning=off")
            #expect(computeMediaSalt(for: input) != nil)
        }
    }

    @Test("ZAYA1-VL processor drops stale history images when template emits no image placeholder")
    func processorDropsStaleHistoryImagesWhenTemplateHasNoImagePlaceholder() async throws {
        try await MLXMetalTestLock.withLock {
            let config = try JSONDecoder.json5().decode(
                Qwen25VLProcessorConfiguration.self,
                from: Self.processorConfigurationData())
            let processor = Zaya1VLProcessor(
                config, tokenizer: VisionPadTokenizer(forcedImageCount: 0))

            let input = try await processor.prepare(input: UserInput(
                prompt: "The rendered follow-up template has no image placeholder.",
                images: [.ciImage(Self.solidImage(width: 56, height: 56, color: .red))],
                additionalContext: ["enable_thinking": false]))

            #expect(input.image == nil)
            #expect(input.cacheScopeSalt == "reasoning=off")
        }
    }

    @Test("Real ZAYA1-VL tokenizer renders image placeholders from sidecar template",
          .enabled(if: hasJANGTQ2TokenizerBundle))
    func realTokenizerRendersImagePlaceholderFromSidecarTemplate() async throws {
        let dir = URL(fileURLWithPath: Self.bundlePath("ZAYA1-VL-8B-JANGTQ2"))
        let tokenizerDir = JangLoader.resolveTokenizerClassSubstitution(
            for: JangLoader.resolveChatTemplateSidecarSubstitution(
                for: JangLoader.resolveTokenizerDirectory(for: dir)))
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: tokenizerDir)
        let messages = Qwen2VLMessageGenerator().generate(from: UserInput(
            prompt: "Describe this image.",
            images: [.ciImage(Self.solidImage(width: 56, height: 56, color: .red))]))
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: ["enable_thinking": false])
        let decoded = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)

        #expect(decoded.contains("<|vision_start|>"), "decoded prompt: \(decoded)")
        #expect(decoded.contains("<image>") || tokens.contains(262_147),
            "decoded prompt: \(decoded)")
    }

    @Test("Real ZAYA1-VL tokenizer renders text-only tool schemas from shimmed template",
          .enabled(if: hasJANGTQ4TokenizerBundle))
    func realTokenizerRendersTextOnlyToolSchemaFromShimmedTemplate() async throws {
        let dir = URL(fileURLWithPath: Self.bundlePath("ZAYA1-VL-8B-JANGTQ4"))
        let tokenizerDir = JangLoader.resolveTokenizerClassSubstitution(
            for: JangLoader.resolveChatTemplateSidecarSubstitution(
                for: JangLoader.resolveTokenizerDirectory(for: dir)))
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: tokenizerDir)
        let messages: [[String: any Sendable]] = [
            [
                "role": "user",
                "content": "Count the lines in exactly this text: alpha\nbeta\ngamma",
            ],
        ]
        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "line_count",
                    "description": "Count newline-delimited lines.",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "text": [
                                "type": "string",
                                "description": "Text whose lines should be counted.",
                            ] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["text"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: tools,
            additionalContext: [
                "enable_thinking": false,
                "tool_choice": "required",
                "tool_choice_name": "line_count",
            ])
        let decoded = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)

        #expect(decoded.contains("# Tools"), "decoded prompt: \(decoded)")
        #expect(decoded.contains("<name>line_count</name>"), "decoded prompt: \(decoded)")
        #expect(decoded.contains("<name>text</name>"), "decoded prompt: \(decoded)")
        #expect(decoded.contains("<required>[\"text\"]</required>"), "decoded prompt: \(decoded)")
        #expect(decoded.contains("<zyphra_tool_call>"), "decoded prompt: \(decoded)")
        #expect(decoded.contains("Use the `line_count` function."), "decoded prompt: \(decoded)")
        #expect(decoded.contains("Required parameters for `line_count`: text."),
            "decoded prompt: \(decoded)")
        #expect(decoded.contains("<function=line_count>"), "decoded prompt: \(decoded)")
        #expect(decoded.contains("<parameter=text>\nalpha\nbeta\ngamma\n</parameter>"),
            "decoded prompt: \(decoded)")
        #expect(decoded.contains("Required call shape for the current request"),
            "decoded prompt: \(decoded)")
        #expect(!decoded.contains("ACTUAL_ARGUMENT_VALUE"), "decoded prompt: \(decoded)")
        #expect(!decoded.contains("PARAMETER_NAME"), "decoded prompt: \(decoded)")
        #expect(!decoded.contains("VALUE_FOR_text"), "decoded prompt: \(decoded)")
        #expect(!decoded.contains("VALUE_FOR_*"), "decoded prompt: \(decoded)")
        #expect(!decoded.contains("value_1"), "decoded prompt: \(decoded)")
    }

    @Test("Real ZAYA1-VL tokenizer grounds required tool text from media messages",
          .enabled(if: hasJANGTQ4TokenizerBundle))
    func realTokenizerGroundsRequiredToolTextFromMediaMessages() async throws {
        let dir = URL(fileURLWithPath: Self.bundlePath("ZAYA1-VL-8B-JANGTQ4"))
        let tokenizerDir = JangLoader.resolveTokenizerClassSubstitution(
            for: JangLoader.resolveChatTemplateSidecarSubstitution(
                for: JangLoader.resolveTokenizerDirectory(for: dir)))
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: tokenizerDir)
        let messages: [[String: any Sendable]] = [
            [
                "role": "user",
                "content": [
                    ["type": "image"] as [String: any Sendable],
                    [
                        "type": "text",
                        "text": "Call line_count on this exact text: alpha\nbeta\ngamma",
                    ] as [String: any Sendable],
                ] as [[String: any Sendable]],
            ],
        ]
        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "line_count",
                    "description": "Count newline-delimited lines.",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "text": [
                                "type": "string",
                                "description": "Text whose lines should be counted.",
                            ] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["text"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: tools,
            additionalContext: [
                "enable_thinking": false,
                "tool_choice": "required",
                "tool_choice_name": "line_count",
            ])
        let decoded = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)

        #expect(decoded.contains("<|vision_start|><image><|vision_end|>"),
            "decoded prompt: \(decoded)")
        #expect(
            decoded.components(separatedBy: "<|vision_start|><image><|vision_end|>").count - 1 == 1,
            "decoded prompt must render the single media image exactly once: \(decoded)")
        #expect(decoded.contains("<function=line_count>"), "decoded prompt: \(decoded)")
        #expect(decoded.contains("<parameter=text>\nalpha\nbeta\ngamma\n</parameter>"),
            "decoded prompt: \(decoded)")
        #expect(!decoded.contains("value_1"), "decoded prompt: \(decoded)")
    }

    @Test("Real ZAYA1-VL processor expands sidecar-rendered image placeholders",
          .enabled(if: hasJANGTQ2TokenizerBundle))
    func realProcessorExpandsSidecarRenderedImagePlaceholder() async throws {
        try await MLXMetalTestLock.withLock {
            let dir = URL(fileURLWithPath: Self.bundlePath("ZAYA1-VL-8B-JANGTQ2"))
            let tokenizerDir = JangLoader.resolveTokenizerClassSubstitution(
                for: JangLoader.resolveChatTemplateSidecarSubstitution(
                    for: JangLoader.resolveTokenizerDirectory(for: dir)))
            let tokenizer = try await #huggingFaceTokenizerLoader().load(from: tokenizerDir)
            let configData = try Data(
                contentsOf: dir.appendingPathComponent("preprocessor_config.json"))
            let config = try JSONDecoder.json5().decode(
                Qwen25VLProcessorConfiguration.self, from: configData)
            let processor = Zaya1VLProcessor(config, tokenizer: tokenizer)

            let input = try await processor.prepare(input: UserInput(
                prompt: "Describe this image.",
                images: [.ciImage(Self.solidImage(width: 56, height: 56, color: .red))],
                additionalContext: ["enable_thinking": false]))
            let processed = try #require(input.image)
            #expect(processed.frames?.count == 1)
            #expect(input.text.tokens.size > 3)
        }
    }

    @Test("ZAYA1-VL processor rejects video explicitly instead of silently dropping it")
    func processorRejectsVideoInput() async throws {
        try await MLXMetalTestLock.withLock {
            let config = try JSONDecoder.json5().decode(
                Qwen25VLProcessorConfiguration.self,
                from: Self.processorConfigurationData())
            let processor = Zaya1VLProcessor(config, tokenizer: VisionPadTokenizer())
            let frame = UserInput.VideoFrame(
                frame: Self.solidImage(width: 56, height: 56, color: .green),
                timeStamp: .zero)

            do {
                _ = try await processor.prepare(input: UserInput(
                    prompt: "Describe this clip.",
                    videos: [.frames([frame])],
                    additionalContext: ["enable_thinking": false]))
                Issue.record("ZAYA1-VL video input should reject until native video support exists")
            } catch let error as VLMError {
                guard case .processing(let message) = error else {
                    Issue.record("Expected VLMError.processing, got \(error)")
                    return
                }
                #expect(message.contains("ZAYA1-VL video input is not implemented"))
            } catch {
                Issue.record("Expected VLMError.processing, got \(error)")
            }
        }
    }

    @Test("ZAYA1-VL processor applies Qwen2.5-VL RGB normalization to image pixels")
    func processorNormalizesSolidBlackAndWhitePixels() throws {
        try MLXMetalTestLock.withLock {
            let config = try JSONDecoder.json5().decode(
                Qwen25VLProcessorConfiguration.self,
                from: Self.processorConfigurationData())
            let processor = Zaya1VLProcessor(config, tokenizer: VisionPadTokenizer())

            let black = try processor.preprocess(
                images: [Self.solidImage(width: 56, height: 56, color: .black)],
                processing: nil)
            let white = try processor.preprocess(
                images: [Self.solidImage(width: 56, height: 56, color: .white)],
                processing: nil)

            #expect(black.1.t == 1)
            #expect(black.1.h == 4)
            #expect(black.1.w == 4)
            #expect(white.1.t == 1)
            #expect(white.1.h == 4)
            #expect(white.1.w == 4)
            #expect(black.0.shape == [16, 588])
            #expect(white.0.shape == [16, 588])

            let blackValues = black.0.asArray(Float.self)
            let whiteValues = white.0.asArray(Float.self)
            let expectedBlack = [
                Float(-0.48145466 / 0.26862954),
                Float(-0.4578275 / 0.26130258),
                Float(-0.40821073 / 0.27577711),
            ]
            let expectedWhite = [
                Float((1.0 - 0.48145466) / 0.26862954),
                Float((1.0 - 0.4578275) / 0.26130258),
                Float((1.0 - 0.40821073) / 0.27577711),
            ]

            for expected in expectedBlack {
                #expect(blackValues.contains { abs($0 - expected) < 0.02 })
            }
            for expected in expectedWhite {
                #expect(whiteValues.contains { abs($0 - expected) < 0.02 })
            }
        }
    }

    @Test("ZAYA1-VL image preprocessing rejects dimensions below patch-merge factor")
    func processorRejectsTooSmallImages() throws {
        try MLXMetalTestLock.withLock {
            let config = try JSONDecoder.json5().decode(
                Qwen25VLProcessorConfiguration.self,
                from: Self.processorConfigurationData())
            let processor = Zaya1VLProcessor(config, tokenizer: VisionPadTokenizer())

            do {
                _ = try processor.preprocess(
                    images: [Self.solidImage(width: 27, height: 56, color: .blue)],
                    processing: nil)
                Issue.record("ZAYA1-VL/Qwen2-VL preprocessing must reject width < patchSize*mergeSize")
            } catch let error as VLMError {
                guard case .imageProcessingFailure(let message) = error else {
                    Issue.record("Expected imageProcessingFailure, got \(error)")
                    return
                }
                #expect(message.contains("Width"))
                #expect(message.contains("factor"))
            } catch {
                Issue.record("Expected VLMError.imageProcessingFailure, got \(error)")
            }
        }
    }

    @Test("ZAYA1-VL cache key contract combines media hash with request scope")
    func cacheSaltCombinesMediaAndRequestScope() {
        MLXMetalTestLock.withLock {
            let text = LMInput.Text(tokens: MLXArray([Int32(255999), 262147, 256000, 42]))
            let pixelsA = MLXArray(Array(UInt8(0)..<UInt8(48)), [3, 4, 4])
            let pixelsB = MLXArray(Array(UInt8(1)..<UInt8(49)), [3, 4, 4])
            let imageA = LMInput.ProcessedImage(pixels: pixelsA, frames: [THW(1, 2, 2)])
            let imageB = LMInput.ProcessedImage(pixels: pixelsB, frames: [THW(1, 2, 2)])

            let textOnly = LMInput(text: text, cacheScopeSalt: "reasoning=off")
            let imageScopedA = LMInput(text: text, image: imageA, cacheScopeSalt: "reasoning=off")
            let imageScopedA2 = LMInput(text: text, image: imageA, cacheScopeSalt: "reasoning=off")
            let imageScopedB = LMInput(text: text, image: imageB, cacheScopeSalt: "reasoning=off")

            #expect(computeMediaSalt(for: textOnly) == nil)
            #expect(computeMediaSalt(for: imageScopedA) != nil)
            #expect(computeCacheSalt(for: imageScopedA) == computeCacheSalt(for: imageScopedA2))
            #expect(computeCacheSalt(for: imageScopedA) != computeCacheSalt(for: imageScopedB))
            #expect(computeCacheSalt(for: imageScopedA) != computeCacheSalt(for: textOnly))
        }
    }

    @Test("ZAYA1-VL image merge replaces only image-token embeddings and preserves mask")
    func imageEmbeddingMergeReplacesOnlyImageTokenPositions() throws {
        try MLXMetalTestLock.withLock {
            let inputIds = MLXArray([Int32(10), 262147, 11, 262147])
            let inputEmbeds = MLXArray([
                    1, 2, 3,
                    4, 5, 6,
                    7, 8, 9,
                    10, 11, 12,
                ] as [Float])
                .reshaped([1, 4, 3])
            let imageFeatures = MLXArray([
                    100, 101, 102,
                    200, 201, 202,
                ] as [Float])
                .reshaped([2, 3])

            let merged = try Zaya1VLRuntimeSupport.mergeImageFeatures(
                inputIds: inputIds,
                inputEmbeds: inputEmbeds,
                imageFeatures: imageFeatures,
                imageTokenId: 262147)
            let imageMask = try #require(merged.imageMask)

            #expect(imageMask.shape == [4])
            #expect(imageMask.asArray(Bool.self) == [false, true, false, true])
            #expect(merged.embeddings.shape == [1, 4, 3])
            #expect(merged.embeddings.asArray(Float.self) == [
                1, 2, 3,
                100, 101, 102,
                7, 8, 9,
                200, 201, 202,
            ])
        }
    }

    @Test("ZAYA1-VL image merge rejects image-token and feature count mismatches")
    func imageEmbeddingMergeRejectsFeatureCountMismatch() {
        MLXMetalTestLock.withLock {
            let inputIds = MLXArray([Int32(10), 262147, 11, 262147])
            let inputEmbeds = MLXArray.zeros([1, 4, 3], dtype: .float32)
            let imageFeatures = MLXArray.zeros([1, 3], dtype: .float32)

            #expect(throws: VLMError.self) {
                _ = try Zaya1VLRuntimeSupport.mergeImageFeatures(
                    inputIds: inputIds,
                    inputEmbeds: inputEmbeds,
                    imageFeatures: imageFeatures,
                    imageTokenId: 262147)
            }
        }
    }

    @Test("ZAYA1-VL image merge supports batch-shaped token ids")
    func imageEmbeddingMergeSupportsBatchShapedInputIds() throws {
        try MLXMetalTestLock.withLock {
            let inputIds = MLXArray([
                Int32(10), 262147, 11,
                262147, 12, 13,
            ]).reshaped([2, 3])
            let inputEmbeds = MLXArray([
                    1, 2, 3,
                    4, 5, 6,
                    7, 8, 9,
                    10, 11, 12,
                    13, 14, 15,
                    16, 17, 18,
                ] as [Float])
                .reshaped([2, 3, 3])
            let imageFeatures = MLXArray([
                    100, 101, 102,
                    200, 201, 202,
                ] as [Float])
                .reshaped([2, 3])

            let merged = try Zaya1VLRuntimeSupport.mergeImageFeatures(
                inputIds: inputIds,
                inputEmbeds: inputEmbeds,
                imageFeatures: imageFeatures,
                imageTokenId: 262147)
            let imageMask = try #require(merged.imageMask)

            #expect(imageMask.shape == [2, 3])
            #expect(imageMask.asArray(Bool.self) == [
                false, true, false,
                true, false, false,
            ])
            #expect(merged.embeddings.shape == [2, 3, 3])
            #expect(merged.embeddings.asArray(Float.self) == [
                1, 2, 3,
                100, 101, 102,
                7, 8, 9,
                200, 201, 202,
                13, 14, 15,
                16, 17, 18,
            ])
        }
    }

    @Test("ZAYA1-VL image merge preserves text embedding dtype")
    func imageEmbeddingMergePreservesEmbeddingDtype() throws {
        try MLXMetalTestLock.withLock {
            let inputIds = MLXArray([Int32(262147)])
            let inputEmbeds = MLXArray.ones([1, 1, 4], dtype: .bfloat16)
            let imageFeatures = MLXArray.ones([1, 4], dtype: .float32)

            let merged = try Zaya1VLRuntimeSupport.mergeImageFeatures(
                inputIds: inputIds,
                inputEmbeds: inputEmbeds,
                imageFeatures: imageFeatures,
                imageTokenId: 262147)

            #expect(merged.embeddings.dtype == .bfloat16)
        }
    }

    @Test("ZAYA1-VL image merge rejects projected feature width mismatches")
    func imageEmbeddingMergeRejectsFeatureWidthMismatch() {
        MLXMetalTestLock.withLock {
            let inputIds = MLXArray([Int32(262147), 262147])
            let inputEmbeds = MLXArray.zeros([1, 2, 3], dtype: .float32)
            let imageFeatures = MLXArray.zeros([2, 2], dtype: .float32)

            #expect(throws: VLMError.self) {
                _ = try Zaya1VLRuntimeSupport.mergeImageFeatures(
                    inputIds: inputIds,
                    inputEmbeds: inputEmbeds,
                    imageFeatures: imageFeatures,
                    imageTokenId: 262147)
            }
        }
    }

    @Test("Real local ZAYA1-VL bundles decode config, preprocessor, and quant-bit metadata",
          .enabled(if: FileManager.default.fileExists(
              atPath: bundlePath("ZAYA1-VL-8B-JANGTQ2") + "/config.json")))
    func realBundleMetadataDecodes() throws {
        let cases: [(String, String, Int?, Int?)] = [
            ("ZAYA1-VL-8B-MXFP4", "mxfp4", nil, nil),
            ("ZAYA1-VL-8B-JANGTQ2", "mxtq", 2, 2),
            ("ZAYA1-VL-8B-JANGTQ4", "mxtq", 4, 4),
            ("ZAYA1-VL-8B-JANGTQ_K", "mxtq", 2, 4),
        ]

        for (bundle, weightFormat, gateUpBits, downBits) in cases {
            guard FileManager.default.fileExists(atPath: Self.bundlePath(bundle) + "/config.json")
            else { continue }

            let config = try JSONDecoder.json5().decode(
                Zaya1VLConfiguration.self,
                from: Self.fileData(bundle, "config.json"))
            let processor = try JSONDecoder.json5().decode(
                Qwen25VLProcessorConfiguration.self,
                from: Self.fileData(bundle, "preprocessor_config.json"))

            #expect(config.modelType == "zaya1_vl")
            #expect(config.architectures == ["Zaya1VLForConditionalGeneration"])
            #expect(config.hiddenSize == 2048)
            #expect(config.numHiddenLayers == 40)
            #expect(config.numAttentionHeads == 8)
            #expect(config.numKeyValueHeads == 2)
            #expect(config.headDim == 128)
            #expect(config.numQueryGroups == 2)
            #expect(config.maxPositionEmbeddings == 32768)
            #expect(config.rotaryBase == 1_000_000)
            #expect(config.ropePct == 0.5)
            #expect(config.ffnHiddenSize == 4096)
            #expect(config.zayaMLPExpansion == 256)
            #expect(config.zayaExpertLayout == "split_switch_mlp")
            #expect(config.normEpsilon == 1e-5)
            #expect(!config.clampTemp)
            #expect(config.projectorHiddenAct == "gelu")
            #expect(config.numExperts == 16)
            #expect(config.moeRouterTopk == 1)
            #expect(config.cca)
            #expect(config.zayaUseEDA)
            #expect(config.zayaUseMOD)
            #expect(config.scaleResidualMerge)
            #expect(!config.residualInFP32)
            #expect(config.tieWordEmbeddings)
            #expect(config.visionLora)
            #expect(config.visionLoraRankAttn == 8)
            #expect(config.visionLoraRankMLP == 32)
            #expect(config.imageTokenId == 262147)
            #expect(config.visionStartTokenId == 255999)
            #expect(config.visionEndTokenId == 256000)
            #expect(config.visionConfiguration.modelType == "qwen2_5_vl")
            #expect(config.visionConfiguration.depth == 32)
            #expect(config.visionConfiguration.hiddenSize == 1280)
            #expect(config.visionConfiguration.intermediateSize == 3420)
            #expect(config.visionConfiguration.outHiddenSize == 2048)
            #expect(config.visionConfiguration.numHeads == 16)
            #expect(config.visionConfiguration.patchSize == 14)
            #expect(config.visionConfiguration.spatialPatchSize == 14)
            #expect(config.visionConfiguration.spatialMergeSize == 2)
            #expect(config.visionConfiguration.temporalPatchSize == 1)
            #expect(config.visionConfiguration.windowSize == 112)
            #expect(config.visionConfiguration.fullattBlockIndexes == [7, 15, 23, 31])
            #expect(config.visionConfiguration.tokensPerSecond == 2)
            #expect(config.visionConfiguration.inChannels == 3)
            #expect(config.visionConfiguration.layerNormEps == 1e-6)
            #expect(!config.visionConfiguration.skipVision)
            #expect(config.visionConfiguration.hiddenAct == "silu")
            let qwenVision = try config.makeQwen25VisionConfiguration()
            #expect(qwenVision.depth == 32)
            #expect(qwenVision.hiddenSize == 1280)
            #expect(qwenVision.intermediateSize == 3420)
            #expect(qwenVision.outHiddenSize == 2048)
            #expect(qwenVision.numHeads == 16)
            #expect(qwenVision.patchSize == 14)
            #expect(qwenVision.spatialMergeSize == 2)
            #expect(qwenVision.temporalPatchSize == 1)
            #expect(qwenVision.windowSize == 112)
            #expect(qwenVision.fullattBlockIndexes == [7, 15, 23, 31])
            #expect(qwenVision.tokensPerSecond == 2)
            #expect(qwenVision.inChannels == 3)
            #expect(qwenVision.layerNormEps == 1e-6)
            #expect(!qwenVision.skipVision)
            #expect(qwenVision.hiddenAct == "silu")
            #expect(config.weightFormat == weightFormat)
            #expect(config.routedExpertBits == gateUpBits)
            #expect(config.routedExpertGateUpBits == gateUpBits)
            #expect(config.routedExpertDownBits == downBits)

            #expect(processor.imageProcessorType == "Qwen2VLImageProcessor")
            #expect(processor.patchSize == 14)
            #expect(processor.mergeSize == 2)
            #expect(processor.temporalPatchSize == 1)
            #expect(processor.minPixels == 3136)
            #expect(processor.maxPixels == 12_845_056)
        }
    }

    @Test("ZAYA1-VL config maps to Zaya text primitive dimensions without 80-layer coercion",
          .enabled(if: FileManager.default.fileExists(
              atPath: bundlePath("ZAYA1-VL-8B-JANGTQ2") + "/config.json")))
    func zayaTextPrimitiveConfigurationMapping() throws {
        let config = try JSONDecoder.json5().decode(
            Zaya1VLConfiguration.self,
            from: Self.fileData("ZAYA1-VL-8B-JANGTQ2", "config.json"))
        let text = config.makeZayaTextConfiguration()

        #expect(text.modelType == "zaya1_vl")
        #expect(text.hiddenSize == 2048)
        #expect(text.numHiddenLayers == 40)
        #expect(text.numAttentionHeads == 8)
        #expect(text.numKeyValueHeads == 2)
        #expect(text.numQueryGroups == 2)
        #expect(text.ccaNumQHeads == 8)
        #expect(text.kvChannels == 128)
        #expect(text.maxPositionEmbeddings == 32768)
        #expect(text.ropeTheta == 1_000_000)
        #expect(text.partialRotaryFactor == 0.5)
        #expect(text.ffnHiddenSize == 4096)
        #expect(text.numExperts == 16)
        #expect(text.moeRouterTopk == 1)
        #expect(text.vocabSize == 262272)
        #expect(text.tieWordEmbeddings)
        #expect(text.scaleResidualMerge)
        #expect(!text.residualInFP32)
        #expect(text.weightFormat == config.weightFormat)
        #expect(text.mxtqBits == 2)
        #expect(text.zayaExpertLayout == "split_switch_mlp")
    }

    @Test("Real local ZAYA1-VL safetensors indexes expose vision tower, text blocks, LoRA, and quant sidecars",
          .enabled(if: FileManager.default.fileExists(
              atPath: bundlePath("ZAYA1-VL-8B-JANGTQ2") + "/model.safetensors.index.json")))
    func realBundleStructureMatchesNativeAdapterRequirements() throws {
        let cases: [(bundle: String, requiresJANGTQSidecar: Bool)] = [
            ("ZAYA1-VL-8B-MXFP4", false),
            ("ZAYA1-VL-8B-JANGTQ2", true),
            ("ZAYA1-VL-8B-JANGTQ4", true),
        ]

        for testCase in cases {
            guard FileManager.default.fileExists(
                atPath: Self.bundlePath(testCase.bundle) + "/model.safetensors.index.json")
            else { continue }

            let index = try JSONDecoder.json5().decode(
                SafetensorsIndex.self,
                from: Self.fileData(testCase.bundle, "model.safetensors.index.json"))
            let keys = Set(index.weightMap.keys)

            #expect(keys.contains("model.embed_tokens.weight"))
            #expect(keys.contains { $0.hasPrefix("vision_tower.patch_embed.") })
            #expect(keys.contains { $0.hasPrefix("vision_tower.blocks.0.attn.") })
            #expect(keys.contains { $0.hasPrefix("vision_tower.merger.mlp.") })
            #expect(!keys.contains { $0.contains("mm_projector") })
            #expect(!keys.contains { $0.contains("vision_projection") })

            // ZAYA1-VL is not text-only Zaya: every text layer exposes both
            // CCA attention and MLP/residual blocks, plus vision-gated LoRA.
            for layer in 0..<40 {
                #expect(keys.contains { $0.hasPrefix("model.layers.\(layer).attn.self_attn.") })
                #expect(keys.contains { $0.hasPrefix("model.layers.\(layer).mlp.") })
                #expect(keys.contains {
                    $0.hasPrefix("model.layers.\(layer).attn.self_attn.lora_linear_")
                })
                #expect(keys.contains {
                    $0.hasPrefix("model.layers.\(layer).mlp.zaya_block.experts.local_experts.")
                        && $0.contains(".lora_fc")
                })
                #expect(keys.contains {
                    $0.hasPrefix("model.layers.\(layer).zaya_block.experts.switch_mlp.")
                })
            }

            if testCase.requiresJANGTQSidecar {
                #expect(FileManager.default.fileExists(
                    atPath: Self.bundlePath(testCase.bundle) + "/jangtq_runtime.safetensors"))
                #expect(keys.contains(
                    "model.layers.0.zaya_block.experts.switch_mlp.gate_proj.tq_packed"))
                #expect(keys.contains(
                    "model.layers.0.zaya_block.experts.switch_mlp.up_proj.tq_norms"))
                #expect(keys.contains(
                    "model.layers.0.zaya_block.experts.switch_mlp.down_proj.tq_bits"))
            } else {
                #expect(!FileManager.default.fileExists(
                    atPath: Self.bundlePath(testCase.bundle) + "/jangtq_runtime.safetensors"))
                #expect(keys.contains(
                    "model.layers.0.zaya_block.experts.switch_mlp.gate_proj.weight"))
                #expect(keys.contains(
                    "model.layers.0.zaya_block.experts.switch_mlp.up_proj.scales"))
                #expect(keys.contains(
                    "model.layers.0.zaya_block.experts.switch_mlp.down_proj.biases"))
            }
        }
    }

    @Test("Real local ZAYA/ZAYA1-VL bundles pin parser and default-template capability stamps",
          .enabled(if: FileManager.default.fileExists(
              atPath: bundlePath("ZAYA1-VL-8B-JANGTQ2") + "/config.json")))
    func realBundleCapabilityStampsAreParserSafe() throws {
        let cases: [(bundle: String, family: String, modality: String)] = [
            // Text ZAYA supports reasoning. Runtime code trusts bundle
            // capability stamps; stale metadata must be corrected in the
            // bundle, not by family-name normalization.
            ("ZAYA1-8B-JANGTQ2", "zaya", "text"),
            ("ZAYA1-8B-JANGTQ4", "zaya", "text"),
            // ZAYA1-VL parser stamps are tracked separately from native
            // generation readiness; no family-name thinking override is
            // applied by the VLM factory.
            ("ZAYA1-VL-8B-MXFP4", "zaya1_vl", "vision"),
            ("ZAYA1-VL-8B-JANGTQ2", "zaya1_vl", "vision"),
            ("ZAYA1-VL-8B-JANGTQ4", "zaya1_vl", "vision"),
        ]

        for testCase in cases {
            for file in ["config.json", "jang_config.json"] {
                guard FileManager.default.fileExists(
                    atPath: Self.bundlePath(testCase.bundle) + "/" + file)
                else { continue }

                let stamped = try JSONDecoder.json5().decode(
                    CapabilityStampedConfig.self,
                    from: Self.fileData(testCase.bundle, file))
                let caps = stamped.capabilities
                #expect(caps.family == testCase.family)
                #expect(caps.modality == testCase.modality)
                #expect(caps.cacheType == "hybrid")
                #expect(caps.reasoningParser == "qwen3")
                #expect(caps.toolParser == "zaya_xml")
                #expect(!caps.thinkInTemplate)
                #expect(caps.supportsTools)
            }
        }
    }

    @Test("Real local ZAYA templates default thinking off and expose opt-in only where present",
          .enabled(if: FileManager.default.fileExists(
              atPath: bundlePath("ZAYA1-VL-8B-JANGTQ2") + "/chat_template.json")))
    func realBundleTemplatesMatchOptInReasoningPolicy() throws {
        for bundle in ["ZAYA1-8B-JANGTQ2", "ZAYA1-8B-JANGTQ4"] {
            guard FileManager.default.fileExists(
                atPath: Self.bundlePath(bundle) + "/chat_template.jinja")
            else { continue }

            let template = String(
                data: try Self.fileData(bundle, "chat_template.jinja"),
                encoding: .utf8) ?? ""
            #expect(template.contains("enable_thinking"))
            #expect(template.contains("<think>"))
            #expect(template.contains("</think>"))
            #expect(template.contains("<zyphra_tool_call>"))
        }

        for bundle in ["ZAYA1-VL-8B-MXFP4", "ZAYA1-VL-8B-JANGTQ2", "ZAYA1-VL-8B-JANGTQ4"] {
            guard FileManager.default.fileExists(
                atPath: Self.bundlePath(bundle) + "/chat_template.json")
            else { continue }

            let templateJSON = try JSONSerialization.jsonObject(
                with: Self.fileData(bundle, "chat_template.json")) as? [String: String]
            let template = templateJSON?["chat_template"] ?? ""
            #expect(template.contains("<|vision_start|><image><|vision_end|>"))
            #expect(template.contains("<|im_start|>assistant"))
            #expect(!template.contains("enable_thinking"))
            #expect(!template.contains("<think>"))
        }
    }
}
